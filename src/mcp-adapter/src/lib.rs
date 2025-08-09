use chrono::Utc;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use reqwest::{Client, StatusCode};
use std::sync::{Arc, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::time::Instant;
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::time::{sleep, timeout};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

pub mod auth;
pub mod client;
pub mod connection;
pub mod error;
pub mod message;
pub mod queue;

pub use auth::*;
pub use connection::*;
pub use error::*;
pub use message::*;
pub use queue::*;

/// Configuration for the MCP adapter
#[derive(Debug, Clone)]
pub struct McpConfig {
    pub endpoint: String,
    pub max_retries: usize,
    pub retry_base_delay_ms: u64,
    pub max_retry_delay_ms: u64,
    pub connection_timeout_ms: u64,
    pub message_timeout_ms: u64,
    pub max_concurrent_messages: usize,
    pub queue_capacity: usize,
    pub auth_refresh_threshold_secs: i64,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8080".to_string(),
            max_retries: 3,
            retry_base_delay_ms: 100,
            max_retry_delay_ms: 30000,
            connection_timeout_ms: 5000,
            message_timeout_ms: 10000,
            max_concurrent_messages: 100,
            queue_capacity: 1000,
            auth_refresh_threshold_secs: 300, // 5 minutes before expiry
        }
    }
}

/// Main MCP Protocol Adapter
pub struct McpAdapter {
    pub config: McpConfig,
    pub client: Client,
    pub connection: Arc<RwLock<Option<Connection>>>,
    pub auth: Arc<RwLock<Option<AuthToken>>>,
    pub message_queue: Arc<MessageQueue>,
    pub auth_handler: AuthHandler,
    pub retry_count: AtomicUsize,
    pub message_counter: AtomicU64,
    pub semaphore: Arc<Semaphore>,
}

impl McpAdapter {
    /// Create a new MCP adapter with default configuration
    pub fn new() -> Self {
        Self::with_config(McpConfig::default())
    }

    /// Create a new MCP adapter with custom configuration
    pub fn with_config(config: McpConfig) -> Self {
        let client = Client::builder()
            .timeout(std::time::Duration::from_millis(config.connection_timeout_ms))
            .build()
            .expect("Failed to create HTTP client");

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_messages));
        let message_queue = Arc::new(MessageQueue::new(config.queue_capacity));

        Self {
            config,
            client,
            connection: Arc::new(RwLock::new(None)),
            auth: Arc::new(RwLock::new(None)),
            message_queue,
            auth_handler: AuthHandler::new(),
            retry_count: AtomicUsize::new(0),
            message_counter: AtomicU64::new(0),
            semaphore,
        }
    }

    /// Connect to the MCP server with exponential backoff retry logic
    pub async fn connect(&self, endpoint: &str) -> Result<Connection> {
        let mut retry_count = 0;
        let mut delay_ms = self.config.retry_base_delay_ms;

        loop {
            match self.attempt_connection(endpoint).await {
                Ok(connection) => {
                    info!("Successfully connected to MCP server at {}", endpoint);
                    *self.connection.write() = Some(connection.clone());
                    self.retry_count.store(retry_count, Ordering::Relaxed);
                    return Ok(connection);
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count > self.config.max_retries {
                        error!("Failed to connect after {} retries: {}", self.config.max_retries, e);
                        self.retry_count.store(retry_count, Ordering::Relaxed);
                        return Err(e);
                    }

                    warn!("Connection attempt {} failed: {}. Retrying in {}ms", retry_count, e, delay_ms);
                    sleep(std::time::Duration::from_millis(delay_ms)).await;
                    
                    // Exponential backoff with jitter
                    delay_ms = std::cmp::min(
                        delay_ms * 2 + (rand::random::<u64>() % 100),
                        self.config.max_retry_delay_ms,
                    );
                }
            }
        }
    }

    /// Attempt a single connection
    async fn attempt_connection(&self, endpoint: &str) -> Result<Connection> {
        let url = url::Url::parse(endpoint)?;
        
        let response = timeout(
            std::time::Duration::from_millis(self.config.connection_timeout_ms),
            self.client.get(&format!("{}/health", endpoint)).send(),
        )
        .await
        .map_err(|_| McpError::MessageTimeout { 
            timeout_ms: self.config.connection_timeout_ms 
        })?
        .map_err(McpError::NetworkError)?;

        if response.status().is_success() {
            Ok(Connection::new(
                Uuid::new_v4(),
                endpoint.to_string(),
                Utc::now(),
            ))
        } else {
            Err(McpError::ConnectionFailed(format!(
                "Health check failed with status: {}",
                response.status()
            )))
        }
    }

    /// Authenticate with OAuth2/JWT and handle token refresh
    pub async fn authenticate(&self, credentials: Credentials) -> Result<AuthToken> {
        let token = self.auth_handler.authenticate(&self.client, &self.config.endpoint, credentials).await?;
        
        info!("Authentication successful, token expires at {}", token.expires_at);
        *self.auth.write() = Some(token.clone());
        
        Ok(token)
    }

    /// Check if current token needs refresh and refresh if necessary
    pub async fn ensure_valid_token(&self) -> Result<()> {
        let needs_refresh = {
            let auth_guard = self.auth.read();
            match auth_guard.as_ref() {
                Some(token) => token.needs_refresh(self.config.auth_refresh_threshold_secs),
                None => return Err(McpError::AuthenticationFailed("No token available".to_string())),
            }
        };

        if needs_refresh {
            let mut auth_guard = self.auth.write();
            if let Some(ref mut current_token) = auth_guard.as_mut() {
                let refreshed_token = self.auth_handler.refresh_token(&self.client, &self.config.endpoint, current_token).await?;
                **current_token = refreshed_token;
                info!("Token refreshed successfully");
            }
        }

        Ok(())
    }

    /// Send a message with timeout and concurrent handling
    pub async fn send_message(&self, message: Message) -> Result<Response> {
        // Ensure we have a valid connection
        if self.connection.read().is_none() {
            return Err(McpError::ConnectionFailed("Not connected to server".to_string()));
        }

        // Acquire semaphore permit for concurrency control
        let _permit = self.semaphore.acquire().await
            .map_err(|_| McpError::Internal("Failed to acquire concurrency permit".to_string()))?;

        // Ensure valid authentication
        self.ensure_valid_token().await?;

        // Increment message counter
        let message_id = self.message_counter.fetch_add(1, Ordering::SeqCst);
        
        // Create message with ID and timestamp
        let mut msg = message;
        msg.id = Some(message_id);
        msg.timestamp = Some(Utc::now());

        // Add to queue for processing
        self.message_queue.enqueue(msg.clone()).await?;

        // Send message with timeout
        let start_time = Instant::now();
        let result = timeout(
            std::time::Duration::from_millis(self.config.message_timeout_ms),
            self.send_message_internal(msg),
        )
        .await
        .map_err(|_| McpError::MessageTimeout { 
            timeout_ms: self.config.message_timeout_ms 
        })?;

        // Log performance metrics
        let duration = start_time.elapsed();
        debug!("Message {} processed in {:?}", message_id, duration);

        result
    }

    /// Internal message sending implementation
    async fn send_message_internal(&self, message: Message) -> Result<Response> {
        let auth_token = {
            let auth_guard = self.auth.read();
            auth_guard.as_ref()
                .map(|token| token.access_token.clone())
                .ok_or_else(|| McpError::AuthenticationFailed("No access token available".to_string()))?
        };

        let endpoint = format!("{}/api/messages", self.config.endpoint);
        let response = self.client
            .post(&endpoint)
            .header("Authorization", format!("Bearer {}", auth_token))
            .header("Content-Type", "application/json")
            .json(&message)
            .send()
            .await?;

        match response.status() {
            StatusCode::OK => {
                let response_data: Response = response.json().await?;
                Ok(response_data)
            }
            StatusCode::UNAUTHORIZED => {
                Err(McpError::AuthenticationFailed("Token expired or invalid".to_string()))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                Err(McpError::RateLimitExceeded)
            }
            status => {
                let error_text = response.text().await.unwrap_or_default();
                Err(McpError::Internal(format!("Request failed with status {}: {}", status, error_text)))
            }
        }
    }

    /// Get current retry count for testing
    pub fn retry_count(&self) -> usize {
        self.retry_count.load(Ordering::Relaxed)
    }

    /// Get current message count
    pub fn message_count(&self) -> u64 {
        self.message_counter.load(Ordering::Relaxed)
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connection.read().is_some()
    }

    /// Get queue statistics
    pub fn queue_stats(&self) -> QueueStats {
        self.message_queue.stats()
    }

    /// Process queued messages concurrently
    pub async fn process_queue(&self, batch_size: usize) -> Result<Vec<Result<Response>>> {
        let messages = self.message_queue.dequeue_batch(batch_size).await;
        if messages.is_empty() {
            return Ok(vec![]);
        }

        let futures: Vec<_> = messages
            .into_iter()
            .map(|msg| Box::pin(self.send_message_internal(msg)) as BoxFuture<Result<Response>>)
            .collect();

        let results = futures::future::join_all(futures).await;
        Ok(results)
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down MCP adapter");
        
        // Process remaining messages in queue
        while !self.message_queue.is_empty() {
            let _ = self.process_queue(10).await;
            sleep(std::time::Duration::from_millis(100)).await;
        }

        // Clear connection and auth
        *self.connection.write() = None;
        *self.auth.write() = None;

        info!("MCP adapter shutdown complete");
        Ok(())
    }
}

impl Default for McpAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// Re-export commonly used types
pub use auth::{AuthHandler, AuthToken, Credentials, GrantType};
pub use client::{McpClient, McpClientConfig};
pub use connection::{Connection, ConnectionPool, ConnectionConfig, PoolStats};
pub use message::{Message, Response, MessageType, MessagePriority, MessageBuilder, ResponseStatus, ResponseError};
pub use queue::{MessageQueue, QueueStats, MultiQueue};

#[cfg(test)]
mod tests {
    pub mod unit;
    
    // Test module available for internal testing
}