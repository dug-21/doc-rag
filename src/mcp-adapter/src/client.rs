//! High-level MCP client with connection management and resilience patterns

use crate::{
    auth::{AuthHandler, AuthToken, Credentials},
    connection::{Connection, ConnectionPool, ConnectionConfig},
    error::{McpError, Result},
    message::{Message, Response},
    queue::{MessageQueue},
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;
use tokio::time::{sleep, timeout};
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;

/// Configuration for the MCP client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpClientConfig {
    pub endpoints: Vec<String>,
    pub max_concurrent_requests: usize,
    pub request_timeout_ms: u64,
    pub connection_timeout_ms: u64,
    pub retry_attempts: usize,
    pub retry_backoff_ms: u64,
    pub health_check_interval_ms: u64,
    pub connection_pool_size: usize,
    pub enable_circuit_breaker: bool,
    pub circuit_breaker_threshold: usize,
    pub auth_refresh_threshold_secs: i64,
}

impl Default for McpClientConfig {
    fn default() -> Self {
        Self {
            endpoints: vec!["http://localhost:8080".to_string()],
            max_concurrent_requests: 100,
            request_timeout_ms: 30000,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            retry_backoff_ms: 1000,
            health_check_interval_ms: 30000,
            connection_pool_size: 10,
            enable_circuit_breaker: true,
            circuit_breaker_threshold: 5,
            auth_refresh_threshold_secs: 300,
        }
    }
}

/// High-level MCP client with advanced features
pub struct McpClient {
    config: McpClientConfig,
    connection_pool: ConnectionPool,
    auth_handler: AuthHandler,
    message_queue: Arc<MessageQueue>,
    request_semaphore: Arc<Semaphore>,
    request_counter: AtomicU64,
    round_robin_counter: AtomicUsize,
    client: reqwest::Client,
    current_token: tokio::sync::RwLock<Option<AuthToken>>,
}

impl McpClient {
    /// Create a new MCP client
    pub fn new(config: McpClientConfig) -> Self {
        let connection_config = ConnectionConfig {
            max_retry_attempts: config.retry_attempts,
            initial_retry_delay: std::time::Duration::from_millis(config.retry_backoff_ms),
            max_retry_delay: std::time::Duration::from_millis(config.retry_backoff_ms * 10),
            health_check_timeout: std::time::Duration::from_millis(config.connection_timeout_ms),
            ping_timeout: std::time::Duration::from_millis(config.connection_timeout_ms / 2),
            heartbeat_interval: std::time::Duration::from_millis(config.health_check_interval_ms),
            max_idle_time: std::time::Duration::from_secs(300),
        };

        let connection_pool = ConnectionPool::with_config(
            config.connection_pool_size, 
            connection_config
        );
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(config.request_timeout_ms))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            config: config.clone(),
            connection_pool,
            auth_handler: AuthHandler::new(),
            message_queue: Arc::new(MessageQueue::new(1000)),
            request_semaphore: Arc::new(Semaphore::new(config.max_concurrent_requests)),
            request_counter: AtomicU64::new(0),
            round_robin_counter: AtomicUsize::new(0),
            client,
            current_token: tokio::sync::RwLock::new(None),
        }
    }

    /// Initialize the client and establish connections
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing MCP client with {} endpoints", self.config.endpoints.len());
        
        // Start health monitoring
        self.connection_pool.start_health_monitoring().await;
        
        // Establish initial connections to all endpoints
        for endpoint in &self.config.endpoints {
            match self.connection_pool.create_connection(endpoint.clone()).await {
                Ok(connection) => {
                    info!("Successfully connected to {}", endpoint);
                }
                Err(e) => {
                    warn!("Failed to connect to {}: {}", endpoint, e);
                }
            }
        }

        // Check if we have at least one healthy connection
        if self.connection_pool.healthy_count().await == 0 {
            return Err(McpError::ConnectionFailed(
                "Failed to establish any connections".to_string()
            ));
        }

        info!("MCP client initialization complete");
        Ok(())
    }

    /// Authenticate with the server
    pub async fn authenticate(&self, credentials: Credentials) -> Result<()> {
        // Get a random endpoint for authentication
        let endpoint = &self.config.endpoints[0]; // Use first endpoint for auth
        
        let token = self.auth_handler.authenticate(&self.client, endpoint, credentials).await?;
        *self.current_token.write().await = Some(token);
        
        info!("Authentication successful");
        Ok(())
    }

    /// Send a message with automatic retry and failover
    #[instrument(skip(self, message), fields(message_id = message.id.unwrap_or(0)))]
    pub async fn send_message(&self, mut message: Message) -> Result<Response> {
        // Acquire semaphore permit for concurrency control
        let _permit = self.request_semaphore.acquire().await
            .map_err(|_| McpError::Internal("Failed to acquire request permit".to_string()))?;

        // Assign message ID
        let message_id = self.request_counter.fetch_add(1, Ordering::Relaxed);
        message.id = Some(message_id);
        message.timestamp = Some(Utc::now());

        debug!("Sending message {} of type {:?}", message_id, message.message_type);

        // Try to send with retry logic
        let mut last_error = None;
        for attempt in 1..=self.config.retry_attempts {
            match self.try_send_message(&message, attempt).await {
                Ok(response) => {
                    debug!("Message {} sent successfully on attempt {}", message_id, attempt);
                    return Ok(response);
                }
                Err(e) => {
                    warn!("Attempt {} failed for message {}: {}", attempt, message_id, e);
                    last_error = Some(e);
                    
                    // Don't retry on certain error types
                    if let Some(ref error) = last_error {
                        if !error.is_retryable() {
                            break;
                        }
                    }

                    // Wait before retry (exponential backoff)
                    if attempt < self.config.retry_attempts {
                        let delay_ms = self.config.retry_backoff_ms * (1 << (attempt - 1));
                        sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| McpError::Internal("No error details".to_string())))
    }

    /// Try to send a message once
    async fn try_send_message(&self, message: &Message, attempt: usize) -> Result<Response> {
        // Ensure we have a valid token
        self.ensure_valid_token().await?;

        // Get the best available connection
        let connection = self.get_connection().await
            .ok_or_else(|| McpError::ConnectionFailed("No healthy connections available".to_string()))?;

        // Send the actual request
        self.send_request_to_connection(&connection, message).await
    }

    /// Ensure we have a valid authentication token
    async fn ensure_valid_token(&self) -> Result<()> {
        let mut token_guard = self.current_token.write().await;
        
        if let Some(ref mut token) = token_guard.as_mut() {
            if token.needs_refresh(self.config.auth_refresh_threshold_secs) {
                if token.can_refresh() {
                    let endpoint = &self.config.endpoints[0]; // Use first endpoint for refresh
                    let refreshed = self.auth_handler.refresh_token(&self.client, endpoint, token).await?;
                    **token = refreshed;
                    info!("Token refreshed successfully");
                } else {
                    return Err(McpError::AuthenticationFailed("Token expired and cannot be refreshed".to_string()));
                }
            }
        } else {
            return Err(McpError::AuthenticationFailed("No authentication token available".to_string()));
        }

        Ok(())
    }

    /// Get a connection using round-robin selection
    async fn get_connection(&self) -> Option<Connection> {
        self.connection_pool.get_best_connection().await
            .or_else(|| {
                // Fallback to any healthy connection
                futures::executor::block_on(async {
                    self.connection_pool.get_healthy_connection().await
                })
            })
    }

    /// Send request to a specific connection
    async fn send_request_to_connection(&self, connection: &Connection, message: &Message) -> Result<Response> {
        let token_guard = self.current_token.read().await;
        let token = token_guard.as_ref()
            .ok_or_else(|| McpError::AuthenticationFailed("No token available".to_string()))?;

        let url = format!("{}/api/messages", connection.endpoint);
        let start_time = Instant::now();

        let response = timeout(
            std::time::Duration::from_millis(self.config.request_timeout_ms),
            self.client
                .post(&url)
                .header("Authorization", format!("Bearer {}", token.access_token))
                .header("Content-Type", "application/json")
                .json(message)
                .send()
        ).await
        .map_err(|_| McpError::MessageTimeout { timeout_ms: self.config.request_timeout_ms })?
        .map_err(McpError::NetworkError)?;

        // Update connection metrics
        let latency = start_time.elapsed().as_millis() as u64;
        connection.update_latency(latency);
        connection.record_sent(
            serde_json::to_vec(message).unwrap_or_default().len() as u64,
            1
        );

        // Handle response
        match response.status() {
            status if status.is_success() => {
                let response_body = response.bytes().await.map_err(McpError::NetworkError)?;
                connection.record_received(response_body.len() as u64, 1);
                
                let response_data: Response = serde_json::from_slice(&response_body)
                    .map_err(McpError::SerializationError)?;
                
                connection.update_heartbeat();
                Ok(response_data)
            }
            reqwest::StatusCode::UNAUTHORIZED => {
                Err(McpError::AuthenticationFailed("Token expired or invalid".to_string()))
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => {
                Err(McpError::RateLimitExceeded)
            }
            status => {
                let error_text = response.text().await.unwrap_or_default();
                connection.mark_unhealthy();
                Err(McpError::Internal(format!("Request failed with status {}: {}", status, error_text)))
            }
        }
    }

    /// Get client statistics
    pub async fn get_stats(&self) -> ClientStats {
        let pool_stats = self.connection_pool.pool_stats().await;
        let queue_stats = self.message_queue.stats();
        
        ClientStats {
            total_requests: self.request_counter.load(Ordering::Relaxed),
            active_connections: pool_stats.healthy_connections,
            total_connections: pool_stats.total_connections,
            queued_messages: queue_stats.current_size,
            pool_stats,
        }
    }

    /// Perform health check on all connections
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let connections = self.connection_pool.get_all_connections().await;
        let mut healthy_count = 0;
        let mut unhealthy_endpoints = Vec::new();

        for connection in &connections {
            match connection.perform_health_check(&self.client, &ConnectionConfig::default()).await {
                Ok(_) => {
                    healthy_count += 1;
                }
                Err(_) => {
                    unhealthy_endpoints.push(connection.endpoint.clone());
                }
            }
        }

        Ok(HealthStatus {
            healthy_connections: healthy_count,
            total_connections: connections.len(),
            unhealthy_endpoints,
            last_check: Utc::now(),
        })
    }

    /// Graceful shutdown
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down MCP client");
        
        // Process remaining messages
        // This would be more complex in a real implementation
        
        // Shutdown connection pool
        self.connection_pool.shutdown().await?;
        
        info!("MCP client shutdown complete");
        Ok(())
    }
}

/// Client statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct ClientStats {
    pub total_requests: u64,
    pub active_connections: usize,
    pub total_connections: usize,
    pub queued_messages: usize,
    pub pool_stats: crate::connection::PoolStats,
}

/// Health check status
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub healthy_connections: usize,
    pub total_connections: usize,
    pub unhealthy_endpoints: Vec<String>,
    pub last_check: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_client_creation() {
        let config = McpClientConfig::default();
        let client = McpClient::new(config);
        
        let stats = client.get_stats().await;
        assert_eq!(stats.total_requests, 0);
    }

    #[tokio::test] 
    async fn test_client_config_defaults() {
        let config = McpClientConfig::default();
        assert_eq!(config.max_concurrent_requests, 100);
        assert_eq!(config.retry_attempts, 3);
        assert!(config.request_timeout_ms > 0);
    }
}