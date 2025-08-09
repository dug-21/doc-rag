//! Robust connection management with retry logic, health checks, and monitoring

use chrono::{DateTime, Duration, Utc};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tokio::time::{sleep, timeout, Instant};
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use url::Url;

use crate::{McpError, Result};

/// Represents an active connection to an MCP server with comprehensive monitoring
#[derive(Debug, Clone)]
pub struct Connection {
    pub id: Uuid,
    pub endpoint: String,
    pub connected_at: DateTime<Utc>,
    pub last_heartbeat: Arc<AtomicU64>,
    pub is_healthy: Arc<AtomicBool>,
    pub bytes_sent: Arc<AtomicU64>,
    pub bytes_received: Arc<AtomicU64>,
    pub messages_sent: Arc<AtomicU64>,
    pub messages_received: Arc<AtomicU64>,
    /// Connection state
    pub state: Arc<AtomicU64>, // 0=disconnected, 1=connecting, 2=connected, 3=error
    /// Last error message
    pub last_error: Arc<RwLock<Option<String>>>,
    /// Connection latency in milliseconds
    pub latency_ms: Arc<AtomicU64>,
    /// Failed connection attempts
    pub failed_attempts: Arc<AtomicU64>,
    /// Last successful operation timestamp
    pub last_success: Arc<AtomicU64>,
}

/// Connection state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    Disconnected = 0,
    Connecting = 1,
    Connected = 2,
    Error = 3,
}

impl From<u64> for ConnectionState {
    fn from(value: u64) -> Self {
        match value {
            1 => ConnectionState::Connecting,
            2 => ConnectionState::Connected,
            3 => ConnectionState::Error,
            _ => ConnectionState::Disconnected,
        }
    }
}

impl From<ConnectionState> for u64 {
    fn from(state: ConnectionState) -> Self {
        state as u64
    }
}

impl Connection {
    /// Create a new connection instance
    pub fn new(id: Uuid, endpoint: String, connected_at: DateTime<Utc>) -> Self {
        let now_ts = connected_at.timestamp() as u64;
        Self {
            id,
            endpoint,
            connected_at,
            last_heartbeat: Arc::new(AtomicU64::new(now_ts)),
            is_healthy: Arc::new(AtomicBool::new(false)), // Start as unhealthy until verified
            bytes_sent: Arc::new(AtomicU64::new(0)),
            bytes_received: Arc::new(AtomicU64::new(0)),
            messages_sent: Arc::new(AtomicU64::new(0)),
            messages_received: Arc::new(AtomicU64::new(0)),
            state: Arc::new(AtomicU64::new(ConnectionState::Connecting.into())),
            last_error: Arc::new(RwLock::new(None)),
            latency_ms: Arc::new(AtomicU64::new(0)),
            failed_attempts: Arc::new(AtomicU64::new(0)),
            last_success: Arc::new(AtomicU64::new(now_ts)),
        }
    }

    /// Create a connection and immediately establish it
    pub async fn establish(client: &Client, endpoint: String, config: &ConnectionConfig) -> Result<Self> {
        let id = Uuid::new_v4();
        let connected_at = Utc::now();
        let connection = Self::new(id, endpoint.clone(), connected_at);
        
        // Validate and parse endpoint
        let url = Url::parse(&endpoint)
            .map_err(|e| McpError::ConnectionFailed(format!("Invalid endpoint URL: {}", e)))?;
        
        debug!("Establishing connection to {}", endpoint);
        connection.set_state(ConnectionState::Connecting).await;
        
        match connection.perform_health_check(client, config).await {
            Ok(_) => {
                connection.mark_healthy();
                connection.set_state(ConnectionState::Connected).await;
                info!("Successfully established connection to {}", endpoint);
                Ok(connection)
            }
            Err(e) => {
                connection.mark_unhealthy();
                connection.set_state(ConnectionState::Error).await;
                connection.set_last_error(&format!("Health check failed: {}", e)).await;
                Err(e)
            }
        }
    }

    /// Update heartbeat timestamp
    pub fn update_heartbeat(&self) {
        let now = Utc::now().timestamp() as u64;
        self.last_heartbeat.store(now, Ordering::Relaxed);
        self.last_success.store(now, Ordering::Relaxed);
    }

    /// Check if connection is healthy based on last heartbeat
    pub fn is_healthy(&self, max_age_secs: u64) -> bool {
        if !self.is_healthy.load(Ordering::Relaxed) {
            return false;
        }

        let now = Utc::now().timestamp() as u64;
        let last_heartbeat = self.last_heartbeat.load(Ordering::Relaxed);
        let age = now.saturating_sub(last_heartbeat);
        
        age <= max_age_secs
    }

    /// Mark connection as unhealthy
    pub fn mark_unhealthy(&self) {
        self.is_healthy.store(false, Ordering::Relaxed);
        self.failed_attempts.fetch_add(1, Ordering::Relaxed);
        debug!("Connection {} marked unhealthy", self.id);
    }

    /// Mark connection as healthy
    pub fn mark_healthy(&self) {
        self.is_healthy.store(true, Ordering::Relaxed);
        self.failed_attempts.store(0, Ordering::Relaxed); // Reset failure count
        self.update_heartbeat();
        debug!("Connection {} marked healthy", self.id);
    }

    /// Get current connection state
    pub fn get_state(&self) -> ConnectionState {
        self.state.load(Ordering::Relaxed).into()
    }

    /// Set connection state
    pub async fn set_state(&self, new_state: ConnectionState) {
        self.state.store(new_state.into(), Ordering::Relaxed);
        debug!("Connection {} state changed to {:?}", self.id, new_state);
    }

    /// Set last error message
    pub async fn set_last_error(&self, error: &str) {
        *self.last_error.write().await = Some(error.to_string());
        warn!("Connection {} error: {}", self.id, error);
    }

    /// Get last error message
    pub async fn get_last_error(&self) -> Option<String> {
        self.last_error.read().await.clone()
    }

    /// Update latency measurement
    pub fn update_latency(&self, latency_ms: u64) {
        self.latency_ms.store(latency_ms, Ordering::Relaxed);
    }

    /// Get current latency
    pub fn get_latency(&self) -> u64 {
        self.latency_ms.load(Ordering::Relaxed)
    }

    /// Get failed attempts count
    pub fn get_failed_attempts(&self) -> u64 {
        self.failed_attempts.load(Ordering::Relaxed)
    }

    /// Record sent data
    pub fn record_sent(&self, bytes: u64, messages: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
        self.messages_sent.fetch_add(messages, Ordering::Relaxed);
    }

    /// Record received data
    pub fn record_received(&self, bytes: u64, messages: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
        self.messages_received.fetch_add(messages, Ordering::Relaxed);
    }

    /// Get connection statistics
    pub async fn stats(&self) -> ConnectionStats {
        ConnectionStats {
            id: self.id,
            endpoint: self.endpoint.clone(),
            connected_at: self.connected_at,
            uptime_secs: (Utc::now() - self.connected_at).num_seconds() as u64,
            last_heartbeat_secs_ago: {
                let now = Utc::now().timestamp() as u64;
                let last_heartbeat = self.last_heartbeat.load(Ordering::Relaxed);
                now.saturating_sub(last_heartbeat)
            },
            is_healthy: self.is_healthy.load(Ordering::Relaxed),
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            state: format!("{:?}", self.get_state()),
            latency_ms: self.get_latency(),
            failed_attempts: self.get_failed_attempts(),
            last_error: self.get_last_error().await,
        }
    }

    /// Calculate throughput metrics
    pub fn throughput_metrics(&self) -> ThroughputMetrics {
        let uptime_secs = (Utc::now() - self.connected_at).num_seconds() as u64;
        let uptime_secs = uptime_secs.max(1); // Avoid division by zero

        let bytes_sent = self.bytes_sent.load(Ordering::Relaxed);
        let bytes_received = self.bytes_received.load(Ordering::Relaxed);
        let messages_sent = self.messages_sent.load(Ordering::Relaxed);
        let messages_received = self.messages_received.load(Ordering::Relaxed);

        ThroughputMetrics {
            bytes_per_sec_sent: bytes_sent / uptime_secs,
            bytes_per_sec_received: bytes_received / uptime_secs,
            messages_per_sec_sent: messages_sent / uptime_secs,
            messages_per_sec_received: messages_received / uptime_secs,
            total_bytes: bytes_sent + bytes_received,
            total_messages: messages_sent + messages_received,
            latency_ms: self.get_latency(),
            failed_attempts: self.get_failed_attempts(),
        }
    }

    /// Perform health check
    pub async fn perform_health_check(&self, client: &Client, config: &ConnectionConfig) -> Result<()> {
        let health_url = format!("{}/health", self.endpoint.trim_end_matches('/'));
        let start_time = Instant::now();
        
        debug!("Performing health check on {}", health_url);
        
        let response = timeout(
            config.health_check_timeout,
            client.get(&health_url).send(),
        )
        .await
        .map_err(|_| McpError::MessageTimeout { 
            timeout_ms: config.health_check_timeout.as_millis() as u64 
        })?;
        
        match response {
            Ok(resp) => {
                let latency = start_time.elapsed().as_millis() as u64;
                self.update_latency(latency);
                
                if resp.status().is_success() {
                    self.update_heartbeat();
                    debug!("Health check passed for {} ({}ms)", self.endpoint, latency);
                    Ok(())
                } else {
                    let error = format!("Health check failed with status: {}", resp.status());
                    self.set_last_error(&error).await;
                    Err(McpError::ConnectionFailed(error))
                }
            }
            Err(e) => {
                let error = format!("Health check request failed: {}", e);
                self.set_last_error(&error).await;
                Err(McpError::NetworkError(e))
            }
        }
    }

    /// Attempt to reconnect with exponential backoff
    pub async fn reconnect(&self, client: &Client, config: &ConnectionConfig) -> Result<()> {
        let mut attempt = 0;
        let mut delay = config.initial_retry_delay;
        
        while attempt < config.max_retry_attempts {
            attempt += 1;
            info!("Reconnection attempt {}/{} for {}", attempt, config.max_retry_attempts, self.endpoint);
            
            self.set_state(ConnectionState::Connecting).await;
            
            match self.perform_health_check(client, config).await {
                Ok(_) => {
                    self.mark_healthy();
                    self.set_state(ConnectionState::Connected).await;
                    info!("Successfully reconnected to {} after {} attempts", self.endpoint, attempt);
                    return Ok(());
                }
                Err(e) => {
                    warn!("Reconnection attempt {} failed: {}", attempt, e);
                    self.mark_unhealthy();
                    
                    if attempt < config.max_retry_attempts {
                        sleep(delay).await;
                        delay = std::cmp::min(delay * 2, config.max_retry_delay);
                    }
                }
            }
        }
        
        self.set_state(ConnectionState::Error).await;
        let error = format!("Failed to reconnect after {} attempts", config.max_retry_attempts);
        self.set_last_error(&error).await;
        Err(McpError::ConnectionFailed(error))
    }

    /// Send a ping to keep connection alive
    pub async fn ping(&self, client: &Client, config: &ConnectionConfig) -> Result<u64> {
        let ping_url = format!("{}/ping", self.endpoint.trim_end_matches('/'));
        let start_time = Instant::now();
        
        let response = timeout(
            config.ping_timeout,
            client.get(&ping_url).send(),
        )
        .await
        .map_err(|_| McpError::MessageTimeout { 
            timeout_ms: config.ping_timeout.as_millis() as u64 
        })?;
        
        match response {
            Ok(resp) => {
                let latency = start_time.elapsed().as_millis() as u64;
                
                if resp.status().is_success() {
                    self.update_latency(latency);
                    self.update_heartbeat();
                    debug!("Ping successful for {} ({}ms)", self.endpoint, latency);
                    Ok(latency)
                } else {
                    let error = format!("Ping failed with status: {}", resp.status());
                    self.set_last_error(&error).await;
                    Err(McpError::ConnectionFailed(error))
                }
            }
            Err(e) => {
                let error = format!("Ping request failed: {}", e);
                self.set_last_error(&error).await;
                Err(McpError::NetworkError(e))
            }
        }
    }
}

/// Connection statistics snapshot
#[derive(Debug, Serialize, Deserialize)]
pub struct ConnectionStats {
    pub id: Uuid,
    pub endpoint: String,
    pub connected_at: DateTime<Utc>,
    pub uptime_secs: u64,
    pub last_heartbeat_secs_ago: u64,
    pub is_healthy: bool,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub state: String,
    pub latency_ms: u64,
    pub failed_attempts: u64,
    pub last_error: Option<String>,
}

/// Throughput metrics for the connection
#[derive(Debug, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    pub bytes_per_sec_sent: u64,
    pub bytes_per_sec_received: u64,
    pub messages_per_sec_sent: u64,
    pub messages_per_sec_received: u64,
    pub total_bytes: u64,
    pub total_messages: u64,
    pub latency_ms: u64,
    pub failed_attempts: u64,
}

/// Configuration for connection management
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    pub max_retry_attempts: usize,
    pub initial_retry_delay: std::time::Duration,
    pub max_retry_delay: std::time::Duration,
    pub health_check_timeout: std::time::Duration,
    pub ping_timeout: std::time::Duration,
    pub heartbeat_interval: std::time::Duration,
    pub max_idle_time: std::time::Duration,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            max_retry_attempts: 3,
            initial_retry_delay: std::time::Duration::from_millis(1000),
            max_retry_delay: std::time::Duration::from_millis(30000),
            health_check_timeout: std::time::Duration::from_millis(5000),
            ping_timeout: std::time::Duration::from_millis(3000),
            heartbeat_interval: std::time::Duration::from_secs(30),
            max_idle_time: std::time::Duration::from_secs(300),
        }
    }
}

/// Advanced connection pool for managing multiple connections with health monitoring
#[derive(Debug)]
pub struct ConnectionPool {
    connections: Arc<RwLock<Vec<Connection>>>,
    max_connections: usize,
    config: ConnectionConfig,
    client: Client,
    /// Semaphore to limit concurrent operations
    operation_semaphore: Arc<Semaphore>,
    /// Health check interval task handle
    health_check_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl ConnectionPool {
    /// Create a new connection pool
    pub fn new(max_connections: usize) -> Self {
        Self::with_config(max_connections, ConnectionConfig::default())
    }

    /// Create a new connection pool with custom configuration
    pub fn with_config(max_connections: usize, config: ConnectionConfig) -> Self {
        let client = Client::builder()
            .timeout(config.health_check_timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            connections: Arc::new(RwLock::new(Vec::new())),
            max_connections,
            config,
            client,
            operation_semaphore: Arc::new(Semaphore::new(max_connections)),
            health_check_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Start background health checking
    pub async fn start_health_monitoring(&self) {
        let connections = Arc::clone(&self.connections);
        let config = self.config.clone();
        let client = self.client.clone();
        let interval = config.heartbeat_interval;

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            
            loop {
                interval.tick().await;
                
                let connections_copy = {
                    let conn_guard = connections.read().await;
                    conn_guard.clone()
                };
                
                for connection in connections_copy {
                    if connection.get_state() == ConnectionState::Connected {
                        if let Err(e) = connection.perform_health_check(&client, &config).await {
                            warn!("Health check failed for {}: {}", connection.endpoint, e);
                            connection.mark_unhealthy();
                            connection.set_state(ConnectionState::Error).await;
                        }
                    }
                }
            }
        });

        *self.health_check_handle.write().await = Some(handle);
        info!("Started health monitoring for connection pool");
    }

    /// Stop background health checking
    pub async fn stop_health_monitoring(&self) {
        if let Some(handle) = self.health_check_handle.write().await.take() {
            handle.abort();
            info!("Stopped health monitoring for connection pool");
        }
    }

    /// Add a connection to the pool
    pub async fn add_connection(&self, connection: Connection) -> Result<()> {
        let mut connections = self.connections.write().await;
        
        if connections.len() >= self.max_connections {
            return Err(McpError::ResourceExhausted("Connection pool is full".to_string()));
        }

        info!("Adding connection {} to pool", connection.id);
        connections.push(connection);
        Ok(())
    }

    /// Create and add a new connection
    pub async fn create_connection(&self, endpoint: String) -> Result<Connection> {
        // Acquire permit to limit concurrent operations
        let _permit = self.operation_semaphore.acquire().await
            .map_err(|_| McpError::Internal("Failed to acquire operation permit".to_string()))?;
        
        let connection = Connection::establish(&self.client, endpoint, &self.config).await?;
        self.add_connection(connection.clone()).await?;
        
        Ok(connection)
    }

    /// Remove a connection from the pool
    pub async fn remove_connection(&self, connection_id: Uuid) -> Option<Connection> {
        let mut connections = self.connections.write().await;
        
        if let Some(pos) = connections.iter().position(|c| c.id == connection_id) {
            let connection = connections.remove(pos);
            info!("Removed connection {} from pool", connection_id);
            Some(connection)
        } else {
            None
        }
    }

    /// Get a healthy connection from the pool (round-robin selection)
    pub async fn get_healthy_connection(&self) -> Option<Connection> {
        let connections = self.connections.read().await;
        let max_age_secs = self.config.max_idle_time.as_secs();
        
        // Find healthy connections
        let healthy_connections: Vec<_> = connections
            .iter()
            .filter(|c| {
                c.is_healthy(max_age_secs) && 
                c.get_state() == ConnectionState::Connected
            })
            .collect();
        
        if healthy_connections.is_empty() {
            return None;
        }
        
        // Simple round-robin selection based on current time
        let index = (Utc::now().timestamp() as usize) % healthy_connections.len();
        Some(healthy_connections[index].clone())
    }

    /// Get the best connection based on latency and load
    pub async fn get_best_connection(&self) -> Option<Connection> {
        let connections = self.connections.read().await;
        let max_age_secs = self.config.max_idle_time.as_secs();
        
        connections
            .iter()
            .filter(|c| {
                c.is_healthy(max_age_secs) && 
                c.get_state() == ConnectionState::Connected
            })
            .min_by_key(|c| {
                // Score based on latency and failed attempts
                let latency = c.get_latency();
                let failures = c.get_failed_attempts();
                latency + (failures * 100) // Penalize connections with failures
            })
            .cloned()
    }

    /// Get all connections
    pub async fn get_all_connections(&self) -> Vec<Connection> {
        self.connections.read().await.clone()
    }

    /// Get healthy connections count
    pub async fn healthy_count(&self) -> usize {
        let connections = self.connections.read().await;
        let max_age_secs = self.config.max_idle_time.as_secs();
        connections
            .iter()
            .filter(|c| {
                c.is_healthy(max_age_secs) && 
                c.get_state() == ConnectionState::Connected
            })
            .count()
    }

    /// Cleanup unhealthy connections
    pub async fn cleanup_unhealthy(&self) -> usize {
        let mut connections = self.connections.write().await;
        let initial_count = connections.len();
        let max_age_secs = self.config.max_idle_time.as_secs();
        
        connections.retain(|c| {
            let is_healthy = c.is_healthy(max_age_secs) && 
                           c.get_state() != ConnectionState::Error;
            
            if !is_healthy {
                debug!("Removing unhealthy connection {}", c.id);
            }
            
            is_healthy
        });
        
        let removed_count = initial_count - connections.len();
        if removed_count > 0 {
            info!("Cleaned up {} unhealthy connections", removed_count);
        }
        
        removed_count
    }

    /// Attempt to reconnect failed connections
    pub async fn reconnect_failed(&self) -> usize {
        let connections = self.get_all_connections().await;
        let mut reconnected = 0;
        
        for connection in connections {
            if connection.get_state() == ConnectionState::Error {
                match connection.reconnect(&self.client, &self.config).await {
                    Ok(_) => {
                        reconnected += 1;
                        info!("Successfully reconnected {}", connection.endpoint);
                    }
                    Err(e) => {
                        debug!("Failed to reconnect {}: {}", connection.endpoint, e);
                    }
                }
            }
        }
        
        reconnected
    }

    /// Get comprehensive pool statistics
    pub async fn pool_stats(&self) -> PoolStats {
        let connections = self.connections.read().await;
        let total_connections = connections.len();
        let max_age_secs = self.config.max_idle_time.as_secs();
        
        let healthy_connections = connections.iter()
            .filter(|c| c.is_healthy(max_age_secs))
            .count();
        
        let connected_connections = connections.iter()
            .filter(|c| c.get_state() == ConnectionState::Connected)
            .count();
        
        let error_connections = connections.iter()
            .filter(|c| c.get_state() == ConnectionState::Error)
            .count();
        
        let (total_bytes_sent, total_bytes_received, total_messages_sent, total_messages_received, total_latency, total_failures) = 
            connections.iter().fold((0u64, 0u64, 0u64, 0u64, 0u64, 0u64), |acc, conn| {
                (
                    acc.0 + conn.bytes_sent.load(Ordering::Relaxed),
                    acc.1 + conn.bytes_received.load(Ordering::Relaxed),
                    acc.2 + conn.messages_sent.load(Ordering::Relaxed),
                    acc.3 + conn.messages_received.load(Ordering::Relaxed),
                    acc.4 + conn.get_latency(),
                    acc.5 + conn.get_failed_attempts(),
                )
            });
            
        let avg_latency = if total_connections > 0 {
            total_latency / total_connections as u64
        } else {
            0
        };

        PoolStats {
            total_connections,
            healthy_connections,
            connected_connections,
            error_connections,
            max_connections: self.max_connections,
            total_bytes_sent,
            total_bytes_received,
            total_messages_sent,
            total_messages_received,
            average_latency_ms: avg_latency,
            total_failed_attempts: total_failures,
        }
    }

    /// Get detailed connection information
    pub async fn get_connection_details(&self) -> Vec<ConnectionStats> {
        let connections = self.connections.read().await;
        let mut details = Vec::new();
        
        for connection in connections.iter() {
            details.push(connection.stats().await);
        }
        
        details
    }

    /// Graceful shutdown of the pool
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down connection pool");
        
        // Stop health monitoring
        self.stop_health_monitoring().await;
        
        // Clear all connections
        let mut connections = self.connections.write().await;
        for connection in connections.iter() {
            connection.set_state(ConnectionState::Disconnected).await;
        }
        connections.clear();
        
        info!("Connection pool shutdown complete");
        Ok(())
    }
}

/// Pool statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct PoolStats {
    pub total_connections: usize,
    pub healthy_connections: usize,
    pub connected_connections: usize,
    pub error_connections: usize,
    pub max_connections: usize,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub total_messages_sent: u64,
    pub total_messages_received: u64,
    pub average_latency_ms: u64,
    pub total_failed_attempts: u64,
}

/// Connection manager for handling multiple endpoints
/// Note: This is a simplified implementation for demonstration purposes
/*
#[derive(Debug)]
pub struct ConnectionManager {
    pools: Arc<RwLock<std::collections::HashMap<String, Arc<ConnectionPool>>>>,
    default_config: ConnectionConfig,
}

impl ConnectionManager {
    /// Create a new connection manager  
    pub fn new(config: ConnectionConfig) -> Self {
        Self {
            pools: Arc::new(RwLock::new(std::collections::HashMap::new())),
            default_config: config,
        }
    }
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;
    use std::time::Duration;

    #[test]
    fn test_connection_creation() {
        let id = Uuid::new_v4();
        let endpoint = "http://test.com".to_string();
        let now = Utc::now();
        
        let conn = Connection::new(id, endpoint.clone(), now);
        
        assert_eq!(conn.id, id);
        assert_eq!(conn.endpoint, endpoint);
        assert_eq!(conn.connected_at, now);
        // Should start unhealthy until verified
        assert!(!conn.is_healthy(60));
        assert_eq!(conn.get_state(), ConnectionState::Connecting);
    }

    #[tokio::test]
    async fn test_connection_state_management() {
        let conn = Connection::new(Uuid::new_v4(), "test".to_string(), Utc::now());
        
        assert_eq!(conn.get_state(), ConnectionState::Connecting);
        
        conn.set_state(ConnectionState::Connected).await;
        assert_eq!(conn.get_state(), ConnectionState::Connected);
        
        conn.set_state(ConnectionState::Error).await;
        assert_eq!(conn.get_state(), ConnectionState::Error);
    }

    #[test]
    fn test_connection_health() {
        let conn = Connection::new(Uuid::new_v4(), "test".to_string(), Utc::now());
        
        // Should be unhealthy initially
        assert!(!conn.is_healthy(60));
        
        // Mark healthy
        conn.mark_healthy();
        assert!(conn.is_healthy(60));
        assert_eq!(conn.get_failed_attempts(), 0);
        
        // Mark unhealthy
        conn.mark_unhealthy();
        assert!(!conn.is_healthy(60));
        assert_eq!(conn.get_failed_attempts(), 1);
        
        // Mark healthy again - should reset failures
        conn.mark_healthy();
        assert!(conn.is_healthy(60));
        assert_eq!(conn.get_failed_attempts(), 0);
    }

    #[tokio::test]
    async fn test_connection_stats() {
        let conn = Connection::new(Uuid::new_v4(), "test".to_string(), Utc::now());
        
        conn.record_sent(100, 5);
        conn.record_received(200, 10);
        conn.update_latency(50);
        
        let stats = conn.stats().await;
        assert_eq!(stats.bytes_sent, 100);
        assert_eq!(stats.bytes_received, 200);
        assert_eq!(stats.messages_sent, 5);
        assert_eq!(stats.messages_received, 10);
        assert_eq!(stats.latency_ms, 50);
    }

    #[test]
    fn test_throughput_metrics() {
        let conn = Connection::new(Uuid::new_v4(), "test".to_string(), Utc::now());
        
        conn.record_sent(1000, 10);
        conn.record_received(2000, 20);
        conn.update_latency(100);
        
        let metrics = conn.throughput_metrics();
        assert_eq!(metrics.total_bytes, 3000);
        assert_eq!(metrics.total_messages, 30);
        assert_eq!(metrics.latency_ms, 100);
        assert!(metrics.bytes_per_sec_sent > 0);
        assert!(metrics.messages_per_sec_sent > 0);
    }

    #[tokio::test]
    async fn test_connection_pool_basic() {
        let pool = ConnectionPool::new(2);
        let conn1 = Connection::new(Uuid::new_v4(), "test1".to_string(), Utc::now());
        let conn2 = Connection::new(Uuid::new_v4(), "test2".to_string(), Utc::now());
        let conn3 = Connection::new(Uuid::new_v4(), "test3".to_string(), Utc::now());
        
        // Add connections
        assert!(pool.add_connection(conn1.clone()).await.is_ok());
        assert!(pool.add_connection(conn2.clone()).await.is_ok());
        assert!(pool.add_connection(conn3).await.is_err()); // Should fail, pool is full
        
        // Check pool stats
        let stats = pool.pool_stats().await;
        assert_eq!(stats.total_connections, 2);
        assert_eq!(stats.max_connections, 2);
        
        // Remove connection
        let removed = pool.remove_connection(conn1.id).await;
        assert!(removed.is_some());
        assert_eq!(pool.pool_stats().await.total_connections, 1);
    }

    #[tokio::test]
    async fn test_connection_pool_health_management() {
        let pool = ConnectionPool::new(3);
        
        let conn1 = Connection::new(Uuid::new_v4(), "test1".to_string(), Utc::now());
        let conn2 = Connection::new(Uuid::new_v4(), "test2".to_string(), Utc::now());
        let conn3 = Connection::new(Uuid::new_v4(), "test3".to_string(), Utc::now());
        
        // Mark some healthy, some not
        conn1.mark_healthy();
        conn1.set_state(ConnectionState::Connected).await;
        
        conn2.mark_unhealthy();
        conn2.set_state(ConnectionState::Error).await;
        
        conn3.mark_healthy();
        conn3.set_state(ConnectionState::Connected).await;
        
        pool.add_connection(conn1).await.unwrap();
        pool.add_connection(conn2).await.unwrap();
        pool.add_connection(conn3).await.unwrap();
        
        // Should have 2 healthy connections
        assert_eq!(pool.healthy_count().await, 2);
        
        // Clean up unhealthy connections
        let removed = pool.cleanup_unhealthy().await;
        assert_eq!(removed, 1);
        assert_eq!(pool.pool_stats().await.total_connections, 2);
    }

    #[test]
    fn test_connection_config() {
        let config = ConnectionConfig::default();
        assert_eq!(config.max_retry_attempts, 3);
        assert_eq!(config.initial_retry_delay, Duration::from_millis(1000));
        assert!(config.health_check_timeout > Duration::from_secs(0));
    }

    #[tokio::test]
    async fn test_connection_error_handling() {
        let conn = Connection::new(Uuid::new_v4(), "test".to_string(), Utc::now());
        
        conn.set_last_error("Test error").await;
        let error = conn.get_last_error().await;
        assert_eq!(error, Some("Test error".to_string()));
        
        assert_eq!(conn.get_failed_attempts(), 0);
        conn.mark_unhealthy();
        assert_eq!(conn.get_failed_attempts(), 1);
    }

    // #[tokio::test]
    // async fn test_connection_manager() {
    //     let config = ConnectionConfig::default();
    //     let manager = ConnectionManager::new(config);
    //     
    //     let stats = manager.get_all_stats().await;
    //     assert!(stats.is_empty());
    //     
    //     // This test would need actual pool implementation to be complete
    //     // The current implementation has some issues with Arc usage
    // }
}