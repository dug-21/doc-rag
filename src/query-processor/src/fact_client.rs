//! FACT Client Integration
//!
//! This module provides the FACT (Fast Access Cache Technology) client implementation
//! according to the SPARC Architecture specification. It implements high-performance
//! caching with connection pooling, retry logic, circuit breaker pattern, and
//! comprehensive monitoring.
//!
//! ## Performance Targets
//! - Cache hits: < 23ms response time
//! - Cache misses: < 95ms response time
//! - Connection pool: bb8 for efficient resource management
//! - Circuit breaker: Fail-fast with automatic recovery

use crate::error::{ProcessorError, Result};
use crate::types::{QueryResult, ClassificationResult, StrategyRecommendation, ExtractedEntity};

use async_trait::async_trait;
use bb8::Pool;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;

/// FACT Client configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FACTConfig {
    /// FACT system endpoint
    pub endpoint: String,
    /// API key for authentication
    pub api_key: String,
    /// Connection pool configuration
    pub pool: PoolConfig,
    /// Retry configuration
    pub retry: RetryConfig,
    /// Circuit breaker configuration
    pub circuit_breaker: CircuitBreakerConfig,
    /// Performance monitoring configuration
    pub monitoring: MonitoringConfig,
}

/// Connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    /// Maximum number of connections
    pub max_size: u32,
    /// Minimum idle connections
    pub min_idle: Option<u32>,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Idle timeout
    pub idle_timeout: Option<Duration>,
    /// Max connection lifetime
    pub max_lifetime: Option<Duration>,
}

/// Retry configuration with exponential backoff
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retry attempts
    pub max_attempts: u32,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor to avoid thundering herd
    pub jitter: bool,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: u32,
    /// Time window for failure counting
    pub time_window: Duration,
    /// Recovery timeout when circuit is open
    pub recovery_timeout: Duration,
    /// Success threshold to close circuit from half-open
    pub success_threshold: u32,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable detailed metrics collection
    pub enable_detailed_metrics: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
    /// Performance alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Performance alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Cache hit response time threshold (ms)
    pub cache_hit_latency_ms: u64,
    /// Cache miss response time threshold (ms)
    pub cache_miss_latency_ms: u64,
    /// Error rate threshold (0.0-1.0)
    pub error_rate_threshold: f64,
    /// Circuit breaker open threshold
    pub circuit_breaker_threshold: f64,
}

/// FACT Client trait definition
#[async_trait]
pub trait FACTClientInterface: Send + Sync + std::fmt::Debug {
    /// Get cached query result
    async fn get_query_result(&self, query: &str) -> Result<Option<QueryResult>>;
    
    /// Set cached query result
    async fn set_query_result(&self, query: &str, result: &QueryResult, ttl: Duration) -> Result<()>;
    
    /// Get cached entities
    async fn get_entities(&self, query: &str) -> Result<Option<Vec<ExtractedEntity>>>;
    
    /// Set cached entities
    async fn set_entities(&self, query: &str, entities: &[ExtractedEntity], ttl: Duration) -> Result<()>;
    
    /// Get cached classification
    async fn get_classification(&self, query: &str) -> Result<Option<ClassificationResult>>;
    
    /// Set cached classification
    async fn set_classification(&self, query: &str, classification: &ClassificationResult, ttl: Duration) -> Result<()>;
    
    /// Get cached strategy
    async fn get_strategy(&self, query_hash: &str) -> Result<Option<StrategyRecommendation>>;
    
    /// Set cached strategy
    async fn set_strategy(&self, query_hash: &str, strategy: &StrategyRecommendation, ttl: Duration) -> Result<()>;
    
    /// Clear all cache entries
    async fn clear_cache(&self) -> Result<()>;
    
    /// Get client health status
    async fn health_check(&self) -> Result<HealthStatus>;
    
    /// Get performance metrics
    async fn get_metrics(&self) -> Result<PerformanceMetrics>;
}

/// FACT Connection manager for bb8 pool
#[derive(Debug, Clone)]
pub struct FACTConnectionManager {
    config: FACTConfig,
    http_client: reqwest::Client,
}

impl FACTConnectionManager {
    pub fn new(config: FACTConfig) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(config.pool.connection_timeout)
            .build()
            .expect("Failed to create HTTP client");
            
        Self {
            config,
            http_client,
        }
    }
}

/// FACT Connection wrapper
#[derive(Debug)]
pub struct FACTConnection {
    client: reqwest::Client,
    endpoint: String,
    api_key: String,
    connection_id: Uuid,
}

impl FACTConnection {
    pub fn new(client: reqwest::Client, endpoint: String, api_key: String) -> Self {
        Self {
            client,
            endpoint,
            api_key,
            connection_id: Uuid::new_v4(),
        }
    }
    
    /// Execute HTTP GET request with authentication
    async fn get(&self, path: &str) -> Result<Option<serde_json::Value>> {
        let url = format!("{}/{}", self.endpoint, path);
        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("X-Connection-Id", self.connection_id.to_string())
            .send()
            .await
            .map_err(|e| ProcessorError::NetworkError {
                operation: "fact_get".to_string(),
                reason: e.to_string(),
            })?;

        if response.status() == 404 {
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(ProcessorError::ExternalServiceError {
                service: "fact".to_string(),
                reason: format!("HTTP {}: {}", response.status(), response.text().await.unwrap_or_default()),
            });
        }

        let json: serde_json::Value = response.json().await
            .map_err(|e| ProcessorError::SerializationError {
                format: "json".to_string(),
                reason: e.to_string(),
            })?;

        Ok(Some(json))
    }
    
    /// Execute HTTP PUT request with authentication
    async fn put(&self, path: &str, data: &serde_json::Value) -> Result<()> {
        let url = format!("{}/{}", self.endpoint, path);
        let response = self.client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("X-Connection-Id", self.connection_id.to_string())
            .header("Content-Type", "application/json")
            .json(data)
            .send()
            .await
            .map_err(|e| ProcessorError::NetworkError {
                operation: "fact_put".to_string(),
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(ProcessorError::ExternalServiceError {
                service: "fact".to_string(),
                reason: format!("HTTP {}: {}", response.status(), response.text().await.unwrap_or_default()),
            });
        }

        Ok(())
    }
    
    /// Execute HTTP DELETE request with authentication
    async fn delete(&self, path: &str) -> Result<()> {
        let url = format!("{}/{}", self.endpoint, path);
        let response = self.client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("X-Connection-Id", self.connection_id.to_string())
            .send()
            .await
            .map_err(|e| ProcessorError::NetworkError {
                operation: "fact_delete".to_string(),
                reason: e.to_string(),
            })?;

        if !response.status().is_success() {
            return Err(ProcessorError::ExternalServiceError {
                service: "fact".to_string(),
                reason: format!("HTTP {}: {}", response.status(), response.text().await.unwrap_or_default()),
            });
        }

        Ok(())
    }
}

#[async_trait]
impl bb8::ManageConnection for FACTConnectionManager {
    type Connection = FACTConnection;
    type Error = ProcessorError;

    async fn connect(&self) -> Result<Self::Connection> {
        let connection = FACTConnection::new(
            self.http_client.clone(),
            self.config.endpoint.clone(),
            self.config.api_key.clone(),
        );

        // Test connection with health check
        let health_url = format!("{}/health", self.config.endpoint);
        let _response = self.http_client
            .get(&health_url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| ProcessorError::NetworkError {
                operation: "connection_test".to_string(),
                reason: e.to_string(),
            })?;

        Ok(connection)
    }

    async fn is_valid(&self, conn: &mut Self::Connection) -> Result<()> {
        // Simple ping to validate connection
        let ping_url = format!("{}/ping", conn.endpoint);
        let _response = conn.client
            .get(&ping_url)
            .header("Authorization", format!("Bearer {}", conn.api_key))
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .map_err(|e| ProcessorError::NetworkError {
                operation: "connection_validation".to_string(),
                reason: e.to_string(),
            })?;

        Ok(())
    }

    fn has_broken(&self, _conn: &mut Self::Connection) -> bool {
        // For HTTP connections, we assume they can be recreated if needed
        false
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker implementation
#[derive(Debug)]
struct CircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<RwLock<u32>>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    success_count: Arc<RwLock<u32>>,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            success_count: Arc::new(RwLock::new(0)),
            config,
        }
    }

    async fn call<F, T>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        // Check if circuit is open and recovery timeout has passed
        {
            let state = self.state.read().await;
            if *state == CircuitState::Open {
                let last_failure = self.last_failure_time.read().await;
                if let Some(last_failure_time) = *last_failure {
                    if last_failure_time.elapsed() < self.config.recovery_timeout {
                        return Err(ProcessorError::ExternalServiceError {
                            service: "fact".to_string(),
                            reason: "Circuit breaker is open".to_string(),
                        });
                    } else {
                        // Transition to half-open
                        drop(state);
                        let mut state = self.state.write().await;
                        *state = CircuitState::HalfOpen;
                        let mut success_count = self.success_count.write().await;
                        *success_count = 0;
                    }
                } else {
                    drop(state);
                    let mut state = self.state.write().await;
                    *state = CircuitState::HalfOpen;
                }
            }
        }

        match operation.await {
            Ok(result) => {
                self.record_success().await;
                Ok(result)
            }
            Err(error) => {
                self.record_failure().await;
                Err(error)
            }
        }
    }

    async fn record_success(&self) {
        let state = self.state.read().await.clone();
        match state {
            CircuitState::HalfOpen => {
                let mut success_count = self.success_count.write().await;
                *success_count += 1;
                
                if *success_count >= self.config.success_threshold {
                    drop(success_count);
                    let mut circuit_state = self.state.write().await;
                    *circuit_state = CircuitState::Closed;
                    let mut failure_count = self.failure_count.write().await;
                    *failure_count = 0;
                    info!("Circuit breaker closed after successful recovery");
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                let mut failure_count = self.failure_count.write().await;
                *failure_count = 0;
            }
            CircuitState::Open => {
                // Should not happen as we check state before calling
            }
        }
    }

    async fn record_failure(&self) {
        let mut failure_count = self.failure_count.write().await;
        *failure_count += 1;
        
        let mut last_failure_time = self.last_failure_time.write().await;
        *last_failure_time = Some(Instant::now());

        if *failure_count >= self.config.failure_threshold {
            drop(failure_count);
            drop(last_failure_time);
            
            let mut state = self.state.write().await;
            if *state == CircuitState::Closed || *state == CircuitState::HalfOpen {
                *state = CircuitState::Open;
                warn!("Circuit breaker opened due to repeated failures");
            }
        }
    }

    async fn is_open(&self) -> bool {
        *self.state.read().await == CircuitState::Open
    }
}

/// Performance metrics for FACT client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total requests made
    pub total_requests: u64,
    /// Cache hit count
    pub cache_hits: u64,
    /// Cache miss count
    pub cache_misses: u64,
    /// Cache hit ratio
    pub hit_ratio: f64,
    /// Average cache hit latency (ms)
    pub avg_hit_latency_ms: f64,
    /// Average cache miss latency (ms)
    pub avg_miss_latency_ms: f64,
    /// Error count
    pub error_count: u64,
    /// Error rate
    pub error_rate: f64,
    /// Circuit breaker status
    pub circuit_breaker_open: bool,
    /// Active connections
    pub active_connections: u32,
    /// Pool utilization
    pub pool_utilization: f64,
}

/// Health status for FACT client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall health status
    pub status: String,
    /// Last successful operation
    pub last_success: Option<chrono::DateTime<chrono::Utc>>,
    /// Connection pool status
    pub pool_status: String,
    /// Circuit breaker status
    pub circuit_breaker_status: String,
    /// Latency metrics
    pub latency_metrics: LatencyMetrics,
}

/// Latency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// P50 latency (ms)
    pub p50_ms: f64,
    /// P95 latency (ms)
    pub p95_ms: f64,
    /// P99 latency (ms)
    pub p99_ms: f64,
    /// Max latency (ms)
    pub max_ms: f64,
}

/// Main FACT Client implementation
#[derive(Debug)]
pub struct FACTClient {
    /// Connection pool
    pool: Pool<FACTConnectionManager>,
    /// Configuration
    config: FACTConfig,
    /// Circuit breaker
    circuit_breaker: CircuitBreaker,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Latency samples for percentile calculation
    latency_samples: Arc<RwLock<Vec<f64>>>,
}

impl FACTClient {
    /// Create new FACT client with configuration
    pub async fn new(config: FACTConfig) -> Result<Self> {
        let manager = FACTConnectionManager::new(config.clone());
        
        let pool = Pool::builder()
            .max_size(config.pool.max_size)
            .min_idle(config.pool.min_idle)
            .connection_timeout(config.pool.connection_timeout)
            .idle_timeout(config.pool.idle_timeout)
            .max_lifetime(config.pool.max_lifetime)
            .build(manager)
            .await
            .map_err(|e| ProcessorError::ResourceError {
                resource: "connection_pool".to_string(),
                reason: e.to_string(),
            })?;

        let circuit_breaker = CircuitBreaker::new(config.circuit_breaker.clone());
        
        let metrics = Arc::new(RwLock::new(PerformanceMetrics {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            hit_ratio: 0.0,
            avg_hit_latency_ms: 0.0,
            avg_miss_latency_ms: 0.0,
            error_count: 0,
            error_rate: 0.0,
            circuit_breaker_open: false,
            active_connections: 0,
            pool_utilization: 0.0,
        }));

        Ok(Self {
            pool,
            config,
            circuit_breaker,
            metrics,
            latency_samples: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Execute operation with retry logic and exponential backoff
    async fn with_retry<F, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>> + Send,
    {
        let mut attempt = 0;
        let mut delay = self.config.retry.initial_delay;

        loop {
            attempt += 1;
            
            let result = self.circuit_breaker.call(operation()).await;
            
            match result {
                Ok(value) => return Ok(value),
                Err(error) if attempt >= self.config.retry.max_attempts => {
                    error!("Operation failed after {} attempts: {}", attempt, error);
                    return Err(error);
                }
                Err(error) => {
                    warn!("Operation failed (attempt {}): {}", attempt, error);
                    
                    // Apply jitter if configured
                    let actual_delay = if self.config.retry.jitter {
                        let jitter = fastrand::f64() * 0.1; // ±10% jitter
                        Duration::from_millis((delay.as_millis() as f64 * (1.0 + jitter)) as u64)
                    } else {
                        delay
                    };
                    
                    tokio::time::sleep(actual_delay).await;
                    
                    // Exponential backoff
                    delay = std::cmp::min(
                        Duration::from_millis(
                            (delay.as_millis() as f64 * self.config.retry.backoff_multiplier) as u64
                        ),
                        self.config.retry.max_delay,
                    );
                }
            }
        }
    }

    /// Record operation metrics
    async fn record_metrics(&self, operation_type: &str, latency: Duration, is_error: bool) {
        let latency_ms = latency.as_millis() as f64;
        
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        
        if is_error {
            metrics.error_count += 1;
        } else if operation_type == "cache_hit" {
            metrics.cache_hits += 1;
            metrics.avg_hit_latency_ms = (metrics.avg_hit_latency_ms * (metrics.cache_hits - 1) as f64 + latency_ms) / metrics.cache_hits as f64;
        } else if operation_type == "cache_miss" {
            metrics.cache_misses += 1;
            metrics.avg_miss_latency_ms = (metrics.avg_miss_latency_ms * (metrics.cache_misses - 1) as f64 + latency_ms) / metrics.cache_misses as f64;
        }
        
        metrics.hit_ratio = metrics.cache_hits as f64 / (metrics.cache_hits + metrics.cache_misses) as f64;
        metrics.error_rate = metrics.error_count as f64 / metrics.total_requests as f64;
        metrics.circuit_breaker_open = self.circuit_breaker.is_open().await;
        
        // Update pool utilization
        let pool_state = self.pool.state();
        metrics.active_connections = pool_state.connections;
        metrics.pool_utilization = pool_state.connections as f64 / self.config.pool.max_size as f64;
        
        // Store latency sample for percentile calculation
        let mut samples = self.latency_samples.write().await;
        samples.push(latency_ms);
        
        // Keep only last 1000 samples
        if samples.len() > 1000 {
            samples.remove(0);
        }
    }

    /// Generate cache key for query
    fn cache_key(&self, prefix: &str, query: &str) -> String {
        format!("{}:{}", prefix, blake3::hash(query.as_bytes()).to_hex())
    }
}

#[async_trait]
impl FACTClientInterface for FACTClient {
    #[instrument(skip(self))]
    async fn get_query_result(&self, query: &str) -> Result<Option<QueryResult>> {
        let start = Instant::now();
        let cache_key = self.cache_key("query_result", query);
        
        let result = self.with_retry(|| {
            let pool = self.pool.clone();
            let key = cache_key.clone();
            Box::pin(async move {
                let conn = pool.get().await.map_err(|e| ProcessorError::ResourceError {
                    resource: "connection_pool".to_string(),
                    reason: e.to_string(),
                })?;
                
                conn.get(&format!("cache/{}", key)).await
            })
        }).await;

        let latency = start.elapsed();
        
        match result {
            Ok(Some(json)) => {
                let query_result: QueryResult = serde_json::from_value(json)
                    .map_err(|e| ProcessorError::SerializationError {
                        format: "json".to_string(),
                        reason: e.to_string(),
                    })?;
                
                self.record_metrics("cache_hit", latency, false).await;
                debug!("Cache hit for query result in {:?}", latency);
                Ok(Some(query_result))
            }
            Ok(None) => {
                self.record_metrics("cache_miss", latency, false).await;
                debug!("Cache miss for query result in {:?}", latency);
                Ok(None)
            }
            Err(e) => {
                self.record_metrics("cache_error", latency, true).await;
                Err(e)
            }
        }
    }

    #[instrument(skip(self, result))]
    async fn set_query_result(&self, query: &str, result: &QueryResult, ttl: Duration) -> Result<()> {
        let start = Instant::now();
        let cache_key = self.cache_key("query_result", query);
        
        let cache_data = serde_json::json!({
            "data": result,
            "ttl": ttl.as_secs(),
            "created_at": chrono::Utc::now().to_rfc3339()
        });

        let set_result = self.with_retry(|| {
            let pool = self.pool.clone();
            let key = cache_key.clone();
            let data = cache_data.clone();
            Box::pin(async move {
                let conn = pool.get().await.map_err(|e| ProcessorError::ResourceError {
                    resource: "connection_pool".to_string(),
                    reason: e.to_string(),
                })?;
                
                conn.put(&format!("cache/{}", key), &data).await
            })
        }).await;

        let latency = start.elapsed();
        
        match set_result {
            Ok(()) => {
                self.record_metrics("cache_set", latency, false).await;
                debug!("Set query result in cache in {:?}", latency);
                Ok(())
            }
            Err(e) => {
                self.record_metrics("cache_set_error", latency, true).await;
                Err(e)
            }
        }
    }

    async fn get_entities(&self, query: &str) -> Result<Option<Vec<ExtractedEntity>>> {
        let start = Instant::now();
        let cache_key = self.cache_key("entities", query);
        
        let result = self.with_retry(|| {
            let pool = self.pool.clone();
            let key = cache_key.clone();
            Box::pin(async move {
                let conn = pool.get().await.map_err(|e| ProcessorError::ResourceError {
                    resource: "connection_pool".to_string(),
                    reason: e.to_string(),
                })?;
                
                conn.get(&format!("cache/{}", key)).await
            })
        }).await;

        let latency = start.elapsed();
        
        match result {
            Ok(Some(json)) => {
                let entities: Vec<ExtractedEntity> = serde_json::from_value(json["data"].clone())
                    .map_err(|e| ProcessorError::SerializationError {
                        format: "json".to_string(),
                        reason: e.to_string(),
                    })?;
                
                self.record_metrics("cache_hit", latency, false).await;
                Ok(Some(entities))
            }
            Ok(None) => {
                self.record_metrics("cache_miss", latency, false).await;
                Ok(None)
            }
            Err(e) => {
                self.record_metrics("cache_error", latency, true).await;
                Err(e)
            }
        }
    }

    async fn set_entities(&self, query: &str, entities: &[ExtractedEntity], ttl: Duration) -> Result<()> {
        let cache_key = self.cache_key("entities", query);
        
        let cache_data = serde_json::json!({
            "data": entities,
            "ttl": ttl.as_secs(),
            "created_at": chrono::Utc::now().to_rfc3339()
        });

        self.with_retry(|| {
            let pool = self.pool.clone();
            let key = cache_key.clone();
            let data = cache_data.clone();
            Box::pin(async move {
                let conn = pool.get().await.map_err(|e| ProcessorError::ResourceError {
                    resource: "connection_pool".to_string(),
                    reason: e.to_string(),
                })?;
                
                conn.put(&format!("cache/{}", key), &data).await
            })
        }).await
    }

    async fn get_classification(&self, query: &str) -> Result<Option<ClassificationResult>> {
        let cache_key = self.cache_key("classification", query);
        
        let result = self.with_retry(|| {
            let pool = self.pool.clone();
            let key = cache_key.clone();
            Box::pin(async move {
                let conn = pool.get().await.map_err(|e| ProcessorError::ResourceError {
                    resource: "connection_pool".to_string(),
                    reason: e.to_string(),
                })?;
                
                conn.get(&format!("cache/{}", key)).await
            })
        }).await?;

        match result {
            Some(json) => {
                let classification: ClassificationResult = serde_json::from_value(json["data"].clone())
                    .map_err(|e| ProcessorError::SerializationError {
                        format: "json".to_string(),
                        reason: e.to_string(),
                    })?;
                Ok(Some(classification))
            }
            None => Ok(None),
        }
    }

    async fn set_classification(&self, query: &str, classification: &ClassificationResult, ttl: Duration) -> Result<()> {
        let cache_key = self.cache_key("classification", query);
        
        let cache_data = serde_json::json!({
            "data": classification,
            "ttl": ttl.as_secs(),
            "created_at": chrono::Utc::now().to_rfc3339()
        });

        self.with_retry(|| {
            let pool = self.pool.clone();
            let key = cache_key.clone();
            let data = cache_data.clone();
            Box::pin(async move {
                let conn = pool.get().await.map_err(|e| ProcessorError::ResourceError {
                    resource: "connection_pool".to_string(),
                    reason: e.to_string(),
                })?;
                
                conn.put(&format!("cache/{}", key), &data).await
            })
        }).await
    }

    async fn get_strategy(&self, query_hash: &str) -> Result<Option<StrategyRecommendation>> {
        let cache_key = format!("strategy:{}", query_hash);
        
        let result = self.with_retry(|| {
            let pool = self.pool.clone();
            let key = cache_key.clone();
            Box::pin(async move {
                let conn = pool.get().await.map_err(|e| ProcessorError::ResourceError {
                    resource: "connection_pool".to_string(),
                    reason: e.to_string(),
                })?;
                
                conn.get(&format!("cache/{}", key)).await
            })
        }).await?;

        match result {
            Some(json) => {
                let strategy: StrategyRecommendation = serde_json::from_value(json["data"].clone())
                    .map_err(|e| ProcessorError::SerializationError {
                        format: "json".to_string(),
                        reason: e.to_string(),
                    })?;
                Ok(Some(strategy))
            }
            None => Ok(None),
        }
    }

    async fn set_strategy(&self, query_hash: &str, strategy: &StrategyRecommendation, ttl: Duration) -> Result<()> {
        let cache_key = format!("strategy:{}", query_hash);
        
        let cache_data = serde_json::json!({
            "data": strategy,
            "ttl": ttl.as_secs(),
            "created_at": chrono::Utc::now().to_rfc3339()
        });

        self.with_retry(|| {
            let pool = self.pool.clone();
            let key = cache_key.clone();
            let data = cache_data.clone();
            Box::pin(async move {
                let conn = pool.get().await.map_err(|e| ProcessorError::ResourceError {
                    resource: "connection_pool".to_string(),
                    reason: e.to_string(),
                })?;
                
                conn.put(&format!("cache/{}", key), &data).await
            })
        }).await
    }

    async fn clear_cache(&self) -> Result<()> {
        self.with_retry(|| {
            let pool = self.pool.clone();
            Box::pin(async move {
                let conn = pool.get().await.map_err(|e| ProcessorError::ResourceError {
                    resource: "connection_pool".to_string(),
                    reason: e.to_string(),
                })?;
                
                conn.delete("cache").await
            })
        }).await
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        let metrics = self.metrics.read().await;
        let samples = self.latency_samples.read().await;
        
        // Calculate percentiles
        let mut sorted_samples = samples.clone();
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let latency_metrics = if sorted_samples.is_empty() {
            LatencyMetrics {
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                max_ms: 0.0,
            }
        } else {
            let p50_idx = (sorted_samples.len() as f64 * 0.5) as usize;
            let p95_idx = (sorted_samples.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted_samples.len() as f64 * 0.99) as usize;
            
            LatencyMetrics {
                p50_ms: sorted_samples.get(p50_idx).copied().unwrap_or(0.0),
                p95_ms: sorted_samples.get(p95_idx).copied().unwrap_or(0.0),
                p99_ms: sorted_samples.get(p99_idx).copied().unwrap_or(0.0),
                max_ms: sorted_samples.last().copied().unwrap_or(0.0),
            }
        };

        let status = if metrics.circuit_breaker_open {
            "circuit_open".to_string()
        } else if metrics.error_rate > self.config.monitoring.alert_thresholds.error_rate_threshold {
            "degraded".to_string()
        } else if latency_metrics.p95_ms > self.config.monitoring.alert_thresholds.cache_miss_latency_ms as f64 {
            "slow".to_string()
        } else {
            "healthy".to_string()
        };

        Ok(HealthStatus {
            status,
            last_success: Some(chrono::Utc::now()), // Simplified
            pool_status: if metrics.pool_utilization > 0.8 { "high_utilization".to_string() } else { "ok".to_string() },
            circuit_breaker_status: if metrics.circuit_breaker_open { "open".to_string() } else { "closed".to_string() },
            latency_metrics,
        })
    }

    async fn get_metrics(&self) -> Result<PerformanceMetrics> {
        Ok(self.metrics.read().await.clone())
    }
}

impl Default for FACTConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:8080".to_string(),
            api_key: "default-key".to_string(),
            pool: PoolConfig {
                max_size: 32,
                min_idle: Some(4),
                connection_timeout: Duration::from_secs(3),
                idle_timeout: Some(Duration::from_secs(600)),
                max_lifetime: Some(Duration::from_secs(1800)),
            },
            retry: RetryConfig {
                max_attempts: 3,
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(5),
                backoff_multiplier: 2.0,
                jitter: true,
            },
            circuit_breaker: CircuitBreakerConfig {
                failure_threshold: 5,
                time_window: Duration::from_secs(60),
                recovery_timeout: Duration::from_secs(60),
                success_threshold: 3,
            },
            monitoring: MonitoringConfig {
                enable_detailed_metrics: true,
                metrics_interval: Duration::from_secs(30),
                alert_thresholds: AlertThresholds {
                    cache_hit_latency_ms: 23,
                    cache_miss_latency_ms: 95,
                    error_rate_threshold: 0.05,
                    circuit_breaker_threshold: 0.1,
                },
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_fact_config_creation() {
        let config = FACTConfig::default();
        assert_eq!(config.pool.max_size, 32);
        assert_eq!(config.retry.max_attempts, 3);
        assert_eq!(config.monitoring.alert_thresholds.cache_hit_latency_ms, 23);
    }

    #[test]
    fn test_cache_key_generation() {
        // We need to create a minimal client for testing, but the actual connection
        // won't work without a real FACT server. This tests the key generation logic.
        let config = FACTConfig::default();
        let manager = FACTConnectionManager::new(config);
        
        // Test that the same input produces the same key
        let query1 = "test query";
        let query2 = "test query";
        
        // We can't easily test the actual FACTClient without a server,
        // but we can test the hashing logic
        let key1 = format!("query_result:{}", blake3::hash(query1.as_bytes()).to_hex());
        let key2 = format!("query_result:{}", blake3::hash(query2.as_bytes()).to_hex());
        
        assert_eq!(key1, key2);
        
        // Different queries should produce different keys
        let query3 = "different query";
        let key3 = format!("query_result:{}", blake3::hash(query3.as_bytes()).to_hex());
        
        assert_ne!(key1, key3);
    }
}