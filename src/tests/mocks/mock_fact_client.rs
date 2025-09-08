//! Mock FACT Client Implementation
//! 
//! Comprehensive mock implementation of FACT client for testing with
//! realistic behavior simulation, test fixtures, and helper utilities.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use async_trait::async_trait;
use mockall::{mock, predicate::*};
use serde_json::{json, Value};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Core FACT client trait for mocking
#[async_trait]
pub trait FACTClientTrait: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, FACTError>;
    async fn set(&self, key: &str, value: &[u8], ttl: Duration) -> Result<(), FACTError>;
    async fn delete(&self, key: &str) -> Result<bool, FACTError>;
    async fn exists(&self, key: &str) -> Result<bool, FACTError>;
    async fn search(&self, query: &str) -> Result<Vec<Value>, FACTError>;
    async fn get_with_citations(&self, key: &str) -> Result<Option<Vec<u8>>, FACTError>;
    async fn validate(&self, result: &QueryResult) -> Result<ConsensusVote, FACTError>;
    async fn get_metrics(&self) -> Result<FACTMetrics, FACTError>;
    async fn health_check(&self) -> Result<HealthStatus, FACTError>;
}

/// FACT Error types for testing
#[derive(Debug, Clone, thiserror::Error, PartialEq)]
pub enum FACTError {
    #[error("Connection timeout")]
    ConnectionTimeout,
    #[error("Connection error: {message}")]
    ConnectionError { message: String },
    #[error("Service unavailable")]
    ServiceUnavailable,
    #[error("Service down")]
    ServiceDown,
    #[error("Authentication failed")]
    AuthenticationFailed,
    #[error("Invalid key: {key}")]
    InvalidKey { key: String },
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },
    #[error("Serialization error: {message}")]
    SerializationError { message: String },
    #[error("Cache full")]
    CacheFull,
    #[error("TTL exceeded")]
    TTLExceeded,
    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Query result structure for consensus testing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryResult {
    pub id: Uuid,
    pub content: String,
    pub confidence: f64,
    pub citations: Vec<Citation>,
    pub metadata: HashMap<String, Value>,
}

/// Citation structure matching the main system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Citation {
    pub source: String,
    pub section: String,
    pub page: Option<u32>,
    pub confidence: f64,
}

/// Consensus vote for Byzantine consensus testing
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsensusVote {
    pub node_id: String,
    pub agrees: bool,
    pub confidence: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// FACT metrics for monitoring
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FACTMetrics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_latency_ms: f64,
    pub hit_rate: f64,
    pub memory_usage_mb: f64,
    pub active_connections: u32,
    pub uptime_seconds: u64,
}

/// Health status for health checks
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    pub healthy: bool,
    pub version: String,
    pub uptime: Duration,
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub cache_status: String,
}

// Create the mock using mockall
mock! {
    pub FACTClient {}
    
    #[async_trait]
    impl FACTClientTrait for FACTClient {
        async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, FACTError>;
        async fn set(&self, key: &str, value: &[u8], ttl: Duration) -> Result<(), FACTError>;
        async fn delete(&self, key: &str) -> Result<bool, FACTError>;
        async fn exists(&self, key: &str) -> Result<bool, FACTError>;
        async fn search(&self, query: &str) -> Result<Vec<Value>, FACTError>;
        async fn get_with_citations(&self, key: &str) -> Result<Option<Vec<u8>>, FACTError>;
        async fn validate(&self, result: &QueryResult) -> Result<ConsensusVote, FACTError>;
        async fn get_metrics(&self) -> Result<FACTMetrics, FACTError>;
        async fn health_check(&self) -> Result<HealthStatus, FACTError>;
    }
}

/// Test fixtures and data generators
pub struct FACTTestFixtures;

impl FACTTestFixtures {
    /// Generate realistic compliance-related test data
    pub fn generate_compliance_data() -> HashMap<String, Value> {
        let mut data = HashMap::new();
        
        data.insert("pci_dss_requirements".to_string(), json!({
            "content": "PCI DSS 4.0 requires strong cryptographic protection of cardholder data",
            "version": "4.0",
            "sections": ["3.4", "3.5", "3.6"],
            "confidence": 0.98,
            "last_updated": "2024-03-15T10:30:00Z"
        }));
        
        data.insert("gdpr_consent_requirements".to_string(), json!({
            "content": "GDPR Article 7 specifies conditions for consent including withdrawal mechanisms",
            "article": "Article 7",
            "sections": ["7.1", "7.2", "7.3"],
            "confidence": 0.95,
            "last_updated": "2024-01-20T14:15:00Z"
        }));
        
        data.insert("iso_27001_access_control".to_string(), json!({
            "content": "ISO 27001 Annex A.9 covers access control management and user access provisioning",
            "standard": "ISO/IEC 27001:2022",
            "annex": "A.9",
            "controls": ["A.9.1", "A.9.2", "A.9.3", "A.9.4"],
            "confidence": 0.92,
            "last_updated": "2024-02-10T09:45:00Z"
        }));
        
        data.insert("sox_documentation_requirements".to_string(), json!({
            "content": "SOX Section 404 requires documented internal controls over financial reporting",
            "section": "404",
            "subsections": ["404(a)", "404(b)"],
            "confidence": 0.89,
            "last_updated": "2023-12-05T16:20:00Z"
        }));
        
        data
    }
    
    /// Generate test citations
    pub fn generate_test_citations() -> Vec<Citation> {
        vec![
            Citation {
                source: "PCI Data Security Standard v4.0".to_string(),
                section: "Requirement 3.4".to_string(),
                page: Some(47),
                confidence: 0.98,
            },
            Citation {
                source: "GDPR - General Data Protection Regulation".to_string(),
                section: "Article 7(3)".to_string(),
                page: Some(23),
                confidence: 0.95,
            },
            Citation {
                source: "ISO/IEC 27001:2022".to_string(),
                section: "Annex A.9.1.2".to_string(),
                page: Some(89),
                confidence: 0.92,
            },
            Citation {
                source: "Sarbanes-Oxley Act".to_string(),
                section: "Section 404(a)".to_string(),
                page: None,
                confidence: 0.89,
            },
        ]
    }
    
    /// Generate performance test data with varying sizes
    pub fn generate_performance_data(size: usize) -> Value {
        let content = "x".repeat(size);
        json!({
            "content": content,
            "size_bytes": size,
            "generated_at": chrono::Utc::now().timestamp(),
            "checksum": format!("{:x}", md5::compute(&content))
        })
    }
    
    /// Create mock FACT metrics
    pub fn create_mock_metrics(hit_rate: f64, avg_latency_ms: f64) -> FACTMetrics {
        let total_requests = 10000;
        let cache_hits = (total_requests as f64 * hit_rate) as u64;
        
        FACTMetrics {
            total_requests,
            cache_hits,
            cache_misses: total_requests - cache_hits,
            average_latency_ms: avg_latency_ms,
            hit_rate,
            memory_usage_mb: 256.7,
            active_connections: 42,
            uptime_seconds: 86400, // 24 hours
        }
    }
}

/// Realistic FACT client simulator for integration testing
pub struct RealisticFACTSimulator {
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    metrics: Arc<RwLock<FACTMetrics>>,
    config: SimulatorConfig,
    start_time: Instant,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    created_at: Instant,
    ttl: Duration,
    access_count: u64,
}

#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    pub cache_hit_latency_ms: u64,
    pub cache_miss_latency_ms: u64,
    pub error_rate: f64,
    pub max_cache_size: usize,
    pub default_ttl: Duration,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            cache_hit_latency_ms: 15,
            cache_miss_latency_ms: 75,
            error_rate: 0.01, // 1% error rate
            max_cache_size: 10000,
            default_ttl: Duration::from_secs(1800),
        }
    }
}

impl RealisticFACTSimulator {
    pub fn new(config: SimulatorConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(FACTMetrics {
                total_requests: 0,
                cache_hits: 0,
                cache_misses: 0,
                average_latency_ms: 0.0,
                hit_rate: 0.0,
                memory_usage_mb: 0.0,
                active_connections: 1,
                uptime_seconds: 0,
            })),
            config,
            start_time: Instant::now(),
        }
    }
    
    /// Pre-populate cache with test data
    pub async fn preload_cache(&self, data: HashMap<String, Vec<u8>>) {
        let mut cache = self.cache.write().await;
        let now = Instant::now();
        
        for (key, value) in data {
            cache.insert(key, CacheEntry {
                data: value,
                created_at: now,
                ttl: self.config.default_ttl,
                access_count: 0,
            });
        }
    }
    
    /// Simulate realistic error conditions
    fn should_simulate_error(&self) -> Option<FACTError> {
        if rand::random::<f64>() < self.config.error_rate {
            match rand::random::<u32>() % 4 {
                0 => Some(FACTError::ConnectionTimeout),
                1 => Some(FACTError::ServiceUnavailable),
                2 => Some(FACTError::ValidationFailed { 
                    reason: "Simulated validation error".to_string() 
                }),
                _ => Some(FACTError::Internal { 
                    message: "Simulated internal error".to_string() 
                }),
            }
        } else {
            None
        }
    }
    
    /// Update metrics after each operation
    async fn update_metrics(&self, latency: Duration, was_hit: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        
        if was_hit {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
        
        // Calculate running average latency
        let total_latency = metrics.average_latency_ms * (metrics.total_requests - 1) as f64 + latency.as_millis() as f64;
        metrics.average_latency_ms = total_latency / metrics.total_requests as f64;
        
        // Update hit rate
        metrics.hit_rate = metrics.cache_hits as f64 / metrics.total_requests as f64;
        
        // Update uptime
        metrics.uptime_seconds = self.start_time.elapsed().as_secs();
        
        // Simulate memory usage based on cache size
        let cache_size = self.cache.read().await.len();
        metrics.memory_usage_mb = (cache_size as f64 * 0.1) + 50.0; // Base memory + cache overhead
    }
}

#[async_trait]
impl FACTClientTrait for RealisticFACTSimulator {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>, FACTError> {
        let start = Instant::now();
        
        // Check for simulated errors first
        if let Some(error) = self.should_simulate_error() {
            return Err(error);
        }
        
        let mut cache = self.cache.write().await;
        let now = Instant::now();
        
        // Check if key exists and is not expired
        if let Some(mut entry) = cache.get_mut(key) {
            if now.duration_since(entry.created_at) < entry.ttl {
                // Cache hit
                entry.access_count += 1;
                let data = entry.data.clone();
                
                // Simulate cache hit latency
                tokio::time::sleep(Duration::from_millis(self.config.cache_hit_latency_ms)).await;
                let latency = start.elapsed();
                
                drop(cache); // Release lock before updating metrics
                self.update_metrics(latency, true).await;
                
                return Ok(Some(data));
            } else {
                // Entry expired, remove it
                cache.remove(key);
            }
        }
        
        // Cache miss - simulate fetching from origin
        tokio::time::sleep(Duration::from_millis(self.config.cache_miss_latency_ms)).await;
        
        // Generate mock data for cache miss
        let mock_data = serde_json::to_vec(&json!({
            "content": format!("Generated content for key: {}", key),
            "cached": false,
            "generated_at": chrono::Utc::now().timestamp(),
            "key": key
        })).map_err(|e| FACTError::SerializationError { 
            message: e.to_string() 
        })?;
        
        // Store in cache if there's room
        if cache.len() < self.config.max_cache_size {
            cache.insert(key.to_string(), CacheEntry {
                data: mock_data.clone(),
                created_at: now,
                ttl: self.config.default_ttl,
                access_count: 1,
            });
        }
        
        let latency = start.elapsed();
        drop(cache); // Release lock
        self.update_metrics(latency, false).await;
        
        Ok(Some(mock_data))
    }
    
    async fn set(&self, key: &str, value: &[u8], ttl: Duration) -> Result<(), FACTError> {
        if let Some(error) = self.should_simulate_error() {
            return Err(error);
        }
        
        let mut cache = self.cache.write().await;
        
        if cache.len() >= self.config.max_cache_size && !cache.contains_key(key) {
            return Err(FACTError::CacheFull);
        }
        
        cache.insert(key.to_string(), CacheEntry {
            data: value.to_vec(),
            created_at: Instant::now(),
            ttl,
            access_count: 0,
        });
        
        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<bool, FACTError> {
        if let Some(error) = self.should_simulate_error() {
            return Err(error);
        }
        
        let mut cache = self.cache.write().await;
        Ok(cache.remove(key).is_some())
    }
    
    async fn exists(&self, key: &str) -> Result<bool, FACTError> {
        if let Some(error) = self.should_simulate_error() {
            return Err(error);
        }
        
        let cache = self.cache.read().await;
        let now = Instant::now();
        
        if let Some(entry) = cache.get(key) {
            Ok(now.duration_since(entry.created_at) < entry.ttl)
        } else {
            Ok(false)
        }
    }
    
    async fn search(&self, query: &str) -> Result<Vec<Value>, FACTError> {
        if let Some(error) = self.should_simulate_error() {
            return Err(error);
        }
        
        // Simulate search latency
        tokio::time::sleep(Duration::from_millis(30)).await;
        
        // Generate mock search results
        let results = (0..5).map(|i| json!({
            "id": format!("result_{}", i),
            "title": format!("Search result {} for query: {}", i + 1, query),
            "relevance": 0.9 - (i as f64 * 0.1),
            "snippet": format!("This is a snippet for result {} matching '{}'", i + 1, query),
            "source": format!("source_{}.pdf", i + 1),
            "page": 10 + i * 5
        })).collect();
        
        Ok(results)
    }
    
    async fn get_with_citations(&self, key: &str) -> Result<Option<Vec<u8>>, FACTError> {
        // Get the base content
        let content = self.get(key).await?;
        
        if let Some(data) = content {
            // Parse and enhance with citations
            let mut parsed: Value = serde_json::from_slice(&data)
                .map_err(|e| FACTError::SerializationError { message: e.to_string() })?;
            
            // Add mock citations
            parsed["citations"] = json!(FACTTestFixtures::generate_test_citations());
            
            let enhanced_data = serde_json::to_vec(&parsed)
                .map_err(|e| FACTError::SerializationError { message: e.to_string() })?;
            
            Ok(Some(enhanced_data))
        } else {
            Ok(None)
        }
    }
    
    async fn validate(&self, result: &QueryResult) -> Result<ConsensusVote, FACTError> {
        if let Some(error) = self.should_simulate_error() {
            return Err(error);
        }
        
        // Simulate validation processing time
        tokio::time::sleep(Duration::from_millis(20)).await;
        
        // Generate consensus vote based on result confidence
        let agrees = result.confidence > 0.7;
        let vote_confidence = if agrees { 
            result.confidence + (rand::random::<f64>() * 0.1 - 0.05) 
        } else { 
            result.confidence - (rand::random::<f64>() * 0.2) 
        }.clamp(0.0, 1.0);
        
        Ok(ConsensusVote {
            node_id: format!("node_{}", uuid::Uuid::new_v4()),
            agrees,
            confidence: vote_confidence,
            timestamp: chrono::Utc::now(),
        })
    }
    
    async fn get_metrics(&self) -> Result<FACTMetrics, FACTError> {
        if let Some(error) = self.should_simulate_error() {
            return Err(error);
        }
        
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }
    
    async fn health_check(&self) -> Result<HealthStatus, FACTError> {
        if let Some(error) = self.should_simulate_error() {
            return Err(error);
        }
        
        let uptime = self.start_time.elapsed();
        let cache_size = self.cache.read().await.len();
        
        Ok(HealthStatus {
            healthy: true,
            version: "1.0.0-test".to_string(),
            uptime,
            memory_usage: (cache_size as f64 * 0.1) + 50.0,
            cpu_usage: rand::random::<f64>() * 0.3 + 0.1, // 10-40% CPU
            cache_status: format!("{} entries", cache_size),
        })
    }
}

/// Factory for creating various test client configurations
pub struct MockFACTClientFactory;

impl MockFACTClientFactory {
    /// Create a client optimized for cache hit testing
    pub fn create_cache_hit_optimized() -> MockFACTClient {
        let mut client = MockFACTClient::new();
        
        client
            .expect_get()
            .returning(|key| {
                // Always return cached data quickly
                let cached_data = serde_json::to_vec(&json!({
                    "content": format!("Cached content for {}", key),
                    "cached": true,
                    "hit_time_ms": 10
                })).unwrap();
                
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    Ok(Some(cached_data))
                })
            });
            
        client
    }
    
    /// Create a client for testing cache misses
    pub fn create_cache_miss_simulation() -> MockFACTClient {
        let mut client = MockFACTClient::new();
        
        client
            .expect_get()
            .returning(|key| {
                Box::pin(async move {
                    // Simulate longer fetch time for cache miss
                    tokio::time::sleep(Duration::from_millis(80)).await;
                    
                    let fresh_data = serde_json::to_vec(&json!({
                        "content": format!("Fresh content for {}", key),
                        "cached": false,
                        "fetch_time_ms": 80
                    })).unwrap();
                    
                    Ok(Some(fresh_data))
                })
            });
            
        client
    }
    
    /// Create a client for error scenario testing
    pub fn create_error_prone() -> MockFACTClient {
        let mut client = MockFACTClient::new();
        
        client
            .expect_get()
            .returning(|_| {
                // Randomly return different types of errors
                match rand::random::<u32>() % 4 {
                    0 => Err(FACTError::ConnectionTimeout),
                    1 => Err(FACTError::ServiceUnavailable),
                    2 => Err(FACTError::ValidationFailed { 
                        reason: "Mock validation failure".to_string() 
                    }),
                    _ => Box::pin(async {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        Ok(Some(b"error recovery data".to_vec()))
                    }),
                }
            });
            
        client
    }
}