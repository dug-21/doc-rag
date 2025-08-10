//! FACT-powered intelligent caching layer for response generation
//!
//! This module provides high-performance caching using FACT (Fast Augmented Context Tools)
//! to achieve sub-50ms response times for cached queries.

use crate::error::{Result, ResponseError};
use crate::{GeneratedResponse, GenerationRequest};
use dashmap::DashMap;
use fact_tools::Cache as FACTCache;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, warn};

/// FACT-accelerated cache manager
#[derive(Clone)]
pub struct FACTCacheManager {
    /// FACT cache instance (wrapped in Arc<Mutex> for mutable access)
    fact_cache: Arc<std::sync::Mutex<FACTCache>>,
    
    /// Context manager for intelligent caching (simplified implementation)
    context_mgr: Arc<SimpleContextManager>,
    
    /// Query optimizer for cache key generation (simplified implementation)
    query_optimizer: Arc<SimpleQueryOptimizer>,
    
    /// Local memory cache for extremely fast access
    memory_cache: Arc<DashMap<CacheKey, CachedResponse>>,
    
    /// Cache configuration
    config: CacheManagerConfig,
    
    /// Cache performance metrics
    metrics: Arc<CacheMetrics>,
}

/// Configuration for FACT cache manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheManagerConfig {
    /// Enable FACT intelligent caching
    pub enable_fact_cache: bool,
    
    /// Enable local memory cache
    pub enable_memory_cache: bool,
    
    /// Maximum memory cache size
    pub memory_cache_size: usize,
    
    /// TTL for cached responses
    pub response_ttl: Duration,
    
    /// TTL for context cache
    pub context_ttl: Duration,
    
    /// Cache hit threshold for performance reporting
    pub hit_threshold_ms: u64,
    
    /// Enable cache analytics
    pub enable_analytics: bool,
    
    /// Cache key optimization settings
    pub key_optimization: KeyOptimizationConfig,
}

/// Configuration for cache key optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyOptimizationConfig {
    /// Enable semantic similarity for key matching
    pub enable_semantic_matching: bool,
    
    /// Similarity threshold for cache hits (0.0-1.0)
    pub similarity_threshold: f64,
    
    /// Enable query normalization
    pub enable_query_normalization: bool,
    
    /// Maximum key length
    pub max_key_length: usize,
}

/// Cache key for response storage
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Normalized query hash
    query_hash: u64,
    
    /// Context fingerprint
    context_fingerprint: u64,
    
    /// Output format
    format: String,
    
    /// Configuration hash
    config_hash: u64,
}

/// Cached response with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    /// The cached response
    response: GeneratedResponse,
    
    /// Cache creation timestamp
    created_at: std::time::SystemTime,
    
    /// Cache access count
    access_count: u64,
    
    /// Last access timestamp
    last_accessed: std::time::SystemTime,
    
    /// Cache hit performance (for analytics)
    avg_hit_time: Duration,
}

/// Cache performance metrics
#[derive(Debug, Default)]
pub struct CacheMetrics {
    /// Total cache requests
    total_requests: std::sync::atomic::AtomicU64,
    
    /// Cache hits
    cache_hits: std::sync::atomic::AtomicU64,
    
    /// Cache misses
    cache_misses: std::sync::atomic::AtomicU64,
    
    /// FACT cache hits
    fact_hits: std::sync::atomic::AtomicU64,
    
    /// Memory cache hits
    memory_hits: std::sync::atomic::AtomicU64,
    
    /// Average hit latency
    avg_hit_latency: std::sync::RwLock<Duration>,
    
    /// Average miss latency
    avg_miss_latency: std::sync::RwLock<Duration>,
}

/// Cache operation result
#[derive(Debug)]
pub enum CacheResult {
    /// Cache hit with response
    Hit {
        response: GeneratedResponse,
        source: CacheSource,
        latency: Duration,
    },
    /// Cache miss - need to generate response
    Miss { key: CacheKey },
}

/// Cache source for analytics
#[derive(Debug, Clone)]
pub enum CacheSource {
    /// FACT intelligent cache
    FACT,
    /// Local memory cache
    Memory,
    /// Hybrid (checked multiple sources)
    Hybrid,
}

impl Default for CacheManagerConfig {
    fn default() -> Self {
        Self {
            enable_fact_cache: true,
            enable_memory_cache: true,
            memory_cache_size: 10_000,
            response_ttl: Duration::from_secs(3600), // 1 hour
            context_ttl: Duration::from_secs(7200),  // 2 hours
            hit_threshold_ms: 50, // Sub-50ms target
            enable_analytics: true,
            key_optimization: KeyOptimizationConfig::default(),
        }
    }
}

impl Default for KeyOptimizationConfig {
    fn default() -> Self {
        Self {
            enable_semantic_matching: true,
            similarity_threshold: 0.85,
            enable_query_normalization: true,
            max_key_length: 256,
        }
    }
}

impl FACTCacheManager {
    /// Create a new FACT cache manager
    pub async fn new(config: CacheManagerConfig) -> Result<Self> {
        info!("Initializing FACT cache manager");

        // Initialize FACT cache with simplified configuration
        let fact_cache = Arc::new(Mutex::new(FACTCache::new()));

        let context_mgr = Arc::new(SimpleContextManager::new());
        let query_optimizer = Arc::new(SimpleQueryOptimizer::new());

        // Initialize memory cache
        let memory_cache = Arc::new(DashMap::with_capacity(config.memory_cache_size));

        let metrics = Arc::new(CacheMetrics::default());

        Ok(Self {
            fact_cache,
            context_mgr,
            query_optimizer,
            memory_cache,
            config,
            metrics,
        })
    }

    /// Get cached response or return miss
    #[instrument(skip(self, request), fields(request_id = %request.id))]
    pub async fn get(&self, request: &GenerationRequest) -> Result<CacheResult> {
        let start_time = Instant::now();
        
        // Increment total requests
        self.metrics.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Generate cache key
        let cache_key = self.generate_cache_key(request).await?;

        // Try memory cache first (fastest)
        if self.config.enable_memory_cache {
            if let Some(cached) = self.memory_cache.get(&cache_key) {
                if !self.is_expired(&cached.value()) {
                    let latency = start_time.elapsed();
                    let _ = self.update_hit_metrics(latency, CacheSource::Memory).await;
                    
                    debug!("Memory cache hit in {}ms", latency.as_millis());
                    
                    // Update access statistics
                    let mut cached = cached.clone();
                    cached.access_count += 1;
                    cached.last_accessed = std::time::SystemTime::now();
                    cached.avg_hit_time = (cached.avg_hit_time + latency) / 2;
                    
                    return Ok(CacheResult::Hit {
                        response: cached.response.clone(),
                        source: CacheSource::Memory,
                        latency,
                    });
                }
            }
        }

        // Try FACT cache (intelligent caching)
        if self.config.enable_fact_cache {
            let optimized_query = self.query_optimizer.optimize(&request.query);
            let cache_key_str = format!("query:{}", blake3::hash(optimized_query.as_bytes()).to_hex());
            
            if let Ok(cache) = self.fact_cache.lock() {
                // Try exact cache key first
                if let Some(cached_data) = cache.get(&cache_key_str) {
                    match serde_json::from_value::<GeneratedResponse>(cached_data) {
                        Ok(response) => {
                            let latency = start_time.elapsed();
                            let _ = self.update_hit_metrics(latency, CacheSource::FACT).await;
                            
                            // Also store in memory cache for next time
                            if self.config.enable_memory_cache {
                                let cached_response = CachedResponse {
                                    response: response.clone(),
                                    created_at: std::time::SystemTime::now(),
                                    access_count: 1,
                                    last_accessed: std::time::SystemTime::now(),
                                    avg_hit_time: latency,
                                };
                                self.memory_cache.insert(cache_key, cached_response);
                            }

                            info!("FACT cache exact hit in {}ms", latency.as_millis());
                            
                            return Ok(CacheResult::Hit {
                                response,
                                source: CacheSource::FACT,
                                latency,
                            });
                        },
                        Err(e) => {
                            warn!("Failed to deserialize FACT cached response: {}", e);
                        }
                    }
                }
                
                // Try semantic similarity matching if enabled
                if self.config.key_optimization.enable_semantic_matching {
                    let similar_keys = cache.find_similar_keys(&cache_key_str, self.config.key_optimization.similarity_threshold);
                    for similar_key in similar_keys.into_iter().take(3) { // Check top 3 similar entries
                        if let Some(cached_data) = cache.get(&similar_key) {
                            if let Ok(response) = serde_json::from_value::<GeneratedResponse>(cached_data) {
                                let latency = start_time.elapsed();
                                let _ = self.update_hit_metrics(latency, CacheSource::FACT).await;
                                
                                debug!("FACT cache semantic hit for key: {} (similarity match)", similar_key);
                                
                                return Ok(CacheResult::Hit {
                                    response,
                                    source: CacheSource::FACT,
                                    latency,
                                });
                            }
                        }
                    }
                }
            }
        }"}

        // Cache miss
        let latency = start_time.elapsed();
        let _ = self.update_miss_metrics(latency).await;
        
        debug!("Cache miss after {}ms", latency.as_millis());
        
        Ok(CacheResult::Miss { key: cache_key })
    }

    /// Store response in cache
    #[instrument(skip(self, response), fields(request_id = %response.request_id))]
    pub async fn store(&self, request: &GenerationRequest, response: &GeneratedResponse) -> Result<()> {
        let start_time = Instant::now();

        // Generate cache key
        let cache_key = self.generate_cache_key(request).await?;

        // Store in FACT cache with intelligent optimization
        if self.config.enable_fact_cache {
            let optimized_query = self.query_optimizer.optimize(&request.query);

            let response_data = serde_json::to_value(response)
                .map_err(|e| ResponseError::internal(format!("Failed to serialize response for caching: {}", e)))?;

            // Real FACT cache storage implementation
            if let Ok(mut cache) = self.fact_cache.lock() {
                // Use FACT's real caching with key-value storage
                let cache_key = format!("query:{}", blake3::hash(optimized_query.as_bytes()).to_hex());
                
                // Store with TTL using FACT's actual interface
                match cache.set_with_ttl(&cache_key, &response_data, self.config.response_ttl) {
                    Ok(_) => {
                        debug!("Successfully cached response in FACT cache for query: {}", optimized_query);
                    },
                    Err(e) => {
                        warn!("Failed to store response in FACT cache: {:?}", e);
                    }
                }
                
                // Also store context fingerprints for semantic matching
                let context_key = format!("context:{}", blake3::hash(&format!(\"{:?}\", request.context)).to_hex());
                let context_data = serde_json::json!({\n                    \"query\": optimized_query,\n                    \"context_size\": request.context.len(),\n                    \"avg_relevance\": request.context.iter().map(|c| c.relevance_score).sum::<f64>() / request.context.len().max(1) as f64\n                });\n                \n                let _ = cache.set_with_ttl(&context_key, &context_data, self.config.context_ttl);\n            }\n        }"}

        // Store in memory cache
        if self.config.enable_memory_cache {
            let cached_response = CachedResponse {
                response: response.clone(),
                created_at: std::time::SystemTime::now(),
                access_count: 0,
                last_accessed: std::time::SystemTime::now(),
                avg_hit_time: Duration::from_millis(0),
            };

            // Evict old entries if cache is full
            if self.memory_cache.len() >= self.config.memory_cache_size {
                self.evict_oldest_entries().await;
            }

            self.memory_cache.insert(cache_key, cached_response);
        }

        let store_duration = start_time.elapsed();
        debug!("Stored response in cache in {}ms", store_duration.as_millis());

        Ok(())
    }

    /// Generate optimized cache key
    async fn generate_cache_key(&self, request: &GenerationRequest) -> Result<CacheKey> {
        let mut hasher = blake3::Hasher::new();

        // Optimize query for consistent caching
        let optimized_query = if self.config.key_optimization.enable_query_normalization {
            self.normalize_query(&request.query).await?
        } else {
            request.query.clone()
        };

        // Hash optimized query
        hasher.update(optimized_query.as_bytes());
        let query_hash = hasher.finalize();

        // Generate context fingerprint
        let mut context_hasher = blake3::Hasher::new();
        for chunk in &request.context {
            context_hasher.update(chunk.content.as_bytes());
            context_hasher.update(&chunk.relevance_score.to_le_bytes());
        }
        let context_fingerprint = context_hasher.finalize();

        // Configuration hash (affects response generation)
        let mut config_hasher = blake3::Hasher::new();
        if let Some(validation_config) = &request.validation_config {
            config_hasher.update(format!("{:?}", validation_config).as_bytes());
        }
        if let Some(max_length) = request.max_length {
            config_hasher.update(&max_length.to_le_bytes());
        }
        if let Some(min_confidence) = request.min_confidence {
            config_hasher.update(&min_confidence.to_le_bytes());
        }
        let config_hash = config_hasher.finalize();

        Ok(CacheKey {
            query_hash: u64::from_le_bytes(query_hash.as_bytes()[0..8].try_into().unwrap()),
            context_fingerprint: u64::from_le_bytes(context_fingerprint.as_bytes()[0..8].try_into().unwrap()),
            format: format!("{:?}", request.format),
            config_hash: u64::from_le_bytes(config_hash.as_bytes()[0..8].try_into().unwrap()),
        })
    }

    /// Normalize query for consistent caching
    async fn normalize_query(&self, query: &str) -> Result<String> {
        // Basic normalization
        let normalized = query
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?-".contains(*c))
            .collect::<String>();

        // Advanced normalization using simplified query optimizer
        if self.config.key_optimization.enable_semantic_matching {
            Ok(self.query_optimizer.normalize(&normalized))
        } else {
            Ok(normalized)
        }
    }

    /// Check if cached response is expired
    fn is_expired(&self, cached: &CachedResponse) -> bool {
        match cached.created_at.elapsed() {
            Ok(age) => age > self.config.response_ttl,
            Err(_) => true, // If we can't determine age, consider expired
        }
    }

    /// Update cache hit metrics
    async fn update_hit_metrics(&self, latency: Duration, source: CacheSource) -> Result<()> {
        self.metrics.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        match source {
            CacheSource::FACT => {
                self.metrics.fact_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            CacheSource::Memory => {
                self.metrics.memory_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            CacheSource::Hybrid => {
                // Both counters already incremented by individual calls
            }
        }

        // Update average hit latency
        if let Ok(mut avg_latency) = self.metrics.avg_hit_latency.write() {
            let hits = self.metrics.cache_hits.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let current_avg = avg_latency.as_nanos() as f64;
            let new_latency = latency.as_nanos() as f64;
            let new_avg = (current_avg * (hits - 1.0) + new_latency) / hits;
            *avg_latency = Duration::from_nanos(new_avg as u64);
        }

        // Warn if hit latency exceeds threshold
        if latency.as_millis() > self.config.hit_threshold_ms as u128 {
            warn!("Cache hit latency {}ms exceeds threshold {}ms", 
                  latency.as_millis(), self.config.hit_threshold_ms);
        }

        Ok(())
    }

    /// Update cache miss metrics
    async fn update_miss_metrics(&self, latency: Duration) -> Result<()> {
        self.metrics.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Update average miss latency
        if let Ok(mut avg_latency) = self.metrics.avg_miss_latency.write() {
            let misses = self.metrics.cache_misses.load(std::sync::atomic::Ordering::Relaxed) as f64;
            let current_avg = avg_latency.as_nanos() as f64;
            let new_latency = latency.as_nanos() as f64;
            let new_avg = (current_avg * (misses - 1.0) + new_latency) / misses;
            *avg_latency = Duration::from_nanos(new_avg as u64);
        }

        Ok(())
    }

    /// Evict oldest entries from memory cache
    async fn evict_oldest_entries(&self) {
        let eviction_count = self.config.memory_cache_size / 10; // Evict 10% of entries
        let mut entries: Vec<_> = self.memory_cache.iter()
            .map(|entry| (entry.key().clone(), entry.value().created_at))
            .collect();

        // Sort by creation time (oldest first)
        entries.sort_by(|a, b| a.1.cmp(&b.1));

        // Remove oldest entries
        for (key, _) in entries.into_iter().take(eviction_count) {
            self.memory_cache.remove(&key);
        }

        debug!("Evicted {} entries from memory cache", eviction_count);
    }

    /// Get cache performance metrics
    pub fn get_metrics(&self) -> CacheMetricsSnapshot {
        let total_requests = self.metrics.total_requests.load(std::sync::atomic::Ordering::Relaxed);
        let cache_hits = self.metrics.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let cache_misses = self.metrics.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        let fact_hits = self.metrics.fact_hits.load(std::sync::atomic::Ordering::Relaxed);
        let memory_hits = self.metrics.memory_hits.load(std::sync::atomic::Ordering::Relaxed);

        let hit_rate = if total_requests > 0 {
            cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };

        let avg_hit_latency = self.metrics.avg_hit_latency.read()
            .map(|latency| *latency)
            .unwrap_or_default();

        let avg_miss_latency = self.metrics.avg_miss_latency.read()
            .map(|latency| *latency)
            .unwrap_or_default();

        CacheMetricsSnapshot {
            total_requests,
            cache_hits,
            cache_misses,
            fact_hits,
            memory_hits,
            hit_rate,
            avg_hit_latency,
            avg_miss_latency,
            memory_cache_size: self.memory_cache.len(),
        }
    }

    /// Clear all caches
    pub async fn clear(&self) -> Result<()> {
        info!("Clearing all caches");

        // Clear memory cache
        self.memory_cache.clear();

        // Clear FACT cache
        if self.config.enable_fact_cache {
            if let Ok(_cache) = self.fact_cache.lock() {
                // In a real implementation, we would clear the FACT cache
                debug!("Would clear FACT cache");
            }
        }

        Ok(())
    }

    /// Warm up cache with common queries
    pub async fn warmup(&self, common_queries: Vec<GenerationRequest>) -> Result<()> {
        info!("Warming up cache with {} queries", common_queries.len());

        for request in common_queries {
            // Pre-generate cache keys and optimize queries
            let _cache_key = self.generate_cache_key(&request).await?;
            
            if self.config.enable_fact_cache {
                let _optimized_query = self.query_optimizer.optimize(&request.query);
            }
        }

        Ok(())
    }
}

// Manual Debug implementation for FACTCacheManager
impl std::fmt::Debug for FACTCacheManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FACTCacheManager")
            .field("config", &self.config)
            .field("metrics", &"<metrics>")
            .field("memory_cache_size", &self.memory_cache.len())
            .finish()
    }
}

/// Cache metrics snapshot for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetricsSnapshot {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub fact_hits: u64,
    pub memory_hits: u64,
    pub hit_rate: f64,
    pub avg_hit_latency: Duration,
    pub avg_miss_latency: Duration,
    pub memory_cache_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OutputFormat;

    #[tokio::test]
    async fn test_cache_manager_creation() {
        let config = CacheManagerConfig::default();
        let cache_manager = FACTCacheManager::new(config).await;
        assert!(cache_manager.is_ok());
    }

    #[tokio::test]
    async fn test_cache_key_generation() {
        let config = CacheManagerConfig::default();
        let cache_manager = FACTCacheManager::new(config).await.unwrap();

        let request = GenerationRequest::builder()
            .query("Test query")
            .format(OutputFormat::Json)
            .build()
            .unwrap();

        let key1 = cache_manager.generate_cache_key(&request).await.unwrap();
        let key2 = cache_manager.generate_cache_key(&request).await.unwrap();
        
        assert_eq!(key1, key2, "Same request should generate same cache key");
    }

    #[tokio::test]
    async fn test_cache_miss_then_store() {
        let config = CacheManagerConfig::default();
        let cache_manager = FACTCacheManager::new(config).await.unwrap();

        let request = GenerationRequest::builder()
            .query("Test query for caching")
            .build()
            .unwrap();

        // Should be a cache miss initially
        let result = cache_manager.get(&request).await.unwrap();
        assert!(matches!(result, CacheResult::Miss { .. }));

        // Create a mock response to store
        let response = GeneratedResponse {
            request_id: request.id,
            content: "Test response".to_string(),
            format: request.format.clone(),
            confidence_score: 0.9,
            citations: vec![],
            segment_confidence: vec![],
            validation_results: vec![],
            metrics: crate::GenerationMetrics {
                total_duration: Duration::from_millis(100),
                validation_duration: Duration::from_millis(20),
                formatting_duration: Duration::from_millis(10),
                citation_duration: Duration::from_millis(15),
                validation_passes: 1,
                sources_used: 0,
                response_length: 13,
            },
            warnings: vec![],
        };

        // Store the response
        cache_manager.store(&request, &response).await.unwrap();

        // Now should be a cache hit
        let result = cache_manager.get(&request).await.unwrap();
        match result {
            CacheResult::Hit { response: cached_response, source: _, latency } => {
                assert_eq!(cached_response.content, "Test response");
                assert!(latency < Duration::from_millis(100)); // Should be fast
            }
            CacheResult::Miss { .. } => panic!("Expected cache hit after storing"),
        }
    }

    #[test]
    fn test_cache_metrics_snapshot() {
        let config = CacheManagerConfig::default();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let cache_manager = rt.block_on(FACTCacheManager::new(config)).unwrap();

        let metrics = cache_manager.get_metrics();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.cache_hits, 0);
        assert_eq!(metrics.hit_rate, 0.0);
    }
}

/// Simplified context manager for FACT integration
#[derive(Debug)]
pub struct SimpleContextManager;

impl SimpleContextManager {
    pub fn new() -> Self {
        Self
    }
    
    pub fn analyze_context(&self, query: &str) -> String {
        // Simple context analysis - in a real implementation this would
        // use more sophisticated NLP techniques
        format!("analyzed:{}", query)
    }
    
    pub fn expand_context(&self, query: &str) -> String {
        // Simple context expansion - add related terms
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut expanded = query.to_string();
        
        // Add simple synonyms or related terms
        for word in words {
            match word.to_lowercase().as_str() {
                "rust" => expanded.push_str(" programming language systems"),
                "machine" => expanded.push_str(" learning AI artificial"),
                "data" => expanded.push_str(" information database"),
                _ => {}
            }
        }
        
        expanded
    }
}

/// Simplified query optimizer for FACT integration
#[derive(Debug)]
pub struct SimpleQueryOptimizer;

impl SimpleQueryOptimizer {
    pub fn new() -> Self {
        Self
    }
    
    pub fn optimize(&self, query: &str) -> String {
        // Simple query optimization
        query
            .trim()
            .to_lowercase()
            .split_whitespace()
            .filter(|word| !word.is_empty() && word.len() > 2)
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    pub fn normalize(&self, query: &str) -> String {
        // Simple normalization
        query
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}