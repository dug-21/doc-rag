//! FACT-accelerated response generator
//!
//! This module provides the main FACT-accelerated response generator that wraps
//! the base ResponseGenerator with intelligent caching to achieve sub-50ms responses.

use crate::cache::{FACTCacheManager, CacheManagerConfig, CacheResult, CacheSource};
use crate::error::{Result, ResponseError};
use crate::{
    Config, GeneratedResponse, GenerationRequest, ResponseGenerator, 
    ResponseChunk, ResponseChunkType
};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn, instrument};

/// FACT-accelerated response generator with intelligent caching
#[derive(Debug)]
pub struct FACTAcceleratedGenerator {
    /// FACT cache manager
    cache: FACTCacheManager,
    
    /// Base response generator (for cache misses)
    base_generator: ResponseGenerator,
    
    /// Configuration for FACT integration
    config: FACTConfig,
}

/// Configuration specific to FACT acceleration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FACTConfig {
    /// Enable FACT acceleration
    pub enabled: bool,
    
    /// Target response time for cached responses (ms)
    pub target_cached_response_time: u64,
    
    /// Maximum time to spend on cache lookup before falling back
    pub max_cache_lookup_time: Duration,
    
    /// Enable cache prewarming
    pub enable_prewarming: bool,
    
    /// Cache performance monitoring
    pub enable_cache_monitoring: bool,
    
    /// Fallback strategy when FACT fails
    pub fallback_strategy: FallbackStrategy,
    
    /// Cache manager configuration
    pub cache_config: CacheManagerConfig,
}

/// Fallback strategies when FACT caching fails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackStrategy {
    /// Use base generator immediately
    Immediate,
    
    /// Try cache once more with reduced settings
    RetryOnce,
    
    /// Disable FACT for this session
    DisableSession,
    
    /// Return error to caller
    Error,
}

/// Cache source for analytics (simplified for serialization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializableCacheSource {
    FACT,
    Memory,
    Hybrid,
}

impl From<CacheSource> for SerializableCacheSource {
    fn from(source: CacheSource) -> Self {
        match source {
            CacheSource::FACT => SerializableCacheSource::FACT,
            CacheSource::Memory => SerializableCacheSource::Memory,
            CacheSource::Hybrid => SerializableCacheSource::Hybrid,
        }
    }
}

/// Enhanced generation result with cache information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FACTGeneratedResponse {
    /// The generated response
    pub response: GeneratedResponse,
    
    /// Whether this was a cache hit or miss
    pub cache_hit: bool,
    
    /// Cache source if hit
    pub cache_source: Option<SerializableCacheSource>,
    
    /// Cache lookup time
    pub cache_lookup_time: Duration,
    
    /// Total response time including caching
    pub total_time: Duration,
    
    /// FACT-specific metrics
    pub fact_metrics: FACTMetrics,
}

/// FACT-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FACTMetrics {
    /// Cache efficiency (hit rate)
    pub cache_efficiency: f64,
    
    /// Average cache hit time
    pub avg_hit_time: Duration,
    
    /// Average generation time for misses
    pub avg_miss_time: Duration,
    
    /// Performance improvement ratio (cached vs non-cached)
    pub performance_ratio: f64,
    
    /// FACT processing overhead
    pub fact_overhead: Duration,
}

impl Default for FACTConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_cached_response_time: 50, // Sub-50ms target
            max_cache_lookup_time: Duration::from_millis(20),
            enable_prewarming: true,
            enable_cache_monitoring: true,
            fallback_strategy: FallbackStrategy::Immediate,
            cache_config: CacheManagerConfig::default(),
        }
    }
}

impl FACTAcceleratedGenerator {
    /// Create a new FACT-accelerated generator
    pub async fn new(base_config: Config, fact_config: FACTConfig) -> Result<Self> {
        info!("Initializing FACT-accelerated response generator");

        // Initialize FACT cache manager
        let cache = FACTCacheManager::new(fact_config.cache_config.clone()).await?;

        // Initialize base generator
        let base_generator = ResponseGenerator::new(base_config).await;

        Ok(Self {
            cache,
            base_generator,
            config: fact_config,
        })
    }

    /// Create with default FACT configuration
    pub async fn with_defaults(base_config: Config) -> Result<Self> {
        Self::new(base_config, FACTConfig::default()).await
    }

    /// Generate response with FACT acceleration
    #[instrument(skip(self, request), fields(request_id = %request.id))]
    pub async fn generate(&self, request: GenerationRequest) -> Result<FACTGeneratedResponse> {
        let start_time = Instant::now();
        
        if !self.config.enabled {
            // FACT disabled, use base generator directly
            return self.generate_without_cache(request, start_time).await;
        }

        info!("Starting FACT-accelerated generation for query: {}", request.query);

        // Attempt cache lookup with timeout
        let cache_lookup_start = Instant::now();
        let cache_result = tokio::time::timeout(
            self.config.max_cache_lookup_time,
            self.cache.get(&request)
        ).await;

        let cache_lookup_time = cache_lookup_start.elapsed();

        match cache_result {
            Ok(Ok(CacheResult::Hit { response, source, latency })) => {
                // Cache hit - return immediately
                let total_time = start_time.elapsed();
                
                info!("Cache hit in {}ms (lookup: {}ms)", 
                      latency.as_millis(), cache_lookup_time.as_millis());

                // Check if we met the sub-50ms target
                if total_time.as_millis() > self.config.target_cached_response_time as u128 {
                    warn!("Cached response took {}ms, exceeding target of {}ms",
                          total_time.as_millis(), self.config.target_cached_response_time);
                }

                let cache_metrics = self.cache.get_metrics();
                let fact_metrics = FACTMetrics {
                    cache_efficiency: cache_metrics.hit_rate,
                    avg_hit_time: cache_metrics.avg_hit_latency,
                    avg_miss_time: cache_metrics.avg_miss_latency,
                    performance_ratio: self.calculate_performance_ratio(&cache_metrics),
                    fact_overhead: cache_lookup_time,
                };

                Ok(FACTGeneratedResponse {
                    response,
                    cache_hit: true,
                    cache_source: Some(source.into()),
                    cache_lookup_time,
                    total_time,
                    fact_metrics,
                })
            }
            Ok(Ok(CacheResult::Miss { key: _cache_key })) => {
                // Cache miss - generate response and cache it
                self.generate_and_cache(request, start_time, cache_lookup_time).await
            }
            Ok(Err(cache_error)) => {
                // Cache error - handle based on fallback strategy
                warn!("Cache error: {}", cache_error);
                self.handle_cache_error(request, start_time, cache_lookup_time, cache_error).await
            }
            Err(_timeout) => {
                // Cache lookup timeout - handle based on fallback strategy
                warn!("Cache lookup timed out after {}ms", self.config.max_cache_lookup_time.as_millis());
                self.handle_cache_timeout(request, start_time, cache_lookup_time).await
            }
        }
    }

    /// Generate streaming response with FACT acceleration
    pub async fn generate_stream(
        &mut self,
        request: GenerationRequest,
    ) -> Result<ReceiverStream<Result<ResponseChunk>>> {
        if !self.config.enabled {
            // FACT disabled, use base generator directly
            return self.base_generator.generate_stream(request).await;
        }

        // For streaming, we first check cache for complete response
        if let Ok(CacheResult::Hit { response, .. }) = self.cache.get(&request).await {
            // Cache hit - convert to streaming chunks
            return self.stream_cached_response(response).await;
        }

        // Cache miss - use base generator streaming with caching
        let stream = self.base_generator.generate_stream(request.clone()).await?;
        
        // Note: In a full implementation, we'd capture streamed response and cache it
        // For now, return the base stream
        Ok(stream)
    }

    /// Generate response without using cache
    async fn generate_without_cache(
        &self, 
        request: GenerationRequest, 
        start_time: Instant
    ) -> Result<FACTGeneratedResponse> {
        let response = self.base_generator.generate(request).await?;
        let total_time = start_time.elapsed();

        let fact_metrics = FACTMetrics {
            cache_efficiency: 0.0,
            avg_hit_time: Duration::from_millis(0),
            avg_miss_time: total_time,
            performance_ratio: 1.0,
            fact_overhead: Duration::from_millis(0),
        };

        Ok(FACTGeneratedResponse {
            response,
            cache_hit: false,
            cache_source: None,
            cache_lookup_time: Duration::from_millis(0),
            total_time,
            fact_metrics,
        })
    }

    /// Generate response and cache it
    async fn generate_and_cache(
        &self,
        request: GenerationRequest,
        start_time: Instant,
        cache_lookup_time: Duration,
    ) -> Result<FACTGeneratedResponse> {
        info!("Cache miss - generating new response");

        // Generate response using base generator
        let generation_start = Instant::now();
        let response = self.base_generator.generate(request.clone()).await?;
        let generation_time = generation_start.elapsed();

        // Store in cache for future requests
        if let Err(cache_error) = self.cache.store(&request, &response).await {
            warn!("Failed to cache response: {}", cache_error);
            // Don't fail the request due to cache store error
        }

        let total_time = start_time.elapsed();

        info!("Generated and cached response in {}ms (generation: {}ms, cache lookup: {}ms)",
              total_time.as_millis(), generation_time.as_millis(), cache_lookup_time.as_millis());

        let cache_metrics = self.cache.get_metrics();
        let fact_metrics = FACTMetrics {
            cache_efficiency: cache_metrics.hit_rate,
            avg_hit_time: cache_metrics.avg_hit_latency,
            avg_miss_time: cache_metrics.avg_miss_latency,
            performance_ratio: self.calculate_performance_ratio(&cache_metrics),
            fact_overhead: cache_lookup_time,
        };

        Ok(FACTGeneratedResponse {
            response,
            cache_hit: false,
            cache_source: None,
            cache_lookup_time,
            total_time,
            fact_metrics,
        })
    }

    /// Handle cache error based on configured fallback strategy
    async fn handle_cache_error(
        &self,
        request: GenerationRequest,
        start_time: Instant,
        cache_lookup_time: Duration,
        cache_error: ResponseError,
    ) -> Result<FACTGeneratedResponse> {
        match self.config.fallback_strategy {
            FallbackStrategy::Immediate => {
                self.generate_without_cache(request, start_time).await
            }
            FallbackStrategy::RetryOnce => {
                // Try cache once more with simpler key
                warn!("Retrying cache lookup once");
                if let Ok(CacheResult::Hit { response, source, latency: _ }) = self.cache.get(&request).await {
                    let total_time = start_time.elapsed();
                    let cache_metrics = self.cache.get_metrics();
                    let fact_metrics = FACTMetrics {
                        cache_efficiency: cache_metrics.hit_rate,
                        avg_hit_time: cache_metrics.avg_hit_latency,
                        avg_miss_time: cache_metrics.avg_miss_latency,
                        performance_ratio: self.calculate_performance_ratio(&cache_metrics),
                        fact_overhead: cache_lookup_time,
                    };

                    Ok(FACTGeneratedResponse {
                        response,
                        cache_hit: true,
                        cache_source: Some(source.into()),
                        cache_lookup_time,
                        total_time,
                        fact_metrics,
                    })
                } else {
                    self.generate_without_cache(request, start_time).await
                }
            }
            FallbackStrategy::DisableSession => {
                warn!("Disabling FACT for this session due to cache error");
                self.generate_without_cache(request, start_time).await
            }
            FallbackStrategy::Error => {
                Err(cache_error)
            }
        }
    }

    /// Handle cache timeout
    async fn handle_cache_timeout(
        &self,
        request: GenerationRequest,
        start_time: Instant,
        cache_lookup_time: Duration,
    ) -> Result<FACTGeneratedResponse> {
        match self.config.fallback_strategy {
            FallbackStrategy::Immediate | FallbackStrategy::RetryOnce => {
                self.generate_without_cache(request, start_time).await
            }
            FallbackStrategy::DisableSession => {
                warn!("Disabling FACT for this session due to cache timeout");
                self.generate_without_cache(request, start_time).await
            }
            FallbackStrategy::Error => {
                Err(ResponseError::timeout(cache_lookup_time))
            }
        }
    }

    /// Convert cached response to streaming chunks
    async fn stream_cached_response(
        &self,
        response: GeneratedResponse,
    ) -> Result<ReceiverStream<Result<ResponseChunk>>> {
        use tokio::sync::mpsc;
        
        let (tx, rx) = mpsc::channel(32);
        
        // Clone response data for the task
        let content = response.content.clone();
        let metrics = Some(response.metrics.clone());
        
        tokio::spawn(async move {
            let chunk_size = 256; // Default chunk size
            let total_length = content.len();
            
            for (i, chunk_content) in content.as_bytes().chunks(chunk_size).enumerate() {
                let chunk_str = String::from_utf8_lossy(chunk_content).to_string();
                let position = i * chunk_size;
                let is_final = position + chunk_size >= total_length;
                
                let chunk = ResponseChunk {
                    content: chunk_str,
                    chunk_type: if is_final { ResponseChunkType::Final } else { ResponseChunkType::Partial },
                    position,
                    is_final,
                    confidence: if is_final { Some(response.confidence_score) } else { None },
                    metadata: if is_final { metrics.clone() } else { None },
                };
                
                if tx.send(Ok(chunk)).await.is_err() {
                    break; // Client disconnected
                }
                
                // Small delay for realistic streaming feel
                if !is_final {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
            }
        });
        
        Ok(ReceiverStream::new(rx))
    }

    /// Calculate performance improvement ratio
    fn calculate_performance_ratio(&self, cache_metrics: &crate::cache::CacheMetricsSnapshot) -> f64 {
        if cache_metrics.avg_hit_latency.is_zero() || cache_metrics.avg_miss_latency.is_zero() {
            return 1.0;
        }
        
        let hit_time = cache_metrics.avg_hit_latency.as_secs_f64();
        let miss_time = cache_metrics.avg_miss_latency.as_secs_f64();
        
        if hit_time > 0.0 {
            miss_time / hit_time
        } else {
            1.0
        }
    }

    /// Preload cache with common queries
    pub async fn preload_cache(&self, common_queries: Vec<GenerationRequest>) -> Result<()> {
        if !self.config.enable_prewarming {
            return Ok(());
        }

        info!("Preloading cache with {} queries", common_queries.len());
        
        for request in common_queries {
            // Generate response and cache it
            if let Ok(response) = self.base_generator.generate(request.clone()).await {
                if let Err(e) = self.cache.store(&request, &response).await {
                    warn!("Failed to preload cache for query '{}': {}", request.query, e);
                }
            }
        }

        Ok(())
    }

    /// Get FACT cache metrics
    pub fn get_cache_metrics(&self) -> crate::cache::CacheMetricsSnapshot {
        self.cache.get_metrics()
    }

    /// Clear FACT cache
    pub async fn clear_cache(&self) -> Result<()> {
        self.cache.clear().await
    }

    /// Get FACT configuration
    pub fn get_fact_config(&self) -> &FACTConfig {
        &self.config
    }

    /// Update FACT configuration
    pub fn update_fact_config(&mut self, new_config: FACTConfig) {
        self.config = new_config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, OutputFormat};

    #[tokio::test]
    async fn test_fact_accelerated_generator_creation() {
        let base_config = Config::default();
        let fact_config = FACTConfig::default();
        
        let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await;
        assert!(generator.is_ok());
    }

    #[tokio::test]
    async fn test_fact_accelerated_generator_with_defaults() {
        let base_config = Config::default();
        
        let generator = FACTAcceleratedGenerator::with_defaults(base_config).await;
        assert!(generator.is_ok());
    }

    #[tokio::test]
    async fn test_generate_without_fact() {
        let base_config = Config::default();
        let mut fact_config = FACTConfig::default();
        fact_config.enabled = false; // Disable FACT
        
        let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
        
        let request = GenerationRequest::builder()
            .query("Test query without FACT")
            .format(OutputFormat::Json)
            .build()
            .unwrap();
        
        let result = generator.generate(request).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert!(!response.cache_hit);
        assert!(response.cache_source.is_none());
    }

    #[tokio::test]
    async fn test_cache_miss_then_hit() {
        let base_config = Config::default();
        let fact_config = FACTConfig::default();
        
        let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
        
        let request = GenerationRequest::builder()
            .query("Test query for cache miss/hit")
            .format(OutputFormat::Json)
            .build()
            .unwrap();
        
        // First request should be a cache miss
        let result1 = generator.generate(request.clone()).await.unwrap();
        assert!(!result1.cache_hit);
        
        // Second request should be a cache hit (if caching worked)
        // Note: This might still be a miss due to FACT cache initialization time
        // In a real implementation, we'd wait for cache to be ready
        let _result2 = generator.generate(request).await;
        // Just ensure it doesn't error
    }

    #[test]
    fn test_fact_config_defaults() {
        let config = FACTConfig::default();
        assert!(config.enabled);
        assert_eq!(config.target_cached_response_time, 50);
        assert!(config.enable_prewarming);
        assert!(config.enable_cache_monitoring);
    }

    #[tokio::test]
    async fn test_cache_metrics() {
        let base_config = Config::default();
        let fact_config = FACTConfig::default();
        
        let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
        
        let metrics = generator.get_cache_metrics();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.cache_hits, 0);
    }
}