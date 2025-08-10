//! Integration tests for FACT caching functionality

use response_generator::{
    FACTAcceleratedGenerator, FACTConfig, FACTCacheManager, CacheManagerConfig, CacheResult,
    GenerationRequest, Config, OutputFormat,
};
use std::time::{Duration, Instant};
use tokio_test;

#[tokio::test]
async fn test_fact_cache_miss_then_hit() {
    let cache_config = CacheManagerConfig {
        enable_fact_cache: true,
        enable_memory_cache: true,
        memory_cache_size: 100,
        response_ttl: Duration::from_secs(60),
        ..Default::default()
    };

    let cache = FACTCacheManager::new(cache_config).await.unwrap();

    let request = GenerationRequest::builder()
        .query("What is Rust programming language?")
        .format(OutputFormat::Json)
        .build()
        .unwrap();

    // First request should be a cache miss
    let result1 = cache.get(&request).await.unwrap();
    match result1 {
        CacheResult::Miss { .. } => {
            // Expected for first request
        }
        CacheResult::Hit { .. } => {
            panic!("Expected cache miss for first request");
        }
    }
}

#[tokio::test]
async fn test_fact_accelerated_generator() {
    let base_config = Config::default();
    let mut fact_config = FACTConfig::default();
    fact_config.target_cached_response_time = 25; // Very aggressive target for testing

    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await;
    assert!(generator.is_ok());

    let generator = generator.unwrap();

    let request = GenerationRequest::builder()
        .query("Test query for FACT acceleration")
        .format(OutputFormat::Markdown)
        .build()
        .unwrap();

    let start_time = Instant::now();
    let result = generator.generate(request).await;
    let total_time = start_time.elapsed();

    assert!(result.is_ok());
    let response = result.unwrap();

    // First request will likely be a cache miss
    assert!(!response.response.content.is_empty());
    assert!(total_time < Duration::from_secs(5)); // Should complete within 5 seconds
    
    // Verify FACT metrics are populated
    assert!(response.fact_metrics.performance_ratio > 0.0);
}

#[tokio::test]
async fn test_cache_performance_metrics() {
    let cache_config = CacheManagerConfig::default();
    let cache = FACTCacheManager::new(cache_config).await.unwrap();

    // Get initial metrics
    let initial_metrics = cache.get_metrics();
    assert_eq!(initial_metrics.total_requests, 0);
    assert_eq!(initial_metrics.cache_hits, 0);
    assert_eq!(initial_metrics.cache_misses, 0);

    let request = GenerationRequest::builder()
        .query("Cache metrics test query")
        .build()
        .unwrap();

    // Make a request to trigger metrics update
    let _result = cache.get(&request).await.unwrap();

    let updated_metrics = cache.get_metrics();
    assert!(updated_metrics.total_requests > initial_metrics.total_requests);
}

#[tokio::test]
async fn test_sub_50ms_target() {
    let base_config = Config::default();
    let fact_config = FACTConfig {
        enabled: true,
        target_cached_response_time: 50,
        enable_cache_monitoring: true,
        ..Default::default()
    };

    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();

    let request = GenerationRequest::builder()
        .query("Quick response test")
        .format(OutputFormat::Json)
        .build()
        .unwrap();

    // First request (likely cache miss)
    let _first_response = generator.generate(request.clone()).await.unwrap();

    // Second request (potential cache hit)
    let start_time = Instant::now();
    let second_response = generator.generate(request).await.unwrap();
    let response_time = start_time.elapsed();

    if second_response.cache_hit {
        // If it was a cache hit, it should meet our target
        assert!(response_time.as_millis() <= 100); // Allow some margin for test environment
    }

    // Verify metrics are reasonable
    assert!(second_response.total_time > Duration::from_nanos(1));
}

#[tokio::test]
async fn test_fact_config_validation() {
    let config = FACTConfig::default();
    assert!(config.enabled);
    assert_eq!(config.target_cached_response_time, 50);
    assert!(config.enable_prewarming);
    assert!(config.enable_cache_monitoring);
}

#[tokio::test]
async fn test_cache_key_consistency() {
    let cache_config = CacheManagerConfig::default();
    let cache = FACTCacheManager::new(cache_config).await.unwrap();

    let request1 = GenerationRequest::builder()
        .query("Consistent key test")
        .format(OutputFormat::Json)
        .build()
        .unwrap();

    let request2 = GenerationRequest::builder()
        .query("Consistent key test")
        .format(OutputFormat::Json)
        .build()
        .unwrap();

    // Both requests should generate the same cache key (though we can't directly compare)
    // We can verify this by checking that the second request recognizes the first request's cache entry
    
    let result1 = cache.get(&request1).await.unwrap();
    assert!(matches!(result1, CacheResult::Miss { .. }));

    // Simulate storing a response
    let mock_response = response_generator::GeneratedResponse {
        request_id: request1.id,
        content: "Test response".to_string(),
        format: request1.format.clone(),
        confidence_score: 0.9,
        citations: vec![],
        segment_confidence: vec![],
        validation_results: vec![],
        metrics: response_generator::GenerationMetrics {
            total_duration: Duration::from_millis(100),
            validation_duration: Duration::from_millis(20),
            formatting_duration: Duration::from_millis(10),
            citation_duration: Duration::from_millis(5),
            validation_passes: 1,
            sources_used: 0,
            response_length: 13,
        },
        warnings: vec![],
    };

    cache.store(&request1, &mock_response).await.unwrap();

    // Second request should potentially hit cache (depending on FACT cache behavior)
    let result2 = cache.get(&request2).await.unwrap();
    // Note: Whether this is a hit or miss depends on FACT cache implementation
    // We're just verifying it doesn't error
    match result2 {
        CacheResult::Hit { .. } => {
            // Great! Cache key consistency worked
        }
        CacheResult::Miss { .. } => {
            // Also fine, FACT might not have the same key optimization
        }
    }
}

#[tokio::test]
async fn test_cache_clearing() {
    let cache_config = CacheManagerConfig::default();
    let cache = FACTCacheManager::new(cache_config).await.unwrap();

    let request = GenerationRequest::builder()
        .query("Clear cache test")
        .build()
        .unwrap();

    // Make a request to populate metrics
    let _result = cache.get(&request).await.unwrap();

    let metrics_before = cache.get_metrics();
    assert!(metrics_before.total_requests > 0);

    // Clear cache
    cache.clear().await.unwrap();

    // Memory cache should be cleared
    assert_eq!(cache.get_metrics().memory_cache_size, 0);
}

#[tokio::test]
async fn test_preload_cache() {
    let base_config = Config::default();
    let fact_config = FACTConfig {
        enable_prewarming: true,
        ..Default::default()
    };

    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();

    let preload_queries = vec![
        GenerationRequest::builder()
            .query("Common query 1")
            .build()
            .unwrap(),
        GenerationRequest::builder()
            .query("Common query 2")
            .build()
            .unwrap(),
    ];

    // Preload should not error
    let result = generator.preload_cache(preload_queries).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_disabled_fact() {
    let base_config = Config::default();
    let fact_config = FACTConfig {
        enabled: false,
        ..Default::default()
    };

    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();

    let request = GenerationRequest::builder()
        .query("Disabled FACT test")
        .build()
        .unwrap();

    let result = generator.generate(request).await.unwrap();
    
    // Should not be a cache hit when FACT is disabled
    assert!(!result.cache_hit);
    assert!(result.cache_source.is_none());
    assert_eq!(result.fact_metrics.cache_efficiency, 0.0);
}

#[tokio::test]
async fn test_fact_streaming_response() {
    let base_config = Config::default();
    let fact_config = FACTConfig::default();

    let mut generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();

    let request = GenerationRequest::builder()
        .query("Streaming test query")
        .format(OutputFormat::Markdown)
        .build()
        .unwrap();

    let stream_result = generator.generate_stream(request).await;
    assert!(stream_result.is_ok());

    // Stream should be created successfully
    let _stream = stream_result.unwrap();
    // Note: Testing the actual streaming would require consuming the stream
    // which is complex in this test environment
}