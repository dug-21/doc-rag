//! Integration tests for FACT functionality
//! 
//! Tests the complete FACT integration including:
//! - Intelligent caching with fact extraction
//! - Citation tracking and source attribution
//! - Performance optimization and sub-50ms response times

use std::time::Duration;
use tokio;
use uuid::Uuid;

// Import the response generator components
use response_generator::{
    ResponseGenerator, GenerationRequest, OutputFormat, Config,
    FACTAcceleratedGenerator, FACTConfig, FACTCacheManager, CacheManagerConfig
};

/// Test FACT cache basic functionality
#[tokio::test]
async fn test_fact_cache_basic_operations() {
    let config = CacheManagerConfig::default();
    let cache_manager = FACTCacheManager::new(config).await.unwrap();

    // Test cache miss for new query
    let request = GenerationRequest::builder()
        .query("What is Rust programming language?")
        .format(OutputFormat::Json)
        .build()
        .unwrap();

    let result = cache_manager.get(&request).await.unwrap();
    assert!(matches!(result, response_generator::CacheResult::Miss { .. }));

    // Get metrics
    let metrics = cache_manager.get_metrics();
    assert_eq!(metrics.total_requests, 1);
    assert_eq!(metrics.cache_hits, 0);
    assert_eq!(metrics.cache_misses, 1);
}

/// Test FACT-accelerated response generation
#[tokio::test]
async fn test_fact_accelerated_generator() {
    let base_config = Config::default();
    let fact_config = FACTConfig::default();
    
    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
    
    let request = GenerationRequest::builder()
        .query("Explain the memory safety features of Rust")
        .format(OutputFormat::Json)
        .build()
        .unwrap();
    
    let start_time = tokio::time::Instant::now();
    let result = generator.generate(request).await.unwrap();
    let generation_time = start_time.elapsed();
    
    // Verify response structure
    assert!(!result.response.content.is_empty());
    assert!(result.response.confidence_score > 0.0);
    assert!(result.response.confidence_score <= 1.0);
    
    // Check FACT metrics
    assert!(result.fact_metrics.cache_efficiency >= 0.0);
    assert!(result.fact_metrics.cache_efficiency <= 1.0);
    
    println!("FACT generation completed in: {:?}", generation_time);
    println!("Cache hit: {}", result.cache_hit);
    println!("Cache source: {:?}", result.cache_source);
}

/// Test cache hit scenario with FACT
#[tokio::test]
async fn test_fact_cache_hit_scenario() {
    let base_config = Config::default();
    let fact_config = FACTConfig::default();
    
    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
    
    let request = GenerationRequest::builder()
        .query("What are the key features of Rust programming?")
        .format(OutputFormat::Json)
        .build()
        .unwrap();
    
    // First request - should be a cache miss
    let result1 = generator.generate(request.clone()).await.unwrap();
    assert!(!result1.cache_hit); // First time should be cache miss
    
    // Second request - should be a cache hit or at least faster
    let start_time = tokio::time::Instant::now();
    let result2 = generator.generate(request).await.unwrap();
    let second_generation_time = start_time.elapsed();
    
    // Verify we got a response
    assert!(!result2.response.content.is_empty());
    
    // If it was a cache hit, it should be very fast
    if result2.cache_hit {
        assert!(second_generation_time < Duration::from_millis(100));
        println!("Cache hit achieved in: {:?}", second_generation_time);
    } else {
        println!("Second request was still a cache miss (cache warming needed)");
    }
}

/// Test FACT cache performance under load
#[tokio::test]
async fn test_fact_cache_performance() {
    let base_config = Config::default();
    let mut fact_config = FACTConfig::default();
    fact_config.target_cached_response_time = 25; // Very aggressive target
    
    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
    
    let queries = vec![
        "What is Rust?",
        "How does Rust ensure memory safety?",
        "What are Rust's performance characteristics?",
        "Explain Rust's ownership system",
        "What is the difference between String and &str in Rust?",
    ];
    
    let mut total_time = Duration::from_millis(0);
    let mut cache_hits = 0;
    
    // Run queries multiple times to test caching
    for iteration in 0..3 {
        println!("=== Iteration {} ===", iteration + 1);
        
        for (i, query) in queries.iter().enumerate() {
            let request = GenerationRequest::builder()
                .query(*query)
                .format(OutputFormat::Json)
                .build()
                .unwrap();
            
            let start_time = tokio::time::Instant::now();
            let result = generator.generate(request).await.unwrap();
            let generation_time = start_time.elapsed();
            
            total_time += generation_time;
            if result.cache_hit {
                cache_hits += 1;
            }
            
            println!("Query {}: {:?} (hit: {})", 
                    i + 1, generation_time, result.cache_hit);
            
            // Verify response quality
            assert!(!result.response.content.is_empty());
            assert!(result.response.confidence_score > 0.0);
        }
    }
    
    let total_queries = queries.len() * 3;
    let avg_time = total_time / total_queries as u32;
    let cache_hit_rate = cache_hits as f64 / total_queries as f64;
    
    println!("=== Performance Summary ===");
    println!("Total queries: {}", total_queries);
    println!("Average time per query: {:?}", avg_time);
    println!("Cache hit rate: {:.2}%", cache_hit_rate * 100.0);
    println!("Total cache hits: {}", cache_hits);
    
    // Performance assertions
    assert!(avg_time < Duration::from_millis(1000)); // Should be reasonable
    
    // After multiple iterations, we should see some cache hits
    if total_queries >= 10 {
        // We expect some caching benefits after running the same queries multiple times
        println!("Cache effectiveness being evaluated...");
    }
}

/// Test FACT semantic similarity matching
#[tokio::test]
async fn test_fact_semantic_similarity() {
    let base_config = Config::default();
    let fact_config = FACTConfig::default();
    
    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
    
    // Original query
    let request1 = GenerationRequest::builder()
        .query("What is Rust programming language used for?")
        .format(OutputFormat::Json)
        .build()
        .unwrap();
    
    // Similar query - should potentially benefit from semantic matching
    let request2 = GenerationRequest::builder()
        .query("What are the applications of Rust programming?")
        .format(OutputFormat::Json)
        .build()
        .unwrap();
    
    // Generate response for first query
    let result1 = generator.generate(request1).await.unwrap();
    assert!(!result1.response.content.is_empty());
    
    // Generate response for similar query
    let start_time = tokio::time::Instant::now();
    let result2 = generator.generate(request2).await.unwrap();
    let second_time = start_time.elapsed();
    
    assert!(!result2.response.content.is_empty());
    
    println!("First query completed");
    println!("Similar query completed in: {:?}", second_time);
    println!("Second query cache hit: {}", result2.cache_hit);
    
    // In a full FACT implementation, semantic similarity might provide cache benefits
    // For now, we just verify both queries work correctly
}

/// Test FACT cache with different output formats
#[tokio::test]
async fn test_fact_cache_different_formats() {
    let base_config = Config::default();
    let fact_config = FACTConfig::default();
    
    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
    
    let query = "Explain Rust's borrowing system";
    
    // Test different output formats
    let formats = vec![OutputFormat::Json, OutputFormat::Markdown, OutputFormat::PlainText];
    
    for format in formats {
        let request = GenerationRequest::builder()
            .query(query)
            .format(format.clone())
            .build()
            .unwrap();
        
        let result = generator.generate(request).await.unwrap();
        
        assert!(!result.response.content.is_empty());
        assert_eq!(result.response.format, format);
        
        println!("Format {:?}: Success (hit: {})", format, result.cache_hit);
    }
}

/// Test FACT cache metrics and monitoring
#[tokio::test]
async fn test_fact_cache_metrics() {
    let base_config = Config::default();
    let fact_config = FACTConfig::default();
    
    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
    
    // Get initial metrics
    let initial_metrics = generator.get_cache_metrics();
    assert_eq!(initial_metrics.total_requests, 0);
    assert_eq!(initial_metrics.cache_hits, 0);
    
    // Make some requests
    for i in 0..5 {
        let request = GenerationRequest::builder()
            .query(&format!("Test query number {}", i))
            .format(OutputFormat::Json)
            .build()
            .unwrap();
        
        let _result = generator.generate(request).await.unwrap();
    }
    
    // Check updated metrics
    let updated_metrics = generator.get_cache_metrics();
    assert!(updated_metrics.total_requests >= 5);
    
    println!("=== FACT Cache Metrics ===");
    println!("Total requests: {}", updated_metrics.total_requests);
    println!("Cache hits: {}", updated_metrics.cache_hits);
    println!("Cache misses: {}", updated_metrics.cache_misses);
    println!("Hit rate: {:.2}%", updated_metrics.hit_rate * 100.0);
    println!("Average hit latency: {:?}", updated_metrics.avg_hit_latency);
    println!("Average miss latency: {:?}", updated_metrics.avg_miss_latency);
}

/// Test FACT preloading and cache warming
#[tokio::test]
async fn test_fact_cache_preloading() {
    let base_config = Config::default();
    let fact_config = FACTConfig::default();
    
    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
    
    // Prepare common queries for preloading
    let common_queries = vec![
        GenerationRequest::builder()
            .query("What is Rust?")
            .format(OutputFormat::Json)
            .build()
            .unwrap(),
        GenerationRequest::builder()
            .query("How does Rust handle memory management?")
            .format(OutputFormat::Json)
            .build()
            .unwrap(),
    ];
    
    // Preload cache
    let preload_result = generator.preload_cache(common_queries.clone()).await;
    assert!(preload_result.is_ok());
    
    // Now test if preloaded queries are faster
    for request in common_queries {
        let start_time = tokio::time::Instant::now();
        let result = generator.generate(request).await.unwrap();
        let response_time = start_time.elapsed();
        
        assert!(!result.response.content.is_empty());
        
        println!("Preloaded query response time: {:?} (hit: {})", 
                response_time, result.cache_hit);
        
        // If cache hit, should be very fast
        if result.cache_hit {
            assert!(response_time < Duration::from_millis(100));
        }
    }
}

/// Integration test for complete FACT workflow
#[tokio::test]
async fn test_complete_fact_workflow() {
    println!("=== Complete FACT Integration Test ===");
    
    let base_config = Config::default();
    let fact_config = FACTConfig::default();
    
    let generator = FACTAcceleratedGenerator::new(base_config, fact_config).await.unwrap();
    
    // Step 1: Clear cache to start fresh
    generator.clear_cache().await.unwrap();
    
    // Step 2: Generate response (cache miss expected)
    let request = GenerationRequest::builder()
        .query("Explain the benefits of Rust for systems programming")
        .format(OutputFormat::Markdown)
        .build()
        .unwrap();
    
    let result1 = generator.generate(request.clone()).await.unwrap();
    assert!(!result1.cache_hit);
    println!("✓ Initial generation (cache miss): {:?}", result1.total_time);
    
    // Step 3: Same query again (cache hit expected)
    let result2 = generator.generate(request.clone()).await.unwrap();
    println!("✓ Second generation (cache hit: {}): {:?}", result2.cache_hit, result2.total_time);
    
    // Step 4: Check that cached response is consistent
    assert_eq!(result1.response.content, result2.response.content);
    assert_eq!(result1.response.format, result2.response.format);
    
    // Step 5: Verify FACT metrics show improvement
    let final_metrics = generator.get_cache_metrics();
    assert!(final_metrics.total_requests >= 2);
    println!("✓ Total requests processed: {}", final_metrics.total_requests);
    println!("✓ Cache hit rate: {:.2}%", final_metrics.hit_rate * 100.0);
    
    // Step 6: Test configuration retrieval
    let fact_config_retrieved = generator.get_fact_config();
    assert!(fact_config_retrieved.enabled);
    assert_eq!(fact_config_retrieved.target_cached_response_time, 50);
    
    println!("✓ FACT integration test completed successfully");
}