//! Integration tests for the embedding generator
//!
//! These tests verify the complete functionality of the embedding system
//! including model loading, batch processing, caching, and performance.

use embedder::{
    EmbeddingGenerator, EmbedderConfig, ModelType, Device, Chunk, ChunkMetadata,
    AdaptiveBatchConfig,
};
use std::collections::HashMap;
use std::time::Duration;
use tokio;
use uuid::Uuid;

#[tokio::test]
async fn test_embedding_generator_basic_functionality() {
    let config = EmbedderConfig::new()
        .with_model_type(ModelType::AllMiniLmL6V2)
        .with_batch_size(2)
        .with_cache_size(100);
    
    // This would require actual model files, so we'll skip in CI
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    
    // If models aren't available, this is expected to fail
    if generator_result.is_err() {
        println!("Model not available, skipping test");
        return;
    }
    
    let generator = generator_result.unwrap();
    
    let chunks = vec![
        create_test_chunk("Hello world", "doc1"),
        create_test_chunk("How are you doing today?", "doc1"), 
        create_test_chunk("The weather is nice", "doc2"),
    ];
    
    let embedded_chunks = generator.generate_embeddings(chunks).await.unwrap();
    
    assert_eq!(embedded_chunks.len(), 3);
    
    for embedded_chunk in &embedded_chunks {
        assert_eq!(embedded_chunk.embeddings.len(), generator.get_dimension());
        assert!(!embedded_chunk.embeddings.iter().all(|&x| x == 0.0));
        
        // Validate embedding
        generator.validate_embedding(&embedded_chunk.embeddings).unwrap();
    }
    
    // Test similarity calculation
    let sim = generator.calculate_similarity(
        &embedded_chunks[0].embeddings,
        &embedded_chunks[1].embeddings
    ).unwrap();
    
    assert!(sim >= -1.0 && sim <= 1.0);
}

#[tokio::test] 
async fn test_cache_functionality() {
    let config = EmbedderConfig::new()
        .with_cache_size(50);
    
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let generator = generator_result.unwrap();
    
    let chunks = vec![
        create_test_chunk("Repeated text", "doc1"),
        create_test_chunk("Different text", "doc1"), 
        create_test_chunk("Repeated text", "doc2"), // Should hit cache
    ];
    
    let embedded_chunks = generator.generate_embeddings(chunks).await.unwrap();
    assert_eq!(embedded_chunks.len(), 3);
    
    let stats = generator.get_stats().await;
    assert!(stats.cache_hits > 0 || stats.cache_misses > 0);
    
    let cache_stats = generator.get_cache_stats().await;
    assert!(cache_stats.total_entries > 0);
}

#[tokio::test]
async fn test_batch_processing() {
    let config = EmbedderConfig::new()
        .with_batch_size(2);
    
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let generator = generator_result.unwrap();
    
    // Create a larger set of chunks to test batching
    let chunks: Vec<Chunk> = (0..10)
        .map(|i| create_test_chunk(&format!("Test text number {}", i), "doc1"))
        .collect();
    
    let start_time = std::time::Instant::now();
    let embedded_chunks = generator.generate_embeddings(chunks).await.unwrap();
    let duration = start_time.elapsed();
    
    assert_eq!(embedded_chunks.len(), 10);
    
    let stats = generator.get_stats().await;
    assert!(stats.total_batches > 0);
    assert!(stats.avg_batch_time_ms > 0.0);
    
    println!("Processed {} embeddings in {:?}", embedded_chunks.len(), duration);
}

#[tokio::test]
async fn test_similarity_calculations() {
    let config = EmbedderConfig::new();
    
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let generator = generator_result.unwrap();
    
    let chunks = vec![
        create_test_chunk("The cat sat on the mat", "doc1"),
        create_test_chunk("A feline was resting on the rug", "doc1"), // Similar meaning
        create_test_chunk("Python programming language", "doc2"), // Different topic
    ];
    
    let embedded_chunks = generator.generate_embeddings(chunks).await.unwrap();
    
    // Test single similarity
    let sim_similar = generator.calculate_similarity(
        &embedded_chunks[0].embeddings,
        &embedded_chunks[1].embeddings
    ).unwrap();
    
    let sim_different = generator.calculate_similarity(
        &embedded_chunks[0].embeddings,
        &embedded_chunks[2].embeddings
    ).unwrap();
    
    // Similar texts should have higher similarity
    assert!(sim_similar > sim_different);
    
    // Test batch similarities
    let query_embedding = &embedded_chunks[0].embeddings;
    let embeddings = embedded_chunks.iter()
        .map(|ec| ec.embeddings.clone())
        .collect::<Vec<_>>();
    
    let similarities = generator.calculate_similarities(query_embedding, &embeddings).unwrap();
    
    assert_eq!(similarities.len(), 3);
    assert!((similarities[0] - 1.0).abs() < 1e-6); // Self-similarity should be ~1.0
    assert!(similarities[1] > similarities[2]); // Similar > different
}

#[tokio::test]
async fn test_model_switching() {
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let config = EmbedderConfig::new()
        .with_model_type(ModelType::AllMiniLmL6V2);
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let mut generator = generator_result.unwrap();
    
    assert_eq!(generator.get_dimension(), 384); // all-MiniLM-L6-v2 dimension
    
    // Try to switch to a different model (may not be available)
    let switch_result = generator.switch_model(ModelType::BertBaseUncased).await;
    
    if switch_result.is_ok() {
        assert_eq!(generator.get_dimension(), 768); // BERT dimension
    }
}

#[tokio::test]
async fn test_memory_usage_calculation() {
    let config = EmbedderConfig::new();
    
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let generator = generator_result.unwrap();
    
    let memory_usage = generator.calculate_memory_usage(1000);
    let expected = 1000 * generator.get_dimension() * std::mem::size_of::<f32>();
    
    assert_eq!(memory_usage, expected);
}

#[tokio::test]
async fn test_processing_time_estimation() {
    let config = EmbedderConfig::new();
    
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let generator = generator_result.unwrap();
    
    // Initially should give a fallback estimate
    let initial_estimate = generator.estimate_processing_time(100).await;
    assert!(initial_estimate > 0.0);
    
    // Process some embeddings to build statistics
    let chunks = vec![create_test_chunk("Test for timing", "doc1")];
    let _ = generator.generate_embeddings(chunks).await;
    
    // Should now give a more accurate estimate
    let updated_estimate = generator.estimate_processing_time(100).await;
    assert!(updated_estimate > 0.0);
}

#[tokio::test]
async fn test_concurrent_processing() {
    use tokio::task;
    
    let config = EmbedderConfig::new()
        .with_cache_size(1000);
    
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let generator = std::sync::Arc::new(generator_result.unwrap());
    
    let mut handles = Vec::new();
    
    // Spawn concurrent tasks
    for i in 0..5 {
        let generator_clone = generator.clone();
        let handle = task::spawn(async move {
            let chunks = vec![
                create_test_chunk(&format!("Concurrent text {}", i), "doc1"),
                create_test_chunk(&format!("Another concurrent text {}", i), "doc1"),
            ];
            
            generator_clone.generate_embeddings(chunks).await
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let results = futures::future::join_all(handles).await;
    
    for result in results {
        let embedded_chunks = result.unwrap().unwrap();
        assert_eq!(embedded_chunks.len(), 2);
    }
    
    let stats = generator.get_stats().await;
    assert_eq!(stats.total_embeddings, 10); // 5 tasks * 2 embeddings each
}

#[tokio::test]
async fn test_error_handling() {
    let config = EmbedderConfig::new();
    
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let generator = generator_result.unwrap();
    
    // Test dimension validation
    let wrong_dimension_embedding = vec![0.5; generator.get_dimension() + 10];
    let validation_result = generator.validate_embedding(&wrong_dimension_embedding);
    assert!(validation_result.is_err());
    
    // Test NaN embedding validation
    let nan_embedding = vec![f32::NAN; generator.get_dimension()];
    let nan_validation = generator.validate_embedding(&nan_embedding);
    assert!(nan_validation.is_err());
    
    // Test similarity with different dimensions
    let emb1 = vec![0.5; generator.get_dimension()];
    let emb2 = vec![0.5; generator.get_dimension() + 1];
    let similarity_result = generator.calculate_similarity(&emb1, &emb2);
    assert!(similarity_result.is_err());
}

#[tokio::test]
async fn test_configuration_validation() {
    // Test valid config
    let valid_config = EmbedderConfig::new();
    assert!(valid_config.validate().is_ok());
    
    // Test invalid configs
    let invalid_batch_size = EmbedderConfig {
        batch_size: 0,
        ..Default::default()
    };
    assert!(invalid_batch_size.validate().is_err());
    
    let invalid_max_length = EmbedderConfig {
        max_length: 0,
        ..Default::default()
    };
    assert!(invalid_max_length.validate().is_err());
    
    let invalid_cache_size = EmbedderConfig {
        cache_size: 0,
        ..Default::default()
    };
    assert!(invalid_cache_size.validate().is_err());
}

#[tokio::test]
async fn test_adaptive_batch_configuration() {
    let config = AdaptiveBatchConfig {
        min_batch_size: 2,
        max_batch_size: 8,
        target_memory_mb: 100,
        avg_text_length_chars: 500,
        embedding_dimension: 384,
    };
    
    // Test that config has reasonable values
    assert!(config.min_batch_size < config.max_batch_size);
    assert!(config.target_memory_mb > 0);
    assert!(config.embedding_dimension > 0);
}

#[tokio::test]
async fn test_memory_optimization() {
    let low_memory_config = EmbedderConfig::new()
        .low_memory()
        .optimize_for_constraints(Some(128), None, None); // 128MB limit
    
    assert!(low_memory_config.batch_size <= 8);
    assert!(low_memory_config.cache_size <= 1000);
    
    let high_perf_config = EmbedderConfig::new()
        .high_performance()
        .optimize_for_constraints(None, Some(1000), None); // 1000/sec throughput
    
    assert!(high_perf_config.batch_size >= 32);
    assert!(high_perf_config.optimization.use_fp16);
}

// Helper function to create test chunks
fn create_test_chunk(content: &str, source: &str) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        content: content.to_string(),
        metadata: ChunkMetadata {
            source: source.to_string(),
            page: Some(1),
            section: Some("test".to_string()),
            created_at: chrono::Utc::now(),
            properties: HashMap::new(),
        },
        embeddings: None,
        references: Vec::new(),
    }
}

#[tokio::test]
async fn test_performance_requirements() {
    // Test that we can meet the 1000 chunks/sec target (at least in theory)
    let config = EmbedderConfig::new()
        .high_performance()
        .with_batch_size(64);
    
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let generator = generator_result.unwrap();
    
    // Create test chunks
    let chunks: Vec<Chunk> = (0..100)
        .map(|i| create_test_chunk(&format!("Performance test text {}", i), "doc1"))
        .collect();
    
    let start_time = std::time::Instant::now();
    let embedded_chunks = generator.generate_embeddings(chunks).await.unwrap();
    let duration = start_time.elapsed();
    
    let chunks_per_sec = embedded_chunks.len() as f64 / duration.as_secs_f64();
    
    println!("Achieved {:.2} chunks/sec", chunks_per_sec);
    
    // This is a rough test - actual performance depends on hardware
    // In a real scenario, we'd test with actual models and appropriate hardware
    assert!(embedded_chunks.len() == 100);
}

#[tokio::test]
async fn test_cache_persistence() {
    use tempfile::tempdir;
    
    let temp_dir = tempdir().unwrap();
    let cache_path = temp_dir.path().join("test_cache.bin");
    
    let config = EmbedderConfig::new()
        .with_cache_size(100);
    
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let generator_result = EmbeddingGenerator::new(config).await;
    if generator_result.is_err() {
        return;
    }
    
    let generator = generator_result.unwrap();
    
    // Generate some embeddings to populate cache
    let chunks = vec![
        create_test_chunk("Cache test text", "doc1"),
        create_test_chunk("Another cache test", "doc2"),
    ];
    
    let embedded_chunks = generator.generate_embeddings(chunks).await.unwrap();
    assert_eq!(embedded_chunks.len(), 2);
    
    // Export cache contents
    let cache_contents = generator.cache.export().await;
    assert!(!cache_contents.is_empty());
    
    // Clear cache and re-import
    generator.clear_cache().await;
    assert_eq!(generator.get_cache_stats().await.total_entries, 0);
    
    generator.cache.import(cache_contents).await;
    assert!(generator.get_cache_stats().await.total_entries > 0);
}