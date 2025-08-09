//! Unit tests for individual components of the embedding generator

use embedder::{
    similarity::*, batch::*, cache::*, config::*, error::*, models::*,
};
use std::time::Duration;
use tokio::time::sleep;

#[test]
fn test_cosine_similarity_basic() {
    // Test identical vectors
    let vec1 = vec![1.0, 0.0, 0.0];
    let vec2 = vec![1.0, 0.0, 0.0];
    let sim = cosine_similarity(&vec1, &vec2).unwrap();
    assert!((sim - 1.0).abs() < 1e-6);
    
    // Test orthogonal vectors
    let vec1 = vec![1.0, 0.0, 0.0];
    let vec2 = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&vec1, &vec2).unwrap();
    assert!(sim.abs() < 1e-6);
    
    // Test opposite vectors
    let vec1 = vec![1.0, 0.0, 0.0];
    let vec2 = vec![-1.0, 0.0, 0.0];
    let sim = cosine_similarity(&vec1, &vec2).unwrap();
    assert!((sim + 1.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_errors() {
    // Different dimensions
    let vec1 = vec![1.0, 0.0];
    let vec2 = vec![1.0, 0.0, 0.0];
    assert!(cosine_similarity(&vec1, &vec2).is_err());
    
    // Empty vectors
    let vec1: Vec<f32> = vec![];
    let vec2: Vec<f32> = vec![];
    assert!(cosine_similarity(&vec1, &vec2).is_err());
    
    // Zero magnitude
    let vec1 = vec![0.0, 0.0, 0.0];
    let vec2 = vec![1.0, 0.0, 0.0];
    assert!(cosine_similarity(&vec1, &vec2).is_err());
}

#[test]
fn test_batch_cosine_similarity() {
    let query = vec![1.0, 0.0, 0.0];
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],  // similarity = 1.0
        vec![0.0, 1.0, 0.0],  // similarity = 0.0
        vec![-1.0, 0.0, 0.0], // similarity = -1.0
        vec![0.5, 0.866, 0.0], // similarity ≈ 0.5
    ];
    
    let similarities = batch_cosine_similarity(&query, &embeddings).unwrap();
    
    assert_eq!(similarities.len(), 4);
    assert!((similarities[0] - 1.0).abs() < 1e-6);
    assert!(similarities[1].abs() < 1e-6);
    assert!((similarities[2] + 1.0).abs() < 1e-6);
    assert!((similarities[3] - 0.5).abs() < 1e-2);
}

#[test]
fn test_euclidean_distance() {
    let vec1 = vec![0.0, 0.0];
    let vec2 = vec![3.0, 4.0];
    let distance = euclidean_distance(&vec1, &vec2).unwrap();
    assert!((distance - 5.0).abs() < 1e-6);
    
    // Same vector
    let distance = euclidean_distance(&vec1, &vec1).unwrap();
    assert!(distance.abs() < 1e-6);
}

#[test]
fn test_manhattan_distance() {
    let vec1 = vec![1.0, 2.0];
    let vec2 = vec![4.0, 6.0];
    let distance = manhattan_distance(&vec1, &vec2).unwrap();
    assert!((distance - 7.0).abs() < 1e-6); // |4-1| + |6-2| = 3 + 4 = 7
}

#[test]
fn test_normalize_l2() {
    let vec = vec![3.0, 4.0, 0.0];
    let normalized = normalize_l2(&vec).unwrap();
    
    // Check values
    assert!((normalized[0] - 0.6).abs() < 1e-6);
    assert!((normalized[1] - 0.8).abs() < 1e-6);
    assert!(normalized[2].abs() < 1e-6);
    
    // Check unit length
    let norm = l2_norm(&normalized);
    assert!((norm - 1.0).abs() < 1e-6);
    
    // Test zero vector error
    let zero_vec = vec![0.0, 0.0, 0.0];
    assert!(normalize_l2(&zero_vec).is_err());
}

#[test]
fn test_find_top_k_similar() {
    let query = vec![1.0, 0.0, 0.0];
    let embeddings = vec![
        (vec![1.0, 0.0, 0.0], 0),  // similarity = 1.0
        (vec![0.0, 1.0, 0.0], 1),  // similarity = 0.0
        (vec![-1.0, 0.0, 0.0], 2), // similarity = -1.0
        (vec![0.707, 0.707, 0.0], 3), // similarity ≈ 0.707
    ];
    
    let top_k = find_top_k_similar(&query, &embeddings, 2).unwrap();
    
    assert_eq!(top_k.len(), 2);
    assert_eq!(top_k[0].0, 0); // Best match (similarity = 1.0)
    assert_eq!(top_k[1].0, 3); // Second best (similarity ≈ 0.707)
    assert!(top_k[0].1 > top_k[1].1); // Descending order
}

#[test]
fn test_pairwise_cosine_similarities() {
    let embeddings = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 0.0], // Same as first
    ];
    
    let similarities = pairwise_cosine_similarities(&embeddings).unwrap();
    
    assert_eq!(similarities.len(), 3);
    assert_eq!(similarities[0].len(), 3);
    
    // Check diagonal (self-similarity)
    for i in 0..3 {
        assert!((similarities[i][i] - 1.0).abs() < 1e-6);
    }
    
    // Check symmetry
    for i in 0..3 {
        for j in 0..3 {
            assert!((similarities[i][j] - similarities[j][i]).abs() < 1e-6);
        }
    }
    
    // Check specific values
    assert!(similarities[0][1].abs() < 1e-6); // Orthogonal
    assert!((similarities[0][2] - 1.0).abs() < 1e-6); // Identical
}

#[test]
fn test_simple_kmeans_clustering() {
    let embeddings = vec![
        vec![1.0, 1.0],   // Cluster 1
        vec![1.1, 1.1],   // Cluster 1
        vec![10.0, 10.0], // Cluster 2
        vec![10.1, 10.1], // Cluster 2
    ];
    
    let (assignments, centroids) = simple_kmeans_clustering(&embeddings, 2, 10).unwrap();
    
    assert_eq!(assignments.len(), 4);
    assert_eq!(centroids.len(), 2);
    
    // Similar points should be in the same cluster
    assert_eq!(assignments[0], assignments[1]);
    assert_eq!(assignments[2], assignments[3]);
    assert_ne!(assignments[0], assignments[2]);
}

#[tokio::test]
async fn test_batch_processor_basic() {
    let processor = BatchProcessor::new(3);
    let texts = vec![
        "text1".to_string(),
        "text2".to_string(),
        "text3".to_string(),
        "text4".to_string(),
        "text5".to_string(),
    ];
    
    let batches = processor.create_batches(&texts);
    
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].len(), 3);
    assert_eq!(batches[1].len(), 2);
    
    // Test progress tracking
    let initial_progress = processor.get_progress();
    assert_eq!(initial_progress.total_items, 5);
    assert_eq!(initial_progress.total_batches, 2);
    assert_eq!(initial_progress.processed_items, 0);
    
    processor.mark_batch_completed(3);
    let updated_progress = processor.get_progress();
    assert_eq!(updated_progress.processed_items, 3);
    assert_eq!(updated_progress.current_batch, 1);
}

#[tokio::test]
async fn test_batch_processor_memory_constraints() {
    let processor = BatchProcessor::with_memory_limit(10, 1); // 1MB limit
    
    let texts = vec!["short".to_string(); 20];
    let batches = processor.create_batches(&texts);
    
    // Should create more, smaller batches due to memory constraints
    assert!(batches.len() > 1);
    
    // Total items should be preserved
    let total_items: usize = batches.iter().map(|b| b.len()).sum();
    assert_eq!(total_items, 20);
}

#[test]
fn test_batch_utils() {
    let items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    
    // Test chunking
    let chunks = BatchUtils::chunk_vector(&items, 3);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0], vec![1, 2, 3]);
    assert_eq!(chunks[1], vec![4, 5, 6]);
    assert_eq!(chunks[2], vec![7, 8, 9]);
    
    // Test splitting into N batches
    let batches = BatchUtils::split_into_n_batches(&items, 4);
    assert_eq!(batches.len(), 4);
    
    // Total items preserved
    let total: usize = batches.iter().map(|b| b.len()).sum();
    assert_eq!(total, 9);
}

#[test]
fn test_optimal_batch_size_calculation() {
    let optimal_size = BatchProcessor::calculate_optimal_batch_size(
        1000, // avg_text_length
        384,  // embedding_dimension
        1024, // available_memory_mb
        100,  // target_throughput_per_sec
    );
    
    assert!(optimal_size >= 1);
    assert!(optimal_size <= 128);
}

#[tokio::test]
async fn test_embedding_cache_basic_operations() {
    let cache = EmbeddingCache::new(10);
    
    // Test put and get
    let embedding = vec![1.0, 2.0, 3.0];
    cache.put("key1".to_string(), embedding.clone()).await;
    
    let retrieved = cache.get("key1").await;
    assert_eq!(retrieved, Some(embedding));
    
    // Test contains_key
    assert!(cache.contains_key("key1").await);
    assert!(!cache.contains_key("nonexistent").await);
    
    // Test remove
    let removed = cache.remove("key1").await;
    assert_eq!(removed, Some(vec![1.0, 2.0, 3.0]));
    assert!(!cache.contains_key("key1").await);
    
    // Test clear
    cache.put("key2".to_string(), vec![4.0, 5.0]).await;
    cache.clear().await;
    assert!(!cache.contains_key("key2").await);
}

#[tokio::test]
async fn test_cache_lru_eviction() {
    let cache = EmbeddingCache::new(2);
    
    cache.put("key1".to_string(), vec![1.0]).await;
    cache.put("key2".to_string(), vec![2.0]).await;
    assert_eq!(cache.get_stats().await.total_entries, 2);
    
    // This should evict key1 (least recently used)
    cache.put("key3".to_string(), vec![3.0]).await;
    assert_eq!(cache.get_stats().await.total_entries, 2);
    assert!(!cache.contains_key("key1").await);
    assert!(cache.contains_key("key2").await);
    assert!(cache.contains_key("key3").await);
}

#[tokio::test]
async fn test_cache_ttl() {
    let cache = EmbeddingCache::with_ttl(10, Duration::from_millis(100));
    
    cache.put("key1".to_string(), vec![1.0]).await;
    assert!(cache.contains_key("key1").await);
    
    // Wait for expiration
    sleep(Duration::from_millis(150)).await;
    
    // Should be expired now
    assert!(!cache.contains_key("key1").await);
    assert_eq!(cache.get("key1").await, None);
}

#[tokio::test]
async fn test_cache_stats() {
    let cache = EmbeddingCache::new(10);
    
    cache.put("key1".to_string(), vec![1.0, 2.0]).await;
    cache.put("key2".to_string(), vec![3.0, 4.0, 5.0]).await;
    
    // Access key1 to increase its access count
    cache.get("key1").await;
    cache.get("key1").await;
    
    let stats = cache.get_stats().await;
    assert_eq!(stats.total_entries, 2);
    assert_eq!(stats.max_capacity, 10);
    assert!(stats.memory_usage_bytes > 0);
    assert!(stats.avg_access_count > 0.0);
}

#[tokio::test]
async fn test_cache_resize() {
    let cache = EmbeddingCache::new(5);
    
    // Fill cache
    for i in 0..5 {
        cache.put(format!("key{}", i), vec![i as f32]).await;
    }
    assert_eq!(cache.get_stats().await.total_entries, 5);
    
    // Resize down
    cache.resize(2).await;
    let stats = cache.get_stats().await;
    assert_eq!(stats.total_entries, 2);
    assert_eq!(stats.max_capacity, 2);
}

#[tokio::test]
async fn test_cache_utilization() {
    let cache = EmbeddingCache::new(10);
    
    assert_eq!(cache.utilization().await, 0.0);
    
    cache.put("key1".to_string(), vec![1.0]).await;
    assert_eq!(cache.utilization().await, 10.0);
    
    for i in 1..10 {
        cache.put(format!("key{}", i), vec![i as f32]).await;
    }
    assert_eq!(cache.utilization().await, 100.0);
}

#[test]
fn test_model_type_properties() {
    let model = ModelType::AllMiniLmL6V2;
    assert_eq!(model.dimension(), 384);
    assert_eq!(model.name(), "all-MiniLM-L6-v2");
    assert_eq!(model.default_max_length(), 512);
    assert!(model.supports_onnx());
    assert!(model.hf_model_id().is_some());
    
    let expected_files = model.expected_files();
    assert!(expected_files.contains(&"config.json"));
    assert!(expected_files.contains(&"vocab.txt"));
}

#[test]
fn test_embedder_config_builder() {
    let config = EmbedderConfig::new()
        .with_model_type(ModelType::BertBaseUncased)
        .with_batch_size(64)
        .with_device(Device::Cuda)
        .with_normalize(false)
        .with_cache_size(5000)
        .with_threads(4);
    
    assert_eq!(config.model_type, ModelType::BertBaseUncased);
    assert_eq!(config.batch_size, 64);
    assert_eq!(config.device, Device::Cuda);
    assert!(!config.normalize);
    assert_eq!(config.cache_size, 5000);
    assert_eq!(config.num_threads, Some(4));
}

#[test]
fn test_config_presets() {
    let high_perf = EmbedderConfig::new().high_performance();
    assert!(high_perf.optimization.use_fp16);
    assert!(high_perf.optimization.memory_optimization);
    assert_eq!(high_perf.batch_size, 64);
    
    let low_mem = EmbedderConfig::new().low_memory();
    assert_eq!(low_mem.batch_size, 8);
    assert_eq!(low_mem.cache_size, 1000);
    assert!(matches!(low_mem.optimization.quantization, QuantizationType::Int8));
}

#[test]
fn test_config_validation() {
    let valid_config = EmbedderConfig::new();
    assert!(valid_config.validate().is_ok());
    
    let invalid_configs = vec![
        EmbedderConfig { batch_size: 0, ..Default::default() },
        EmbedderConfig { max_length: 0, ..Default::default() },
        EmbedderConfig { cache_size: 0, ..Default::default() },
    ];
    
    for config in invalid_configs {
        assert!(config.validate().is_err());
    }
}

#[test]
fn test_memory_estimation() {
    let config = EmbedderConfig::new();
    let memory = config.estimated_memory_usage(1000);
    
    // Should include embedding storage, cache, and model overhead
    assert!(memory > 0);
    
    // Should scale with number of embeddings
    let memory_2000 = config.estimated_memory_usage(2000);
    assert!(memory_2000 > memory);
}

#[test]
fn test_constraint_optimization() {
    let config = EmbedderConfig::new()
        .optimize_for_constraints(Some(256), Some(1000), Some(50));
    
    // Should adjust for low memory
    assert!(config.cache_size <= 10000);
    
    // Should optimize for low latency
    assert_eq!(config.batch_size, 1);
}

#[test]
fn test_error_types() {
    let error = EmbedderError::DimensionMismatch { expected: 384, actual: 768 };
    let error_string = format!("{}", error);
    assert!(error_string.contains("384"));
    assert!(error_string.contains("768"));
    
    // Test helper functions
    let config_err = config_error("Invalid batch size");
    match config_err {
        EmbedderError::ConfigError { message } => {
            assert_eq!(message, "Invalid batch size");
        }
        _ => panic!("Expected ConfigError"),
    }
}

#[test]
fn test_device_display() {
    assert_eq!(format!("{}", Device::Cpu), "cpu");
    assert_eq!(format!("{}", Device::Cuda), "cuda");
}

#[test]
fn test_model_type_display() {
    assert_eq!(format!("{}", ModelType::AllMiniLmL6V2), "all-MiniLM-L6-v2");
    assert_eq!(format!("{}", ModelType::BertBaseUncased), "bert-base-uncased");
}

#[tokio::test]
async fn test_model_manager_basic() {
    let manager = ModelManager::new();
    
    assert!(!manager.is_model_loaded(&ModelType::AllMiniLmL6V2));
    assert!(manager.loaded_models().is_empty());
    
    // Test path management
    let mut manager = ModelManager::new();
    let custom_path = std::path::PathBuf::from("/custom/path");
    manager.set_model_path(ModelType::AllMiniLmL6V2, custom_path.clone());
    
    // Would need actual model files to test loading
}

#[test]
fn test_adaptive_batch_config() {
    let config = AdaptiveBatchConfig::default();
    assert!(config.min_batch_size < config.max_batch_size);
    assert!(config.target_memory_mb > 0);
    assert!(config.embedding_dimension > 0);
    
    let custom_config = AdaptiveBatchConfig {
        min_batch_size: 2,
        max_batch_size: 16,
        target_memory_mb: 256,
        avg_text_length_chars: 800,
        embedding_dimension: 768,
    };
    
    assert_eq!(custom_config.min_batch_size, 2);
    assert_eq!(custom_config.max_batch_size, 16);
}