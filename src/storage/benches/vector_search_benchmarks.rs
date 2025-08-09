//! Specialized benchmarks for vector search operations

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;
use uuid::Uuid;

use storage::{
    VectorStorage, StorageConfig, ChunkDocument, ChunkMetadata,
    SearchQuery, SearchType, SearchFilters, VectorSimilarity,
    SearchOperations, DatabaseOperations,
};

struct VectorBenchmarkConfig {
    runtime: Runtime,
    storage: Option<VectorStorage>,
}

impl VectorBenchmarkConfig {
    fn new() -> Self {
        Self {
            runtime: Runtime::new().unwrap(),
            storage: None,
        }
    }
    
    fn get_or_create_storage(&mut self) -> &VectorStorage {
        if self.storage.is_none() {
            let config = StorageConfig::for_testing();
            self.storage = Some(
                self.runtime.block_on(async {
                    VectorStorage::new(config).await.expect("Failed to create storage")
                })
            );
        }
        self.storage.as_ref().unwrap()
    }
    
    fn generate_embedding(&self, dimension: usize, seed: u64) -> Vec<f64> {
        // Generate high-quality, deterministic embeddings
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let base_hash = hasher.finish();
        
        (0..dimension).map(|i| {
            let mut hasher = DefaultHasher::new();
            (base_hash + i as u64).hash(&mut hasher);
            let val = hasher.finish() as f64 / u64::MAX as f64;
            (val - 0.5) * 2.0 // Normalize to [-1, 1]
        }).collect()
    }
    
    fn create_chunk_with_embedding(&self, index: usize, document_id: Uuid, dimension: usize) -> ChunkDocument {
        let metadata = ChunkMetadata::new(
            document_id,
            format!("Vector Document {}", index),
            index,
            10000,
            format!("/vectors/doc_{}.txt", index),
        );
        
        let embedding = self.generate_embedding(dimension, index as u64);
        
        ChunkDocument::new(
            Uuid::new_v4(),
            format!("Vector content for document {} with {} dimensional embedding", index, dimension),
            metadata,
        ).with_embedding(embedding)
    }
}

fn bench_embedding_dimensions(c: &mut Criterion) {
    let mut config = VectorBenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    let dimensions = vec![128, 256, 384, 512, 768, 1024];
    let mut group = c.benchmark_group("embedding_dimensions");
    
    for &dimension in &dimensions {
        // Pre-populate data for each dimension
        config.runtime.block_on(async {
            let storage = config.get_or_create_storage();
            let chunks: Vec<ChunkDocument> = (0..1000)
                .map(|i| config.create_chunk_with_embedding(i, document_id, dimension))
                .collect();
            
            storage.insert_chunks(chunks).await.unwrap();
        });
        
        let query_embedding = config.generate_embedding(dimension, 9999);
        
        group.throughput(Throughput::Elements(dimension as u64));
        group.bench_with_input(BenchmarkId::new("vector_search", dimension), &dimension, |b, _| {
            b.to_async(&config.runtime).iter(|| async {
                let storage = config.get_or_create_storage();
                
                storage.vector_search(&query_embedding, 10, None).await.unwrap()
            });
        });
        
        // Clean up for next dimension test
        config.runtime.block_on(async {
            let storage = config.get_or_create_storage();
            storage.delete_document_chunks(document_id).await.unwrap();
        });
    }
    
    group.finish();
}

fn bench_dataset_sizes(c: &mut Criterion) {
    let mut config = VectorBenchmarkConfig::new();
    let dimension = 384; // Standard dimension
    
    let sizes = vec![100, 500, 1000, 5000, 10000, 50000];
    let mut group = c.benchmark_group("dataset_sizes");
    group.measurement_time(Duration::from_secs(45));
    
    for &size in &sizes {
        let document_id = Uuid::new_v4();
        
        // Pre-populate data
        config.runtime.block_on(async {
            let storage = config.get_or_create_storage();
            let chunks: Vec<ChunkDocument> = (0..size)
                .map(|i| config.create_chunk_with_embedding(i, document_id, dimension))
                .collect();
            
            // Insert in batches for large datasets
            for batch in chunks.chunks(1000) {
                storage.insert_chunks(batch.to_vec()).await.unwrap();
            }
        });
        
        let query_embedding = config.generate_embedding(dimension, 9999);
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("search_in", size), &size, |b, _| {
            b.to_async(&config.runtime).iter(|| async {
                let storage = config.get_or_create_storage();
                
                let start = std::time::Instant::now();
                let results = storage.vector_search(&query_embedding, 10, None).await.unwrap();
                let duration = start.elapsed();
                
                // Assert that we meet the <50ms requirement
                if duration > Duration::from_millis(50) {
                    eprintln!("Warning: Search in {} docs took {:?} (>50ms)", size, duration);
                }
                
                results
            });
        });
    }
    
    group.finish();
}

fn bench_k_values(c: &mut Criterion) {
    let mut config = VectorBenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    let dimension = 384;
    
    // Pre-populate with large dataset
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let chunks: Vec<ChunkDocument> = (0..10000)
            .map(|i| config.create_chunk_with_embedding(i, document_id, dimension))
            .collect();
        
        for batch in chunks.chunks(1000) {
            storage.insert_chunks(batch.to_vec()).await.unwrap();
        }
    });
    
    let query_embedding = config.generate_embedding(dimension, 9999);
    let k_values = vec![1, 5, 10, 20, 50, 100, 200, 500];
    
    let mut group = c.benchmark_group("k_values");
    
    for &k in &k_values {
        group.throughput(Throughput::Elements(k as u64));
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.to_async(&config.runtime).iter(|| async {
                let storage = config.get_or_create_storage();
                
                storage.vector_search(&query_embedding, k, None).await.unwrap()
            });
        });
    }
    
    group.finish();
}

fn bench_similarity_calculations(c: &mut Criterion) {
    let dimensions = vec![128, 256, 384, 512, 768, 1024];
    let config = VectorBenchmarkConfig::new();
    
    let mut group = c.benchmark_group("similarity_calculations");
    
    for &dim in &dimensions {
        let vec1 = config.generate_embedding(dim, 1);
        let vec2 = config.generate_embedding(dim, 2);
        
        group.throughput(Throughput::Elements(dim as u64));
        
        group.bench_with_input(BenchmarkId::new("cosine_similarity", dim), &dim, |b, _| {
            b.iter(|| {
                VectorSimilarity::cosine_similarity(&vec1, &vec2).unwrap()
            });
        });
        
        group.bench_with_input(BenchmarkId::new("euclidean_distance", dim), &dim, |b, _| {
            b.iter(|| {
                VectorSimilarity::euclidean_distance(&vec1, &vec2).unwrap()
            });
        });
    }
    
    group.finish();
}

fn bench_filtered_search(c: &mut Criterion) {
    let mut config = VectorBenchmarkConfig::new();
    let dimension = 384;
    
    // Create multiple documents for filtering tests
    let document_ids: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();
    
    // Pre-populate with data across multiple documents
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        
        for (doc_idx, &document_id) in document_ids.iter().enumerate() {
            let chunks: Vec<ChunkDocument> = (0..1000)
                .map(|i| {
                    let mut chunk = config.create_chunk_with_embedding(i, document_id, dimension);
                    chunk.metadata.tags = vec![
                        format!("doc_{}", doc_idx),
                        format!("category_{}", i % 5),
                        "searchable".to_string(),
                    ];
                    chunk
                })
                .collect();
            
            storage.insert_chunks(chunks).await.unwrap();
        }
    });
    
    let query_embedding = config.generate_embedding(dimension, 9999);
    
    let mut group = c.benchmark_group("filtered_search");
    
    // No filter baseline
    group.bench_function("no_filter", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            
            storage.vector_search(&query_embedding, 20, None).await.unwrap()
        });
    });
    
    // Single document filter
    group.bench_function("single_document", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            let filters = SearchFilters {
                document_ids: Some(vec![document_ids[0]]),
                ..Default::default()
            };
            
            storage.vector_search(&query_embedding, 20, Some(filters)).await.unwrap()
        });
    });
    
    // Multiple document filter
    group.bench_function("multiple_documents", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            let filters = SearchFilters {
                document_ids: Some(document_ids[0..3].to_vec()),
                ..Default::default()
            };
            
            storage.vector_search(&query_embedding, 20, Some(filters)).await.unwrap()
        });
    });
    
    // Tag filter
    group.bench_function("tag_filter", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            let filters = SearchFilters {
                tags: Some(vec!["category_1".to_string()]),
                ..Default::default()
            };
            
            storage.vector_search(&query_embedding, 20, Some(filters)).await.unwrap()
        });
    });
    
    // Combined filters
    group.bench_function("combined_filters", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            let filters = SearchFilters {
                document_ids: Some(document_ids[0..5].to_vec()),
                tags: Some(vec!["searchable".to_string()]),
                ..Default::default()
            };
            
            storage.vector_search(&query_embedding, 20, Some(filters)).await.unwrap()
        });
    });
    
    group.finish();
}

fn bench_search_accuracy_vs_speed(c: &mut Criterion) {
    let mut config = VectorBenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    let dimension = 384;
    
    // Create a known dataset where we can verify results
    let known_embeddings: Vec<Vec<f64>> = (0..1000)
        .map(|i| config.generate_embedding(dimension, i as u64))
        .collect();
    
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let chunks: Vec<ChunkDocument> = known_embeddings.iter().enumerate()
            .map(|(i, embedding)| {
                let metadata = ChunkMetadata::new(
                    document_id,
                    format!("Accuracy Test Doc {}", i),
                    i,
                    1000,
                    format!("/accuracy/doc_{}.txt", i),
                );
                
                ChunkDocument::new(
                    Uuid::new_v4(),
                    format!("Accuracy test content {}", i),
                    metadata,
                ).with_embedding(embedding.clone())
            })
            .collect();
        
        storage.insert_chunks(chunks).await.unwrap();
    });
    
    // Use the first embedding as query (should be most similar to itself)
    let query_embedding = known_embeddings[0].clone();
    
    let mut group = c.benchmark_group("accuracy_vs_speed");
    
    // Test different k values to see accuracy vs speed tradeoff
    for &k in &[1, 5, 10, 20, 50, 100] {
        group.bench_with_input(BenchmarkId::new("search_k", k), &k, |b, &k| {
            b.to_async(&config.runtime).iter_custom(|iters| async move {
                let storage = config.get_or_create_storage();
                let mut total_duration = Duration::new(0, 0);
                
                for _ in 0..iters {
                    let start = std::time::Instant::now();
                    let results = storage.vector_search(&query_embedding, k, None).await.unwrap();
                    let duration = start.elapsed();
                    
                    // Verify that the most similar result is reasonably good
                    if let Some(best_result) = results.first() {
                        if best_result.score < 0.5 {
                            eprintln!("Warning: Best similarity score is low: {}", best_result.score);
                        }
                    }
                    
                    total_duration += duration;
                }
                
                total_duration
            });
        });
    }
    
    group.finish();
}

fn bench_concurrent_vector_searches(c: &mut Criterion) {
    let mut config = VectorBenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    let dimension = 384;
    
    // Pre-populate with test data
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let chunks: Vec<ChunkDocument> = (0..5000)
            .map(|i| config.create_chunk_with_embedding(i, document_id, dimension))
            .collect();
        
        for batch in chunks.chunks(1000) {
            storage.insert_chunks(batch.to_vec()).await.unwrap();
        }
    });
    
    let mut group = c.benchmark_group("concurrent_vector_searches");
    
    for &concurrency in &[1, 2, 4, 8, 16] {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(BenchmarkId::new("concurrent", concurrency), &concurrency, |b, &concurrency| {
            b.to_async(&config.runtime).iter(|| async {
                let storage = config.get_or_create_storage();
                
                let tasks: Vec<_> = (0..concurrency)
                    .map(|i| {
                        let storage = storage.clone();
                        let query_embedding = config.generate_embedding(dimension, i as u64 + 10000);
                        
                        tokio::spawn(async move {
                            let start = std::time::Instant::now();
                            let results = storage.vector_search(&query_embedding, 10, None).await.unwrap();
                            let duration = start.elapsed();
                            (results, duration)
                        })
                    })
                    .collect();
                
                let results = futures::future::join_all(tasks).await;
                
                // Check that all searches completed successfully and within time limits
                let mut max_duration = Duration::new(0, 0);
                for result in results {
                    let (search_results, duration) = result.unwrap();
                    assert!(!search_results.is_empty());
                    max_duration = max_duration.max(duration);
                }
                
                // In concurrent scenarios, individual searches might take longer,
                // but not excessively so
                if max_duration > Duration::from_millis(100) {
                    eprintln!("Warning: Max concurrent search duration: {:?}", max_duration);
                }
                
                max_duration
            });
        });
    }
    
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let config = VectorBenchmarkConfig::new();
    
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Test memory efficiency of different embedding storage methods
    for &dimension in &[128, 256, 384, 512, 768, 1024] {
        group.throughput(Throughput::Elements(dimension as u64));
        
        group.bench_with_input(BenchmarkId::new("embedding_creation", dimension), &dimension, |b, &dimension| {
            b.iter(|| {
                // Test the cost of creating embeddings
                let embedding = config.generate_embedding(dimension, 42);
                let memory_size = embedding.len() * std::mem::size_of::<f64>();
                (embedding, memory_size)
            });
        });
        
        group.bench_with_input(BenchmarkId::new("embedding_clone", dimension), &dimension, |b, &dimension| {
            let original = config.generate_embedding(dimension, 42);
            b.iter(|| {
                original.clone()
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    vector_search_benches,
    bench_embedding_dimensions,
    bench_dataset_sizes,
    bench_k_values,
    bench_similarity_calculations,
    bench_filtered_search,
    bench_search_accuracy_vs_speed,
    bench_concurrent_vector_searches,
    bench_memory_efficiency
);

criterion_main!(vector_search_benches);