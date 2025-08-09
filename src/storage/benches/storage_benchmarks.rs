//! Performance benchmarks for MongoDB Vector Storage

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;
use uuid::Uuid;
use std::collections::HashMap;

use storage::{
    VectorStorage, StorageConfig, ChunkDocument, ChunkMetadata, CustomFieldValue,
    SearchQuery, SearchType, SearchFilters, DatabaseOperations, SearchOperations,
    BulkInsertRequest,
};

/// Benchmark configuration
struct BenchmarkConfig {
    runtime: Runtime,
    storage: Option<VectorStorage>,
}

impl BenchmarkConfig {
    fn new() -> Self {
        let runtime = Runtime::new().unwrap();
        
        Self {
            runtime,
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
    
    fn create_test_chunk(&self, index: usize, document_id: Uuid) -> ChunkDocument {
        let metadata = ChunkMetadata {
            document_id,
            title: format!("Benchmark Document {}", index),
            chunk_index: index,
            total_chunks: 1000,
            chunk_size: 512,
            overlap_size: 50,
            source_path: format!("/benchmark/doc_{}.txt", index),
            mime_type: "text/plain".to_string(),
            language: "en".to_string(),
            tags: vec!["benchmark".to_string(), format!("batch_{}", index / 100)],
            custom_fields: {
                let mut fields = HashMap::new();
                fields.insert("priority".to_string(), CustomFieldValue::String("normal".to_string()));
                fields.insert("batch".to_string(), CustomFieldValue::Number((index / 100) as f64));
                fields
            },
            content_hash: format!("hash_{}", index),
            boundary_confidence: Some(0.85),
        };
        
        // Generate high-quality embeddings for benchmarking
        let embedding: Vec<f64> = (0..384).map(|i| {
            let freq = (i + index) as f64 * 0.1;
            (freq.sin() + freq.cos()) * 0.5
        }).collect();
        
        ChunkDocument::new(
            Uuid::new_v4(),
            format!("Benchmark content for chunk {}. This contains detailed information about the benchmark test case and provides realistic text length for performance testing.", index),
            metadata,
        ).with_embedding(embedding)
    }
}

fn bench_single_insert(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    c.bench_function("single_insert", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            let chunk = config.create_test_chunk(0, document_id);
            
            storage.insert_chunk(chunk).await.unwrap()
        });
    });
}

fn bench_bulk_insert(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    let mut group = c.benchmark_group("bulk_insert");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("chunks", size), size, |b, &size| {
            b.to_async(&config.runtime).iter(|| async {
                let storage = config.get_or_create_storage();
                let chunks: Vec<ChunkDocument> = (0..size)
                    .map(|i| config.create_test_chunk(i, document_id))
                    .collect();
                
                storage.insert_chunks(chunks).await.unwrap()
            });
        });
    }
    
    group.finish();
}

fn bench_chunk_retrieval(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    // Pre-populate with test data
    let chunk_ids: Vec<Uuid> = config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let chunks: Vec<ChunkDocument> = (0..1000)
            .map(|i| config.create_test_chunk(i, document_id))
            .collect();
        
        let chunk_ids: Vec<Uuid> = chunks.iter().map(|c| c.chunk_id).collect();
        storage.insert_chunks(chunks).await.unwrap();
        chunk_ids
    });
    
    let mut group = c.benchmark_group("chunk_retrieval");
    
    group.bench_function("get_single_chunk", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            let chunk_id = chunk_ids[fastrand::usize(..chunk_ids.len())];
            
            storage.get_chunk(chunk_id).await.unwrap()
        });
    });
    
    group.bench_function("get_document_chunks", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            
            storage.get_document_chunks(document_id).await.unwrap()
        });
    });
    
    group.bench_function("count_document_chunks", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            
            storage.count_document_chunks(document_id).await.unwrap()
        });
    });
    
    group.finish();
}

fn bench_vector_search(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    // Pre-populate with test data
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let chunks: Vec<ChunkDocument> = (0..10000)
            .map(|i| config.create_test_chunk(i, document_id))
            .collect();
        
        storage.insert_chunks(chunks).await.unwrap();
    });
    
    let query_embedding: Vec<f64> = (0..384).map(|i| {
        let freq = i as f64 * 0.1;
        (freq.sin() + freq.cos()) * 0.5
    }).collect();
    
    let mut group = c.benchmark_group("vector_search");
    group.measurement_time(Duration::from_secs(30));
    
    for k in [1, 5, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*k as u64));
        group.bench_with_input(BenchmarkId::new("k_nearest", k), k, |b, &k| {
            b.to_async(&config.runtime).iter(|| async {
                let storage = config.get_or_create_storage();
                
                storage.vector_search(&query_embedding, k, None).await.unwrap()
            });
        });
    }
    
    group.finish();
}

fn bench_text_search(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    // Pre-populate with varied content
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let mut chunks = Vec::new();
        
        for i in 0..1000 {
            let mut chunk = config.create_test_chunk(i, document_id);
            chunk.content = format!(
                "Document {} discusses {} and related topics like {} and {}.",
                i,
                match i % 10 {
                    0 => "machine learning algorithms",
                    1 => "database optimization techniques",
                    2 => "vector search implementations", 
                    3 => "natural language processing",
                    4 => "distributed systems architecture",
                    5 => "data storage solutions",
                    6 => "search engine design",
                    7 => "information retrieval methods",
                    8 => "artificial intelligence applications",
                    _ => "software engineering practices",
                },
                match (i + 1) % 8 {
                    0 => "performance optimization",
                    1 => "scalability patterns",
                    2 => "security considerations",
                    3 => "user experience design",
                    4 => "system reliability",
                    5 => "data consistency",
                    6 => "fault tolerance",
                    _ => "monitoring strategies",
                },
                match (i + 2) % 6 {
                    0 => "best practices",
                    1 => "implementation details",
                    2 => "theoretical foundations",
                    3 => "practical applications",
                    4 => "comparative analysis",
                    _ => "future developments",
                }
            );
            chunks.push(chunk);
        }
        
        storage.insert_chunks(chunks).await.unwrap();
    });
    
    let queries = vec![
        "machine learning",
        "database optimization",
        "vector search",
        "natural language",
        "distributed systems",
        "performance optimization",
        "best practices",
    ];
    
    let mut group = c.benchmark_group("text_search");
    
    for query in queries.iter() {
        group.bench_with_input(BenchmarkId::new("query", query), query, |b, query| {
            b.to_async(&config.runtime).iter(|| async {
                let storage = config.get_or_create_storage();
                
                storage.text_search(query, 20, None).await.unwrap()
            });
        });
    }
    
    group.finish();
}

fn bench_hybrid_search(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    // Pre-populate with test data
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let chunks: Vec<ChunkDocument> = (0..5000)
            .map(|i| {
                let mut chunk = config.create_test_chunk(i, document_id);
                chunk.content = format!(
                    "Hybrid search document {} containing information about vector similarity and text matching capabilities with relevance scoring.",
                    i
                );
                chunk
            })
            .collect();
        
        storage.insert_chunks(chunks).await.unwrap();
    });
    
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.01).collect();
    
    let mut group = c.benchmark_group("hybrid_search");
    
    // Test different search configurations
    let search_configs = vec![
        ("vector_only", SearchType::Vector, Some(query_embedding.clone()), None),
        ("text_only", SearchType::Text, None, Some("vector similarity".to_string())),
        ("hybrid", SearchType::Hybrid, Some(query_embedding.clone()), Some("text matching".to_string())),
    ];
    
    for (name, search_type, embedding, text) in search_configs {
        group.bench_function(name, |b| {
            b.to_async(&config.runtime).iter(|| async {
                let storage = config.get_or_create_storage();
                
                let query = SearchQuery {
                    query_embedding: embedding.clone(),
                    text_query: text.clone(),
                    search_type: search_type.clone(),
                    limit: 20,
                    offset: 0,
                    min_score: Some(0.1),
                    filters: SearchFilters::default(),
                    sort: Default::default(),
                };
                
                storage.hybrid_search(query).await.unwrap()
            });
        });
    }
    
    group.finish();
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    // Pre-populate with test data
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let chunks: Vec<ChunkDocument> = (0..1000)
            .map(|i| config.create_test_chunk(i, document_id))
            .collect();
        
        storage.insert_chunks(chunks).await.unwrap();
    });
    
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.01).collect();
    
    let mut group = c.benchmark_group("concurrent_operations");
    
    for concurrency in [1, 2, 5, 10].iter() {
        group.throughput(Throughput::Elements(*concurrency as u64));
        group.bench_with_input(BenchmarkId::new("concurrent_search", concurrency), concurrency, |b, &concurrency| {
            b.to_async(&config.runtime).iter(|| async {
                let storage = config.get_or_create_storage();
                
                let tasks: Vec<_> = (0..concurrency)
                    .map(|_| {
                        let storage = storage.clone();
                        let embedding = query_embedding.clone();
                        tokio::spawn(async move {
                            storage.vector_search(&embedding, 10, None).await.unwrap()
                        })
                    })
                    .collect();
                
                futures::future::join_all(tasks).await
            });
        });
    }
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    let mut group = c.benchmark_group("memory_usage");
    
    // Test memory usage for different data sizes
    for chunk_count in [100, 500, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*chunk_count as u64));
        group.bench_with_input(BenchmarkId::new("memory_load", chunk_count), chunk_count, |b, &chunk_count| {
            b.iter(|| {
                // Create chunks in memory (not inserted to DB for this test)
                let chunks: Vec<ChunkDocument> = (0..chunk_count)
                    .map(|i| config.create_test_chunk(i, document_id))
                    .collect();
                
                // Calculate total memory usage estimate
                let total_size: usize = chunks.iter().map(|chunk| {
                    chunk.content.len() + 
                    chunk.embedding.as_ref().map_or(0, |e| e.len() * 8) + // f64 = 8 bytes
                    chunk.metadata.title.len() +
                    chunk.metadata.source_path.len() +
                    chunk.metadata.tags.iter().map(|t| t.len()).sum::<usize>() +
                    1000 // Estimated overhead
                }).sum();
                
                (chunks.len(), total_size)
            });
        });
    }
    
    group.finish();
}

fn bench_search_latency_distribution(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    // Pre-populate with large dataset
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let chunks: Vec<ChunkDocument> = (0..20000)
            .map(|i| config.create_test_chunk(i, document_id))
            .collect();
        
        // Insert in batches to avoid memory issues
        for batch in chunks.chunks(1000) {
            storage.insert_chunks(batch.to_vec()).await.unwrap();
        }
    });
    
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.01).collect();
    
    let mut group = c.benchmark_group("search_latency");
    group.measurement_time(Duration::from_secs(60)); // Longer measurement for better statistics
    
    group.bench_function("p50_latency", |b| {
        b.to_async(&config.runtime).iter(|| async {
            let storage = config.get_or_create_storage();
            let start = std::time::Instant::now();
            
            storage.vector_search(&query_embedding, 10, None).await.unwrap();
            
            start.elapsed()
        });
    });
    
    group.finish();
}

// Custom benchmark for the <50ms search latency requirement
fn bench_search_latency_requirement(c: &mut Criterion) {
    let mut config = BenchmarkConfig::new();
    let document_id = Uuid::new_v4();
    
    // Pre-populate with realistic dataset size
    config.runtime.block_on(async {
        let storage = config.get_or_create_storage();
        let chunks: Vec<ChunkDocument> = (0..10000)
            .map(|i| config.create_test_chunk(i, document_id))
            .collect();
        
        storage.insert_chunks(chunks).await.unwrap();
    });
    
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.01).collect();
    
    let mut group = c.benchmark_group("latency_requirement");
    group.measurement_time(Duration::from_secs(30));
    
    group.bench_function("search_under_50ms", |b| {
        b.to_async(&config.runtime).iter_custom(|iters| async move {
            let storage = config.get_or_create_storage();
            let mut total_duration = Duration::new(0, 0);
            let mut violations = 0;
            
            for _ in 0..iters {
                let start = std::time::Instant::now();
                storage.vector_search(&query_embedding, 10, None).await.unwrap();
                let duration = start.elapsed();
                
                total_duration += duration;
                
                if duration > Duration::from_millis(50) {
                    violations += 1;
                }
            }
            
            // Report violations if any
            if violations > 0 {
                eprintln!("Warning: {} out of {} searches exceeded 50ms requirement", violations, iters);
            }
            
            total_duration
        });
    });
    
    group.finish();
}

criterion_group!(
    storage_benches,
    bench_single_insert,
    bench_bulk_insert,
    bench_chunk_retrieval,
    bench_vector_search,
    bench_text_search,
    bench_hybrid_search,
    bench_concurrent_operations,
    bench_memory_usage,
    bench_search_latency_distribution,
    bench_search_latency_requirement
);

criterion_main!(storage_benches);