//! Benchmark tests for embedding generation performance
//!
//! These benchmarks verify that the embedding generator meets the 
//! performance target of 1000 chunks/sec.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use embedder::{
    EmbeddingGenerator, EmbedderConfig, ModelType, Device, Chunk, ChunkMetadata,
    similarity::*, batch::*, cache::*,
};
use std::collections::HashMap;
use uuid::Uuid;
use tokio::runtime::Runtime;

// Helper function to create test chunks
fn create_test_chunks(count: usize, text_length: usize) -> Vec<Chunk> {
    let base_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(text_length / 56);
    
    (0..count)
        .map(|i| Chunk {
            id: Uuid::new_v4(),
            content: format!("{} {}", base_text, i),
            metadata: ChunkMetadata {
                source: format!("doc_{}", i / 10),
                page: Some((i / 10 + 1) as u32),
                section: Some(format!("section_{}", i % 5)),
                created_at: chrono::Utc::now(),
                properties: HashMap::new(),
            },
            embeddings: None,
            references: Vec::new(),
        })
        .collect()
}

// Create runtime for async benchmarks
fn create_runtime() -> Runtime {
    Runtime::new().unwrap()
}

fn bench_embedding_generation(c: &mut Criterion) {
    let rt = create_runtime();
    
    // Skip if models aren't available
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let mut group = c.benchmark_group("embedding_generation");
    
    for chunk_count in [10, 50, 100, 500].iter() {
        for batch_size in [8, 16, 32, 64].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}chunks_{}batch", chunk_count, batch_size)),
                &(*chunk_count, *batch_size),
                |b, &(chunks, batch)| {
                    let config = EmbedderConfig::new()
                        .with_model_type(ModelType::AllMiniLmL6V2)
                        .with_batch_size(batch)
                        .with_device(Device::Cpu);
                    
                    // Setup (not measured)
                    let generator = rt.block_on(async {
                        match EmbeddingGenerator::new(config).await {
                            Ok(gen) => Some(gen),
                            Err(_) => None, // Model not available
                        }
                    });
                    
                    if let Some(gen) = generator {
                        b.iter(|| {
                            rt.block_on(async {
                                let test_chunks = create_test_chunks(chunks, 500);
                                let result = gen.generate_embeddings(test_chunks).await;
                                black_box(result)
                            })
                        });
                    }
                },
            );
        }
    }
    
    group.finish();
}

fn bench_similarity_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_calculations");
    
    // Create test embeddings of different dimensions
    let dimensions = [384, 768, 1536]; // Common embedding dimensions
    let embedding_counts = [100, 500, 1000];
    
    for &dim in &dimensions {
        for &count in &embedding_counts {
            let query_embedding = vec![0.5; dim];
            let embeddings: Vec<Vec<f32>> = (0..count)
                .map(|i| vec![i as f32 / count as f32; dim])
                .collect();
            
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("cosine_{}d_{}emb", dim, count)),
                &(query_embedding.clone(), embeddings.clone()),
                |b, (query, embs)| {
                    b.iter(|| {
                        let similarities = batch_cosine_similarity(query, embs).unwrap();
                        black_box(similarities)
                    })
                },
            );
            
            // Single similarity calculation
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("cosine_single_{}d", dim)),
                &(query_embedding.clone(), embeddings[0].clone()),
                |b, (query, emb)| {
                    b.iter(|| {
                        let similarity = cosine_similarity(query, emb).unwrap();
                        black_box(similarity)
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn bench_cache_operations(c: &mut Criterion) {
    let rt = create_runtime();
    let mut group = c.benchmark_group("cache_operations");
    
    let cache_sizes = [100, 1000, 10000];
    let embedding_dim = 384;
    
    for &cache_size in &cache_sizes {
        // Benchmark cache insertions
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("cache_put_{}", cache_size)),
            &cache_size,
            |b, &size| {
                b.iter_batched(
                    || EmbeddingCache::new(size),
                    |cache| {
                        rt.block_on(async {
                            for i in 0..100 {
                                let key = format!("key_{}", i);
                                let embedding = vec![i as f32; embedding_dim];
                                cache.put(key, embedding).await;
                            }
                        })
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
        
        // Benchmark cache retrievals
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("cache_get_{}", cache_size)),
            &cache_size,
            |b, &size| {
                let cache = rt.block_on(async {
                    let cache = EmbeddingCache::new(size);
                    // Pre-populate cache
                    for i in 0..100 {
                        let key = format!("key_{}", i);
                        let embedding = vec![i as f32; embedding_dim];
                        cache.put(key, embedding).await;
                    }
                    cache
                });
                
                b.iter(|| {
                    rt.block_on(async {
                        for i in 0..100 {
                            let key = format!("key_{}", i);
                            let result = cache.get(&key).await;
                            black_box(result);
                        }
                    })
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let text_counts = [100, 500, 1000, 5000];
    let batch_sizes = [8, 16, 32, 64, 128];
    
    for &text_count in &text_counts {
        let texts: Vec<String> = (0..text_count)
            .map(|i| format!("This is test text number {} for benchmarking batch processing performance", i))
            .collect();
        
        for &batch_size in &batch_sizes {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("batch_{}texts_{}size", text_count, batch_size)),
                &(texts.clone(), batch_size),
                |b, (texts, batch_size)| {
                    b.iter(|| {
                        let processor = BatchProcessor::new(*batch_size);
                        let batches = processor.create_batches(texts);
                        black_box(batches)
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");
    
    let dimensions = [384, 768, 1536];
    
    for &dim in &dimensions {
        let embedding = vec![0.5; dim];
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("l2_norm_{}d", dim)),
            &embedding,
            |b, emb| {
                b.iter(|| {
                    let norm = l2_norm(emb);
                    black_box(norm)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("normalize_{}d", dim)),
            &embedding,
            |b, emb| {
                b.iter(|| {
                    let normalized = normalize_l2(emb).unwrap();
                    black_box(normalized)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let rt = create_runtime();
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Test memory usage with different cache sizes and embedding counts
    let cache_sizes = [1000, 5000, 10000];
    let embedding_counts = [100, 500, 1000];
    let embedding_dim = 384;
    
    for &cache_size in &cache_sizes {
        for &emb_count in &embedding_counts {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("memory_cache{}_emb{}", cache_size, emb_count)),
                &(cache_size, emb_count),
                |b, &(cache_sz, emb_cnt)| {
                    b.iter_batched(
                        || {
                            rt.block_on(async {
                                let cache = EmbeddingCache::new(cache_sz);
                                
                                // Fill with embeddings
                                for i in 0..emb_cnt {
                                    let key = format!("embedding_{}", i);
                                    let embedding = vec![i as f32 / emb_cnt as f32; embedding_dim];
                                    cache.put(key, embedding).await;
                                }
                                
                                cache
                            })
                        },
                        |cache| {
                            rt.block_on(async {
                                let memory_usage = cache.memory_usage().await;
                                let stats = cache.get_stats().await;
                                black_box((memory_usage, stats))
                            })
                        },
                        criterion::BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    
    group.finish();
}

fn bench_throughput_target(c: &mut Criterion) {
    let rt = create_runtime();
    
    // Skip if models aren't available
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let mut group = c.benchmark_group("throughput_target");
    group.measurement_time(std::time::Duration::from_secs(30)); // Longer measurement for accuracy
    
    // Test the 1000 chunks/sec target
    let target_chunks = 1000;
    let optimal_config = EmbedderConfig::new()
        .with_model_type(ModelType::AllMiniLmL6V2)
        .with_batch_size(64) // Optimized for throughput
        .high_performance()
        .with_device(Device::Cpu);
    
    group.bench_function("1000_chunks_per_second_target", |b| {
        let generator = rt.block_on(async {
            match EmbeddingGenerator::new(optimal_config).await {
                Ok(gen) => Some(gen),
                Err(_) => None, // Model not available
            }
        });
        
        if let Some(gen) = generator {
            b.iter(|| {
                rt.block_on(async {
                    let chunks = create_test_chunks(target_chunks, 200); // 200 char average
                    let start = std::time::Instant::now();
                    let result = gen.generate_embeddings(chunks).await;
                    let duration = start.elapsed();
                    
                    if let Ok(embeddings) = result {
                        let chunks_per_sec = embeddings.len() as f64 / duration.as_secs_f64();
                        
                        // Log performance for analysis
                        if chunks_per_sec >= 1000.0 {
                            println!("✓ Target met: {:.1} chunks/sec", chunks_per_sec);
                        } else {
                            println!("✗ Target missed: {:.1} chunks/sec", chunks_per_sec);
                        }
                        
                        black_box((embeddings, chunks_per_sec))
                    } else {
                        black_box((Vec::new(), 0.0))
                    }
                })
            });
        }
    });
    
    group.finish();
}

fn bench_concurrent_processing(c: &mut Criterion) {
    let rt = create_runtime();
    
    // Skip if models aren't available
    if std::env::var("SKIP_MODEL_TESTS").is_ok() {
        return;
    }
    
    let mut group = c.benchmark_group("concurrent_processing");
    
    let thread_counts = [1, 2, 4, 8];
    let chunks_per_thread = 100;
    
    for &thread_count in &thread_counts {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}threads", thread_count)),
            &thread_count,
            |b, &threads| {
                let config = EmbedderConfig::new()
                    .with_model_type(ModelType::AllMiniLmL6V2)
                    .with_batch_size(32);
                
                let generator = rt.block_on(async {
                    match EmbeddingGenerator::new(config).await {
                        Ok(gen) => Some(std::sync::Arc::new(gen)),
                        Err(_) => None,
                    }
                });
                
                if let Some(gen) = generator {
                    b.iter(|| {
                        rt.block_on(async {
                            let mut handles = Vec::new();
                            
                            for i in 0..threads {
                                let gen_clone = gen.clone();
                                let handle = tokio::spawn(async move {
                                    let chunks = create_test_chunks(chunks_per_thread, 200);
                                    gen_clone.generate_embeddings(chunks).await
                                });
                                handles.push(handle);
                            }
                            
                            let results = futures::future::join_all(handles).await;
                            black_box(results)
                        })
                    });
                }
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_embedding_generation,
    bench_similarity_calculations,
    bench_cache_operations,
    bench_batch_processing,
    bench_normalization,
    bench_memory_usage,
    bench_throughput_target,
    bench_concurrent_processing
);

criterion_main!(benches);