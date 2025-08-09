//! Specialized benchmarks for similarity calculations
//!
//! These benchmarks focus on optimizing similarity computations which are
//! critical for retrieval performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use embedder::similarity::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// Generate random embeddings for testing
fn generate_random_embeddings(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    
    (0..count)
        .map(|_| {
            (0..dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

// Generate normalized random embeddings
fn generate_normalized_embeddings(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    generate_random_embeddings(count, dimension, seed)
        .into_iter()
        .map(|emb| normalize_l2(&emb).unwrap())
        .collect()
}

fn bench_cosine_similarity_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity_single");
    
    let dimensions = [128, 256, 384, 512, 768, 1024, 1536, 2048];
    
    for &dim in &dimensions {
        let emb1 = generate_normalized_embeddings(1, dim, 42)[0].clone();
        let emb2 = generate_normalized_embeddings(1, dim, 43)[0].clone();
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}d", dim)),
            &(emb1, emb2),
            |b, (e1, e2)| {
                b.iter(|| {
                    let similarity = cosine_similarity(e1, e2).unwrap();
                    black_box(similarity)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_cosine_similarity_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity_batch");
    
    let dimensions = [384, 768, 1536]; // Common embedding dimensions
    let batch_sizes = [10, 50, 100, 500, 1000, 5000];
    
    for &dim in &dimensions {
        for &batch_size in &batch_sizes {
            let query = generate_normalized_embeddings(1, dim, 42)[0].clone();
            let embeddings = generate_normalized_embeddings(batch_size, dim, 123);
            
            group.throughput(Throughput::Elements(batch_size as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}d_{}batch", dim, batch_size)),
                &(query, embeddings),
                |b, (q, embs)| {
                    b.iter(|| {
                        let similarities = batch_cosine_similarity(q, embs).unwrap();
                        black_box(similarities)
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");
    
    let dimensions = [128, 384, 768, 1536];
    
    for &dim in &dimensions {
        let emb1 = generate_random_embeddings(1, dim, 42)[0].clone();
        let emb2 = generate_random_embeddings(1, dim, 43)[0].clone();
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}d", dim)),
            &(emb1, emb2),
            |b, (e1, e2)| {
                b.iter(|| {
                    let distance = euclidean_distance(e1, e2).unwrap();
                    black_box(distance)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_manhattan_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("manhattan_distance");
    
    let dimensions = [128, 384, 768, 1536];
    
    for &dim in &dimensions {
        let emb1 = generate_random_embeddings(1, dim, 42)[0].clone();
        let emb2 = generate_random_embeddings(1, dim, 43)[0].clone();
        
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}d", dim)),
            &(emb1, emb2),
            |b, (e1, e2)| {
                b.iter(|| {
                    let distance = manhattan_distance(e1, e2).unwrap();
                    black_box(distance)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_normalization");
    
    let dimensions = [128, 256, 384, 512, 768, 1024, 1536, 2048];
    
    for &dim in &dimensions {
        let embedding = generate_random_embeddings(1, dim, 42)[0].clone();
        
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}d", dim)),
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

fn bench_l2_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_norm_calculation");
    
    let dimensions = [128, 256, 384, 512, 768, 1024, 1536, 2048];
    
    for &dim in &dimensions {
        let embedding = generate_random_embeddings(1, dim, 42)[0].clone();
        
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}d", dim)),
            &embedding,
            |b, emb| {
                b.iter(|| {
                    let norm = l2_norm(emb);
                    black_box(norm)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_top_k_similar(c: &mut Criterion) {
    let mut group = c.benchmark_group("top_k_similar");
    
    let dim = 384; // Fixed dimension for this benchmark
    let corpus_sizes = [100, 500, 1000, 5000, 10000];
    let k_values = [1, 5, 10, 50, 100];
    
    for &corpus_size in &corpus_sizes {
        for &k in &k_values {
            if k > corpus_size {
                continue;
            }
            
            let query = generate_normalized_embeddings(1, dim, 42)[0].clone();
            let corpus: Vec<(Vec<f32>, usize)> = generate_normalized_embeddings(corpus_size, dim, 123)
                .into_iter()
                .enumerate()
                .map(|(i, emb)| (emb, i))
                .collect();
            
            group.throughput(Throughput::Elements(corpus_size as u64));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("corpus{}_k{}", corpus_size, k)),
                &(query, corpus, k),
                |b, (q, corp, k_val)| {
                    b.iter(|| {
                        let top_k = find_top_k_similar(q, corp, *k_val).unwrap();
                        black_box(top_k)
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn bench_pairwise_similarities(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairwise_similarities");
    
    let dim = 384;
    let embedding_counts = [10, 20, 50, 100, 200];
    
    for &count in &embedding_counts {
        let embeddings = generate_normalized_embeddings(count, dim, 42);
        
        // Number of similarity calculations is n*(n-1)/2
        let num_calculations = (count * (count - 1)) / 2;
        group.throughput(Throughput::Elements(num_calculations as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}embeddings", count)),
            &embeddings,
            |b, embs| {
                b.iter(|| {
                    let similarities = pairwise_cosine_similarities(embs).unwrap();
                    black_box(similarities)
                })
            },
        );
    }
    
    group.finish();
}

fn bench_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_clustering");
    
    let dim = 384;
    let embedding_counts = [100, 500, 1000];
    let cluster_counts = [2, 5, 10, 20];
    let max_iterations = 10;
    
    for &emb_count in &embedding_counts {
        for &k in &cluster_counts {
            if k > emb_count {
                continue;
            }
            
            let embeddings = generate_normalized_embeddings(emb_count, dim, 42);
            
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}embs_{}clusters", emb_count, k)),
                &(embeddings, k),
                |b, (embs, clusters)| {
                    b.iter(|| {
                        let (assignments, centroids) = simple_kmeans_clustering(
                            embs,
                            *clusters,
                            max_iterations,
                        ).unwrap();
                        black_box((assignments, centroids))
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn bench_similarity_matrix_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_matrix_computation");
    group.sample_size(10); // Fewer samples for expensive operations
    
    let dimensions = [128, 384, 768];
    let matrix_sizes = [10, 25, 50, 100];
    
    for &dim in &dimensions {
        for &size in &matrix_sizes {
            let embeddings = generate_normalized_embeddings(size, dim, 42);
            
            // Total number of similarity calculations
            let total_calculations = size * size;
            group.throughput(Throughput::Elements(total_calculations as u64));
            
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}x{}_{}d", size, size, dim)),
                &embeddings,
                |b, embs| {
                    b.iter(|| {
                        let matrix = pairwise_cosine_similarities(embs).unwrap();
                        black_box(matrix)
                    })
                },
            );
        }
    }
    
    group.finish();
}

fn bench_memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access_patterns");
    
    let dim = 384;
    let query = generate_normalized_embeddings(1, dim, 42)[0].clone();
    
    // Test different memory layouts and access patterns
    let sizes = [100, 1000, 10000];
    
    for &size in &sizes {
        let embeddings = generate_normalized_embeddings(size, dim, 123);
        
        // Sequential access (cache-friendly)
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("sequential_{}", size)),
            &(query.clone(), embeddings.clone()),
            |b, (q, embs)| {
                b.iter(|| {
                    let mut similarities = Vec::with_capacity(embs.len());
                    for emb in embs {
                        let sim = cosine_similarity(q, emb).unwrap();
                        similarities.push(sim);
                    }
                    black_box(similarities)
                })
            },
        );
        
        // Batch computation (vectorized)
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("batch_{}", size)),
            &(query.clone(), embeddings),
            |b, (q, embs)| {
                b.iter(|| {
                    let similarities = batch_cosine_similarity(q, embs).unwrap();
                    black_box(similarities)
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    similarity_benches,
    bench_cosine_similarity_single,
    bench_cosine_similarity_batch,
    bench_euclidean_distance,
    bench_manhattan_distance,
    bench_normalization,
    bench_l2_norm,
    bench_top_k_similar,
    bench_pairwise_similarities,
    bench_clustering,
    bench_similarity_matrix_computation,
    bench_memory_access_patterns
);

criterion_main!(similarity_benches);