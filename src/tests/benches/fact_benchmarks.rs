//! FACT Performance Benchmarks
//! 
//! Criterion-based benchmarks for FACT integration performance validation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use std::sync::Arc;
use tokio::runtime::Runtime;
use futures::future::join_all;

// Import our test modules
use fact_integration_tests::mocks::{MockFACTClientFactory, RealisticFACTSimulator, SimulatorConfig};
use fact_integration_tests::test_utils::{generate_benchmark_queries, create_compliance_response};

/// Benchmark cache hit operations with various batch sizes
fn benchmark_cache_hits(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let fact_client = runtime.block_on(async {
        Arc::new(MockFACTClientFactory::create_cache_hit_optimized())
    });

    let mut group = c.benchmark_group("fact_cache_hits");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(10));
    
    // Single cache hit benchmark
    group.bench_function("single_hit", |b| {
        b.to_async(&runtime).iter(|| async {
            let result = fact_client.get(black_box("benchmark_key")).await;
            assert!(result.is_ok());
            result
        });
    });
    
    // Batch cache hit benchmarks
    for batch_size in [1, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_hits", batch_size),
            batch_size,
            |b, &size| {
                let queries = generate_benchmark_queries(size);
                b.to_async(&runtime).iter(|| {
                    let client = fact_client.clone();
                    let queries = queries.clone();
                    async move {
                        let mut handles = Vec::with_capacity(size);
                        for query in queries {
                            let client = client.clone();
                            handles.push(async move {
                                client.get(&query).await
                            });
                        }
                        let results = join_all(handles).await;
                        black_box(results)
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache miss operations
fn benchmark_cache_misses(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let fact_client = runtime.block_on(async {
        Arc::new(MockFACTClientFactory::create_cache_miss_simulation())
    });

    let mut group = c.benchmark_group("fact_cache_misses");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(15)); // Longer for cache misses
    
    group.bench_function("single_miss", |b| {
        b.to_async(&runtime).iter(|| async {
            let key = format!("miss_key_{}", uuid::Uuid::new_v4());
            let result = fact_client.get(black_box(&key)).await;
            assert!(result.is_ok());
            result
        });
    });
    
    group.finish();
}

/// Benchmark concurrent access patterns
fn benchmark_concurrent_access(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let simulator = runtime.block_on(async {
        let config = SimulatorConfig {
            cache_hit_latency_ms: 15,
            cache_miss_latency_ms: 75,
            error_rate: 0.0, // No errors for clean benchmarks
            ..Default::default()
        };
        Arc::new(RealisticFACTSimulator::new(config))
    });

    let mut group = c.benchmark_group("concurrent_access");
    group.measurement_time(Duration::from_secs(20));
    
    for concurrency in [1, 10, 25, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_requests", concurrency),
            concurrency,
            |b, &concurrent_count| {
                let queries = generate_benchmark_queries(concurrent_count);
                b.to_async(&runtime).iter(|| {
                    let simulator = simulator.clone();
                    let queries = queries.clone();
                    async move {
                        let mut handles = Vec::with_capacity(concurrent_count);
                        
                        for (i, query) in queries.iter().enumerate() {
                            let client = simulator.clone();
                            let query = query.clone();
                            
                            handles.push(tokio::spawn(async move {
                                client.get(&query).await
                            }));
                        }
                        
                        let results = join_all(handles).await;
                        black_box(results)
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark search operations
fn benchmark_search_operations(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let simulator = runtime.block_on(async {
        Arc::new(RealisticFACTSimulator::new(SimulatorConfig::default()))
    });

    let mut group = c.benchmark_group("search_operations");
    group.throughput(Throughput::Elements(1));
    
    let search_queries = vec![
        "PCI DSS encryption requirements",
        "GDPR consent mechanisms", 
        "ISO 27001 access control",
        "SOX financial reporting controls",
        "HIPAA data protection",
    ];
    
    for query in search_queries.iter() {
        group.bench_with_input(
            BenchmarkId::new("search", query),
            query,
            |b, &query| {
                b.to_async(&runtime).iter(|| async {
                    let result = simulator.search(black_box(query)).await;
                    assert!(result.is_ok());
                    result
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark citation retrieval operations
fn benchmark_citation_operations(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let simulator = runtime.block_on(async {
        Arc::new(RealisticFACTSimulator::new(SimulatorConfig::default()))
    });

    let mut group = c.benchmark_group("citation_operations");
    group.throughput(Throughput::Elements(1));
    
    let citation_queries = vec![
        "pci_dss_requirements",
        "gdpr_consent_requirements", 
        "iso_27001_access_control",
        "sox_documentation_requirements",
    ];
    
    for query in citation_queries.iter() {
        group.bench_with_input(
            BenchmarkId::new("with_citations", query),
            query,
            |b, &query| {
                b.to_async(&runtime).iter(|| async {
                    let result = simulator.get_with_citations(black_box(query)).await;
                    assert!(result.is_ok());
                    result
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark consensus operations
fn benchmark_consensus_operations(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let simulator = runtime.block_on(async {
        Arc::new(RealisticFACTSimulator::new(SimulatorConfig::default()))
    });

    let mut group = c.benchmark_group("consensus_operations");
    group.throughput(Throughput::Elements(1));
    
    // Create sample query results for validation
    let query_results = vec![
        create_test_query_result("High confidence result", 0.95),
        create_test_query_result("Medium confidence result", 0.75),
        create_test_query_result("Low confidence result", 0.55),
    ];
    
    for (i, result) in query_results.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("validate", format!("confidence_{}", i)),
            result,
            |b, result| {
                b.to_async(&runtime).iter(|| async {
                    let validation_result = simulator.validate(black_box(result)).await;
                    assert!(validation_result.is_ok());
                    validation_result
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark metrics collection operations
fn benchmark_metrics_operations(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    
    let simulator = runtime.block_on(async {
        // Pre-load some operations to generate metrics
        let sim = Arc::new(RealisticFACTSimulator::new(SimulatorConfig::default()));
        
        // Perform some operations to generate realistic metrics
        for i in 0..100 {
            let _ = sim.get(&format!("warmup_key_{}", i)).await;
        }
        
        sim
    });

    let mut group = c.benchmark_group("metrics_operations");
    group.throughput(Throughput::Elements(1));
    
    group.bench_function("get_metrics", |b| {
        b.to_async(&runtime).iter(|| async {
            let result = simulator.get_metrics().await;
            assert!(result.is_ok());
            black_box(result)
        });
    });
    
    group.bench_function("health_check", |b| {
        b.to_async(&runtime).iter(|| async {
            let result = simulator.health_check().await;
            assert!(result.is_ok());
            black_box(result)
        });
    });
    
    group.finish();
}

/// Benchmark data serialization/deserialization
fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    
    // Test with various data sizes
    let data_sizes = [100, 1024, 10240, 102400]; // 100B, 1KB, 10KB, 100KB
    
    for size in data_sizes.iter() {
        let test_data = create_compliance_response(&format!("test_query_size_{}", size));
        
        group.bench_with_input(
            BenchmarkId::new("serialize", format!("{}B", size)),
            &test_data,
            |b, data| {
                b.iter(|| {
                    let serialized = serde_json::to_vec(black_box(data));
                    assert!(serialized.is_ok());
                    black_box(serialized)
                });
            },
        );
        
        let serialized = serde_json::to_vec(&test_data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("deserialize", format!("{}B", size)),
            &serialized,
            |b, data| {
                b.iter(|| {
                    let deserialized: Result<serde_json::Value, _> = serde_json::from_slice(black_box(data));
                    assert!(deserialized.is_ok());
                    black_box(deserialized)
                });
            },
        );
    }
    
    group.finish();
}

/// Helper function to create test query results
fn create_test_query_result(content: &str, confidence: f64) -> fact_integration_tests::mocks::QueryResult {
    use fact_integration_tests::mocks::{QueryResult, Citation};
    use std::collections::HashMap;
    
    QueryResult {
        id: uuid::Uuid::new_v4(),
        content: content.to_string(),
        confidence,
        citations: vec![
            Citation {
                source: "Test Source".to_string(),
                section: "Section 1".to_string(),
                page: Some(42),
                confidence: confidence,
            }
        ],
        metadata: HashMap::new(),
    }
}

criterion_group!(
    benches,
    benchmark_cache_hits,
    benchmark_cache_misses,
    benchmark_concurrent_access,
    benchmark_search_operations,
    benchmark_citation_operations,
    benchmark_consensus_operations,
    benchmark_metrics_operations,
    benchmark_serialization
);

criterion_main!(benches);