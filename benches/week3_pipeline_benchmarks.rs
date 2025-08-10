//! Week 3 RAG Pipeline Performance Benchmarks
//! 
//! Comprehensive benchmarking suite for the complete RAG pipeline including:
//! - Query processing performance
//! - Response generation speed
//! - End-to-end latency measurements
//! - Throughput under load
//! - Memory efficiency
//! - Component-specific benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;
use uuid::Uuid;

// Re-use the mock components from integration tests
include!("../tests/week3_integration_tests.rs");

/// Benchmark configuration
struct BenchmarkConfig {
    pub small_doc_size: usize,
    pub medium_doc_size: usize,
    pub large_doc_size: usize,
    pub batch_sizes: Vec<usize>,
    pub query_complexity_levels: Vec<&'static str>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            small_doc_size: 500,
            medium_doc_size: 2000,
            large_doc_size: 10000,
            batch_sizes: vec![1, 5, 10, 25, 50],
            query_complexity_levels: vec![
                "simple",
                "medium complexity query with multiple terms",
                "complex analytical query requiring comprehensive understanding of multiple interconnected concepts and their relationships within the domain knowledge",
            ],
        }
    }
}

/// Generate test documents of various sizes
fn generate_test_documents(config: &BenchmarkConfig) -> Vec<(String, String)> {
    let base_content = "This is test content for benchmarking the RAG system performance. It includes various types of information including technical details, explanations, examples, and references. The content is designed to test different aspects of the system including chunking, embedding generation, storage, retrieval, and response generation capabilities.";

    vec![
        (
            "small_doc".to_string(),
            base_content.repeat(config.small_doc_size / base_content.len() + 1)[..config.small_doc_size].to_string(),
        ),
        (
            "medium_doc".to_string(),
            base_content.repeat(config.medium_doc_size / base_content.len() + 1)[..config.medium_doc_size].to_string(),
        ),
        (
            "large_doc".to_string(),
            base_content.repeat(config.large_doc_size / base_content.len() + 1)[..config.large_doc_size].to_string(),
        ),
    ]
}

/// Benchmark query processing performance
fn bench_query_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();
    
    let mut group = c.benchmark_group("query_processing");
    
    for query_complexity in &config.query_complexity_levels {
        let system = rt.block_on(async {
            let test_config = TestConfig::default();
            RagSystemIntegration::new(test_config)
        });
        
        group.bench_with_input(
            BenchmarkId::new("process_query", query_complexity.len()),
            query_complexity,
            |b, query| {
                b.to_async(&rt).iter(|| async {
                    let result = system.query_processor.process(black_box(query)).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark response generation performance
fn bench_response_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();
    
    let mut group = c.benchmark_group("response_generation");
    
    // Pre-setup system and data
    let (system, search_results, processed_query) = rt.block_on(async {
        let test_config = TestConfig::default();
        let system = RagSystemIntegration::new(test_config);
        
        // Create mock search results
        let search_results = vec![
            SearchResult {
                chunk_id: Uuid::new_v4(),
                content: "Sample search result content for benchmarking response generation".to_string(),
                similarity_score: 0.9,
                metadata: std::collections::HashMap::new(),
            },
            SearchResult {
                chunk_id: Uuid::new_v4(),
                content: "Additional context for comprehensive response generation testing".to_string(),
                similarity_score: 0.8,
                metadata: std::collections::HashMap::new(),
            },
        ];
        
        let processed_query = ProcessedQuery {
            id: Uuid::new_v4(),
            original_query: "benchmark query".to_string(),
            processed_query: "processed benchmark query".to_string(),
            intent: QueryIntent::Factual,
            entities: vec!["benchmark".to_string(), "query".to_string()],
            confidence_score: 0.9,
            processing_time: Duration::from_millis(30),
            search_strategy: "hybrid".to_string(),
        };
        
        (system, search_results, processed_query)
    });
    
    for num_results in [1, 3, 5, 10] {
        let test_results = &search_results[..num_results.min(search_results.len())];
        
        group.bench_with_input(
            BenchmarkId::new("generate_response", num_results),
            &num_results,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = system.response_generator.generate(
                        black_box(&processed_query),
                        black_box(test_results)
                    ).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark end-to-end query processing
fn bench_end_to_end_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();
    
    let mut group = c.benchmark_group("end_to_end");
    group.sample_size(20); // Reduce sample size for slower end-to-end tests
    
    // Pre-setup system with indexed documents
    let system = rt.block_on(async {
        let test_config = TestConfig::default();
        let system = RagSystemIntegration::new(test_config);
        
        let documents = generate_test_documents(&config);
        let _ = system.index_documents(&documents).await.unwrap();
        
        system
    });
    
    for query_complexity in &config.query_complexity_levels {
        group.bench_with_input(
            BenchmarkId::new("end_to_end", query_complexity.len()),
            query_complexity,
            |b, query| {
                b.to_async(&rt).iter(|| async {
                    let result = system.process_query_end_to_end(black_box(query)).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark document chunking performance
fn bench_document_chunking(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();
    let system = RagSystemIntegration::new(TestConfig::default());
    
    let mut group = c.benchmark_group("document_chunking");
    
    let test_documents = generate_test_documents(&config);
    
    for (doc_name, content) in &test_documents {
        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("chunk_document", doc_name),
            content,
            |b, content| {
                b.to_async(&rt).iter(|| async {
                    let result = system.chunker.chunk_document(
                        black_box(content),
                        black_box("benchmark_doc")
                    ).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark embedding generation performance
fn bench_embedding_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();
    let system = RagSystemIntegration::new(TestConfig::default());
    
    let mut group = c.benchmark_group("embedding_generation");
    
    // Pre-create chunks of different batch sizes
    let base_chunk = DocumentChunk {
        id: Uuid::new_v4(),
        content: "This is a test chunk for embedding generation benchmarking.".to_string(),
        embeddings: None,
        metadata: std::collections::HashMap::new(),
        references: vec![],
    };
    
    for &batch_size in &config.batch_sizes {
        let mut chunks = vec![base_chunk.clone(); batch_size];
        
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("generate_embeddings", batch_size),
            &batch_size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let mut test_chunks = chunks.clone();
                    let result = system.embedder.generate_embeddings(black_box(&mut test_chunks)).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark vector similarity search performance
fn bench_vector_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();
    
    let mut group = c.benchmark_group("vector_search");
    
    // Pre-setup system with various amounts of data
    let systems = rt.block_on(async {
        let mut systems = Vec::new();
        
        for &doc_count in &[10, 50, 100, 500] {
            let test_config = TestConfig::default();
            let system = RagSystemIntegration::new(test_config);
            
            // Generate and index documents
            let documents: Vec<_> = (0..doc_count).map(|i| {
                (
                    format!("search_doc_{}", i),
                    format!("Search benchmark document {} with unique content for testing vector similarity search performance.", i)
                )
            }).collect();
            
            let _ = system.index_documents(&documents).await.unwrap();
            systems.push((doc_count, system));
        }
        
        systems
    });
    
    for (doc_count, system) in systems {
        // Create query embedding
        let query_embedding = system.embedder.generate_mock_embedding("benchmark search query");
        
        group.bench_with_input(
            BenchmarkId::new("search_similar", doc_count),
            &doc_count,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = system.storage.search_similar(
                        black_box(&query_embedding),
                        black_box(10),
                        black_box(0.7)
                    ).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent query processing
fn bench_concurrent_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::default();
    
    let mut group = c.benchmark_group("concurrent_processing");
    group.sample_size(10); // Reduce for expensive concurrent tests
    
    // Pre-setup system
    let system = rt.block_on(async {
        let test_config = TestConfig::default();
        let system = RagSystemIntegration::new(test_config);
        
        let documents = generate_test_documents(&config);
        let _ = system.index_documents(&documents).await.unwrap();
        
        system
    });
    
    for &concurrent_queries in &[1, 2, 5, 10] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_queries", concurrent_queries),
            &concurrent_queries,
            |b, &num_queries| {
                b.to_async(&rt).iter(|| async {
                    let mut handles = Vec::new();
                    
                    for i in 0..num_queries {
                        let system = &system;
                        let query = format!("concurrent benchmark query {}", i);
                        
                        let handle = tokio::spawn(async move {
                            system.process_query_end_to_end(&query).await
                        });
                        
                        handles.push(handle);
                    }
                    
                    let results = futures::future::join_all(handles).await;
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_efficiency");
    group.sample_size(10);
    
    for &doc_count in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("memory_usage", doc_count),
            &doc_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let test_config = TestConfig::default();
                    let system = RagSystemIntegration::new(test_config);
                    
                    // Generate documents
                    let documents: Vec<_> = (0..count).map(|i| {
                        (
                            format!("memory_doc_{}", i),
                            format!("Memory efficiency test document {} with content.", i)
                        )
                    }).collect();
                    
                    // Index and query
                    let _ = system.index_documents(black_box(&documents)).await.unwrap();
                    let result = system.process_query_end_to_end("memory test query").await;
                    
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark citation processing performance
fn bench_citation_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("citation_processing");
    
    let system = RagSystemIntegration::new(TestConfig::default());
    
    // Create search results with varying numbers of sources
    for &num_sources in &[1, 3, 5, 10, 20] {
        let search_results: Vec<_> = (0..num_sources).map(|i| {
            SearchResult {
                chunk_id: Uuid::new_v4(),
                content: format!("Citation source {} with detailed content for testing citation processing performance.", i),
                similarity_score: 0.9 - (i as f64 * 0.05),
                metadata: {
                    let mut meta = std::collections::HashMap::new();
                    meta.insert("source".to_string(), format!("Source Document {}", i));
                    meta.insert("page".to_string(), format!("{}", i + 1));
                    meta
                },
            }
        }).collect();
        
        let processed_query = ProcessedQuery {
            id: Uuid::new_v4(),
            original_query: "citation benchmark query".to_string(),
            processed_query: "processed citation benchmark query".to_string(),
            intent: QueryIntent::Factual,
            entities: vec!["citation".to_string(), "benchmark".to_string()],
            confidence_score: 0.9,
            processing_time: Duration::from_millis(30),
            search_strategy: "hybrid".to_string(),
        };
        
        group.bench_with_input(
            BenchmarkId::new("process_citations", num_sources),
            &num_sources,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = system.response_generator.generate(
                        black_box(&processed_query),
                        black_box(&search_results)
                    ).await;
                    black_box(result)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark validation pipeline performance
fn bench_validation_pipeline(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("validation_pipeline");
    
    let system = RagSystemIntegration::new(TestConfig::default());
    
    // Create responses of varying complexity for validation
    let test_responses = vec![
        ("short", "Short response for validation testing."),
        ("medium", "This is a medium-length response for validation testing. It contains multiple sentences with various claims and assertions that need to be validated for accuracy and completeness."),
        ("long", "This is a comprehensive, long-form response for validation testing. It includes detailed explanations, multiple supporting arguments, extensive citations, and complex reasoning chains. The validation system must process all aspects including factual accuracy, citation completeness, logical consistency, and overall response quality. This type of response represents the most challenging validation scenario."),
    ];
    
    for (response_type, content) in test_responses {
        let mock_response = GeneratedResponse {
            query_id: Uuid::new_v4(),
            content: content.to_string(),
            confidence_score: 0.88,
            citations: vec![Citation {
                id: Uuid::new_v4(),
                source: "Test Source".to_string(),
                page: Some(1),
                confidence: 0.9,
                relevance_score: 0.85,
            }],
            generation_time: Duration::from_millis(70),
            validation_results: vec![],
            format: OutputFormat::Markdown,
        };
        
        group.bench_with_input(
            BenchmarkId::new("validate_response", response_type),
            &response_type,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    // Simulate validation processing
                    let validated = system.response_generator.validate_response(black_box(&mock_response.content));
                    black_box(validated)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark throughput under sustained load
fn bench_sustained_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("sustained_throughput");
    group.measurement_time(Duration::from_secs(30)); // Longer measurement for throughput
    group.sample_size(10);
    
    // Pre-setup system
    let system = rt.block_on(async {
        let test_config = TestConfig::default();
        let system = RagSystemIntegration::new(test_config);
        
        let documents = generate_test_documents(&BenchmarkConfig::default());
        let _ = system.index_documents(&documents).await.unwrap();
        
        system
    });
    
    let queries = vec![
        "What is the main topic?",
        "Explain the key concepts",
        "Summarize the important points",
        "How does this work?",
        "What are the benefits?",
    ];
    
    group.bench_function("sustained_queries", |b| {
        b.to_async(&rt).iter(|| async {
            let mut query_count = 0;
            let start_time = std::time::Instant::now();
            
            // Process queries for a fixed duration
            while start_time.elapsed() < Duration::from_millis(100) {
                let query = &queries[query_count % queries.len()];
                let _ = system.process_query_end_to_end(black_box(query)).await;
                query_count += 1;
            }
            
            black_box(query_count)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_query_processing,
    bench_response_generation,
    bench_end_to_end_processing,
    bench_document_chunking,
    bench_embedding_generation,
    bench_vector_search,
    bench_concurrent_processing,
    bench_memory_efficiency,
    bench_citation_processing,
    bench_validation_pipeline,
    bench_sustained_throughput
);

criterion_main!(benches);