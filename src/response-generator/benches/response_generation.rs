//! Benchmarks for response generation performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use response_generator::{
    Config, ResponseGenerator, GenerationRequest, ContextChunk, Source, OutputFormat,
};
use std::collections::HashMap;
use tokio::runtime::Runtime;
use uuid::Uuid;

/// Benchmark basic response generation
fn bench_basic_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = Config::default();
    let generator = ResponseGenerator::new(config);
    
    let queries = vec![
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain neural networks",
        "What are the benefits of cloud computing?",
        "Describe software development processes",
    ];
    
    let mut group = c.benchmark_group("basic_generation");
    
    for query in queries {
        group.bench_with_input(
            BenchmarkId::new("basic", query.len()),
            query,
            |b, query| {
                b.to_async(&rt).iter(|| async {
                    let request = GenerationRequest::builder()
                        .query(black_box(query.to_string()))
                        .build()
                        .unwrap();
                    
                    black_box(generator.generate(request).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark response generation with context
fn bench_generation_with_context(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = Config::default();
    let generator = ResponseGenerator::new(config);
    
    // Create context chunks of different sizes
    let context_sizes = vec![100, 500, 1000, 2000];
    
    let mut group = c.benchmark_group("generation_with_context");
    
    for size in context_sizes {
        group.throughput(Throughput::Bytes(size as u64));
        
        let context_content = "a".repeat(size);
        let context_chunk = ContextChunk {
            content: context_content,
            source: Source {
                id: Uuid::new_v4(),
                title: "Benchmark Source".to_string(),
                url: None,
                document_type: "text".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.8,
            position: Some(0),
            metadata: HashMap::new(),
        };
        
        group.bench_with_input(
            BenchmarkId::new("with_context", size),
            &context_chunk,
            |b, chunk| {
                b.to_async(&rt).iter(|| async {
                    let request = GenerationRequest::builder()
                        .query("Summarize the provided content")
                        .add_context(black_box(chunk.clone()))
                        .build()
                        .unwrap();
                    
                    black_box(generator.generate(request).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark different output formats
fn bench_output_formats(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = Config::default();
    let generator = ResponseGenerator::new(config);
    
    let formats = vec![
        OutputFormat::Json,
        OutputFormat::Markdown,
        OutputFormat::Text,
        OutputFormat::Html,
        OutputFormat::Xml,
    ];
    
    let mut group = c.benchmark_group("output_formats");
    
    for format in formats {
        group.bench_with_input(
            BenchmarkId::new("format", format!("{:?}", format)),
            &format,
            |b, fmt| {
                b.to_async(&rt).iter(|| async {
                    let request = GenerationRequest::builder()
                        .query("Test query for format benchmarking")
                        .format(black_box(fmt.clone()))
                        .build()
                        .unwrap();
                    
                    black_box(generator.generate(request).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent request handling
fn bench_concurrent_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = Config::default();
    let generator = ResponseGenerator::new(config);
    
    let concurrency_levels = vec![1, 2, 4, 8, 16];
    
    let mut group = c.benchmark_group("concurrent_generation");
    
    for concurrency in concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent", concurrency),
            &concurrency,
            |b, &level| {
                b.to_async(&rt).iter(|| async {
                    let requests: Vec<_> = (0..level).map(|i| {
                        GenerationRequest::builder()
                            .query(format!("Concurrent query {}", i))
                            .build()
                            .unwrap()
                    }).collect();
                    
                    let futures: Vec<_> = requests.into_iter().map(|request| {
                        generator.generate(black_box(request))
                    }).collect();
                    
                    black_box(futures::future::join_all(futures).await)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark validation layers
fn bench_validation_layers(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Test with different validation configurations
    let validation_configs = vec![
        ("minimal", create_minimal_validation_config()),
        ("standard", Config::default()),
        ("strict", create_strict_validation_config()),
    ];
    
    let mut group = c.benchmark_group("validation_layers");
    
    for (config_name, config) in validation_configs {
        let generator = ResponseGenerator::new(config);
        
        group.bench_with_input(
            BenchmarkId::new("validation", config_name),
            config_name,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let request = GenerationRequest::builder()
                        .query("Test query for validation benchmarking")
                        .build()
                        .unwrap();
                    
                    black_box(generator.generate(request).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark citation processing
fn bench_citation_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = Config::default();
    let generator = ResponseGenerator::new(config);
    
    // Test with different numbers of sources
    let source_counts = vec![1, 5, 10, 20];
    
    let mut group = c.benchmark_group("citation_processing");
    
    for count in source_counts {
        let context_chunks: Vec<ContextChunk> = (0..count).map(|i| {
            ContextChunk {
                content: format!("Content from source {} with information relevant to the query.", i),
                source: Source {
                    id: Uuid::new_v4(),
                    title: format!("Source {}", i),
                    url: Some(format!("https://source{}.com", i)),
                    document_type: "article".to_string(),
                    metadata: HashMap::new(),
                },
                relevance_score: 0.8,
                position: Some(i),
                metadata: HashMap::new(),
            }
        }).collect();
        
        group.bench_with_input(
            BenchmarkId::new("citations", count),
            &context_chunks,
            |b, chunks| {
                b.to_async(&rt).iter(|| async {
                    let request = GenerationRequest::builder()
                        .query("What do the sources say about this topic?")
                        .context(black_box(chunks.clone()))
                        .build()
                        .unwrap();
                    
                    black_box(generator.generate(request).await.unwrap())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark streaming response generation
fn bench_streaming_generation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = Config::default();
    let generator = ResponseGenerator::new(config);
    
    let mut group = c.benchmark_group("streaming_generation");
    
    group.bench_function("streaming", |b| {
        b.to_async(&rt).iter(|| async {
            let request = GenerationRequest::builder()
                .query("Generate a comprehensive response about machine learning")
                .build()
                .unwrap();
            
            let stream = generator.generate_stream(black_box(request)).await.unwrap();
            let chunks: Vec<_> = tokio_stream::StreamExt::collect(stream).await;
            black_box(chunks)
        });
    });
    
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = Config::default();
    let generator = ResponseGenerator::new(config);
    
    let mut group = c.benchmark_group("memory_usage");
    
    // Test with large context
    let large_context = ContextChunk {
        content: "a".repeat(10_000), // 10KB content
        source: Source {
            id: Uuid::new_v4(),
            title: "Large Source".to_string(),
            url: None,
            document_type: "text".to_string(),
            metadata: HashMap::new(),
        },
        relevance_score: 0.8,
        position: Some(0),
        metadata: HashMap::new(),
    };
    
    group.bench_function("large_context", |b| {
        b.to_async(&rt).iter(|| async {
            let request = GenerationRequest::builder()
                .query("Summarize the large content provided")
                .add_context(black_box(large_context.clone()))
                .build()
                .unwrap();
            
            black_box(generator.generate(request).await.unwrap())
        });
    });
    
    group.finish();
}

/// Create minimal validation configuration for benchmarking
fn create_minimal_validation_config() -> Config {
    let mut config = Config::default();
    config.validation.min_confidence_threshold = 0.3;
    config.validation.max_validation_time = std::time::Duration::from_millis(10);
    config
}

/// Create strict validation configuration for benchmarking
fn create_strict_validation_config() -> Config {
    let mut config = Config::default();
    config.validation.min_confidence_threshold = 0.9;
    config.validation.strict_mode = true;
    config.validation.max_validation_time = std::time::Duration::from_millis(100);
    config
}

criterion_group!(
    benches,
    bench_basic_generation,
    bench_generation_with_context,
    bench_output_formats,
    bench_concurrent_generation,
    bench_validation_layers,
    bench_citation_processing,
    bench_streaming_generation,
    bench_memory_usage
);

criterion_main!(benches);