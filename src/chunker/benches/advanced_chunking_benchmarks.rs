use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use chunker::{DocumentChunker, ChunkerConfig, BoundaryDetector};
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmarks for advanced chunking scenarios
fn benchmark_chunking_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("chunking_sizes");
    
    // Test different chunk sizes
    let chunk_sizes = vec![256, 512, 1024, 2048];
    let test_content = "This is a test sentence. ".repeat(10000); // ~250KB
    
    for chunk_size in chunk_sizes {
        group.bench_with_input(
            BenchmarkId::new("chunk_size", chunk_size),
            &chunk_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let chunker = DocumentChunker::new(size, 50).await.unwrap();
                    let chunks = chunker.chunk_document(
                        black_box(&test_content), 
                        "benchmark-doc".to_string()
                    ).await.unwrap();
                    black_box(chunks.len())
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmarks for neural vs non-neural boundary detection
fn benchmark_boundary_detection_modes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("boundary_detection");
    
    let test_content = "First paragraph with content.\n\nSecond paragraph here.\n\nThird paragraph content.".repeat(1000);
    
    group.bench_function("neural_boundaries", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ChunkerConfig {
                enable_neural_boundaries: true,
                enable_async: false,
                ..Default::default()
            };
            let chunker = DocumentChunker::with_config(512, 50, config).await.unwrap();
            let chunks = chunker.chunk_document(
                black_box(&test_content), 
                "neural-test".to_string()
            ).await.unwrap();
            black_box(chunks.len())
        });
    });
    
    group.bench_function("simple_boundaries", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ChunkerConfig {
                enable_neural_boundaries: false,
                enable_async: false,
                ..Default::default()
            };
            let chunker = DocumentChunker::with_config(512, 50, config).await.unwrap();
            let chunks = chunker.chunk_document(
                black_box(&test_content), 
                "simple-test".to_string()
            ).await.unwrap();
            black_box(chunks.len())
        });
    });
    
    group.finish();
}

/// Benchmarks for async vs sequential processing
fn benchmark_processing_modes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("processing_modes");
    
    let large_content = "This is a large document with many paragraphs. ".repeat(50000); // ~2.5MB
    
    group.bench_function("async_processing", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ChunkerConfig {
                enable_neural_boundaries: false,
                enable_async: true,
                max_concurrent_tasks: 4,
                ..Default::default()
            };
            let chunker = DocumentChunker::with_config(512, 50, config).await.unwrap();
            let chunks = chunker.chunk_document(
                black_box(&large_content), 
                "async-test".to_string()
            ).await.unwrap();
            black_box(chunks.len())
        });
    });
    
    group.bench_function("sequential_processing", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ChunkerConfig {
                enable_neural_boundaries: false,
                enable_async: false,
                ..Default::default()
            };
            let chunker = DocumentChunker::with_config(512, 50, config).await.unwrap();
            let chunks = chunker.chunk_document(
                black_box(&large_content), 
                "sequential-test".to_string()
            ).await.unwrap();
            black_box(chunks.len())
        });
    });
    
    group.finish();
}

/// Benchmarks for different document types
fn benchmark_document_types(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("document_types");
    
    // Plain text
    let plain_text = "This is plain text content. ".repeat(5000);
    
    // Markdown text
    let markdown_text = r#"# Heading
This is markdown content with **bold** and *italic* text.

## Subheading
- List item 1
- List item 2
- List item 3

```rust
fn example() {
    println!("Hello, world!");
}
```
"#.repeat(1000);
    
    // Code content
    let code_content = r#"
fn main() {
    let x = 42;
    println!("The answer is {}", x);
}

struct Example {
    field1: String,
    field2: i32,
}

impl Example {
    fn new(field1: String, field2: i32) -> Self {
        Self { field1, field2 }
    }
}
"#.repeat(1000);
    
    group.bench_function("plain_text", |b| {
        b.to_async(&rt).iter(|| async {
            let chunker = DocumentChunker::new(512, 50).await.unwrap();
            let chunks = chunker.chunk_document(
                black_box(&plain_text), 
                "plain-text".to_string()
            ).await.unwrap();
            black_box(chunks.len())
        });
    });
    
    group.bench_function("markdown_text", |b| {
        b.to_async(&rt).iter(|| async {
            let chunker = DocumentChunker::new(512, 50).await.unwrap();
            let chunks = chunker.chunk_document(
                black_box(&markdown_text), 
                "markdown-text".to_string()
            ).await.unwrap();
            black_box(chunks.len())
        });
    });
    
    group.bench_function("code_content", |b| {
        b.to_async(&rt).iter(|| async {
            let chunker = DocumentChunker::new(512, 50).await.unwrap();
            let chunks = chunker.chunk_document(
                black_box(&code_content), 
                "code-content".to_string()
            ).await.unwrap();
            black_box(chunks.len())
        });
    });
    
    group.finish();
}

/// Benchmarks for memory usage patterns
fn benchmark_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("memory_usage");
    
    // Test with different overlap ratios
    let overlap_ratios = vec![0, 10, 50, 100];
    let test_content = "Memory usage test content. ".repeat(10000);
    
    for overlap in overlap_ratios {
        group.bench_with_input(
            BenchmarkId::new("overlap", overlap),
            &overlap,
            |b, &overlap_size| {
                b.to_async(&rt).iter(|| async {
                    let chunker = DocumentChunker::new(512, overlap_size).await.unwrap();
                    let chunks = chunker.chunk_document(
                        black_box(&test_content), 
                        "memory-test".to_string()
                    ).await.unwrap();
                    // Force chunk content to stay in memory for measurement
                    let total_chars: usize = chunks.iter().map(|c| c.content.len()).sum();
                    black_box(total_chars)
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmarks for quality scoring overhead
fn benchmark_quality_scoring(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("quality_scoring");
    
    let test_content = "Quality scoring test content with proper sentences. This content should have decent quality scores. It contains multiple sentences and paragraphs for analysis.".repeat(2000);
    
    group.bench_function("with_quality_scoring", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ChunkerConfig {
                enable_neural_boundaries: false,
                quality_threshold: 0.0, // Don't filter, just score
                ..Default::default()
            };
            let chunker = DocumentChunker::with_config(512, 50, config).await.unwrap();
            let chunks = chunker.chunk_document(
                black_box(&test_content), 
                "quality-test".to_string()
            ).await.unwrap();
            let avg_quality: f64 = chunks.iter().map(|c| c.metadata.quality_score).sum::<f64>() / chunks.len() as f64;
            black_box(avg_quality)
        });
    });
    
    group.bench_function("with_quality_filtering", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ChunkerConfig {
                enable_neural_boundaries: false,
                quality_threshold: 0.7, // Filter low-quality chunks
                ..Default::default()
            };
            let chunker = DocumentChunker::with_config(512, 50, config).await.unwrap();
            let chunks = chunker.chunk_document(
                black_box(&test_content), 
                "filtered-test".to_string()
            ).await.unwrap();
            black_box(chunks.len())
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_chunking_sizes,
    benchmark_boundary_detection_modes,
    benchmark_processing_modes,
    benchmark_document_types,
    benchmark_memory_usage,
    benchmark_quality_scoring
);

criterion_main!(benches);