//! Comprehensive performance benchmarks for the chunker system
//!
//! These benchmarks target >100MB/sec processing performance and measure
//! various aspects of chunking performance including throughput, latency,
//! memory usage, and scalability.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput, BatchSize};
use chunker::{DocumentChunker, BoundaryDetector};
use std::time::{Duration, Instant};
use std::sync::Arc;

fn create_test_documents() -> Vec<(String, String)> {
    vec![
        (
            "small".to_string(),
            "This is a small test document. ".repeat(100), // ~3KB
        ),
        (
            "medium".to_string(),
            "This is a medium test document with more content. ".repeat(1000), // ~47KB
        ),
        (
            "large".to_string(),
            "This is a large test document with lots of content for performance testing. ".repeat(10000), // ~770KB
        ),
        (
            "complex".to_string(),
            create_complex_document(),
        ),
    ]
}

fn create_complex_document() -> String {
    format!(
        r#"
# Document Title

## Section 1: Introduction

This document outlines security requirements and compliance standards.
See section 2.1 for detailed requirements.

### Subsection 1.1: Overview

Here's what we'll cover:
- Security policies
- Data protection measures  
- Compliance frameworks

## Section 2: Requirements

### Section 2.1: Data Security

The following table shows encryption requirements:

| Data Type | Encryption | Key Length |
|-----------|------------|------------|
| PCI Data  | AES-256    | 256 bits   |
| PII Data  | AES-128    | 128 bits   |

Code example:

```rust
fn encrypt_data(data: &str) -> Result<String, Error> {{
    // Implementation here
    Ok(encrypted_data)
}}
```

> Important: All data must be encrypted at rest [1].

For more information, visit https://example.com/security-guide.

### Section 2.2: Access Control

{}

## References

[1] Security Standards Document, 2024
"#,
        "Access control requirements and implementation details. ".repeat(500)
    )
}

fn bench_chunking_performance(c: &mut Criterion) {
    let test_documents = create_test_documents();
    let chunker = DocumentChunker::new(512, 50).unwrap();

    let mut group = c.benchmark_group("chunking_performance");
    
    for (name, content) in test_documents {
        let size = content.len();
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("chunk_document", &name),
            &content,
            |b, content| {
                b.iter(|| chunker.chunk_document(content));
            },
        );
    }
    
    group.finish();
}

fn bench_chunking_different_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunking_sizes");
    let content = "This is test content for chunking benchmarks. ".repeat(2000); // ~94KB
    
    let chunk_sizes = vec![256, 512, 1024, 2048];
    let overlap_sizes = vec![25, 50, 100];
    
    for chunk_size in chunk_sizes {
        for overlap in &overlap_sizes {
            if *overlap < chunk_size {
                let chunker = DocumentChunker::new(chunk_size, *overlap).unwrap();
                let name = format!("chunk_{}_{}", chunk_size, overlap);
                
                group.throughput(Throughput::Bytes(content.len() as u64));
                group.bench_with_input(
                    BenchmarkId::new("chunk_sizes", name),
                    &content,
                    |b, content| {
                        b.iter(|| chunker.chunk_document(content));
                    },
                );
            }
        }
    }
    
    group.finish();
}

fn bench_boundary_detection(c: &mut Criterion) {
    let detector = BoundaryDetector::new().unwrap();
    let test_cases = vec![
        ("simple", "Sentence one. Sentence two.\n\nNew paragraph.".repeat(100)),
        ("complex", create_complex_document()),
        ("no_boundaries", "Continuous text without clear boundaries or punctuation marks".repeat(500)),
        ("many_boundaries", "Short. Very. Many. Sentences. With. Boundaries.\n\n".repeat(200)),
    ];

    let mut group = c.benchmark_group("boundary_detection");
    
    for (name, content) in test_cases {
        let size = content.len();
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("detect_boundaries", name),
            &content,
            |b, content| {
                b.iter(|| detector.detect_boundaries(content));
            },
        );
    }
    
    group.finish();
}

fn bench_concurrent_chunking(c: &mut Criterion) {
    use std::sync::Arc;
    use std::thread;
    
    let chunker = Arc::new(DocumentChunker::new(512, 50).unwrap());
    let content = "Concurrent test content. ".repeat(1000);
    
    let mut group = c.benchmark_group("concurrent_chunking");
    
    let thread_counts = vec![1, 2, 4, 8];
    
    for thread_count in thread_counts {
        group.throughput(Throughput::Bytes((content.len() * thread_count) as u64));
        
        group.bench_with_input(
            BenchmarkId::new("concurrent", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count)
                        .map(|_| {
                            let chunker_clone = Arc::clone(&chunker);
                            let content_clone = content.clone();
                            thread::spawn(move || {
                                chunker_clone.chunk_document(&content_clone)
                            })
                        })
                        .collect();
                    
                    let results: Vec<_> = handles
                        .into_iter()
                        .map(|h| h.join().unwrap())
                        .collect();
                    
                    results
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(10));
    
    let chunker = DocumentChunker::new(512, 50).unwrap();
    let large_content = "Memory usage test content. ".repeat(50000); // ~1.35MB
    
    group.throughput(Throughput::Bytes(large_content.len() as u64));
    group.bench_function("large_document_memory", |b| {
        b.iter(|| {
            let chunks = chunker.chunk_document(&large_content);
            // Force usage of chunks to prevent optimization
            criterion::black_box(chunks.len());
        });
    });
    
    group.finish();
}

fn bench_real_world_scenarios(c: &mut Criterion) {
    let chunker = DocumentChunker::new(512, 50).unwrap();
    
    // Simulate real-world document types
    let scenarios = vec![
        ("technical_doc", create_technical_document()),
        ("legal_doc", create_legal_document()),
        ("mixed_content", create_mixed_content_document()),
    ];

    let mut group = c.benchmark_group("real_world_scenarios");
    
    for (name, content) in scenarios {
        let size = content.len();
        group.throughput(Throughput::Bytes(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("real_world", name),
            &content,
            |b, content| {
                b.iter(|| {
                    let chunks = chunker.chunk_document(content);
                    // Simulate processing results
                    criterion::black_box(chunks.iter().map(|c| c.metadata.quality_score).sum::<f32>());
                });
            },
        );
    }
    
    group.finish();
}

fn create_technical_document() -> String {
    format!(
        r#"
# Technical Specification

## Overview

This document describes the technical implementation details.

### Architecture

```rust
pub struct System {{
    components: Vec<Component>,
    config: Configuration,
}}

impl System {{
    pub fn initialize() -> Result<Self, Error> {{
        // Implementation
    }}
}}
```

## Requirements

### Functional Requirements

1. The system must process data at >100MB/sec
2. Latency must be <10ms for 99th percentile
3. Memory usage must not exceed 2GB

### Non-Functional Requirements

- Availability: 99.99%
- Scalability: Handle 10K concurrent users
- Security: Encrypt all data in transit and at rest

## Implementation

{}

### Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Throughput | 100MB/s | 95MB/s |
| Latency | <10ms | 8ms |
| Memory | <2GB | 1.5GB |

## Testing

Test cases cover:
- Unit tests (>90% coverage)
- Integration tests
- Performance benchmarks
- Security validation
"#,
        "Detailed implementation notes and code examples. ".repeat(300)
    )
}

fn create_legal_document() -> String {
    format!(
        r#"
# TERMS AND CONDITIONS

## 1. DEFINITIONS

For the purposes of this Agreement:

1.1 "Company" means the entity providing the services.
1.2 "User" means any person or entity using the services.
1.3 "Services" means the software and related services.

## 2. TERMS OF USE

2.1 License Grant
Subject to these Terms, Company grants User a limited, non-exclusive license.

2.2 Restrictions
User shall not:
- Reverse engineer the software
- Distribute without authorization
- Use for illegal purposes

## 3. PRIVACY POLICY

{}

## 4. LIABILITY

4.1 Limitation of Liability
Company's liability is limited to the maximum extent permitted by law.

4.2 Indemnification
User agrees to indemnify Company against all claims.

## 5. TERMINATION

This Agreement may be terminated by either party with 30 days notice.
"#,
        "Privacy policy details and data handling procedures. ".repeat(200)
    )
}

fn create_mixed_content_document() -> String {
    format!(
        r#"
# Mixed Content Document

## Section A: Text Content

{}

## Section B: Code Examples

```python
def process_data(input_data):
    """Process input data and return results."""
    results = []
    for item in input_data:
        processed = transform(item)
        results.append(processed)
    return results
```

## Section C: Tables and Lists

### Requirements Table

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| REQ-001 | User authentication | High | Complete |
| REQ-002 | Data encryption | High | In Progress |
| REQ-003 | Audit logging | Medium | Pending |

### Feature List

- Feature A: Basic functionality
  - Sub-feature A1
  - Sub-feature A2
- Feature B: Advanced options
  - Sub-feature B1
  - Sub-feature B2

## Section D: References

For more details, see:
- Section 2.1 of the technical specification
- Reference [1] for implementation guidelines
- https://example.com/documentation

### Footnotes

[1] Implementation Guidelines Document, Version 2.0
"#,
        "Mixed content with various formatting and structures. ".repeat(150)
    )
}

// Performance regression tests
fn bench_performance_regression(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_regression");
    group.measurement_time(Duration::from_secs(15));
    
    let chunker = DocumentChunker::new(512, 50).unwrap();
    let test_content = "Performance regression test content. ".repeat(25000); // ~925KB
    
    // This benchmark helps detect performance regressions
    group.throughput(Throughput::Bytes(test_content.len() as u64));
    group.bench_function("regression_baseline", |b| {
        b.iter(|| {
            let start = std::time::Instant::now();
            let chunks = chunker.chunk_document(&test_content);
            let duration = start.elapsed();
            
            // Verify performance target (100MB/sec minimum)
            let mb_per_sec = (test_content.len() as f64 / 1_000_000.0) / duration.as_secs_f64();
            assert!(mb_per_sec > 100.0, "Performance regression detected: {:.2} MB/s", mb_per_sec);
            
            criterion::black_box(chunks)
        });
    });
    
    group.finish();
}

// High-performance benchmarks targeting 100MB/sec
fn bench_high_performance_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_performance");
    group.measurement_time(Duration::from_secs(20));
    group.warm_up_time(Duration::from_secs(3));
    group.sample_size(30);
    
    let chunker = DocumentChunker::new(1024, 100).unwrap();
    
    // Test different sizes to verify 100MB/sec target
    let test_sizes = vec![
        ("1MB", 1_000_000),
        ("5MB", 5_000_000),
        ("10MB", 10_000_000),
        ("25MB", 25_000_000),
        ("50MB", 50_000_000),
    ];
    
    for (name, size_bytes) in test_sizes {
        let content = "High performance test content with various punctuation and structure. ".repeat(size_bytes / 70);
        let actual_size = content.len();
        
        group.throughput(Throughput::Bytes(actual_size as u64));
        group.bench_with_input(
            BenchmarkId::new("throughput_target", name),
            &content,
            |b, content| {
                b.iter_batched(
                    || content.clone(),
                    |content| {
                        let start = Instant::now();
                        let chunks = chunker.chunk_document(&content);
                        let duration = start.elapsed();
                        
                        // Verify performance target
                        let mb_per_sec = (content.len() as f64 / 1_000_000.0) / duration.as_secs_f64();
                        if mb_per_sec < 100.0 && content.len() > 1_000_000 {
                            eprintln!("⚠️  Performance below target: {:.2} MB/sec for {} bytes", mb_per_sec, content.len());
                        }
                        
                        criterion::black_box(chunks)
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    
    group.finish();
}

// Latency-focused benchmarks
fn bench_latency_characteristics(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency");
    group.measurement_time(Duration::from_secs(10));
    
    let chunker = DocumentChunker::new(512, 50).unwrap();
    
    let latency_tests = vec![
        ("small_doc", "Quick latency test.".repeat(100)),
        ("medium_doc", "Medium latency test document. ".repeat(1000)),
        ("large_doc", "Large latency test document for timing. ".repeat(10000)),
    ];
    
    for (name, content) in latency_tests {
        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("p99_latency", name),
            &content,
            |b, content| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::new(0, 0);
                    let mut durations = Vec::with_capacity(iters as usize);
                    
                    for _ in 0..iters {
                        let start = Instant::now();
                        let chunks = chunker.chunk_document(content);
                        let duration = start.elapsed();
                        
                        criterion::black_box(chunks);
                        durations.push(duration);
                        total_duration += duration;
                    }
                    
                    // Calculate p99 latency
                    durations.sort();
                    let p99_idx = ((durations.len() as f64) * 0.99) as usize;
                    let p99_latency = durations[p99_idx.min(durations.len() - 1)];
                    
                    if p99_latency.as_millis() > 100 {
                        eprintln!("⚠️  High p99 latency: {}ms for {}", p99_latency.as_millis(), name);
                    }
                    
                    total_duration
                });
            },
        );
    }
    
    group.finish();
}

// Memory efficiency benchmarks
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.measurement_time(Duration::from_secs(15));
    
    let chunker = DocumentChunker::new(1024, 100).unwrap();
    
    // Test memory usage patterns
    let memory_tests = vec![
        ("streaming_simulation", create_streaming_content()),
        ("peak_memory", create_peak_memory_content()),
        ("fragmentation_test", create_fragmentation_content()),
    ];
    
    for (name, content) in memory_tests {
        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("memory_profile", name),
            &content,
            |b, content| {
                b.iter_batched(
                    || {
                        // Pre-allocate to simulate real conditions
                        let mut buffer = String::with_capacity(content.len());
                        buffer.push_str(content);
                        buffer
                    },
                    |content| {
                        let chunks = chunker.chunk_document(&content);
                        
                        // Simulate processing chunks (memory retention)
                        let total_chunk_size: usize = chunks.iter().map(|c| c.content.len()).sum();
                        let memory_ratio = total_chunk_size as f64 / content.len() as f64;
                        
                        if memory_ratio > 2.5 {
                            eprintln!("⚠️  High memory ratio: {:.2}x for {}", memory_ratio, name);
                        }
                        
                        criterion::black_box(chunks)
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    
    group.finish();
}

// Scalability benchmarks
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.measurement_time(Duration::from_secs(20));
    
    let chunker = Arc::new(DocumentChunker::new(512, 50).unwrap());
    
    // Test concurrent processing scalability
    let thread_counts = vec![1, 2, 4, 8, 16];
    let base_content = "Scalability test content with various structures and lengths. ".repeat(5000);
    
    for thread_count in thread_counts {
        let total_size = base_content.len() * thread_count;
        group.throughput(Throughput::Bytes(total_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("concurrent_throughput", thread_count),
            &thread_count,
            |b, &thread_count| {
                b.iter_batched(
                    || {
                        let chunker = Arc::clone(&chunker);
                        let content = base_content.clone();
                        (chunker, content)
                    },
                    |(chunker, content)| {
                        use std::thread;
                        
                        let start = Instant::now();
                        let handles: Vec<_> = (0..thread_count)
                            .map(|i| {
                                let chunker_clone = Arc::clone(&chunker);
                                let content_clone = format!("{} Thread {}", content, i);
                                
                                thread::spawn(move || {
                                    chunker_clone.chunk_document(&content_clone)
                                })
                            })
                            .collect();
                        
                        let results: Vec<_> = handles
                            .into_iter()
                            .map(|h| h.join().unwrap())
                            .collect();
                        
                        let duration = start.elapsed();
                        let total_processed = results.iter().map(|chunks| {
                            chunks.iter().map(|c| c.content.len()).sum::<usize>()
                        }).sum::<usize>();
                        
                        let mb_per_sec = (total_processed as f64 / 1_000_000.0) / duration.as_secs_f64();
                        
                        if thread_count == 1 {
                            eprintln!("📊 Single-thread baseline: {:.2} MB/sec", mb_per_sec);
                        } else {
                            eprintln!("📊 {}-thread throughput: {:.2} MB/sec", thread_count, mb_per_sec);
                        }
                        
                        criterion::black_box(results)
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    
    group.finish();
}

// Specialized content benchmarks
fn bench_specialized_content(c: &mut Criterion) {
    let mut group = c.benchmark_group("specialized_content");
    group.measurement_time(Duration::from_secs(12));
    
    let chunker = DocumentChunker::new(1024, 100).unwrap();
    
    let specialized_tests = vec![
        ("code_heavy", create_code_heavy_document()),
        ("table_heavy", create_table_heavy_document()),
        ("reference_heavy", create_reference_heavy_document()),
        ("unicode_heavy", create_unicode_heavy_document()),
        ("mixed_structure", create_mixed_structure_document()),
    ];
    
    for (name, content) in specialized_tests {
        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("specialized", name),
            &content,
            |b, content| {
                b.iter_batched(
                    || content.clone(),
                    |content| {
                        let start = Instant::now();
                        let chunks = chunker.chunk_document(&content);
                        let duration = start.elapsed();
                        
                        // Verify quality for specialized content
                        let avg_quality: f32 = chunks.iter().map(|c| c.metadata.quality_score).sum::<f32>() / chunks.len() as f32;
                        let reference_count: usize = chunks.iter().map(|c| c.references.len()).sum();
                        
                        if avg_quality < 0.3 {
                            eprintln!("⚠️  Low average quality {:.2} for {}", avg_quality, name);
                        }
                        
                        eprintln!("📈 {} - Quality: {:.2}, References: {}, Speed: {:.2} MB/sec", 
                                name, avg_quality, reference_count,
                                (content.len() as f64 / 1_000_000.0) / duration.as_secs_f64());
                        
                        criterion::black_box(chunks)
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    
    group.finish();
}

// Boundary detection performance
fn bench_boundary_detection_advanced(c: &mut Criterion) {
    let mut group = c.benchmark_group("boundary_detection_advanced");
    group.measurement_time(Duration::from_secs(10));
    
    let detector = BoundaryDetector::new().unwrap();
    
    let boundary_tests = vec![
        ("dense_boundaries", create_dense_boundary_content()),
        ("sparse_boundaries", create_sparse_boundary_content()),
        ("complex_structure", create_complex_structure_content()),
        ("no_clear_boundaries", "Continuous text without clear boundaries or punctuation marks flowing seamlessly together ".repeat(2000)),
    ];
    
    for (name, content) in boundary_tests {
        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("boundary_perf", name),
            &content,
            |b, content| {
                b.iter_batched(
                    || content.clone(),
                    |content| {
                        let start = Instant::now();
                        let boundaries = detector.detect_boundaries(&content).unwrap_or_default();
                        let duration = start.elapsed();
                        
                        let boundaries_per_sec = boundaries.len() as f64 / duration.as_secs_f64();
                        let chars_per_sec = content.len() as f64 / duration.as_secs_f64();
                        
                        eprintln!("🎯 {} - Boundaries: {}, Rate: {:.0} boundaries/sec, {:.2} MB/sec",
                                name, boundaries.len(), boundaries_per_sec, chars_per_sec / 1_000_000.0);
                        
                        criterion::black_box(boundaries)
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
    
    group.finish();
}

// Helper functions for creating test content
fn create_streaming_content() -> String {
    (0..10000).map(|i| format!("Streaming content block {} with data. ", i)).collect()
}

fn create_peak_memory_content() -> String {
    let mut content = String::with_capacity(5_000_000);
    for i in 0..50000 {
        content.push_str(&format!("Peak memory test {} with longer content to simulate real world usage patterns. ", i));
    }
    content
}

fn create_fragmentation_content() -> String {
    let mut content = String::new();
    for i in 0..10000 {
        if i % 3 == 0 {
            content.push_str(&format!("# Heading {}\n\n", i / 3));
        } else if i % 5 == 0 {
            content.push_str("```rust\nfn test() { println!(\"fragmentation\"); }\n```\n\n");
        } else {
            content.push_str(&format!("Fragmentation test content {} with mixed structures. ", i));
        }
    }
    content
}

fn create_code_heavy_document() -> String {
    let code_blocks = vec![
        "```rust\nfn main() {\n    println!(\"Hello, world!\");\n}\n```",
        "```python\ndef process_data(data):\n    return [x * 2 for x in data]\n```",
        "```javascript\nfunction calculate(x, y) {\n    return x + y * 2;\n}\n```",
        "```sql\nSELECT * FROM users WHERE active = 1;\n```",
    ];
    
    let mut content = String::new();
    for i in 0..1000 {
        content.push_str(&format!("## Code Example {}\n\n", i));
        content.push_str(code_blocks[i % code_blocks.len()]);
        content.push_str("\n\nExplanation of the above code and its usage patterns.\n\n");
    }
    content
}

fn create_table_heavy_document() -> String {
    let mut content = String::new();
    for i in 0..500 {
        content.push_str(&format!("## Table {}\n\n", i));
        content.push_str("| Column A | Column B | Column C | Column D |\n");
        content.push_str("|----------|----------|----------|----------|\n");
        for j in 0..10 {
            content.push_str(&format!("| Data {}-{} | Value {}-{} | Info {}-{} | Result {}-{} |\n", i, j, i, j, i, j, i, j));
        }
        content.push_str("\n\nTable analysis and description follows.\n\n");
    }
    content
}

fn create_reference_heavy_document() -> String {
    let mut content = String::new();
    for i in 0..1000 {
        content.push_str(&format!(
            "Section {} discusses important topics. See section {}.{} for details. \
            Check reference [{0}] and visit https://example{0}.com for more information. \
            Table {0} shows the data, while figure {0} illustrates the concept.\n\n",
            i, i + 1, (i % 10) + 1
        ));
    }
    content
}

fn create_unicode_heavy_document() -> String {
    let mut content = String::new();
    let unicode_samples = vec![
        "Café naïve résumé 🚀 💻 🌟",
        "中文测试内容 with mixed languages",
        "العربية 测试 тест プルーフ",
        "Emoji heavy: 🎉🎊🎈🎁🎂🍰🎪🎨🎭🎪",
        "Special chars: §±¿¡™€£¥₹₩₽",
    ];
    
    for i in 0..2000 {
        content.push_str(&format!("## Section {} - {}", i, unicode_samples[i % unicode_samples.len()]));
        content.push_str("\n\nContent with mixed unicode characters and standard text for comprehensive testing purposes.\n\n");
    }
    content
}

fn create_mixed_structure_document() -> String {
    let mut content = String::new();
    
    for i in 0..500 {
        match i % 6 {
            0 => {
                content.push_str(&format!("# Major Section {}\n\n", i));
                content.push_str("Introduction to this major section with overview and context.\n\n");
            }
            1 => {
                content.push_str("## Subsection\n\n");
                content.push_str("- List item one\n- List item two\n- List item three\n\n");
            }
            2 => {
                content.push_str("```code\nfunction example() { return true; }\n```\n\n");
            }
            3 => {
                content.push_str("| A | B |\n|---|---|\n| 1 | 2 |\n\n");
            }
            4 => {
                content.push_str("> This is a quote block\n> with multiple lines\n> for testing purposes\n\n");
            }
            _ => {
                content.push_str(&format!("Regular paragraph {} with normal text content and proper sentence structure. See reference [{}] for more details.\n\n", i, i));
            }
        }
    }
    content
}

fn create_dense_boundary_content() -> String {
    "Short. Very. Dense. Boundaries. Here. Every. Word. Ends. Sentences.\n\n".repeat(1000)
}

fn create_sparse_boundary_content() -> String {
    "This is a very long sentence that goes on and on without much punctuation or clear boundary markers making it challenging for boundary detection algorithms to find natural breaking points in the text flow ".repeat(500)
}

fn create_complex_structure_content() -> String {
    let mut content = String::new();
    for i in 0..200 {
        content.push_str(&format!(
            "### Complex Section {} \
            \n\nParagraph with mixed content: code `inline_code()`, references [{}], \
            URLs https://test{}.com, and **bold** text.\n\n\
            > Quote within complex structure\n\
            > Second line of quote\n\n\
            | Table | In | Complex |\n\
            |-------|----|---------|\
            | Data  | {} | Content |\n\n",
            i, i, i, i
        ));
    }
    content
}

criterion_group!(
    benches,
    bench_chunking_performance,
    bench_chunking_different_sizes,
    bench_boundary_detection,
    bench_concurrent_chunking,
    bench_memory_usage,
    bench_real_world_scenarios,
    bench_performance_regression,
    bench_high_performance_targets,
    bench_latency_characteristics,
    bench_memory_efficiency,
    bench_scalability,
    bench_specialized_content,
    bench_boundary_detection_advanced
);

criterion_main!(benches);