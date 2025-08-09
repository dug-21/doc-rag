# Phase 1 Test Specifications

## Test Philosophy
Every component must be thoroughly tested before integration. Tests are not an afterthought but a fundamental part of development.

## Test Categories

### 1. Unit Tests
**Coverage Target**: >90% per component

#### MCP Adapter Tests
```rust
// mcp-adapter/src/tests/unit.rs

#[cfg(test)]
mod mcp_adapter_tests {
    use super::*;
    use mockito::{mock, server_url};
    
    #[tokio::test]
    async fn test_connection_with_retry() {
        let _m = mock("POST", "/connect")
            .with_status(500)
            .expect(2)
            .create();
            
        let adapter = McpAdapter::new();
        let result = adapter.connect(&server_url()).await;
        
        assert!(result.is_err());
        assert_eq!(adapter.retry_count(), 3);
    }
    
    #[tokio::test]
    async fn test_authentication_success() {
        let _m = mock("POST", "/auth")
            .with_body(r#"{"token":"valid_token","expires_in":3600}"#)
            .create();
            
        let adapter = McpAdapter::new();
        let token = adapter.authenticate(test_credentials()).await.unwrap();
        
        assert_eq!(token.value, "valid_token");
        assert!(token.is_valid());
    }
    
    #[tokio::test]
    async fn test_concurrent_messages() {
        let adapter = Arc::new(McpAdapter::new());
        let mut handles = vec![];
        
        for i in 0..100 {
            let adapter_clone = adapter.clone();
            handles.push(tokio::spawn(async move {
                adapter_clone.send_message(Message::new(i)).await
            }));
        }
        
        let results: Vec<_> = futures::future::join_all(handles).await;
        assert_eq!(results.len(), 100);
        assert!(results.iter().all(|r| r.is_ok()));
    }
    
    #[bench]
    fn bench_message_serialization(b: &mut Bencher) {
        let msg = create_large_message();
        b.iter(|| {
            black_box(msg.serialize())
        });
    }
}
```

#### Document Chunker Tests
```rust
// chunker/src/tests/unit.rs

#[cfg(test)]
mod chunker_tests {
    use super::*;
    
    #[test]
    fn test_basic_chunking() {
        let chunker = DocumentChunker::new(512, 50);
        let content = "a".repeat(2000);
        let chunks = chunker.chunk_document(&content);
        
        assert_eq!(chunks.len(), 4);
        assert!(chunks.iter().all(|c| c.content.len() <= 512));
    }
    
    #[test]
    fn test_semantic_boundaries() {
        let chunker = DocumentChunker::new(512, 50);
        let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunker.chunk_document(content);
        
        // Should respect paragraph boundaries
        assert!(chunks.iter().all(|c| !c.content.starts_with('\n')));
    }
    
    #[test]
    fn test_metadata_preservation() {
        let chunker = DocumentChunker::new(512, 50);
        let content = "# Title\n\n## Section 1\n\nContent here.";
        let chunks = chunker.chunk_document(content);
        
        assert_eq!(chunks[0].metadata.section, Some("Title".to_string()));
        assert_eq!(chunks[1].metadata.section, Some("Section 1".to_string()));
    }
    
    #[test]
    fn test_cross_reference_tracking() {
        let chunker = DocumentChunker::new(512, 50);
        let content = "See section 2.1 for details. [1] Reference to footnote.";
        let chunks = chunker.chunk_document(content);
        
        assert!(!chunks[0].references.is_empty());
    }
    
    #[bench]
    fn bench_large_document_chunking(b: &mut Bencher) {
        let chunker = DocumentChunker::new(512, 50);
        let content = std::fs::read_to_string("test_data/large_doc.txt").unwrap();
        
        b.iter(|| {
            black_box(chunker.chunk_document(&content))
        });
    }
}
```

### 2. Integration Tests
**Goal**: Verify component interactions

```rust
// tests/integration/pipeline.rs

#[tokio::test]
async fn test_chunk_and_embed_pipeline() {
    // Setup
    let chunker = DocumentChunker::new(512, 50);
    let embedder = EmbeddingGenerator::new("all-MiniLM-L6-v2").await.unwrap();
    
    // Execute
    let content = load_test_document("pci_dss_sample.txt");
    let chunks = chunker.chunk_document(&content);
    let embedded = embedder.generate_embeddings(chunks).await.unwrap();
    
    // Verify
    assert!(!embedded.is_empty());
    assert!(embedded.iter().all(|c| c.embeddings.is_some()));
    assert!(embedded.iter().all(|c| c.embeddings.as_ref().unwrap().len() == 384));
}

#[tokio::test]
async fn test_storage_and_retrieval() {
    // Setup
    let storage = VectorStorage::connect("mongodb://localhost:27017").await.unwrap();
    let embedder = EmbeddingGenerator::new("all-MiniLM-L6-v2").await.unwrap();
    
    // Store
    let chunks = create_test_chunks();
    let embedded = embedder.generate_embeddings(chunks).await.unwrap();
    storage.store_chunks(embedded).await.unwrap();
    
    // Retrieve
    let query = "data encryption requirements";
    let query_embedding = embedder.generate_embedding(query).await.unwrap();
    let results = storage.vector_search(&query_embedding, 5).await.unwrap();
    
    // Verify
    assert_eq!(results.len(), 5);
    assert!(results[0].score > 0.7);
}

#[tokio::test]
async fn test_end_to_end_query() {
    // Setup all components
    let system = setup_test_system().await;
    
    // Execute query
    let query = "What are the requirements for storing payment card data?";
    let response = system.process_query(query).await.unwrap();
    
    // Verify response
    assert!(!response.answer.is_empty());
    assert!(!response.citations.is_empty());
    assert!(response.confidence > 0.8);
    assert!(response.latency_ms < 2000);
}
```

### 3. Performance Tests
**Target**: Meet all latency and throughput requirements

```rust
// tests/performance/benchmarks.rs

#[bench]
fn bench_mcp_message_throughput(b: &mut Bencher) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let adapter = rt.block_on(McpAdapter::connect("test")).unwrap();
    
    b.iter(|| {
        rt.block_on(async {
            for _ in 0..100 {
                black_box(adapter.send_message(test_message()).await);
            }
        })
    });
}

#[bench]
fn bench_vector_search_latency(b: &mut Bencher) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let storage = rt.block_on(VectorStorage::connect("test")).unwrap();
    let query_embedding = random_embedding();
    
    b.iter(|| {
        rt.block_on(async {
            black_box(storage.vector_search(&query_embedding, 10).await)
        })
    });
}

#[test]
fn test_concurrent_query_handling() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let system = rt.block_on(setup_test_system());
    
    let start = Instant::now();
    let handles: Vec<_> = (0..100).map(|i| {
        let system_clone = system.clone();
        rt.spawn(async move {
            system_clone.process_query(&format!("Query {}", i)).await
        })
    }).collect();
    
    rt.block_on(async {
        futures::future::join_all(handles).await
    });
    
    let duration = start.elapsed();
    assert!(duration.as_secs() < 10); // 100 queries in < 10 seconds
}
```

### 4. Property-Based Tests
**Using**: proptest for comprehensive edge case coverage

```rust
// tests/property/chunker_properties.rs

use proptest::prelude::*;

proptest! {
    #[test]
    fn chunks_never_exceed_max_size(
        content in "\\PC{100,10000}",
        chunk_size in 100..1000usize,
        overlap in 0..50usize
    ) {
        let chunker = DocumentChunker::new(chunk_size, overlap);
        let chunks = chunker.chunk_document(&content);
        
        prop_assert!(chunks.iter().all(|c| c.content.len() <= chunk_size));
    }
    
    #[test]
    fn overlaps_are_consistent(
        content in "\\PC{500,5000}",
        overlap in 10..100usize
    ) {
        let chunker = DocumentChunker::new(512, overlap);
        let chunks = chunker.chunk_document(&content);
        
        for i in 1..chunks.len() {
            let prev_end = &chunks[i-1].content[chunks[i-1].content.len()-overlap..];
            let curr_start = &chunks[i].content[..overlap];
            prop_assert_eq!(prev_end, curr_start);
        }
    }
    
    #[test]
    fn embeddings_are_normalized(
        text in "\\PC{10,1000}"
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let embedder = rt.block_on(EmbeddingGenerator::new("test")).unwrap();
        let embedding = rt.block_on(embedder.generate_embedding(&text)).unwrap();
        
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assert!((magnitude - 1.0).abs() < 0.01);
    }
}
```

### 5. Compliance Tests
**Ensure**: System meets all requirements

```rust
// tests/compliance/requirements.rs

#[tokio::test]
async fn test_citation_coverage() {
    let system = setup_test_system().await;
    let queries = load_compliance_test_queries();
    
    for query in queries {
        let response = system.process_query(&query).await.unwrap();
        
        // Every claim must have a citation
        for claim in response.claims {
            assert!(!claim.citations.is_empty(), 
                   "Claim '{}' has no citations", claim.text);
        }
    }
}

#[tokio::test]
async fn test_response_accuracy() {
    let system = setup_test_system().await;
    let test_cases = load_accuracy_test_cases();
    
    let mut correct = 0;
    for (query, expected) in test_cases {
        let response = system.process_query(&query).await.unwrap();
        if response.answer.contains(&expected) {
            correct += 1;
        }
    }
    
    let accuracy = correct as f64 / test_cases.len() as f64;
    assert!(accuracy >= 0.99, "Accuracy {} is below 99%", accuracy);
}

#[tokio::test]
async fn test_latency_requirements() {
    let system = setup_test_system().await;
    let queries = load_performance_test_queries();
    
    let mut latencies = Vec::new();
    for query in queries {
        let start = Instant::now();
        system.process_query(&query).await.unwrap();
        latencies.push(start.elapsed().as_millis());
    }
    
    latencies.sort();
    let p99 = latencies[latencies.len() * 99 / 100];
    assert!(p99 < 2000, "P99 latency {}ms exceeds 2000ms", p99);
}
```

## Test Data Management

### Test Document Structure
```
test_data/
├── documents/
│   ├── pci_dss_sample.txt
│   ├── small_doc.txt
│   ├── large_doc.txt
│   └── edge_cases/
│       ├── tables.txt
│       ├── code_blocks.txt
│       └── references.txt
├── queries/
│   ├── simple_queries.json
│   ├── complex_queries.json
│   └── compliance_queries.json
└── expected/
    ├── responses.json
    └── citations.json
```

### Test Data Generation
```rust
// tests/helpers/data_generation.rs

pub fn generate_test_document(size: usize) -> String {
    let mut rng = rand::thread_rng();
    let paragraphs: Vec<String> = (0..size/100)
        .map(|_| generate_paragraph(&mut rng))
        .collect();
    
    paragraphs.join("\n\n")
}

pub fn generate_test_chunks(count: usize) -> Vec<Chunk> {
    (0..count).map(|i| Chunk {
        id: Uuid::new_v4(),
        content: format!("Test chunk {}", i),
        metadata: ChunkMetadata::default(),
        embeddings: None,
        references: vec![],
    }).collect()
}

pub fn generate_random_embedding(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let vec: Vec<f32> = (0..dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    // Normalize
    let magnitude = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.iter().map(|x| x / magnitude).collect()
}
```

## Test Execution Strategy

### Continuous Testing
```yaml
# .github/workflows/test.yml
name: Continuous Testing

on:
  push:
  pull_request:

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Unit Tests
        run: cargo test --lib --all-features
      - name: Generate Coverage
        run: cargo tarpaulin --out Xml
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        
  integration-tests:
    runs-on: ubuntu-latest
    services:
      mongodb:
        image: mongo:7.0
        ports:
          - 27017:27017
    steps:
      - uses: actions/checkout@v3
      - name: Run Integration Tests
        run: cargo test --test integration
        
  performance-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Benchmarks
        run: cargo bench --all
      - name: Check Performance Regression
        run: ./scripts/check_performance.sh
```

### Test Coverage Requirements

| Component | Unit | Integration | Performance | Total |
|-----------|------|-------------|-------------|-------|
| MCP Adapter | 95% | 90% | 100% | 93% |
| Chunker | 95% | 90% | 100% | 93% |
| Embedder | 90% | 85% | 100% | 90% |
| Storage | 90% | 95% | 100% | 92% |
| Query Processor | 95% | 90% | 100% | 93% |
| Response Generator | 95% | 90% | 100% | 93% |
| **Overall** | **93%** | **90%** | **100%** | **92%** |

## Test Documentation

### Test Case Template
```markdown
## Test ID: TC-001
**Component**: MCP Adapter
**Type**: Unit Test
**Priority**: High

### Description
Test connection retry logic when server is unavailable

### Preconditions
- MCP server mock configured
- Retry count set to 3

### Steps
1. Attempt connection to unavailable server
2. Verify retry attempts
3. Check final error state

### Expected Results
- Exactly 3 retry attempts made
- Appropriate error returned
- Retry delays follow exponential backoff

### Actual Results
[To be filled during execution]

### Status
[Pass/Fail]
```

## Testing Best Practices

1. **Test First**: Write tests before implementation
2. **Test Independence**: Each test should be independent
3. **Clear Naming**: Test names should describe what they test
4. **Fast Execution**: Unit tests should run in < 1 second
5. **Deterministic**: No flaky tests allowed
6. **Documentation**: Complex tests need comments
7. **Coverage**: Aim for >90% but focus on critical paths
8. **Performance**: Include benchmarks for critical operations
9. **Data Management**: Use fixtures for consistent test data
10. **Continuous Integration**: All tests run on every commit