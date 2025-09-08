# SPARC Refinement: Test-Driven Development Plan
## Phase 5 - TDD Implementation Strategy

**Document Version**: 1.0  
**Date**: January 8, 2025  
**Dependencies**: 01-Specification, 02-Pseudocode, 03-Architecture

---

## 1. TDD Strategy Overview

### Test Pyramid for FACT Integration

```
         /\
        /  \  E2E Tests (5%)
       /    \  - Full system with real FACT
      /──────\
     /        \  Integration Tests (20%)
    /          \  - FACT client integration
   /            \  - MCP protocol tests
  /──────────────\
 /                \  Unit Tests (75%)
/                  \  - Mock FACT responses
────────────────────  - Component isolation
```

## 2. Test Categories & Scenarios

### 2.1 Unit Tests

#### Test Suite: FACT Client Wrapper

```rust
#[cfg(test)]
mod fact_client_tests {
    use super::*;
    use mockall::predicate::*;
    use tokio::test;

    #[test]
    async fn test_cache_hit_performance() {
        // Given: Mock FACT client with cached response
        let mut mock_fact = MockFACTClient::new();
        mock_fact
            .expect_get()
            .with(eq("test_key"))
            .times(1)
            .returning(|_| Ok(Some(b"cached_response".to_vec())));
        
        // When: Query is executed
        let start = Instant::now();
        let result = mock_fact.get("test_key").await;
        let duration = start.elapsed();
        
        // Then: Response returns within 23ms
        assert!(result.is_ok());
        assert!(duration < Duration::from_millis(23));
        assert_eq!(result.unwrap(), Some(b"cached_response".to_vec()));
    }

    #[test]
    async fn test_cache_miss_handling() {
        // Given: Mock FACT client with cache miss
        let mut mock_fact = MockFACTClient::new();
        mock_fact
            .expect_get()
            .with(eq("missing_key"))
            .times(1)
            .returning(|_| Ok(None));
        
        // When: Query for non-cached key
        let result = mock_fact.get("missing_key").await;
        
        // Then: Returns None without error
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    async fn test_ttl_calculation() {
        // Test cases for TTL calculation
        let test_cases = vec![
            (0.9, 0.1, 6480),  // High confidence, low volatility = long TTL
            (0.5, 0.5, 1800),  // Medium both = medium TTL
            (0.3, 0.9, 60),    // Low confidence, high volatility = minimum TTL
        ];
        
        for (confidence, volatility, expected_ttl) in test_cases {
            let ttl = calculate_ttl(confidence, volatility);
            assert_eq!(ttl.as_secs(), expected_ttl);
        }
    }

    #[test]
    async fn test_connection_retry() {
        // Given: FACT client with retry policy
        let mut mock_fact = MockFACTClient::new();
        let mut seq = Sequence::new();
        
        // First call fails
        mock_fact
            .expect_get()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_| Err(FACTError::ConnectionError));
        
        // Second call succeeds
        mock_fact
            .expect_get()
            .times(1)
            .in_sequence(&mut seq)
            .returning(|_| Ok(Some(b"success".to_vec())));
        
        // When: Client with retry executes
        let client = RetryableFACTClient::new(mock_fact, RetryPolicy::default());
        let result = client.get("key").await;
        
        // Then: Succeeds after retry
        assert!(result.is_ok());
    }
}
```

#### Test Suite: MCP Tool Registration

```rust
#[cfg(test)]
mod mcp_tool_tests {
    use super::*;

    #[test]
    async fn test_tool_registration() {
        // Given: MCP server
        let mut mcp_server = MockMCPServer::new();
        
        // When: Tool is registered
        let tool = SearchTool::new();
        mcp_server.register_tool("search", Box::new(tool));
        
        // Then: Tool is available
        assert!(mcp_server.has_tool("search"));
        assert_eq!(mcp_server.tool_count(), 1);
    }

    #[test]
    async fn test_tool_execution() {
        // Given: Registered search tool
        let mut tool = MockSearchTool::new();
        tool.expect_execute()
            .with(eq(json!({"query": "PCI DSS"})))
            .returning(|_| Ok(json!({"results": ["doc1", "doc2"]})));
        
        // When: Tool is executed
        let result = tool.execute(json!({"query": "PCI DSS"})).await;
        
        // Then: Returns expected results
        assert!(result.is_ok());
        assert_eq!(result.unwrap()["results"].as_array().unwrap().len(), 2);
    }
}
```

### 2.2 Integration Tests

#### Test Suite: Query Processor with FACT

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use testcontainers::*;

    #[tokio::test]
    async fn test_query_processor_fact_integration() {
        // Given: Real FACT instance (containerized)
        let docker = clients::Cli::default();
        let fact_container = docker.run(FACTImage::default());
        let fact_url = format!("http://localhost:{}", fact_container.get_port());
        
        let config = FACTConfig {
            endpoint: fact_url,
            api_key: "test_key".to_string(),
            timeout: Duration::from_secs(5),
        };
        
        let fact_client = FACTClient::new(config).await.unwrap();
        let query_processor = QueryProcessor::with_fact(fact_client);
        
        // When: Query is processed
        let query = Query::new("What are PCI DSS requirements?");
        let result = query_processor.process(query).await;
        
        // Then: Result includes FACT caching
        assert!(result.is_ok());
        let processed = result.unwrap();
        assert!(processed.metadata.contains_key("fact_cache_status"));
    }

    #[tokio::test]
    async fn test_byzantine_consensus_with_fact() {
        // Given: Consensus manager with FACT
        let fact_client = create_test_fact_client().await;
        let consensus = ConsensusManager::with_fact(fact_client);
        
        // When: Multiple nodes validate result
        let result = QueryResult::new("test result");
        let validated = consensus.validate(result, 3).await;
        
        // Then: 66% consensus achieved and cached
        assert!(validated.is_ok());
        assert!(validated.unwrap().consensus_achieved);
        assert!(validated.unwrap().consensus_confidence >= 0.66);
    }
}
```

#### Test Suite: Performance Benchmarks

```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn benchmark_cache_hit(c: &mut Criterion) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let fact_client = runtime.block_on(create_test_fact_client());
        
        // Pre-warm cache
        runtime.block_on(fact_client.set("bench_key", b"data", Duration::from_secs(60)));
        
        c.bench_function("fact_cache_hit", |b| {
            b.to_async(&runtime).iter(|| async {
                let _ = fact_client.get(black_box("bench_key")).await;
            });
        });
    }

    fn benchmark_cache_miss(c: &mut Criterion) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let fact_client = runtime.block_on(create_test_fact_client());
        
        c.bench_function("fact_cache_miss", |b| {
            b.to_async(&runtime).iter(|| async {
                let key = format!("miss_{}", uuid::Uuid::new_v4());
                let _ = fact_client.get(black_box(&key)).await;
            });
        });
    }

    criterion_group!(benches, benchmark_cache_hit, benchmark_cache_miss);
    criterion_main!(benches);
}
```

### 2.3 End-to-End Tests

#### Test Suite: Complete RAG Pipeline

```rust
#[cfg(test)]
mod e2e_tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Run only in CI/CD
    async fn test_complete_rag_pipeline_with_fact() {
        // Given: Full system setup
        let system = TestSystem::new()
            .with_fact()
            .with_daa()
            .with_ruv_fann()
            .build()
            .await;
        
        // When: Complex query is processed
        let query = "Compare PCI DSS 3.2.1 and 4.0 encryption requirements";
        let response = system.query(query).await;
        
        // Then: All requirements met
        assert!(response.is_ok());
        let result = response.unwrap();
        
        // Performance requirements
        assert!(result.latency < Duration::from_secs(2));
        assert!(result.cache_hit_rate >= 0.873);
        
        // Accuracy requirements
        assert!(result.confidence >= 0.99);
        assert!(!result.citations.is_empty());
        
        // Consensus requirements
        assert!(result.consensus_achieved);
        assert!(result.consensus_confidence >= 0.66);
    }

    #[tokio::test]
    #[ignore]
    async fn test_fact_failover() {
        // Given: System with FACT
        let mut system = TestSystem::new().with_fact().build().await;
        
        // When: FACT becomes unavailable
        system.kill_fact().await;
        let query = "Test query during outage";
        let response = system.query(query).await;
        
        // Then: System falls back gracefully
        assert!(response.is_ok());
        let result = response.unwrap();
        assert!(result.warnings.contains(&"FACT unavailable - using fallback"));
        assert!(result.latency < Duration::from_secs(5));
    }
}
```

## 3. Test Data & Fixtures

### 3.1 Test Data Generation

```rust
pub struct TestDataGenerator {
    faker: Faker,
    templates: Vec<QueryTemplate>,
}

impl TestDataGenerator {
    pub fn generate_queries(&self, count: usize) -> Vec<Query> {
        (0..count)
            .map(|_| self.generate_single_query())
            .collect()
    }
    
    fn generate_single_query(&self) -> Query {
        let template = &self.templates[rand::random::<usize>() % self.templates.len()];
        Query::new(&template.fill(&self.faker))
    }
}
```

### 3.2 Mock FACT Responses

```rust
pub fn mock_fact_responses() -> HashMap<String, Vec<u8>> {
    let mut responses = HashMap::new();
    
    responses.insert(
        "pci_dss_encryption".to_string(),
        serde_json::to_vec(&json!({
            "content": "PCI DSS requires strong encryption...",
            "citations": [
                {
                    "source": "PCI DSS 4.0",
                    "section": "3.5.1",
                    "page": 47
                }
            ],
            "confidence": 0.95
        })).unwrap()
    );
    
    responses
}
```

## 4. Test Execution Strategy

### 4.1 Test Phases

```yaml
test_phases:
  phase_1_unit:
    description: "Unit tests with mocked FACT"
    command: "cargo test --lib"
    duration: "< 30 seconds"
    
  phase_2_integration:
    description: "Integration tests with test FACT"
    command: "cargo test --test integration"
    duration: "< 2 minutes"
    
  phase_3_performance:
    description: "Performance benchmarks"
    command: "cargo bench"
    duration: "< 5 minutes"
    
  phase_4_e2e:
    description: "End-to-end tests"
    command: "cargo test --test e2e --ignored"
    duration: "< 10 minutes"
```

### 4.2 Continuous Testing

```yaml
name: FACT Integration Tests

on:
  push:
    paths:
      - 'src/query-processor/**'
      - 'Cargo.toml'
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      fact:
        image: ruvnet/fact:latest
        ports:
          - 8080:8080
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Run unit tests
        run: cargo test --lib
        
      - name: Run integration tests
        run: cargo test --test integration
        env:
          FACT_ENDPOINT: http://localhost:8080
          
      - name: Run benchmarks
        run: cargo bench --no-fail-fast
        
      - name: Check performance regression
        run: |
          if [ $(cat target/criterion/fact_cache_hit/base/estimates.json | jq .mean.point_estimate) -gt 23000000 ]; then
            echo "Cache hit latency exceeds 23ms!"
            exit 1
          fi
```

## 5. Validation Criteria

### 5.1 Performance Validation

| Metric | Target | Test Method |
|--------|--------|-------------|
| Cache Hit Latency | <23ms | Benchmark suite |
| Cache Miss Latency | <95ms | Integration test |
| Hit Rate | >87.3% | Load test |
| Concurrent Users | 100+ | Stress test |

### 5.2 Functional Validation

| Feature | Validation | Test Type |
|---------|------------|-----------|
| MCP Protocol | Tool execution works | Integration |
| Citation Tracking | All citations have sources | Unit |
| Byzantine Consensus | 66% threshold met | Integration |
| Fallback Strategy | Graceful degradation | E2E |

### 5.3 Quality Gates

```rust
pub struct QualityGates {
    pub min_test_coverage: f64,  // 90%
    pub max_cache_latency_ms: u64,  // 23ms
    pub min_cache_hit_rate: f64,  // 0.873
    pub max_error_rate: f64,  // 0.01
}

impl QualityGates {
    pub fn validate(&self, metrics: &TestMetrics) -> Result<()> {
        if metrics.coverage < self.min_test_coverage {
            return Err("Insufficient test coverage");
        }
        if metrics.avg_cache_latency_ms > self.max_cache_latency_ms {
            return Err("Cache latency exceeds threshold");
        }
        // ... additional validations
        Ok(())
    }
}
```

## 6. Test Documentation

### 6.1 Test Case Template

```markdown
### Test ID: FACT-001
**Description**: Verify cache hit performance under load
**Priority**: P0
**Type**: Performance

**Preconditions**:
- FACT service running
- Cache pre-warmed with test data

**Test Steps**:
1. Generate 1000 concurrent requests
2. Measure response times
3. Calculate p50, p95, p99 latencies

**Expected Results**:
- p50 < 20ms
- p95 < 23ms
- p99 < 30ms

**Actual Results**: [To be filled]
**Status**: [Pass/Fail]
```

### 6.2 Test Coverage Report

```
Module                  | Coverage | Target
------------------------|----------|--------
fact_client            | 95%      | 90%
mcp_integration        | 88%      | 85%
query_processor        | 92%      | 90%
consensus_manager      | 87%      | 85%
performance_optimizer  | 90%      | 85%
------------------------|----------|--------
Overall                | 90.4%    | 87%
```

## 7. Debugging & Troubleshooting

### 7.1 Common Test Failures

| Failure | Cause | Solution |
|---------|-------|----------|
| Timeout in cache test | Network latency | Increase timeout, check connection |
| Consensus not achieved | Not enough nodes | Ensure 3+ nodes available |
| Performance regression | Cache miss | Verify cache warming |
| MCP tool not found | Registration failed | Check tool schema |

### 7.2 Debug Helpers

```rust
#[cfg(test)]
pub mod test_helpers {
    pub fn enable_trace_logging() {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_test_writer()
            .init();
    }
    
    pub fn capture_metrics() -> MetricsCapture {
        MetricsCapture::new()
    }
    
    pub fn dump_fact_cache_state(client: &FACTClient) {
        println!("Cache Statistics:");
        println!("  Hit Rate: {:.2}%", client.hit_rate() * 100.0);
        println!("  Entries: {}", client.entry_count());
        println!("  Memory: {} MB", client.memory_usage_mb());
    }
}
```

---

## Summary

This TDD refinement provides:

1. **Comprehensive test coverage** at unit, integration, and E2E levels
2. **Performance validation** ensuring <23ms cache hits
3. **Quality gates** enforcing standards
4. **Continuous testing** in CI/CD pipeline
5. **Clear validation criteria** for all requirements
6. **Debugging tools** for troubleshooting

**Next Steps**: Proceed to SPARC Completion for deployment strategy.