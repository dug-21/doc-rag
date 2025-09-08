# FACT Integration Test Suite

Comprehensive test suite for FACT (Fast Adaptive Content Transport) integration following London TDD methodology.

## Overview

This test suite provides comprehensive coverage for FACT integration including:
- **Integration Tests**: End-to-end testing with mock FACT clients
- **Performance Tests**: Benchmarks and load testing to validate performance requirements
- **Mock Implementation**: Realistic FACT client simulation for testing

## Test Structure

```
src/tests/
├── fact_integration_tests.rs    # Main integration test suite
├── fact_performance_tests.rs    # Performance benchmarks and load tests
├── mocks/
│   ├── mock_fact_client.rs     # Mock FACT client implementation
│   └── mod.rs                  # Mock module exports
├── mod.rs                      # Test module organization
├── Cargo.toml                  # Test-specific dependencies
└── README.md                   # This file
```

## Performance Requirements

The test suite validates against these specific requirements:

| Metric | Target | Test Coverage |
|--------|--------|---------------|
| Cache Hit Latency | <23ms | ✅ Benchmark + Load Tests |
| Cache Miss Latency | <95ms | ✅ Integration Tests |
| Hit Rate | >87.3% | ✅ Load Tests with Hit Rate Validation |
| Concurrent Users | 100+ | ✅ Stress Tests |
| Byzantine Consensus | 66% threshold | ✅ Consensus Integration Tests |

## Running Tests

### Unit Tests
```bash
cargo test --lib fact_integration_tests
```

### Performance Benchmarks
```bash
cargo bench --bench fact_benchmarks
```

### Load Tests (100+ concurrent users)
```bash
cargo test test_100_concurrent_users_load --release
```

### All Integration Tests
```bash
cargo test --test '*' --features integration
```

### Stress Tests
```bash
cargo test test_stress_conditions --release --features stress
```

## Test Categories

### 1. Cache Performance Tests
- **Cache Hit Performance**: Validates <23ms response time
- **Cache Miss Handling**: Validates <95ms response time
- **Hit Rate Validation**: Ensures >87.3% hit rate under load

### 2. MCP Protocol Integration
- **Tool Registration**: MCP tool registration with FACT backend
- **Tool Execution**: MCP protocol message handling
- **Error Handling**: Graceful degradation scenarios

### 3. Byzantine Consensus Tests
- **Consensus Achievement**: 66% threshold validation
- **Multi-node Validation**: Distributed consensus simulation
- **Fault Tolerance**: Byzantine fault tolerance testing

### 4. Citation Tracking
- **Citation Extraction**: Metadata citation tracking
- **Citation Validation**: Source verification
- **Citation Usage**: Usage statistics and analytics

### 5. Load and Stress Testing
- **Concurrent Users**: 100+ simultaneous users
- **Throughput**: 1000+ requests per second capacity
- **Error Rate**: <1% error rate under normal load
- **Graceful Degradation**: <5% error rate under stress

## Mock Implementation Features

### RealisticFACTSimulator
- **Configurable Latency**: Separate cache hit/miss latencies
- **Error Simulation**: Configurable error rates and types
- **Cache Management**: TTL-based cache with size limits
- **Metrics Collection**: Comprehensive performance metrics
- **Health Monitoring**: Health check simulation

### MockFACTClientFactory
Provides pre-configured clients for different test scenarios:
- `create_cache_hit_optimized()`: Always returns cached data quickly
- `create_cache_miss_simulation()`: Simulates cache miss scenarios
- `create_error_prone()`: Tests error handling and recovery

## Test Data and Fixtures

### FACTTestFixtures
Provides realistic compliance-related test data:
- PCI DSS requirements and controls
- GDPR articles and consent mechanisms
- ISO 27001 security controls
- SOX documentation requirements

### Citation Generation
Automatically generates realistic citations with:
- Source document references
- Section/article numbers
- Page numbers where applicable
- Confidence scores

## Validation Criteria

### Performance Gates
```rust
pub struct QualityGates {
    pub min_test_coverage: f64,     // 90%
    pub max_cache_latency_ms: u64,  // 23ms
    pub min_cache_hit_rate: f64,    // 0.873
    pub max_error_rate: f64,        // 0.01
}
```

### Functional Validation
- All MCP tools must register successfully
- All citations must have valid source references  
- Byzantine consensus must achieve 66% agreement
- System must fallback gracefully when FACT is unavailable

## Test Configuration

### Standard Test Config
```rust
FACTTestConfig {
    cache_ttl: Duration::from_secs(1800),
    retry_attempts: 3,
    timeout: Duration::from_secs(5),
    consensus_threshold: 0.66,
}
```

### Load Test Config
```rust
LoadTestConfig {
    concurrent_users: 100,
    duration: Duration::from_secs(30),
    target_rps: 1000.0,
    cache_warm_queries: vec![/* pre-warm queries */],
}
```

## Debugging and Troubleshooting

### Common Test Issues

1. **Cache Hit Latency Exceeds 23ms**
   - Check network conditions in test environment
   - Verify mock client is configured for cache hits
   - Consider test system performance impact

2. **Hit Rate Below 87.3%**
   - Verify cache warm-up is working correctly
   - Check query distribution (too many unique queries)
   - Validate cache TTL settings

3. **Consensus Not Achieved**
   - Ensure at least 3 nodes are configured
   - Check Byzantine threshold calculation
   - Verify node response simulation

4. **Load Test Failures**
   - Increase timeout values for CI/CD environments
   - Check system resources during test execution
   - Validate concurrent user limits

### Test Metrics and Monitoring

All tests collect detailed metrics:
- Request latency histograms (p50, p95, p99)
- Cache hit/miss ratios
- Error rates by type
- Memory usage patterns
- Consensus participation rates

## Integration with CI/CD

The test suite is designed for automated execution:
- Fast unit tests run on every commit
- Integration tests run on pull requests
- Performance benchmarks run nightly
- Stress tests run weekly

## Dependencies

Key testing dependencies:
- `mockall` - Mock generation and expectations
- `criterion` - Performance benchmarking
- `tokio-test` - Async test utilities
- `proptest` - Property-based testing
- `fake` - Test data generation

## Contributing

When adding new tests:
1. Follow London TDD methodology (mock external dependencies)
2. Include both positive and negative test cases
3. Add performance assertions for timing-critical code
4. Update this README with new test categories
5. Ensure tests are deterministic and don't rely on external services