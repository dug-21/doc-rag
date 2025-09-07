# 🧪 Doc-RAG Testing Infrastructure

## Overview

This document describes the optimal testing infrastructure designed for the Doc-RAG system, providing both fast development feedback loops and comprehensive CI/CD validation.

## 🏗️ Architecture

### Test Environments

1. **Local Development** (`docker-compose.test.yml`)
   - Hot-reload capabilities
   - Interactive testing
   - In-memory databases for speed
   - Mock services for isolated testing

2. **CI/CD Pipeline** (`docker-compose.ci.yml`)
   - Streamlined for automated testing
   - Parallel test execution
   - Resource-optimized containers
   - Comprehensive test coverage

## 🚀 Quick Start

```bash
# Start test environment
./scripts/test-runner.sh start

# Run all tests
./scripts/test-runner.sh ci

# Interactive development
./scripts/test-runner.sh shell

# Watch mode for continuous testing
./scripts/test-runner.sh watch integration
```

## 📁 File Structure

```
├── docker-compose.test.yml     # Local development environment
├── docker-compose.ci.yml       # CI/CD environment
├── Dockerfile.test             # Multi-stage dev containers
├── Dockerfile.ci               # Optimized CI containers
├── scripts/
│   ├── test-runner.sh          # Interactive test runner
│   ├── health-check.sh         # Service health monitoring
│   ├── seed_test_data.py       # Test data seeding
│   └── generate_test_report.py # Test report generation
├── tests/
│   └── mocks/                  # Mock services
├── config/
│   └── test/                   # Test configurations
└── README-Testing.md           # This file
```

## 🔧 Test Components

### 1. Database Services

**MongoDB Test Instance**
- Port: 27018 (non-conflicting)
- In-memory storage for speed
- Auto-seeded with realistic test data
- Health checks with 10s timeout

**Redis Test Cache**
- Port: 6380 (non-conflicting)
- 256MB memory limit
- No persistence (speed optimized)
- Pre-populated with cache entries

**Qdrant Vector Database**
- Port: 6334 (non-conflicting)
- In-memory vector storage
- Mock embeddings for testing
- RESTful health endpoints

### 2. Application Services

**API Service (`api-test`)**
- Hot-reload with `cargo watch`
- Debug logging enabled
- Mock integrations available
- Metrics endpoint on port 9091

**Storage Service (`storage-test`)**
- Watches source changes
- Test database connections
- Mock vector operations
- Performance profiling enabled

**Embedder Service (`embedder-test`)**
- Lightweight model for testing
- Batch processing simulation
- Configurable response times
- Memory usage monitoring

### 3. Test Utilities

**Test Runner Container**
- Interactive Bash shell
- All Rust toolchain included
- Test data access
- Hot-reload capabilities

**Mock Services**
- Node.js based mocks
- Realistic response simulation
- Error scenario testing
- Performance timing control

**Test Seeder**
- Automated data population
- Realistic document corpus
- Query/response pairs
- Performance benchmarks

## 🎯 Testing Strategies

### Unit Testing
```bash
# Run all unit tests
./scripts/test-runner.sh unit

# Specific test pattern
./scripts/test-runner.sh test api_tests

# Watch mode
./scripts/test-runner.sh watch "api"
```

### Integration Testing
```bash
# Full integration suite
./scripts/test-runner.sh integration

# Service-specific integration
docker-compose -f docker-compose.test.yml exec test-runner \
  cargo test --test integration_tests
```

### Performance Testing
```bash
# Run benchmarks
./scripts/test-runner.sh ci
# Performance results in test-results/perf-results.json

# Load testing with k6
docker-compose -f docker-compose.ci.yml run --rm load-tests
```

### Security Testing
```bash
# Security audit
docker-compose -f docker-compose.ci.yml run --rm security-tests

# Results in test-results/security-audit.json
```

## 🔍 Monitoring & Debugging

### Health Checks
```bash
# Check all services
./scripts/health-check.sh

# Specific service
./scripts/health-check.sh api

# JSON output for automation
./scripts/health-check.sh --json --verbose
```

### Log Aggregation
```bash
# View all logs
./scripts/test-runner.sh logs

# Service-specific logs
./scripts/test-runner.sh logs api-test

# Live monitoring
docker-compose -f docker-compose.test.yml logs -f
```

### Performance Profiling
- Built-in metrics endpoints
- Memory usage tracking
- Response time monitoring
- Bottleneck identification

## 📊 Test Reporting

### Automated Reports
- HTML test reports with visual dashboards
- JSON output for CI/CD integration
- Performance trend analysis
- Security vulnerability summaries

### Coverage Analysis
- Line coverage per component
- Integration test coverage
- Performance regression detection
- Dependency vulnerability scanning

## 🚀 CI/CD Integration

### GitHub Actions Integration
```yaml
- name: Run Test Suite
  run: |
    ./scripts/test-runner.sh ci
    
- name: Upload Test Reports
  uses: actions/upload-artifact@v3
  with:
    name: test-reports
    path: test-results/
```

### Parallel Execution
- Unit tests: 4 parallel threads
- Integration tests: Service-based parallelism
- Performance tests: Isolated execution
- Security scans: Background processing

## 💡 Development Workflow

### 1. Start Development Environment
```bash
./scripts/test-runner.sh start
```

### 2. Seed Test Data
```bash
# Automatic seeding on startup
docker-compose -f docker-compose.test.yml run --rm test-seeder
```

### 3. Interactive Development
```bash
# Enter interactive shell
./scripts/test-runner.sh shell

# Inside container:
cargo test --workspace
cargo watch -x "test integration"
```

### 4. Health Monitoring
```bash
# Continuous health checks
./scripts/health-check.sh --verbose
```

## 🔧 Configuration

### Environment Variables
```bash
# Test database URLs
TEST_DATABASE_URL=mongodb://test-mongodb:27017/doc_rag_test
TEST_REDIS_URL=redis://test-redis:6379
TEST_QDRANT_URL=http://test-qdrant:6333

# Test modes
TEST_MODE=true
RUST_LOG=debug
RUST_BACKTRACE=1
```

### Resource Limits
- MongoDB: 500MB memory, no persistence
- Redis: 256MB memory limit
- API Service: 1GB memory, 2 CPU cores
- Test containers: Shared cargo cache

## 🚦 Performance Targets

### Response Times
- Health checks: < 100ms
- Unit tests: < 5 minutes total
- Integration tests: < 10 minutes total
- Full CI suite: < 20 minutes total

### Resource Usage
- Memory: < 4GB total for full test suite
- CPU: Efficient parallel execution
- Network: Minimal external dependencies
- Storage: Ephemeral, no persistent volumes

## 🔒 Security Features

### Container Security
- Non-root user execution
- Read-only filesystems
- Minimal base images
- Secrets management

### Network Isolation
- Separate test networks
- No external internet access during tests
- Service-to-service communication only
- Port isolation from production

## 📈 Scalability

### Horizontal Scaling
- Service-based parallelization
- Independent test execution
- Distributed load testing
- Multi-environment support

### Vertical Scaling
- Configurable resource limits
- Memory-optimized containers
- CPU-efficient test execution
- Storage optimization

## 🛠️ Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker resources
docker system df
docker system prune

# Restart with clean state
./scripts/test-runner.sh clean
./scripts/test-runner.sh start
```

**Test timeouts:**
```bash
# Increase timeout
./scripts/health-check.sh --timeout 30

# Check service logs
./scripts/test-runner.sh logs
```

**Memory issues:**
```bash
# Monitor resource usage
docker stats

# Clean up containers
./scripts/test-runner.sh clean
```

### Debug Mode
```bash
# Enable verbose logging
RUST_LOG=debug ./scripts/test-runner.sh start

# Interactive debugging
./scripts/test-runner.sh shell
# Inside: cargo test -- --nocapture
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Rust Testing Guide](https://doc.rust-lang.org/book/ch11-00-testing.html)
- [k6 Load Testing](https://k6.io/docs/)
- [Fluentd Logging](https://docs.fluentd.org/)

---

**Test Environment Status**: ✅ Production Ready
**Last Updated**: $(date +'%Y-%m-%d')
**Maintainer**: Doc-RAG Team