# Doc-RAG Testing Workflow Guide

## Overview

This guide describes the refined testing workflow for the Doc-RAG system, including script usage, CI/CD integration, and development practices.

## Quick Reference

### Primary Commands

```bash
# 🚀 Quick development validation (< 2 minutes)
./scripts/test_quick_enhanced.sh

# 🧪 Comprehensive testing (< 15 minutes)
./scripts/test_all_enhanced.sh

# 🏗️ Development environment setup
./scripts/dev_start.sh

# 📊 Performance monitoring
./scripts/performance_monitor.sh
```

### Script Purposes

| Script | Purpose | Time | When to Use |
|--------|---------|------|-------------|
| `test_quick_enhanced.sh` | Fast API validation | < 2 min | Every commit, rapid feedback |
| `test_all_enhanced.sh` | Comprehensive testing | 5-15 min | Before PR, release validation |
| `dev_start.sh` | Environment setup | 1-2 min | Initial setup, after changes |
| `performance_monitor.sh` | Performance tracking | 2-5 min | Performance validation |

## Development Workflow

### Daily Development Cycle

```bash
# 1. Start development environment
./scripts/dev_start.sh

# 2. Make code changes
# ... coding ...

# 3. Quick validation after changes
./scripts/test_quick_enhanced.sh

# 4. Full testing before commit
./scripts/test_all_enhanced.sh --unit-only --integration-only

# 5. Final validation before push
./scripts/test_all_enhanced.sh
```

### Feature Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Initial validation
./scripts/test_quick_enhanced.sh

# 3. Develop feature with TDD
# Write test -> Run test -> Implement -> Refactor
./scripts/test_all_enhanced.sh --unit-only

# 4. Integration testing
./scripts/test_all_enhanced.sh --integration-only

# 5. Performance validation
./scripts/test_all_enhanced.sh --performance-only

# 6. Full validation before PR
./scripts/test_all_enhanced.sh --comprehensive
```

## Testing Categories Explained

### 1. Unit Tests
**Purpose**: Test individual components in isolation
**Speed**: < 30 seconds
**Coverage**: Individual functions and modules

```bash
# Run only unit tests
./scripts/test_all_enhanced.sh --unit-only

# With coverage
./scripts/test_all_enhanced.sh --unit-only --coverage-only
```

**What's tested**:
- Individual function behavior
- Component interfaces
- Error handling
- Edge cases

### 2. Integration Tests  
**Purpose**: Test component interactions
**Speed**: 1-2 minutes
**Coverage**: Service integrations

```bash
# Run integration tests
./scripts/test_all_enhanced.sh --integration-only

# Verbose output for debugging
./scripts/test_all_enhanced.sh --integration-only --verbose
```

**What's tested**:
- Database connections
- Service communication
- API endpoint interactions
- Data flow between components

### 3. End-to-End Tests
**Purpose**: Test complete user workflows
**Speed**: 2-5 minutes
**Coverage**: Full system behavior

```bash
# Run E2E tests
./scripts/test_all_enhanced.sh --e2e-only
```

**What's tested**:
- Document upload and processing
- Query and response generation
- Citation tracking
- Complete RAG pipeline

### 4. Load Tests
**Purpose**: Test system under concurrent load
**Speed**: 5-10 minutes
**Coverage**: Concurrent user simulation

```bash
# Run load tests
./scripts/test_all_enhanced.sh --load-only

# Extended load testing
./scripts/test_all_enhanced.sh --load-only --timeout 1200
```

**What's tested**:
- Concurrent user handling
- Database connection pooling
- Memory usage under load
- Response time degradation

### 5. Performance Tests
**Purpose**: Validate performance targets
**Speed**: 2-5 minutes
**Coverage**: Latency and throughput

```bash
# Run performance tests
./scripts/test_all_enhanced.sh --performance-only

# With regression checking
./scripts/test_all_enhanced.sh --performance-only --regression-check
```

**Performance Targets**:
- Query processing: < 50ms
- Response generation: < 100ms
- End-to-end: < 200ms
- Throughput: > 100 QPS

### 6. Accuracy Tests
**Purpose**: Validate ML model accuracy
**Speed**: 5-10 minutes
**Coverage**: RAG accuracy and citations

```bash
# Run accuracy tests
./scripts/test_all_enhanced.sh --accuracy-only
```

**What's tested**:
- Search result relevance
- Citation accuracy
- Response quality
- Model performance

## CI/CD Integration

### Pull Request Workflow

```yaml
# .github/workflows/pr-validation.yml
name: PR Validation

on: [pull_request]

jobs:
  fast-feedback:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Quick Tests
        run: ./scripts/test_quick_enhanced.sh
        
  comprehensive-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Full Test Suite
        run: ./scripts/test_all_enhanced.sh --ci --parallel
```

### CI Optimization Options

```bash
# CI-optimized execution (faster, less verbose)
./scripts/test_all_enhanced.sh --ci

# Parallel execution with more workers
./scripts/test_all_enhanced.sh --parallel --jobs 8

# Timeout control for CI environments
./scripts/test_all_enhanced.sh --timeout 600
```

## Environment-Specific Usage

### Local Development

```bash
# Verbose output for debugging
./scripts/test_all_enhanced.sh --local --verbose

# Quick iteration cycle
./scripts/test_quick_enhanced.sh --verbose
```

### Docker Environment

```bash
# Skip service checks (assume services running)
./scripts/test_quick_enhanced.sh --skip-services

# Docker-optimized testing
./scripts/test_all_enhanced.sh --timeout 900
```

### Production Validation

```bash
# Full comprehensive suite with security
./scripts/test_all_enhanced.sh --comprehensive --security

# Performance regression monitoring
./scripts/performance_monitor.sh --check-regression
```

## Troubleshooting

### Common Issues and Solutions

#### Services Not Responding
```bash
# Check service status
./scripts/dev_start.sh

# Manual service startup
docker-compose up -d postgres mongodb redis qdrant

# Check service logs
docker-compose logs postgres mongodb redis
```

#### Tests Taking Too Long
```bash
# Run specific test categories
./scripts/test_all_enhanced.sh --unit-only

# Increase parallel jobs
./scripts/test_all_enhanced.sh --jobs 8

# Skip slow tests
./scripts/test_quick_enhanced.sh
```

#### Performance Issues
```bash
# Check performance baseline
./scripts/performance_monitor.sh --baseline

# Monitor resource usage
./scripts/performance_monitor.sh --monitor

# Check for regressions
./scripts/performance_monitor.sh --check-regression
```

#### Coverage Issues
```bash
# Generate coverage report only
./scripts/test_all_enhanced.sh --coverage-only

# Check coverage threshold
./scripts/test_all_enhanced.sh --unit-only --coverage-only
```

### Debug Mode

```bash
# Maximum verbosity
VERBOSE=true ./scripts/test_all_enhanced.sh --verbose

# Debug specific test category
RUST_LOG=debug ./scripts/test_all_enhanced.sh --integration-only --verbose

# Keep test artifacts for inspection
CLEAN_ARTIFACTS=false ./scripts/test_all_enhanced.sh
```

## Performance Monitoring

### Establishing Baselines

```bash
# Set performance baseline after major changes
./scripts/performance_monitor.sh --baseline

# Store baseline in version control
git add performance/baseline.json
git commit -m "Update performance baseline"
```

### Regression Detection

```bash
# Automatic regression checking
./scripts/test_all_enhanced.sh --regression-check

# Manual regression analysis
./scripts/performance_monitor.sh --check-regression --detailed
```

### Performance Trends

```bash
# Generate performance trends report
./scripts/performance_monitor.sh --trends --period 30d

# Compare with previous version
./scripts/performance_monitor.sh --compare main
```

## Test Data Management

### Test Database Setup

```bash
# Initialize test database
psql $POSTGRES_URL_TEST < scripts/init-db.sql

# Create test data
./scripts/test_data_manager.sh --create --samples 100

# Clean test data
./scripts/test_data_manager.sh --clean
```

### Test Isolation

```bash
# Run tests in isolated environment
./scripts/test_all_enhanced.sh --isolated

# Parallel test execution with isolation
./scripts/test_all_enhanced.sh --parallel --isolated
```

## Reporting and Analysis

### Generated Reports

After running tests, the following reports are available:

- **HTML Summary**: `test-reports/test-summary.html`
- **Coverage Report**: `coverage/tarpaulin-report.html`
- **Performance Results**: `performance/latest-results.json`
- **Full Logs**: `test-reports/test-output.log`

### Report Integration

```bash
# Generate reports only
./scripts/test_all_enhanced.sh --unit-only --coverage-only

# Skip report generation (faster execution)
./scripts/test_all_enhanced.sh --no-reports

# Custom report directory
REPORTS_DIR=/tmp/test-reports ./scripts/test_all_enhanced.sh
```

## Best Practices

### Development Testing

1. **Run quick tests frequently** - After every significant change
2. **Run full tests before commits** - Ensure nothing breaks
3. **Use appropriate test categories** - Don't run load tests for minor changes
4. **Monitor performance** - Watch for regressions early

### CI/CD Testing  

1. **Optimize for speed** - Use parallel execution and appropriate timeouts
2. **Generate artifacts** - Save reports and logs for analysis
3. **Use matrix testing** - Test across different environments
4. **Cache dependencies** - Speed up repeated runs

### Performance Testing

1. **Establish baselines** - After major changes or releases
2. **Monitor trends** - Track performance over time
3. **Set realistic targets** - Based on actual usage patterns
4. **Automate regression detection** - Catch issues early

This guide provides comprehensive coverage of the testing workflow while maintaining simplicity for daily development tasks.