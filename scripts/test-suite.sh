#!/bin/bash
# Unified Test Suite Runner - Replaces fragmented test scripts
# Supports both interactive development and CI/CD automation

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_MODE="${1:-all}"
ENVIRONMENT="${2:-local}"
PARALLEL="${3:-true}"
COVERAGE="${4:-false}"

# Timing
START_TIME=$(date +%s)

# Functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    cat << EOF
Usage: $0 [MODE] [ENVIRONMENT] [PARALLEL] [COVERAGE]

MODES:
  all         - Run complete test suite (default)
  unit        - Unit tests only
  integration - Integration tests only
  e2e         - End-to-end tests
  perf        - Performance tests
  security    - Security scanning
  quick       - Fast smoke tests
  watch       - Continuous testing with hot-reload

ENVIRONMENTS:
  local   - Local development (default)
  docker  - Docker environment
  ci      - CI/CD pipeline

OPTIONS:
  PARALLEL - true/false (default: true)
  COVERAGE - true/false (default: false)

Examples:
  $0 quick local           # Quick local tests
  $0 all ci true true      # Full CI suite with coverage
  $0 watch docker          # Watch mode in Docker
EOF
    exit 0
}

# Check prerequisites
check_requirements() {
    log_info "Checking requirements..."
    
    local missing=()
    
    command -v docker >/dev/null 2>&1 || missing+=("docker")
    command -v cargo >/dev/null 2>&1 || missing+=("cargo")
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing requirements: ${missing[*]}"
        exit 1
    fi
    
    log_success "All requirements met"
}

# Setup test environment
setup_environment() {
    log_info "Setting up $ENVIRONMENT environment..."
    
    case $ENVIRONMENT in
        local)
            export MONGODB_URL="mongodb://localhost:27017/doc_rag_test"
            export RUST_LOG="debug"
            ;;
        docker)
            docker-compose -f docker-compose.test.yml up -d
            export MONGODB_URL="mongodb://localhost:27018/doc_rag_test"
            ;;
        ci)
            docker-compose -f docker-compose.ci.yml up -d
            export CI=true
            export RUST_LOG="warn"
            ;;
    esac
    
    # Wait for services
    if [ "$ENVIRONMENT" != "local" ]; then
        log_info "Waiting for services to be healthy..."
        sleep 5
        
        if ! docker-compose -f docker-compose.${ENVIRONMENT}.yml ps | grep -q "healthy"; then
            log_warning "Some services may not be fully ready"
        fi
    fi
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    local cmd="cargo test --workspace --lib"
    
    if [ "$PARALLEL" = "true" ]; then
        cmd="$cmd -- --test-threads=4"
    fi
    
    if [ "$COVERAGE" = "true" ]; then
        export CARGO_INCREMENTAL=0
        export RUSTFLAGS="-Cinstrument-coverage"
        export LLVM_PROFILE_FILE="target/coverage/unit-%p-%m.profraw"
    fi
    
    if $cmd; then
        log_success "Unit tests passed"
        return 0
    else
        log_error "Unit tests failed"
        return 1
    fi
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    local cmd="cargo test --workspace --test '*'"
    
    if [ "$PARALLEL" = "true" ]; then
        cmd="$cmd -- --test-threads=2"
    fi
    
    if $cmd; then
        log_success "Integration tests passed"
        return 0
    else
        log_error "Integration tests failed"
        return 1
    fi
}

# Run E2E tests
run_e2e_tests() {
    log_info "Running end-to-end tests..."
    
    # Start API server in background
    cargo run --release --bin api &
    local api_pid=$!
    
    # Wait for API to be ready
    sleep 5
    
    # Run E2E test suite
    if ./scripts/test_api.sh; then
        log_success "E2E tests passed"
        kill $api_pid
        return 0
    else
        log_error "E2E tests failed"
        kill $api_pid
        return 1
    fi
}

# Run performance tests
run_perf_tests() {
    log_info "Running performance tests..."
    
    cargo test --release --features perf-tests perf_ -- --nocapture
    
    # Check performance regression
    if [ -f "perf-baseline.json" ]; then
        log_info "Checking for performance regressions..."
        # Compare with baseline (implement comparison logic)
    fi
    
    log_success "Performance tests completed"
}

# Run security scanning
run_security_scan() {
    log_info "Running security scan..."
    
    # Cargo audit
    if command -v cargo-audit >/dev/null 2>&1; then
        cargo audit
    fi
    
    # Trivy scan if in Docker
    if [ "$ENVIRONMENT" = "docker" ] || [ "$ENVIRONMENT" = "ci" ]; then
        docker run --rm -v "$PWD":/workspace aquasec/trivy fs /workspace
    fi
    
    log_success "Security scan completed"
}

# Quick smoke tests
run_quick_tests() {
    log_info "Running quick smoke tests..."
    
    cargo test --workspace --lib test_health
    cargo test --workspace --lib test_basic
    
    log_success "Quick tests passed"
}

# Watch mode for development
run_watch_mode() {
    log_info "Starting watch mode..."
    
    if [ "$ENVIRONMENT" = "docker" ]; then
        docker-compose -f docker-compose.test.yml run --rm test-runner \
            cargo watch -x 'test --workspace' -w src -w tests
    else
        cargo watch -x 'test --workspace' -w src -w tests
    fi
}

# Generate coverage report
generate_coverage_report() {
    if [ "$COVERAGE" = "true" ]; then
        log_info "Generating coverage report..."
        
        grcov target/coverage \
            --binary-path ./target/debug/deps \
            -s . \
            -t html \
            --branch \
            --ignore-not-existing \
            -o ./coverage-report
        
        log_success "Coverage report generated at ./coverage-report/index.html"
    fi
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    case $ENVIRONMENT in
        docker)
            docker-compose -f docker-compose.test.yml down -v
            ;;
        ci)
            docker-compose -f docker-compose.ci.yml down -v
            ;;
    esac
    
    # Calculate elapsed time
    local end_time=$(date +%s)
    local elapsed=$((end_time - START_TIME))
    
    log_info "Test suite completed in ${elapsed} seconds"
}

# Main execution
main() {
    # Parse arguments
    [ "$1" = "help" ] || [ "$1" = "-h" ] || [ "$1" = "--help" ] && show_usage
    
    # Setup trap for cleanup
    trap cleanup EXIT
    
    # Run checks
    check_requirements
    setup_environment
    
    # Execute test mode
    case $TEST_MODE in
        all)
            run_unit_tests
            run_integration_tests
            run_e2e_tests
            run_perf_tests
            run_security_scan
            ;;
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        e2e)
            run_e2e_tests
            ;;
        perf)
            run_perf_tests
            ;;
        security)
            run_security_scan
            ;;
        quick)
            run_quick_tests
            ;;
        watch)
            run_watch_mode
            ;;
        *)
            log_error "Unknown test mode: $TEST_MODE"
            show_usage
            ;;
    esac
    
    # Generate coverage if requested
    generate_coverage_report
    
    log_success "All tests completed successfully!"
}

# Run main function
main "$@"