#!/bin/bash
# Interactive Test Runner with Hot-Reload Capabilities
# Usage: ./scripts/test-runner.sh [command] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE_TEST="docker-compose.test.yml"
COMPOSE_FILE_CI="docker-compose.ci.yml"
PROJECT_NAME="doc-rag-test"

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are available
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    log_success "Dependencies check passed"
}

# Start test environment
start_test_env() {
    log_info "Starting test environment..."
    
    # Clean up any existing containers
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME down -v 2>/dev/null || true
    
    # Start infrastructure services first
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME up -d \
        test-mongodb test-redis test-qdrant
    
    log_info "Waiting for infrastructure services to be ready..."
    
    # Wait for services to be healthy
    local retries=30
    while [ $retries -gt 0 ]; do
        if docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME ps | grep -q "healthy"; then
            log_success "Infrastructure services are ready"
            break
        fi
        log_info "Waiting for services... ($retries attempts remaining)"
        sleep 2
        ((retries--))
    done
    
    if [ $retries -eq 0 ]; then
        log_error "Infrastructure services failed to start within timeout"
        docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME logs
        exit 1
    fi
    
    # Start application services
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME up -d
    
    log_success "Test environment started successfully"
}

# Stop test environment
stop_test_env() {
    log_info "Stopping test environment..."
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME down -v
    log_success "Test environment stopped"
}

# Run unit tests
run_unit_tests() {
    log_info "Running unit tests..."
    
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME exec test-runner \
        cargo test --lib --bins --workspace --color always -- --nocapture
    
    log_success "Unit tests completed"
}

# Run integration tests
run_integration_tests() {
    log_info "Running integration tests..."
    
    # Ensure services are running
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME up -d api-test storage-test embedder-test
    
    # Wait for services to be ready
    sleep 10
    
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME exec test-runner \
        cargo test --test integration_tests --color always -- --nocapture
    
    log_success "Integration tests completed"
}

# Run specific test file
run_specific_test() {
    local test_name=$1
    if [ -z "$test_name" ]; then
        log_error "Test name required"
        exit 1
    fi
    
    log_info "Running specific test: $test_name"
    
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME exec test-runner \
        cargo test $test_name --color always -- --nocapture
    
    log_success "Test $test_name completed"
}

# Interactive test shell
interactive_shell() {
    log_info "Starting interactive test shell..."
    
    # Ensure test environment is running
    start_test_env
    
    # Seed test data
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME run --rm test-seeder
    
    log_success "Interactive shell ready. Run tests with 'cargo test'"
    
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME exec test-runner bash
}

# Watch mode for continuous testing
watch_tests() {
    local test_pattern=${1:-""}
    
    log_info "Starting watch mode for tests..."
    log_info "Test pattern: ${test_pattern:-'all tests'}"
    
    # Start test environment
    start_test_env
    
    # Run cargo watch in test runner
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME exec test-runner \
        cargo watch -x "test $test_pattern --color always"
}

# Health check for test environment
health_check() {
    log_info "Checking test environment health..."
    
    local services=("test-mongodb:27017" "test-redis:6379" "test-qdrant:6333")
    local api_services=("api-test:8080" "storage-test:8080" "embedder-test:8080")
    
    for service in "${services[@]}"; do
        local name=$(echo $service | cut -d: -f1)
        local port=$(echo $service | cut -d: -f2)
        
        if docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME exec $name nc -z localhost $port 2>/dev/null; then
            log_success "$name is healthy"
        else
            log_error "$name is not responding"
        fi
    done
    
    for service in "${api_services[@]}"; do
        local name=$(echo $service | cut -d: -f1)
        local port=$(echo $service | cut -d: -f2)
        
        local health_url="http://localhost:$port/health"
        if docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME exec $name curl -f $health_url 2>/dev/null; then
            log_success "$name API is healthy"
        else
            log_warning "$name API is not responding (may not be started)"
        fi
    done
}

# CI mode - runs all tests in sequence
run_ci_tests() {
    log_info "Running CI test suite..."
    
    # Use CI compose file
    docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" down -v
    
    # Start infrastructure
    docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" up -d \
        ci-mongodb ci-redis ci-qdrant
    
    log_info "Waiting for CI infrastructure..."
    sleep 15
    
    # Run test suites in parallel
    local exit_code=0
    
    # Unit tests
    docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" run --rm unit-tests || exit_code=$?
    
    # Integration tests  
    docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" run --rm integration-tests || exit_code=$?
    
    # API tests
    docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" run --rm api-tests || exit_code=$?
    
    # Security tests
    docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" run --rm security-tests || exit_code=$?
    
    # Performance tests (optional in CI)
    if [ "${RUN_PERF_TESTS:-false}" = "true" ]; then
        docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" run --rm performance-tests || exit_code=$?
    fi
    
    # Generate report
    docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" run --rm test-reporter
    
    # Cleanup
    docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" down -v
    
    if [ $exit_code -eq 0 ]; then
        log_success "All CI tests passed"
    else
        log_error "Some CI tests failed"
    fi
    
    exit $exit_code
}

# Show usage information
show_usage() {
    echo "Doc-RAG Test Runner"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start           Start test environment"
    echo "  stop            Stop test environment"
    echo "  unit            Run unit tests"
    echo "  integration     Run integration tests"
    echo "  test [NAME]     Run specific test"
    echo "  shell           Interactive test shell"
    echo "  watch [PATTERN] Watch mode for continuous testing"
    echo "  health          Check test environment health"
    echo "  ci              Run full CI test suite"
    echo "  logs [SERVICE]  Show service logs"
    echo "  clean           Clean up all test resources"
    echo ""
    echo "Examples:"
    echo "  $0 start                    # Start test environment"
    echo "  $0 unit                     # Run unit tests"
    echo "  $0 test api_tests          # Run specific test"
    echo "  $0 watch integration       # Watch integration tests"
    echo "  $0 shell                   # Interactive testing"
}

# Show service logs
show_logs() {
    local service=${1:-""}
    
    if [ -z "$service" ]; then
        docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME logs -f
    else
        docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME logs -f $service
    fi
}

# Clean up all test resources
clean_all() {
    log_info "Cleaning up all test resources..."
    
    docker-compose -f $COMPOSE_FILE_TEST -p $PROJECT_NAME down -v --remove-orphans
    docker-compose -f $COMPOSE_FILE_CI -p "${PROJECT_NAME}-ci" down -v --remove-orphans
    
    # Remove test images
    docker images | grep doc-rag-test | awk '{print $3}' | xargs docker rmi -f 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main command dispatcher
main() {
    check_dependencies
    
    case "${1:-help}" in
        start)
            start_test_env
            ;;
        stop)
            stop_test_env
            ;;
        unit)
            run_unit_tests
            ;;
        integration)
            run_integration_tests
            ;;
        test)
            run_specific_test "$2"
            ;;
        shell)
            interactive_shell
            ;;
        watch)
            watch_tests "$2"
            ;;
        health)
            health_check
            ;;
        ci)
            run_ci_tests
            ;;
        logs)
            show_logs "$2"
            ;;
        clean)
            clean_all
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"