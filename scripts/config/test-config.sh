#!/bin/bash
# Shared Test Configuration for Doc-RAG System
# This file provides common configuration variables and functions used across all testing scripts

# =============================================================================
# CORE TEST CONFIGURATION
# =============================================================================

# Default timeouts (in seconds)
export TEST_TIMEOUT_DEFAULT=300        # 5 minutes
export TEST_TIMEOUT_EXTENDED=600       # 10 minutes  
export TEST_TIMEOUT_QUICK=30           # 30 seconds
export TEST_TIMEOUT_STRESS=1200        # 20 minutes for stress tests

# Parallel execution
export PARALLEL_JOBS_DEFAULT=4
export PARALLEL_JOBS_CI=8
export PARALLEL_JOBS_STRESS=2

# Performance targets
export TARGET_QUERY_MS=50              # Query processing under 50ms
export TARGET_RESPONSE_MS=100          # Response generation under 100ms
export TARGET_E2E_MS=200              # End-to-end under 200ms
export TARGET_THROUGHPUT_QPS=100       # 100 queries per second
export TARGET_MEMORY_MB=2048          # Memory usage under 2GB

# Quality gates
export COVERAGE_THRESHOLD=90           # 90% code coverage minimum
export PERFORMANCE_REGRESSION_THRESHOLD=0.05  # 5% performance regression threshold
export ACCURACY_THRESHOLD=0.95        # 95% accuracy minimum

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# Service endpoints
export POSTGRES_URL_TEST="postgres://docrag:docrag_password@localhost:5432/docrag_test"
export MONGODB_URL_TEST="mongodb://localhost:27017/docrag_test"
export REDIS_URL_TEST="redis://localhost:6379/0"
export QDRANT_URL_TEST="http://localhost:6333"
export MINIO_URL_TEST="http://localhost:9000"
export API_URL_DEFAULT="http://localhost:8080"

# Docker configuration
export DOCKER_COMPOSE_FILE="docker-compose.yml"
export DOCKER_COMPOSE_TEST_FILE="docker-compose.test.yml"
export DOCKER_NETWORK="doc-rag-test-network"

# File paths
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPORTS_DIR="$PROJECT_ROOT/test-reports"
export COVERAGE_DIR="$PROJECT_ROOT/coverage"
export PERFORMANCE_DIR="$PROJECT_ROOT/performance"
export TEST_DATA_DIR="$PROJECT_ROOT/test-data"

# =============================================================================
# RUST/CARGO CONFIGURATION  
# =============================================================================

# Cargo test configuration
export CARGO_TEST_FLAGS="--color always"
export CARGO_BUILD_FLAGS="--color always"
export RUST_BACKTRACE=1
export RUST_LOG="info"
export CARGO_INCREMENTAL=1

# Development tools
export REQUIRED_TOOLS=(
    "cargo-tarpaulin:coverage analysis"
    "cargo-audit:security scanning"  
    "cargo-criterion:benchmarking"
    "cargo-nextest:faster testing"
)

# Test categories
export TEST_CATEGORIES=(
    "unit:Unit Tests"
    "integration:Integration Tests"
    "e2e:End-to-End Tests"
    "load:Load Tests"
    "performance:Performance Tests"
    "accuracy:Accuracy Tests"
    "security:Security Tests"
)

# =============================================================================
# LOGGING AND REPORTING
# =============================================================================

# Colors for output
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export PURPLE='\033[0;35m'
export CYAN='\033[0;36m'
export NC='\033[0m' # No Color

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Environment detection
detect_test_environment() {
    if [[ "$CI" == "true" ]]; then
        echo "ci"
    elif [[ -f "/.dockerenv" ]]; then
        echo "docker"
    elif [[ "$CODESPACES" == "true" ]]; then
        echo "codespaces"
    else
        echo "local"
    fi
}

# Load environment-specific configuration
load_environment_config() {
    local env=$(detect_test_environment)
    case "$env" in
        "ci")
            export PARALLEL_JOBS=${PARALLEL_JOBS:-$PARALLEL_JOBS_CI}
            export TEST_TIMEOUT=${TEST_TIMEOUT:-$TEST_TIMEOUT_DEFAULT}
            export GENERATE_REPORTS=${GENERATE_REPORTS:-true}
            export VERBOSE=${VERBOSE:-false}
            ;;
        "docker")
            export PARALLEL_JOBS=${PARALLEL_JOBS:-$PARALLEL_JOBS_DEFAULT}
            export TEST_TIMEOUT=${TEST_TIMEOUT:-$TEST_TIMEOUT_EXTENDED}
            export GENERATE_REPORTS=${GENERATE_REPORTS:-true}
            export VERBOSE=${VERBOSE:-false}
            ;;
        "local")
            export PARALLEL_JOBS=${PARALLEL_JOBS:-$PARALLEL_JOBS_DEFAULT}
            export TEST_TIMEOUT=${TEST_TIMEOUT:-$TEST_TIMEOUT_DEFAULT}
            export GENERATE_REPORTS=${GENERATE_REPORTS:-true}
            export VERBOSE=${VERBOSE:-true}
            ;;
    esac
}

# Check if running in CI
is_ci() {
    [[ "$CI" == "true" ]] || [[ "$GITHUB_ACTIONS" == "true" ]] || [[ "$GITLAB_CI" == "true" ]]
}

# Get optimal parallel jobs for current system
get_optimal_parallel_jobs() {
    local cpu_count
    if command -v nproc &> /dev/null; then
        cpu_count=$(nproc)
    elif command -v sysctl &> /dev/null; then
        cpu_count=$(sysctl -n hw.ncpu 2>/dev/null || echo "4")
    else
        cpu_count=4
    fi
    
    # Use 75% of available CPUs, minimum 2, maximum 16
    local optimal=$(( cpu_count * 3 / 4 ))
    [[ $optimal -lt 2 ]] && optimal=2
    [[ $optimal -gt 16 ]] && optimal=16
    
    echo "$optimal"
}

# Create test directories
create_test_directories() {
    local dirs=(
        "$REPORTS_DIR"
        "$COVERAGE_DIR" 
        "$PERFORMANCE_DIR"
        "$TEST_DATA_DIR"
        "$PROJECT_ROOT/logs"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
}

# Check if required tools are available
check_required_tools() {
    local missing_tools=()
    
    # Check essential tools
    local essential_tools=("cargo" "rustc" "docker" "curl")
    for tool in "${essential_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool:essential tool")
        fi
    done
    
    # Check optional development tools
    for tool_info in "${REQUIRED_TOOLS[@]}"; do
        local tool="${tool_info%%:*}"
        local desc="${tool_info#*:}"
        
        if ! cargo install --list | grep -q "$tool" 2>/dev/null; then
            missing_tools+=("$tool:$desc (optional)")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        echo "Missing tools detected:"
        for tool_info in "${missing_tools[@]}"; do
            local tool="${tool_info%%:*}"
            local desc="${tool_info#*:}"
            
            if [[ "$desc" == *"optional"* ]]; then
                echo "  ⚠️  $tool - $desc"
            else
                echo "  ❌ $tool - $desc"
            fi
        done
        
        # Count essential missing tools
        local essential_missing=0
        for tool_info in "${missing_tools[@]}"; do
            [[ "$tool_info" != *"optional"* ]] && ((essential_missing++))
        done
        
        return $essential_missing
    fi
    
    return 0
}

# Service health check
check_service_health() {
    local service_name="$1"
    local host="$2"
    local port="$3"
    local timeout="${4:-5}"
    
    if timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Wait for services to be ready
wait_for_services() {
    local services=(
        "postgres:localhost:5432"
        "mongodb:localhost:27017"
        "redis:localhost:6379"
        "qdrant:localhost:6333"
    )
    
    local max_wait=120
    local wait_interval=2
    local total_wait=0
    
    echo "Waiting for services to be ready..."
    
    while [[ $total_wait -lt $max_wait ]]; do
        local all_ready=true
        
        for service_info in "${services[@]}"; do
            local service="${service_info%%:*}"
            local host_port="${service_info#*:}"
            local host="${host_port%%:*}"
            local port="${host_port#*:}"
            
            if ! check_service_health "$service" "$host" "$port" 1; then
                all_ready=false
                break
            fi
        done
        
        if [[ "$all_ready" == "true" ]]; then
            echo "✅ All services are ready"
            return 0
        fi
        
        echo -n "."
        sleep $wait_interval
        ((total_wait += wait_interval))
    done
    
    echo ""
    echo "⚠️  Services may not be fully ready, continuing..."
    return 1
}

# Generate timestamp for reports
get_report_timestamp() {
    date '+%Y-%m-%d_%H-%M-%S'
}

# Parse test results from log
parse_test_results() {
    local log_file="$1"
    local results_file="$2"
    
    if [[ ! -f "$log_file" ]]; then
        echo "No log file found: $log_file"
        return 1
    fi
    
    # Extract test results using various patterns
    {
        grep -o "test result: [^,]*" "$log_file" || true
        grep -o "[0-9]* passed; [0-9]* failed" "$log_file" || true
        grep -E "PASS|FAIL" "$log_file" || true
    } > "$results_file"
}

# Calculate test success rate
calculate_success_rate() {
    local passed="$1"
    local total="$2"
    
    if [[ $total -eq 0 ]]; then
        echo "0"
    else
        echo $(( passed * 100 / total ))
    fi
}

# =============================================================================
# INITIALIZATION
# =============================================================================

# Auto-load environment configuration when sourced
load_environment_config

# Create test directories
create_test_directories

# Export functions for use in other scripts
export -f detect_test_environment
export -f load_environment_config
export -f is_ci
export -f get_optimal_parallel_jobs
export -f create_test_directories
export -f check_required_tools
export -f check_service_health
export -f wait_for_services
export -f get_report_timestamp
export -f parse_test_results
export -f calculate_success_rate

# Log configuration load (only if not in quiet mode)
if [[ "${QUIET_CONFIG:-false}" != "true" ]]; then
    echo "📋 Test configuration loaded (Environment: $(detect_test_environment))"
fi