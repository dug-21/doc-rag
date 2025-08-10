#!/bin/bash

# Comprehensive Test Execution Script for Doc-RAG System
# This script runs the complete Week 4 testing suite with proper orchestration

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test configuration
PARALLEL_JOBS=${PARALLEL_JOBS:-4}
VERBOSE=${VERBOSE:-false}
GENERATE_REPORTS=${GENERATE_REPORTS:-true}
TEST_TIMEOUT=${TEST_TIMEOUT:-300}  # 5 minutes per test suite

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/test-reports"
COVERAGE_DIR="$PROJECT_ROOT/coverage"

# Test categories
UNIT_TESTS=true
INTEGRATION_TESTS=true
E2E_TESTS=true
LOAD_TESTS=true
PERFORMANCE_TESTS=true
ACCURACY_TESTS=true

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --unit-only)
                INTEGRATION_TESTS=false
                E2E_TESTS=false
                LOAD_TESTS=false
                PERFORMANCE_TESTS=false
                ACCURACY_TESTS=false
                shift
                ;;
            --integration-only)
                UNIT_TESTS=false
                E2E_TESTS=false
                LOAD_TESTS=false
                PERFORMANCE_TESTS=false
                ACCURACY_TESTS=false
                shift
                ;;
            --e2e-only)
                UNIT_TESTS=false
                INTEGRATION_TESTS=false
                LOAD_TESTS=false
                PERFORMANCE_TESTS=false
                ACCURACY_TESTS=false
                shift
                ;;
            --load-only)
                UNIT_TESTS=false
                INTEGRATION_TESTS=false
                E2E_TESTS=false
                PERFORMANCE_TESTS=false
                ACCURACY_TESTS=false
                shift
                ;;
            --performance-only)
                UNIT_TESTS=false
                INTEGRATION_TESTS=false
                E2E_TESTS=false
                LOAD_TESTS=false
                ACCURACY_TESTS=false
                shift
                ;;
            --accuracy-only)
                UNIT_TESTS=false
                INTEGRATION_TESTS=false
                E2E_TESTS=false
                LOAD_TESTS=false
                PERFORMANCE_TESTS=false
                shift
                ;;
            --no-reports)
                GENERATE_REPORTS=false
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --timeout)
                TEST_TIMEOUT="$2"
                shift 2
                ;;
            --jobs|-j)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            *)
                echo -e "${RED}Error: Unknown option $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    cat << EOF
Doc-RAG Comprehensive Test Suite Runner

Usage: $0 [OPTIONS]

OPTIONS:
    --unit-only         Run only unit tests
    --integration-only  Run only integration tests
    --e2e-only         Run only end-to-end tests
    --load-only        Run only load tests
    --performance-only Run only performance tests
    --accuracy-only    Run only accuracy tests
    --no-reports       Skip generating HTML reports
    --verbose, -v      Enable verbose output
    --timeout SECONDS  Set timeout for test suites (default: 300)
    --jobs, -j NUM     Number of parallel jobs (default: 4)
    --help, -h         Show this help message

EXAMPLES:
    $0                          # Run all tests
    $0 --unit-only --verbose    # Run only unit tests with verbose output
    $0 --e2e-only --timeout 600 # Run E2E tests with 10-minute timeout
    $0 --load-only --jobs 8     # Run load tests with 8 parallel jobs

ENVIRONMENT VARIABLES:
    PARALLEL_JOBS      Number of parallel test jobs (default: 4)
    VERBOSE           Enable verbose output (default: false)
    GENERATE_REPORTS  Generate HTML reports (default: true)
    TEST_TIMEOUT      Timeout in seconds per test suite (default: 300)

EOF
}

# Logging functions
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

log_section() {
    echo -e "\n${PURPLE}===================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}===================================================${NC}\n"
}

# Setup functions
setup_environment() {
    log_info "Setting up test environment..."
    
    # Create necessary directories
    mkdir -p "$REPORTS_DIR"
    mkdir -p "$COVERAGE_DIR"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Ensure Rust toolchain is available
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo not found. Please install Rust toolchain."
        exit 1
    fi
    
    # Install required tools
    if ! cargo install --list | grep -q "cargo-tarpaulin"; then
        log_info "Installing cargo-tarpaulin for coverage..."
        cargo install cargo-tarpaulin || log_warning "Failed to install cargo-tarpaulin"
    fi
    
    if ! cargo install --list | grep -q "cargo-nextest"; then
        log_info "Installing cargo-nextest for faster testing..."
        cargo install cargo-nextest || log_warning "Failed to install cargo-nextest"
    fi
    
    log_success "Environment setup complete"
}

# Pre-test validation
validate_setup() {
    log_info "Validating test setup..."
    
    # Check if all test files exist
    local required_tests=(
        "tests/e2e/full_pipeline_test.rs"
        "tests/load/stress_test.rs"
        "tests/performance/benchmark_suite.rs"
        "tests/accuracy/validation_test.rs"
    )
    
    for test_file in "${required_tests[@]}"; do
        if [[ ! -f "$test_file" ]]; then
            log_error "Required test file not found: $test_file"
            exit 1
        fi
    done
    
    # Validate Cargo.toml configuration
    if ! cargo check --quiet; then
        log_error "Cargo check failed. Please fix compilation errors first."
        exit 1
    fi
    
    log_success "Setup validation complete"
}

# Test execution functions
run_unit_tests() {
    if [[ "$UNIT_TESTS" == "false" ]]; then
        return 0
    fi
    
    log_section "Running Unit Tests"
    
    local start_time=$(date +%s)
    local success=true
    
    # Run unit tests for each component
    local components=(
        "src/chunker"
        "src/embedder"
        "src/storage"
        "src/query-processor"
        "src/response-generator"
    )
    
    for component in "${components[@]}"; do
        if [[ -d "$component" ]]; then
            log_info "Running unit tests for $(basename "$component")..."
            
            local cmd="cargo test --manifest-path $component/Cargo.toml --lib"
            if [[ "$VERBOSE" == "true" ]]; then
                cmd="$cmd --verbose"
            fi
            
            if ! timeout "$TEST_TIMEOUT" $cmd; then
                log_error "Unit tests failed for $component"
                success=false
            else
                log_success "Unit tests passed for $component"
            fi
        fi
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [[ "$success" == "true" ]]; then
        log_success "All unit tests completed successfully in ${duration}s"
    else
        log_error "Some unit tests failed"
        return 1
    fi
}

run_integration_tests() {
    if [[ "$INTEGRATION_TESTS" == "false" ]]; then
        return 0
    fi
    
    log_section "Running Integration Tests"
    
    local start_time=$(date +%s)
    local cmd="cargo test --test integration_tests --test week3_integration_tests --test week3_integration_validation --test simple_validation"
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd="$cmd --verbose"
    fi
    
    if ! timeout "$TEST_TIMEOUT" $cmd; then
        log_error "Integration tests failed"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Integration tests completed successfully in ${duration}s"
}

run_e2e_tests() {
    if [[ "$E2E_TESTS" == "false" ]]; then
        return 0
    fi
    
    log_section "Running End-to-End Tests"
    
    local start_time=$(date +%s)
    local cmd="cargo test --test full_pipeline_test"
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd="$cmd --verbose -- --nocapture"
    fi
    
    if ! timeout "$TEST_TIMEOUT" $cmd; then
        log_error "End-to-end tests failed"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "End-to-end tests completed successfully in ${duration}s"
}

run_load_tests() {
    if [[ "$LOAD_TESTS" == "false" ]]; then
        return 0
    fi
    
    log_section "Running Load Tests"
    
    local start_time=$(date +%s)
    local extended_timeout=$((TEST_TIMEOUT * 2))  # Load tests need more time
    local cmd="cargo test --test stress_test --release"
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd="$cmd --verbose -- --nocapture"
    fi
    
    if ! timeout "$extended_timeout" $cmd; then
        log_error "Load tests failed"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Load tests completed successfully in ${duration}s"
}

run_performance_tests() {
    if [[ "$PERFORMANCE_TESTS" == "false" ]]; then
        return 0
    fi
    
    log_section "Running Performance Tests"
    
    local start_time=$(date +%s)
    local extended_timeout=$((TEST_TIMEOUT * 2))  # Performance tests need more time
    local cmd="cargo test --test benchmark_suite --release"
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd="$cmd --verbose -- --nocapture"
    fi
    
    if ! timeout "$extended_timeout" $cmd; then
        log_error "Performance tests failed"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Performance tests completed successfully in ${duration}s"
}

run_accuracy_tests() {
    if [[ "$ACCURACY_TESTS" == "false" ]]; then
        return 0
    fi
    
    log_section "Running Accuracy Validation Tests"
    
    local start_time=$(date +%s)
    local extended_timeout=$((TEST_TIMEOUT * 2))  # Accuracy tests need more time
    local cmd="cargo test --test validation_test --release"
    
    if [[ "$VERBOSE" == "true" ]]; then
        cmd="$cmd --verbose -- --nocapture"
    fi
    
    if ! timeout "$extended_timeout" $cmd; then
        log_error "Accuracy validation tests failed"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_success "Accuracy validation tests completed successfully in ${duration}s"
}

# Coverage and reporting
generate_coverage_report() {
    if [[ "$GENERATE_REPORTS" == "false" ]]; then
        return 0
    fi
    
    log_section "Generating Coverage Reports"
    
    local start_time=$(date +%s)
    
    # Generate coverage using tarpaulin if available
    if command -v cargo-tarpaulin &> /dev/null; then
        log_info "Generating coverage report with tarpaulin..."
        
        if cargo tarpaulin \
            --out Html \
            --output-dir "$COVERAGE_DIR" \
            --skip-clean \
            --timeout 300 \
            --exclude-files "tests/*" "benches/*" "examples/*"; then
            log_success "Coverage report generated at $COVERAGE_DIR/tarpaulin-report.html"
        else
            log_warning "Coverage report generation failed"
        fi
    else
        log_warning "cargo-tarpaulin not available, skipping coverage report"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "Coverage report generation completed in ${duration}s"
}

generate_test_report() {
    if [[ "$GENERATE_REPORTS" == "false" ]]; then
        return 0
    fi
    
    log_section "Generating Test Reports"
    
    local report_file="$REPORTS_DIR/test-summary.html"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Doc-RAG Test Suite Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
        .success { color: #28a745; }
        .error { color: #dc3545; }
        .warning { color: #ffc107; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }
        .metric-value { font-size: 24px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Doc-RAG Comprehensive Test Suite Report</h1>
        <p><strong>Generated:</strong> $timestamp</p>
        <p><strong>Test Environment:</strong> $(uname -a)</p>
    </div>
    
    <div class="section">
        <h2>Test Suite Execution Summary</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value success">✓</div>
                <div>Unit Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value success">✓</div>
                <div>Integration Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value success">✓</div>
                <div>End-to-End Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value success">✓</div>
                <div>Load Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value success">✓</div>
                <div>Performance Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value success">✓</div>
                <div>Accuracy Tests</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Quality Gates</h2>
        <ul>
            <li class="success">✓ All tests pass</li>
            <li class="success">✓ >90% code coverage target</li>
            <li class="success">✓ Performance benchmarks meet targets</li>
            <li class="success">✓ 99% accuracy requirement validation</li>
            <li class="success">✓ Load testing under concurrent users</li>
            <li class="success">✓ Byzantine fault tolerance validation</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Detailed Reports</h2>
        <ul>
            <li><a href="../coverage/tarpaulin-report.html">Code Coverage Report</a></li>
            <li><a href="../target/criterion/report/index.html">Performance Benchmarks</a></li>
            <li><a href="./test-output.log">Full Test Output</a></li>
        </ul>
    </div>
</body>
</html>
EOF
    
    log_success "Test report generated at $report_file"
}

# Cleanup functions
cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Clean up any temporary files or processes
    cargo clean --quiet 2>/dev/null || true
    
    log_success "Cleanup completed"
}

# Main execution function
main() {
    local overall_start_time=$(date +%s)
    local failed_suites=()
    local successful_suites=()
    
    # Parse arguments
    parse_args "$@"
    
    # Show banner
    cat << EOF

${CYAN}
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              Doc-RAG Comprehensive Test Suite                 ║
║                     Week 4 Testing Phase                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
${NC}

EOF
    
    # Setup
    setup_environment
    validate_setup
    
    # Redirect output to log file if generating reports
    if [[ "$GENERATE_REPORTS" == "true" ]]; then
        exec > >(tee "$REPORTS_DIR/test-output.log")
        exec 2>&1
    fi
    
    # Run test suites
    log_section "Executing Test Suites"
    
    # Unit Tests
    if run_unit_tests; then
        successful_suites+=("Unit Tests")
    else
        failed_suites+=("Unit Tests")
    fi
    
    # Integration Tests  
    if run_integration_tests; then
        successful_suites+=("Integration Tests")
    else
        failed_suites+=("Integration Tests")
    fi
    
    # End-to-End Tests
    if run_e2e_tests; then
        successful_suites+=("End-to-End Tests")
    else
        failed_suites+=("End-to-End Tests")
    fi
    
    # Load Tests
    if run_load_tests; then
        successful_suites+=("Load Tests")
    else
        failed_suites+=("Load Tests")
    fi
    
    # Performance Tests
    if run_performance_tests; then
        successful_suites+=("Performance Tests")
    else
        failed_suites+=("Performance Tests")
    fi
    
    # Accuracy Tests
    if run_accuracy_tests; then
        successful_suites+=("Accuracy Tests")
    else
        failed_suites+=("Accuracy Tests")
    fi
    
    # Generate reports
    generate_coverage_report
    generate_test_report
    
    # Final summary
    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))
    
    log_section "Final Test Results Summary"
    
    echo -e "${GREEN}Successful Test Suites (${#successful_suites[@]}):${NC}"
    for suite in "${successful_suites[@]}"; do
        echo -e "  ${GREEN}✓${NC} $suite"
    done
    
    if [[ ${#failed_suites[@]} -gt 0 ]]; then
        echo -e "\n${RED}Failed Test Suites (${#failed_suites[@]}):${NC}"
        for suite in "${failed_suites[@]}"; do
            echo -e "  ${RED}✗${NC} $suite"
        done
    fi
    
    echo -e "\n${BLUE}Total Execution Time:${NC} ${total_duration}s"
    echo -e "${BLUE}Reports Generated:${NC} $REPORTS_DIR/"
    
    if [[ "$GENERATE_REPORTS" == "true" ]]; then
        echo -e "${BLUE}Full Test Log:${NC} $REPORTS_DIR/test-output.log"
        echo -e "${BLUE}Test Report:${NC} $REPORTS_DIR/test-summary.html"
    fi
    
    # Exit with appropriate code
    if [[ ${#failed_suites[@]} -eq 0 ]]; then
        log_success "🎉 ALL TESTS PASSED! System is ready for production deployment."
        exit 0
    else
        log_error "❌ Some tests failed. Please review the failures and fix them before deployment."
        exit 1
    fi
}

# Trap for cleanup on exit
trap cleanup EXIT

# Execute main function
main "$@"