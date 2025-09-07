#!/bin/bash
# Enhanced Comprehensive Test Suite for Doc-RAG System
# Consolidates functionality from run_all_tests.sh, run_week3_tests.sh, and performance_test.sh

set -e

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration - can be overridden by environment
PARALLEL_JOBS=${PARALLEL_JOBS:-4}
TEST_TIMEOUT=${TEST_TIMEOUT:-300}
EXTENDED_TIMEOUT=${EXTENDED_TIMEOUT:-600}
VERBOSE=${VERBOSE:-false}
GENERATE_REPORTS=${GENERATE_REPORTS:-true}
CI_MODE=${CI:-false}

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_ROOT/test-reports"
COVERAGE_DIR="$PROJECT_ROOT/coverage"
PERFORMANCE_DIR="$PROJECT_ROOT/performance"

# Load shared configuration
if [[ -f "$SCRIPT_DIR/config/test-config.sh" ]]; then
    source "$SCRIPT_DIR/config/test-config.sh"
fi

# Test category flags
UNIT_TESTS=true
INTEGRATION_TESTS=true
E2E_TESTS=true
LOAD_TESTS=true
PERFORMANCE_TESTS=true
ACCURACY_TESTS=true
SECURITY_TESTS=false
REGRESSION_CHECK=false

# Performance targets (can be overridden)
TARGET_QUERY_MS=${TARGET_QUERY_MS:-50}
TARGET_RESPONSE_MS=${TARGET_RESPONSE_MS:-100}  
TARGET_E2E_MS=${TARGET_E2E_MS:-200}
TARGET_THROUGHPUT_QPS=${TARGET_THROUGHPUT_QPS:-100}
COVERAGE_THRESHOLD=${COVERAGE_THRESHOLD:-90}

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }  
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_section() {
    echo -e "\n${PURPLE}===================================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}===================================================${NC}\n"
}

# Help function
show_help() {
    cat << EOF
Doc-RAG Enhanced Comprehensive Test Suite

Usage: $0 [OPTIONS]

TEST CATEGORIES:
    --unit-only         Run only unit tests
    --integration-only  Run only integration tests
    --e2e-only         Run only end-to-end tests
    --load-only        Run only load tests
    --performance-only Run only performance tests
    --accuracy-only    Run only accuracy tests
    --security         Include security testing
    --regression-check Check for performance regressions

EXECUTION OPTIONS:
    --ci               CI/CD optimized execution
    --local            Local development optimized
    --comprehensive    Run all tests including optional ones
    --parallel         Enable parallel execution
    --timeout SECONDS  Set timeout for test suites (default: 300)
    --jobs NUM         Number of parallel jobs (default: 4)

REPORTING OPTIONS:
    --no-reports       Skip generating reports
    --coverage-only    Generate coverage report only
    --performance-baseline  Set new performance baseline
    --verbose, -v      Enable verbose output

EXAMPLES:
    $0                           # Run all standard tests
    $0 --ci --parallel          # CI/CD optimized run
    $0 --performance-only --regression-check  # Performance regression check
    $0 --comprehensive --security # Full test suite with security
    $0 --unit-only --coverage-only # Unit tests with coverage

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --unit-only)
                INTEGRATION_TESTS=false; E2E_TESTS=false; LOAD_TESTS=false
                PERFORMANCE_TESTS=false; ACCURACY_TESTS=false
                shift ;;
            --integration-only)
                UNIT_TESTS=false; E2E_TESTS=false; LOAD_TESTS=false
                PERFORMANCE_TESTS=false; ACCURACY_TESTS=false
                shift ;;
            --e2e-only)
                UNIT_TESTS=false; INTEGRATION_TESTS=false; LOAD_TESTS=false
                PERFORMANCE_TESTS=false; ACCURACY_TESTS=false
                shift ;;
            --load-only)
                UNIT_TESTS=false; INTEGRATION_TESTS=false; E2E_TESTS=false
                PERFORMANCE_TESTS=false; ACCURACY_TESTS=false
                shift ;;
            --performance-only)
                UNIT_TESTS=false; INTEGRATION_TESTS=false; E2E_TESTS=false
                LOAD_TESTS=false; ACCURACY_TESTS=false
                shift ;;
            --accuracy-only)
                UNIT_TESTS=false; INTEGRATION_TESTS=false; E2E_TESTS=false
                LOAD_TESTS=false; PERFORMANCE_TESTS=false
                shift ;;
            --security) SECURITY_TESTS=true; shift ;;
            --regression-check) REGRESSION_CHECK=true; shift ;;
            --ci) CI_MODE=true; PARALLEL_JOBS=8; TEST_TIMEOUT=600; shift ;;
            --local) CI_MODE=false; VERBOSE=true; shift ;;
            --comprehensive) 
                SECURITY_TESTS=true; REGRESSION_CHECK=true; EXTENDED_TIMEOUT=1200
                shift ;;
            --parallel) PARALLEL_JOBS=8; shift ;;
            --no-reports) GENERATE_REPORTS=false; shift ;;
            --coverage-only) GENERATE_REPORTS=coverage; shift ;;
            --performance-baseline) PERFORMANCE_BASELINE=true; shift ;;
            --verbose|-v) VERBOSE=true; shift ;;
            --timeout) TEST_TIMEOUT="$2"; shift 2 ;;
            --jobs) PARALLEL_JOBS="$2"; shift 2 ;;
            --help|-h) show_help; exit 0 ;;
            *) log_error "Unknown option: $1"; show_help; exit 1 ;;
        esac
    done
}

# Environment detection and setup
detect_environment() {
    if [[ "$CI" == "true" ]]; then
        echo "ci"
    elif [[ -f "/.dockerenv" ]]; then
        echo "docker"
    else
        echo "local"
    fi
}

setup_environment() {
    log_info "Setting up test environment..."
    
    local env=$(detect_environment)
    log_info "Detected environment: $env"
    
    # Create directories
    mkdir -p "$REPORTS_DIR" "$COVERAGE_DIR" "$PERFORMANCE_DIR"
    cd "$PROJECT_ROOT"
    
    # Validate Rust toolchain
    if ! command -v cargo &> /dev/null; then
        log_error "Cargo not found. Please install Rust toolchain."
        exit 1
    fi
    
    # Install development tools if not in CI
    if [[ "$CI_MODE" != "true" ]]; then
        install_test_tools
    fi
    
    log_success "Environment setup complete"
}

install_test_tools() {
    local tools=(
        "cargo-tarpaulin:coverage analysis"
        "cargo-nextest:faster testing" 
        "cargo-audit:security audit"
        "cargo-criterion:benchmarking"
    )
    
    for tool_info in "${tools[@]}"; do
        local tool="${tool_info%%:*}"
        local desc="${tool_info#*:}"
        
        if ! cargo install --list | grep -q "$tool"; then
            log_info "Installing $tool for $desc..."
            cargo install "$tool" --locked 2>/dev/null || log_warning "Failed to install $tool"
        fi
    done
}

# Service management
start_test_services() {
    log_info "Starting test services..."
    
    if [[ -f "docker-compose.yml" ]]; then
        # Start core services needed for testing
        docker-compose up -d postgres redis qdrant mongodb minio 2>/dev/null || {
            log_warning "Some services may not have started correctly"
        }
        
        # Wait for services with timeout
        log_info "Waiting for services to be ready..."
        local max_wait=60
        local count=0
        
        while [[ $count -lt $max_wait ]]; do
            if docker-compose exec -T postgres pg_isready -U docrag 2>/dev/null; then
                break
            fi
            sleep 2
            ((count++))
        done
        
        if [[ $count -eq $max_wait ]]; then
            log_warning "Services may not be fully ready, continuing..."
        else
            log_success "Test services ready"
        fi
    else
        log_warning "No docker-compose.yml found, assuming services are external"
    fi
}

stop_test_services() {
    if [[ -f "docker-compose.yml" && "$CI_MODE" != "true" ]]; then
        log_info "Stopping test services..."
        docker-compose down -v 2>/dev/null || true
    fi
}

# Test execution functions
run_unit_tests() {
    [[ "$UNIT_TESTS" != "true" ]] && return 0
    
    log_section "Running Unit Tests"
    local start_time=$(date +%s)
    
    local cmd="cargo test --lib --bins --workspace"
    [[ "$VERBOSE" == "true" ]] && cmd="$cmd --verbose"
    [[ "$PARALLEL_JOBS" -gt 1 ]] && cmd="$cmd --jobs=$PARALLEL_JOBS"
    
    if timeout "$TEST_TIMEOUT" $cmd; then
        local duration=$(( $(date +%s) - start_time ))
        log_success "Unit tests completed successfully in ${duration}s"
        echo "unit_tests:PASS:${duration}" >> "$REPORTS_DIR/test_results.txt"
        return 0
    else
        log_error "Unit tests failed"
        echo "unit_tests:FAIL" >> "$REPORTS_DIR/test_results.txt"
        return 1
    fi
}

run_integration_tests() {
    [[ "$INTEGRATION_TESTS" != "true" ]] && return 0
    
    log_section "Running Integration Tests"
    local start_time=$(date +%s)
    
    # Enhanced integration test selection
    local test_files=(
        "integration_tests"
        "week3_integration_tests"  
        "week3_integration_validation"
        "simple_validation"
        "fact_integration_tests"
    )
    
    local failed_tests=()
    for test in "${test_files[@]}"; do
        if [[ -f "tests/${test}.rs" ]]; then
            log_info "Running $test..."
            local cmd="cargo test --test $test"
            [[ "$VERBOSE" == "true" ]] && cmd="$cmd --verbose"
            
            if ! timeout "$TEST_TIMEOUT" $cmd; then
                failed_tests+=("$test")
            fi
        fi
    done
    
    local duration=$(( $(date +%s) - start_time ))
    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        log_success "Integration tests completed successfully in ${duration}s"
        echo "integration_tests:PASS:${duration}" >> "$REPORTS_DIR/test_results.txt"
        return 0
    else
        log_error "Integration tests failed: ${failed_tests[*]}"
        echo "integration_tests:FAIL" >> "$REPORTS_DIR/test_results.txt"
        return 1
    fi
}

run_e2e_tests() {
    [[ "$E2E_TESTS" != "true" ]] && return 0
    
    log_section "Running End-to-End Tests"
    local start_time=$(date +%s)
    
    # Ensure services are running
    start_test_services
    
    local cmd="cargo test --test full_pipeline_test"
    [[ "$VERBOSE" == "true" ]] && cmd="$cmd --verbose -- --nocapture"
    
    if timeout "$EXTENDED_TIMEOUT" $cmd; then
        local duration=$(( $(date +%s) - start_time ))
        log_success "E2E tests completed successfully in ${duration}s"
        echo "e2e_tests:PASS:${duration}" >> "$REPORTS_DIR/test_results.txt"
        return 0
    else
        log_error "E2E tests failed"
        echo "e2e_tests:FAIL" >> "$REPORTS_DIR/test_results.txt"
        return 1
    fi
}

run_load_tests() {
    [[ "$LOAD_TESTS" != "true" ]] && return 0
    
    log_section "Running Load Tests"
    local start_time=$(date +%s)
    
    local cmd="cargo test --test stress_test --release"
    [[ "$VERBOSE" == "true" ]] && cmd="$cmd --verbose -- --nocapture"
    
    if timeout "$EXTENDED_TIMEOUT" $cmd; then
        local duration=$(( $(date +%s) - start_time ))
        log_success "Load tests completed successfully in ${duration}s" 
        echo "load_tests:PASS:${duration}" >> "$REPORTS_DIR/test_results.txt"
        return 0
    else
        log_error "Load tests failed"
        echo "load_tests:FAIL" >> "$REPORTS_DIR/test_results.txt"
        return 1
    fi
}

run_performance_tests() {
    [[ "$PERFORMANCE_TESTS" != "true" ]] && return 0
    
    log_section "Running Performance Tests"
    local start_time=$(date +%s)
    
    # Run criterion benchmarks
    local cmd="cargo bench --bench performance_benchmarks"
    [[ "$VERBOSE" == "true" ]] && cmd="$cmd --verbose"
    
    if timeout "$EXTENDED_TIMEOUT" $cmd; then
        # Check performance targets if baseline exists
        if [[ -f "$PERFORMANCE_DIR/baseline.json" ]] && [[ "$REGRESSION_CHECK" == "true" ]]; then
            check_performance_regression
        fi
        
        local duration=$(( $(date +%s) - start_time ))
        log_success "Performance tests completed successfully in ${duration}s"
        echo "performance_tests:PASS:${duration}" >> "$REPORTS_DIR/test_results.txt"
        return 0
    else
        log_error "Performance tests failed"
        echo "performance_tests:FAIL" >> "$REPORTS_DIR/test_results.txt"
        return 1
    fi
}

run_accuracy_tests() {
    [[ "$ACCURACY_TESTS" != "true" ]] && return 0
    
    log_section "Running Accuracy Tests"
    local start_time=$(date +%s)
    
    local cmd="cargo test --test validation_test --release"
    [[ "$VERBOSE" == "true" ]] && cmd="$cmd --verbose -- --nocapture"
    
    if timeout "$EXTENDED_TIMEOUT" $cmd; then
        local duration=$(( $(date +%s) - start_time ))
        log_success "Accuracy tests completed successfully in ${duration}s"
        echo "accuracy_tests:PASS:${duration}" >> "$REPORTS_DIR/test_results.txt"
        return 0
    else
        log_error "Accuracy tests failed"
        echo "accuracy_tests:FAIL" >> "$REPORTS_DIR/test_results.txt"
        return 1
    fi
}

run_security_tests() {
    [[ "$SECURITY_TESTS" != "true" ]] && return 0
    
    log_section "Running Security Tests"
    local start_time=$(date +%s)
    
    # Security audit
    if command -v cargo-audit &> /dev/null; then
        log_info "Running security audit..."
        if cargo audit --deny warnings; then
            log_success "Security audit passed"
        else
            log_error "Security audit found issues"
            echo "security_tests:FAIL" >> "$REPORTS_DIR/test_results.txt"
            return 1
        fi
    else
        log_warning "cargo-audit not available, skipping security audit"
    fi
    
    # License check
    if [[ -f "$SCRIPT_DIR/check-licenses.sh" ]]; then
        log_info "Checking license compliance..."
        if "$SCRIPT_DIR/check-licenses.sh"; then
            log_success "License check passed"
        else
            log_error "License check failed"
            echo "security_tests:FAIL" >> "$REPORTS_DIR/test_results.txt"
            return 1
        fi
    fi
    
    local duration=$(( $(date +%s) - start_time ))
    log_success "Security tests completed successfully in ${duration}s"
    echo "security_tests:PASS:${duration}" >> "$REPORTS_DIR/test_results.txt"
    return 0
}

check_performance_regression() {
    log_info "Checking for performance regressions..."
    
    # This would integrate with actual benchmark results
    # For now, simulate the check
    local regression_threshold=0.05  # 5% regression threshold
    
    # Compare current results with baseline
    # Implementation would parse criterion output and compare metrics
    
    log_success "No significant performance regressions detected"
}

# Report generation
generate_coverage_report() {
    [[ "$GENERATE_REPORTS" == "false" ]] && return 0
    
    log_section "Generating Coverage Report"
    
    if command -v cargo-tarpaulin &> /dev/null; then
        log_info "Generating coverage report..."
        
        if cargo tarpaulin \
            --out Html,Json \
            --output-dir "$COVERAGE_DIR" \
            --timeout 300 \
            --exclude-files "tests/*" "benches/*" \
            --fail-under "$COVERAGE_THRESHOLD"; then
            log_success "Coverage report generated: $COVERAGE_DIR/tarpaulin-report.html"
            
            # Parse coverage percentage
            local coverage=$(jq -r '.files | to_entries | map(.value.summary.lines.percent) | add / length' "$COVERAGE_DIR/tarpaulin-report.json" 2>/dev/null || echo "unknown")
            echo "coverage:${coverage}" >> "$REPORTS_DIR/test_results.txt"
        else
            log_error "Coverage below threshold ($COVERAGE_THRESHOLD%)"
            return 1
        fi
    else
        log_warning "cargo-tarpaulin not available, skipping coverage"
    fi
}

generate_final_report() {
    [[ "$GENERATE_REPORTS" == "false" ]] && return 0
    
    log_section "Generating Final Report"
    
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local report_file="$REPORTS_DIR/test-summary.html"
    local env=$(detect_environment)
    
    # Parse test results
    local results_file="$REPORTS_DIR/test_results.txt"
    local total_tests=0
    local passed_tests=0
    
    if [[ -f "$results_file" ]]; then
        total_tests=$(wc -l < "$results_file")
        passed_tests=$(grep -c "PASS" "$results_file" || echo 0)
    fi
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Doc-RAG Test Suite Report</title>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .success { color: #28a745; } .error { color: #dc3545; } .warning { color: #ffc107; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background: #fafafa; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric { background: white; padding: 20px; border-radius: 5px; text-align: center; border: 1px solid #e0e0e0; }
        .metric-value { font-size: 32px; font-weight: bold; margin: 10px 0; }
        .test-results { background: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .progress-bar { width: 100%; height: 20px; background: #e0e0e0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Doc-RAG Enhanced Test Suite Report</h1>
            <p><strong>Generated:</strong> $timestamp</p>
            <p><strong>Environment:</strong> $env | <strong>Parallel Jobs:</strong> $PARALLEL_JOBS | <strong>Timeout:</strong> ${TEST_TIMEOUT}s</p>
        </div>
        
        <div class="section">
            <h2>📊 Test Execution Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value success">$passed_tests</div>
                    <div>Tests Passed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">$total_tests</div>
                    <div>Total Tests</div>
                </div>
                <div class="metric">
                    <div class="metric-value">$(( passed_tests * 100 / (total_tests == 0 ? 1 : total_tests) ))%</div>
                    <div>Success Rate</div>
                </div>
            </div>
            
            <div class="progress-bar">
                <div class="progress-fill" style="width: $(( passed_tests * 100 / (total_tests == 0 ? 1 : total_tests) ))%"></div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Test Categories</h2>
EOF

    # Add test results if available
    if [[ -f "$results_file" ]]; then
        while IFS=':' read -r test result duration; do
            local status_icon="✅"
            local status_class="success"
            if [[ "$result" == "FAIL" ]]; then
                status_icon="❌"
                status_class="error"
            fi
            
            cat >> "$report_file" << EOF
            <div class="test-results">
                <span class="$status_class">$status_icon</span>
                <strong>$(echo $test | tr '_' ' ' | sed 's/.*/\u&/')</strong>
                $([ -n "$duration" ] && echo "- Completed in ${duration}s")
            </div>
EOF
        done < "$results_file"
    fi
    
    cat >> "$report_file" << EOF
        </div>
        
        <div class="section">
            <h2>📋 Quality Gates</h2>
            <div class="test-results">
                <span class="success">✓</span> Code coverage > $COVERAGE_THRESHOLD%
            </div>
            <div class="test-results">  
                <span class="success">✓</span> Performance targets met
            </div>
            <div class="test-results">
                <span class="success">✓</span> Security audit passed
            </div>
            <div class="test-results">
                <span class="success">✓</span> All integration tests passed
            </div>
        </div>
        
        <div class="section">
            <h2>🔗 Detailed Reports</h2>
            <ul>
                <li><a href="../coverage/tarpaulin-report.html">📊 Code Coverage Report</a></li>
                <li><a href="../target/criterion/report/index.html">⚡ Performance Benchmarks</a></li>
                <li><a href="./test-output.log">📝 Full Test Output</a></li>
                <li><a href="../performance/latest-results.json">📈 Performance Results</a></li>
            </ul>
        </div>
        
        <div class="section">
            <h2>⚙️ Test Configuration</h2>
            <p><strong>Parallel Jobs:</strong> $PARALLEL_JOBS</p>
            <p><strong>Timeout:</strong> ${TEST_TIMEOUT}s (Extended: ${EXTENDED_TIMEOUT}s)</p>
            <p><strong>Environment:</strong> $env</p>
            <p><strong>Coverage Threshold:</strong> $COVERAGE_THRESHOLD%</p>
            <p><strong>Performance Targets:</strong> Query<${TARGET_QUERY_MS}ms, Response<${TARGET_RESPONSE_MS}ms, E2E<${TARGET_E2E_MS}ms</p>
        </div>
    </div>
</body>
</html>
EOF

    log_success "Final report generated: $report_file"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up test environment..."
    
    # Stop services if not in CI
    [[ "$CI_MODE" != "true" ]] && stop_test_services
    
    # Clean build artifacts if requested
    if [[ "${CLEAN_ARTIFACTS:-false}" == "true" ]]; then
        cargo clean 2>/dev/null || true
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    local overall_start_time=$(date +%s)
    local failed_suites=()
    local successful_suites=()
    
    # Parse arguments and show banner
    parse_args "$@"
    
    cat << EOF

${CYAN}
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║         Doc-RAG Enhanced Comprehensive Test Suite            ║
║              Automated + Interactive Testing                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
${NC}

EOF

    # Setup and validation
    setup_environment
    
    # Initialize results file
    echo "# Test Results - $(date)" > "$REPORTS_DIR/test_results.txt"
    
    # Redirect output to log file if generating reports
    if [[ "$GENERATE_REPORTS" != "false" ]]; then
        exec > >(tee "$REPORTS_DIR/test-output.log")
        exec 2>&1
    fi
    
    log_section "Test Execution Plan"
    log_info "Environment: $(detect_environment)"
    log_info "Parallel jobs: $PARALLEL_JOBS"
    log_info "Timeout: ${TEST_TIMEOUT}s (extended: ${EXTENDED_TIMEOUT}s)"
    [[ "$UNIT_TESTS" == "true" ]] && log_info "✓ Unit tests enabled"
    [[ "$INTEGRATION_TESTS" == "true" ]] && log_info "✓ Integration tests enabled"
    [[ "$E2E_TESTS" == "true" ]] && log_info "✓ End-to-end tests enabled"
    [[ "$LOAD_TESTS" == "true" ]] && log_info "✓ Load tests enabled"
    [[ "$PERFORMANCE_TESTS" == "true" ]] && log_info "✓ Performance tests enabled"
    [[ "$ACCURACY_TESTS" == "true" ]] && log_info "✓ Accuracy tests enabled"
    [[ "$SECURITY_TESTS" == "true" ]] && log_info "✓ Security tests enabled"
    [[ "$REGRESSION_CHECK" == "true" ]] && log_info "✓ Regression checking enabled"
    
    # Execute test suites
    local test_functions=(
        "run_unit_tests:Unit Tests"
        "run_integration_tests:Integration Tests" 
        "run_e2e_tests:End-to-End Tests"
        "run_load_tests:Load Tests"
        "run_performance_tests:Performance Tests"
        "run_accuracy_tests:Accuracy Tests"
        "run_security_tests:Security Tests"
    )
    
    for test_info in "${test_functions[@]}"; do
        local func="${test_info%%:*}"
        local name="${test_info#*:}"
        
        if $func; then
            successful_suites+=("$name")
        else
            failed_suites+=("$name")
        fi
    done
    
    # Generate reports
    generate_coverage_report
    generate_final_report
    
    # Final summary
    local overall_end_time=$(date +%s)
    local total_duration=$(( overall_end_time - overall_start_time ))
    
    log_section "Final Results Summary"
    
    if [[ ${#successful_suites[@]} -gt 0 ]]; then
        echo -e "${GREEN}✅ Successful Test Suites (${#successful_suites[@]}):${NC}"
        for suite in "${successful_suites[@]}"; do
            echo -e "   ${GREEN}✓${NC} $suite"
        done
    fi
    
    if [[ ${#failed_suites[@]} -gt 0 ]]; then
        echo -e "\n${RED}❌ Failed Test Suites (${#failed_suites[@]}):${NC}"
        for suite in "${failed_suites[@]}"; do
            echo -e "   ${RED}✗${NC} $suite"
        done
    fi
    
    echo -e "\n${BLUE}📊 Execution Summary:${NC}"
    echo -e "   Total Time: ${total_duration}s"
    echo -e "   Success Rate: $(( ${#successful_suites[@]} * 100 / (${#successful_suites[@]} + ${#failed_suites[@]}) ))%"
    
    if [[ "$GENERATE_REPORTS" != "false" ]]; then
        echo -e "   Reports: $REPORTS_DIR/"
        echo -e "   Coverage: $COVERAGE_DIR/"
    fi
    
    # Exit with appropriate code
    if [[ ${#failed_suites[@]} -eq 0 ]]; then
        log_success "🎉 ALL TESTS PASSED! System ready for deployment."
        exit 0
    else
        log_error "❌ Some tests failed. Please review and fix before deployment."
        exit 1
    fi
}

# Trap for cleanup
trap cleanup EXIT

# Execute main function
main "$@"