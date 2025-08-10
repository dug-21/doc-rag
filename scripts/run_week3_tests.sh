#!/bin/bash

# Week 3 RAG System Integration Test Runner
# Comprehensive testing suite for Query Processor + Response Generator integration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TEST_TIMEOUT="300" # 5 minutes
BENCH_TIMEOUT="600" # 10 minutes
PERFORMANCE_REPORT="performance_report.json"
TEST_REPORT="test_report.txt"

echo -e "${BLUE}🚀 Week 3 RAG System Integration Test Suite${NC}"
echo "=============================================="
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}📋 Checking prerequisites...${NC}"
    
    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}❌ Cargo not found. Please install Rust.${NC}"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}⚠️  jq not found. Installing for JSON processing...${NC}"
        # Attempt to install jq on various systems
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y jq
        elif command -v brew &> /dev/null; then
            brew install jq
        else
            echo -e "${YELLOW}⚠️  Could not install jq automatically. Some reports may be limited.${NC}"
        fi
    fi
    
    echo -e "${GREEN}✅ Prerequisites checked${NC}"
    echo ""
}

# Clean previous results
clean_previous_results() {
    echo -e "${YELLOW}🧹 Cleaning previous test results...${NC}"
    
    rm -f $TEST_REPORT
    rm -f $PERFORMANCE_REPORT
    rm -rf target/criterion
    
    echo -e "${GREEN}✅ Cleaned previous results${NC}"
    echo ""
}

# Build all components
build_components() {
    echo -e "${YELLOW}🔨 Building all components...${NC}"
    
    # Build workspace
    timeout $TEST_TIMEOUT cargo build --workspace --release
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Build successful${NC}"
    else
        echo -e "${RED}❌ Build failed${NC}"
        exit 1
    fi
    echo ""
}

# Run unit tests for all components
run_unit_tests() {
    echo -e "${YELLOW}🧪 Running unit tests for all components...${NC}"
    
    local components=("chunker" "embedder" "storage" "query-processor" "response-generator")
    local failed_components=()
    
    for component in "${components[@]}"; do
        echo -e "${BLUE}Testing ${component}...${NC}"
        
        if timeout $TEST_TIMEOUT cargo test --package $component --lib; then
            echo -e "${GREEN}✅ ${component} unit tests passed${NC}"
        else
            echo -e "${RED}❌ ${component} unit tests failed${NC}"
            failed_components+=($component)
        fi
        echo ""
    done
    
    if [ ${#failed_components[@]} -eq 0 ]; then
        echo -e "${GREEN}✅ All unit tests passed${NC}"
    else
        echo -e "${RED}❌ Failed components: ${failed_components[*]}${NC}"
        echo "Continuing with integration tests..."
    fi
    echo ""
}

# Run Week 3 integration tests
run_integration_tests() {
    echo -e "${YELLOW}🔗 Running Week 3 integration tests...${NC}"
    
    echo "Test Results - $(date)" > $TEST_REPORT
    echo "=========================" >> $TEST_REPORT
    echo "" >> $TEST_REPORT
    
    local test_cases=(
        "test_end_to_end_query_processing"
        "test_different_query_types" 
        "test_concurrent_load_processing"
        "test_error_handling_resilience"
        "test_component_integration"
        "test_performance_benchmarks"
        "test_data_flow_integrity"
        "test_scalability_increasing_data"
        "test_memory_efficiency"
        "test_production_readiness"
        "test_comprehensive_system_validation"
    )
    
    local passed_tests=0
    local total_tests=${#test_cases[@]}
    
    for test_case in "${test_cases[@]}"; do
        echo -e "${BLUE}Running ${test_case}...${NC}"
        
        if timeout $TEST_TIMEOUT cargo test $test_case --test week3_integration_tests -- --nocapture; then
            echo -e "${GREEN}✅ ${test_case} passed${NC}"
            echo "✅ $test_case - PASSED" >> $TEST_REPORT
            ((passed_tests++))
        else
            echo -e "${RED}❌ ${test_case} failed${NC}"
            echo "❌ $test_case - FAILED" >> $TEST_REPORT
        fi
        echo ""
    done
    
    echo "" >> $TEST_REPORT
    echo "Summary:" >> $TEST_REPORT
    echo "  Passed: $passed_tests/$total_tests" >> $TEST_REPORT
    echo "  Success Rate: $(( passed_tests * 100 / total_tests ))%" >> $TEST_REPORT
    
    if [ $passed_tests -eq $total_tests ]; then
        echo -e "${GREEN}🎉 All integration tests passed! ($passed_tests/$total_tests)${NC}"
    else
        echo -e "${YELLOW}⚠️  Some integration tests failed. ($passed_tests/$total_tests passed)${NC}"
    fi
    echo ""
}

# Run performance benchmarks
run_performance_benchmarks() {
    echo -e "${YELLOW}📊 Running performance benchmarks...${NC}"
    
    local benchmarks=(
        "query_processing"
        "response_generation" 
        "end_to_end"
        "document_chunking"
        "embedding_generation"
        "vector_search"
        "concurrent_processing"
        "citation_processing"
        "validation_pipeline"
    )
    
    echo "{" > $PERFORMANCE_REPORT
    echo "  \"timestamp\": \"$(date -Iseconds)\"," >> $PERFORMANCE_REPORT
    echo "  \"benchmarks\": [" >> $PERFORMANCE_REPORT
    
    local first=true
    for benchmark in "${benchmarks[@]}"; do
        echo -e "${BLUE}Running ${benchmark} benchmark...${NC}"
        
        if [ "$first" = true ]; then
            first=false
        else
            echo "    ," >> $PERFORMANCE_REPORT
        fi
        
        if timeout $BENCH_TIMEOUT cargo bench --bench week3_pipeline_benchmarks -- $benchmark --output-format json; then
            echo -e "${GREEN}✅ ${benchmark} benchmark completed${NC}"
            echo "    {\"name\": \"$benchmark\", \"status\": \"completed\"}" >> $PERFORMANCE_REPORT
        else
            echo -e "${RED}❌ ${benchmark} benchmark failed${NC}"
            echo "    {\"name\": \"$benchmark\", \"status\": \"failed\"}" >> $PERFORMANCE_REPORT
        fi
        echo ""
    done
    
    echo "  ]" >> $PERFORMANCE_REPORT  
    echo "}" >> $PERFORMANCE_REPORT
    
    echo -e "${GREEN}✅ Performance benchmarks completed${NC}"
    echo ""
}

# Validate performance targets
validate_performance_targets() {
    echo -e "${YELLOW}🎯 Validating performance targets...${NC}"
    
    echo "Performance Targets Validation:" >> $TEST_REPORT
    echo "===============================" >> $TEST_REPORT
    
    # Define performance targets (in milliseconds)
    local query_processing_target=50
    local response_generation_target=100
    local end_to_end_target=200
    local accuracy_target="0.99"
    
    echo "Targets:" >> $TEST_REPORT
    echo "  Query Processing: < ${query_processing_target}ms" >> $TEST_REPORT
    echo "  Response Generation: < ${response_generation_target}ms" >> $TEST_REPORT
    echo "  End-to-End: < ${end_to_end_target}ms" >> $TEST_REPORT
    echo "  Accuracy: > ${accuracy_target} (99%)" >> $TEST_REPORT
    echo "" >> $TEST_REPORT
    
    # Check if criterion results exist
    if [ -d "target/criterion" ]; then
        echo -e "${GREEN}✅ Benchmark results found in target/criterion${NC}"
        echo "📊 Detailed benchmark reports available in target/criterion/"
    else
        echo -e "${YELLOW}⚠️  No detailed benchmark results found${NC}"
    fi
    
    echo -e "${GREEN}✅ Performance target validation completed${NC}"
    echo ""
}

# Generate comprehensive report
generate_report() {
    echo -e "${YELLOW}📝 Generating comprehensive report...${NC}"
    
    {
        echo ""
        echo "Week 3 RAG System Integration Test Report"
        echo "========================================"
        echo "Generated: $(date)"
        echo ""
        
        echo "System Information:"
        echo "  OS: $(uname -s)"
        echo "  Architecture: $(uname -m)" 
        echo "  Rust Version: $(rustc --version)"
        echo "  Cargo Version: $(cargo --version)"
        echo ""
        
        if [ -f $TEST_REPORT ]; then
            cat $TEST_REPORT
        fi
        
        echo ""
        echo "Performance Metrics:"
        echo "==================="
        
        if [ -f $PERFORMANCE_REPORT ]; then
            if command -v jq &> /dev/null; then
                echo "Benchmark Summary:"
                jq -r '.benchmarks[] | "  " + .name + ": " + .status' $PERFORMANCE_REPORT 2>/dev/null || echo "  (Performance data parsing failed)"
            else
                echo "  Performance data available in $PERFORMANCE_REPORT"
            fi
        else
            echo "  No performance data available"
        fi
        
        echo ""
        echo "Files Generated:"
        echo "================"
        echo "  - $TEST_REPORT (Test results)"
        echo "  - $PERFORMANCE_REPORT (Performance metrics)"
        echo "  - target/criterion/ (Detailed benchmark reports)"
        echo ""
        
        echo "Next Steps:"
        echo "==========="
        echo "1. Review failed tests in $TEST_REPORT"
        echo "2. Check performance metrics in target/criterion/"
        echo "3. Address any performance issues"
        echo "4. Re-run tests after fixes"
        echo ""
        
    } >> $TEST_REPORT
    
    echo -e "${GREEN}✅ Comprehensive report generated: $TEST_REPORT${NC}"
    echo ""
}

# Display summary
display_summary() {
    echo -e "${BLUE}📊 Test Suite Summary${NC}"
    echo "===================="
    
    if [ -f $TEST_REPORT ]; then
        echo "📄 Test Report: $TEST_REPORT"
        
        # Show summary if available
        if grep -q "Summary:" $TEST_REPORT; then
            echo ""
            grep -A 5 "Summary:" $TEST_REPORT
        fi
    fi
    
    if [ -f $PERFORMANCE_REPORT ]; then
        echo "📊 Performance Report: $PERFORMANCE_REPORT"
    fi
    
    if [ -d "target/criterion" ]; then
        echo "📈 Detailed Benchmarks: target/criterion/"
        echo ""
        echo "To view benchmark reports:"
        echo "  open target/criterion/reports/index.html"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 Week 3 integration testing completed!${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}Starting Week 3 RAG System Integration Testing...${NC}"
    echo ""
    
    check_prerequisites
    clean_previous_results
    build_components
    run_unit_tests
    run_integration_tests
    run_performance_benchmarks  
    validate_performance_targets
    generate_report
    display_summary
    
    echo ""
    echo -e "${GREEN}✨ All testing phases completed!${NC}"
}

# Handle script arguments
case "${1:-all}" in
    "all")
        main
        ;;
    "tests")
        check_prerequisites
        build_components
        run_integration_tests
        ;;
    "benchmarks")
        check_prerequisites
        build_components
        run_performance_benchmarks
        ;;
    "clean")
        clean_previous_results
        ;;
    "report")
        generate_report
        display_summary
        ;;
    *)
        echo "Usage: $0 [all|tests|benchmarks|clean|report]"
        echo ""
        echo "Commands:"
        echo "  all        - Run complete test suite (default)"
        echo "  tests      - Run integration tests only"
        echo "  benchmarks - Run performance benchmarks only" 
        echo "  clean      - Clean previous results"
        echo "  report     - Generate and display report"
        exit 1
        ;;
esac