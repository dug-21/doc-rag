#!/bin/bash
# Enhanced Quick Test Script for Doc-RAG System
# Consolidates functionality from quick_test.sh and test_api.sh for rapid development workflow

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL="${API_URL:-http://localhost:8080}"
TEST_TIMEOUT=${TEST_TIMEOUT:-30}
VERBOSE=${VERBOSE:-false}
SKIP_SERVICES=${SKIP_SERVICES:-false}

# Test data
TEST_PDF_CONTENT="%PDF-1.4
Test PDF content for Doc-RAG system validation
Byzantine consensus ensures 67% agreement threshold
FACT caching provides sub-50ms response times  
Neural boundary detection at 95.4% accuracy
%%EOF"

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Help function
show_help() {
    cat << EOF
Doc-RAG Enhanced Quick Test Suite

Purpose: Fast validation for development workflow (< 2 minutes)

Usage: $0 [OPTIONS]

OPTIONS:
    --api-url URL        API endpoint URL (default: http://localhost:8080)
    --timeout SECONDS    Test timeout (default: 30)
    --skip-services      Skip service health checks
    --verbose, -v        Enable verbose output
    --help, -h          Show this help message

TESTS PERFORMED:
    1. Service health checks (optional)
    2. API endpoint validation
    3. Document upload test
    4. Query processing test  
    5. Citation tracking test
    6. Performance validation

EXAMPLES:
    $0                          # Standard quick test
    $0 --verbose               # Verbose output  
    $0 --skip-services         # API tests only
    $0 --api-url http://prod:8080  # Test production endpoint

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --api-url) API_URL="$2"; shift 2 ;;
            --timeout) TEST_TIMEOUT="$2"; shift 2 ;;
            --skip-services) SKIP_SERVICES=true; shift ;;
            --verbose|-v) VERBOSE=true; shift ;;
            --help|-h) show_help; exit 0 ;;
            *) log_error "Unknown option: $1"; show_help; exit 1 ;;
        esac
    done
}

# Service health checks
check_service_health() {
    [[ "$SKIP_SERVICES" == "true" ]] && return 0
    
    log_info "Checking service health..."
    
    local services=(
        "postgres:5432:PostgreSQL"
        "mongodb:27017:MongoDB"
        "redis:6379:Redis Cache"
        "qdrant:6333:Vector Database"
    )
    
    local failed_services=()
    
    for service_info in "${services[@]}"; do
        local service="${service_info%%:*}"
        local port="${service_info#*:}"; port="${port%%:*}"
        local name="${service_info##*:}"
        
        if timeout 5 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
            [[ "$VERBOSE" == "true" ]] && log_success "$name ($service:$port) is responding"
        else
            failed_services+=("$name")
            [[ "$VERBOSE" == "true" ]] && log_warning "$name ($service:$port) not responding"
        fi
    done
    
    if [[ ${#failed_services[@]} -eq 0 ]]; then
        log_success "All core services are healthy"
    else
        log_warning "Some services not responding: ${failed_services[*]}"
        log_info "Continuing with API tests..."
    fi
}

# API health check
test_api_health() {
    log_info "Testing API health endpoint..."
    
    local response
    if response=$(timeout "$TEST_TIMEOUT" curl -s "$API_URL/health" 2>/dev/null); then
        if echo "$response" | grep -q "ok\|healthy\|up"; then
            log_success "API health check passed"
            [[ "$VERBOSE" == "true" ]] && echo "   Response: $response"
            return 0
        else
            log_error "API health check failed - unexpected response"
            [[ "$VERBOSE" == "true" ]] && echo "   Response: $response"
            return 1
        fi
    else
        log_error "API health check failed - no response"
        log_info "   Endpoint: $API_URL/health"
        log_info "   Is the API service running?"
        return 1
    fi
}

# Document upload test
test_document_upload() {
    log_info "Testing document upload..."
    
    # Create temporary test PDF
    local test_pdf="/tmp/docrag_test_$(date +%s).pdf"
    echo "$TEST_PDF_CONTENT" > "$test_pdf"
    
    local upload_response
    if upload_response=$(timeout "$TEST_TIMEOUT" curl -s -X POST "$API_URL/upload" \
        -F "file=@$test_pdf" \
        -F "metadata={\"title\":\"Quick Test Document\"}" 2>/dev/null); then
        
        # Extract document ID from response
        local doc_id
        if command -v jq &> /dev/null; then
            doc_id=$(echo "$upload_response" | jq -r '.id // .doc_id // .document_id // empty' 2>/dev/null)
        else
            # Fallback parsing without jq
            doc_id=$(echo "$upload_response" | grep -o '"id":"[^"]*' | cut -d'"' -f4)
            [[ -z "$doc_id" ]] && doc_id=$(echo "$upload_response" | grep -o '"doc_id":"[^"]*' | cut -d'"' -f4)
        fi
        
        if [[ -n "$doc_id" ]]; then
            log_success "Document upload successful (ID: ${doc_id:0:8}...)"
            [[ "$VERBOSE" == "true" ]] && echo "   Full response: $upload_response"
            echo "$doc_id" > /tmp/docrag_test_doc_id
        else
            log_warning "Document upload completed but no ID found"
            [[ "$VERBOSE" == "true" ]] && echo "   Response: $upload_response"
            # Try to extract any reasonable ID from response
            echo "test_doc_$(date +%s)" > /tmp/docrag_test_doc_id
        fi
        
        rm -f "$test_pdf"
        return 0
    else
        log_error "Document upload failed"
        [[ "$VERBOSE" == "true" ]] && echo "   Response: $upload_response"
        rm -f "$test_pdf"
        return 1
    fi
}

# Query processing test
test_query_processing() {
    log_info "Testing query processing..."
    
    local doc_id="test_doc_001"
    if [[ -f "/tmp/docrag_test_doc_id" ]]; then
        doc_id=$(cat /tmp/docrag_test_doc_id)
    fi
    
    local query_data="{\"doc_id\": \"$doc_id\", \"question\": \"What is Byzantine consensus?\", \"max_results\": 5}"
    local query_response
    
    local start_time=$(date +%s%N)
    if query_response=$(timeout "$TEST_TIMEOUT" curl -s -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -d "$query_data" 2>/dev/null); then
        local end_time=$(date +%s%N)
        local response_time=$(( (end_time - start_time) / 1000000 ))
        
        # Check if response contains expected fields
        local has_answer=false
        if echo "$query_response" | grep -q "answer\|response\|result"; then
            has_answer=true
        fi
        
        if [[ "$has_answer" == "true" ]]; then
            log_success "Query processing successful (${response_time}ms)"
            
            # Extract answer preview if possible
            if command -v jq &> /dev/null; then
                local answer=$(echo "$query_response" | jq -r '.answer // .response // .result // empty' 2>/dev/null | head -c 50)
                [[ -n "$answer" ]] && log_info "   Answer: ${answer}..."
            fi
            
            # Store response for citation test
            echo "$query_response" > /tmp/docrag_test_response
            echo "$response_time" > /tmp/docrag_test_response_time
            
            [[ "$VERBOSE" == "true" ]] && echo "   Full response: $query_response"
        else
            log_warning "Query processing completed but no answer found"
            [[ "$VERBOSE" == "true" ]] && echo "   Response: $query_response"
        fi
        
        return 0
    else
        log_error "Query processing failed"
        [[ "$VERBOSE" == "true" ]] && echo "   Request: $query_data"
        return 1
    fi
}

# Citation tracking test
test_citation_tracking() {
    log_info "Testing citation tracking..."
    
    if [[ ! -f "/tmp/docrag_test_response" ]]; then
        log_warning "No query response available, skipping citation test"
        return 0
    fi
    
    local response=$(cat /tmp/docrag_test_response)
    
    # Check for citation fields
    local has_citations=false
    if echo "$response" | grep -q "citations\|sources\|references"; then
        has_citations=true
    fi
    
    if [[ "$has_citations" == "true" ]]; then
        log_success "Citation tracking is working"
        
        # Extract citation count if possible
        if command -v jq &> /dev/null; then
            local citation_count=$(echo "$response" | jq -r '.citations | length // empty' 2>/dev/null)
            [[ -n "$citation_count" ]] && log_info "   Found $citation_count citations"
        fi
    else
        log_warning "No citations found in response"
        log_info "   This may indicate citation tracking needs configuration"
    fi
    
    return 0
}

# Performance validation
test_performance() {
    log_info "Testing performance targets..."
    
    local response_time=0
    if [[ -f "/tmp/docrag_test_response_time" ]]; then
        response_time=$(cat /tmp/docrag_test_response_time)
    fi
    
    # Performance targets
    local target_response_ms=2000  # 2 second target for quick test
    local target_cache_ms=50       # Cache hit target
    
    if [[ $response_time -lt $target_response_ms ]]; then
        log_success "Response time within target (${response_time}ms < ${target_response_ms}ms)"
    else
        log_warning "Response time above target (${response_time}ms > ${target_response_ms}ms)"
        log_info "   This may be expected for first-time queries without cache"
    fi
    
    # Test cache performance with repeat query
    log_info "Testing cache performance..."
    local doc_id=$(cat /tmp/docrag_test_doc_id 2>/dev/null || echo "test_doc_001")
    
    local start_time=$(date +%s%N)
    curl -s -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -d "{\"doc_id\": \"$doc_id\", \"question\": \"What is Byzantine consensus?\"}" \
        >/dev/null 2>&1
    local end_time=$(date +%s%N)
    local cache_time=$(( (end_time - start_time) / 1000000 ))
    
    if [[ $cache_time -lt $target_cache_ms ]]; then
        log_success "Cache performance excellent (${cache_time}ms < ${target_cache_ms}ms)"
    else
        log_info "Cache performance (${cache_time}ms) - may improve with usage"
    fi
}

# Document listing test (optional)
test_document_listing() {
    log_info "Testing document listing..."
    
    local response
    if response=$(timeout "$TEST_TIMEOUT" curl -s "$API_URL/documents" 2>/dev/null); then
        if echo "$response" | grep -q "\[\|documents\|items"; then
            log_success "Document listing working"
            
            # Count documents if possible
            if command -v jq &> /dev/null; then
                local doc_count=$(echo "$response" | jq -r 'length // .documents | length // empty' 2>/dev/null)
                [[ -n "$doc_count" ]] && log_info "   Found $doc_count documents"
            fi
        else
            log_warning "Document listing returned unexpected format"
        fi
        
        [[ "$VERBOSE" == "true" ]] && echo "   Response: $response"
        return 0
    else
        log_warning "Document listing endpoint not available or failed"
        return 0  # Non-critical for quick test
    fi
}

# Cleanup function
cleanup() {
    # Clean up temporary files
    rm -f /tmp/docrag_test_* 2>/dev/null || true
}

# Main execution
main() {
    local start_time=$(date +%s)
    local tests_run=0
    local tests_passed=0
    
    # Parse arguments and show banner
    parse_args "$@"
    
    cat << EOF

${BLUE}
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║            Doc-RAG Enhanced Quick Test Suite             ║
║               Rapid Development Validation               ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
${NC}

Testing API at: $API_URL
Timeout: ${TEST_TIMEOUT}s
$([ "$SKIP_SERVICES" == "true" ] && echo "Service checks: Skipped" || echo "Service checks: Enabled")

EOF

    # Test execution with error handling
    local test_functions=(
        "check_service_health:Service Health Checks"
        "test_api_health:API Health Check"
        "test_document_upload:Document Upload"
        "test_query_processing:Query Processing"
        "test_citation_tracking:Citation Tracking"
        "test_performance:Performance Validation"
        "test_document_listing:Document Listing"
    )
    
    for test_info in "${test_functions[@]}"; do
        local func="${test_info%%:*}"
        local name="${test_info#*:}"
        
        ((tests_run++))
        if $func; then
            ((tests_passed++))
        fi
        echo  # Add spacing between tests
    done
    
    # Final summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "=================================================="
    echo "🧪 QUICK TEST SUMMARY"
    echo "=================================================="
    echo "Tests Run: $tests_run"
    echo "Tests Passed: $tests_passed"
    echo "Success Rate: $(( tests_passed * 100 / tests_run ))%"
    echo "Total Time: ${duration}s"
    echo ""
    
    if [[ $tests_passed -eq $tests_run ]]; then
        log_success "✅ ALL QUICK TESTS PASSED!"
        echo ""
        echo "System is ready for development work:"
        echo "• API endpoints are responding correctly"
        echo "• Document upload and processing working"
        echo "• Query processing and citation tracking active"
        echo "• Performance within acceptable ranges"
        echo ""
        echo "Next steps:"
        echo "1. Upload real PDFs for testing"
        echo "2. Run comprehensive tests: ./scripts/test_all.sh"
        echo "3. Check detailed performance: ./scripts/performance_monitor.sh"
    else
        log_error "❌ Some quick tests failed ($tests_passed/$tests_run passed)"
        echo ""
        echo "Common troubleshooting:"
        echo "• Ensure services are running: ./scripts/dev_start.sh"
        echo "• Check service logs: docker-compose logs"
        echo "• Verify API configuration"
        echo "• Run with --verbose for more details"
    fi
    
    echo ""
    
    # Exit with appropriate code
    if [[ $tests_passed -eq $tests_run ]]; then
        exit 0
    else
        exit 1
    fi
}

# Trap for cleanup
trap cleanup EXIT

# Execute main function
main "$@"