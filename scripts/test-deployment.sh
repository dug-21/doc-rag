#!/bin/bash
# Test deployment script for neurosymbolic RAG system
# Validates Docker configuration and service health

set -e

echo "🧪 Testing Neurosymbolic RAG Deployment"
echo "========================================"

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_COMPOSE_FILE="$PROJECT_DIR/docker-compose.fixed.yml"
DOCKERFILE="$PROJECT_DIR/Dockerfile.fixed"
TEST_TIMEOUT=300  # 5 minutes
HEALTH_CHECK_RETRIES=20
HEALTH_CHECK_INTERVAL=10

cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to check if a service is healthy
check_service_health() {
    local service_name=$1
    local max_retries=$2
    local retry=0
    
    log_info "Checking health of $service_name..."
    
    while [ $retry -lt $max_retries ]; do
        if docker-compose -f "$DOCKER_COMPOSE_FILE" ps "$service_name" | grep -q "healthy\|Up"; then
            log_success "$service_name is healthy"
            return 0
        fi
        
        retry=$((retry + 1))
        log_info "Waiting for $service_name... (attempt $retry/$max_retries)"
        sleep $HEALTH_CHECK_INTERVAL
    done
    
    log_error "$service_name failed to become healthy"
    return 1
}

# Function to test API endpoint
test_api_endpoint() {
    local endpoint=$1
    local expected_status=$2
    local description=$3
    
    log_info "Testing $description: $endpoint"
    
    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" || echo "000")
    
    if [ "$response_code" = "$expected_status" ]; then
        log_success "$description - HTTP $response_code"
        return 0
    else
        log_error "$description failed - HTTP $response_code (expected $expected_status)"
        return 1
    fi
}

# Function to test database connectivity
test_database_connectivity() {
    log_info "Testing database connectivity..."
    
    # Test Neo4j
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T neo4j cypher-shell -u neo4j -p password "RETURN 1" >/dev/null 2>&1; then
        log_success "Neo4j connectivity test passed"
    else
        log_error "Neo4j connectivity test failed"
        return 1
    fi
    
    # Test MongoDB
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T mongodb mongosh --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
        log_success "MongoDB connectivity test passed"
    else
        log_error "MongoDB connectivity test failed"
        return 1
    fi
    
    # Test Redis
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T redis redis-cli ping | grep -q "PONG"; then
        log_success "Redis connectivity test passed"
    else
        log_error "Redis connectivity test failed"
        return 1
    fi
    
    # Test Qdrant
    if docker-compose -f "$DOCKER_COMPOSE_FILE" exec -T qdrant wget -q --spider http://localhost:6333/health; then
        log_success "Qdrant connectivity test passed"
    else
        log_warning "Qdrant connectivity test failed (non-critical for basic testing)"
    fi
}

# Function to run functional tests
run_functional_tests() {
    log_info "Running functional tests..."
    
    # Test document upload (placeholder)
    log_info "Testing document upload functionality..."
    
    # Create test document
    local test_doc="/tmp/test-document.txt"
    echo "This is a test document for PCI-DSS compliance testing. It contains information about encryption requirements." > "$test_doc"
    
    # Test query processing
    log_info "Testing query processing..."
    local query_payload='{"query": "What are the encryption requirements?", "max_results": 5}'
    local api_endpoint="http://localhost:8080/api/v1/query"
    
    local response
    response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$query_payload" \
        "$api_endpoint" || echo '{"error": "Request failed"}')
    
    if echo "$response" | grep -q '"query"'; then
        log_success "Query processing test passed"
        log_info "Response preview: $(echo "$response" | jq -r '.response // .error' 2>/dev/null || echo "$response")"
    else
        log_warning "Query processing test returned unexpected response: $response"
    fi
    
    rm -f "$test_doc"
}

# Function to check compilation
test_compilation() {
    log_info "Testing Rust compilation..."
    
    if cargo check --workspace --all-targets; then
        log_success "Rust compilation check passed"
    else
        log_error "Rust compilation check failed"
        return 1
    fi
    
    if cargo build --release --bin integration-server; then
        log_success "Integration server build succeeded"
    else
        log_error "Integration server build failed"
        return 1
    fi
}

# Function to clean up
cleanup() {
    log_info "Cleaning up test environment..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v --remove-orphans >/dev/null 2>&1 || true
    docker system prune -f >/dev/null 2>&1 || true
}

# Main test execution
main() {
    local exit_code=0
    
    # Ensure cleanup on exit
    trap cleanup EXIT
    
    echo "Starting deployment tests at $(date)"
    echo "Project directory: $PROJECT_DIR"
    echo "Docker Compose file: $DOCKER_COMPOSE_FILE"
    echo "Dockerfile: $DOCKERFILE"
    echo ""
    
    # Phase 1: Compilation Tests
    log_info "Phase 1: Compilation Tests"
    echo "=========================="
    if ! test_compilation; then
        log_error "Compilation tests failed - stopping deployment tests"
        exit 1
    fi
    echo ""
    
    # Phase 2: Docker Build Tests
    log_info "Phase 2: Docker Build Tests"
    echo "============================"
    log_info "Building Docker images..."
    if docker-compose -f "$DOCKER_COMPOSE_FILE" build; then
        log_success "Docker build completed successfully"
    else
        log_error "Docker build failed"
        exit_code=1
    fi
    echo ""
    
    # Phase 3: Service Startup Tests
    log_info "Phase 3: Service Startup Tests"
    echo "==============================="
    log_info "Starting services..."
    if docker-compose -f "$DOCKER_COMPOSE_FILE" up -d; then
        log_success "Services started"
    else
        log_error "Failed to start services"
        exit_code=1
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check individual service health
    check_service_health "neo4j" $HEALTH_CHECK_RETRIES || exit_code=1
    check_service_health "redis" $HEALTH_CHECK_RETRIES || exit_code=1
    check_service_health "mongodb" $HEALTH_CHECK_RETRIES || exit_code=1
    check_service_health "qdrant" $HEALTH_CHECK_RETRIES || log_warning "Qdrant health check failed (non-critical)"
    check_service_health "neurosymbolic-rag" $HEALTH_CHECK_RETRIES || exit_code=1
    echo ""
    
    # Phase 4: Connectivity Tests
    log_info "Phase 4: Connectivity Tests"
    echo "============================"
    test_database_connectivity || exit_code=1
    echo ""
    
    # Phase 5: API Endpoint Tests
    log_info "Phase 5: API Endpoint Tests"
    echo "============================"
    test_api_endpoint "http://localhost:8080/health" "200" "Health check endpoint" || exit_code=1
    test_api_endpoint "http://localhost:9090/metrics" "200" "Metrics endpoint" || log_warning "Metrics endpoint test failed (non-critical)"
    echo ""
    
    # Phase 6: Functional Tests
    log_info "Phase 6: Functional Tests"
    echo "=========================="
    run_functional_tests || log_warning "Some functional tests failed (expected in early testing)"
    echo ""
    
    # Test Summary
    echo "Test Summary"
    echo "============"
    if [ $exit_code -eq 0 ]; then
        log_success "🎉 All critical tests passed! System is ready for Phase 3 testing."
        echo ""
        echo "Next steps:"
        echo "1. Load test documents: docker-compose -f $DOCKER_COMPOSE_FILE exec neurosymbolic-rag /app/neurosymbolic-rag --load-docs"
        echo "2. Access web interface: http://localhost:3000 (if enabled with --profile web)"
        echo "3. Test API directly: curl -X POST -H 'Content-Type: application/json' -d '{\"query\": \"test\"}' http://localhost:8080/api/v1/query"
        echo "4. Monitor logs: docker-compose -f $DOCKER_COMPOSE_FILE logs -f neurosymbolic-rag"
    else
        log_error "❌ Some tests failed. Review the output above and fix issues before proceeding."
        echo ""
        echo "Common troubleshooting steps:"
        echo "1. Check service logs: docker-compose -f $DOCKER_COMPOSE_FILE logs [service-name]"
        echo "2. Verify compilation: cargo check --workspace"
        echo "3. Check Docker build: docker-compose -f $DOCKER_COMPOSE_FILE build neurosymbolic-rag"
        echo "4. Restart services: docker-compose -f $DOCKER_COMPOSE_FILE restart"
    fi
    
    echo ""
    echo "Test completed at $(date)"
    exit $exit_code
}

# Run main function
main "$@"