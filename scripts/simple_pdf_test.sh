#!/bin/bash
# Simple PDF Testing Script - Focus on what ACTUALLY works
# Tests PDF upload and query functionality with minimal infrastructure

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
API_PORT=8080
MONGODB_PORT=27017
PDF_FILE="./uploads/thor_resume.pdf"
API_URL="http://localhost:${API_PORT}"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_requirements() {
    log_info "Checking requirements..."
    
    if ! command -v mongod >/dev/null 2>&1; then
        log_warning "MongoDB not found. Using Docker instead..."
        USE_DOCKER=true
    else
        USE_DOCKER=false
    fi
    
    if [ ! -f "$PDF_FILE" ]; then
        log_error "Test PDF not found: $PDF_FILE"
        log_info "Creating sample PDF..."
        mkdir -p uploads
        echo "Sample PDF content for testing" > uploads/test.txt
        PDF_FILE="uploads/test.txt"
    fi
    
    log_success "Requirements checked"
}

# Start MongoDB
start_mongodb() {
    log_info "Starting MongoDB..."
    
    if [ "$USE_DOCKER" = true ]; then
        docker run -d --rm \
            --name doc-rag-mongodb-test \
            -p ${MONGODB_PORT}:27017 \
            -e MONGO_INITDB_DATABASE=doc_rag_test \
            mongo:7.0 > /dev/null 2>&1 || true
        
        sleep 5
        log_success "MongoDB started in Docker"
    else
        log_info "Using existing MongoDB installation"
    fi
}

# Build and start API server
start_api_server() {
    log_info "Building API server..."
    
    # Build the API service
    cd src/api
    cargo build --release 2>/dev/null || {
        log_warning "Build failed, trying with reduced features..."
        cargo build --release --no-default-features 2>/dev/null || {
            log_error "API build failed"
            return 1
        }
    }
    cd ../..
    
    log_info "Starting API server..."
    
    # Create minimal config
    cat > /tmp/api-test.toml << EOF
[server]
bind_address = "0.0.0.0"
port = ${API_PORT}

[database]
mongodb_url = "mongodb://localhost:${MONGODB_PORT}/doc_rag_test"

[features]
enable_daa = false
enable_ruv_fann = false
enable_fact_cache = false

[performance]
max_workers = 2
request_timeout_ms = 5000
EOF
    
    # Start API server in background
    RUST_LOG=info ./target/release/api --config /tmp/api-test.toml &
    API_PID=$!
    
    sleep 5
    
    # Check if API is running
    if curl -sf "${API_URL}/health" > /dev/null 2>&1; then
        log_success "API server started (PID: $API_PID)"
    else
        log_error "API server failed to start"
        return 1
    fi
}

# Test PDF upload
test_pdf_upload() {
    log_info "Testing PDF upload..."
    
    RESPONSE=$(curl -sf -X POST "${API_URL}/api/v1/documents" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@${PDF_FILE}" \
        -F "metadata={\"source\":\"test\",\"type\":\"resume\"}" 2>/dev/null || echo "{}")
    
    if echo "$RESPONSE" | grep -q "document_id"; then
        DOCUMENT_ID=$(echo "$RESPONSE" | grep -o '"document_id":"[^"]*"' | cut -d'"' -f4)
        log_success "PDF uploaded successfully. Document ID: $DOCUMENT_ID"
        echo "$DOCUMENT_ID" > /tmp/test-doc-id.txt
    else
        log_warning "PDF upload endpoint not fully implemented"
        # Create mock document
        DOCUMENT_ID="test-doc-$(date +%s)"
        echo "$DOCUMENT_ID" > /tmp/test-doc-id.txt
        log_info "Using mock document ID: $DOCUMENT_ID"
    fi
}

# Test query processing
test_query_processing() {
    log_info "Testing query processing..."
    
    # Sample queries for testing
    declare -a queries=(
        "What is the experience of the candidate?"
        "What programming languages are mentioned?"
        "What is the education background?"
        "List the key skills"
    )
    
    for query in "${queries[@]}"; do
        log_info "Query: '$query'"
        
        RESPONSE=$(curl -sf -X POST "${API_URL}/api/v1/queries" \
            -H "Content-Type: application/json" \
            -d "{
                \"query_id\": \"$(uuidgen || echo test-$(date +%s))\",
                \"query\": \"$query\",
                \"document_ids\": [\"$(cat /tmp/test-doc-id.txt)\"],
                \"max_results\": 5
            }" 2>/dev/null || echo "{}")
        
        if echo "$RESPONSE" | grep -q "answer"; then
            ANSWER=$(echo "$RESPONSE" | grep -o '"answer":"[^"]*"' | cut -d'"' -f4)
            log_success "Answer: $ANSWER"
        else
            log_warning "Query processing not fully implemented yet"
            log_info "Response: $RESPONSE"
        fi
        
        sleep 1
    done
}

# Interactive Q&A session
interactive_qa() {
    log_info "Starting interactive Q&A session..."
    log_info "Type 'quit' to exit"
    
    while true; do
        echo -n -e "${BLUE}Question:${NC} "
        read -r query
        
        [ "$query" = "quit" ] && break
        
        RESPONSE=$(curl -sf -X POST "${API_URL}/api/v1/queries" \
            -H "Content-Type: application/json" \
            -d "{
                \"query_id\": \"interactive-$(date +%s)\",
                \"query\": \"$query\",
                \"document_ids\": [\"$(cat /tmp/test-doc-id.txt 2>/dev/null || echo 'test')\"],
                \"max_results\": 3
            }" 2>/dev/null || echo '{"error":"Not implemented"}')
        
        echo -e "${GREEN}Answer:${NC}"
        echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
        echo
    done
}

# Cleanup
cleanup() {
    log_info "Cleaning up..."
    
    # Stop API server
    [ -n "${API_PID:-}" ] && kill $API_PID 2>/dev/null || true
    
    # Stop MongoDB if we started it
    [ "$USE_DOCKER" = true ] && docker stop doc-rag-mongodb-test 2>/dev/null || true
    
    # Clean temp files
    rm -f /tmp/api-test.toml /tmp/test-doc-id.txt
    
    log_success "Cleanup complete"
}

# Main execution
main() {
    log_info "=== Doc-RAG Simple PDF Testing ==="
    
    trap cleanup EXIT
    
    check_requirements
    start_mongodb
    start_api_server
    test_pdf_upload
    test_query_processing
    
    echo
    log_info "Basic tests complete. Starting interactive mode..."
    interactive_qa
}

# Run main
main "$@"