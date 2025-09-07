#!/bin/bash
# Test with the actual thor_resume.pdf file
# Focuses on real document processing

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== Testing with Real PDF: thor_resume.pdf ===${NC}"

# Check if PDF exists
if [ ! -f "uploads/thor_resume.pdf" ]; then
    echo -e "${YELLOW}thor_resume.pdf not found!${NC}"
    exit 1
fi

# Start minimal services
echo -e "${BLUE}Starting services...${NC}"
docker-compose up -d mongodb 2>/dev/null || {
    echo "Starting MongoDB with Docker..."
    docker run -d --rm --name test-mongodb \
        -p 27017:27017 \
        mongo:7.0
}

sleep 5

# Build and run API
echo -e "${BLUE}Building API...${NC}"
cargo build --release --bin api 2>/dev/null

# Start API with minimal config
echo -e "${BLUE}Starting API server...${NC}"
RUST_LOG=debug MONGODB_URL=mongodb://localhost:27017/doc_rag_test \
    ./target/release/api &
API_PID=$!

sleep 3

# Process the PDF
echo -e "${BLUE}Processing thor_resume.pdf...${NC}"

# Upload PDF
DOC_ID=$(curl -s -X POST http://localhost:8080/api/v1/documents \
    -F "file=@uploads/thor_resume.pdf" \
    -F 'metadata={"type":"resume","name":"Thor"}' \
    | grep -o '"document_id":"[^"]*"' | cut -d'"' -f4)

echo -e "${GREEN}Document uploaded: $DOC_ID${NC}"

# Test queries specific to the resume
echo -e "${BLUE}Testing resume-specific queries...${NC}"

queries=(
    "What is Thor's current position?"
    "What programming languages does Thor know?"
    "What are Thor's main achievements?"
    "Where did Thor study?"
    "What projects has Thor worked on?"
)

for query in "${queries[@]}"; do
    echo -e "\n${BLUE}Q: $query${NC}"
    
    response=$(curl -s -X POST http://localhost:8080/api/v1/queries \
        -H "Content-Type: application/json" \
        -d "{
            \"query_id\": \"test-$(date +%s)\",
            \"query\": \"$query\",
            \"document_ids\": [\"$DOC_ID\"]
        }")
    
    echo -e "${GREEN}A:${NC} $response" | python3 -m json.tool 2>/dev/null || echo "$response"
done

# Cleanup
echo -e "\n${BLUE}Cleaning up...${NC}"
kill $API_PID 2>/dev/null
docker stop test-mongodb 2>/dev/null || true

echo -e "${GREEN}Test complete!${NC}"