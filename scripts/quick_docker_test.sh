#!/bin/bash

# Quick Docker-based PDF Testing
# Uses minimal Docker Compose setup

set -e

echo "🐳 Doc-RAG Docker PDF Test"
echo "=========================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_step() {
    echo -e "▶ $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/.."

print_step "Starting minimal Docker environment..."

# Stop any existing containers
docker-compose -f docker-compose.minimal.yml down 2>/dev/null || true

# Start services
print_step "Building and starting services..."
docker-compose -f docker-compose.minimal.yml up --build -d

# Wait for services to be ready
print_step "Waiting for services to be ready..."

# Wait for MongoDB
for i in {1..30}; do
    if docker-compose -f docker-compose.minimal.yml exec -T mongodb mongosh --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

print_success "MongoDB is ready"

# Wait for API
for i in {1..30}; do
    if curl -s http://localhost:8080/health >/dev/null 2>&1; then
        break
    fi
    sleep 2
done

if curl -s http://localhost:8080/health >/dev/null 2>&1; then
    print_success "API server is ready"
else
    print_error "API server failed to start"
    docker-compose -f docker-compose.minimal.yml logs api
    exit 1
fi

print_success "All services are running!"

echo
echo "🎯 Available services:"
echo "   📊 API Server: http://localhost:8080"
echo "   🗄️  MongoDB: localhost:27017"
echo
echo "🧪 Test commands:"
echo "   python3 scripts/live_pdf_qa.py    # Interactive Q&A"
echo "   curl http://localhost:8080/health  # Health check"
echo
echo "📋 Management:"
echo "   docker-compose -f docker-compose.minimal.yml logs    # View logs"
echo "   docker-compose -f docker-compose.minimal.yml down    # Stop services"
echo

# Offer to start interactive session
echo -n "Start interactive Q&A session now? (y/N): "
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo
    print_step "Starting interactive Q&A session..."
    python3 scripts/live_pdf_qa.py
fi

echo
echo "🐳 Docker environment is ready for testing!"