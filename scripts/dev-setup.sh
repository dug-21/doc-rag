#!/bin/bash

# Development setup script for neurosymbolic RAG
# Quickly get the entire stack running for development and testing

set -e

echo "🚀 Setting up neurosymbolic RAG development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
print_status "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Docker and Docker Compose are available"

# Create required directories
print_status "Creating required directories..."
mkdir -p data/documents data/models logs config

# Create basic configuration files if they don't exist
print_status "Setting up configuration files..."

if [ ! -f "config/app.toml" ]; then
    cat > config/app.toml << EOF
[server]
host = "0.0.0.0"
port = 8080
metrics_port = 9090

[neural]
chunker_enabled = true
model_path = "/app/models"

[symbolic]
reasoning_enabled = true
max_inference_depth = 5

[cache]
fact_cache_ttl = 3600
query_cache_ttl = 1800

[logging]
level = "info"
format = "json"
EOF
    print_success "Created default app.toml configuration"
fi

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Build and start services
print_status "Building and starting neurosymbolic RAG services..."
print_warning "This may take several minutes on first run..."

# Start infrastructure services first
print_status "Starting infrastructure services (Neo4j, Redis, MongoDB, Qdrant)..."
docker-compose up -d neo4j redis mongodb qdrant

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 30

# Check service health
print_status "Checking service health..."

# Check Neo4j
if curl -s http://localhost:7474 > /dev/null; then
    print_success "Neo4j is running (http://localhost:7474)"
else
    print_warning "Neo4j may still be starting up"
fi

# Check Redis
if redis-cli -h localhost ping 2>/dev/null | grep -q PONG; then
    print_success "Redis is running"
else
    print_warning "Redis may not be accessible"
fi

# Check MongoDB
if curl -s http://localhost:27017 > /dev/null 2>&1; then
    print_success "MongoDB is running"
else
    print_warning "MongoDB may still be starting up"
fi

# Check Qdrant
if curl -s http://localhost:6333/health > /dev/null; then
    print_success "Qdrant is running (http://localhost:6333)"
else
    print_warning "Qdrant may still be starting up"
fi

# Build and start the main application
print_status "Building and starting the main application..."
docker-compose up -d --build neurosymbolic-rag

# Wait for application to start
print_status "Waiting for application to start..."
sleep 60

# Check application health
print_status "Checking application health..."
for i in {1..10}; do
    if curl -s http://localhost:8080/health > /dev/null; then
        print_success "Neurosymbolic RAG application is running!"
        break
    elif [ $i -eq 10 ]; then
        print_error "Application health check failed after 10 attempts"
        print_status "Checking logs..."
        docker-compose logs neurosymbolic-rag
        exit 1
    else
        print_status "Waiting for application... (attempt $i/10)"
        sleep 10
    fi
done

# Display status
echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "📋 Service Status:"
echo "  • Main Application: http://localhost:8080"
echo "  • Application Health: http://localhost:8080/health"
echo "  • Metrics: http://localhost:9090/metrics"
echo "  • Neo4j Browser: http://localhost:7474 (neo4j/password)"
echo "  • Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "🧪 Quick Test Commands:"
echo "  # Check health"
echo "  curl http://localhost:8080/health"
echo ""
echo "  # Upload a document (place PDF in data/documents/)"
echo "  curl -X POST -F 'file=@data/documents/sample.pdf' http://localhost:8080/api/v1/upload"
echo ""
echo "  # Query the system"
echo "  curl -X POST -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"What are the main requirements?\"}' \\"
echo "    http://localhost:8080/api/v1/query"
echo ""
echo "📊 Monitoring:"
echo "  # View logs"
echo "  docker-compose logs -f neurosymbolic-rag"
echo ""
echo "  # Stop everything"
echo "  docker-compose down"
echo ""
echo "🔧 Development workflow:"
echo "  1. Make code changes"
echo "  2. Run: docker-compose up -d --build neurosymbolic-rag"
echo "  3. Test with curl commands above"
echo ""

print_success "Setup complete! Happy developing! 🚀"