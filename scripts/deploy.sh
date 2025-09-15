#!/bin/bash

# Neurosymbolic RAG Deployment Script
# Deploys the corrected Rust application with proper configuration management

set -e

echo "🚀 Deploying Neurosymbolic RAG System"

# Configuration
IMAGE_NAME="neurosymbolic-rag"
IMAGE_TAG="latest"
CONTAINER_NAME="neurosymbolic-rag"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME:$IMAGE_TAG .

# Stop existing container if running
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    echo "🛑 Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Start database services using docker-compose
echo "📊 Starting database services..."
docker-compose up -d neo4j mongodb redis qdrant

# Wait for services to be ready
echo "⏳ Waiting for database services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking database service health..."

# Check Neo4j
echo "Checking Neo4j..."
timeout 60 bash -c 'until docker-compose exec neo4j cypher-shell -u neo4j -p password "RETURN 1" 2>/dev/null; do sleep 2; done'

# Check Qdrant
echo "Checking Qdrant..."
timeout 60 bash -c 'until curl -f http://localhost:6333/health >/dev/null 2>&1; do sleep 2; done'

# Check MongoDB
echo "Checking MongoDB..."
timeout 60 bash -c 'until docker-compose exec mongodb mongosh --eval "db.adminCommand(\"ping\")" >/dev/null 2>&1; do sleep 2; done'

# Check Redis
echo "Checking Redis..."
timeout 60 bash -c 'until docker-compose exec redis redis-cli ping >/dev/null 2>&1; do sleep 2; done'

echo "✅ All database services are ready"

# Start the main application
echo "🚀 Starting Neurosymbolic RAG application..."
docker run -d \
    --name $CONTAINER_NAME \
    --network neurosymbolic-rag-network \
    -p 8080:8080 \
    -p 9090:9090 \
    -e APP_ENV=docker \
    -e DOCKER_ENV=1 \
    -e RUST_LOG=info \
    -e RUST_BACKTRACE=1 \
    -e NEO4J_URI=bolt://neo4j:7687 \
    -e NEO4J_USER=neo4j \
    -e NEO4J_PASSWORD=password \
    -e MONGODB_URI=mongodb://mongodb:27017/neurosymbolic-rag \
    -e REDIS_URL=redis://redis:6379 \
    -e QDRANT_URL=http://qdrant:6333 \
    -e FACT_CACHE_ENABLED=true \
    -e NEURAL_CHUNKER_ENABLED=true \
    -e SYMBOLIC_REASONING_ENABLED=true \
    -v ./data/documents:/app/documents \
    -v ./data/models:/app/models \
    -v ./logs:/app/logs \
    $IMAGE_NAME:$IMAGE_TAG

# Wait for application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Health check
echo "🔍 Performing health check..."
HEALTH_CHECK_RETRIES=30
for i in $(seq 1 $HEALTH_CHECK_RETRIES); do
    if curl -f http://localhost:8080/health >/dev/null 2>&1; then
        echo "✅ Application health check passed"
        break
    elif [ $i -eq $HEALTH_CHECK_RETRIES ]; then
        echo "❌ Application health check failed after $HEALTH_CHECK_RETRIES attempts"
        echo "Container logs:"
        docker logs $CONTAINER_NAME --tail 50
        exit 1
    else
        echo "⏳ Health check attempt $i/$HEALTH_CHECK_RETRIES failed, retrying..."
        sleep 5
    fi
done

# Display deployment information
echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Service Information:"
echo "  Application:     http://localhost:8080"
echo "  Health Check:    http://localhost:8080/health"
echo "  Metrics:         http://localhost:9090/metrics"
echo "  Query API:       http://localhost:8080/api/v1/query"
echo ""
echo "📊 Database Services:"
echo "  Neo4j Web UI:    http://localhost:7474 (neo4j/password)"
echo "  Qdrant API:      http://localhost:6333"
echo "  MongoDB:         localhost:27017"
echo "  Redis:           localhost:6379"
echo ""
echo "🐳 Container Management:"
echo "  View logs:       docker logs $CONTAINER_NAME -f"
echo "  Stop service:    docker stop $CONTAINER_NAME"
echo "  Remove service:  docker rm $CONTAINER_NAME"
echo "  Stop all:        docker-compose down"
echo ""

# Test query endpoint
echo "🧪 Testing query endpoint..."
QUERY_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"query": "health check test", "max_results": 1}' \
    http://localhost:8080/api/v1/query || echo "FAILED")

if [[ "$QUERY_RESPONSE" == *"FAILED"* ]]; then
    echo "⚠️  Query endpoint test failed (normal during startup)"
else
    echo "✅ Query endpoint responding"
fi

echo ""
echo "🎯 Quick Test Commands:"
echo "  curl http://localhost:8080/health"
echo "  curl http://localhost:8080/metrics"
echo "  curl -X POST -H 'Content-Type: application/json' -d '{\"query\":\"test\"}' http://localhost:8080/api/v1/query"
echo ""
echo "🔧 Troubleshooting:"
echo "  Check logs:      docker logs $CONTAINER_NAME"
echo "  Check services:  docker-compose ps"
echo "  Restart:         ./scripts/deploy.sh"