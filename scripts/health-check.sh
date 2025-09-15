#!/bin/bash

# Health check script for neurosymbolic RAG application
# Tests basic connectivity and service health

set -e

echo "🔍 Starting health check for neurosymbolic RAG system..."

# Application health endpoint
echo "📊 Checking application health..."
APP_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health || echo "000")

if [ "$APP_HEALTH" = "200" ]; then
    echo "✅ Application health: OK"
else
    echo "❌ Application health: FAILED (HTTP $APP_HEALTH)"
    exit 1
fi

# Check if metrics endpoint is available
echo "📈 Checking metrics endpoint..."
METRICS_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9090/metrics || echo "000")

if [ "$METRICS_HEALTH" = "200" ]; then
    echo "✅ Metrics endpoint: OK"
else
    echo "⚠️  Metrics endpoint: UNAVAILABLE (HTTP $METRICS_HEALTH)"
fi

# Test basic query functionality (if application is ready)
echo "🧠 Testing basic query functionality..."
QUERY_RESPONSE=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d '{"query": "health check test", "max_results": 1}' \
    http://localhost:8080/api/v1/query || echo "FAILED")

if [[ "$QUERY_RESPONSE" == *"FAILED"* ]]; then
    echo "⚠️  Query endpoint: Not yet ready (normal during startup)"
else
    echo "✅ Query endpoint: Responding"
fi

echo "🎉 Health check completed successfully!"
exit 0