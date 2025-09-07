#!/bin/bash
# Service Health Check Script - Monitor all services
set -euo pipefail

# Configuration
OUTPUT_FORMAT="${1:-human}"  # human or json

# Service endpoints
declare -A SERVICES=(
    ["api"]="http://localhost:8080/health"
    ["mongodb"]="mongodb://localhost:27017"
    ["metrics"]="http://localhost:9090/metrics"
)

# Check service health
check_service() {
    local name=$1
    local endpoint=$2
    
    if [[ $endpoint == http* ]]; then
        # HTTP health check
        if curl -sf "$endpoint" > /dev/null 2>&1; then
            echo "UP"
        else
            echo "DOWN"
        fi
    elif [[ $endpoint == mongodb* ]]; then
        # MongoDB health check
        if mongosh --quiet --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
            echo "UP"
        else
            echo "DOWN"
        fi
    fi
}

# Output results
if [ "$OUTPUT_FORMAT" = "json" ]; then
    echo "{"
    for service in "${!SERVICES[@]}"; do
        status=$(check_service "$service" "${SERVICES[$service]}")
        echo "  \"$service\": \"$status\","
    done | sed '$ s/,$//'
    echo "}"
else
    echo "Service Health Status:"
    echo "====================="
    for service in "${!SERVICES[@]}"; do
        status=$(check_service "$service" "${SERVICES[$service]}")
        if [ "$status" = "UP" ]; then
            echo "✅ $service: $status"
        else
            echo "❌ $service: $status"
        fi
    done
fi