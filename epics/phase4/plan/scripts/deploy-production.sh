#!/bin/bash
# Production Deployment Script for Doc-RAG System
# Usage: ./deploy-production.sh [version] [environment]

set -euo pipefail

# Configuration
VERSION=${1:-latest}
ENVIRONMENT=${2:-production}
NAMESPACE="doc-rag-prod"
BACKUP_RETENTION_DAYS=30
DEPLOYMENT_TIMEOUT=900  # 15 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Trap function for cleanup
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        error "Deployment failed with exit code $exit_code"
        log "Starting rollback procedure..."
        perform_rollback
    fi
    exit $exit_code
}

trap cleanup EXIT

# Pre-deployment checks
pre_deployment_checks() {
    log "🔍 Running pre-deployment checks..."
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &>/dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace $NAMESPACE &>/dev/null; then
        log "Creating namespace $NAMESPACE"
        kubectl create namespace $NAMESPACE
    fi
    
    # Validate Kubernetes configurations
    log "📋 Validating Kubernetes configurations..."
    if ! kubectl apply --dry-run=client -f k8s/production/ &>/dev/null; then
        error "Invalid Kubernetes configurations"
        exit 1
    fi
    
    # Check resource availability
    log "💾 Checking cluster resource availability..."
    local available_cpu=$(kubectl top nodes --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum}' | cut -d% -f1 || echo "0")
    local available_mem=$(kubectl top nodes --no-headers 2>/dev/null | awk '{sum+=$5} END {print sum}' | cut -d% -f1 || echo "0")
    
    if [[ ${available_cpu:-0} -gt 80 ]]; then
        warning "High CPU utilization: ${available_cpu}%"
    fi
    
    if [[ ${available_mem:-0} -gt 80 ]]; then
        warning "High memory utilization: ${available_mem}%"
    fi
    
    # Check Docker images availability
    log "🐳 Verifying Docker images..."
    local images=(
        "doc-rag/api:$VERSION"
        "doc-rag/neural-chunker:$VERSION"
        "postgres:16-alpine"
        "redis:7-alpine"
        "qdrant/qdrant:v1.6.1"
    )
    
    for image in "${images[@]}"; do
        if ! docker manifest inspect "$image" &>/dev/null; then
            error "Docker image not found: $image"
            exit 1
        fi
    done
    
    # Check external dependencies
    log "🌐 Checking external dependencies..."
    local dependencies=(
        "prometheus:9090"
        "grafana:3000"
    )
    
    for dep in "${dependencies[@]}"; do
        local host=$(echo "$dep" | cut -d: -f1)
        local port=$(echo "$dep" | cut -d: -f2)
        if ! nc -z "$host" "$port" 2>/dev/null; then
            warning "Cannot reach $dep - monitoring may be affected"
        fi
    done
    
    success "Pre-deployment checks completed successfully"
}

# Create backup before deployment
create_backup() {
    log "💾 Creating pre-deployment backup..."
    
    local backup_date=$(date +%Y%m%d_%H%M%S)
    local backup_dir="/backups/pre-deployment-$backup_date"
    
    mkdir -p "$backup_dir"
    
    # Database backup
    if kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=postgres &>/dev/null; then
        log "Backing up PostgreSQL database..."
        kubectl exec -n $NAMESPACE deployment/postgres -- pg_dump -U docrag docrag | gzip > "$backup_dir/postgres-backup.sql.gz"
    fi
    
    # Configuration backup
    log "Backing up Kubernetes configurations..."
    kubectl get all,configmap,secret,pvc -n $NAMESPACE -o yaml > "$backup_dir/k8s-resources.yaml"
    
    # Vector database backup
    if kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=qdrant &>/dev/null; then
        log "Backing up vector database..."
        kubectl exec -n $NAMESPACE deployment/qdrant -- tar -czf - /qdrant/storage > "$backup_dir/qdrant-backup.tar.gz"
    fi
    
    success "Backup created at $backup_dir"
    echo "BACKUP_DIR=$backup_dir" > /tmp/deployment-backup-path
}

# Deploy application components
deploy_components() {
    log "🚀 Starting deployment of version $VERSION to $ENVIRONMENT..."
    
    # Apply namespace and RBAC
    log "📦 Applying namespace and RBAC configurations..."
    kubectl apply -f k8s/production/namespace.yaml
    kubectl apply -f k8s/production/rbac.yaml
    
    # Apply ConfigMaps and Secrets
    log "⚙️ Applying configurations and secrets..."
    kubectl apply -f k8s/production/configmap.yaml
    
    # Check if secrets exist, create if needed
    if ! kubectl get secret doc-rag-secrets -n $NAMESPACE &>/dev/null; then
        log "Creating default secrets (replace with proper secret management in production)"
        kubectl create secret generic doc-rag-secrets -n $NAMESPACE \
            --from-literal=database-url="postgres://docrag:$(openssl rand -base64 32)@postgres-service:5432/docrag" \
            --from-literal=redis-url="redis://:$(openssl rand -base64 32)@redis-service:6379" \
            --from-literal=jwt-secret="$(openssl rand -base64 64)"
    fi
    
    # Deploy persistent storage
    log "💾 Deploying persistent storage..."
    kubectl apply -f k8s/production/storage.yaml
    
    # Deploy databases first (PostgreSQL, Redis, Qdrant)
    log "🗄️ Deploying database services..."
    kubectl apply -f k8s/production/postgres.yaml
    kubectl apply -f k8s/production/redis.yaml  
    kubectl apply -f k8s/production/qdrant.yaml
    
    # Wait for databases to be ready
    log "⏳ Waiting for database services to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgres -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=qdrant -n $NAMESPACE --timeout=300s
    
    # Deploy application services
    log "🔧 Deploying application services..."
    kubectl apply -f k8s/production/neural-chunker.yaml
    kubectl apply -f k8s/production/api-deployment.yaml
    
    # Deploy ingress and load balancer
    log "🌐 Deploying ingress and load balancer..."
    kubectl apply -f k8s/production/nginx.yaml
    kubectl apply -f k8s/production/ingress.yaml
    
    # Deploy monitoring services
    log "📊 Deploying monitoring services..."
    kubectl apply -f k8s/production/monitoring.yaml
    
    success "All components deployed successfully"
}

# Wait for rollout completion
wait_for_rollout() {
    log "⏳ Waiting for rollout completion..."
    
    local deployments=(
        "api-deployment"
        "neural-chunker-deployment"
        "postgres"
        "redis"
        "qdrant"
        "nginx"
    )
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n $NAMESPACE &>/dev/null; then
            log "Waiting for $deployment rollout..."
            kubectl rollout status deployment/"$deployment" -n $NAMESPACE --timeout=${DEPLOYMENT_TIMEOUT}s
        elif kubectl get statefulset "$deployment" -n $NAMESPACE &>/dev/null; then
            log "Waiting for $deployment statefulset..."
            kubectl rollout status statefulset/"$deployment" -n $NAMESPACE --timeout=${DEPLOYMENT_TIMEOUT}s
        fi
    done
    
    success "All rollouts completed successfully"
}

# Verify deployment health
verify_deployment() {
    log "✅ Verifying deployment health..."
    
    # Check pod status
    log "📊 Checking pod status..."
    kubectl get pods -n $NAMESPACE -o wide
    
    # Check if all pods are running
    local failed_pods=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    if [[ $failed_pods -gt 0 ]]; then
        error "$failed_pods pods are not in running state"
        kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running
        return 1
    fi
    
    # Check service endpoints
    log "🌐 Checking service endpoints..."
    kubectl get endpoints -n $NAMESPACE
    
    # Health checks
    log "🏥 Running health checks..."
    
    # API health check
    local api_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}')
    if [[ -n "$api_pod" ]]; then
        kubectl exec -n $NAMESPACE "$api_pod" -- curl -sf http://localhost:8080/health || {
            error "API health check failed"
            return 1
        }
    fi
    
    # Database connectivity check
    local postgres_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=postgres -o jsonpath='{.items[0].metadata.name}')
    if [[ -n "$postgres_pod" ]]; then
        kubectl exec -n $NAMESPACE "$postgres_pod" -- pg_isready -U docrag || {
            error "PostgreSQL health check failed"
            return 1
        }
    fi
    
    # Redis connectivity check
    local redis_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=redis -o jsonpath='{.items[0].metadata.name}')
    if [[ -n "$redis_pod" ]]; then
        kubectl exec -n $NAMESPACE "$redis_pod" -- redis-cli ping || {
            error "Redis health check failed"  
            return 1
        }
    fi
    
    success "All health checks passed"
}

# Run smoke tests
run_smoke_tests() {
    log "🧪 Running smoke tests..."
    
    # Get API service endpoint
    local api_service=$(kubectl get service api-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    
    if [[ -n "$api_service" ]]; then
        # Test API endpoints
        kubectl run smoke-test-pod --rm -i --restart=Never --image=curlimages/curl -- \
            curl -sf "http://$api_service:8080/health" || {
            error "Smoke test failed for API health endpoint"
            return 1
        }
        
        kubectl run smoke-test-pod --rm -i --restart=Never --image=curlimages/curl -- \
            curl -sf "http://$api_service:9090/metrics" | head -5 || {
            warning "Metrics endpoint may not be ready yet"
        }
    fi
    
    success "Smoke tests completed"
}

# Update monitoring and alerting
update_monitoring() {
    log "📊 Updating monitoring configuration..."
    
    # Update Prometheus targets
    kubectl patch configmap prometheus-config -n doc-rag-monitoring --patch "
data:
  prometheus.yml: |
    $(cat config/prometheus.yml | sed "s/VERSION_PLACEHOLDER/$VERSION/g")
"
    
    # Restart Prometheus to reload config
    if kubectl get deployment prometheus -n doc-rag-monitoring &>/dev/null; then
        kubectl rollout restart deployment/prometheus -n doc-rag-monitoring
    fi
    
    success "Monitoring configuration updated"
}

# Cleanup old resources
cleanup_old_resources() {
    log "🧹 Cleaning up old resources..."
    
    # Remove old replica sets
    kubectl delete replicaset -n $NAMESPACE --cascade=orphan --all
    
    # Remove completed jobs older than 24 hours
    kubectl delete jobs -n $NAMESPACE --field-selector=status.conditions[0].type=Complete \
        --ignore-not-found=true --all
    
    # Clean up old config map revisions (keep last 3)
    local old_configmaps=$(kubectl get configmap -n $NAMESPACE -o name | grep -E '.*-[0-9]+$' | sort -V | head -n -3)
    if [[ -n "$old_configmaps" ]]; then
        echo "$old_configmaps" | xargs -r kubectl delete -n $NAMESPACE
    fi
    
    success "Resource cleanup completed"
}

# Perform rollback if deployment fails
perform_rollback() {
    log "🔄 Performing rollback..."
    
    local deployments=(
        "api-deployment"
        "neural-chunker-deployment"
    )
    
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n $NAMESPACE &>/dev/null; then
            log "Rolling back $deployment..."
            kubectl rollout undo deployment/"$deployment" -n $NAMESPACE
            kubectl rollout status deployment/"$deployment" -n $NAMESPACE --timeout=300s
        fi
    done
    
    # Restore from backup if needed
    if [[ -f /tmp/deployment-backup-path ]]; then
        local backup_dir=$(cat /tmp/deployment-backup-path | cut -d= -f2)
        if [[ -d "$backup_dir" ]]; then
            warning "Consider restoring from backup at: $backup_dir"
        fi
    fi
    
    error "Rollback completed, but manual intervention may be required"
}

# Generate deployment report
generate_deployment_report() {
    log "📋 Generating deployment report..."
    
    local report_file="/tmp/deployment-report-$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Doc-RAG Production Deployment Report
====================================
Date: $(date)
Version: $VERSION
Environment: $ENVIRONMENT
Namespace: $NAMESPACE

Deployment Status:
$(kubectl get deployments -n $NAMESPACE)

Pod Status:
$(kubectl get pods -n $NAMESPACE -o wide)

Service Status:
$(kubectl get services -n $NAMESPACE)

Resource Usage:
$(kubectl top pods -n $NAMESPACE)

Recent Events:
$(kubectl get events -n $NAMESPACE --sort-by=.metadata.creationTimestamp | tail -20)

Health Check Results:
- API Health: $(kubectl exec -n $NAMESPACE deployment/api-deployment -- curl -sf http://localhost:8080/health 2>/dev/null && echo "PASS" || echo "FAIL")
- Database: $(kubectl exec -n $NAMESPACE deployment/postgres -- pg_isready -U docrag 2>/dev/null && echo "PASS" || echo "FAIL")
- Cache: $(kubectl exec -n $NAMESPACE deployment/redis -- redis-cli ping 2>/dev/null && echo "PASS" || echo "FAIL")

Next Steps:
1. Monitor application performance for the next 24 hours
2. Verify all integrations are working correctly
3. Update DNS records if necessary
4. Schedule performance testing
5. Update documentation with any changes

EOF

    log "Deployment report generated: $report_file"
    cat "$report_file"
}

# Main deployment function
main() {
    log "🚀 Starting Doc-RAG production deployment..."
    log "Version: $VERSION"
    log "Environment: $ENVIRONMENT"
    log "Namespace: $NAMESPACE"
    
    # Execute deployment steps
    pre_deployment_checks
    create_backup
    deploy_components
    wait_for_rollout
    verify_deployment
    run_smoke_tests
    update_monitoring
    cleanup_old_resources
    generate_deployment_report
    
    success "🎉 Production deployment completed successfully!"
    
    log "📊 System Status:"
    kubectl get pods -n $NAMESPACE
    
    log "🔗 Access Points:"
    log "  API: https://$(kubectl get ingress api-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}')"
    log "  Monitoring: Check Grafana dashboard for system metrics"
    
    log "📝 Next Steps:"
    log "  1. Verify all services are functioning correctly"
    log "  2. Monitor system performance and resource usage"
    log "  3. Run comprehensive integration tests"
    log "  4. Update monitoring alerts and thresholds"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi