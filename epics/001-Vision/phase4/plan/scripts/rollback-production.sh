#!/bin/bash
# Production Rollback Script for Doc-RAG System
# Usage: ./rollback-production.sh [target-revision] [namespace]

set -euo pipefail

# Configuration
TARGET_REVISION=${1:-""}
NAMESPACE=${2:-doc-rag-prod}
ROLLBACK_TIMEOUT=600  # 10 minutes
BACKUP_RESTORE=${3:-false}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
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

# Emergency stop function
emergency_stop() {
    error "Emergency stop initiated!"
    log "Attempting to restore last known good state..."
    
    # Scale down all deployments to prevent further damage
    kubectl scale deployment --all --replicas=0 -n $NAMESPACE
    
    # Wait a moment for pods to terminate
    sleep 30
    
    # Try to restore from emergency backup if available
    if [[ -f "/tmp/emergency-backup-path" ]]; then
        local backup_path=$(cat /tmp/emergency-backup-path)
        warning "Emergency backup available at: $backup_path"
        echo "Manual intervention required to restore from backup"
    fi
    
    exit 1
}

# Trap for emergency situations
trap 'emergency_stop' ERR

# Pre-rollback checks
pre_rollback_checks() {
    log "🔍 Running pre-rollback checks..."
    
    # Check kubectl connectivity
    if ! kubectl cluster-info &>/dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace $NAMESPACE &>/dev/null; then
        error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    # Check if there are deployments to rollback
    local deployments=$(kubectl get deployments -n $NAMESPACE --no-headers 2>/dev/null | wc -l)
    if [[ $deployments -eq 0 ]]; then
        error "No deployments found in namespace $NAMESPACE"
        exit 1
    fi
    
    success "Pre-rollback checks completed"
}

# Create emergency backup
create_emergency_backup() {
    log "💾 Creating emergency backup before rollback..."
    
    local backup_date=$(date +%Y%m%d_%H%M%S)
    local backup_dir="/backups/emergency-rollback-$backup_date"
    
    mkdir -p "$backup_dir"
    
    # Backup current Kubernetes state
    log "Backing up current Kubernetes state..."
    kubectl get all,configmap,secret,pvc -n $NAMESPACE -o yaml > "$backup_dir/current-k8s-state.yaml"
    
    # Backup current application data if possible
    if kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=postgres --no-headers &>/dev/null; then
        log "Creating emergency database backup..."
        local postgres_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=postgres -o jsonpath='{.items[0].metadata.name}')
        kubectl exec -n $NAMESPACE "$postgres_pod" -- pg_dump -U docrag docrag | gzip > "$backup_dir/emergency-db-backup.sql.gz" || warning "Database backup failed"
    fi
    
    # Save backup location
    echo "EMERGENCY_BACKUP_DIR=$backup_dir" > /tmp/emergency-backup-path
    
    success "Emergency backup created at $backup_dir"
}

# Get rollback information
get_rollback_info() {
    log "📋 Getting rollback information..."
    
    local deployments=(
        "api-deployment"
        "neural-chunker-deployment"
        "postgres"
        "redis" 
        "qdrant"
        "nginx"
    )
    
    echo ""
    log "Current deployment status:"
    for deployment in "${deployments[@]}"; do
        if kubectl get deployment "$deployment" -n $NAMESPACE &>/dev/null; then
            echo "📦 $deployment:"
            kubectl rollout history deployment/"$deployment" -n $NAMESPACE | tail -5
            echo ""
        elif kubectl get statefulset "$deployment" -n $NAMESPACE &>/dev/null; then
            echo "📦 $deployment (StatefulSet):"
            kubectl rollout history statefulset/"$deployment" -n $NAMESPACE | tail -5
            echo ""
        fi
    done
}

# Determine rollback strategy
determine_rollback_strategy() {
    log "🎯 Determining rollback strategy..."
    
    if [[ -n "$TARGET_REVISION" ]]; then
        log "Rolling back to specific revision: $TARGET_REVISION"
    else
        log "Rolling back to previous revision (automatic)"
        warning "No specific target revision specified - using previous revision"
    fi
    
    # Check if rollback is safe based on current system state
    local failed_pods=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    if [[ $failed_pods -gt 5 ]]; then
        warning "Many pods are in failed state ($failed_pods), rollback may not be sufficient"
        warning "Consider full system restore from backup"
        read -p "Continue with rollback anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Perform application rollback
rollback_applications() {
    log "🔄 Starting application rollback..."
    
    local rollback_deployments=(
        "api-deployment"
        "neural-chunker-deployment"
    )
    
    # Rollback application services first
    for deployment in "${rollback_deployments[@]}"; do
        if kubectl get deployment "$deployment" -n $NAMESPACE &>/dev/null; then
            log "Rolling back deployment: $deployment"
            
            if [[ -n "$TARGET_REVISION" ]]; then
                kubectl rollout undo deployment/"$deployment" -n $NAMESPACE --to-revision="$TARGET_REVISION"
            else
                kubectl rollout undo deployment/"$deployment" -n $NAMESPACE
            fi
            
            # Wait for rollback to complete
            log "Waiting for $deployment rollback to complete..."
            if ! kubectl rollout status deployment/"$deployment" -n $NAMESPACE --timeout=${ROLLBACK_TIMEOUT}s; then
                error "Rollback failed for $deployment"
                return 1
            fi
            
            success "Rollback completed for $deployment"
        else
            warning "Deployment $deployment not found, skipping"
        fi
    done
}

# Verify rollback health
verify_rollback_health() {
    log "🏥 Verifying rollback health..."
    
    # Wait a moment for services to stabilize
    sleep 30
    
    # Check pod status
    local failed_pods=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
    if [[ $failed_pods -gt 0 ]]; then
        warning "$failed_pods pods are not in running state after rollback"
        kubectl get pods -n $NAMESPACE --field-selector=status.phase!=Running
    else
        success "All pods are running after rollback"
    fi
    
    # Test API health endpoint
    local api_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/component=api -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [[ -n "$api_pod" ]]; then
        log "Testing API health endpoint..."
        if kubectl exec -n $NAMESPACE "$api_pod" -- curl -sf http://localhost:8080/health &>/dev/null; then
            success "API health endpoint is responding"
        else
            error "API health endpoint is not responding after rollback"
            return 1
        fi
    fi
    
    # Test database connectivity
    local postgres_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=postgres -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [[ -n "$postgres_pod" ]]; then
        log "Testing database connectivity..."
        if kubectl exec -n $NAMESPACE "$postgres_pod" -- pg_isready -U docrag &>/dev/null; then
            success "Database connectivity is working"
        else
            error "Database connectivity failed after rollback"
            return 1
        fi
    fi
    
    # Test Redis connectivity
    local redis_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=redis -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    if [[ -n "$redis_pod" ]]; then
        log "Testing Redis connectivity..."
        if kubectl exec -n $NAMESPACE "$redis_pod" -- redis-cli ping &>/dev/null; then
            success "Redis connectivity is working"
        else
            error "Redis connectivity failed after rollback"
            return 1
        fi
    fi
}

# Restore from backup if needed
restore_from_backup() {
    if [[ "$BACKUP_RESTORE" == "true" ]]; then
        log "💾 Restoring from backup as requested..."
        
        # Find the most recent backup
        local backup_dir=""
        if [[ -f "/tmp/deployment-backup-path" ]]; then
            backup_dir=$(cat /tmp/deployment-backup-path | cut -d= -f2)
        else
            # Look for most recent backup
            backup_dir=$(ls -dt /backups/pre-deployment-* 2>/dev/null | head -1 || echo "")
        fi
        
        if [[ -z "$backup_dir" || ! -d "$backup_dir" ]]; then
            error "No backup directory found for restoration"
            return 1
        fi
        
        log "Restoring from backup: $backup_dir"
        
        # Restore database if backup exists
        if [[ -f "$backup_dir/postgres-backup.sql.gz" ]]; then
            local postgres_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=postgres -o jsonpath='{.items[0].metadata.name}')
            if [[ -n "$postgres_pod" ]]; then
                log "Restoring database from backup..."
                
                # Create a temporary restore database
                kubectl exec -n $NAMESPACE "$postgres_pod" -- createdb -U docrag docrag_restore || true
                
                # Restore data to temporary database
                gunzip -c "$backup_dir/postgres-backup.sql.gz" | kubectl exec -i -n $NAMESPACE "$postgres_pod" -- psql -U docrag docrag_restore
                
                # Verify backup integrity
                local restore_count=$(kubectl exec -n $NAMESPACE "$postgres_pod" -- psql -U docrag -d docrag_restore -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
                
                if [[ $restore_count -gt 0 ]]; then
                    # Switch databases
                    kubectl exec -n $NAMESPACE "$postgres_pod" -- dropdb -U docrag docrag || true
                    kubectl exec -n $NAMESPACE "$postgres_pod" -- psql -U docrag -c "ALTER DATABASE docrag_restore RENAME TO docrag;"
                    success "Database restored from backup"
                else
                    error "Backup restoration verification failed"
                    return 1
                fi
            fi
        fi
        
        # Restore vector database if backup exists
        if [[ -f "$backup_dir/qdrant-backup.tar.gz" ]]; then
            local qdrant_pod=$(kubectl get pods -n $NAMESPACE -l app.kubernetes.io/name=qdrant -o jsonpath='{.items[0].metadata.name}')
            if [[ -n "$qdrant_pod" ]]; then
                log "Restoring vector database from backup..."
                kubectl exec -n $NAMESPACE "$qdrant_pod" -- tar -xzf - -C /qdrant/storage < "$backup_dir/qdrant-backup.tar.gz"
                kubectl delete pod "$qdrant_pod" -n $NAMESPACE  # Force restart to load backup
                success "Vector database restored from backup"
            fi
        fi
    fi
}

# Update monitoring after rollback
update_monitoring_after_rollback() {
    log "📊 Updating monitoring after rollback..."
    
    # Send alert to monitoring system about rollback
    if kubectl get service prometheus-service -n doc-rag-monitoring &>/dev/null; then
        # Create a custom metric to indicate rollback
        kubectl run rollback-metric-sender --rm -i --restart=Never --image=curlimages/curl -- \
            curl -X POST http://prometheus-service.doc-rag-monitoring.svc.cluster.local:9090/api/v1/admin/tsdb/delete_series \
            -d 'match[]=doc_rag_rollback_event{}'
        
        # Send rollback event metric
        local current_time=$(date +%s)
        kubectl run rollback-metric-sender --rm -i --restart=Never --image=appropriate/curl -- \
            curl -X POST http://prometheus-service.doc-rag-monitoring.svc.cluster.local:9090/api/v1/write \
            -H 'Content-Type: application/x-protobuf' \
            --data-binary "doc_rag_rollback_event{type=\"production\",timestamp=\"$current_time\"} 1"
    fi
    
    # Update Grafana annotations if possible
    if kubectl get service grafana-service -n doc-rag-monitoring &>/dev/null; then
        warning "Consider adding rollback annotation to Grafana dashboards manually"
    fi
}

# Generate rollback report
generate_rollback_report() {
    log "📋 Generating rollback report..."
    
    local report_file="/tmp/rollback-report-$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Doc-RAG Production Rollback Report
==================================
Date: $(date)
Target Revision: ${TARGET_REVISION:-"Previous"}
Namespace: $NAMESPACE
Backup Restore: $BACKUP_RESTORE

Rollback Summary:
$(kubectl rollout history deployment/api-deployment -n $NAMESPACE | tail -3)

Current System Status:
$(kubectl get pods -n $NAMESPACE -o wide)

Service Status:
$(kubectl get services -n $NAMESPACE)

Resource Usage Post-Rollback:
$(kubectl top pods -n $NAMESPACE 2>/dev/null || echo "Resource metrics not available")

Recent Events:
$(kubectl get events -n $NAMESPACE --sort-by='.metadata.creationTimestamp' | tail -10)

Health Check Results:
- API Health: $(kubectl exec -n $NAMESPACE deployment/api-deployment -- curl -sf http://localhost:8080/health 2>/dev/null && echo "PASS" || echo "FAIL")
- Database: $(kubectl exec -n $NAMESPACE deployment/postgres -- pg_isready -U docrag 2>/dev/null && echo "PASS" || echo "FAIL")
- Cache: $(kubectl exec -n $NAMESPACE deployment/redis -- redis-cli ping 2>/dev/null && echo "PASS" || echo "FAIL")

Rollback Actions Taken:
1. Created emergency backup before rollback
2. Rolled back application deployments to previous/target revision
3. Verified system health after rollback
$([ "$BACKUP_RESTORE" == "true" ] && echo "4. Restored data from backup")

Next Steps:
1. Monitor system performance for next 2 hours
2. Verify all user-facing functionality works correctly
3. Check error logs for any ongoing issues
4. Plan investigation of root cause that led to rollback
5. Schedule follow-up deployment with fixes

Emergency Contacts:
- On-call Engineer: [Contact Info]
- Team Lead: [Contact Info]
- Platform Team: [Contact Info]

EOF

    log "Rollback report generated: $report_file"
    cat "$report_file"
}

# Cleanup rollback artifacts
cleanup_rollback_artifacts() {
    log "🧹 Cleaning up rollback artifacts..."
    
    # Remove old replica sets
    kubectl delete replicaset -n $NAMESPACE --cascade=orphan --all || true
    
    # Clean up any failed pods
    kubectl delete pods -n $NAMESPACE --field-selector=status.phase=Failed || true
    
    # Remove temporary resources
    kubectl delete pods -n $NAMESPACE -l app=rollback-temp || true
    
    success "Rollback artifacts cleaned up"
}

# Send notifications
send_rollback_notifications() {
    log "📨 Sending rollback notifications..."
    
    # Slack notification (if webhook configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{'text':'🔄 Production rollback completed for Doc-RAG system\nNamespace: $NAMESPACE\nTime: $(date)\nStatus: Rollback successful, system health verified'}" \
            "$SLACK_WEBHOOK_URL" || warning "Slack notification failed"
    fi
    
    # Email notification (if configured)
    if command -v mail &>/dev/null && [[ -n "${ALERT_EMAIL:-}" ]]; then
        echo "Production rollback completed for Doc-RAG system at $(date)" | \
            mail -s "Doc-RAG Production Rollback Completed" "$ALERT_EMAIL" || warning "Email notification failed"
    fi
    
    log "Rollback notifications sent"
}

# Main rollback function
main() {
    log "🔄 Starting Doc-RAG production rollback..."
    log "Target revision: ${TARGET_REVISION:-Previous}"
    log "Namespace: $NAMESPACE"
    log "Backup restore: $BACKUP_RESTORE"
    
    # Confirmation prompt for production
    echo ""
    warning "You are about to perform a PRODUCTION ROLLBACK!"
    warning "This will affect live user traffic and data."
    echo ""
    read -p "Are you sure you want to continue? Type 'ROLLBACK' to confirm: " -r
    if [[ $REPLY != "ROLLBACK" ]]; then
        log "Rollback cancelled by user"
        exit 0
    fi
    
    # Execute rollback steps
    pre_rollback_checks
    create_emergency_backup
    get_rollback_info
    determine_rollback_strategy
    
    log "🚨 Starting rollback operations - no turning back!"
    rollback_applications
    verify_rollback_health
    restore_from_backup
    update_monitoring_after_rollback
    cleanup_rollback_artifacts
    generate_rollback_report
    send_rollback_notifications
    
    success "🎉 Production rollback completed successfully!"
    
    log "📊 Current System Status:"
    kubectl get pods -n $NAMESPACE
    
    log "📝 Important Next Steps:"
    log "  1. Monitor system performance for the next 2 hours"
    log "  2. Verify all user-facing functionality"
    log "  3. Investigate root cause of the issue that required rollback"
    log "  4. Plan corrective deployment with proper fixes"
    log "  5. Update incident documentation"
    
    warning "Remember: Rollback is a temporary measure. Root cause analysis and proper fix deployment are required."
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi