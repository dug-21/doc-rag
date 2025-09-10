#!/bin/bash
# Comprehensive Health Check Script for Doc-RAG Production System
# Usage: ./health-check-comprehensive.sh [namespace] [verbose]

set -euo pipefail

# Configuration
NAMESPACE=${1:-doc-rag-prod}
VERBOSE=${2:-false}
HEALTH_CHECK_TIMEOUT=30
CRITICAL_THRESHOLD=90  # CPU/Memory percentage
WARNING_THRESHOLD=80

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Health check results
HEALTH_STATUS="PASS"
CRITICAL_ISSUES=0
WARNING_ISSUES=0
INFO_ISSUES=0

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNING_ISSUES++))
    if [[ "$HEALTH_STATUS" == "PASS" ]]; then
        HEALTH_STATUS="WARNING"
    fi
}

critical() {
    echo -e "${RED}❌ $1${NC}"
    ((CRITICAL_ISSUES++))
    HEALTH_STATUS="CRITICAL"
}

info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
    ((INFO_ISSUES++))
}

verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

# Check if kubectl is available and cluster is accessible
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        critical "kubectl command not found"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        critical "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        critical "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Check overall cluster health
check_cluster_health() {
    log "🏥 Checking cluster health..."
    
    # Check node status
    local nodes_not_ready=$(kubectl get nodes --no-headers | grep -v Ready | wc -l)
    if [[ $nodes_not_ready -gt 0 ]]; then
        critical "$nodes_not_ready nodes are not in Ready state"
    else
        success "All cluster nodes are healthy"
    fi
    
    # Check cluster resource usage
    verbose "Checking cluster resource utilization..."
    if command -v kubectl-top &> /dev/null; then
        local cluster_cpu=$(kubectl top nodes --no-headers 2>/dev/null | awk '{sum+=$3} END {print sum}' | cut -d% -f1 || echo "0")
        local cluster_mem=$(kubectl top nodes --no-headers 2>/dev/null | awk '{sum+=$5} END {print sum}' | cut -d% -f1 || echo "0")
        
        if [[ ${cluster_cpu:-0} -gt $CRITICAL_THRESHOLD ]]; then
            critical "High cluster CPU usage: ${cluster_cpu}%"
        elif [[ ${cluster_cpu:-0} -gt $WARNING_THRESHOLD ]]; then
            warning "Elevated cluster CPU usage: ${cluster_cpu}%"
        fi
        
        if [[ ${cluster_mem:-0} -gt $CRITICAL_THRESHOLD ]]; then
            critical "High cluster memory usage: ${cluster_mem}%"
        elif [[ ${cluster_mem:-0} -gt $WARNING_THRESHOLD ]]; then
            warning "Elevated cluster memory usage: ${cluster_mem}%"
        fi
    fi
}

# Check pod health and status
check_pod_health() {
    log "🏃 Checking pod health..."
    
    # Get all pods in namespace
    local pods_info=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null)
    
    if [[ -z "$pods_info" ]]; then
        critical "No pods found in namespace $NAMESPACE"
        return 1
    fi
    
    # Check pod statuses
    local running_pods=0
    local total_pods=0
    
    while IFS= read -r line; do
        if [[ -z "$line" ]]; then continue; fi
        
        local pod_name=$(echo "$line" | awk '{print $1}')
        local ready=$(echo "$line" | awk '{print $2}')
        local status=$(echo "$line" | awk '{print $3}')
        local restarts=$(echo "$line" | awk '{print $4}')
        local age=$(echo "$line" | awk '{print $5}')
        
        ((total_pods++))
        
        verbose "Checking pod: $pod_name"
        
        case $status in
            "Running")
                ((running_pods++))
                
                # Check if pod is ready
                local ready_containers=$(echo "$ready" | cut -d/ -f1)
                local total_containers=$(echo "$ready" | cut -d/ -f2)
                
                if [[ "$ready_containers" != "$total_containers" ]]; then
                    warning "Pod $pod_name is running but not ready ($ready)"
                fi
                
                # Check restart count
                if [[ $restarts -gt 5 ]]; then
                    warning "Pod $pod_name has high restart count: $restarts"
                elif [[ $restarts -gt 0 ]]; then
                    info "Pod $pod_name has been restarted $restarts times"
                fi
                ;;
            "Pending")
                warning "Pod $pod_name is in Pending state"
                verbose "$(kubectl describe pod "$pod_name" -n "$NAMESPACE" | grep -A 5 "Events:")"
                ;;
            "CrashLoopBackOff"|"Error"|"Failed")
                critical "Pod $pod_name is in failed state: $status"
                verbose "$(kubectl logs "$pod_name" -n "$NAMESPACE" --tail=10 2>/dev/null || echo 'No logs available')"
                ;;
            *)
                warning "Pod $pod_name is in unexpected state: $status"
                ;;
        esac
    done <<< "$pods_info"
    
    if [[ $running_pods -eq $total_pods ]]; then
        success "All $total_pods pods are running"
    else
        warning "$running_pods/$total_pods pods are running"
    fi
}

# Check service health and endpoints
check_service_health() {
    log "🌐 Checking service health..."
    
    local services=$(kubectl get services -n "$NAMESPACE" --no-headers 2>/dev/null)
    
    if [[ -z "$services" ]]; then
        warning "No services found in namespace $NAMESPACE"
        return 0
    fi
    
    while IFS= read -r line; do
        if [[ -z "$line" ]]; then continue; fi
        
        local service_name=$(echo "$line" | awk '{print $1}')
        local service_type=$(echo "$line" | awk '{print $2}')
        
        verbose "Checking service: $service_name"
        
        # Check if service has endpoints
        local endpoints=$(kubectl get endpoints "$service_name" -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null)
        
        if [[ -z "$endpoints" ]]; then
            warning "Service $service_name has no endpoints"
        else
            local endpoint_count=$(echo "$endpoints" | wc -w)
            success "Service $service_name has $endpoint_count endpoint(s)"
        fi
        
        # Check LoadBalancer services
        if [[ "$service_type" == "LoadBalancer" ]]; then
            local external_ip=$(echo "$line" | awk '{print $4}')
            if [[ "$external_ip" == "<pending>" ]]; then
                warning "LoadBalancer service $service_name has no external IP"
            fi
        fi
        
    done <<< "$services"
}

# Check application-specific health endpoints
check_application_health() {
    log "🔍 Checking application health endpoints..."
    
    # API Service Health Check
    local api_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api --no-headers 2>/dev/null | awk '{print $1}')
    
    if [[ -n "$api_pods" ]]; then
        while IFS= read -r pod; do
            if [[ -z "$pod" ]]; then continue; fi
            
            verbose "Checking API health on pod: $pod"
            
            # Health endpoint
            if kubectl exec -n "$NAMESPACE" "$pod" -- timeout $HEALTH_CHECK_TIMEOUT curl -sf http://localhost:8080/health &>/dev/null; then
                success "API health endpoint is responding on $pod"
            else
                critical "API health endpoint is not responding on $pod"
            fi
            
            # Ready endpoint
            if kubectl exec -n "$NAMESPACE" "$pod" -- timeout $HEALTH_CHECK_TIMEOUT curl -sf http://localhost:8080/ready &>/dev/null; then
                success "API ready endpoint is responding on $pod"
            else
                warning "API ready endpoint is not responding on $pod"
            fi
            
            # Metrics endpoint
            if kubectl exec -n "$NAMESPACE" "$pod" -- timeout $HEALTH_CHECK_TIMEOUT curl -sf http://localhost:9090/metrics &>/dev/null; then
                success "API metrics endpoint is responding on $pod"
            else
                warning "API metrics endpoint is not responding on $pod"
            fi
            
        done <<< "$api_pods"
    else
        warning "No API pods found"
    fi
}

# Check database connectivity and health
check_database_health() {
    log "🗄️ Checking database health..."
    
    # PostgreSQL Health Check
    local postgres_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=postgres --no-headers 2>/dev/null | awk '{print $1}')
    
    if [[ -n "$postgres_pods" ]]; then
        while IFS= read -r pod; do
            if [[ -z "$pod" ]]; then continue; fi
            
            verbose "Checking PostgreSQL on pod: $pod"
            
            # Check if PostgreSQL is accepting connections
            if kubectl exec -n "$NAMESPACE" "$pod" -- pg_isready -U docrag &>/dev/null; then
                success "PostgreSQL is accepting connections on $pod"
            else
                critical "PostgreSQL is not accepting connections on $pod"
            fi
            
            # Check database existence and basic query
            if kubectl exec -n "$NAMESPACE" "$pod" -- psql -U docrag -d docrag -c "SELECT 1;" &>/dev/null; then
                success "PostgreSQL database query successful on $pod"
            else
                critical "PostgreSQL database query failed on $pod"
            fi
            
            # Check database size and connection count
            if [[ "$VERBOSE" == "true" ]]; then
                local db_stats=$(kubectl exec -n "$NAMESPACE" "$pod" -- psql -U docrag -d docrag -t -c "
                SELECT 
                    pg_size_pretty(pg_database_size('docrag')) as db_size,
                    (SELECT count(*) FROM pg_stat_activity) as connections;
                " 2>/dev/null)
                
                if [[ -n "$db_stats" ]]; then
                    info "Database stats: $db_stats"
                fi
            fi
            
        done <<< "$postgres_pods"
    else
        warning "No PostgreSQL pods found"
    fi
    
    # Redis Health Check
    local redis_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=redis --no-headers 2>/dev/null | awk '{print $1}')
    
    if [[ -n "$redis_pods" ]]; then
        while IFS= read -r pod; do
            if [[ -z "$pod" ]]; then continue; fi
            
            verbose "Checking Redis on pod: $pod"
            
            # Check Redis ping
            if kubectl exec -n "$NAMESPACE" "$pod" -- redis-cli ping &>/dev/null; then
                success "Redis is responding to ping on $pod"
            else
                critical "Redis is not responding to ping on $pod"
            fi
            
            # Check Redis memory usage
            if [[ "$VERBOSE" == "true" ]]; then
                local redis_info=$(kubectl exec -n "$NAMESPACE" "$pod" -- redis-cli info memory 2>/dev/null | grep used_memory_human || echo "")
                if [[ -n "$redis_info" ]]; then
                    info "Redis memory usage: $redis_info"
                fi
            fi
            
        done <<< "$redis_pods"
    else
        warning "No Redis pods found"
    fi
    
    # Qdrant Health Check
    local qdrant_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=qdrant --no-headers 2>/dev/null | awk '{print $1}')
    
    if [[ -n "$qdrant_pods" ]]; then
        while IFS= read -r pod; do
            if [[ -z "$pod" ]]; then continue; fi
            
            verbose "Checking Qdrant on pod: $pod"
            
            # Check Qdrant health endpoint
            if kubectl exec -n "$NAMESPACE" "$pod" -- timeout $HEALTH_CHECK_TIMEOUT curl -sf http://localhost:6333/health &>/dev/null; then
                success "Qdrant is healthy on $pod"
            else
                critical "Qdrant health check failed on $pod"
            fi
            
        done <<< "$qdrant_pods"
    else
        warning "No Qdrant pods found"
    fi
}

# Check resource utilization
check_resource_utilization() {
    log "📊 Checking resource utilization..."
    
    if ! command -v kubectl-top &> /dev/null; then
        warning "kubectl top not available, skipping resource utilization checks"
        return 0
    fi
    
    # Pod resource utilization
    local pod_resources=$(kubectl top pods -n "$NAMESPACE" --no-headers 2>/dev/null)
    
    if [[ -n "$pod_resources" ]]; then
        while IFS= read -r line; do
            if [[ -z "$line" ]]; then continue; fi
            
            local pod_name=$(echo "$line" | awk '{print $1}')
            local cpu_usage=$(echo "$line" | awk '{print $2}' | sed 's/m//')
            local memory_usage=$(echo "$line" | awk '{print $3}' | sed 's/Mi//')
            
            verbose "Pod $pod_name: CPU=${cpu_usage}m, Memory=${memory_usage}Mi"
            
            # Check CPU usage (assuming 1000m = 100%)
            if [[ ${cpu_usage:-0} -gt 1800 ]]; then  # 180% of 1 core
                critical "Pod $pod_name has very high CPU usage: ${cpu_usage}m"
            elif [[ ${cpu_usage:-0} -gt 800 ]]; then  # 80% of 1 core
                warning "Pod $pod_name has high CPU usage: ${cpu_usage}m"
            fi
            
            # Check memory usage
            if [[ ${memory_usage:-0} -gt 1800 ]]; then  # 1.8GB
                critical "Pod $pod_name has very high memory usage: ${memory_usage}Mi"
            elif [[ ${memory_usage:-0} -gt 1200 ]]; then  # 1.2GB
                warning "Pod $pod_name has high memory usage: ${memory_usage}Mi"
            fi
            
        done <<< "$pod_resources"
    else
        info "No pod resource information available"
    fi
}

# Check storage and persistent volumes
check_storage_health() {
    log "💾 Checking storage health..."
    
    # Check PVC status
    local pvcs=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null)
    
    if [[ -n "$pvcs" ]]; then
        while IFS= read -r line; do
            if [[ -z "$line" ]]; then continue; fi
            
            local pvc_name=$(echo "$line" | awk '{print $1}')
            local status=$(echo "$line" | awk '{print $2}')
            local capacity=$(echo "$line" | awk '{print $4}')
            
            verbose "Checking PVC: $pvc_name"
            
            if [[ "$status" == "Bound" ]]; then
                success "PVC $pvc_name is bound (Capacity: $capacity)"
            else
                warning "PVC $pvc_name is in state: $status"
            fi
            
        done <<< "$pvcs"
    else
        info "No PVCs found in namespace $NAMESPACE"
    fi
    
    # Check disk usage on pods
    local storage_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api --no-headers 2>/dev/null | awk '{print $1}' | head -1)
    
    if [[ -n "$storage_pods" ]]; then
        verbose "Checking disk usage on pod: $storage_pods"
        
        local disk_usage=$(kubectl exec -n "$NAMESPACE" "$storage_pods" -- df -h /app/storage 2>/dev/null | tail -1 | awk '{print $5}' | sed 's/%//' || echo "0")
        
        if [[ ${disk_usage:-0} -gt 90 ]]; then
            critical "Storage usage is critically high: ${disk_usage}%"
        elif [[ ${disk_usage:-0} -gt 80 ]]; then
            warning "Storage usage is high: ${disk_usage}%"
        elif [[ ${disk_usage:-0} -gt 0 ]]; then
            success "Storage usage is normal: ${disk_usage}%"
        fi
    fi
}

# Check network connectivity
check_network_health() {
    log "🌐 Checking network connectivity..."
    
    # Test internal service connectivity
    local api_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api --no-headers 2>/dev/null | awk '{print $1}' | head -1)
    
    if [[ -n "$api_pods" ]]; then
        verbose "Testing network connectivity from pod: $api_pods"
        
        # Test connectivity to PostgreSQL
        if kubectl exec -n "$NAMESPACE" "$api_pods" -- timeout 10 nc -z postgres-service 5432 &>/dev/null; then
            success "Network connectivity to PostgreSQL service is working"
        else
            critical "Network connectivity to PostgreSQL service failed"
        fi
        
        # Test connectivity to Redis
        if kubectl exec -n "$NAMESPACE" "$api_pods" -- timeout 10 nc -z redis-service 6379 &>/dev/null; then
            success "Network connectivity to Redis service is working"
        else
            critical "Network connectivity to Redis service failed"
        fi
        
        # Test connectivity to Qdrant
        if kubectl exec -n "$NAMESPACE" "$api_pods" -- timeout 10 nc -z qdrant-service 6333 &>/dev/null; then
            success "Network connectivity to Qdrant service is working"
        else
            critical "Network connectivity to Qdrant service failed"
        fi
        
        # Test external connectivity (DNS resolution)
        if kubectl exec -n "$NAMESPACE" "$api_pods" -- timeout 10 nslookup google.com &>/dev/null; then
            success "External DNS resolution is working"
        else
            warning "External DNS resolution may be impacted"
        fi
    fi
}

# Check monitoring and metrics
check_monitoring_health() {
    log "📊 Checking monitoring health..."
    
    # Check if monitoring namespace exists
    if kubectl get namespace doc-rag-monitoring &>/dev/null; then
        
        # Check Prometheus
        local prometheus_pods=$(kubectl get pods -n doc-rag-monitoring -l app.kubernetes.io/name=prometheus --no-headers 2>/dev/null | awk '{print $1}')
        if [[ -n "$prometheus_pods" ]]; then
            local prom_pod=$(echo "$prometheus_pods" | head -1)
            if kubectl exec -n doc-rag-monitoring "$prom_pod" -- timeout $HEALTH_CHECK_TIMEOUT wget -q --spider http://localhost:9090/-/healthy &>/dev/null; then
                success "Prometheus is healthy"
            else
                warning "Prometheus health check failed"
            fi
        else
            warning "No Prometheus pods found"
        fi
        
        # Check Grafana
        local grafana_pods=$(kubectl get pods -n doc-rag-monitoring -l app.kubernetes.io/name=grafana --no-headers 2>/dev/null | awk '{print $1}')
        if [[ -n "$grafana_pods" ]]; then
            local grafana_pod=$(echo "$grafana_pods" | head -1)
            if kubectl exec -n doc-rag-monitoring "$grafana_pod" -- timeout $HEALTH_CHECK_TIMEOUT curl -sf http://localhost:3000/api/health &>/dev/null; then
                success "Grafana is healthy"
            else
                warning "Grafana health check failed"
            fi
        else
            warning "No Grafana pods found"
        fi
        
    else
        warning "Monitoring namespace not found"
    fi
}

# Check recent events and logs
check_recent_events() {
    log "📝 Checking recent events..."
    
    # Get recent warning/error events
    local warning_events=$(kubectl get events -n "$NAMESPACE" --field-selector type=Warning --sort-by='.metadata.creationTimestamp' -o custom-columns=TIME:.firstTimestamp,REASON:.reason,OBJECT:.involvedObject.name,MESSAGE:.message --no-headers 2>/dev/null | tail -5)
    
    if [[ -n "$warning_events" && $(echo "$warning_events" | wc -l) -gt 0 ]]; then
        warning "Recent warning events found:"
        echo "$warning_events" | while IFS= read -r event; do
            if [[ -n "$event" ]]; then
                echo "  - $event"
            fi
        done
    else
        success "No recent warning events"
    fi
    
    # Check for error logs in key pods
    if [[ "$VERBOSE" == "true" ]]; then
        local key_pods=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=api --no-headers 2>/dev/null | awk '{print $1}' | head -2)
        
        if [[ -n "$key_pods" ]]; then
            while IFS= read -r pod; do
                if [[ -n "$pod" ]]; then
                    verbose "Checking recent logs for errors on pod: $pod"
                    local error_count=$(kubectl logs "$pod" -n "$NAMESPACE" --tail=100 2>/dev/null | grep -i error | wc -l)
                    if [[ $error_count -gt 0 ]]; then
                        warning "Found $error_count error log entries in pod $pod"
                    fi
                fi
            done <<< "$key_pods"
        fi
    fi
}

# Generate health summary report
generate_health_report() {
    log "📋 Generating health check report..."
    
    local report_file="/tmp/health-check-report-$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Doc-RAG System Health Check Report
==================================
Date: $(date)
Namespace: $NAMESPACE
Overall Status: $HEALTH_STATUS

Summary:
- Critical Issues: $CRITICAL_ISSUES
- Warning Issues: $WARNING_ISSUES  
- Info Issues: $INFO_ISSUES

System Overview:
$(kubectl get pods -n "$NAMESPACE" -o wide 2>/dev/null)

Service Status:
$(kubectl get services -n "$NAMESPACE" 2>/dev/null)

Resource Usage:
$(kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Resource metrics not available")

Recent Events:
$(kubectl get events -n "$NAMESPACE" --sort-by='.metadata.creationTimestamp' | tail -10)

Recommendations:
EOF

    # Add recommendations based on issues found
    if [[ $CRITICAL_ISSUES -gt 0 ]]; then
        cat >> "$report_file" << EOF
- URGENT: Address $CRITICAL_ISSUES critical issues immediately
- Consider scaling or restarting affected services
- Check system logs for detailed error information
EOF
    fi
    
    if [[ $WARNING_ISSUES -gt 0 ]]; then
        cat >> "$report_file" << EOF
- Monitor $WARNING_ISSUES warning conditions closely
- Consider preventive maintenance or resource scaling
- Review performance metrics and trends
EOF
    fi
    
    if [[ $CRITICAL_ISSUES -eq 0 && $WARNING_ISSUES -eq 0 ]]; then
        cat >> "$report_file" << EOF
- System is healthy, continue normal monitoring
- Consider performance optimization opportunities
- Schedule regular maintenance windows
EOF
    fi
    
    echo "" >> "$report_file"
    echo "Health check completed at: $(date)" >> "$report_file"
    
    log "Health check report saved to: $report_file"
    
    if [[ "$VERBOSE" == "true" ]]; then
        cat "$report_file"
    fi
}

# Main health check function
main() {
    log "🏥 Starting comprehensive health check for Doc-RAG system..."
    log "Namespace: $NAMESPACE"
    log "Verbose mode: $VERBOSE"
    
    # Run all health checks
    check_prerequisites
    check_cluster_health
    check_pod_health
    check_service_health
    check_application_health
    check_database_health
    check_resource_utilization
    check_storage_health
    check_network_health
    check_monitoring_health
    check_recent_events
    generate_health_report
    
    # Final status
    echo ""
    log "🏁 Health check completed"
    
    case $HEALTH_STATUS in
        "PASS")
            success "Overall system health: HEALTHY ✅"
            ;;
        "WARNING")
            warning "Overall system health: WARNING ⚠️"
            echo "  - Critical issues: $CRITICAL_ISSUES"
            echo "  - Warning issues: $WARNING_ISSUES"
            ;;
        "CRITICAL")
            critical "Overall system health: CRITICAL ❌"
            echo "  - Critical issues: $CRITICAL_ISSUES"  
            echo "  - Warning issues: $WARNING_ISSUES"
            ;;
    esac
    
    # Exit with appropriate code
    if [[ $CRITICAL_ISSUES -gt 0 ]]; then
        exit 2
    elif [[ $WARNING_ISSUES -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi