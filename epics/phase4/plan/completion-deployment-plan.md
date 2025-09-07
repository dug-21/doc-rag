# Phase 4 - Production Deployment Plan

## Overview
Comprehensive production deployment strategy for the Doc-RAG system using Docker and Kubernetes infrastructure. This plan ensures zero-downtime deployment with full observability and rollback capabilities.

## 🎯 Deployment Objectives
- **Zero-downtime deployment** using blue-green strategy
- **Auto-scaling** based on load and resource utilization
- **High availability** with 99.9% uptime SLA
- **Security-first** approach with hardened configurations
- **Full observability** from day one

## 🏗️ Infrastructure Architecture

### Kubernetes Cluster Configuration
```yaml
# Cluster Specifications
Nodes: 3-5 worker nodes (auto-scaling)
Node Size: 4 vCPU, 16GB RAM minimum
Storage: SSD with 1000 IOPS minimum
Network: Private subnet with NAT gateway
Load Balancer: Application Load Balancer with SSL termination
```

### Namespace Strategy
```yaml
# Production namespaces
doc-rag-prod        # Main application services
doc-rag-monitoring  # Prometheus, Grafana, Jaeger
doc-rag-security    # Security services and scanners
doc-rag-ingress     # Ingress controllers and load balancers
```

## 🐳 Docker Container Configuration

### Base Image Strategy
- **Multi-stage builds** for optimized image sizes
- **Distroless images** for security
- **Non-root users** for all containers
- **Security scanning** integrated in CI/CD

### Container Specifications
```dockerfile
# API Service Container
FROM rust:1.74-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin api

FROM gcr.io/distroless/cc-debian12
COPY --from=builder /app/target/release/api /app/api
USER 1001:1001
EXPOSE 8080 9090
CMD ["/app/api"]
```

## 🚀 Deployment Strategy

### Blue-Green Deployment Process

#### Phase 1: Preparation
1. **Infrastructure Validation**
   ```bash
   # Validate cluster health
   kubectl cluster-info
   kubectl get nodes -o wide
   kubectl top nodes
   
   # Check storage classes
   kubectl get storageclass
   kubectl get pv,pvc --all-namespaces
   ```

2. **Pre-deployment Checks**
   ```bash
   # Run pre-deployment validation
   ./scripts/pre-deployment-check.sh
   
   # Validate configurations
   kubectl apply --dry-run=client -f k8s/production/
   
   # Check resource quotas
   kubectl describe quota --all-namespaces
   ```

#### Phase 2: Green Environment Deployment
1. **Deploy Green Environment**
   ```bash
   # Create green deployment
   kubectl apply -f k8s/production/green/
   
   # Wait for rollout completion
   kubectl rollout status deployment/api-deployment-green -n doc-rag-prod
   
   # Verify health checks
   kubectl get pods -l version=green -n doc-rag-prod
   ```

2. **Smoke Testing**
   ```bash
   # Internal health checks
   kubectl exec -it deploy/api-deployment-green -n doc-rag-prod -- curl http://localhost:8080/health
   
   # Database connectivity
   kubectl exec -it deploy/api-deployment-green -n doc-rag-prod -- nc -zv postgres-service 5432
   
   # Cache connectivity  
   kubectl exec -it deploy/api-deployment-green -n doc-rag-prod -- redis-cli -h redis-service ping
   ```

#### Phase 3: Traffic Switching
1. **Canary Deployment (5% traffic)**
   ```yaml
   # Update ingress for canary
   apiVersion: networking.k8s.io/v1
   kind: Ingress
   metadata:
     name: api-ingress
     annotations:
       nginx.ingress.kubernetes.io/canary: "true"
       nginx.ingress.kubernetes.io/canary-weight: "5"
   ```

2. **Progressive Traffic Shift**
   ```bash
   # 25% traffic
   kubectl patch ingress api-ingress -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"25"}}}'
   
   # 50% traffic
   kubectl patch ingress api-ingress -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"50"}}}'
   
   # 100% traffic
   kubectl patch service api-service -p '{"spec":{"selector":{"version":"green"}}}'
   ```

#### Phase 4: Blue Environment Cleanup
1. **Monitor Green Environment**
   ```bash
   # Wait for stability (15 minutes minimum)
   sleep 900
   
   # Check error rates
   kubectl logs -l version=green -n doc-rag-prod --tail=1000 | grep -i error
   
   # Check metrics
   curl -s http://prometheus:9090/api/v1/query?query=rate(http_requests_total[5m])
   ```

2. **Cleanup Blue Environment**
   ```bash
   # Scale down blue deployment
   kubectl scale deployment api-deployment-blue --replicas=0 -n doc-rag-prod
   
   # Remove blue resources (after 24h)
   kubectl delete -f k8s/production/blue/
   ```

## ⚙️ Service Configuration

### API Service Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-deployment
  namespace: doc-rag-prod
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 2
  template:
    spec:
      containers:
      - name: api
        image: doc-rag/api:v1.0.0
        ports:
        - containerPort: 8080
        - containerPort: 9090
        env:
        - name: RUST_LOG
          value: "info"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: doc-rag-secrets
              key: database-url
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2
            memory: 2Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Database Service Configuration
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: doc-rag-prod
spec:
  serviceName: postgres-service
  replicas: 1
  template:
    spec:
      containers:
      - name: postgres
        image: postgres:16-alpine
        env:
        - name: POSTGRES_DB
          value: "docrag"
        - name: POSTGRES_USER
          value: "docrag"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: doc-rag-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
```

## 🔧 Service Mesh Configuration

### Istio Service Mesh
```yaml
# Gateway configuration
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: doc-rag-gateway
  namespace: doc-rag-prod
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: doc-rag-tls
    hosts:
    - "api.docrag.example.com"

# Virtual Service
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: doc-rag-vs
  namespace: doc-rag-prod
spec:
  hosts:
  - "api.docrag.example.com"
  gateways:
  - doc-rag-gateway
  http:
  - match:
    - uri:
        prefix: /api/
    route:
    - destination:
        host: api-service
        port:
          number: 8080
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 5s
    retries:
      attempts: 3
      perTryTimeout: 10s
```

## 🌐 Load Balancing Setup

### Application Load Balancer
```yaml
# AWS Load Balancer Controller
apiVersion: v1
kind: Service
metadata:
  name: api-service-alb
  namespace: doc-rag-prod
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "http"
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: "arn:aws:acm:region:account:certificate/cert-id"
    service.beta.kubernetes.io/aws-load-balancer-ssl-ports: "443"
spec:
  type: LoadBalancer
  ports:
  - port: 443
    targetPort: 8080
    protocol: TCP
    name: https
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  selector:
    app.kubernetes.io/name: doc-rag
    app.kubernetes.io/component: api
```

### NGINX Ingress Configuration
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: doc-rag-ingress
  namespace: doc-rag-prod
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    nginx.ingress.kubernetes.io/rate-limit-rps: "100"
    nginx.ingress.kubernetes.io/rate-limit-connections: "10"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.docrag.example.com
    secretName: doc-rag-tls
  rules:
  - host: api.docrag.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
```

## 📊 Monitoring Integration

### Prometheus Configuration
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: doc-rag-monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    rule_files:
    - "/etc/prometheus/rules/*.yml"
    scrape_configs:
    - job_name: 'doc-rag-api'
      kubernetes_sd_configs:
      - role: endpoints
        namespaces:
          names:
          - doc-rag-prod
      relabel_configs:
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
```

### Grafana Dashboard Provisioning
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboards
  namespace: doc-rag-monitoring
data:
  doc-rag-overview.json: |
    {
      "dashboard": {
        "id": null,
        "title": "Doc-RAG System Overview",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(http_requests_total[5m])",
                "legendFormat": "{{method}} {{status}}"
              }
            ]
          },
          {
            "title": "Response Time",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                "legendFormat": "95th percentile"
              }
            ]
          }
        ]
      }
    }
```

## 🔐 Security Configuration

### Network Policies
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: doc-rag-network-policy
  namespace: doc-rag-prod
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: doc-rag
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: doc-rag-ingress
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: postgres
    ports:
    - protocol: TCP
      port: 5432
```

### Pod Security Policy
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: doc-rag-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
  - ALL
  volumes:
  - 'configMap'
  - 'emptyDir'
  - 'projected'
  - 'secret'
  - 'downwardAPI'
  - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## 🚀 Deployment Scripts

### Pre-deployment Validation Script
```bash
#!/bin/bash
# scripts/pre-deployment-check.sh

set -euo pipefail

echo "🔍 Running pre-deployment validation..."

# Check kubectl connectivity
kubectl cluster-info > /dev/null || { echo "❌ Cannot connect to cluster"; exit 1; }

# Validate configurations
echo "📋 Validating Kubernetes configurations..."
kubectl apply --dry-run=client -f k8s/production/ > /dev/null || { echo "❌ Invalid configurations"; exit 1; }

# Check resource availability
echo "💾 Checking resource availability..."
AVAILABLE_CPU=$(kubectl top nodes --no-headers | awk '{sum+=$3} END {print sum}' | cut -d% -f1)
AVAILABLE_MEM=$(kubectl top nodes --no-headers | awk '{sum+=$5} END {print sum}' | cut -d% -f1)

if [[ $AVAILABLE_CPU -gt 80 ]]; then
    echo "⚠️  CPU utilization high: ${AVAILABLE_CPU}%"
fi

if [[ $AVAILABLE_MEM -gt 80 ]]; then
    echo "⚠️  Memory utilization high: ${AVAILABLE_MEM}%"
fi

# Check external dependencies
echo "🌐 Checking external dependencies..."
curl -sf http://prometheus:9090/-/healthy > /dev/null || { echo "❌ Prometheus not healthy"; exit 1; }
curl -sf http://grafana:3000/api/health > /dev/null || { echo "❌ Grafana not healthy"; exit 1; }

echo "✅ Pre-deployment validation passed!"
```

### Deployment Automation Script
```bash
#!/bin/bash
# scripts/deploy-production.sh

set -euo pipefail

VERSION=${1:-latest}
ENVIRONMENT=${2:-production}
NAMESPACE="doc-rag-prod"

echo "🚀 Starting deployment of version $VERSION to $ENVIRONMENT"

# Run pre-deployment checks
./scripts/pre-deployment-check.sh

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply configurations
echo "📦 Applying configurations..."
kubectl apply -f k8s/production/namespace.yaml
kubectl apply -f k8s/production/configmap.yaml
kubectl apply -f k8s/production/secrets.yaml

# Deploy services
echo "🔧 Deploying services..."
kubectl apply -f k8s/production/postgres.yaml
kubectl apply -f k8s/production/redis.yaml
kubectl apply -f k8s/production/api-deployment.yaml

# Wait for rollout
echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/api-deployment -n $NAMESPACE --timeout=600s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

# Run smoke tests
echo "🧪 Running smoke tests..."
./scripts/smoke-tests.sh $NAMESPACE

echo "🎉 Deployment completed successfully!"
```

## 📋 Post-Deployment Validation

### Health Check Script
```bash
#!/bin/bash
# scripts/health-check.sh

NAMESPACE="doc-rag-prod"
API_URL="https://api.docrag.example.com"

echo "🏥 Running health checks..."

# Check pod health
echo "📊 Checking pod health..."
kubectl get pods -n $NAMESPACE -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,READY:.status.containerStatuses[*].ready

# Check service endpoints
echo "🌐 Checking service endpoints..."
kubectl get endpoints -n $NAMESPACE

# API health check
echo "🔍 Checking API health..."
curl -sf $API_URL/health | jq '.'

# Database connectivity
echo "💾 Checking database connectivity..."
kubectl exec -n $NAMESPACE deployment/api-deployment -- nc -zv postgres-service 5432

# Cache connectivity
echo "🗄️ Checking cache connectivity..."
kubectl exec -n $NAMESPACE deployment/api-deployment -- redis-cli -h redis-service ping

echo "✅ All health checks passed!"
```

## 🔄 Rollback Plan

### Automated Rollback
```bash
#!/bin/bash
# scripts/rollback.sh

NAMESPACE="doc-rag-prod"
DEPLOYMENT="api-deployment"

echo "🔄 Starting rollback procedure..."

# Get current revision
CURRENT_REVISION=$(kubectl rollout history deployment/$DEPLOYMENT -n $NAMESPACE --revision=0 | tail -1 | awk '{print $1}')
PREVIOUS_REVISION=$((CURRENT_REVISION - 1))

echo "📝 Rolling back from revision $CURRENT_REVISION to $PREVIOUS_REVISION"

# Perform rollback
kubectl rollout undo deployment/$DEPLOYMENT -n $NAMESPACE --to-revision=$PREVIOUS_REVISION

# Wait for rollback completion
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=300s

# Verify rollback
./scripts/health-check.sh

echo "✅ Rollback completed successfully!"
```

---

**Next Steps**: Execute deployment plan after completion checklist validation and team sign-off.