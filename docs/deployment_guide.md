# Doc-RAG Production Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Security Configuration](#security-configuration)
6. [Monitoring Setup](#monitoring-setup)
7. [Load Testing](#load-testing)
8. [Maintenance Procedures](#maintenance-procedures)

## Prerequisites

### System Requirements

**Minimum Production Requirements:**
- CPU: 8 cores
- RAM: 32GB
- Storage: 500GB SSD
- Network: 1Gbps

**Recommended Production Requirements:**
- CPU: 16 cores
- RAM: 64GB
- Storage: 1TB NVMe SSD
- Network: 10Gbps
- GPU: NVIDIA V100 or better (for embeddings)

### Software Dependencies

**Required:**
- Docker 24.0+
- Docker Compose 2.20+
- OpenSSL 3.0+
- Git 2.35+

**For Kubernetes:**
- Kubernetes 1.27+
- Helm 3.12+
- kubectl 1.27+

**For Security:**
- TLS certificates (Let's Encrypt or CA-signed)
- Secrets management system (HashiCorp Vault or cloud-native)

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/doc-rag.git
cd doc-rag
git checkout v1.0.0  # Use stable release
```

### 2. Environment Variables

Create production environment file:

```bash
# Copy template
cp .env.template .env.production

# Edit environment variables
nano .env.production
```

**Required Environment Variables:**

```bash
# Database Credentials
POSTGRES_PASSWORD=your_secure_postgres_password
POSTGRES_DB=docrag
POSTGRES_USER=docrag

# Redis Configuration
REDIS_PASSWORD=your_secure_redis_password

# Security
JWT_SECRET=your_256_bit_jwt_secret_key_here
API_KEY_SECRET=your_api_key_encryption_secret

# External Services
MINIO_ACCESS_KEY=your_minio_access_key
MINIO_SECRET_KEY=your_minio_secret_key

# Monitoring
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your_secure_grafana_password

# Application Settings
CORS_ORIGINS=https://your-domain.com,https://api.your-domain.com
ALLOWED_HOSTS=your-domain.com,api.your-domain.com
RATE_LIMIT_RPM=100
MAX_REQUEST_SIZE=10485760

# Performance
WORKER_THREADS=4
MAX_CONNECTIONS=100
CACHE_TTL=3600
```

### 3. SSL Certificate Setup

```bash
# Create SSL directory
mkdir -p config/ssl

# For Let's Encrypt (recommended)
certbot certonly --standalone -d your-domain.com -d api.your-domain.com

# Copy certificates
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem config/ssl/tls.crt
cp /etc/letsencrypt/live/your-domain.com/privkey.pem config/ssl/tls.key

# Set proper permissions
chmod 600 config/ssl/tls.key
chmod 644 config/ssl/tls.crt
```

## Docker Deployment

### 1. Build and Deploy

```bash
# Load environment variables
export $(cat .env.production | xargs)

# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Build custom images
docker-compose -f docker-compose.production.yml build

# Deploy all services
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
docker-compose -f docker-compose.production.yml ps
```

### 2. Database Initialization

```bash
# Wait for PostgreSQL to be ready
docker-compose -f docker-compose.production.yml exec postgres pg_isready -U docrag

# Run database migrations
docker-compose -f docker-compose.production.yml exec api ./run-migrations.sh

# Verify database setup
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U docrag -d docrag -c "SELECT version();"
```

### 3. Service Verification

```bash
# Check service health
curl -f http://localhost:8080/health
curl -f http://localhost:8081/health  # MCP Adapter
curl -f http://localhost:8082/health  # Embedder
curl -f http://localhost:8083/health  # Storage

# Check Prometheus metrics
curl http://localhost:9091/metrics

# Verify Grafana
curl http://localhost:3000/api/health
```

### 4. Load Balancer Configuration

```bash
# Test Nginx configuration
docker-compose -f docker-compose.production.yml exec nginx nginx -t

# Reload Nginx
docker-compose -f docker-compose.production.yml exec nginx nginx -s reload

# Test HTTPS
curl -k https://localhost/api/health
```

## Kubernetes Deployment

### 1. Cluster Preparation

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets
kubectl create secret generic doc-rag-secrets \
  --namespace=doc-rag-prod \
  --from-literal=database-url="postgres://docrag:${POSTGRES_PASSWORD}@postgres-service:5432/docrag" \
  --from-literal=redis-url="redis://:${REDIS_PASSWORD}@redis-service:6379" \
  --from-literal=jwt-secret="${JWT_SECRET}"

# Create TLS secret for ingress
kubectl create secret tls doc-rag-tls \
  --namespace=doc-rag-prod \
  --cert=config/ssl/tls.crt \
  --key=config/ssl/tls.key
```

### 2. Deploy Infrastructure

```bash
# Deploy ConfigMaps
kubectl apply -f k8s/configmap.yaml

# Deploy databases
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/mongodb-deployment.yaml
kubectl apply -f k8s/qdrant-deployment.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod \
  --namespace=doc-rag-prod \
  --selector=app.kubernetes.io/component=database \
  --timeout=300s
```

### 3. Deploy Application Services

```bash
# Deploy core services
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/security-deployment.yaml
kubectl apply -f k8s/query-processor-deployment.yaml
kubectl apply -f k8s/response-generator-deployment.yaml
kubectl apply -f k8s/embedder-deployment.yaml
kubectl apply -f k8s/storage-deployment.yaml

# Deploy load balancer
kubectl apply -f k8s/nginx-deployment.yaml
kubectl apply -f k8s/ingress.yaml

# Wait for all services
kubectl wait --for=condition=ready pod \
  --namespace=doc-rag-prod \
  --selector=app.kubernetes.io/name=doc-rag \
  --timeout=300s
```

### 4. Helm Chart Deployment (Alternative)

```bash
# Install using Helm
helm install doc-rag-prod ./helm/doc-rag \
  --namespace doc-rag-prod \
  --create-namespace \
  --values helm/doc-rag/values-production.yaml \
  --set secrets.jwtSecret="${JWT_SECRET}" \
  --set secrets.postgresPassword="${POSTGRES_PASSWORD}" \
  --set secrets.redisPassword="${REDIS_PASSWORD}"

# Verify deployment
helm status doc-rag-prod --namespace doc-rag-prod

# Check all pods are running
kubectl get pods --namespace doc-rag-prod
```

### 5. Monitoring Setup

```bash
# Deploy monitoring namespace
kubectl apply -f k8s/monitoring-namespace.yaml

# Deploy Prometheus
kubectl apply -f k8s/prometheus-deployment.yaml

# Deploy Grafana
kubectl apply -f k8s/grafana-deployment.yaml

# Deploy Jaeger
kubectl apply -f k8s/jaeger-deployment.yaml

# Set up service monitors
kubectl apply -f k8s/service-monitors.yaml
```

## Security Configuration

### 1. Network Security

```bash
# Apply network policies
kubectl apply -f k8s/network-policy.yaml

# Configure firewall rules (iptables example)
iptables -A INPUT -p tcp --dport 22 -j ACCEPT      # SSH
iptables -A INPUT -p tcp --dport 80 -j ACCEPT       # HTTP
iptables -A INPUT -p tcp --dport 443 -j ACCEPT      # HTTPS
iptables -A INPUT -p tcp --dport 9091 -s 10.0.0.0/8 -j ACCEPT  # Prometheus (internal)
iptables -A INPUT -j DROP                           # Drop all other traffic
```

### 2. Authentication Setup

```bash
# Generate initial admin user
docker-compose -f docker-compose.production.yml exec api \
  ./create-admin-user.sh admin@your-domain.com

# Test authentication
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@your-domain.com","password":"generated_password"}'
```

### 3. Rate Limiting Configuration

```bash
# Test rate limiting
for i in {1..10}; do
  curl -H "X-API-Key: test-key" http://localhost:8080/api/health
  echo "Request $i completed"
done

# Should see 429 responses after limit exceeded
```

### 4. Security Scanning

```bash
# Run security scan with Trivy
trivy image doc-rag/api:latest
trivy image doc-rag/security:latest

# Scan Kubernetes configurations
trivy config k8s/

# OWASP ZAP security test
docker run -v $(pwd):/zap/wrk/:rw \
  -t owasp/zap2docker-stable zap-baseline.py \
  -t https://your-domain.com \
  -g gen.conf -r baseline_report.html
```

## Monitoring Setup

### 1. Verify Prometheus Targets

```bash
# Check Prometheus targets
curl http://localhost:9091/api/v1/targets

# Expected targets:
# - doc-rag-api:9090
# - doc-rag-security:9090  
# - postgres-exporter:9187
# - redis-exporter:9121
# - qdrant:6333
# - node-exporter:9100
```

### 2. Configure Grafana Dashboards

```bash
# Import dashboards
curl -X POST http://admin:${GRAFANA_ADMIN_PASSWORD}@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @config/grafana/dashboards/main-dashboard.json

curl -X POST http://admin:${GRAFANA_ADMIN_PASSWORD}@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @config/grafana/dashboards/security-dashboard.json
```

### 3. Alert Configuration

```bash
# Test alert rules
curl http://localhost:9091/api/v1/rules

# Simulate alert condition (high error rate)
for i in {1..100}; do
  curl http://localhost:8080/api/nonexistent-endpoint
done

# Check alerts
curl http://localhost:9091/api/v1/alerts
```

### 4. Log Aggregation

```bash
# Configure log rotation
cat > /etc/logrotate.d/doc-rag << EOF
/opt/docrag/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 644 docrag docrag
    postrotate
        docker-compose -f docker-compose.production.yml kill -s USR1 nginx
    endscript
}
EOF

# Test log rotation
logrotate -d /etc/logrotate.d/doc-rag
```

## Load Testing

### 1. Basic Performance Test

```bash
# Install Apache Bench
apt-get update && apt-get install -y apache2-utils

# Test API endpoint
ab -n 1000 -c 10 http://localhost:8080/api/health

# Test with authentication
ab -n 1000 -c 10 -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  http://localhost:8080/api/documents
```

### 2. Comprehensive Load Test

```bash
# Install wrk
git clone https://github.com/wg/wrk.git
cd wrk && make && sudo cp wrk /usr/local/bin/

# Create test script
cat > load_test.lua << 'EOF'
wrk.method = "POST"
wrk.body = '{"query": "What is machine learning?", "limit": 10}'
wrk.headers["Content-Type"] = "application/json"
wrk.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"
EOF

# Run load test
wrk -t12 -c400 -d30s -s load_test.lua http://localhost:8080/api/query
```

### 3. Database Load Test

```bash
# PostgreSQL connection test
pgbench -i -s 50 -U docrag -d docrag -h localhost
pgbench -c 10 -j 2 -t 1000 -U docrag -d docrag -h localhost

# Redis performance test
docker exec doc-rag-redis-prod redis-benchmark \
  -a ${REDIS_PASSWORD} -c 50 -n 10000 -d 3
```

### 4. Vector Search Performance Test

```bash
# Create test vectors
python3 << 'EOF'
import requests
import numpy as np
import time

base_url = "http://localhost:8080/api"
headers = {"Authorization": "Bearer YOUR_JWT_TOKEN"}

# Generate random test vectors
test_vectors = np.random.rand(100, 384).tolist()

start_time = time.time()
for i, vector in enumerate(test_vectors):
    response = requests.post(
        f"{base_url}/search/vector",
        json={"vector": vector, "limit": 10},
        headers=headers
    )
    if i % 10 == 0:
        print(f"Processed {i} queries, avg time: {(time.time() - start_time)/(i+1):.3f}s")

print(f"Total time: {time.time() - start_time:.2f}s")
EOF
```

## Maintenance Procedures

### 1. Rolling Updates

```bash
# Docker Compose rolling update
./scripts/rolling-update.sh

# Kubernetes rolling update
kubectl set image deployment/api-deployment \
  api=doc-rag/api:v1.1.0 \
  --namespace doc-rag-prod

# Monitor rollout
kubectl rollout status deployment/api-deployment --namespace doc-rag-prod
```

### 2. Database Maintenance

```bash
# PostgreSQL maintenance
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U docrag -d docrag -c "VACUUM ANALYZE;"

# Check database size
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U docrag -d docrag -c "
    SELECT 
      schemaname,
      tablename,
      pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as size
    FROM pg_tables 
    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
    ORDER BY pg_relation_size(schemaname||'.'||tablename) DESC;"

# Backup database
./scripts/backup-database.sh production
```

### 3. Log Management

```bash
# Compress old logs
find /opt/docrag/logs -name "*.log" -mtime +7 -exec gzip {} \;

# Clean up old compressed logs
find /opt/docrag/logs -name "*.log.gz" -mtime +30 -delete

# Monitor log disk usage
du -sh /opt/docrag/logs/*
```

### 4. Certificate Renewal

```bash
# Renew Let's Encrypt certificates
certbot renew --dry-run

# For Docker deployment
certbot renew
cp /etc/letsencrypt/live/your-domain.com/fullchain.pem config/ssl/tls.crt
cp /etc/letsencrypt/live/your-domain.com/privkey.pem config/ssl/tls.key
docker-compose -f docker-compose.production.yml restart nginx

# For Kubernetes deployment
kubectl create secret tls doc-rag-tls \
  --namespace=doc-rag-prod \
  --cert=config/ssl/tls.crt \
  --key=config/ssl/tls.key \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 5. Scaling Operations

```bash
# Scale API service (Docker)
docker-compose -f docker-compose.production.yml up -d --scale api=3

# Scale API service (Kubernetes)
kubectl scale deployment/api-deployment --replicas=5 --namespace doc-rag-prod

# Auto-scaling configuration (Kubernetes)
kubectl apply -f k8s/hpa.yaml

# Monitor scaling
kubectl get hpa --namespace doc-rag-prod --watch
```

### 6. Backup and Restore

```bash
# Full system backup
./scripts/backup-production.sh

# Restore from backup
./scripts/restore-production.sh backup-20240810-120000

# Test restore procedure
./scripts/test-restore.sh backup-20240810-120000
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**: Check logs and dependencies
2. **Database Connection Issues**: Verify credentials and network
3. **High Memory Usage**: Review configuration and optimize queries
4. **SSL Certificate Issues**: Check certificate validity and paths
5. **Performance Degradation**: Monitor metrics and scale resources

### Support Contacts

- **Operations**: ops@your-domain.com
- **Engineering**: engineering@your-domain.com  
- **Security**: security@your-domain.com
- **On-call**: +1-555-0100

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-08-10  
**Next Review**: 2025-09-10