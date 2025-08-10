# Doc-RAG Production Troubleshooting Guide

## Table of Contents
1. [General Diagnostic Commands](#general-diagnostic-commands)
2. [Service-Specific Issues](#service-specific-issues)
3. [Database Problems](#database-problems)
4. [Performance Issues](#performance-issues)
5. [Security Incidents](#security-incidents)
6. [Network and Connectivity](#network-and-connectivity)
7. [Container and Kubernetes Issues](#container-and-kubernetes-issues)
8. [Monitoring and Alerting](#monitoring-and-alerting)

## General Diagnostic Commands

### System Health Check
```bash
# Check all service status (Docker)
docker-compose -f docker-compose.production.yml ps

# Check all pod status (Kubernetes)
kubectl get pods -n doc-rag-prod -o wide

# Check system resources
top -bn1 | head -20
free -h
df -h

# Check network connectivity
ss -tulnp | grep -E "(8080|5432|6379|6333|9000)"

# Check log files
tail -f /opt/docrag/logs/*.log
```

### Service Logs Analysis
```bash
# Get last 100 lines of all services
for service in api security-service query-processor response-generator; do
    echo "=== $service logs ==="
    docker logs --tail 100 doc-rag-${service}-prod
    echo
done

# Search for errors across all logs
docker-compose -f docker-compose.production.yml logs | grep -i error | tail -20

# Real-time log monitoring
docker-compose -f docker-compose.production.yml logs -f --tail 100
```

## Service-Specific Issues

### API Service Issues

**Symptom**: API returning 503 Service Unavailable

**Diagnosis**:
```bash
# Check API service status
curl -f http://localhost:8080/health

# Check API service logs
docker logs doc-rag-api-prod --tail 50

# Check database connectivity from API container
docker exec doc-rag-api-prod pg_isready -h postgres -p 5432 -U docrag

# Check Redis connectivity
docker exec doc-rag-api-prod redis-cli -h redis -p 6379 -a $REDIS_PASSWORD ping
```

**Solutions**:
1. Restart API service: `docker-compose -f docker-compose.production.yml restart api`
2. Check database connections and credentials
3. Verify environment variables are set correctly
4. Check available memory and CPU resources

---

**Symptom**: High API latency (>2 seconds)

**Diagnosis**:
```bash
# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/api/health

# Create curl-format.txt:
cat > curl-format.txt << 'EOF'
     time_namelookup:  %{time_namelookup}s\n
        time_connect:  %{time_connect}s\n
     time_appconnect:  %{time_appconnect}s\n
    time_pretransfer:  %{time_pretransfer}s\n
       time_redirect:  %{time_redirect}s\n
  time_starttransfer:  %{time_starttransfer}s\n
                     ----------\n
          time_total:  %{time_total}s\n
EOF

# Check database query performance
docker exec doc-rag-postgres-prod psql -U docrag -d docrag -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;"
```

**Solutions**:
1. Scale API service horizontally
2. Optimize database queries
3. Increase connection pool size
4. Review and optimize code paths

### Security Service Issues

**Symptom**: Authentication failures

**Diagnosis**:
```bash
# Check security service logs
docker logs doc-rag-security-service-prod --tail 50

# Test JWT token generation
curl -X POST http://localhost:8087/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@docrag.com","password":"test"}'

# Check Redis connectivity for rate limiting
docker exec doc-rag-security-service-prod redis-cli -h redis -p 6379 -a $REDIS_PASSWORD info
```

**Solutions**:
1. Verify JWT secret is correctly configured
2. Check Redis connectivity for session storage
3. Verify user credentials in database
4. Check rate limiting configuration

### Query Processor Issues

**Symptom**: Query processing timeouts

**Diagnosis**:
```bash
# Check query processor health
curl -f http://localhost:8084/health

# Monitor query processing metrics
curl http://localhost:8084/metrics | grep query_processing

# Check vector database connectivity
curl http://localhost:6333/collections
```

**Solutions**:
1. Increase processing timeout limits
2. Check vector database performance
3. Optimize query processing algorithms
4. Scale query processor service

## Database Problems

### PostgreSQL Issues

**Symptom**: Database connection errors

**Diagnosis**:
```bash
# Check PostgreSQL status
docker exec doc-rag-postgres-prod pg_isready -U docrag

# Check active connections
docker exec doc-rag-postgres-prod psql -U docrag -d docrag -c "
SELECT count(*), state 
FROM pg_stat_activity 
GROUP BY state;"

# Check for blocking queries
docker exec doc-rag-postgres-prod psql -U docrag -d docrag -c "
SELECT pid, wait_event, query_start, query
FROM pg_stat_activity 
WHERE wait_event IS NOT NULL
ORDER BY query_start;"

# Check database size
docker exec doc-rag-postgres-prod psql -U docrag -d docrag -c "
SELECT pg_size_pretty(pg_database_size('docrag'));"
```

**Solutions**:
1. Increase max_connections in postgresql.conf
2. Kill blocking queries: `SELECT pg_terminate_backend(pid);`
3. Run VACUUM ANALYZE to optimize database
4. Check and optimize slow queries

---

**Symptom**: High database CPU usage

**Diagnosis**:
```bash
# Check top queries by CPU time
docker exec doc-rag-postgres-prod psql -U docrag -d docrag -c "
SELECT query, calls, total_time, mean_time, rows
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;"

# Check for missing indexes
docker exec doc-rag-postgres-prod psql -U docrag -d docrag -c "
SELECT schemaname, tablename, seq_scan, seq_tup_read, idx_scan, idx_tup_fetch
FROM pg_stat_user_tables 
WHERE seq_scan > idx_scan 
ORDER BY seq_tup_read DESC;"
```

**Solutions**:
1. Add missing indexes on frequently queried columns
2. Optimize expensive queries
3. Consider read replicas for read-heavy workloads
4. Increase database resources (CPU, memory)

### Redis Issues

**Symptom**: Redis memory issues

**Diagnosis**:
```bash
# Check Redis memory usage
docker exec doc-rag-redis-prod redis-cli -a $REDIS_PASSWORD info memory

# Check key statistics
docker exec doc-rag-redis-prod redis-cli -a $REDIS_PASSWORD info keyspace

# Check eviction policy
docker exec doc-rag-redis-prod redis-cli -a $REDIS_PASSWORD config get maxmemory-policy

# Monitor slow commands
docker exec doc-rag-redis-prod redis-cli -a $REDIS_PASSWORD slowlog get 10
```

**Solutions**:
1. Increase Redis memory limit
2. Implement key expiration policies
3. Use appropriate data structures
4. Consider Redis Cluster for scaling

### MongoDB Issues

**Symptom**: MongoDB performance degradation

**Diagnosis**:
```bash
# Check MongoDB status
docker exec doc-rag-mongodb mongosh --eval "db.adminCommand('serverStatus')"

# Check slow operations
docker exec doc-rag-mongodb mongosh --eval "db.adminCommand('currentOp')"

# Check index usage
docker exec doc-rag-mongodb mongosh docrag --eval "db.documents.getIndexes()"

# Check database statistics
docker exec doc-rag-mongodb mongosh docrag --eval "db.stats()"
```

**Solutions**:
1. Add appropriate indexes
2. Optimize query patterns
3. Consider sharding for large datasets
4. Increase MongoDB memory allocation

### Qdrant Vector Database Issues

**Symptom**: Vector search performance issues

**Diagnosis**:
```bash
# Check Qdrant health
curl http://localhost:6333/health

# Check collection information
curl http://localhost:6333/collections

# Check collection statistics
curl "http://localhost:6333/collections/documents"

# Monitor search performance
curl -X POST "http://localhost:6333/collections/documents/points/search" \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3], "limit": 10}' \
  -w "Time: %{time_total}s\n"
```

**Solutions**:
1. Optimize vector indexing parameters
2. Increase Qdrant memory allocation
3. Use appropriate similarity metrics
4. Consider collection sharding

## Performance Issues

### High CPU Usage

**Diagnosis**:
```bash
# Find CPU-intensive processes
top -bn1 | head -20

# Check per-container CPU usage
docker stats --no-stream

# Profile application (if profiling enabled)
curl http://localhost:8080/debug/pprof/profile?seconds=30
```

**Solutions**:
1. Scale services horizontally
2. Optimize CPU-intensive algorithms
3. Implement caching for expensive operations
4. Use async processing for heavy tasks

### Memory Issues

**Diagnosis**:
```bash
# Check system memory
free -h

# Check container memory usage
docker stats --no-stream

# Check for memory leaks
for container in $(docker ps --format "table {{.Names}}" | grep doc-rag); do
    echo "=== $container ==="
    docker exec $container ps aux --sort=-%mem | head -10
done
```

**Solutions**:
1. Increase container memory limits
2. Implement proper garbage collection
3. Optimize data structures
4. Use memory profiling tools

### Disk I/O Issues

**Diagnosis**:
```bash
# Check disk usage
df -h

# Monitor disk I/O
iostat -x 1 10

# Check database I/O
docker exec doc-rag-postgres-prod psql -U docrag -d docrag -c "
SELECT schemaname, tablename, heap_blks_read, heap_blks_hit
FROM pg_statio_user_tables
ORDER BY heap_blks_read DESC;"
```

**Solutions**:
1. Use faster storage (SSD/NVMe)
2. Optimize database queries
3. Implement proper indexing
4. Consider database partitioning

## Security Incidents

### Suspicious Activity

**Diagnosis**:
```bash
# Check for failed authentication attempts
docker logs doc-rag-security-service-prod | grep -i "authentication failed"

# Check rate limiting logs
docker logs doc-rag-nginx-prod | grep "429"

# Check for suspicious IP addresses
docker logs doc-rag-nginx-prod | awk '{print $1}' | sort | uniq -c | sort -nr | head -20

# Check security headers
curl -I https://your-domain.com
```

**Response Actions**:
1. Block suspicious IP addresses
2. Rotate JWT secrets if compromised
3. Enable additional logging
4. Notify security team

### DDoS Attack

**Diagnosis**:
```bash
# Check connection counts
ss -s

# Monitor incoming connections
netstat -an | grep :80 | wc -l
netstat -an | grep :443 | wc -l

# Check Nginx rate limiting
docker logs doc-rag-nginx-prod | grep "limiting requests"
```

**Response Actions**:
1. Enable DDoS protection at load balancer
2. Implement aggressive rate limiting
3. Use cloud-based DDoS protection
4. Scale infrastructure if needed

## Network and Connectivity

### Service Discovery Issues

**Diagnosis**:
```bash
# Test internal service connectivity
docker exec doc-rag-api-prod nslookup postgres
docker exec doc-rag-api-prod telnet redis 6379

# Check network configuration
docker network ls
docker network inspect doc-rag-network
```

**Solutions**:
1. Verify service names match compose file
2. Check network configuration
3. Restart Docker daemon if needed
4. Verify firewall rules

### Load Balancer Issues

**Diagnosis**:
```bash
# Check Nginx configuration
docker exec doc-rag-nginx-prod nginx -t

# Check upstream server health
curl -H "Host: your-domain.com" http://localhost/health

# Check SSL certificate
openssl s_client -connect your-domain.com:443 -servername your-domain.com
```

**Solutions**:
1. Fix Nginx configuration syntax
2. Update upstream server definitions
3. Renew SSL certificates
4. Check backend service health

## Container and Kubernetes Issues

### Docker Issues

**Diagnosis**:
```bash
# Check Docker daemon status
systemctl status docker

# Check container resource usage
docker stats --no-stream

# Check Docker logs
journalctl -u docker.service --since "1 hour ago"

# Check for container exits
docker ps -a | grep -v "Up"
```

**Solutions**:
1. Restart Docker daemon
2. Increase container resource limits
3. Check disk space for Docker data
4. Clean up unused containers and images

### Kubernetes Issues

**Diagnosis**:
```bash
# Check pod status
kubectl get pods -n doc-rag-prod -o wide

# Check pod events
kubectl describe pod <pod-name> -n doc-rag-prod

# Check resource usage
kubectl top pods -n doc-rag-prod

# Check service endpoints
kubectl get endpoints -n doc-rag-prod
```

**Solutions**:
1. Scale deployments: `kubectl scale deployment/api-deployment --replicas=3`
2. Check resource quotas and limits
3. Verify service selectors match pod labels
4. Check cluster node health

## Monitoring and Alerting

### Prometheus Issues

**Diagnosis**:
```bash
# Check Prometheus targets
curl http://localhost:9091/api/v1/targets

# Check Prometheus configuration
curl http://localhost:9091/api/v1/status/config

# Check alert rules
curl http://localhost:9091/api/v1/rules
```

**Solutions**:
1. Verify target service endpoints
2. Check service discovery configuration
3. Reload Prometheus configuration
4. Check alert rule syntax

### Grafana Issues

**Diagnosis**:
```bash
# Check Grafana health
curl http://localhost:3000/api/health

# Check data source connectivity
curl -u admin:password http://localhost:3000/api/datasources

# Check dashboard configuration
docker logs doc-rag-grafana-prod --tail 50
```

**Solutions**:
1. Verify Prometheus data source URL
2. Check dashboard JSON syntax
3. Restart Grafana service
4. Check authentication configuration

## Emergency Procedures

### Service Recovery Checklist

1. **Assess Impact**
   - Check affected services and users
   - Estimate recovery time
   - Notify stakeholders

2. **Immediate Response**
   - Stop traffic to affected services
   - Scale healthy services
   - Enable circuit breakers

3. **Investigation**
   - Collect logs and metrics
   - Identify root cause
   - Document findings

4. **Recovery**
   - Apply fixes
   - Gradually restore traffic
   - Monitor closely

5. **Post-Incident**
   - Conduct post-mortem
   - Update documentation
   - Implement preventive measures

### Escalation Contacts

- **L1 Support**: ops@docrag.com, +1-555-0100
- **L2 Engineering**: engineering@docrag.com, +1-555-0101  
- **L3 Security**: security@docrag.com, +1-555-0102
- **Management**: cto@docrag.com, +1-555-0103

---

**Document Version**: 1.0.0  
**Last Updated**: 2025-08-10  
**Next Review**: 2025-09-10