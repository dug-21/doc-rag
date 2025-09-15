# Rust Application Configuration and Deployment Guide

## Overview

This document describes the corrected Rust application configuration and build system designed for robust Docker deployment with proper environment management, database connections, and service lifecycle management.

## Architecture

### Core Components

1. **Configuration Management** (`src/integration/src/config.rs`)
   - Environment-aware configuration loading
   - Docker detection and automatic environment setup
   - Database connection management with retry logic
   - Validation and health checks

2. **Service Manager** (`src/integration/src/service.rs`)
   - Robust service lifecycle management
   - Graceful startup and shutdown
   - Signal handling for containers
   - Pre-flight checks and resource validation

3. **API Gateway** (`src/integration/src/api.rs`)
   - Health and metrics endpoints
   - Query processing endpoints
   - Proper error handling and responses

## Configuration System

### Environment Detection

The system automatically detects Docker environments using:
- `APP_ENV=docker` environment variable
- `DOCKER_ENV` environment variable presence
- `/.dockerenv` file existence

### Configuration Loading Priority

1. **Config File** (if `CONFIG_FILE` env var is set)
2. **Docker Defaults** (if Docker environment detected)
3. **Environment Variables** (override defaults)
4. **Default Configuration** (fallback)

### Environment-Specific Configurations

#### Development (`config/environments/development.toml`)
```toml
[server]
host = "127.0.0.1"
port = 8080

[database]
neo4j_uri = "bolt://localhost:7687"
mongodb_uri = "mongodb://localhost:27017/neurosymbolic-rag-dev"
redis_url = "redis://localhost:6379"
qdrant_url = "http://localhost:6333"

[logging]
level = "debug"
format = "pretty"
```

#### Production (`config/environments/production.toml`)
```toml
[server]
host = "0.0.0.0"
port = 8080

[database]
neo4j_uri = "${NEO4J_URI}"
mongodb_uri = "${MONGODB_URI}"
redis_url = "${REDIS_URL}"
qdrant_url = "${QDRANT_URL}"

[logging]
level = "warn"
format = "json"

[security]
enable_tls = true
api_key_required = true
```

#### Docker (`config/docker.toml`)
```toml
[server]
host = "0.0.0.0"
port = 8080

[database]
neo4j_uri = "bolt://neo4j:7687"
mongodb_uri = "mongodb://mongodb:27017/neurosymbolic-rag"
redis_url = "redis://redis:6379"
qdrant_url = "http://qdrant:6333"

[logging]
level = "info"
format = "json"
```

## Database Connection Management

### Retry Logic
- **Attempts**: Configurable retry attempts (default: 3-5)
- **Delay**: Exponential backoff with configurable base delay
- **Timeout**: Per-connection timeout with circuit breaker pattern

### Health Checks
- **Service Readiness**: Wait for database services during startup
- **Connection Validation**: Test connections before service start
- **Continuous Monitoring**: Periodic health checks during runtime

### Connection Pooling
- **Pool Size**: Environment-specific pool sizing
- **Timeout**: Connection acquisition timeouts
- **Lifecycle**: Proper connection cleanup on shutdown

## Service Lifecycle Management

### Startup Sequence

1. **Initialization**
   - Load configuration with environment detection
   - Run pre-flight checks
   - Validate system resources

2. **Pre-flight Checks**
   - Configuration validation
   - Database service availability
   - Required directory creation
   - System resource validation

3. **Service Start**
   - Initialize system integration
   - Start all components concurrently
   - Setup signal handlers
   - Update service state

4. **Health Monitoring**
   - Continuous health checks
   - Component status tracking
   - Metrics collection

### Shutdown Sequence

1. **Signal Handling**
   - SIGTERM and SIGINT handling
   - Graceful shutdown initiation

2. **Component Shutdown**
   - Stop API gateway
   - Stop health monitors
   - Stop processing pipeline
   - Close database connections

3. **Cleanup**
   - Release resources
   - Final logging
   - State cleanup

## Docker Integration

### Build Configuration

The Dockerfile uses multi-stage builds for optimization:

```dockerfile
# Builder stage
FROM rust:1.75 AS builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/*/Cargo.toml ./src/*/
# Build dependencies separately for caching
RUN cargo build --release --bin integration-server

# Runtime stage  
FROM debian:bookworm-slim
COPY --from=builder /app/target/release/integration-server /app/neurosymbolic-rag
```

### Environment Variables

```bash
# Core configuration
APP_ENV=docker
DOCKER_ENV=1
RUST_LOG=info
RUST_BACKTRACE=1

# Database connections
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
MONGODB_URI=mongodb://mongodb:27017/neurosymbolic-rag
REDIS_URL=redis://redis:6379
QDRANT_URL=http://qdrant:6333

# Service configuration
DB_CONNECTION_TIMEOUT_SECS=60
DB_CONNECTION_POOL_SIZE=5
DB_CONNECTION_RETRY_ATTEMPTS=5
DB_CONNECTION_RETRY_DELAY_MS=2000
```

### Health Checks

```bash
# Docker health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

## API Endpoints

### Health Monitoring
- `GET /health` - System health status
- `GET /metrics` - System metrics
- `GET /components/{component}/health` - Component health

### Core Functionality
- `POST /api/v1/query` - Process RAG queries
- `POST /api/v1/upload` - Document upload

### Administrative
- `POST /admin/shutdown` - Graceful shutdown
- `POST /admin/reload` - Configuration reload

## Error Handling

### Error Classification
- **Client Errors** (4xx): Validation, authentication, not found
- **Server Errors** (5xx): Internal errors, timeouts, circuit breaker

### Recovery Strategies
- **Retry with Exponential Backoff**: Network and timeout errors
- **Circuit Breaker**: Component failures
- **Graceful Degradation**: Service unavailability
- **Fail Fast**: Validation and authentication errors

### Error Context
All errors include:
- Error code for programmatic handling
- Human-readable message
- Recovery suggestions
- Request ID for tracing
- Component information

## Observability

### Logging
- **Structured Logging**: JSON format for production
- **Log Levels**: Environment-specific log levels
- **Request Tracing**: Request ID tracking
- **Component Logging**: Per-component log filtering

### Metrics
- **System Metrics**: CPU, memory, disk usage
- **Application Metrics**: Request counts, response times
- **Database Metrics**: Connection pool status, query performance
- **Custom Metrics**: Business logic metrics

### Tracing
- **Distributed Tracing**: OpenTelemetry integration
- **Jaeger Integration**: Trace collection and analysis
- **Request Correlation**: Cross-component request tracking

## Deployment Strategies

### Development
```bash
# Local development
cargo run --bin integration-server

# With specific environment
CONFIG_FILE=config/environments/development.toml cargo run --bin integration-server
```

### Docker Compose
```bash
# Start all services
docker-compose up -d

# Check health
docker-compose ps
curl http://localhost:8080/health
```

### Production Deployment
```bash
# Build production image
docker build -t neurosymbolic-rag:latest .

# Deploy with environment-specific configuration
docker run -d \
  --name neurosymbolic-rag \
  -p 8080:8080 \
  -p 9090:9090 \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e MONGODB_URI=mongodb://mongodb:27017/neurosymbolic-rag \
  neurosymbolic-rag:latest
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   - Check database connectivity
   - Verify configuration validity
   - Check port availability
   - Review logs for specific errors

2. **Database Connection Failures**
   - Verify database service availability
   - Check connection strings
   - Review network connectivity
   - Validate credentials

3. **High Memory Usage**
   - Check connection pool sizes
   - Review cache configurations
   - Monitor for memory leaks
   - Adjust resource limits

### Debug Commands

```bash
# Check service status
curl http://localhost:8080/health

# View metrics
curl http://localhost:8080/metrics

# Check logs
docker logs neurosymbolic-rag

# Test database connectivity
docker exec neurosymbolic-rag curl -f http://qdrant:6333/health
```

## Security Considerations

### Production Security
- Enable TLS for all connections
- Use secure credential management
- Implement API key authentication
- Configure proper CORS policies
- Enable audit logging

### Network Security
- Use internal networks for database connections
- Implement proper firewall rules
- Use secure database connections
- Enable connection encryption

## Performance Optimization

### Configuration Tuning
- Connection pool sizing
- Timeout configurations
- Circuit breaker thresholds
- Cache configurations

### Resource Management
- Memory limits
- CPU allocation
- Disk I/O optimization
- Network buffer tuning

## Monitoring and Alerting

### Key Metrics to Monitor
- Service availability
- Response times
- Error rates
- Database connection health
- Resource utilization

### Alert Thresholds
- Response time > 2 seconds
- Error rate > 5%
- Memory usage > 80%
- Database connections > 80% of pool

This configuration system provides a robust, Docker-compatible deployment strategy with proper error handling, monitoring, and operational capabilities for production use.