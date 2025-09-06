# Phase 3: CI/CD Architecture & Automation Framework

## Executive Summary

This document outlines the comprehensive CI/CD architecture for Phase 3 of the Document RAG system, focusing on production-ready automation, security, monitoring, and deployment strategies. The architecture implements DevSecOps best practices with multi-stage pipelines, automated testing, security scanning, and observability.

## Architecture Overview

```mermaid
graph TB
    subgraph "Development Environment"
        DEV[Developer Workstation]
        PRE[Pre-commit Hooks]
        DEV --> PRE
    end

    subgraph "Source Control"
        GIT[GitHub Repository]
        PRE --> GIT
    end

    subgraph "CI/CD Pipeline"
        subgraph "Build Stage"
            LINT[Linting & Format]
            SEC[Security Scan]
            TEST[Unit Tests]
            BUILD[Build Artifacts]
        end

        subgraph "Test Stage"
            INTEGRATION[Integration Tests]
            E2E[E2E Tests]
            PERF[Performance Tests]
            LOAD[Load Tests]
        end

        subgraph "Security Stage"
            SAST[Static Analysis]
            DAST[Dynamic Analysis]
            DEPS[Dependency Scan]
            SECRETS[Secret Scan]
        end

        subgraph "Package Stage"
            DOCKER[Docker Build]
            REGISTRY[Container Registry]
            HELM[Helm Charts]
            ARTIFACTS[Release Artifacts]
        end

        GIT --> LINT
        LINT --> SEC
        SEC --> TEST
        TEST --> BUILD
        BUILD --> INTEGRATION
        INTEGRATION --> E2E
        E2E --> PERF
        PERF --> LOAD
        LOAD --> SAST
        SAST --> DAST
        DAST --> DEPS
        DEPS --> SECRETS
        SECRETS --> DOCKER
        DOCKER --> REGISTRY
        REGISTRY --> HELM
        HELM --> ARTIFACTS
    end

    subgraph "Deployment Environments"
        subgraph "Staging"
            STAGE_K8S[Kubernetes Staging]
            STAGE_DB[Staging Database]
            STAGE_MONITOR[Staging Monitoring]
        end

        subgraph "Production"
            PROD_K8S[Kubernetes Production]
            PROD_DB[Production Database]
            PROD_MONITOR[Production Monitoring]
            CANARY[Canary Deployment]
            BLUE_GREEN[Blue-Green Strategy]
        end

        ARTIFACTS --> STAGE_K8S
        STAGE_K8S --> PROD_K8S
        PROD_K8S --> CANARY
        CANARY --> BLUE_GREEN
    end

    subgraph "Monitoring & Observability"
        PROMETHEUS[Prometheus]
        GRAFANA[Grafana]
        JAEGER[Jaeger Tracing]
        ALERTS[Alert Manager]
        LOGS[Log Aggregation]
    end

    PROD_K8S --> PROMETHEUS
    PROMETHEUS --> GRAFANA
    PROD_K8S --> JAEGER
    PROMETHEUS --> ALERTS
    PROD_K8S --> LOGS
```

## 1. CI/CD Pipeline Design

### 1.1 Pipeline Stages

#### Stage 1: Code Quality & Linting (2-3 minutes)
```yaml
# .github/workflows/quality.yml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - name: Rust Format Check
        run: cargo fmt --all -- --check
      
      - name: Clippy Analysis
        run: cargo clippy --workspace --all-targets --all-features -- -D warnings
      
      - name: Documentation Check
        run: cargo doc --workspace --all-features --no-deps
      
      - name: License Compliance
        run: cargo license --json | jq '.[] | select(.license == null)'
```

#### Stage 2: Security Scanning (3-5 minutes)
```yaml
# Security-first approach with multiple scanning tools
security:
  parallel:
    - name: Dependency Audit
      run: cargo audit --deny warnings
    
    - name: Secret Scanning
      uses: trufflesecurity/trufflehog@v3
      
    - name: License Check
      uses: fossa-contrib/fossa-action@v2
      
    - name: Code Quality
      uses: github/super-linter@v4
```

#### Stage 3: Multi-Matrix Testing (8-15 minutes)
```yaml
test:
  strategy:
    matrix:
      os: [ubuntu-latest, macos-latest, windows-latest]
      rust: [stable, beta, 1.75.0]  # MSRV
      features: [default, all-features, no-default-features]
  
  services:
    postgres: { image: postgres:16, options: --health-cmd pg_isready }
    redis: { image: redis:7-alpine, options: --health-cmd "redis-cli ping" }
    qdrant: { image: qdrant/qdrant:v1.7.4 }
    
  parallel_tests:
    - Unit Tests: cargo test --lib --bins
    - Integration Tests: cargo test --test '*'
    - Doc Tests: cargo test --doc
    - Benchmark Tests: cargo bench --no-run
```

#### Stage 4: Container & Package Building (5-8 minutes)
```yaml
docker:
  strategy:
    matrix:
      service: [api, chunker, embedder, retriever, query-processor, response-generator]
      platform: [linux/amd64, linux/arm64]
  
  build_optimization:
    - Multi-stage builds with dependency caching
    - BuildKit for advanced caching strategies  
    - Registry caching with GitHub Container Registry
    - Distroless base images for security
    - Image size optimization (< 100MB per service)
```

#### Stage 5: Integration & E2E Testing (10-15 minutes)
```yaml
integration:
  deployment:
    - Docker Compose stack deployment
    - Service health verification
    - Database migration testing
    - API contract validation
    - Performance baseline verification
  
  e2e_tests:
    - Document ingestion workflow
    - Query processing pipeline
    - Response generation accuracy
    - Multi-user concurrent access
    - Failure recovery scenarios
```

### 1.2 Advanced Pipeline Features

#### Parallel Test Execution
```yaml
# Advanced parallel test execution for 4x speed improvement
test-parallel:
  strategy:
    matrix:
      test_group: [unit-core, unit-api, unit-storage, integration-basic, integration-advanced, e2e]
  
  steps:
    - name: Run Test Group
      run: |
        case "${{ matrix.test_group }}" in
          "unit-core") cargo test --package chunker --package embedder --package retriever ;;
          "unit-api") cargo test --package api --package mcp-adapter ;;
          "unit-storage") cargo test --package storage --package query-processor ;;
          "integration-basic") cargo test --test basic_integration ;;
          "integration-advanced") cargo test --test advanced_integration ;;
          "e2e") cargo test --test e2e_workflow ;;
        esac
      timeout-minutes: 20
```

#### Dynamic Pipeline Configuration
```yaml
# Pipeline adapts based on changes detected
pipeline-intelligence:
  detect-changes:
    outputs:
      api-changed: ${{ steps.changes.outputs.api }}
      chunker-changed: ${{ steps.changes.outputs.chunker }}
      embedder-changed: ${{ steps.changes.outputs.embedder }}
      # ... other services
  
  conditional-stages:
    needs: detect-changes
    if: needs.detect-changes.outputs.api-changed == 'true'
    # Only run relevant tests and builds
```

### 1.3 Performance Optimization

#### Cache Strategy
```yaml
# Multi-layer caching for maximum efficiency
cache_optimization:
  rust_cache:
    - uses: Swatinem/rust-cache@v2
      with:
        shared-key: "ci-${{ matrix.os }}-${{ matrix.rust }}"
        cache-targets: "~/.cargo/registry,~/.cargo/git,target/"
        
  docker_cache:
    - uses: docker/setup-buildx-action@v3
      with:
        buildkitd-flags: --allow-insecure-entitlement security.insecure
    - uses: docker/build-push-action@v5
      with:
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

## 2. Automation Framework

### 2.1 Pre-commit Hooks Framework

#### Setup Script
```bash
#!/bin/bash
# scripts/setup-pre-commit.sh

# Install pre-commit hooks
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e

echo "🔍 Running pre-commit checks..."

# Format check
echo "📝 Checking code formatting..."
if ! cargo fmt --all -- --check; then
    echo "❌ Code formatting issues found. Run: cargo fmt --all"
    exit 1
fi

# Clippy check
echo "🔧 Running Clippy analysis..."
if ! cargo clippy --workspace --all-targets --all-features -- -D warnings; then
    echo "❌ Clippy warnings found. Fix them before committing."
    exit 1
fi

# Security audit
echo "🛡️ Running security audit..."
if ! cargo audit; then
    echo "❌ Security vulnerabilities found. Address them before committing."
    exit 1
fi

# License check
echo "📋 Checking license compliance..."
./scripts/check-licenses.sh

# Test critical paths
echo "🧪 Running critical tests..."
cargo test --lib --bins --tests --quiet

echo "✅ All pre-commit checks passed!"
EOF

chmod +x .git/hooks/pre-commit
echo "✅ Pre-commit hooks installed successfully!"
```

#### License Compliance Check
```bash
#!/bin/bash
# scripts/check-licenses.sh

ALLOWED_LICENSES=("MIT" "Apache-2.0" "BSD-3-Clause" "ISC" "Apache-2.0 WITH LLVM-exception")

echo "🔍 Checking license compliance..."

# Generate license report
cargo license --json > license_report.json

# Check for forbidden licenses
FORBIDDEN=$(jq -r '.[] | select(.license | IN("GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0") | not) | .name + " (" + .license + ")"' license_report.json)

if [ -n "$FORBIDDEN" ]; then
    echo "❌ Forbidden licenses detected:"
    echo "$FORBIDDEN"
    rm license_report.json
    exit 1
fi

echo "✅ All licenses are compliant"
rm license_report.json
```

### 2.2 Automated Code Quality Checks

#### Quality Gates Configuration
```yaml
# .github/quality-gates.yml
quality_gates:
  code_coverage:
    minimum: 85%
    exclude_paths: ["tests/", "benches/", "examples/"]
  
  complexity:
    max_cyclomatic: 15
    max_cognitive: 25
  
  duplication:
    max_percentage: 3%
    min_token_length: 100
  
  security:
    sast_severity: "HIGH"
    dependency_severity: "MEDIUM" 
    secrets_scan: true
  
  performance:
    build_time_max: "10m"
    test_time_max: "15m"
    binary_size_max: "50MB"
```

#### Automated Code Review Bot
```yaml
# .github/workflows/code-review-bot.yml
name: Automated Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  ai-code-review:
    runs-on: ubuntu-latest
    steps:
      - uses: coderabbitai/ai-pr-reviewer@latest
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        with:
          debug: false
          review_simple_changes: true
          review_comment_lgtm: true
          path_filters: |
            !dist/**
            !**/*.lock
            !**/*.md
```

### 2.3 Security Scanning Framework

#### Multi-layered Security Pipeline
```yaml
# .github/workflows/security-comprehensive.yml
name: Security Comprehensive Scan

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  pull_request:
    branches: [main, develop]

jobs:
  sast-analysis:
    name: Static Application Security Testing
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: CodeQL Analysis
        uses: github/codeql-action/init@v3
        with:
          languages: rust
          
      - name: Build for Analysis
        run: cargo build --workspace --all-features
        
      - name: CodeQL Analysis
        uses: github/codeql-action/analyze@v3
      
      - name: Semgrep Security Scan
        uses: returntocorp/semgrep-action@v1
        with:
          config: >-
            p/security-audit
            p/secrets
            p/rust
            
      - name: Rust Security Audit
        run: |
          cargo install cargo-audit
          cargo audit --deny warnings --json > audit-report.json
          
      - name: Upload Security Reports
        uses: actions/upload-artifact@v4
        with:
          name: security-reports
          path: |
            audit-report.json
            semgrep-results.json
            codeql-results.sarif

  dependency-vulnerability-scan:
    name: Dependency Vulnerability Scanning
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Cache Dependencies
        uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target/
          key: deps-${{ runner.os }}-${{ hashFiles('**/Cargo.lock') }}
          
      - name: Install Tools
        run: |
          cargo install cargo-audit
          cargo install cargo-deny
          
      - name: Vulnerability Scan
        run: |
          cargo audit --deny warnings
          cargo deny --all-features check
          
      - name: License Compliance
        run: |
          cargo deny --all-features check licenses
          
  container-security-scan:
    name: Container Security Scanning
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [api, chunker, embedder, retriever]
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Build Container
        run: |
          docker build -f src/${{ matrix.service }}/Dockerfile -t doc-rag-${{ matrix.service }}:security-scan .
          
      - name: Trivy Vulnerability Scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: doc-rag-${{ matrix.service }}:security-scan
          format: 'sarif'
          output: 'trivy-${{ matrix.service }}.sarif'
          
      - name: Grype Vulnerability Scanner
        run: |
          curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
          grype doc-rag-${{ matrix.service }}:security-scan --output json > grype-${{ matrix.service }}.json
          
      - name: Upload Container Scan Results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-${{ matrix.service }}.sarif

  secrets-detection:
    name: Secrets Detection
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: TruffleHog OSS
        uses: trufflesecurity/trufflehog@v3.63.2-beta
        with:
          path: ./
          base: main
          head: HEAD
          extra_args: --debug --only-verified

  dast-analysis:
    name: Dynamic Application Security Testing
    runs-on: ubuntu-latest
    needs: [sast-analysis]
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: docrag_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Setup Test Environment
        run: |
          docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d
          sleep 30
          
      - name: Wait for Services
        run: |
          timeout 120 bash -c 'until curl -f http://localhost:8080/health; do sleep 2; done'
          
      - name: OWASP ZAP Baseline Scan
        uses: zaproxy/action-baseline@v0.10.0
        with:
          target: 'http://localhost:8080'
          rules_file_name: '.zap/rules.tsv'
          cmd_options: '-a'
          
      - name: Nuclei Security Scanner
        run: |
          docker run -v $(pwd):/tmp projectdiscovery/nuclei:latest \
            -t /tmp/.nuclei-templates/ \
            -u http://localhost:8080 \
            -o /tmp/nuclei-report.json -json
            
      - name: Cleanup
        if: always()
        run: docker-compose down -v
```

### 2.4 Performance Benchmarking Automation

#### Comprehensive Performance Suite
```yaml
# .github/workflows/performance-benchmarks.yml
name: Performance Benchmarks

on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Mondays at 4 AM UTC
  workflow_dispatch:
  push:
    branches: [main]
    paths:
      - 'src/**/*.rs'
      - 'Cargo.toml'
      - 'Cargo.lock'

jobs:
  cargo-benchmarks:
    name: Cargo Criterion Benchmarks
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
        
      - name: Cache Dependencies
        uses: Swatinem/rust-cache@v2
        with:
          shared-key: "bench"
          
      - name: Install Criterion
        run: cargo install cargo-criterion
        
      - name: Run Benchmarks
        run: |
          cargo criterion --workspace --message-format=json > benchmark-results.json
          
      - name: Performance Analysis
        run: |
          # Compare with baseline
          python scripts/analyze-performance.py benchmark-results.json
          
      - name: Upload Results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          name: Rust Criterion Benchmark
          tool: 'cargo'
          output-file-path: benchmark-results.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: '200%'
          comment-on-alert: true

  load-testing:
    name: Load Testing
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Start Services
        run: |
          docker-compose -f docker-compose.yml up -d --scale api=3
          sleep 60
          
      - name: K6 Load Testing
        run: |
          docker run --rm --network=host \
            -v $(pwd)/tests/load:/scripts \
            grafana/k6:latest run /scripts/load-test.js \
            --out json=/scripts/results.json
            
      - name: Artillery Load Testing
        run: |
          npm install -g artillery
          artillery run tests/load/artillery-config.yml --output load-test-results.json
          
      - name: JMeter Performance Testing
        run: |
          docker run --rm --network=host \
            -v $(pwd)/tests/jmeter:/jmeter \
            justb4/jmeter -n -t /jmeter/doc-rag-test-plan.jmx \
            -l /jmeter/results.jtl -e -o /jmeter/reports
            
      - name: Performance Report
        run: |
          python scripts/generate-performance-report.py \
            --k6 tests/load/results.json \
            --artillery load-test-results.json \
            --jmeter tests/jmeter/results.jtl \
            --output performance-report.html
            
      - name: Upload Performance Report
        uses: actions/upload-artifact@v4
        with:
          name: performance-report
          path: |
            performance-report.html
            tests/jmeter/reports/
            
      - name: Performance Regression Check
        run: |
          # Compare with historical data
          python scripts/check-performance-regression.py

  memory-profiling:
    name: Memory and Resource Profiling
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Install Profiling Tools
        run: |
          cargo install cargo-profdata
          cargo install valgrind
          sudo apt-get update
          sudo apt-get install -y valgrind heaptrack
          
      - name: Memory Leak Detection
        run: |
          cargo build --release
          valgrind --tool=memcheck --leak-check=full \
            --show-leak-kinds=all --track-origins=yes \
            target/release/api-server --version
            
      - name: Heap Profiling
        run: |
          heaptrack target/release/api-server --version
          heaptrack_print heaptrack.api-server.*.gz > heap-report.txt
          
      - name: CPU Profiling
        run: |
          perf record -g target/release/api-server --version
          perf report --stdio > cpu-profile.txt
```

## 3. Infrastructure as Code

### 3.1 Docker Compose for Local Development

#### Enhanced Development Environment
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # Development overrides
  api:
    build:
      target: development
    volumes:
      - .:/app:cached
      - /app/target  # Anonymous volume for target directory
      - dev_cargo_cache:/usr/local/cargo/registry
    environment:
      - RUST_LOG=debug
      - RUST_BACKTRACE=1
      - HOT_RELOAD=true
    ports:
      - "8080:8080"
      - "8000:8000"  # Debug port
      - "9615:9615"  # Substrate debug
    command: ["cargo", "watch", "-x", "run -- --config /app/config/api-dev.toml"]

  # Development databases with test data
  postgres:
    environment:
      - POSTGRES_DB=docrag_dev
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./scripts/dev-seed-data.sql:/docker-entrypoint-initdb.d/99-seed.sql:ro

  # Development monitoring
  jaeger:
    environment:
      - COLLECTOR_OTLP_ENABLED=true
      - SPAN_STORAGE_TYPE=memory
      - QUERY_BASE_PATH=/jaeger

  # Development tools
  pgadmin:
    image: dpage/pgadmin4:latest
    environment:
      - PGADMIN_DEFAULT_EMAIL=dev@docrag.local
      - PGADMIN_DEFAULT_PASSWORD=dev_password
    ports:
      - "5050:80"
    depends_on:
      - postgres

  redis-commander:
    image: rediscommander/redis-commander:latest
    environment:
      - REDIS_HOSTS=local:redis:6379
      - REDIS_PASSWORD=redis_secret_2024
    ports:
      - "8081:8081"
    depends_on:
      - redis

  qdrant-dashboard:
    image: qdrant/qdrant:v1.7.4
    command: ./qdrant --config-path /qdrant/config/config.yaml
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__WEB_UI__ENABLED=true
    volumes:
      - ./config/qdrant-dev.yaml:/qdrant/config/config.yaml:ro

volumes:
  postgres_dev_data:
    driver: local
  dev_cargo_cache:
    driver: local
```

#### Test Environment Configuration
```yaml
# docker-compose.test.yml
version: '3.8'

services:
  api:
    build:
      target: production
    environment:
      - RUST_LOG=debug
      - DATABASE_URL=postgres://docrag:test_password@postgres:5432/docrag_test
      - REDIS_URL=redis://redis:6379
      - TEST_MODE=true
      - INTEGRATION_TESTS=true
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    command: ["api-server", "--config", "/app/config/api-test.toml"]

  postgres:
    environment:
      - POSTGRES_DB=docrag_test
      - POSTGRES_PASSWORD=test_password
    volumes:
      - ./scripts/test-schema.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro
      - ./scripts/test-fixtures.sql:/docker-entrypoint-initdb.d/02-fixtures.sql:ro

  # Test-specific services
  testcontainers-proxy:
    image: testcontainers/ryuk:latest
    ports:
      - "8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    environment:
      - RYUK_CONNECTION_TIMEOUT=60
      - RYUK_RECONNECTION_TIMEOUT=10

networks:
  default:
    name: docrag-test-network
```

### 3.2 Kubernetes Production Manifests

#### Namespace and Security
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: doc-rag-prod
  labels:
    app.kubernetes.io/name: doc-rag
    security-policy: restricted
  annotations:
    security.alpha.kubernetes.io/seccomp: runtime/default

---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: doc-rag-quota
  namespace: doc-rag-prod
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "10"
    services: "20"
    secrets: "30"
    configmaps: "30"

---
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: doc-rag-psp
  namespace: doc-rag-prod
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
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

#### Advanced Deployment with Blue-Green Strategy
```yaml
# k8s/api-deployment-advanced.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: api-rollout
  namespace: doc-rag-prod
spec:
  replicas: 5
  strategy:
    blueGreen:
      activeService: api-active
      previewService: api-preview
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 30
      prePromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: api-preview
      postPromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: api-active
  selector:
    matchLabels:
      app: doc-rag-api
  template:
    metadata:
      labels:
        app: doc-rag-api
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        vault.hashicorp.com/agent-inject: "true"
        vault.hashicorp.com/agent-inject-secret-config: "secret/doc-rag/api"
    spec:
      serviceAccountName: doc-rag-api
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: api
        image: doc-rag/api:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        env:
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        envFrom:
        - configMapRef:
            name: doc-rag-config
        - secretRef:
            name: doc-rag-secrets
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
            ephemeral-storage: 1Gi
          limits:
            cpu: 2
            memory: 4Gi
            ephemeral-storage: 2Gi
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 2
          timeoutSeconds: 1
          failureThreshold: 30
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1001
          capabilities:
            drop:
            - ALL
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
        - name: logs
          mountPath: /app/logs
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: tmp
        emptyDir:
          sizeLimit: 1Gi
      - name: cache
        emptyDir:
          sizeLimit: 2Gi
      - name: logs
        emptyDir:
          sizeLimit: 1Gi
      - name: config
        configMap:
          name: doc-rag-config
      nodeSelector:
        kubernetes.io/arch: amd64
        node-type: compute
      tolerations:
      - key: "doc-rag"
        operator: "Equal"
        value: "api"
        effect: "NoSchedule"
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - doc-rag-api
            topologyKey: kubernetes.io/hostname
```

#### Horizontal Pod Autoscaler with Custom Metrics
```yaml
# k8s/hpa-custom-metrics.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: doc-rag-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-deployment
  minReplicas: 3
  maxReplicas: 20
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: request_latency_p99
      target:
        type: AverageValue
        averageValue: "500m"  # 500ms
  - type: External
    external:
      metric:
        name: queue_length
        selector:
          matchLabels:
            queue: doc-rag-processing
      target:
        type: AverageValue
        averageValue: "10"
```

### 3.3 Helm Charts for Scalable Deployment

#### Chart Structure
```yaml
# helm/doc-rag/Chart.yaml
apiVersion: v2
name: doc-rag
description: Document RAG System Helm Chart
version: 1.0.0
appVersion: "1.0.0"
keywords:
  - rag
  - document-processing
  - ai
  - rust
home: https://github.com/doc-rag/doc-rag
sources:
  - https://github.com/doc-rag/doc-rag
maintainers:
  - name: Doc-RAG Team
    email: team@doc-rag.io
dependencies:
  - name: postgresql
    version: 12.x.x
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: 18.x.x
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
  - name: qdrant
    version: 1.x.x
    repository: https://qdrant.github.io/helm-charts
    condition: qdrant.enabled
```

#### Advanced Values Configuration
```yaml
# helm/doc-rag/values.yaml
global:
  imageRegistry: "docker.io"
  imageTag: "latest"
  pullPolicy: Always
  storageClass: "fast-ssd"

# Application Configuration
api:
  replicaCount: 3
  image:
    repository: doc-rag/api
    tag: ""
  service:
    type: ClusterIP
    port: 8080
    targetPort: 8080
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
  resources:
    limits:
      cpu: 2
      memory: 4Gi
    requests:
      cpu: 500m
      memory: 1Gi
  nodeSelector:
    node-type: compute
  tolerations:
    - key: "doc-rag"
      operator: "Equal"
      value: "api"
      effect: "NoSchedule"

# Processing Services
chunker:
  enabled: true
  replicaCount: 2
  resources:
    limits:
      nvidia.com/gpu: 1
      cpu: 4
      memory: 8Gi
    requests:
      cpu: 2
      memory: 4Gi
  nodeSelector:
    accelerator: nvidia-tesla-v100

embedder:
  enabled: true
  replicaCount: 2
  resources:
    limits:
      nvidia.com/gpu: 1
      cpu: 4
      memory: 8Gi
    requests:
      cpu: 2
      memory: 4Gi

# Infrastructure
postgresql:
  enabled: true
  auth:
    enablePostgresUser: true
    postgresPassword: ""  # Set via secret
    username: "docrag"
    password: ""  # Set via secret
    database: "docrag"
  primary:
    persistence:
      enabled: true
      size: 100Gi
      storageClass: "fast-ssd"
  metrics:
    enabled: true
    serviceMonitor:
      enabled: true

redis:
  enabled: true
  auth:
    enabled: true
    password: ""  # Set via secret
  master:
    persistence:
      enabled: true
      size: 20Gi
      storageClass: "fast-ssd"
  metrics:
    enabled: true
    serviceMonitor:
      enabled: true

# Monitoring
monitoring:
  prometheus:
    enabled: true
    serviceMonitor:
      enabled: true
      interval: 30s
  grafana:
    enabled: true
    dashboards:
      enabled: true

# Security
security:
  networkPolicies:
    enabled: true
  podSecurityPolicy:
    enabled: true
  rbac:
    create: true

# Ingress
ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: api.doc-rag.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: doc-rag-tls
      hosts:
        - api.doc-rag.com
```

## 4. Deployment Strategies

### 4.1 Blue-Green Deployment

#### Implementation with Argo Rollouts
```yaml
# Blue-Green deployment strategy for zero-downtime deployments
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: api-rollout
spec:
  strategy:
    blueGreen:
      # Service names for active and preview environments
      activeService: api-active
      previewService: api-preview
      
      # Disable automatic promotion (manual approval required)
      autoPromotionEnabled: false
      
      # Time to wait before scaling down old version
      scaleDownDelaySeconds: 30
      
      # Health checks before promotion
      prePromotionAnalysis:
        templates:
        - templateName: api-health-check
        - templateName: api-performance-check
        args:
        - name: service-name
          value: api-preview
      
      # Health checks after promotion
      postPromotionAnalysis:
        templates:
        - templateName: api-monitoring
        args:
        - name: service-name
          value: api-active
```

#### Blue-Green Service Configuration
```yaml
# Services for Blue-Green deployment
apiVersion: v1
kind: Service
metadata:
  name: api-active
spec:
  selector:
    app: doc-rag-api
  ports:
  - port: 8080
    targetPort: 8080

---
apiVersion: v1
kind: Service
metadata:
  name: api-preview
spec:
  selector:
    app: doc-rag-api
  ports:
  - port: 8080
    targetPort: 8080

---
apiVersion: v1
kind: Service
metadata:
  name: api-public
spec:
  selector:
    app: doc-rag-api
  ports:
  - port: 80
    targetPort: 8080
```

### 4.2 Canary Deployment Strategy

#### Progressive Canary with Traffic Splitting
```yaml
# Canary deployment with progressive traffic shifting
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: api-canary-rollout
spec:
  strategy:
    canary:
      # Canary service for new version
      canaryService: api-canary
      
      # Stable service for current version
      stableService: api-stable
      
      # Traffic routing
      trafficRouting:
        istio:
          virtualService:
            name: api-virtualservice
          destinationRule:
            name: api-destinationrule
      
      # Progressive traffic shifting steps
      steps:
      - setWeight: 5   # Start with 5% traffic
      - pause: {}      # Manual approval
      
      - setWeight: 10
      - pause:
          duration: 2m
      
      - setWeight: 20
      - pause:
          duration: 5m
      
      - setWeight: 50
      - pause:
          duration: 10m
      
      - setWeight: 80
      - pause:
          duration: 10m
      
      - setWeight: 100  # Full rollout
      
      # Analysis during canary deployment
      analysis:
        templates:
        - templateName: canary-success-rate
        - templateName: canary-latency
        args:
        - name: canary-hash
          valueFrom:
            podTemplateHashValue: Latest
        startingStep: 2
        interval: 60s
```

#### Canary Analysis Templates
```yaml
# Analysis templates for canary deployments
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: canary-success-rate
spec:
  args:
  - name: canary-hash
  metrics:
  - name: success-rate
    interval: 60s
    count: 5
    successCondition: result[0] >= 0.95
    provider:
      prometheus:
        address: http://prometheus:9090
        query: |
          sum(rate(http_requests_total{status!~"5.*",rollouts_pod_template_hash="{{args.canary-hash}}"}[2m])) /
          sum(rate(http_requests_total{rollouts_pod_template_hash="{{args.canary-hash}}"}[2m]))

---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: canary-latency
spec:
  args:
  - name: canary-hash
  metrics:
  - name: p99-latency
    interval: 60s
    count: 5
    successCondition: result[0] <= 0.5
    provider:
      prometheus:
        address: http://prometheus:9090
        query: |
          histogram_quantile(0.99,
            rate(http_request_duration_seconds_bucket{rollouts_pod_template_hash="{{args.canary-hash}}"}[2m])
          )
```

### 4.3 Multi-Environment Pipeline

#### Environment Promotion Pipeline
```yaml
# .github/workflows/environment-promotion.yml
name: Environment Promotion Pipeline

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target Environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

jobs:
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to Staging
        uses: ./.github/actions/k8s-deploy
        with:
          environment: staging
          image-tag: ${{ github.event.release.tag_name }}
          kubeconfig: ${{ secrets.STAGING_KUBECONFIG }}
          
      - name: Run Smoke Tests
        run: |
          ./scripts/smoke-tests.sh staging
          
      - name: Performance Tests
        run: |
          ./scripts/performance-tests.sh staging

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    environment: production
    needs: deploy-staging
    if: github.event_name == 'release' || inputs.environment == 'production'
    steps:
      - name: Production Deployment Approval
        uses: trstringer/manual-approval@v1
        with:
          secret: ${{ github.TOKEN }}
          approvers: admin-team,senior-devs
          
      - name: Deploy to Production
        uses: ./.github/actions/k8s-deploy
        with:
          environment: production
          image-tag: ${{ github.event.release.tag_name }}
          kubeconfig: ${{ secrets.PRODUCTION_KUBECONFIG }}
          deployment-strategy: blue-green
          
      - name: Verify Production Deployment
        run: |
          ./scripts/production-verification.sh
          
      - name: Update Monitoring
        run: |
          ./scripts/update-monitoring-dashboard.sh production
```

## 5. Monitoring & Observability

### 5.1 Comprehensive Monitoring Stack

#### Prometheus Configuration with Advanced Rules
```yaml
# config/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'doc-rag-prod'
    environment: 'production'

rule_files:
  - "/etc/prometheus/rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Application metrics
  - job_name: 'doc-rag-api'
    static_configs:
      - targets: ['api:9090']
    scrape_interval: 10s
    metrics_path: '/metrics'
    scheme: http
    
  - job_name: 'doc-rag-services'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)

  # Infrastructure metrics
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node-exporter'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - action: replace
        source_labels: [__address__]
        regex: '(.+):10250'
        target_label: __address__
        replacement: '${1}:9100'

  # Custom business metrics
  - job_name: 'business-metrics'
    static_configs:
      - targets: ['business-metrics-exporter:8080']
    scrape_interval: 30s
```

#### Advanced Alerting Rules
```yaml
# config/alerting/alert_rules.yml
groups:
- name: doc-rag.api
  rules:
  - alert: APIHighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
      service: api
      team: platform
    annotations:
      summary: "High error rate detected in API"
      description: "API error rate is {{ $value }} errors per second"
      runbook_url: "https://wiki.doc-rag.io/runbooks/api-high-error-rate"
      
  - alert: APIHighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
    for: 10m
    labels:
      severity: warning
      service: api
      team: platform
    annotations:
      summary: "High latency detected in API"
      description: "95th percentile latency is {{ $value }}s"
      
  - alert: APILowThroughput
    expr: rate(http_requests_total[5m]) < 10
    for: 15m
    labels:
      severity: warning
      service: api
      team: platform
    annotations:
      summary: "Low API throughput detected"
      description: "API throughput is only {{ $value }} requests per second"

- name: doc-rag.infrastructure
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage_percentage > 80
    for: 10m
    labels:
      severity: warning
      component: infrastructure
    annotations:
      summary: "High CPU usage on {{ $labels.instance }}"
      description: "CPU usage is {{ $value }}%"
      
  - alert: HighMemoryUsage
    expr: memory_usage_percentage > 85
    for: 5m
    labels:
      severity: critical
      component: infrastructure
    annotations:
      summary: "High memory usage on {{ $labels.instance }}"
      description: "Memory usage is {{ $value }}%"
      
  - alert: DiskSpaceLow
    expr: disk_free_percentage < 15
    for: 5m
    labels:
      severity: critical
      component: infrastructure
    annotations:
      summary: "Low disk space on {{ $labels.instance }}"
      description: "Only {{ $value }}% disk space remaining"

- name: doc-rag.business
  rules:
  - alert: DocumentProcessingBacklog
    expr: document_processing_queue_length > 1000
    for: 30m
    labels:
      severity: warning
      business_impact: high
    annotations:
      summary: "Large document processing backlog"
      description: "Queue has {{ $value }} documents pending"
      
  - alert: LowQueryAccuracy
    expr: avg(query_accuracy_score) < 0.8
    for: 1h
    labels:
      severity: critical
      business_impact: high
    annotations:
      summary: "Query accuracy has degraded"
      description: "Average accuracy is {{ $value }}"
```

### 5.2 Distributed Tracing with Jaeger

#### Tracing Configuration
```yaml
# config/jaeger-config.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: jaeger-config
data:
  jaeger-config.yml: |
    extensions:
      health_check: {}
      
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318
      jaeger:
        protocols:
          grpc:
            endpoint: 0.0.0.0:14250
          thrift_http:
            endpoint: 0.0.0.0:14268
          thrift_compact:
            endpoint: 0.0.0.0:6831
    
    processors:
      batch:
        timeout: 1s
        send_batch_size: 50
      memory_limiter:
        limit_mib: 512
        
    exporters:
      jaeger:
        endpoint: jaeger-collector:14250
        tls:
          insecure: true
          
    service:
      extensions: [health_check]
      pipelines:
        traces:
          receivers: [otlp, jaeger]
          processors: [memory_limiter, batch]
          exporters: [jaeger]
```

### 5.3 Log Aggregation with ELK Stack

#### Fluentd Configuration
```yaml
# config/fluentd/fluent.conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<source>
  @type tail
  path /var/log/containers/*.log
  pos_file /var/log/fluentd-containers.log.pos
  tag kubernetes.*
  format json
  read_from_head true
</source>

<filter kubernetes.**>
  @type kubernetes_metadata
  @id filter_kube_metadata
  kubernetes_url "#{ENV['FLUENT_FILTER_KUBERNETES_URL'] || 'https://' + ENV.fetch('KUBERNETES_SERVICE_HOST') + ':' + ENV.fetch('KUBERNETES_SERVICE_PORT') + '/api'}"
  verify_ssl "#{ENV['KUBERNETES_VERIFY_SSL'] || true}"
  ca_file "#{ENV['KUBERNETES_CA_FILE']}"
  skip_labels false
  skip_container_metadata false
  skip_master_url false
  skip_namespace_metadata false
</filter>

<filter doc-rag.**>
  @type parser
  key_name message
  reserve_data true
  remove_key_name_field false
  <parse>
    @type json
  </parse>
</filter>

<filter doc-rag.api>
  @type record_transformer
  <record>
    service_name api
    log_level ${record["level"]}
    request_id ${record["request_id"]}
    trace_id ${record["trace_id"]}
  </record>
</filter>

<match doc-rag.**>
  @type elasticsearch
  host elasticsearch
  port 9200
  index_name doc-rag
  type_name _doc
  logstash_format true
  logstash_prefix doc-rag
  <buffer>
    @type file
    path /var/log/fluentd-buffers/kubernetes.system.buffer
    flush_mode interval
    retry_type exponential_backoff
    flush_thread_count 2
    flush_interval 5s
    retry_forever
    retry_max_interval 30
    chunk_limit_size 2M
    queue_limit_length 8
    overflow_action block
  </buffer>
</match>
```

### 5.4 Custom Metrics and Dashboards

#### Business Metrics Exporter (Rust)
```rust
// src/metrics-exporter/src/main.rs
use axum::{extract::Query, response::IntoResponse, routing::get, Router};
use prometheus::{
    register_counter, register_gauge, register_histogram, Counter, Encoder, Gauge, Histogram,
    TextEncoder,
};
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct MetricsQuery {
    service: Option<String>,
}

struct BusinessMetrics {
    documents_processed: Counter,
    query_accuracy: Gauge,
    response_time: Histogram,
    active_users: Gauge,
    storage_usage: Gauge,
}

impl BusinessMetrics {
    fn new() -> Self {
        Self {
            documents_processed: register_counter!(
                "doc_rag_documents_processed_total",
                "Total number of documents processed"
            ).unwrap(),
            query_accuracy: register_gauge!(
                "doc_rag_query_accuracy",
                "Current query accuracy score"
            ).unwrap(),
            response_time: register_histogram!(
                "doc_rag_response_time_seconds",
                "Response time for queries in seconds"
            ).unwrap(),
            active_users: register_gauge!(
                "doc_rag_active_users",
                "Number of currently active users"
            ).unwrap(),
            storage_usage: register_gauge!(
                "doc_rag_storage_usage_bytes",
                "Storage usage in bytes"
            ).unwrap(),
        }
    }

    fn update_metrics(&self) {
        // Update metrics from database/cache
        // This would typically query your database for current values
        self.documents_processed.inc();
        self.query_accuracy.set(0.95);
        self.active_users.set(150.0);
        self.storage_usage.set(1024.0 * 1024.0 * 1024.0); // 1GB
    }
}

async fn metrics_handler(Query(params): Query<MetricsQuery>) -> impl IntoResponse {
    let metrics = BusinessMetrics::new();
    metrics.update_metrics();

    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut output = Vec::new();
    encoder.encode(&metric_families, &mut output).unwrap();

    (
        [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
        output,
    )
}

#[tokio::main]
async fn main() {
    let app = Router::new().route("/metrics", get(metrics_handler));

    println!("Business metrics exporter starting on :8080/metrics");
    axum::Server::bind(&"0.0.0.0:8080".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

## 6. Secrets Management

### 6.1 HashiCorp Vault Integration

#### Vault Configuration
```hcl
# config/vault/vault.hcl
ui = true
disable_mlock = true

storage "file" {
  path = "/vault/data"
}

listener "tcp" {
  address = "0.0.0.0:8200"
  tls_disable = 1
}

api_addr = "http://127.0.0.1:8200"
cluster_addr = "https://127.0.0.1:8201"
```

#### Kubernetes Integration with External Secrets Operator
```yaml
# config/secrets/external-secrets.yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
  namespace: doc-rag-prod
spec:
  provider:
    vault:
      server: "https://vault.doc-rag.io"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "doc-rag-role"

---
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: doc-rag-secrets
  namespace: doc-rag-prod
spec:
  refreshInterval: 10m
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: doc-rag-secrets
    creationPolicy: Owner
  data:
  - secretKey: database-url
    remoteRef:
      key: doc-rag/database
      property: url
  - secretKey: redis-password
    remoteRef:
      key: doc-rag/redis
      property: password
  - secretKey: jwt-secret
    remoteRef:
      key: doc-rag/auth
      property: jwt-secret
```

## 7. Security Hardening

### 7.1 Container Security Scanning

#### Comprehensive Security Pipeline
```yaml
# .github/workflows/security-comprehensive.yml (Extended)
container-security:
  name: Container Security Analysis
  runs-on: ubuntu-latest
  strategy:
    matrix:
      scanner: [trivy, grype, snyk, anchore]
      service: [api, chunker, embedder, retriever]
  
  steps:
    - name: Build Image
      run: docker build -t doc-rag-${{ matrix.service }}:scan .
      
    - name: Trivy Vulnerability Scanner
      if: matrix.scanner == 'trivy'
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: doc-rag-${{ matrix.service }}:scan
        format: 'sarif'
        output: 'trivy-results.sarif'
        
    - name: Grype Vulnerability Scanner  
      if: matrix.scanner == 'grype'
      run: |
        curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
        grype doc-rag-${{ matrix.service }}:scan -o sarif > grype-results.sarif
        
    - name: Snyk Container Test
      if: matrix.scanner == 'snyk'
      uses: snyk/actions/docker@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        image: doc-rag-${{ matrix.service }}:scan
        args: --severity-threshold=medium --sarif-file-output=snyk-results.sarif
```

### 7.2 Runtime Security

#### Falco Rules for Runtime Security
```yaml
# config/security/falco-rules.yml
- rule: Unexpected Network Connection from Doc-RAG Container
  desc: Detect unexpected network connections from doc-rag containers
  condition: >
    inbound_outbound and
    container and
    container.image contains "doc-rag" and
    not proc.name in (curl, wget, api-server) and
    not fd.sport in (8080, 9090, 5432, 6379, 6333)
  output: >
    Unexpected network connection from doc-rag container 
    (command=%proc.cmdline pid=%proc.pid container=%container.id 
    image=%container.image.repository connection=%fd.name)
  priority: WARNING
  tags: [network, doc-rag]

- rule: File Access Outside Expected Paths in Doc-RAG
  desc: Detect file access outside expected paths
  condition: >
    open_read and
    container and
    container.image contains "doc-rag" and
    not fd.name startswith /app/ and
    not fd.name startswith /tmp/ and
    not fd.name startswith /proc/ and
    not fd.name startswith /sys/
  output: >
    File access outside expected paths in doc-rag container
    (file=%fd.name command=%proc.cmdline pid=%proc.pid container=%container.id)
  priority: ERROR
  tags: [filesystem, doc-rag]
```

## 8. Performance Optimization Scripts

### 8.1 Load Testing Suite
```javascript
// tests/load/k6-load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
let errorRate = new Rate('errors');
let queryLatency = new Trend('query_latency');

export let options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp up
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 100 },  // Ramp to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 0 },    // Scale down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests under 500ms
    http_req_failed: ['rate<0.01'],   // Error rate under 1%
    errors: ['rate<0.01'],
  },
};

export default function() {
  // Test document upload
  let uploadResponse = http.post('http://localhost:8080/api/v1/documents', {
    content: 'Test document content for load testing',
    metadata: JSON.stringify({ source: 'load-test', type: 'text' })
  });
  
  check(uploadResponse, {
    'upload status is 201': (r) => r.status === 201,
  }) || errorRate.add(1);

  sleep(1);

  // Test query
  let queryStart = Date.now();
  let queryResponse = http.post('http://localhost:8080/api/v1/query', 
    JSON.stringify({
      query: 'What is the content about?',
      limit: 5
    }),
    { headers: { 'Content-Type': 'application/json' } }
  );
  
  let queryDuration = Date.now() - queryStart;
  queryLatency.add(queryDuration);
  
  check(queryResponse, {
    'query status is 200': (r) => r.status === 200,
    'query has results': (r) => JSON.parse(r.body).results.length > 0,
  }) || errorRate.add(1);

  sleep(2);
}
```

### 8.2 Performance Monitoring Scripts
```python
#!/usr/bin/env python3
# scripts/performance-monitor.py

import time
import requests
import psutil
import docker
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class PerformanceMetrics:
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    response_time: float
    error_rate: float
    throughput: float

class PerformanceMonitor:
    def __init__(self, api_endpoint: str = "http://localhost:8080"):
        self.api_endpoint = api_endpoint
        self.docker_client = docker.from_env()
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
            "network": psutil.net_io_counters()._asdict()
        }
    
    def collect_container_metrics(self) -> Dict[str, Any]:
        """Collect Docker container metrics"""
        containers = {}
        for container in self.docker_client.containers.list():
            if "doc-rag" in container.name:
                stats = container.stats(stream=False)
                containers[container.name] = {
                    "cpu_usage": self._calculate_cpu_percent(stats),
                    "memory_usage": stats["memory_stats"]["usage"],
                    "memory_limit": stats["memory_stats"]["limit"],
                    "network_rx": stats["networks"]["bridge"]["rx_bytes"],
                    "network_tx": stats["networks"]["bridge"]["tx_bytes"]
                }
        return containers
    
    def collect_application_metrics(self) -> Dict[str, Any]:
        """Collect application-specific metrics"""
        try:
            # Health check
            health_response = requests.get(f"{self.api_endpoint}/health", timeout=5)
            health_status = health_response.status_code == 200
            
            # Performance test query
            start_time = time.time()
            query_response = requests.post(
                f"{self.api_endpoint}/api/v1/query",
                json={"query": "performance test", "limit": 1},
                timeout=30
            )
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Metrics endpoint
            metrics_response = requests.get(f"{self.api_endpoint}/metrics", timeout=5)
            
            return {
                "health_status": health_status,
                "response_time_ms": response_time,
                "query_success": query_response.status_code == 200,
                "metrics_available": metrics_response.status_code == 200
            }
        except Exception as e:
            return {
                "health_status": False,
                "response_time_ms": -1,
                "query_success": False,
                "metrics_available": False,
                "error": str(e)
            }
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats"""
        cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                   stats["precpu_stats"]["cpu_usage"]["total_usage"]
        system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                      stats["precpu_stats"]["system_cpu_usage"]
        
        if system_delta > 0:
            return (cpu_delta / system_delta) * len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"]) * 100.0
        return 0.0
    
    def generate_report(self, duration_minutes: int = 60) -> None:
        """Generate performance monitoring report"""
        print(f"Starting {duration_minutes}-minute performance monitoring...")
        
        metrics_log = []
        start_time = time.time()
        
        while (time.time() - start_time) < (duration_minutes * 60):
            timestamp = datetime.now().isoformat()
            
            # Collect all metrics
            system_metrics = self.collect_system_metrics()
            container_metrics = self.collect_container_metrics()
            app_metrics = self.collect_application_metrics()
            
            # Combine metrics
            combined_metrics = {
                "timestamp": timestamp,
                "system": system_metrics,
                "containers": container_metrics,
                "application": app_metrics
            }
            
            metrics_log.append(combined_metrics)
            
            # Print current status
            print(f"[{timestamp}] CPU: {system_metrics['cpu_percent']:.1f}% | "
                  f"Memory: {system_metrics['memory']['percent']:.1f}% | "
                  f"Health: {'OK' if app_metrics['health_status'] else 'FAIL'} | "
                  f"Response: {app_metrics['response_time_ms']:.1f}ms")
            
            time.sleep(60)  # Collect metrics every minute
        
        # Save report
        report_filename = f"performance-report-{int(start_time)}.json"
        with open(report_filename, 'w') as f:
            json.dump(metrics_log, f, indent=2)
        
        print(f"Performance report saved to {report_filename}")
        self._analyze_and_print_summary(metrics_log)
    
    def _analyze_and_print_summary(self, metrics_log: List[Dict]) -> None:
        """Analyze collected metrics and print summary"""
        if not metrics_log:
            return
        
        # Calculate averages
        cpu_values = [m["system"]["cpu_percent"] for m in metrics_log]
        memory_values = [m["system"]["memory"]["percent"] for m in metrics_log]
        response_times = [m["application"]["response_time_ms"] for m in metrics_log 
                         if m["application"]["response_time_ms"] > 0]
        
        print("\n" + "="*50)
        print("PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Duration: {len(metrics_log)} minutes")
        print(f"Average CPU Usage: {sum(cpu_values)/len(cpu_values):.1f}%")
        print(f"Peak CPU Usage: {max(cpu_values):.1f}%")
        print(f"Average Memory Usage: {sum(memory_values)/len(memory_values):.1f}%")
        print(f"Peak Memory Usage: {max(memory_values):.1f}%")
        
        if response_times:
            print(f"Average Response Time: {sum(response_times)/len(response_times):.1f}ms")
            print(f"Max Response Time: {max(response_times):.1f}ms")
            print(f"Min Response Time: {min(response_times):.1f}ms")
        
        # Health check success rate
        health_checks = [m["application"]["health_status"] for m in metrics_log]
        success_rate = (sum(health_checks) / len(health_checks)) * 100
        print(f"Health Check Success Rate: {success_rate:.1f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Doc-RAG Performance Monitor")
    parser.add_argument("--duration", type=int, default=60, 
                       help="Monitoring duration in minutes")
    parser.add_argument("--endpoint", default="http://localhost:8080",
                       help="API endpoint to monitor")
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.endpoint)
    monitor.generate_report(args.duration)
```

## Conclusion

This comprehensive CI/CD architecture provides:

1. **Automated Quality Assurance**: Multi-stage pipeline with extensive testing, security scanning, and performance benchmarking
2. **Security-First Approach**: Integrated security scanning, secrets management, and runtime protection
3. **Scalable Infrastructure**: Kubernetes-native deployment with auto-scaling and advanced deployment strategies
4. **Comprehensive Monitoring**: Full observability stack with metrics, logs, traces, and alerting
5. **Performance Optimization**: Automated performance testing and optimization recommendations

The architecture supports both development velocity and production reliability, with automated rollback capabilities and comprehensive monitoring to ensure system health and performance.

## Implementation Timeline

- **Week 1**: CI/CD pipeline setup and basic automation
- **Week 2**: Security integration and container optimization  
- **Week 3**: Kubernetes deployment and monitoring setup
- **Week 4**: Performance optimization and documentation

This architecture scales from development through production and provides the foundation for reliable, secure, and performant operations.