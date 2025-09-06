#!/bin/bash
# CI/CD Setup Script for Doc-RAG Phase 3
# Automated setup of development environment and CI/CD tools

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/config"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
DOCKER_DIR="${PROJECT_ROOT}/docker"

# Function definitions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Command '$1' not found. Please install it first."
        return 1
    fi
    return 0
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Essential tools
    check_command "docker" || missing_tools+=("docker")
    check_command "docker-compose" || missing_tools+=("docker-compose") 
    check_command "git" || missing_tools+=("git")
    check_command "curl" || missing_tools+=("curl")
    check_command "jq" || missing_tools+=("jq")
    
    # Rust toolchain
    check_command "rustc" || missing_tools+=("rust")
    check_command "cargo" || missing_tools+=("cargo")
    
    # Optional but recommended tools
    if ! check_command "kubectl"; then
        log_warning "kubectl not found - Kubernetes features will be limited"
    fi
    
    if ! check_command "helm"; then
        log_warning "helm not found - Helm chart deployment will be limited"
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and run this script again."
        exit 1
    fi
    
    log_success "All prerequisites met"
}

# Setup directory structure
setup_directories() {
    log_info "Setting up directory structure..."
    
    local dirs=(
        ".github/workflows"
        ".github/actions"
        "config/grafana/dashboards"
        "config/grafana/datasources"
        "config/prometheus"
        "config/alerting"
        "config/nginx"
        "config/ssl"
        "config/security"
        "config/fluentd"
        "scripts/ci"
        "scripts/deployment"
        "scripts/monitoring"
        "tests/load"
        "tests/integration"
        "tests/e2e"
        "k8s/base"
        "k8s/overlays/staging"
        "k8s/overlays/production"
        "helm/doc-rag/templates"
        "helm/doc-rag/charts"
        "docker/images"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
        log_info "Created directory: $dir"
    done
    
    log_success "Directory structure created"
}

# Setup Git hooks
setup_git_hooks() {
    log_info "Setting up Git hooks..."
    
    # Pre-commit hook
    cat > "$PROJECT_ROOT/.git/hooks/pre-commit" << 'EOF'
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

# Quick tests
echo "🧪 Running quick tests..."
if ! cargo test --lib --bins --quiet; then
    echo "❌ Tests failed. Fix them before committing."
    exit 1
fi

# License check
if [[ -f "./scripts/check-licenses.sh" ]]; then
    echo "📋 Checking license compliance..."
    ./scripts/check-licenses.sh
fi

echo "✅ All pre-commit checks passed!"
EOF

    chmod +x "$PROJECT_ROOT/.git/hooks/pre-commit"
    
    # Pre-push hook
    cat > "$PROJECT_ROOT/.git/hooks/pre-push" << 'EOF'
#!/bin/bash
set -e

echo "🔍 Running pre-push checks..."

# Security audit
echo "🛡️ Running security audit..."
if command -v cargo-audit &> /dev/null; then
    cargo audit
else
    echo "⚠️ cargo-audit not installed. Run: cargo install cargo-audit"
fi

# Full test suite
echo "🧪 Running full test suite..."
cargo test --workspace --all-features

echo "✅ All pre-push checks passed!"
EOF

    chmod +x "$PROJECT_ROOT/.git/hooks/pre-push"
    
    log_success "Git hooks installed"
}

# Install development tools
install_dev_tools() {
    log_info "Installing development tools..."
    
    # Rust tools
    local rust_tools=(
        "cargo-audit"
        "cargo-deny"
        "cargo-tarpaulin"
        "cargo-criterion"
        "cargo-watch"
        "cargo-edit"
        "cargo-outdated"
    )
    
    for tool in "${rust_tools[@]}"; do
        if ! cargo install --list | grep -q "$tool"; then
            log_info "Installing $tool..."
            cargo install "$tool" --locked || log_warning "Failed to install $tool"
        else
            log_info "$tool already installed"
        fi
    done
    
    log_success "Development tools installed"
}

# Setup Docker development environment
setup_docker_dev() {
    log_info "Setting up Docker development environment..."
    
    # Create development override
    cat > "$PROJECT_ROOT/docker-compose.override.yml" << 'EOF'
version: '3.8'

services:
  api:
    build:
      target: development
    volumes:
      - .:/app:cached
      - /app/target
      - cargo_cache:/usr/local/cargo/registry
    environment:
      - RUST_LOG=debug
      - RUST_BACKTRACE=1
    ports:
      - "8080:8080"
      - "8000:8000"  # Debug port

volumes:
  cargo_cache:
    driver: local
EOF

    # Create .dockerignore
    cat > "$PROJECT_ROOT/.dockerignore" << 'EOF'
target/
Cargo.lock
.git
.gitignore
README.md
docs/
tests/
benches/
examples/
.env*
docker-compose*.yml
Dockerfile*
.dockerignore
node_modules/
*.log
.DS_Store
EOF

    log_success "Docker development environment configured"
}

# Setup monitoring configuration
setup_monitoring() {
    log_info "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > "$CONFIG_DIR/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'doc-rag-dev'
    environment: 'development'

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'doc-rag-api'
    static_configs:
      - targets: ['api:9090']
    scrape_interval: 10s
    
  - job_name: 'doc-rag-services'
    static_configs:
      - targets: ['chunker:9090', 'embedder:9090', 'retriever:9090']
      
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
EOF

    # Grafana datasource
    mkdir -p "$CONFIG_DIR/grafana/datasources"
    cat > "$CONFIG_DIR/grafana/datasources/prometheus.yml" << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
EOF

    log_success "Monitoring configuration created"
}

# Create essential scripts
create_scripts() {
    log_info "Creating essential scripts..."
    
    # Development start script
    cat > "$SCRIPTS_DIR/dev-start.sh" << 'EOF'
#!/bin/bash
# Start development environment

set -e

echo "🚀 Starting Doc-RAG development environment..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start services
echo "📦 Starting services with docker-compose..."
docker-compose up -d postgres redis qdrant minio

# Wait for services
echo "⏳ Waiting for services to be ready..."
timeout 120 bash -c 'until docker-compose exec -T postgres pg_isready -U docrag; do sleep 2; done'
timeout 60 bash -c 'until docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; do sleep 2; done'

echo "✅ Development environment ready!"
echo ""
echo "Services available at:"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo "  - Qdrant: localhost:6333"
echo "  - MinIO: localhost:9000"
echo ""
echo "Start the API with: cargo run --bin api-server"
EOF

    chmod +x "$SCRIPTS_DIR/dev-start.sh"
    
    # Build script
    cat > "$SCRIPTS_DIR/build.sh" << 'EOF'
#!/bin/bash
# Build script for CI/CD

set -e

BUILD_MODE=${1:-release}
FEATURES=${2:-all-features}

echo "🏗️ Building Doc-RAG ($BUILD_MODE mode, $FEATURES)..."

# Format check
echo "📝 Checking format..."
cargo fmt --all -- --check

# Clippy
echo "🔧 Running Clippy..."
cargo clippy --workspace --all-targets --${FEATURES} -- -D warnings

# Build
echo "🔨 Building..."
if [[ "$BUILD_MODE" == "release" ]]; then
    cargo build --release --workspace --${FEATURES}
else
    cargo build --workspace --${FEATURES}
fi

# Test
echo "🧪 Running tests..."
cargo test --workspace --${FEATURES}

echo "✅ Build completed successfully!"
EOF

    chmod +x "$SCRIPTS_DIR/build.sh"
    
    # License check script
    cat > "$SCRIPTS_DIR/check-licenses.sh" << 'EOF'
#!/bin/bash
# License compliance check

set -e

ALLOWED_LICENSES=("MIT" "Apache-2.0" "BSD-3-Clause" "ISC" "Apache-2.0 WITH LLVM-exception")

echo "🔍 Checking license compliance..."

# Install cargo-license if not available
if ! command -v cargo-license &> /dev/null; then
    echo "Installing cargo-license..."
    cargo install cargo-license
fi

# Generate license report
cargo license --json > license_report.json

# Check for forbidden licenses
FORBIDDEN=$(jq -r '.[] | select(.license | IN("GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0") | not | not) | .name + " (" + .license + ")"' license_report.json)

if [ -n "$FORBIDDEN" ]; then
    echo "❌ Forbidden licenses detected:"
    echo "$FORBIDDEN"
    rm -f license_report.json
    exit 1
fi

echo "✅ All licenses are compliant"
rm -f license_report.json
EOF

    chmod +x "$SCRIPTS_DIR/check-licenses.sh"
    
    log_success "Essential scripts created"
}

# Setup GitHub Actions workflow
setup_github_actions() {
    log_info "Setting up GitHub Actions workflow..."
    
    # Copy the comprehensive CI workflow
    cat > "$PROJECT_ROOT/.github/workflows/ci-comprehensive.yml" << 'EOF'
name: Comprehensive CI/CD Pipeline

on:
  push:
    branches: [main, develop]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  pull_request:
    branches: [main, develop]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1
  RUSTFLAGS: "-D warnings"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # Fast feedback - code quality
  quality-check:
    name: Code Quality
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
          
      - name: Cache
        uses: Swatinem/rust-cache@v2
        
      - name: Format Check
        run: cargo fmt --all -- --check
        
      - name: Clippy
        run: cargo clippy --workspace --all-targets --all-features -- -D warnings

  # Security scanning
  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4
        
      - name: Install Rust
        uses: dtolnay/rust-toolchain@stable
        
      - name: Security Audit
        run: |
          cargo install cargo-audit
          cargo audit --deny warnings
          
      - name: License Check
        run: |
          ./scripts/check-licenses.sh

  # Comprehensive testing
  test-suite:
    name: Test Suite
    runs-on: ${{ matrix.os }}
    timeout-minutes: 45
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        rust: [stable, beta]
        
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
        ports:
          - 5432:5432
          
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v4
        
      - name: Install Rust
        uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
          
      - name: Cache
        uses: Swatinem/rust-cache@v2
        with:
          shared-key: "test-${{ matrix.os }}-${{ matrix.rust }}"
          
      - name: Test
        run: cargo test --workspace --all-features
        env:
          DATABASE_URL: postgres://postgres:test_password@localhost:5432/docrag_test
          REDIS_URL: redis://localhost:6379

  # Docker builds
  docker-build:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: [quality-check, security-scan]
    strategy:
      matrix:
        service: [api, chunker, embedder, retriever]
    steps:
      - uses: actions/checkout@v4
        
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Build
        uses: docker/build-push-action@v5
        with:
          context: .
          file: src/${{ matrix.service }}/Dockerfile
          target: production
          tags: doc-rag-${{ matrix.service }}:test
          load: true
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # Integration tests
  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: [docker-build]
    steps:
      - uses: actions/checkout@v4
        
      - name: Start Test Environment
        run: |
          docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d
          sleep 30
          
      - name: Run Integration Tests
        run: |
          # Wait for services
          timeout 120 bash -c 'until curl -f http://localhost:8080/health; do sleep 2; done'
          
          # Run tests
          cargo test --test integration_tests
          
      - name: Cleanup
        if: always()
        run: docker-compose down -v
EOF

    log_success "GitHub Actions workflow created"
}

# Main execution
main() {
    echo "🚀 Setting up CI/CD for Doc-RAG Phase 3"
    echo "========================================"
    
    check_prerequisites
    setup_directories
    setup_git_hooks
    install_dev_tools
    setup_docker_dev
    setup_monitoring
    create_scripts
    setup_github_actions
    
    echo ""
    echo "🎉 CI/CD setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run './scripts/dev-start.sh' to start the development environment"
    echo "2. Use './scripts/build.sh' to build and test the project"
    echo "3. Commit your changes to trigger the CI pipeline"
    echo "4. Review the monitoring dashboards at http://localhost:3000 (Grafana)"
    echo ""
    echo "For more information, see: epics/phase3/plan/cicd-architecture.md"
}

# Run main function
main "$@"