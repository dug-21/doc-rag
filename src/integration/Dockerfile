# Multi-stage Docker build for Doc-RAG Integration System
# Following design principle of reproducible builds

# Build stage
FROM rust:1.75-slim-bullseye AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/

# Build dependencies (cached layer)
RUN mkdir -p src/bin && echo 'fn main() {}' > src/bin/dummy.rs
RUN cargo build --release --bin dummy && rm -rf src/bin/dummy.rs target/release/deps/dummy*

# Build the actual application
RUN cargo build --release --bin integration-server

# Runtime stage
FROM debian:bullseye-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root for security)
RUN useradd -m -u 1001 appuser

# Create app directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/integration-server /app/integration-server

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV RUST_LOG=integration=info
ENV RUST_BACKTRACE=1

# Resource limits
ENV GOMAXPROCS=2
LABEL max_memory="1GB"
LABEL max_cpu="2"

# Metadata labels
LABEL maintainer="Doc-RAG Team <team@doc-rag.io>"
LABEL version="0.1.0"
LABEL description="Doc-RAG System Integration Orchestrator"
LABEL org.opencontainers.image.title="doc-rag-integration"
LABEL org.opencontainers.image.description="High-performance document retrieval and augmented generation system integration"
LABEL org.opencontainers.image.source="https://github.com/doc-rag/doc-rag"
LABEL org.opencontainers.image.vendor="Doc-RAG Team"
LABEL org.opencontainers.image.licenses="MIT"

# Run the application
CMD ["./integration-server"]
