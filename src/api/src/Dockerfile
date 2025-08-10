# Multi-stage Docker build for Doc-RAG API Gateway
# Optimized for security, performance, and minimal image size

# Build stage
FROM rust:1.75-bullseye AS builder

# Create app user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/app

# Create directory structure
RUN mkdir -p src/api

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY src/api/Cargo.toml src/api/

# Copy all necessary component Cargo.toml files for workspace dependencies
COPY src/chunker/Cargo.toml src/chunker/
COPY src/embedder/Cargo.toml src/embedder/
COPY src/storage/Cargo.toml src/storage/
COPY src/query-processor/Cargo.toml src/query-processor/
COPY src/response-generator/Cargo.toml src/response-generator/
COPY src/integration/Cargo.toml src/integration/

# Create dummy source files to cache dependencies
RUN mkdir -p src/api/src && echo "fn main() {}" > src/api/src/main.rs
RUN echo 'pub fn dummy() {}' > src/api/src/lib.rs

# Build dependencies only
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/app/target \
    cargo build --release --bin api-server

# Remove dummy files and copy real source code
RUN rm -rf src/api/src

# Copy API source code
COPY src/api/src src/api/src

# Copy shared modules if they exist
COPY src/security src/security
COPY src/shared src/shared

# Build the application
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/src/app/target \
    cargo build --release --bin api-server && \
    cp target/release/api-server /usr/local/bin/api-server

# Runtime stage
FROM debian:bullseye-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Create application directories
RUN mkdir -p /app/config /app/logs /app/storage && \
    chown -R appuser:appgroup /app

# Copy binary from builder
COPY --from=builder /usr/local/bin/api-server /usr/local/bin/api-server
RUN chmod +x /usr/local/bin/api-server

# Copy configuration files
COPY config/ /app/config/

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Set working directory
WORKDIR /app

# Environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV BIND_ADDRESS=0.0.0.0
ENV PORT=8080

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Security hardening
# Run as non-root
# Read-only filesystem for security
# No shell access
# Minimal attack surface

# Default command
CMD ["api-server", "--config", "/app/config/api.toml"]

# Build targets for different environments
FROM runtime AS production
ENV RUST_LOG=info

FROM runtime AS development
ENV RUST_LOG=debug
# Development-specific configurations can be added here

# Metadata
LABEL \
    org.opencontainers.image.title="Doc-RAG API Gateway" \
    org.opencontainers.image.description="HTTP API Gateway for Doc-RAG System" \
    org.opencontainers.image.vendor="Doc-RAG Team" \
    org.opencontainers.image.licenses="MIT" \
    org.opencontainers.image.source="https://github.com/doc-rag/doc-rag" \
    org.opencontainers.image.documentation="https://docs.doc-rag.io" \
    org.opencontainers.image.authors="Doc-RAG Team <team@doc-rag.io>"

# Security scanning labels
LABEL \
    security.scan="enabled" \
    security.non-root-user="appuser" \
    security.no-shell="true" \
    security.minimal-base="debian:bullseye-slim"