# Multi-stage Docker build for the embedder service
# This Dockerfile creates an optimized production image for the embedding generator

# Build stage
FROM rust:1.75-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src
COPY benches ./benches
COPY examples ./examples
COPY tests ./tests

# Build the application in release mode
RUN cargo build --release

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash embedder

# Create necessary directories
RUN mkdir -p /models /cache /data && \
    chown -R embedder:embedder /models /cache /data

# Copy the binary from builder stage
COPY --from=builder /usr/src/app/target/release/embedder /usr/local/bin/embedder

# Copy configuration files
COPY config/ /etc/embedder/

# Set up non-root user
USER embedder

# Set working directory
WORKDIR /app

# Create volume mounts for models, cache, and data
VOLUME ["/models", "/cache", "/data"]

# Environment variables
ENV RUST_LOG=info \
    EMBEDDER_MODEL_PATH=/models \
    EMBEDDER_CACHE_PATH=/cache \
    EMBEDDER_DATA_PATH=/data \
    EMBEDDER_PORT=8080 \
    EMBEDDER_HOST=0.0.0.0 \
    EMBEDDER_WORKERS=4 \
    EMBEDDER_MAX_BATCH_SIZE=64 \
    EMBEDDER_CACHE_SIZE=10000

# Health check script
COPY --chown=embedder:embedder <<EOF /usr/local/bin/health-check.sh
#!/bin/bash
set -e
# Basic health check - verify service is responding
curl -f http://localhost:\${EMBEDDER_PORT}/health || exit 1
EOF

RUN chmod +x /usr/local/bin/health-check.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/health-check.sh

# Expose port
EXPOSE 8080

# Labels for metadata
LABEL maintainer="RAG System Team <team@ragsystem.com>" \
      version="1.0.0" \
      description="High-performance embedding generator service" \
      documentation="https://github.com/your-org/doc-rag/tree/main/src/embedder" \
      source="https://github.com/your-org/doc-rag"

# Default command
CMD ["embedder", "--config", "/etc/embedder/config.toml"]

# Alternative Dockerfile for development
# FROM rust:1.75-slim-bullseye as development
# 
# RUN apt-get update && apt-get install -y \
#     pkg-config \
#     libssl-dev \
#     ca-certificates \
#     curl \
#     wget \
#     git \
#     && rm -rf /var/lib/apt/lists/*
# 
# WORKDIR /usr/src/app
# 
# # Install cargo tools for development
# RUN cargo install cargo-watch cargo-edit
# 
# # Copy source
# COPY . .
# 
# # Development mode with hot reloading
# CMD ["cargo", "watch", "-x", "run"]

# Multi-architecture build support
# Use buildx for ARM64 and AMD64:
# docker buildx build --platform linux/amd64,linux/arm64 -t embedder:latest .

# Build arguments for customization
ARG RUST_VERSION=1.75
ARG MODEL_VERSION=latest
ARG BUILD_FEATURES=""

# Production optimizations
RUN strip /usr/local/bin/embedder

# Security: Run as non-root and set proper permissions
RUN chmod 755 /usr/local/bin/embedder

# Add build info
ARG BUILD_DATE
ARG BUILD_VERSION
ARG BUILD_COMMIT

LABEL build.date=$BUILD_DATE \
      build.version=$BUILD_VERSION \
      build.commit=$BUILD_COMMIT