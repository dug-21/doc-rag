# Multi-stage Dockerfile for Response Generator
# Optimized for security, size, and performance

# Build stage
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create dummy src to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (for caching)
RUN cargo build --release && rm -rf src target/release/deps/response_generator*

# Copy source code
COPY src ./src
COPY benches ./benches
COPY examples ./examples
COPY tests ./tests

# Build application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root for security)
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# Create directories for config and data
RUN mkdir -p config data logs && \
    chown -R appuser:appuser /app

# Copy binary from builder
COPY --from=builder /app/target/release/response-generator /usr/local/bin/response-generator

# Copy default configuration
COPY config/default.toml config/

# Set user
USER appuser

# Expose port (if running as web service)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD response-generator validate-config config/default.toml || exit 1

# Environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV CONFIG_FILE=/app/config/default.toml

# Default command
CMD ["response-generator", "--config", "/app/config/default.toml", "interactive"]

# Labels for metadata
LABEL org.opencontainers.image.title="Response Generator"
LABEL org.opencontainers.image.description="High-accuracy response generation system with citation tracking"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="RAG System Team"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/your-org/doc-rag"
LABEL org.opencontainers.image.documentation="https://github.com/your-org/doc-rag/tree/main/src/response-generator"

# Security labels
LABEL security.non-root="true"
LABEL security.readonly-rootfs="false"