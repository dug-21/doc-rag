# Multi-stage build for minimal production image
# Stage 1: Build dependencies and compile
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /usr/src/app

# Copy manifests first for better caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy src to build dependencies
RUN mkdir src && echo "fn main() {}" > src/lib.rs

# Build dependencies (this layer will be cached unless Cargo.toml changes)
RUN cargo build --release && rm -rf src target/release/deps/chunker*

# Copy source code
COPY src ./src
COPY benches ./benches

# Build the actual application
RUN cargo build --release

# Run tests to ensure everything works
RUN cargo test --release

# Stage 2: Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user (same UID as builder stage)
RUN useradd -m -u 1001 appuser

# Create app directory
WORKDIR /app

# Copy the compiled binary
COPY --from=builder /usr/src/app/target/release/chunker /usr/local/bin/chunker
COPY --from=builder /usr/src/app/target/release/libchunker* /usr/local/lib/

# Copy any additional files if needed
# COPY --from=builder /usr/src/app/config /app/config

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD test -f /usr/local/bin/chunker && echo "Chunker binary is available" || exit 1

# Set environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Resource limits (can be overridden)
ENV MEMORY_LIMIT=2G
ENV CPU_LIMIT=2

# Expose any ports if this becomes a service
# EXPOSE 8080

# Default command (can be overridden)
CMD ["chunker", "--help"]

# Metadata
LABEL maintainer="RAG System Team"
LABEL version="0.1.0"
LABEL description="Intelligent document chunker with semantic boundary detection"
LABEL org.opencontainers.image.source="https://github.com/your-org/doc-rag"
LABEL org.opencontainers.image.title="Document Chunker"
LABEL org.opencontainers.image.description="High-performance document chunking service with neural boundary detection"
LABEL org.opencontainers.image.vendor="RAG System Team"
LABEL org.opencontainers.image.licenses="MIT"

# Security settings
USER 1001:1001
WORKDIR /app

# Ensure no root access
RUN chmod -R 755 /app && \
    find /app -type f -executable -exec chmod 755 {} \;

# Set secure defaults
ENV TMPDIR=/tmp
RUN chmod 1777 /tmp

# Final verification
RUN whoami && id && pwd && ls -la /usr/local/bin/chunker || true