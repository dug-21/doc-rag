# Multi-stage Rust Docker build for neurosymbolic RAG
FROM rust:1.75 AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace Cargo files first (for dependency caching)
COPY Cargo.toml Cargo.lock ./
COPY src/*/Cargo.toml ./src/*/

# Create minimal dummy source structure for dependency caching
RUN mkdir -p src/integration/src && echo "fn main() {}" > src/integration/src/main.rs

# Build dependencies only
RUN cargo build --release --bin integration-server
RUN rm -rf src

# Copy source code
COPY . .

# Build final application
RUN cargo build --release --bin integration-server

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 appuser

# Create application directory
WORKDIR /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/integration-server /app/neurosymbolic-rag
RUN chmod +x /app/neurosymbolic-rag

# Create required directories
RUN mkdir -p /app/documents /app/models /app/logs \
    && chown -R appuser:appuser /app

# Copy configuration files
COPY config/ /app/config/

# Switch to app user
USER appuser

# Create simple health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8080/health || exit 1' > /app/health-check.sh && chmod +x /app/health-check.sh

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV APP_ENV=docker
ENV DOCKER_ENV=1

# Start application
CMD ["./neurosymbolic-rag"]