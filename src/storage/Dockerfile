# MongoDB Vector Storage - Multi-stage Dockerfile

# Build stage
FROM rust:1.75-alpine as builder

# Install required packages for building
RUN apk add --no-cache musl-dev openssl-dev

# Set working directory
WORKDIR /usr/src/app

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Create a dummy main.rs for caching dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (this will be cached if Cargo.toml doesn't change)
RUN cargo build --release && rm src/main.rs

# Copy source code
COPY src ./src

# Build the application
RUN cargo build --release --bin storage

# Runtime stage
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    openssl \
    libgcc \
    && rm -rf /var/cache/apk/*

# Create app user
RUN addgroup -g 1001 -S appgroup && \
    adduser -u 1001 -S appuser -G appgroup

# Set working directory
WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /usr/src/app/target/release/storage /usr/local/bin/storage

# Copy configuration files
COPY storage.toml /app/storage.toml

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port (if storage has a web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD storage --health-check || exit 1

# Default command
CMD ["storage", "--config", "/app/storage.toml"]

# Metadata
LABEL maintainer="RAG Team"
LABEL description="MongoDB Vector Storage for RAG system"
LABEL version="1.0.0"