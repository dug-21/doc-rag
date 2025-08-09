# Multi-stage build for minimal image size
FROM rust:1.75-slim-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Cargo files first for better caching
COPY Cargo.toml ./

# Create a dummy src directory to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release --bin dummy 2>/dev/null || true

# Copy source code
COPY src ./src

# Build the actual application
RUN cargo build --release --lib

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r mcp && useradd -r -g mcp mcp

# Create application directory
WORKDIR /app

# Copy the binary from builder stage
COPY --from=builder /app/target/release/libmcp_adapter.so /usr/local/lib/
COPY --from=builder /app/target/release/deps/ /usr/local/lib/deps/

# Set permissions
RUN chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Expose default ports (if needed)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD echo "Health check placeholder - implement actual health check endpoint"

# Environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Default command (this would typically run your application)
CMD ["echo", "MCP Adapter library ready"]

# Build-time metadata
LABEL maintainer="RAG System Team"
LABEL description="MCP Protocol Adapter - Async communication foundation"
LABEL version="0.1.0"
LABEL org.opencontainers.image.title="MCP Adapter"
LABEL org.opencontainers.image.description="Async communication foundation for RAG system components"
LABEL org.opencontainers.image.vendor="RAG System"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/your-org/doc-rag"
LABEL org.opencontainers.image.documentation="https://github.com/your-org/doc-rag/blob/main/src/mcp-adapter/README.md"

# Resource limits (recommended settings)
# These can be overridden at runtime
ENV MEMORY_LIMIT=512m
ENV CPU_LIMIT=0.5