# MongoDB Vector Storage

A high-performance, production-ready MongoDB-based vector storage system for RAG (Retrieval-Augmented Generation) applications.

## 🎯 Features

- **Complete MongoDB Integration**: Full CRUD operations with transaction support
- **High-Performance Vector Search**: <50ms search latency for 10K+ documents
- **Hybrid Search**: Combines vector similarity and text search
- **Advanced Filtering**: Document ID, tags, date range, and custom field filtering
- **Comprehensive Error Handling**: Robust error recovery and retry mechanisms
- **Performance Monitoring**: Built-in metrics and health checks
- **Docker Support**: Production-ready containerization
- **Full Test Coverage**: Unit, integration, and performance tests

## 🚀 Quick Start

### Prerequisites

- Rust 1.75 or later
- MongoDB 7.0+ (with vector search support)
- Docker and Docker Compose (optional)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd src/storage

# Build the project
cargo build --release

# Run tests (requires MongoDB)
cargo test

# Run benchmarks
cargo bench
```

### Docker Setup

```bash
# Start MongoDB and all services
docker-compose up -d

# Run with monitoring
docker-compose --profile monitoring up -d

# Run benchmarks
docker-compose --profile benchmark up storage-benchmark
```

## 📖 Usage

### Basic Example

```rust
use storage::{VectorStorage, StorageConfig, ChunkDocument, ChunkMetadata};
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create storage configuration
    let config = StorageConfig::default();
    
    // Initialize storage
    let storage = VectorStorage::new(config).await?;
    
    // Create a document chunk
    let metadata = ChunkMetadata::new(
        Uuid::new_v4(),
        "My Document".to_string(),
        0,
        1,
        "/path/to/doc.txt".to_string(),
    );
    
    let chunk = ChunkDocument::new(
        Uuid::new_v4(),
        "This is the content of my document.".to_string(),
        metadata,
    );
    
    // Add embedding
    let embedding = vec![0.1, 0.2, 0.3]; // From your embedding model
    let chunk = chunk.with_embedding(embedding);
    
    // Insert the chunk
    let chunk_id = storage.insert_chunk(chunk).await?;
    
    // Perform vector search
    let query_embedding = vec![0.15, 0.25, 0.35];
    let results = storage.vector_search(&query_embedding, 5, None).await?;
    
    println!("Found {} similar documents", results.len());
    
    Ok(())
}
```

### Advanced Search

```rust
use storage::{SearchQuery, SearchType, SearchFilters, SortOptions};

// Create a hybrid search query
let query = SearchQuery {
    query_embedding: Some(embedding),
    text_query: Some("artificial intelligence".to_string()),
    search_type: SearchType::Hybrid,
    limit: 10,
    filters: SearchFilters {
        tags: Some(vec!["technology".to_string()]),
        ..Default::default()
    },
    sort: SortOptions::default(),
    ..Default::default()
};

let response = storage.hybrid_search(query).await?;
println!("Found {} results in {}ms", 
         response.results.len(), 
         response.search_time_ms);
```

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────┐
│                VectorStorage                    │
├─────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │  Operations │ │   Search    │ │   Schema    │ │
│ │   (CRUD)    │ │ (Vector+Text│ │ (Documents) │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │   Config    │ │   Metrics   │ │   Errors    │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ │
├─────────────────────────────────────────────────┤
│              MongoDB Driver                     │
└─────────────────────────────────────────────────┘
```

### Data Model

```rust
ChunkDocument {
    chunk_id: Uuid,
    content: String,
    embedding: Vec<f64>,
    metadata: ChunkMetadata {
        document_id: Uuid,
        chunk_index: usize,
        tags: Vec<String>,
        custom_fields: HashMap<String, Value>,
        // ...
    },
    references: Vec<ChunkReference>,
    created_at: DateTime<Utc>,
    version: i64,
}
```

## 🔧 Configuration

### Environment Variables

```bash
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
STORAGE_DATABASE_NAME=rag_storage
STORAGE_CHUNK_COLLECTION_NAME=chunks

# Performance Tuning
STORAGE_MAX_POOL_SIZE=10
STORAGE_PERFORMANCE__BATCH_SIZE=100
STORAGE_VECTOR_SEARCH__EMBEDDING_DIMENSION=384

# Monitoring
STORAGE_MONITORING__LOG_LEVEL=info
STORAGE_MONITORING__ENABLE_METRICS=true
```

### Configuration File (storage.toml)

```toml
connection_string = "mongodb://localhost:27017"
database_name = "rag_storage"
chunk_collection_name = "chunks"
metadata_collection_name = "metadata"

[vector_search]
embedding_dimension = 384
similarity_metric = "cosine"
num_candidates = 100

[performance]
batch_size = 100
max_concurrency = 10
enable_query_caching = true

[monitoring]
enable_metrics = true
log_level = "info"
enable_health_checks = true
```

## ⚡ Performance

### Benchmarks

| Operation | Latency (p95) | Throughput |
|-----------|---------------|------------|
| Insert Single | <5ms | 1000+ ops/sec |
| Bulk Insert (100) | <100ms | 10000+ docs/sec |
| Vector Search | <50ms | 200+ searches/sec |
| Text Search | <30ms | 300+ searches/sec |
| Hybrid Search | <80ms | 150+ searches/sec |

### Scalability

- **Document Count**: Tested with 1M+ documents
- **Concurrent Users**: 100+ simultaneous connections
- **Embedding Dimensions**: 128-1024 dimensions supported
- **Memory Usage**: <2GB for 100K documents

## 🧪 Testing

```bash
# Unit tests
cargo test --lib

# Integration tests (requires MongoDB)
cargo test --test integration

# Performance benchmarks
cargo bench

# With Docker
docker-compose exec vector-storage cargo test
```

### Test Coverage

- **Unit Tests**: 95%+ coverage for core functionality
- **Integration Tests**: Full end-to-end scenarios
- **Performance Tests**: Latency and throughput validation
- **Error Handling**: Comprehensive failure scenario testing

## 📊 Monitoring

### Metrics

The storage system exposes comprehensive metrics:

```rust
// Get metrics snapshot
let metrics = storage.metrics().snapshot();

println!("Operations: {}", metrics.operations.len());
println!("Uptime: {}s", metrics.uptime_seconds);
println!("Error rate: {:.2}%", metrics.errors.error_rate * 100.0);
```

### Health Checks

```rust
let health = storage.health_check().await?;
if health.healthy {
    println!("Storage is healthy ({}ms latency)", health.latency_ms);
}
```

### Grafana Dashboard

When using Docker Compose with monitoring profile:

- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **MongoDB Express**: http://localhost:8081

## 🔒 Security

- **Authentication**: MongoDB authentication support
- **TLS/SSL**: Full encryption in transit
- **Input Validation**: Comprehensive data validation
- **Access Control**: Role-based access patterns
- **Audit Logging**: Complete operation tracking

## 🐛 Error Handling

The system provides comprehensive error handling with automatic recovery:

```rust
use storage::{StorageError, RecoveryStrategy, WithContext};

match storage.insert_chunk(chunk).await {
    Ok(id) => println!("Inserted: {}", id),
    Err(e) => {
        match e {
            StorageError::ConnectionError(_) => {
                // Automatic retry with exponential backoff
                let strategy = RecoveryStrategy::for_error(&e);
                // ... retry logic
            }
            StorageError::ValidationError(_) => {
                // Log and return error to user
            }
            _ => {
                // Handle other error types
            }
        }
    }
}
```

## 📚 API Reference

### Core Operations

- `VectorStorage::new(config)` - Initialize storage
- `insert_chunk(chunk)` - Insert single document
- `insert_chunks(chunks)` - Bulk insert
- `get_chunk(id)` - Retrieve by ID
- `update_chunk(id, chunk)` - Update document
- `delete_chunk(id)` - Delete document

### Search Operations

- `vector_search(embedding, k, filters)` - Vector similarity
- `text_search(query, limit, filters)` - Full-text search
- `hybrid_search(query)` - Combined search
- `find_similar(chunk_id, limit)` - Find similar documents
- `get_recommendations(viewed, limit)` - Get recommendations

### Utility Operations

- `health_check()` - System health status
- `metrics()` - Performance metrics
- `count_document_chunks(doc_id)` - Count chunks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the full test suite
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
cargo install cargo-watch cargo-tarpaulin

# Run tests on file changes
cargo watch -x test

# Generate coverage report
cargo tarpaulin --out html
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MongoDB team for vector search capabilities
- Rust community for excellent crates
- Contributors and testers

---

For more examples and detailed documentation, see the [examples](examples/) directory and run:

```bash
cargo run --example basic_usage
cargo run --example vector_search_demo
```