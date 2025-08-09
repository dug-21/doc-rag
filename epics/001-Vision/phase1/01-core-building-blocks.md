# Phase 1: Core Building Blocks

## Overview
Phase 1 implements the foundational components of the RAG system using a building-block approach. Each component is fully functional, tested, and valuable on its own before integration.

## Building Block Sequence

### Block 1: MCP Protocol Adapter (Week 1)
**Purpose**: Async communication foundation for all components

#### Implementation
```rust
// mcp-adapter/src/lib.rs
pub struct McpAdapter {
    client: McpClient,
    auth: AuthHandler,
    message_queue: Arc<RwLock<MessageQueue>>,
}

impl McpAdapter {
    pub async fn connect(&mut self, endpoint: &str) -> Result<Connection> {
        // Full implementation with retry logic
    }
    
    pub async fn authenticate(&mut self, credentials: Credentials) -> Result<Token> {
        // OAuth2/JWT authentication
    }
    
    pub async fn send_message(&self, msg: Message) -> Result<Response> {
        // Async message handling with timeout
    }
}
```

#### Test Criteria
- [ ] Connection establishment with retry
- [ ] Authentication flow with token refresh
- [ ] Message serialization/deserialization
- [ ] Timeout handling
- [ ] Concurrent message handling
- [ ] Performance: <10ms message latency

#### Dependencies
- tokio = "1.35"
- serde = "1.0"
- async-trait = "0.1"

---

### Block 2: Document Chunker (Week 1-2)
**Purpose**: Intelligent document segmentation with semantic boundaries

#### Implementation
```rust
// chunker/src/lib.rs
pub struct DocumentChunker {
    chunk_size: usize,
    overlap: usize,
    boundary_detector: BoundaryDetector,
}

impl DocumentChunker {
    pub fn chunk_document(&self, content: &str) -> Vec<Chunk> {
        // Intelligent chunking with semantic boundaries
    }
    
    pub fn preserve_context(&self, chunks: &mut Vec<Chunk>) {
        // Maintain cross-references and context
    }
}

pub struct Chunk {
    pub id: Uuid,
    pub content: String,
    pub metadata: ChunkMetadata,
    pub embeddings: Option<Vec<f32>>,
    pub references: Vec<ChunkReference>,
}
```

#### Test Criteria
- [ ] Chunks maintain semantic coherence
- [ ] No critical information split
- [ ] Metadata preservation
- [ ] Cross-reference tracking
- [ ] Performance: 100MB/sec processing
- [ ] Edge cases: tables, lists, code blocks

#### Dependencies
- ruv-fann = "0.3.0" (for boundary detection)
- uuid = "1.6"

---

### Block 3: Embedding Generator (Week 2)
**Purpose**: Vector representations for chunks

#### Implementation
```rust
// embedder/src/lib.rs
pub struct EmbeddingGenerator {
    model: EmbeddingModel,
    dimension: usize,
    batch_size: usize,
}

impl EmbeddingGenerator {
    pub async fn generate_embeddings(&self, chunks: Vec<Chunk>) -> Result<Vec<EmbeddedChunk>> {
        // Batch processing with progress tracking
    }
    
    pub fn calculate_similarity(&self, emb1: &[f32], emb2: &[f32]) -> f32 {
        // Cosine similarity calculation
    }
}
```

#### Test Criteria
- [ ] Embedding quality validation
- [ ] Batch processing efficiency
- [ ] Memory management for large batches
- [ ] Similarity calculations accurate
- [ ] Performance: 1000 chunks/sec
- [ ] Error handling for model failures

#### Dependencies
- candle = "0.3" (for embedding models)
- ndarray = "0.15"

---

### Block 4: MongoDB Vector Storage (Week 2-3)
**Purpose**: Persistent storage with vector search

#### Implementation
```rust
// storage/src/lib.rs
pub struct VectorStorage {
    client: MongoClient,
    database: Database,
    chunk_collection: Collection<Chunk>,
    vector_index: VectorIndex,
}

impl VectorStorage {
    pub async fn store_chunks(&self, chunks: Vec<EmbeddedChunk>) -> Result<()> {
        // Bulk insert with transaction support
    }
    
    pub async fn vector_search(&self, query_embedding: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // KNN search with filtering
    }
    
    pub async fn hybrid_search(&self, query: SearchQuery) -> Result<Vec<SearchResult>> {
        // Combined vector + keyword search
    }
}
```

#### Test Criteria
- [ ] CRUD operations functional
- [ ] Vector indexing working
- [ ] Search accuracy validated
- [ ] Transaction support tested
- [ ] Performance: <50ms search latency
- [ ] Concurrent access handling

#### Dependencies
- mongodb = "2.7" (with vector support)
- bson = "2.7"

---

### Block 5: Query Processor (Week 3)
**Purpose**: Query understanding and decomposition

#### Implementation
```rust
// query/src/lib.rs
pub struct QueryProcessor {
    chunker: DocumentChunker,
    embedder: EmbeddingGenerator,
    intent_classifier: IntentClassifier,
}

impl QueryProcessor {
    pub async fn process_query(&self, query: &str) -> Result<ProcessedQuery> {
        // Query analysis and decomposition
    }
    
    pub fn extract_entities(&self, query: &str) -> Vec<Entity> {
        // Entity extraction for targeted search
    }
    
    pub async fn generate_search_strategy(&self, query: ProcessedQuery) -> SearchStrategy {
        // Multi-stage search planning
    }
}
```

#### Test Criteria
- [ ] Query intent classification accurate
- [ ] Entity extraction validated
- [ ] Search strategy generation tested
- [ ] Complex queries handled
- [ ] Performance: <100ms processing
- [ ] Edge cases covered

#### Dependencies
- Previous blocks
- regex = "1.10"

---

### Block 6: Response Generator (Week 3-4)
**Purpose**: Generate validated responses with citations

#### Implementation
```rust
// response/src/lib.rs
pub struct ResponseGenerator {
    validator: ResponseValidator,
    citation_tracker: CitationTracker,
}

impl ResponseGenerator {
    pub async fn generate_response(&self, context: Context, query: ProcessedQuery) -> Result<Response> {
        // Response generation with validation
    }
    
    pub fn add_citations(&self, response: &mut Response, sources: Vec<Source>) {
        // Citation attachment and formatting
    }
    
    pub fn validate_response(&self, response: &Response) -> ValidationResult {
        // Multi-layer validation
    }
}
```

#### Test Criteria
- [ ] Response relevance validated
- [ ] Citations correctly attached
- [ ] No hallucinations detected
- [ ] Validation layers functional
- [ ] Performance: <500ms generation
- [ ] Error handling comprehensive

#### Dependencies
- All previous blocks

---

## Integration Tests (Week 4)

### End-to-End Test Scenarios
1. **Simple Query Flow**
   - Input: "What is required for data encryption?"
   - Expected: Response with specific requirements and citations

2. **Complex Multi-Part Query**
   - Input: "Compare encryption requirements between PCI DSS 3.2 and 4.0"
   - Expected: Detailed comparison with citations from both versions

3. **Edge Case Handling**
   - Malformed queries
   - No relevant content
   - Ambiguous queries

### Performance Benchmarks
- Query latency: p50 < 1s, p99 < 2s
- Throughput: 100 QPS sustained
- Memory usage: < 2GB per service
- CPU usage: < 80% under load

## Docker Architecture

```yaml
# docker-compose.yml
version: '3.8'
services:
  mongodb:
    image: mongodb/mongodb-community-server:7.0-ubuntu2204
    environment:
      MONGODB_INITDB_ROOT_USERNAME: admin
      MONGODB_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
    volumes:
      - mongo_data:/data/db
    ports:
      - "27017:27017"
    command: ["--replSet", "rs0", "--bind_ip_all"]
    
  mcp-adapter:
    build:
      context: ./mcp-adapter
      dockerfile: Dockerfile
    environment:
      MCP_ENDPOINT: ${MCP_ENDPOINT}
      MCP_AUTH_TOKEN: ${MCP_AUTH_TOKEN}
    depends_on:
      - mongodb
    
  chunker:
    build:
      context: ./chunker
      dockerfile: Dockerfile
    environment:
      CHUNK_SIZE: 512
      OVERLAP_SIZE: 50
    
  embedder:
    build:
      context: ./embedder
      dockerfile: Dockerfile
    environment:
      MODEL_PATH: /models/all-MiniLM-L6-v2
    volumes:
      - ./models:/models
    
  query-processor:
    build:
      context: ./query
      dockerfile: Dockerfile
    depends_on:
      - mcp-adapter
      - chunker
      - embedder
    
  response-generator:
    build:
      context: ./response
      dockerfile: Dockerfile
    depends_on:
      - query-processor
      - mongodb

volumes:
  mongo_data:
  models:
```

## CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          components: rustfmt, clippy
      
      - name: Cache
        uses: Swatinem/rust-cache@v2
      
      - name: Format Check
        run: cargo fmt --all -- --check
      
      - name: Clippy
        run: cargo clippy --all-targets --all-features -- -D warnings
      
      - name: Test
        run: cargo test --all --verbose
      
      - name: Benchmark
        run: cargo bench --all
      
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Docker Build
        run: docker-compose build
      
      - name: Integration Tests
        run: docker-compose run tests
      
      - name: Push to Registry
        if: github.ref == 'refs/heads/main'
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker-compose push
```

## Success Criteria

### Phase 1 Completion
- [ ] All 6 building blocks implemented and tested
- [ ] Integration tests passing
- [ ] Docker containers running
- [ ] CI/CD pipeline operational
- [ ] Performance targets met
- [ ] Documentation complete

### Validation Metrics
- Code coverage > 90%
- All benchmarks passing
- No critical security issues
- Memory leaks detected: 0
- Race conditions detected: 0

## Next Steps (Phase 2 Preview)
- DAA integration for orchestration
- ruv-FANN neural processing integration
- FACT system adaptation for compliance
- Byzantine consensus implementation
- Multi-agent coordination