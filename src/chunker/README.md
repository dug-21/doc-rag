# Document Chunker

High-performance intelligent document chunker with semantic boundary detection using ruv-FANN neural networks.

## Features

- **Intelligent Chunking**: Uses ruv-FANN neural networks for semantic boundary detection
- **Context Preservation**: Maintains cross-references and document structure
- **Metadata Extraction**: Comprehensive metadata tracking for each chunk
- **Cross-Reference Handling**: Detects and tracks various reference types
- **Multi-Content Support**: Handles tables, lists, code blocks, headers, and quotes
- **High Performance**: Target >100MB/sec processing speed
- **Comprehensive Testing**: >90% test coverage with property-based tests

## Quick Start

```rust
use chunker::DocumentChunker;

let chunker = DocumentChunker::new(512, 50).unwrap();
let content = "Your document content here...";
let chunks = chunker.chunk_document(content);

for chunk in chunks {
    println!("Chunk ID: {}", chunk.id);
    println!("Content: {}", chunk.content);
    println!("Metadata: {:?}", chunk.metadata);
    println!("References: {} found", chunk.references.len());
}
```

## Configuration

### Chunk Settings
- `chunk_size`: Maximum size per chunk (recommended: 512-1024 chars)
- `overlap`: Overlap between consecutive chunks (recommended: 10-20% of chunk_size)

### Content Types Supported
- Plain text
- Code blocks (fenced with ```)
- Tables (pipe-delimited)
- Lists (bulleted and numbered)
- Headers (Markdown-style)
- Quotes (> prefixed)
- Mathematical expressions

### Reference Types Detected
- Cross-references (e.g., "see section 2.1")
- Citations (e.g., [1], (Smith et al., 2024))
- Footnotes
- Table/Figure references
- External links (URLs)

## Architecture

### Core Components

1. **DocumentChunker**: Main orchestrator
2. **BoundaryDetector**: Neural network-based semantic boundary detection
3. **MetadataExtractor**: Extracts comprehensive chunk metadata
4. **ReferenceTracker**: Identifies and tracks cross-references

### Neural Network Features

The boundary detector uses a 50-25-12-1 neural network with:
- 50 input features (character, word, sentence, semantic, context)
- Trained on boundary detection patterns
- Confidence threshold filtering
- Fallback to pattern-based detection

## Performance

### Benchmarks
- **Target**: >100MB/sec processing speed
- **Latency**: <10ms per chunk for typical documents
- **Memory**: Efficient processing with minimal memory overhead
- **Concurrency**: Thread-safe for concurrent processing

### Test Coverage
- Unit tests: >95% coverage
- Integration tests: Full pipeline testing
- Property-based tests: Edge case validation
- Performance benchmarks: Regression detection

## Building

```bash
# Build library
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Build Docker image
docker build -t chunker:latest .
```

## Testing

```bash
# Unit tests
cargo test --lib

# Integration tests
cargo test --test integration

# Property-based tests (slower)
cargo test prop_

# Performance tests
cargo bench

# Coverage report
cargo tarpaulin --out Html
```

## Docker Usage

```bash
# Build image
docker build -t chunker:latest .

# Run tests in container
docker run --rm chunker:latest cargo test

# Interactive development
docker run -it --rm -v $(pwd):/app chunker:latest bash
```

## Dependencies

### Core Dependencies
- `ruv-fann`: Neural network for boundary detection
- `uuid`: Unique chunk identification
- `serde`: Serialization support
- `tokio`: Async runtime
- `regex`: Pattern matching
- `unicode-segmentation`: Text processing

### Development Dependencies
- `criterion`: Performance benchmarking
- `proptest`: Property-based testing
- `tempfile`: Test file management

## Examples

### Basic Usage
```rust
use chunker::DocumentChunker;

let chunker = DocumentChunker::new(512, 50)?;
let chunks = chunker.chunk_document("Your content...");
```

### Advanced Configuration
```rust
use chunker::{DocumentChunker, BoundaryDetector};

let chunker = DocumentChunker::new(1024, 100)?;
let chunks = chunker.chunk_document(complex_document);

// Process results
for chunk in chunks {
    println!("Section: {:?}", chunk.metadata.section);
    println!("Type: {:?}", chunk.metadata.content_type);
    println!("Quality: {:.2}", chunk.metadata.quality_score);
}
```

### Reference Processing
```rust
for chunk in chunks {
    for reference in &chunk.references {
        println!("Found {:?}: {} (confidence: {:.2})", 
                reference.reference_type, 
                reference.target_text,
                reference.confidence);
    }
}
```

## Contributing

1. Follow Rust standard formatting (`cargo fmt`)
2. Ensure all tests pass (`cargo test`)
3. Maintain >90% test coverage
4. Run clippy for linting (`cargo clippy`)
5. Update benchmarks for performance-critical changes

## License

MIT License - see LICENSE file for details.