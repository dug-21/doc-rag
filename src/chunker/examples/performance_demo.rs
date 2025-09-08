use chunker::DocumentChunker;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a chunker with default settings
    let mut chunker = DocumentChunker::new(512, 50)?;
    
    // Create test content
    let small_doc = "This is a small test document. ".repeat(100); // ~3KB
    let medium_doc = "This is a medium test document with more content. ".repeat(1000); // ~47KB
    let large_doc = "This is a large test document with lots of content for performance testing. ".repeat(10000); // ~770KB
    
    println!("Document Chunker Performance Demo");
    println!("================================");
    
    // Test small document
    println!("\n1. Small Document (~3KB):");
    let start = Instant::now();
    let chunks = chunker.chunk_document(&small_doc)?;
    let duration = start.elapsed();
    let mb_per_sec = (small_doc.len() as f64 / 1_000_000.0) / duration.as_secs_f64();
    println!("   Chunks created: {}", chunks.len());
    println!("   Processing time: {:?}", duration);
    println!("   Speed: {:.2} MB/sec", mb_per_sec);
    
    // Test medium document
    println!("\n2. Medium Document (~47KB):");
    let start = Instant::now();
    let chunks = chunker.chunk_document(&medium_doc)?;
    let duration = start.elapsed();
    let mb_per_sec = (medium_doc.len() as f64 / 1_000_000.0) / duration.as_secs_f64();
    println!("   Chunks created: {}", chunks.len());
    println!("   Processing time: {:?}", duration);
    println!("   Speed: {:.2} MB/sec", mb_per_sec);
    
    // Test large document
    println!("\n3. Large Document (~770KB):");
    let start = Instant::now();
    let chunks = chunker.chunk_document(&large_doc)?;
    let duration = start.elapsed();
    let mb_per_sec = (large_doc.len() as f64 / 1_000_000.0) / duration.as_secs_f64();
    println!("   Chunks created: {}", chunks.len());
    println!("   Processing time: {:?}", duration);
    println!("   Speed: {:.2} MB/sec", mb_per_sec);
    
    // Verify performance target
    if mb_per_sec > 100.0 {
        println!("\n✓ Performance target ACHIEVED: {:.2} MB/sec > 100 MB/sec", mb_per_sec);
    } else {
        println!("\n⚠ Performance target MISSED: {:.2} MB/sec < 100 MB/sec", mb_per_sec);
    }
    
    // Test complex document with various content types
    let complex_doc = create_complex_document();
    println!("\n4. Complex Document with Mixed Content:");
    let start = Instant::now();
    let chunks = chunker.chunk_document(&complex_doc)?;
    let duration = start.elapsed();
    let mb_per_sec = (complex_doc.len() as f64 / 1_000_000.0) / duration.as_secs_f64();
    println!("   Chunks created: {}", chunks.len());
    println!("   Processing time: {:?}", duration);
    println!("   Speed: {:.2} MB/sec", mb_per_sec);
    
    // Analyze chunk quality
    let mut content_types = std::collections::HashMap::new();
    let mut total_references = 0;
    let mut quality_scores = Vec::new();
    
    for chunk in &chunks {
        *content_types.entry(chunk.metadata.content_type.clone()).or_insert(0) += 1;
        total_references += chunk.references.len();
        quality_scores.push(chunk.metadata.quality_score);
    }
    
    println!("\n5. Quality Analysis:");
    println!("   Content types detected: {:?}", content_types);
    println!("   Total references found: {}", total_references);
    println!("   Average quality score: {:.3}", quality_scores.iter().sum::<f64>() / quality_scores.len() as f64);
    println!("   Chunks with semantic tags: {}", chunks.iter().filter(|c| !c.metadata.semantic_tags.is_empty()).count());
    
    // Verify chunk linking
    let mut properly_linked = 0;
    for i in 1..chunks.len() {
        if chunks[i].prev_chunk_id == Some(chunks[i-1].id) && chunks[i-1].next_chunk_id == Some(chunks[i].id) {
            properly_linked += 1;
        }
    }
    
    println!("   Properly linked chunks: {}/{}", properly_linked, chunks.len().saturating_sub(1));
    
    println!("\n✓ Document Chunker Demo Complete!");
    
    Ok(())
}

fn create_complex_document() -> String {
    format!(
        r#"# Document Processing System

## Overview

This document describes a high-performance document processing system with intelligent chunking capabilities.

### Key Features

- **Semantic Boundary Detection**: Uses neural networks to identify natural break points
- **Context Preservation**: Maintains document structure and relationships
- **Reference Tracking**: Identifies and tracks cross-references, citations, and links
- **Multi-Format Support**: Handles various content types seamlessly

## Technical Implementation

### Core Components

The system consists of several key components:

1. Document Chunker
2. Boundary Detector  
3. Metadata Extractor
4. Reference Tracker

```rust
pub struct DocumentChunker {{
    chunk_size: usize,
    overlap: usize,
    boundary_detector: BoundaryDetector,
    metadata_extractor: MetadataExtractor,
    reference_tracker: ReferenceTracker,
}}

impl DocumentChunker {{
    pub fn new(chunk_size: usize, overlap: usize) -> Result<Self> {{
        // Implementation details...
        Ok(DocumentChunker {{
            chunk_size,
            overlap,
            boundary_detector: BoundaryDetector::new()?,
            metadata_extractor: MetadataExtractor::new(),
            reference_tracker: ReferenceTracker::new(),
        }})
    }}
}}
```

### Performance Specifications

The system is designed to meet the following performance targets:

| Metric | Target | Current |
|--------|--------|---------|
| Throughput | >100 MB/sec | 150+ MB/sec |
| Latency | <10ms | 8ms |
| Memory | <2GB | 1.2GB |
| Accuracy | >95%% | 97.3%% |

> **Important**: All performance measurements are taken under standard test conditions with representative document samples.

### Usage Examples

Basic usage is straightforward:

```rust
let chunker = DocumentChunker::new(512, 50)?;
let chunks = chunker.chunk_document(content);

for chunk in chunks {{
    println!("Chunk ID: {{}}", chunk.id);
    println!("Content: {{}}", chunk.content);
    println!("References: {{}} found", chunk.references.len());
}}
```

For more advanced scenarios, see section 3.2 of the technical specification [1].

## References

[1] Technical Specification Document, Version 2.1
[2] Performance Benchmarking Results: https://example.com/benchmarks
[3] Neural Network Architecture Details (Internal Document)

## Appendix A: Configuration Options

The chunker supports various configuration parameters:

- `chunk_size`: Target size per chunk (default: 512)
- `overlap`: Overlap between chunks (default: 50)
- `min_chunk_size`: Minimum viable chunk size (default: 100)
- `semantic_boundaries`: Enable neural boundary detection (default: true)

For complete configuration details, consult the API documentation.

---

*This document was processed using the intelligent document chunker described herein.*
"#
    )
}