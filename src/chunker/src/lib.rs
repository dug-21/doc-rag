//! Intelligent document chunker with semantic boundary detection
//! 
//! This library provides comprehensive document chunking capabilities with:
//! - Semantic boundary detection using pattern matching and heuristics
//! - Comprehensive metadata extraction for each chunk
//! - Cross-reference detection and preservation
//! - Support for tables, lists, code blocks, and other structured content

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

// Re-export modules
pub mod boundary;
pub mod chunk;
pub mod metadata;
pub mod references_simple;
pub use references_simple as references;

// Re-export important types for convenience
pub use boundary::{BoundaryDetector, BoundaryInfo, BoundaryType};
pub use chunk::{Chunk, ChunkMetadata, ChunkReference, ChunkValidationError};
pub use metadata::{MetadataExtractor, ExtendedMetadata, ContentAnalysis, StructureInfo, QualityMetrics};
pub use references::{ReferenceTracker, ExtendedReference, ReferenceGraph, ReferenceValidationResult};

/// Main error types for the chunker
#[derive(Error, Debug)]
pub enum ChunkerError {
    #[error("Invalid chunk size: {0}")]
    InvalidChunkSize(usize),
    #[error("Neural network error: {0}")]
    NeuralNetworkError(String),
    #[error("Boundary detection failed: {0}")]
    BoundaryDetectionError(String),
    #[error("Metadata extraction error: {0}")]
    MetadataError(String),
}

pub type Result<T> = std::result::Result<T, ChunkerError>;

/// Content types supported by the chunker
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ContentType {
    PlainText,
    CodeBlock,
    Table,
    List,
    Header,
    Quote,
    Footnote,
    Reference,
    Mathematical,
}

// Re-export ReferenceType from chunk module to avoid duplication
pub use chunk::ReferenceType;

/// Main document chunker with comprehensive analysis capabilities
pub struct DocumentChunker {
    chunk_size: usize,
    overlap: usize,
    boundary_detector: BoundaryDetector,
    metadata_extractor: metadata::MetadataExtractor,
    reference_tracker: references::ReferenceTracker,
}

/// Extended chunk with comprehensive metadata and reference analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedChunk {
    /// Unique chunk identifier
    pub id: Uuid,
    /// Chunk content text
    pub content: String,
    /// Source document identifier
    pub document_id: String,
    /// Extended metadata with detailed analysis
    pub extended_metadata: metadata::ExtendedMetadata,
    /// Extended references with full analysis
    pub extended_references: Vec<references::ExtendedReference>,
    /// Vector embeddings (populated by embedding service)
    pub embeddings: Option<Vec<f32>>,
    /// Previous chunk in sequence
    pub prev_chunk_id: Option<Uuid>,
    /// Next chunk in sequence
    pub next_chunk_id: Option<Uuid>,
    /// Parent chunk (for hierarchical chunking)
    pub parent_chunk_id: Option<Uuid>,
    /// Child chunks (for hierarchical chunking)
    pub child_chunk_ids: Vec<Uuid>,
    /// Creation timestamp
    pub creation_timestamp: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Chunk validation report
#[derive(Debug, Clone)]
pub struct ChunkValidationReport {
    /// Total number of chunks validated
    pub total_chunks: usize,
    /// Number of valid chunks
    pub valid_chunks: usize,
    /// Validation errors
    pub errors: Vec<ChunkValidationError>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Overall validation score
    pub validation_score: f32,
}

impl DocumentChunker {
    /// Creates a new document chunker with specified parameters
    pub fn new(chunk_size: usize, overlap: usize) -> Result<Self> {
        if chunk_size < 50 {
            return Err(ChunkerError::InvalidChunkSize(chunk_size));
        }
        
        if overlap >= chunk_size {
            return Err(ChunkerError::InvalidChunkSize(overlap));
        }

        Ok(DocumentChunker {
            chunk_size,
            overlap,
            boundary_detector: BoundaryDetector::new()?,
            metadata_extractor: metadata::MetadataExtractor::new(),
            reference_tracker: references::ReferenceTracker::new(),
        })
    }

    /// Chunks document using semantic boundary detection
    pub fn chunk_document(&self, content: &str) -> Result<Vec<Chunk>> {
        let boundaries = self.boundary_detector.detect_boundaries(content)?;
        let chunk_positions = self.calculate_optimal_chunk_positions(content, &boundaries);
        let mut chunks: Vec<Chunk> = Vec::new();
        let mut previous_chunk_id: Option<Uuid> = None;
        
        for (i, (start, end)) in chunk_positions.iter().enumerate() {
            let chunk_content = &content[*start..*end];
            let chunk_id = Uuid::new_v4();
            
            let metadata = self.metadata_extractor.extract_metadata(
                chunk_content,
                *start,
                i,
                content,
            );
            
            let references = self.reference_tracker.extract_references(chunk_content);
            
            let chunk = Chunk {
                id: chunk_id,
                content: chunk_content.to_string(),
                metadata,
                embeddings: None,
                references,
                prev_chunk_id: previous_chunk_id,
                next_chunk_id: None,
                parent_chunk_id: None,
                child_chunk_ids: Vec::new(),
            };
            
            // Update previous chunk's next_chunk_id
            if previous_chunk_id.is_some() {
                if let Some(prev_chunk) = chunks.last_mut() {
                    prev_chunk.next_chunk_id = Some(chunk_id);
                }
            }
            
            chunks.push(chunk);
            previous_chunk_id = Some(chunk_id);
        }
        
        self.preserve_context(&mut chunks);
        Ok(chunks)
    }
    
    /// Chunks document with extended metadata and reference analysis
    pub fn chunk_document_extended(&self, content: &str, document_id: &str) -> Result<Vec<ExtendedChunk>> {
        let boundaries = self.boundary_detector.detect_boundaries(content)?;
        let chunk_positions = self.calculate_optimal_chunk_positions(content, &boundaries);
        let mut chunks: Vec<ExtendedChunk> = Vec::new();
        let mut previous_chunk_id: Option<Uuid> = None;
        
        for (i, (start, end)) in chunk_positions.iter().enumerate() {
            let chunk_content = &content[*start..*end];
            let chunk_id = Uuid::new_v4();
            
            // Extract extended metadata
            let extended_metadata = self.metadata_extractor.extract_extended_metadata(
                chunk_content,
                *start,
                i,
                content,
            );
            
            // Extract extended references
            let extended_references = self.reference_tracker.extract_extended_references(
                chunk_content,
                chunk_id,
            );
            
            let chunk = ExtendedChunk {
                id: chunk_id,
                content: chunk_content.to_string(),
                document_id: document_id.to_string(),
                extended_metadata,
                extended_references,
                embeddings: None,
                prev_chunk_id: previous_chunk_id,
                next_chunk_id: None,
                parent_chunk_id: None,
                child_chunk_ids: Vec::new(),
                creation_timestamp: chrono::Utc::now(),
                last_updated: chrono::Utc::now(),
            };
            
            chunks.push(chunk);
            previous_chunk_id = Some(chunk_id);
        }
        
        Ok(chunks)
    }
    
    /// Validates and optimizes chunk quality
    pub fn validate_chunks(&self, chunks: &mut Vec<Chunk>) -> ChunkValidationReport {
        let mut report = ChunkValidationReport::new();
        
        for chunk in chunks.iter_mut() {
            // Update quality scores based on comprehensive analysis
            chunk.update_quality_score();
            
            // Validate chunk consistency
            if let Err(error) = chunk.validate() {
                report.errors.push(ChunkValidationError::ValidationFailed {
                    chunk_id: chunk.id,
                    error: error.to_string(),
                });
            }
            
            // Validate references
            let ref_validations = self.reference_tracker.validate_references(&chunk.references);
            for validation in ref_validations {
                if !validation.is_valid {
                    report.warnings.push(format!("Chunk {} has invalid references", chunk.id));
                }
            }
            
            // Check for quality issues
            if chunk.metadata.quality_score < 0.3 {
                report.warnings.push(format!("Chunk {} has low quality score: {:.2}", 
                    chunk.id, chunk.metadata.quality_score));
            }
        }
        
        report.total_chunks = chunks.len();
        report.valid_chunks = chunks.len() - report.errors.len();
        report.calculate_score();
        report
    }

    /// Calculates optimal chunk positions based on boundaries
    fn calculate_optimal_chunk_positions(&self, content: &str, boundaries: &[BoundaryInfo]) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();
        let mut current_start = 0;
        
        while current_start < content.len() {
            let ideal_end = std::cmp::min(current_start + self.chunk_size, content.len());
            
            // Find the best boundary near the ideal end
            let best_boundary = self.find_best_boundary_near(boundaries, ideal_end, self.chunk_size / 4);
            
            let actual_end = best_boundary.map_or(ideal_end, |b| b.position);
            let chunk_end = std::cmp::min(actual_end, content.len());
            
            if chunk_end > current_start {
                positions.push((current_start, chunk_end));
            }
            
            // Calculate next start with overlap
            current_start = if chunk_end >= content.len() {
                content.len()
            } else {
                chunk_end.saturating_sub(self.overlap)
            };
            
            if current_start >= content.len() {
                break;
            }
        }
        
        positions
    }

    /// Finds the best boundary near a target position
    fn find_best_boundary_near<'a>(&self, boundaries: &'a [BoundaryInfo], target: usize, tolerance: usize) -> Option<&'a BoundaryInfo> {
        boundaries
            .iter()
            .filter(|b| b.position.abs_diff(target) <= tolerance)
            .max_by(|a, b| {
                let a_score = a.confidence * a.semantic_strength;
                let b_score = b.confidence * b.semantic_strength;
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Preserves context between chunks
    fn preserve_context(&self, chunks: &mut Vec<Chunk>) {
        let total_chunks = chunks.len();
        for (i, chunk) in chunks.iter_mut().enumerate() {
            let mut context_tags = Vec::new();
            
            // Add section context
            if let Some(section) = &chunk.metadata.section {
                context_tags.push(format!("section:{}", section));
            }
            
            // Add position context
            context_tags.push(format!("position:{}/{}", i + 1, total_chunks));
            
            // Add content type context
            context_tags.push(format!("type:{}", chunk.metadata.content_type));
            
            chunk.metadata.semantic_tags = context_tags;
        }
    }
}

impl ChunkValidationReport {
    fn new() -> Self {
        Self {
            total_chunks: 0,
            valid_chunks: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
            validation_score: 0.0,
        }
    }
    
    /// Calculates the overall validation score
    pub fn calculate_score(&mut self) {
        if self.total_chunks == 0 {
            self.validation_score = 0.0;
            return;
        }
        
        let base_score = self.valid_chunks as f32 / self.total_chunks as f32;
        let warning_penalty = (self.warnings.len() as f32 * 0.1).min(0.3);
        
        self.validation_score = (base_score - warning_penalty).max(0.0);
    }
}

impl ExtendedChunk {
    /// Returns the content length in characters
    pub fn content_length(&self) -> usize {
        self.content.len()
    }
    
    /// Returns the word count
    pub fn word_count(&self) -> usize {
        self.extended_metadata.base.word_count
    }
    
    /// Returns the overall quality score
    pub fn quality_score(&self) -> f32 {
        self.extended_metadata.quality_metrics.overall_score
    }
    
    /// Checks if chunk has high-quality references
    pub fn has_high_quality_references(&self) -> bool {
        self.extended_references.iter()
            .any(|r| r.quality.overall_score > 0.8)
    }
}

/// Benchmarking utilities for performance testing
pub mod benches {
    use super::*;
    use std::time::Instant;

    /// Benchmarks basic chunking performance
    pub fn benchmark_chunking_performance() -> std::time::Duration {
        let chunker = DocumentChunker::new(512, 50).unwrap();
        let content = "This is a test document. ".repeat(100000); // ~2.4MB
        
        let start = Instant::now();
        let _chunks = chunker.chunk_document(&content).unwrap();
        start.elapsed()
    }

    /// Benchmarks boundary detection performance
    pub fn benchmark_boundary_detection() -> std::time::Duration {
        let detector = BoundaryDetector::new().unwrap();
        let content = "Paragraph one.\n\nParagraph two.\n\nParagraph three.".repeat(1000);
        
        let start = Instant::now();
        let _boundaries = detector.detect_boundaries(&content).unwrap();
        start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunker_creation() {
        let chunker = DocumentChunker::new(512, 50);
        assert!(chunker.is_ok());
    }

    #[test]
    fn test_invalid_chunk_size() {
        let chunker = DocumentChunker::new(10, 5);
        assert!(chunker.is_err());
    }

    #[test]
    fn test_basic_chunking() {
        let chunker = DocumentChunker::new(100, 10).unwrap();
        let content = "a".repeat(250);
        let chunks = chunker.chunk_document(&content).unwrap();
        
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.content.len() <= 100 + 10)); // Allow for overlap
    }

    #[test]
    fn test_chunk_linking() {
        let chunker = DocumentChunker::new(50, 5).unwrap();
        let content = "First chunk content. Second chunk content. Third chunk content.";
        let chunks = chunker.chunk_document(content).unwrap();
        
        if chunks.len() > 1 {
            assert_eq!(chunks[0].next_chunk_id, Some(chunks[1].id));
            assert_eq!(chunks[1].prev_chunk_id, Some(chunks[0].id));
        }
    }

    #[test]
    fn test_metadata_extraction() {
        let chunker = DocumentChunker::new(512, 50).unwrap();
        let content = "# Main Title\n\nThis is some content with multiple words and sentences.";
        let chunks = chunker.chunk_document(content).unwrap();
        
        assert!(!chunks.is_empty());
        assert!(chunks[0].metadata.word_count > 0);
        assert!(chunks[0].metadata.character_count > 0);
    }

    #[test]
    fn test_reference_detection() {
        let chunker = DocumentChunker::new(512, 50).unwrap();
        let content = "Please see section 2.1 for more details. Also check [1] for references.";
        let chunks = chunker.chunk_document(content).unwrap();
        
        assert!(!chunks.is_empty());
        // References should be detected
        assert!(!chunks[0].references.is_empty());
    }

    #[test]
    fn test_extended_chunking() {
        let chunker = DocumentChunker::new(512, 50).unwrap();
        let content = "# Header\n\nThis is test content with a table:\n\n| Col1 | Col2 |\n|------|------|\n| A    | B    |\n\nAnd a list:\n- Item 1\n- Item 2";
        let chunks = chunker.chunk_document_extended(content, "test-doc").unwrap();
        
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].document_id, "test-doc");
        assert!(chunks[0].quality_score() > 0.0);
    }

    #[test]
    fn test_chunk_validation() {
        let chunker = DocumentChunker::new(512, 50).unwrap();
        let content = "Valid content for testing chunk validation.";
        let mut chunks = chunker.chunk_document(content).unwrap();
        
        let report = chunker.validate_chunks(&mut chunks);
        
        assert_eq!(report.total_chunks, chunks.len());
        assert!(report.validation_score >= 0.0);
        assert!(report.validation_score <= 1.0);
    }
}