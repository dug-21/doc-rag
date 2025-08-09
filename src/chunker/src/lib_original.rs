//! Intelligent Document Chunker with Neural Boundary Detection
//! 
//! This crate provides intelligent document chunking capabilities with semantic
//! boundary detection using ruv-FANN neural networks. It's designed for high-performance
//! document processing in RAG systems.
//!
//! # Features
//!
//! - Semantic boundary detection with neural networks
//! - Adaptive chunk sizing based on content analysis
//! - Cross-chunk reference tracking
//! - Comprehensive metadata extraction
//! - Async/concurrent processing capabilities
//! - High-performance optimization

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use uuid::Uuid;
use tracing::{info, debug};

// Public modules
mod boundary_simple;
pub use boundary_simple as boundary;
pub mod chunk;
pub mod metadata;
pub mod references;

// Re-exports for convenience
pub use boundary::{BoundaryDetector, BoundaryConfig, BoundaryStats};
pub use chunk::{Chunk, ChunkMetadata, ChunkReference, ReferenceType, ChunkValidationError};
pub use metadata::{MetadataExtractor, MetadataConfig, ContentAnalyzer};
pub use references::{ReferenceTracker, ReferenceResolver, CrossReferenceGraph};

// Core types
pub use uuid::Uuid as ChunkId;
pub type Result<T> = std::result::Result<T, ChunkerError>;

/// Main error type for the chunker system
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
    #[error("Reference resolution error: {0}")]
    ReferenceError(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Async task error: {0}")]
    TaskError(String),
}

/// Main document chunker with intelligent boundary detection
#[derive(Clone)]
pub struct DocumentChunker {
    /// Target chunk size in characters
    chunk_size: usize,
    /// Overlap between chunks in characters
    overlap: usize,
    /// Neural boundary detector
    boundary_detector: Arc<BoundaryDetector>,
    /// Metadata extractor
    metadata_extractor: Arc<metadata::MetadataExtractor>,
    /// Reference tracker
    reference_tracker: Arc<references::ReferenceTracker>,
    /// Chunker configuration
    config: ChunkerConfig,
}

/// Configuration for the document chunker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkerConfig {
    /// Enable neural boundary detection
    pub enable_neural_boundaries: bool,
    /// Enable async processing
    pub enable_async: bool,
    /// Maximum concurrent processing tasks
    pub max_concurrent_tasks: usize,
    /// Quality threshold for chunks (0.0-1.0)
    pub quality_threshold: f64,
    /// Enable cross-chunk reference tracking
    pub enable_references: bool,
    /// Enable comprehensive metadata extraction
    pub enable_metadata: bool,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            enable_neural_boundaries: true,
            enable_async: true,
            max_concurrent_tasks: 4,
            quality_threshold: 0.6,
            enable_references: true,
            enable_metadata: true,
        }
    }
}

impl DocumentChunker {
    /// Creates a new document chunker with default configuration
    pub async fn new(chunk_size: usize, overlap: usize) -> Result<Self> {
        Self::with_config(chunk_size, overlap, ChunkerConfig::default()).await
    }

    /// Creates a new document chunker with custom configuration
    pub async fn with_config(
        chunk_size: usize, 
        overlap: usize, 
        config: ChunkerConfig
    ) -> Result<Self> {
        if chunk_size < 50 {
            return Err(ChunkerError::InvalidChunkSize(chunk_size));
        }
        
        if overlap >= chunk_size {
            return Err(ChunkerError::InvalidChunkSize(overlap));
        }

        let boundary_detector = Arc::new(
            BoundaryDetector::new(config.enable_neural_boundaries).await
                .map_err(|e| ChunkerError::BoundaryDetectionError(e.to_string()))?
        );

        let metadata_extractor = Arc::new(
            metadata::MetadataExtractor::new().await
                .map_err(|e| ChunkerError::MetadataError(e.to_string()))?
        );

        let reference_tracker = Arc::new(
            references::ReferenceTracker::new().await
                .map_err(|e| ChunkerError::ReferenceError(e.to_string()))?
        );

        info!("DocumentChunker initialized with chunk_size: {}, overlap: {}", chunk_size, overlap);

        Ok(DocumentChunker {
            chunk_size,
            overlap,
            boundary_detector,
            metadata_extractor,
            reference_tracker,
            config,
        })
    }

    /// Chunks a document with intelligent boundary detection
    pub async fn chunk_document(&self, content: &str, document_id: String) -> Result<Vec<chunk::Chunk>> {
        info!("Chunking document '{}' with {} characters", document_id, content.len());
        
        let mut chunks: Vec<chunk::Chunk> = Vec::new();
        
        // Detect boundaries using neural network or fallback
        let boundaries = self.boundary_detector.detect_boundaries(content).await
            .map_err(|e| ChunkerError::BoundaryDetectionError(e.to_string()))?;
        
        let chunk_positions = self.calculate_optimal_chunk_positions(content, &boundaries);
        
        // Process chunks concurrently if enabled
        if self.config.enable_async && chunk_positions.len() > 1 {
            chunks = self.process_chunks_async(content, &document_id, &chunk_positions).await?;
        } else {
            chunks = self.process_chunks_sequential(content, &document_id, &chunk_positions).await?;
        }
        
        // Post-process chunks
        self.post_process_chunks(&mut chunks).await?;
        
        info!("Generated {} chunks for document '{}'", chunks.len(), document_id);
        Ok(chunks)
    }

    /// Calculates optimal chunk positions based on boundaries
    fn calculate_optimal_chunk_positions(&self, content: &str, boundaries: &[usize]) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();
        let mut current_start = 0;
        
        while current_start < content.len() {
            let ideal_end = std::cmp::min(current_start + self.chunk_size, content.len());
            
            // Find the best boundary near the ideal end
            let best_boundary = self.find_best_boundary_near(boundaries, ideal_end, self.chunk_size / 4);
            
            let actual_end = best_boundary.unwrap_or(ideal_end);
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
        
        debug!("Calculated {} chunk positions", positions.len());
        positions
    }

    /// Finds the best boundary near the target position
    fn find_best_boundary_near(&self, boundaries: &[usize], target: usize, tolerance: usize) -> Option<usize> {
        boundaries
            .iter()
            .filter(|&&pos| pos.abs_diff(target) <= tolerance)
            .min_by_key(|&&pos| pos.abs_diff(target))
            .copied()
    }

    /// Processes chunks sequentially
    async fn process_chunks_sequential(
        &self,
        content: &str,
        document_id: &str,
        chunk_positions: &[(usize, usize)],
    ) -> Result<Vec<chunk::Chunk>> {
        let mut chunks = Vec::new();
        
        for (i, (start, end)) in chunk_positions.iter().enumerate() {
            let chunk = self.create_chunk(
                content, document_id, i, chunk_positions.len(), *start, *end
            ).await?;
            
            chunks.push(chunk);
        }
        
        debug!("Processed {} chunks sequentially", chunks.len());
        Ok(chunks)
    }

    /// Processes chunks asynchronously
    async fn process_chunks_async(
        &self,
        content: &str,
        document_id: &str,
        chunk_positions: &[(usize, usize)],
    ) -> Result<Vec<chunk::Chunk>> {
        use tokio::task::JoinSet;
        
        let mut join_set = JoinSet::new();
        let content = content.to_string();
        let document_id = document_id.to_string();
        let total_chunks = chunk_positions.len();
        
        // Limit concurrent tasks
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrent_tasks));
        
        for (i, (start, end)) in chunk_positions.iter().enumerate() {
            let content = content.clone();
            let document_id = document_id.clone();
            let start = *start;
            let end = *end;
            let semaphore = semaphore.clone();
            let chunker = self.clone();
            
            join_set.spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                chunker.create_chunk(&content, &document_id, i, total_chunks, start, end).await
            });
        }
        
        let mut chunks = Vec::new();
        while let Some(result) = join_set.join_next().await {
            let chunk = result.map_err(|e| ChunkerError::TaskError(e.to_string()))??;
            chunks.push(chunk);
        }
        
        // Sort chunks by position to maintain order
        chunks.sort_by(|a, b| a.metadata.start_offset.cmp(&b.metadata.start_offset));
        
        debug!("Processed {} chunks asynchronously", chunks.len());
        Ok(chunks)
    }

    /// Creates a single chunk with metadata and references
    async fn create_chunk(
        &self,
        content: &str,
        document_id: &str,
        chunk_index: usize,
        total_chunks: usize,
        start: usize,
        end: usize,
    ) -> Result<chunk::Chunk> {
        let chunk_content = &content[start..end];
        let chunk_id = Uuid::new_v4();
        
        // Create basic chunk structure
        let mut chunk = chunk::Chunk::new(chunk_id, chunk_content.to_string(), document_id.to_string());
        
        // Update basic metadata
        chunk.metadata.chunk_index = chunk_index;
        chunk.metadata.total_chunks = total_chunks;
        chunk.metadata.start_offset = start;
        chunk.metadata.end_offset = end;
        
        // Extract references if enabled
        if self.config.enable_references {
            let references = self.reference_tracker.extract_references(
                chunk_id,
                chunk_content,
                document_id.to_string(),
                chunk_index,
            ).await.map_err(|e| ChunkerError::ReferenceError(e.to_string()))?;
            
            // Convert and add references (simplified - would need proper conversion)
            for reference in references {
                chunk.add_reference(references::ChunkReference {
                    reference_type: reference.reference_type.into(), // Conversion needed
                    target_id: reference.target_chunk_id.to_string(),
                    context: reference.context,
                    confidence: reference.confidence,
                });
            }
        }
        
        // Update quality score
        chunk.update_quality_score();
        
        Ok(chunk)
    }

    /// Post-processes chunks for quality and consistency
    async fn post_process_chunks(&self, chunks: &mut Vec<chunk::Chunk>) -> Result<()> {
        // Link chunks sequentially by updating references
        for i in 0..chunks.len() {
            // Set previous chunk reference
            if i > 0 {
                let prev_ref = chunk::ChunkReference::new(
                    chunk::ReferenceType::PreviousChunk,
                    chunks[i - 1].id.to_string(),
                    0.9
                );
                chunks[i].add_reference(prev_ref);
            }
            
            // Set next chunk reference
            if i < chunks.len() - 1 {
                let next_ref = chunk::ChunkReference::new(
                    chunk::ReferenceType::NextChunk,
                    chunks[i + 1].id.to_string(),
                    0.9
                );
                chunks[i].add_reference(next_ref);
            }
        }
        
        // Filter low-quality chunks if threshold is set
        if self.config.quality_threshold > 0.0 {
            let initial_count = chunks.len();
            chunks.retain(|chunk| chunk.metadata.quality_score >= self.config.quality_threshold);
            let filtered_count = initial_count - chunks.len();
            if filtered_count > 0 {
                debug!("Filtered out {} low-quality chunks", filtered_count);
            }
        }
        
        Ok(())
    }

    /// Returns chunker statistics
    pub async fn get_stats(&self) -> ChunkerStats {
        let boundary_stats = self.boundary_detector.get_statistics().await;
        // Note: These would need proper integration once metadata and references modules are complete
        let metadata_stats = metadata::ExtractionStats::default();
        let reference_stats = references::TrackerStats::default();
        
        ChunkerStats {
            boundary_stats,
            metadata_stats,
            reference_stats,
        }
    }
    
    /// Performs health check on all components
    pub async fn health_check(&self) -> bool {
        let boundary_health = self.boundary_detector.health_check().await;
        // Add other health checks as needed
        boundary_health
    }

    /// Returns configuration
    pub fn get_config(&self) -> &ChunkerConfig {
        &self.config
    }
}

/// Combined statistics from all chunker components
#[derive(Debug, Clone)]
pub struct ChunkerStats {
    pub boundary_stats: BoundaryStats,
    pub metadata_stats: metadata::ExtractionStats,
    pub reference_stats: references::TrackerStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chunker_creation() {
        let chunker = DocumentChunker::new(512, 50).await.unwrap();
        assert_eq!(chunker.chunk_size, 512);
        assert_eq!(chunker.overlap, 50);
        assert!(chunker.health_check().await);
    }

    #[tokio::test]
    async fn test_invalid_chunk_size() {
        let result = DocumentChunker::new(10, 5).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_basic_chunking() {
        let chunker = DocumentChunker::new(100, 10).await.unwrap();
        let content = "a".repeat(250);
        let chunks = chunker.chunk_document(&content, "test-doc".to_string()).await.unwrap();
        
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.content.len() <= 100));
    }

    #[tokio::test]
    async fn test_chunker_config() {
        let config = ChunkerConfig {
            enable_neural_boundaries: false,
            enable_async: false,
            ..Default::default()
        };
        
        let chunker = DocumentChunker::with_config(512, 50, config).await.unwrap();
        assert!(!chunker.config.enable_neural_boundaries);
        assert!(!chunker.config.enable_async);
    }
}

pub mod benches {
    use super::*;
    use std::time::Instant;

    pub async fn benchmark_chunking_performance() -> std::time::Duration {
        let chunker = DocumentChunker::new(512, 50).await.unwrap();
        let content = "This is a test document. ".repeat(100000); // ~2.4MB
        
        let start = Instant::now();
        let _chunks = chunker.chunk_document(&content, "benchmark-doc".to_string()).await.unwrap();
        start.elapsed()
    }

    pub async fn benchmark_boundary_detection() -> std::time::Duration {
        let detector = BoundaryDetector::new(true).await.unwrap();
        let content = "Paragraph one.\n\nParagraph two.\n\nParagraph three.".repeat(1000);
        
        let start = Instant::now();
        let _boundaries = detector.detect_boundaries(&content).await.unwrap();
        start.elapsed()
    }
}