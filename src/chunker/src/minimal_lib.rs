//! Minimal chunker implementation demonstrating ruv-fann neural boundary detection

// Import the boundary module
use crate::boundary;

// Minimal chunk structure
use uuid::Uuid;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub content: String,
    pub start_offset: usize,
    pub end_offset: usize,
}

impl Chunk {
    pub fn new(content: String, start_offset: usize, end_offset: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            content,
            start_offset,
            end_offset,
        }
    }
    
    pub fn content_length(&self) -> usize {
        self.content.len()
    }
    
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

// Error handling
#[derive(Debug, thiserror::Error)]
pub enum ChunkerError {
    #[error("Invalid chunk size: {0}")]
    InvalidChunkSize(usize),
    #[error("Neural network error: {0}")]
    NeuralNetworkError(String),
    #[error("Boundary detection failed: {0}")]
    BoundaryDetectionError(String),
}

pub type Result<T> = std::result::Result<T, ChunkerError>;

// Minimal DocumentChunker that uses the boundary detection
pub struct DocumentChunker {
    chunk_size: usize,
    overlap: usize,
    boundary_detector: boundary::BoundaryDetector,
}

impl DocumentChunker {
    /// Creates a new document chunker with neural boundary detection enabled
    pub async fn new(chunk_size: usize, overlap: usize) -> Result<Self> {
        Self::with_neural_detection(chunk_size, overlap, true).await
    }
    
    /// Creates a new document chunker with configurable neural detection
    pub async fn with_neural_detection(chunk_size: usize, overlap: usize, enable_neural: bool) -> Result<Self> {
        if chunk_size < 50 {
            return Err(ChunkerError::InvalidChunkSize(chunk_size));
        }
        
        if overlap >= chunk_size {
            return Err(ChunkerError::InvalidChunkSize(overlap));
        }

        let boundary_detector = boundary::BoundaryDetector::new(enable_neural)
            .await
            .map_err(|e| ChunkerError::BoundaryDetectionError(format!("Failed to create boundary detector: {:?}", e)))?;

        Ok(DocumentChunker {
            chunk_size,
            overlap,
            boundary_detector,
        })
    }

    /// Chunks a document using semantic boundary detection
    pub async fn chunk_document(&self, content: &str) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        
        // Get semantic boundaries from neural network
        let boundaries = self.boundary_detector.detect_boundaries(content)
            .await
            .map_err(|e| ChunkerError::BoundaryDetectionError(format!("Boundary detection failed: {:?}", e)))?;
        
        // Calculate optimal chunk positions using boundaries
        let chunk_positions = self.calculate_optimal_chunk_positions(content, &boundaries);
        
        for (start, end) in chunk_positions {
            let chunk_content = &content[start..end];
            let chunk = Chunk::new(chunk_content.to_string(), start, end);
            chunks.push(chunk);
        }
        
        Ok(chunks)
    }

    /// Calculate optimal chunk positions using detected boundaries
    fn calculate_optimal_chunk_positions(&self, content: &str, boundaries: &[usize]) -> Vec<(usize, usize)> {
        let mut positions = Vec::new();
        let mut current_start = 0;
        
        while current_start < content.len() {
            let ideal_end = std::cmp::min(current_start + self.chunk_size, content.len());
            
            // Find the best boundary near the ideal end
            let best_boundary = boundaries
                .iter()
                .filter(|&&pos| pos >= current_start)
                .min_by_key(|&&pos| (pos as i32 - ideal_end as i32).abs())
                .copied()
                .unwrap_or(ideal_end);
            
            let chunk_end = std::cmp::min(best_boundary, content.len());
            
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chunker_creation() {
        let chunker = DocumentChunker::new(512, 50).await;
        assert!(chunker.is_ok());
    }

    #[tokio::test] 
    async fn test_chunker_without_neural() {
        let chunker = DocumentChunker::with_neural_detection(512, 50, false).await;
        assert!(chunker.is_ok());
    }

    #[tokio::test]
    async fn test_invalid_chunk_size() {
        let chunker = DocumentChunker::new(10, 5).await;
        assert!(chunker.is_err());
    }

    #[tokio::test]
    async fn test_basic_chunking() {
        let chunker = DocumentChunker::with_neural_detection(100, 10, false).await.unwrap();
        let content = "a".repeat(250);
        let chunks = chunker.chunk_document(&content).await.unwrap();
        
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| c.content.len() <= 100));
    }

    #[tokio::test]
    async fn test_neural_boundary_detection() {
        let chunker = DocumentChunker::with_neural_detection(512, 50, true).await.unwrap();
        let content = "This is the first paragraph with some content.\n\nThis is the second paragraph with different content.\n\nAnd this is the third paragraph.";
        let chunks = chunker.chunk_document(content).await.unwrap();
        
        assert!(!chunks.is_empty());
        // Should detect paragraph boundaries with neural network
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_chunk_methods() {
        let chunk = Chunk::new("This is a test chunk.".to_string(), 0, 21);
        
        assert_eq!(chunk.content_length(), 21);
        assert_eq!(chunk.word_count(), 5);
    }
}