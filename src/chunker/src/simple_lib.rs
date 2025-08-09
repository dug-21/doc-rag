//! Simplified chunker implementation with semantic boundary detection using ruv-fann

use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use std::sync::Arc;
use ruv_fann::{Network as FannNetwork, ActivationFunction, TrainingAlgorithm, TrainingData as TrainData};

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

// Chunk data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub content: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub quality_score: f64,
}

impl Chunk {
    pub fn new(id: Uuid, content: String, start_offset: usize, end_offset: usize) -> Self {
        Self {
            id,
            content,
            start_offset,
            end_offset,
            quality_score: 0.8, // Default score
        }
    }

    pub fn content_length(&self) -> usize {
        self.content.len()
    }

    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

// Neural boundary detector
pub struct BoundaryDetector {
    network: Arc<RwLock<FannNetwork<f64>>>,
    feature_dimension: usize,
    enabled: bool,
}

impl BoundaryDetector {
    /// Creates a new boundary detector using ruv-fann neural network
    pub async fn new(enabled: bool) -> Result<Self> {
        let feature_dimension = 20;
        let layers = [feature_dimension, 30, 20, 10, 1];
        
        let mut network = FannNetwork::new(&layers);
        
        if enabled {
            // Configure activation functions
            network.set_activation_function_hidden(ActivationFunction::Sigmoid);
            network.set_activation_function_output(ActivationFunction::Sigmoid);
            network.set_training_algorithm(TrainingAlgorithm::RProp);
            
            // Train with synthetic data
            let training_data = Self::create_synthetic_training_data(feature_dimension)?;
            network.train_on_data(&training_data.inputs, &training_data.outputs, 500, 0.001);
            
            println!("[INFO] BoundaryDetector initialized with neural network training");
        } else {
            println!("[INFO] BoundaryDetector initialized without neural network");
        }

        Ok(Self {
            network: Arc::new(RwLock::new(network)),
            feature_dimension,
            enabled,
        })
    }

    /// Creates synthetic training data for boundary detection
    fn create_synthetic_training_data(feature_dim: usize) -> Result<TrainData<f64>> {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        // Create 200 training examples
        for i in 0..100 {
            // Boundary examples (high punctuation, paragraph breaks, etc.)
            let mut features = vec![0.0; feature_dim];
            features[0] = 0.8; // High punctuation density
            features[1] = 0.9; // Paragraph break indicator
            features[2] = 0.7; // Sentence completeness
            features[3] = 0.6; // Topic shift
            
            // Add noise to other features
            for j in 4..feature_dim {
                features[j] = fastrand::f64() * 0.5;
            }
            
            inputs.push(features);
            outputs.push(vec![1.0]); // Is boundary
        }

        for i in 0..100 {
            // Non-boundary examples (low punctuation, continuation, etc.)
            let mut features = vec![0.0; feature_dim];
            features[0] = 0.2; // Low punctuation density
            features[1] = 0.1; // No paragraph break
            features[2] = 0.3; // Sentence continuation
            features[3] = 0.2; // No topic shift
            
            // Add noise
            for j in 4..feature_dim {
                features[j] = fastrand::f64() * 0.3;
            }
            
            inputs.push(features);
            outputs.push(vec![0.0]); // Not boundary
        }

        Ok(TrainData::from(inputs, outputs))
    }

    /// Extract features for boundary detection at given position
    fn extract_features(&self, text: &str, position: usize) -> Vec<f64> {
        let mut features = vec![0.0; self.feature_dimension];
        
        let window_size = 100;
        let start = position.saturating_sub(window_size / 2);
        let end = std::cmp::min(position + window_size / 2, text.len());
        
        if start >= end {
            return features;
        }
        
        let window_text = &text[start..end];
        let before_text = &text[start..position.min(text.len())];
        let after_text = &text[position.min(text.len())..end];

        // Feature 0: Punctuation density
        let punct_chars = window_text.chars().filter(|c| c.is_ascii_punctuation()).count();
        features[0] = if window_text.is_empty() { 0.0 } else { punct_chars as f64 / window_text.len() as f64 };
        
        // Feature 1: Paragraph break indicator
        features[1] = if before_text.ends_with("\n\n") || after_text.starts_with("\n\n") { 1.0 } else { 0.0 };
        
        // Feature 2: Sentence completeness
        let sentence_endings = window_text.matches(&['.', '!', '?'][..]).count();
        let estimated_sentences = window_text.split_whitespace().count() / 15; // ~15 words per sentence
        features[2] = if estimated_sentences == 0 { 0.5 } else {
            (sentence_endings as f64 / estimated_sentences as f64).min(1.0)
        };
        
        // Feature 3: Topic shift (word overlap between before/after)
        let before_words: std::collections::HashSet<&str> = before_text
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        let after_words: std::collections::HashSet<&str> = after_text
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        
        features[3] = if before_words.is_empty() || after_words.is_empty() {
            0.5
        } else {
            let intersection_size = before_words.intersection(&after_words).count();
            let union_size = before_words.union(&after_words).count();
            1.0 - (intersection_size as f64 / union_size as f64) // High value indicates topic shift
        };

        // Fill remaining features with contextual information
        for i in 4..self.feature_dimension {
            features[i] = match i % 4 {
                0 => if window_text.contains('\n') { 1.0 } else { 0.0 },
                1 => window_text.chars().filter(|c| c.is_uppercase()).count() as f64 / window_text.len().max(1) as f64,
                2 => window_text.chars().filter(|c| c.is_numeric()).count() as f64 / window_text.len().max(1) as f64,
                _ => window_text.split_whitespace().count() as f64 / 50.0, // Normalized word count
            };
        }

        features
    }

    /// Detect semantic boundaries in text using neural network
    pub async fn detect_boundaries(&self, text: &str) -> Result<Vec<usize>> {
        if !self.enabled {
            return self.detect_simple_boundaries(text).await;
        }

        let mut boundaries = vec![0]; // Always start at position 0
        let step_size = 50; // Check every 50 characters
        
        let mut pos = step_size;
        while pos < text.len() - step_size {
            let features = self.extract_features(text, pos);
            let probability = self.predict_boundary_probability(&features).await?;
            
            if probability >= 0.7 {
                boundaries.push(pos);
            }
            
            pos += step_size;
        }
        
        // Always end at text length
        if boundaries.last() != Some(&text.len()) {
            boundaries.push(text.len());
        }
        
        Ok(boundaries)
    }

    /// Predict boundary probability using neural network
    async fn predict_boundary_probability(&self, features: &[f64]) -> Result<f64> {
        if !self.enabled {
            return Ok(0.5);
        }

        let mut network = self.network.write().await;
        let output = network.run(features);
        
        Ok(output.get(0).copied().unwrap_or(0.0))
    }

    /// Simple fallback boundary detection without neural network
    async fn detect_simple_boundaries(&self, text: &str) -> Result<Vec<usize>> {
        let mut boundaries = vec![0];
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            // Look for paragraph breaks
            if i > 0 && chars[i] == '\n' && 
               i + 1 < chars.len() && chars[i + 1] == '\n' {
                boundaries.push(i + 2);
                i += 2;
                continue;
            }

            // Look for sentence boundaries followed by significant whitespace
            if (chars[i] == '.' || chars[i] == '!' || chars[i] == '?') &&
               i + 1 < chars.len() && chars[i + 1] == ' ' {
                
                // Check if there's a capital letter soon after
                let mut j = i + 2;
                while j < chars.len() && j < i + 10 && chars[j].is_whitespace() {
                    j += 1;
                }
                
                if j < chars.len() && chars[j].is_uppercase() {
                    boundaries.push(i + 1);
                }
            }

            i += 1;
        }

        if boundaries.last() != Some(&text.len()) {
            boundaries.push(text.len());
        }

        Ok(boundaries)
    }
}

// Document chunker with neural boundary detection
pub struct DocumentChunker {
    chunk_size: usize,
    overlap: usize,
    boundary_detector: BoundaryDetector,
}

impl DocumentChunker {
    /// Create a new document chunker with neural boundary detection
    pub async fn new(chunk_size: usize, overlap: usize) -> Result<Self> {
        Self::with_neural_detection(chunk_size, overlap, true).await
    }

    /// Create a document chunker with configurable neural detection
    pub async fn with_neural_detection(chunk_size: usize, overlap: usize, enable_neural: bool) -> Result<Self> {
        if chunk_size < 50 {
            return Err(ChunkerError::InvalidChunkSize(chunk_size));
        }
        
        if overlap >= chunk_size {
            return Err(ChunkerError::InvalidChunkSize(overlap));
        }

        let boundary_detector = BoundaryDetector::new(enable_neural)
            .await
            .map_err(|e| ChunkerError::BoundaryDetectionError(format!("Failed to create boundary detector: {:?}", e)))?;

        Ok(Self {
            chunk_size,
            overlap,
            boundary_detector,
        })
    }

    /// Chunk a document using semantic boundary detection
    pub async fn chunk_document(&self, content: &str) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        
        // Get semantic boundaries from neural network
        let boundaries = self.boundary_detector.detect_boundaries(content).await?;
        
        // Calculate optimal chunk positions using boundaries
        let chunk_positions = self.calculate_optimal_chunk_positions(content, &boundaries);
        
        for (i, (start, end)) in chunk_positions.iter().enumerate() {
            let chunk_content = &content[*start..*end];
            let chunk_id = Uuid::new_v4();
            
            let chunk = Chunk::new(chunk_id, chunk_content.to_string(), *start, *end);
            chunks.push(chunk);
        }
        
        println!("[INFO] Created {} chunks using neural boundary detection", chunks.len());
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
                .filter(|&&pos| pos >= current_start && (pos as i32 - ideal_end as i32).abs() <= (self.chunk_size / 4) as i32)
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
    async fn test_boundary_detector_creation() {
        let detector = BoundaryDetector::new(false).await.unwrap();
        // Basic test - just ensure it creates without error
        assert!(!detector.enabled);
    }

    #[tokio::test] 
    async fn test_boundary_detector_with_neural() {
        let detector = BoundaryDetector::new(true).await.unwrap();
        assert!(detector.enabled);
    }

    #[tokio::test]
    async fn test_chunker_creation() {
        let chunker = DocumentChunker::new(512, 50).await.unwrap();
        assert_eq!(chunker.chunk_size, 512);
        assert_eq!(chunker.overlap, 50);
    }

    #[tokio::test]
    async fn test_basic_chunking() {
        let chunker = DocumentChunker::with_neural_detection(100, 10, false).await.unwrap();
        let content = "This is a test document with multiple sentences. Each sentence should help with boundary detection. The neural network should identify good places to split.".repeat(5);
        let chunks = chunker.chunk_document(&content).await.unwrap();
        
        assert!(!chunks.is_empty());
        // Verify chunks don't exceed max size (allowing some flexibility for boundary detection)
        assert!(chunks.iter().all(|c| c.content.len() <= 150)); // Allow some buffer
    }

    #[tokio::test]
    async fn test_neural_boundary_detection() {
        let chunker = DocumentChunker::with_neural_detection(200, 20, true).await.unwrap();
        let content = "First paragraph with some content.\n\nSecond paragraph with different content.\n\nThird paragraph with more information.";
        let chunks = chunker.chunk_document(content).await.unwrap();
        
        assert!(!chunks.is_empty());
        println!("Created {} chunks with neural detection", chunks.len());
    }

    #[test]
    fn test_chunk_methods() {
        let chunk = Chunk::new(
            Uuid::new_v4(),
            "This is a test chunk with multiple words.".to_string(),
            0,
            42
        );
        
        assert_eq!(chunk.content_length(), 42);
        assert_eq!(chunk.word_count(), 8);
        assert_eq!(chunk.quality_score, 0.8);
    }
}