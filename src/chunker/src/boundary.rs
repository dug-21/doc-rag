//! Semantic boundary detection for document chunking
//!
//! This module provides boundary detection capabilities using neural networks
//! via ruv-FANN for intelligent boundary identification.

use crate::{Result, neural_chunker::NeuralChunker};
use serde::{Deserialize, Serialize};

/// Boundary detector for identifying optimal chunk boundaries using neural networks
#[derive(Debug)]
pub struct BoundaryDetector {
    /// Neural boundary detector using ruv-FANN
    neural_chunker: NeuralChunker,
    /// Configuration parameters
    config: BoundaryConfig,
}

/// Configuration for boundary detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryConfig {
    /// Minimum confidence threshold for boundary detection
    pub confidence_threshold: f64,
    /// Maximum distance between boundaries
    pub max_boundary_distance: usize,
    /// Minimum distance between boundaries
    pub min_boundary_distance: usize,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            max_boundary_distance: 2000,
            min_boundary_distance: 50,
        }
    }
}

/// Boundary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryInfo {
    pub position: usize,
    pub confidence: f32,
    pub boundary_type: BoundaryType,
    pub semantic_strength: f32,
}

/// Types of boundaries
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BoundaryType {
    Paragraph,
    Header,
    Sentence,
    Semantic,
    Pattern,
}

impl BoundaryDetector {
    /// Creates a new neural boundary detector
    pub fn new() -> Result<Self> {
        let config = BoundaryConfig::default();
        let neural_chunker = NeuralChunker::new()?;
        
        Ok(Self {
            neural_chunker,
            config,
        })
    }

    /// Creates boundary detector with custom configuration
    pub fn with_config(config: BoundaryConfig) -> Result<Self> {
        let neural_chunker = NeuralChunker::new()?;
        
        Ok(Self {
            neural_chunker,
            config,
        })
    }

    /// Detects boundaries in text using neural network analysis
    pub fn detect_boundaries(&mut self, text: &str) -> Result<Vec<BoundaryInfo>> {
        // Use neural chunker for boundary detection
        let mut boundaries = self.neural_chunker.detect_boundaries(text)?;
        
        // Apply configuration filters
        boundaries.retain(|b| b.confidence >= self.config.confidence_threshold as f32);
        
        Ok(boundaries)
    }

    /// Performs health check on neural networks
    pub fn health_check(&mut self) -> bool {
        self.neural_chunker.health_check()
    }
}

impl Default for BoundaryDetector {
    fn default() -> Self {
        Self::new().expect("Failed to create default boundary detector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_detector_creation() {
        let mut detector = BoundaryDetector::new().unwrap();
        assert!(detector.health_check());
    }

    #[test]
    fn test_boundary_detection() {
        let mut detector = BoundaryDetector::new().unwrap();
        let text = "First paragraph.\n\nSecond paragraph.\n\n# Header\n\nThird paragraph.";
        
        let boundaries = detector.detect_boundaries(text).unwrap();
        
        // Should have at least start and end boundaries
        assert!(boundaries.len() >= 2);
        assert_eq!(boundaries.first().unwrap().position, 0);
        assert_eq!(boundaries.last().unwrap().position, text.len());
    }

    #[test]
    fn test_paragraph_boundaries() {
        let mut detector = BoundaryDetector::new().unwrap();
        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
        
        let boundaries = detector.detect_boundaries(text).unwrap();
        
        // Should detect paragraph breaks
        assert!(boundaries.len() >= 2);
    }

    #[test]
    fn test_header_boundaries() {
        let mut detector = BoundaryDetector::new().unwrap();
        let text = "Some text before.\n\n# Main Header\n\nText after header.";
        
        let boundaries = detector.detect_boundaries(text).unwrap();
        
        // Should detect header boundary
        let header_boundary = boundaries.iter().find(|b| b.boundary_type == BoundaryType::Header || b.boundary_type == BoundaryType::Semantic);
        assert!(header_boundary.is_some());
    }
}