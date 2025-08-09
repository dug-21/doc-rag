//! Simplified boundary detection without neural networks for testing
//! 
//! This module provides basic boundary detection using pattern matching
//! and heuristic rules, focusing on reliability and testing coverage.

use crate::{ChunkerError, Result};
use std::collections::HashMap;
use regex::Regex;
use once_cell::sync::Lazy;

/// Simple boundary detector using pattern matching
#[derive(Clone)]
pub struct BoundaryDetector {
    config: BoundaryConfig,
}

/// Configuration for boundary detection
#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    pub window_size: usize,
    pub confidence_threshold: f64,
    pub max_boundary_distance: usize,
    pub min_boundary_distance: usize,
}

impl Default for BoundaryConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            confidence_threshold: 0.7,
            max_boundary_distance: 2000,
            min_boundary_distance: 50,
        }
    }
}

/// Information about detected boundaries
#[derive(Debug, Clone)]
pub struct BoundaryInfo {
    pub position: usize,
    pub confidence: f32,
    pub boundary_type: BoundaryType,
    pub semantic_strength: f32,
}

/// Types of detected boundaries
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryType {
    Paragraph,
    Header,
    Sentence,
    Semantic,
    Pattern,
}

static BOUNDARY_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        Regex::new(r"\n\s*\n\s*").unwrap(), // Paragraph breaks
        Regex::new(r"(?m)^#+\s").unwrap(),   // Markdown headers
        Regex::new(r"(?m)^\d+\.\s").unwrap(), // Numbered lists
        Regex::new(r"(?m)^[-*+]\s").unwrap(), // Bullet points
        Regex::new(r"```[\s\S]*?```").unwrap(), // Code blocks
        Regex::new(r"\|\s*[^|\n]*\s*\|").unwrap(), // Tables
        Regex::new(r"(?m)^\s*>").unwrap(),    // Quotes
        Regex::new(r"\[\d+\]|\(\d+\)").unwrap(), // References
    ]
});

impl BoundaryDetector {
    /// Create a new boundary detector
    pub fn new() -> Result<Self> {
        Ok(BoundaryDetector {
            config: BoundaryConfig::default(),
        })
    }

    /// Create boundary detector with custom configuration
    pub fn with_config(config: BoundaryConfig) -> Result<Self> {
        Ok(BoundaryDetector { config })
    }

    /// Detect semantic boundaries in text
    pub fn detect_boundaries(&self, text: &str) -> Result<Vec<BoundaryInfo>> {
        let mut boundaries = Vec::new();
        
        // Add pattern-based boundaries
        self.add_pattern_based_boundaries(text, &mut boundaries);
        
        // Add sentence-based boundaries
        self.add_sentence_boundaries(text, &mut boundaries);
        
        // Add paragraph boundaries
        self.add_paragraph_boundaries(text, &mut boundaries);
        
        // Sort and filter boundaries
        boundaries.sort_by(|a, b| a.position.cmp(&b.position));
        self.filter_boundaries(boundaries, text.len())
    }

    /// Add boundaries based on regex patterns
    fn add_pattern_based_boundaries(&self, text: &str, boundaries: &mut Vec<BoundaryInfo>) {
        for (i, pattern) in BOUNDARY_PATTERNS.iter().enumerate() {
            for match_info in pattern.find_iter(text) {
                let position = match_info.start();
                
                // Avoid duplicates
                if !boundaries.iter().any(|b| (b.position as i32 - position as i32).abs() < 10) {
                    boundaries.push(BoundaryInfo {
                        position,
                        confidence: 0.8 + (i as f32 * 0.02), // Slight variation per pattern
                        boundary_type: BoundaryType::Pattern,
                        semantic_strength: 0.7,
                    });
                }
            }
        }
    }

    /// Add sentence-based boundaries
    fn add_sentence_boundaries(&self, text: &str, boundaries: &mut Vec<BoundaryInfo>) {
        let sentence_endings = Regex::new(r"[.!?]\s+[A-Z]").unwrap();
        
        for match_info in sentence_endings.find_iter(text) {
            let position = match_info.start() + 1; // Position after punctuation
            
            if !boundaries.iter().any(|b| (b.position as i32 - position as i32).abs() < 5) {
                boundaries.push(BoundaryInfo {
                    position,
                    confidence: 0.6,
                    boundary_type: BoundaryType::Sentence,
                    semantic_strength: 0.5,
                });
            }
        }
    }

    /// Add paragraph boundaries
    fn add_paragraph_boundaries(&self, text: &str, boundaries: &mut Vec<BoundaryInfo>) {
        let paragraph_breaks = Regex::new(r"\n\s*\n").unwrap();
        
        for match_info in paragraph_breaks.find_iter(text) {
            let position = match_info.end();
            
            if !boundaries.iter().any(|b| (b.position as i32 - position as i32).abs() < 5) {
                boundaries.push(BoundaryInfo {
                    position,
                    confidence: 0.9,
                    boundary_type: BoundaryType::Paragraph,
                    semantic_strength: 0.8,
                });
            }
        }
    }

    /// Filter boundaries to maintain minimum distances
    fn filter_boundaries(&self, mut boundaries: Vec<BoundaryInfo>, text_length: usize) -> Result<Vec<BoundaryInfo>> {
        if boundaries.is_empty() {
            return Ok(vec![]);
        }

        // Sort by confidence descending, then by position
        boundaries.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.position.cmp(&b.position))
        });

        let mut filtered = Vec::new();
        let mut last_position = 0;

        for boundary in boundaries {
            // Check minimum distance from last boundary
            if boundary.position >= last_position + self.config.min_boundary_distance {
                // Check not too close to end
                if text_length >= boundary.position + self.config.min_boundary_distance {
                    filtered.push(boundary.clone());
                    last_position = boundary.position;
                }
            }
        }

        // Sort final results by position
        filtered.sort_by(|a, b| a.position.cmp(&b.position));
        Ok(filtered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_detector_creation() {
        let detector = BoundaryDetector::new();
        assert!(detector.is_ok());
    }

    #[test]
    fn test_paragraph_boundary_detection() {
        let detector = BoundaryDetector::new().unwrap();
        let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        
        let boundaries = detector.detect_boundaries(content).unwrap();
        assert!(!boundaries.is_empty());
        
        // Should detect paragraph breaks
        let has_paragraph_boundaries = boundaries.iter()
            .any(|b| b.boundary_type == BoundaryType::Paragraph);
        assert!(has_paragraph_boundaries);
    }

    #[test]
    fn test_header_boundary_detection() {
        let detector = BoundaryDetector::new().unwrap();
        let content = "# Main Header\n\nContent here.\n\n## Sub Header\n\nMore content.";
        
        let boundaries = detector.detect_boundaries(content).unwrap();
        assert!(!boundaries.is_empty());
        
        // Should detect header boundaries
        let has_pattern_boundaries = boundaries.iter()
            .any(|b| b.boundary_type == BoundaryType::Pattern);
        assert!(has_pattern_boundaries);
    }

    #[test]
    fn test_sentence_boundary_detection() {
        let detector = BoundaryDetector::new().unwrap();
        let content = "First sentence. Second sentence! Third sentence?";
        
        let boundaries = detector.detect_boundaries(content).unwrap();
        
        // Should detect some sentence boundaries
        let has_sentence_boundaries = boundaries.iter()
            .any(|b| b.boundary_type == BoundaryType::Sentence);
        assert!(has_sentence_boundaries);
    }

    #[test]
    fn test_boundary_filtering() {
        let detector = BoundaryDetector::new().unwrap();
        let content = "a".repeat(1000); // Long text without clear boundaries
        
        let boundaries = detector.detect_boundaries(content).unwrap();
        
        // Should not produce too many boundaries for uniform text
        assert!(boundaries.len() < 10);
    }

    #[test]
    fn test_confidence_scores() {
        let detector = BoundaryDetector::new().unwrap();
        let content = "Text with various boundaries.\n\nParagraph break here.\n\n# Header here\n\nMore text.";
        
        let boundaries = detector.detect_boundaries(content).unwrap();
        
        // All boundaries should have valid confidence scores
        for boundary in &boundaries {
            assert!(boundary.confidence >= 0.0 && boundary.confidence <= 1.0);
            assert!(boundary.semantic_strength >= 0.0 && boundary.semantic_strength <= 1.0);
        }
    }

    #[test]
    fn test_minimum_distance_enforcement() {
        let mut config = BoundaryConfig::default();
        config.min_boundary_distance = 100;
        let detector = BoundaryDetector::with_config(config).unwrap();
        
        let content = "Short. Sentence. After. Sentence. In. Quick. Succession.";
        let boundaries = detector.detect_boundaries(content).unwrap();
        
        // Should enforce minimum distance between boundaries
        for i in 1..boundaries.len() {
            let distance = boundaries[i].position - boundaries[i-1].position;
            assert!(distance >= 100);
        }
    }

    #[test]
    fn test_empty_text() {
        let detector = BoundaryDetector::new().unwrap();
        let boundaries = detector.detect_boundaries("").unwrap();
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_deterministic_results() {
        let detector = BoundaryDetector::new().unwrap();
        let content = "Test content for determinism.\n\nSecond paragraph.";
        
        let boundaries1 = detector.detect_boundaries(content).unwrap();
        let boundaries2 = detector.detect_boundaries(content).unwrap();
        
        assert_eq!(boundaries1.len(), boundaries2.len());
        for (b1, b2) in boundaries1.iter().zip(boundaries2.iter()) {
            assert_eq!(b1.position, b2.position);
            assert!((b1.confidence - b2.confidence).abs() < f32::EPSILON);
        }
    }
}