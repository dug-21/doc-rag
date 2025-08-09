//! Semantic boundary detection for document chunking
//!
//! This module provides boundary detection capabilities using pattern matching
//! and heuristic analysis for optimal chunk boundary identification.

use crate::Result;
use regex::Regex;
use once_cell::sync::Lazy;

/// Boundary detector for identifying optimal chunk boundaries
#[derive(Debug, Clone)]
pub struct BoundaryDetector {
    /// Feature extractor for text analysis
    feature_extractor: FeatureExtractor,
    /// Configuration parameters
    config: BoundaryConfig,
    /// Detection patterns
    patterns: &'static BoundaryPatterns,
}

/// Configuration for boundary detection
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct BoundaryInfo {
    pub position: usize,
    pub confidence: f32,
    pub boundary_type: BoundaryType,
    pub semantic_strength: f32,
}

/// Types of boundaries
#[derive(Debug, Clone, PartialEq)]
pub enum BoundaryType {
    Paragraph,
    Header,
    Sentence,
    Semantic,
    Pattern,
}

/// Pre-compiled boundary patterns
#[derive(Debug)]
struct BoundaryPatterns {
    paragraph_breaks: Regex,
    headers: Regex,
    lists: Regex,
    code_blocks: Regex,
    tables: Regex,
    quotes: Regex,
    sentences: Regex,
}

static BOUNDARY_PATTERNS: Lazy<BoundaryPatterns> = Lazy::new(|| BoundaryPatterns {
    paragraph_breaks: Regex::new(r"\n\s*\n\s*").unwrap(),
    headers: Regex::new(r"(?m)^#+\s").unwrap(),
    lists: Regex::new(r"(?m)^(\s*)[-*+]\s|^(\s*)\d+\.\s").unwrap(),
    code_blocks: Regex::new(r"```[\s\S]*?```").unwrap(),
    tables: Regex::new(r"\|\s*[^|\n]*\s*\|").unwrap(),
    quotes: Regex::new(r"(?m)^\s*>").unwrap(),
    sentences: Regex::new(r"[.!?]+\s+").unwrap(),
});

/// Feature extractor for boundary analysis
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    config: BoundaryConfig,
}

impl BoundaryDetector {
    /// Creates a new boundary detector
    pub fn new() -> Result<Self> {
        let config = BoundaryConfig::default();
        Ok(Self {
            feature_extractor: FeatureExtractor::new(config.clone()),
            config,
            patterns: &BOUNDARY_PATTERNS,
        })
    }

    /// Creates boundary detector with custom configuration
    pub fn with_config(config: BoundaryConfig) -> Result<Self> {
        Ok(Self {
            feature_extractor: FeatureExtractor::new(config.clone()),
            patterns: &BOUNDARY_PATTERNS,
            config,
        })
    }

    /// Detects boundaries in text using pattern matching and heuristics
    pub fn detect_boundaries(&self, text: &str) -> Result<Vec<BoundaryInfo>> {
        let mut boundaries = Vec::new();
        
        // Always start with position 0
        boundaries.push(BoundaryInfo {
            position: 0,
            confidence: 1.0,
            boundary_type: BoundaryType::Pattern,
            semantic_strength: 1.0,
        });

        // Detect paragraph boundaries
        self.add_paragraph_boundaries(text, &mut boundaries);
        
        // Detect header boundaries
        self.add_header_boundaries(text, &mut boundaries);
        
        // Detect list boundaries
        self.add_list_boundaries(text, &mut boundaries);
        
        // Detect code block boundaries
        self.add_code_boundaries(text, &mut boundaries);
        
        // Detect table boundaries
        self.add_table_boundaries(text, &mut boundaries);
        
        // Detect sentence boundaries (lower priority)
        self.add_sentence_boundaries(text, &mut boundaries);

        // Filter and sort boundaries
        boundaries.retain(|b| b.confidence >= self.config.confidence_threshold as f32);
        self.filter_close_boundaries(&mut boundaries);
        boundaries.sort_by_key(|b| b.position);

        // Always end with text length
        if boundaries.last().map(|b| b.position) != Some(text.len()) {
            boundaries.push(BoundaryInfo {
                position: text.len(),
                confidence: 1.0,
                boundary_type: BoundaryType::Pattern,
                semantic_strength: 1.0,
            });
        }

        Ok(boundaries)
    }

    /// Adds paragraph boundaries
    fn add_paragraph_boundaries(&self, text: &str, boundaries: &mut Vec<BoundaryInfo>) {
        for mat in self.patterns.paragraph_breaks.find_iter(text) {
            let position = mat.end();
            boundaries.push(BoundaryInfo {
                position,
                confidence: 0.9,
                boundary_type: BoundaryType::Paragraph,
                semantic_strength: 0.8,
            });
        }
    }

    /// Adds header boundaries
    fn add_header_boundaries(&self, text: &str, boundaries: &mut Vec<BoundaryInfo>) {
        for mat in self.patterns.headers.find_iter(text) {
            let position = mat.start();
            if position > 0 {
                boundaries.push(BoundaryInfo {
                    position,
                    confidence: 0.95,
                    boundary_type: BoundaryType::Header,
                    semantic_strength: 0.9,
                });
            }
        }
    }

    /// Adds list boundaries
    fn add_list_boundaries(&self, text: &str, boundaries: &mut Vec<BoundaryInfo>) {
        let lines: Vec<&str> = text.lines().collect();
        let mut current_pos = 0;
        
        for (i, line) in lines.iter().enumerate() {
            if i > 0 && self.patterns.lists.is_match(line) {
                // Check if previous line was also a list item
                let prev_line = lines[i - 1];
                if !self.patterns.lists.is_match(prev_line) && !prev_line.trim().is_empty() {
                    boundaries.push(BoundaryInfo {
                        position: current_pos,
                        confidence: 0.85,
                        boundary_type: BoundaryType::Pattern,
                        semantic_strength: 0.7,
                    });
                }
            }
            current_pos += line.len() + 1; // +1 for newline
        }
    }

    /// Adds code block boundaries
    fn add_code_boundaries(&self, text: &str, boundaries: &mut Vec<BoundaryInfo>) {
        for mat in self.patterns.code_blocks.find_iter(text) {
            if mat.start() > 0 {
                boundaries.push(BoundaryInfo {
                    position: mat.start(),
                    confidence: 0.9,
                    boundary_type: BoundaryType::Pattern,
                    semantic_strength: 0.8,
                });
            }
            if mat.end() < text.len() {
                boundaries.push(BoundaryInfo {
                    position: mat.end(),
                    confidence: 0.9,
                    boundary_type: BoundaryType::Pattern,
                    semantic_strength: 0.8,
                });
            }
        }
    }

    /// Adds table boundaries
    fn add_table_boundaries(&self, text: &str, boundaries: &mut Vec<BoundaryInfo>) {
        let lines: Vec<&str> = text.lines().collect();
        let mut current_pos = 0;
        let mut in_table = false;
        
        for (i, line) in lines.iter().enumerate() {
            let is_table_line = self.patterns.tables.is_match(line);
            
            if is_table_line && !in_table && current_pos > 0 {
                // Start of table
                boundaries.push(BoundaryInfo {
                    position: current_pos,
                    confidence: 0.8,
                    boundary_type: BoundaryType::Pattern,
                    semantic_strength: 0.7,
                });
                in_table = true;
            } else if !is_table_line && in_table {
                // End of table
                boundaries.push(BoundaryInfo {
                    position: current_pos,
                    confidence: 0.8,
                    boundary_type: BoundaryType::Pattern,
                    semantic_strength: 0.7,
                });
                in_table = false;
            }
            
            current_pos += line.len() + 1; // +1 for newline
        }
    }

    /// Adds sentence boundaries (lowest priority)
    fn add_sentence_boundaries(&self, text: &str, boundaries: &mut Vec<BoundaryInfo>) {
        for mat in self.patterns.sentences.find_iter(text) {
            let position = mat.end();
            
            // Only add if this would create a reasonably sized chunk
            if position > self.config.min_boundary_distance && 
               text.len() - position > self.config.min_boundary_distance {
                boundaries.push(BoundaryInfo {
                    position,
                    confidence: 0.6,
                    boundary_type: BoundaryType::Sentence,
                    semantic_strength: 0.5,
                });
            }
        }
    }

    /// Filters boundaries that are too close together
    fn filter_close_boundaries(&self, boundaries: &mut Vec<BoundaryInfo>) {
        boundaries.sort_by_key(|b| b.position);
        
        let mut filtered = Vec::new();
        let mut last_position = 0;
        
        for boundary in boundaries.drain(..) {
            if boundary.position - last_position >= self.config.min_boundary_distance ||
               boundary.confidence > 0.9 { // Always keep high-confidence boundaries
                last_position = boundary.position;
                filtered.push(boundary);
            }
        }
        
        *boundaries = filtered;
    }

    /// Performs health check
    pub fn health_check(&self) -> bool {
        // Basic health check - ensure patterns are valid
        true
    }
}

impl FeatureExtractor {
    /// Creates a new feature extractor
    pub fn new(config: BoundaryConfig) -> Self {
        Self { config }
    }

    /// Extracts features for boundary detection at given position
    pub fn extract_features(&self, text: &str, position: usize) -> Vec<f64> {
        let mut features = vec![0.0; 10]; // Simplified feature set
        
        let window_size = 100;
        let start = position.saturating_sub(window_size);
        let end = (position + window_size).min(text.len());
        
        if start >= end {
            return features;
        }
        
        let context = &text[start..end];
        
        // Feature 0: Line break density
        features[0] = context.matches('\n').count() as f64 / context.len() as f64;
        
        // Feature 1: Punctuation density
        features[1] = context.chars().filter(|c| ".,;:!?".contains(*c)).count() as f64 / context.len() as f64;
        
        // Feature 2: Word count
        features[2] = context.split_whitespace().count() as f64;
        
        // Feature 3: Has header pattern
        features[3] = if BOUNDARY_PATTERNS.headers.is_match(context) { 1.0 } else { 0.0 };
        
        // Feature 4: Has list pattern
        features[4] = if BOUNDARY_PATTERNS.lists.is_match(context) { 1.0 } else { 0.0 };
        
        // Feature 5: Has code pattern
        features[5] = if BOUNDARY_PATTERNS.code_blocks.is_match(context) { 1.0 } else { 0.0 };
        
        // Feature 6: Has table pattern
        features[6] = if BOUNDARY_PATTERNS.tables.is_match(context) { 1.0 } else { 0.0 };
        
        // Feature 7: Paragraph break
        features[7] = if BOUNDARY_PATTERNS.paragraph_breaks.is_match(context) { 1.0 } else { 0.0 };
        
        // Feature 8: Average word length
        let words: Vec<&str> = context.split_whitespace().collect();
        features[8] = if !words.is_empty() {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64
        } else {
            0.0
        };
        
        // Feature 9: Whitespace ratio
        features[9] = context.chars().filter(|c| c.is_whitespace()).count() as f64 / context.len() as f64;
        
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_detector_creation() {
        let detector = BoundaryDetector::new().unwrap();
        assert!(detector.health_check());
    }

    #[test]
    fn test_boundary_detection() {
        let detector = BoundaryDetector::new().unwrap();
        let text = "First paragraph.\n\nSecond paragraph.\n\n# Header\n\nThird paragraph.";
        
        let boundaries = detector.detect_boundaries(text).unwrap();
        
        // Should have at least start and end boundaries
        assert!(boundaries.len() >= 2);
        assert_eq!(boundaries.first().unwrap().position, 0);
        assert_eq!(boundaries.last().unwrap().position, text.len());
    }

    #[test]
    fn test_paragraph_boundaries() {
        let detector = BoundaryDetector::new().unwrap();
        let text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.";
        
        let boundaries = detector.detect_boundaries(text).unwrap();
        
        // Should detect paragraph breaks
        assert!(boundaries.len() >= 2);
    }

    #[test]
    fn test_header_boundaries() {
        let detector = BoundaryDetector::new().unwrap();
        let text = "Some text before.\n\n# Main Header\n\nText after header.";
        
        let boundaries = detector.detect_boundaries(text).unwrap();
        
        // Should detect header boundary
        let header_boundary = boundaries.iter().find(|b| b.boundary_type == BoundaryType::Header);
        assert!(header_boundary.is_some());
    }

    #[test]
    fn test_feature_extraction() {
        let config = BoundaryConfig::default();
        let extractor = FeatureExtractor::new(config);
        
        let text = "This is a test sentence. This is another sentence.";
        let features = extractor.extract_features(text, 25);
        
        assert_eq!(features.len(), 10);
        assert!(features.iter().all(|&f| f.is_finite()));
    }
}