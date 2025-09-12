//! Feature extraction for neural classification systems
//!
//! Provides optimized feature extraction for:
//! - Document type classification (PCI-DSS, ISO-27001, SOC2, NIST)
//! - Section type classification (Requirements, Definitions, Procedures)
//! - Query routing classification (symbolic vs graph vs vector)
//!
//! Design optimized for <10ms inference performance per CONSTRAINT-003

use crate::{Result, ChunkerError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use regex::Regex;
use chrono::{DateTime, Utc};

/// Feature extractor for neural classification systems
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Compliance-specific keyword patterns
    keyword_patterns: KeywordPatterns,
    /// Document structure patterns
    structure_patterns: StructurePatterns,
    /// Query intent patterns
    query_patterns: QueryPatterns,
    /// Performance metrics
    metrics: FeatureMetrics,
}

/// Pre-compiled patterns for document classification
#[derive(Debug, Clone)]
pub struct KeywordPatterns {
    /// PCI-DSS specific patterns
    pub pci_patterns: Vec<Regex>,
    /// ISO-27001 specific patterns  
    pub iso_patterns: Vec<Regex>,
    /// SOC2 specific patterns
    pub soc2_patterns: Vec<Regex>,
    /// NIST specific patterns
    pub nist_patterns: Vec<Regex>,
    /// General compliance patterns
    pub compliance_patterns: Vec<Regex>,
}

/// Document structure detection patterns
#[derive(Debug, Clone)]
pub struct StructurePatterns {
    /// Header patterns
    pub header_patterns: Vec<Regex>,
    /// Section numbering patterns
    pub section_patterns: Vec<Regex>,
    /// Requirement indicator patterns
    pub requirement_patterns: Vec<Regex>,
    /// Definition patterns
    pub definition_patterns: Vec<Regex>,
    /// Procedure patterns
    pub procedure_patterns: Vec<Regex>,
}

/// Query intent classification patterns
#[derive(Debug, Clone)]
pub struct QueryPatterns {
    /// Symbolic reasoning triggers
    pub symbolic_patterns: Vec<Regex>,
    /// Graph traversal triggers
    pub graph_patterns: Vec<Regex>,
    /// Vector search triggers
    pub vector_patterns: Vec<Regex>,
    /// Question type patterns
    pub question_patterns: Vec<Regex>,
}

/// Feature extraction performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMetrics {
    pub total_extractions: u64,
    pub average_extraction_time_ms: f64,
    pub cache_hit_rate: f64,
    pub feature_cache_size: usize,
    pub last_updated: DateTime<Utc>,
}

/// Document-level features for classification
#[derive(Debug, Clone)]
pub struct DocumentFeatures {
    /// Text-based features (256 dimensions)
    pub text_features: Vec<f32>,
    /// Structure-based features (128 dimensions) 
    pub structure_features: Vec<f32>,
    /// Metadata features (64 dimensions)
    pub metadata_features: Vec<f32>,
    /// Domain-specific features (64 dimensions)
    pub domain_features: Vec<f32>,
    /// Total feature vector (512 dimensions)
    pub combined_features: Vec<f32>,
    /// Feature extraction confidence
    pub confidence: f64,
    /// Extraction time in milliseconds
    pub extraction_time_ms: f64,
}

/// Section-level features for classification
#[derive(Debug, Clone)]
pub struct SectionFeatures {
    /// Text content features (128 dimensions)
    pub content_features: Vec<f32>,
    /// Structure features (64 dimensions)
    pub structure_features: Vec<f32>,
    /// Context features (32 dimensions)
    pub context_features: Vec<f32>,
    /// Position features (32 dimensions)
    pub position_features: Vec<f32>,
    /// Combined feature vector (256 dimensions)
    pub combined_features: Vec<f32>,
    /// Section type hints
    pub type_hints: Vec<String>,
    /// Feature confidence
    pub confidence: f64,
}

/// Query-level features for routing
#[derive(Debug, Clone)]
pub struct QueryFeatures {
    /// Intent features (64 dimensions)
    pub intent_features: Vec<f32>,
    /// Complexity features (32 dimensions)
    pub complexity_features: Vec<f32>,
    /// Domain features (32 dimensions)
    pub domain_features: Vec<f32>,
    /// Combined feature vector (128 dimensions)
    pub combined_features: Vec<f32>,
    /// Query type indicators
    pub query_indicators: Vec<String>,
    /// Routing confidence
    pub confidence: f64,
}

impl FeatureExtractor {
    /// Creates a new feature extractor with optimized patterns
    pub fn new() -> Result<Self> {
        let keyword_patterns = KeywordPatterns::new()?;
        let structure_patterns = StructurePatterns::new()?;
        let query_patterns = QueryPatterns::new()?;
        
        Ok(Self {
            keyword_patterns,
            structure_patterns,
            query_patterns,
            metrics: FeatureMetrics::new(),
        })
    }

    /// Extracts document-level features for classification
    pub fn extract_document_features(&self, text: &str, metadata: Option<&HashMap<String, String>>) -> Result<DocumentFeatures> {
        let start_time = std::time::Instant::now();
        
        // Extract text-based features (256 dimensions)
        let text_features = self.extract_text_features(text)?;
        
        // Extract structure-based features (128 dimensions)
        let structure_features = self.extract_document_structure_features(text)?;
        
        // Extract metadata features (64 dimensions)
        let metadata_features = self.extract_metadata_features(metadata)?;
        
        // Extract domain-specific features (64 dimensions)
        let domain_features = self.extract_domain_features(text)?;
        
        // Combine all features (total: 512 dimensions)
        let mut combined_features = Vec::with_capacity(512);
        combined_features.extend(text_features.iter());
        combined_features.extend(structure_features.iter());
        combined_features.extend(metadata_features.iter());
        combined_features.extend(domain_features.iter());
        
        // Normalize features to [-1, 1] range for neural network
        self.normalize_features(&mut combined_features);
        
        let extraction_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        // Calculate confidence based on feature quality
        let confidence = self.calculate_feature_confidence(&combined_features);
        
        Ok(DocumentFeatures {
            text_features,
            structure_features,
            metadata_features,
            domain_features,
            combined_features,
            confidence,
            extraction_time_ms: extraction_time,
        })
    }

    /// Extracts section-level features for classification
    pub fn extract_section_features(&self, text: &str, context: Option<&str>, position: usize) -> Result<SectionFeatures> {
        let start_time = std::time::Instant::now();
        
        // Extract content features (128 dimensions)
        let content_features = self.extract_section_content_features(text)?;
        
        // Extract structure features (64 dimensions) 
        let structure_features = self.extract_section_structure_features(text)?;
        
        // Extract context features (32 dimensions)
        let context_features = self.extract_context_features(text, context)?;
        
        // Extract position features (32 dimensions)
        let position_features = self.extract_position_features(text, position)?;
        
        // Combine features (total: 256 dimensions)
        let mut combined_features = Vec::with_capacity(256);
        combined_features.extend(content_features.iter());
        combined_features.extend(structure_features.iter());
        combined_features.extend(context_features.iter());
        combined_features.extend(position_features.iter());
        
        // Normalize features
        self.normalize_features(&mut combined_features);
        
        // Extract type hints from content
        let type_hints = self.extract_section_type_hints(text);
        
        let confidence = self.calculate_feature_confidence(&combined_features);
        
        Ok(SectionFeatures {
            content_features,
            structure_features,
            context_features,
            position_features,
            combined_features,
            type_hints,
            confidence,
        })
    }

    /// Extracts query features for routing classification
    pub fn extract_query_features(&self, query: &str) -> Result<QueryFeatures> {
        let start_time = std::time::Instant::now();
        
        // Extract intent features (64 dimensions)
        let intent_features = self.extract_query_intent_features(query)?;
        
        // Extract complexity features (32 dimensions)
        let complexity_features = self.extract_query_complexity_features(query)?;
        
        // Extract domain features (32 dimensions)
        let domain_features = self.extract_query_domain_features(query)?;
        
        // Combine features (total: 128 dimensions)
        let mut combined_features = Vec::with_capacity(128);
        combined_features.extend(intent_features.iter());
        combined_features.extend(complexity_features.iter());
        combined_features.extend(domain_features.iter());
        
        // Normalize features
        self.normalize_features(&mut combined_features);
        
        // Extract query indicators for routing
        let query_indicators = self.extract_query_indicators(query);
        
        let confidence = self.calculate_feature_confidence(&combined_features);
        
        Ok(QueryFeatures {
            intent_features,
            complexity_features,
            domain_features,
            combined_features,
            query_indicators,
            confidence,
        })
    }

    /// Extract text-based features (256 dimensions)
    fn extract_text_features(&self, text: &str) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(256);
        
        // Basic text statistics (20 features)
        features.push(text.len() as f32 / 10000.0); // Normalized length
        features.push(text.lines().count() as f32 / 1000.0); // Line count
        features.push(text.split_whitespace().count() as f32 / 5000.0); // Word count
        features.push(text.chars().filter(|c| c.is_uppercase()).count() as f32 / text.len() as f32); // Uppercase ratio
        features.push(text.chars().filter(|c| c.is_numeric()).count() as f32 / text.len() as f32); // Numeric ratio
        features.push(text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32 / text.len() as f32); // Punctuation ratio
        
        // Sentence complexity features (10 features)
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?').collect();
        if !sentences.is_empty() {
            let avg_sentence_length = sentences.iter().map(|s| s.len()).sum::<usize>() as f32 / sentences.len() as f32;
            features.push((avg_sentence_length / 100.0).min(1.0)); // Average sentence length
        } else {
            features.push(0.0);
        }
        
        // Add more basic features
        for _ in features.len()..30 {
            features.push(0.0);
        }
        
        // Document type specific keywords (80 features - 20 per type)
        features.extend(self.extract_keyword_features(text, &self.keyword_patterns.pci_patterns, 20));
        features.extend(self.extract_keyword_features(text, &self.keyword_patterns.iso_patterns, 20));
        features.extend(self.extract_keyword_features(text, &self.keyword_patterns.soc2_patterns, 20));
        features.extend(self.extract_keyword_features(text, &self.keyword_patterns.nist_patterns, 20));
        
        // General compliance patterns (50 features)
        features.extend(self.extract_keyword_features(text, &self.keyword_patterns.compliance_patterns, 50));
        
        // N-gram features (96 features - simplified)
        let words: Vec<&str> = text.split_whitespace().take(1000).collect(); // Limit for performance
        features.extend(self.extract_ngram_features(&words, 96));
        
        // Ensure exactly 256 features
        features.resize(256, 0.0);
        Ok(features)
    }

    /// Extract document structure features (128 dimensions)
    fn extract_document_structure_features(&self, text: &str) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(128);
        
        // Header structure (20 features)
        for pattern in &self.structure_patterns.header_patterns {
            let count = pattern.find_iter(text).count() as f32;
            features.push((count / 100.0).min(1.0));
        }
        // Pad to 20
        while features.len() < 20 {
            features.push(0.0);
        }
        
        // Section numbering (20 features)
        for pattern in &self.structure_patterns.section_patterns {
            let count = pattern.find_iter(text).count() as f32;
            features.push((count / 50.0).min(1.0));
        }
        while features.len() < 40 {
            features.push(0.0);
        }
        
        // List and table structures (20 features)
        features.push(text.matches("- ").count() as f32 / 100.0); // Bullet lists
        features.push(text.matches("* ").count() as f32 / 100.0); // Bullet lists
        features.push(text.matches("|").count() as f32 / 100.0); // Tables
        features.push(text.matches("```").count() as f32 / 20.0); // Code blocks
        // Pad to 60
        while features.len() < 60 {
            features.push(0.0);
        }
        
        // Document organization features (68 features)
        let toc_indicators = ["table of contents", "contents", "index"];
        for indicator in &toc_indicators {
            features.push(if text.to_lowercase().contains(indicator) { 1.0 } else { 0.0 });
        }
        
        // Pad to exactly 128 features
        while features.len() < 128 {
            features.push(0.0);
        }
        features.resize(128, 0.0);
        Ok(features)
    }

    /// Extract metadata features (64 dimensions)
    fn extract_metadata_features(&self, metadata: Option<&HashMap<String, String>>) -> Result<Vec<f32>> {
        let mut features = vec![0.0; 64];
        
        if let Some(meta) = metadata {
            // Document type hints from metadata
            if let Some(title) = meta.get("title") {
                let title_lower = title.to_lowercase();
                features[0] = if title_lower.contains("pci") { 1.0 } else { 0.0 };
                features[1] = if title_lower.contains("iso") { 1.0 } else { 0.0 };
                features[2] = if title_lower.contains("soc") { 1.0 } else { 0.0 };
                features[3] = if title_lower.contains("nist") { 1.0 } else { 0.0 };
            }
            
            // Version information
            if let Some(version) = meta.get("version") {
                // Encode version as normalized number
                let version_num = version.chars()
                    .filter(|c| c.is_numeric())
                    .collect::<String>()
                    .parse::<f32>()
                    .unwrap_or(0.0);
                features[4] = (version_num / 10.0).min(1.0);
            }
            
            // Page count indication
            if let Some(pages) = meta.get("pages") {
                if let Ok(page_count) = pages.parse::<f32>() {
                    features[5] = (page_count / 1000.0).min(1.0);
                }
            }
        }
        
        Ok(features)
    }

    /// Extract domain-specific features (64 dimensions)
    fn extract_domain_features(&self, text: &str) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(64);
        
        // Security domain terms (16 features)
        let security_terms = [
            "encryption", "authentication", "authorization", "firewall",
            "vulnerability", "threat", "risk", "security",
            "compliance", "audit", "control", "policy",
            "procedure", "access", "data", "protection"
        ];
        
        for term in &security_terms {
            let count = text.to_lowercase().matches(term).count() as f32;
            features.push((count / text.len() as f32 * 10000.0).min(1.0));
        }
        
        // Compliance-specific terms (16 features)
        let compliance_terms = [
            "requirement", "must", "shall", "should",
            "mandatory", "optional", "recommended", "prohibited",
            "standard", "framework", "guideline", "specification",
            "assessment", "validation", "verification", "certification"
        ];
        
        for term in &compliance_terms {
            let count = text.to_lowercase().matches(term).count() as f32;
            features.push((count / text.len() as f32 * 10000.0).min(1.0));
        }
        
        // Technical terms (16 features)
        let technical_terms = [
            "system", "network", "database", "application",
            "software", "hardware", "server", "client",
            "protocol", "algorithm", "configuration", "implementation",
            "architecture", "infrastructure", "deployment", "monitoring"
        ];
        
        for term in &technical_terms {
            let count = text.to_lowercase().matches(term).count() as f32;
            features.push((count / text.len() as f32 * 10000.0).min(1.0));
        }
        
        // Process terms (16 features)
        let process_terms = [
            "process", "workflow", "procedure", "step",
            "phase", "stage", "activity", "task",
            "method", "approach", "technique", "practice",
            "operation", "execution", "performance", "maintenance"
        ];
        
        for term in &process_terms {
            let count = text.to_lowercase().matches(term).count() as f32;
            features.push((count / text.len() as f32 * 10000.0).min(1.0));
        }
        
        Ok(features)
    }

    /// Extract section content features (128 dimensions)
    fn extract_section_content_features(&self, text: &str) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(128);
        
        // Basic content statistics (20 features)
        features.push(text.len() as f32 / 5000.0); // Length
        features.push(text.split_whitespace().count() as f32 / 1000.0); // Word count
        features.push(text.lines().count() as f32 / 100.0); // Line count
        
        // Requirement indicators (20 features)
        for pattern in self.structure_patterns.requirement_patterns.iter().take(17) {
            let count = pattern.find_iter(text).count() as f32;
            features.push((count / 10.0).min(1.0));
        }
        while features.len() < 40 {
            features.push(0.0);
        }
        
        // Definition indicators (20 features)
        for pattern in self.structure_patterns.definition_patterns.iter().take(17) {
            let count = pattern.find_iter(text).count() as f32;
            features.push((count / 10.0).min(1.0));
        }
        while features.len() < 60 {
            features.push(0.0);
        }
        
        // Procedure indicators (20 features)
        for pattern in self.structure_patterns.procedure_patterns.iter().take(17) {
            let count = pattern.find_iter(text).count() as f32;
            features.push((count / 10.0).min(1.0));
        }
        while features.len() < 80 {
            features.push(0.0);
        }
        
        // Content type features (48 features)
        let content_indicators = [
            ("definition:", 1.0), ("defined as", 1.0), ("means", 1.0), ("refers to", 1.0),
            ("step ", 1.0), ("procedure", 1.0), ("process", 1.0), ("method", 1.0),
            ("example", 1.0), ("illustration", 1.0), ("note:", 1.0), ("warning:", 1.0),
            ("caution:", 1.0), ("important:", 1.0), ("see also", 1.0), ("reference", 1.0),
        ];
        
        for (indicator, weight) in &content_indicators {
            let count = text.to_lowercase().matches(indicator).count() as f32;
            features.push((count * weight / 5.0).min(1.0));
        }
        
        // Pad to exactly 128 features
        while features.len() < 128 {
            features.push(0.0);
        }
        features.resize(128, 0.0);
        Ok(features)
    }

    /// Extract section structure features (64 dimensions)
    fn extract_section_structure_features(&self, text: &str) -> Result<Vec<f32>> {
        let mut features = vec![0.0; 64];
        
        // Numbering patterns (16 features)
        let numbering_patterns = [
            r"\d+\.\d+", r"\d+\.\d+\.\d+", r"\(\d+\)", r"\d+\)",
            r"[a-z]\)", r"[A-Z]\)", r"[ivx]+\.", r"[IVX]+\."
        ];
        
        for (i, pattern) in numbering_patterns.iter().enumerate() {
            if let Ok(regex) = Regex::new(pattern) {
                let count = regex.find_iter(text).count() as f32;
                features[i] = (count / 10.0).min(1.0);
            }
        }
        
        // List structures (16 features)  
        features[16] = text.matches("- ").count() as f32 / 20.0;
        features[17] = text.matches("* ").count() as f32 / 20.0;
        features[18] = text.matches("+ ").count() as f32 / 20.0;
        
        // Code and technical content (16 features)
        features[32] = text.matches("```").count() as f32 / 5.0;
        features[33] = text.matches("`").count() as f32 / 50.0;
        features[34] = text.matches("    ").count() as f32 / 20.0; // Indented code
        
        // Table structures (16 features)
        features[48] = text.matches("|").count() as f32 / 20.0;
        features[49] = text.matches("---").count() as f32 / 5.0;
        features[50] = text.matches("===").count() as f32 / 5.0;
        
        Ok(features)
    }

    /// Extract context features (32 dimensions)
    fn extract_context_features(&self, text: &str, context: Option<&str>) -> Result<Vec<f32>> {
        let mut features = vec![0.0; 32];
        
        if let Some(ctx) = context {
            // Context similarity features (16 features)
            let common_words = self.count_common_words(text, ctx);
            features[0] = (common_words as f32 / 100.0).min(1.0);
            
            // Context type indicators (16 features)
            if ctx.to_lowercase().contains("requirement") {
                features[16] = 1.0;
            }
            if ctx.to_lowercase().contains("definition") {
                features[17] = 1.0;
            }
            if ctx.to_lowercase().contains("procedure") {
                features[18] = 1.0;
            }
        }
        
        Ok(features)
    }

    /// Extract position features (32 dimensions)
    fn extract_position_features(&self, text: &str, position: usize) -> Result<Vec<f32>> {
        let mut features = vec![0.0; 32];
        
        // Normalized position
        features[0] = (position as f32 / 1000.0).min(1.0);
        
        // Section positioning indicators
        if text.to_lowercase().starts_with("introduction") {
            features[1] = 1.0;
        }
        if text.to_lowercase().contains("conclusion") || text.to_lowercase().contains("summary") {
            features[2] = 1.0;
        }
        if text.to_lowercase().contains("appendix") {
            features[3] = 1.0;
        }
        
        Ok(features)
    }

    /// Extract query intent features (64 dimensions)
    fn extract_query_intent_features(&self, query: &str) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(64);
        
        // Question type features (16 features)
        let question_words = ["what", "where", "when", "why", "how", "who", "which", "whose"];
        for word in &question_words {
            features.push(if query.to_lowercase().contains(word) { 1.0 } else { 0.0 });
        }
        while features.len() < 16 {
            features.push(0.0);
        }
        
        // Symbolic reasoning indicators (16 features)
        for pattern in self.query_patterns.symbolic_patterns.iter().take(16) {
            let count = pattern.find_iter(query).count() as f32;
            features.push((count / 3.0).min(1.0));
        }
        while features.len() < 32 {
            features.push(0.0);
        }
        
        // Graph traversal indicators (16 features)
        for pattern in self.query_patterns.graph_patterns.iter().take(16) {
            let count = pattern.find_iter(query).count() as f32;
            features.push((count / 3.0).min(1.0));
        }
        while features.len() < 48 {
            features.push(0.0);
        }
        
        // Vector search indicators (16 features)
        for pattern in self.query_patterns.vector_patterns.iter().take(16) {
            let count = pattern.find_iter(query).count() as f32;
            features.push((count / 3.0).min(1.0));
        }
        while features.len() < 64 {
            features.push(0.0);
        }
        
        features.resize(64, 0.0);
        Ok(features)
    }

    /// Extract query complexity features (32 dimensions)
    fn extract_query_complexity_features(&self, query: &str) -> Result<Vec<f32>> {
        let mut features = vec![0.0; 32];
        
        // Basic complexity metrics
        features[0] = (query.len() as f32 / 1000.0).min(1.0);
        features[1] = (query.split_whitespace().count() as f32 / 50.0).min(1.0);
        features[2] = (query.matches("?").count() as f32 / 3.0).min(1.0);
        features[3] = (query.matches("and").count() as f32 / 5.0).min(1.0);
        features[4] = (query.matches("or").count() as f32 / 5.0).min(1.0);
        
        // Complex query patterns
        features[5] = if query.contains("related to") { 1.0 } else { 0.0 };
        features[6] = if query.contains("depends on") { 1.0 } else { 0.0 };
        features[7] = if query.contains("similar to") { 1.0 } else { 0.0 };
        
        Ok(features)
    }

    /// Extract query domain features (32 dimensions)  
    fn extract_query_domain_features(&self, query: &str) -> Result<Vec<f32>> {
        let mut features = vec![0.0; 32];
        
        // Domain-specific terms
        let query_lower = query.to_lowercase();
        features[0] = if query_lower.contains("security") { 1.0 } else { 0.0 };
        features[1] = if query_lower.contains("compliance") { 1.0 } else { 0.0 };
        features[2] = if query_lower.contains("audit") { 1.0 } else { 0.0 };
        features[3] = if query_lower.contains("control") { 1.0 } else { 0.0 };
        features[4] = if query_lower.contains("requirement") { 1.0 } else { 0.0 };
        features[5] = if query_lower.contains("policy") { 1.0 } else { 0.0 };
        features[6] = if query_lower.contains("procedure") { 1.0 } else { 0.0 };
        features[7] = if query_lower.contains("risk") { 1.0 } else { 0.0 };
        
        Ok(features)
    }

    /// Helper methods for feature extraction
    fn extract_keyword_features(&self, text: &str, patterns: &[Regex], num_features: usize) -> Vec<f32> {
        let mut features = Vec::with_capacity(num_features);
        let text_lower = text.to_lowercase();
        
        for pattern in patterns.iter().take(num_features) {
            let count = pattern.find_iter(&text_lower).count() as f32;
            let normalized = (count / text.len() as f32 * 10000.0).min(1.0);
            features.push(normalized);
        }
        
        // Pad if necessary
        while features.len() < num_features {
            features.push(0.0);
        }
        
        features
    }

    fn extract_ngram_features(&self, words: &[&str], num_features: usize) -> Vec<f32> {
        let mut features = vec![0.0; num_features];
        
        if words.len() >= 2 {
            // Simple hash-based n-gram features for performance
            for i in 0..(words.len() - 1) {
                let bigram = format!("{} {}", words[i], words[i + 1]);
                let hash = self.simple_hash(&bigram) % num_features;
                features[hash] += 1.0;
            }
            
            // Normalize by number of bigrams
            let total_bigrams = (words.len() - 1) as f32;
            for feature in &mut features {
                *feature /= total_bigrams;
            }
        }
        
        features
    }

    fn extract_section_type_hints(&self, text: &str) -> Vec<String> {
        let mut hints = Vec::new();
        let text_lower = text.to_lowercase();
        
        // Check for requirement indicators
        if text_lower.contains("must") || text_lower.contains("shall") || text_lower.contains("required") {
            hints.push("requirement".to_string());
        }
        
        // Check for definition indicators
        if text_lower.contains("definition") || text_lower.contains("defined as") || text_lower.contains("means") {
            hints.push("definition".to_string());
        }
        
        // Check for procedure indicators
        if text_lower.contains("step") || text_lower.contains("procedure") || text_lower.contains("process") {
            hints.push("procedure".to_string());
        }
        
        // Check for appendix indicators
        if text_lower.contains("appendix") || text_lower.contains("annex") {
            hints.push("appendix".to_string());
        }
        
        hints
    }

    fn extract_query_indicators(&self, query: &str) -> Vec<String> {
        let mut indicators = Vec::new();
        let query_lower = query.to_lowercase();
        
        // Symbolic reasoning indicators
        if query_lower.contains("comply") || query_lower.contains("requirement") {
            indicators.push("symbolic".to_string());
        }
        
        // Graph traversal indicators
        if query_lower.contains("related") || query_lower.contains("depend") {
            indicators.push("graph".to_string());
        }
        
        // Vector search indicators
        if query_lower.contains("similar") || query_lower.contains("like") {
            indicators.push("vector".to_string());
        }
        
        indicators
    }

    fn normalize_features(&self, features: &mut Vec<f32>) {
        // Simple min-max normalization to [-1, 1] range
        if !features.is_empty() {
            let min_val = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            
            if (max_val - min_val).abs() > f32::EPSILON {
                for feature in features {
                    *feature = 2.0 * (*feature - min_val) / (max_val - min_val) - 1.0;
                }
            }
        }
    }

    fn calculate_feature_confidence(&self, features: &[f32]) -> f64 {
        if features.is_empty() {
            return 0.0;
        }
        
        // Calculate confidence based on feature variance and non-zero count
        let non_zero_count = features.iter().filter(|&&f| f.abs() > 0.001).count();
        let non_zero_ratio = non_zero_count as f64 / features.len() as f64;
        
        // Higher confidence with more informative features
        (non_zero_ratio * 0.8 + 0.2).min(1.0)
    }

    fn count_common_words(&self, text1: &str, text2: &str) -> usize {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
        words1.intersection(&words2).count()
    }

    fn simple_hash(&self, s: &str) -> usize {
        s.bytes().fold(0, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize))
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &FeatureMetrics {
        &self.metrics
    }
}

impl KeywordPatterns {
    fn new() -> Result<Self> {
        Ok(Self {
            pci_patterns: Self::compile_pci_patterns()?,
            iso_patterns: Self::compile_iso_patterns()?,
            soc2_patterns: Self::compile_soc2_patterns()?,
            nist_patterns: Self::compile_nist_patterns()?,
            compliance_patterns: Self::compile_compliance_patterns()?,
        })
    }

    fn compile_pci_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"pci.?dss", r"payment.*card.*industry", r"cardholder.*data", r"card.*data.*environment",
            r"primary.*account.*number", r"pan", r"sensitive.*authentication", r"tokenization",
            r"acquiring.*bank", r"issuing.*bank", r"card.*brand", r"payment.*processor",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_iso_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"iso.*27001", r"information.*security.*management", r"isms", r"security.*controls",
            r"annex.*a", r"risk.*assessment", r"risk.*treatment", r"statement.*of.*applicability",
            r"management.*review", r"internal.*audit", r"continual.*improvement", r"security.*policy",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_soc2_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"soc.*2", r"service.*organization.*control", r"trust.*services.*criteria", r"security.*criterion",
            r"availability.*criterion", r"processing.*integrity", r"confidentiality.*criterion", r"privacy.*criterion",
            r"type.*i.*report", r"type.*ii.*report", r"control.*environment", r"service.*auditor",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_nist_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"nist", r"cybersecurity.*framework", r"csf", r"identify.*function", r"protect.*function",
            r"detect.*function", r"respond.*function", r"recover.*function", r"risk.*management.*framework",
            r"rmf", r"security.*control.*baseline", r"sp.*800", r"federal.*information",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_compliance_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"compliance", r"audit", r"assessment", r"control", r"requirement", r"mandatory",
            r"shall", r"must", r"should", r"may", r"policy", r"procedure", r"guideline",
            r"standard", r"framework", r"regulation", r"governance", r"oversight",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_patterns(patterns: &[&str]) -> Result<Vec<Regex>> {
        patterns.iter()
            .map(|&p| Regex::new(&format!("(?i){}", p))
                .map_err(|e| ChunkerError::NeuralError(format!("Failed to compile regex '{}': {}", p, e))))
            .collect()
    }
}

impl StructurePatterns {
    fn new() -> Result<Self> {
        Ok(Self {
            header_patterns: Self::compile_header_patterns()?,
            section_patterns: Self::compile_section_patterns()?,
            requirement_patterns: Self::compile_requirement_patterns()?,
            definition_patterns: Self::compile_definition_patterns()?,
            procedure_patterns: Self::compile_procedure_patterns()?,
        })
    }

    fn compile_header_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"^#{1,6}\s+", r"^[A-Z][A-Z\s]+$", r"^\d+\..*$", r"^[IVX]+\..*$",
            r"^Chapter\s+\d+", r"^Section\s+\d+", r"^Part\s+[A-Z]", r"^Appendix\s+[A-Z]",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_section_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"\d+\.\d+", r"\d+\.\d+\.\d+", r"\d+\.\d+\.\d+\.\d+", r"[A-Z]\.\d+",
            r"\(\d+\)", r"\([a-z]\)", r"\([A-Z]\)", r"[ivx]+\.",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_requirement_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"(?i)requirement\s+\d+", r"(?i)req\s+\d+", r"(?i)control\s+\d+", r"(?i)\bmust\b",
            r"(?i)\bshall\b", r"(?i)\brequired\b", r"(?i)\bmandatory\b", r"(?i)it\s+is\s+required",
            r"(?i)organizations?\s+must", r"(?i)systems?\s+shall", r"(?i)entities?\s+must",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_definition_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"(?i)definition", r"(?i)defined\s+as", r"(?i)\bmeans\b", r"(?i)refers\s+to",
            r"(?i)is\s+defined", r"(?i)terminology", r"(?i)glossary", r"(?i)for\s+purposes\s+of",
            r"(?i)in\s+this\s+standard", r"(?i)the\s+term\s+.+\s+means",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_procedure_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"(?i)procedure", r"(?i)process", r"(?i)step\s+\d+", r"(?i)follow\s+these\s+steps",
            r"(?i)methodology", r"(?i)approach", r"(?i)method", r"(?i)technique",
            r"(?i)implementation", r"(?i)execution", r"(?i)performance", r"(?i)operation",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_patterns(patterns: &[&str]) -> Result<Vec<Regex>> {
        patterns.iter()
            .map(|&p| Regex::new(p)
                .map_err(|e| ChunkerError::NeuralError(format!("Failed to compile regex '{}': {}", p, e))))
            .collect()
    }
}

impl QueryPatterns {
    fn new() -> Result<Self> {
        Ok(Self {
            symbolic_patterns: Self::compile_symbolic_patterns()?,
            graph_patterns: Self::compile_graph_patterns()?,
            vector_patterns: Self::compile_vector_patterns()?,
            question_patterns: Self::compile_question_patterns()?,
        })
    }

    fn compile_symbolic_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"(?i)compliant?\s+with", r"(?i)requirement\s+for", r"(?i)must\s+.*\s+do",
            r"(?i)is\s+required", r"(?i)does\s+.*\s+comply", r"(?i)what\s+.*\s+required",
            r"(?i)mandatory\s+for", r"(?i)shall\s+.*\s+be", r"(?i)policy\s+states",
            r"(?i)according\s+to", r"(?i)as\s+per\s+the", r"(?i)in\s+accordance",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_graph_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"(?i)related\s+to", r"(?i)depends?\s+on", r"(?i)connected\s+to",
            r"(?i)associated\s+with", r"(?i)linked\s+to", r"(?i)references?",
            r"(?i)cross.?references?", r"(?i)see\s+also", r"(?i)refers?\s+to",
            r"(?i)mentions?", r"(?i)cites?", r"(?i)points?\s+to",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_vector_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"(?i)similar\s+to", r"(?i)like", r"(?i)resembles?", r"(?i)comparable\s+to",
            r"(?i)analogous\s+to", r"(?i)equivalent\s+to", r"(?i)same\s+as",
            r"(?i)matches?", r"(?i)corresponds?\s+to", r"(?i)parallel\s+to",
            r"(?i)find\s+.*\s+about", r"(?i)search\s+for", r"(?i)look\s+for",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_question_patterns() -> Result<Vec<Regex>> {
        let patterns = [
            r"(?i)^what\s+", r"(?i)^where\s+", r"(?i)^when\s+", r"(?i)^why\s+",
            r"(?i)^how\s+", r"(?i)^who\s+", r"(?i)^which\s+", r"(?i)^whose\s+",
            r"(?i)^can\s+", r"(?i)^could\s+", r"(?i)^would\s+", r"(?i)^should\s+",
        ];
        
        Self::compile_patterns(&patterns)
    }

    fn compile_patterns(patterns: &[&str]) -> Result<Vec<Regex>> {
        patterns.iter()
            .map(|&p| Regex::new(p)
                .map_err(|e| ChunkerError::NeuralError(format!("Failed to compile regex '{}': {}", p, e))))
            .collect()
    }
}

impl FeatureMetrics {
    fn new() -> Self {
        Self {
            total_extractions: 0,
            average_extraction_time_ms: 0.0,
            cache_hit_rate: 0.0,
            feature_cache_size: 0,
            last_updated: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::new();
        assert!(extractor.is_ok());
    }

    #[test]
    fn test_document_feature_extraction() {
        let extractor = FeatureExtractor::new().unwrap();
        let text = "PCI DSS compliance requires cardholder data protection. Organizations must implement security controls.";
        
        let features = extractor.extract_document_features(text, None).unwrap();
        assert_eq!(features.combined_features.len(), 512);
        assert!(features.confidence > 0.0);
    }

    #[test]
    fn test_section_feature_extraction() {
        let extractor = FeatureExtractor::new().unwrap();
        let text = "Requirement 3.2.1: Cardholder data must be encrypted using strong cryptography.";
        
        let features = extractor.extract_section_features(text, None, 0).unwrap();
        assert_eq!(features.combined_features.len(), 256);
        assert!(!features.type_hints.is_empty());
    }

    #[test]
    fn test_query_feature_extraction() {
        let extractor = FeatureExtractor::new().unwrap();
        let query = "What security controls are required for cardholder data?";
        
        let features = extractor.extract_query_features(query).unwrap();
        assert_eq!(features.combined_features.len(), 128);
        assert!(!features.query_indicators.is_empty());
    }

    #[test]
    fn test_pci_pattern_detection() {
        let extractor = FeatureExtractor::new().unwrap();
        let text = "Payment Card Industry Data Security Standard (PCI DSS) requirements";
        
        let features = extractor.extract_document_features(text, None).unwrap();
        // PCI-related features should be non-zero
        let pci_features = &features.text_features[30..50]; // PCI features range
        assert!(pci_features.iter().any(|&f| f > 0.0));
    }

    #[test]
    fn test_requirement_pattern_detection() {
        let extractor = FeatureExtractor::new().unwrap();
        let text = "Organizations must implement multi-factor authentication for all users.";
        
        let features = extractor.extract_section_features(text, None, 0).unwrap();
        assert!(features.type_hints.contains(&"requirement".to_string()));
    }

    #[test]
    fn test_feature_normalization() {
        let extractor = FeatureExtractor::new().unwrap();
        let text = "Test document with various features.";
        
        let features = extractor.extract_document_features(text, None).unwrap();
        
        // All features should be in [-1, 1] range after normalization
        for &feature in &features.combined_features {
            assert!(feature >= -1.0 && feature <= 1.0, "Feature {} out of range", feature);
        }
    }

    #[test]
    fn test_performance_metrics() {
        let extractor = FeatureExtractor::new().unwrap();
        let text = "Test document for performance measurement.";
        
        let start = std::time::Instant::now();
        let _features = extractor.extract_document_features(text, None).unwrap();
        let duration = start.elapsed();
        
        // Should extract features in reasonable time (<10ms target)
        assert!(duration.as_millis() < 50); // Allow some margin for test environment
    }
}