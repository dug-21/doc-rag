//! FACT-powered intelligent cache implementation
//! 
//! This module provides the core FACT functionality including:
//! - Intelligent caching with fact extraction
//! - Citation tracking and source attribution
//! - Semantic similarity matching for cache hits

use crate::error::{Result, ResponseError};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// FACT-powered intelligent cache with fact extraction and citation tracking
#[derive(Debug)]
pub struct IntelligentFACTCache {
    /// Internal storage for cached responses
    storage: DashMap<String, CachedFACTResponse>,
    /// Fact extraction enabled
    fact_extraction_enabled: bool,
}

/// Enhanced cached response with fact extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedFACTResponse {
    /// The actual response data
    pub data: serde_json::Value,
    /// Extracted facts for intelligent matching
    pub extracted_facts: Vec<ExtractedFact>,
    /// Citation information
    pub citations: Vec<String>,
    /// Cache metadata
    pub metadata: CacheMetadata,
}

/// Extracted fact for intelligent cache matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    /// Fact content
    pub content: String,
    /// Confidence score
    pub confidence: f64,
    /// Source information
    pub source: String,
    /// Entity tags
    pub entities: Vec<String>,
}

/// Cache metadata for analytics and optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Last accessed
    pub last_accessed: SystemTime,
}

impl IntelligentFACTCache {
    /// Create new FACT cache instance
    pub fn new() -> Self {
        Self {
            storage: DashMap::new(),
            fact_extraction_enabled: true,
        }
    }
    
    /// Get cached response with intelligent fact matching
    pub fn get(&self, key: &str) -> Option<serde_json::Value> {
        if let Some(mut entry) = self.storage.get_mut(key) {
            // Update access statistics
            entry.metadata.access_count += 1;
            entry.metadata.last_accessed = SystemTime::now();
            
            Some(entry.data.clone())
        } else if self.fact_extraction_enabled {
            // Try semantic fact-based matching
            self.find_similar_by_facts(key)
        } else {
            None
        }
    }
    
    /// Store response with FACT processing
    pub fn store(&self, key: &str, value: serde_json::Value, response_text: Option<&str>) -> Result<()> {
        let extracted_facts = if let Some(text) = response_text {
            self.extract_facts(text)?
        } else {
            vec![]
        };
        
        let citations = if let Some(text) = response_text {
            self.extract_citations(text)?
        } else {
            vec![]
        };
        
        let cached_response = CachedFACTResponse {
            data: value,
            extracted_facts,
            citations,
            metadata: CacheMetadata {
                created_at: SystemTime::now(),
                access_count: 0,
                last_accessed: SystemTime::now(),
            },
        };
        
        self.storage.insert(key.to_string(), cached_response);
        Ok(())
    }
    
    /// Find similar responses based on extracted facts
    fn find_similar_by_facts(&self, query: &str) -> Option<serde_json::Value> {
        let Ok(query_facts) = self.extract_facts(query) else {
            return None;
        };
        
        let mut best_match = None;
        let mut best_similarity = 0.0;
        
        for entry in self.storage.iter() {
            let similarity = self.calculate_fact_similarity(&query_facts, &entry.value().extracted_facts);
            if similarity > best_similarity && similarity > 0.8 { // 80% similarity threshold
                best_similarity = similarity;
                best_match = Some(entry.value().data.clone());
            }
        }
        
        best_match
    }
    
    /// Extract facts from text using FACT methodology
    fn extract_facts(&self, text: &str) -> Result<Vec<ExtractedFact>> {
        let mut facts = Vec::new();
        
        // Extract sentences as potential facts
        let sentences: Vec<&str> = text.split('.').filter(|s| !s.trim().is_empty()).collect();
        
        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.len() > 10 { // Filter out very short sentences
                let entities = self.extract_entities(sentence);
                
                facts.push(ExtractedFact {
                    content: sentence.to_string(),
                    confidence: self.calculate_fact_confidence(sentence),
                    source: "response_text".to_string(),
                    entities,
                });
            }
        }
        
        Ok(facts)
    }
    
    /// Extract entities from text
    fn extract_entities(&self, text: &str) -> Vec<String> {
        let mut entities = Vec::new();
        
        // Simple pattern-based entity extraction
        // In a full implementation, this would use advanced NLP
        for word in text.split_whitespace() {
            if word.len() > 3 && word.chars().next().unwrap().is_uppercase() {
                entities.push(format!("ENTITY:{}", word));
            }
        }
        
        entities
    }
    
    /// Calculate confidence score for extracted facts
    fn calculate_fact_confidence(&self, sentence: &str) -> f64 {
        let mut confidence = 0.5; // Base confidence
        
        // Boost confidence for factual patterns
        if sentence.contains(" is ") || sentence.contains(" are ") {
            confidence += 0.2;
        }
        if sentence.len() > 50 {
            confidence += 0.1;
        }
        if sentence.chars().filter(|c| c.is_uppercase()).count() > 2 {
            confidence += 0.1;
        }
        
        confidence.min(1.0)
    }
    
    /// Extract citations from text
    fn extract_citations(&self, text: &str) -> Result<Vec<String>> {
        let mut citations = Vec::new();
        
        // Simple citation pattern matching
        // In production, this would be more sophisticated
        if text.contains("http") {
            for word in text.split_whitespace() {
                if word.starts_with("http") {
                    citations.push(word.to_string());
                }
            }
        }
        
        // Look for reference patterns
        for line in text.lines() {
            if line.contains("[") && line.contains("]") {
                citations.push(line.trim().to_string());
            }
        }
        
        // Deduplicate
        citations.sort();
        citations.dedup();
        
        Ok(citations)
    }
    
    /// Calculate similarity between fact sets
    fn calculate_fact_similarity(&self, facts1: &[ExtractedFact], facts2: &[ExtractedFact]) -> f64 {
        if facts1.is_empty() && facts2.is_empty() {
            return 1.0;
        }
        
        if facts1.is_empty() || facts2.is_empty() {
            return 0.0;
        }
        
        let mut matches = 0;
        let total = facts1.len().max(facts2.len());
        
        for fact1 in facts1 {
            for fact2 in facts2 {
                if self.facts_match(fact1, fact2) {
                    matches += 1;
                    break;
                }
            }
        }
        
        matches as f64 / total as f64
    }
    
    /// Check if two facts match semantically
    fn facts_match(&self, fact1: &ExtractedFact, fact2: &ExtractedFact) -> bool {
        // Simple text similarity
        let similarity = self.text_similarity(&fact1.content, &fact2.content);
        similarity > 0.7 || fact1.entities.iter().any(|e1| fact2.entities.contains(e1))
    }
    
    /// Calculate text similarity (simplified Jaccard similarity)
    fn text_similarity(&self, text1: &str, text2: &str) -> f64 {
        let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
        let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
        
        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();
        
        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }
    
    /// Clear all cached data
    pub fn clear(&self) {
        self.storage.clear();
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let total_entries = self.storage.len();
        let total_facts: usize = self.storage.iter()
            .map(|entry| entry.value().extracted_facts.len())
            .sum();
        let total_citations: usize = self.storage.iter()
            .map(|entry| entry.value().citations.len())
            .sum();
        
        CacheStats {
            total_entries,
            total_facts,
            total_citations,
        }
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_facts: usize,
    pub total_citations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fact_cache_creation() {
        let cache = IntelligentFACTCache::new();
        let stats = cache.get_stats();
        assert_eq!(stats.total_entries, 0);
    }
    
    #[test]
    fn test_fact_extraction() {
        let cache = IntelligentFACTCache::new();
        let text = "Rust is a systems programming language. It focuses on safety and performance.";
        let facts = cache.extract_facts(text).unwrap();
        assert!(!facts.is_empty());
        assert!(facts.len() >= 2); // Should extract at least 2 facts
    }
    
    #[test]
    fn test_citation_extraction() {
        let cache = IntelligentFACTCache::new();
        let text = "According to the study [Smith et al. 2021], Rust provides memory safety. See https://example.com for details.";
        let citations = cache.extract_citations(text).unwrap();
        assert!(!citations.is_empty());
    }
    
    #[test]
    fn test_cache_store_and_retrieve() {
        let cache = IntelligentFACTCache::new();
        let key = "test_key";
        let value = serde_json::json!({"content": "test response"});
        
        // Store with fact extraction
        cache.store(key, value.clone(), Some("Test response content.")).unwrap();
        
        // Retrieve
        let retrieved = cache.get(key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), value);
        
        // Check stats
        let stats = cache.get_stats();
        assert_eq!(stats.total_entries, 1);
    }
}