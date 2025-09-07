//! FACT System - Fast, Accurate Citation Tracking
//! 
//! This module provides intelligent caching and citation tracking capabilities
//! as required by Phase 2 architecture requirements for 99% accuracy.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error types for FACT system
#[derive(Error, Debug)]
pub enum FactError {
    #[error("Cache miss")]
    CacheMiss,
    
    #[error("Invalid citation")]
    InvalidCitation,
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
}

pub type Result<T> = std::result::Result<T, FactError>;

/// Citation information for source tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub source: String,
    pub page: Option<u32>,
    pub section: Option<String>,
    pub relevance_score: f32,
    pub timestamp: u64,
}

/// Cached response with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResponse {
    pub content: String,
    pub citations: Vec<Citation>,
    pub confidence: f32,
    pub cached_at: u64, // Unix timestamp in seconds
    pub ttl: u64, // TTL in seconds
}

/// Intelligent cache implementation
pub struct FactCache {
    cache: Arc<parking_lot::RwLock<HashMap<String, CachedResponse>>>,
    max_size: usize,
    hit_rate: Arc<parking_lot::RwLock<f32>>,
}

impl FactCache {
    /// Create a new FACT cache instance
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(parking_lot::RwLock::new(HashMap::new())),
            max_size,
            hit_rate: Arc::new(parking_lot::RwLock::new(0.0)),
        }
    }
    
    /// Get a cached response with <50ms SLA
    pub fn get(&self, key: &str) -> Result<CachedResponse> {
        let start = Instant::now();
        let cache = self.cache.read();
        
        if let Some(response) = cache.get(key) {
            // Check if TTL expired
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            if current_time < response.cached_at + response.ttl {
                // Update hit rate
                let mut hit_rate = self.hit_rate.write();
                *hit_rate = (*hit_rate * 0.9) + 0.1; // Exponential moving average
                
                // Ensure <50ms response time
                assert!(start.elapsed() < Duration::from_millis(50), "Cache retrieval exceeded 50ms SLA");
                
                return Ok(response.clone());
            }
        }
        
        // Update hit rate for miss
        let mut hit_rate = self.hit_rate.write();
        *hit_rate = *hit_rate * 0.9; // Decay on miss
        
        Err(FactError::CacheMiss)
    }
    
    /// Store a response in cache
    pub fn put(&self, key: String, response: CachedResponse) {
        let mut cache = self.cache.write();
        
        // Evict LRU if at capacity
        if cache.len() >= self.max_size {
            // Simple eviction - in production would use proper LRU
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }
        
        cache.insert(key, response);
    }
    
    /// Get current cache hit rate
    pub fn hit_rate(&self) -> f32 {
        *self.hit_rate.read()
    }
    
    /// Clear the cache
    pub fn clear(&self) {
        self.cache.write().clear();
    }
}

/// Citation tracker for source attribution
pub struct CitationTracker {
    citations: Arc<parking_lot::RwLock<Vec<Citation>>>,
}

impl CitationTracker {
    /// Create a new citation tracker
    pub fn new() -> Self {
        Self {
            citations: Arc::new(parking_lot::RwLock::new(Vec::new())),
        }
    }
    
    /// Add a citation
    pub fn add_citation(&self, citation: Citation) {
        self.citations.write().push(citation);
    }
    
    /// Get all citations
    pub fn get_citations(&self) -> Vec<Citation> {
        self.citations.read().clone()
    }
    
    /// Get citations above relevance threshold
    pub fn get_relevant_citations(&self, threshold: f32) -> Vec<Citation> {
        self.citations
            .read()
            .iter()
            .filter(|c| c.relevance_score >= threshold)
            .cloned()
            .collect()
    }
    
    /// Clear all citations
    pub fn clear(&self) {
        self.citations.write().clear();
    }
}

/// Fact extractor for document processing
pub struct FactExtractor;

impl FactExtractor {
    /// Extract facts from text
    pub fn extract_facts(&self, text: &str) -> Vec<String> {
        // Simplified fact extraction - in production would use NLP
        text.split('.')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }
    
    /// Verify fact accuracy
    pub fn verify_fact(&self, fact: &str, source: &str) -> f32 {
        // Simplified verification - in production would use semantic matching
        if source.contains(fact) {
            1.0
        } else {
            0.0
        }
    }
}

/// Main FACT system coordinator
pub struct FactSystem {
    pub cache: FactCache,
    pub tracker: CitationTracker,
    pub extractor: FactExtractor,
}

impl FactSystem {
    /// Create a new FACT system instance
    pub fn new(cache_size: usize) -> Self {
        Self {
            cache: FactCache::new(cache_size),
            tracker: CitationTracker::new(),
            extractor: FactExtractor,
        }
    }
    
    /// Process a query with caching and citation tracking
    pub fn process_query(&self, query: &str) -> Result<CachedResponse> {
        // Try cache first
        if let Ok(cached) = self.cache.get(query) {
            return Ok(cached);
        }
        
        // If not cached, return miss (actual processing would happen in integration)
        Err(FactError::CacheMiss)
    }
    
    /// Store a response with citations
    pub fn store_response(&self, query: String, content: String, citations: Vec<Citation>) {
        let response = CachedResponse {
            content,
            citations: citations.clone(),
            confidence: 0.95,
            cached_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ttl: 3600, // 1 hour TTL in seconds
        };
        
        // Store in cache
        self.cache.put(query, response);
        
        // Track citations
        for citation in citations {
            self.tracker.add_citation(citation);
        }
    }
}

impl Default for CitationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_operations() {
        let cache = FactCache::new(10);
        
        let response = CachedResponse {
            content: "Test response".to_string(),
            citations: vec![],
            confidence: 0.9,
            cached_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            ttl: 60, // 60 seconds TTL
        };
        
        cache.put("test_key".to_string(), response.clone());
        
        let retrieved = cache.get("test_key").unwrap();
        assert_eq!(retrieved.content, "Test response");
    }
    
    #[test]
    fn test_citation_tracking() {
        let tracker = CitationTracker::new();
        
        tracker.add_citation(Citation {
            source: "doc1.pdf".to_string(),
            page: Some(42),
            section: Some("Section 3.2".to_string()),
            relevance_score: 0.85,
            timestamp: 1234567890,
        });
        
        let citations = tracker.get_citations();
        assert_eq!(citations.len(), 1);
        assert_eq!(citations[0].source, "doc1.pdf");
    }
}