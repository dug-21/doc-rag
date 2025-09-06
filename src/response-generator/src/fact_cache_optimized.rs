//! Optimized FACT-powered cache implementation for sub-50ms performance
//!
//! This module provides a high-performance FACT (Fast, Accurate, Citation-Tracked) cache
//! implementation targeting sub-50ms cache hits with advanced optimization techniques.

use crate::error::{Result, ResponseError};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// High-performance FACT cache with sub-50ms target performance
#[derive(Debug)]
pub struct OptimizedFACTCache {
    /// L1 cache - hot data in memory with ultra-fast access
    l1_cache: Arc<DashMap<String, L1CacheEntry>>,
    /// L2 cache - warm data with fact extraction
    l2_cache: Arc<DashMap<String, L2CacheEntry>>,
    /// Fact index for intelligent similarity matching
    fact_index: Arc<DashMap<String, FactIndexEntry>>,
    /// Configuration parameters
    config: OptimizedCacheConfig,
    /// Performance metrics
    metrics: Arc<CachePerformanceMetrics>,
    /// Parallel access semaphore
    access_semaphore: Arc<Semaphore>,
    /// Background cleanup handle
    _cleanup_handle: tokio::task::JoinHandle<()>,
}

/// Optimized cache configuration for maximum performance
#[derive(Debug, Clone)]
pub struct OptimizedCacheConfig {
    /// L1 cache size (hot data)
    pub l1_max_entries: usize,
    /// L2 cache size (warm data) 
    pub l2_max_entries: usize,
    /// Fact index size
    pub fact_index_max_entries: usize,
    /// L1 TTL for ultra-hot data
    pub l1_ttl_ms: u64,
    /// L2 TTL for warm data
    pub l2_ttl_ms: u64,
    /// Fact similarity threshold (0.8 = 80% similarity for cache hit)
    pub similarity_threshold: f64,
    /// Maximum parallel access operations
    pub max_concurrent_access: usize,
    /// Prefetch prediction threshold
    pub prefetch_threshold: f64,
    /// Enable advanced optimization features
    pub enable_advanced_features: bool,
}

impl Default for OptimizedCacheConfig {
    fn default() -> Self {
        Self {
            l1_max_entries: 1000,        // Hot data
            l2_max_entries: 10000,       // Warm data
            fact_index_max_entries: 50000, // Fact index
            l1_ttl_ms: 300_000,          // 5 minutes for hot data
            l2_ttl_ms: 1_800_000,        // 30 minutes for warm data
            similarity_threshold: 0.85,   // 85% similarity threshold
            max_concurrent_access: 1000,  // High concurrency support
            prefetch_threshold: 0.7,      // Prefetch prediction threshold
            enable_advanced_features: true,
        }
    }
}

/// L1 cache entry for ultra-fast access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L1CacheEntry {
    /// Cache key
    pub key: String,
    /// Cached data
    pub data: serde_json::Value,
    /// Creation timestamp
    pub created_at: Instant,
    /// Last access timestamp  
    pub last_accessed: Instant,
    /// Access count for popularity tracking
    pub access_count: AtomicU64,
    /// Entry size in bytes
    pub size_bytes: usize,
}

/// L2 cache entry with fact extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2CacheEntry {
    /// Cache key
    pub key: String,
    /// Cached response data
    pub data: serde_json::Value,
    /// Extracted facts for intelligent matching
    pub extracted_facts: Vec<OptimizedExtractedFact>,
    /// Citations
    pub citations: Vec<String>,
    /// Metadata
    pub metadata: CacheEntryMetadata,
    /// Fact signature for fast comparison
    pub fact_signature: u64,
    /// Entry size in bytes
    pub size_bytes: usize,
}

/// Optimized extracted fact for high-speed processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedExtractedFact {
    /// Fact content (shortened for speed)
    pub content_hash: u64,
    /// Confidence score
    pub confidence: f32,
    /// Source identifier
    pub source_id: String,
    /// Entity hash for fast matching
    pub entity_hash: u64,
    /// Semantic vector (simplified)
    pub semantic_features: [f32; 8], // 8-dimensional feature vector for speed
}

/// Fact index entry for O(1) similarity lookups
#[derive(Debug, Clone)]
pub struct FactIndexEntry {
    /// Original cache key
    pub cache_key: String,
    /// Fact signature
    pub fact_signature: u64,
    /// Semantic centroid for similarity
    pub semantic_centroid: [f32; 8],
    /// Entry popularity score
    pub popularity_score: f32,
    /// Last accessed time
    pub last_accessed: Instant,
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntryMetadata {
    /// Creation timestamp
    pub created_at: SystemTime,
    /// Access count
    pub access_count: u64,
    /// Last accessed
    pub last_accessed: SystemTime,
    /// Response quality score
    pub quality_score: f32,
    /// Processing time when created
    pub original_processing_time_ms: u64,
}

/// Performance metrics for cache optimization
#[derive(Debug, Default)]
pub struct CachePerformanceMetrics {
    /// Total cache requests
    pub total_requests: AtomicU64,
    /// L1 cache hits
    pub l1_hits: AtomicU64,
    /// L2 cache hits
    pub l2_hits: AtomicU64,
    /// Fact-based similarity hits
    pub fact_similarity_hits: AtomicU64,
    /// Cache misses
    pub misses: AtomicU64,
    /// Total access time in microseconds
    pub total_access_time_us: AtomicU64,
    /// Current L1 size
    pub l1_size: AtomicUsize,
    /// Current L2 size
    pub l2_size: AtomicUsize,
    /// Evictions count
    pub evictions: AtomicU64,
    /// Prefetch hits
    pub prefetch_hits: AtomicU64,
}

impl OptimizedFACTCache {
    /// Create new optimized FACT cache
    pub fn new(config: OptimizedCacheConfig) -> Self {
        let l1_cache = Arc::new(DashMap::with_capacity(config.l1_max_entries));
        let l2_cache = Arc::new(DashMap::with_capacity(config.l2_max_entries));
        let fact_index = Arc::new(DashMap::with_capacity(config.fact_index_max_entries));
        let metrics = Arc::new(CachePerformanceMetrics::default());
        let access_semaphore = Arc::new(Semaphore::new(config.max_concurrent_access));
        
        // Start background cleanup and optimization
        let cleanup_handle = Self::start_background_tasks(
            l1_cache.clone(),
            l2_cache.clone(),
            fact_index.clone(),
            metrics.clone(),
            config.clone(),
        );
        
        Self {
            l1_cache,
            l2_cache,
            fact_index,
            config,
            metrics,
            access_semaphore,
            _cleanup_handle: cleanup_handle,
        }
    }

    /// Ultra-fast cache get with <50ms target
    pub async fn get(&self, key: &str) -> Option<serde_json::Value> {
        let start_time = Instant::now();
        let _permit = self.access_semaphore.acquire().await.ok()?;
        
        self.metrics.total_requests.fetch_add(1, Ordering::Relaxed);
        
        // L1 cache lookup (should be <5ms)
        if let Some(entry) = self.l1_cache.get(key) {
            entry.access_count.fetch_add(1, Ordering::Relaxed);
            let mut entry_mut = entry.clone();
            entry_mut.last_accessed = Instant::now();
            
            self.metrics.l1_hits.fetch_add(1, Ordering::Relaxed);
            self.record_access_time(start_time);
            return Some(entry_mut.data);
        }
        
        // L2 cache lookup (should be <20ms)
        if let Some(entry) = self.l2_cache.get(key) {
            let data = entry.data.clone();
            
            // Promote to L1 if frequently accessed
            if entry.metadata.access_count > 5 {
                self.promote_to_l1(key, &data).await;
            }
            
            self.metrics.l2_hits.fetch_add(1, Ordering::Relaxed);
            self.record_access_time(start_time);
            return Some(data);
        }
        
        // Fact-based similarity search (should be <25ms)
        if let Some(similar_entry) = self.find_similar_by_facts_fast(key).await {
            self.metrics.fact_similarity_hits.fetch_add(1, Ordering::Relaxed);
            self.record_access_time(start_time);
            return Some(similar_entry);
        }
        
        // Cache miss
        self.metrics.misses.fetch_add(1, Ordering::Relaxed);
        self.record_access_time(start_time);
        None
    }

    /// Ultra-fast cache put with background processing
    pub async fn put(&self, key: String, value: serde_json::Value, response_text: Option<&str>) -> Result<()> {
        let _permit = self.access_semaphore.acquire().await.map_err(|_| ResponseError::CacheError("Failed to acquire access permit".to_string()))?;
        
        let size_bytes = self.estimate_size(&value);
        
        // Always put in L1 for immediate access (should be <2ms)
        let l1_entry = L1CacheEntry {
            key: key.clone(),
            data: value.clone(),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: AtomicU64::new(1),
            size_bytes,
        };
        
        self.l1_cache.insert(key.clone(), l1_entry);
        self.metrics.l1_size.store(self.l1_cache.len(), Ordering::Relaxed);
        
        // Background processing for L2 and fact extraction
        if let Some(text) = response_text {
            let l2_cache = self.l2_cache.clone();
            let fact_index = self.fact_index.clone();
            let metrics = self.metrics.clone();
            let key_clone = key.clone();
            let value_clone = value.clone();
            let text_clone = text.to_string();
            
            tokio::spawn(async move {
                let _ = Self::process_for_l2_cache(
                    l2_cache,
                    fact_index,
                    metrics,
                    key_clone,
                    value_clone,
                    text_clone,
                    size_bytes,
                ).await;
            });
        }
        
        // Trigger cleanup if needed
        if self.l1_cache.len() > self.config.l1_max_entries {
            self.trigger_l1_cleanup().await;
        }
        
        Ok(())
    }

    /// Process entry for L2 cache with fact extraction (background task)
    async fn process_for_l2_cache(
        l2_cache: Arc<DashMap<String, L2CacheEntry>>,
        fact_index: Arc<DashMap<String, FactIndexEntry>>,
        metrics: Arc<CachePerformanceMetrics>,
        key: String,
        value: serde_json::Value,
        response_text: String,
        size_bytes: usize,
    ) -> Result<()> {
        // Extract facts optimized for speed
        let extracted_facts = Self::extract_facts_optimized(&response_text)?;
        let citations = Self::extract_citations_fast(&response_text)?;
        
        // Calculate fact signature for fast comparison
        let fact_signature = Self::calculate_fact_signature(&extracted_facts);
        
        // Create L2 entry
        let l2_entry = L2CacheEntry {
            key: key.clone(),
            data: value,
            extracted_facts: extracted_facts.clone(),
            citations,
            metadata: CacheEntryMetadata {
                created_at: SystemTime::now(),
                access_count: 1,
                last_accessed: SystemTime::now(),
                quality_score: 0.85, // Default quality score
                original_processing_time_ms: 1500, // Estimated
            },
            fact_signature,
            size_bytes,
        };
        
        l2_cache.insert(key.clone(), l2_entry);
        metrics.l2_size.store(l2_cache.len(), Ordering::Relaxed);
        
        // Update fact index
        if !extracted_facts.is_empty() {
            let semantic_centroid = Self::calculate_semantic_centroid(&extracted_facts);
            let fact_index_entry = FactIndexEntry {
                cache_key: key.clone(),
                fact_signature,
                semantic_centroid,
                popularity_score: 1.0,
                last_accessed: Instant::now(),
            };
            
            fact_index.insert(key, fact_index_entry);
        }
        
        Ok(())
    }

    /// Ultra-fast fact-based similarity search
    async fn find_similar_by_facts_fast(&self, query: &str) -> Option<serde_json::Value> {
        let start_time = Instant::now();
        
        // Quick fact extraction from query
        let query_facts = Self::extract_facts_optimized(query).ok()?;
        if query_facts.is_empty() {
            return None;
        }
        
        let query_signature = Self::calculate_fact_signature(&query_facts);
        let query_centroid = Self::calculate_semantic_centroid(&query_facts);
        
        // Fast similarity search through fact index
        let mut best_match = None;
        let mut best_similarity = 0.0;
        
        for entry in self.fact_index.iter() {
            let fact_entry = entry.value();
            
            // Quick signature-based pre-filtering
            let signature_similarity = Self::signature_similarity(query_signature, fact_entry.fact_signature);
            if signature_similarity < 0.5 {
                continue;
            }
            
            // Semantic similarity calculation
            let semantic_similarity = Self::cosine_similarity(&query_centroid, &fact_entry.semantic_centroid);
            let combined_similarity = (signature_similarity + semantic_similarity) / 2.0;
            
            if combined_similarity > best_similarity && combined_similarity > self.config.similarity_threshold {
                best_similarity = combined_similarity;
                best_match = Some(fact_entry.cache_key.clone());
            }
            
            // Early termination if excellent match found
            if best_similarity > 0.95 {
                break;
            }
        }
        
        // If we found a good match, return from L2 cache
        if let Some(cache_key) = best_match {
            if let Some(l2_entry) = self.l2_cache.get(&cache_key) {
                debug!("Found similar cache entry with {:.2}% similarity in {:?}", 
                       best_similarity * 100.0, start_time.elapsed());
                return Some(l2_entry.data.clone());
            }
        }
        
        None
    }

    /// Promote entry from L2 to L1 cache
    async fn promote_to_l1(&self, key: &str, data: &serde_json::Value) {
        let l1_entry = L1CacheEntry {
            key: key.to_string(),
            data: data.clone(),
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: AtomicU64::new(1),
            size_bytes: self.estimate_size(data),
        };
        
        self.l1_cache.insert(key.to_string(), l1_entry);
        self.metrics.l1_size.store(self.l1_cache.len(), Ordering::Relaxed);
    }

    /// Extract facts optimized for speed
    fn extract_facts_optimized(text: &str) -> Result<Vec<OptimizedExtractedFact>> {
        let mut facts = Vec::new();
        
        // Fast sentence splitting
        let sentences: Vec<&str> = text.split('.').filter(|s| s.trim().len() > 10).take(20).collect();
        
        for (i, sentence) in sentences.iter().enumerate() {
            let sentence = sentence.trim();
            if sentence.len() > 15 { // Only process substantial sentences
                
                let content_hash = Self::fast_hash(sentence);
                let entities = Self::extract_entities_fast(sentence);
                let entity_hash = Self::fast_hash(&entities.join(","));
                
                // Calculate semantic features quickly
                let semantic_features = Self::calculate_semantic_features_fast(sentence);
                
                facts.push(OptimizedExtractedFact {
                    content_hash,
                    confidence: Self::calculate_fact_confidence_fast(sentence),
                    source_id: format!("sent_{}", i),
                    entity_hash,
                    semantic_features,
                });
            }
        }
        
        Ok(facts)
    }

    /// Fast entity extraction
    fn extract_entities_fast(text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter(|word| word.len() > 3 && word.chars().next().unwrap().is_uppercase())
            .take(5) // Limit to 5 entities for speed
            .map(|s| s.to_string())
            .collect()
    }

    /// Fast confidence calculation
    fn calculate_fact_confidence_fast(sentence: &str) -> f32 {
        let mut confidence = 0.5;
        
        // Quick patterns
        if sentence.contains(" is ") || sentence.contains(" are ") { confidence += 0.2; }
        if sentence.len() > 50 { confidence += 0.1; }
        if sentence.chars().filter(|c| c.is_uppercase()).count() > 2 { confidence += 0.1; }
        
        confidence.min(1.0)
    }

    /// Calculate semantic features fast (8-dimensional vector)
    fn calculate_semantic_features_fast(text: &str) -> [f32; 8] {
        let mut features = [0.0f32; 8];
        
        let text_lower = text.to_lowercase();
        let word_count = text.split_whitespace().count() as f32;
        
        // Feature 0: Technical content
        features[0] = ["api", "system", "function", "data"]
            .iter()
            .map(|&term| text_lower.matches(term).count() as f32)
            .sum::<f32>() / word_count.max(1.0);
        
        // Feature 1: Narrative content
        features[1] = ["the", "and", "but", "however"]
            .iter()
            .map(|&term| text_lower.matches(term).count() as f32)
            .sum::<f32>() / word_count.max(1.0);
        
        // Feature 2: Length factor
        features[2] = (text.len() as f32 / 100.0).min(1.0);
        
        // Feature 3: Punctuation density
        features[3] = text.chars().filter(|c| ".,;:!?".contains(*c)).count() as f32 / text.len() as f32;
        
        // Feature 4: Number content
        features[4] = if text.chars().any(|c| c.is_numeric()) { 1.0 } else { 0.0 };
        
        // Feature 5: Question content
        features[5] = if text.contains('?') { 1.0 } else { 0.0 };
        
        // Feature 6: Uppercase ratio
        features[6] = text.chars().filter(|c| c.is_uppercase()).count() as f32 / text.len() as f32;
        
        // Feature 7: Complexity
        features[7] = (word_count / 20.0).min(1.0);
        
        features
    }

    /// Fast citation extraction
    fn extract_citations_fast(text: &str) -> Result<Vec<String>> {
        let mut citations = Vec::new();
        
        // Quick URL detection
        for word in text.split_whitespace().take(50) { // Limit search for speed
            if word.starts_with("http") {
                citations.push(word.trim_end_matches(|c: char| !c.is_ascii_alphanumeric()).to_string());
            }
        }
        
        // Quick reference pattern
        if text.contains('[') && text.contains(']') {
            citations.push("reference_pattern_detected".to_string());
        }
        
        Ok(citations)
    }

    /// Calculate fact signature for fast comparison
    fn calculate_fact_signature(facts: &[OptimizedExtractedFact]) -> u64 {
        facts.iter()
            .map(|fact| fact.content_hash ^ fact.entity_hash)
            .fold(0u64, |acc, hash| acc.wrapping_add(hash))
    }

    /// Calculate semantic centroid
    fn calculate_semantic_centroid(facts: &[OptimizedExtractedFact]) -> [f32; 8] {
        if facts.is_empty() {
            return [0.0; 8];
        }
        
        let mut centroid = [0.0f32; 8];
        for fact in facts {
            for i in 0..8 {
                centroid[i] += fact.semantic_features[i];
            }
        }
        
        for i in 0..8 {
            centroid[i] /= facts.len() as f32;
        }
        
        centroid
    }

    /// Fast hash function
    fn fast_hash(text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// Signature similarity (Hamming distance based)
    fn signature_similarity(sig1: u64, sig2: u64) -> f64 {
        let diff = sig1 ^ sig2;
        let hamming_distance = diff.count_ones() as f64;
        1.0 - (hamming_distance / 64.0) // 64 bits
    }

    /// Fast cosine similarity
    fn cosine_similarity(vec1: &[f32; 8], vec2: &[f32; 8]) -> f64 {
        let mut dot_product = 0.0f64;
        let mut norm1 = 0.0f64;
        let mut norm2 = 0.0f64;
        
        for i in 0..8 {
            dot_product += (vec1[i] * vec2[i]) as f64;
            norm1 += (vec1[i] * vec1[i]) as f64;
            norm2 += (vec2[i] * vec2[i]) as f64;
        }
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm1.sqrt() * norm2.sqrt())
    }

    /// Estimate entry size
    fn estimate_size(&self, value: &serde_json::Value) -> usize {
        serde_json::to_string(value).map(|s| s.len()).unwrap_or(100)
    }

    /// Record access time for metrics
    fn record_access_time(&self, start_time: Instant) {
        let elapsed_us = start_time.elapsed().as_micros() as u64;
        self.metrics.total_access_time_us.fetch_add(elapsed_us, Ordering::Relaxed);
    }

    /// Trigger L1 cache cleanup
    async fn trigger_l1_cleanup(&self) {
        if self.l1_cache.len() <= self.config.l1_max_entries {
            return;
        }
        
        // Quick LRU eviction
        let mut entries_to_remove = Vec::new();
        let target_remove = self.l1_cache.len() - (self.config.l1_max_entries * 4 / 5);
        
        for entry in self.l1_cache.iter() {
            let (key, value) = entry.pair();
            let age = value.last_accessed.elapsed();
            if age > Duration::from_millis(self.config.l1_ttl_ms / 2) {
                entries_to_remove.push(key.clone());
                if entries_to_remove.len() >= target_remove {
                    break;
                }
            }
        }
        
        for key in entries_to_remove {
            self.l1_cache.remove(&key);
        }
        
        self.metrics.evictions.fetch_add(1, Ordering::Relaxed);
        self.metrics.l1_size.store(self.l1_cache.len(), Ordering::Relaxed);
    }

    /// Start background tasks for maintenance and optimization
    fn start_background_tasks(
        l1_cache: Arc<DashMap<String, L1CacheEntry>>,
        l2_cache: Arc<DashMap<String, L2CacheEntry>>,
        fact_index: Arc<DashMap<String, FactIndexEntry>>,
        metrics: Arc<CachePerformanceMetrics>,
        config: OptimizedCacheConfig,
    ) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30)); // Every 30 seconds
            
            loop {
                interval.tick().await;
                
                // Cleanup expired entries
                Self::cleanup_expired_entries(&l1_cache, &l2_cache, &fact_index, &config).await;
                
                // Update metrics
                metrics.l1_size.store(l1_cache.len(), Ordering::Relaxed);
                metrics.l2_size.store(l2_cache.len(), Ordering::Relaxed);
                
                // Log performance stats
                let total_requests = metrics.total_requests.load(Ordering::Relaxed);
                if total_requests > 0 {
                    let l1_hits = metrics.l1_hits.load(Ordering::Relaxed);
                    let l2_hits = metrics.l2_hits.load(Ordering::Relaxed);
                    let fact_hits = metrics.fact_similarity_hits.load(Ordering::Relaxed);
                    let hit_rate = (l1_hits + l2_hits + fact_hits) as f64 / total_requests as f64;
                    let avg_access_time_us = metrics.total_access_time_us.load(Ordering::Relaxed) / total_requests;
                    
                    debug!("Cache stats - Hit rate: {:.1}%, Avg access: {}μs, L1: {}, L2: {}", 
                           hit_rate * 100.0, avg_access_time_us, l1_cache.len(), l2_cache.len());
                }
            }
        })
    }

    /// Cleanup expired entries from all cache levels
    async fn cleanup_expired_entries(
        l1_cache: &Arc<DashMap<String, L1CacheEntry>>,
        l2_cache: &Arc<DashMap<String, L2CacheEntry>>,
        fact_index: &Arc<DashMap<String, FactIndexEntry>>,
        config: &OptimizedCacheConfig,
    ) {
        let now = Instant::now();
        let l1_ttl = Duration::from_millis(config.l1_ttl_ms);
        let l2_ttl = Duration::from_millis(config.l2_ttl_ms);
        
        // L1 cleanup
        let mut l1_expired = Vec::new();
        for entry in l1_cache.iter() {
            if now.duration_since(entry.last_accessed) > l1_ttl {
                l1_expired.push(entry.key().clone());
            }
        }
        for key in l1_expired {
            l1_cache.remove(&key);
        }
        
        // L2 cleanup
        let mut l2_expired = Vec::new();
        for entry in l2_cache.iter() {
            let age = now.duration_since(entry.created_at);
            if age > l2_ttl {
                l2_expired.push(entry.key().clone());
            }
        }
        for key in &l2_expired {
            l2_cache.remove(key);
            fact_index.remove(key); // Also remove from fact index
        }
        
        if !l1_expired.is_empty() || !l2_expired.is_empty() {
            debug!("Cache cleanup: removed {} L1 entries, {} L2 entries", 
                   l1_expired.len(), l2_expired.len());
        }
    }

    /// Get comprehensive cache performance metrics
    pub fn get_performance_metrics(&self) -> CachePerformanceSnapshot {
        let total_requests = self.metrics.total_requests.load(Ordering::Relaxed);
        let l1_hits = self.metrics.l1_hits.load(Ordering::Relaxed);
        let l2_hits = self.metrics.l2_hits.load(Ordering::Relaxed);
        let fact_hits = self.metrics.fact_similarity_hits.load(Ordering::Relaxed);
        let misses = self.metrics.misses.load(Ordering::Relaxed);
        
        let total_hits = l1_hits + l2_hits + fact_hits;
        let hit_rate = if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        };
        
        let avg_access_time_us = if total_requests > 0 {
            self.metrics.total_access_time_us.load(Ordering::Relaxed) / total_requests
        } else {
            0
        };
        
        CachePerformanceSnapshot {
            total_requests,
            hit_rate,
            l1_hit_rate: if total_requests > 0 { l1_hits as f64 / total_requests as f64 } else { 0.0 },
            l2_hit_rate: if total_requests > 0 { l2_hits as f64 / total_requests as f64 } else { 0.0 },
            fact_similarity_hit_rate: if total_requests > 0 { fact_hits as f64 / total_requests as f64 } else { 0.0 },
            average_access_time_us: avg_access_time_us,
            l1_size: self.metrics.l1_size.load(Ordering::Relaxed),
            l2_size: self.metrics.l2_size.load(Ordering::Relaxed),
            fact_index_size: self.fact_index.len(),
            evictions: self.metrics.evictions.load(Ordering::Relaxed),
            sub_50ms_performance: avg_access_time_us < 50_000, // Target: <50ms
        }
    }

    /// Clear all cache data
    pub async fn clear(&self) {
        self.l1_cache.clear();
        self.l2_cache.clear();
        self.fact_index.clear();
        
        // Reset metrics
        self.metrics.total_requests.store(0, Ordering::Relaxed);
        self.metrics.l1_hits.store(0, Ordering::Relaxed);
        self.metrics.l2_hits.store(0, Ordering::Relaxed);
        self.metrics.fact_similarity_hits.store(0, Ordering::Relaxed);
        self.metrics.misses.store(0, Ordering::Relaxed);
        self.metrics.total_access_time_us.store(0, Ordering::Relaxed);
        self.metrics.l1_size.store(0, Ordering::Relaxed);
        self.metrics.l2_size.store(0, Ordering::Relaxed);
        self.metrics.evictions.store(0, Ordering::Relaxed);
        
        info!("Cache cleared successfully");
    }
}

/// Cache performance snapshot for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceSnapshot {
    pub total_requests: u64,
    pub hit_rate: f64,
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub fact_similarity_hit_rate: f64,
    pub average_access_time_us: u64,
    pub l1_size: usize,
    pub l2_size: usize,
    pub fact_index_size: usize,
    pub evictions: u64,
    pub sub_50ms_performance: bool, // Target achievement
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_optimized_cache_creation() {
        let config = OptimizedCacheConfig::default();
        let cache = OptimizedFACTCache::new(config);
        
        let metrics = cache.get_performance_metrics();
        assert_eq!(metrics.total_requests, 0);
        assert_eq!(metrics.l1_size, 0);
        assert_eq!(metrics.l2_size, 0);
    }
    
    #[tokio::test]
    async fn test_cache_put_and_get() {
        let config = OptimizedCacheConfig::default();
        let cache = OptimizedFACTCache::new(config);
        
        let key = "test_key".to_string();
        let value = serde_json::json!({"content": "test response"});
        let response_text = "This is a test response with some factual content.";
        
        // Test put
        cache.put(key.clone(), value.clone(), Some(response_text)).await.unwrap();
        
        // Test get (should hit L1)
        let result = cache.get(&key).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap(), value);
        
        let metrics = cache.get_performance_metrics();
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.l1_hit_rate, 1.0);
        assert!(metrics.average_access_time_us < 50_000); // <50ms target
    }
    
    #[tokio::test]
    async fn test_fact_similarity_matching() {
        let config = OptimizedCacheConfig::default();
        let cache = OptimizedFACTCache::new(config);
        
        // Store original content
        let original_text = "The API provides secure authentication using JWT tokens for user authorization.";
        cache.put(
            "original".to_string(),
            serde_json::json!({"answer": "JWT authentication details"}),
            Some(original_text)
        ).await.unwrap();
        
        // Wait for background processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Search with similar content
        let similar_query = "How does the API handle authentication and JWT tokens?";
        let result = cache.get(similar_query).await;
        
        // Should find similar content
        assert!(result.is_some());
        
        let metrics = cache.get_performance_metrics();
        assert!(metrics.fact_similarity_hit_rate > 0.0);
    }
    
    #[tokio::test]
    async fn test_performance_target() {
        let config = OptimizedCacheConfig::default();
        let cache = OptimizedFACTCache::new(config);
        
        // Perform multiple operations
        for i in 0..100 {
            let key = format!("key_{}", i);
            let value = serde_json::json!({"id": i, "content": format!("Content {}", i)});
            cache.put(key.clone(), value, Some(&format!("Content {} with facts", i))).await.unwrap();
            
            // Immediate read (L1 hit)
            let result = cache.get(&key).await;
            assert!(result.is_some());
        }
        
        let metrics = cache.get_performance_metrics();
        println!("Performance metrics: {:#?}", metrics);
        
        // Verify sub-50ms performance target
        assert!(metrics.sub_50ms_performance, "Failed to achieve <50ms cache performance");
        assert!(metrics.hit_rate > 0.9, "Hit rate should be >90%");
        assert!(metrics.average_access_time_us < 50_000, "Average access time should be <50ms");
    }
    
    #[tokio::test]
    async fn test_cache_cleanup() {
        let mut config = OptimizedCacheConfig::default();
        config.l1_ttl_ms = 50; // Very short TTL for testing
        
        let cache = OptimizedFACTCache::new(config);
        
        // Add entry
        cache.put(
            "temp_key".to_string(),
            serde_json::json!({"temp": "data"}),
            Some("Temporary content")
        ).await.unwrap();
        
        // Should be available immediately
        assert!(cache.get("temp_key").await.is_some());
        
        // Wait for expiry
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        
        // Should be cleaned up
        assert!(cache.get("temp_key").await.is_none());
    }
}