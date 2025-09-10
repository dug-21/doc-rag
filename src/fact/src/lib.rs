//! FACT - Fast Accurate Caching Technology
//! 
//! Provides <50ms intelligent caching for the RAG system per Phase 2 requirements.
//! This is a Rust implementation following the FACT architecture principles.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use blake3::Hasher;
use async_trait::async_trait;
use tracing::{info, warn};
use chrono::{DateTime, Utc};

/// Configuration for FACT caching system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FACTConfig {
    /// Enable FACT acceleration
    pub enabled: bool,
    
    /// Maximum cache size
    pub cache_size: usize,
    
    /// Target response time for cached responses (ms)
    pub target_cached_response_time: u64,
    
    /// Cache TTL in seconds
    pub ttl_seconds: u64,
}

impl Default for FACTConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_size: 10000,
            target_cached_response_time: 50,
            ttl_seconds: 3600,
        }
    }
}

/// Cached entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEntry<T> {
    pub data: T,
    pub created_at: DateTime<Utc>,
    pub hit_count: u64,
    pub last_accessed: DateTime<Utc>,
}

/// Main FACT cache implementation
pub struct FACTCache<T: Clone + Send + Sync> {
    cache: Arc<DashMap<String, CachedEntry<T>>>,
    config: FACTConfig,
}

impl<T: Clone + Send + Sync + 'static> FACTCache<T> {
    /// Create new FACT cache instance
    pub fn new(config: FACTConfig) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            config,
        }
    }
    
    /// Generate cache key from query
    pub fn generate_key(&self, query: &str) -> String {
        let mut hasher = Hasher::new();
        hasher.update(query.as_bytes());
        let hash = hasher.finalize();
        hash.to_string()
    }
    
    /// Get cached value
    pub async fn get(&self, key: &str) -> Option<T> {
        if !self.config.enabled {
            return None;
        }
        
        let mut entry = self.cache.get_mut(key)?;
        let now = Utc::now();
        let age = (now - entry.last_accessed).num_seconds() as u64;
        
        // Check TTL
        if age > self.config.ttl_seconds {
            drop(entry);
            self.cache.remove(key);
            return None;
        }
        
        // Update metadata
        entry.hit_count += 1;
        entry.last_accessed = now;
        
        Some(entry.data.clone())
    }
    
    /// Set cached value
    pub async fn set(&self, key: String, value: T) {
        if !self.config.enabled {
            return;
        }
        
        // Check cache size limit
        if self.cache.len() >= self.config.cache_size {
            // Simple LRU eviction - remove oldest entry
            if let Some(oldest_key) = self.find_oldest_entry() {
                self.cache.remove(&oldest_key);
            }
        }
        
        let entry = CachedEntry {
            data: value,
            created_at: Utc::now(),
            hit_count: 0,
            last_accessed: Utc::now(),
        };
        
        self.cache.insert(key, entry);
    }
    
    /// Find oldest entry for eviction
    fn find_oldest_entry(&self) -> Option<String> {
        self.cache
            .iter()
            .min_by_key(|entry| entry.last_accessed)
            .map(|entry| entry.key().clone())
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_entries: self.cache.len(),
            enabled: self.config.enabled,
            target_response_time_ms: self.config.target_cached_response_time,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub enabled: bool,
    pub target_response_time_ms: u64,
}

/// Trait for FACT-compatible caching
#[async_trait]
pub trait FACTClient: Send + Sync {
    type Item: Clone + Send + Sync;
    
    /// Get cached item
    async fn get(&self, key: &str) -> Option<Self::Item>;
    
    /// Set cached item
    async fn set(&self, key: String, value: Self::Item);
    
    /// Generate cache key
    fn generate_key(&self, query: &str) -> String;
}

// Re-export commonly used types
pub use FACTCache as Cache;
pub use FACTConfig as Config;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_caching() {
        let config = FACTConfig::default();
        let cache: FACTCache<String> = FACTCache::new(config);
        
        let key = cache.generate_key("test query");
        cache.set(key.clone(), "test result".to_string()).await;
        
        let result = cache.get(&key).await;
        assert_eq!(result, Some("test result".to_string()));
    }
    
    #[tokio::test]
    async fn test_cache_miss() {
        let config = FACTConfig::default();
        let cache: FACTCache<String> = FACTCache::new(config);
        
        let key = cache.generate_key("non-existent");
        let result = cache.get(&key).await;
        assert_eq!(result, None);
    }
}