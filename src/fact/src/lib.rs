//! FACT - Fast Accurate Caching Technology
//! 
//! Provides <50ms intelligent caching for the RAG system per Phase 2 requirements.
//! This is a Rust implementation following the FACT architecture principles.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use blake3::Hasher;
use async_trait::async_trait;
use tracing::{info, warn, debug};
use chrono::{DateTime, Utc};
use std::time::Instant;

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
    // Performance tracking
    hit_count: Arc<std::sync::atomic::AtomicU64>,
    miss_count: Arc<std::sync::atomic::AtomicU64>,
}

impl<T: Clone + Send + Sync + 'static> FACTCache<T> {
    /// Create new FACT cache instance
    pub fn new(config: FACTConfig) -> Self {
        info!("Initializing FACT cache with target response time: {}ms", config.target_cached_response_time);
        Self {
            cache: Arc::new(DashMap::new()),
            config,
            hit_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            miss_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
    
    /// Generate cache key from query
    pub fn generate_key(&self, query: &str) -> String {
        let mut hasher = Hasher::new();
        hasher.update(query.as_bytes());
        let hash = hasher.finalize();
        hash.to_string()
    }
    
    /// Get cached value with performance tracking
    pub async fn get(&self, key: &str) -> Option<T> {
        let start = Instant::now();
        
        if !self.config.enabled {
            self.miss_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
        
        let entry = self.cache.get_mut(key);
        
        let result = if let Some(mut entry) = entry {
            let now = Utc::now();
            let age = (now - entry.last_accessed).num_seconds() as u64;
            
            // Check TTL
            if age > self.config.ttl_seconds {
                drop(entry);
                self.cache.remove(key);
                self.miss_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            } else {
                // Update metadata
                entry.hit_count += 1;
                entry.last_accessed = now;
                self.hit_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(entry.data.clone())
            }
        } else {
            self.miss_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        };
        
        let duration = start.elapsed();
        if duration.as_millis() as u64 > self.config.target_cached_response_time {
            warn!("FACT cache get operation took {}ms, target is {}ms", 
                  duration.as_millis(), self.config.target_cached_response_time);
        } else {
            debug!("FACT cache get operation completed in {}ms", duration.as_millis());
        }
        
        result
    }
    
    /// Set cached value with performance tracking
    pub async fn set(&self, key: String, value: T) {
        let start = Instant::now();
        
        if !self.config.enabled {
            return;
        }
        
        // Check cache size limit and evict if necessary
        if self.cache.len() >= self.config.cache_size {
            if let Some(oldest_key) = self.find_oldest_entry() {
                self.cache.remove(&oldest_key);
                debug!("Evicted oldest cache entry: {}", oldest_key);
            }
        }
        
        let entry = CachedEntry {
            data: value,
            created_at: Utc::now(),
            hit_count: 0,
            last_accessed: Utc::now(),
        };
        
        self.cache.insert(key.clone(), entry);
        
        let duration = start.elapsed();
        if duration.as_millis() as u64 > self.config.target_cached_response_time {
            warn!("FACT cache set operation took {}ms, target is {}ms", 
                  duration.as_millis(), self.config.target_cached_response_time);
        } else {
            debug!("FACT cache set operation completed in {}ms for key: {}", 
                   duration.as_millis(), key);
        }
    }
    
    /// Find oldest entry for eviction
    fn find_oldest_entry(&self) -> Option<String> {
        self.cache
            .iter()
            .min_by_key(|entry| entry.last_accessed)
            .map(|entry| entry.key().clone())
    }
    
    /// Get cache statistics with performance metrics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hit_count.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.miss_count.load(std::sync::atomic::Ordering::Relaxed);
        let total_requests = hits + misses;
        let hit_rate = if total_requests > 0 { 
            (hits as f64 / total_requests as f64) * 100.0 
        } else { 
            0.0 
        };
        
        CacheStats {
            total_entries: self.cache.len(),
            enabled: self.config.enabled,
            target_response_time_ms: self.config.target_cached_response_time,
            hit_count: hits,
            miss_count: misses,
            hit_rate_percent: hit_rate,
        }
    }
    
    /// Clear cache and reset statistics
    pub fn clear(&self) {
        self.cache.clear();
        self.hit_count.store(0, std::sync::atomic::Ordering::Relaxed);
        self.miss_count.store(0, std::sync::atomic::Ordering::Relaxed);
        info!("FACT cache cleared");
    }
    
    /// Health check - returns true if performing within target response time
    pub async fn health_check(&self) -> bool
    where
        T: Default,
    {
        let test_key = "__health_check__";
        let test_value = T::default();
        
        let start = Instant::now();
        
        // Test set operation
        self.cache.insert(test_key.to_string(), CachedEntry {
            data: test_value,
            created_at: Utc::now(),
            hit_count: 0,
            last_accessed: Utc::now(),
        });
        
        // Test get operation
        let _result = self.cache.get(test_key);
        
        // Cleanup
        self.cache.remove(test_key);
        
        let duration = start.elapsed();
        let healthy = duration.as_millis() as u64 <= self.config.target_cached_response_time;
        
        if !healthy {
            warn!("FACT cache health check failed: {}ms > {}ms target", 
                  duration.as_millis(), self.config.target_cached_response_time);
        } else {
            debug!("FACT cache health check passed: {}ms", duration.as_millis());
        }
        
        healthy
    }
}

/// Cache statistics with performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub enabled: bool,
    pub target_response_time_ms: u64,
    pub hit_count: u64,
    pub miss_count: u64,
    pub hit_rate_percent: f64,
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
    
    #[tokio::test]
    async fn test_performance_tracking() {
        let config = FACTConfig::default();
        let cache: FACTCache<String> = FACTCache::new(config);
        
        let key = cache.generate_key("test query");
        
        // Test miss
        let result = cache.get(&key).await;
        assert_eq!(result, None);
        
        // Set value
        cache.set(key.clone(), "test result".to_string()).await;
        
        // Test hit
        let result = cache.get(&key).await;
        assert_eq!(result, Some("test result".to_string()));
        
        let stats = cache.stats();
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
        assert_eq!(stats.hit_rate_percent, 50.0);
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let config = FACTConfig::default();
        let cache: FACTCache<&str> = FACTCache::new(config);
        
        let healthy = cache.health_check().await;
        assert!(healthy);
    }
    
    #[tokio::test]
    async fn test_ttl_expiration() {
        let mut config = FACTConfig::default();
        config.ttl_seconds = 1; // 1 second TTL
        let cache: FACTCache<String> = FACTCache::new(config);
        
        let key = cache.generate_key("ttl test");
        cache.set(key.clone(), "test".to_string()).await;
        
        // Should be available immediately
        let result = cache.get(&key).await;
        assert_eq!(result, Some("test".to_string()));
        
        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        // Should be expired
        let result = cache.get(&key).await;
        assert_eq!(result, None);
    }
}