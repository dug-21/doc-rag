//! Embedding cache for improved performance
//!
//! This module provides an efficient caching system for embeddings with
//! LRU eviction, TTL support, and memory management.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use serde::{Deserialize, Serialize};

/// An LRU cache for embeddings with TTL support
pub struct EmbeddingCache {
    cache: Arc<RwLock<LruCache>>,
    max_size: usize,
    default_ttl: Option<Duration>,
}

/// Internal LRU cache implementation
struct LruCache {
    entries: HashMap<String, CacheEntry>,
    access_order: Vec<String>,
    max_size: usize,
}

/// A cached embedding entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    embedding: Vec<f32>,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    ttl: Option<Duration>,
}

/// Statistics for the embedding cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub max_capacity: usize,
    pub memory_usage_bytes: usize,
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub total_hits: u64,
    pub total_misses: u64,
    pub total_requests: u64,
    pub eviction_count: u64,
    pub expired_count: u64,
    pub avg_access_count: f64,
}

impl EmbeddingCache {
    /// Create a new embedding cache with specified capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(max_size))),
            max_size,
            default_ttl: Some(Duration::from_secs(3600)), // 1 hour default TTL
        }
    }
    
    /// Create a new cache with custom TTL
    pub fn with_ttl(max_size: usize, ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(max_size))),
            max_size,
            default_ttl: Some(ttl),
        }
    }
    
    /// Create a cache without TTL (items never expire)
    pub fn without_ttl(max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(LruCache::new(max_size))),
            max_size,
            default_ttl: None,
        }
    }
    
    /// Get an embedding from the cache
    pub async fn get(&self, key: &str) -> Option<Vec<f32>> {
        let mut cache = self.cache.write().await;
        cache.get(key)
    }
    
    /// Put an embedding into the cache
    pub async fn put(&self, key: String, embedding: Vec<f32>) {
        let mut cache = self.cache.write().await;
        cache.put(key, embedding, self.default_ttl);
    }
    
    /// Put an embedding with custom TTL
    pub async fn put_with_ttl(&self, key: String, embedding: Vec<f32>, ttl: Duration) {
        let mut cache = self.cache.write().await;
        cache.put(key, embedding, Some(ttl));
    }
    
    /// Check if a key exists in the cache
    pub async fn contains_key(&self, key: &str) -> bool {
        let cache = self.cache.read().await;
        cache.contains_key(key)
    }
    
    /// Remove an entry from the cache
    pub async fn remove(&self, key: &str) -> Option<Vec<f32>> {
        let mut cache = self.cache.write().await;
        cache.remove(key)
    }
    
    /// Clear all entries from the cache
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        info!("Cache cleared");
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        cache.get_stats()
    }
    
    /// Cleanup expired entries
    pub async fn cleanup_expired(&self) -> usize {
        let mut cache = self.cache.write().await;
        let removed_count = cache.cleanup_expired();
        if removed_count > 0 {
            debug!("Removed {} expired cache entries", removed_count);
        }
        removed_count
    }
    
    /// Resize the cache
    pub async fn resize(&self, new_size: usize) {
        let mut cache = self.cache.write().await;
        cache.resize(new_size);
        info!("Cache resized to {} entries", new_size);
    }
    
    /// Get cache utilization percentage
    pub async fn utilization(&self) -> f64 {
        let cache = self.cache.read().await;
        if cache.max_size == 0 {
            0.0
        } else {
            cache.entries.len() as f64 / cache.max_size as f64 * 100.0
        }
    }
    
    /// Get the keys of all cached entries (for debugging)
    pub async fn keys(&self) -> Vec<String> {
        let cache = self.cache.read().await;
        cache.entries.keys().cloned().collect()
    }
    
    /// Get memory usage in bytes
    pub async fn memory_usage(&self) -> usize {
        let cache = self.cache.read().await;
        cache.calculate_memory_usage()
    }
    
    /// Warmup cache with commonly accessed embeddings
    pub async fn warmup(&self, embeddings: Vec<(String, Vec<f32>)>) {
        let mut cache = self.cache.write().await;
        for (key, embedding) in embeddings {
            cache.put(key, embedding, self.default_ttl);
        }
        info!("Cache warmed up with {} entries", cache.entries.len());
    }
    
    /// Export cache contents for persistence
    pub async fn export(&self) -> Vec<(String, Vec<f32>)> {
        let cache = self.cache.read().await;
        cache.entries.iter()
            .filter(|(_, entry)| !entry.is_expired())
            .map(|(key, entry)| (key.clone(), entry.embedding.clone()))
            .collect()
    }
    
    /// Import cache contents from persistence
    pub async fn import(&self, entries: Vec<(String, Vec<f32>)>) {
        let mut cache = self.cache.write().await;
        cache.clear();
        for (key, embedding) in entries {
            cache.put(key, embedding, self.default_ttl);
        }
        info!("Imported {} entries into cache", cache.entries.len());
    }
}

impl LruCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_order: Vec::new(),
            max_size,
        }
    }
    
    fn get(&mut self, key: &str) -> Option<Vec<f32>> {
        if let Some(entry) = self.entries.get_mut(key) {
            if entry.is_expired() {
                self.remove(key);
                return None;
            }
            
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Move to end of access order (most recently used)
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            self.access_order.push(key.to_string());
            
            Some(entry.embedding.clone())
        } else {
            None
        }
    }
    
    fn put(&mut self, key: String, embedding: Vec<f32>, ttl: Option<Duration>) {
        let now = Instant::now();
        
        // Remove existing entry if present
        if self.entries.contains_key(&key) {
            self.remove(&key);
        }
        
        // Make room if necessary
        while self.entries.len() >= self.max_size && self.max_size > 0 {
            self.evict_lru();
        }
        
        let entry = CacheEntry {
            embedding,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            ttl,
        };
        
        self.entries.insert(key.clone(), entry);
        self.access_order.push(key);
    }
    
    fn contains_key(&self, key: &str) -> bool {
        if let Some(entry) = self.entries.get(key) {
            !entry.is_expired()
        } else {
            false
        }
    }
    
    fn remove(&mut self, key: &str) -> Option<Vec<f32>> {
        if let Some(entry) = self.entries.remove(key) {
            self.access_order.retain(|k| k != key);
            Some(entry.embedding)
        } else {
            None
        }
    }
    
    fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }
    
    fn evict_lru(&mut self) -> Option<String> {
        if let Some(lru_key) = self.access_order.first().cloned() {
            self.remove(&lru_key);
            Some(lru_key)
        } else {
            None
        }
    }
    
    fn cleanup_expired(&mut self) -> usize {
        let expired_keys: Vec<String> = self.entries.iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();
        
        let count = expired_keys.len();
        for key in expired_keys {
            self.remove(&key);
        }
        count
    }
    
    fn resize(&mut self, new_size: usize) {
        self.max_size = new_size;
        
        // Evict entries if new size is smaller
        while self.entries.len() > new_size && new_size > 0 {
            self.evict_lru();
        }
    }
    
    fn get_stats(&self) -> CacheStats {
        let total_entries = self.entries.len();
        let memory_usage = self.calculate_memory_usage();
        
        let total_access_count: u64 = self.entries.values()
            .map(|entry| entry.access_count)
            .sum();
        
        let avg_access_count = if total_entries > 0 {
            total_access_count as f64 / total_entries as f64
        } else {
            0.0
        };
        
        // Note: These would be tracked separately in a real implementation
        let total_hits = total_access_count; // Approximation
        let total_misses = 0; // Would need separate tracking
        let total_requests = total_hits + total_misses;
        
        let hit_rate = if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        };
        
        CacheStats {
            total_entries,
            max_capacity: self.max_size,
            memory_usage_bytes: memory_usage,
            hit_rate,
            miss_rate: 1.0 - hit_rate,
            total_hits,
            total_misses,
            total_requests,
            eviction_count: 0, // Would need separate tracking
            expired_count: 0,  // Would need separate tracking
            avg_access_count,
        }
    }
    
    fn calculate_memory_usage(&self) -> usize {
        let key_memory: usize = self.entries.keys()
            .map(|key| key.len())
            .sum();
        
        let embedding_memory: usize = self.entries.values()
            .map(|entry| entry.embedding.len() * std::mem::size_of::<f32>())
            .sum();
        
        let metadata_memory = self.entries.len() * std::mem::size_of::<CacheEntry>();
        let access_order_memory = self.access_order.capacity() * std::mem::size_of::<String>();
        
        key_memory + embedding_memory + metadata_memory + access_order_memory
    }
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }
}

/// Persistent cache that can save/load to disk
pub struct PersistentEmbeddingCache {
    cache: EmbeddingCache,
    persistence_path: std::path::PathBuf,
    auto_save_interval: Option<Duration>,
}

impl PersistentEmbeddingCache {
    /// Create a new persistent cache
    pub fn new(max_size: usize, persistence_path: std::path::PathBuf) -> Self {
        Self {
            cache: EmbeddingCache::new(max_size),
            persistence_path,
            auto_save_interval: Some(Duration::from_secs(300)), // 5 minutes
        }
    }
    
    /// Load cache from disk
    pub async fn load(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.persistence_path.exists() {
            info!("Cache persistence file not found, starting with empty cache");
            return Ok(());
        }
        
        let data = tokio::fs::read(&self.persistence_path).await?;
        let entries: Vec<(String, Vec<f32>)> = bincode::deserialize(&data)?;
        
        self.cache.import(entries).await;
        info!("Loaded cache from {:?}", self.persistence_path);
        
        Ok(())
    }
    
    /// Save cache to disk
    pub async fn save(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let entries = self.cache.export().await;
        let data = bincode::serialize(&entries)?;
        
        // Create parent directories if they don't exist
        if let Some(parent) = self.persistence_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        tokio::fs::write(&self.persistence_path, data).await?;
        info!("Saved cache to {:?}", self.persistence_path);
        
        Ok(())
    }
    
    /// Get the underlying cache
    pub fn cache(&self) -> &EmbeddingCache {
        &self.cache
    }
    
    /// Start auto-save background task
    pub async fn start_auto_save(
        &self,
    ) -> Result<tokio::task::JoinHandle<()>, Box<dyn std::error::Error + Send + Sync>> {
        if let Some(interval) = self.auto_save_interval {
            let cache_clone = self.cache.clone();
            let path_clone = self.persistence_path.clone();
            
            let handle = tokio::spawn(async move {
                let mut interval_timer = tokio::time::interval(interval);
                
                loop {
                    interval_timer.tick().await;
                    
                    let entries = cache_clone.export().await;
                    if !entries.is_empty() {
                        match bincode::serialize(&entries) {
                            Ok(data) => {
                                if let Err(e) = tokio::fs::write(&path_clone, data).await {
                                    warn!("Failed to auto-save cache: {}", e);
                                } else {
                                    debug!("Auto-saved cache with {} entries", entries.len());
                                }
                            }
                            Err(e) => {
                                warn!("Failed to serialize cache for auto-save: {}", e);
                            }
                        }
                    }
                }
            });
            
            Ok(handle)
        } else {
            Err("Auto-save interval not configured".into())
        }
    }
}

impl Clone for EmbeddingCache {
    fn clone(&self) -> Self {
        Self {
            cache: Arc::clone(&self.cache),
            max_size: self.max_size,
            default_ttl: self.default_ttl,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_cache_basic_operations() {
        let cache = EmbeddingCache::new(10);
        
        // Test put and get
        let embedding = vec![1.0, 2.0, 3.0];
        cache.put("key1".to_string(), embedding.clone()).await;
        
        let retrieved = cache.get("key1").await;
        assert_eq!(retrieved, Some(embedding));
        
        // Test contains_key
        assert!(cache.contains_key("key1").await);
        assert!(!cache.contains_key("nonexistent").await);
        
        // Test remove
        let removed = cache.remove("key1").await;
        assert_eq!(removed, Some(vec![1.0, 2.0, 3.0]));
        assert!(!cache.contains_key("key1").await);
    }
    
    #[tokio::test]
    async fn test_cache_lru_eviction() {
        let cache = EmbeddingCache::new(2); // Very small cache
        
        cache.put("key1".to_string(), vec![1.0]).await;
        cache.put("key2".to_string(), vec![2.0]).await;
        cache.put("key3".to_string(), vec![3.0]).await; // Should evict key1
        
        assert!(!cache.contains_key("key1").await); // Evicted
        assert!(cache.contains_key("key2").await);
        assert!(cache.contains_key("key3").await);
    }
    
    #[tokio::test]
    async fn test_cache_ttl() {
        let cache = EmbeddingCache::with_ttl(10, Duration::from_millis(100));
        
        cache.put("key1".to_string(), vec![1.0]).await;
        assert!(cache.contains_key("key1").await);
        
        // Wait for expiration
        sleep(Duration::from_millis(150)).await;
        
        assert!(!cache.contains_key("key1").await); // Should be expired
        assert_eq!(cache.get("key1").await, None);
    }
    
    #[tokio::test]
    async fn test_cache_stats() {
        let cache = EmbeddingCache::new(10);
        
        cache.put("key1".to_string(), vec![1.0, 2.0]).await;
        cache.put("key2".to_string(), vec![3.0, 4.0, 5.0]).await;
        
        let stats = cache.get_stats().await;
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.max_capacity, 10);
        assert!(stats.memory_usage_bytes > 0);
    }
    
    #[tokio::test]
    async fn test_cache_cleanup() {
        let cache = EmbeddingCache::with_ttl(10, Duration::from_millis(50));
        
        cache.put("key1".to_string(), vec![1.0]).await;
        cache.put("key2".to_string(), vec![2.0]).await;
        
        // Wait for expiration
        sleep(Duration::from_millis(100)).await;
        
        let removed_count = cache.cleanup_expired().await;
        assert_eq!(removed_count, 2);
        assert_eq!(cache.get_stats().await.total_entries, 0);
    }
    
    #[tokio::test]
    async fn test_cache_resize() {
        let cache = EmbeddingCache::new(5);
        
        // Fill cache
        for i in 0..5 {
            cache.put(format!("key{}", i), vec![i as f32]).await;
        }
        
        assert_eq!(cache.get_stats().await.total_entries, 5);
        
        // Resize to smaller
        cache.resize(2).await;
        
        let stats = cache.get_stats().await;
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.max_capacity, 2);
    }
    
    #[tokio::test]
    async fn test_cache_warmup_and_export() {
        let cache = EmbeddingCache::new(10);
        
        let warmup_data = vec![
            ("key1".to_string(), vec![1.0, 2.0]),
            ("key2".to_string(), vec![3.0, 4.0]),
        ];
        
        cache.warmup(warmup_data.clone()).await;
        
        assert!(cache.contains_key("key1").await);
        assert!(cache.contains_key("key2").await);
        
        let exported = cache.export().await;
        assert_eq!(exported.len(), 2);
    }
    
    #[tokio::test]
    async fn test_persistent_cache() {
        let temp_dir = tempdir().unwrap();
        let cache_path = temp_dir.path().join("cache.bin");
        
        {
            let cache = PersistentEmbeddingCache::new(10, cache_path.clone());
            
            cache.cache().put("key1".to_string(), vec![1.0, 2.0, 3.0]).await;
            cache.cache().put("key2".to_string(), vec![4.0, 5.0, 6.0]).await;
            
            cache.save().await.unwrap();
        }
        
        // Create new cache and load
        {
            let cache = PersistentEmbeddingCache::new(10, cache_path);
            cache.load().await.unwrap();
            
            assert!(cache.cache().contains_key("key1").await);
            assert!(cache.cache().contains_key("key2").await);
            
            let retrieved = cache.cache().get("key1").await.unwrap();
            assert_eq!(retrieved, vec![1.0, 2.0, 3.0]);
        }
    }
    
    #[tokio::test]
    async fn test_cache_utilization() {
        let cache = EmbeddingCache::new(10);
        
        assert_eq!(cache.utilization().await, 0.0);
        
        cache.put("key1".to_string(), vec![1.0]).await;
        assert_eq!(cache.utilization().await, 10.0); // 1/10 * 100%
        
        for i in 2..10 {
            cache.put(format!("key{}", i), vec![i as f32]).await;
        }
        
        assert_eq!(cache.utilization().await, 90.0); // 9/10 * 100%
    }
}