//! High-performance query caching layer with LRU eviction and multi-level caching
//! 
//! This module provides comprehensive caching functionality for the query processor:
//! - Multi-level caching (L1: in-memory, L2: distributed)
//! - LRU eviction policy with adaptive sizing
//! - Query result caching with confidence-based invalidation
//! - Entity extraction caching with TTL
//! - Classification result caching with version tracking
//! - Strategy caching with performance metrics
//! - Cache warming and preloading capabilities
//! - Distributed cache synchronization
//! - Cache metrics and monitoring

use crate::error::{Result, QueryProcessorError};
use crate::types::{
    QueryIntent, SearchStrategy, QueryResult, ExtractedEntity, ProcessedQuery,
    ClassificationResult, StrategyRecommendation, ConsensusResult, ValidationResult
};
use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use tracing::{debug, info, warn, instrument};
use uuid::Uuid;

/// Cache entry with metadata and TTL support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    /// Cached value
    pub value: T,
    
    /// Entry creation timestamp
    pub created_at: u64,
    
    /// Time-to-live in seconds (0 = no expiration)
    pub ttl: u64,
    
    /// Access count for LRU tracking
    pub access_count: u64,
    
    /// Last access timestamp
    pub last_accessed: u64,
    
    /// Confidence score (for result caching)
    pub confidence: Option<f64>,
    
    /// Entry version for consistency
    pub version: u32,
    
    /// Entry tags for selective invalidation
    pub tags: Vec<String>,
    
    /// Entry size in bytes (for memory management)
    pub size_bytes: usize,
}

/// Cache configuration with performance tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// L1 cache maximum entries
    pub l1_max_entries: usize,
    
    /// L2 cache maximum entries (distributed)
    pub l2_max_entries: usize,
    
    /// Maximum memory usage in bytes
    pub max_memory_bytes: usize,
    
    /// Default TTL for entries (seconds)
    pub default_ttl: u64,
    
    /// LRU eviction batch size
    pub eviction_batch_size: usize,
    
    /// Cache warming settings
    pub enable_warming: bool,
    
    /// Preload common queries
    pub enable_preloading: bool,
    
    /// Background cleanup interval
    pub cleanup_interval: Duration,
    
    /// Distributed cache settings
    pub distributed: DistributedCacheConfig,
    
    /// Performance monitoring
    pub enable_metrics: bool,
    
    /// Cache levels configuration
    pub levels: CacheLevelConfig,
}

/// Distributed cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedCacheConfig {
    /// Enable distributed caching
    pub enabled: bool,
    
    /// Redis connection URL
    pub redis_url: Option<String>,
    
    /// Cluster nodes for distributed caching
    pub cluster_nodes: Vec<String>,
    
    /// Replication factor
    pub replication_factor: u32,
    
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    
    /// Synchronization interval
    pub sync_interval: Duration,
}

/// Cache consistency level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// Eventually consistent
    Eventual,
    
    /// Strong consistency
    Strong,
    
    /// Session consistency
    Session,
    
    /// Monotonic read consistency
    MonotonicRead,
}

/// Cache level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLevelConfig {
    /// Query result cache settings
    pub query_results: CacheLevelSettings,
    
    /// Entity extraction cache settings
    pub entity_extraction: CacheLevelSettings,
    
    /// Classification cache settings
    pub classification: CacheLevelSettings,
    
    /// Strategy cache settings
    pub strategy: CacheLevelSettings,
    
    /// Consensus cache settings
    pub consensus: CacheLevelSettings,
    
    /// Validation cache settings
    pub validation: CacheLevelSettings,
}

/// Individual cache level settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheLevelSettings {
    /// Enable this cache level
    pub enabled: bool,
    
    /// Maximum entries for this level
    pub max_entries: usize,
    
    /// Default TTL for this level
    pub ttl: u64,
    
    /// Minimum confidence for caching
    pub min_confidence: f64,
    
    /// Enable compression for this level
    pub enable_compression: bool,
}

/// Cache statistics and metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Total cache hits
    pub hits: u64,
    
    /// Total cache misses
    pub misses: u64,
    
    /// Cache hit ratio
    pub hit_ratio: f64,
    
    /// Total entries across all levels
    pub total_entries: usize,
    
    /// Current memory usage in bytes
    pub memory_usage: usize,
    
    /// Memory usage ratio
    pub memory_ratio: f64,
    
    /// Average access time in microseconds
    pub avg_access_time_us: u64,
    
    /// Eviction statistics
    pub evictions: EvictionMetrics,
    
    /// Per-level metrics
    pub level_metrics: HashMap<String, LevelMetrics>,
    
    /// Distributed cache metrics
    pub distributed_metrics: Option<DistributedMetrics>,
}

/// Eviction metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvictionMetrics {
    /// Total evictions
    pub total_evictions: u64,
    
    /// LRU evictions
    pub lru_evictions: u64,
    
    /// TTL evictions
    pub ttl_evictions: u64,
    
    /// Size-based evictions
    pub size_evictions: u64,
    
    /// Confidence-based evictions
    pub confidence_evictions: u64,
}

/// Per-level cache metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LevelMetrics {
    /// Level-specific hits
    pub hits: u64,
    
    /// Level-specific misses
    pub misses: u64,
    
    /// Current entries in this level
    pub entries: usize,
    
    /// Memory usage for this level
    pub memory_bytes: usize,
    
    /// Average entry size
    pub avg_entry_size: usize,
    
    /// Average access time for this level
    pub avg_access_time_us: u64,
}

/// Distributed cache metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedMetrics {
    /// Network roundtrip time
    pub network_latency_us: u64,
    
    /// Synchronization events
    pub sync_events: u64,
    
    /// Consistency conflicts
    pub conflicts: u64,
    
    /// Replication lag
    pub replication_lag_ms: u64,
}

/// Cache key generator for consistent hashing
pub trait CacheKeyGenerator<T> {
    /// Generate cache key for the given item
    fn generate_key(&self, item: &T) -> String;
    
    /// Generate versioned key
    fn generate_versioned_key(&self, item: &T, version: u32) -> String {
        format!("{}:v{}", self.generate_key(item), version)
    }
    
    /// Generate tagged key
    fn generate_tagged_key(&self, item: &T, tags: &[String]) -> String {
        let base_key = self.generate_key(item);
        let tag_suffix = tags.join(",");
        format!("{}:tags:{}", base_key, tag_suffix)
    }
}

/// Multi-level query cache with LRU eviction
pub struct QueryCache {
    /// L1 cache (in-memory, fastest access)
    l1_cache: Arc<DashMap<String, CacheEntry<Vec<u8>>>>,
    
    /// L2 cache (distributed, larger capacity)
    l2_cache: Option<Arc<dyn DistributedCache + Send + Sync>>,
    
    /// Cache configuration
    config: Arc<CacheConfig>,
    
    /// Cache metrics
    metrics: Arc<RwLock<CacheMetrics>>,
    
    /// Memory usage tracker
    memory_usage: Arc<Mutex<usize>>,
    
    /// Background cleanup task handle
    cleanup_task: Option<tokio::task::JoinHandle<()>>,
    
    /// Cache warming manager
    warming_manager: Option<Arc<CacheWarmingManager>>,
    
    /// Key generators for different types
    key_generators: KeyGeneratorRegistry,
}

/// Distributed cache trait for L2 caching
#[async_trait]
pub trait DistributedCache: Debug + Send + Sync {
    /// Get value from distributed cache
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// Set value in distributed cache
    async fn set(&self, key: &str, value: Vec<u8>, ttl: Duration) -> Result<()>;
    
    /// Remove value from distributed cache
    async fn remove(&self, key: &str) -> Result<bool>;
    
    /// Check if key exists
    async fn exists(&self, key: &str) -> Result<bool>;
    
    /// Get multiple values
    async fn mget(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>>;
    
    /// Set multiple values
    async fn mset(&self, entries: &[(String, Vec<u8>, Duration)]) -> Result<()>;
    
    /// Clear all entries
    async fn clear(&self) -> Result<()>;
    
    /// Get cache size
    async fn size(&self) -> Result<usize>;
    
    /// Get distributed cache metrics
    async fn metrics(&self) -> Result<DistributedMetrics>;
}

/// Cache warming manager for preloading common queries
pub struct CacheWarmingManager {
    /// Common query patterns
    warming_patterns: Arc<RwLock<Vec<String>>>,
    
    /// Warming scheduler
    scheduler: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    
    /// Warming configuration
    config: WarmingConfig,
}

/// Cache warming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmingConfig {
    /// Enable warming on startup
    pub warm_on_startup: bool,
    
    /// Warming interval
    pub warming_interval: Duration,
    
    /// Number of patterns to warm per cycle
    pub patterns_per_cycle: usize,
    
    /// Maximum warming time per cycle
    pub max_warming_time: Duration,
    
    /// Common query patterns file
    pub patterns_file: Option<String>,
}

/// Key generator registry
#[derive(Debug)]
pub struct KeyGeneratorRegistry {
    /// Query result key generator
    pub query_results: Box<dyn CacheKeyGenerator<QueryResult> + Send + Sync>,
    
    /// Entity extraction key generator
    pub entity_extraction: Box<dyn CacheKeyGenerator<String> + Send + Sync>,
    
    /// Classification key generator
    pub classification: Box<dyn CacheKeyGenerator<String> + Send + Sync>,
    
    /// Strategy key generator
    pub strategy: Box<dyn CacheKeyGenerator<ProcessedQuery> + Send + Sync>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_entries: 10_000,
            l2_max_entries: 100_000,
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            default_ttl: 3600, // 1 hour
            eviction_batch_size: 100,
            enable_warming: true,
            enable_preloading: true,
            cleanup_interval: Duration::from_secs(60),
            distributed: DistributedCacheConfig::default(),
            enable_metrics: true,
            levels: CacheLevelConfig::default(),
        }
    }
}

impl Default for DistributedCacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            redis_url: None,
            cluster_nodes: Vec::new(),
            replication_factor: 1,
            consistency_level: ConsistencyLevel::Eventual,
            sync_interval: Duration::from_secs(30),
        }
    }
}

impl Default for CacheLevelConfig {
    fn default() -> Self {
        Self {
            query_results: CacheLevelSettings {
                enabled: true,
                max_entries: 5_000,
                ttl: 1800, // 30 minutes
                min_confidence: 0.7,
                enable_compression: true,
            },
            entity_extraction: CacheLevelSettings {
                enabled: true,
                max_entries: 20_000,
                ttl: 7200, // 2 hours
                min_confidence: 0.8,
                enable_compression: false,
            },
            classification: CacheLevelSettings {
                enabled: true,
                max_entries: 15_000,
                ttl: 3600, // 1 hour
                min_confidence: 0.75,
                enable_compression: false,
            },
            strategy: CacheLevelSettings {
                enabled: true,
                max_entries: 8_000,
                ttl: 1800, // 30 minutes
                min_confidence: 0.8,
                enable_compression: true,
            },
            consensus: CacheLevelSettings {
                enabled: true,
                max_entries: 3_000,
                ttl: 900, // 15 minutes
                min_confidence: 0.9,
                enable_compression: true,
            },
            validation: CacheLevelSettings {
                enabled: true,
                max_entries: 5_000,
                ttl: 1200, // 20 minutes
                min_confidence: 0.85,
                enable_compression: false,
            },
        }
    }
}

impl Default for WarmingConfig {
    fn default() -> Self {
        Self {
            warm_on_startup: true,
            warming_interval: Duration::from_secs(300), // 5 minutes
            patterns_per_cycle: 50,
            max_warming_time: Duration::from_secs(30),
            patterns_file: None,
        }
    }
}

impl<T> CacheEntry<T> {
    /// Create a new cache entry
    pub fn new(value: T, ttl: u64, confidence: Option<f64>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        Self {
            value,
            created_at: now,
            ttl,
            access_count: 1,
            last_accessed: now,
            confidence,
            version: 1,
            tags: Vec::new(),
            size_bytes: 0, // Will be calculated externally
        }
    }
    
    /// Create entry with tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    /// Create entry with version
    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }
    
    /// Check if entry has expired
    pub fn is_expired(&self) -> bool {
        if self.ttl == 0 {
            return false; // No expiration
        }
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        now >= self.created_at + self.ttl
    }
    
    /// Update access tracking
    pub fn mark_accessed(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }
    
    /// Get time since last access
    pub fn time_since_access(&self) -> Duration {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
            
        Duration::from_secs(now.saturating_sub(self.last_accessed))
    }
    
    /// Calculate LRU score (higher = less recently used)
    pub fn lru_score(&self) -> u64 {
        let time_weight = self.time_since_access().as_secs();
        let access_weight = if self.access_count > 0 {
            1000 / self.access_count
        } else {
            1000
        };
        
        time_weight + access_weight
    }
}

impl QueryCache {
    /// Create a new query cache with configuration
    pub fn new(config: CacheConfig) -> Self {
        let config = Arc::new(config);
        let l1_cache = Arc::new(DashMap::new());
        let metrics = Arc::new(RwLock::new(CacheMetrics::default()));
        let memory_usage = Arc::new(Mutex::new(0));
        
        // Initialize key generators
        let key_generators = KeyGeneratorRegistry {
            query_results: Box::new(QueryResultKeyGenerator),
            entity_extraction: Box::new(StringKeyGenerator::new("entity")),
            classification: Box::new(StringKeyGenerator::new("classify")),
            strategy: Box::new(StrategyKeyGenerator),
        };
        
        let mut cache = Self {
            l1_cache,
            l2_cache: None,
            config: config.clone(),
            metrics,
            memory_usage,
            cleanup_task: None,
            warming_manager: None,
            key_generators,
        };
        
        // Initialize warming manager if enabled
        if config.enable_warming {
            cache.warming_manager = Some(Arc::new(
                CacheWarmingManager::new(WarmingConfig::default())
            ));
        }
        
        cache
    }
    
    /// Initialize cache with distributed backend
    pub async fn with_distributed_cache(
        mut self,
        distributed_cache: Arc<dyn DistributedCache + Send + Sync>
    ) -> Result<Self> {
        self.l2_cache = Some(distributed_cache);
        Ok(self)
    }
    
    /// Start background tasks (cleanup, warming, etc.)
    pub async fn start_background_tasks(&mut self) -> Result<()> {
        // Start cleanup task
        self.start_cleanup_task().await?;
        
        // Start warming task if enabled
        if let Some(ref warming_manager) = self.warming_manager {
            warming_manager.start_warming_task().await?;
        }
        
        info!("Query cache background tasks started");
        Ok(())
    }
    
    /// Get query result from cache
    #[instrument(skip(self, query))]
    pub async fn get_query_result<T>(&self, query: &QueryResult) -> Result<Option<T>>
    where
        T: DeserializeOwned + Send + 'static,
    {
        let start_time = Instant::now();
        let key = self.key_generators.query_results.generate_key(query);
        
        // Try L1 cache first
        if let Some(entry) = self.get_from_l1(&key).await? {
            let result: T = bincode::deserialize(&entry.value)
                .map_err(|e| QueryProcessorError::cache(format!("Failed to deserialize L1 cache entry: {}", e)))?;
            
            self.record_hit("query_results", start_time.elapsed()).await;
            return Ok(Some(result));
        }
        
        // Try L2 cache if available
        if let Some(ref l2_cache) = self.l2_cache {
            if let Some(data) = l2_cache.get(&key).await? {
                let entry: CacheEntry<Vec<u8>> = bincode::deserialize(&data)
                    .map_err(|e| QueryProcessorError::cache(format!("Failed to deserialize L2 cache entry: {}", e)))?;
                
                if !entry.is_expired() {
                    // Promote to L1 cache
                    self.set_to_l1(key.clone(), entry.clone()).await?;
                    
                    let result: T = bincode::deserialize(&entry.value)
                        .map_err(|e| QueryProcessorError::cache(format!("Failed to deserialize L2 cache result: {}", e)))?;
                    
                    self.record_hit("query_results", start_time.elapsed()).await;
                    return Ok(Some(result));
                }
            }
        }
        
        self.record_miss("query_results", start_time.elapsed()).await;
        Ok(None)
    }
    
    /// Set query result in cache
    #[instrument(skip(self, query, result))]
    pub async fn set_query_result<T>(&self, query: &QueryResult, result: &T, confidence: Option<f64>) -> Result<()>
    where
        T: Serialize + Send,
    {
        let settings = &self.config.levels.query_results;
        if !settings.enabled {
            return Ok(());
        }
        
        // Check confidence threshold
        if let Some(conf) = confidence {
            if conf < settings.min_confidence {
                debug!("Skipping cache set due to low confidence: {}", conf);
                return Ok(());
            }
        }
        
        let key = self.key_generators.query_results.generate_key(query);
        let serialized = bincode::serialize(result)
            .map_err(|e| QueryProcessorError::cache(format!("Failed to serialize query result: {}", e)))?;
        
        let mut entry = CacheEntry::new(serialized, settings.ttl, confidence);
        entry.size_bytes = entry.value.len();
        
        // Set in L1 cache
        self.set_to_l1(key.clone(), entry.clone()).await?;
        
        // Set in L2 cache if available
        if let Some(ref l2_cache) = self.l2_cache {
            let entry_data = bincode::serialize(&entry)
                .map_err(|e| QueryProcessorError::cache(format!("Failed to serialize cache entry: {}", e)))?;
            
            l2_cache.set(&key, entry_data, Duration::from_secs(settings.ttl)).await?;
        }
        
        Ok(())
    }
    
    /// Get entity extraction result from cache
    pub async fn get_entities(&self, query: &str) -> Result<Option<Vec<ExtractedEntity>>> {
        let start_time = Instant::now();
        let key = self.key_generators.entity_extraction.generate_key(query);
        
        if let Some(entry) = self.get_from_l1(&key).await? {
            let entities: Vec<ExtractedEntity> = bincode::deserialize(&entry.value)
                .map_err(|e| QueryProcessorError::cache(format!("Failed to deserialize entities: {}", e)))?;
            
            self.record_hit("entity_extraction", start_time.elapsed()).await;
            return Ok(Some(entities));
        }
        
        self.record_miss("entity_extraction", start_time.elapsed()).await;
        Ok(None)
    }
    
    /// Set entity extraction result in cache
    pub async fn set_entities(&self, query: &str, entities: &[ExtractedEntity], confidence: f64) -> Result<()> {
        let settings = &self.config.levels.entity_extraction;
        if !settings.enabled || confidence < settings.min_confidence {
            return Ok(());
        }
        
        let key = self.key_generators.entity_extraction.generate_key(query);
        let serialized = bincode::serialize(entities)
            .map_err(|e| QueryProcessorError::cache(format!("Failed to serialize entities: {}", e)))?;
        
        let mut entry = CacheEntry::new(serialized, settings.ttl, Some(confidence));
        entry.size_bytes = entry.value.len();
        
        self.set_to_l1(key, entry).await?;
        Ok(())
    }
    
    /// Get classification result from cache
    pub async fn get_classification(&self, query: &str) -> Result<Option<ClassificationResult>> {
        let start_time = Instant::now();
        let key = self.key_generators.classification.generate_key(query);
        
        if let Some(entry) = self.get_from_l1(&key).await? {
            let classification: ClassificationResult = bincode::deserialize(&entry.value)
                .map_err(|e| QueryProcessorError::cache(format!("Failed to deserialize classification: {}", e)))?;
            
            self.record_hit("classification", start_time.elapsed()).await;
            return Ok(Some(classification));
        }
        
        self.record_miss("classification", start_time.elapsed()).await;
        Ok(None)
    }
    
    /// Set classification result in cache
    pub async fn set_classification(&self, query: &str, result: &ClassificationResult) -> Result<()> {
        let settings = &self.config.levels.classification;
        if !settings.enabled || result.confidence < settings.min_confidence {
            return Ok(());
        }
        
        let key = self.key_generators.classification.generate_key(query);
        let serialized = bincode::serialize(result)
            .map_err(|e| QueryProcessorError::cache(format!("Failed to serialize classification: {}", e)))?;
        
        let mut entry = CacheEntry::new(serialized, settings.ttl, Some(result.confidence));
        entry.size_bytes = entry.value.len();
        
        self.set_to_l1(key, entry).await?;
        Ok(())
    }
    
    /// Get strategy recommendation from cache
    pub async fn get_strategy(&self, query: &ProcessedQuery) -> Result<Option<StrategyRecommendation>> {
        let start_time = Instant::now();
        let key = self.key_generators.strategy.generate_key(query);
        
        if let Some(entry) = self.get_from_l1(&key).await? {
            let strategy: StrategyRecommendation = bincode::deserialize(&entry.value)
                .map_err(|e| QueryProcessorError::cache(format!("Failed to deserialize strategy: {}", e)))?;
            
            self.record_hit("strategy", start_time.elapsed()).await;
            return Ok(Some(strategy));
        }
        
        self.record_miss("strategy", start_time.elapsed()).await;
        Ok(None)
    }
    
    /// Set strategy recommendation in cache
    pub async fn set_strategy(&self, query: &ProcessedQuery, strategy: &StrategyRecommendation) -> Result<()> {
        let settings = &self.config.levels.strategy;
        if !settings.enabled || strategy.confidence < settings.min_confidence {
            return Ok(());
        }
        
        let key = self.key_generators.strategy.generate_key(query);
        let serialized = bincode::serialize(strategy)
            .map_err(|e| QueryProcessorError::cache(format!("Failed to serialize strategy: {}", e)))?;
        
        let mut entry = CacheEntry::new(serialized, settings.ttl, Some(strategy.confidence));
        entry.size_bytes = entry.value.len();
        
        self.set_to_l1(key, entry).await?;
        Ok(())
    }
    
    /// Invalidate entries by tag
    pub async fn invalidate_by_tag(&self, tag: &str) -> Result<u64> {
        let mut removed_count = 0;
        
        // Collect keys to remove
        let keys_to_remove: Vec<String> = self.l1_cache.iter()
            .filter_map(|entry| {
                let cache_entry: Result<CacheEntry<Vec<u8>>, _> = bincode::deserialize(entry.value());
                if let Ok(cached) = cache_entry {
                    if cached.tags.contains(&tag.to_string()) {
                        return Some(entry.key().clone());
                    }
                }
                None
            })
            .collect();
        
        // Remove entries
        for key in keys_to_remove {
            if self.l1_cache.remove(&key).is_some() {
                removed_count += 1;
                
                // Also remove from L2 cache if available
                if let Some(ref l2_cache) = self.l2_cache {
                    l2_cache.remove(&key).await?;
                }
            }
        }
        
        info!("Invalidated {} cache entries with tag: {}", removed_count, tag);
        Ok(removed_count)
    }
    
    /// Clear all cache entries
    pub async fn clear(&self) -> Result<()> {
        self.l1_cache.clear();
        
        if let Some(ref l2_cache) = self.l2_cache {
            l2_cache.clear().await?;
        }
        
        // Reset memory usage
        *self.memory_usage.lock().await = 0;
        
        info!("Cache cleared");
        Ok(())
    }
    
    /// Get cache metrics
    pub async fn get_metrics(&self) -> CacheMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get from L1 cache with access tracking
    async fn get_from_l1(&self, key: &str) -> Result<Option<CacheEntry<Vec<u8>>>> {
        if let Some(mut entry_ref) = self.l1_cache.get_mut(key) {
            let mut entry: CacheEntry<Vec<u8>> = bincode::deserialize(entry_ref.value())
                .map_err(|e| QueryProcessorError::cache(format!("Failed to deserialize L1 entry: {}", e)))?;
            
            if entry.is_expired() {
                drop(entry_ref);
                self.l1_cache.remove(key);
                self.update_memory_usage_subtract(entry.size_bytes).await;
                return Ok(None);
            }
            
            entry.mark_accessed();
            let serialized = bincode::serialize(&entry)
                .map_err(|e| QueryProcessorError::cache(format!("Failed to serialize updated entry: {}", e)))?;
            *entry_ref.value_mut() = serialized;
            
            return Ok(Some(entry));
        }
        
        Ok(None)
    }
    
    /// Set to L1 cache with eviction
    async fn set_to_l1(&self, key: String, entry: CacheEntry<Vec<u8>>) -> Result<()> {
        // Check if we need to evict entries
        if self.l1_cache.len() >= self.config.l1_max_entries {
            self.evict_lru_entries().await?;
        }
        
        let serialized = bincode::serialize(&entry)
            .map_err(|e| QueryProcessorError::cache(format!("Failed to serialize cache entry: {}", e)))?;
        
        self.l1_cache.insert(key, serialized);
        self.update_memory_usage_add(entry.size_bytes).await;
        
        Ok(())
    }
    
    /// Evict LRU entries from L1 cache
    async fn evict_lru_entries(&self) -> Result<()> {
        let mut entries_to_evict = Vec::new();
        
        // Collect entries with their LRU scores
        for entry_ref in self.l1_cache.iter() {
            if let Ok(entry) = bincode::deserialize::<CacheEntry<Vec<u8>>>(entry_ref.value()) {
                entries_to_evict.push((entry_ref.key().clone(), entry.lru_score(), entry.size_bytes));
            }
        }
        
        // Sort by LRU score (highest = least recently used)
        entries_to_evict.sort_by(|a, b| b.1.cmp(&a.1));
        
        // Evict batch
        let evict_count = std::cmp::min(self.config.eviction_batch_size, entries_to_evict.len());
        for (key, _, size) in entries_to_evict.iter().take(evict_count) {
            self.l1_cache.remove(key);
            self.update_memory_usage_subtract(*size).await;
        }
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.evictions.lru_evictions += evict_count as u64;
        metrics.evictions.total_evictions += evict_count as u64;
        
        debug!("Evicted {} LRU entries from L1 cache", evict_count);
        Ok(())
    }
    
    /// Update memory usage (add)
    async fn update_memory_usage_add(&self, size: usize) {
        let mut usage = self.memory_usage.lock().await;
        *usage += size;
    }
    
    /// Update memory usage (subtract)
    async fn update_memory_usage_subtract(&self, size: usize) {
        let mut usage = self.memory_usage.lock().await;
        *usage = usage.saturating_sub(size);
    }
    
    /// Record cache hit
    async fn record_hit(&self, level: &str, access_time: Duration) {
        let mut metrics = self.metrics.write().await;
        metrics.hits += 1;
        metrics.hit_ratio = metrics.hits as f64 / (metrics.hits + metrics.misses) as f64;
        
        let level_metrics = metrics.level_metrics.entry(level.to_string()).or_insert_with(LevelMetrics::default);
        level_metrics.hits += 1;
        level_metrics.avg_access_time_us = access_time.as_micros() as u64;
    }
    
    /// Record cache miss
    async fn record_miss(&self, level: &str, access_time: Duration) {
        let mut metrics = self.metrics.write().await;
        metrics.misses += 1;
        metrics.hit_ratio = metrics.hits as f64 / (metrics.hits + metrics.misses) as f64;
        
        let level_metrics = metrics.level_metrics.entry(level.to_string()).or_insert_with(LevelMetrics::default);
        level_metrics.misses += 1;
        level_metrics.avg_access_time_us = access_time.as_micros() as u64;
    }
    
    /// Start cleanup task for expired entries
    async fn start_cleanup_task(&mut self) -> Result<()> {
        let l1_cache = self.l1_cache.clone();
        let memory_usage = self.memory_usage.clone();
        let metrics = self.metrics.clone();
        let interval_duration = self.config.cleanup_interval;
        
        let cleanup_task = tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            
            loop {
                interval.tick().await;
                
                // Clean expired entries
                let mut expired_keys = Vec::new();
                let mut total_size_freed = 0;
                
                for entry_ref in l1_cache.iter() {
                    if let Ok(entry) = bincode::deserialize::<CacheEntry<Vec<u8>>>(entry_ref.value()) {
                        if entry.is_expired() {
                            expired_keys.push(entry_ref.key().clone());
                            total_size_freed += entry.size_bytes;
                        }
                    }
                }
                
                // Remove expired entries
                for key in &expired_keys {
                    l1_cache.remove(key);
                }
                
                // Update memory usage
                if total_size_freed > 0 {
                    let mut usage = memory_usage.lock().await;
                    *usage = usage.saturating_sub(total_size_freed);
                }
                
                // Update metrics
                if !expired_keys.is_empty() {
                    let mut metrics_guard = metrics.write().await;
                    metrics_guard.evictions.ttl_evictions += expired_keys.len() as u64;
                    metrics_guard.evictions.total_evictions += expired_keys.len() as u64;
                    
                    debug!("Cleaned {} expired cache entries, freed {} bytes", 
                           expired_keys.len(), total_size_freed);
                }
            }
        });
        
        self.cleanup_task = Some(cleanup_task);
        Ok(())
    }
}

impl CacheWarmingManager {
    /// Create a new cache warming manager
    pub fn new(config: WarmingConfig) -> Self {
        Self {
            warming_patterns: Arc::new(RwLock::new(Vec::new())),
            scheduler: Arc::new(Mutex::new(None)),
            config,
        }
    }
    
    /// Start warming task
    pub async fn start_warming_task(&self) -> Result<()> {
        let patterns = self.warming_patterns.clone();
        let interval_duration = self.config.warming_interval;
        let patterns_per_cycle = self.config.patterns_per_cycle;
        
        let warming_task = tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            
            loop {
                interval.tick().await;
                
                let patterns_guard = patterns.read().await;
                let pattern_count = std::cmp::min(patterns_per_cycle, patterns_guard.len());
                
                if pattern_count > 0 {
                    debug!("Warming cache with {} patterns", pattern_count);
                    // Cache warming logic would go here
                }
            }
        });
        
        *self.scheduler.lock().await = Some(warming_task);
        Ok(())
    }
    
    /// Add warming pattern
    pub async fn add_pattern(&self, pattern: String) -> Result<()> {
        self.warming_patterns.write().await.push(pattern);
        Ok(())
    }
}

// Key generator implementations
pub struct QueryResultKeyGenerator;

impl CacheKeyGenerator<QueryResult> for QueryResultKeyGenerator {
    fn generate_key(&self, result: &QueryResult) -> String {
        // Generate key based on query content and parameters
        format!("qr:{}:{}", 
                hash_string(&result.query), 
                hash_string(&format!("{:?}", result.search_strategy)))
    }
}

pub struct StringKeyGenerator {
    prefix: String,
}

impl StringKeyGenerator {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }
}

impl CacheKeyGenerator<String> for StringKeyGenerator {
    fn generate_key(&self, item: &String) -> String {
        format!("{}:{}", self.prefix, hash_string(item))
    }
}

pub struct StrategyKeyGenerator;

impl CacheKeyGenerator<ProcessedQuery> for StrategyKeyGenerator {
    fn generate_key(&self, query: &ProcessedQuery) -> String {
        format!("strat:{}:{}:{}", 
                hash_string(&query.original_query),
                hash_string(&format!("{:?}", query.intent)),
                hash_string(&format!("{:?}", query.entities)))
    }
}

// Utility function for consistent hashing
fn hash_string(input: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

impl Drop for QueryCache {
    fn drop(&mut self) {
        // Cleanup background tasks
        if let Some(cleanup_task) = self.cleanup_task.take() {
            cleanup_task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{QueryIntent, SearchStrategy};
    
    #[tokio::test]
    async fn test_cache_entry_creation() {
        let entry = CacheEntry::new("test_value".to_string(), 3600, Some(0.8));
        assert!(!entry.is_expired());
        assert_eq!(entry.confidence, Some(0.8));
        assert_eq!(entry.access_count, 1);
    }
    
    #[tokio::test]
    async fn test_cache_entry_expiration() {
        let mut entry = CacheEntry::new("test_value".to_string(), 0, Some(0.8)); // No expiration
        assert!(!entry.is_expired());
        
        entry.ttl = 1; // 1 second TTL
        entry.created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() - 2; // Created 2 seconds ago
            
        assert!(entry.is_expired());
    }
    
    #[tokio::test]
    async fn test_cache_basic_operations() {
        let config = CacheConfig::default();
        let cache = QueryCache::new(config);
        
        // Test entity caching
        let entities = vec![
            ExtractedEntity {
                text: "test entity".to_string(),
                entity_type: "TEST".to_string(),
                confidence: 0.9,
                position: (0, 11),
                metadata: HashMap::new(),
                relationships: Vec::new(),
            }
        ];
        
        cache.set_entities("test query", &entities, 0.9).await.unwrap();
        let cached_entities = cache.get_entities("test query").await.unwrap();
        
        assert!(cached_entities.is_some());
        let retrieved = cached_entities.unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].text, "test entity");
    }
    
    #[tokio::test]
    async fn test_lru_eviction() {
        let mut config = CacheConfig::default();
        config.l1_max_entries = 2; // Small cache for testing
        config.eviction_batch_size = 1;
        
        let cache = QueryCache::new(config);
        
        // Fill cache beyond capacity
        let entities1 = vec![ExtractedEntity {
            text: "entity1".to_string(),
            entity_type: "TEST".to_string(),
            confidence: 0.9,
            position: (0, 7),
            metadata: HashMap::new(),
            relationships: Vec::new(),
        }];
        
        let entities2 = vec![ExtractedEntity {
            text: "entity2".to_string(),
            entity_type: "TEST".to_string(),
            confidence: 0.9,
            position: (0, 7),
            metadata: HashMap::new(),
            relationships: Vec::new(),
        }];
        
        let entities3 = vec![ExtractedEntity {
            text: "entity3".to_string(),
            entity_type: "TEST".to_string(),
            confidence: 0.9,
            position: (0, 7),
            metadata: HashMap::new(),
            relationships: Vec::new(),
        }];
        
        cache.set_entities("query1", &entities1, 0.9).await.unwrap();
        cache.set_entities("query2", &entities2, 0.9).await.unwrap();
        
        // This should trigger eviction
        cache.set_entities("query3", &entities3, 0.9).await.unwrap();
        
        // Verify eviction occurred
        let metrics = cache.get_metrics().await;
        assert!(metrics.evictions.total_evictions > 0);
    }
    
    #[tokio::test]
    async fn test_confidence_filtering() {
        let mut config = CacheConfig::default();
        config.levels.entity_extraction.min_confidence = 0.8;
        
        let cache = QueryCache::new(config);
        
        let entities = vec![ExtractedEntity {
            text: "low confidence entity".to_string(),
            entity_type: "TEST".to_string(),
            confidence: 0.5,
            position: (0, 20),
            metadata: HashMap::new(),
            relationships: Vec::new(),
        }];
        
        // Should not cache due to low confidence
        cache.set_entities("test query", &entities, 0.5).await.unwrap();
        let cached = cache.get_entities("test query").await.unwrap();
        assert!(cached.is_none());
        
        // Should cache with high confidence
        cache.set_entities("test query", &entities, 0.9).await.unwrap();
        let cached = cache.get_entities("test query").await.unwrap();
        assert!(cached.is_some());
    }
    
    #[test]
    fn test_key_generation() {
        let query_gen = QueryResultKeyGenerator;
        let result = QueryResult {
            query: "test query".to_string(),
            search_strategy: SearchStrategy::Hybrid,
            confidence: 0.8,
            processing_time: Duration::from_millis(10),
            metadata: HashMap::new(),
        };
        
        let key1 = query_gen.generate_key(&result);
        let key2 = query_gen.generate_key(&result);
        assert_eq!(key1, key2); // Keys should be consistent
        
        let versioned_key = query_gen.generate_versioned_key(&result, 1);
        assert!(versioned_key.contains(":v1"));
    }
    
    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::default();
        assert_eq!(config.l1_max_entries, 10_000);
        assert_eq!(config.default_ttl, 3600);
        assert!(config.enable_warming);
        assert!(config.enable_metrics);
    }
    
    #[tokio::test]
    async fn test_tag_invalidation() {
        let cache = QueryCache::new(CacheConfig::default());
        
        // This would require implementing tag support in set operations
        // For now, just test the invalidation method exists
        let removed = cache.invalidate_by_tag("test_tag").await.unwrap();
        assert_eq!(removed, 0); // No entries with tag should exist
    }
}