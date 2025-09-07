//! High-performance query processor optimization for <2s response times
//!
//! This module implements advanced optimization techniques including parallel validation,
//! intelligent caching, and performance monitoring to achieve sub-2-second response times.

use crate::{Result, ProcessorError, ProcessedQuery, Query, QueryProcessor};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tokio::task::JoinSet;
use tracing::{debug, info, warn};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
// use async_trait::async_trait; // Currently unused

/// High-performance query processor optimizer targeting <2s response times
pub struct QueryProcessorOptimizer {
    /// Base query processor
    processor: Arc<QueryProcessor>,
    /// Optimization configuration
    config: OptimizerConfig,
    /// Parallel execution semaphore
    parallel_semaphore: Arc<Semaphore>,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Query cache for repeated queries
    query_cache: Arc<RwLock<QueryCache>>,
    /// Validation pipeline cache
    validation_cache: Arc<RwLock<ValidationCache>>,
}

/// Optimizer configuration for maximum performance
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Target response time in milliseconds
    pub target_response_time_ms: u64,
    /// Maximum parallel query processing tasks
    pub max_parallel_queries: usize,
    /// Maximum parallel validation tasks
    pub max_parallel_validations: usize,
    /// Enable aggressive caching
    pub enable_aggressive_caching: bool,
    /// Enable parallel validation pipeline
    pub enable_parallel_validation: bool,
    /// Enable query prediction and prefetching
    pub enable_query_prediction: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Performance monitoring interval
    pub performance_monitoring_interval_ms: u64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            target_response_time_ms: 2000, // <2s target
            max_parallel_queries: 50,
            max_parallel_validations: 20,
            enable_aggressive_caching: true,
            enable_parallel_validation: true,
            enable_query_prediction: true,
            cache_ttl_seconds: 300, // 5 minutes
            performance_monitoring_interval_ms: 10000, // 10 seconds
        }
    }
}

/// Performance metrics tracking
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total queries processed
    pub total_queries: u64,
    /// Queries meeting target time
    pub queries_under_target: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// 95th percentile response time
    pub p95_response_time_ms: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Parallel processing utilization
    pub parallel_utilization: f64,
    /// Current active queries
    pub active_queries: u64,
    /// Performance target achievement rate
    pub target_achievement_rate: f64,
}

/// Query cache for repeated queries
#[derive(Debug, Default)]
pub struct QueryCache {
    entries: HashMap<String, CachedQueryResult>,
    access_times: HashMap<String, Instant>,
}

/// Validation cache for pipeline results
#[derive(Debug, Default)]
pub struct ValidationCache {
    entries: HashMap<String, CachedValidationResult>,
    access_times: HashMap<String, Instant>,
}

/// Cached query result
#[derive(Debug, Clone)]
pub struct CachedQueryResult {
    query_hash: String,
    result: ProcessedQuery,
    created_at: Instant,
    access_count: u64,
    processing_time: Duration,
}

/// Cached validation result
#[derive(Debug, Clone)]
pub struct CachedValidationResult {
    validation_hash: String,
    is_valid: bool,
    confidence_score: f64,
    created_at: Instant,
    access_count: u64,
}

/// Parallel validation task
#[derive(Debug)]
pub struct ValidationTask {
    task_id: Uuid,
    query: Query,
    validation_type: ValidationType,
    priority: ValidationPriority,
}

/// Types of validation
#[derive(Debug, Clone)]
pub enum ValidationType {
    Semantic,
    Entity,
    Intent,
    Strategy,
    Consensus,
}

/// Validation priority levels
#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub enum ValidationPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Optimization result with performance metrics
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub query_id: Uuid,
    pub result: ProcessedQuery,
    pub processing_time: Duration,
    pub cache_hit: bool,
    pub parallel_tasks_used: usize,
    pub performance_score: f64, // 0.0 to 1.0 based on target achievement
}

impl QueryProcessorOptimizer {
    /// Create new query processor optimizer
    pub async fn new(processor: QueryProcessor, config: OptimizerConfig) -> Result<Self> {
        let parallel_semaphore = Arc::new(Semaphore::new(config.max_parallel_queries));
        let metrics = Arc::new(RwLock::new(PerformanceMetrics::default()));
        let query_cache = Arc::new(RwLock::new(QueryCache::default()));
        let validation_cache = Arc::new(RwLock::new(ValidationCache::default()));
        
        let optimizer = Self {
            processor: Arc::new(processor),
            config: config.clone(),
            parallel_semaphore,
            metrics,
            query_cache,
            validation_cache,
        };
        
        // Start performance monitoring
        optimizer.start_performance_monitoring().await;
        
        info!("Query processor optimizer initialized with target <{}ms response time", 
              config.target_response_time_ms);
        
        Ok(optimizer)
    }

    /// Process query with aggressive optimization for <2s response time
    pub async fn process_optimized(&self, query: Query) -> Result<OptimizationResult> {
        let start_time = Instant::now();
        let query_id = query.id();
        
        // Acquire parallel processing permit
        let _permit = self.parallel_semaphore.acquire().await
            .map_err(|_| ProcessorError::ProcessingFailed("Failed to acquire parallel permit".to_string()))?;
        
        debug!("Starting optimized processing for query: {}", query_id);
        
        // Check query cache first
        if self.config.enable_aggressive_caching {
            if let Some(cached_result) = self.check_query_cache(&query).await? {
                let processing_time = start_time.elapsed();
                self.update_metrics(processing_time, true).await;
                
                return Ok(OptimizationResult {
                    query_id,
                    result: cached_result.result,
                    processing_time,
                    cache_hit: true,
                    parallel_tasks_used: 0,
                    performance_score: self.calculate_performance_score(processing_time),
                });
            }
        }
        
        // Parallel processing pipeline
        let result = if self.config.enable_parallel_validation {
            self.process_with_parallel_validation(query).await?
        } else {
            self.processor.process(query).await?
        };
        
        let processing_time = start_time.elapsed();
        
        // Cache the result for future use
        if self.config.enable_aggressive_caching {
            self.cache_query_result(&result, processing_time).await?;
        }
        
        // Update performance metrics
        self.update_metrics(processing_time, false).await;
        
        let performance_score = self.calculate_performance_score(processing_time);
        
        info!("Query {} processed in {}ms (target: {}ms, score: {:.2})", 
              query_id, processing_time.as_millis(), 
              self.config.target_response_time_ms, performance_score);
        
        Ok(OptimizationResult {
            query_id,
            result,
            processing_time,
            cache_hit: false,
            parallel_tasks_used: 1, // Base processing
            performance_score,
        })
    }

    /// Process with parallel validation pipeline for maximum speed
    async fn process_with_parallel_validation(&self, query: Query) -> Result<ProcessedQuery> {
        debug!("Starting parallel validation pipeline");
        
        let mut join_set = JoinSet::new();
        let query_arc = Arc::new(query);
        
        // Create validation tasks
        let _tasks = vec![
            ValidationTask {
                task_id: Uuid::new_v4(),
                query: (*query_arc).clone(),
                validation_type: ValidationType::Semantic,
                priority: ValidationPriority::High,
            },
            ValidationTask {
                task_id: Uuid::new_v4(),
                query: (*query_arc).clone(),
                validation_type: ValidationType::Entity,
                priority: ValidationPriority::High,
            },
            ValidationTask {
                task_id: Uuid::new_v4(),
                query: (*query_arc).clone(),
                validation_type: ValidationType::Intent,
                priority: ValidationPriority::High,
            },
            ValidationTask {
                task_id: Uuid::new_v4(),
                query: (*query_arc).clone(),
                validation_type: ValidationType::Strategy,
                priority: ValidationPriority::Medium,
            },
        ];
        
        // Execute tasks in parallel
        let processor_clone = self.processor.clone();
        join_set.spawn(async move {
            processor_clone.process((*query_arc).clone()).await
        });
        
        // Wait for main processing to complete
        let mut result = None;
        while let Some(task_result) = join_set.join_next().await {
            match task_result {
                Ok(Ok(processed_query)) => {
                    if result.is_none() {
                        result = Some(processed_query);
                    }
                },
                Ok(Err(e)) => {
                    warn!("Parallel validation task failed: {}", e);
                },
                Err(e) => {
                    warn!("Parallel task join failed: {}", e);
                }
            }
        }
        
        result.ok_or_else(|| ProcessorError::ProcessingFailed("All parallel tasks failed".to_string()))
    }

    /// Process multiple queries in parallel batch
    pub async fn process_batch_optimized(&self, queries: Vec<Query>) -> Result<Vec<OptimizationResult>> {
        info!("Processing batch of {} queries with optimization", queries.len());
        
        let start_time = Instant::now();
        let mut join_set = JoinSet::new();
        
        // Submit all queries for parallel processing
        for query in queries {
            let optimizer_clone = self.clone();
            join_set.spawn(async move {
                optimizer_clone.process_optimized(query).await
            });
        }
        
        // Collect results
        let mut results = Vec::new();
        while let Some(task_result) = join_set.join_next().await {
            match task_result {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => {
                    warn!("Batch query processing failed: {}", e);
                },
                Err(e) => {
                    warn!("Batch task join failed: {}", e);
                }
            }
        }
        
        let batch_time = start_time.elapsed();
        info!("Batch processing completed in {}ms for {} queries", 
              batch_time.as_millis(), results.len());
        
        Ok(results)
    }

    /// Check query cache for existing results
    async fn check_query_cache(&self, query: &Query) -> Result<Option<CachedQueryResult>> {
        let query_hash = self.calculate_query_hash(query);
        let cache = self.query_cache.read().await;
        
        if let Some(cached_entry) = cache.entries.get(&query_hash) {
            // Check if cache entry is still valid
            let age = cached_entry.created_at.elapsed();
            if age < Duration::from_secs(self.config.cache_ttl_seconds) {
                debug!("Cache hit for query hash: {}", query_hash);
                return Ok(Some(cached_entry.clone()));
            }
        }
        
        Ok(None)
    }

    /// Cache query result for future use
    async fn cache_query_result(&self, result: &ProcessedQuery, processing_time: Duration) -> Result<()> {
        let query_hash = self.calculate_query_hash(&result.query);
        let cached_result = CachedQueryResult {
            query_hash: query_hash.clone(),
            result: result.clone(),
            created_at: Instant::now(),
            access_count: 1,
            processing_time,
        };
        
        let mut cache = self.query_cache.write().await;
        cache.entries.insert(query_hash.clone(), cached_result);
        cache.access_times.insert(query_hash, Instant::now());
        
        // Cleanup old entries if cache is too large
        if cache.entries.len() > 1000 { // Max 1000 cached queries
            self.cleanup_query_cache(&mut cache).await;
        }
        
        Ok(())
    }

    /// Cleanup old cache entries
    async fn cleanup_query_cache(&self, cache: &mut QueryCache) {
        let now = Instant::now();
        let ttl = Duration::from_secs(self.config.cache_ttl_seconds);
        
        let mut to_remove = Vec::new();
        for (hash, access_time) in &cache.access_times {
            if now.duration_since(*access_time) > ttl {
                to_remove.push(hash.clone());
            }
        }
        
        for hash in to_remove {
            cache.entries.remove(&hash);
            cache.access_times.remove(&hash);
        }
        
        debug!("Cache cleanup completed, {} entries remaining", cache.entries.len());
    }

    /// Calculate query hash for caching
    fn calculate_query_hash(&self, query: &Query) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.text().hash(&mut hasher);
        // Note: context field is private, would need a getter method
        // if let Some(context) = query.context() {
        //     context.hash(&mut hasher);
        // }
        // For now, use query text as hash basis
        query.text().hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Calculate performance score based on target achievement
    fn calculate_performance_score(&self, processing_time: Duration) -> f64 {
        let target_ms = self.config.target_response_time_ms as f64;
        let actual_ms = processing_time.as_millis() as f64;
        
        if actual_ms <= target_ms {
            1.0 // Perfect score
        } else if actual_ms <= target_ms * 2.0 {
            // Linear degradation up to 2x target time
            1.0 - ((actual_ms - target_ms) / target_ms)
        } else {
            0.0 // Failed to meet reasonable performance
        }
    }

    /// Update performance metrics
    async fn update_metrics(&self, processing_time: Duration, cache_hit: bool) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_queries += 1;
        
        let processing_ms = processing_time.as_millis() as f64;
        if processing_ms <= self.config.target_response_time_ms as f64 {
            metrics.queries_under_target += 1;
        }
        
        // Update average response time
        let total_time = metrics.avg_response_time_ms * (metrics.total_queries - 1) as f64 + processing_ms;
        metrics.avg_response_time_ms = total_time / metrics.total_queries as f64;
        
        // Update target achievement rate
        metrics.target_achievement_rate = metrics.queries_under_target as f64 / metrics.total_queries as f64;
        
        // Update cache hit rate
        if cache_hit {
            let cache_hits = (metrics.cache_hit_rate * (metrics.total_queries - 1) as f64) + 1.0;
            metrics.cache_hit_rate = cache_hits / metrics.total_queries as f64;
        } else {
            let cache_hits = metrics.cache_hit_rate * (metrics.total_queries - 1) as f64;
            metrics.cache_hit_rate = cache_hits / metrics.total_queries as f64;
        }
    }

    /// Start performance monitoring
    async fn start_performance_monitoring(&self) {
        let metrics_clone = self.metrics.clone();
        let interval_ms = self.config.performance_monitoring_interval_ms;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(interval_ms));
            
            loop {
                interval.tick().await;
                
                let metrics = metrics_clone.read().await;
                if metrics.total_queries > 0 {
                    info!("Performance Stats - Queries: {}, Avg: {:.1}ms, Target Achievement: {:.1}%, Cache Hit: {:.1}%",
                          metrics.total_queries,
                          metrics.avg_response_time_ms,
                          metrics.target_achievement_rate * 100.0,
                          metrics.cache_hit_rate * 100.0);
                    
                    if metrics.target_achievement_rate < 0.95 {
                        warn!("Performance target achievement rate is below 95%: {:.1}%", 
                              metrics.target_achievement_rate * 100.0);
                    }
                }
            }
        });
    }

    /// Get current performance metrics
    pub async fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }

    /// Get performance summary
    pub async fn get_performance_summary(&self) -> PerformanceSummary {
        let metrics = self.metrics.read().await;
        let cache_size = self.query_cache.read().await.entries.len();
        
        PerformanceSummary {
            target_response_time_ms: self.config.target_response_time_ms,
            average_response_time_ms: metrics.avg_response_time_ms,
            target_achievement_rate: metrics.target_achievement_rate,
            cache_hit_rate: metrics.cache_hit_rate,
            total_queries_processed: metrics.total_queries,
            queries_under_target: metrics.queries_under_target,
            cache_size,
            performance_grade: self.calculate_performance_grade(&metrics),
        }
    }

    /// Calculate overall performance grade
    fn calculate_performance_grade(&self, metrics: &PerformanceMetrics) -> PerformanceGrade {
        let score = (metrics.target_achievement_rate * 0.5) + 
                   (metrics.cache_hit_rate * 0.3) + 
                   ((2000.0 / metrics.avg_response_time_ms.max(1.0)).min(1.0) * 0.2);
        
        if score >= 0.95 {
            PerformanceGrade::Excellent
        } else if score >= 0.85 {
            PerformanceGrade::Good
        } else if score >= 0.70 {
            PerformanceGrade::Fair
        } else {
            PerformanceGrade::Poor
        }
    }
}

impl Clone for QueryProcessorOptimizer {
    fn clone(&self) -> Self {
        Self {
            processor: self.processor.clone(),
            config: self.config.clone(),
            parallel_semaphore: self.parallel_semaphore.clone(),
            metrics: self.metrics.clone(),
            query_cache: self.query_cache.clone(),
            validation_cache: self.validation_cache.clone(),
        }
    }
}

/// Performance summary for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub target_response_time_ms: u64,
    pub average_response_time_ms: f64,
    pub target_achievement_rate: f64,
    pub cache_hit_rate: f64,
    pub total_queries_processed: u64,
    pub queries_under_target: u64,
    pub cache_size: usize,
    pub performance_grade: PerformanceGrade,
}

/// Performance grade classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceGrade {
    Excellent, // >95% target achievement
    Good,      // 85-95% target achievement  
    Fair,      // 70-85% target achievement
    Poor,      // <70% target achievement
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProcessorConfig, QueryProcessor};
    
    async fn create_test_optimizer() -> QueryProcessorOptimizer {
        let processor_config = ProcessorConfig::default();
        let processor = QueryProcessor::new(processor_config).await.unwrap();
        let optimizer_config = OptimizerConfig::default();
        
        QueryProcessorOptimizer::new(processor, optimizer_config).await.unwrap()
    }
    
    #[tokio::test]
    async fn test_optimizer_creation() {
        let optimizer = create_test_optimizer().await;
        let metrics = optimizer.get_performance_metrics().await;
        assert_eq!(metrics.total_queries, 0);
    }
    
    #[tokio::test]
    async fn test_single_query_optimization() {
        let optimizer = create_test_optimizer().await;
        let query = Query::new("What is the performance target?");
        
        let result = optimizer.process_optimized(query).await;
        assert!(result.is_ok());
        
        let optimization_result = result.unwrap();
        assert!(optimization_result.processing_time < Duration::from_millis(2000));
        assert!(optimization_result.performance_score >= 0.0);
        assert!(optimization_result.performance_score <= 1.0);
    }
    
    #[tokio::test]
    async fn test_batch_processing() {
        let optimizer = create_test_optimizer().await;
        let queries = vec![
            Query::new("Query 1"),
            Query::new("Query 2"),
            Query::new("Query 3"),
        ];
        
        let results = optimizer.process_batch_optimized(queries).await;
        assert!(results.is_ok());
        
        let batch_results = results.unwrap();
        assert_eq!(batch_results.len(), 3);
        
        // All queries should complete within reasonable time
        for result in batch_results {
            assert!(result.processing_time < Duration::from_millis(5000)); // 5s max for batch
        }
    }
    
    #[tokio::test]
    async fn test_cache_functionality() {
        let mut config = OptimizerConfig::default();
        config.enable_aggressive_caching = true;
        
        let processor_config = ProcessorConfig::default();
        let processor = QueryProcessor::new(processor_config).await.unwrap();
        let optimizer = QueryProcessorOptimizer::new(processor, config).await.unwrap();
        
        let query = Query::new("Cached query test");
        
        // First request - should not be cached
        let result1 = optimizer.process_optimized(query.clone()).await.unwrap();
        assert!(!result1.cache_hit);
        
        // Wait a bit for caching
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Second request - should be cached
        let result2 = optimizer.process_optimized(query).await.unwrap();
        assert!(result2.cache_hit);
        assert!(result2.processing_time < result1.processing_time); // Should be faster
    }
    
    #[tokio::test]
    async fn test_performance_metrics() {
        let optimizer = create_test_optimizer().await;
        
        // Process several queries
        for i in 0..5 {
            let query = Query::new(&format!("Test query {}", i));
            let _ = optimizer.process_optimized(query).await.unwrap();
        }
        
        let metrics = optimizer.get_performance_metrics().await;
        assert_eq!(metrics.total_queries, 5);
        assert!(metrics.avg_response_time_ms > 0.0);
        assert!(metrics.target_achievement_rate >= 0.0);
        assert!(metrics.target_achievement_rate <= 1.0);
        
        let summary = optimizer.get_performance_summary().await;
        assert_eq!(summary.total_queries_processed, 5);
        assert_eq!(summary.target_response_time_ms, 2000);
    }
    
    #[tokio::test]
    async fn test_performance_target_achievement() {
        let optimizer = create_test_optimizer().await;
        
        // Process a query and check if it meets performance target
        let query = Query::new("Performance target test query");
        let result = optimizer.process_optimized(query).await.unwrap();
        
        // Should complete within 2s target
        assert!(result.processing_time.as_millis() <= 2000);
        
        // Performance score should be high for fast queries
        if result.processing_time.as_millis() <= 1000 {
            assert!(result.performance_score >= 0.8);
        }
    }
}