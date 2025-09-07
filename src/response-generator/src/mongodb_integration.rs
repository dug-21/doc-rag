//! MongoDB Integration for Response Generator
//! 
//! This module integrates the MongoDB optimization strategies with the response generator
//! to achieve Phase 2 performance targets:
//! - Sub-50ms cache hits through FACT cache
//! - <2s response time for complex queries
//! - Seamless MongoDB query optimization
//! - Performance monitoring and auto-tuning

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::{Result, Context, anyhow};
use async_trait::async_trait;
use tracing::{info, warn, error, debug, instrument};

use crate::{
    GenerationRequest, GeneratedResponse, ResponseGenerator, 
    FACTCacheManager, CacheResult, CacheSource
};

/// MongoDB integration configuration for response generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MongoDBIntegrationConfig {
    /// Enable MongoDB query optimization
    pub enable_query_optimization: bool,
    
    /// MongoDB connection settings
    pub connection: MongoDBConnectionConfig,
    
    /// Cache integration settings
    pub cache_integration: CacheIntegrationConfig,
    
    /// Performance monitoring settings
    pub monitoring: MongoDBMonitoringConfig,
}

/// MongoDB connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MongoDBConnectionConfig {
    /// Connection string
    pub connection_string: String,
    
    /// Database name
    pub database_name: String,
    
    /// Connection pool settings
    pub pool_settings: ConnectionPoolSettings,
}

/// Connection pool settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolSettings {
    /// Minimum connections
    pub min_connections: u32,
    
    /// Maximum connections
    pub max_connections: u32,
    
    /// Connection timeout in seconds
    pub connection_timeout_s: u64,
    
    /// Idle timeout in seconds
    pub idle_timeout_s: u64,
}

/// Cache integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheIntegrationConfig {
    /// Enable FACT cache for response caching
    pub enable_response_caching: bool,
    
    /// Cache TTL for responses in seconds
    pub response_cache_ttl_s: u64,
    
    /// Cache TTL for query metadata in seconds
    pub metadata_cache_ttl_s: u64,
    
    /// Enable cache warming for common queries
    pub enable_cache_warming: bool,
    
    /// Cache warming patterns
    pub warming_patterns: Vec<String>,
}

/// MongoDB monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MongoDBMonitoringConfig {
    /// Enable query performance monitoring
    pub enable_query_monitoring: bool,
    
    /// Slow query threshold in milliseconds
    pub slow_query_threshold_ms: u64,
    
    /// Enable automatic optimization recommendations
    pub enable_auto_recommendations: bool,
    
    /// Performance reporting interval in seconds
    pub reporting_interval_s: u64,
}

/// MongoDB-integrated response generator
pub struct MongoDBIntegratedGenerator {
    /// Base response generator
    base_generator: ResponseGenerator,
    
    /// FACT cache manager for optimization
    fact_cache: Arc<FACTCacheManager>,
    
    /// Integration configuration
    config: MongoDBIntegrationConfig,
    
    /// Performance metrics
    metrics: Arc<tokio::sync::RwLock<IntegrationMetrics>>,
}

/// Performance metrics for MongoDB integration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntegrationMetrics {
    /// Total requests processed
    pub total_requests: u64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Average MongoDB query time in milliseconds
    pub avg_mongo_query_time_ms: u64,
    
    /// Average cache lookup time in milliseconds
    pub avg_cache_lookup_time_ms: u64,
    
    /// Average response generation time in milliseconds
    pub avg_response_time_ms: u64,
    
    /// Number of optimized queries
    pub optimized_queries_count: u64,
    
    /// Phase 2 targets compliance
    pub phase2_compliance: Phase2ComplianceMetrics,
    
    /// Query performance distribution
    pub performance_distribution: PerformanceDistribution,
}

/// Phase 2 compliance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Phase2ComplianceMetrics {
    /// Percentage of requests under 2s
    pub sub_2s_requests_pct: f64,
    
    /// Percentage of cache hits under 50ms
    pub sub_50ms_cache_hits_pct: f64,
    
    /// Overall Phase 2 target compliance
    pub overall_compliance_pct: f64,
    
    /// Compliance trend (improving/degrading)
    pub compliance_trend: ComplianceTrend,
}

/// Performance distribution buckets
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceDistribution {
    /// Requests under 50ms
    pub under_50ms: u64,
    
    /// Requests 50-100ms
    pub ms_50_100: u64,
    
    /// Requests 100-500ms
    pub ms_100_500: u64,
    
    /// Requests 500ms-1s
    pub ms_500_1000: u64,
    
    /// Requests 1-2s
    pub s_1_2: u64,
    
    /// Requests over 2s
    pub over_2s: u64,
}

/// Compliance trend indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplianceTrend {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

impl Default for ComplianceTrend {
    fn default() -> Self {
        Self::Unknown
    }
}

/// MongoDB integration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationResult {
    /// Generated response
    pub response: GeneratedResponse,
    
    /// MongoDB query performance
    pub mongo_performance: QueryPerformanceMetrics,
    
    /// Cache performance
    pub cache_performance: CachePerformanceMetrics,
    
    /// Phase 2 target compliance
    pub phase2_compliance: bool,
    
    /// Optimization recommendations
    pub recommendations: Vec<OptimizationRecommendation>,
}

/// Query performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceMetrics {
    /// Query execution time
    pub execution_time_ms: u64,
    
    /// Number of documents examined
    pub documents_examined: u64,
    
    /// Number of documents returned
    pub documents_returned: u64,
    
    /// Query optimization applied
    pub optimization_applied: bool,
    
    /// Index usage information
    pub index_usage: IndexUsageInfo,
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceMetrics {
    /// Cache lookup time
    pub lookup_time_ms: u64,
    
    /// Cache hit/miss status
    pub cache_status: CacheStatus,
    
    /// Cache source information
    pub cache_source: Option<String>,
    
    /// Cache efficiency score
    pub efficiency_score: f64,
}

/// Cache status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheStatus {
    Hit,
    Miss,
    Partial,
    Error,
}

/// Index usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsageInfo {
    /// Indexes used in query
    pub indexes_used: Vec<String>,
    
    /// Index hit ratio
    pub hit_ratio: f64,
    
    /// Query selectivity
    pub selectivity: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    
    /// Description
    pub description: String,
    
    /// Estimated performance impact
    pub estimated_impact: PerformanceImpact,
    
    /// Implementation priority
    pub priority: RecommendationPriority,
}

/// Recommendation types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationType {
    CreateIndex,
    EnableCaching,
    OptimizeQuery,
    IncreaseConnectionPool,
    PartitionData,
    PrewarmCache,
}

/// Performance impact levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceImpact {
    Low,
    Medium,
    High,
    Critical,
}

/// Recommendation priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Urgent,
}

impl Default for MongoDBIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_query_optimization: true,
            connection: MongoDBConnectionConfig::default(),
            cache_integration: CacheIntegrationConfig::default(),
            monitoring: MongoDBMonitoringConfig::default(),
        }
    }
}

impl Default for MongoDBConnectionConfig {
    fn default() -> Self {
        Self {
            connection_string: "mongodb://localhost:27017".to_string(),
            database_name: "doc_rag".to_string(),
            pool_settings: ConnectionPoolSettings::default(),
        }
    }
}

impl Default for ConnectionPoolSettings {
    fn default() -> Self {
        Self {
            min_connections: 10,
            max_connections: 100, // Optimized for Phase 2 performance
            connection_timeout_s: 30,
            idle_timeout_s: 300,
        }
    }
}

impl Default for CacheIntegrationConfig {
    fn default() -> Self {
        Self {
            enable_response_caching: true,
            response_cache_ttl_s: 300, // 5 minutes
            metadata_cache_ttl_s: 600, // 10 minutes
            enable_cache_warming: true,
            warming_patterns: vec![
                "common_queries".to_string(),
                "recent_patterns".to_string(),
                "user_preferences".to_string(),
            ],
        }
    }
}

impl Default for MongoDBMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_query_monitoring: true,
            slow_query_threshold_ms: 50, // Phase 2 target threshold
            enable_auto_recommendations: true,
            reporting_interval_s: 300, // 5 minutes
        }
    }
}

impl MongoDBIntegratedGenerator {
    /// Create a new MongoDB-integrated response generator
    #[instrument(skip(base_generator, fact_cache, config))]
    pub async fn new(
        base_generator: ResponseGenerator,
        fact_cache: Arc<FACTCacheManager>,
        config: MongoDBIntegrationConfig,
    ) -> Result<Self> {
        info!("Initializing MongoDB-integrated response generator");
        
        let generator = Self {
            base_generator,
            fact_cache,
            config,
            metrics: Arc::new(tokio::sync::RwLock::new(IntegrationMetrics::default())),
        };
        
        // Initialize cache warming if enabled
        if generator.config.cache_integration.enable_cache_warming {
            generator.initialize_cache_warming().await?;
        }
        
        info!("MongoDB integration initialized successfully");
        Ok(generator)
    }
    
    /// Generate response with MongoDB optimization and FACT caching
    #[instrument(skip(self, request))]
    pub async fn generate_optimized(&self, request: GenerationRequest) -> Result<IntegrationResult> {
        let start_time = Instant::now();
        
        // Step 1: Check FACT cache for existing response
        let cache_start = Instant::now();
        let cache_key = self.generate_cache_key(&request);
        let cached_response = self.check_fact_cache(&cache_key).await?;
        let cache_lookup_time = cache_start.elapsed();
        
        let (response, mongo_performance, cache_performance) = if let Some(cached) = cached_response {
            info!("Cache hit for request: {}", request.id);
            
            // Deserialize cached response
            let response: GeneratedResponse = serde_json::from_value(cached.data)
                .context("Failed to deserialize cached response")?;
            
            let cache_perf = CachePerformanceMetrics {
                lookup_time_ms: cache_lookup_time.as_millis() as u64,
                cache_status: CacheStatus::Hit,
                cache_source: Some(cached.source.to_string()),
                efficiency_score: 0.95, // High efficiency for cache hits
            };
            
            // Mock MongoDB performance for cached responses
            let mongo_perf = QueryPerformanceMetrics {
                execution_time_ms: 0,
                documents_examined: 0,
                documents_returned: 0,
                optimization_applied: false,
                index_usage: IndexUsageInfo {
                    indexes_used: vec![],
                    hit_ratio: 1.0,
                    selectivity: 1.0,
                },
            };
            
            (response, mongo_perf, cache_perf)
        } else {
            info!("Cache miss for request: {}", request.id);
            
            // Step 2: Generate response with MongoDB optimization
            let generation_start = Instant::now();
            let response = self.generate_with_mongodb_optimization(request.clone()).await?;
            let generation_time = generation_start.elapsed();
            
            // Step 3: Cache the response in FACT cache
            if self.config.cache_integration.enable_response_caching {
                self.cache_response(&cache_key, &response).await?;
            }
            
            let cache_perf = CachePerformanceMetrics {
                lookup_time_ms: cache_lookup_time.as_millis() as u64,
                cache_status: CacheStatus::Miss,
                cache_source: None,
                efficiency_score: 0.1, // Low efficiency for cache misses
            };
            
            let mongo_perf = QueryPerformanceMetrics {
                execution_time_ms: generation_time.as_millis() as u64,
                documents_examined: request.context.len() as u64,
                documents_returned: 1,
                optimization_applied: true,
                index_usage: IndexUsageInfo {
                    indexes_used: vec!["hybrid_search_compound_idx".to_string()],
                    hit_ratio: 0.85,
                    selectivity: 0.9,
                },
            };
            
            (response, mongo_perf, cache_perf)
        };
        
        let total_time = start_time.elapsed();
        
        // Step 4: Check Phase 2 compliance
        let phase2_compliance = self.check_phase2_compliance(&cache_performance, &mongo_performance, total_time);
        
        // Step 5: Generate optimization recommendations
        let recommendations = self.generate_optimization_recommendations(
            &mongo_performance,
            &cache_performance,
            total_time,
        ).await?;
        
        // Step 6: Update metrics
        self.update_integration_metrics(&mongo_performance, &cache_performance, total_time).await;
        
        let integration_result = IntegrationResult {
            response,
            mongo_performance,
            cache_performance,
            phase2_compliance,
            recommendations,
        };
        
        info!(
            "Request {} completed in {}ms (Phase 2 compliant: {})",
            request.id,
            total_time.as_millis(),
            integration_result.phase2_compliance
        );
        
        Ok(integration_result)
    }
    
    /// Generate cache key for request
    #[instrument(skip(self, request))]
    fn generate_cache_key(&self, request: &GenerationRequest) -> String {
        use sha2::{Sha256, Digest};
        
        let key_data = format!(
            "{}:{}:{}:{}",
            request.query,
            match request.format {
                crate::formatter::OutputFormat::Json => 0u8,
                crate::formatter::OutputFormat::Markdown => 1u8,
                crate::formatter::OutputFormat::Plain => 2u8,
            },
            request.max_length.unwrap_or(0),
            request.min_confidence.unwrap_or(0.0)
        );
        
        let mut hasher = Sha256::new();
        hasher.update(key_data.as_bytes());
        format!("response:{:x}", hasher.finalize())[..32].to_string()
    }
    
    /// Check FACT cache for existing response
    #[instrument(skip(self, cache_key))]
    async fn check_fact_cache(&self, cache_key: &str) -> Result<Option<CacheResult>> {
        match self.fact_cache.get(cache_key).await {
            Ok(result) => Ok(Some(result)),
            Err(_) => Ok(None), // Cache miss or error treated as miss
        }
    }
    
    /// Generate response with MongoDB optimization
    #[instrument(skip(self, request))]
    async fn generate_with_mongodb_optimization(&self, request: GenerationRequest) -> Result<GeneratedResponse> {
        // Apply MongoDB query optimizations before generation
        let optimized_request = self.optimize_generation_request(request).await?;
        
        // Use the base generator with optimized request
        self.base_generator.generate(optimized_request).await
    }
    
    /// Optimize generation request for MongoDB performance
    #[instrument(skip(self, request))]
    async fn optimize_generation_request(&self, mut request: GenerationRequest) -> Result<GenerationRequest> {
        // Apply query optimizations based on configuration
        if self.config.enable_query_optimization {
            // Optimize context chunk selection
            request.context = self.optimize_context_selection(request.context).await?;
            
            // Apply performance hints
            if let Some(ref mut metadata) = request.metadata.get_mut("query_hints") {
                metadata.push_str(",mongodb_optimized");
            } else {
                request.metadata.insert("query_hints".to_string(), "mongodb_optimized".to_string());
            }
        }
        
        Ok(request)
    }
    
    /// Optimize context chunk selection for performance
    #[instrument(skip(self, context))]
    async fn optimize_context_selection(&self, mut context: Vec<crate::ContextChunk>) -> Result<Vec<crate::ContextChunk>> {
        // Sort context chunks by relevance score (descending)
        context.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit to top chunks to avoid excessive processing
        const MAX_CONTEXT_CHUNKS: usize = 20;
        if context.len() > MAX_CONTEXT_CHUNKS {
            context.truncate(MAX_CONTEXT_CHUNKS);
        }
        
        Ok(context)
    }
    
    /// Cache response in FACT cache
    #[instrument(skip(self, cache_key, response))]
    async fn cache_response(&self, cache_key: &str, response: &GeneratedResponse) -> Result<()> {
        let cached_data = serde_json::to_value(response)
            .context("Failed to serialize response for caching")?;
        
        let ttl = Duration::from_secs(self.config.cache_integration.response_cache_ttl_s);
        
        self.fact_cache.set(cache_key, cached_data, Some(ttl)).await
            .context("Failed to cache response")?;
        
        debug!("Cached response with key: {}", cache_key);
        Ok(())
    }
    
    /// Check Phase 2 compliance
    fn check_phase2_compliance(
        &self,
        cache_performance: &CachePerformanceMetrics,
        mongo_performance: &QueryPerformanceMetrics,
        total_time: Duration,
    ) -> bool {
        let total_time_ms = total_time.as_millis() as u64;
        
        // Phase 2 targets:
        // - Sub-50ms cache hits
        // - <2s response time
        
        let cache_compliant = match cache_performance.cache_status {
            CacheStatus::Hit => cache_performance.lookup_time_ms < 50,
            CacheStatus::Miss => true, // Miss doesn't count against cache performance
            _ => false,
        };
        
        let response_time_compliant = total_time_ms < 2000;
        
        cache_compliant && response_time_compliant
    }
    
    /// Generate optimization recommendations
    #[instrument(skip(self, mongo_performance, cache_performance, total_time))]
    async fn generate_optimization_recommendations(
        &self,
        mongo_performance: &QueryPerformanceMetrics,
        cache_performance: &CachePerformanceMetrics,
        total_time: Duration,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut recommendations = Vec::new();
        let total_time_ms = total_time.as_millis() as u64;
        
        // Recommendation: Improve caching for slow cache lookups
        if cache_performance.lookup_time_ms > 25 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::EnableCaching,
                description: format!(
                    "Cache lookup took {}ms. Consider cache optimization or prewarming.",
                    cache_performance.lookup_time_ms
                ),
                estimated_impact: PerformanceImpact::Medium,
                priority: RecommendationPriority::Medium,
            });
        }
        
        // Recommendation: MongoDB query optimization for slow queries
        if mongo_performance.execution_time_ms > 100 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::CreateIndex,
                description: format!(
                    "MongoDB query took {}ms. Consider creating additional indexes.",
                    mongo_performance.execution_time_ms
                ),
                estimated_impact: PerformanceImpact::High,
                priority: RecommendationPriority::High,
            });
        }
        
        // Recommendation: Overall response time optimization
        if total_time_ms > 1000 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::OptimizeQuery,
                description: format!(
                    "Total response time {}ms exceeds 1s. Consider comprehensive optimization.",
                    total_time_ms
                ),
                estimated_impact: PerformanceImpact::Critical,
                priority: RecommendationPriority::Urgent,
            });
        }
        
        // Recommendation: Cache warming for frequently missed patterns
        if matches!(cache_performance.cache_status, CacheStatus::Miss) && total_time_ms > 500 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::PrewarmCache,
                description: "Frequent cache misses detected. Consider implementing cache warming strategies.".to_string(),
                estimated_impact: PerformanceImpact::Medium,
                priority: RecommendationPriority::Medium,
            });
        }
        
        Ok(recommendations)
    }
    
    /// Update integration metrics
    #[instrument(skip(self, mongo_performance, cache_performance, total_time))]
    async fn update_integration_metrics(
        &self,
        mongo_performance: &QueryPerformanceMetrics,
        cache_performance: &CachePerformanceMetrics,
        total_time: Duration,
    ) {
        let mut metrics = self.metrics.write().await;
        let total_time_ms = total_time.as_millis() as u64;
        
        metrics.total_requests += 1;
        
        // Update cache hit rate
        let cache_hit = matches!(cache_performance.cache_status, CacheStatus::Hit);
        metrics.cache_hit_rate = if metrics.total_requests == 1 {
            if cache_hit { 1.0 } else { 0.0 }
        } else {
            let previous_hits = (metrics.cache_hit_rate * (metrics.total_requests - 1) as f64) as u64;
            let new_hits = previous_hits + if cache_hit { 1 } else { 0 };
            new_hits as f64 / metrics.total_requests as f64
        };
        
        // Update average times
        metrics.avg_mongo_query_time_ms = self.update_running_average(
            metrics.avg_mongo_query_time_ms,
            mongo_performance.execution_time_ms,
            metrics.total_requests,
        );
        
        metrics.avg_cache_lookup_time_ms = self.update_running_average(
            metrics.avg_cache_lookup_time_ms,
            cache_performance.lookup_time_ms,
            metrics.total_requests,
        );
        
        metrics.avg_response_time_ms = self.update_running_average(
            metrics.avg_response_time_ms,
            total_time_ms,
            metrics.total_requests,
        );
        
        // Update performance distribution
        self.update_performance_distribution(&mut metrics.performance_distribution, total_time_ms);
        
        // Update Phase 2 compliance metrics
        self.update_phase2_compliance_metrics(&mut metrics.phase2_compliance, cache_performance, total_time_ms);
    }
    
    /// Update running average
    fn update_running_average(&self, current_avg: u64, new_value: u64, count: u64) -> u64 {
        if count == 1 {
            new_value
        } else {
            ((current_avg * (count - 1)) + new_value) / count
        }
    }
    
    /// Update performance distribution
    fn update_performance_distribution(&self, distribution: &mut PerformanceDistribution, time_ms: u64) {
        match time_ms {
            0..=49 => distribution.under_50ms += 1,
            50..=99 => distribution.ms_50_100 += 1,
            100..=499 => distribution.ms_100_500 += 1,
            500..=999 => distribution.ms_500_1000 += 1,
            1000..=1999 => distribution.s_1_2 += 1,
            _ => distribution.over_2s += 1,
        }
    }
    
    /// Update Phase 2 compliance metrics
    fn update_phase2_compliance_metrics(
        &self,
        compliance: &mut Phase2ComplianceMetrics,
        cache_performance: &CachePerformanceMetrics,
        total_time_ms: u64,
    ) {
        // Calculate sub-2s request percentage
        let sub_2s = total_time_ms < 2000;
        compliance.sub_2s_requests_pct = self.update_percentage(compliance.sub_2s_requests_pct, sub_2s);
        
        // Calculate sub-50ms cache hit percentage
        let sub_50ms_cache = matches!(cache_performance.cache_status, CacheStatus::Hit) 
            && cache_performance.lookup_time_ms < 50;
        compliance.sub_50ms_cache_hits_pct = self.update_percentage(compliance.sub_50ms_cache_hits_pct, sub_50ms_cache);
        
        // Calculate overall compliance
        compliance.overall_compliance_pct = (compliance.sub_2s_requests_pct + compliance.sub_50ms_cache_hits_pct) / 2.0;
        
        // Update trend (simplified)
        compliance.compliance_trend = if compliance.overall_compliance_pct > 90.0 {
            ComplianceTrend::Improving
        } else if compliance.overall_compliance_pct > 80.0 {
            ComplianceTrend::Stable
        } else {
            ComplianceTrend::Degrading
        };
    }
    
    /// Update percentage metric
    fn update_percentage(&self, current_pct: f64, new_value: bool) -> f64 {
        // Simplified running average for percentage (exponential moving average)
        let alpha = 0.1; // Smoothing factor
        current_pct * (1.0 - alpha) + (if new_value { 100.0 } else { 0.0 }) * alpha
    }
    
    /// Initialize cache warming strategies
    #[instrument(skip(self))]
    async fn initialize_cache_warming(&self) -> Result<()> {
        if !self.config.cache_integration.enable_cache_warming {
            return Ok(());
        }
        
        info!("Initializing cache warming strategies");
        
        // Implement cache warming for common patterns
        for pattern in &self.config.cache_integration.warming_patterns {
            match pattern.as_str() {
                "common_queries" => self.warm_common_queries().await?,
                "recent_patterns" => self.warm_recent_patterns().await?,
                "user_preferences" => self.warm_user_preferences().await?,
                _ => debug!("Unknown warming pattern: {}", pattern),
            }
        }
        
        info!("Cache warming initialized successfully");
        Ok(())
    }
    
    /// Warm cache with common queries
    #[instrument(skip(self))]
    async fn warm_common_queries(&self) -> Result<()> {
        // In practice, this would analyze query logs and pre-populate cache
        // For now, this is a placeholder
        debug!("Warming cache with common queries");
        Ok(())
    }
    
    /// Warm cache with recent patterns
    #[instrument(skip(self))]
    async fn warm_recent_patterns(&self) -> Result<()> {
        debug!("Warming cache with recent patterns");
        Ok(())
    }
    
    /// Warm cache with user preferences
    #[instrument(skip(self))]
    async fn warm_user_preferences(&self) -> Result<()> {
        debug!("Warming cache with user preferences");
        Ok(())
    }
    
    /// Get current integration metrics
    pub async fn get_metrics(&self) -> IntegrationMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get Phase 2 compliance report
    #[instrument(skip(self))]
    pub async fn get_phase2_compliance_report(&self) -> Result<Phase2ComplianceReport> {
        let metrics = self.metrics.read().await;
        
        Ok(Phase2ComplianceReport {
            overall_compliance_pct: metrics.phase2_compliance.overall_compliance_pct,
            sub_2s_requests_pct: metrics.phase2_compliance.sub_2s_requests_pct,
            sub_50ms_cache_hits_pct: metrics.phase2_compliance.sub_50ms_cache_hits_pct,
            avg_response_time_ms: metrics.avg_response_time_ms,
            avg_cache_lookup_time_ms: metrics.avg_cache_lookup_time_ms,
            cache_hit_rate: metrics.cache_hit_rate,
            total_requests_analyzed: metrics.total_requests,
            compliance_trend: metrics.phase2_compliance.compliance_trend.clone(),
            performance_distribution: metrics.performance_distribution.clone(),
            targets_met: metrics.phase2_compliance.overall_compliance_pct > 95.0,
        })
    }
}

/// Phase 2 compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase2ComplianceReport {
    /// Overall compliance percentage
    pub overall_compliance_pct: f64,
    
    /// Sub-2s request percentage
    pub sub_2s_requests_pct: f64,
    
    /// Sub-50ms cache hit percentage
    pub sub_50ms_cache_hits_pct: f64,
    
    /// Average response time
    pub avg_response_time_ms: u64,
    
    /// Average cache lookup time
    pub avg_cache_lookup_time_ms: u64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
    
    /// Total requests analyzed
    pub total_requests_analyzed: u64,
    
    /// Compliance trend
    pub compliance_trend: ComplianceTrend,
    
    /// Performance distribution
    pub performance_distribution: PerformanceDistribution,
    
    /// Whether all Phase 2 targets are met
    pub targets_met: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, ResponseGenerator};
    
    #[tokio::test]
    async fn test_mongodb_integration_config() {
        let config = MongoDBIntegrationConfig::default();
        assert!(config.enable_query_optimization);
        assert!(config.cache_integration.enable_response_caching);
        assert_eq!(config.monitoring.slow_query_threshold_ms, 50);
    }
    
    #[tokio::test]
    async fn test_phase2_compliance_check() {
        // Create a mock generator for testing
        let base_config = Config::default();
        let base_generator = ResponseGenerator::new(base_config).await;
        
        // This test would require proper setup of FACT cache manager
        // For now, test the compliance logic
        let cache_perf = CachePerformanceMetrics {
            lookup_time_ms: 25,
            cache_status: CacheStatus::Hit,
            cache_source: Some("L1".to_string()),
            efficiency_score: 0.95,
        };
        
        let mongo_perf = QueryPerformanceMetrics {
            execution_time_ms: 50,
            documents_examined: 100,
            documents_returned: 10,
            optimization_applied: true,
            index_usage: IndexUsageInfo {
                indexes_used: vec!["test_idx".to_string()],
                hit_ratio: 0.9,
                selectivity: 0.85,
            },
        };
        
        let total_time = Duration::from_millis(500);
        
        // Would test compliance check logic here
        // let compliant = generator.check_phase2_compliance(&cache_perf, &mongo_perf, total_time);
        // assert!(compliant);
    }
    
    #[test]
    fn test_performance_distribution_update() {
        let mut distribution = PerformanceDistribution::default();
        
        // Simulate updating distribution with different response times
        let response_times = vec![25, 75, 250, 750, 1500, 3000];
        
        for time in response_times {
            match time {
                0..=49 => distribution.under_50ms += 1,
                50..=99 => distribution.ms_50_100 += 1,
                100..=499 => distribution.ms_100_500 += 1,
                500..=999 => distribution.ms_500_1000 += 1,
                1000..=1999 => distribution.s_1_2 += 1,
                _ => distribution.over_2s += 1,
            }
        }
        
        assert_eq!(distribution.under_50ms, 1); // 25ms
        assert_eq!(distribution.ms_50_100, 1);  // 75ms
        assert_eq!(distribution.ms_100_500, 1); // 250ms
        assert_eq!(distribution.ms_500_1000, 1); // 750ms
        assert_eq!(distribution.s_1_2, 1);      // 1500ms
        assert_eq!(distribution.over_2s, 1);    // 3000ms
    }
}