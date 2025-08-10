//! Performance module
//! 
//! Comprehensive performance optimization and profiling tools for the Doc-RAG system.

pub mod profiler;
pub mod optimizer;
pub mod integration;

pub use profiler::*;
pub use optimizer::*;
pub use integration::*;

/// Performance module configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub profiler_config: ProfilerConfig,
    pub optimizer_config: OptimizerConfig,
    pub enable_real_time_monitoring: bool,
    pub performance_targets: PerformanceTargets,
}

/// Performance targets from requirements
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    pub query_processing_ms: u64,
    pub response_generation_ms: u64,
    pub end_to_end_ms: u64,
    pub throughput_qps: f64,
    pub memory_usage_mb: u64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            query_processing_ms: 50,
            response_generation_ms: 100,
            end_to_end_ms: 200,
            throughput_qps: 100.0,
            memory_usage_mb: 2048,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            profiler_config: ProfilerConfig::default(),
            optimizer_config: OptimizerConfig::default(),
            enable_real_time_monitoring: true,
            performance_targets: PerformanceTargets::default(),
        }
    }
}

/// Integrated performance management system
pub struct PerformanceManager {
    profiler: PerformanceProfiler,
    optimizer: PerformanceOptimizer,
    config: PerformanceConfig,
}

impl PerformanceManager {
    /// Create new performance manager
    pub fn new(config: PerformanceConfig) -> Self {
        let profiler = PerformanceProfiler::new(config.profiler_config.clone());
        let optimizer = PerformanceOptimizer::new(config.optimizer_config.clone());
        
        Self {
            profiler,
            optimizer,
            config,
        }
    }
    
    /// Start performance monitoring and optimization
    pub async fn start(&self) {
        tracing::info!("Starting integrated performance management system");
        
        // Start profiler
        self.profiler.start_profiling().await;
        
        // Start optimizer
        self.optimizer.start_adaptive_optimization().await;
        
        if self.config.enable_real_time_monitoring {
            self.start_real_time_monitoring().await;
        }
    }
    
    /// Profile an operation with automatic optimization
    pub async fn profile_and_optimize<F, T>(&self, component: &str, operation: &str, func: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        // Check cache first
        let cache_key = format!("{}::{}", component, operation);
        if let Some(cached_result) = self.optimizer.cache_get(&cache_key).await {
            // Return cached result if applicable (this is a simplified example)
            // In practice, you'd need to deserialize the result properly
            tracing::debug!("Cache hit for operation: {}::{}", component, operation);
        }
        
        // Profile the operation
        let result = self.profiler.profile_operation(component, operation, func).await;
        
        // Optionally cache the result
        // self.optimizer.cache_put(cache_key, serialized_result, None).await;
        
        result
    }
    
    /// Get comprehensive performance report
    pub async fn get_performance_report(&self) -> IntegratedPerformanceReport {
        let profiler_report = self.profiler.generate_report().await;
        let optimizer_metrics = self.optimizer.get_metrics().await;
        
        IntegratedPerformanceReport {
            timestamp: chrono::Utc::now().timestamp() as u64,
            profiler_report,
            optimizer_metrics,
            performance_targets: self.config.performance_targets.clone(),
            overall_health: self.assess_overall_health(&profiler_report).await,
        }
    }
    
    async fn start_real_time_monitoring(&self) {
        tracing::info!("Starting real-time performance monitoring");
        
        let profiler = self.profiler.clone();
        let optimizer = self.optimizer.clone();
        let targets = self.config.performance_targets.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                let metrics = profiler.get_metrics().await;
                let optimizer_metrics = optimizer.get_metrics().await;
                
                // Check if performance targets are being met
                Self::check_performance_targets(&metrics, &optimizer_metrics, &targets).await;
            }
        });
    }
    
    async fn check_performance_targets(
        _profiler_metrics: &ProfilerMetrics,
        optimizer_metrics: &OptimizerMetrics,
        targets: &PerformanceTargets
    ) {
        // Check memory usage
        if optimizer_metrics.memory_pool_metrics.total_allocated_mb > targets.memory_usage_mb {
            tracing::warn!(
                "Memory usage ({} MB) exceeds target ({} MB)",
                optimizer_metrics.memory_pool_metrics.total_allocated_mb,
                targets.memory_usage_mb
            );
        }
        
        // Check cache performance
        if optimizer_metrics.cache_metrics.hit_rate < 0.7 {
            tracing::warn!(
                "Cache hit rate ({:.2}) below optimal threshold (0.7)",
                optimizer_metrics.cache_metrics.hit_rate
            );
        }
    }
    
    async fn assess_overall_health(&self, report: &PerformanceReport) -> PerformanceHealth {
        let mut health_score = 100.0;
        let mut issues = Vec::new();
        
        // Check for critical bottlenecks
        if report.bottleneck_summary.critical_bottlenecks > 0 {
            health_score -= 30.0;
            issues.push("Critical performance bottlenecks detected".to_string());
        }
        
        // Check error rate
        if report.system_summary.error_rate_percent > 5.0 {
            health_score -= 20.0;
            issues.push("High error rate detected".to_string());
        }
        
        // Check latency
        if report.system_summary.avg_system_latency_ms > self.config.performance_targets.end_to_end_ms as f64 {
            health_score -= 25.0;
            issues.push("System latency exceeds targets".to_string());
        }
        
        // Check throughput
        if report.system_summary.system_throughput_ops_per_sec < self.config.performance_targets.throughput_qps {
            health_score -= 15.0;
            issues.push("System throughput below targets".to_string());
        }
        
        let status = match health_score {
            score if score >= 90.0 => HealthStatus::Excellent,
            score if score >= 80.0 => HealthStatus::Good,
            score if score >= 70.0 => HealthStatus::Fair,
            score if score >= 60.0 => HealthStatus::Poor,
            _ => HealthStatus::Critical,
        };
        
        PerformanceHealth {
            status,
            health_score,
            issues,
        }
    }
}

/// Integrated performance report
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntegratedPerformanceReport {
    pub timestamp: u64,
    pub profiler_report: PerformanceReport,
    pub optimizer_metrics: OptimizerMetrics,
    pub performance_targets: PerformanceTargets,
    pub overall_health: PerformanceHealth,
}

/// Overall system performance health
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceHealth {
    pub status: HealthStatus,
    pub health_score: f64,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_manager_creation() {
        let config = PerformanceConfig::default();
        let manager = PerformanceManager::new(config);
        
        let report = manager.get_performance_report().await;
        assert!(report.timestamp > 0);
    }
    
    #[tokio::test]
    async fn test_performance_targets() {
        let targets = PerformanceTargets::default();
        
        assert_eq!(targets.query_processing_ms, 50);
        assert_eq!(targets.response_generation_ms, 100);
        assert_eq!(targets.end_to_end_ms, 200);
        assert_eq!(targets.throughput_qps, 100.0);
        assert_eq!(targets.memory_usage_mb, 2048);
    }
    
    #[tokio::test]
    async fn test_integrated_profiling_and_optimization() {
        let config = PerformanceConfig::default();
        let manager = PerformanceManager::new(config);
        
        let result = manager.profile_and_optimize("test", "operation", async {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            "test_result"
        }).await;
        
        assert_eq!(result, "test_result");
    }
}