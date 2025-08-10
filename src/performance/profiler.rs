//! Comprehensive Performance Profiler
//! 
//! Advanced profiling system for identifying bottlenecks and optimization opportunities
//! across the entire Doc-RAG system with real-time monitoring and analysis.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

/// Performance profiler with real-time monitoring
#[derive(Debug, Clone)]
pub struct PerformanceProfiler {
    metrics: Arc<RwLock<ProfilerMetrics>>,
    config: ProfilerConfig,
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    pub enable_cpu_profiling: bool,
    pub enable_memory_profiling: bool,
    pub enable_io_profiling: bool,
    pub enable_network_profiling: bool,
    pub sampling_interval_ms: u64,
    pub max_samples: usize,
    pub alert_thresholds: AlertThresholds,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enable_cpu_profiling: true,
            enable_memory_profiling: true,
            enable_io_profiling: true,
            enable_network_profiling: true,
            sampling_interval_ms: 100,
            max_samples: 10000,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert thresholds for performance monitoring
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub latency_ms: u64,
    pub error_rate_percent: f64,
    pub throughput_qps: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 80.0,
            memory_usage_mb: 1500,
            latency_ms: 150,
            error_rate_percent: 5.0,
            throughput_qps: 80.0,
        }
    }
}

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerMetrics {
    pub system_metrics: SystemMetrics,
    pub component_metrics: HashMap<String, ComponentMetrics>,
    pub bottleneck_analysis: Vec<Bottleneck>,
    pub performance_trends: PerformanceTrends,
    pub alerts: Vec<PerformanceAlert>,
    pub recommendations: Vec<OptimizationRecommendation>,
}

impl Default for ProfilerMetrics {
    fn default() -> Self {
        Self {
            system_metrics: SystemMetrics::default(),
            component_metrics: HashMap::new(),
            bottleneck_analysis: Vec::new(),
            performance_trends: PerformanceTrends::default(),
            alerts: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}

/// System-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: u64,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub memory_available_mb: u64,
    pub gc_pressure_score: f64,
    pub thread_count: u32,
    pub open_file_descriptors: u32,
    pub network_connections: u32,
    pub io_read_mbps: f64,
    pub io_write_mbps: f64,
    pub network_rx_mbps: f64,
    pub network_tx_mbps: f64,
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self {
            timestamp: 0,
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0,
            memory_available_mb: 0,
            gc_pressure_score: 0.0,
            thread_count: 0,
            open_file_descriptors: 0,
            network_connections: 0,
            io_read_mbps: 0.0,
            io_write_mbps: 0.0,
            network_rx_mbps: 0.0,
            network_tx_mbps: 0.0,
        }
    }
}

/// Component-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    pub component_name: String,
    pub operation_count: u64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub max_latency_ms: f64,
    pub error_count: u64,
    pub success_count: u64,
    pub throughput_ops_per_sec: f64,
    pub resource_usage: ResourceUsage,
    pub latency_histogram: HashMap<u64, u64>, // latency_bucket -> count
}

/// Resource usage per component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_time_ms: u64,
    pub memory_allocated_mb: u64,
    pub io_bytes_read: u64,
    pub io_bytes_written: u64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
}

/// Identified performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub id: String,
    pub component: String,
    pub bottleneck_type: BottleneckType,
    pub severity: BottleneckSeverity,
    pub impact_score: f64,
    pub description: String,
    pub suggested_optimizations: Vec<String>,
    pub detected_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
    Database,
    Algorithm,
    Concurrency,
    GarbageCollection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Critical,  // >50% performance impact
    High,      // 25-50% performance impact
    Medium,    // 10-25% performance impact
    Low,       // <10% performance impact
}

/// Performance trends over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub latency_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub error_rate_trend: TrendDirection,
    pub trend_period_hours: u64,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            latency_trend: TrendDirection::Stable,
            throughput_trend: TrendDirection::Stable,
            memory_trend: TrendDirection::Stable,
            error_rate_trend: TrendDirection::Stable,
            trend_period_hours: 24,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub component: String,
    pub message: String,
    pub threshold_value: f64,
    pub current_value: f64,
    pub triggered_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighLatency,
    LowThroughput,
    HighCPU,
    HighMemory,
    HighErrorRate,
    SystemOverload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub id: String,
    pub recommendation_type: RecommendationType,
    pub priority: RecommendationPriority,
    pub component: String,
    pub title: String,
    pub description: String,
    pub expected_improvement_percent: f64,
    pub implementation_effort: ImplementationEffort,
    pub code_changes_required: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    Caching,
    ConnectionPooling,
    AlgorithmOptimization,
    MemoryOptimization,
    DatabaseOptimization,
    ConcurrencyOptimization,
    ResourceOptimization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,      // <1 day
    Medium,   // 1-3 days
    High,     // 1-2 weeks
    VeryHigh, // >2 weeks
}

impl PerformanceProfiler {
    /// Create new performance profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(ProfilerMetrics::default())),
            config,
        }
    }
    
    /// Start continuous profiling
    pub async fn start_profiling(&self) {
        info!("Starting performance profiler with config: {:?}", self.config);
        
        let metrics = self.metrics.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                Duration::from_millis(config.sampling_interval_ms)
            );
            
            loop {
                interval.tick().await;
                
                let mut metrics_guard = metrics.write().await;
                
                // Collect system metrics
                if config.enable_cpu_profiling || config.enable_memory_profiling {
                    let system_metrics = Self::collect_system_metrics(&config).await;
                    metrics_guard.system_metrics = system_metrics;
                }
                
                // Analyze bottlenecks
                let bottlenecks = Self::analyze_bottlenecks(&metrics_guard).await;
                metrics_guard.bottleneck_analysis = bottlenecks;
                
                // Check for alerts
                let alerts = Self::check_alerts(&metrics_guard, &config).await;
                metrics_guard.alerts = alerts;
                
                // Generate recommendations
                let recommendations = Self::generate_recommendations(&metrics_guard).await;
                metrics_guard.recommendations = recommendations;
                
                drop(metrics_guard);
            }
        });
    }
    
    /// Profile a specific operation
    pub async fn profile_operation<F, T>(&self, component: &str, operation: &str, func: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = func.await;
        let duration = start.elapsed();
        
        self.record_operation(component, operation, duration, true).await;
        result
    }
    
    /// Profile an operation that may fail
    pub async fn profile_fallible_operation<F, T, E>(&self, component: &str, operation: &str, func: F) -> Result<T, E>
    where
        F: std::future::Future<Output = Result<T, E>>,
    {
        let start = Instant::now();
        let result = func.await;
        let duration = start.elapsed();
        let success = result.is_ok();
        
        self.record_operation(component, operation, duration, success).await;
        result
    }
    
    /// Record operation metrics
    pub async fn record_operation(&self, component: &str, _operation: &str, duration: Duration, success: bool) {
        let mut metrics = self.metrics.write().await;
        
        let component_metrics = metrics.component_metrics
            .entry(component.to_string())
            .or_insert_with(|| ComponentMetrics {
                component_name: component.to_string(),
                operation_count: 0,
                avg_latency_ms: 0.0,
                p95_latency_ms: 0.0,
                p99_latency_ms: 0.0,
                max_latency_ms: 0.0,
                error_count: 0,
                success_count: 0,
                throughput_ops_per_sec: 0.0,
                resource_usage: ResourceUsage {
                    cpu_time_ms: 0,
                    memory_allocated_mb: 0,
                    io_bytes_read: 0,
                    io_bytes_written: 0,
                    network_bytes_sent: 0,
                    network_bytes_received: 0,
                },
                latency_histogram: HashMap::new(),
            });
        
        let latency_ms = duration.as_millis() as f64;
        component_metrics.operation_count += 1;
        
        if success {
            component_metrics.success_count += 1;
        } else {
            component_metrics.error_count += 1;
        }
        
        // Update average latency
        let total_ops = component_metrics.operation_count as f64;
        component_metrics.avg_latency_ms = (component_metrics.avg_latency_ms * (total_ops - 1.0) + latency_ms) / total_ops;
        
        // Update max latency
        if latency_ms > component_metrics.max_latency_ms {
            component_metrics.max_latency_ms = latency_ms;
        }
        
        // Update histogram
        let bucket = Self::get_latency_bucket(duration.as_millis() as u64);
        *component_metrics.latency_histogram.entry(bucket).or_insert(0) += 1;
        
        debug!("Recorded operation: {} took {:.2}ms, success: {}", component, latency_ms, success);
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> ProfilerMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Generate performance report
    pub async fn generate_report(&self) -> PerformanceReport {
        let metrics = self.metrics.read().await;
        
        PerformanceReport {
            timestamp: chrono::Utc::now().timestamp() as u64,
            system_summary: SystemSummary {
                total_components: metrics.component_metrics.len(),
                total_operations: metrics.component_metrics.values()
                    .map(|m| m.operation_count)
                    .sum(),
                avg_system_latency_ms: Self::calculate_avg_latency(&metrics.component_metrics),
                system_throughput_ops_per_sec: Self::calculate_system_throughput(&metrics.component_metrics),
                error_rate_percent: Self::calculate_error_rate(&metrics.component_metrics),
            },
            bottleneck_summary: BottleneckSummary {
                critical_bottlenecks: metrics.bottleneck_analysis.iter()
                    .filter(|b| matches!(b.severity, BottleneckSeverity::Critical))
                    .count(),
                high_impact_bottlenecks: metrics.bottleneck_analysis.iter()
                    .filter(|b| matches!(b.severity, BottleneckSeverity::High))
                    .count(),
                top_bottlenecks: metrics.bottleneck_analysis.iter()
                    .take(5)
                    .cloned()
                    .collect(),
            },
            optimization_summary: OptimizationSummary {
                total_recommendations: metrics.recommendations.len(),
                high_priority_recommendations: metrics.recommendations.iter()
                    .filter(|r| matches!(r.priority, RecommendationPriority::High | RecommendationPriority::Critical))
                    .count(),
                expected_performance_improvement: metrics.recommendations.iter()
                    .map(|r| r.expected_improvement_percent)
                    .sum::<f64>(),
                top_recommendations: metrics.recommendations.iter()
                    .take(10)
                    .cloned()
                    .collect(),
            },
            component_details: metrics.component_metrics.clone(),
            alerts: metrics.alerts.clone(),
            trends: metrics.performance_trends.clone(),
        }
    }
    
    // Internal helper methods
    
    async fn collect_system_metrics(config: &ProfilerConfig) -> SystemMetrics {
        let timestamp = chrono::Utc::now().timestamp() as u64;
        
        // In a real implementation, these would use actual system monitoring
        // For now, we'll simulate with reasonable values
        SystemMetrics {
            timestamp,
            cpu_usage_percent: Self::simulate_cpu_usage(),
            memory_usage_mb: Self::simulate_memory_usage(),
            memory_available_mb: 8192 - Self::simulate_memory_usage(), // Assume 8GB total
            gc_pressure_score: Self::simulate_gc_pressure(),
            thread_count: Self::simulate_thread_count(),
            open_file_descriptors: Self::simulate_file_descriptors(),
            network_connections: Self::simulate_network_connections(),
            io_read_mbps: if config.enable_io_profiling { Self::simulate_io_throughput() } else { 0.0 },
            io_write_mbps: if config.enable_io_profiling { Self::simulate_io_throughput() } else { 0.0 },
            network_rx_mbps: if config.enable_network_profiling { Self::simulate_network_throughput() } else { 0.0 },
            network_tx_mbps: if config.enable_network_profiling { Self::simulate_network_throughput() } else { 0.0 },
        }
    }
    
    async fn analyze_bottlenecks(metrics: &ProfilerMetrics) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();
        
        // Analyze each component for bottlenecks
        for (name, component_metrics) in &metrics.component_metrics {
            // High latency bottleneck
            if component_metrics.avg_latency_ms > 100.0 {
                bottlenecks.push(Bottleneck {
                    id: Uuid::new_v4().to_string(),
                    component: name.clone(),
                    bottleneck_type: BottleneckType::Algorithm,
                    severity: if component_metrics.avg_latency_ms > 200.0 { 
                        BottleneckSeverity::Critical 
                    } else { 
                        BottleneckSeverity::High 
                    },
                    impact_score: component_metrics.avg_latency_ms / 10.0,
                    description: format!("High average latency: {:.1}ms", component_metrics.avg_latency_ms),
                    suggested_optimizations: vec![
                        "Implement caching".to_string(),
                        "Optimize algorithm complexity".to_string(),
                        "Add connection pooling".to_string(),
                    ],
                    detected_at: chrono::Utc::now().timestamp() as u64,
                });
            }
            
            // High error rate bottleneck
            let error_rate = if component_metrics.operation_count > 0 {
                component_metrics.error_count as f64 / component_metrics.operation_count as f64 * 100.0
            } else {
                0.0
            };
            
            if error_rate > 5.0 {
                bottlenecks.push(Bottleneck {
                    id: Uuid::new_v4().to_string(),
                    component: name.clone(),
                    bottleneck_type: BottleneckType::Algorithm,
                    severity: if error_rate > 15.0 { 
                        BottleneckSeverity::Critical 
                    } else { 
                        BottleneckSeverity::Medium 
                    },
                    impact_score: error_rate,
                    description: format!("High error rate: {:.1}%", error_rate),
                    suggested_optimizations: vec![
                        "Improve error handling".to_string(),
                        "Add retry mechanisms".to_string(),
                        "Validate inputs more thoroughly".to_string(),
                    ],
                    detected_at: chrono::Utc::now().timestamp() as u64,
                });
            }
        }
        
        // System-level bottlenecks
        if metrics.system_metrics.memory_usage_mb > 1500 {
            bottlenecks.push(Bottleneck {
                id: Uuid::new_v4().to_string(),
                component: "system".to_string(),
                bottleneck_type: BottleneckType::Memory,
                severity: BottleneckSeverity::High,
                impact_score: metrics.system_metrics.memory_usage_mb as f64 / 100.0,
                description: format!("High memory usage: {}MB", metrics.system_metrics.memory_usage_mb),
                suggested_optimizations: vec![
                    "Implement memory pooling".to_string(),
                    "Optimize data structures".to_string(),
                    "Add garbage collection tuning".to_string(),
                ],
                detected_at: chrono::Utc::now().timestamp() as u64,
            });
        }
        
        bottlenecks
    }
    
    async fn check_alerts(metrics: &ProfilerMetrics, config: &ProfilerConfig) -> Vec<PerformanceAlert> {
        let mut alerts = Vec::new();
        
        // System alerts
        if metrics.system_metrics.cpu_usage_percent > config.alert_thresholds.cpu_usage_percent {
            alerts.push(PerformanceAlert {
                id: Uuid::new_v4().to_string(),
                alert_type: AlertType::HighCPU,
                severity: AlertSeverity::Warning,
                component: "system".to_string(),
                message: format!("High CPU usage: {:.1}%", metrics.system_metrics.cpu_usage_percent),
                threshold_value: config.alert_thresholds.cpu_usage_percent,
                current_value: metrics.system_metrics.cpu_usage_percent,
                triggered_at: chrono::Utc::now().timestamp() as u64,
            });
        }
        
        if metrics.system_metrics.memory_usage_mb > config.alert_thresholds.memory_usage_mb {
            alerts.push(PerformanceAlert {
                id: Uuid::new_v4().to_string(),
                alert_type: AlertType::HighMemory,
                severity: AlertSeverity::Critical,
                component: "system".to_string(),
                message: format!("High memory usage: {}MB", metrics.system_metrics.memory_usage_mb),
                threshold_value: config.alert_thresholds.memory_usage_mb as f64,
                current_value: metrics.system_metrics.memory_usage_mb as f64,
                triggered_at: chrono::Utc::now().timestamp() as u64,
            });
        }
        
        // Component alerts
        for (name, component_metrics) in &metrics.component_metrics {
            if component_metrics.avg_latency_ms > config.alert_thresholds.latency_ms as f64 {
                alerts.push(PerformanceAlert {
                    id: Uuid::new_v4().to_string(),
                    alert_type: AlertType::HighLatency,
                    severity: AlertSeverity::Warning,
                    component: name.clone(),
                    message: format!("High latency in {}: {:.1}ms", name, component_metrics.avg_latency_ms),
                    threshold_value: config.alert_thresholds.latency_ms as f64,
                    current_value: component_metrics.avg_latency_ms,
                    triggered_at: chrono::Utc::now().timestamp() as u64,
                });
            }
        }
        
        alerts
    }
    
    async fn generate_recommendations(metrics: &ProfilerMetrics) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        
        for (name, component_metrics) in &metrics.component_metrics {
            // Caching recommendation for high-latency components
            if component_metrics.avg_latency_ms > 50.0 {
                recommendations.push(OptimizationRecommendation {
                    id: Uuid::new_v4().to_string(),
                    recommendation_type: RecommendationType::Caching,
                    priority: RecommendationPriority::High,
                    component: name.clone(),
                    title: format!("Implement caching for {}", name),
                    description: format!("Component {} has high latency ({:.1}ms). Implementing caching could reduce response times significantly.", name, component_metrics.avg_latency_ms),
                    expected_improvement_percent: 40.0,
                    implementation_effort: ImplementationEffort::Medium,
                    code_changes_required: vec![
                        "Add cache layer interface".to_string(),
                        "Implement Redis/in-memory caching".to_string(),
                        "Add cache invalidation logic".to_string(),
                    ],
                });
            }
            
            // Connection pooling for database operations
            if name.contains("storage") || name.contains("database") {
                recommendations.push(OptimizationRecommendation {
                    id: Uuid::new_v4().to_string(),
                    recommendation_type: RecommendationType::ConnectionPooling,
                    priority: RecommendationPriority::Medium,
                    component: name.clone(),
                    title: format!("Add connection pooling for {}", name),
                    description: format!("Database component {} could benefit from connection pooling to reduce connection overhead.", name),
                    expected_improvement_percent: 25.0,
                    implementation_effort: ImplementationEffort::Low,
                    code_changes_required: vec![
                        "Configure connection pool".to_string(),
                        "Update database client initialization".to_string(),
                        "Add connection health checks".to_string(),
                    ],
                });
            }
        }
        
        recommendations
    }
    
    // Simulation methods (in real implementation, these would query actual system metrics)
    
    fn simulate_cpu_usage() -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(20.0..90.0)
    }
    
    fn simulate_memory_usage() -> u64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(500..2000)
    }
    
    fn simulate_gc_pressure() -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(0.1..0.8)
    }
    
    fn simulate_thread_count() -> u32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(10..50)
    }
    
    fn simulate_file_descriptors() -> u32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(50..200)
    }
    
    fn simulate_network_connections() -> u32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(5..50)
    }
    
    fn simulate_io_throughput() -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(10.0..100.0)
    }
    
    fn simulate_network_throughput() -> f64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(5.0..50.0)
    }
    
    fn get_latency_bucket(latency_ms: u64) -> u64 {
        match latency_ms {
            0..=10 => 10,
            11..=25 => 25,
            26..=50 => 50,
            51..=100 => 100,
            101..=250 => 250,
            251..=500 => 500,
            _ => 1000,
        }
    }
    
    fn calculate_avg_latency(components: &HashMap<String, ComponentMetrics>) -> f64 {
        if components.is_empty() {
            return 0.0;
        }
        
        let total_weighted_latency: f64 = components.values()
            .map(|m| m.avg_latency_ms * m.operation_count as f64)
            .sum();
        let total_operations: u64 = components.values()
            .map(|m| m.operation_count)
            .sum();
        
        if total_operations > 0 {
            total_weighted_latency / total_operations as f64
        } else {
            0.0
        }
    }
    
    fn calculate_system_throughput(components: &HashMap<String, ComponentMetrics>) -> f64 {
        components.values()
            .map(|m| m.throughput_ops_per_sec)
            .sum()
    }
    
    fn calculate_error_rate(components: &HashMap<String, ComponentMetrics>) -> f64 {
        let total_errors: u64 = components.values().map(|m| m.error_count).sum();
        let total_operations: u64 = components.values().map(|m| m.operation_count).sum();
        
        if total_operations > 0 {
            (total_errors as f64 / total_operations as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Performance report structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: u64,
    pub system_summary: SystemSummary,
    pub bottleneck_summary: BottleneckSummary,
    pub optimization_summary: OptimizationSummary,
    pub component_details: HashMap<String, ComponentMetrics>,
    pub alerts: Vec<PerformanceAlert>,
    pub trends: PerformanceTrends,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSummary {
    pub total_components: usize,
    pub total_operations: u64,
    pub avg_system_latency_ms: f64,
    pub system_throughput_ops_per_sec: f64,
    pub error_rate_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckSummary {
    pub critical_bottlenecks: usize,
    pub high_impact_bottlenecks: usize,
    pub top_bottlenecks: Vec<Bottleneck>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSummary {
    pub total_recommendations: usize,
    pub high_priority_recommendations: usize,
    pub expected_performance_improvement: f64,
    pub top_recommendations: Vec<OptimizationRecommendation>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_profiler_creation() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        
        let metrics = profiler.get_metrics().await;
        assert!(metrics.component_metrics.is_empty());
    }
    
    #[tokio::test]
    async fn test_operation_recording() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        
        profiler.record_operation("test_component", "test_operation", Duration::from_millis(50), true).await;
        
        let metrics = profiler.get_metrics().await;
        assert!(metrics.component_metrics.contains_key("test_component"));
        
        let component_metrics = &metrics.component_metrics["test_component"];
        assert_eq!(component_metrics.operation_count, 1);
        assert_eq!(component_metrics.success_count, 1);
        assert_eq!(component_metrics.avg_latency_ms, 50.0);
    }
    
    #[tokio::test]
    async fn test_profiled_operation() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        
        let result = profiler.profile_operation("test", "async_op", async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            "success"
        }).await;
        
        assert_eq!(result, "success");
        
        let metrics = profiler.get_metrics().await;
        assert!(metrics.component_metrics.contains_key("test"));
    }
    
    #[tokio::test]
    async fn test_report_generation() {
        let config = ProfilerConfig::default();
        let profiler = PerformanceProfiler::new(config);
        
        // Record some operations
        profiler.record_operation("component1", "op1", Duration::from_millis(30), true).await;
        profiler.record_operation("component2", "op2", Duration::from_millis(150), false).await;
        
        let report = profiler.generate_report().await;
        
        assert_eq!(report.system_summary.total_components, 2);
        assert_eq!(report.system_summary.total_operations, 2);
        assert!(report.system_summary.error_rate_percent > 0.0);
    }
}