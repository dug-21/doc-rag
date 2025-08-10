//! # System Metrics Collection
//!
//! Comprehensive metrics collection and monitoring for the integration system
//! with Prometheus integration and real-time performance tracking.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::error;

/// System-wide metrics collection
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// System uptime
    pub uptime: Duration,
    /// Total requests processed
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Peak response time
    pub peak_response_time: Duration,
    /// Current active requests
    pub active_requests: u64,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// CPU usage statistics
    pub cpu_stats: CpuStats,
    /// Network statistics
    pub network_stats: NetworkStats,
    /// Component metrics
    pub component_metrics: HashMap<String, ComponentMetrics>,
    /// Error metrics
    pub error_metrics: ErrorMetrics,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Memory usage statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage: u64,
    /// Peak memory usage in bytes
    pub peak_usage: u64,
    /// Available memory in bytes
    pub available: u64,
    /// Memory usage percentage
    pub usage_percent: f64,
    /// Garbage collection statistics
    pub gc_stats: GcStats,
}

/// CPU usage statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    /// Current CPU usage percentage
    pub current_usage: f64,
    /// Average CPU usage percentage
    pub avg_usage: f64,
    /// Peak CPU usage percentage
    pub peak_usage: f64,
    /// Number of CPU cores
    pub cores: usize,
    /// Load averages
    pub load_averages: [f64; 3], // 1min, 5min, 15min
}

/// Network statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Network errors
    pub errors: u64,
    /// Active connections
    pub active_connections: u64,
}

/// Garbage collection statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GcStats {
    /// Number of GC cycles
    pub cycles: u64,
    /// Total GC time
    pub total_time: Duration,
    /// Average GC time
    pub avg_time: Duration,
    /// Memory freed by GC
    pub memory_freed: u64,
}

/// Component-specific metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ComponentMetrics {
    /// Component name
    pub name: String,
    /// Component status
    pub status: String,
    /// Total requests to component
    pub requests: u64,
    /// Successful requests
    pub successes: u64,
    /// Failed requests
    pub failures: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Peak response time
    pub peak_response_time: Duration,
    /// Current active requests
    pub active_requests: u64,
    /// Error rate
    pub error_rate: f64,
    /// Throughput (requests per second)
    pub throughput: f64,
    /// Circuit breaker state
    pub circuit_breaker_state: String,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Error tracking metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Total errors
    pub total_errors: u64,
    /// Errors by type
    pub errors_by_type: HashMap<String, u64>,
    /// Errors by component
    pub errors_by_component: HashMap<String, u64>,
    /// Error rate (errors per minute)
    pub error_rate: f64,
    /// Most common errors
    pub top_errors: Vec<ErrorInfo>,
    /// Recent errors
    pub recent_errors: Vec<RecentError>,
}

/// Error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    /// Error type
    pub error_type: String,
    /// Error count
    pub count: u64,
    /// Error percentage
    pub percentage: f64,
}

/// Recent error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentError {
    /// Error message
    pub message: String,
    /// Component where error occurred
    pub component: String,
    /// Error timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Request ID if applicable
    pub request_id: Option<uuid::Uuid>,
}

/// Performance metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Percentile response times
    pub response_time_percentiles: ResponseTimePercentiles,
    /// Throughput statistics
    pub throughput_stats: ThroughputStats,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Cache statistics
    pub cache_stats: CacheStats,
    /// Database performance
    pub database_performance: DatabasePerformance,
}

/// Response time percentiles
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ResponseTimePercentiles {
    /// 50th percentile (median)
    pub p50: Duration,
    /// 90th percentile
    pub p90: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// 99.9th percentile
    pub p999: Duration,
}

/// Throughput statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    /// Current throughput (requests/second)
    pub current: f64,
    /// Average throughput
    pub average: f64,
    /// Peak throughput
    pub peak: f64,
    /// Throughput trend
    pub trend: ThroughputTrend,
}

/// Throughput trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThroughputTrend {
    Increasing,
    Decreasing,
    Stable,
    Unknown,
}

impl Default for ThroughputTrend {
    fn default() -> Self {
        ThroughputTrend::Unknown
    }
}

/// Resource utilization metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// Memory utilization percentage
    pub memory_utilization: f64,
    /// Disk utilization percentage
    pub disk_utilization: f64,
    /// Network utilization percentage
    pub network_utilization: f64,
    /// Thread pool utilization
    pub thread_pool_utilization: f64,
}

/// Cache statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Hit rate percentage
    pub hit_rate: f64,
    /// Cache size
    pub size: u64,
    /// Cache capacity
    pub capacity: u64,
    /// Evictions
    pub evictions: u64,
}

/// Database performance metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DatabasePerformance {
    /// Query count
    pub query_count: u64,
    /// Average query time
    pub avg_query_time: Duration,
    /// Slow queries
    pub slow_queries: u64,
    /// Connection pool stats
    pub connection_pool_stats: ConnectionPoolStats,
    /// Transaction stats
    pub transaction_stats: TransactionStats,
}

/// Connection pool statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ConnectionPoolStats {
    /// Active connections
    pub active: u64,
    /// Idle connections
    pub idle: u64,
    /// Maximum connections
    pub max: u64,
    /// Connection waits
    pub waits: u64,
    /// Average wait time
    pub avg_wait_time: Duration,
}

/// Transaction statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TransactionStats {
    /// Committed transactions
    pub commits: u64,
    /// Rolled back transactions
    pub rollbacks: u64,
    /// Average transaction time
    pub avg_transaction_time: Duration,
    /// Deadlocks
    pub deadlocks: u64,
}

/// Metrics collector and aggregator
pub struct MetricsCollector {
    /// Start time for uptime calculation
    start_time: Instant,
    /// Metrics storage
    metrics: Arc<RwLock<SystemMetrics>>,
    /// Response time samples for percentile calculation
    response_times: Arc<RwLock<Vec<Duration>>>,
    /// Collection interval
    collection_interval: Duration,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            metrics: Arc::new(RwLock::new(SystemMetrics::default())),
            response_times: Arc::new(RwLock::new(Vec::new())),
            collection_interval: Duration::from_secs(30),
        }
    }
    
    /// Start metrics collection
    pub async fn start(&self) {
        let collector = self.clone();
        tokio::spawn(async move {
            collector.collect_metrics_loop().await;
        });
        
        let collector = self.clone();
        tokio::spawn(async move {
            collector.calculate_percentiles_loop().await;
        });
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> SystemMetrics {
        let mut metrics = self.metrics.read().await.clone();
        metrics.uptime = self.start_time.elapsed();
        metrics
    }
    
    /// Record request
    pub async fn record_request(&self, component: &str, success: bool, duration: Duration) {
        let mut metrics = self.metrics.write().await;
        
        // Update system-wide metrics
        metrics.total_requests += 1;
        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }
        
        // Update average response time
        let total_time = metrics.avg_response_time.as_millis() as f64 * (metrics.total_requests - 1) as f64;
        metrics.avg_response_time = Duration::from_millis(
            ((total_time + duration.as_millis() as f64) / metrics.total_requests as f64) as u64
        );
        
        // Update peak response time
        if duration > metrics.peak_response_time {
            metrics.peak_response_time = duration;
        }
        
        // Update component metrics
        let component_metrics = metrics.component_metrics
            .entry(component.to_string())
            .or_insert_with(|| ComponentMetrics {
                name: component.to_string(),
                ..Default::default()
            });
        
        component_metrics.requests += 1;
        if success {
            component_metrics.successes += 1;
        } else {
            component_metrics.failures += 1;
        }
        
        // Update component average response time
        let comp_total_time = component_metrics.avg_response_time.as_millis() as f64 
            * (component_metrics.requests - 1) as f64;
        component_metrics.avg_response_time = Duration::from_millis(
            ((comp_total_time + duration.as_millis() as f64) / component_metrics.requests as f64) as u64
        );
        
        if duration > component_metrics.peak_response_time {
            component_metrics.peak_response_time = duration;
        }
        
        component_metrics.error_rate = component_metrics.failures as f64 / component_metrics.requests as f64;
        
        // Add to response time samples
        let mut response_times = self.response_times.write().await;
        response_times.push(duration);
        
        // Keep only recent samples (last 1000)
        if response_times.len() > 1000 {
            let len = response_times.len();
            response_times.drain(0..len - 1000);
        }
    }
    
    /// Record error
    pub async fn record_error(&self, error_type: &str, component: &str, message: &str, request_id: Option<uuid::Uuid>) {
        let mut metrics = self.metrics.write().await;
        
        metrics.error_metrics.total_errors += 1;
        
        *metrics.error_metrics.errors_by_type
            .entry(error_type.to_string())
            .or_insert(0) += 1;
        
        *metrics.error_metrics.errors_by_component
            .entry(component.to_string())
            .or_insert(0) += 1;
        
        // Add to recent errors (keep last 100)
        metrics.error_metrics.recent_errors.push(RecentError {
            message: message.to_string(),
            component: component.to_string(),
            timestamp: chrono::Utc::now(),
            request_id,
        });
        
        if metrics.error_metrics.recent_errors.len() > 100 {
            metrics.error_metrics.recent_errors.drain(0..1);
        }
    }
    
    /// Update active requests count
    pub async fn update_active_requests(&self, component: &str, active_count: u64) {
        let mut metrics = self.metrics.write().await;
        
        if let Some(component_metrics) = metrics.component_metrics.get_mut(component) {
            component_metrics.active_requests = active_count;
        }
        
        // Update system-wide active requests
        metrics.active_requests = metrics.component_metrics
            .values()
            .map(|m| m.active_requests)
            .sum();
    }
    
    /// Collect system metrics periodically
    async fn collect_metrics_loop(&self) {
        let mut interval = tokio::time::interval(self.collection_interval);
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.collect_system_metrics().await {
                error!("Failed to collect system metrics: {}", e);
            }
        }
    }
    
    /// Collect system-level metrics
    async fn collect_system_metrics(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut metrics = self.metrics.write().await;
        
        // Collect memory stats
        #[cfg(target_os = "linux")]
        {
            if let Ok(info) = sys_info::mem_info() {
                metrics.memory_stats.current_usage = (info.total - info.free) * 1024;
                metrics.memory_stats.available = info.free * 1024;
                metrics.memory_stats.usage_percent = 
                    ((info.total - info.free) as f64 / info.total as f64) * 100.0;
                
                if metrics.memory_stats.current_usage > metrics.memory_stats.peak_usage {
                    metrics.memory_stats.peak_usage = metrics.memory_stats.current_usage;
                }
            }
        }
        
        // Collect CPU stats
        #[cfg(target_os = "linux")]
        {
            if let Ok(load_avg) = sys_info::loadavg() {
                metrics.cpu_stats.load_averages = [load_avg.one, load_avg.five, load_avg.fifteen];
            }
            
            if let Ok(cpu_num) = sys_info::cpu_num() {
                metrics.cpu_stats.cores = cpu_num as usize;
            }
        }
        
        // Calculate error rate
        let time_window_minutes = 1.0; // Last minute
        metrics.error_metrics.error_rate = metrics.error_metrics.total_errors as f64 / time_window_minutes;
        
        // Update top errors
        let mut error_counts: Vec<_> = metrics.error_metrics.errors_by_type
            .iter()
            .map(|(error_type, count)| {
                ErrorInfo {
                    error_type: error_type.clone(),
                    count: *count,
                    percentage: (*count as f64 / metrics.error_metrics.total_errors as f64) * 100.0,
                }
            })
            .collect();
        
        error_counts.sort_by(|a, b| b.count.cmp(&a.count));
        metrics.error_metrics.top_errors = error_counts.into_iter().take(10).collect();
        
        Ok(())
    }
    
    /// Calculate response time percentiles
    async fn calculate_percentiles_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            self.calculate_response_time_percentiles().await;
        }
    }
    
    /// Calculate response time percentiles
    async fn calculate_response_time_percentiles(&self) {
        let response_times = self.response_times.read().await;
        
        if response_times.is_empty() {
            return;
        }
        
        let mut sorted_times: Vec<_> = response_times.iter().copied().collect();
        sorted_times.sort();
        
        let len = sorted_times.len();
        
        let p50_idx = len * 50 / 100;
        let p90_idx = len * 90 / 100;
        let p95_idx = len * 95 / 100;
        let p99_idx = len * 99 / 100;
        let p999_idx = len * 999 / 1000;
        
        let mut metrics = self.metrics.write().await;
        
        metrics.performance_metrics.response_time_percentiles = ResponseTimePercentiles {
            p50: sorted_times.get(p50_idx).copied().unwrap_or_default(),
            p90: sorted_times.get(p90_idx).copied().unwrap_or_default(),
            p95: sorted_times.get(p95_idx).copied().unwrap_or_default(),
            p99: sorted_times.get(p99_idx).copied().unwrap_or_default(),
            p999: sorted_times.get(p999_idx).copied().unwrap_or_default(),
        };
    }
    
    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> String {
        let metrics = self.get_metrics().await;
        let mut output = String::new();
        
        // System-wide metrics
        output.push_str(&format!("# HELP system_uptime_seconds System uptime in seconds\n"));
        output.push_str(&format!("# TYPE system_uptime_seconds counter\n"));
        output.push_str(&format!("system_uptime_seconds {}\n", metrics.uptime.as_secs()));
        
        output.push_str(&format!("# HELP system_requests_total Total number of requests\n"));
        output.push_str(&format!("# TYPE system_requests_total counter\n"));
        output.push_str(&format!("system_requests_total {}\n", metrics.total_requests));
        
        output.push_str(&format!("# HELP system_requests_successful Total number of successful requests\n"));
        output.push_str(&format!("# TYPE system_requests_successful counter\n"));
        output.push_str(&format!("system_requests_successful {}\n", metrics.successful_requests));
        
        output.push_str(&format!("# HELP system_requests_failed Total number of failed requests\n"));
        output.push_str(&format!("# TYPE system_requests_failed counter\n"));
        output.push_str(&format!("system_requests_failed {}\n", metrics.failed_requests));
        
        output.push_str(&format!("# HELP system_response_time_avg Average response time in milliseconds\n"));
        output.push_str(&format!("# TYPE system_response_time_avg gauge\n"));
        output.push_str(&format!("system_response_time_avg {}\n", metrics.avg_response_time.as_millis()));
        
        // Component metrics
        for (component, comp_metrics) in &metrics.component_metrics {
            let labels = format!("{{component=\"{}\"}}", component);
            
            output.push_str(&format!("component_requests_total{} {}\n", labels, comp_metrics.requests));
            output.push_str(&format!("component_requests_successful{} {}\n", labels, comp_metrics.successes));
            output.push_str(&format!("component_requests_failed{} {}\n", labels, comp_metrics.failures));
            output.push_str(&format!("component_response_time_avg{} {}\n", labels, comp_metrics.avg_response_time.as_millis()));
            output.push_str(&format!("component_active_requests{} {}\n", labels, comp_metrics.active_requests));
            output.push_str(&format!("component_error_rate{} {}\n", labels, comp_metrics.error_rate));
        }
        
        // Memory metrics
        output.push_str(&format!("system_memory_usage_bytes {}\n", metrics.memory_stats.current_usage));
        output.push_str(&format!("system_memory_available_bytes {}\n", metrics.memory_stats.available));
        output.push_str(&format!("system_memory_usage_percent {}\n", metrics.memory_stats.usage_percent));
        
        // Error metrics
        output.push_str(&format!("system_errors_total {}\n", metrics.error_metrics.total_errors));
        output.push_str(&format!("system_error_rate {}\n", metrics.error_metrics.error_rate));
        
        // Response time percentiles
        output.push_str(&format!("system_response_time_p50 {}\n", metrics.performance_metrics.response_time_percentiles.p50.as_millis()));
        output.push_str(&format!("system_response_time_p90 {}\n", metrics.performance_metrics.response_time_percentiles.p90.as_millis()));
        output.push_str(&format!("system_response_time_p95 {}\n", metrics.performance_metrics.response_time_percentiles.p95.as_millis()));
        output.push_str(&format!("system_response_time_p99 {}\n", metrics.performance_metrics.response_time_percentiles.p99.as_millis()));
        
        output
    }
}

impl Clone for MetricsCollector {
    fn clone(&self) -> Self {
        Self {
            start_time: self.start_time,
            metrics: self.metrics.clone(),
            response_times: self.response_times.clone(),
            collection_interval: self.collection_interval,
        }
    }
}

// Optional sys_info replacement for non-Linux platforms
#[cfg(not(target_os = "linux"))]
mod sys_info {
    pub struct MemInfo {
        pub total: u64,
        pub free: u64,
    }
    
    pub struct LoadAvg {
        pub one: f64,
        pub five: f64,
        pub fifteen: f64,
    }
    
    pub fn mem_info() -> Result<MemInfo, &'static str> {
        // Mock implementation for non-Linux platforms
        Ok(MemInfo { total: 8 * 1024 * 1024, free: 4 * 1024 * 1024 })
    }
    
    pub fn loadavg() -> Result<LoadAvg, &'static str> {
        // Mock implementation for non-Linux platforms
        Ok(LoadAvg { one: 0.5, five: 0.7, fifteen: 0.8 })
    }
    
    pub fn cpu_num() -> Result<u32, &'static str> {
        // Mock implementation for non-Linux platforms
        Ok(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        // Record some requests
        collector.record_request("test-component", true, Duration::from_millis(100)).await;
        collector.record_request("test-component", false, Duration::from_millis(200)).await;
        collector.record_request("another-component", true, Duration::from_millis(150)).await;
        
        let metrics = collector.get_metrics().await;
        
        assert_eq!(metrics.total_requests, 3);
        assert_eq!(metrics.successful_requests, 2);
        assert_eq!(metrics.failed_requests, 1);
        assert_eq!(metrics.component_metrics.len(), 2);
        
        let test_component = metrics.component_metrics.get("test-component").unwrap();
        assert_eq!(test_component.requests, 2);
        assert_eq!(test_component.successes, 1);
        assert_eq!(test_component.failures, 1);
        assert_eq!(test_component.error_rate, 0.5);
    }
    
    #[tokio::test]
    async fn test_error_recording() {
        let collector = MetricsCollector::new();
        
        collector.record_error("ValidationError", "gateway", "Test error", None).await;
        collector.record_error("NetworkError", "pipeline", "Connection failed", Some(uuid::Uuid::new_v4())).await;
        
        let metrics = collector.get_metrics().await;
        
        assert_eq!(metrics.error_metrics.total_errors, 2);
        assert_eq!(metrics.error_metrics.errors_by_type.get("ValidationError"), Some(&1));
        assert_eq!(metrics.error_metrics.errors_by_type.get("NetworkError"), Some(&1));
        assert_eq!(metrics.error_metrics.errors_by_component.get("gateway"), Some(&1));
        assert_eq!(metrics.error_metrics.errors_by_component.get("pipeline"), Some(&1));
        assert_eq!(metrics.error_metrics.recent_errors.len(), 2);
    }
    
    #[tokio::test]
    async fn test_prometheus_export() {
        let collector = MetricsCollector::new();
        
        collector.record_request("test", true, Duration::from_millis(100)).await;
        
        let prometheus_output = collector.export_prometheus().await;
        
        assert!(prometheus_output.contains("system_requests_total 1"));
        assert!(prometheus_output.contains("system_requests_successful 1"));
        assert!(prometheus_output.contains("component_requests_total{component=\"test\"} 1"));
    }
}
