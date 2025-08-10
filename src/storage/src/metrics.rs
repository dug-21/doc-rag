//! Metrics and monitoring for MongoDB Vector Storage

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Storage metrics collector
#[derive(Debug)]
pub struct StorageMetrics {
    /// Operation counters
    operation_counters: RwLock<HashMap<String, OperationMetrics>>,
    
    /// Performance metrics
    performance_metrics: RwLock<PerformanceMetrics>,
    
    /// Error metrics
    error_metrics: RwLock<ErrorMetrics>,
    
    /// Connection metrics
    connection_metrics: RwLock<ConnectionMetrics>,
    
    /// Start time for metrics collection
    start_time: Instant,
}

/// Metrics for individual operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OperationMetrics {
    /// Total number of operations
    pub count: u64,
    
    /// Number of successful operations
    pub success_count: u64,
    
    /// Number of failed operations
    pub error_count: u64,
    
    /// Total duration of all operations
    pub total_duration_ms: u64,
    
    /// Minimum operation duration
    pub min_duration_ms: u64,
    
    /// Maximum operation duration
    pub max_duration_ms: u64,
    
    /// Last operation timestamp
    pub last_operation: Option<DateTime<Utc>>,
    
    /// Operation durations for percentile calculation (last 1000 operations)
    pub recent_durations: Vec<u64>,
}

/// Overall performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total documents processed
    pub documents_processed: u64,
    
    /// Total bytes processed
    pub bytes_processed: u64,
    
    /// Average throughput (documents per second)
    pub avg_throughput_dps: f64,
    
    /// Average throughput (bytes per second)
    pub avg_throughput_bps: f64,
    
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    
    /// Cache miss ratio
    pub cache_miss_ratio: f64,
    
    /// Total cache hits (for calculation)
    #[serde(skip)]
    pub cache_hits: u64,
    
    /// Total cache misses (for calculation)
    #[serde(skip)]
    pub cache_misses: u64,
    
    /// Index utilization statistics
    pub index_stats: HashMap<String, IndexStats>,
    
    /// Query optimization statistics
    pub query_stats: QueryStats,
}

/// Index utilization statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    /// Number of times index was used
    pub usage_count: u64,
    
    /// Average documents examined per query
    pub avg_docs_examined: f64,
    
    /// Average keys examined per query
    pub avg_keys_examined: f64,
    
    /// Index effectiveness ratio
    pub effectiveness_ratio: f64,
}

/// Query performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    /// Vector search statistics
    pub vector_search: SearchTypeStats,
    
    /// Text search statistics
    pub text_search: SearchTypeStats,
    
    /// Hybrid search statistics
    pub hybrid_search: SearchTypeStats,
    
    /// Average query planning time
    pub avg_planning_time_ms: f64,
    
    /// Average query execution time
    pub avg_execution_time_ms: f64,
}

/// Statistics for specific search types
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchTypeStats {
    /// Number of searches performed
    pub search_count: u64,
    
    /// Average search duration
    pub avg_duration_ms: f64,
    
    /// Average number of results returned
    pub avg_results_returned: f64,
    
    /// Average relevance score
    pub avg_relevance_score: f64,
    
    /// 95th percentile search duration
    pub p95_duration_ms: f64,
    
    /// 99th percentile search duration
    pub p99_duration_ms: f64,
}

/// Error tracking metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorMetrics {
    /// Error counts by type
    pub error_counts: HashMap<String, u64>,
    
    /// Error counts by operation
    pub operation_errors: HashMap<String, u64>,
    
    /// Recent errors (last 100)
    pub recent_errors: Vec<ErrorEvent>,
    
    /// Total error rate
    pub error_rate: f64,
    
    /// Errors per hour
    pub errors_per_hour: f64,
}

/// Individual error event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    /// Error type/category
    pub error_type: String,
    
    /// Operation that failed
    pub operation: String,
    
    /// Error message
    pub message: String,
    
    /// Timestamp when error occurred
    pub timestamp: DateTime<Utc>,
    
    /// Request ID if available
    pub request_id: Option<String>,
}

/// Connection pool metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectionMetrics {
    /// Current active connections
    pub active_connections: u32,
    
    /// Total connections created
    pub connections_created: u64,
    
    /// Total connections closed
    pub connections_closed: u64,
    
    /// Connection pool utilization ratio
    pub pool_utilization: f64,
    
    /// Average connection wait time
    pub avg_wait_time_ms: f64,
    
    /// Connection timeouts
    pub connection_timeouts: u64,
    
    /// Connection errors
    pub connection_errors: u64,
}

/// Comprehensive metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Uptime in seconds
    pub uptime_seconds: u64,
    
    /// Operation metrics
    pub operations: HashMap<String, OperationMetrics>,
    
    /// Performance metrics
    pub performance: PerformanceMetrics,
    
    /// Error metrics
    pub errors: ErrorMetrics,
    
    /// Connection metrics
    pub connections: ConnectionMetrics,
}

impl StorageMetrics {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            operation_counters: RwLock::new(HashMap::new()),
            performance_metrics: RwLock::new(PerformanceMetrics::default()),
            error_metrics: RwLock::new(ErrorMetrics::default()),
            connection_metrics: RwLock::new(ConnectionMetrics::default()),
            start_time: Instant::now(),
        }
    }
    
    /// Record an operation duration
    pub fn record_operation_duration(&self, operation: &str, duration: Duration) {
        let duration_ms = duration.as_millis() as u64;
        
        if let Ok(mut counters) = self.operation_counters.write() {
            let metrics = counters.entry(operation.to_string()).or_default();
            
            metrics.count += 1;
            metrics.success_count += 1;
            metrics.total_duration_ms += duration_ms;
            
            if metrics.min_duration_ms == 0 || duration_ms < metrics.min_duration_ms {
                metrics.min_duration_ms = duration_ms;
            }
            
            if duration_ms > metrics.max_duration_ms {
                metrics.max_duration_ms = duration_ms;
            }
            
            metrics.last_operation = Some(Utc::now());
            
            // Keep track of recent durations for percentile calculations
            metrics.recent_durations.push(duration_ms);
            if metrics.recent_durations.len() > 1000 {
                metrics.recent_durations.remove(0);
            }
        }
    }
    
    /// Increment operation counter
    pub fn increment_operation_count(&self, operation: &str) {
        if let Ok(mut counters) = self.operation_counters.write() {
            let metrics = counters.entry(operation.to_string()).or_default();
            metrics.count += 1;
        }
    }
    
    /// Record an operation error
    pub fn record_operation_error(&self, operation: &str, error_type: &str, message: &str) {
        // Update operation counters
        if let Ok(mut counters) = self.operation_counters.write() {
            let metrics = counters.entry(operation.to_string()).or_default();
            metrics.count += 1;
            metrics.error_count += 1;
            metrics.last_operation = Some(Utc::now());
        }
        
        // Update error metrics
        if let Ok(mut errors) = self.error_metrics.write() {
            *errors.error_counts.entry(error_type.to_string()).or_insert(0) += 1;
            *errors.operation_errors.entry(operation.to_string()).or_insert(0) += 1;
            
            // Add to recent errors
            errors.recent_errors.push(ErrorEvent {
                error_type: error_type.to_string(),
                operation: operation.to_string(),
                message: message.to_string(),
                timestamp: Utc::now(),
                request_id: None,
            });
            
            // Keep only last 100 errors
            if errors.recent_errors.len() > 100 {
                errors.recent_errors.remove(0);
            }
            
            // Update error rates
            self.update_error_rates(&mut errors);
        }
    }
    
    /// Record bytes processed
    pub fn record_bytes_processed(&self, bytes: u64) {
        if let Ok(mut perf) = self.performance_metrics.write() {
            perf.bytes_processed += bytes;
            self.update_throughput_metrics(&mut perf);
        }
    }
    
    /// Record documents processed
    pub fn record_documents_processed(&self, count: u64) {
        if let Ok(mut perf) = self.performance_metrics.write() {
            perf.documents_processed += count;
            self.update_throughput_metrics(&mut perf);
        }
    }
    
    /// Record cache hit
    pub fn record_cache_hit(&self) {
        if let Ok(mut perf) = self.performance_metrics.write() {
            perf.cache_hits += 1;
            let total_ops = perf.cache_hits + perf.cache_misses;
            if total_ops > 0 {
                perf.cache_hit_ratio = perf.cache_hits as f64 / total_ops as f64;
                perf.cache_miss_ratio = perf.cache_misses as f64 / total_ops as f64;
            }
        }
    }
    
    /// Record cache miss
    pub fn record_cache_miss(&self) {
        if let Ok(mut perf) = self.performance_metrics.write() {
            perf.cache_misses += 1;
            let total_ops = perf.cache_hits + perf.cache_misses;
            if total_ops > 0 {
                perf.cache_hit_ratio = perf.cache_hits as f64 / total_ops as f64;
                perf.cache_miss_ratio = perf.cache_misses as f64 / total_ops as f64;
            }
        }
    }
    
    /// Record search metrics
    pub fn record_search_metrics(&self, search_type: &str, duration: Duration, results_count: usize, avg_score: f32) {
        if let Ok(mut perf) = self.performance_metrics.write() {
            let duration_ms = duration.as_millis() as f64;
            
            let stats = match search_type {
                "vector" => &mut perf.query_stats.vector_search,
                "text" => &mut perf.query_stats.text_search,
                "hybrid" => &mut perf.query_stats.hybrid_search,
                _ => return,
            };
            
            // Update search statistics
            let count = stats.search_count;
            stats.search_count += 1;
            
            // Update averages using incremental formula
            stats.avg_duration_ms = (stats.avg_duration_ms * count as f64 + duration_ms) / (count + 1) as f64;
            stats.avg_results_returned = (stats.avg_results_returned * count as f64 + results_count as f64) / (count + 1) as f64;
            stats.avg_relevance_score = (stats.avg_relevance_score * count as f64 + avg_score as f64) / (count + 1) as f64;
        }
    }
    
    /// Update connection metrics
    pub fn update_connection_metrics(&self, active: u32, created: u64, closed: u64) {
        if let Ok(mut conn) = self.connection_metrics.write() {
            conn.active_connections = active;
            conn.connections_created = created;
            conn.connections_closed = closed;
            
            // Calculate pool utilization (assuming max pool size of 10 by default)
            conn.pool_utilization = active as f64 / 10.0;
        }
    }
    
    /// Get a snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        let timestamp = Utc::now();
        let uptime_seconds = self.start_time.elapsed().as_secs();
        
        let operations = self.operation_counters.read().unwrap().clone();
        let performance = self.performance_metrics.read().unwrap().clone();
        let errors = self.error_metrics.read().unwrap().clone();
        let connections = self.connection_metrics.read().unwrap().clone();
        
        MetricsSnapshot {
            timestamp,
            uptime_seconds,
            operations,
            performance,
            errors,
            connections,
        }
    }
    
    /// Get operation metrics for a specific operation
    pub fn get_operation_metrics(&self, operation: &str) -> Option<OperationMetrics> {
        self.operation_counters.read().ok()?
            .get(operation)
            .cloned()
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        self.performance_metrics.read().unwrap().clone()
    }
    
    /// Get error metrics
    pub fn get_error_metrics(&self) -> ErrorMetrics {
        self.error_metrics.read().unwrap().clone()
    }
    
    /// Reset all metrics
    pub fn reset(&self) {
        if let Ok(mut counters) = self.operation_counters.write() {
            counters.clear();
        }
        
        if let Ok(mut perf) = self.performance_metrics.write() {
            *perf = PerformanceMetrics::default();
        }
        
        if let Ok(mut errors) = self.error_metrics.write() {
            *errors = ErrorMetrics::default();
        }
        
        if let Ok(mut conn) = self.connection_metrics.write() {
            *conn = ConnectionMetrics::default();
        }
    }
    
    /// Update throughput metrics
    fn update_throughput_metrics(&self, perf: &mut PerformanceMetrics) {
        let elapsed = self.start_time.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        
        // Use at least 1 millisecond to avoid division by zero
        let elapsed_for_calc = if elapsed_seconds > 0.0 {
            elapsed_seconds
        } else {
            0.001 // 1ms minimum to prevent division by zero
        };
        
        perf.avg_throughput_dps = perf.documents_processed as f64 / elapsed_for_calc;
        perf.avg_throughput_bps = perf.bytes_processed as f64 / elapsed_for_calc;
    }
    
    /// Update error rates
    fn update_error_rates(&self, errors: &mut ErrorMetrics) {
        let total_operations: u64 = if let Ok(counters) = self.operation_counters.read() {
            counters.values().map(|m| m.count).sum()
        } else {
            0
        };
        
        let total_errors: u64 = errors.error_counts.values().sum();
        
        if total_operations > 0 {
            errors.error_rate = total_errors as f64 / total_operations as f64;
        }
        
        // Calculate errors per hour based on recent errors
        let one_hour_ago = Utc::now() - chrono::Duration::hours(1);
        let recent_error_count = errors.recent_errors
            .iter()
            .filter(|e| e.timestamp >= one_hour_ago)
            .count();
        
        errors.errors_per_hour = recent_error_count as f64;
    }
}

impl OperationMetrics {
    /// Calculate average duration
    pub fn avg_duration_ms(&self) -> f64 {
        if self.count > 0 {
            self.total_duration_ms as f64 / self.count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.count > 0 {
            self.success_count as f64 / self.count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate error rate
    pub fn error_rate(&self) -> f64 {
        if self.count > 0 {
            self.error_count as f64 / self.count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate 95th percentile duration
    pub fn p95_duration_ms(&self) -> f64 {
        self.percentile(0.95)
    }
    
    /// Calculate 99th percentile duration
    pub fn p99_duration_ms(&self) -> f64 {
        self.percentile(0.99)
    }
    
    /// Calculate percentile from recent durations
    fn percentile(&self, p: f64) -> f64 {
        if self.recent_durations.is_empty() {
            return 0.0;
        }
        
        let mut sorted = self.recent_durations.clone();
        sorted.sort_unstable();
        
        let index = (p * (sorted.len() - 1) as f64) as usize;
        sorted[index] as f64
    }
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_operation_metrics() {
        let metrics = StorageMetrics::new();
        
        // Record some operations
        metrics.record_operation_duration("insert", Duration::from_millis(100));
        metrics.record_operation_duration("insert", Duration::from_millis(200));
        metrics.record_operation_error("insert", "validation", "Invalid data");
        
        let insert_metrics = metrics.get_operation_metrics("insert").unwrap();
        assert_eq!(insert_metrics.count, 3);
        assert_eq!(insert_metrics.success_count, 2);
        assert_eq!(insert_metrics.error_count, 1);
        assert_eq!(insert_metrics.avg_duration_ms(), 100.0); // (100 + 200) / 2
    }
    
    #[test]
    fn test_performance_metrics() {
        let metrics = StorageMetrics::new();
        
        // Record some processing
        metrics.record_documents_processed(100);
        metrics.record_bytes_processed(1024);
        
        thread::sleep(Duration::from_millis(100)); // Allow some time to pass
        
        let perf = metrics.get_performance_metrics();
        assert_eq!(perf.documents_processed, 100);
        assert_eq!(perf.bytes_processed, 1024);
        assert!(perf.avg_throughput_dps > 0.0, "Document throughput should be > 0, got {}", perf.avg_throughput_dps);
        assert!(perf.avg_throughput_bps > 0.0, "Byte throughput should be > 0, got {}", perf.avg_throughput_bps);
        
        // Test incrementing counters
        metrics.record_documents_processed(50);
        metrics.record_bytes_processed(512);
        
        let updated_perf = metrics.get_performance_metrics();
        assert_eq!(updated_perf.documents_processed, 150);
        assert_eq!(updated_perf.bytes_processed, 1536);
        
        // Throughput should be positive and reasonable
        // Note: Throughput might decrease if more time has elapsed relative to documents added
        assert!(updated_perf.avg_throughput_dps > 0.0, 
                "Document throughput should still be positive, got {}", updated_perf.avg_throughput_dps);
        assert!(updated_perf.avg_throughput_bps > 0.0,
                "Byte throughput should still be positive, got {}", updated_perf.avg_throughput_bps);
    }
    
    #[test]
    fn test_cache_metrics() {
        let metrics = StorageMetrics::new();
        
        // Record cache hits and misses
        metrics.record_cache_hit();
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        
        let perf = metrics.get_performance_metrics();
        
        // Check that cache metrics are properly calculated
        // With 2 hits and 1 miss, hit ratio should be 2/3 ≈ 0.67
        let expected_hit_ratio = 2.0 / 3.0;
        let expected_miss_ratio = 1.0 / 3.0;
        
        assert!((perf.cache_hit_ratio - expected_hit_ratio).abs() < 0.01, 
                "Expected hit ratio ~{}, got {}", expected_hit_ratio, perf.cache_hit_ratio);
        assert!((perf.cache_miss_ratio - expected_miss_ratio).abs() < 0.01,
                "Expected miss ratio ~{}, got {}", expected_miss_ratio, perf.cache_miss_ratio);
        
        // Test that hit + miss ratio equals 1.0
        let total_ratio = perf.cache_hit_ratio + perf.cache_miss_ratio;
        assert!((total_ratio - 1.0).abs() < 0.01, 
                "Hit + miss ratio should equal 1.0, got {}", total_ratio);
    }
    
    #[test]
    fn test_search_metrics() {
        let metrics = StorageMetrics::new();
        
        metrics.record_search_metrics("vector", Duration::from_millis(50), 10, 0.8);
        metrics.record_search_metrics("vector", Duration::from_millis(100), 5, 0.6);
        
        let perf = metrics.get_performance_metrics();
        assert_eq!(perf.query_stats.vector_search.search_count, 2);
        assert_eq!(perf.query_stats.vector_search.avg_duration_ms, 75.0);
        assert_eq!(perf.query_stats.vector_search.avg_results_returned, 7.5);
    }
    
    #[test]
    fn test_metrics_snapshot() {
        let metrics = StorageMetrics::new();
        
        metrics.record_operation_duration("test", Duration::from_millis(100));
        metrics.record_documents_processed(50);
        
        let snapshot = metrics.snapshot();
        assert!(!snapshot.operations.is_empty());
        assert_eq!(snapshot.performance.documents_processed, 50);
        assert!(snapshot.uptime_seconds >= 0);
    }
    
    #[test]
    fn test_percentile_calculations() {
        let mut metrics = OperationMetrics::default();
        
        // Add some durations
        for i in 1..=100 {
            metrics.recent_durations.push(i);
        }
        
        assert_eq!(metrics.p95_duration_ms(), 95.0);
        assert_eq!(metrics.p99_duration_ms(), 99.0);
    }
}