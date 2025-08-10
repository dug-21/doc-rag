//! Observability module providing metrics, logging, and tracing
//! Integrates with both Query Processor and Response Generator

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::interval;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

pub mod metrics;
pub mod logging;
pub mod tracing;
pub mod alerts;

pub use metrics::*;
pub use logging::*;
pub use tracing::*;
pub use alerts::*;

/// Central observability manager
#[derive(Debug, Clone)]
pub struct ObservabilityManager {
    /// Metrics collector
    pub metrics: Arc<MetricsCollector>,
    
    /// Logging configuration
    pub logger: Arc<LoggingManager>,
    
    /// Tracing configuration
    pub tracer: Arc<TracingManager>,
    
    /// Alert manager
    pub alerts: Arc<AlertManager>,
    
    /// Performance monitor
    pub performance: Arc<PerformanceMonitor>,
}

/// Configuration for observability features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    
    /// Metrics export endpoint
    pub metrics_endpoint: Option<String>,
    
    /// Enable structured logging
    pub enable_logging: bool,
    
    /// Log level (trace, debug, info, warn, error)
    pub log_level: String,
    
    /// Enable distributed tracing
    pub enable_tracing: bool,
    
    /// Tracing service endpoint
    pub tracing_endpoint: Option<String>,
    
    /// Enable alerting
    pub enable_alerts: bool,
    
    /// Alert webhook URLs
    pub alert_webhooks: Vec<String>,
    
    /// Performance monitoring configuration
    pub performance_config: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// CPU usage threshold for alerts
    pub cpu_threshold: f64,
    
    /// Memory usage threshold for alerts
    pub memory_threshold: f64,
    
    /// Latency threshold in milliseconds
    pub latency_threshold: u64,
    
    /// Error rate threshold (0.0-1.0)
    pub error_rate_threshold: f64,
    
    /// Monitoring interval in seconds
    pub monitoring_interval: u64,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_endpoint: Some("http://localhost:9090/metrics".to_string()),
            enable_logging: true,
            log_level: "info".to_string(),
            enable_tracing: true,
            tracing_endpoint: Some("http://localhost:14268/api/traces".to_string()),
            enable_alerts: true,
            alert_webhooks: vec![],
            performance_config: PerformanceConfig {
                cpu_threshold: 80.0,
                memory_threshold: 85.0,
                latency_threshold: 100,
                error_rate_threshold: 0.05,
                monitoring_interval: 30,
            },
        }
    }
}

impl ObservabilityManager {
    /// Create new observability manager
    pub async fn new(config: ObservabilityConfig) -> Result<Self, ObservabilityError> {
        let metrics = Arc::new(MetricsCollector::new(&config).await?);
        let logger = Arc::new(LoggingManager::new(&config).await?);
        let tracer = Arc::new(TracingManager::new(&config).await?);
        let alerts = Arc::new(AlertManager::new(&config).await?);
        let performance = Arc::new(PerformanceMonitor::new(config.performance_config).await?);
        
        Ok(Self {
            metrics,
            logger,
            tracer,
            alerts,
            performance,
        })
    }
    
    /// Start observability services
    pub async fn start(&self) -> Result<(), ObservabilityError> {
        // Start metrics collection
        self.metrics.start().await?;
        
        // Initialize logging
        self.logger.initialize().await?;
        
        // Start tracing
        self.tracer.start().await?;
        
        // Start alert monitoring
        self.alerts.start().await?;
        
        // Start performance monitoring
        self.performance.start().await?;
        
        tracing::info!("Observability services started successfully");
        Ok(())
    }
    
    /// Stop observability services
    pub async fn stop(&self) -> Result<(), ObservabilityError> {
        self.performance.stop().await?;
        self.alerts.stop().await?;
        self.tracer.stop().await?;
        self.metrics.stop().await?;
        
        tracing::info!("Observability services stopped");
        Ok(())
    }
    
    /// Record a query processing event
    pub async fn record_query_event(&self, event: QueryEvent) {
        // Record metrics
        self.metrics.record_query_metrics(&event).await;
        
        // Log the event
        self.logger.log_query_event(&event).await;
        
        // Create trace span
        let span = self.tracer.create_query_span(&event).await;
        
        // Check for alerts
        if let Err(e) = self.alerts.check_query_alerts(&event).await {
            tracing::warn!("Alert check failed: {}", e);
        }
        
        drop(span); // End span
    }
    
    /// Record a response generation event
    pub async fn record_response_event(&self, event: ResponseEvent) {
        // Record metrics
        self.metrics.record_response_metrics(&event).await;
        
        // Log the event
        self.logger.log_response_event(&event).await;
        
        // Create trace span
        let span = self.tracer.create_response_span(&event).await;
        
        // Check for alerts
        if let Err(e) = self.alerts.check_response_alerts(&event).await {
            tracing::warn!("Alert check failed: {}", e);
        }
        
        drop(span); // End span
    }
    
    /// Get current system health status
    pub async fn get_health_status(&self) -> HealthStatus {
        let metrics_status = self.metrics.get_health().await;
        let performance_status = self.performance.get_current_status().await;
        let alert_status = self.alerts.get_status().await;
        
        HealthStatus {
            timestamp: chrono::Utc::now(),
            overall_status: if metrics_status.is_healthy && performance_status.is_healthy && alert_status.is_healthy {
                ServiceStatus::Healthy
            } else if metrics_status.is_degraded || performance_status.is_degraded {
                ServiceStatus::Degraded
            } else {
                ServiceStatus::Unhealthy
            },
            components: vec![
                ComponentHealth {
                    name: "metrics".to_string(),
                    status: if metrics_status.is_healthy { ServiceStatus::Healthy } else { ServiceStatus::Unhealthy },
                    details: metrics_status.details,
                },
                ComponentHealth {
                    name: "performance".to_string(),
                    status: if performance_status.is_healthy { ServiceStatus::Healthy } else { ServiceStatus::Degraded },
                    details: performance_status.details,
                },
                ComponentHealth {
                    name: "alerts".to_string(),
                    status: if alert_status.is_healthy { ServiceStatus::Healthy } else { ServiceStatus::Unhealthy },
                    details: alert_status.details,
                },
            ],
            performance_metrics: performance_status.current_metrics,
        }
    }
    
    /// Export observability data
    pub async fn export_data(&self, format: ExportFormat) -> Result<String, ObservabilityError> {
        let metrics_data = self.metrics.export_metrics().await?;
        let logs_data = self.logger.export_logs().await?;
        let traces_data = self.tracer.export_traces().await?;
        
        match format {
            ExportFormat::Json => {
                let export_data = ExportData {
                    timestamp: chrono::Utc::now(),
                    metrics: metrics_data,
                    logs: logs_data,
                    traces: traces_data,
                };
                
                serde_json::to_string_pretty(&export_data)
                    .map_err(|e| ObservabilityError::ExportError(e.to_string()))
            }
            ExportFormat::Prometheus => {
                // Convert to Prometheus format
                Ok(self.metrics.export_prometheus_format().await?)
            }
            ExportFormat::OpenTelemetry => {
                // Convert to OpenTelemetry format
                Ok(self.tracer.export_otel_format().await?)
            }
        }
    }
}

/// Query processing event for observability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryEvent {
    pub query_id: Uuid,
    pub query_text: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub duration: Duration,
    pub success: bool,
    pub confidence_score: f64,
    pub entities_extracted: usize,
    pub intent: Option<String>,
    pub validation_results: Vec<ValidationResult>,
    pub error_message: Option<String>,
    pub processing_stages: Vec<ProcessingStage>,
}

/// Response generation event for observability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseEvent {
    pub response_id: Uuid,
    pub query_id: Uuid,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub duration: Duration,
    pub success: bool,
    pub confidence_score: f64,
    pub response_length: usize,
    pub citation_count: usize,
    pub format: String,
    pub validation_results: Vec<ValidationResult>,
    pub error_message: Option<String>,
    pub generation_stages: Vec<GenerationStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub layer_name: String,
    pub passed: bool,
    pub confidence: f64,
    pub findings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStage {
    pub name: String,
    pub duration: Duration,
    pub success: bool,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStage {
    pub name: String,
    pub duration: Duration,
    pub success: bool,
    pub metrics: HashMap<String, f64>,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub overall_status: ServiceStatus,
    pub components: Vec<ComponentHealth>,
    pub performance_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: ServiceStatus,
    pub details: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Export formats for observability data
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Prometheus,
    OpenTelemetry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportData {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metrics: serde_json::Value,
    pub logs: serde_json::Value,
    pub traces: serde_json::Value,
}

/// Observability-related errors
#[derive(Debug, thiserror::Error)]
pub enum ObservabilityError {
    #[error("Metrics error: {0}")]
    MetricsError(String),
    
    #[error("Logging error: {0}")]
    LoggingError(String),
    
    #[error("Tracing error: {0}")]
    TracingError(String),
    
    #[error("Alert error: {0}")]
    AlertError(String),
    
    #[error("Performance monitoring error: {0}")]
    PerformanceError(String),
    
    #[error("Export error: {0}")]
    ExportError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
}

/// Convenience macros for instrumentation
#[macro_export]
macro_rules! observe_query {
    ($manager:expr, $query_id:expr, $query:expr, $block:block) => {{
        let start_time = std::time::Instant::now();
        let result = $block;
        let duration = start_time.elapsed();
        
        let event = QueryEvent {
            query_id: $query_id,
            query_text: $query.to_string(),
            timestamp: chrono::Utc::now(),
            duration,
            success: result.is_ok(),
            confidence_score: if let Ok(ref r) = result { r.confidence_score } else { 0.0 },
            entities_extracted: if let Ok(ref r) = result { r.entities.len() } else { 0 },
            intent: if let Ok(ref r) = result { r.intent.as_ref().map(|i| format!("{:?}", i)) } else { None },
            validation_results: vec![], // Would be populated from actual result
            error_message: if let Err(ref e) = result { Some(e.to_string()) } else { None },
            processing_stages: vec![], // Would be populated from actual processing
        };
        
        $manager.record_query_event(event).await;
        result
    }};
}

#[macro_export]
macro_rules! observe_response {
    ($manager:expr, $response_id:expr, $query_id:expr, $block:block) => {{
        let start_time = std::time::Instant::now();
        let result = $block;
        let duration = start_time.elapsed();
        
        let event = ResponseEvent {
            response_id: $response_id,
            query_id: $query_id,
            timestamp: chrono::Utc::now(),
            duration,
            success: result.is_ok(),
            confidence_score: if let Ok(ref r) = result { r.confidence_score } else { 0.0 },
            response_length: if let Ok(ref r) = result { r.content.len() } else { 0 },
            citation_count: if let Ok(ref r) = result { r.citations.len() } else { 0 },
            format: if let Ok(ref r) = result { format!("{:?}", r.format) } else { "unknown".to_string() },
            validation_results: vec![], // Would be populated from actual result
            error_message: if let Err(ref e) = result { Some(e.to_string()) } else { None },
            generation_stages: vec![], // Would be populated from actual generation
        };
        
        $manager.record_response_event(event).await;
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_observability_manager_creation() {
        let config = ObservabilityConfig::default();
        let manager = ObservabilityManager::new(config).await;
        
        assert!(manager.is_ok());
    }
    
    #[tokio::test]
    async fn test_health_status() {
        let config = ObservabilityConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();
        
        let health = manager.get_health_status().await;
        assert!(!health.components.is_empty());
    }
    
    #[tokio::test]
    async fn test_event_recording() {
        let config = ObservabilityConfig::default();
        let manager = ObservabilityManager::new(config).await.unwrap();
        
        let query_event = QueryEvent {
            query_id: Uuid::new_v4(),
            query_text: "test query".to_string(),
            timestamp: chrono::Utc::now(),
            duration: Duration::from_millis(50),
            success: true,
            confidence_score: 0.85,
            entities_extracted: 3,
            intent: Some("factual".to_string()),
            validation_results: vec![],
            error_message: None,
            processing_stages: vec![],
        };
        
        // Should not panic
        manager.record_query_event(query_event).await;
    }
}