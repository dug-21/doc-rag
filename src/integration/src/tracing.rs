//! # Distributed Tracing System
//!
//! OpenTelemetry-based distributed tracing for end-to-end request tracking
//! across all system components with performance monitoring and debugging support.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, instrument};
use uuid::Uuid;

use crate::{Result, IntegrationError};

/// Trace context for request tracking
#[derive(Debug, Clone)]
pub struct TraceContext {
    /// Trace ID
    pub trace_id: String,
    /// Span ID
    pub span_id: String,
    /// Parent span ID
    pub parent_span_id: Option<String>,
    /// Request ID
    pub request_id: Uuid,
    /// Service name
    pub service_name: String,
    /// Operation name
    pub operation_name: String,
    /// Start timestamp
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Tags/attributes
    pub tags: HashMap<String, String>,
    /// Baggage items
    pub baggage: HashMap<String, String>,
}

/// Span information for tracking
#[derive(Debug, Clone)]
pub struct SpanInfo {
    /// Span ID
    pub span_id: String,
    /// Trace ID
    pub trace_id: String,
    /// Parent span ID
    pub parent_span_id: Option<String>,
    /// Service name
    pub service_name: String,
    /// Operation name
    pub operation_name: String,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Duration
    pub duration: Option<Duration>,
    /// Status
    pub status: SpanStatus,
    /// Tags
    pub tags: HashMap<String, String>,
    /// Events
    pub events: Vec<SpanEvent>,
    /// Logs
    pub logs: Vec<SpanLog>,
}

/// Span status enumeration
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanStatus {
    /// Span is active
    Active,
    /// Span completed successfully
    Ok,
    /// Span completed with error
    Error,
    /// Span was cancelled
    Cancelled,
    /// Span timed out
    Timeout,
}

/// Span event
#[derive(Debug, Clone)]
pub struct SpanEvent {
    /// Event name
    pub name: String,
    /// Event timestamp
    pub timestamp: Instant,
    /// Event attributes
    pub attributes: HashMap<String, String>,
}

/// Span log entry
#[derive(Debug, Clone)]
pub struct SpanLog {
    /// Log level
    pub level: String,
    /// Log message
    pub message: String,
    /// Log timestamp
    pub timestamp: Instant,
    /// Log fields
    pub fields: HashMap<String, String>,
}

/// Trace statistics
#[derive(Debug, Default, Clone)]
pub struct TraceStats {
    /// Total traces
    pub total_traces: u64,
    /// Active traces
    pub active_traces: u64,
    /// Completed traces
    pub completed_traces: u64,
    /// Error traces
    pub error_traces: u64,
    /// Average trace duration
    pub avg_trace_duration: Duration,
    /// Trace statistics by service
    pub service_stats: HashMap<String, ServiceTraceStats>,
}

/// Service-specific trace statistics
#[derive(Debug, Default, Clone)]
pub struct ServiceTraceStats {
    /// Service name
    pub service_name: String,
    /// Total spans
    pub total_spans: u64,
    /// Error spans
    pub error_spans: u64,
    /// Average span duration
    pub avg_span_duration: Duration,
    /// Request rate (spans/sec)
    pub request_rate: f64,
    /// Error rate
    pub error_rate: f64,
}

/// Distributed tracing system
pub struct TracingSystem {
    /// System ID
    id: Uuid,
    /// Configuration
    config: Arc<crate::IntegrationConfig>,
    /// Active spans
    active_spans: Arc<RwLock<HashMap<String, SpanInfo>>>,
    /// Completed traces
    completed_traces: Arc<RwLock<HashMap<String, Vec<SpanInfo>>>>,
    /// Trace statistics
    stats: Arc<RwLock<TraceStats>>,
    /// Jaeger tracer (if configured)
    #[cfg(feature = "tracing")]
    tracer: Option<opentelemetry::global::BoxedTracer>,
}

impl TracingSystem {
    /// Create new tracing system
    pub async fn new(config: Arc<crate::IntegrationConfig>) -> Result<Self> {
        let mut system = Self {
            id: Uuid::new_v4(),
            config: config.clone(),
            active_spans: Arc::new(RwLock::new(HashMap::new())),
            completed_traces: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(TraceStats::default())),
            #[cfg(feature = "tracing")]
            tracer: None,
        };
        
        // Initialize OpenTelemetry if enabled
        #[cfg(feature = "tracing")]
        {
            if let Some(jaeger_endpoint) = &config.jaeger_endpoint {
                system.tracer = Some(system.initialize_jaeger(jaeger_endpoint).await?);
            }
        }
        
        Ok(system)
    }
    
    /// Initialize tracing system
    #[instrument(skip(self))]
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing Distributed Tracing System: {}", self.id);
        
        // Configure global tracing subscriber
        self.configure_global_tracing().await?;
        
        info!("Distributed Tracing System initialized successfully");
        Ok(())
    }
    
    /// Start tracing system
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting Distributed Tracing System...");
        
        // Start trace collection
        let system = self.clone();
        tokio::spawn(async move {
            system.collect_trace_statistics().await;
        });
        
        // Start trace cleanup
        let system = self.clone();
        tokio::spawn(async move {
            system.cleanup_old_traces().await;
        });
        
        info!("Distributed Tracing System started successfully");
        Ok(())
    }
    
    /// Stop tracing system
    #[instrument(skip(self))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Distributed Tracing System...");
        
        // Finish all active spans
        let active_spans: Vec<String> = {
            let spans = self.active_spans.read().await;
            spans.keys().cloned().collect()
        };
        
        for span_id in active_spans {
            self.finish_span(&span_id, SpanStatus::Cancelled).await;
        }
        
        #[cfg(feature = "tracing")]
        {
            // Shutdown OpenTelemetry
            opentelemetry::global::shutdown_tracer_provider();
        }
        
        info!("Distributed Tracing System stopped successfully");
        Ok(())
    }
    
    /// Start a new trace
    #[instrument(skip(self))]
    pub async fn start_trace(
        &self,
        request_id: Uuid,
        service_name: &str,
        operation_name: &str,
    ) -> Result<TraceContext> {
        let trace_id = self.generate_trace_id();
        let span_id = self.generate_span_id();
        
        let context = TraceContext {
            trace_id: trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id: None,
            request_id,
            service_name: service_name.to_string(),
            operation_name: operation_name.to_string(),
            start_time: chrono::Utc::now(),
            tags: HashMap::new(),
            baggage: HashMap::new(),
        };
        
        // Create root span
        let span_info = SpanInfo {
            span_id: span_id.clone(),
            trace_id: trace_id.clone(),
            parent_span_id: None,
            service_name: service_name.to_string(),
            operation_name: operation_name.to_string(),
            start_time: Instant::now(),
            end_time: None,
            duration: None,
            status: SpanStatus::Active,
            tags: HashMap::new(),
            events: Vec::new(),
            logs: Vec::new(),
        };
        
        // Store active span
        {
            let mut active_spans = self.active_spans.write().await;
            active_spans.insert(span_id, span_info);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_traces += 1;
            stats.active_traces += 1;
        }
        
        info!("Started trace: {} for request: {}", trace_id, request_id);
        Ok(context)
    }
    
    /// Start a child span
    #[instrument(skip(self, parent_context))]
    pub async fn start_span(
        &self,
        parent_context: &TraceContext,
        service_name: &str,
        operation_name: &str,
    ) -> Result<TraceContext> {
        let span_id = self.generate_span_id();
        
        let context = TraceContext {
            trace_id: parent_context.trace_id.clone(),
            span_id: span_id.clone(),
            parent_span_id: Some(parent_context.span_id.clone()),
            request_id: parent_context.request_id,
            service_name: service_name.to_string(),
            operation_name: operation_name.to_string(),
            start_time: chrono::Utc::now(),
            tags: parent_context.tags.clone(),
            baggage: parent_context.baggage.clone(),
        };
        
        // Create child span
        let span_info = SpanInfo {
            span_id: span_id.clone(),
            trace_id: parent_context.trace_id.clone(),
            parent_span_id: Some(parent_context.span_id.clone()),
            service_name: service_name.to_string(),
            operation_name: operation_name.to_string(),
            start_time: Instant::now(),
            end_time: None,
            duration: None,
            status: SpanStatus::Active,
            tags: HashMap::new(),
            events: Vec::new(),
            logs: Vec::new(),
        };
        
        // Store active span
        {
            let mut active_spans = self.active_spans.write().await;
            active_spans.insert(span_id, span_info);
        }
        
        Ok(context)
    }
    
    /// Finish a span
    #[instrument(skip(self))]
    pub async fn finish_span(&self, span_id: &str, status: SpanStatus) {
        let span_info = {
            let mut active_spans = self.active_spans.write().await;
            active_spans.remove(span_id)
        };
        
        if let Some(mut span) = span_info {
            let end_time = Instant::now();
            span.end_time = Some(end_time);
            span.duration = Some(end_time.duration_since(span.start_time));
            span.status = status;
            
            // Store completed span
            {
                let mut completed_traces = self.completed_traces.write().await;
                completed_traces
                    .entry(span.trace_id.clone())
                    .or_insert_with(Vec::new)
                    .push(span.clone());
            }
            
            // Update statistics
            self.update_span_statistics(&span).await;
            
            info!("Finished span: {} ({})", span_id, span.operation_name);
        }
    }
    
    /// Add event to active span
    #[instrument(skip(self, attributes))]
    pub async fn add_span_event(
        &self,
        span_id: &str,
        name: &str,
        attributes: HashMap<String, String>,
    ) {
        let mut active_spans = self.active_spans.write().await;
        
        if let Some(span) = active_spans.get_mut(span_id) {
            span.events.push(SpanEvent {
                name: name.to_string(),
                timestamp: Instant::now(),
                attributes,
            });
        }
    }
    
    /// Add log to active span
    #[instrument(skip(self, fields))]
    pub async fn add_span_log(
        &self,
        span_id: &str,
        level: &str,
        message: &str,
        fields: HashMap<String, String>,
    ) {
        let mut active_spans = self.active_spans.write().await;
        
        if let Some(span) = active_spans.get_mut(span_id) {
            span.logs.push(SpanLog {
                level: level.to_string(),
                message: message.to_string(),
                timestamp: Instant::now(),
                fields,
            });
        }
    }
    
    /// Add tags to active span
    #[instrument(skip(self, tags))]
    pub async fn add_span_tags(&self, span_id: &str, tags: HashMap<String, String>) {
        let mut active_spans = self.active_spans.write().await;
        
        if let Some(span) = active_spans.get_mut(span_id) {
            span.tags.extend(tags);
        }
    }
    
    /// Get trace by ID
    pub async fn get_trace(&self, trace_id: &str) -> Option<Vec<SpanInfo>> {
        let completed_traces = self.completed_traces.read().await;
        completed_traces.get(trace_id).cloned()
    }
    
    /// Get active spans
    pub async fn get_active_spans(&self) -> HashMap<String, SpanInfo> {
        self.active_spans.read().await.clone()
    }
    
    /// Get trace statistics
    pub async fn get_stats(&self) -> TraceStats {
        self.stats.read().await.clone()
    }
    
    /// Generate trace ID
    fn generate_trace_id(&self) -> String {
        format!("{:016x}", rand::random::<u64>())
    }
    
    /// Generate span ID
    fn generate_span_id(&self) -> String {
        format!("{:016x}", rand::random::<u64>())
    }
    
    /// Configure global tracing subscriber
    async fn configure_global_tracing(&self) -> Result<()> {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
        
        let subscriber = tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer().json())
            .with(tracing_subscriber::EnvFilter::from_default_env());
        
        #[cfg(feature = "tracing")]
        let subscriber = {
            if self.tracer.is_some() {
                // Note: OpenTelemetry tracing layer temporarily disabled due to BoxedTracer clone issues
                subscriber
            } else {
                subscriber
            }
        };
        
        subscriber.try_init()
            .map_err(|e| IntegrationError::Internal(format!("Failed to initialize tracing: {}", e)))?;
        
        Ok(())
    }
    
    /// Initialize Jaeger tracer
    #[cfg(feature = "tracing")]
    async fn initialize_jaeger(&self, _endpoint: &str) -> Result<opentelemetry::global::BoxedTracer> {
        warn!("Jaeger tracing not fully implemented due to API changes in opentelemetry-jaeger");
        
        // Return a no-op tracer for now
        let tracer = opentelemetry::global::tracer("doc-rag-integration");
        
        Ok(tracer)
    }
    
    /// Update span statistics
    async fn update_span_statistics(&self, span: &SpanInfo) {
        let mut stats = self.stats.write().await;
        
        if span.status == SpanStatus::Error {
            stats.error_traces += 1;
        } else {
            stats.completed_traces += 1;
        }
        
        stats.active_traces = stats.active_traces.saturating_sub(1);
        
        // Update service statistics
        let service_stats = stats.service_stats
            .entry(span.service_name.clone())
            .or_insert_with(|| ServiceTraceStats {
                service_name: span.service_name.clone(),
                ..Default::default()
            });
        
        service_stats.total_spans += 1;
        if span.status == SpanStatus::Error {
            service_stats.error_spans += 1;
        }
        
        if let Some(duration) = span.duration {
            // Update average duration (simple moving average)
            let total_duration = service_stats.avg_span_duration.as_millis() as f64 
                * (service_stats.total_spans - 1) as f64;
            service_stats.avg_span_duration = Duration::from_millis(
                ((total_duration + duration.as_millis() as f64) / service_stats.total_spans as f64) as u64
            );
        }
        
        // Update error rate
        service_stats.error_rate = service_stats.error_spans as f64 / service_stats.total_spans as f64;
    }
    
    /// Collect trace statistics
    async fn collect_trace_statistics(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            let stats = self.stats.read().await;
            info!("Trace Statistics: {} total, {} active, {} completed, {} errors",
                stats.total_traces,
                stats.active_traces,
                stats.completed_traces,
                stats.error_traces
            );
            
            for (service, service_stats) in &stats.service_stats {
                info!("Service {}: {} spans, {:.2}% error rate, avg duration: {:?}",
                    service,
                    service_stats.total_spans,
                    service_stats.error_rate * 100.0,
                    service_stats.avg_span_duration
                );
            }
        }
    }
    
    /// Clean up old traces
    async fn cleanup_old_traces(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(3600)); // 1 hour
        
        loop {
            interval.tick().await;
            
            let cutoff = Instant::now() - Duration::from_secs(3600 * 24); // 24 hours
            let mut cleanup_count = 0;
            
            {
                let mut completed_traces = self.completed_traces.write().await;
                let trace_ids: Vec<String> = completed_traces.keys().cloned().collect();
                
                for trace_id in trace_ids {
                    if let Some(spans) = completed_traces.get(&trace_id) {
                        // Check if all spans in the trace are old
                        if spans.iter().all(|span| span.start_time < cutoff) {
                            completed_traces.remove(&trace_id);
                            cleanup_count += 1;
                        }
                    }
                }
            }
            
            if cleanup_count > 0 {
                info!("Cleaned up {} old traces", cleanup_count);
            }
        }
    }
}

impl Clone for TracingSystem {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            config: self.config.clone(),
            active_spans: self.active_spans.clone(),
            completed_traces: self.completed_traces.clone(),
            stats: self.stats.clone(),
            #[cfg(feature = "tracing")]
            tracer: None, // Simplified for compilation
        }
    }
}

/// Tracing utilities for easy span management
pub struct TracingUtils;

impl TracingUtils {
    /// Create a traced async block
    pub async fn traced<F, T>(
        tracing_system: &TracingSystem,
        parent_context: Option<&TraceContext>,
        service_name: &str,
        operation_name: &str,
        f: F,
    ) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let context = match parent_context {
            Some(parent) => tracing_system.start_span(parent, service_name, operation_name).await?,
            None => tracing_system.start_trace(
                Uuid::new_v4(),
                service_name,
                operation_name,
            ).await?,
        };
        
        let result = f.await;
        
        let status = match result {
            Ok(_) => SpanStatus::Ok,
            Err(_) => SpanStatus::Error,
        };
        
        tracing_system.finish_span(&context.span_id, status).await;
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntegrationConfig;
    
    #[tokio::test]
    async fn test_tracing_system_creation() {
        let config = Arc::new(IntegrationConfig::default());
        let tracing_system = TracingSystem::new(config).await;
        assert!(tracing_system.is_ok());
    }
    
    #[tokio::test]
    async fn test_trace_lifecycle() {
        let config = Arc::new(IntegrationConfig::default());
        let tracing_system = TracingSystem::new(config).await.unwrap();
        
        // Start trace
        let request_id = Uuid::new_v4();
        let context = tracing_system.start_trace(
            request_id,
            "test-service",
            "test-operation",
        ).await.unwrap();
        
        // Start child span
        let child_context = tracing_system.start_span(
            &context,
            "child-service",
            "child-operation",
        ).await.unwrap();
        
        // Finish spans
        tracing_system.finish_span(&child_context.span_id, SpanStatus::Ok).await;
        tracing_system.finish_span(&context.span_id, SpanStatus::Ok).await;
        
        // Check trace exists
        let trace = tracing_system.get_trace(&context.trace_id).await;
        assert!(trace.is_some());
        assert_eq!(trace.unwrap().len(), 2);
    }
    
    #[tokio::test]
    async fn test_span_events_and_logs() {
        let config = Arc::new(IntegrationConfig::default());
        let tracing_system = TracingSystem::new(config).await.unwrap();
        
        let request_id = Uuid::new_v4();
        let context = tracing_system.start_trace(
            request_id,
            "test-service",
            "test-operation",
        ).await.unwrap();
        
        // Add event
        let mut attributes = HashMap::new();
        attributes.insert("key".to_string(), "value".to_string());
        tracing_system.add_span_event(&context.span_id, "test-event", attributes).await;
        
        // Add log
        let mut fields = HashMap::new();
        fields.insert("field".to_string(), "data".to_string());
        tracing_system.add_span_log(
            &context.span_id,
            "INFO",
            "Test log message",
            fields,
        ).await;
        
        tracing_system.finish_span(&context.span_id, SpanStatus::Ok).await;
        
        let trace = tracing_system.get_trace(&context.trace_id).await.unwrap();
        assert!(!trace[0].events.is_empty());
        assert!(!trace[0].logs.is_empty());
    }
}
