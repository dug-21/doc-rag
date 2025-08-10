//! Metrics collection and reporting for the API layer

use prometheus::{Counter, Histogram, Gauge, Registry};
use std::sync::Arc;

/// API metrics collector
#[derive(Clone)]
pub struct ApiMetrics {
    /// HTTP requests counter
    pub http_requests_total: Counter,
    /// HTTP request duration histogram
    pub http_request_duration_seconds: Histogram,
    /// Active connections gauge
    pub active_connections: Gauge,
    /// Registry for metrics
    pub registry: Arc<Registry>,
}

impl ApiMetrics {
    /// Create new API metrics instance
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = Arc::new(Registry::new());
        
        let http_requests_total = Counter::new(
            "api_http_requests_total",
            "Total number of HTTP requests"
        )?;
        
        let http_request_duration_seconds = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "api_http_request_duration_seconds",
                "HTTP request duration in seconds"
            )
        )?;
        
        let active_connections = Gauge::new(
            "api_active_connections",
            "Number of active connections"
        )?;
        
        registry.register(Box::new(http_requests_total.clone()))?;
        registry.register(Box::new(http_request_duration_seconds.clone()))?;
        registry.register(Box::new(active_connections.clone()))?;
        
        Ok(Self {
            http_requests_total,
            http_request_duration_seconds,
            active_connections,
            registry,
        })
    }
    
    /// Record HTTP request
    pub fn record_request(&self) {
        self.http_requests_total.inc();
    }
    
    /// Record request duration
    pub fn record_duration(&self, duration: f64) {
        self.http_request_duration_seconds.observe(duration);
    }
    
    /// Increment active connections
    pub fn inc_connections(&self) {
        self.active_connections.inc();
    }
    
    /// Decrement active connections
    pub fn dec_connections(&self) {
        self.active_connections.dec();
    }
}

impl Default for ApiMetrics {
    fn default() -> Self {
        Self::new().expect("Failed to create default metrics")
    }
}