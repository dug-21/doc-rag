use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
};
use prometheus::{
    CounterVec, HistogramVec, IntCounter, IntGauge, Opts, Registry,
};
use std::{
    sync::Arc,
    time::Instant,
};
use tower::{Layer, Service};
use tracing::warn;

/// Metrics collection middleware
#[derive(Clone)]
pub struct MetricsMiddleware {
    registry: Arc<MetricsRegistry>,
}

impl MetricsMiddleware {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(MetricsRegistry::new().expect("Failed to create metrics registry")),
        }
    }

    pub fn with_registry(registry: Arc<MetricsRegistry>) -> Self {
        Self { registry }
    }
}

impl<S> Layer<S> for MetricsMiddleware {
    type Service = MetricsService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MetricsService {
            inner,
            registry: self.registry.clone(),
        }
    }
}

#[derive(Clone)]
pub struct MetricsService<S> {
    inner: S,
    registry: Arc<MetricsRegistry>,
}

impl<S> Service<Request> for MetricsService<S>
where
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        let start_time = Instant::now();
        let method = request.method().to_string();
        let path = request.uri().path().to_string();
        
        // Normalize path for metrics (remove IDs and parameters)
        let normalized_path = normalize_path(&path);
        
        let registry = self.registry.clone();
        let future = self.inner.call(request);

        Box::pin(async move {
            // Increment request counter
            registry.requests_total
                .with_label_values(&[&method, &normalized_path])
                .inc();

            registry.active_requests.inc();

            let result = future.await;
            let duration = start_time.elapsed();

            registry.active_requests.dec();

            match &result {
                Ok(response) => {
                    let status_code = response.status().as_u16().to_string();
                    let status_class = get_status_class(response.status().as_u16());

                    // Record response time
                    registry.request_duration_seconds
                        .with_label_values(&[&method, &normalized_path, &status_class])
                        .observe(duration.as_secs_f64());

                    // Increment response counter
                    registry.responses_total
                        .with_label_values(&[&method, &normalized_path, &status_code])
                        .inc();

                    // Track content length if available
                    if let Some(content_length) = response.headers().get("content-length") {
                        if let Ok(length_str) = content_length.to_str() {
                            if let Ok(length) = length_str.parse::<f64>() {
                                registry.response_size_bytes
                                    .with_label_values(&[&method, &normalized_path])
                                    .observe(length);
                            }
                        }
                    }
                }
                Err(_) => {
                    // Record error
                    registry.request_duration_seconds
                        .with_label_values(&[&method, &normalized_path, "error"])
                        .observe(duration.as_secs_f64());

                    registry.responses_total
                        .with_label_values(&[&method, &normalized_path, "error"])
                        .inc();
                }
            }

            result
        })
    }
}

/// Centralized metrics registry
pub struct MetricsRegistry {
    pub requests_total: CounterVec,
    pub responses_total: CounterVec,
    pub request_duration_seconds: HistogramVec,
    pub response_size_bytes: HistogramVec,
    pub active_requests: IntGauge,
    pub component_requests_total: CounterVec,
    pub component_errors_total: CounterVec,
    pub component_duration_seconds: HistogramVec,
    pub database_connections_active: IntGauge,
    pub cache_hits_total: IntCounter,
    pub cache_misses_total: IntCounter,
    pub auth_attempts_total: CounterVec,
    pub auth_failures_total: CounterVec,
    pub registry: Registry,
}

impl MetricsRegistry {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let registry = Registry::new();

        let requests_total = CounterVec::new(
            Opts::new("http_requests_total", "Total number of HTTP requests")
                .const_label("service", "api-gateway"),
            &["method", "path"],
        )?;
        registry.register(Box::new(requests_total.clone()))?;

        let responses_total = CounterVec::new(
            Opts::new("http_responses_total", "Total number of HTTP responses")
                .const_label("service", "api-gateway"),
            &["method", "path", "status_code"],
        )?;
        registry.register(Box::new(responses_total.clone()))?;

        let request_duration_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new("http_request_duration_seconds", "HTTP request duration in seconds")
                .const_label("service", "api-gateway")
                .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
            &["method", "path", "status_class"],
        )?;
        registry.register(Box::new(request_duration_seconds.clone()))?;

        let response_size_bytes = HistogramVec::new(
            prometheus::HistogramOpts::new("http_response_size_bytes", "HTTP response size in bytes")
                .const_label("service", "api-gateway")
                .buckets(vec![100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]),
            &["method", "path"],
        )?;
        registry.register(Box::new(response_size_bytes.clone()))?;

        let active_requests = IntGauge::with_opts(
            Opts::new("http_requests_active", "Number of active HTTP requests")
                .const_label("service", "api-gateway"),
        )?;
        registry.register(Box::new(active_requests.clone()))?;

        let component_requests_total = CounterVec::new(
            Opts::new("component_requests_total", "Total requests to components")
                .const_label("service", "api-gateway"),
            &["component", "operation"],
        )?;
        registry.register(Box::new(component_requests_total.clone()))?;

        let component_errors_total = CounterVec::new(
            Opts::new("component_errors_total", "Total component errors")
                .const_label("service", "api-gateway"),
            &["component", "operation"],
        )?;
        registry.register(Box::new(component_errors_total.clone()))?;

        let component_duration_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new("component_request_duration_seconds", "Component request duration")
                .const_label("service", "api-gateway"),
            &["component", "operation"],
        )?;
        registry.register(Box::new(component_duration_seconds.clone()))?;

        let database_connections_active = IntGauge::with_opts(
            Opts::new("database_connections_active", "Active database connections")
                .const_label("service", "api-gateway"),
        )?;
        registry.register(Box::new(database_connections_active.clone()))?;

        let cache_hits_total = IntCounter::with_opts(
            Opts::new("cache_hits_total", "Total cache hits")
                .const_label("service", "api-gateway"),
        )?;
        registry.register(Box::new(cache_hits_total.clone()))?;

        let cache_misses_total = IntCounter::with_opts(
            Opts::new("cache_misses_total", "Total cache misses")
                .const_label("service", "api-gateway"),
        )?;
        registry.register(Box::new(cache_misses_total.clone()))?;

        let auth_attempts_total = CounterVec::new(
            Opts::new("auth_attempts_total", "Total authentication attempts")
                .const_label("service", "api-gateway"),
            &["method"],
        )?;
        registry.register(Box::new(auth_attempts_total.clone()))?;

        let auth_failures_total = CounterVec::new(
            Opts::new("auth_failures_total", "Total authentication failures")
                .const_label("service", "api-gateway"),
            &["method"],
        )?;
        registry.register(Box::new(auth_failures_total.clone()))?;

        Ok(Self {
            requests_total,
            responses_total,
            request_duration_seconds,
            response_size_bytes,
            active_requests,
            component_requests_total,
            component_errors_total,
            component_duration_seconds,
            database_connections_active,
            cache_hits_total,
            cache_misses_total,
            auth_attempts_total,
            auth_failures_total,
            registry,
        })
    }

    /// Record component request metrics
    pub fn record_component_request(
        &self,
        component: &str,
        operation: &str,
        duration: std::time::Duration,
        success: bool,
    ) {
        self.component_requests_total
            .with_label_values(&[component, operation])
            .inc();

        self.component_duration_seconds
            .with_label_values(&[component, operation])
            .observe(duration.as_secs_f64());

        if !success {
            self.component_errors_total
                .with_label_values(&[component, operation])
                .inc();
        }
    }

    /// Record authentication attempt
    pub fn record_auth_attempt(&self, method: &str, success: bool) {
        self.auth_attempts_total
            .with_label_values(&[method])
            .inc();

        if !success {
            self.auth_failures_total
                .with_label_values(&[method])
                .inc();
        }
    }

    /// Record cache hit/miss
    pub fn record_cache_access(&self, cache_type: &str, hit: bool) {
        if hit {
            self.cache_hits_total.inc();
        } else {
            self.cache_misses_total.inc();
        }
    }

    /// Update database connection count
    pub fn set_database_connections(&self, count: i64) {
        self.database_connections_active.set(count);
    }

    /// Export final metrics for shutdown
    pub async fn export_final_metrics(&self) -> Result<(), Box<dyn std::error::Error>> {
        use prometheus::{Encoder, TextEncoder};
        
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        
        let metrics_output = String::from_utf8(buffer)?;
        tracing::info!("Final metrics exported: {} bytes", metrics_output.len());
        
        Ok(())
    }
}

/// Normalize path for metrics (remove variable path segments)
fn normalize_path(path: &str) -> String {
    // Replace UUIDs and numeric IDs with placeholders
    let uuid_regex = regex::Regex::new(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}").unwrap();
    let id_regex = regex::Regex::new(r"/\d+(/|$)").unwrap();
    
    let normalized = uuid_regex.replace_all(path, "{uuid}");
    let normalized = id_regex.replace_all(&normalized, "/{id}$1");
    
    normalized.to_string()
}

/// Get status class for metrics grouping
fn get_status_class(status_code: u16) -> String {
    match status_code {
        100..=199 => "1xx".to_string(),
        200..=299 => "2xx".to_string(),
        300..=399 => "3xx".to_string(),
        400..=499 => "4xx".to_string(),
        500..=599 => "5xx".to_string(),
        _ => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path() {
        assert_eq!(
            normalize_path("/api/v1/documents/550e8400-e29b-41d4-a716-446655440000"),
            "/api/v1/documents/{uuid}"
        );
        
        assert_eq!(
            normalize_path("/api/v1/users/123/posts"),
            "/api/v1/users/{id}/posts"
        );
        
        assert_eq!(
            normalize_path("/api/v1/health"),
            "/api/v1/health"
        );
    }

    #[test]
    fn test_get_status_class() {
        assert_eq!(get_status_class(200), "2xx");
        assert_eq!(get_status_class(404), "4xx");
        assert_eq!(get_status_class(500), "5xx");
        assert_eq!(get_status_class(600), "unknown");
    }

    #[test]
    fn test_metrics_registry_creation() {
        let result = MetricsRegistry::new();
        assert!(result.is_ok());
        
        let registry = result.unwrap();
        
        // Test that metrics can be recorded without panicking
        registry.record_component_request("test", "operation", std::time::Duration::from_millis(100), true);
        registry.record_auth_attempt("jwt", true);
        registry.record_cache_access("redis", true);
        registry.set_database_connections(10);
    }

    #[tokio::test]
    async fn test_metrics_export() {
        let registry = MetricsRegistry::new().unwrap();
        
        // Record some test metrics
        registry.record_component_request("chunker", "process", std::time::Duration::from_millis(150), true);
        registry.record_auth_attempt("login", false);
        
        // Export should not fail
        let result = registry.export_final_metrics().await;
        assert!(result.is_ok());
    }
}