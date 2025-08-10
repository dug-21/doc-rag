use anyhow::{Context, Result};
use tracing::Level;
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Registry,
};

/// Initialize tracing with configurable output format
pub fn init_tracing(log_level: &str) -> Result<()> {
    // Parse log level
    let level: Level = log_level
        .parse()
        .unwrap_or_else(|_| {
            eprintln!("Invalid log level '{}', defaulting to 'info'", log_level);
            Level::INFO
        });

    // Create environment filter
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            EnvFilter::new(format!("{}={}", env!("CARGO_PKG_NAME").replace("-", "_"), level))
        });

    // Determine if we're in a container/production environment
    let is_production = std::env::var("ENVIRONMENT")
        .map(|env| env == "production" || env == "prod")
        .unwrap_or(false);

    let is_container = std::env::var("KUBERNETES_SERVICE_HOST").is_ok() ||
                       std::env::var("DOCKER_CONTAINER").is_ok();

    if is_production || is_container {
        // Production/container: JSON format for structured logging
        init_json_tracing(env_filter)
    } else {
        // Development: Pretty format for readability
        init_pretty_tracing(env_filter)
    }
}

/// Initialize JSON-formatted tracing for production
fn init_json_tracing(env_filter: EnvFilter) -> Result<()> {
    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_current_span(false)
        .flatten_event(true);

    // Initialize with OpenTelemetry if Jaeger endpoint is available
    if let Ok(jaeger_endpoint) = std::env::var("JAEGER_AGENT_ENDPOINT") {
        #[cfg(feature = "tracing")]
        {
            use tracing_opentelemetry::OpenTelemetryLayer;
            use opentelemetry::trace::TracerProvider;
            use opentelemetry_jaeger::new_agent_pipeline;

            let tracer = new_agent_pipeline()
                .with_service_name(env!("CARGO_PKG_NAME"))
                .with_endpoint(jaeger_endpoint)
                .install_batch(opentelemetry::runtime::Tokio)
                .context("Failed to initialize Jaeger tracing")?;

            let opentelemetry_layer = OpenTelemetryLayer::new(tracer);

            Registry::default()
                .with(env_filter)
                .with(json_layer)
                .with(opentelemetry_layer)
                .init();
        }

        #[cfg(not(feature = "tracing"))]
        {
            Registry::default()
                .with(env_filter)
                .with(json_layer)
                .init();
        }
    } else {
        Registry::default()
            .with(env_filter)
            .with(json_layer)
            .init();
    }

    Ok(())
}

/// Initialize pretty-formatted tracing for development
fn init_pretty_tracing(env_filter: EnvFilter) -> Result<()> {
    let pretty_layer = tracing_subscriber::fmt::layer()
        .pretty()
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_file(true)
        .with_line_number(true);

    Registry::default()
        .with(env_filter)
        .with(pretty_layer)
        .init();

    Ok(())
}

/// Create a structured logging context for request processing
pub fn create_request_span(
    request_id: &str,
    method: &str,
    path: &str,
    user_id: Option<&str>,
) -> tracing::Span {
    tracing::info_span!(
        "request_processing",
        request_id = request_id,
        method = method,
        path = path,
        user_id = user_id,
        component = "api-gateway"
    )
}

/// Create a span for component communication
pub fn create_component_span(
    component: &str,
    operation: &str,
    request_id: Option<&str>,
) -> tracing::Span {
    tracing::info_span!(
        "component_call",
        component = component,
        operation = operation,
        request_id = request_id,
        source = "api-gateway"
    )
}

/// Log performance metrics
pub fn log_performance_metrics(
    operation: &str,
    duration_ms: u64,
    success: bool,
    metadata: Option<&serde_json::Value>,
) {
    if success {
        tracing::info!(
            operation = operation,
            duration_ms = duration_ms,
            success = success,
            metadata = ?metadata,
            "Operation completed"
        );
    } else {
        tracing::warn!(
            operation = operation,
            duration_ms = duration_ms,
            success = success,
            metadata = ?metadata,
            "Operation failed"
        );
    }
}

/// Log security events
pub fn log_security_event(
    event_type: &str,
    user_id: Option<&str>,
    client_ip: &str,
    details: Option<&serde_json::Value>,
) {
    tracing::warn!(
        event_type = event_type,
        user_id = user_id,
        client_ip = client_ip,
        details = ?details,
        component = "api-gateway",
        "Security event detected"
    );
}

/// Log component health status changes
pub fn log_health_status_change(
    component: &str,
    previous_status: &str,
    new_status: &str,
    details: Option<&str>,
) {
    tracing::warn!(
        component = component,
        previous_status = previous_status,
        new_status = new_status,
        details = details,
        "Component health status changed"
    );
}

/// Shutdown tracing gracefully
pub fn shutdown_tracing() {
    #[cfg(feature = "tracing")]
    {
        use opentelemetry::global;
        global::shutdown_tracer_provider();
    }
    
    tracing::info!("Tracing shutdown completed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_init_tracing_with_valid_level() {
        // This test would typically run in isolation
        // For now, we'll just test that the function doesn't panic
        
        // Test with different log levels
        let levels = ["trace", "debug", "info", "warn", "error"];
        for level in &levels {
            // In a real test environment, we'd want to reset tracing state
            // For now, just ensure parsing works
            let parsed_level: Result<Level, _> = level.parse();
            assert!(parsed_level.is_ok());
        }
    }

    #[test]
    fn test_invalid_log_level_handling() {
        let invalid_level = "invalid";
        let parsed_level: Result<Level, _> = invalid_level.parse();
        assert!(parsed_level.is_err());
    }

    #[test]
    fn test_span_creation() {
        let span = create_request_span("req-123", "GET", "/api/v1/health", Some("user-456"));
        assert!(!span.is_disabled());
        
        let span = create_component_span("chunker", "process", Some("req-123"));
        assert!(!span.is_disabled());
    }

    #[test]
    fn test_environment_detection() {
        // Test production environment detection
        env::set_var("ENVIRONMENT", "production");
        let is_production = env::var("ENVIRONMENT")
            .map(|env| env == "production" || env == "prod")
            .unwrap_or(false);
        assert!(is_production);
        
        env::set_var("ENVIRONMENT", "development");
        let is_production = env::var("ENVIRONMENT")
            .map(|env| env == "production" || env == "prod")
            .unwrap_or(false);
        assert!(!is_production);
        
        // Clean up
        env::remove_var("ENVIRONMENT");
    }

    #[test] 
    fn test_container_detection() {
        // Test Kubernetes detection
        env::set_var("KUBERNETES_SERVICE_HOST", "10.0.0.1");
        let is_container = env::var("KUBERNETES_SERVICE_HOST").is_ok();
        assert!(is_container);
        env::remove_var("KUBERNETES_SERVICE_HOST");
        
        // Test Docker detection
        env::set_var("DOCKER_CONTAINER", "true");
        let is_container = env::var("DOCKER_CONTAINER").is_ok();
        assert!(is_container);
        env::remove_var("DOCKER_CONTAINER");
    }
}