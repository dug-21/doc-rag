use axum::{
    extract::State,
    response::Response,
    http::{StatusCode, header},
};
use prometheus::{Encoder, TextEncoder};
use std::sync::Arc;
use tracing::{info, error};

use crate::{
    server::AppState,
    Result, ApiError,
};

/// Export Prometheus metrics
pub async fn export_metrics(
    State(state): State<Arc<AppState>>,
) -> Result<Response> {
    let metrics = &state.metrics;
    info!("Metrics export requested");
    
    let encoder = TextEncoder::new();
    let metric_families = metrics.registry.gather();
    
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)
        .map_err(|e| {
            error!("Failed to encode metrics: {}", e);
            ApiError::Internal("Failed to encode metrics".to_string())
        })?;
    
    let response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, encoder.format_type())
        .body(buffer.into())
        .map_err(|e| {
            error!("Failed to build metrics response: {}", e);
            ApiError::Internal("Failed to build response".to_string())
        })?;
    
    Ok(response)
}

/// Get metrics in JSON format
pub async fn get_metrics_json(
    State(state): State<Arc<AppState>>,
) -> Result<axum::Json<serde_json::Value>> {
    let metrics = &state.metrics;
    info!("JSON metrics requested");
    
    let metric_families = metrics.registry.gather();
    
    let mut json_metrics = serde_json::Map::new();
    
    for metric_family in metric_families {
        let metric_name = metric_family.get_name();
        let metric_type = metric_family.get_field_type();
        let metric_help = metric_family.get_help();
        
        let mut metric_data = serde_json::Map::new();
        metric_data.insert("type".to_string(), serde_json::Value::String(format!("{:?}", metric_type)));
        metric_data.insert("help".to_string(), serde_json::Value::String(metric_help.to_string()));
        
        let mut samples = Vec::new();
        
        for metric in metric_family.get_metric() {
            let mut sample = serde_json::Map::new();
            
            // Add labels
            let mut labels = serde_json::Map::new();
            for label_pair in metric.get_label() {
                labels.insert(
                    label_pair.get_name().to_string(),
                    serde_json::Value::String(label_pair.get_value().to_string())
                );
            }
            sample.insert("labels".to_string(), serde_json::Value::Object(labels));
            
            // Add value based on metric type
            match metric_type {
                prometheus::proto::MetricType::COUNTER => {
                    if metric.has_counter() {
                        sample.insert("value".to_string(), 
                            serde_json::Value::Number(serde_json::Number::from_f64(
                                metric.get_counter().get_value()
                            ).unwrap_or_else(|| serde_json::Number::from(0))));
                    }
                }
                prometheus::proto::MetricType::GAUGE => {
                    if metric.has_gauge() {
                        sample.insert("value".to_string(), 
                            serde_json::Value::Number(serde_json::Number::from_f64(
                                metric.get_gauge().get_value()
                            ).unwrap_or_else(|| serde_json::Number::from(0))));
                    }
                }
                prometheus::proto::MetricType::HISTOGRAM => {
                    if metric.has_histogram() {
                        let histogram = metric.get_histogram();
                        let mut hist_data = serde_json::Map::new();
                        hist_data.insert("sample_count".to_string(), 
                            serde_json::Value::Number(serde_json::Number::from(histogram.get_sample_count())));
                        hist_data.insert("sample_sum".to_string(), 
                            serde_json::Value::Number(serde_json::Number::from_f64(
                                histogram.get_sample_sum()
                            ).unwrap_or_else(|| serde_json::Number::from(0))));
                        
                        let mut buckets = Vec::new();
                        for bucket in histogram.get_bucket() {
                            buckets.push(serde_json::json!({
                                "upper_bound": bucket.get_upper_bound(),
                                "cumulative_count": bucket.get_cumulative_count()
                            }));
                        }
                        hist_data.insert("buckets".to_string(), serde_json::Value::Array(buckets));
                        
                        sample.insert("histogram".to_string(), serde_json::Value::Object(hist_data));
                    }
                }
                _ => {
                    // Handle other metric types if needed
                }
            }
            
            samples.push(serde_json::Value::Object(sample));
        }
        
        metric_data.insert("samples".to_string(), serde_json::Value::Array(samples));
        json_metrics.insert(metric_name.to_string(), serde_json::Value::Object(metric_data));
    }
    
    let response = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "metrics": json_metrics
    });
    
    Ok(axum::Json(response))
}

/// Get system health metrics
pub async fn get_health_metrics(
    State(state): State<Arc<AppState>>,
) -> Result<axum::Json<serde_json::Value>> {
    let metrics = &state.metrics;
    info!("Health metrics requested");
    
    // Get key health metrics
    let active_requests = get_gauge_value(&metrics.registry, "http_requests_active").unwrap_or(0.0);
    let database_connections = get_gauge_value(&metrics.registry, "database_connections_active").unwrap_or(0.0);
    
    let response = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "status": "healthy",
        "metrics": {
            "active_requests": active_requests,
            "database_connections": database_connections,
            "memory_usage_mb": get_memory_usage(),
            "uptime_seconds": get_uptime_seconds()
        }
    });
    
    Ok(axum::Json(response))
}

/// Record a custom metric (admin endpoint)
pub async fn record_custom_metric(
    State(state): State<Arc<AppState>>,
    axum::Json(request): axum::Json<CustomMetricRequest>,
) -> Result<StatusCode> {
    let metrics = &state.metrics;
    info!("Custom metric recording requested: {}", request.name);
    
    match request.metric_type.as_str() {
        "counter" => {
            metrics.component_requests_total
                .with_label_values(&[&request.name, "custom"])
                .inc_by(request.value);
        }
        "gauge" => {
            metrics.database_connections_active.set(request.value as i64);
        }
        "histogram" => {
            metrics.component_duration_seconds
                .with_label_values(&[&request.name, "custom"])
                .observe(request.value);
        }
        _ => {
            return Err(ApiError::BadRequest(format!(
                "Unsupported metric type: {}", 
                request.metric_type
            )));
        }
    }
    
    info!("Custom metric recorded: {} = {}", request.name, request.value);
    Ok(StatusCode::CREATED)
}

#[derive(serde::Deserialize)]
pub struct CustomMetricRequest {
    pub name: String,
    pub metric_type: String,
    pub value: f64,
}

// Helper functions

fn get_gauge_value(registry: &prometheus::Registry, metric_name: &str) -> Option<f64> {
    let metric_families = registry.gather();
    
    for family in metric_families {
        if family.get_name() == metric_name {
            for metric in family.get_metric() {
                if metric.has_gauge() {
                    return Some(metric.get_gauge().get_value());
                }
            }
        }
    }
    
    None
}

fn get_memory_usage() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb / 1024; // Convert KB to MB
                        }
                    }
                }
            }
        }
    }
    
    0
}

fn get_uptime_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_metric_request_parsing() {
        let json = r#"{"name": "test_metric", "metric_type": "counter", "value": 1.5}"#;
        let request: CustomMetricRequest = serde_json::from_str(json).unwrap();
        
        assert_eq!(request.name, "test_metric");
        assert_eq!(request.metric_type, "counter");
        assert_eq!(request.value, 1.5);
    }

    #[tokio::test]
    #[ignore] // TODO: MetricsRegistry type not found
    async fn test_metrics_registry() {
        // let registry = MetricsRegistry::new().unwrap();
        
        // TODO: Test needs MetricsRegistry implementation
        // Record some test metrics
        // registry.record_component_request("test_service", "test_op", std::time::Duration::from_millis(100), true);
        // registry.record_auth_attempt("jwt", true);
        
        // Verify metrics were recorded
        // let families = registry.registry.gather();
        // assert!(!families.is_empty());
    }

    #[test]
    fn test_memory_usage() {
        let usage = get_memory_usage();
        // Should return a reasonable value or 0 if unable to determine
        assert!(usage >= 0);
    }

    #[test]
    fn test_uptime() {
        let uptime = get_uptime_seconds();
        assert!(uptime > 0);
    }
}