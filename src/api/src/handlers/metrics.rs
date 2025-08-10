use axum::response::{IntoResponse, Response};
use axum::http::{StatusCode, header};
use std::sync::Arc;
use prometheus::{Encoder, TextEncoder};
use tracing::{debug, error};

use crate::{Result, ApiError};

/// Get Prometheus metrics
pub async fn get_metrics() -> Result<Response> {
    debug!("Serving Prometheus metrics");

    match gather_metrics().await {
        Ok(metrics_output) => {
            Ok((
                StatusCode::OK,
                [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
                metrics_output,
            ).into_response())
        }
        Err(e) => {
            error!("Failed to gather metrics: {}", e);
            Err(ApiError::Internal("Failed to gather metrics".to_string()))
        }
    }
}

async fn gather_metrics() -> Result<String> {
    // Create encoder
    let encoder = TextEncoder::new();
    
    // Gather all metrics from the default registry
    let metric_families = prometheus::gather();
    
    // Encode metrics
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)
        .map_err(|e| ApiError::Internal(format!("Failed to encode metrics: {}", e)))?;
    
    String::from_utf8(buffer)
        .map_err(|e| ApiError::Internal(format!("Invalid UTF-8 in metrics: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gather_metrics() {
        let result = gather_metrics().await;
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(!metrics.is_empty());
        // Should contain some default metrics
        assert!(metrics.contains("go_"));
    }
}