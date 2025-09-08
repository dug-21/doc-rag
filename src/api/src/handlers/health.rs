use axum::{
    extract::State,
    response::Json,
    http::StatusCode,
};
use serde_json::json;
use std::{sync::Arc, time::SystemTime};
use tracing::{info, warn};

use crate::{
    clients::ComponentClients,
    models::{HealthResponse, HealthStatus, ComponentHealth, ComponentStatusResponse},
    server::AppState,
    Result, VERSION,
};

/// Basic health check endpoint
pub async fn health_check(
    State(state): State<Arc<AppState>>,
) -> Result<Json<HealthResponse>> {
    let clients = &state.clients;
    info!("Health check requested");
    
    let start_time = SystemTime::now();
    let uptime = start_time
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    
    // Perform basic health checks
    let mut all_healthy = true;
    let mut checks = Vec::new();
    
    // Check database connectivity
    match clients.check_database_health().await {
        Ok(_) => {
            checks.push(ComponentHealth {
                name: "database".to_string(),
                status: HealthStatus::Healthy,
                details: Some("Connection successful".to_string()),
                last_check: chrono::Utc::now(),
            });
        }
        Err(e) => {
            all_healthy = false;
            checks.push(ComponentHealth {
                name: "database".to_string(),
                status: HealthStatus::Unhealthy,
                details: Some(format!("Connection failed: {}", e)),
                last_check: chrono::Utc::now(),
            });
        }
    }
    
    // Check Redis connectivity
    match clients.check_redis_health().await {
        Ok(_) => {
            checks.push(ComponentHealth {
                name: "redis".to_string(),
                status: HealthStatus::Healthy,
                details: Some("Connection successful".to_string()),
                last_check: chrono::Utc::now(),
            });
        }
        Err(e) => {
            warn!("Redis health check failed: {}", e);
            checks.push(ComponentHealth {
                name: "redis".to_string(),
                status: HealthStatus::Degraded,
                details: Some(format!("Connection failed: {}", e)),
                last_check: chrono::Utc::now(),
            });
            // Redis failure is not critical for basic health
        }
    }
    
    let overall_status = if all_healthy {
        HealthStatus::Healthy
    } else {
        HealthStatus::Unhealthy
    };
    
    let health_response = HealthResponse {
        status: overall_status,
        version: VERSION.to_string(),
        timestamp: chrono::Utc::now(),
        uptime_seconds: uptime,
        checks,
    };
    
    Ok(Json(health_response))
}

/// Kubernetes readiness probe endpoint
pub async fn readiness_check(
    State(state): State<Arc<AppState>>,
) -> Result<StatusCode> {
    info!("Readiness check requested");
    
    // Check if we can connect to essential services
    let clients = &state.clients;
    match clients.check_database_health().await {
        Ok(_) => {
            info!("Readiness check passed");
            Ok(StatusCode::OK)
        }
        Err(e) => {
            warn!("Readiness check failed: {}", e);
            Ok(StatusCode::SERVICE_UNAVAILABLE)
        }
    }
}

/// Kubernetes liveness probe endpoint
pub async fn liveness_check() -> Result<StatusCode> {
    // Simple liveness check - if we can respond, we're alive
    Ok(StatusCode::OK)
}

/// Detailed component health check
pub async fn component_health(
    State(state): State<Arc<AppState>>,
) -> Result<Json<ComponentStatusResponse>> {
    let clients = &state.clients;
    info!("Component health check requested");
    
    let health_results = clients.detailed_health_check().await;
    let mut healthy_count = 0;
    let mut total_count = 0;
    
    let components = health_results.into_iter().map(|(name, status_info)| {
        total_count += 1;
        
        let status = if status_info.get("status").and_then(|v| v.as_str()) == Some("healthy") {
            healthy_count += 1;
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy
        };
        
        let component_status = crate::models::ComponentStatus {
            status,
            url: status_info.get("url").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
            last_check: chrono::Utc::now(),
            response_time_ms: status_info.get("response_time_ms").and_then(|v| v.as_u64()),
            error_message: status_info.get("error").and_then(|v| v.as_str()).map(|s| s.to_string()),
            version: status_info.get("version").and_then(|v| v.as_str()).map(|s| s.to_string()),
        };
        
        (name, component_status)
    }).collect();
    
    let overall_status = if healthy_count == total_count {
        HealthStatus::Healthy
    } else if healthy_count > 0 {
        HealthStatus::Degraded
    } else {
        HealthStatus::Unhealthy
    };
    
    let response = ComponentStatusResponse {
        components,
        overall_status,
        last_updated: chrono::Utc::now(),
    };
    
    Ok(Json(response))
}

/// Simple health check that returns basic system status
pub async fn simple_health() -> Result<Json<serde_json::Value>> {
    let response = json!({
        "status": "healthy",
        "service": "doc-rag-api-gateway",
        "version": VERSION,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    Ok(Json(response))
}

/// Extended health check with system metrics
pub async fn extended_health(
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<serde_json::Value>> {
    let start_time = std::time::Instant::now();
    
    // Gather system information
    let memory_usage = get_memory_usage_mb();
    let uptime = get_uptime_seconds();
    
    // Check core dependencies
    let mut dependencies = json!({});
    
    // Database check
    match clients.check_database_health().await {
        Ok(_) => {
            dependencies["database"] = json!({
                "status": "healthy",
                "response_time_ms": start_time.elapsed().as_millis()
            });
        }
        Err(e) => {
            dependencies["database"] = json!({
                "status": "unhealthy",
                "error": e.to_string(),
                "response_time_ms": start_time.elapsed().as_millis()
            });
        }
    }
    
    // Redis check
    match clients.check_redis_health().await {
        Ok(_) => {
            dependencies["redis"] = json!({
                "status": "healthy",
                "response_time_ms": start_time.elapsed().as_millis()
            });
        }
        Err(e) => {
            dependencies["redis"] = json!({
                "status": "degraded",
                "error": e.to_string(),
                "response_time_ms": start_time.elapsed().as_millis()
            });
        }
    }
    
    let response = json!({
        "status": "healthy",
        "service": "doc-rag-api-gateway", 
        "version": VERSION,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "uptime_seconds": uptime,
        "memory_usage_mb": memory_usage,
        "dependencies": dependencies,
        "health_check_duration_ms": start_time.elapsed().as_millis()
    });
    
    Ok(Json(response))
}

// Helper functions

fn get_memory_usage_mb() -> u64 {
    // Simplified memory usage calculation for Linux
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
    
    0 // Default if we can't determine
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
    use crate::config::ApiConfig;

    #[tokio::test]
    async fn test_liveness_check() {
        let result = liveness_check().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_simple_health() {
        let result = simple_health().await;
        assert!(result.is_ok());
        
        let response = result.unwrap().0;
        assert_eq!(response["status"], "healthy");
        assert_eq!(response["service"], "doc-rag-api-gateway");
        assert_eq!(response["version"], VERSION);
    }

    #[test]
    fn test_memory_usage() {
        let usage = get_memory_usage_mb();
        // Should return a reasonable value or 0 if unable to determine
        assert!(usage >= 0);
    }

    #[test]
    fn test_uptime() {
        let uptime = get_uptime_seconds();
        assert!(uptime > 0);
    }

    #[tokio::test]
    async fn test_health_check_structure() {
        // Test that we can create a health response structure
        let health = HealthResponse {
            status: HealthStatus::Healthy,
            version: VERSION.to_string(),
            timestamp: chrono::Utc::now(),
            uptime_seconds: 3600,
            checks: vec![
                ComponentHealth {
                    name: "test".to_string(),
                    status: HealthStatus::Healthy,
                    details: Some("OK".to_string()),
                    last_check: chrono::Utc::now(),
                }
            ],
        };
        
        assert!(health.status.is_healthy());
        assert_eq!(health.version, VERSION);
        assert_eq!(health.checks.len(), 1);
    }
}