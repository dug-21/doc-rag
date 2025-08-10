use axum::{
    extract::State,
    response::Json,
    http::StatusCode,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{info, warn};

use crate::{
    clients::ComponentClients,
    models::{HealthResponse, ComponentHealth, HealthStatus},
    Result, ApiError, API_VERSION,
};

/// Basic health check endpoint
/// Returns OK if the API gateway is running
pub async fn health_check() -> Result<Json<HealthResponse>> {
    Ok(Json(HealthResponse {
        status: HealthStatus::Healthy,
        version: API_VERSION.to_string(),
        timestamp: chrono::Utc::now(),
        uptime_seconds: get_uptime_seconds(),
        checks: vec![],
    }))
}

/// Readiness check endpoint
/// Returns OK only if all dependencies are available
pub async fn readiness_check(
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<HealthResponse>> {
    info!("Performing readiness check");

    let mut checks = Vec::new();
    let mut overall_status = HealthStatus::Healthy;

    // Check all component services
    let component_results = clients.health_check_all().await;

    // Database check
    match clients.check_database_health().await {
        Ok(_) => checks.push(ComponentHealth {
            name: "database".to_string(),
            status: HealthStatus::Healthy,
            details: Some("PostgreSQL connection OK".to_string()),
            last_check: chrono::Utc::now(),
        }),
        Err(e) => {
            warn!("Database health check failed: {}", e);
            checks.push(ComponentHealth {
                name: "database".to_string(),
                status: HealthStatus::Unhealthy,
                details: Some(format!("Database error: {}", e)),
                last_check: chrono::Utc::now(),
            });
            overall_status = HealthStatus::Unhealthy;
        }
    }

    // Redis check
    match clients.check_redis_health().await {
        Ok(_) => checks.push(ComponentHealth {
            name: "redis".to_string(),
            status: HealthStatus::Healthy,
            details: Some("Redis connection OK".to_string()),
            last_check: chrono::Utc::now(),
        }),
        Err(e) => {
            warn!("Redis health check failed: {}", e);
            checks.push(ComponentHealth {
                name: "redis".to_string(),
                status: HealthStatus::Unhealthy,
                details: Some(format!("Redis error: {}", e)),
                last_check: chrono::Utc::now(),
            });
            overall_status = HealthStatus::Unhealthy;
        }
    }

    // Component services checks
    for (component, result) in component_results {
        match result {
            Ok(_) => checks.push(ComponentHealth {
                name: component,
                status: HealthStatus::Healthy,
                details: Some("Service responding".to_string()),
                last_check: chrono::Utc::now(),
            }),
            Err(e) => {
                warn!("Component {} health check failed: {}", component, e);
                checks.push(ComponentHealth {
                    name: component,
                    status: HealthStatus::Unhealthy,
                    details: Some(format!("Service error: {}", e)),
                    last_check: chrono::Utc::now(),
                });
                overall_status = HealthStatus::Degraded;
            }
        }
    }

    let response = HealthResponse {
        status: overall_status,
        version: API_VERSION.to_string(),
        timestamp: chrono::Utc::now(),
        uptime_seconds: get_uptime_seconds(),
        checks,
    };

    match response.status {
        HealthStatus::Healthy => Ok(Json(response)),
        HealthStatus::Degraded => {
            // Return 200 but indicate degraded performance
            Ok(Json(response))
        }
        HealthStatus::Unhealthy => {
            // Return 503 Service Unavailable
            Err(ApiError::ServiceUnavailable(
                "One or more critical dependencies are unhealthy".to_string()
            ))
        }
    }
}

/// Detailed component health check
/// Returns status of each component service
pub async fn component_health(
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<Value>> {
    info!("Performing component health checks");

    let results = clients.detailed_health_check().await;
    
    Ok(Json(json!({
        "timestamp": chrono::Utc::now(),
        "components": results,
        "summary": {
            "total": results.len(),
            "healthy": results.values().filter(|v| v.get("status") == Some(&json!("healthy"))).count(),
            "unhealthy": results.values().filter(|v| v.get("status") == Some(&json!("unhealthy"))).count(),
            "degraded": results.values().filter(|v| v.get("status") == Some(&json!("degraded"))).count(),
        }
    })))
}

fn get_uptime_seconds() -> u64 {
    // Simple uptime calculation - in production, you might want to track
    // the actual start time and calculate from there
    use std::time::{SystemTime, UNIX_EPOCH};
    
    static START_TIME: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    
    let start = START_TIME.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    });

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .saturating_sub(*start)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await.unwrap();
        assert_eq!(response.0.status, HealthStatus::Healthy);
        assert_eq!(response.0.version, API_VERSION);
    }

    #[test]
    fn test_uptime_calculation() {
        let uptime1 = get_uptime_seconds();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let uptime2 = get_uptime_seconds();
        assert!(uptime2 >= uptime1);
    }
}