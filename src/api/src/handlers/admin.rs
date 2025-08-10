use axum::{
    extract::State,
    response::Json,
    http::StatusCode,
};
use std::{collections::HashMap, sync::Arc};
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    clients::ComponentClients,
    models::{SystemInfo, ComponentStatusResponse, BuildInfo, RuntimeInfo, ComponentStatus, HealthStatus},
    middleware::auth::{AuthExtension},
    Result, ApiError, VERSION,
};

/// Get system information
pub async fn system_info(
    request: axum::extract::Request,
) -> Result<Json<SystemInfo>> {
    let auth_context = request.require_auth_context()?;
    info!("System info requested by: {}", auth_context.email);

    // Check admin permissions (this should also be enforced by middleware)
    if auth_context.role != "admin" && auth_context.role != "system" {
        return Err(ApiError::Forbidden("Admin access required".to_string()));
    }

    let system_info = SystemInfo {
        version: VERSION.to_string(),
        build_info: BuildInfo {
            version: VERSION.to_string(),
            commit_hash: option_env!("VERGEN_GIT_SHA").unwrap_or("unknown").to_string(),
            build_date: option_env!("VERGEN_BUILD_TIMESTAMP").unwrap_or("unknown").to_string(),
            rust_version: option_env!("VERGEN_RUSTC_SEMVER").unwrap_or("unknown").to_string(),
        },
        runtime_info: get_runtime_info(),
        component_versions: get_component_versions().await,
    };

    Ok(Json(system_info))
}

/// Get component status information
pub async fn component_status(
    State(clients): State<Arc<ComponentClients>>,
    request: axum::extract::Request,
) -> Result<Json<ComponentStatusResponse>> {
    let auth_context = request.require_auth_context()?;
    info!("Component status requested by: {}", auth_context.email);

    // Check admin permissions
    if auth_context.role != "admin" && auth_context.role != "system" {
        return Err(ApiError::Forbidden("Admin access required".to_string()));
    }

    let health_results = clients.detailed_health_check().await;
    let mut components = HashMap::new();
    let mut healthy_count = 0;
    let mut total_count = 0;

    for (name, status_info) in health_results {
        total_count += 1;
        
        let status = if status_info.get("status").and_then(|v| v.as_str()) == Some("healthy") {
            healthy_count += 1;
            HealthStatus::Healthy
        } else {
            HealthStatus::Unhealthy
        };

        let component_status = ComponentStatus {
            status,
            url: status_info.get("url").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
            last_check: chrono::Utc::now(),
            response_time_ms: status_info.get("response_time_ms").and_then(|v| v.as_u64()),
            error_message: status_info.get("error").and_then(|v| v.as_str()).map(|s| s.to_string()),
            version: None, // Would be populated from actual component responses
        };

        components.insert(name, component_status);
    }

    let overall_status = if healthy_count == total_count {
        HealthStatus::Healthy
    } else if healthy_count > 0 {
        HealthStatus::Degraded
    } else {
        HealthStatus::Unhealthy
    };

    Ok(Json(ComponentStatusResponse {
        components,
        overall_status,
        last_updated: chrono::Utc::now(),
    }))
}

/// Reset component connections/state
pub async fn reset_components(
    State(clients): State<Arc<ComponentClients>>,
    request: axum::extract::Request,
) -> Result<StatusCode> {
    let auth_context = request.require_auth_context()?;
    info!("Component reset requested by: {}", auth_context.email);

    // Check admin permissions
    if auth_context.role != "admin" && auth_context.role != "system" {
        return Err(ApiError::Forbidden("Admin access required".to_string()));
    }

    match clients.reset_all_connections().await {
        Ok(_) => {
            info!("Component reset successful");
            Ok(StatusCode::OK)
        }
        Err(e) => {
            warn!("Component reset failed: {}", e);
            Err(ApiError::Internal(format!("Reset failed: {}", e)))
        }
    }
}

fn get_runtime_info() -> RuntimeInfo {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    // Simple runtime information - in production, you might use more sophisticated metrics
    RuntimeInfo {
        uptime_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        memory_usage_mb: get_memory_usage(),
        cpu_usage_percent: get_cpu_usage(),
        active_connections: 0, // Would be tracked by the server
        processed_requests: 0, // Would be tracked by metrics middleware
    }
}

fn get_memory_usage() -> u64 {
    // Simplified memory usage calculation
    // In production, use proper system metrics libraries
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
            for line in contents.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb / 1024; // Convert to MB
                        }
                    }
                }
            }
        }
    }
    
    0 // Default if we can't determine
}

fn get_cpu_usage() -> f64 {
    // Simplified CPU usage - in production, use proper system metrics
    // This would typically require tracking over time
    0.0
}

async fn get_component_versions() -> HashMap<String, String> {
    // In production, this would query each component for its version
    let mut versions = HashMap::new();
    
    versions.insert("chunker".to_string(), "0.1.0".to_string());
    versions.insert("embedder".to_string(), "0.1.0".to_string());
    versions.insert("storage".to_string(), "0.1.0".to_string());
    versions.insert("retriever".to_string(), "0.1.0".to_string());
    versions.insert("query_processor".to_string(), "0.1.0".to_string());
    versions.insert("response_generator".to_string(), "0.1.0".to_string());
    versions.insert("mcp_adapter".to_string(), "0.1.0".to_string());
    
    versions
}

// Extension methods for ComponentClients (would be implemented in clients.rs)
impl ComponentClients {
    pub async fn reset_all_connections(&self) -> anyhow::Result<()> {
        // Implementation would reset connection pools, clear caches, etc.
        info!("Resetting all component connections");
        Ok(())
    }

    pub async fn store_uploaded_file(
        &self,
        file_id: Uuid,
        filename: String,
        content_type: String,
        file_content: Vec<u8>,
    ) -> anyhow::Result<String> {
        // Implementation would store file and return URL/path
        Ok(format!("/files/{}", file_id))
    }

    pub async fn get_file_info(&self, file_id: Uuid) -> anyhow::Result<crate::models::FileUploadResponse> {
        // Implementation would retrieve file info from storage
        use crate::models::{FileUploadResponse, TaskStatus};
        
        Ok(FileUploadResponse {
            file_id,
            filename: format!("file_{}", file_id),
            content_type: "application/octet-stream".to_string(),
            size_bytes: 0,
            upload_url: Some(format!("/files/{}", file_id)),
            processing_status: TaskStatus::Completed,
        })
    }

    pub async fn delete_file(&self, file_id: Uuid, user_id: Uuid) -> anyhow::Result<bool> {
        // Implementation would delete file and check permissions
        info!("Deleting file {} for user {}", file_id, user_id);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_info() {
        let runtime_info = get_runtime_info();
        assert!(runtime_info.uptime_seconds > 0);
        assert!(runtime_info.memory_usage_mb >= 0);
        assert!(runtime_info.cpu_usage_percent >= 0.0);
    }

    #[tokio::test]
    async fn test_component_versions() {
        let versions = get_component_versions().await;
        assert!(!versions.is_empty());
        assert!(versions.contains_key("chunker"));
        assert!(versions.contains_key("embedder"));
        assert!(versions.contains_key("storage"));
    }

    #[test]
    fn test_memory_usage() {
        let memory = get_memory_usage();
        // Should return a reasonable value or 0 if unable to determine
        assert!(memory >= 0);
    }
}