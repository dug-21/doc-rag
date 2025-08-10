use axum::{
    extract::{Multipart, Path, Query, State},
    http::StatusCode,
    response::{Json, Sse},
    routing::{get, post},
    Router,
};
use std::sync::Arc;
// tokio-stream dependency needs to be added
// use tokio_stream::StreamExt;
use uuid::Uuid;

use crate::{
    config::ApiConfig,
    handlers::{
        documents::{ingest_document, batch_ingest_documents, get_document_status},
        queries::{process_query, stream_query_response, get_query_history},
        health::{health_check, readiness_check, component_health},
        metrics::get_metrics,
        auth::{login, logout, refresh_token, user_info},
        files::{upload_file, get_file, delete_file},
        admin::{system_info, component_status, reset_components},
    },
    middleware::{auth::AuthMiddleware, metrics::MetricsMiddleware},
    models::{
        IngestRequest, IngestResponse, QueryRequest, QueryResponse,
        BatchIngestRequest, BatchIngestResponse, LoginRequest, AuthResponse,
        HealthResponse, SystemInfo, ComponentStatusResponse
    },
    security::RateLimitLayer,
    ApiError, Result,
};

pub fn create_routes(config: Arc<ApiConfig>) -> Router {
    Router::new()
        // Health and system endpoints (no auth required)
        .route("/health", get(health_check))
        .route("/health/ready", get(readiness_check))
        .route("/health/components", get(component_health))
        .route("/metrics", get(get_metrics))
        
        // Authentication endpoints
        .route("/auth/login", post(login))
        .route("/auth/logout", post(logout))
        .route("/auth/refresh", post(refresh_token))
        .route("/auth/me", get(user_info))
        
        // Document ingestion endpoints
        .route("/ingest", post(ingest_document))
        .route("/ingest/batch", post(batch_ingest_documents))
        .route("/ingest/status/:task_id", get(get_document_status))
        
        // File upload endpoints
        .route("/files/upload", post(upload_file))
        .route("/files/:file_id", get(get_file))
        .route("/files/:file_id", axum::routing::delete(delete_file))
        
        // Query processing endpoints
        .route("/query", post(process_query))
        .route("/query/stream", post(stream_query_response))
        .route("/query/history", get(get_query_history))
        
        // Administrative endpoints (requires admin role)
        .route("/admin/system", get(system_info))
        .route("/admin/components", get(component_status))
        .route("/admin/components/reset", post(reset_components))
        
        // Add middleware layers
        .layer(AuthMiddleware::new(config.clone()))
        .layer(MetricsMiddleware::new())
        .layer(RateLimitLayer::new(&config.security))
        .with_state(config)
}

// Route handlers are defined in separate handler modules for better organization

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_config() -> Arc<ApiConfig> {
        Arc::new(ApiConfig::default())
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_routes(test_config());
        
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_metrics_endpoint() {
        let app = create_routes(test_config());
        
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_endpoints_exist() {
        let app = create_routes(test_config());
        
        // Test login endpoint exists (will fail auth but endpoint should exist)
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/auth/login")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();

        // Should not be 404 (not found)
        assert_ne!(response.status(), StatusCode::NOT_FOUND);
    }
}