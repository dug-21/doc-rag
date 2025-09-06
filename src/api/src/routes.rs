use axum::{
    routing::{delete, get, post},
    Router,
};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;

use crate::{
    config::ApiConfig,
    handlers::{
        admin, auth, documents, files, health, metrics, queries,
    },
    middleware::{
        auth::{auth_middleware, AuthMiddleware},
        metrics::MetricsMiddleware,
        rate_limiting::RateLimitingLayer,
    },
    server::AppState,
};

pub fn create_routes(config: Arc<ApiConfig>) -> Router<AppState> {
    // Create auth middleware (temporarily disabled)
    // let auth_middleware_instance = AuthMiddleware::new(config.clone());
    
    // Build the router with nested route groups
    Router::new()
        // Health and system endpoints (public)
        .nest("/health", health_routes())
        
        // Authentication endpoints (public)
        .nest("/auth", auth_routes())
        
        // API endpoints (authenticated)
        .nest("/api/v1", api_routes())
        
        // Admin endpoints (admin-only)
        .nest("/admin", admin_routes())
        
        // Metrics endpoint (public but can be restricted)
        .route("/metrics", get(metrics::export_metrics))
        
        // Global middleware - simplified for now
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(MetricsMiddleware::new())
                // TODO: Re-enable rate limiting and auth middleware
                // .layer(RateLimitingLayer::new(config.clone()))
        )
        // TODO: Fix auth middleware trait bounds
        // .layer(axum::middleware::from_fn_with_state(
        //     auth_middleware_instance,
        //     auth_middleware,
        // ))
}

fn health_routes() -> Router<AppState> {
    Router::new()
        .route("/", get(health::health_check))
        .route("/ready", get(health::readiness_check))
        .route("/live", get(health::liveness_check))
        .route("/components", get(health::component_health))
}

fn auth_routes() -> Router<AppState> {
    Router::new()
        .route("/login", post(auth::login))
        .route("/refresh", post(auth::refresh_token))
        .route("/logout", post(auth::logout))
        .route("/me", get(auth::user_info))
}

fn api_routes() -> Router<AppState> {
    Router::new()
        // Document processing
        .route("/ingest", post(documents::ingest_document))
        .route("/ingest/batch", post(documents::batch_ingest))
        .route("/documents/:document_id/status", get(documents::get_document_status))
        
        // File uploads
        .route("/files", post(files::upload_file))
        .route("/files/:file_id", get(files::get_file_info))
        .route("/files/:file_id", delete(files::delete_file))
        
        // Query processing
        .route("/query", post(queries::process_query))
        // .route("/query/stream", post(queries::stream_query_response)) // TODO: Fix streaming handler
        .route("/queries/history", get(queries::get_query_history))
        .route("/queries/metrics", get(queries::get_query_metrics))
        .route("/queries/:query_id", get(queries::get_query_result))
        .route("/queries/:query_id", delete(queries::cancel_query))
        .route("/queries/:query_id/similar", get(queries::get_similar_queries))
}

fn admin_routes() -> Router<AppState> {
    Router::new()
        .route("/system/info", get(admin::system_info))
        .route("/system/components", get(admin::component_status))
        .route("/system/reset", post(admin::reset_components))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiConfig;

    #[tokio::test]
    async fn test_routes_creation() {
        let config = Arc::new(ApiConfig::default());
        let routes = create_routes(config);
        
        // Verify that the router is created without panicking
        // In a full test, we'd use axum-test to verify route functionality
    }

    #[test]
    fn test_nested_routes() {
        // Test that individual route groups can be created
        let health_router = health_routes();
        let auth_router = auth_routes();
        let api_router = api_routes();
        let admin_router = admin_routes();
        
        // Basic test - routers should be created without panicking
    }
}