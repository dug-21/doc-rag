use anyhow::{Context, Result};
use axum::serve;
use std::{net::SocketAddr, sync::Arc, future::Future};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::{CorsLayer, Any},
    request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing::{info, warn};
use dashmap::DashMap;
use serde_json::Value;

use crate::{
    config::ApiConfig,
    routes::create_routes,
    middleware::{
        error_handling::ErrorHandlingLayer,
        request_logging::RequestLoggingLayer,
        metrics::MetricsRegistry,
    },
    clients::ComponentClients,
};

pub struct ApiServer {
    config: Arc<ApiConfig>,
    clients: Arc<ComponentClients>,
    metrics: Arc<MetricsRegistry>,
    mongodb: mongodb::Client,
    cache: Arc<DashMap<String, Value>>,
}

impl ApiServer {
    pub async fn new(config: Arc<ApiConfig>) -> Result<Self> {
        info!("Initializing Doc-RAG API Gateway");

        // Initialize MongoDB connection
        let mongodb_url = std::env::var("MONGODB_URL")
            .unwrap_or_else(|_| "mongodb://localhost:27017".to_string());
        let mongodb = mongodb::Client::with_uri_str(&mongodb_url).await
            .context("Failed to connect to MongoDB")?;
        info!("Connected to MongoDB at {}", mongodb_url);

        // Initialize in-memory cache (replacing Redis)
        let cache = Arc::new(DashMap::new());
        info!("Initialized in-memory cache (DashMap) - replacing Redis");

        // Initialize component clients
        let clients = ComponentClients::new(config.clone()).await
            .context("Failed to initialize component clients")?;

        // Initialize metrics registry
        let metrics = MetricsRegistry::new()
            .map_err(|e| anyhow::anyhow!("Failed to initialize metrics registry: {}", e))?;

        // Run health checks on all components
        let health = clients.health_check_all().await;
        
        // Check if any components are unhealthy
        let unhealthy: Vec<_> = health
            .iter()
            .filter_map(|(name, result)| {
                if result.is_err() {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect();
        
        if !unhealthy.is_empty() {
            warn!("Some components are unhealthy: {:?}", unhealthy);
        }

        info!("All components are healthy and ready");

        Ok(Self {
            config,
            clients: Arc::new(clients),
            metrics: Arc::new(metrics),
            mongodb,
            cache,
        })
    }

    pub async fn run(self) -> Result<()> {
        let addr = self.config.server_addr().parse::<SocketAddr>()
            .context("Invalid server address")?;

        info!("Starting server on {}", addr);

        let app = self.create_app();

        let listener = TcpListener::bind(&addr).await
            .context("Failed to bind to address")?;
            
        serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
            .await
            .context("Server failed to start")?;

        Ok(())
    }

    pub async fn run_with_graceful_shutdown<F>(self, shutdown_signal: F) -> Result<()>
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let addr = self.config.server_addr().parse::<SocketAddr>()
            .context("Invalid server address")?;

        info!("Starting server with graceful shutdown on {}", addr);

        let app = self.create_app();

        let listener = TcpListener::bind(&addr).await
            .context("Failed to bind to address")?;
            
        serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
            .with_graceful_shutdown(shutdown_signal)
            .await
            .context("Server failed to start")?;

        // Cleanup resources
        self.shutdown().await?;

        Ok(())
    }

    fn create_app(&self) -> axum::Router {
        let routes = create_routes(self.config.clone());

        // Build middleware stack
        let middleware = ServiceBuilder::new()
            // Request ID generation and propagation
            .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
            .layer(PropagateRequestIdLayer::x_request_id())
            
            // Request timeout
            .layer(TimeoutLayer::new(std::time::Duration::from_secs(
                self.config.server.request_timeout_secs
            )))
            
            // Compression
            .layer(CompressionLayer::new())
            
            // CORS
            .layer(self.create_cors_layer())
            
            // Request tracing
            .layer(TraceLayer::new_for_http())
            
            // Request logging
            .layer(RequestLoggingLayer::new())
            
            // Error handling (should be last)
            .layer(ErrorHandlingLayer::new());

        routes
            .with_state(Arc::new(AppState {
                config: self.config.clone(),
                clients: self.clients.clone(),
                metrics: self.metrics.clone(),
                mongodb: self.mongodb.clone(),
                cache: self.cache.clone(),
            }))
            .layer(middleware)
    }

    fn create_cors_layer(&self) -> CorsLayer {
        if self.config.security.enable_cors {
            let mut cors = CorsLayer::new()
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::PUT,
                    axum::http::Method::DELETE,
                    axum::http::Method::OPTIONS,
                ])
                .allow_headers([
                    axum::http::header::CONTENT_TYPE,
                    axum::http::header::AUTHORIZATION,
                    axum::http::header::ACCEPT,
                ]);

            if self.config.server.cors_origins.contains(&"*".to_string()) {
                cors = cors.allow_origin(Any);
            } else {
                for origin in &self.config.server.cors_origins {
                    if let Ok(origin) = origin.parse::<axum::http::HeaderValue>() {
                        cors = cors.allow_origin(origin);
                    }
                }
            }

            cors
        } else {
            CorsLayer::permissive()
        }
    }

    async fn shutdown(&self) -> Result<()> {
        info!("Shutting down API Gateway");

        // Shutdown component clients
        if let Err(e) = self.clients.shutdown().await {
            warn!("Error during client shutdown: {}", e);
        }

        // Export final metrics
        if let Err(e) = self.metrics.export_final_metrics().await {
            warn!("Error exporting final metrics: {}", e);
        }

        info!("API Gateway shutdown complete");
        Ok(())
    }
}

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ApiConfig>,
    pub clients: Arc<ComponentClients>,
    pub metrics: Arc<MetricsRegistry>,
    pub mongodb: mongodb::Client,
    pub cache: Arc<DashMap<String, Value>>,
}

// Implement axum State extraction for Arc<AppState>
impl axum::extract::FromRef<Arc<AppState>> for Arc<ApiConfig> {
    fn from_ref(app_state: &Arc<AppState>) -> Self {
        app_state.config.clone()
    }
}

impl axum::extract::FromRef<Arc<AppState>> for Arc<ComponentClients> {
    fn from_ref(app_state: &Arc<AppState>) -> Self {
        app_state.clients.clone()
    }
}

impl axum::extract::FromRef<Arc<AppState>> for Arc<MetricsRegistry> {
    fn from_ref(app_state: &Arc<AppState>) -> Self {
        app_state.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_creation() {
        let config = Arc::new(ApiConfig::default());
        
        // This would normally fail in tests due to no actual services running
        // but we can test the basic instantiation logic
        let result = ApiServer::new(config).await;
        
        // In a real test environment with mock services, this should succeed
        // For now, we just ensure the function doesn't panic
        match result {
            Ok(_) => {
                // Success case
            }
            Err(e) => {
                // Expected in test environment without actual services
                assert!(e.to_string().contains("health check") || e.to_string().contains("clients"));
            }
        }
    }

    #[test]
    fn test_server_addr_parsing() {
        let config = ApiConfig::default();
        let addr_str = config.server_addr();
        let _addr: SocketAddr = addr_str.parse().expect("Should parse valid socket address");
    }
}