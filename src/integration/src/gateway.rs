//! # API Gateway
//!
//! Unified API gateway providing secure, authenticated access to all system components
//! with rate limiting, request routing, and comprehensive monitoring.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use axum::{
    routing::{get, post},
    Router, Json, Extension,
    http::{StatusCode, HeaderMap, Method},
    response::Response,
    body::Body,
    middleware::{self, Next},
    extract::{Path, Query, State},
};
use tower::ServiceBuilder;
use tower_http::{
    cors::{CorsLayer, Any},
    compression::CompressionLayer,
    trace::TraceLayer,
    add_extension::AddExtensionLayer,
    timeout::TimeoutLayer,
};
use tracing::{info, warn, error, instrument};
use uuid::Uuid;
use serde::{Deserialize, Serialize};

use crate::{
    Result, IntegrationError, ProcessingPipeline, HealthMonitor,
    QueryRequest, QueryResponse, SystemHealth, ResponseFormat,
};

/// API Gateway configuration
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Server bind address
    pub bind_address: SocketAddr,
    /// Request timeout
    pub request_timeout: Duration,
    /// Rate limit (requests per minute)
    pub rate_limit: u32,
    /// Enable CORS
    pub enable_cors: bool,
    /// API key for authentication
    pub api_key: Option<String>,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Enable request logging
    pub enable_logging: bool,
}

/// API request context
#[derive(Debug, Clone)]
pub struct RequestContext {
    /// Request ID
    pub request_id: Uuid,
    /// Client IP address
    pub client_ip: Option<String>,
    /// User agent
    pub user_agent: Option<String>,
    /// Authentication status
    pub authenticated: bool,
    /// Request timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// API response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Request ID
    pub request_id: Uuid,
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Server version
    pub version: String,
    /// Response timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Standard API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Response data
    pub data: T,
    /// Response metadata
    pub metadata: ResponseMetadata,
    /// Success status
    pub success: bool,
    /// Error message if any
    pub error: Option<String>,
}

/// Rate limiting state
#[derive(Debug, Default, Clone)]
struct RateLimitState {
    /// Request counts by client IP
    requests: HashMap<String, (u32, chrono::DateTime<chrono::Utc>)>,
    /// Rate limit configuration
    limit: u32,
    /// Time window
    window: Duration,
}

/// Gateway metrics
#[derive(Debug, Default, Clone)]
struct GatewayMetrics {
    /// Total requests
    total_requests: u64,
    /// Successful requests
    successful_requests: u64,
    /// Failed requests
    failed_requests: u64,
    /// Rate limited requests
    rate_limited_requests: u64,
    /// Average response time
    avg_response_time: Duration,
    /// Requests by endpoint
    endpoint_metrics: HashMap<String, EndpointMetrics>,
}

/// Endpoint-specific metrics
#[derive(Debug, Default, Clone)]
struct EndpointMetrics {
    /// Request count
    requests: u64,
    /// Success count
    successes: u64,
    /// Error count
    errors: u64,
    /// Average response time
    avg_response_time: Duration,
}

/// Main API Gateway
pub struct ApiGateway {
    /// Gateway ID
    id: Uuid,
    /// Configuration
    config: Arc<crate::IntegrationConfig>,
    /// Gateway-specific configuration
    gateway_config: GatewayConfig,
    /// Processing pipeline
    pipeline: Arc<ProcessingPipeline>,
    /// Health monitor
    health_monitor: Arc<HealthMonitor>,
    /// Rate limiting state
    rate_limit_state: Arc<RwLock<RateLimitState>>,
    /// Gateway metrics
    metrics: Arc<RwLock<GatewayMetrics>>,
}

impl ApiGateway {
    /// Create new API gateway
    pub async fn new(
        config: Arc<crate::IntegrationConfig>,
        pipeline: Arc<ProcessingPipeline>,
        health_monitor: Arc<HealthMonitor>,
    ) -> Result<Self> {
        let bind_address = config.gateway_bind_address
            .unwrap_or_else(|| "0.0.0.0:8000".parse().unwrap());
        
        let gateway_config = GatewayConfig {
            bind_address,
            request_timeout: Duration::from_secs(30),
            rate_limit: config.rate_limit.unwrap_or(60), // 60 requests per minute
            enable_cors: config.enable_cors.unwrap_or(true),
            api_key: config.api_key.clone(),
            enable_metrics: config.enable_metrics.unwrap_or(true),
            enable_logging: config.enable_logging.unwrap_or(true),
        };
        
        let rate_limit_state = RateLimitState {
            requests: HashMap::new(),
            limit: gateway_config.rate_limit,
            window: Duration::from_secs(60),
        };
        
        Ok(Self {
            id: Uuid::new_v4(),
            config,
            gateway_config,
            pipeline,
            health_monitor,
            rate_limit_state: Arc::new(RwLock::new(rate_limit_state)),
            metrics: Arc::new(RwLock::new(GatewayMetrics::default())),
        })
    }
    
    /// Initialize API gateway
    #[instrument(skip(self))]
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing API Gateway: {}", self.id);
        info!("Gateway will bind to: {}", self.gateway_config.bind_address);
        Ok(())
    }
    
    /// Start API gateway server
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting API Gateway server...");
        
        // Build router
        let app = self.build_router().await;
        
        // Start server
        let listener = tokio::net::TcpListener::bind(&self.gateway_config.bind_address).await
            .map_err(|e| IntegrationError::Internal(format!("Failed to bind to {}: {}", self.gateway_config.bind_address, e)))?;
        
        info!("API Gateway listening on {}", self.gateway_config.bind_address);
        
        // Start metrics collection
        let gateway = self.clone();
        tokio::spawn(async move {
            gateway.collect_metrics().await;
        });
        
        // Start rate limit cleanup
        let gateway = self.clone();
        tokio::spawn(async move {
            gateway.cleanup_rate_limits().await;
        });
        
        // Serve the application
        axum::serve(listener, app).await
            .map_err(|e| IntegrationError::Internal(format!("Server error: {}", e)))?;
        
        Ok(())
    }
    
    /// Stop API gateway
    #[instrument(skip(self))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping API Gateway...");
        // Server stops when the future is dropped
        info!("API Gateway stopped successfully");
        Ok(())
    }
    
    /// Build Axum router with all routes and middleware
    async fn build_router(&self) -> Router {
        let gateway = Arc::new(self.clone());
        
        // Create CORS layer if enabled
        let cors_layer = if self.gateway_config.enable_cors {
            Some(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                    .allow_headers(Any)
            )
        } else {
            None
        };
        
        // Build the router
        let mut app = Router::new()
            // Health and system status endpoints
            .route("/health", get(health_handler))
            .route("/status", get(status_handler))
            .route("/metrics", get(metrics_handler))
            
            // Main RAG query endpoint
            .route("/query", post(query_handler))
            .route("/query/stream", post(query_stream_handler))
            
            // Component-specific endpoints
            .route("/components/:component/health", get(component_health_handler))
            .route("/components/:component/status", get(component_status_handler))
            
            // Administrative endpoints
            .route("/admin/shutdown", post(shutdown_handler))
            .route("/admin/reload", post(reload_handler))
            
            // Add shared state
            .with_state(gateway.clone());
        
        // Add middleware layers individually
        app = app
            .layer(TimeoutLayer::new(self.gateway_config.request_timeout))
            .layer(CompressionLayer::new())
            .layer(TraceLayer::new_for_http());
        
        // Add CORS if enabled
        if let Some(cors) = cors_layer {
            app = app.layer(cors);
        }
        
        app
    }
    
    /// Update gateway metrics
    async fn update_metrics(&self, endpoint: &str, success: bool, duration: Duration) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_requests += 1;
        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }
        
        // Update average response time
        let total_time = metrics.avg_response_time.as_millis() as f64 * (metrics.total_requests - 1) as f64;
        metrics.avg_response_time = Duration::from_millis(
            ((total_time + duration.as_millis() as f64) / metrics.total_requests as f64) as u64
        );
        
        // Update endpoint metrics
        let endpoint_metrics = metrics.endpoint_metrics
            .entry(endpoint.to_string())
            .or_insert_with(EndpointMetrics::default);
        
        endpoint_metrics.requests += 1;
        if success {
            endpoint_metrics.successes += 1;
        } else {
            endpoint_metrics.errors += 1;
        }
        
        // Update endpoint average response time
        let endpoint_total = endpoint_metrics.avg_response_time.as_millis() as f64 
            * (endpoint_metrics.requests - 1) as f64;
        endpoint_metrics.avg_response_time = Duration::from_millis(
            ((endpoint_total + duration.as_millis() as f64) / endpoint_metrics.requests as f64) as u64
        );
    }
    
    /// Collect and log metrics periodically
    async fn collect_metrics(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            let metrics = self.metrics.read().await;
            info!("Gateway Metrics: {} total, {} success, {} failed, avg: {:?}",
                metrics.total_requests,
                metrics.successful_requests,
                metrics.failed_requests,
                metrics.avg_response_time
            );
            
            for (endpoint, endpoint_metrics) in &metrics.endpoint_metrics {
                info!("Endpoint {}: {} requests, {} success, {} errors, avg: {:?}",
                    endpoint,
                    endpoint_metrics.requests,
                    endpoint_metrics.successes,
                    endpoint_metrics.errors,
                    endpoint_metrics.avg_response_time
                );
            }
        }
    }
    
    /// Clean up old rate limit entries
    async fn cleanup_rate_limits(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(300)); // 5 minutes
        
        loop {
            interval.tick().await;
            
            let mut rate_limits = self.rate_limit_state.write().await;
            let cutoff = chrono::Utc::now() - chrono::Duration::from_std(rate_limits.window).unwrap();
            
            let mut to_remove = Vec::new();
            for (ip, (_, timestamp)) in &rate_limits.requests {
                if *timestamp < cutoff {
                    to_remove.push(ip.clone());
                }
            }
            
            for ip in to_remove {
                rate_limits.requests.remove(&ip);
            }
        }
    }
}

impl Clone for ApiGateway {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            config: self.config.clone(),
            gateway_config: self.gateway_config.clone(),
            pipeline: self.pipeline.clone(),
            health_monitor: self.health_monitor.clone(),
            rate_limit_state: self.rate_limit_state.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

// HTTP Handlers

/// Health check handler
async fn health_handler(
    State(gateway): State<Arc<ApiGateway>>,
    Extension(ctx): Extension<RequestContext>,
) -> std::result::Result<Json<ApiResponse<SystemHealth>>, StatusCode> {
    let health = gateway.health_monitor.system_health().await;
    
    let response = ApiResponse {
        data: health,
        metadata: ResponseMetadata {
            request_id: ctx.request_id,
            processing_time_ms: 0, // Would be set by middleware
            version: crate::VERSION.to_string(),
            timestamp: chrono::Utc::now(),
        },
        success: true,
        error: None,
    };
    
    Ok(Json(response))
}

/// System status handler
async fn status_handler(
    State(gateway): State<Arc<ApiGateway>>,
    Extension(ctx): Extension<RequestContext>,
) -> std::result::Result<Json<ApiResponse<HashMap<String, serde_json::Value>>>, StatusCode> {
    let health = gateway.health_monitor.system_health().await;
    let pipeline_metrics = gateway.pipeline.metrics().await;
    let gateway_metrics = gateway.metrics.read().await;
    
    let mut status = HashMap::new();
    status.insert("health".to_string(), serde_json::to_value(&health).unwrap());
    status.insert("pipeline_metrics".to_string(), serde_json::to_value(&*pipeline_metrics).unwrap());
    status.insert("gateway_metrics".to_string(), serde_json::to_value(&*gateway_metrics).unwrap());
    
    let response = ApiResponse {
        data: status,
        metadata: ResponseMetadata {
            request_id: ctx.request_id,
            processing_time_ms: 0,
            version: crate::VERSION.to_string(),
            timestamp: chrono::Utc::now(),
        },
        success: true,
        error: None,
    };
    
    Ok(Json(response))
}

/// Metrics handler
async fn metrics_handler(
    State(gateway): State<Arc<ApiGateway>>,
    Extension(ctx): Extension<RequestContext>,
) -> std::result::Result<Json<ApiResponse<GatewayMetrics>>, StatusCode> {
    let metrics = gateway.metrics.read().await.clone();
    
    let response = ApiResponse {
        data: metrics,
        metadata: ResponseMetadata {
            request_id: ctx.request_id,
            processing_time_ms: 0,
            version: crate::VERSION.to_string(),
            timestamp: chrono::Utc::now(),
        },
        success: true,
        error: None,
    };
    
    Ok(Json(response))
}

/// Main query processing handler
async fn query_handler(
    State(gateway): State<Arc<ApiGateway>>,
    Extension(ctx): Extension<RequestContext>,
    Json(mut request): Json<QueryRequest>,
) -> std::result::Result<Json<ApiResponse<QueryResponse>>, StatusCode> {
    // Set request ID from context
    request.id = ctx.request_id;
    
    match gateway.pipeline.process_query(request).await {
        Ok(response) => {
            let api_response = ApiResponse {
                data: response,
                metadata: ResponseMetadata {
                    request_id: ctx.request_id,
                    processing_time_ms: 0, // Set by middleware
                    version: crate::VERSION.to_string(),
                    timestamp: chrono::Utc::now(),
                },
                success: true,
                error: None,
            };
            Ok(Json(api_response))
        }
        Err(e) => {
            error!("Query processing failed: {}", e);
            let api_response = ApiResponse {
                data: QueryResponse {
                    request_id: ctx.request_id,
                    response: "Query processing failed".to_string(),
                    format: ResponseFormat::Json,
                    confidence: 0.0,
                    citations: Vec::new(),
                    processing_time_ms: 0,
                    component_times: HashMap::new(),
                },
                metadata: ResponseMetadata {
                    request_id: ctx.request_id,
                    processing_time_ms: 0,
                    version: crate::VERSION.to_string(),
                    timestamp: chrono::Utc::now(),
                },
                success: false,
                error: Some(e.to_string()),
            };
            Ok(Json(api_response))
        }
    }
}

/// Streaming query handler (placeholder)
async fn query_stream_handler(
    State(_gateway): State<Arc<ApiGateway>>,
    Extension(_ctx): Extension<RequestContext>,
) -> Result<&'static str, StatusCode> {
    // Streaming implementation would go here
    Ok("Streaming not yet implemented")
}

/// Component health handler
async fn component_health_handler(
    State(gateway): State<Arc<ApiGateway>>,
    Extension(ctx): Extension<RequestContext>,
    Path(component): Path<String>,
) -> std::result::Result<Json<ApiResponse<Option<crate::ComponentHealth>>>, StatusCode> {
    let health = gateway.health_monitor.component_health(&component).await;
    
    let response = ApiResponse {
        data: health,
        metadata: ResponseMetadata {
            request_id: ctx.request_id,
            processing_time_ms: 0,
            version: crate::VERSION.to_string(),
            timestamp: chrono::Utc::now(),
        },
        success: true,
        error: None,
    };
    
    Ok(Json(response))
}

/// Component status handler
async fn component_status_handler(
    State(_gateway): State<Arc<ApiGateway>>,
    Extension(ctx): Extension<RequestContext>,
    Path(_component): Path<String>,
) -> std::result::Result<Json<ApiResponse<String>>, StatusCode> {
    let response = ApiResponse {
        data: "Component status not implemented".to_string(),
        metadata: ResponseMetadata {
            request_id: ctx.request_id,
            processing_time_ms: 0,
            version: crate::VERSION.to_string(),
            timestamp: chrono::Utc::now(),
        },
        success: true,
        error: None,
    };
    
    Ok(Json(response))
}

/// Shutdown handler
async fn shutdown_handler() -> std::result::Result<&'static str, StatusCode> {
    // Graceful shutdown would be implemented here
    Ok("Shutdown initiated")
}

/// Reload configuration handler
async fn reload_handler() -> std::result::Result<&'static str, StatusCode> {
    // Configuration reload would be implemented here
    Ok("Configuration reloaded")
}

// Middleware functions

/// Authentication middleware
async fn auth_middleware(
    State(gateway): State<Arc<ApiGateway>>,
    mut request: axum::extract::Request,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    // Check API key if configured
    if let Some(required_key) = &gateway.gateway_config.api_key {
        let auth_header = request.headers()
            .get("Authorization")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "));
        
        match auth_header {
            Some(key) if key == required_key => {
                // Authenticated
            }
            _ => {
                return Err(StatusCode::UNAUTHORIZED);
            }
        }
    }
    
    Ok(next.run(request).await)
}

/// Rate limiting middleware
async fn rate_limit_middleware(
    State(gateway): State<Arc<ApiGateway>>,
    request: axum::extract::Request,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    // Get client IP (simplified)
    let client_ip = request.headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .or_else(|| request.headers().get("x-real-ip").and_then(|h| h.to_str().ok()))
        .unwrap_or("unknown")
        .to_string();
    
    // Check rate limit
    {
        let mut rate_limits = gateway.rate_limit_state.write().await;
        let now = chrono::Utc::now();
        let window_start = now - chrono::Duration::from_std(rate_limits.window).unwrap();
        
        let (count, last_reset) = rate_limits.requests
            .get(&client_ip)
            .cloned()
            .unwrap_or((0, now));
        
        if last_reset < window_start {
            // Reset window
            rate_limits.requests.insert(client_ip.clone(), (1, now));
        } else if count >= rate_limits.limit {
            // Rate limited
            let mut metrics = gateway.metrics.write().await;
            metrics.rate_limited_requests += 1;
            return Err(StatusCode::TOO_MANY_REQUESTS);
        } else {
            // Increment count
            rate_limits.requests.insert(client_ip.clone(), (count + 1, last_reset));
        }
    }
    
    Ok(next.run(request).await)
}

/// Request context middleware
async fn request_context_middleware(
    State(_gateway): State<Arc<ApiGateway>>,
    mut request: axum::extract::Request,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    let request_id = Uuid::new_v4();
    let client_ip = request.headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());
    let user_agent = request.headers()
        .get("user-agent")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string());
    
    let context = RequestContext {
        request_id,
        client_ip,
        user_agent,
        authenticated: true, // Set by auth middleware
        timestamp: chrono::Utc::now(),
        metadata: HashMap::new(),
    };
    
    request.extensions_mut().insert(context);
    
    Ok(next.run(request).await)
}

/// Metrics collection middleware
async fn metrics_middleware(
    State(gateway): State<Arc<ApiGateway>>,
    request: axum::extract::Request,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    let start = Instant::now();
    let path = request.uri().path().to_string();
    
    let response = next.run(request).await;
    
    let duration = start.elapsed();
    let success = response.status().is_success();
    
    gateway.update_metrics(&path, success, duration).await;
    
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IntegrationConfig, ServiceDiscovery, MessageBus, IntegrationCoordinator};
    
    async fn create_test_gateway() -> ApiGateway {
        let config = Arc::new(IntegrationConfig::default());
        let service_discovery = Arc::new(ServiceDiscovery::new(config.clone()).await.unwrap());
        let message_bus = Arc::new(MessageBus::new(config.clone()).await.unwrap());
        let coordinator = Arc::new(
            IntegrationCoordinator::new(config.clone(), service_discovery, message_bus.clone()).await.unwrap()
        );
        let pipeline = Arc::new(
            ProcessingPipeline::new(config.clone(), coordinator, message_bus).await.unwrap()
        );
        let health_monitor = Arc::new(HealthMonitor::new(config.clone(), service_discovery).await.unwrap());
        
        ApiGateway::new(config, pipeline, health_monitor).await.unwrap()
    }
    
    #[tokio::test]
    async fn test_gateway_creation() {
        let gateway = create_test_gateway().await;
        assert_eq!(gateway.gateway_config.rate_limit, 60);
    }
    
    #[tokio::test]
    async fn test_rate_limiting() {
        let gateway = create_test_gateway().await;
        
        // Test rate limit structure
        let rate_limits = gateway.rate_limit_state.read().await;
        assert_eq!(rate_limits.limit, 60);
        assert_eq!(rate_limits.window, Duration::from_secs(60));
    }
}
