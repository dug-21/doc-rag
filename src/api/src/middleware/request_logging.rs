use axum::{
    extract::{ConnectInfo, MatchedPath, Request},
    http::{HeaderMap, Method, StatusCode, Uri, Version},
    middleware::Next,
    response::Response,
};
use std::{
    net::SocketAddr,
    time::Instant,
};
use tower::{Layer, Service};
use tracing::{info, warn, Span};
use uuid::Uuid;

#[derive(Clone)]
pub struct RequestLoggingLayer;

impl RequestLoggingLayer {
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for RequestLoggingLayer {
    type Service = RequestLoggingMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestLoggingMiddleware { inner }
    }
}

#[derive(Clone)]
pub struct RequestLoggingMiddleware<S> {
    inner: S,
}

impl<S> Service<Request> for RequestLoggingMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
    S::Error: std::fmt::Display,
{
    type Response = Response;
    type Error = S::Error;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        // Remove the stricter requirement on S::Error
        let start_time = Instant::now();
        
        // Extract request information
        let method = request.method().clone();
        let uri = request.uri().clone();
        let version = request.version();
        let headers = request.headers().clone();
        
        // Extract client IP
        let client_ip = request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ConnectInfo(addr)| addr.ip().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        // Extract matched path for route logging
        let matched_path = request
            .extensions()
            .get::<MatchedPath>()
            .map(|path| path.as_str().to_string())
            .unwrap_or_else(|| uri.path().to_string());

        // Generate request ID if not present
        let request_id = headers
            .get("x-request-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        // Extract user agent
        let user_agent = headers
            .get("user-agent")
            .and_then(|h| h.to_str().ok())
            .unwrap_or_else(|| "unknown")
            .to_string();

        // Create tracing span for this request
        let span = tracing::info_span!(
            "request",
            method = %method,
            path = %matched_path,
            request_id = %request_id,
            client_ip = %client_ip,
            user_agent = %user_agent
        );

        let future = self.inner.call(request);

        Box::pin(async move {
            let _enter = span.enter();
            
            info!(
                method = %method,
                path = %matched_path,
                version = ?version,
                client_ip = %client_ip,
                user_agent = %user_agent,
                request_id = %request_id,
                "Request started"
            );

            let result = future.await;

            let duration = start_time.elapsed();

            match result {
                Ok(response) => {
                    let status = response.status();
                    let response_size = response
                        .headers()
                        .get("content-length")
                        .and_then(|h| h.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok())
                        .unwrap_or(0);

                    // Log based on status code
                    if status.is_server_error() {
                        warn!(
                            method = %method,
                            path = %matched_path,
                            status = %status,
                            duration_ms = duration.as_millis(),
                            response_size_bytes = response_size,
                            request_id = %request_id,
                            client_ip = %client_ip,
                            "Request completed with server error"
                        );
                    } else if status.is_client_error() {
                        warn!(
                            method = %method,
                            path = %matched_path,
                            status = %status,
                            duration_ms = duration.as_millis(),
                            response_size_bytes = response_size,
                            request_id = %request_id,
                            client_ip = %client_ip,
                            "Request completed with client error"
                        );
                    } else {
                        info!(
                            method = %method,
                            path = %matched_path,
                            status = %status,
                            duration_ms = duration.as_millis(),
                            response_size_bytes = response_size,
                            request_id = %request_id,
                            client_ip = %client_ip,
                            "Request completed successfully"
                        );
                    }

                    Ok(response)
                }
                Err(error) => {
                    warn!(
                        method = %method,
                        path = %matched_path,
                        duration_ms = duration.as_millis(),
                        request_id = %request_id,
                        client_ip = %client_ip,
                        error = %error,
                        "Request failed with error"
                    );
                    
                    Err(error)
                }
            }
        })
    }
}

/// Extract common request metadata for structured logging
#[derive(Debug)]
pub struct RequestMetadata {
    pub request_id: String,
    pub method: Method,
    pub path: String,
    pub client_ip: String,
    pub user_agent: String,
    pub content_length: Option<u64>,
    pub referer: Option<String>,
}

impl RequestMetadata {
    pub fn from_request(request: &Request) -> Self {
        let headers = request.headers();
        
        Self {
            request_id: headers
                .get("x-request-id")
                .and_then(|h| h.to_str().ok())
                .map(|s| s.to_string())
                .unwrap_or_else(|| Uuid::new_v4().to_string()),
            method: request.method().clone(),
            path: request.uri().path().to_string(),
            client_ip: request
                .extensions()
                .get::<ConnectInfo<SocketAddr>>()
                .map(|ConnectInfo(addr)| addr.ip().to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            user_agent: headers
                .get("user-agent")
                .and_then(|h| h.to_str().ok())
                .unwrap_or("unknown")
                .to_string(),
            content_length: headers
                .get("content-length")
                .and_then(|h| h.to_str().ok())
                .and_then(|s| s.parse().ok()),
            referer: headers
                .get("referer")
                .and_then(|h| h.to_str().ok())
                .map(|s| s.to_string()),
        }
    }
}

/// Middleware for security-focused request logging
pub async fn security_request_logging(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let metadata = RequestMetadata::from_request(&request);
    
    // Log potentially suspicious activity
    log_suspicious_patterns(&metadata);
    
    let response = next.run(request).await;
    
    // Log security-relevant response patterns
    log_security_response(&metadata, &response);
    
    Ok(response)
}

fn log_suspicious_patterns(metadata: &RequestMetadata) {
    // Log potential security threats
    let suspicious_patterns = [
        "admin", "config", "env", ".git", "backup", "dump",
        "password", "secret", "key", "token", "auth",
    ];
    
    let path_lower = metadata.path.to_lowercase();
    for pattern in &suspicious_patterns {
        if path_lower.contains(pattern) {
            warn!(
                request_id = %metadata.request_id,
                client_ip = %metadata.client_ip,
                path = %metadata.path,
                user_agent = %metadata.user_agent,
                pattern = pattern,
                "Suspicious path pattern detected"
            );
        }
    }
    
    // Log unusual user agents
    if metadata.user_agent.to_lowercase().contains("bot") ||
       metadata.user_agent.to_lowercase().contains("crawler") ||
       metadata.user_agent.to_lowercase().contains("spider") {
        info!(
            request_id = %metadata.request_id,
            client_ip = %metadata.client_ip,
            user_agent = %metadata.user_agent,
            "Bot/crawler access detected"
        );
    }
    
    // Log large requests
    if let Some(content_length) = metadata.content_length {
        if content_length > 10 * 1024 * 1024 { // 10MB
            warn!(
                request_id = %metadata.request_id,
                client_ip = %metadata.client_ip,
                content_length = content_length,
                "Large request detected"
            );
        }
    }
}

fn log_security_response(metadata: &RequestMetadata, response: &Response) {
    // Log authentication failures
    if response.status() == StatusCode::UNAUTHORIZED {
        warn!(
            request_id = %metadata.request_id,
            client_ip = %metadata.client_ip,
            path = %metadata.path,
            user_agent = %metadata.user_agent,
            "Authentication failure"
        );
    }
    
    // Log forbidden access attempts
    if response.status() == StatusCode::FORBIDDEN {
        warn!(
            request_id = %metadata.request_id,
            client_ip = %metadata.client_ip,
            path = %metadata.path,
            user_agent = %metadata.user_agent,
            "Forbidden access attempt"
        );
    }
    
    // Log rate limiting
    if response.status() == StatusCode::TOO_MANY_REQUESTS {
        warn!(
            request_id = %metadata.request_id,
            client_ip = %metadata.client_ip,
            user_agent = %metadata.user_agent,
            "Rate limit exceeded"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn test_request_metadata_extraction() {
        let request = Request::builder()
            .method(Method::GET)
            .uri("/test/path")
            .header("user-agent", "test-agent")
            .header("x-request-id", "test-123")
            .header("content-length", "1024")
            .header("referer", "https://example.com")
            .body(Body::empty())
            .unwrap();

        let metadata = RequestMetadata::from_request(&request);
        
        assert_eq!(metadata.method, Method::GET);
        assert_eq!(metadata.path, "/test/path");
        assert_eq!(metadata.request_id, "test-123");
        assert_eq!(metadata.user_agent, "test-agent");
        assert_eq!(metadata.content_length, Some(1024));
        assert_eq!(metadata.referer, Some("https://example.com".to_string()));
    }

    #[test] 
    fn test_request_metadata_defaults() {
        let request = Request::builder()
            .method(Method::POST)
            .uri("/api")
            .body(Body::empty())
            .unwrap();

        let metadata = RequestMetadata::from_request(&request);
        
        assert_eq!(metadata.method, Method::POST);
        assert_eq!(metadata.path, "/api");
        assert_eq!(metadata.user_agent, "unknown");
        assert_eq!(metadata.client_ip, "unknown");
        assert!(metadata.content_length.is_none());
        assert!(metadata.referer.is_none());
        // request_id should be generated
        assert!(!metadata.request_id.is_empty());
        assert_ne!(metadata.request_id, "unknown");
    }

    #[tokio::test]
    async fn test_request_logging_middleware() {
        use tower::{ServiceExt, service_fn};

        let service = service_fn(|_request: Request| async {
            Ok::<_, std::convert::Infallible>(
                (StatusCode::OK, "Hello").into_response()
            )
        });

        let service = RequestLoggingLayer::new().layer(service);
        
        let request = Request::builder()
            .method(Method::GET)
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let response = service.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }
}