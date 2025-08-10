use axum::{
    extract::{ConnectInfo, Request},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use dashmap::DashMap;
use std::{
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::time::sleep;
use tower::{Layer, Service};
use tracing::{debug, warn};

use crate::{config::ApiConfig, ApiError, middleware::error_handling::handle_rate_limit_error};

/// Rate limiting configuration
#[derive(Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per window
    pub max_requests: u32,
    /// Time window duration
    pub window_duration: Duration,
    /// Whether to use sliding window (true) or fixed window (false)
    pub sliding_window: bool,
    /// List of endpoints that are exempt from rate limiting
    pub exempt_paths: Vec<String>,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window_duration: Duration::from_secs(60),
            sliding_window: true,
            exempt_paths: vec![
                "/health".to_string(),
                "/health/ready".to_string(),
                "/health/live".to_string(),
                "/metrics".to_string(),
            ],
        }
    }
}

/// Rate limiting layer
#[derive(Clone)]
pub struct RateLimitingLayer {
    config: Arc<RateLimitConfig>,
    store: Arc<RateLimitStore>,
}

impl RateLimitingLayer {
    pub fn new(api_config: Arc<ApiConfig>) -> Self {
        let config = Arc::new(RateLimitConfig {
            max_requests: api_config.security.rate_limit_requests,
            window_duration: Duration::from_secs(api_config.security.rate_limit_window_secs),
            sliding_window: true,
            exempt_paths: vec![
                "/health".to_string(),
                "/health/ready".to_string(),
                "/health/live".to_string(),
                "/metrics".to_string(),
            ],
        });
        
        Self {
            config: config.clone(),
            store: Arc::new(RateLimitStore::new(config)),
        }
    }

    pub fn with_config(config: RateLimitConfig) -> Self {
        let config = Arc::new(config);
        Self {
            config: config.clone(),
            store: Arc::new(RateLimitStore::new(config)),
        }
    }
}

impl<S> Layer<S> for RateLimitingLayer {
    type Service = RateLimitingMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RateLimitingMiddleware {
            inner,
            config: self.config.clone(),
            store: self.store.clone(),
        }
    }
}

/// Rate limiting middleware service
#[derive(Clone)]
pub struct RateLimitingMiddleware<S> {
    inner: S,
    config: Arc<RateLimitConfig>,
    store: Arc<RateLimitStore>,
}

impl<S> Service<Request> for RateLimitingMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Into<Box<dyn std::error::Error + Send + Sync>>,
{
    type Response = Response;
    type Error = Box<dyn std::error::Error + Send + Sync>;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        match self.inner.poll_ready(cx) {
            std::task::Poll::Ready(Ok(())) => std::task::Poll::Ready(Ok(())),
            std::task::Poll::Ready(Err(e)) => std::task::Poll::Ready(Err(e.into())),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }

    fn call(&mut self, request: Request) -> Self::Future {
        let path = request.uri().path().to_string();
        
        // Check if path is exempt from rate limiting
        if self.config.exempt_paths.iter().any(|exempt| path.starts_with(exempt)) {
            debug!("Path {} is exempt from rate limiting", path);
            let future = self.inner.call(request);
            return Box::pin(async move {
                future.await.map_err(|e| e.into())
            });
        }

        let client_id = extract_client_identifier(&request);
        let store = self.store.clone();
        let config = self.config.clone();
        let future = self.inner.call(request);

        Box::pin(async move {
            // Check rate limit
            let rate_limit_result = store.check_rate_limit(&client_id).await;
            
            match rate_limit_result {
                Ok(remaining) => {
                    debug!("Rate limit check passed for {}: {} remaining", client_id, remaining);
                    
                    // Proceed with request
                    let mut response = future.await.map_err(|e| e.into())?;
                    
                    // Add rate limit headers
                    add_rate_limit_headers(&mut response, &config, remaining, None);
                    
                    Ok(response)
                }
                Err(retry_after) => {
                    warn!("Rate limit exceeded for {}: retry after {}s", client_id, retry_after);
                    
                    let error = handle_rate_limit_error(Some(&client_id), Some(retry_after));
                    let mut response = error.into_response();
                    
                    // Add rate limit headers
                    add_rate_limit_headers(&mut response, &config, 0, Some(retry_after));
                    
                    Ok(response)
                }
            }
        })
    }
}

/// Rate limiting storage
pub struct RateLimitStore {
    // Map of client_id -> rate limit state
    requests: DashMap<String, ClientRateLimit>,
    config: Arc<RateLimitConfig>,
}

impl RateLimitStore {
    pub fn new(config: Arc<RateLimitConfig>) -> Self {
        let store = Self {
            requests: DashMap::new(),
            config,
        };
        
        // Start cleanup task
        store.start_cleanup_task();
        
        store
    }

    /// Check rate limit for a client
    pub async fn check_rate_limit(&self, client_id: &str) -> Result<u32, u64> {
        let now = Instant::now();
        
        let mut entry = self.requests.entry(client_id.to_string()).or_insert_with(|| {
            ClientRateLimit::new(self.config.clone())
        });
        
        entry.check_and_update(now)
    }

    /// Start background cleanup task to remove expired entries
    fn start_cleanup_task(&self) {
        let requests = self.requests.clone();
        let cleanup_interval = self.config.window_duration;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                let now = Instant::now();
                let mut to_remove = Vec::new();
                
                // Find expired entries
                for entry in requests.iter() {
                    if entry.value().is_expired(now) {
                        to_remove.push(entry.key().clone());
                    }
                }
                
                // Remove expired entries
                for key in to_remove {
                    requests.remove(&key);
                }
            }
        });
    }

    /// Reset rate limit for a specific client (admin function)
    pub fn reset_client_rate_limit(&self, client_id: &str) {
        self.requests.remove(client_id);
        debug!("Rate limit reset for client: {}", client_id);
    }

    /// Get current statistics
    pub fn get_stats(&self) -> RateLimitStats {
        let total_clients = self.requests.len();
        let mut limited_clients = 0;
        
        let now = Instant::now();
        for entry in self.requests.iter() {
            if entry.value().is_limited(now) {
                limited_clients += 1;
            }
        }
        
        RateLimitStats {
            total_clients,
            limited_clients,
            config: self.config.clone(),
        }
    }
}

/// Rate limit state for a single client
#[derive(Debug)]
struct ClientRateLimit {
    requests: Vec<Instant>,
    config: Arc<RateLimitConfig>,
    last_access: Instant,
}

impl ClientRateLimit {
    fn new(config: Arc<RateLimitConfig>) -> Self {
        Self {
            requests: Vec::new(),
            config,
            last_access: Instant::now(),
        }
    }

    fn check_and_update(&mut self, now: Instant) -> Result<u32, u64> {
        self.last_access = now;
        
        if self.config.sliding_window {
            self.check_sliding_window(now)
        } else {
            self.check_fixed_window(now)
        }
    }

    fn check_sliding_window(&mut self, now: Instant) -> Result<u32, u64> {
        let window_start = now - self.config.window_duration;
        
        // Remove requests outside the window
        self.requests.retain(|&timestamp| timestamp > window_start);
        
        if self.requests.len() >= self.config.max_requests as usize {
            // Rate limit exceeded - calculate retry after
            let oldest_request = self.requests.first().copied().unwrap_or(now);
            let retry_after = (oldest_request + self.config.window_duration)
                .saturating_duration_since(now)
                .as_secs();
            
            Err(retry_after.max(1))
        } else {
            // Add current request
            self.requests.push(now);
            let remaining = self.config.max_requests - self.requests.len() as u32;
            Ok(remaining)
        }
    }

    fn check_fixed_window(&mut self, now: Instant) -> Result<u32, u64> {
        // For fixed window, we reset the counter at the start of each window
        let current_window = now.elapsed().as_secs() / self.config.window_duration.as_secs();
        let last_window = self.last_access.elapsed().as_secs() / self.config.window_duration.as_secs();
        
        if current_window > last_window {
            // New window, reset counter
            self.requests.clear();
        }
        
        if self.requests.len() >= self.config.max_requests as usize {
            // Rate limit exceeded
            let retry_after = self.config.window_duration.as_secs() - (now.elapsed().as_secs() % self.config.window_duration.as_secs());
            Err(retry_after.max(1))
        } else {
            // Add current request
            self.requests.push(now);
            let remaining = self.config.max_requests - self.requests.len() as u32;
            Ok(remaining)
        }
    }

    fn is_expired(&self, now: Instant) -> bool {
        now.duration_since(self.last_access) > self.config.window_duration * 2
    }

    fn is_limited(&self, now: Instant) -> bool {
        if self.config.sliding_window {
            let window_start = now - self.config.window_duration;
            let valid_requests = self.requests.iter().filter(|&&timestamp| timestamp > window_start).count();
            valid_requests >= self.config.max_requests as usize
        } else {
            self.requests.len() >= self.config.max_requests as usize
        }
    }
}

/// Rate limiting statistics
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    pub total_clients: usize,
    pub limited_clients: usize,
    pub config: Arc<RateLimitConfig>,
}

/// Extract client identifier from request
fn extract_client_identifier(request: &Request) -> String {
    // Try to get authenticated user ID first
    if let Some(user_id) = extract_user_id_from_request(request) {
        return format!("user:{}", user_id);
    }
    
    // Try to get API key
    if let Some(api_key) = extract_api_key_from_request(request) {
        return format!("apikey:{}", api_key);
    }
    
    // Fall back to IP address
    if let Some(ConnectInfo(addr)) = request.extensions().get::<ConnectInfo<SocketAddr>>() {
        return format!("ip:{}", addr.ip());
    }
    
    // Last resort - use a default identifier
    "unknown".to_string()
}

fn extract_user_id_from_request(request: &Request) -> Option<String> {
    // This would extract user ID from JWT token or session
    // For now, we'll check for a simple user-id header
    request
        .headers()
        .get("x-user-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn extract_api_key_from_request(request: &Request) -> Option<String> {
    request
        .headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Add rate limit headers to response
fn add_rate_limit_headers(
    response: &mut Response,
    config: &RateLimitConfig,
    remaining: u32,
    retry_after: Option<u64>,
) {
    let headers = response.headers_mut();
    
    headers.insert(
        "X-RateLimit-Limit",
        config.max_requests.to_string().parse().unwrap(),
    );
    
    headers.insert(
        "X-RateLimit-Remaining",
        remaining.to_string().parse().unwrap(),
    );
    
    headers.insert(
        "X-RateLimit-Window",
        config.window_duration.as_secs().to_string().parse().unwrap(),
    );
    
    if let Some(retry_after) = retry_after {
        headers.insert(
            "Retry-After",
            retry_after.to_string().parse().unwrap(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use std::net::{IpAddr, Ipv4Addr};

    #[tokio::test]
    async fn test_rate_limit_store() {
        let config = Arc::new(RateLimitConfig {
            max_requests: 2,
            window_duration: Duration::from_secs(60),
            sliding_window: true,
            exempt_paths: vec![],
        });
        
        let store = RateLimitStore::new(config);
        let client_id = "test_client";
        
        // First request should succeed
        assert!(store.check_rate_limit(client_id).await.is_ok());
        
        // Second request should succeed
        assert!(store.check_rate_limit(client_id).await.is_ok());
        
        // Third request should be rate limited
        assert!(store.check_rate_limit(client_id).await.is_err());
    }

    #[tokio::test]
    async fn test_sliding_window() {
        let config = Arc::new(RateLimitConfig {
            max_requests: 2,
            window_duration: Duration::from_millis(100),
            sliding_window: true,
            exempt_paths: vec![],
        });
        
        let store = RateLimitStore::new(config);
        let client_id = "test_client";
        
        // Use up the limit
        assert!(store.check_rate_limit(client_id).await.is_ok());
        assert!(store.check_rate_limit(client_id).await.is_ok());
        assert!(store.check_rate_limit(client_id).await.is_err());
        
        // Wait for window to slide
        sleep(Duration::from_millis(150)).await;
        
        // Should be able to make requests again
        assert!(store.check_rate_limit(client_id).await.is_ok());
    }

    #[test]
    fn test_client_identifier_extraction() {
        let request = Request::builder()
            .header("x-user-id", "user123")
            .body(Body::empty())
            .unwrap();
        
        let client_id = extract_client_identifier(&request);
        assert_eq!(client_id, "user:user123");
    }

    #[test]
    fn test_exempt_paths() {
        let config = RateLimitConfig::default();
        assert!(config.exempt_paths.contains(&"/health".to_string()));
        assert!(config.exempt_paths.contains(&"/metrics".to_string()));
    }

    #[tokio::test]
    async fn test_rate_limit_stats() {
        let config = Arc::new(RateLimitConfig {
            max_requests: 1,
            window_duration: Duration::from_secs(60),
            sliding_window: true,
            exempt_paths: vec![],
        });
        
        let store = RateLimitStore::new(config);
        
        // Make some requests
        let _ = store.check_rate_limit("client1").await;
        let _ = store.check_rate_limit("client2").await;
        let _ = store.check_rate_limit("client1").await; // This should be rate limited
        
        let stats = store.get_stats();
        assert_eq!(stats.total_clients, 2);
    }
}