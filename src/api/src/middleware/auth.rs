use axum::{
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
};
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::{config::ApiConfig, ApiError};

#[derive(Debug, Clone)]
pub struct AuthMiddleware {
    config: Arc<ApiConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,      // Subject (user ID)
    pub email: String,    // User email
    pub role: String,     // User role
    pub exp: usize,       // Expiry time
    pub iat: usize,       // Issued at
    pub jti: String,      // JWT ID
}

#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: Uuid,
    pub email: String,
    pub role: String,
    pub token_id: String,
}

impl AuthMiddleware {
    pub fn new(config: Arc<ApiConfig>) -> Self {
        Self { config }
    }

    /// Extract and validate JWT token from request
    async fn extract_and_validate_token(&self, request: &Request) -> Result<Claims, ApiError> {
        // Extract Authorization header
        let auth_header = request
            .headers()
            .get("Authorization")
            .ok_or_else(|| ApiError::Unauthorized("Missing Authorization header".to_string()))?;

        let auth_str = auth_header
            .to_str()
            .map_err(|_| ApiError::Unauthorized("Invalid Authorization header format".to_string()))?;

        // Check for Bearer token format
        if !auth_str.starts_with("Bearer ") {
            return Err(ApiError::Unauthorized(
                "Authorization header must use Bearer scheme".to_string(),
            ));
        }

        let token = &auth_str[7..]; // Remove "Bearer " prefix

        // Decode and validate JWT
        let decoding_key = DecodingKey::from_secret(self.config.security.jwt_secret.as_bytes());
        let validation = Validation::new(Algorithm::HS256);

        let token_data = decode::<Claims>(token, &decoding_key, &validation)
            .map_err(|e| {
                warn!("JWT validation failed: {}", e);
                ApiError::Unauthorized("Invalid or expired token".to_string())
            })?;

        debug!("Token validated for user: {}", token_data.claims.email);
        Ok(token_data.claims)
    }

    /// Check if the endpoint requires authentication
    fn requires_auth(&self, path: &str) -> bool {
        // Public endpoints that don't require authentication
        let public_paths = [
            "/health",
            "/health/ready", 
            "/health/components",
            "/metrics",
            "/auth/login",
            "/auth/refresh",
        ];

        !public_paths.iter().any(|&public_path| path.starts_with(public_path))
    }

    /// Check if user has required role for admin endpoints
    fn check_admin_access(&self, path: &str, role: &str) -> bool {
        if path.starts_with("/admin") {
            role == "admin" || role == "system"
        } else {
            true // Non-admin endpoints allow all authenticated users
        }
    }
}

/// Middleware function for authentication
pub async fn auth_middleware(
    State(auth_middleware): State<AuthMiddleware>,
    mut request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path();

    // Skip authentication for public endpoints
    if !auth_middleware.requires_auth(path) {
        debug!("Skipping auth for public path: {}", path);
        return next.run(request).await;
    }

    // Extract and validate token
    let claims = match auth_middleware.extract_and_validate_token(&request).await {
        Ok(claims) => claims,
        Err(err) => {
            return err.into_response();
        }
    };

    // Check admin access for admin endpoints
    if !auth_middleware.check_admin_access(path, &claims.role) {
        warn!("Admin access denied for user {} on path {}", claims.email, path);
        return ApiError::Forbidden(
            "Insufficient permissions for this endpoint".to_string(),
        ).into_response();
    }

    // Create auth context and add to request extensions
    let auth_context = AuthContext {
        user_id: match Uuid::parse_str(&claims.sub) {
            Ok(id) => id,
            Err(_) => {
                return ApiError::Unauthorized("Invalid user ID in token".to_string()).into_response();
            }
        },
        email: claims.email.clone(),
        role: claims.role.clone(),
        token_id: claims.jti.clone(),
    };

    request.extensions_mut().insert(auth_context);

    debug!("Authentication successful for user: {} ({})", claims.email, claims.role);

    // Continue to next middleware/handler
    next.run(request).await
}

/// Axum extension trait to extract auth context from request
pub trait AuthExtension {
    fn auth_context(&self) -> Option<&AuthContext>;
    fn require_auth_context(&self) -> Result<&AuthContext, ApiError>;
}

impl AuthExtension for Request {
    fn auth_context(&self) -> Option<&AuthContext> {
        self.extensions().get::<AuthContext>()
    }

    fn require_auth_context(&self) -> Result<&AuthContext, ApiError> {
        self.auth_context()
            .ok_or_else(|| ApiError::Unauthorized("Authentication required".to_string()))
    }
}

/// Helper function to generate JWT tokens (for login endpoint)
pub fn generate_jwt_token(
    user_id: Uuid,
    email: String,
    role: String,
    secret: &str,
    expiry_hours: u64,
) -> Result<String, ApiError> {
    use jsonwebtoken::{encode, EncodingKey, Header};

    let now = chrono::Utc::now();
    let exp = (now + chrono::Duration::hours(expiry_hours as i64)).timestamp() as usize;
    let iat = now.timestamp() as usize;

    let claims = Claims {
        sub: user_id.to_string(),
        email,
        role,
        exp,
        iat,
        jti: Uuid::new_v4().to_string(),
    };

    let encoding_key = EncodingKey::from_secret(secret.as_bytes());
    let token = encode(&Header::default(), &claims, &encoding_key)
        .map_err(|e| ApiError::Internal(format!("Failed to generate token: {}", e)))?;

    Ok(token)
}

/// Helper function to create refresh tokens
pub fn generate_refresh_token() -> String {
    use rand::Rng;
    
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                            abcdefghijklmnopqrstuvwxyz\
                            0123456789";
    const TOKEN_LEN: usize = 32;
    let mut rng = rand::thread_rng();

    (0..TOKEN_LEN)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{HeaderValue, Method, Request};

    #[test]
    fn test_jwt_token_generation() {
        let user_id = Uuid::new_v4();
        let email = "test@example.com".to_string();
        let role = "user".to_string();
        let secret = "test-secret-key-32-characters-long";

        let token = generate_jwt_token(user_id, email.clone(), role.clone(), secret, 24);
        assert!(token.is_ok());

        let token_str = token.unwrap();
        assert!(!token_str.is_empty());
        assert!(token_str.contains('.'));
    }

    #[test]
    fn test_refresh_token_generation() {
        let token1 = generate_refresh_token();
        let token2 = generate_refresh_token();
        
        assert_eq!(token1.len(), 32);
        assert_eq!(token2.len(), 32);
        assert_ne!(token1, token2);
        assert!(token1.chars().all(|c| c.is_alphanumeric()));
    }

    #[tokio::test]
    async fn test_auth_middleware_public_paths() {
        let config = Arc::new(ApiConfig::default());
        let middleware = AuthMiddleware::new(config);

        // Test public paths
        let public_paths = vec![
            "/health",
            "/health/ready",
            "/metrics",
            "/auth/login",
        ];

        for path in public_paths {
            assert!(!middleware.requires_auth(path), "Path {} should not require auth", path);
        }

        // Test protected paths
        let protected_paths = vec![
            "/api/v1/ingest",
            "/api/v1/query",
            "/admin/system",
        ];

        for path in protected_paths {
            assert!(middleware.requires_auth(path), "Path {} should require auth", path);
        }
    }

    #[test]
    fn test_admin_access_check() {
        let config = Arc::new(ApiConfig::default());
        let middleware = AuthMiddleware::new(config);

        // Admin paths require admin role
        assert!(!middleware.check_admin_access("/admin/system", "user"));
        assert!(middleware.check_admin_access("/admin/system", "admin"));
        assert!(middleware.check_admin_access("/admin/system", "system"));

        // Non-admin paths allow all roles
        assert!(middleware.check_admin_access("/api/v1/query", "user"));
        assert!(middleware.check_admin_access("/api/v1/query", "admin"));
    }

    #[tokio::test]
    async fn test_token_extraction() {
        let config = Arc::new(ApiConfig::default());
        let middleware = AuthMiddleware::new(config);

        // Test missing authorization header
        let request = Request::builder()
            .method(Method::GET)
            .uri("/api/v1/query")
            .body(Body::empty())
            .unwrap();

        let result = middleware.extract_and_validate_token(&request).await;
        assert!(result.is_err());

        // Test invalid authorization header format
        let request = Request::builder()
            .method(Method::GET)
            .uri("/api/v1/query")
            .header("Authorization", "InvalidFormat")
            .body(Body::empty())
            .unwrap();

        let result = middleware.extract_and_validate_token(&request).await;
        assert!(result.is_err());

        // Test Bearer format but invalid token
        let request = Request::builder()
            .method(Method::GET)
            .uri("/api/v1/query")
            .header("Authorization", "Bearer invalid-token")
            .body(Body::empty())
            .unwrap();

        let result = middleware.extract_and_validate_token(&request).await;
        assert!(result.is_err());
    }
}