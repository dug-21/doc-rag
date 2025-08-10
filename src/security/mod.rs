// Security module for Doc-RAG production readiness
// Implements OWASP-compliant security measures

pub mod auth;
pub mod validation;
pub mod rate_limiting;
pub mod headers;
pub mod secrets;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    #[error("Authorization denied: {0}")]
    AuthorizationDenied(String),
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),
    #[error("Input validation failed: {0}")]
    ValidationFailed(String),
    #[error("Security policy violation: {0}")]
    PolicyViolation(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub auth: AuthConfig,
    pub rate_limiting: RateLimitConfig,
    pub validation: ValidationConfig,
    pub headers: SecurityHeadersConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub jwt_secret: String,
    pub jwt_expiration_hours: u64,
    pub require_auth: bool,
    pub allowed_origins: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub enabled: bool,
    pub requests_per_minute: u32,
    pub burst_size: u32,
    pub redis_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub max_request_size: usize,
    pub allowed_content_types: Vec<String>,
    pub sanitize_inputs: bool,
    pub strict_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityHeadersConfig {
    pub cors_enabled: bool,
    pub csp_enabled: bool,
    pub hsts_enabled: bool,
    pub frame_options: String,
    pub content_type_options: String,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            auth: AuthConfig {
                jwt_secret: "change-me-in-production".to_string(),
                jwt_expiration_hours: 24,
                require_auth: true,
                allowed_origins: vec!["http://localhost:3000".to_string()],
            },
            rate_limiting: RateLimitConfig {
                enabled: true,
                requests_per_minute: 100,
                burst_size: 20,
                redis_url: "redis://localhost:6379".to_string(),
            },
            validation: ValidationConfig {
                max_request_size: 10 * 1024 * 1024, // 10MB
                allowed_content_types: vec![
                    "application/json".to_string(),
                    "application/pdf".to_string(),
                    "text/plain".to_string(),
                ],
                sanitize_inputs: true,
                strict_validation: true,
            },
            headers: SecurityHeadersConfig {
                cors_enabled: true,
                csp_enabled: true,
                hsts_enabled: true,
                frame_options: "DENY".to_string(),
                content_type_options: "nosniff".to_string(),
            },
        }
    }
}

pub type SecurityResult<T> = Result<T, SecurityError>;