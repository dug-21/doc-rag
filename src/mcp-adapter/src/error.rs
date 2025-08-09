//! Error handling for MCP adapter operations

use thiserror::Error;

/// Result type alias for MCP adapter operations
pub type Result<T> = std::result::Result<T, McpError>;

/// Comprehensive error types for MCP adapter operations
#[derive(Error, Debug)]
pub enum McpError {
    /// Connection-related errors
    #[error("Connection failed: {0}")]
    Connection(String),

    /// Authentication failures
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// Network communication errors
    #[error("Network error: {0}")]
    Network(String),

    /// Request timeout errors
    #[error("Operation timed out")]
    Timeout,

    /// Message serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid configuration errors
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Rate limiting errors
    #[error("Rate limit exceeded, retry after {retry_after} seconds")]
    RateLimit { retry_after: u64 },

    /// Service unavailable errors
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    /// Internal adapter errors
    #[error("Internal error: {0}")]
    Internal(String),

    /// Validation errors for input data
    #[error("Validation error: {0}")]
    Validation(String),

    /// Resource exhaustion errors
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    /// Connection-related errors
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// Authentication failures
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),

    /// Message timeout with specific duration
    #[error("Message timeout after {timeout_ms}ms")]
    MessageTimeout { timeout_ms: u64 },

    /// Serialization error from serde_json
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Network error from reqwest
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// Token error from jsonwebtoken
    #[error("Token error: {0}")]
    TokenError(#[from] jsonwebtoken::errors::Error),

    /// Invalid URL error
    #[error("Invalid URL: {0}")]
    InvalidUrl(#[from] url::ParseError),

    /// Rate limit exceeded
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

impl McpError {
    /// Returns true if the error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            McpError::Network(_) |
            McpError::Timeout |
            McpError::ServiceUnavailable(_) |
            McpError::RateLimit { .. } |
            McpError::NetworkError(_) |
            McpError::MessageTimeout { .. } |
            McpError::ConnectionFailed(_) |
            McpError::RateLimitExceeded
        )
    }

    /// Returns the recommended retry delay in seconds
    pub fn retry_delay(&self) -> u64 {
        match self {
            McpError::RateLimit { retry_after } => *retry_after,
            McpError::RateLimitExceeded => 5,
            McpError::Network(_) | McpError::NetworkError(_) => 1,
            McpError::Timeout | McpError::MessageTimeout { .. } => 2,
            McpError::ServiceUnavailable(_) | McpError::ConnectionFailed(_) => 5,
            _ => 0,
        }
    }

    /// Returns the error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            McpError::Configuration(_) | McpError::Validation(_) => ErrorSeverity::Critical,
            McpError::Authentication(_) | McpError::AuthenticationFailed(_) | McpError::TokenError(_) => ErrorSeverity::High,
            McpError::Connection(_) | McpError::ConnectionFailed(_) | McpError::ServiceUnavailable(_) => ErrorSeverity::Medium,
            McpError::Network(_) | McpError::NetworkError(_) | McpError::Timeout | McpError::MessageTimeout { .. } | McpError::RateLimit { .. } | McpError::RateLimitExceeded => ErrorSeverity::Low,
            McpError::Serialization(_) | McpError::SerializationError(_) | McpError::Internal(_) | McpError::ResourceExhausted(_) | McpError::InvalidUrl(_) => ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels for monitoring and alerting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl From<tokio::time::error::Elapsed> for McpError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        McpError::Timeout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_retryable() {
        assert!(McpError::Network("test".to_string()).is_retryable());
        assert!(McpError::Timeout.is_retryable());
        assert!(!McpError::Configuration("test".to_string()).is_retryable());
        assert!(!McpError::Authentication("test".to_string()).is_retryable());
    }

    #[test]
    fn test_retry_delay() {
        assert_eq!(McpError::RateLimit { retry_after: 10 }.retry_delay(), 10);
        assert_eq!(McpError::Network("test".to_string()).retry_delay(), 1);
        assert_eq!(McpError::Timeout.retry_delay(), 2);
        assert_eq!(McpError::Authentication("test".to_string()).retry_delay(), 0);
    }

    #[test]
    fn test_error_severity() {
        assert_eq!(McpError::Configuration("test".to_string()).severity(), ErrorSeverity::Critical);
        assert_eq!(McpError::Authentication("test".to_string()).severity(), ErrorSeverity::High);
        assert_eq!(McpError::Network("test".to_string()).severity(), ErrorSeverity::Low);
    }
}