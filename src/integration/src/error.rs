//! # Integration Error Types
//!
//! Comprehensive error handling for the integration system with
//! detailed error information and recovery suggestions.

use std::time::Duration;
use thiserror::Error;

/// Main integration error type
#[derive(Error, Debug)]
pub enum IntegrationError {
    /// Component not found
    #[error("Component not found: {0}")]
    ComponentNotFound(String),
    
    /// Component is unhealthy
    #[error("Component {stage} is unhealthy: {error}")]
    StageUnhealthy {
        stage: String,
        error: String,
    },
    
    /// Stage timeout
    #[error("Stage {stage} timed out after {timeout:?}")]
    StageTimeout {
        stage: String,
        timeout: Duration,
    },
    
    /// Pipeline error
    #[error("Pipeline error: {0}")]
    PipelineError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Consensus error
    #[error("Consensus error: {0}")]
    Consensus(String),
    
    /// Network error
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    
    /// JSON serialization error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Service discovery error
    #[error("Service discovery error: {0}")]
    ServiceDiscoveryError(String),
    
    /// Message bus error
    #[error("Message bus error: {0}")]
    MessageBusError(String),
    
    /// Health check error
    #[error("Health check error: {0}")]
    HealthCheckError(String),
    
    /// Tracing error
    #[error("Tracing error: {0}")]
    TracingError(String),
    
    /// Circuit breaker open
    #[error("Circuit breaker is open for component: {0}")]
    CircuitBreakerOpen(String),
    
    /// Rate limit exceeded
    #[error("Rate limit exceeded for client: {0}")]
    RateLimitExceeded(String),
    
    /// Authentication error
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),
    
    /// Authorization error
    #[error("Authorization failed: {0}")]
    AuthorizationError(String),
    
    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    /// Resource not found
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),
    
    /// Resource conflict
    #[error("Resource conflict: {0}")]
    ResourceConflict(String),
    
    /// Internal system error
    #[error("Internal error: {0}")]
    Internal(String),
    
    /// External dependency error
    #[error("External dependency error: {dependency} - {error}")]
    ExternalDependencyError {
        dependency: String,
        error: String,
    },
    
    /// Database error
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),
    
    /// Metrics error
    #[error("Metrics error: {0}")]
    MetricsError(String),
    
    /// Feature not implemented
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
    
    /// DAA orchestrator initialization failed
    #[error("DAA orchestrator initialization failed: {reason}")]
    InitializationFailed { reason: String },
    
    /// Agent spawning failed
    #[error("Agent spawn failed: {reason}")]
    AgentSpawnFailed { reason: String },
    
    /// Operation cancelled
    #[error("Operation cancelled: {0}")]
    OperationCancelled(String),
    
    /// Resource exhausted
    #[error("Resource exhausted: {resource} - {details}")]
    ResourceExhausted {
        resource: String,
        details: String,
    },
    
    /// Compatibility error
    #[error("Compatibility error: {0}")]
    CompatibilityError(String),
}

/// Result type alias for integration operations
pub type Result<T> = std::result::Result<T, IntegrationError>;

/// Error context for enhanced error reporting
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorContext {
    /// Error code for programmatic handling
    pub code: String,
    /// Human-readable error message
    pub message: String,
    /// Detailed error description
    pub details: Option<String>,
    /// Suggested recovery actions
    pub recovery_suggestions: Vec<String>,
    /// Additional metadata
    pub metadata: std::collections::HashMap<String, String>,
    /// Error timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Request ID if applicable
    pub request_id: Option<uuid::Uuid>,
    /// Component that generated the error
    pub component: Option<String>,
}

/// Enhanced error with context
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnhancedError {
    /// The original error
    pub error: String,
    /// Error context
    pub context: ErrorContext,
    /// Stack trace if available
    pub stack_trace: Option<String>,
    /// Related errors
    pub related_errors: Vec<String>,
}

impl IntegrationError {
    /// Create enhanced error with context
    pub fn with_context(
        self,
        component: Option<String>,
        request_id: Option<uuid::Uuid>,
        additional_context: std::collections::HashMap<String, String>,
    ) -> EnhancedError {
        let (code, recovery_suggestions) = match &self {
            IntegrationError::ComponentNotFound(_) => (
                "COMPONENT_NOT_FOUND".to_string(),
                vec![
                    "Check if the component is registered".to_string(),
                    "Verify component configuration".to_string(),
                    "Check service discovery settings".to_string(),
                ],
            ),
            IntegrationError::StageUnhealthy { .. } => (
                "STAGE_UNHEALTHY".to_string(),
                vec![
                    "Check component health status".to_string(),
                    "Review component logs".to_string(),
                    "Verify external dependencies".to_string(),
                    "Consider restarting the component".to_string(),
                ],
            ),
            IntegrationError::StageTimeout { .. } => (
                "STAGE_TIMEOUT".to_string(),
                vec![
                    "Increase timeout configuration".to_string(),
                    "Check component performance".to_string(),
                    "Review resource allocation".to_string(),
                    "Optimize query complexity".to_string(),
                ],
            ),
            IntegrationError::NetworkError(_) => (
                "NETWORK_ERROR".to_string(),
                vec![
                    "Check network connectivity".to_string(),
                    "Verify endpoint URLs".to_string(),
                    "Check firewall settings".to_string(),
                    "Review DNS configuration".to_string(),
                ],
            ),
            IntegrationError::CircuitBreakerOpen(_) => (
                "CIRCUIT_BREAKER_OPEN".to_string(),
                vec![
                    "Wait for circuit breaker to reset".to_string(),
                    "Check component health".to_string(),
                    "Review error patterns".to_string(),
                    "Consider manual circuit breaker reset".to_string(),
                ],
            ),
            IntegrationError::RateLimitExceeded(_) => (
                "RATE_LIMIT_EXCEEDED".to_string(),
                vec![
                    "Reduce request frequency".to_string(),
                    "Implement exponential backoff".to_string(),
                    "Consider request batching".to_string(),
                    "Review rate limit configuration".to_string(),
                ],
            ),
            IntegrationError::AuthenticationError(_) => (
                "AUTHENTICATION_ERROR".to_string(),
                vec![
                    "Check API key or credentials".to_string(),
                    "Verify authentication configuration".to_string(),
                    "Review token expiration".to_string(),
                    "Check authorization headers".to_string(),
                ],
            ),
            IntegrationError::ResourceExhausted { .. } => (
                "RESOURCE_EXHAUSTED".to_string(),
                vec![
                    "Scale up resources".to_string(),
                    "Optimize resource usage".to_string(),
                    "Implement resource pooling".to_string(),
                    "Review resource limits".to_string(),
                ],
            ),
            _ => (
                "INTERNAL_ERROR".to_string(),
                vec![
                    "Check system logs".to_string(),
                    "Review configuration".to_string(),
                    "Contact system administrator".to_string(),
                ],
            ),
        };
        
        let context = ErrorContext {
            code,
            message: self.to_string(),
            details: None,
            recovery_suggestions,
            metadata: additional_context,
            timestamp: chrono::Utc::now(),
            request_id,
            component,
        };
        
        EnhancedError {
            error: self.to_string(),
            context,
            stack_trace: None,
            related_errors: Vec::new(),
        }
    }
    
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(self,
            IntegrationError::NetworkError(_) |
            IntegrationError::StageTimeout { .. } |
            IntegrationError::ResourceExhausted { .. } |
            IntegrationError::ExternalDependencyError { .. }
        )
    }
    
    /// Check if error is client error (4xx)
    pub fn is_client_error(&self) -> bool {
        matches!(self,
            IntegrationError::ValidationError(_) |
            IntegrationError::AuthenticationError(_) |
            IntegrationError::AuthorizationError(_) |
            IntegrationError::ResourceNotFound(_) |
            IntegrationError::ResourceConflict(_) |
            IntegrationError::RateLimitExceeded(_)
        )
    }
    
    /// Check if error is server error (5xx)
    pub fn is_server_error(&self) -> bool {
        matches!(self,
            IntegrationError::Internal(_) |
            IntegrationError::StageUnhealthy { .. } |
            IntegrationError::StageTimeout { .. } |
            IntegrationError::CircuitBreakerOpen(_) |
            IntegrationError::ResourceExhausted { .. } |
            IntegrationError::ExternalDependencyError { .. }
        )
    }
    
    /// Get HTTP status code for this error
    pub fn http_status_code(&self) -> u16 {
        match self {
            IntegrationError::ValidationError(_) => 400,
            IntegrationError::AuthenticationError(_) => 401,
            IntegrationError::AuthorizationError(_) => 403,
            IntegrationError::ResourceNotFound(_) => 404,
            IntegrationError::ResourceConflict(_) => 409,
            IntegrationError::RateLimitExceeded(_) => 429,
            IntegrationError::NotImplemented(_) => 501,
            IntegrationError::ExternalDependencyError { .. } => 502,
            IntegrationError::ResourceExhausted { .. } => 503,
            IntegrationError::StageTimeout { .. } => 504,
            _ => 500,
        }
    }
}

/// Error recovery strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Retry with exponential backoff
    RetryExponential {
        max_retries: usize,
        base_delay_ms: u64,
        max_delay_ms: u64,
    },
    /// Retry with fixed delay
    RetryFixed {
        max_retries: usize,
        delay_ms: u64,
    },
    /// Fail fast - no retry
    FailFast,
    /// Circuit breaker pattern
    CircuitBreaker {
        failure_threshold: usize,
        timeout_ms: u64,
        half_open_max_calls: usize,
    },
    /// Graceful degradation
    GracefulDegradation,
    /// Manual intervention required
    ManualIntervention,
}

impl IntegrationError {
    /// Get recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            IntegrationError::NetworkError(_) => RecoveryStrategy::RetryExponential {
                max_retries: 3,
                base_delay_ms: 100,
                max_delay_ms: 5000,
            },
            IntegrationError::StageTimeout { .. } => RecoveryStrategy::RetryFixed {
                max_retries: 2,
                delay_ms: 1000,
            },
            IntegrationError::ResourceExhausted { .. } => RecoveryStrategy::RetryExponential {
                max_retries: 5,
                base_delay_ms: 500,
                max_delay_ms: 10000,
            },
            IntegrationError::StageUnhealthy { .. } => RecoveryStrategy::CircuitBreaker {
                failure_threshold: 5,
                timeout_ms: 30000,
                half_open_max_calls: 3,
            },
            IntegrationError::RateLimitExceeded(_) => RecoveryStrategy::RetryExponential {
                max_retries: 10,
                base_delay_ms: 1000,
                max_delay_ms: 60000,
            },
            IntegrationError::ExternalDependencyError { .. } => RecoveryStrategy::RetryExponential {
                max_retries: 3,
                base_delay_ms: 200,
                max_delay_ms: 2000,
            },
            IntegrationError::ValidationError(_) |
            IntegrationError::AuthenticationError(_) |
            IntegrationError::AuthorizationError(_) |
            IntegrationError::ResourceNotFound(_) |
            IntegrationError::ResourceConflict(_) => RecoveryStrategy::FailFast,
            
            IntegrationError::Internal(_) |
            IntegrationError::ConfigurationError(_) |
            IntegrationError::CompatibilityError(_) => RecoveryStrategy::ManualIntervention,
            
            _ => RecoveryStrategy::GracefulDegradation,
        }
    }
}

/// Error metrics for monitoring
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorMetrics {
    /// Total errors
    pub total_errors: u64,
    /// Errors by type
    pub errors_by_type: std::collections::HashMap<String, u64>,
    /// Errors by component
    pub errors_by_component: std::collections::HashMap<String, u64>,
    /// Error rate (errors per minute)
    pub error_rate: f64,
    /// Recovery success rate
    pub recovery_success_rate: f64,
    /// Most common errors
    pub top_errors: Vec<String>,
}

impl ErrorMetrics {
    /// Record an error
    pub fn record_error(&mut self, error: &IntegrationError, component: Option<&str>) {
        self.total_errors += 1;
        
        let error_type = match error {
            IntegrationError::ComponentNotFound(_) => "ComponentNotFound",
            IntegrationError::StageUnhealthy { .. } => "StageUnhealthy",
            IntegrationError::StageTimeout { .. } => "StageTimeout",
            IntegrationError::NetworkError(_) => "NetworkError",
            IntegrationError::CircuitBreakerOpen(_) => "CircuitBreakerOpen",
            IntegrationError::RateLimitExceeded(_) => "RateLimitExceeded",
            IntegrationError::AuthenticationError(_) => "AuthenticationError",
            IntegrationError::ValidationError(_) => "ValidationError",
            _ => "Other",
        };
        
        *self.errors_by_type.entry(error_type.to_string()).or_insert(0) += 1;
        
        if let Some(comp) = component {
            *self.errors_by_component.entry(comp.to_string()).or_insert(0) += 1;
        }
    }
    
    /// Calculate error rate
    pub fn calculate_error_rate(&mut self, window_minutes: f64) {
        self.error_rate = self.total_errors as f64 / window_minutes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_classification() {
        let client_error = IntegrationError::ValidationError("test".to_string());
        assert!(client_error.is_client_error());
        assert!(!client_error.is_server_error());
        
        let server_error = IntegrationError::Internal("test".to_string());
        assert!(server_error.is_server_error());
        assert!(!server_error.is_client_error());
    }
    
    #[test]
    fn test_retryable_errors() {
        // Create a mock network error using a timeout simulation
        let mock_error = std::io::Error::new(std::io::ErrorKind::TimedOut, "Connection timed out");
        let retryable = IntegrationError::NetworkError(
            reqwest::Error::from(mock_error)
        );
        assert!(retryable.is_retryable());
        
        let non_retryable = IntegrationError::ValidationError("test".to_string());
        assert!(!non_retryable.is_retryable());
    }
    
    #[test]
    fn test_http_status_codes() {
        assert_eq!(IntegrationError::ValidationError("test".to_string()).http_status_code(), 400);
        assert_eq!(IntegrationError::AuthenticationError("test".to_string()).http_status_code(), 401);
        assert_eq!(IntegrationError::ResourceNotFound("test".to_string()).http_status_code(), 404);
        assert_eq!(IntegrationError::Internal("test".to_string()).http_status_code(), 500);
    }
    
    #[test]
    fn test_recovery_strategies() {
        let network_error = IntegrationError::NetworkError(
            reqwest::Error::from(url::ParseError::EmptyHost)
        );
        
        matches!(
            network_error.recovery_strategy(),
            RecoveryStrategy::RetryExponential { .. }
        );
        
        let validation_error = IntegrationError::ValidationError("test".to_string());
        assert_eq!(validation_error.recovery_strategy(), RecoveryStrategy::FailFast);
    }
    
    #[test]
    fn test_enhanced_error() {
        let error = IntegrationError::ComponentNotFound("test-component".to_string());
        let enhanced = error.with_context(
            Some("gateway".to_string()),
            Some(uuid::Uuid::new_v4()),
            std::collections::HashMap::new(),
        );
        
        assert_eq!(enhanced.context.code, "COMPONENT_NOT_FOUND");
        assert!(!enhanced.context.recovery_suggestions.is_empty());
        assert_eq!(enhanced.context.component, Some("gateway".to_string()));
    }
    
    #[test]
    fn test_error_metrics() {
        let mut metrics = ErrorMetrics::default();
        
        let error1 = IntegrationError::ValidationError("test".to_string());
        // Create a simple network error using a URL parse error
        let error2 = IntegrationError::NetworkError(
            reqwest::Error::from(url::ParseError::EmptyHost)
        );
        
        metrics.record_error(&error1, Some("gateway"));
        metrics.record_error(&error2, Some("pipeline"));
        
        assert_eq!(metrics.total_errors, 2);
        assert_eq!(metrics.errors_by_type.get("ValidationError"), Some(&1));
        assert_eq!(metrics.errors_by_type.get("NetworkError"), Some(&1));
        assert_eq!(metrics.errors_by_component.get("gateway"), Some(&1));
        assert_eq!(metrics.errors_by_component.get("pipeline"), Some(&1));
    }
}
