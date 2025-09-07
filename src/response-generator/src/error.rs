//! Error types and handling for the response generator

use std::time::Duration;
use thiserror::Error;

/// Result type alias for response generator operations
pub type Result<T> = std::result::Result<T, ResponseError>;

/// Comprehensive error types for response generation
#[derive(Error, Debug)]
pub enum ResponseError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Invalid request parameters
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Context processing errors
    #[error("Context processing failed: {source}")]
    ContextProcessing {
        #[from]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Response generation failures
    #[error("Response generation failed: {reason}")]
    GenerationFailed { reason: String },

    /// Validation failures
    #[error("Validation failed: {details}")]
    ValidationFailed { details: String },

    /// Insufficient confidence in generated response
    #[error("Response confidence {actual:.3} below required threshold {required:.3}")]
    InsufficientConfidence { actual: f64, required: f64 },

    /// Citation processing errors
    #[error("Citation processing failed: {message}")]
    CitationError { message: String },

    /// Cache operation errors
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Formatting errors
    #[error("Response formatting failed for format {format:?}: {reason}")]
    FormattingError { format: String, reason: String },

    /// Performance target violations
    #[error("Performance target violated: operation took {actual:?}, target was {target:?}")]
    PerformanceViolation { actual: Duration, target: Duration },

    /// Resource exhaustion
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },

    /// Network/external service errors
    #[error("External service error: {service} - {message}")]
    ExternalService { service: String, message: String },

    /// Pipeline processing errors
    #[error("Pipeline stage '{stage}' failed: {reason}")]
    PipelineError { stage: String, reason: String },

    /// Streaming response errors
    #[error("Streaming error: {message}")]
    StreamingError { message: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// YAML serialization errors
    #[error("YAML serialization error: {0}")]
    YamlSerialization(#[from] serde_yaml::Error),

    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP client errors
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Timeout errors
    #[error("Operation timed out after {duration:?}")]
    Timeout { duration: Duration },

    /// Concurrency errors
    #[error("Concurrency error: {message}")]
    Concurrency { message: String },

    /// Internal errors (should not occur in production)
    #[error("Internal error: {message}")]
    Internal { message: String },
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry with exponential backoff
    Retry { max_attempts: u32, base_delay: Duration },
    
    /// Fallback to simpler processing
    Fallback { strategy: String },
    
    /// Graceful degradation with reduced quality
    Degrade { quality_reduction: f64 },
    
    /// Fail fast - no recovery possible
    FailFast,
}

/// Error context for better debugging and monitoring
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Request ID associated with the error
    pub request_id: Option<uuid::Uuid>,
    
    /// Processing stage where error occurred
    pub stage: Option<String>,
    
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
    
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Recommended recovery strategy
    pub recovery_strategy: RecoveryStrategy,
}

impl ResponseError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Configuration { message: message.into() }
    }

    /// Create an invalid request error
    pub fn invalid_request<S: Into<String>>(message: S) -> Self {
        Self::InvalidRequest(message.into())
    }

    /// Create a generation failed error
    pub fn generation_failed<S: Into<String>>(reason: S) -> Self {
        Self::GenerationFailed { reason: reason.into() }
    }

    /// Create a validation failed error
    pub fn validation_failed<S: Into<String>>(details: S) -> Self {
        Self::ValidationFailed { details: details.into() }
    }

    /// Create a citation error
    pub fn citation<S: Into<String>>(message: S) -> Self {
        Self::CitationError { message: message.into() }
    }

    /// Create a formatting error
    pub fn formatting<S: Into<String>>(format: S, reason: S) -> Self {
        Self::FormattingError { 
            format: format.into(), 
            reason: reason.into() 
        }
    }

    /// Create a performance violation error
    pub fn performance_violation(actual: Duration, target: Duration) -> Self {
        Self::PerformanceViolation { actual, target }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted<S: Into<String>>(resource: S) -> Self {
        Self::ResourceExhausted { resource: resource.into() }
    }

    /// Create an external service error
    pub fn external_service<S: Into<String>>(service: S, message: S) -> Self {
        Self::ExternalService { 
            service: service.into(), 
            message: message.into() 
        }
    }

    /// Create a pipeline error
    pub fn pipeline<S: Into<String>>(stage: S, reason: S) -> Self {
        Self::PipelineError { 
            stage: stage.into(), 
            reason: reason.into() 
        }
    }

    /// Create a streaming error
    pub fn streaming<S: Into<String>>(message: S) -> Self {
        Self::StreamingError { message: message.into() }
    }

    /// Create a timeout error
    pub fn timeout(duration: Duration) -> Self {
        Self::Timeout { duration }
    }

    /// Create a concurrency error
    pub fn concurrency<S: Into<String>>(message: S) -> Self {
        Self::Concurrency { message: message.into() }
    }

    /// Create an internal error
    pub fn internal<S: Into<String>>(message: S) -> Self {
        Self::Internal { message: message.into() }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::ExternalService { .. } => true,
            Self::Timeout { .. } => true,
            Self::Http(_) => true,
            Self::Io(_) => true,
            Self::PerformanceViolation { .. } => false,
            Self::ValidationFailed { .. } => false,
            Self::InvalidRequest(_) => false,
            Self::Configuration { .. } => false,
            Self::InsufficientConfidence { .. } => false,
            _ => true,
        }
    }

    /// Get recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            Self::ExternalService { .. } => RecoveryStrategy::Retry {
                max_attempts: 3,
                base_delay: Duration::from_millis(100),
            },
            Self::Timeout { .. } => RecoveryStrategy::Retry {
                max_attempts: 2,
                base_delay: Duration::from_millis(500),
            },
            Self::PerformanceViolation { .. } => RecoveryStrategy::Degrade {
                quality_reduction: 0.1,
            },
            Self::ValidationFailed { .. } => RecoveryStrategy::Fallback {
                strategy: "simplified_validation".to_string(),
            },
            Self::ResourceExhausted { .. } => RecoveryStrategy::Fallback {
                strategy: "reduced_complexity".to_string(),
            },
            Self::InvalidRequest(_) => RecoveryStrategy::FailFast,
            Self::Configuration { .. } => RecoveryStrategy::FailFast,
            _ => RecoveryStrategy::Retry {
                max_attempts: 2,
                base_delay: Duration::from_millis(200),
            },
        }
    }

    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Internal { .. } => ErrorSeverity::Critical,
            Self::Configuration { .. } => ErrorSeverity::Critical,
            Self::ValidationFailed { .. } => ErrorSeverity::High,
            Self::InsufficientConfidence { .. } => ErrorSeverity::High,
            Self::PerformanceViolation { .. } => ErrorSeverity::Medium,
            Self::ExternalService { .. } => ErrorSeverity::Medium,
            Self::Timeout { .. } => ErrorSeverity::Medium,
            Self::InvalidRequest(_) => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium,
        }
    }

    /// Convert to structured error context
    pub fn to_context(&self, request_id: Option<uuid::Uuid>, stage: Option<String>) -> ErrorContext {
        let mut context = std::collections::HashMap::new();
        context.insert("error_type".to_string(), format!("{:?}", self));
        context.insert("severity".to_string(), format!("{:?}", self.severity()));
        context.insert("retryable".to_string(), self.is_retryable().to_string());

        ErrorContext {
            request_id,
            stage,
            context,
            timestamp: chrono::Utc::now(),
            recovery_strategy: self.recovery_strategy(),
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - user errors, non-critical issues
    Low,
    /// Medium severity - processing errors, performance issues
    Medium,
    /// High severity - validation failures, quality issues
    High,
    /// Critical severity - system errors, configuration problems
    Critical,
}

/// Error reporter for structured logging and monitoring
pub struct ErrorReporter {
    /// Whether to enable structured logging
    structured_logging: bool,
    
    /// Metrics endpoint for error reporting
    metrics_endpoint: Option<String>,
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self {
            structured_logging: true,
            metrics_endpoint: None,
        }
    }
}

impl ErrorReporter {
    /// Create new error reporter
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable metrics reporting
    pub fn with_metrics<S: Into<String>>(mut self, endpoint: S) -> Self {
        self.metrics_endpoint = Some(endpoint.into());
        self
    }

    /// Report an error with full context
    pub async fn report(&self, error: &ResponseError, context: ErrorContext) -> Result<()> {
        // Structured logging
        if self.structured_logging {
            match error.severity() {
                ErrorSeverity::Critical => {
                    tracing::error!(
                        error = %error,
                        request_id = ?context.request_id,
                        stage = ?context.stage,
                        severity = ?error.severity(),
                        retryable = error.is_retryable(),
                        recovery_strategy = ?context.recovery_strategy,
                        "Critical error occurred"
                    );
                }
                ErrorSeverity::High => {
                    tracing::error!(
                        error = %error,
                        request_id = ?context.request_id,
                        stage = ?context.stage,
                        "High severity error occurred"
                    );
                }
                ErrorSeverity::Medium => {
                    tracing::warn!(
                        error = %error,
                        request_id = ?context.request_id,
                        stage = ?context.stage,
                        "Medium severity error occurred"
                    );
                }
                ErrorSeverity::Low => {
                    tracing::info!(
                        error = %error,
                        request_id = ?context.request_id,
                        "Low severity error occurred"
                    );
                }
            }
        }

        // Metrics reporting
        if let Some(endpoint) = &self.metrics_endpoint {
            self.report_metrics(error, &context, endpoint).await?;
        }

        Ok(())
    }

    /// Report metrics to external endpoint
    async fn report_metrics(
        &self,
        error: &ResponseError,
        context: &ErrorContext,
        endpoint: &str,
    ) -> Result<()> {
        let client = reqwest::Client::new();
        let metric = serde_json::json!({
            "error_type": format!("{:?}", error),
            "severity": format!("{:?}", error.severity()),
            "retryable": error.is_retryable(),
            "request_id": context.request_id,
            "stage": context.stage,
            "timestamp": context.timestamp,
        });

        client.post(endpoint)
            .json(&metric)
            .send()
            .await?;

        Ok(())
    }
}

/// Utility macros for error handling
#[macro_export]
macro_rules! ensure_config {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return Err(ResponseError::config($msg));
        }
    };
}

#[macro_export]
macro_rules! ensure_valid {
    ($cond:expr, $msg:expr) => {
        if !$cond {
            return Err(ResponseError::invalid_request($msg));
        }
    };
}

#[macro_export]
macro_rules! ensure_performance {
    ($duration:expr, $target:expr) => {
        if $duration > $target {
            return Err(ResponseError::performance_violation($duration, $target));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_error_creation() {
        let err = ResponseError::config("test message");
        assert!(matches!(err, ResponseError::Configuration { .. }));

        let err = ResponseError::performance_violation(
            Duration::from_millis(200),
            Duration::from_millis(100),
        );
        assert!(matches!(err, ResponseError::PerformanceViolation { .. }));
    }

    #[test]
    fn test_error_retryable() {
        assert!(ResponseError::external_service("test", "message").is_retryable());
        assert!(!ResponseError::invalid_request("test").is_retryable());
        assert!(!ResponseError::config("test").is_retryable());
    }

    #[test]
    fn test_error_severity() {
        assert_eq!(ResponseError::internal("test").severity(), ErrorSeverity::Critical);
        assert_eq!(ResponseError::validation_failed("test").severity(), ErrorSeverity::High);
        assert_eq!(ResponseError::invalid_request("test").severity(), ErrorSeverity::Low);
    }

    #[test]
    fn test_recovery_strategy() {
        let err = ResponseError::external_service("test", "message");
        match err.recovery_strategy() {
            RecoveryStrategy::Retry { max_attempts, .. } => {
                assert_eq!(max_attempts, 3);
            }
            _ => panic!("Expected retry strategy"),
        }
    }

    #[tokio::test]
    async fn test_error_reporter() {
        let reporter = ErrorReporter::new();
        let error = ResponseError::config("test error");
        let context = error.to_context(None, Some("test_stage".to_string()));

        // Should not panic
        reporter.report(&error, context).await.unwrap();
    }
}