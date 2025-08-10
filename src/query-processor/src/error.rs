//! Error handling for the Query Processor
//!
//! This module provides comprehensive error types and handling for all processor components,
//! with proper error chaining, context, and recovery strategies.

// use std::fmt; // Unused
use thiserror::Error;

/// Result type alias for Query Processor operations
pub type Result<T> = std::result::Result<T, ProcessorError>;

/// Main error type for Query Processor operations
#[derive(Error, Debug, Clone)]
pub enum ProcessorError {
    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Query validation errors
    #[error("Invalid query: {reason}")]
    InvalidQuery { reason: String },

    /// Analysis errors
    #[error("Query analysis failed: {stage} - {reason}")]
    AnalysisFailed { stage: String, reason: String },

    /// Entity extraction errors
    #[error("Entity extraction failed: {reason}")]
    EntityExtractionFailed { reason: String },

    /// Term extraction errors
    #[error("Term extraction failed: {reason}")]
    TermExtractionFailed { reason: String },

    /// Intent classification errors
    #[error("Intent classification failed: {reason}")]
    IntentClassificationFailed { reason: String },

    /// Strategy selection errors
    #[error("Strategy selection failed: {reason}")]
    StrategySelectionFailed { reason: String },

    /// Consensus errors
    #[error("Consensus validation failed: {reason}")]
    ConsensusFailed { reason: String },

    /// Validation errors
    #[error("Validation failed: {field} - {reason}")]
    ValidationFailed { field: String, reason: String },

    /// Performance errors
    #[error("Performance constraint violated: {constraint} - {details}")]
    PerformanceViolation { constraint: String, details: String },

    /// Resource errors
    #[error("Resource error: {resource} - {reason}")]
    ResourceError { resource: String, reason: String },

    /// Network/IO errors
    #[error("Network error: {operation} - {reason}")]
    NetworkError { operation: String, reason: String },

    /// Timeout errors
    #[error("Operation timed out: {operation} after {duration:?}")]
    Timeout {
        operation: String,
        duration: std::time::Duration,
    },

    /// Concurrency errors
    #[error("Concurrency error: {reason}")]
    ConcurrencyError { reason: String },

    /// External service errors
    #[error("External service error: {service} - {reason}")]
    ExternalServiceError { service: String, reason: String },

    /// Data processing errors
    #[error("Data processing error: {stage} - {reason}")]
    DataProcessingError { stage: String, reason: String },

    /// Serialization/deserialization errors
    #[error("Serialization error: {format} - {reason}")]
    SerializationError { format: String, reason: String },

    /// Authentication/authorization errors
    #[error("Authentication error: {reason}")]
    AuthenticationError { reason: String },

    /// Rate limiting errors
    #[error("Rate limit exceeded: {limit} requests per {window:?}")]
    RateLimitExceeded {
        limit: u32,
        window: std::time::Duration,
    },

    /// Memory errors
    #[error("Memory error: {reason}")]
    MemoryError { reason: String },

    /// Neural network errors
    #[error("Neural network error: {model} - {reason}")]
    NeuralNetworkError { model: String, reason: String },

    /// Consensus algorithm errors
    #[error("Consensus algorithm error: {algorithm} - {reason}")]
    ConsensusAlgorithmError { algorithm: String, reason: String },

    /// Byzantine fault errors
    #[error("Byzantine fault detected: {details}")]
    ByzantineFault { details: String },

    /// Citation tracking errors
    #[error("Citation tracking error: {reason}")]
    CitationError { reason: String },

    /// Database errors
    #[error("Database error: {operation} - {reason}")]
    DatabaseError { operation: String, reason: String },

    /// Cache errors
    #[error("Cache error: {operation} - {reason}")]
    CacheError { operation: String, reason: String },

    /// Metrics errors
    #[error("Metrics error: {metric} - {reason}")]
    MetricsError { metric: String, reason: String },

    /// General processing error
    #[error("Processing failed: {0}")]
    ProcessingFailed(String),

    /// Internal errors (should not occur in normal operation)
    #[error("Internal error: {reason}")]
    Internal { reason: String },
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorSeverity {
    /// Low severity - operation can continue with degraded functionality
    Low,
    /// Medium severity - operation should be retried or alternative approach used
    Medium,
    /// High severity - operation should fail but system can continue
    High,
    /// Critical severity - system integrity compromised, immediate attention required
    Critical,
}

/// Error recovery strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// Retry the operation with the same parameters
    Retry,
    /// Retry with modified parameters
    RetryWithModification,
    /// Use fallback approach
    Fallback,
    /// Skip this operation and continue
    Skip,
    /// Fail fast - don't attempt recovery
    FailFast,
}

/// Error context with additional metadata
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Error severity level
    pub severity: ErrorSeverity,
    /// Suggested recovery strategy
    pub recovery: RecoveryStrategy,
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Correlation ID for tracing
    pub correlation_id: Option<uuid::Uuid>,
}

impl ProcessorError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ProcessorError::Configuration { .. } => ErrorSeverity::High,
            ProcessorError::InvalidQuery { .. } => ErrorSeverity::Medium,
            ProcessorError::AnalysisFailed { .. } => ErrorSeverity::Medium,
            ProcessorError::EntityExtractionFailed { .. } => ErrorSeverity::Low,
            ProcessorError::TermExtractionFailed { .. } => ErrorSeverity::Low,
            ProcessorError::IntentClassificationFailed { .. } => ErrorSeverity::Medium,
            ProcessorError::StrategySelectionFailed { .. } => ErrorSeverity::Medium,
            ProcessorError::ConsensusFailed { .. } => ErrorSeverity::High,
            ProcessorError::ValidationFailed { .. } => ErrorSeverity::High,
            ProcessorError::PerformanceViolation { .. } => ErrorSeverity::Medium,
            ProcessorError::ResourceError { .. } => ErrorSeverity::High,
            ProcessorError::NetworkError { .. } => ErrorSeverity::Medium,
            ProcessorError::Timeout { .. } => ErrorSeverity::Medium,
            ProcessorError::ConcurrencyError { .. } => ErrorSeverity::High,
            ProcessorError::ExternalServiceError { .. } => ErrorSeverity::Medium,
            ProcessorError::DataProcessingError { .. } => ErrorSeverity::Medium,
            ProcessorError::SerializationError { .. } => ErrorSeverity::Medium,
            ProcessorError::AuthenticationError { .. } => ErrorSeverity::High,
            ProcessorError::RateLimitExceeded { .. } => ErrorSeverity::Low,
            ProcessorError::MemoryError { .. } => ErrorSeverity::High,
            ProcessorError::NeuralNetworkError { .. } => ErrorSeverity::Medium,
            ProcessorError::ConsensusAlgorithmError { .. } => ErrorSeverity::High,
            ProcessorError::ByzantineFault { .. } => ErrorSeverity::Critical,
            ProcessorError::CitationError { .. } => ErrorSeverity::Medium,
            ProcessorError::DatabaseError { .. } => ErrorSeverity::High,
            ProcessorError::CacheError { .. } => ErrorSeverity::Low,
            ProcessorError::MetricsError { .. } => ErrorSeverity::Low,
            ProcessorError::ProcessingFailed(_) => ErrorSeverity::Medium,
            ProcessorError::Internal { .. } => ErrorSeverity::Critical,
        }
    }

    /// Get the suggested recovery strategy for this error
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            ProcessorError::Configuration { .. } => RecoveryStrategy::FailFast,
            ProcessorError::InvalidQuery { .. } => RecoveryStrategy::FailFast,
            ProcessorError::AnalysisFailed { .. } => RecoveryStrategy::Fallback,
            ProcessorError::EntityExtractionFailed { .. } => RecoveryStrategy::Skip,
            ProcessorError::TermExtractionFailed { .. } => RecoveryStrategy::Skip,
            ProcessorError::IntentClassificationFailed { .. } => RecoveryStrategy::Fallback,
            ProcessorError::StrategySelectionFailed { .. } => RecoveryStrategy::Fallback,
            ProcessorError::ConsensusFailed { .. } => RecoveryStrategy::Retry,
            ProcessorError::ValidationFailed { .. } => RecoveryStrategy::FailFast,
            ProcessorError::PerformanceViolation { .. } => RecoveryStrategy::RetryWithModification,
            ProcessorError::ResourceError { .. } => RecoveryStrategy::Retry,
            ProcessorError::NetworkError { .. } => RecoveryStrategy::Retry,
            ProcessorError::Timeout { .. } => RecoveryStrategy::RetryWithModification,
            ProcessorError::ConcurrencyError { .. } => RecoveryStrategy::Retry,
            ProcessorError::ExternalServiceError { .. } => RecoveryStrategy::Retry,
            ProcessorError::DataProcessingError { .. } => RecoveryStrategy::Fallback,
            ProcessorError::SerializationError { .. } => RecoveryStrategy::FailFast,
            ProcessorError::AuthenticationError { .. } => RecoveryStrategy::FailFast,
            ProcessorError::RateLimitExceeded { .. } => RecoveryStrategy::RetryWithModification,
            ProcessorError::MemoryError { .. } => RecoveryStrategy::RetryWithModification,
            ProcessorError::NeuralNetworkError { .. } => RecoveryStrategy::Fallback,
            ProcessorError::ConsensusAlgorithmError { .. } => RecoveryStrategy::Fallback,
            ProcessorError::ByzantineFault { .. } => RecoveryStrategy::FailFast,
            ProcessorError::CitationError { .. } => RecoveryStrategy::Skip,
            ProcessorError::DatabaseError { .. } => RecoveryStrategy::Retry,
            ProcessorError::CacheError { .. } => RecoveryStrategy::Skip,
            ProcessorError::MetricsError { .. } => RecoveryStrategy::Skip,
            ProcessorError::ProcessingFailed(_) => RecoveryStrategy::Fallback,
            ProcessorError::Internal { .. } => RecoveryStrategy::FailFast,
        }
    }

    /// Create error context with default values
    pub fn with_context(self) -> (Self, ErrorContext) {
        let context = ErrorContext {
            severity: self.severity(),
            recovery: self.recovery_strategy(),
            context: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            correlation_id: None,
        };
        (self, context)
    }

    /// Create error context with correlation ID
    pub fn with_correlation_id(self, correlation_id: uuid::Uuid) -> (Self, ErrorContext) {
        let mut context = ErrorContext {
            severity: self.severity(),
            recovery: self.recovery_strategy(),
            context: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            correlation_id: Some(correlation_id),
        };
        context.context.insert("correlation_id".to_string(), correlation_id.to_string());
        (self, context)
    }

    /// Add additional context information
    pub fn add_context(self, key: &str, value: &str) -> (Self, ErrorContext) {
        let mut context_map = std::collections::HashMap::new();
        context_map.insert(key.to_string(), value.to_string());
        
        let context = ErrorContext {
            severity: self.severity(),
            recovery: self.recovery_strategy(),
            context: context_map,
            timestamp: chrono::Utc::now(),
            correlation_id: None,
        };
        (self, context)
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        !matches!(self.recovery_strategy(), RecoveryStrategy::FailFast)
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.recovery_strategy(),
            RecoveryStrategy::Retry | RecoveryStrategy::RetryWithModification
        )
    }

    /// Get error code for structured logging
    pub fn error_code(&self) -> &'static str {
        match self {
            ProcessorError::Configuration { .. } => "CONFIG_ERROR",
            ProcessorError::InvalidQuery { .. } => "INVALID_QUERY",
            ProcessorError::AnalysisFailed { .. } => "ANALYSIS_FAILED",
            ProcessorError::EntityExtractionFailed { .. } => "ENTITY_EXTRACTION_FAILED",
            ProcessorError::TermExtractionFailed { .. } => "TERM_EXTRACTION_FAILED",
            ProcessorError::IntentClassificationFailed { .. } => "INTENT_CLASSIFICATION_FAILED",
            ProcessorError::StrategySelectionFailed { .. } => "STRATEGY_SELECTION_FAILED",
            ProcessorError::ConsensusFailed { .. } => "CONSENSUS_FAILED",
            ProcessorError::ValidationFailed { .. } => "VALIDATION_FAILED",
            ProcessorError::PerformanceViolation { .. } => "PERFORMANCE_VIOLATION",
            ProcessorError::ResourceError { .. } => "RESOURCE_ERROR",
            ProcessorError::NetworkError { .. } => "NETWORK_ERROR",
            ProcessorError::Timeout { .. } => "TIMEOUT",
            ProcessorError::ConcurrencyError { .. } => "CONCURRENCY_ERROR",
            ProcessorError::ExternalServiceError { .. } => "EXTERNAL_SERVICE_ERROR",
            ProcessorError::DataProcessingError { .. } => "DATA_PROCESSING_ERROR",
            ProcessorError::SerializationError { .. } => "SERIALIZATION_ERROR",
            ProcessorError::AuthenticationError { .. } => "AUTHENTICATION_ERROR",
            ProcessorError::RateLimitExceeded { .. } => "RATE_LIMIT_EXCEEDED",
            ProcessorError::MemoryError { .. } => "MEMORY_ERROR",
            ProcessorError::NeuralNetworkError { .. } => "NEURAL_NETWORK_ERROR",
            ProcessorError::ConsensusAlgorithmError { .. } => "CONSENSUS_ALGORITHM_ERROR",
            ProcessorError::ByzantineFault { .. } => "BYZANTINE_FAULT",
            ProcessorError::CitationError { .. } => "CITATION_ERROR",
            ProcessorError::DatabaseError { .. } => "DATABASE_ERROR",
            ProcessorError::CacheError { .. } => "CACHE_ERROR",
            ProcessorError::MetricsError { .. } => "METRICS_ERROR",
            ProcessorError::ProcessingFailed(_) => "PROCESSING_FAILED",
            ProcessorError::Internal { .. } => "INTERNAL_ERROR",
        }
    }
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(severity: ErrorSeverity, recovery: RecoveryStrategy) -> Self {
        Self {
            severity,
            recovery,
            context: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            correlation_id: None,
        }
    }

    /// Add context information
    pub fn add_context(&mut self, key: &str, value: &str) {
        self.context.insert(key.to_string(), value.to_string());
    }

    /// Set correlation ID
    pub fn set_correlation_id(&mut self, correlation_id: uuid::Uuid) {
        self.correlation_id = Some(correlation_id);
        self.context.insert("correlation_id".to_string(), correlation_id.to_string());
    }

    /// Get context value
    pub fn get_context(&self, key: &str) -> Option<&String> {
        self.context.get(key)
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new(ErrorSeverity::Medium, RecoveryStrategy::Fallback)
    }
}

/// Error metrics for monitoring and alerting
#[derive(Debug, Clone)]
pub struct ErrorMetrics {
    /// Total error count
    pub total_errors: u64,
    /// Error count by type
    pub errors_by_type: std::collections::HashMap<String, u64>,
    /// Error count by severity
    pub errors_by_severity: std::collections::HashMap<ErrorSeverity, u64>,
    /// Recovery success rate
    pub recovery_success_rate: f64,
    /// Average time to recovery
    pub average_recovery_time: std::time::Duration,
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_by_type: std::collections::HashMap::new(),
            errors_by_severity: std::collections::HashMap::new(),
            recovery_success_rate: 0.0,
            average_recovery_time: std::time::Duration::ZERO,
        }
    }
}

impl ErrorMetrics {
    /// Record an error occurrence
    pub fn record_error(&mut self, error: &ProcessorError) {
        self.total_errors += 1;
        
        let error_code = error.error_code();
        *self.errors_by_type.entry(error_code.to_string()).or_insert(0) += 1;
        
        let severity = error.severity();
        *self.errors_by_severity.entry(severity).or_insert(0) += 1;
    }

    /// Record recovery success
    pub fn record_recovery(&mut self, success: bool, duration: std::time::Duration) {
        // Update recovery success rate using exponential moving average
        let alpha = 0.1;
        let success_value = if success { 1.0 } else { 0.0 };
        self.recovery_success_rate = alpha * success_value + (1.0 - alpha) * self.recovery_success_rate;
        
        // Update average recovery time
        if self.average_recovery_time.is_zero() {
            self.average_recovery_time = duration;
        } else {
            let current_nanos = self.average_recovery_time.as_nanos() as f64;
            let new_nanos = duration.as_nanos() as f64;
            let avg_nanos = alpha * new_nanos + (1.0 - alpha) * current_nanos;
            self.average_recovery_time = std::time::Duration::from_nanos(avg_nanos as u64);
        }
    }

    /// Get error rate for a specific type
    pub fn error_rate_for_type(&self, error_type: &str) -> f64 {
        if self.total_errors == 0 {
            return 0.0;
        }
        
        let count = self.errors_by_type.get(error_type).copied().unwrap_or(0);
        count as f64 / self.total_errors as f64
    }

    /// Get error rate for a specific severity
    pub fn error_rate_for_severity(&self, severity: ErrorSeverity) -> f64 {
        if self.total_errors == 0 {
            return 0.0;
        }
        
        let count = self.errors_by_severity.get(&severity).copied().unwrap_or(0);
        count as f64 / self.total_errors as f64
    }
}

// Implement standard error conversions
impl From<std::io::Error> for ProcessorError {
    fn from(err: std::io::Error) -> Self {
        ProcessorError::ResourceError {
            resource: "filesystem".to_string(),
            reason: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for ProcessorError {
    fn from(err: serde_json::Error) -> Self {
        ProcessorError::SerializationError {
            format: "json".to_string(),
            reason: err.to_string(),
        }
    }
}

impl From<reqwest::Error> for ProcessorError {
    fn from(err: reqwest::Error) -> Self {
        ProcessorError::NetworkError {
            operation: "http_request".to_string(),
            reason: err.to_string(),
        }
    }
}

impl From<tokio::time::error::Elapsed> for ProcessorError {
    fn from(err: tokio::time::error::Elapsed) -> Self {
        ProcessorError::Timeout {
            operation: "async_operation".to_string(),
            duration: std::time::Duration::from_secs(5), // Default timeout
        }
    }
}

impl From<validator::ValidationErrors> for ProcessorError {
    fn from(err: validator::ValidationErrors) -> Self {
        ProcessorError::ValidationFailed {
            field: "configuration".to_string(),
            reason: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_severity() {
        let error = ProcessorError::ByzantineFault {
            details: "Multiple conflicting responses".to_string(),
        };
        assert_eq!(error.severity(), ErrorSeverity::Critical);
    }

    #[test]
    fn test_recovery_strategy() {
        let error = ProcessorError::NetworkError {
            operation: "api_call".to_string(),
            reason: "Connection timeout".to_string(),
        };
        assert_eq!(error.recovery_strategy(), RecoveryStrategy::Retry);
    }

    #[test]
    fn test_error_context() {
        let error = ProcessorError::ValidationFailed {
            field: "query".to_string(),
            reason: "Empty query text".to_string(),
        };
        
        let (_, context) = error.with_context();
        assert_eq!(context.severity, ErrorSeverity::High);
        assert_eq!(context.recovery, RecoveryStrategy::FailFast);
    }

    #[test]
    fn test_error_metrics() {
        let mut metrics = ErrorMetrics::default();
        
        let error = ProcessorError::ConsensusFailed {
            reason: "Insufficient responses".to_string(),
        };
        
        metrics.record_error(&error);
        assert_eq!(metrics.total_errors, 1);
        assert!(metrics.errors_by_type.contains_key("CONSENSUS_FAILED"));
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable_error = ProcessorError::NetworkError {
            operation: "test".to_string(),
            reason: "timeout".to_string(),
        };
        assert!(recoverable_error.is_recoverable());
        
        let non_recoverable_error = ProcessorError::ByzantineFault {
            details: "attack detected".to_string(),
        };
        assert!(!non_recoverable_error.is_recoverable());
    }
}