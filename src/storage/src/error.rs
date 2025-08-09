//! Error types and handling for MongoDB Vector Storage

use std::fmt;
use thiserror::Error;
use uuid::Uuid;

/// Main error type for storage operations
#[derive(Error, Debug)]
pub enum StorageError {
    /// MongoDB connection or client errors
    #[error("Database connection error: {0}")]
    ConnectionError(String),
    
    /// Database operation errors
    #[error("Database operation failed: {0}")]
    DatabaseError(String),
    
    /// Document validation errors
    #[error("Document validation failed: {0}")]
    ValidationError(String),
    
    /// Chunk not found errors
    #[error("Chunk not found: {0}")]
    ChunkNotFound(Uuid),
    
    /// Document not found errors
    #[error("Document not found: {0}")]
    DocumentNotFound(Uuid),
    
    /// Embedding-related errors
    #[error("Embedding not found for chunk: {0}")]
    EmbeddingNotFound(Uuid),
    
    /// Invalid embedding dimension
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidEmbeddingDimension { expected: usize, actual: usize },
    
    /// Search query errors
    #[error("Invalid search query: {0}")]
    InvalidQuery(String),
    
    /// Index operation errors
    #[error("Index operation failed: {0}")]
    IndexError(String),
    
    /// Serialization/deserialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    /// Timeout errors
    #[error("Operation timed out: {0}")]
    TimeoutError(String),
    
    /// Resource limit errors
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitError(String),
    
    /// Concurrent access errors
    #[error("Concurrent access conflict: {0}")]
    ConcurrencyError(String),
    
    /// Transaction errors
    #[error("Transaction failed: {0}")]
    TransactionError(String),
    
    /// Bulk operation errors
    #[error("Bulk operation failed: {inserted} succeeded, {failed} failed")]
    BulkOperationError { inserted: u64, failed: u64 },
    
    /// Vector operation errors
    #[error("Vector operation failed: {0}")]
    VectorError(String),
    
    /// Authentication/authorization errors
    #[error("Authentication failed: {0}")]
    AuthError(String),
    
    /// Generic internal errors
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl StorageError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            StorageError::ConnectionError(_) => true,
            StorageError::TimeoutError(_) => true,
            StorageError::ConcurrencyError(_) => true,
            StorageError::TransactionError(_) => true,
            StorageError::DatabaseError(msg) if msg.contains("connection") => true,
            StorageError::DatabaseError(msg) if msg.contains("timeout") => true,
            _ => false,
        }
    }
    
    /// Check if error is a client error (4xx equivalent)
    pub fn is_client_error(&self) -> bool {
        match self {
            StorageError::ValidationError(_) => true,
            StorageError::ChunkNotFound(_) => true,
            StorageError::DocumentNotFound(_) => true,
            StorageError::InvalidQuery(_) => true,
            StorageError::InvalidEmbeddingDimension { .. } => true,
            StorageError::ConfigError(_) => true,
            StorageError::AuthError(_) => true,
            _ => false,
        }
    }
    
    /// Check if error is a server error (5xx equivalent)
    pub fn is_server_error(&self) -> bool {
        match self {
            StorageError::ConnectionError(_) => true,
            StorageError::DatabaseError(_) => true,
            StorageError::IndexError(_) => true,
            StorageError::TimeoutError(_) => true,
            StorageError::ResourceLimitError(_) => true,
            StorageError::ConcurrencyError(_) => true,
            StorageError::TransactionError(_) => true,
            StorageError::InternalError(_) => true,
            _ => false,
        }
    }
    
    /// Get error category for metrics and logging
    pub fn category(&self) -> &'static str {
        match self {
            StorageError::ConnectionError(_) => "connection",
            StorageError::DatabaseError(_) => "database",
            StorageError::ValidationError(_) => "validation",
            StorageError::ChunkNotFound(_) => "not_found",
            StorageError::DocumentNotFound(_) => "not_found",
            StorageError::EmbeddingNotFound(_) => "embedding",
            StorageError::InvalidEmbeddingDimension { .. } => "embedding",
            StorageError::InvalidQuery(_) => "query",
            StorageError::IndexError(_) => "index",
            StorageError::SerializationError(_) => "serialization",
            StorageError::ConfigError(_) => "config",
            StorageError::TimeoutError(_) => "timeout",
            StorageError::ResourceLimitError(_) => "resource",
            StorageError::ConcurrencyError(_) => "concurrency",
            StorageError::TransactionError(_) => "transaction",
            StorageError::BulkOperationError { .. } => "bulk",
            StorageError::VectorError(_) => "vector",
            StorageError::AuthError(_) => "auth",
            StorageError::InternalError(_) => "internal",
        }
    }
    
    /// Get HTTP status code equivalent
    pub fn status_code(&self) -> u16 {
        match self {
            StorageError::ChunkNotFound(_) => 404,
            StorageError::DocumentNotFound(_) => 404,
            StorageError::ValidationError(_) => 400,
            StorageError::InvalidQuery(_) => 400,
            StorageError::InvalidEmbeddingDimension { .. } => 400,
            StorageError::ConfigError(_) => 400,
            StorageError::AuthError(_) => 401,
            StorageError::ResourceLimitError(_) => 429,
            StorageError::TimeoutError(_) => 408,
            StorageError::ConnectionError(_) => 503,
            StorageError::DatabaseError(_) => 503,
            StorageError::IndexError(_) => 503,
            StorageError::ConcurrencyError(_) => 409,
            StorageError::TransactionError(_) => 500,
            StorageError::BulkOperationError { .. } => 207, // Multi-status
            _ => 500,
        }
    }
}

/// Convert from MongoDB errors
impl From<mongodb::error::Error> for StorageError {
    fn from(err: mongodb::error::Error) -> Self {
        use mongodb::error::ErrorKind;
        
        match err.kind.as_ref() {
            ErrorKind::Authentication { .. } => {
                StorageError::AuthError(err.to_string())
            }
            ErrorKind::ConnectionPoolCleared { .. } => {
                StorageError::ConnectionError(err.to_string())
            }
            ErrorKind::Io(_) => {
                StorageError::ConnectionError(err.to_string())
            }
            ErrorKind::ServerSelection { .. } => {
                StorageError::ConnectionError(err.to_string())
            }
            ErrorKind::Command(cmd_err) => {
                if cmd_err.code == 11000 {
                    StorageError::ConcurrencyError("Duplicate key error".to_string())
                } else if cmd_err.code == 50 {
                    StorageError::TimeoutError("Operation exceeded time limit".to_string())
                } else {
                    StorageError::DatabaseError(err.to_string())
                }
            }
            ErrorKind::BulkWrite(_) => {
                StorageError::BulkOperationError { inserted: 0, failed: 1 }
            }
            ErrorKind::Transaction { .. } => {
                StorageError::TransactionError(err.to_string())
            }
            _ => StorageError::DatabaseError(err.to_string()),
        }
    }
}

/// Convert from BSON serialization errors
impl From<bson::ser::Error> for StorageError {
    fn from(err: bson::ser::Error) -> Self {
        StorageError::SerializationError(format!("BSON serialization error: {}", err))
    }
}

/// Convert from BSON deserialization errors
impl From<bson::de::Error> for StorageError {
    fn from(err: bson::de::Error) -> Self {
        StorageError::SerializationError(format!("BSON deserialization error: {}", err))
    }
}

/// Convert from serde_json errors
impl From<serde_json::Error> for StorageError {
    fn from(err: serde_json::Error) -> Self {
        StorageError::SerializationError(format!("JSON serialization error: {}", err))
    }
}

/// Convert from UUID parsing errors
impl From<uuid::Error> for StorageError {
    fn from(err: uuid::Error) -> Self {
        StorageError::ValidationError(format!("Invalid UUID: {}", err))
    }
}

/// Result type alias for storage operations
pub type StorageResult<T> = Result<T, StorageError>;

/// Error context for detailed error reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that was being performed
    pub operation: String,
    
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
    
    /// Timestamp when error occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Request ID for tracing
    pub request_id: Option<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            context: std::collections::HashMap::new(),
            timestamp: chrono::Utc::now(),
            request_id: None,
        }
    }
    
    /// Add context information
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }
    
    /// Set request ID for tracing
    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }
}

/// Enhanced error with context information
#[derive(Debug)]
pub struct ContextualError {
    pub error: StorageError,
    pub context: ErrorContext,
}

impl ContextualError {
    /// Create a new contextual error
    pub fn new(error: StorageError, context: ErrorContext) -> Self {
        Self { error, context }
    }
    
    /// Create with basic context
    pub fn with_operation(error: StorageError, operation: impl Into<String>) -> Self {
        Self {
            error,
            context: ErrorContext::new(operation),
        }
    }
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Error in operation '{}': {} (at {})",
            self.context.operation,
            self.error,
            self.context.timestamp
        )?;
        
        if let Some(ref request_id) = self.context.request_id {
            write!(f, " [request_id: {}]", request_id)?;
        }
        
        if !self.context.context.is_empty() {
            write!(f, " Context: {:?}", self.context.context)?;
        }
        
        Ok(())
    }
}

impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Trait for adding context to errors
pub trait WithContext<T> {
    /// Add operation context to an error
    fn with_context(self, operation: impl Into<String>) -> Result<T, ContextualError>;
    
    /// Add detailed context to an error
    fn with_detailed_context(self, context: ErrorContext) -> Result<T, ContextualError>;
}

impl<T> WithContext<T> for Result<T, StorageError> {
    fn with_context(self, operation: impl Into<String>) -> Result<T, ContextualError> {
        self.map_err(|e| ContextualError::with_operation(e, operation))
    }
    
    fn with_detailed_context(self, context: ErrorContext) -> Result<T, ContextualError> {
        self.map_err(|e| ContextualError::new(e, context))
    }
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation with exponential backoff
    Retry { max_attempts: u32, base_delay_ms: u64 },
    
    /// Fail fast without retry
    FailFast,
    
    /// Degrade gracefully with reduced functionality
    Degrade,
    
    /// Fallback to alternative implementation
    Fallback,
}

impl RecoveryStrategy {
    /// Get recovery strategy for a specific error
    pub fn for_error(error: &StorageError) -> Self {
        match error {
            StorageError::ConnectionError(_) => Self::Retry {
                max_attempts: 3,
                base_delay_ms: 1000,
            },
            StorageError::TimeoutError(_) => Self::Retry {
                max_attempts: 2,
                base_delay_ms: 500,
            },
            StorageError::ConcurrencyError(_) => Self::Retry {
                max_attempts: 5,
                base_delay_ms: 100,
            },
            StorageError::TransactionError(_) => Self::Retry {
                max_attempts: 3,
                base_delay_ms: 200,
            },
            StorageError::ChunkNotFound(_) => Self::FailFast,
            StorageError::DocumentNotFound(_) => Self::FailFast,
            StorageError::ValidationError(_) => Self::FailFast,
            StorageError::InvalidQuery(_) => Self::FailFast,
            StorageError::ResourceLimitError(_) => Self::Degrade,
            StorageError::BulkOperationError { .. } => Self::Fallback,
            _ => Self::Retry {
                max_attempts: 2,
                base_delay_ms: 1000,
            },
        }
    }
    
    /// Calculate delay for retry attempt
    pub fn calculate_delay(&self, attempt: u32) -> Option<std::time::Duration> {
        match self {
            Self::Retry { max_attempts, base_delay_ms } => {
                if attempt <= *max_attempts {
                    let delay_ms = base_delay_ms * (1 << (attempt - 1).min(10)); // Cap at 1024x
                    Some(std::time::Duration::from_millis(delay_ms))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_categorization() {
        let connection_error = StorageError::ConnectionError("Failed to connect".to_string());
        assert!(connection_error.is_retryable());
        assert!(connection_error.is_server_error());
        assert_eq!(connection_error.category(), "connection");
        assert_eq!(connection_error.status_code(), 503);
        
        let validation_error = StorageError::ValidationError("Invalid data".to_string());
        assert!(!validation_error.is_retryable());
        assert!(validation_error.is_client_error());
        assert_eq!(validation_error.category(), "validation");
        assert_eq!(validation_error.status_code(), 400);
    }
    
    #[test]
    fn test_recovery_strategy() {
        let connection_error = StorageError::ConnectionError("Failed to connect".to_string());
        let strategy = RecoveryStrategy::for_error(&connection_error);
        
        match strategy {
            RecoveryStrategy::Retry { max_attempts, base_delay_ms } => {
                assert_eq!(max_attempts, 3);
                assert_eq!(base_delay_ms, 1000);
                
                let delay1 = strategy.calculate_delay(1).unwrap();
                let delay2 = strategy.calculate_delay(2).unwrap();
                assert!(delay2 > delay1); // Exponential backoff
            }
            _ => panic!("Expected retry strategy for connection error"),
        }
    }
    
    #[test]
    fn test_contextual_error() {
        let storage_error = StorageError::ChunkNotFound(uuid::Uuid::new_v4());
        let context = ErrorContext::new("get_chunk")
            .with_context("chunk_id", "test-id")
            .with_request_id("req-123");
        
        let contextual_error = ContextualError::new(storage_error, context);
        let error_string = contextual_error.to_string();
        
        assert!(error_string.contains("get_chunk"));
        assert!(error_string.contains("req-123"));
    }
}