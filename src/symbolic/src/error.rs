// src/symbolic/src/error.rs
// Error types for symbolic reasoning system

use thiserror::Error;

/// Symbolic reasoning error types
#[derive(Error, Debug)]
pub enum SymbolicError {
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Rule compilation error: {0}")]
    RuleCompilationError(String),
    
    #[error("Rule conflict: {0}")]
    RuleConflict(String),
    
    #[error("Query execution error: {0}")]
    QueryError(String),
    
    #[error("Performance constraint violation: {message} (took {duration_ms}ms, limit {limit_ms}ms)")]
    PerformanceViolation {
        message: String,
        duration_ms: u64,
        limit_ms: u64,
    },
    
    #[error("Proof chain validation error: {0}")]
    ProofValidationError(String),
    
    #[error("Knowledge base error: {0}")]
    KnowledgeBaseError(String),
    
    #[error("Logic engine error: {0}")]
    EngineError(String),
    
    #[error("Invalid syntax: {0}")]
    SyntaxError(String),
    
    #[error("Initialization error: {0}")]
    InitializationError(String),
    
    #[error("Cache error: {0}")]
    CacheError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Generic error: {0}")]
    GenericError(String),
}

/// Result type for symbolic reasoning operations
pub type Result<T> = std::result::Result<T, SymbolicError>;

impl From<anyhow::Error> for SymbolicError {
    fn from(err: anyhow::Error) -> Self {
        SymbolicError::GenericError(err.to_string())
    }
}

impl From<String> for SymbolicError {
    fn from(err: String) -> Self {
        SymbolicError::GenericError(err)
    }
}

impl From<regex::Error> for SymbolicError {
    fn from(err: regex::Error) -> Self {
        SymbolicError::ParseError(format!("Regex error: {}", err))
    }
}