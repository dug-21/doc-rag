//! Error types for the embedding generator

use thiserror::Error;

/// Errors that can occur during embedding generation
#[derive(Error, Debug)]
pub enum EmbedderError {
    /// Model file not found
    #[error("Model not found")]
    ModelNotFound,
    
    /// Model not loaded in memory
    #[error("Model not loaded: {model_type}")]
    ModelNotLoaded { model_type: String },
    
    /// Invalid model format
    #[error("Invalid model format: {message}")]
    InvalidModelFormat { message: String },
    
    /// Tokenization error
    #[error("Tokenization failed: {message}")]
    TokenizationError { message: String },
    
    /// Inference error
    #[error("Inference failed: {message}")]
    InferenceError { message: String },
    
    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    /// Invalid embedding values
    #[error("Invalid embedding: {message}")]
    InvalidEmbedding { message: String },
    
    /// Batch processing error
    #[error("Batch processing failed: {message}")]
    BatchProcessingError { message: String },
    
    /// Model download error
    #[error("Failed to download model: {message}")]
    DownloadError { message: String },
    
    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
    
    /// I/O error
    #[error("I/O error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },
    
    /// Serialization error
    #[error("Serialization error: {source}")]
    SerializationError {
        #[from]
        source: serde_json::Error,
    },
    
    /// ONNX Runtime error
    #[error("ONNX Runtime error: {message}")]
    OnnxError { message: String },
    
    /// Candle error
    #[error("Candle error: {message}")]
    CandleError { message: String },
    
    /// Threading error
    #[error("Threading error: {message}")]
    ThreadingError { message: String },
    
    /// Memory allocation error
    #[error("Memory allocation error: {message}")]
    MemoryError { message: String },
    
    /// Timeout error
    #[error("Operation timed out after {timeout_ms}ms")]
    TimeoutError { timeout_ms: u64 },
}

impl From<ort::OrtError> for EmbedderError {
    fn from(err: ort::OrtError) -> Self {
        EmbedderError::OnnxError {
            message: err.to_string(),
        }
    }
}

impl From<candle_core::Error> for EmbedderError {
    fn from(err: candle_core::Error) -> Self {
        EmbedderError::CandleError {
            message: err.to_string(),
        }
    }
}

impl From<ndarray::ShapeError> for EmbedderError {
    fn from(err: ndarray::ShapeError) -> Self {
        EmbedderError::BatchProcessingError {
            message: format!("Shape error: {}", err),
        }
    }
}

impl From<reqwest::Error> for EmbedderError {
    fn from(err: reqwest::Error) -> Self {
        EmbedderError::DownloadError {
            message: err.to_string(),
        }
    }
}

/// Result type for embedding operations
pub type EmbedderResult<T> = Result<T, EmbedderError>;

/// Helper function to create a configuration error
pub fn config_error(message: impl Into<String>) -> EmbedderError {
    EmbedderError::ConfigError {
        message: message.into(),
    }
}

/// Helper function to create an inference error
pub fn inference_error(message: impl Into<String>) -> EmbedderError {
    EmbedderError::InferenceError {
        message: message.into(),
    }
}

/// Helper function to create a tokenization error
pub fn tokenization_error(message: impl Into<String>) -> EmbedderError {
    EmbedderError::TokenizationError {
        message: message.into(),
    }
}

/// Helper function to create a batch processing error
pub fn batch_error(message: impl Into<String>) -> EmbedderError {
    EmbedderError::BatchProcessingError {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let error = EmbedderError::DimensionMismatch {
            expected: 384,
            actual: 768,
        };
        let display = format!("{}", error);
        assert!(display.contains("384"));
        assert!(display.contains("768"));
    }
    
    #[test]
    fn test_error_helpers() {
        let error = config_error("Invalid batch size");
        match error {
            EmbedderError::ConfigError { message } => {
                assert_eq!(message, "Invalid batch size");
            }
            _ => panic!("Expected ConfigError"),
        }
    }
    
    #[test]
    fn test_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let embedder_error = EmbedderError::from(io_error);
        
        match embedder_error {
            EmbedderError::IoError { .. } => {}
            _ => panic!("Expected IoError"),
        }
    }
}