use axum::{
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde_json::json;
use thiserror::Error;
use tracing::error;
use uuid::Uuid;

#[derive(Error, Debug, PartialEq)]
pub enum ApiError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Forbidden: {0}")]
    Forbidden(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Unprocessable entity: {0}")]
    UnprocessableEntity(String),

    #[error("Too many requests: {0}")]
    TooManyRequests(String),

    #[error("Internal server error: {0}")]
    Internal(String),

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Gateway timeout: {0}")]
    GatewayTimeout(String),

    #[error("Query processing failed for query {query_id}: {message} (took {processing_time_ms}ms)")]
    QueryProcessingFailed {
        query_id: Uuid,
        message: String,
        processing_time_ms: u64,
    },

    #[error("Document ingestion failed: {0}")]
    DocumentIngestionFailed(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Authentication error: {0}")]
    AuthenticationError(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("File processing error: {0}")]
    FileProcessingError(String),

    #[error("Component communication error: {component} - {message}")]
    ComponentError { component: String, message: String },

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Request timeout: {0}")]
    Timeout(String),

    #[error("Validation failed for field {field}: {message}")]
    ValidationFailed { field: String, message: String },

    #[error("External service error - {service}: {message}")]
    ExternalServiceError { service: String, message: String },

    #[error("Insufficient storage")]
    InsufficientStorage,

    #[error("Payload too large")]
    PayloadTooLarge,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_code, message) = match &self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "BAD_REQUEST", msg.clone()),
            ApiError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, "UNAUTHORIZED", msg.clone()),
            ApiError::Forbidden(msg) => (StatusCode::FORBIDDEN, "FORBIDDEN", msg.clone()),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "NOT_FOUND", msg.clone()),
            ApiError::Conflict(msg) => (StatusCode::CONFLICT, "CONFLICT", msg.clone()),
            ApiError::UnprocessableEntity(msg) => (StatusCode::UNPROCESSABLE_ENTITY, "UNPROCESSABLE_ENTITY", msg.clone()),
            ApiError::TooManyRequests(msg) => (StatusCode::TOO_MANY_REQUESTS, "TOO_MANY_REQUESTS", msg.clone()),
            ApiError::ServiceUnavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, "SERVICE_UNAVAILABLE", msg.clone()),
            ApiError::GatewayTimeout(msg) => (StatusCode::GATEWAY_TIMEOUT, "GATEWAY_TIMEOUT", msg.clone()),
            ApiError::QueryProcessingFailed { query_id, message, processing_time_ms } => {
                (StatusCode::INTERNAL_SERVER_ERROR, "QUERY_PROCESSING_FAILED", 
                 format!("Query {} failed: {} ({}ms)", query_id, message, processing_time_ms))
            }
            ApiError::DocumentIngestionFailed(msg) => (StatusCode::UNPROCESSABLE_ENTITY, "DOCUMENT_INGESTION_FAILED", msg.clone()),
            ApiError::ValidationError(msg) => (StatusCode::BAD_REQUEST, "VALIDATION_ERROR", msg.clone()),
            ApiError::AuthenticationError(msg) => (StatusCode::UNAUTHORIZED, "AUTHENTICATION_ERROR", msg.clone()),
            ApiError::RateLimitExceeded(msg) => (StatusCode::TOO_MANY_REQUESTS, "RATE_LIMIT_EXCEEDED", msg.clone()),
            ApiError::FileProcessingError(msg) => (StatusCode::UNPROCESSABLE_ENTITY, "FILE_PROCESSING_ERROR", msg.clone()),
            ApiError::ComponentError { component, message } => {
                (StatusCode::BAD_GATEWAY, "COMPONENT_ERROR", format!("{}: {}", component, message))
            }
            ApiError::DatabaseError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "DATABASE_ERROR", msg.clone()),
            ApiError::CacheError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "CACHE_ERROR", msg.clone()),
            ApiError::Timeout(msg) => (StatusCode::REQUEST_TIMEOUT, "TIMEOUT", msg.clone()),
            ApiError::ValidationFailed { field, message } => {
                (StatusCode::BAD_REQUEST, "VALIDATION_FAILED", format!("Field '{}': {}", field, message))
            }
            ApiError::ExternalServiceError { service, message } => {
                (StatusCode::BAD_GATEWAY, "EXTERNAL_SERVICE_ERROR", format!("{}: {}", service, message))
            }
            ApiError::InsufficientStorage => (StatusCode::INSUFFICIENT_STORAGE, "INSUFFICIENT_STORAGE", "Insufficient storage space".to_string()),
            ApiError::PayloadTooLarge => (StatusCode::PAYLOAD_TOO_LARGE, "PAYLOAD_TOO_LARGE", "Request payload too large".to_string()),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_SERVER_ERROR", msg.clone()),
        };

        // Log the error for debugging
        match status {
            StatusCode::INTERNAL_SERVER_ERROR | StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE => {
                error!("API Error ({}): {}", status, message);
            }
            _ => {
                tracing::warn!("API Error ({}): {}", status, message);
            }
        }

        let body = json!({
            "error": {
                "code": error_code,
                "message": message,
                "timestamp": chrono::Utc::now(),
                "status": status.as_u16()
            }
        });

        (status, Json(body)).into_response()
    }
}

// Convert from common error types
impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        ApiError::Internal(error.to_string())
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(error: serde_json::Error) -> Self {
        ApiError::BadRequest(format!("JSON parsing error: {}", error))
    }
}

impl From<validator::ValidationErrors> for ApiError {
    fn from(error: validator::ValidationErrors) -> Self {
        let messages: Vec<String> = error
            .field_errors()
            .into_iter()
            .flat_map(|(field, errors)| {
                errors.iter().map(move |error| {
                    format!("Field '{}': {}", field, error.message.as_ref().unwrap_or(&std::borrow::Cow::Borrowed("validation failed")))
                })
            })
            .collect();

        ApiError::ValidationError(messages.join(", "))
    }
}

impl From<reqwest::Error> for ApiError {
    fn from(error: reqwest::Error) -> Self {
        if error.is_timeout() {
            ApiError::GatewayTimeout("Request timeout".to_string())
        } else if error.is_connect() {
            ApiError::ServiceUnavailable("Service connection failed".to_string())
        } else {
            ApiError::ComponentError {
                component: "http_client".to_string(),
                message: error.to_string(),
            }
        }
    }
}

impl From<tokio::time::error::Elapsed> for ApiError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        ApiError::GatewayTimeout("Operation timeout".to_string())
    }
}

// Helper functions for creating common errors
impl ApiError {
    pub fn invalid_input(message: impl Into<String>) -> Self {
        ApiError::BadRequest(message.into())
    }

    pub fn not_authenticated() -> Self {
        ApiError::Unauthorized("Authentication required".to_string())
    }

    pub fn insufficient_permissions() -> Self {
        ApiError::Forbidden("Insufficient permissions".to_string())
    }

    pub fn resource_not_found(resource: impl Into<String>) -> Self {
        ApiError::NotFound(format!("{} not found", resource.into()))
    }

    pub fn service_error(service: impl Into<String>, message: impl Into<String>) -> Self {
        ApiError::ComponentError {
            component: service.into(),
            message: message.into(),
        }
    }

    pub fn validation_failed(field: impl Into<String>, message: impl Into<String>) -> Self {
        ApiError::ValidationError(format!("Field '{}': {}", field.into(), message.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(
            ApiError::BadRequest("test".to_string()).into_response().status(),
            StatusCode::BAD_REQUEST
        );

        assert_eq!(
            ApiError::Unauthorized("test".to_string()).into_response().status(),
            StatusCode::UNAUTHORIZED
        );

        assert_eq!(
            ApiError::NotFound("test".to_string()).into_response().status(),
            StatusCode::NOT_FOUND
        );

        assert_eq!(
            ApiError::Internal("test".to_string()).into_response().status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn test_error_conversion() {
        let anyhow_error = anyhow::anyhow!("test error");
        let api_error: ApiError = anyhow_error.into();
        
        match api_error {
            ApiError::Internal(msg) => assert!(msg.contains("test error")),
            _ => panic!("Expected Internal error"),
        }
    }

    #[test]
    fn test_helper_functions() {
        let error = ApiError::invalid_input("test message");
        match error {
            ApiError::BadRequest(msg) => assert_eq!(msg, "test message"),
            _ => panic!("Expected BadRequest error"),
        }

        let error = ApiError::not_authenticated();
        match error {
            ApiError::Unauthorized(msg) => assert!(msg.contains("Authentication required")),
            _ => panic!("Expected Unauthorized error"),
        }

        let error = ApiError::resource_not_found("user");
        match error {
            ApiError::NotFound(msg) => assert_eq!(msg, "user not found"),
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn test_component_error() {
        let error = ApiError::service_error("chunker", "connection failed");
        match error {
            ApiError::ComponentError { component, message } => {
                assert_eq!(component, "chunker");
                assert_eq!(message, "connection failed");
            }
            _ => panic!("Expected ComponentError"),
        }
    }
}