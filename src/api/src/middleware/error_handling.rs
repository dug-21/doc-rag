use axum::{
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::convert::Infallible;
use tower::{Layer, Service};
use tracing::{error, warn, debug};
use uuid::Uuid;

use crate::ApiError;

/// Error handling middleware layer
#[derive(Clone)]
pub struct ErrorHandlingLayer;

impl ErrorHandlingLayer {
    pub fn new() -> Self {
        Self
    }
}

impl<S> Layer<S> for ErrorHandlingLayer {
    type Service = ErrorHandlingMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ErrorHandlingMiddleware { inner }
    }
}

#[derive(Clone)]
pub struct ErrorHandlingMiddleware<S> {
    inner: S,
}

impl<S> Service<Request> for ErrorHandlingMiddleware<S>
where
    S: Service<Request, Response = Response, Error = Infallible> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = Infallible;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut std::task::Context<'_>) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        let future = self.inner.call(request);

        Box::pin(async move {
            let response = future.await?;
            
            // Check if response indicates an error that needs special handling
            if response.status().is_server_error() {
                let (parts, _body) = response.into_parts();
                
                // Try to extract error information from the response
                let error_response = create_error_response(
                    parts.status,
                    "Internal server error occurred",
                    None,
                    get_request_id_from_headers(&parts.headers),
                );
                
                Ok((StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response())
            } else {
                Ok(response)
            }
        })
    }
}

/// Axum error handler for unhandled errors
pub async fn handle_error(error: BoxError) -> impl IntoResponse {
    error!("Unhandled error: {:?}", error);
    
    let error_response = create_error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "An unexpected error occurred",
        Some(error.to_string()),
        None,
    );
    
    (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response))
}

/// Create a standardized error response
pub fn create_error_response(
    status: StatusCode,
    message: &str,
    details: Option<String>,
    request_id: Option<String>,
) -> serde_json::Value {
    let mut response = json!({
        "error": {
            "code": status.as_u16(),
            "status": status.canonical_reason().unwrap_or("Unknown"),
            "message": message,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        }
    });
    
    if let Some(details) = details {
        response["error"]["details"] = json!(details);
    }
    
    if let Some(request_id) = request_id {
        response["error"]["request_id"] = json!(request_id);
    }
    
    response
}

/// Extract request ID from headers
fn get_request_id_from_headers(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

// IntoResponse implementation is provided in errors.rs to avoid conflicts
/* Removed duplicate impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message, details) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "Bad Request", Some(msg)),
            ApiError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, "Unauthorized", Some(msg)),
            ApiError::Forbidden(msg) => (StatusCode::FORBIDDEN, "Forbidden", Some(msg)),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, "Not Found", Some(msg)),
            ApiError::Conflict(msg) => (StatusCode::CONFLICT, "Conflict", Some(msg)),
            ApiError::UnprocessableEntity(msg) => (StatusCode::UNPROCESSABLE_ENTITY, "Unprocessable Entity", Some(msg)),
            ApiError::TooManyRequests(msg) => (StatusCode::TOO_MANY_REQUESTS, "Too Many Requests", Some(msg)),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error", Some(msg)),
            ApiError::ServiceUnavailable(msg) => (StatusCode::SERVICE_UNAVAILABLE, "Service Unavailable", Some(msg)),
            ApiError::Timeout(msg) => (StatusCode::REQUEST_TIMEOUT, "Request Timeout", Some(msg)),
            ApiError::ValidationFailed { field, message } => (
                StatusCode::BAD_REQUEST, 
                "Validation Failed", 
                Some(format!("Field '{}': {}", field, message))
            ),
            ApiError::DatabaseError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "Database Error", Some(msg)),
            ApiError::ExternalServiceError { service, message } => (
                StatusCode::BAD_GATEWAY,
                "External Service Error",
                Some(format!("Service '{}': {}", service, message))
            ),
            ApiError::QueryProcessingFailed { query_id, message, processing_time_ms } => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Query Processing Failed",
                Some(format!("Query ID: {}, Processing time: {}ms, Error: {}", query_id, processing_time_ms, message))
            ),
            ApiError::InsufficientStorage => (StatusCode::INSUFFICIENT_STORAGE, "Insufficient Storage", None),
            ApiError::PayloadTooLarge => (StatusCode::PAYLOAD_TOO_LARGE, "Payload Too Large", None),
        };
        
        // Log the error based on severity
        match status {
            StatusCode::INTERNAL_SERVER_ERROR | StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE => {
                error!("API Error: {} - {}", status, message);
                if let Some(ref details) = details {
                    error!("Error details: {}", details);
                }
            }
            StatusCode::BAD_REQUEST | StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN | StatusCode::NOT_FOUND => {
                warn!("API Error: {} - {}", status, message);
                if let Some(ref details) = details {
                    debug!("Error details: {}", details);
                }
            }
            _ => {
                debug!("API Error: {} - {}", status, message);
            }
        }
        
        let error_response = create_error_response(status, message, details, None);
        
        (status, Json(error_response)).into_response()
    }
} */

/// Type alias for boxed errors
type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Handle timeout errors specifically
pub fn handle_timeout_error(operation: &str) -> ApiError {
    warn!("Operation timed out: {}", operation);
    ApiError::Timeout(format!("Operation '{}' timed out", operation))
}

/// Handle database connection errors
pub fn handle_database_error(error: impl std::error::Error) -> ApiError {
    error!("Database error: {}", error);
    ApiError::DatabaseError("Database operation failed".to_string())
}

/// Handle external service errors with retry information
pub fn handle_external_service_error(
    service: &str, 
    error: impl std::error::Error,
    retry_count: u32,
) -> ApiError {
    warn!("External service error: service={}, retry={}, error={}", service, retry_count, error);
    
    if retry_count >= 3 {
        ApiError::ComponentError {
            component: service.to_string(),
            message: format!("Service unavailable after {} retries", retry_count),
        }
    } else {
        ApiError::ComponentError {
            component: service.to_string(),
            message: error.to_string(),
        }
    }
}

/// Handle validation errors with field context
pub fn handle_validation_error(field: &str, message: &str) -> ApiError {
    debug!("Validation error: field={}, message={}", field, message);
    ApiError::ValidationError(format!("Field '{}': {}", field, message))
}

/// Handle rate limiting errors
pub fn handle_rate_limit_error(client_id: Option<&str>, retry_after: Option<u64>) -> ApiError {
    let message = match (client_id, retry_after) {
        (Some(id), Some(seconds)) => {
            warn!("Rate limit exceeded for client: {}, retry after: {}s", id, seconds);
            format!("Rate limit exceeded. Retry after {} seconds", seconds)
        }
        (Some(id), None) => {
            warn!("Rate limit exceeded for client: {}", id);
            "Rate limit exceeded".to_string()
        }
        (None, Some(seconds)) => {
            warn!("Rate limit exceeded, retry after: {}s", seconds);
            format!("Rate limit exceeded. Retry after {} seconds", seconds)
        }
        (None, None) => {
            warn!("Rate limit exceeded");
            "Rate limit exceeded".to_string()
        }
    };
    
    ApiError::TooManyRequests(message)
}

/// Handle file upload errors
pub fn handle_file_upload_error(error: impl std::error::Error) -> ApiError {
    warn!("File upload error: {}", error);
    
    let error_str = error.to_string().to_lowercase();
    
    if error_str.contains("size") || error_str.contains("large") {
        ApiError::BadRequest("Request payload too large".to_string())
    } else if error_str.contains("format") || error_str.contains("type") {
        ApiError::BadRequest("Unsupported file format".to_string())
    } else if error_str.contains("storage") || error_str.contains("space") {
        ApiError::Internal("Insufficient storage space".to_string())
    } else {
        ApiError::Internal("File upload failed".to_string())
    }
}

/// Create a structured error response with correlation ID
pub fn create_structured_error_response(
    status: StatusCode,
    error_code: &str,
    message: &str,
    details: Option<serde_json::Value>,
    correlation_id: Option<Uuid>,
) -> serde_json::Value {
    let mut response = json!({
        "error": {
            "code": error_code,
            "http_status": status.as_u16(),
            "message": message,
            "timestamp": chrono::Utc::now().to_rfc3339(),
        }
    });
    
    if let Some(details) = details {
        response["error"]["details"] = details;
    }
    
    if let Some(correlation_id) = correlation_id {
        response["error"]["correlation_id"] = json!(correlation_id);
    }
    
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn test_error_response_creation() {
        let response = create_error_response(
            StatusCode::BAD_REQUEST,
            "Test error",
            Some("Additional details".to_string()),
            Some("req-123".to_string()),
        );
        
        assert_eq!(response["error"]["code"], 400);
        assert_eq!(response["error"]["message"], "Test error");
        assert_eq!(response["error"]["details"], "Additional details");
        assert_eq!(response["error"]["request_id"], "req-123");
    }

    #[test]
    fn test_api_error_conversion() {
        let error = ApiError::BadRequest("Invalid input".to_string());
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_structured_error_response() {
        let correlation_id = Uuid::new_v4();
        let details = json!({
            "field": "email",
            "constraint": "format"
        });
        
        let response = create_structured_error_response(
            StatusCode::BAD_REQUEST,
            "VALIDATION_ERROR",
            "Invalid email format",
            Some(details.clone()),
            Some(correlation_id),
        );
        
        assert_eq!(response["error"]["code"], "VALIDATION_ERROR");
        assert_eq!(response["error"]["http_status"], 400);
        assert_eq!(response["error"]["details"], details);
        assert_eq!(response["error"]["correlation_id"], correlation_id.to_string());
    }

    #[test]
    fn test_timeout_error_handling() {
        let error = handle_timeout_error("database_query");
        match error {
            ApiError::Timeout(msg) => assert!(msg.contains("database_query")),
            _ => panic!("Expected Timeout error"),
        }
    }

    #[test]
    fn test_rate_limit_error_handling() {
        let error = handle_rate_limit_error(Some("client-123"), Some(60));
        match error {
            ApiError::TooManyRequests(msg) => {
                assert!(msg.contains("60"));
                assert!(msg.contains("seconds"));
            }
            _ => panic!("Expected TooManyRequests error"),
        }
    }
}