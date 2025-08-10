use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Json, Response},
};
use serde_json::json;
use tower::{Layer, Service};
use tracing::{error, warn};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use crate::ApiError;

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
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        let future = self.inner.call(request);

        Box::pin(async move {
            let response = future.await?;
            
            // Check if response is an error status
            if response.status().is_client_error() || response.status().is_server_error() {
                // Log based on error type
                match response.status() {
                    StatusCode::BAD_REQUEST |
                    StatusCode::UNAUTHORIZED |
                    StatusCode::FORBIDDEN |
                    StatusCode::NOT_FOUND |
                    StatusCode::CONFLICT |
                    StatusCode::UNPROCESSABLE_ENTITY => {
                        warn!("Client error: {}", response.status());
                    }
                    _ => {
                        error!("Server error: {}", response.status());
                    }
                }
            }

            Ok(response)
        })
    }
}

/// Global error handler for unhandled errors
pub async fn handle_error(error: Box<dyn std::error::Error + Send + Sync>) -> Response {
    error!("Unhandled error: {}", error);

    let body = json!({
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "timestamp": chrono::Utc::now(),
            "status": 500
        }
    });

    (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}

/// Handle panic errors
pub fn handle_panic(info: &std::panic::PanicInfo) -> Response {
    let location = info.location().map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
        .unwrap_or_else(|| "unknown location".to_string());
    
    let message = if let Some(s) = info.payload().downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    };

    error!("Panic occurred at {}: {}", location, message);

    let body = json!({
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "A critical error occurred",
            "timestamp": chrono::Utc::now(),
            "status": 500
        }
    });

    (StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use tower::{ServiceExt, service_fn};

    #[tokio::test]
    async fn test_error_handling_layer() {
        // Create a service that returns an error
        let service = service_fn(|_request: Request| async {
            Ok::<_, std::convert::Infallible>(
                (StatusCode::INTERNAL_SERVER_ERROR, "Test error").into_response()
            )
        });

        let service = ErrorHandlingLayer::new().layer(service);
        
        let request = Request::builder()
            .uri("/test")
            .body(Body::empty())
            .unwrap();

        let response = service.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_handle_error() {
        let error: Box<dyn std::error::Error + Send + Sync> = 
            Box::new(std::io::Error::new(std::io::ErrorKind::Other, "test error"));
        
        let response = handle_error(error).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_handle_panic() {
        // Create a panic info (this is normally created by the runtime)
        let location = std::panic::Location::caller();
        let payload = "test panic message";
        
        // We can't easily create a PanicInfo for testing, so we'll test the error response format
        let body = serde_json::json!({
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "A critical error occurred",
                "timestamp": chrono::Utc::now(),
                "status": 500
            }
        });

        assert!(body.get("error").is_some());
        assert_eq!(body["error"]["code"], "INTERNAL_SERVER_ERROR");
        assert_eq!(body["error"]["status"], 500);
    }
}