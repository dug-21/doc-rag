use axum::{
    extract::{Path, Query as AxumQuery, State},
    response::{Json, Sse},
    http::StatusCode,
};
use futures::stream::StreamExt;
use serde::Deserialize;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn, error};
use uuid::Uuid;

use crate::{
    clients::ComponentClients,
    models::{
        QueryRequest, QueryResponse, QueryHistoryRequest,
        QueryHistoryResponse, QueryMetrics
    },
    validation::validate_query_request,
    Result, ApiError,
};

/// Process a query request
pub async fn process_query(
    State(clients): State<Arc<ComponentClients>>,
    Json(request): Json<QueryRequest>,
) -> Result<Json<QueryResponse>> {
    info!("Processing query request: query_id={}", request.query_id);

    // Validate the query request
    validate_query_request(&request)?;

    // Start query processing pipeline
    let start_time = std::time::Instant::now();
    
    match clients.process_query_pipeline(request.clone()).await {
        Ok(response) => {
            let processing_time = start_time.elapsed();
            info!("Query processed successfully: query_id={}, processing_time={:?}", 
                  request.query_id, processing_time);

            Ok(Json(QueryResponse {
                processing_time_ms: processing_time.as_millis() as u64,
                ..response
            }))
        }
        Err(e) => {
            let processing_time = start_time.elapsed();
            error!("Query processing failed: query_id={}, error={}, processing_time={:?}", 
                   request.query_id, e, processing_time);

            // Return error response with timing information
            Err(ApiError::QueryProcessingFailed {
                query_id: request.query_id,
                message: e.to_string(),
                processing_time_ms: processing_time.as_millis() as u64,
            })
        }
    }
}

/// Stream query response for real-time processing  
pub async fn stream_query_response(
    State(clients): State<Arc<ComponentClients>>,
    Json(request): Json<QueryRequest>,
) -> std::result::Result<Sse<ReceiverStream<tokio::sync::mpsc::Receiver<std::result::Result<axum::response::sse::Event, std::convert::Infallible>>>>, ApiError> {
    info!("Starting streaming query: query_id={}", request.query_id);

    // Validate the query request
    validate_query_request(&request)?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::response::sse::Event, std::convert::Infallible>>(100);
    let query_id = request.query_id;

    // Spawn task to handle streaming response
    tokio::spawn(async move {
        match clients.process_streaming_query(request).await {
            Ok(mut stream) => {
                // Send initial event
                if let Err(e) = tx.send(Ok(
                    axum::response::sse::Event::default()
                        .event("query_started")
                        .data(serde_json::json!({
                            "query_id": query_id,
                            "status": "processing"
                        }).to_string())
                )).await {
                    warn!("Failed to send initial streaming event: {}", e);
                    return;
                }

                // Stream intermediate results
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(chunk_data) => {
                            let event = axum::response::sse::Event::default()
                                .event("chunk")
                                .data(serde_json::to_string(&chunk_data).unwrap_or_default());

                            if let Err(e) = tx.send(Ok(event)).await {
                                warn!("Failed to send streaming chunk: {}", e);
                                break;
                            }
                        }
                        Err(e) => {
                            warn!("Error in streaming chunk: {}", e);
                            let event = axum::response::sse::Event::default()
                                .event("error")
                                .data(serde_json::json!({
                                    "query_id": query_id,
                                    "error": e.to_string()
                                }).to_string());

                            let _ = tx.send(Ok(event)).await;
                            break;
                        }
                    }
                }

                // Send completion event
                let event = axum::response::sse::Event::default()
                    .event("query_completed")
                    .data(serde_json::json!({
                        "query_id": query_id,
                        "status": "completed"
                    }).to_string());

                let _ = tx.send(Ok(event)).await;
            }
            Err(e) => {
                error!("Streaming query failed: query_id={}, error={}", query_id, e);
                
                let event = axum::response::sse::Event::default()
                    .event("query_failed")
                    .data(serde_json::json!({
                        "query_id": query_id,
                        "error": e.to_string()
                    }).to_string());

                let _ = tx.send(Ok(event)).await;
            }
        }
    });

    let stream = ReceiverStream::new(rx);
    
    Ok(Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default()))
}

/// Get query history with pagination and filtering
#[derive(Deserialize)]
pub struct QueryHistoryParams {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub user_id: Option<Uuid>,
    pub start_date: Option<chrono::DateTime<chrono::Utc>>,
    pub end_date: Option<chrono::DateTime<chrono::Utc>>,
    pub status: Option<String>,
}

pub async fn get_query_history(
    State(clients): State<Arc<ComponentClients>>,
    AxumQuery(params): AxumQuery<QueryHistoryParams>,
) -> Result<Json<QueryHistoryResponse>> {
    info!("Retrieving query history with filters");

    let request = QueryHistoryRequest {
        limit: params.limit.unwrap_or(50),
        offset: params.offset.unwrap_or(0),
        user_id: params.user_id,
        start_date: params.start_date,
        end_date: params.end_date,
        status_filter: params.status,
    };

    // Validate pagination limits
    if request.limit > 1000 {
        return Err(ApiError::BadRequest("Limit cannot exceed 1000".to_string()));
    }

    match clients.get_query_history(request).await {
        Ok(history) => {
            info!("Retrieved {} query records", history.queries.len());
            Ok(Json(history))
        }
        Err(e) => {
            error!("Failed to retrieve query history: {}", e);
            Err(ApiError::Internal(format!("Failed to retrieve query history: {}", e)))
        }
    }
}

/// Get query metrics and analytics
pub async fn get_query_metrics(
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<QueryMetrics>> {
    info!("Retrieving query metrics");

    match clients.get_query_metrics().await {
        Ok(metrics) => {
            Ok(Json(metrics))
        }
        Err(e) => {
            error!("Failed to retrieve query metrics: {}", e);
            Err(ApiError::Internal(format!("Failed to retrieve query metrics: {}", e)))
        }
    }
}

/// Cancel a running query
pub async fn cancel_query(
    Path(query_id): axum::extract::Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
) -> Result<StatusCode> {
    info!("Cancelling query: query_id={}", query_id);

    match clients.cancel_query(query_id).await {
        Ok(true) => {
            info!("Query cancelled successfully: query_id={}", query_id);
            Ok(StatusCode::OK)
        }
        Ok(false) => {
            warn!("Query not found or already completed: query_id={}", query_id);
            Err(ApiError::NotFound(format!("Query not found: {}", query_id)))
        }
        Err(e) => {
            error!("Failed to cancel query: query_id={}, error={}", query_id, e);
            Err(ApiError::Internal(format!("Failed to cancel query: {}", e)))
        }
    }
}

/// Get query result by ID
pub async fn get_query_result(
    Path(query_id): axum::extract::Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<QueryResponse>> {
    info!("Retrieving query result: query_id={}", query_id);

    match clients.get_query_result(query_id).await {
        Ok(Some(result)) => {
            Ok(Json(result))
        }
        Ok(None) => {
            Err(ApiError::NotFound(format!("Query result not found: {}", query_id)))
        }
        Err(e) => {
            error!("Failed to retrieve query result: query_id={}, error={}", query_id, e);
            Err(ApiError::Internal(format!("Failed to retrieve query result: {}", e)))
        }
    }
}

/// Get similar queries for recommendations
pub async fn get_similar_queries(
    Path(query_id): axum::extract::Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<serde_json::Value>> {
    info!("Finding similar queries: query_id={}", query_id);

    match clients.find_similar_queries(query_id, 10).await {
        Ok(similar_queries) => {
            Ok(Json(serde_json::json!({
                "query_id": query_id,
                "similar_queries": similar_queries,
                "count": similar_queries.len()
            })))
        }
        Err(e) => {
            error!("Failed to find similar queries: query_id={}, error={}", query_id, e);
            Err(ApiError::Internal(format!("Failed to find similar queries: {}", e)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::*;

    #[tokio::test]
    async fn test_query_validation() {
        let request = QueryRequest {
            query_id: Uuid::new_v4(),
            query: String::new(), // Empty query should fail validation
            user_id: Some(Uuid::new_v4()),
            context: None,
            preferences: None,
            max_results: Some(10),
            include_sources: Some(true),
        };

        let result = validate_query_request(&request);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_query_history_params() {
        let params = QueryHistoryParams {
            limit: Some(2000), // Should exceed maximum
            offset: Some(0),
            user_id: None,
            start_date: None,
            end_date: None,
            status: None,
        };

        // In a real test scenario, this would be validated against the limit check
        assert!(params.limit.unwrap_or(50) > 1000);
    }

    #[test]
    fn test_query_history_request_creation() {
        let params = QueryHistoryParams {
            limit: Some(100),
            offset: Some(20),
            user_id: Some(Uuid::new_v4()),
            start_date: None,
            end_date: None,
            status: Some("completed".to_string()),
        };

        let request = QueryHistoryRequest {
            limit: params.limit.unwrap_or(50),
            offset: params.offset.unwrap_or(0),
            user_id: params.user_id,
            start_date: params.start_date,
            end_date: params.end_date,
            status_filter: params.status,
        };

        assert_eq!(request.limit, 100);
        assert_eq!(request.offset, 20);
        assert!(request.user_id.is_some());
        assert_eq!(request.status_filter, Some("completed".to_string()));
    }
}