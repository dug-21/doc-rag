use axum::{
    extract::{Multipart, Path, State},
    response::Json,
    http::StatusCode,
};
use std::sync::Arc;
use tracing::{info, warn, error};
use uuid::Uuid;

use crate::{
    clients::ComponentClients,
    models::{
        IngestRequest, IngestResponse, BatchIngestRequest, BatchIngestResponse,
        DocumentStatus, IngestTask, TaskStatus
    },
    validation::{validate_document_content, validate_file_type},
    Result, ApiError,
};

/// Ingest a single document
pub async fn ingest_document(
    State(clients): State<Arc<ComponentClients>>,
    Json(request): Json<IngestRequest>,
) -> Result<Json<IngestResponse>> {
    info!("Processing document ingestion request");

    // Validate the request
    validate_document_content(&request.content)?;
    if let Some(ref mime_type) = request.content_type {
        validate_file_type(mime_type)?;
    }

    // Generate task ID for tracking
    let task_id = Uuid::new_v4();
    
    // Start the ingestion pipeline
    let pipeline_result = clients.process_document_ingestion(
        task_id,
        request.content.clone(),
        request.metadata.clone(),
        request.chunking_strategy.clone(),
    ).await;

    match pipeline_result {
        Ok(document_id) => {
            info!("Document ingestion successful: task_id={}, document_id={}", task_id, document_id);
            
            Ok(Json(IngestResponse {
                task_id,
                document_id,
                status: TaskStatus::Completed,
                message: "Document ingested successfully".to_string(),
                chunks_created: Some(1), // This would come from the actual pipeline
                processing_time_ms: 0, // This would be calculated
            }))
        }
        Err(e) => {
            error!("Document ingestion failed: task_id={}, error={}", task_id, e);
            
            // Return partial success with error details
            Ok(Json(IngestResponse {
                task_id,
                document_id: Uuid::nil(),
                status: TaskStatus::Failed,
                message: format!("Ingestion failed: {}", e),
                chunks_created: None,
                processing_time_ms: 0,
            }))
        }
    }
}

/// Batch ingest multiple documents
pub async fn batch_ingest_documents(
    State(clients): State<Arc<ComponentClients>>,
    Json(request): Json<BatchIngestRequest>,
) -> Result<Json<BatchIngestResponse>> {
    info!("Processing batch document ingestion: {} documents", request.documents.len());

    if request.documents.is_empty() {
        return Err(ApiError::BadRequest("No documents provided".to_string()));
    }

    if request.documents.len() > 100 {
        return Err(ApiError::BadRequest("Too many documents (max 100)".to_string()));
    }

    let batch_id = Uuid::new_v4();
    let mut results = Vec::new();
    let mut successful = 0;
    let mut failed = 0;

    // Process documents in parallel (with concurrency limit)
    use futures::stream::{self, StreamExt};
    
    let results_stream = stream::iter(request.documents.into_iter().enumerate())
        .map(|(index, doc_request)| {
            let clients = clients.clone();
            let batch_id = batch_id;
            
            async move {
                let task_id = Uuid::new_v4();
                
                // Validate each document
                if let Err(e) = validate_document_content(&doc_request.content) {
                    return IngestResponse {
                        task_id,
                        document_id: Uuid::nil(),
                        status: TaskStatus::Failed,
                        message: format!("Validation failed: {}", e),
                        chunks_created: None,
                        processing_time_ms: 0,
                    };
                }

                // Process the document
                match clients.process_document_ingestion(
                    task_id,
                    doc_request.content,
                    doc_request.metadata,
                    doc_request.chunking_strategy,
                ).await {
                    Ok(document_id) => {
                        info!("Batch document {} ingested: batch_id={}, task_id={}, document_id={}", 
                              index, batch_id, task_id, document_id);
                        
                        IngestResponse {
                            task_id,
                            document_id,
                            status: TaskStatus::Completed,
                            message: "Document ingested successfully".to_string(),
                            chunks_created: Some(1),
                            processing_time_ms: 0,
                        }
                    }
                    Err(e) => {
                        warn!("Batch document {} failed: batch_id={}, task_id={}, error={}", 
                              index, batch_id, task_id, e);
                        
                        IngestResponse {
                            task_id,
                            document_id: Uuid::nil(),
                            status: TaskStatus::Failed,
                            message: format!("Ingestion failed: {}", e),
                            chunks_created: None,
                            processing_time_ms: 0,
                        }
                    }
                }
            }
        })
        .buffer_unordered(10); // Process up to 10 documents concurrently

    let results: Vec<IngestResponse> = results_stream.collect().await;

    // Count successes and failures
    for result in &results {
        match result.status {
            TaskStatus::Completed => successful += 1,
            TaskStatus::Failed => failed += 1,
            _ => {}
        }
    }

    info!("Batch ingestion completed: batch_id={}, successful={}, failed={}", 
          batch_id, successful, failed);

    Ok(Json(BatchIngestResponse {
        batch_id,
        total_documents: results.len(),
        successful_ingestions: successful,
        failed_ingestions: failed,
        results,
        processing_time_ms: 0, // This would be calculated
    }))
}

/// Get document ingestion status
pub async fn get_document_status(
    Path(task_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<DocumentStatus>> {
    info!("Retrieving document status for task_id={}", task_id);

    match clients.get_document_status(task_id).await {
        Ok(status) => {
            Ok(Json(status))
        }
        Err(e) => {
            warn!("Failed to retrieve document status: task_id={}, error={}", task_id, e);
            Err(ApiError::NotFound(format!("Task not found: {}", task_id)))
        }
    }
}

/// Upload and process a file
pub async fn upload_and_ingest_file(
    State(clients): State<Arc<ComponentClients>>,
    mut multipart: Multipart,
) -> Result<Json<IngestResponse>> {
    info!("Processing file upload and ingestion");

    let mut file_content = Vec::new();
    let mut filename = None;
    let mut content_type = None;

    // Process multipart form data
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        ApiError::BadRequest(format!("Invalid multipart data: {}", e))
    })? {
        let name = field.name().unwrap_or_default().to_string();
        
        match name.as_str() {
            "file" => {
                filename = field.file_name().map(|s| s.to_string());
                content_type = field.content_type().map(|s| s.to_string());
                file_content = field.bytes().await.map_err(|e| {
                    ApiError::BadRequest(format!("Failed to read file: {}", e))
                })?.to_vec();
            }
            _ => {
                // Skip unknown fields
                continue;
            }
        }
    }

    if file_content.is_empty() {
        return Err(ApiError::BadRequest("No file provided".to_string()));
    }

    // Validate file type
    if let Some(ref ct) = content_type {
        validate_file_type(ct)?;
    }

    // Convert file content to text based on content type
    let text_content = clients.extract_text_from_file(file_content, content_type.as_deref()).await
        .map_err(|e| ApiError::BadRequest(format!("Failed to extract text: {}", e)))?;

    // Create ingestion request
    let request = IngestRequest {
        content: text_content,
        content_type,
        metadata: Some(serde_json::json!({
            "filename": filename,
            "uploaded_at": chrono::Utc::now(),
            "source": "file_upload"
        })),
        chunking_strategy: None,
    };

    // Process using the regular ingestion handler
    ingest_document(State(clients), Json(request)).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::*;
    use mockall::predicate::*;

    #[tokio::test]
    async fn test_ingest_document_validation() {
        // Test empty content
        let request = IngestRequest {
            content: String::new(),
            content_type: None,
            metadata: None,
            chunking_strategy: None,
        };

        let result = validate_document_content(&request.content);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_batch_ingest_empty() {
        let request = BatchIngestRequest {
            documents: vec![],
        };

        // Create mock clients (in real implementation)
        // let clients = Arc::new(MockComponentClients::new());
        
        // This would test the empty batch handling
        // let result = batch_ingest_documents(State(clients), Json(request)).await;
        // assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_batch_ingest_too_many() {
        let mut documents = Vec::new();
        for _ in 0..101 {
            documents.push(IngestRequest {
                content: "Test content".to_string(),
                content_type: None,
                metadata: None,
                chunking_strategy: None,
            });
        }

        let request = BatchIngestRequest { documents };

        // This should fail due to too many documents
        // In a real test with proper mocking, we would verify this
    }
}