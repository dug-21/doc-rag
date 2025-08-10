use axum::{
    extract::{Path, State},
    response::Json,
};
use std::sync::Arc;
use tracing::{info, warn, error};
use uuid::Uuid;

use crate::{
    clients::ComponentClients,
    models::{
        IngestRequest, IngestResponse, BatchIngestRequest, BatchIngestResponse,
        DocumentStatus, TaskStatus
    },
    validation::{validate_ingest_request, validate_batch_ingest_request, validate_chunking_strategy},
    Result, ApiError,
};

/// Process a single document ingestion
pub async fn ingest_document(
    State(clients): State<Arc<ComponentClients>>,
    Json(request): Json<IngestRequest>,
) -> Result<Json<IngestResponse>> {
    info!("Processing document ingestion");
    
    // Validate the ingestion request
    validate_ingest_request(&request)?;
    
    // Validate chunking strategy if provided
    if let Some(ref strategy) = request.chunking_strategy {
        validate_chunking_strategy(strategy)?;
    }
    
    let task_id = Uuid::new_v4();
    let start_time = std::time::Instant::now();
    
    match clients.process_document_ingestion(
        task_id,
        request.content,
        request.metadata,
        request.chunking_strategy,
    ).await {
        Ok(document_id) => {
            let processing_time = start_time.elapsed();
            
            info!("Document ingestion completed: task_id={}, document_id={}, processing_time={:?}", 
                  task_id, document_id, processing_time);
            
            Ok(Json(IngestResponse {
                task_id,
                document_id,
                status: TaskStatus::Completed,
                message: "Document processed successfully".to_string(),
                chunks_created: Some(0), // Would be populated from actual processing
                processing_time_ms: processing_time.as_millis() as u64,
            }))
        }
        Err(e) => {
            let processing_time = start_time.elapsed();
            
            error!("Document ingestion failed: task_id={}, error={}, processing_time={:?}", 
                   task_id, e, processing_time);
            
            Err(ApiError::Internal(format!("Document ingestion failed: {}", e)))
        }
    }
}

/// Process batch document ingestion
pub async fn batch_ingest(
    State(clients): State<Arc<ComponentClients>>,
    Json(request): Json<BatchIngestRequest>,
) -> Result<Json<BatchIngestResponse>> {
    info!("Processing batch document ingestion: {} documents", request.documents.len());
    
    // Validate the batch request
    validate_batch_ingest_request(&request)?;
    
    let batch_id = Uuid::new_v4();
    let start_time = std::time::Instant::now();
    let total_documents = request.documents.len();
    
    let mut successful_ingestions = 0;
    let mut failed_ingestions = 0;
    let mut results = Vec::with_capacity(total_documents);
    
    // Process each document in the batch
    for (index, doc_request) in request.documents.iter().enumerate() {
        let task_id = Uuid::new_v4();
        let doc_start_time = std::time::Instant::now();
        
        info!("Processing batch document {}/{}: task_id={}", index + 1, total_documents, task_id);
        
        match clients.process_document_ingestion(
            task_id,
            doc_request.content.clone(),
            doc_request.metadata.clone(),
            doc_request.chunking_strategy.clone(),
        ).await {
            Ok(document_id) => {
                successful_ingestions += 1;
                let processing_time = doc_start_time.elapsed();
                
                results.push(IngestResponse {
                    task_id,
                    document_id,
                    status: TaskStatus::Completed,
                    message: "Document processed successfully".to_string(),
                    chunks_created: Some(0), // Would be populated from actual processing
                    processing_time_ms: processing_time.as_millis() as u64,
                });
            }
            Err(e) => {
                failed_ingestions += 1;
                let processing_time = doc_start_time.elapsed();
                
                warn!("Batch document processing failed: task_id={}, error={}", task_id, e);
                
                results.push(IngestResponse {
                    task_id,
                    document_id: Uuid::new_v4(), // Generate placeholder ID for failed docs
                    status: TaskStatus::Failed,
                    message: format!("Processing failed: {}", e),
                    chunks_created: None,
                    processing_time_ms: processing_time.as_millis() as u64,
                });
            }
        }
    }
    
    let total_processing_time = start_time.elapsed();
    
    info!("Batch ingestion completed: batch_id={}, successful={}, failed={}, total_time={:?}",
          batch_id, successful_ingestions, failed_ingestions, total_processing_time);
    
    Ok(Json(BatchIngestResponse {
        batch_id,
        total_documents,
        successful_ingestions,
        failed_ingestions,
        results,
        processing_time_ms: total_processing_time.as_millis() as u64,
    }))
}

/// Get document processing status
pub async fn get_document_status(
    Path(document_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<DocumentStatus>> {
    info!("Retrieving document status: document_id={}", document_id);
    
    match clients.get_document_status(document_id).await {
        Ok(status) => {
            Ok(Json(status))
        }
        Err(e) => {
            error!("Failed to retrieve document status: document_id={}, error={}", document_id, e);
            Err(ApiError::Internal(format!("Failed to retrieve document status: {}", e)))
        }
    }
}

/// Get document processing history
pub async fn get_processing_history(
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<serde_json::Value>> {
    info!("Retrieving document processing history");
    
    // In a real implementation, this would query the database for processing history
    // For now, return a placeholder response
    let history = serde_json::json!({
        "total_documents": 0,
        "successful_processing": 0,
        "failed_processing": 0,
        "average_processing_time_ms": 0,
        "recent_documents": []
    });
    
    Ok(Json(history))
}

/// Cancel document processing
pub async fn cancel_document_processing(
    Path(task_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<serde_json::Value>> {
    info!("Cancelling document processing: task_id={}", task_id);
    
    // In a real implementation, this would:
    // 1. Check if the task is still processing
    // 2. Send cancellation signal to processing pipeline
    // 3. Update task status to cancelled
    // 4. Clean up any partial processing results
    
    let response = serde_json::json!({
        "task_id": task_id,
        "status": "cancelled",
        "message": "Document processing cancelled successfully"
    });
    
    Ok(Json(response))
}

/// Retry failed document processing
pub async fn retry_document_processing(
    Path(task_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<IngestResponse>> {
    info!("Retrying document processing: task_id={}", task_id);
    
    // In a real implementation, this would:
    // 1. Retrieve the original document and parameters
    // 2. Reset the task status to pending
    // 3. Re-initiate the processing pipeline
    // 4. Return new processing status
    
    let start_time = std::time::Instant::now();
    
    // For demo purposes, return a success response
    let response = IngestResponse {
        task_id,
        document_id: Uuid::new_v4(),
        status: TaskStatus::Processing,
        message: "Document processing retried successfully".to_string(),
        chunks_created: None,
        processing_time_ms: start_time.elapsed().as_millis() as u64,
    };
    
    Ok(Json(response))
}

/// Get processing statistics
pub async fn get_processing_stats(
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<serde_json::Value>> {
    info!("Retrieving processing statistics");
    
    // In a real implementation, this would aggregate statistics from the database
    let stats = serde_json::json!({
        "total_documents_processed": 0,
        "documents_processed_today": 0,
        "average_processing_time_ms": 0,
        "success_rate_percent": 100.0,
        "active_processing_tasks": 0,
        "queue_size": 0,
        "processing_rate_per_hour": 0,
        "storage_usage_mb": 0,
        "by_content_type": {
            "application/pdf": 0,
            "text/plain": 0,
            "application/json": 0
        }
    });
    
    Ok(Json(stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiConfig;
    use crate::models::ChunkingStrategy;

    #[tokio::test]
    async fn test_ingest_request_validation() {
        let valid_request = IngestRequest {
            content: "Test content".to_string(),
            content_type: Some("text/plain".to_string()),
            metadata: None,
            chunking_strategy: None,
        };
        
        assert!(validate_ingest_request(&valid_request).is_ok());
        
        let invalid_request = IngestRequest {
            content: "".to_string(), // Empty content should fail
            content_type: None,
            metadata: None,
            chunking_strategy: None,
        };
        
        assert!(validate_ingest_request(&invalid_request).is_err());
    }

    #[tokio::test]
    async fn test_batch_ingest_validation() {
        let valid_batch = BatchIngestRequest {
            documents: vec![
                IngestRequest {
                    content: "Test content 1".to_string(),
                    content_type: Some("text/plain".to_string()),
                    metadata: None,
                    chunking_strategy: None,
                },
                IngestRequest {
                    content: "Test content 2".to_string(),
                    content_type: Some("text/plain".to_string()),
                    metadata: None,
                    chunking_strategy: None,
                },
            ],
        };
        
        assert!(validate_batch_ingest_request(&valid_batch).is_ok());
        
        let empty_batch = BatchIngestRequest {
            documents: vec![],
        };
        
        assert!(validate_batch_ingest_request(&empty_batch).is_err());
    }

    #[test]
    fn test_chunking_strategy_validation() {
        let valid_strategy = ChunkingStrategy {
            strategy_type: crate::models::ChunkingType::Adaptive,
            max_chunk_size: Some(1024),
            overlap_size: Some(128),
            preserve_structure: Some(true),
        };
        
        assert!(validate_chunking_strategy(&valid_strategy).is_ok());
        
        let invalid_strategy = ChunkingStrategy {
            strategy_type: crate::models::ChunkingType::Fixed,
            max_chunk_size: Some(100),
            overlap_size: Some(200), // Overlap larger than chunk size
            preserve_structure: Some(false),
        };
        
        assert!(validate_chunking_strategy(&invalid_strategy).is_err());
    }

    #[test]
    fn test_document_status_creation() {
        let status = DocumentStatus {
            task_id: Uuid::new_v4(),
            document_id: Some(Uuid::new_v4()),
            status: TaskStatus::Processing,
            progress_percent: 50,
            current_stage: crate::models::ProcessingStage::Embedding,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            error_message: None,
            chunks_processed: Some(10),
            total_chunks: Some(20),
        };
        
        assert_eq!(status.progress_percent, 50);
        assert!(status.status.is_processing());
        assert!(status.chunks_processed.is_some());
    }
}