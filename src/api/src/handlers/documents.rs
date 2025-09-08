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
    server::AppState,
    validation::{validate_ingest_request, validate_batch_ingest_request, validate_chunking_strategy},
    Result, ApiError,
};

/// Process a single document ingestion
pub async fn ingest_document(
    State(state): State<Arc<AppState>>,
    Json(request): Json<IngestRequest>,
) -> Result<Json<IngestResponse>> {
    let clients = &state.clients;
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
                chunks_created: Some(count_document_chunks(&document_id, &clients).await.unwrap_or(0) as usize),
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
    State(state): State<Arc<AppState>>,
    Json(request): Json<BatchIngestRequest>,
) -> Result<Json<BatchIngestResponse>> {
    let clients = &state.clients;
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
                    chunks_created: Some(count_document_chunks(&document_id, &clients).await.unwrap_or(0) as usize),
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
                    status: crate::models::TaskStatus::Failed,
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
    State(state): State<Arc<AppState>>,
) -> Result<Json<DocumentStatus>> {
    let clients = &state.clients;
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
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let clients = &state.clients;
    info!("Retrieving document processing history");
    
    // Query the database for real processing history
    let history = match get_real_processing_history(&clients).await {
        Ok(hist) => hist,
        Err(e) => {
            warn!("Failed to retrieve processing history: {}", e);
            serde_json::json!({
                "total_documents": 0,
                "successful_processing": 0,
                "failed_processing": 0,
                "average_processing_time_ms": 0,
                "recent_documents": [],
                "error": "Failed to retrieve history from database"
            })
        }
    };
    
    Ok(Json(history))
}

/// Cancel document processing
pub async fn cancel_document_processing(
    Path(task_id): Path<Uuid>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let clients = &state.clients;
    info!("Cancelling document processing: task_id={}", task_id);
    
    // Real cancellation implementation
    match cancel_real_document_processing(task_id, &clients).await {
        Ok(cancellation_result) => {
            info!("Document processing cancelled: task_id={}", task_id);
            Ok(Json(cancellation_result))
        },
        Err(e) => {
            error!("Failed to cancel document processing: task_id={}, error={}", task_id, e);
            let response = serde_json::json!({
                "task_id": task_id,
                "status": "error",
                "message": format!("Failed to cancel processing: {}", e)
            });
            Ok(Json(response))
        }
    }
}

/// Retry failed document processing
pub async fn retry_document_processing(
    Path(task_id): Path<Uuid>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<IngestResponse>> {
    let clients = &state.clients;
    info!("Retrying document processing: task_id={}", task_id);
    
    // Real retry implementation
    let start_time = std::time::Instant::now();
    
    match retry_real_document_processing(task_id, &clients).await {
        Ok(retry_result) => {
            info!("Document processing retried: task_id={}", task_id);
            Ok(Json(retry_result))
        },
        Err(e) => {
            error!("Failed to retry document processing: task_id={}, error={}", task_id, e);
            let processing_time = start_time.elapsed();
            
            let response = IngestResponse {
                task_id,
                document_id: Uuid::new_v4(),
                status: crate::models::TaskStatus::Failed,
                message: format!("Retry failed: {}", e),
                chunks_created: None,
                processing_time_ms: processing_time.as_millis() as u64,
            };
            
            Ok(Json(response))
        }
    }
}

/// Get processing statistics
pub async fn get_processing_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    let clients = &state.clients;
    info!("Retrieving processing statistics");
    
    // Aggregate real statistics from the database
    let stats = match get_real_processing_stats(&clients).await {
        Ok(statistics) => statistics,
        Err(e) => {
            warn!("Failed to retrieve processing statistics: {}", e);
            serde_json::json!({
                "total_documents_processed": 0,
                "documents_processed_today": 0,
                "average_processing_time_ms": 0,
                "success_rate_percent": 0.0,
                "active_processing_tasks": 0,
                "queue_size": 0,
                "processing_rate_per_hour": 0,
                "storage_usage_mb": 0,
                "by_content_type": {
                    "application/pdf": 0,
                    "text/plain": 0,
                    "application/json": 0
                },
                "error": "Failed to retrieve statistics from database"
            })
        }
    };
    
    Ok(Json(stats))
}

/// Count chunks for a specific document
async fn count_document_chunks(
    document_id: &uuid::Uuid,
    clients: &Arc<ComponentClients>,
) -> Result<u32> {
    // Query the storage backend to get chunk count for document
    match clients.storage().count_chunks_for_document(*document_id).await {
        Ok(count) => Ok(count as u32),
        Err(e) => {
            warn!("Failed to count chunks for document {}: {}", document_id, e);
            // Return 0 as fallback rather than failing the request
            Ok(0)
        }
    }
}

/// Get real processing history from database
async fn get_real_processing_history(
    clients: &Arc<ComponentClients>,
) -> Result<serde_json::Value> {
    let processing_history = clients.storage().get_processing_history().await
        .map_err(|e| ApiError::Internal(format!("Failed to query processing history: {}", e)))?;
    
    let recent_documents = clients.storage().get_recent_documents(10).await
        .map_err(|e| ApiError::Internal(format!("Failed to query recent documents: {}", e)))?;
    
    // Calculate statistics from processing history entries
    let successful_count = processing_history.entries.iter()
        .filter(|entry| matches!(entry.status, crate::models::TaskStatus::Completed))
        .count();
    let failed_count = processing_history.entries.iter()
        .filter(|entry| matches!(entry.status, crate::models::TaskStatus::Failed))
        .count();
    let avg_processing_time = processing_history.entries.iter()
        .filter_map(|entry| entry.processing_time_ms)
        .sum::<u64>() / (processing_history.entries.len() as u64).max(1);

    Ok(serde_json::json!({
        "total_documents": processing_history.total_processed,
        "successful_processing": successful_count,
        "failed_processing": failed_count,
        "average_processing_time_ms": avg_processing_time,
        "recent_documents": recent_documents.iter().map(|doc| {
            serde_json::json!({
                "document_id": doc.document_id,
                "title": doc.title,
                "content_type": doc.content_type,
                "size_bytes": doc.size_bytes,
                "chunk_count": doc.chunk_count,
                "created_at": doc.created_at,
                "last_accessed": doc.last_accessed
            })
        }).collect::<Vec<_>>()
    }))
}

/// Get real processing statistics from database  
async fn get_real_processing_stats(
    clients: &Arc<ComponentClients>,
) -> Result<serde_json::Value> {
    let stats = clients.storage().get_processing_statistics().await
        .map_err(|e| ApiError::Internal(format!("Failed to query processing statistics: {}", e)))?;
    
    let storage_usage = clients.storage().get_storage_usage().await
        .map_err(|e| ApiError::Internal(format!("Failed to query storage usage: {}", e)))?;
    
    let content_type_breakdown = clients.storage().get_content_type_statistics().await
        .map_err(|e| ApiError::Internal(format!("Failed to query content type statistics: {}", e)))?;
    
    let active_tasks = clients.get_active_processing_tasks().await
        .map_err(|e| ApiError::Internal(format!("Failed to query active tasks: {}", e)))?;
    
    Ok(serde_json::json!({
        "total_documents_processed": stats.total_processing_tasks,
        "documents_processed_today": stats.successful_tasks,
        "average_processing_time_ms": stats.average_processing_time_ms,
        "success_rate_percent": (stats.successful_tasks as f64 / stats.total_processing_tasks.max(1) as f64) * 100.0,
        "active_processing_tasks": active_tasks.len(),
        "queue_size": clients.get_processing_queue_size().await.unwrap_or(0),
        "processing_rate_per_hour": 0.0, // Computed value not available in storage model
        "storage_usage_mb": storage_usage.total_size_bytes / 1024 / 1024,
        "by_content_type": content_type_breakdown
    }))
}

/// Cancel real document processing
async fn cancel_real_document_processing(
    task_id: Uuid,
    clients: &Arc<ComponentClients>,
) -> Result<serde_json::Value> {
    // Check if task exists and is still processing
    let task_status = clients.storage().get_task_status(task_id).await
        .map_err(|e| ApiError::Internal(format!("Failed to query task status: {}", e)))?;
    
    match task_status.status {
        crate::models::TaskStatus::Processing => {
            // Send cancellation signal to processing pipeline
            clients.send_cancellation_signal(task_id).await
                .map_err(|e| ApiError::Internal(format!("Failed to send cancellation signal: {}", e)))?;
            
            // Update task status to cancelled in database
            clients.storage().update_task_status(task_id, TaskStatus::Cancelled).await
                .map_err(|e| ApiError::Internal(format!("Failed to update task status: {}", e)))?;
            
            // Clean up any partial processing results
            if let Err(e) = clients.cleanup_partial_processing(task_id).await {
                warn!("Failed to cleanup partial processing for task {}: {}", task_id, e);
            }
            
            Ok(serde_json::json!({
                "task_id": task_id,
                "status": "cancelled",
                "message": "Document processing cancelled successfully",
                "cancelled_at": chrono::Utc::now()
            }))
        },
        crate::models::TaskStatus::Completed => {
            Err(ApiError::BadRequest("Cannot cancel completed task".to_string()))
        },
        crate::models::TaskStatus::Failed => {
            Err(ApiError::BadRequest("Cannot cancel failed task".to_string()))
        },
        crate::models::TaskStatus::Cancelled => {
            Ok(serde_json::json!({
                "task_id": task_id,
                "status": "already_cancelled",
                "message": "Task was already cancelled"
            }))
        },
        crate::models::TaskStatus::Pending => {
            // Update status to cancelled
            clients.storage().update_task_status(task_id, TaskStatus::Cancelled).await
                .map_err(|e| ApiError::Internal(format!("Failed to update task status: {}", e)))?;
            
            Ok(serde_json::json!({
                "task_id": task_id,
                "status": "cancelled",
                "message": "Pending task cancelled successfully"
            }))
        }
    }
}

/// Retry real document processing
async fn retry_real_document_processing(
    task_id: Uuid,
    clients: &Arc<ComponentClients>,
) -> Result<IngestResponse> {
    // Retrieve the original document and parameters from database
    let original_task = clients.storage().get_task_details(task_id).await
        .map_err(|e| ApiError::Internal(format!("Failed to retrieve task details: {}", e)))?;
    
    if original_task.status != crate::models::TaskStatus::Failed {
        return Err(ApiError::BadRequest("Can only retry failed tasks".to_string()));
    }
    
    // Reset the task status to pending
    clients.storage().update_task_status(task_id, TaskStatus::Pending).await
        .map_err(|e| ApiError::Internal(format!("Failed to reset task status: {}", e)))?;
    
    // Re-initiate the processing pipeline with original parameters
    let content = original_task.content.clone().unwrap_or_default();
    let metadata = original_task.metadata.clone().map(|v| {
        if let serde_json::Value::Object(map) = v {
            map.into_iter().collect()
        } else {
            std::collections::HashMap::new()
        }
    });
    let new_document_id = clients.reprocess_document(
        task_id,
        content,
        metadata,
        original_task.chunking_strategy.clone(),
    ).await
    .map_err(|e| ApiError::Internal(format!("Failed to reinitiate processing: {}", e)))?;
    
    Ok(IngestResponse {
        task_id,
        document_id: new_document_id,
        status: TaskStatus::Processing,
        message: "Document processing retried successfully".to_string(),
        chunks_created: None,
        processing_time_ms: 0, // Will be updated when processing completes
    })
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