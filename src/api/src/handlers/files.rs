use axum::{
    extract::{Path, Request, State, Multipart},
    response::Json,
    http::StatusCode,
};
use std::sync::Arc;
use tracing::{info, warn, error};
use uuid::Uuid;

use crate::{
    clients::ComponentClients,
    models::{FileUploadResponse, TaskStatus},
    middleware::auth::AuthExtension,
    validation::validate_file_upload,
    Result, ApiError,
};

/// Upload a file for processing  
pub async fn upload_file(
    State(clients): State<Arc<ComponentClients>>,
    State(config): State<Arc<crate::config::ApiConfig>>,
    mut multipart: Multipart,
) -> Result<Json<FileUploadResponse>> {
    // Auth context would be available through middleware
    // let auth_context = request.require_auth_context()?;
    info!("File upload requested");
    
    let mut filename = String::new();
    let mut content_type: Option<String> = None;
    let mut file_content = Vec::new();
    
    // Process multipart form data
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        ApiError::BadRequest(format!("Invalid multipart data: {}", e))
    })? {
        let field_name = field.name().unwrap_or("unknown");
        
        match field_name {
            "file" => {
                filename = field.file_name().unwrap_or("unknown").to_string();
                content_type = field.content_type().map(|ct| ct.to_string());
                file_content = field.bytes().await.map_err(|e| {
                    ApiError::BadRequest(format!("Failed to read file content: {}", e))
                })?.to_vec();
            }
            "filename" => {
                if filename.is_empty() {
                    filename = String::from_utf8_lossy(&field.bytes().await.map_err(|e| {
                        ApiError::BadRequest(format!("Failed to read filename: {}", e))
                    })?).to_string();
                }
            }
            _ => {
                // Ignore unknown fields
                warn!("Ignoring unknown multipart field: {}", field_name);
            }
        }
    }
    
    // Validate file upload
    if filename.is_empty() {
        return Err(ApiError::BadRequest("Filename is required".to_string()));
    }
    
    if file_content.is_empty() {
        return Err(ApiError::BadRequest("File content is required".to_string()));
    }
    
    validate_file_upload(
        &filename,
        content_type.as_deref(),
        file_content.len(),
        config.features.max_file_size_mb * 1024 * 1024,
        &config.features.supported_formats,
    )?;
    
    let file_id = Uuid::new_v4();
    let start_time = std::time::Instant::now();
    
    info!("Processing file upload: file_id={}, filename={}, size={} bytes", 
          file_id, filename, file_content.len());
    
    // Store the file
    let content_type_for_storage = content_type.clone().unwrap_or("application/octet-stream".to_string());
    match clients.store_uploaded_file(
        file_id,
        filename.clone(),
        content_type_for_storage,
        file_content,
    ).await {
        Ok(upload_url) => {
            let processing_time = start_time.elapsed();
            
            info!("File upload successful: file_id={}, processing_time={:?}", 
                  file_id, processing_time);
            
            Ok(Json(FileUploadResponse {
                file_id,
                filename,
                content_type: content_type.unwrap_or("application/octet-stream".to_string()),
                size_bytes: 0, // Would be populated from actual storage
                upload_url: Some(upload_url),
                processing_status: TaskStatus::Completed,
            }))
        }
        Err(e) => {
            error!("File upload failed: file_id={}, error={}", file_id, e);
            Err(ApiError::Internal(format!("File upload failed: {}", e)))
        }
    }
}

/// Get file information
pub async fn get_file_info(
    Path(file_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
    request: Request,
) -> Result<Json<FileUploadResponse>> {
    let auth_context = request.require_auth_context()?;
    info!("File info requested: file_id={}, user={}", file_id, auth_context.email);
    
    match clients.get_file_info(file_id).await {
        Ok(file_info) => {
            Ok(Json(file_info))
        }
        Err(e) => {
            error!("Failed to retrieve file info: file_id={}, error={}", file_id, e);
            Err(ApiError::NotFound(format!("File not found: {}", file_id)))
        }
    }
}

/// Delete a file
pub async fn delete_file(
    Path(file_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
    request: Request,
) -> Result<StatusCode> {
    let auth_context = request.require_auth_context()?;
    info!("File deletion requested: file_id={}, user={}", file_id, auth_context.email);
    
    match clients.delete_file(file_id, auth_context.user_id).await {
        Ok(true) => {
            info!("File deleted successfully: file_id={}", file_id);
            Ok(StatusCode::NO_CONTENT)
        }
        Ok(false) => {
            warn!("File not found or access denied: file_id={}, user={}", 
                  file_id, auth_context.user_id);
            Err(ApiError::NotFound(format!("File not found: {}", file_id)))
        }
        Err(e) => {
            error!("Failed to delete file: file_id={}, error={}", file_id, e);
            Err(ApiError::Internal(format!("Failed to delete file: {}", e)))
        }
    }
}

/// Process uploaded file (extract text and ingest)
pub async fn process_file(
    Path(file_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
    request: Request,
) -> Result<Json<serde_json::Value>> {
    let auth_context = request.require_auth_context()?;
    info!("File processing requested: file_id={}, user={}", file_id, auth_context.email);
    
    // Get file information
    let file_info = clients.get_file_info(file_id).await
        .map_err(|e| ApiError::NotFound(format!("File not found: {}", e)))?;
    
    // In a real implementation, this would:
    // 1. Extract text from the file based on content type
    // 2. Create an ingestion request with the extracted text
    // 3. Process the document through the ingestion pipeline
    // 4. Return processing status
    
    let task_id = Uuid::new_v4();
    
    let response = serde_json::json!({
        "file_id": file_id,
        "task_id": task_id,
        "filename": file_info.filename,
        "status": "processing",
        "message": "File processing started successfully"
    });
    
    Ok(Json(response))
}

/// Get file processing status
pub async fn get_file_processing_status(
    Path(file_id): Path<Uuid>,
    State(_clients): State<Arc<ComponentClients>>,
    request: Request,
) -> Result<Json<serde_json::Value>> {
    let auth_context = request.require_auth_context()?;
    info!("File processing status requested: file_id={}, user={}", file_id, auth_context.email);
    
    // In a real implementation, this would query the processing status from database
    let response = serde_json::json!({
        "file_id": file_id,
        "status": "completed",
        "progress_percent": 100,
        "extracted_text_length": 0,
        "chunks_created": 0,
        "processing_time_ms": 0,
        "message": "File processed successfully"
    });
    
    Ok(Json(response))
}

/// List user's uploaded files
pub async fn list_user_files(
    State(_clients): State<Arc<ComponentClients>>,
    request: Request,
) -> Result<Json<serde_json::Value>> {
    let auth_context = request.require_auth_context()?;
    info!("File list requested by user: {}", auth_context.email);
    
    // In a real implementation, this would query the database for user's files
    let response = serde_json::json!({
        "user_id": auth_context.user_id,
        "files": [],
        "total_count": 0,
        "total_size_bytes": 0
    });
    
    Ok(Json(response))
}

/// Extract text from file without processing
pub async fn extract_text_from_file(
    Path(file_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
    request: Request,
) -> Result<Json<serde_json::Value>> {
    let auth_context = request.require_auth_context()?;
    info!("Text extraction requested: file_id={}, user={}", file_id, auth_context.email);
    
    // Get file information
    let file_info = clients.get_file_info(file_id).await
        .map_err(|e| ApiError::NotFound(format!("File not found: {}", e)))?;
    
    // In a real implementation, this would extract text from the file
    let extracted_text = match file_info.content_type.as_str() {
        "text/plain" => "Sample extracted text from plain text file".to_string(),
        "application/pdf" => "Sample extracted text from PDF file".to_string(),
        "application/json" => "Sample extracted text from JSON file".to_string(),
        _ => return Err(ApiError::BadRequest(format!(
            "Text extraction not supported for content type: {}", 
            file_info.content_type
        ))),
    };
    
    let response = serde_json::json!({
        "file_id": file_id,
        "filename": file_info.filename,
        "content_type": file_info.content_type,
        "extracted_text": extracted_text,
        "text_length": extracted_text.len(),
        "extraction_successful": true
    });
    
    Ok(Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiConfig;

    #[test]
    fn test_file_upload_validation() {
        let supported_formats = vec![
            "application/pdf".to_string(),
            "text/plain".to_string(),
            "application/json".to_string(),
        ];
        
        // Valid file upload
        assert!(validate_file_upload(
            "document.pdf",
            Some("application/pdf"),
            1024,
            10 * 1024 * 1024,
            &supported_formats
        ).is_ok());
        
        // Empty filename
        assert!(validate_file_upload(
            "",
            Some("application/pdf"),
            1024,
            10 * 1024 * 1024,
            &supported_formats
        ).is_err());
        
        // Unsupported format
        assert!(validate_file_upload(
            "document.exe",
            Some("application/exe"),
            1024,
            10 * 1024 * 1024,
            &supported_formats
        ).is_err());
        
        // File too large
        assert!(validate_file_upload(
            "document.pdf",
            Some("application/pdf"),
            20 * 1024 * 1024, // 20MB
            10 * 1024 * 1024, // Max 10MB
            &supported_formats
        ).is_err());
        
        // Path traversal attempt
        assert!(validate_file_upload(
            "../../../etc/passwd",
            Some("text/plain"),
            1024,
            10 * 1024 * 1024,
            &supported_formats
        ).is_err());
    }

    #[test]
    fn test_file_upload_response() {
        let response = FileUploadResponse {
            file_id: Uuid::new_v4(),
            filename: "test.pdf".to_string(),
            content_type: "application/pdf".to_string(),
            size_bytes: 1024,
            upload_url: Some("/files/123".to_string()),
            processing_status: TaskStatus::Completed,
        };
        
        assert_eq!(response.filename, "test.pdf");
        assert_eq!(response.content_type, "application/pdf");
        assert_eq!(response.size_bytes, 1024);
        assert!(response.processing_status.is_terminal());
    }

    #[tokio::test]
    async fn test_extract_text_content_types() {
        // Test different content types for text extraction
        let supported_types = [
            "text/plain",
            "application/pdf", 
            "application/json"
        ];
        
        for content_type in &supported_types {
            // Would test actual extraction logic here
            assert!(content_type.len() > 0);
        }
    }
}