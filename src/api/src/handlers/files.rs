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
    models::{FileUploadResponse, TaskStatus},
    middleware::auth::{AuthExtension},
    validation::{validate_file_type, validate_file_size},
    Result, ApiError,
};

/// Upload a file for processing
pub async fn upload_file(
    State(clients): State<Arc<ComponentClients>>,
    mut multipart: Multipart,
) -> Result<Json<FileUploadResponse>> {
    info!("Processing file upload");

    let mut file_content = Vec::new();
    let mut filename = None;
    let mut content_type = None;

    // Process multipart form data
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        ApiError::BadRequest(format!("Invalid multipart data: {}", e))
    })? {
        let field_name = field.name().unwrap_or_default().to_string();
        
        match field_name.as_str() {
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

    // Validate file type and size
    if let Some(ref ct) = content_type {
        validate_file_type(ct)?;
    }
    validate_file_size(file_content.len())?;

    let file_id = Uuid::new_v4();
    let filename = filename.unwrap_or_else(|| format!("upload_{}", file_id));

    // Store file temporarily and get processing status
    let upload_result = clients.store_uploaded_file(
        file_id,
        filename.clone(),
        content_type.clone().unwrap_or_default(),
        file_content,
    ).await;

    match upload_result {
        Ok(upload_url) => {
            info!("File uploaded successfully: file_id={}, filename={}", file_id, filename);
            
            Ok(Json(FileUploadResponse {
                file_id,
                filename,
                content_type: content_type.unwrap_or_default(),
                size_bytes: file_content.len() as u64,
                upload_url: Some(upload_url),
                processing_status: TaskStatus::Pending,
            }))
        }
        Err(e) => {
            error!("File upload failed: file_id={}, error={}", file_id, e);
            Err(ApiError::Internal(format!("File upload failed: {}", e)))
        }
    }
}

/// Get file information
pub async fn get_file(
    Path(file_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
) -> Result<Json<FileUploadResponse>> {
    info!("Retrieving file information: file_id={}", file_id);

    match clients.get_file_info(file_id).await {
        Ok(file_info) => Ok(Json(file_info)),
        Err(e) => {
            warn!("File not found: file_id={}, error={}", file_id, e);
            Err(ApiError::NotFound(format!("File not found: {}", file_id)))
        }
    }
}

/// Delete a file
pub async fn delete_file(
    Path(file_id): Path<Uuid>,
    State(clients): State<Arc<ComponentClients>>,
    request: axum::extract::Request,
) -> Result<StatusCode> {
    let auth_context = request.require_auth_context()?;
    info!("Deleting file: file_id={}, user={}", file_id, auth_context.email);

    match clients.delete_file(file_id, auth_context.user_id).await {
        Ok(true) => {
            info!("File deleted successfully: file_id={}", file_id);
            Ok(StatusCode::NO_CONTENT)
        }
        Ok(false) => {
            warn!("File not found or access denied: file_id={}", file_id);
            Err(ApiError::NotFound(format!("File not found: {}", file_id)))
        }
        Err(e) => {
            error!("Failed to delete file: file_id={}, error={}", file_id, e);
            Err(ApiError::Internal(format!("Failed to delete file: {}", e)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_id_generation() {
        let file_id1 = Uuid::new_v4();
        let file_id2 = Uuid::new_v4();
        
        assert_ne!(file_id1, file_id2);
        assert_ne!(file_id1, Uuid::nil());
    }
}