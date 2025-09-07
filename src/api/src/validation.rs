use validator::Validate;
use crate::{
    models::{
        QueryRequest, IngestRequest, LoginRequest, 
        BatchIngestRequest
    },
    ApiError,
};

/// Validate a query request
pub fn validate_query_request(request: &QueryRequest) -> Result<(), ApiError> {
    request.validate().map_err(|e| {
        ApiError::BadRequest(format!("Query validation failed: {}", format_validation_errors(&e)))
    })
}

/// Validate an ingest request
pub fn validate_ingest_request(request: &IngestRequest) -> Result<(), ApiError> {
    request.validate().map_err(|e| {
        ApiError::BadRequest(format!("Ingest validation failed: {}", format_validation_errors(&e)))
    })
}

/// Validate a batch ingest request
pub fn validate_batch_ingest_request(request: &BatchIngestRequest) -> Result<(), ApiError> {
    request.validate().map_err(|e| {
        ApiError::BadRequest(format!("Batch ingest validation failed: {}", format_validation_errors(&e)))
    })?;
    
    // Validate each individual document in the batch
    for (index, doc) in request.documents.iter().enumerate() {
        if let Err(e) = doc.validate() {
            return Err(ApiError::BadRequest(format!(
                "Document {} validation failed: {}", 
                index, 
                format_validation_errors(&e)
            )));
        }
    }
    
    Ok(())
}

/// Validate a login request
pub fn validate_login_request(request: &LoginRequest) -> Result<(), ApiError> {
    request.validate().map_err(|e| {
        ApiError::BadRequest(format!("Login validation failed: {}", format_validation_errors(&e)))
    })
}

/// Validate query history parameters
pub fn validate_query_history_params(params: &crate::handlers::queries::QueryHistoryParams) -> Result<(), ApiError> {
    // Custom validation for query history parameters
    if let Some(limit) = params.limit {
        if limit == 0 {
            return Err(ApiError::BadRequest("Limit must be greater than 0".to_string()));
        }
        if limit > 1000 {
            return Err(ApiError::BadRequest("Limit cannot exceed 1000".to_string()));
        }
    }
    
    if let Some(start_date) = params.start_date {
        if let Some(end_date) = params.end_date {
            if start_date > end_date {
                return Err(ApiError::BadRequest("Start date cannot be after end date".to_string()));
            }
        }
    }
    
    Ok(())
}

/// Validate file upload parameters
pub fn validate_file_upload(
    filename: &str,
    content_type: Option<&str>,
    file_size: usize,
    max_file_size: usize,
    supported_formats: &[String],
) -> Result<(), ApiError> {
    // Validate filename
    if filename.is_empty() {
        return Err(ApiError::BadRequest("Filename cannot be empty".to_string()));
    }
    
    // Validate file size
    if file_size == 0 {
        return Err(ApiError::BadRequest("File cannot be empty".to_string()));
    }
    
    if file_size > max_file_size {
        return Err(ApiError::BadRequest(format!(
            "File size {} exceeds maximum allowed size {}",
            file_size, max_file_size
        )));
    }
    
    // Validate content type
    if let Some(content_type) = content_type {
        if !supported_formats.iter().any(|format| content_type.starts_with(format)) {
            return Err(ApiError::BadRequest(format!(
                "Unsupported file format: {}. Supported formats: {}",
                content_type,
                supported_formats.join(", ")
            )));
        }
    }
    
    // Validate filename for security (no path traversal)
    if filename.contains("..") || filename.contains('/') || filename.contains('\\') {
        return Err(ApiError::BadRequest("Invalid filename - contains path separators".to_string()));
    }
    
    Ok(())
}

/// Validate pagination parameters
pub fn validate_pagination(limit: Option<usize>, offset: Option<usize>) -> Result<(usize, usize), ApiError> {
    let limit = limit.unwrap_or(50);
    let offset = offset.unwrap_or(0);
    
    if limit == 0 {
        return Err(ApiError::BadRequest("Limit must be greater than 0".to_string()));
    }
    
    if limit > 1000 {
        return Err(ApiError::BadRequest("Limit cannot exceed 1000".to_string()));
    }
    
    Ok((limit, offset))
}

/// Custom validation for UUID strings
pub fn validate_uuid_string(uuid_str: &str) -> Result<uuid::Uuid, ApiError> {
    uuid::Uuid::parse_str(uuid_str)
        .map_err(|_| ApiError::BadRequest(format!("Invalid UUID format: {}", uuid_str)))
}

/// Validate query preferences
pub fn validate_query_preferences(preferences: &crate::models::QueryPreferences) -> Result<(), ApiError> {
    if let Some(temperature) = preferences.temperature {
        if temperature < 0.0 || temperature > 2.0 {
            return Err(ApiError::BadRequest(
                "Temperature must be between 0.0 and 2.0".to_string()
            ));
        }
    }
    
    if let Some(max_context_length) = preferences.max_context_length {
        if max_context_length == 0 || max_context_length > 32000 {
            return Err(ApiError::BadRequest(
                "Max context length must be between 1 and 32000".to_string()
            ));
        }
    }
    
    Ok(())
}

/// Validate chunking strategy
pub fn validate_chunking_strategy(strategy: &crate::models::ChunkingStrategy) -> Result<(), ApiError> {
    if let Some(max_chunk_size) = strategy.max_chunk_size {
        if max_chunk_size == 0 || max_chunk_size > 10000 {
            return Err(ApiError::BadRequest(
                "Max chunk size must be between 1 and 10000".to_string()
            ));
        }
    }
    
    if let Some(overlap_size) = strategy.overlap_size {
        if let Some(max_chunk_size) = strategy.max_chunk_size {
            if overlap_size >= max_chunk_size {
                return Err(ApiError::BadRequest(
                    "Overlap size must be smaller than max chunk size".to_string()
                ));
            }
        }
    }
    
    Ok(())
}

/// Format validation errors into a readable string
fn format_validation_errors(errors: &validator::ValidationErrors) -> String {
    errors
        .field_errors()
        .iter()
        .map(|(field, field_errors)| {
            let field_messages: Vec<String> = field_errors
                .iter()
                .map(|error| {
                    error.message
                        .as_ref()
                        .map(|msg| msg.to_string())
                        .unwrap_or_else(|| format!("Invalid value for field '{}'", field))
                })
                .collect();
            format!("{}: {}", field, field_messages.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Sanitize text input to prevent common security issues
pub fn sanitize_text_input(input: &str) -> String {
    // Basic HTML entity encoding and removal of control characters
    input
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
        .collect::<String>()
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

/// Validate email format (more thorough than basic regex)
pub fn validate_email(email: &str) -> Result<(), ApiError> {
    if email.is_empty() {
        return Err(ApiError::BadRequest("Email cannot be empty".to_string()));
    }
    
    if email.len() > 320 {
        return Err(ApiError::BadRequest("Email is too long".to_string()));
    }
    
    let email_regex = regex::Regex::new(
        r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
    ).unwrap();
    
    if !email_regex.is_match(email) {
        return Err(ApiError::BadRequest("Invalid email format".to_string()));
    }
    
    Ok(())
}

/// Validate password strength
pub fn validate_password_strength(password: &str, min_length: usize) -> Result<(), ApiError> {
    if password.len() < min_length {
        return Err(ApiError::BadRequest(format!(
            "Password must be at least {} characters long",
            min_length
        )));
    }
    
    let has_lowercase = password.chars().any(|c| c.is_lowercase());
    let has_uppercase = password.chars().any(|c| c.is_uppercase());
    let has_digit = password.chars().any(|c| c.is_ascii_digit());
    let has_special = password.chars().any(|c| "!@#$%^&*()_+-=[]{}|;':\",./<>?".contains(c));
    
    let strength_score = [has_lowercase, has_uppercase, has_digit, has_special]
        .iter()
        .map(|&b| if b { 1 } else { 0 })
        .sum::<i32>();
    
    if strength_score < 3 {
        return Err(ApiError::BadRequest(
            "Password must contain at least 3 of: lowercase, uppercase, digit, special character".to_string()
        ));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::*;
    use uuid::Uuid;

    #[test]
    fn test_query_validation() {
        let valid_query = QueryRequest {
            query_id: Uuid::new_v4(),
            query: "test query".to_string(),
            user_id: None,
            context: None,
            preferences: None,
            max_results: Some(10),
            include_sources: Some(true),
        };
        
        assert!(validate_query_request(&valid_query).is_ok());
        
        let invalid_query = QueryRequest {
            query_id: Uuid::new_v4(),
            query: "".to_string(), // Empty query
            user_id: None,
            context: None,
            preferences: None,
            max_results: Some(10),
            include_sources: Some(true),
        };
        
        assert!(validate_query_request(&invalid_query).is_err());
    }

    #[test]
    fn test_pagination_validation() {
        assert_eq!(validate_pagination(Some(50), Some(0)), Ok((50, 0)));
        assert_eq!(validate_pagination(None, None), Ok((50, 0)));
        assert!(validate_pagination(Some(0), Some(0)).is_err());
        assert!(validate_pagination(Some(2000), Some(0)).is_err());
    }

    #[test]
    fn test_uuid_validation() {
        let valid_uuid = "550e8400-e29b-41d4-a716-446655440000";
        let invalid_uuid = "not-a-uuid";
        
        assert!(validate_uuid_string(valid_uuid).is_ok());
        assert!(validate_uuid_string(invalid_uuid).is_err());
    }

    #[test]
    fn test_text_sanitization() {
        let malicious_input = "<script>alert('xss')</script>";
        let sanitized = sanitize_text_input(malicious_input);
        assert!(!sanitized.contains("<script>"));
        assert!(sanitized.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_email_validation() {
        assert!(validate_email("user@example.com").is_ok());
        assert!(validate_email("invalid-email").is_err());
        assert!(validate_email("").is_err());
    }

    #[test]
    fn test_password_validation() {
        assert!(validate_password_strength("Password123!", 8).is_ok());
        assert!(validate_password_strength("weak", 8).is_err());
        assert!(validate_password_strength("NoNumber!", 8).is_err());
    }

    #[test]
    fn test_file_validation() {
        let supported_formats = vec!["application/pdf".to_string(), "text/plain".to_string()];
        
        assert!(validate_file_upload(
            "document.pdf",
            Some("application/pdf"),
            1000,
            10000,
            &supported_formats
        ).is_ok());
        
        assert!(validate_file_upload(
            "document.exe",
            Some("application/exe"),
            1000,
            10000,
            &supported_formats
        ).is_err());
        
        assert!(validate_file_upload(
            "../../../etc/passwd",
            Some("text/plain"),
            1000,
            10000,
            &supported_formats
        ).is_err());
    }
}