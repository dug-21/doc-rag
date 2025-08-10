use crate::{models::*, ApiError, Result};
use validator::Validate;

/// Validate document content
pub fn validate_document_content(content: &str) -> Result<()> {
    if content.trim().is_empty() {
        return Err(ApiError::ValidationError("Document content cannot be empty".to_string()));
    }

    if content.len() > 10_000_000 {
        return Err(ApiError::ValidationError("Document content too large (max 10MB)".to_string()));
    }

    // Check for potentially problematic content
    if content.chars().filter(|c| c.is_control()).count() > content.len() / 2 {
        return Err(ApiError::ValidationError("Document contains too many control characters".to_string()));
    }

    Ok(())
}

/// Validate file type
pub fn validate_file_type(content_type: &str) -> Result<()> {
    let allowed_types = [
        "application/pdf",
        "text/plain",
        "text/markdown", 
        "text/html",
        "application/json",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // DOCX
        "application/msword", // DOC
        "text/csv",
        "application/rtf",
    ];

    if !allowed_types.contains(&content_type) {
        return Err(ApiError::ValidationError(
            format!("Unsupported file type: {}. Allowed types: {}", 
                   content_type, 
                   allowed_types.join(", "))
        ));
    }

    Ok(())
}

/// Validate file size
pub fn validate_file_size(size_bytes: usize) -> Result<()> {
    const MAX_FILE_SIZE: usize = 100 * 1024 * 1024; // 100MB

    if size_bytes == 0 {
        return Err(ApiError::ValidationError("File cannot be empty".to_string()));
    }

    if size_bytes > MAX_FILE_SIZE {
        return Err(ApiError::ValidationError(
            format!("File too large: {} bytes (max {} bytes)", size_bytes, MAX_FILE_SIZE)
        ));
    }

    Ok(())
}

/// Validate query request using validator crate
pub fn validate_query_request(request: &QueryRequest) -> Result<()> {
    // Use validator crate for struct validation
    request.validate()
        .map_err(|e| ApiError::ValidationError(format!("Query validation failed: {}", e)))?;

    // Additional business logic validation
    if request.query.trim().is_empty() {
        return Err(ApiError::ValidationError("Query cannot be empty".to_string()));
    }

    if let Some(max_results) = request.max_results {
        if max_results == 0 {
            return Err(ApiError::ValidationError("Max results must be greater than 0".to_string()));
        }
        if max_results > 100 {
            return Err(ApiError::ValidationError("Max results cannot exceed 100".to_string()));
        }
    }

    // Validate preferences if present
    if let Some(ref prefs) = request.preferences {
        validate_query_preferences(prefs)?;
    }

    Ok(())
}

/// Validate query preferences
fn validate_query_preferences(preferences: &QueryPreferences) -> Result<()> {
    if let Some(temperature) = preferences.temperature {
        if temperature < 0.0 || temperature > 2.0 {
            return Err(ApiError::ValidationError(
                "Temperature must be between 0.0 and 2.0".to_string()
            ));
        }
    }

    if let Some(max_context) = preferences.max_context_length {
        if max_context > 32000 {
            return Err(ApiError::ValidationError(
                "Max context length cannot exceed 32,000".to_string()
            ));
        }
    }

    Ok(())
}

/// Validate login request
pub fn validate_login_request(request: &LoginRequest) -> Result<()> {
    request.validate()
        .map_err(|e| ApiError::ValidationError(format!("Login validation failed: {}", e)))?;

    // Additional validation
    if request.email.len() > 255 {
        return Err(ApiError::ValidationError("Email too long".to_string()));
    }

    if request.password.len() > 128 {
        return Err(ApiError::ValidationError("Password too long".to_string()));
    }

    // Check for common security issues
    if request.password.to_lowercase().contains(&request.email.to_lowercase().split('@').next().unwrap_or("")) {
        return Err(ApiError::ValidationError("Password cannot contain email username".to_string()));
    }

    Ok(())
}

/// Validate batch ingest request
pub fn validate_batch_ingest_request(request: &BatchIngestRequest) -> Result<()> {
    if request.documents.is_empty() {
        return Err(ApiError::ValidationError("Batch cannot be empty".to_string()));
    }

    if request.documents.len() > 100 {
        return Err(ApiError::ValidationError("Batch too large (max 100 documents)".to_string()));
    }

    // Validate each document in the batch
    for (index, doc) in request.documents.iter().enumerate() {
        doc.validate()
            .map_err(|e| ApiError::ValidationError(
                format!("Document {} validation failed: {}", index + 1, e)
            ))?;

        validate_document_content(&doc.content)
            .map_err(|e| match e {
                ApiError::ValidationError(msg) => ApiError::ValidationError(
                    format!("Document {}: {}", index + 1, msg)
                ),
                other => other,
            })?;
    }

    // Check total content size
    let total_content_size: usize = request.documents
        .iter()
        .map(|doc| doc.content.len())
        .sum();

    if total_content_size > 50_000_000 { // 50MB total
        return Err(ApiError::ValidationError(
            "Total batch content size exceeds 50MB".to_string()
        ));
    }

    Ok(())
}

/// Validate chunking strategy
pub fn validate_chunking_strategy(strategy: &ChunkingStrategy) -> Result<()> {
    if let Some(chunk_size) = strategy.max_chunk_size {
        if chunk_size < 100 || chunk_size > 8192 {
            return Err(ApiError::ValidationError(
                "Chunk size must be between 100 and 8192 characters".to_string()
            ));
        }
    }

    if let Some(overlap_size) = strategy.overlap_size {
        if let Some(chunk_size) = strategy.max_chunk_size {
            if overlap_size >= chunk_size {
                return Err(ApiError::ValidationError(
                    "Overlap size must be less than chunk size".to_string()
                ));
            }
        }

        if overlap_size > 1000 {
            return Err(ApiError::ValidationError(
                "Overlap size cannot exceed 1000 characters".to_string()
            ));
        }
    }

    Ok(())
}

/// Validate UUID format
pub fn validate_uuid_string(uuid_str: &str) -> Result<uuid::Uuid> {
    uuid::Uuid::parse_str(uuid_str)
        .map_err(|_| ApiError::ValidationError(format!("Invalid UUID format: {}", uuid_str)))
}

/// Validate pagination parameters
pub fn validate_pagination(limit: Option<usize>, offset: Option<usize>) -> Result<(usize, usize)> {
    let limit = limit.unwrap_or(50);
    let offset = offset.unwrap_or(0);

    if limit == 0 {
        return Err(ApiError::ValidationError("Limit must be greater than 0".to_string()));
    }

    if limit > 1000 {
        return Err(ApiError::ValidationError("Limit cannot exceed 1000".to_string()));
    }

    if offset > 1_000_000 {
        return Err(ApiError::ValidationError("Offset too large".to_string()));
    }

    Ok((limit, offset))
}

/// Sanitize text input (remove potentially harmful content)
pub fn sanitize_text_input(input: &str) -> String {
    // Remove null bytes and other control characters except newlines and tabs
    input.chars()
        .filter(|&c| {
            !c.is_control() || c == '\n' || c == '\t' || c == '\r'
        })
        .collect::<String>()
        .trim()
        .to_string()
}

/// Validate email format (more strict than just regex)
pub fn validate_email_format(email: &str) -> Result<()> {
    if email.is_empty() {
        return Err(ApiError::ValidationError("Email cannot be empty".to_string()));
    }

    if email.len() > 254 {
        return Err(ApiError::ValidationError("Email too long".to_string()));
    }

    // Basic email format check
    if !email.contains('@') || !email.contains('.') {
        return Err(ApiError::ValidationError("Invalid email format".to_string()));
    }

    let parts: Vec<&str> = email.split('@').collect();
    if parts.len() != 2 {
        return Err(ApiError::ValidationError("Invalid email format".to_string()));
    }

    let (local, domain) = (parts[0], parts[1]);

    if local.is_empty() || domain.is_empty() {
        return Err(ApiError::ValidationError("Invalid email format".to_string()));
    }

    if local.len() > 64 {
        return Err(ApiError::ValidationError("Email local part too long".to_string()));
    }

    if domain.len() > 253 {
        return Err(ApiError::ValidationError("Email domain too long".to_string()));
    }

    // Check for dangerous patterns
    let dangerous_patterns = ["<script", "javascript:", "data:", "vbscript:"];
    for pattern in &dangerous_patterns {
        if email.to_lowercase().contains(pattern) {
            return Err(ApiError::ValidationError("Invalid email format".to_string()));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_validate_document_content() {
        // Valid content
        assert!(validate_document_content("This is valid content").is_ok());

        // Empty content
        assert!(validate_document_content("").is_err());
        assert!(validate_document_content("   ").is_err());

        // Too large content
        let large_content = "a".repeat(10_000_001);
        assert!(validate_document_content(&large_content).is_err());

        // Too many control characters
        let control_content = "\x01\x02\x03\x04";
        assert!(validate_document_content(control_content).is_err());
    }

    #[test]
    fn test_validate_file_type() {
        // Valid types
        assert!(validate_file_type("application/pdf").is_ok());
        assert!(validate_file_type("text/plain").is_ok());
        assert!(validate_file_type("application/json").is_ok());

        // Invalid types
        assert!(validate_file_type("image/jpeg").is_err());
        assert!(validate_file_type("video/mp4").is_err());
        assert!(validate_file_type("application/x-executable").is_err());
    }

    #[test]
    fn test_validate_file_size() {
        // Valid sizes
        assert!(validate_file_size(1000).is_ok());
        assert!(validate_file_size(50 * 1024 * 1024).is_ok()); // 50MB

        // Empty file
        assert!(validate_file_size(0).is_err());

        // Too large
        assert!(validate_file_size(101 * 1024 * 1024).is_err()); // 101MB
    }

    #[test]
    fn test_validate_query_preferences() {
        let mut prefs = QueryPreferences::default();

        // Valid preferences
        assert!(validate_query_preferences(&prefs).is_ok());

        // Invalid temperature
        prefs.temperature = Some(-1.0);
        assert!(validate_query_preferences(&prefs).is_err());

        prefs.temperature = Some(3.0);
        assert!(validate_query_preferences(&prefs).is_err());

        prefs.temperature = Some(1.0);
        assert!(validate_query_preferences(&prefs).is_ok());

        // Invalid max context length
        prefs.max_context_length = Some(50000);
        assert!(validate_query_preferences(&prefs).is_err());
    }

    #[test]
    fn test_validate_pagination() {
        // Valid pagination
        assert!(validate_pagination(Some(10), Some(0)).is_ok());
        assert!(validate_pagination(None, None).is_ok());

        // Invalid limit
        assert!(validate_pagination(Some(0), Some(0)).is_err());
        assert!(validate_pagination(Some(1001), Some(0)).is_err());

        // Invalid offset
        assert!(validate_pagination(Some(10), Some(1_000_001)).is_err());
    }

    #[test]
    fn test_sanitize_text_input() {
        assert_eq!(sanitize_text_input("Hello World"), "Hello World");
        assert_eq!(sanitize_text_input("  Hello World  "), "Hello World");
        assert_eq!(sanitize_text_input("Hello\nWorld"), "Hello\nWorld");
        assert_eq!(sanitize_text_input("Hello\tWorld"), "Hello\tWorld");
        
        // Remove null bytes and other control characters
        assert_eq!(sanitize_text_input("Hello\x00World"), "HelloWorld");
        assert_eq!(sanitize_text_input("Hello\x01\x02World"), "HelloWorld");
    }

    #[test]
    fn test_validate_email_format() {
        // Valid emails
        assert!(validate_email_format("user@example.com").is_ok());
        assert!(validate_email_format("test.email@domain.co.uk").is_ok());

        // Invalid emails
        assert!(validate_email_format("").is_err());
        assert!(validate_email_format("invalid").is_err());
        assert!(validate_email_format("@example.com").is_err());
        assert!(validate_email_format("user@").is_err());
        assert!(validate_email_format("user@@example.com").is_err());

        // Dangerous patterns
        assert!(validate_email_format("user<script>@example.com").is_err());
        assert!(validate_email_format("javascript:alert@example.com").is_err());
    }

    #[test]
    fn test_validate_uuid_string() {
        let valid_uuid = Uuid::new_v4().to_string();
        assert!(validate_uuid_string(&valid_uuid).is_ok());

        assert!(validate_uuid_string("invalid-uuid").is_err());
        assert!(validate_uuid_string("").is_err());
        assert!(validate_uuid_string("not-a-uuid-at-all").is_err());
    }

    #[test]
    fn test_validate_chunking_strategy() {
        let mut strategy = ChunkingStrategy::default();

        // Valid strategy
        assert!(validate_chunking_strategy(&strategy).is_ok());

        // Invalid chunk size
        strategy.max_chunk_size = Some(50); // Too small
        assert!(validate_chunking_strategy(&strategy).is_err());

        strategy.max_chunk_size = Some(10000); // Too large
        assert!(validate_chunking_strategy(&strategy).is_err());

        strategy.max_chunk_size = Some(1000); // Valid
        assert!(validate_chunking_strategy(&strategy).is_ok());

        // Invalid overlap size (greater than chunk size)
        strategy.overlap_size = Some(1500);
        assert!(validate_chunking_strategy(&strategy).is_err());

        strategy.overlap_size = Some(100); // Valid
        assert!(validate_chunking_strategy(&strategy).is_ok());
    }
}