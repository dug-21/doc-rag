// Input validation and sanitization module
// Implements OWASP guidelines for input validation

use crate::security::{SecurityError, SecurityResult, ValidationConfig};
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use validator::{Validate, ValidationError};

pub struct InputValidator {
    config: ValidationConfig,
    patterns: HashMap<String, Regex>,
}

impl InputValidator {
    pub fn new(config: ValidationConfig) -> SecurityResult<Self> {
        let mut patterns = HashMap::new();
        
        // SQL injection patterns
        patterns.insert(
            "sql_injection".to_string(),
            Regex::new(r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute|\bor\b|\band\b|--|/\*|\*/|;|'|\"|<|>)").unwrap()
        );
        
        // XSS patterns
        patterns.insert(
            "xss".to_string(),
            Regex::new(r"(?i)(<script|</script>|javascript:|on\w+\s*=|<iframe|</iframe>|<object|</object>)").unwrap()
        );
        
        // Path traversal patterns
        patterns.insert(
            "path_traversal".to_string(),
            Regex::new(r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)").unwrap()
        );
        
        // Command injection patterns
        patterns.insert(
            "command_injection".to_string(),
            Regex::new(r"[;&|`$\(\)]").unwrap()
        );

        Ok(Self { config, patterns })
    }

    pub fn validate_input(&self, input: &str, field_name: &str) -> SecurityResult<String> {
        // Check for malicious patterns
        for (pattern_name, regex) in &self.patterns {
            if regex.is_match(input) {
                return Err(SecurityError::ValidationFailed(
                    format!("Field '{}' contains potentially malicious content ({})", field_name, pattern_name)
                ));
            }
        }

        // Sanitize if enabled
        if self.config.sanitize_inputs {
            Ok(self.sanitize_string(input))
        } else {
            Ok(input.to_string())
        }
    }

    pub fn validate_json(&self, json: &Value) -> SecurityResult<Value> {
        match json {
            Value::String(s) => {
                let validated = self.validate_input(s, "json_string")?;
                Ok(Value::String(validated))
            }
            Value::Object(map) => {
                let mut result = serde_json::Map::new();
                for (key, value) in map {
                    let validated_key = self.validate_input(key, "json_key")?;
                    let validated_value = self.validate_json(value)?;
                    result.insert(validated_key, validated_value);
                }
                Ok(Value::Object(result))
            }
            Value::Array(arr) => {
                let mut result = Vec::new();
                for (index, item) in arr.iter().enumerate() {
                    let validated_item = self.validate_json(item)?;
                    result.push(validated_item);
                }
                Ok(Value::Array(result))
            }
            _ => Ok(json.clone()),
        }
    }

    pub fn validate_file_upload(&self, content_type: &str, size: usize) -> SecurityResult<()> {
        // Check size limits
        if size > self.config.max_request_size {
            return Err(SecurityError::ValidationFailed(
                format!("File size {} exceeds maximum allowed size {}", size, self.config.max_request_size)
            ));
        }

        // Check content type
        if !self.config.allowed_content_types.contains(&content_type.to_lowercase()) {
            return Err(SecurityError::ValidationFailed(
                format!("Content type '{}' is not allowed", content_type)
            ));
        }

        Ok(())
    }

    pub fn validate_email(&self, email: &str) -> SecurityResult<String> {
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
        if !email_regex.is_match(email) {
            return Err(SecurityError::ValidationFailed("Invalid email format".to_string()));
        }
        
        self.validate_input(email, "email")
    }

    pub fn validate_url(&self, url: &str) -> SecurityResult<String> {
        // Basic URL validation
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(SecurityError::ValidationFailed("URL must use HTTP or HTTPS protocol".to_string()));
        }

        // Check for suspicious patterns
        self.validate_input(url, "url")
    }

    fn sanitize_string(&self, input: &str) -> String {
        input
            .chars()
            .map(|c| match c {
                '<' => "&lt;".to_string(),
                '>' => "&gt;".to_string(),
                '"' => "&quot;".to_string(),
                '\'' => "&#x27;".to_string(),
                '&' => "&amp;".to_string(),
                _ => c.to_string(),
            })
            .collect()
    }
}

#[derive(Debug, Validate)]
pub struct QueryRequest {
    #[validate(length(min = 1, max = 1000))]
    pub query: String,
    
    #[validate(range(min = 1, max = 100))]
    pub limit: Option<u32>,
    
    #[validate(range(min = 0))]
    pub offset: Option<u32>,
    
    #[validate(length(max = 100))]
    pub filters: Option<Vec<String>>,
}

#[derive(Debug, Validate)]
pub struct DocumentUpload {
    #[validate(length(min = 1, max = 255))]
    pub filename: String,
    
    #[validate(length(min = 1, max = 100))]
    pub content_type: String,
    
    #[validate(range(min = 1, max = 104857600))] // 100MB max
    pub size: u64,
    
    #[validate(length(max = 500))]
    pub description: Option<String>,
}

pub fn validate_request_size(size: usize, max_size: usize) -> SecurityResult<()> {
    if size > max_size {
        return Err(SecurityError::ValidationFailed(
            format!("Request size {} exceeds maximum allowed size {}", size, max_size)
        ));
    }
    Ok(())
}

pub fn validate_content_type(content_type: &str, allowed_types: &[String]) -> SecurityResult<()> {
    let normalized = content_type.to_lowercase();
    if !allowed_types.iter().any(|t| normalized.starts_with(&t.to_lowercase())) {
        return Err(SecurityError::ValidationFailed(
            format!("Content type '{}' is not allowed", content_type)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sql_injection_detection() {
        let config = ValidationConfig::default();
        let validator = InputValidator::new(config).unwrap();
        
        assert!(validator.validate_input("'; DROP TABLE users; --", "test").is_err());
        assert!(validator.validate_input("normal text", "test").is_ok());
    }

    #[test]
    fn test_xss_detection() {
        let config = ValidationConfig::default();
        let validator = InputValidator::new(config).unwrap();
        
        assert!(validator.validate_input("<script>alert('xss')</script>", "test").is_err());
        assert!(validator.validate_input("normal text", "test").is_ok());
    }

    #[test]
    fn test_path_traversal_detection() {
        let config = ValidationConfig::default();
        let validator = InputValidator::new(config).unwrap();
        
        assert!(validator.validate_input("../../../etc/passwd", "test").is_err());
        assert!(validator.validate_input("normal/path", "test").is_ok());
    }

    #[test]
    fn test_email_validation() {
        let config = ValidationConfig::default();
        let validator = InputValidator::new(config).unwrap();
        
        assert!(validator.validate_email("test@example.com").is_ok());
        assert!(validator.validate_email("invalid-email").is_err());
    }

    #[test]
    fn test_sanitization() {
        let config = ValidationConfig {
            sanitize_inputs: true,
            ..Default::default()
        };
        let validator = InputValidator::new(config).unwrap();
        
        let result = validator.sanitize_string("<script>alert('test')</script>");
        assert_eq!(result, "&lt;script&gt;alert(&#x27;test&#x27;)&lt;/script&gt;");
    }
}