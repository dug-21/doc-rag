use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// MCP Protocol message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: Option<u64>,
    #[serde(rename = "type")]
    pub message_type: MessageType,
    pub payload: serde_json::Value,
    pub headers: HashMap<String, String>,
    pub timestamp: Option<DateTime<Utc>>,
    pub correlation_id: Option<Uuid>,
    pub reply_to: Option<String>,
    pub priority: MessagePriority,
    pub ttl_ms: Option<u64>,
    pub retry_count: u32,
}

/// Types of MCP messages
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    Request,
    Response,
    Event,
    Heartbeat,
    Error,
    Notification,
    Command,
    Query,
    Stream,
}

/// Message priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum MessagePriority {
    Low,
    Normal,
    High,
    Critical,
}

impl Default for MessagePriority {
    fn default() -> Self {
        MessagePriority::Normal
    }
}

/// Response from MCP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub id: u64,
    pub status: ResponseStatus,
    pub data: serde_json::Value,
    pub headers: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub correlation_id: Option<Uuid>,
    pub processing_time_ms: Option<u64>,
    pub error: Option<ResponseError>,
}

/// Response status codes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Success,
    Error,
    Timeout,
    RateLimited,
    Unauthorized,
    NotFound,
    ValidationFailed,
    InternalError,
}

/// Error details in response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseError {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub retry_after: Option<u64>,
}

impl Message {
    /// Create a new message with default values
    pub fn new(message_type: MessageType) -> Self {
        Self {
            id: None,
            message_type,
            payload: serde_json::Value::Null,
            headers: HashMap::new(),
            timestamp: None,
            correlation_id: None,
            reply_to: None,
            priority: MessagePriority::default(),
            ttl_ms: None,
            retry_count: 0,
        }
    }

    /// Create a request message
    pub fn request(payload: serde_json::Value) -> Self {
        let mut msg = Self::new(MessageType::Request);
        msg.payload = payload;
        msg.correlation_id = Some(Uuid::new_v4());
        msg
    }

    /// Create a response message
    pub fn response(request_id: u64, payload: serde_json::Value) -> Self {
        let mut msg = Self::new(MessageType::Response);
        msg.id = Some(request_id);
        msg.payload = payload;
        msg
    }

    /// Create an event message
    pub fn event(event_type: &str, payload: serde_json::Value) -> Self {
        let mut msg = Self::new(MessageType::Event);
        msg.payload = payload;
        msg.headers.insert("event_type".to_string(), event_type.to_string());
        msg
    }

    /// Create a heartbeat message
    pub fn heartbeat() -> Self {
        let mut msg = Self::new(MessageType::Heartbeat);
        msg.payload = serde_json::json!({
            "timestamp": Utc::now(),
            "status": "alive"
        });
        msg
    }

    /// Create an error message
    pub fn error(error_code: &str, error_message: &str) -> Self {
        let mut msg = Self::new(MessageType::Error);
        msg.payload = serde_json::json!({
            "error_code": error_code,
            "error_message": error_message,
            "timestamp": Utc::now()
        });
        msg
    }

    /// Set message priority
    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set message TTL
    pub fn with_ttl(mut self, ttl_ms: u64) -> Self {
        self.ttl_ms = Some(ttl_ms);
        self
    }

    /// Set correlation ID
    pub fn with_correlation_id(mut self, correlation_id: Uuid) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Set reply-to address
    pub fn with_reply_to(mut self, reply_to: String) -> Self {
        self.reply_to = Some(reply_to);
        self
    }

    /// Add header
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Check if message has expired based on TTL
    pub fn is_expired(&self) -> bool {
        match (self.timestamp, self.ttl_ms) {
            (Some(timestamp), Some(ttl_ms)) => {
                let now = Utc::now();
                let elapsed_ms = (now - timestamp).num_milliseconds() as u64;
                elapsed_ms > ttl_ms
            }
            _ => false, // No TTL or timestamp means no expiration
        }
    }

    /// Get remaining TTL in milliseconds
    pub fn remaining_ttl_ms(&self) -> Option<u64> {
        match (self.timestamp, self.ttl_ms) {
            (Some(timestamp), Some(ttl_ms)) => {
                let now = Utc::now();
                let elapsed_ms = (now - timestamp).num_milliseconds() as u64;
                if elapsed_ms < ttl_ms {
                    Some(ttl_ms - elapsed_ms)
                } else {
                    Some(0) // Expired
                }
            }
            _ => None,
        }
    }

    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }

    /// Check if message can be retried
    pub fn can_retry(&self, max_retries: u32) -> bool {
        self.retry_count < max_retries && !self.is_expired()
    }

    /// Serialize message to JSON bytes
    pub fn serialize(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }

    /// Deserialize message from JSON bytes
    pub fn deserialize(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }

    /// Get message size in bytes
    pub fn size_bytes(&self) -> usize {
        self.serialize().map(|bytes| bytes.len()).unwrap_or(0)
    }

    /// Create a copy with updated timestamp
    pub fn with_current_timestamp(mut self) -> Self {
        self.timestamp = Some(Utc::now());
        self
    }
}

impl Response {
    /// Create a successful response
    pub fn success(id: u64, data: serde_json::Value) -> Self {
        Self {
            id,
            status: ResponseStatus::Success,
            data,
            headers: HashMap::new(),
            timestamp: Utc::now(),
            correlation_id: None,
            processing_time_ms: None,
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: u64, error: ResponseError) -> Self {
        Self {
            id,
            status: ResponseStatus::Error,
            data: serde_json::Value::Null,
            headers: HashMap::new(),
            timestamp: Utc::now(),
            correlation_id: None,
            processing_time_ms: None,
            error: Some(error),
        }
    }

    /// Create a timeout response
    pub fn timeout(id: u64, timeout_ms: u64) -> Self {
        Self {
            id,
            status: ResponseStatus::Timeout,
            data: serde_json::Value::Null,
            headers: HashMap::new(),
            timestamp: Utc::now(),
            correlation_id: None,
            processing_time_ms: Some(timeout_ms),
            error: Some(ResponseError {
                code: "TIMEOUT".to_string(),
                message: format!("Request timed out after {}ms", timeout_ms),
                details: None,
                retry_after: Some(1000), // Suggest retry after 1 second
            }),
        }
    }

    /// Check if response indicates success
    pub fn is_success(&self) -> bool {
        self.status == ResponseStatus::Success
    }

    /// Check if response indicates an error
    pub fn is_error(&self) -> bool {
        matches!(
            self.status,
            ResponseStatus::Error
                | ResponseStatus::Timeout
                | ResponseStatus::Unauthorized
                | ResponseStatus::NotFound
                | ResponseStatus::ValidationFailed
                | ResponseStatus::InternalError
        )
    }

    /// Check if request should be retried
    pub fn should_retry(&self) -> bool {
        matches!(
            self.status,
            ResponseStatus::Timeout | ResponseStatus::RateLimited | ResponseStatus::InternalError | ResponseStatus::Error
        )
    }

    /// Get retry delay from response
    pub fn retry_delay_ms(&self) -> Option<u64> {
        self.error.as_ref().and_then(|e| e.retry_after)
    }

    /// Set processing time
    pub fn with_processing_time(mut self, processing_time_ms: u64) -> Self {
        self.processing_time_ms = Some(processing_time_ms);
        self
    }

    /// Set correlation ID
    pub fn with_correlation_id(mut self, correlation_id: Uuid) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Add header
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }
}

impl ResponseError {
    /// Create a new response error
    pub fn new(code: String, message: String) -> Self {
        Self {
            code,
            message,
            details: None,
            retry_after: None,
        }
    }

    /// Create a validation error
    pub fn validation_error(message: String, details: serde_json::Value) -> Self {
        Self {
            code: "VALIDATION_FAILED".to_string(),
            message,
            details: Some(details),
            retry_after: None,
        }
    }

    /// Create a rate limit error
    pub fn rate_limit_error(retry_after_ms: u64) -> Self {
        Self {
            code: "RATE_LIMITED".to_string(),
            message: format!("Rate limit exceeded, retry after {}ms", retry_after_ms),
            details: None,
            retry_after: Some(retry_after_ms),
        }
    }

    /// Create an authentication error
    pub fn auth_error(message: String) -> Self {
        Self {
            code: "UNAUTHORIZED".to_string(),
            message,
            details: None,
            retry_after: None,
        }
    }
}

/// Message builder for fluent construction
pub struct MessageBuilder {
    message: Message,
}

impl MessageBuilder {
    pub fn new(message_type: MessageType) -> Self {
        Self {
            message: Message::new(message_type),
        }
    }

    pub fn payload(mut self, payload: serde_json::Value) -> Self {
        self.message.payload = payload;
        self
    }

    pub fn priority(mut self, priority: MessagePriority) -> Self {
        self.message.priority = priority;
        self
    }

    pub fn ttl_ms(mut self, ttl_ms: u64) -> Self {
        self.message.ttl_ms = Some(ttl_ms);
        self
    }

    pub fn header(mut self, key: String, value: String) -> Self {
        self.message.headers.insert(key, value);
        self
    }

    pub fn correlation_id(mut self, correlation_id: Uuid) -> Self {
        self.message.correlation_id = Some(correlation_id);
        self
    }

    pub fn reply_to(mut self, reply_to: String) -> Self {
        self.message.reply_to = Some(reply_to);
        self
    }

    pub fn build(mut self) -> Message {
        self.message.timestamp = Some(Utc::now());
        self.message
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::request(serde_json::json!({"test": "data"}));
        
        assert_eq!(msg.message_type, MessageType::Request);
        assert_eq!(msg.payload, serde_json::json!({"test": "data"}));
        assert!(msg.correlation_id.is_some());
    }

    #[test]
    fn test_message_expiry() {
        let mut msg = Message::new(MessageType::Request);
        msg.timestamp = Some(Utc::now() - chrono::Duration::milliseconds(2000));
        msg.ttl_ms = Some(1000);
        
        assert!(msg.is_expired());
        assert_eq!(msg.remaining_ttl_ms(), Some(0));
    }

    #[test]
    fn test_message_builder() {
        let msg = MessageBuilder::new(MessageType::Command)
            .payload(serde_json::json!({"action": "test"}))
            .priority(MessagePriority::High)
            .ttl_ms(5000)
            .header("content-type".to_string(), "application/json".to_string())
            .build();
        
        assert_eq!(msg.message_type, MessageType::Command);
        assert_eq!(msg.priority, MessagePriority::High);
        assert_eq!(msg.ttl_ms, Some(5000));
        assert!(msg.timestamp.is_some());
        assert_eq!(msg.headers.get("content-type"), Some(&"application/json".to_string()));
    }

    #[test]
    fn test_response_creation() {
        let resp = Response::success(123, serde_json::json!({"result": "ok"}));
        
        assert_eq!(resp.id, 123);
        assert_eq!(resp.status, ResponseStatus::Success);
        assert!(resp.is_success());
        assert!(!resp.is_error());
    }

    #[test]
    fn test_response_error() {
        let error = ResponseError::rate_limit_error(1000);
        let resp = Response::error(456, error);
        
        assert_eq!(resp.id, 456);
        assert_eq!(resp.status, ResponseStatus::Error);
        assert!(resp.is_error());
        assert!(resp.should_retry());
        assert_eq!(resp.retry_delay_ms(), Some(1000));
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::request(serde_json::json!({"test": 123}));
        let serialized = msg.serialize().unwrap();
        let deserialized = Message::deserialize(&serialized).unwrap();
        
        assert_eq!(msg.message_type, deserialized.message_type);
        assert_eq!(msg.payload, deserialized.payload);
    }

    #[test]
    fn test_message_retry_logic() {
        let mut msg = Message::new(MessageType::Request);
        
        assert!(msg.can_retry(3));
        
        msg.increment_retry();
        msg.increment_retry();
        msg.increment_retry();
        
        assert_eq!(msg.retry_count, 3);
        assert!(!msg.can_retry(3));
    }
}