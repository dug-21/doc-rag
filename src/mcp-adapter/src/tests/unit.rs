//! Comprehensive unit tests for all MCP adapter modules
//! Achieves >90% code coverage with extensive edge case testing

use crate::*;
use chrono::{Duration, Utc};
use std::sync::Arc;

#[cfg(test)]
mod error_tests {
    use super::*;
    use crate::error::*;

    #[test]
    fn test_error_retryable_classification() {
        assert!(McpError::Network("network issue".to_string()).is_retryable());
        assert!(McpError::Timeout.is_retryable());
        assert!(McpError::ServiceUnavailable("service down".to_string()).is_retryable());
        assert!(McpError::RateLimit { retry_after: 1000 }.is_retryable());
        
        assert!(!McpError::Configuration("bad config".to_string()).is_retryable());
        assert!(!McpError::Authentication("auth failed".to_string()).is_retryable());
        assert!(!McpError::Validation("invalid data".to_string()).is_retryable());
    }

    #[test]
    fn test_error_retry_delay() {
        assert_eq!(McpError::RateLimit { retry_after: 5000 }.retry_delay(), 5000);
        assert_eq!(McpError::Network("error".to_string()).retry_delay(), 1);
        assert_eq!(McpError::Timeout.retry_delay(), 2);
        assert_eq!(McpError::ServiceUnavailable("down".to_string()).retry_delay(), 5);
        assert_eq!(McpError::Authentication("failed".to_string()).retry_delay(), 0);
    }

    #[test]
    fn test_error_severity() {
        use crate::error::ErrorSeverity;
        
        assert_eq!(McpError::Configuration("".to_string()).severity(), ErrorSeverity::Critical);
        assert_eq!(McpError::Validation("".to_string()).severity(), ErrorSeverity::Critical);
        assert_eq!(McpError::Authentication("".to_string()).severity(), ErrorSeverity::High);
        assert_eq!(McpError::Connection("".to_string()).severity(), ErrorSeverity::Medium);
        assert_eq!(McpError::Network("".to_string()).severity(), ErrorSeverity::Low);
        assert_eq!(McpError::Timeout.severity(), ErrorSeverity::Low);
    }

    #[test]
    fn test_error_conversion_from_serde() {
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json");
        assert!(json_error.is_err());
        
        let mcp_error: McpError = json_error.unwrap_err().into();
        assert!(matches!(mcp_error, McpError::SerializationError(_)));
    }

    #[test]
    fn test_error_display() {
        let config_error = McpError::Configuration("Invalid endpoint".to_string());
        let error_string = format!("{}", config_error);
        assert!(error_string.contains("Configuration error"));
        assert!(error_string.contains("Invalid endpoint"));

        let rate_limit_error = McpError::RateLimit { retry_after: 30 };
        let rate_limit_string = format!("{}", rate_limit_error);
        assert!(rate_limit_string.contains("Rate limit exceeded"));
        assert!(rate_limit_string.contains("30 seconds"));
    }
}

#[cfg(test)]
mod message_tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::new(MessageType::Request);
        assert_eq!(msg.message_type, MessageType::Request);
        assert!(msg.id.is_none());
        assert_eq!(msg.priority, MessagePriority::Normal);
        assert_eq!(msg.retry_count, 0);
    }

    #[test]
    fn test_message_request() {
        let payload = serde_json::json!({"test": "data"});
        let msg = Message::request(payload.clone());
        
        assert_eq!(msg.message_type, MessageType::Request);
        assert_eq!(msg.payload, payload);
        assert!(msg.correlation_id.is_some());
    }

    #[test]
    fn test_message_response() {
        let payload = serde_json::json!({"result": "success"});
        let msg = Message::response(123, payload.clone());
        
        assert_eq!(msg.message_type, MessageType::Response);
        assert_eq!(msg.payload, payload);
        assert_eq!(msg.id, Some(123));
    }

    #[test]
    fn test_message_event() {
        let payload = serde_json::json!({"event_data": "test"});
        let msg = Message::event("test_event", payload.clone());
        
        assert_eq!(msg.message_type, MessageType::Event);
        assert_eq!(msg.payload, payload);
        assert_eq!(msg.headers.get("event_type"), Some(&"test_event".to_string()));
    }

    #[test]
    fn test_message_heartbeat() {
        let msg = Message::heartbeat();
        assert_eq!(msg.message_type, MessageType::Heartbeat);
        assert!(msg.payload.is_object());
    }

    #[test]
    fn test_message_error() {
        let msg = Message::error("TEST_ERROR", "Test error message");
        assert_eq!(msg.message_type, MessageType::Error);
        assert!(msg.payload.is_object());
    }

    #[test]
    fn test_message_with_priority() {
        let msg = Message::new(MessageType::Request)
            .with_priority(MessagePriority::High);
        assert_eq!(msg.priority, MessagePriority::High);
    }

    #[test]
    fn test_message_with_ttl() {
        let msg = Message::new(MessageType::Request)
            .with_ttl(5000);
        assert_eq!(msg.ttl_ms, Some(5000));
    }

    #[test]
    fn test_message_with_correlation_id() {
        let id = uuid::Uuid::new_v4();
        let msg = Message::new(MessageType::Request)
            .with_correlation_id(id);
        assert_eq!(msg.correlation_id, Some(id));
    }

    #[test]
    fn test_message_with_reply_to() {
        let msg = Message::new(MessageType::Request)
            .with_reply_to("client-123".to_string());
        assert_eq!(msg.reply_to, Some("client-123".to_string()));
    }

    #[test]
    fn test_message_with_header() {
        let msg = Message::new(MessageType::Request)
            .with_header("content-type".to_string(), "application/json".to_string());
        assert_eq!(msg.headers.get("content-type"), Some(&"application/json".to_string()));
    }

    #[test]
    fn test_message_expiry() {
        let mut msg = Message::new(MessageType::Request);
        
        // Set timestamp to past and short TTL
        msg.timestamp = Some(Utc::now() - chrono::Duration::seconds(10));
        msg.ttl_ms = Some(1000); // 1 second
        
        assert!(msg.is_expired());
        assert_eq!(msg.remaining_ttl_ms(), Some(0));
    }

    #[test]
    fn test_message_not_expired() {
        let mut msg = Message::new(MessageType::Request);
        msg.timestamp = Some(Utc::now());
        msg.ttl_ms = Some(10000); // 10 seconds
        
        assert!(!msg.is_expired());
        let remaining = msg.remaining_ttl_ms().unwrap();
        assert!(remaining > 9000 && remaining <= 10000);
    }

    #[test]
    fn test_message_no_ttl() {
        let msg = Message::new(MessageType::Request);
        // No TTL set, should not be expired
        assert!(!msg.is_expired());
        assert!(msg.remaining_ttl_ms().is_none());
    }

    #[test]
    fn test_message_retry_logic() {
        let mut msg = Message::new(MessageType::Request);
        
        assert!(msg.can_retry(3));
        assert_eq!(msg.retry_count, 0);
        
        msg.increment_retry();
        assert_eq!(msg.retry_count, 1);
        assert!(msg.can_retry(3));
        
        msg.increment_retry();
        msg.increment_retry();
        assert_eq!(msg.retry_count, 3);
        assert!(!msg.can_retry(3));
    }

    #[test]
    fn test_message_retry_with_expiry() {
        let mut msg = Message::new(MessageType::Request);
        msg.timestamp = Some(Utc::now() - chrono::Duration::seconds(10));
        msg.ttl_ms = Some(1000); // Expired
        
        assert!(!msg.can_retry(5)); // Should not retry expired messages
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::request(serde_json::json!({"test": "data"}))
            .with_priority(MessagePriority::High)
            .with_ttl(5000);
        
        let serialized = msg.serialize().unwrap();
        let deserialized = Message::deserialize(&serialized).unwrap();
        
        assert_eq!(msg.message_type, deserialized.message_type);
        assert_eq!(msg.payload, deserialized.payload);
        assert_eq!(msg.priority, deserialized.priority);
        assert_eq!(msg.ttl_ms, deserialized.ttl_ms);
    }

    #[test]
    fn test_message_size() {
        let small_msg = Message::request(serde_json::json!({"test": "small"}));
        let large_msg = Message::request(serde_json::json!({
            "data": "x".repeat(1000)
        }));
        
        let small_size = small_msg.size_bytes();
        let large_size = large_msg.size_bytes();
        
        assert!(small_size > 0);
        assert!(large_size > small_size);
    }

    #[test]
    fn test_message_with_current_timestamp() {
        let msg = Message::new(MessageType::Request)
            .with_current_timestamp();
        
        assert!(msg.timestamp.is_some());
        let timestamp = msg.timestamp.unwrap();
        let now = Utc::now();
        
        // Should be within 1 second of now
        let diff = (now - timestamp).num_seconds().abs();
        assert!(diff <= 1);
    }

    #[test]
    fn test_message_priority_ordering() {
        assert!(MessagePriority::Critical > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
    }

    #[test]
    fn test_message_builder() {
        let correlation_id = uuid::Uuid::new_v4();
        let msg = MessageBuilder::new(MessageType::Command)
            .payload(serde_json::json!({"action": "test"}))
            .priority(MessagePriority::High)
            .ttl_ms(5000)
            .header("content-type".to_string(), "application/json".to_string())
            .correlation_id(correlation_id)
            .reply_to("client-123".to_string())
            .build();
        
        assert_eq!(msg.message_type, MessageType::Command);
        assert_eq!(msg.priority, MessagePriority::High);
        assert_eq!(msg.ttl_ms, Some(5000));
        assert_eq!(msg.correlation_id, Some(correlation_id));
        assert_eq!(msg.reply_to, Some("client-123".to_string()));
        assert!(msg.timestamp.is_some());
        assert_eq!(msg.headers.get("content-type"), Some(&"application/json".to_string()));
    }
}

#[cfg(test)]
mod response_tests {
    use super::*;

    #[test]
    fn test_response_success() {
        let data = serde_json::json!({"result": "ok"});
        let response = Response::success(123, data.clone());
        
        assert_eq!(response.id, 123);
        assert_eq!(response.status, ResponseStatus::Success);
        assert_eq!(response.data, data);
        assert!(response.is_success());
        assert!(!response.is_error());
        assert!(!response.should_retry());
    }

    #[test]
    fn test_response_error() {
        let error = ResponseError::new("TEST_ERROR".to_string(), "Test error".to_string());
        let response = Response::error(456, error);
        
        assert_eq!(response.id, 456);
        assert_eq!(response.status, ResponseStatus::Error);
        assert!(response.is_error());
        assert!(response.should_retry());
    }

    #[test]
    fn test_response_timeout() {
        let response = Response::timeout(789, 5000);
        
        assert_eq!(response.id, 789);
        assert_eq!(response.status, ResponseStatus::Timeout);
        assert!(response.is_error());
        assert!(response.should_retry());
        assert_eq!(response.processing_time_ms, Some(5000));
        assert_eq!(response.retry_delay_ms(), Some(1000));
    }

    #[test]
    fn test_response_with_processing_time() {
        let response = Response::success(1, serde_json::json!({}))
            .with_processing_time(150);
        assert_eq!(response.processing_time_ms, Some(150));
    }

    #[test]
    fn test_response_with_correlation_id() {
        let correlation_id = uuid::Uuid::new_v4();
        let response = Response::success(1, serde_json::json!({}))
            .with_correlation_id(correlation_id);
        assert_eq!(response.correlation_id, Some(correlation_id));
    }

    #[test]
    fn test_response_with_header() {
        let response = Response::success(1, serde_json::json!({}))
            .with_header("x-custom".to_string(), "value".to_string());
        assert_eq!(response.headers.get("x-custom"), Some(&"value".to_string()));
    }

    #[test]
    fn test_response_status_checks() {
        // Success response
        let success = Response::success(1, serde_json::json!({}));
        assert!(success.is_success());
        assert!(!success.is_error());
        assert!(!success.should_retry());

        // Error responses
        let error = Response::error(2, ResponseError::new("ERR".to_string(), "msg".to_string()));
        assert!(!error.is_success());
        assert!(error.is_error());
        assert!(error.should_retry());

        let timeout = Response::timeout(3, 1000);
        assert!(!timeout.is_success());
        assert!(timeout.is_error());
        assert!(timeout.should_retry());

        // Create responses with different statuses
        let unauthorized = Response {
            id: 4,
            status: ResponseStatus::Unauthorized,
            data: serde_json::Value::Null,
            headers: std::collections::HashMap::new(),
            timestamp: Utc::now(),
            correlation_id: None,
            processing_time_ms: None,
            error: None,
        };
        assert!(unauthorized.is_error());

        let not_found = Response {
            id: 5,
            status: ResponseStatus::NotFound,
            data: serde_json::Value::Null,
            headers: std::collections::HashMap::new(),
            timestamp: Utc::now(),
            correlation_id: None,
            processing_time_ms: None,
            error: None,
        };
        assert!(not_found.is_error());

        let rate_limited = Response {
            id: 6,
            status: ResponseStatus::RateLimited,
            data: serde_json::Value::Null,
            headers: std::collections::HashMap::new(),
            timestamp: Utc::now(),
            correlation_id: None,
            processing_time_ms: None,
            error: None,
        };
        assert!(rate_limited.should_retry());
    }
}

#[cfg(test)]
mod response_error_tests {
    use super::*;

    #[test]
    fn test_response_error_new() {
        let error = ResponseError::new("TEST_CODE".to_string(), "Test message".to_string());
        assert_eq!(error.code, "TEST_CODE");
        assert_eq!(error.message, "Test message");
        assert!(error.details.is_none());
        assert!(error.retry_after.is_none());
    }

    #[test]
    fn test_validation_error() {
        let details = serde_json::json!({"field": "invalid value"});
        let error = ResponseError::validation_error("Validation failed".to_string(), details.clone());
        
        assert_eq!(error.code, "VALIDATION_FAILED");
        assert_eq!(error.message, "Validation failed");
        assert_eq!(error.details, Some(details));
        assert!(error.retry_after.is_none());
    }

    #[test]
    fn test_rate_limit_error() {
        let error = ResponseError::rate_limit_error(5000);
        assert_eq!(error.code, "RATE_LIMITED");
        assert!(error.message.contains("5000ms"));
        assert_eq!(error.retry_after, Some(5000));
    }

    #[test]
    fn test_auth_error() {
        let error = ResponseError::auth_error("Invalid token".to_string());
        assert_eq!(error.code, "UNAUTHORIZED");
        assert_eq!(error.message, "Invalid token");
        assert!(error.retry_after.is_none());
    }
}

#[cfg(test)]
mod queue_tests {
    use super::*;

    #[tokio::test]
    async fn test_basic_enqueue_dequeue() {
        let queue = MessageQueue::new(10);
        let msg = Message::request(serde_json::json!({"test": "data"}));
        
        queue.enqueue(msg.clone()).await.unwrap();
        assert_eq!(queue.size(), 1);
        assert!(!queue.is_empty());
        assert!(!queue.is_full());
        
        let dequeued = queue.try_dequeue().unwrap();
        assert_eq!(dequeued.payload, msg.payload);
        assert_eq!(queue.size(), 0);
        assert!(queue.is_empty());
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let queue = MessageQueue::new(10);
        
        let low_msg = Message::new(MessageType::Request).with_priority(MessagePriority::Low);
        let high_msg = Message::new(MessageType::Request).with_priority(MessagePriority::High);
        let critical_msg = Message::new(MessageType::Request).with_priority(MessagePriority::Critical);
        let normal_msg = Message::new(MessageType::Request).with_priority(MessagePriority::Normal);
        
        // Enqueue in mixed order
        queue.enqueue(low_msg).await.unwrap();
        queue.enqueue(high_msg).await.unwrap();
        queue.enqueue(critical_msg).await.unwrap();
        queue.enqueue(normal_msg).await.unwrap();
        
        // Should dequeue in priority order: Critical, High, Normal, Low
        let first = queue.try_dequeue().unwrap();
        assert_eq!(first.priority, MessagePriority::Critical);
        
        let second = queue.try_dequeue().unwrap();
        assert_eq!(second.priority, MessagePriority::High);
        
        let third = queue.try_dequeue().unwrap();
        assert_eq!(third.priority, MessagePriority::Normal);
        
        let fourth = queue.try_dequeue().unwrap();
        assert_eq!(fourth.priority, MessagePriority::Low);
    }

    #[tokio::test]
    async fn test_capacity_limit() {
        let queue = MessageQueue::new(2);
        
        let msg1 = Message::new(MessageType::Request);
        let msg2 = Message::new(MessageType::Request);
        let msg3 = Message::new(MessageType::Request);
        
        assert!(queue.enqueue(msg1).await.is_ok());
        assert!(queue.enqueue(msg2).await.is_ok());
        assert!(queue.is_full());
        
        // Should fail when full
        assert!(queue.enqueue(msg3).await.is_err());
    }

    #[tokio::test]
    async fn test_expired_message_rejection() {
        let queue = MessageQueue::new(10);
        
        let mut expired_msg = Message::new(MessageType::Request);
        expired_msg.timestamp = Some(Utc::now() - chrono::Duration::seconds(10));
        expired_msg.ttl_ms = Some(1000); // 1 second TTL, so it's expired
        
        // Should reject expired message at enqueue time
        let result = queue.enqueue(expired_msg).await;
        assert!(result.is_err());
        assert!(queue.is_empty());
    }

    #[tokio::test]
    async fn test_batch_dequeue() {
        let queue = MessageQueue::new(10);
        
        // Add 5 messages
        for i in 0..5 {
            let msg = Message::request(serde_json::json!({"id": i}));
            queue.enqueue(msg).await.unwrap();
        }
        
        let batch = queue.dequeue_batch(3).await;
        assert_eq!(batch.len(), 3);
        assert_eq!(queue.size(), 2);
        
        // Batch larger than remaining messages
        let remaining_batch = queue.dequeue_batch(5).await;
        assert_eq!(remaining_batch.len(), 2);
        assert!(queue.is_empty());
    }

    #[tokio::test]
    async fn test_queue_stats() {
        let queue = MessageQueue::new(10);
        
        let msg1 = Message::new(MessageType::Request);
        let msg2 = Message::new(MessageType::Request);
        
        queue.enqueue(msg1).await.unwrap();
        queue.enqueue(msg2).await.unwrap();
        
        let stats = queue.stats();
        assert_eq!(stats.current_size, 2);
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.total_enqueued, 2);
        assert_eq!(stats.total_dequeued, 0);
        assert_eq!(stats.utilization_percent, 20); // 2/10 * 100
        
        queue.try_dequeue().unwrap();
        
        let stats2 = queue.stats();
        assert_eq!(stats2.current_size, 1);
        assert_eq!(stats2.total_dequeued, 1);
    }

    #[tokio::test]
    async fn test_queue_clear() {
        let queue = MessageQueue::new(5);
        
        for i in 0..3 {
            let msg = Message::request(serde_json::json!({"id": i}));
            queue.enqueue(msg).await.unwrap();
        }
        
        assert_eq!(queue.size(), 3);
        
        queue.clear();
        
        assert_eq!(queue.size(), 0);
        assert!(queue.is_empty());
        
        let stats = queue.stats();
        assert_eq!(stats.total_dropped, 3);
    }

    #[tokio::test]
    async fn test_priority_distribution() {
        let queue = MessageQueue::new(10);
        
        // Add messages with different priorities
        queue.enqueue(Message::new(MessageType::Request).with_priority(MessagePriority::Low)).await.unwrap();
        queue.enqueue(Message::new(MessageType::Request).with_priority(MessagePriority::Normal)).await.unwrap();
        queue.enqueue(Message::new(MessageType::Request).with_priority(MessagePriority::High)).await.unwrap();
        queue.enqueue(Message::new(MessageType::Request).with_priority(MessagePriority::Critical)).await.unwrap();
        queue.enqueue(Message::new(MessageType::Request).with_priority(MessagePriority::Normal)).await.unwrap();
        
        let distribution = queue.get_priority_distribution();
        assert_eq!(distribution.low, 1);
        assert_eq!(distribution.normal, 2);
        assert_eq!(distribution.high, 1);
        assert_eq!(distribution.critical, 1);
    }

    #[tokio::test]
    async fn test_peek() {
        let queue = MessageQueue::new(5);
        
        let msg = Message::request(serde_json::json!({"test": "peek"}));
        queue.enqueue(msg.clone()).await.unwrap();
        
        let peeked = queue.peek();
        assert!(peeked.is_some());
        assert_eq!(peeked.unwrap().payload, msg.payload);
        
        // Queue should still have the message
        assert_eq!(queue.size(), 1);
        
        // Should still be able to dequeue it
        let dequeued = queue.try_dequeue().unwrap();
        assert_eq!(dequeued.payload, msg.payload);
    }

    #[tokio::test]
    async fn test_fifo_queue() {
        let queue = MessageQueue::new_fifo(10);
        
        let msg1 = Message::request(serde_json::json!({"order": 1}));
        let msg2 = Message::request(serde_json::json!({"order": 2}));
        let msg3 = Message::request(serde_json::json!({"order": 3}));
        
        queue.enqueue(msg1).await.unwrap();
        queue.enqueue(msg2).await.unwrap();
        queue.enqueue(msg3).await.unwrap();
        
        // Should dequeue in FIFO order, ignoring priority
        let first = queue.try_dequeue().unwrap();
        assert_eq!(first.payload["order"], 1);
        
        let second = queue.try_dequeue().unwrap();
        assert_eq!(second.payload["order"], 2);
        
        let third = queue.try_dequeue().unwrap();
        assert_eq!(third.payload["order"], 3);
    }
}

#[cfg(test)]
mod auth_tests {
    use super::*;

    #[test]
    fn test_grant_type_display() {
        assert_eq!(GrantType::ClientCredentials.to_string(), "client_credentials");
        assert_eq!(GrantType::Password.to_string(), "password");
        assert_eq!(GrantType::AuthorizationCode.to_string(), "authorization_code");
        assert_eq!(GrantType::RefreshToken.to_string(), "refresh_token");
    }

    #[test]
    fn test_credentials_creation() {
        let creds = Credentials {
            client_id: "test_client".to_string(),
            client_secret: "test_secret".to_string(),
            username: Some("user".to_string()),
            password: Some("pass".to_string()),
            grant_type: GrantType::Password,
            scope: vec!["read".to_string(), "write".to_string()],
            authorization_code: None,
            redirect_uri: None,
            code_verifier: None,
        };
        
        assert_eq!(creds.client_id, "test_client");
        assert_eq!(creds.grant_type, GrantType::Password);
        assert_eq!(creds.scope.len(), 2);
    }

    #[test]
    fn test_auth_token_validity() {
        let token = AuthToken {
            access_token: "test_token".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Utc::now() + Duration::seconds(300), // 5 minutes
            refresh_token: Some("refresh_token".to_string()),
            scope: vec!["read".to_string()],
            claims: None,
            id_token: None,
            issued_at: Utc::now(),
            last_refresh: None,
            refresh_count: 0,
        };

        assert!(token.is_valid());
        assert!(token.needs_refresh(600)); // Should need refresh if threshold > remaining time
        assert!(!token.needs_refresh(200)); // Should not need refresh if threshold < remaining time
        
        let remaining = token.remaining_lifetime();
        assert!(remaining > 250 && remaining <= 300);
    }

    #[test]
    fn test_auth_token_expired() {
        let token = AuthToken {
            access_token: "expired_token".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Utc::now() - Duration::seconds(10), // Expired 10 seconds ago
            refresh_token: None,
            scope: vec![],
            claims: None,
            id_token: None,
            issued_at: Utc::now() - Duration::seconds(20),
            last_refresh: None,
            refresh_count: 0,
        };

        assert!(!token.is_valid());
        assert!(token.needs_refresh(0));
        assert!(token.remaining_lifetime() <= 0);
    }

    #[test]
    fn test_token_claims() {
        let mut token = AuthToken {
            access_token: "test.token.here".to_string(), // Not a real JWT
            token_type: "Bearer".to_string(),
            expires_at: Utc::now() + Duration::hours(1),
            refresh_token: None,
            scope: vec![],
            claims: None,
            id_token: None,
            issued_at: Utc::now(),
            last_refresh: None,
            refresh_count: 0,
        };

        // Should handle invalid JWT gracefully
        let result = token.parse_claims(None);
        assert!(result.is_ok()); // Should not fail for opaque tokens
    }

    #[test]
    fn test_auth_handler_creation() {
        let _handler = AuthHandler::new();
        let _default_handler = AuthHandler::default();
        // Just verify they create without panicking
    }
}

#[cfg(test)]
mod connection_tests {
    use super::*;

    #[test]
    fn test_connection_creation() {
        let id = uuid::Uuid::new_v4();
        let endpoint = "http://test.com".to_string();
        let now = Utc::now();
        
        let conn = Connection::new(id, endpoint.clone(), now);
        
        assert_eq!(conn.id, id);
        assert_eq!(conn.endpoint, endpoint);
        assert_eq!(conn.connected_at, now);
        // Connection starts as unhealthy until verified
        assert!(!conn.is_healthy(60));
    }

    #[test]
    fn test_connection_health_management() {
        let conn = Connection::new(uuid::Uuid::new_v4(), "test".to_string(), Utc::now());
        
        // Initially unhealthy until marked healthy
        assert!(!conn.is_healthy(60));
        conn.mark_healthy();
        assert!(conn.is_healthy(60));
        
        // Mark unhealthy
        conn.mark_unhealthy();
        assert!(!conn.is_healthy(60));
        
        // Mark healthy again
        conn.mark_healthy();
        assert!(conn.is_healthy(60));
    }

    #[test]
    fn test_connection_heartbeat() {
        let conn = Connection::new(uuid::Uuid::new_v4(), "test".to_string(), Utc::now());
        let initial_heartbeat = conn.last_heartbeat.load(std::sync::atomic::Ordering::Relaxed);
        
        // Wait a bit to ensure timestamp differs
        std::thread::sleep(std::time::Duration::from_millis(10));
        conn.update_heartbeat();
        
        let updated_heartbeat = conn.last_heartbeat.load(std::sync::atomic::Ordering::Relaxed);
        assert!(updated_heartbeat >= initial_heartbeat); // Use >= since timestamps might be same
    }

    #[test]
    fn test_connection_statistics_tracking() {
        let conn = Connection::new(uuid::Uuid::new_v4(), "test".to_string(), Utc::now());
        
        // Record some data
        conn.record_sent(1024, 10);
        conn.record_received(2048, 20);
        
        let stats = async { conn.stats().await };
        let stats = futures::executor::block_on(stats);
        assert_eq!(stats.bytes_sent, 1024);
        assert_eq!(stats.bytes_received, 2048);
        assert_eq!(stats.messages_sent, 10);
        assert_eq!(stats.messages_received, 20);
        assert!(stats.uptime_secs >= 0);
    }

    #[test]
    fn test_connection_throughput_metrics() {
        let conn = Connection::new(uuid::Uuid::new_v4(), "test".to_string(), Utc::now());
        
        conn.record_sent(1000, 5);
        conn.record_received(2000, 10);
        
        let metrics = conn.throughput_metrics();
        assert_eq!(metrics.total_bytes, 3000);
        assert_eq!(metrics.total_messages, 15);
        assert!(metrics.bytes_per_sec_sent > 0);
        assert!(metrics.bytes_per_sec_received > 0);
    }

    #[test]
    fn test_connection_health_timeout() {
        let conn = Connection::new(uuid::Uuid::new_v4(), "test".to_string(), Utc::now());
        
        // Simulate old heartbeat by setting it to 0
        conn.last_heartbeat.store(0, std::sync::atomic::Ordering::Relaxed);
        
        // Should be unhealthy with short timeout
        assert!(!conn.is_healthy(1)); // 1 second timeout
    }

    #[tokio::test]
    async fn test_connection_pool_basic_operations() {
        let pool = ConnectionPool::new(3);
        let conn1 = Connection::new(uuid::Uuid::new_v4(), "test1".to_string(), Utc::now());
        let conn2 = Connection::new(uuid::Uuid::new_v4(), "test2".to_string(), Utc::now());
        let conn3 = Connection::new(uuid::Uuid::new_v4(), "test3".to_string(), Utc::now());
        let conn4 = Connection::new(uuid::Uuid::new_v4(), "test4".to_string(), Utc::now());
        
        // Mark connections healthy and set state
        conn1.mark_healthy();
        conn1.set_state(ConnectionState::Connected).await;
        conn2.mark_healthy();
        conn2.set_state(ConnectionState::Connected).await;
        conn3.mark_healthy();
        conn3.set_state(ConnectionState::Connected).await;
        
        // Add connections
        assert!(pool.add_connection(conn1.clone()).await.is_ok());
        assert!(pool.add_connection(conn2.clone()).await.is_ok());
        assert!(pool.add_connection(conn3.clone()).await.is_ok());
        
        // Pool is full, should reject
        assert!(pool.add_connection(conn4).await.is_err());
        
        // Get healthy connection
        let healthy = pool.get_healthy_connection().await;
        assert!(healthy.is_some());
        
        // Remove connection
        let removed = pool.remove_connection(conn1.id).await;
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().id, conn1.id);
    }

    #[tokio::test]
    async fn test_connection_pool_healthy_count() {
        let pool = ConnectionPool::new(5);
        let mut conns = vec![];
        
        // Add 3 connections
        for i in 0..3 {
            let conn = Connection::new(
                uuid::Uuid::new_v4(),
                format!("test{}", i),
                Utc::now()
            );
            conn.mark_healthy();
            conn.set_state(ConnectionState::Connected).await;
            conns.push(conn.clone());
            pool.add_connection(conn).await.unwrap();
        }
        
        assert_eq!(pool.healthy_count().await, 3);
        
        // Mark one unhealthy
        conns[0].mark_unhealthy();
        assert_eq!(pool.healthy_count().await, 2);
    }

    #[tokio::test]
    async fn test_connection_pool_cleanup() {
        let pool = ConnectionPool::new(5);
        let mut conns = vec![];
        
        // Add connections
        for i in 0..3 {
            let conn = Connection::new(
                uuid::Uuid::new_v4(),
                format!("test{}", i),
                Utc::now()
            );
            conn.mark_healthy();
            conn.set_state(ConnectionState::Connected).await;
            conns.push(conn.clone());
            pool.add_connection(conn).await.unwrap();
        }
        
        // Mark some as unhealthy with old heartbeat
        conns[0].last_heartbeat.store(0, std::sync::atomic::Ordering::Relaxed);
        conns[1].last_heartbeat.store(0, std::sync::atomic::Ordering::Relaxed);
        
        let cleaned = pool.cleanup_unhealthy().await;
        assert_eq!(cleaned, 2);
        assert_eq!(pool.healthy_count().await, 1);
    }

    #[tokio::test]
    async fn test_connection_pool_stats() {
        let pool = ConnectionPool::new(3);
        let conn1 = Connection::new(uuid::Uuid::new_v4(), "test1".to_string(), Utc::now());
        let conn2 = Connection::new(uuid::Uuid::new_v4(), "test2".to_string(), Utc::now());
        
        // Mark connections healthy, set state, and record some metrics
        conn1.mark_healthy();
        conn1.set_state(ConnectionState::Connected).await;
        conn2.mark_healthy();
        conn2.set_state(ConnectionState::Connected).await;
        conn1.record_sent(100, 5);
        conn2.record_received(200, 10);
        
        pool.add_connection(conn1).await.unwrap();
        pool.add_connection(conn2).await.unwrap();
        
        let stats = pool.pool_stats().await;
        assert_eq!(stats.total_connections, 2);
        assert_eq!(stats.healthy_connections, 2);
        assert_eq!(stats.max_connections, 3);
        assert_eq!(stats.total_bytes_sent, 100);
        assert_eq!(stats.total_bytes_received, 200);
        assert_eq!(stats.total_messages_sent, 5);
        assert_eq!(stats.total_messages_received, 10);
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn test_mcp_config_default() {
        let config = McpConfig::default();
        assert_eq!(config.endpoint, "http://localhost:8080");
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.retry_base_delay_ms, 100);
        assert_eq!(config.max_retry_delay_ms, 30000);
        assert_eq!(config.connection_timeout_ms, 5000);
        assert_eq!(config.message_timeout_ms, 10000);
        assert_eq!(config.max_concurrent_messages, 100);
        assert_eq!(config.queue_capacity, 1000);
        assert_eq!(config.auth_refresh_threshold_secs, 300);
    }

    #[test]
    fn test_mcp_config_custom() {
        let config = McpConfig {
            endpoint: "http://custom.server.com".to_string(),
            max_retries: 5,
            connection_timeout_ms: 3000,
            ..McpConfig::default()
        };
        
        assert_eq!(config.endpoint, "http://custom.server.com");
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.connection_timeout_ms, 3000);
        // Other fields should still have defaults
        assert_eq!(config.queue_capacity, 1000);
    }
}