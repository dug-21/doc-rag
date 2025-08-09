//! Integration tests for MCP Adapter

use mcp_adapter::{
    McpAdapter, Message, MessageType, MessagePriority, Response, ResponseStatus, ResponseError,
    Credentials, GrantType, AuthToken, MessageQueue, QueueStats, McpError
};

#[tokio::test]
async fn test_adapter_full_workflow() {
    // Create adapter instance
    let adapter = McpAdapter::new();
    
    // Test initial state
    assert!(!adapter.is_connected());
    let queue_stats = adapter.queue_stats();
    assert_eq!(queue_stats.current_size, 0);
    
    // Test graceful shutdown
    assert!(adapter.shutdown().await.is_ok());
}

#[tokio::test]
async fn test_authentication_workflows() {
    let adapter = McpAdapter::new();
    
    // Test client credentials authentication
    let client_creds = Credentials {
        client_id: "test-client".to_string(),
        client_secret: "test-secret".to_string(),
        username: None,
        password: None,
        grant_type: GrantType::ClientCredentials,
        scope: vec!["read".to_string(), "write".to_string()],
        authorization_code: None,
        redirect_uri: None,
        code_verifier: None,
    };
    
    let token_result = adapter.authenticate(client_creds).await;
    // This will fail due to network error, but we're testing the API
    assert!(token_result.is_err());
}

#[tokio::test]
async fn test_message_queue_integration() {
    let _adapter = McpAdapter::new();
    
    // Create test messages
    let normal_message = Message::request(serde_json::json!({"query": "test normal priority"}));
    
    let high_message = Message::request(serde_json::json!({"document": "urgent processing"}))
        .with_priority(MessagePriority::High);
    
    let critical_message = Message::error("CRITICAL_ERROR", "critical system error")
        .with_priority(MessagePriority::Critical);
    
    // Test that we can create messages without errors
    assert_eq!(normal_message.priority, MessagePriority::Normal);
    assert_eq!(high_message.priority, MessagePriority::High);
    assert_eq!(critical_message.priority, MessagePriority::Critical);
    
    // Test message serialization/deserialization
    let serialized = serde_json::to_string(&normal_message).unwrap();
    let deserialized: Message = serde_json::from_str(&serialized).unwrap();
    assert_eq!(deserialized.id, normal_message.id);
    assert_eq!(deserialized.message_type, normal_message.message_type);
}

#[tokio::test]
async fn test_concurrent_message_handling() {
    let adapter = McpAdapter::new();
    
    // Create multiple messages
    let messages: Vec<Message> = (0..20)
        .map(|i| {
            Message::request(serde_json::json!({"test_id": i}))
                .with_correlation_id(uuid::Uuid::new_v4())
        })
        .collect();
    
    // Test that messages can be created properly
    assert_eq!(messages.len(), 20);
    
    // Test individual message sending (will fail without connection)
    for message in messages.into_iter().take(3) {
        let result = adapter.send_message(message).await;
        assert!(result.is_err());
        // Should be connection errors since we're not connected
        match result.unwrap_err() {
            McpError::ConnectionFailed(_) => {},
            other => panic!("Unexpected error type: {:?}", other),
        }
    }
}

#[tokio::test]
async fn test_error_handling_and_retry_logic() {
    let adapter = McpAdapter::new();
    
    // Test connection to invalid endpoint (should trigger retry logic)
    let connection_result = adapter.connect("http://invalid-endpoint-12345.com").await;
    assert!(connection_result.is_err());
    
    // Should be a connection error after retries (could be NetworkError or ConnectionFailed)
    match connection_result.unwrap_err() {
        McpError::ConnectionFailed(_) | McpError::NetworkError(_) => {},
        other => panic!("Expected connection or network error, got: {:?}", other),
    }
    
    // Test message sending without connection (should fail quickly)
    let test_message = Message::heartbeat();
    
    let message_result = adapter.send_message(test_message).await;
    assert!(message_result.is_err());
}

#[tokio::test]
async fn test_adapter_metrics_and_monitoring() {
    let adapter = McpAdapter::new();
    
    // Test initial metrics
    let queue_stats = adapter.queue_stats();
    assert_eq!(queue_stats.current_size, 0);
    assert_eq!(queue_stats.total_enqueued, 0);
    assert_eq!(queue_stats.total_dequeued, 0);
    assert_eq!(queue_stats.utilization_percent, 0);
    
    // Test connection status
    let is_connected = adapter.is_connected();
    assert!(!is_connected); // Should not be connected initially
}

#[tokio::test]
async fn test_message_expiration_and_cleanup() {
    // Test message expiration
    let mut expired_message = Message::request(serde_json::json!({}));
    
    // Set message timestamp to 10 seconds ago and TTL to 5 seconds
    expired_message.timestamp = Some(chrono::Utc::now() - chrono::Duration::seconds(10));
    expired_message.ttl_ms = Some(5000); // 5 second TTL, so it's expired
    
    // Message should be expired
    assert!(expired_message.is_expired());
    
    // Test non-expired message
    let fresh_message = Message::request(serde_json::json!({}))
        .with_ttl(300000); // 5 minutes in milliseconds
    
    assert!(!fresh_message.is_expired());
}

#[tokio::test]
async fn test_response_creation_and_handling() {
    // Test successful response creation
    let success_response = Response::success(
        123,
        serde_json::json!({"result": "operation completed successfully"}),
    );
    
    assert_eq!(success_response.status, ResponseStatus::Success);
    assert_eq!(success_response.id, 123);
    
    // Test error response creation
    let error = ResponseError::new(
        "VALIDATION_ERROR".to_string(),
        "Invalid input data".to_string(),
    );
    let error_response = Response::error(456, error);
    
    assert_eq!(error_response.status, ResponseStatus::Error);
    assert_eq!(error_response.id, 456);
    
    // Test response with processing time
    let response_with_time = success_response
        .with_processing_time(150)
        .with_correlation_id(uuid::Uuid::new_v4());
    
    assert_eq!(response_with_time.processing_time_ms, Some(150));
    assert!(response_with_time.correlation_id.is_some());
}

#[tokio::test]
async fn test_queue_statistics_and_health() {
    let queue = MessageQueue::new(10);
    
    // Test initial state
    let stats = queue.stats();
    assert_eq!(stats.current_size, 0);
    assert_eq!(stats.total_enqueued, 0);
    assert_eq!(stats.total_dequeued, 0);
    assert!(queue.is_empty());
    
    // Add some messages
    for i in 0..5 {
        let message = Message::request(serde_json::json!({"id": i}));
        queue.enqueue(message).await.unwrap();
    }
    
    let updated_stats = queue.stats();
    assert_eq!(updated_stats.current_size, 5);
    assert_eq!(updated_stats.total_enqueued, 5);
    
    // Process some messages
    for _ in 0..3 {
        let _ = queue.dequeue().await;
    }
    
    let final_stats = queue.stats();
    assert_eq!(final_stats.current_size, 2);
    assert_eq!(final_stats.total_dequeued, 3);
}

#[tokio::test]
async fn test_priority_queue_behavior() {
    let queue = MessageQueue::new(10);
    
    // Add messages in mixed order
    let low_message = Message::request(serde_json::json!({"status": "low priority update"}))
        .with_priority(MessagePriority::Low);
    
    let critical_message = Message::error("SYSTEM_FAILURE", "system failure")
        .with_priority(MessagePriority::Critical);
    
    let high_message = Message::request(serde_json::json!({"auth": "urgent auth request"}))
        .with_priority(MessagePriority::High);
    
    // Enqueue in non-priority order
    queue.enqueue(low_message).await.unwrap();
    queue.enqueue(critical_message).await.unwrap();
    queue.enqueue(high_message).await.unwrap();
    
    // Should dequeue in priority order: Critical, High, Low
    let first = queue.dequeue().await.unwrap();
    assert_eq!(first.priority, MessagePriority::Critical);
    
    let second = queue.dequeue().await.unwrap();
    assert_eq!(second.priority, MessagePriority::High);
    
    let third = queue.dequeue().await.unwrap();
    assert_eq!(third.priority, MessagePriority::Low);
}

#[tokio::test]
async fn test_adapter_shutdown_and_cleanup() {
    let adapter = McpAdapter::new();
    
    // Verify adapter initial state
    assert!(!adapter.is_connected());
    
    // Test graceful shutdown
    let shutdown_result = adapter.shutdown().await;
    assert!(shutdown_result.is_ok());
    
    // Verify state after shutdown
    assert!(!adapter.is_connected());
    let queue_stats = adapter.queue_stats();
    assert_eq!(queue_stats.current_size, 0);
}