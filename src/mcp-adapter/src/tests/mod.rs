pub mod unit;

// Integration test helpers
use crate::*;
use mockito::Server;
use std::sync::Arc;
use tokio::sync::OnceCell;

static TEST_SERVER: OnceCell<Arc<Server>> = OnceCell::const_new();

/// Get or create a shared mock server for tests
pub async fn get_test_server() -> Arc<Server> {
    TEST_SERVER
        .get_or_init(|| async {
            let server = Server::new_async().await;
            Arc::new(server)
        })
        .await
        .clone()
}

/// Create test credentials for authentication
pub fn test_credentials() -> Credentials {
    Credentials {
        client_id: "test_client".to_string(),
        client_secret: "test_secret".to_string(),
        username: None,
        password: None,
        grant_type: GrantType::ClientCredentials,
        scope: vec!["read".to_string(), "write".to_string()],
    }
}

/// Create a test message with specified size
pub fn create_test_message(size_bytes: usize) -> Message {
    let payload = serde_json::json!({
        "data": "x".repeat(size_bytes),
        "test": true
    });
    Message::request(payload)
}

/// Create a large message for performance testing
pub fn create_large_message() -> Message {
    let payload = serde_json::json!({
        "data": "x".repeat(10240), // 10KB
        "metadata": {
            "version": "1.0",
            "timestamp": chrono::Utc::now(),
            "tags": (0..100).map(|i| format!("tag{}", i)).collect::<Vec<_>>()
        }
    });
    Message::request(payload)
}

/// Setup a test system with mock server
pub async fn setup_test_system() -> McpAdapter {
    let server = get_test_server().await;
    
    // Setup default mocks
    let _health_mock = server
        .mock("GET", "/health")
        .with_status(200)
        .create_async()
        .await;

    let config = McpConfig {
        endpoint: server.url(),
        ..McpConfig::default()
    };
    
    let adapter = McpAdapter::with_config(config);
    
    // Connect to mock server
    adapter.connect(&server.url()).await.unwrap();
    
    adapter
}