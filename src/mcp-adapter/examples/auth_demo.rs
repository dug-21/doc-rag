//! Demonstration of OAuth2/JWT authentication implementation

use mcp_adapter::{
    auth::{AuthHandler, Credentials, GrantType},
    connection::{Connection, ConnectionPool, ConnectionConfig},
    error::Result,
};
use chrono::{Duration, Utc};
use reqwest::Client;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🚀 MCP Adapter Authentication Demo");
    println!("===================================");

    // 1. Test token validation and refresh logic
    demo_token_operations().await?;

    // 2. Test connection management
    demo_connection_management().await?;

    // 3. Test authentication handler
    demo_auth_handler().await?;

    println!("\n✅ All demos completed successfully!");
    Ok(())
}

async fn demo_token_operations() -> Result<()> {
    println!("\n📱 Token Operations Demo");
    println!("------------------------");

    // Create a mock token that expires soon
    let token = mcp_adapter::auth::AuthToken {
        access_token: "test_access_token".to_string(),
        token_type: "Bearer".to_string(),
        expires_at: Utc::now() + Duration::seconds(100), // Expires in 100 seconds
        refresh_token: Some("test_refresh_token".to_string()),
        scope: vec!["read".to_string(), "write".to_string()],
        claims: None,
        id_token: None,
        issued_at: Utc::now(),
        last_refresh: None,
        refresh_count: 0,
    };

    println!("Token is valid: {}", token.is_valid());
    println!("Token needs refresh (threshold 200s): {}", token.needs_refresh(200));
    println!("Token needs refresh (threshold 50s): {}", token.needs_refresh(50));
    println!("Token can be refreshed: {}", token.can_refresh());
    println!("Remaining lifetime: {} seconds", token.remaining_lifetime());

    // Test claims validation
    match token.validate_claims(Some("test-audience"), Some("test-issuer")) {
        Ok(valid) => println!("Claims validation (no claims present): {}", valid),
        Err(e) => println!("Claims validation error: {}", e),
    }

    Ok(())
}

async fn demo_connection_management() -> Result<()> {
    println!("\n🔗 Connection Management Demo");
    println!("-----------------------------");

    // Create a connection
    let connection = Connection::new(
        Uuid::new_v4(),
        "http://localhost:8080".to_string(),
        Utc::now(),
    );

    println!("Connection ID: {}", connection.id);
    println!("Connection endpoint: {}", connection.endpoint);
    println!("Initial health status: {}", connection.is_healthy(60));

    // Mark as healthy and test
    connection.mark_healthy();
    println!("After marking healthy: {}", connection.is_healthy(60));

    // Record some metrics
    connection.record_sent(1024, 5);
    connection.record_received(2048, 10);
    connection.update_latency(150);

    let stats = connection.stats().await;
    println!("Connection stats:");
    println!("  - Bytes sent: {}", stats.bytes_sent);
    println!("  - Bytes received: {}", stats.bytes_received);
    println!("  - Messages sent: {}", stats.messages_sent);
    println!("  - Messages received: {}", stats.messages_received);
    println!("  - Latency: {}ms", stats.latency_ms);

    let throughput = connection.throughput_metrics();
    println!("Throughput metrics:");
    println!("  - Total bytes: {}", throughput.total_bytes);
    println!("  - Total messages: {}", throughput.total_messages);

    // Test connection pool
    let config = ConnectionConfig::default();
    let pool = ConnectionPool::with_config(5, config);

    pool.add_connection(connection).await?;
    let pool_stats = pool.pool_stats().await;
    println!("Pool stats: {} total connections", pool_stats.total_connections);

    Ok(())
}

async fn demo_auth_handler() -> Result<()> {
    println!("\n🔐 Authentication Handler Demo");
    println!("-------------------------------");

    let auth_handler = AuthHandler::new();
    println!("Created AuthHandler with client ID: (generated)");

    // Test PKCE challenge generation
    let pkce = mcp_adapter::auth::PkceChallenge::new();
    println!("Generated PKCE challenge:");
    println!("  - Code verifier length: {}", pkce.code_verifier.len());
    println!("  - Code challenge length: {}", pkce.code_challenge.len());
    println!("  - Challenge method: {}", pkce.code_challenge_method);

    // Test different grant types
    let grant_types = vec![
        GrantType::ClientCredentials,
        GrantType::Password,
        GrantType::AuthorizationCode,
        GrantType::RefreshToken,
        GrantType::DeviceCode,
        GrantType::JwtBearer,
    ];

    println!("Grant type string representations:");
    for grant_type in grant_types {
        println!("  - {:?} -> {}", grant_type, grant_type.to_string());
    }

    // Test credentials creation
    let credentials = Credentials {
        client_id: "test_client".to_string(),
        client_secret: "test_secret".to_string(),
        username: Some("test_user".to_string()),
        password: Some("test_pass".to_string()),
        grant_type: GrantType::ClientCredentials,
        scope: vec!["read".to_string(), "write".to_string()],
        authorization_code: None,
        redirect_uri: None,
        code_verifier: None,
    };

    println!("Created credentials for client: {}", credentials.client_id);
    println!("Grant type: {}", credentials.grant_type);
    println!("Scopes: {:?}", credentials.scope);

    Ok(())
}