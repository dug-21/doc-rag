use axum::{
    extract::State,
    response::Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn, error};
use uuid::Uuid;

use crate::{
    config::ApiConfig,
    middleware::auth::{generate_jwt_token, generate_refresh_token, AuthContext, AuthExtension},
    models::{LoginRequest, AuthResponse, UserInfo, UserRole},
    validation::validate_login_request,
    Result, ApiError,
};

#[derive(Debug, Deserialize)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

#[derive(Debug, Serialize)]
pub struct LogoutResponse {
    pub message: String,
    pub logged_out_at: chrono::DateTime<chrono::Utc>,
}

/// Handle user login
pub async fn login(
    State(config): State<Arc<ApiConfig>>,
    Json(request): Json<LoginRequest>,
) -> Result<Json<AuthResponse>> {
    info!("Login attempt for email: {}", request.email);

    // Validate the login request
    validate_login_request(&request)?;

    // In a real implementation, you would:
    // 1. Hash the password and compare with database
    // 2. Check user account status (active, locked, etc.)
    // 3. Implement rate limiting for failed attempts
    // 4. Log security events

    // For demo purposes, we'll simulate a successful login
    // In production, implement proper password verification
    let user = authenticate_user(&request.email, &request.password).await?;

    // Generate JWT token
    let access_token = generate_jwt_token(
        user.id,
        user.email.clone(),
        user.role.to_string(),
        &config.security.jwt_secret,
        config.security.jwt_expiration_hours,
    )?;

    // Generate refresh token
    let refresh_token = generate_refresh_token();

    // Store refresh token in database/cache (not implemented here)
    store_refresh_token(user.id, &refresh_token).await?;

    info!("Login successful for user: {}", user.email);

    Ok(Json(AuthResponse {
        access_token,
        refresh_token,
        expires_in: config.security.jwt_expiration_hours * 3600, // Convert to seconds
        user,
    }))
}

/// Handle token refresh
pub async fn refresh_token(
    State(config): State<Arc<ApiConfig>>,
    Json(request): Json<RefreshTokenRequest>,
) -> Result<Json<AuthResponse>> {
    info!("Token refresh attempt");

    // Validate refresh token
    let user = validate_refresh_token(&request.refresh_token).await?;

    // Generate new tokens
    let access_token = generate_jwt_token(
        user.id,
        user.email.clone(),
        user.role.to_string(),
        &config.security.jwt_secret,
        config.security.jwt_expiration_hours,
    )?;

    let new_refresh_token = generate_refresh_token();

    // Update refresh token in storage
    store_refresh_token(user.id, &new_refresh_token).await?;
    invalidate_refresh_token(&request.refresh_token).await?;

    info!("Token refresh successful for user: {}", user.email);

    Ok(Json(AuthResponse {
        access_token,
        refresh_token: new_refresh_token,
        expires_in: config.security.jwt_expiration_hours * 3600,
        user,
    }))
}

/// Handle user logout
pub async fn logout(
    request: axum::extract::Request,
) -> Result<Json<LogoutResponse>> {
    let auth_context = request.require_auth_context()?;
    
    info!("Logout request for user: {}", auth_context.email);

    // Invalidate the current token (add to blacklist)
    invalidate_access_token(&auth_context.token_id).await?;

    // Invalidate all refresh tokens for this user
    invalidate_user_refresh_tokens(auth_context.user_id).await?;

    info!("Logout successful for user: {}", auth_context.email);

    Ok(Json(LogoutResponse {
        message: "Successfully logged out".to_string(),
        logged_out_at: chrono::Utc::now(),
    }))
}

/// Get current user information
pub async fn user_info(
    request: axum::extract::Request,
) -> Result<Json<UserInfo>> {
    let auth_context = request.require_auth_context()?;
    
    info!("User info request for: {}", auth_context.email);

    // Fetch fresh user data from database
    let user = get_user_by_id(auth_context.user_id).await?;

    Ok(Json(user))
}

// Helper functions (in a real implementation, these would interact with a database)

async fn authenticate_user(email: &str, password: &str) -> Result<UserInfo> {
    // This is a demo implementation - in production:
    // 1. Hash the provided password using the same algorithm as stored
    // 2. Compare with database hash
    // 3. Check account status, attempt limits, etc.
    
    // Demo user for testing
    if email == "admin@doc-rag.io" && password == "admin123" {
        Ok(UserInfo {
            id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            email: email.to_string(),
            name: "Admin User".to_string(),
            role: UserRole::Admin,
            created_at: chrono::Utc::now() - chrono::Duration::days(30),
            last_login: Some(chrono::Utc::now()),
        })
    } else if email == "user@doc-rag.io" && password == "user123" {
        Ok(UserInfo {
            id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440001").unwrap(),
            email: email.to_string(),
            name: "Regular User".to_string(),
            role: UserRole::User,
            created_at: chrono::Utc::now() - chrono::Duration::days(15),
            last_login: Some(chrono::Utc::now()),
        })
    } else {
        warn!("Authentication failed for email: {}", email);
        Err(ApiError::Unauthorized("Invalid email or password".to_string()))
    }
}

async fn validate_refresh_token(refresh_token: &str) -> Result<UserInfo> {
    // In production:
    // 1. Look up refresh token in database/cache
    // 2. Check if token is valid and not expired
    // 3. Get associated user data
    
    // Demo implementation
    if refresh_token.len() == 32 {
        // Return demo user - in production, get from database
        Ok(UserInfo {
            id: Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap(),
            email: "admin@doc-rag.io".to_string(),
            name: "Admin User".to_string(),
            role: UserRole::Admin,
            created_at: chrono::Utc::now() - chrono::Duration::days(30),
            last_login: Some(chrono::Utc::now()),
        })
    } else {
        Err(ApiError::Unauthorized("Invalid refresh token".to_string()))
    }
}

async fn store_refresh_token(user_id: Uuid, refresh_token: &str) -> Result<()> {
    // In production:
    // 1. Store in Redis/database with expiration
    // 2. Associate with user ID
    // 3. Handle cleanup of expired tokens
    
    info!("Storing refresh token for user: {}", user_id);
    Ok(())
}

async fn invalidate_refresh_token(refresh_token: &str) -> Result<()> {
    // In production: Remove from storage
    info!("Invalidating refresh token: {}", &refresh_token[..8]);
    Ok(())
}

async fn invalidate_access_token(token_id: &str) -> Result<()> {
    // In production: Add token ID to blacklist with expiration
    info!("Invalidating access token: {}", token_id);
    Ok(())
}

async fn invalidate_user_refresh_tokens(user_id: Uuid) -> Result<()> {
    // In production: Remove all refresh tokens for user
    info!("Invalidating all refresh tokens for user: {}", user_id);
    Ok(())
}

async fn get_user_by_id(user_id: Uuid) -> Result<UserInfo> {
    // In production: Fetch from database
    
    // Demo implementation
    if user_id == Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap() {
        Ok(UserInfo {
            id: user_id,
            email: "admin@doc-rag.io".to_string(),
            name: "Admin User".to_string(),
            role: UserRole::Admin,
            created_at: chrono::Utc::now() - chrono::Duration::days(30),
            last_login: Some(chrono::Utc::now()),
        })
    } else if user_id == Uuid::parse_str("550e8400-e29b-41d4-a716-446655440001").unwrap() {
        Ok(UserInfo {
            id: user_id,
            email: "user@doc-rag.io".to_string(),
            name: "Regular User".to_string(),
            role: UserRole::User,
            created_at: chrono::Utc::now() - chrono::Duration::days(15),
            last_login: Some(chrono::Utc::now()),
        })
    } else {
        Err(ApiError::NotFound("User not found".to_string()))
    }
}

// Helper trait for UserRole string conversion
impl ToString for UserRole {
    fn to_string(&self) -> String {
        match self {
            UserRole::Admin => "admin".to_string(),
            UserRole::User => "user".to_string(),
            UserRole::System => "system".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_authenticate_user_success() {
        let result = authenticate_user("admin@doc-rag.io", "admin123").await;
        assert!(result.is_ok());
        
        let user = result.unwrap();
        assert_eq!(user.email, "admin@doc-rag.io");
        assert_eq!(user.role, UserRole::Admin);
    }

    #[tokio::test]
    async fn test_authenticate_user_failure() {
        let result = authenticate_user("admin@doc-rag.io", "wrongpassword").await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ApiError::Unauthorized(_) => {
                // Expected error type
            }
            _ => panic!("Expected Unauthorized error"),
        }
    }

    #[tokio::test]
    async fn test_validate_refresh_token() {
        // Valid length token should work in demo
        let result = validate_refresh_token("abcdefghijklmnopqrstuvwxyz123456").await;
        assert!(result.is_ok());

        // Invalid length should fail
        let result = validate_refresh_token("short").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_token_operations() {
        let user_id = Uuid::new_v4();
        let refresh_token = generate_refresh_token();

        // Test storage operations (should not fail)
        assert!(store_refresh_token(user_id, &refresh_token).await.is_ok());
        assert!(invalidate_refresh_token(&refresh_token).await.is_ok());
        assert!(invalidate_user_refresh_tokens(user_id).await.is_ok());
        assert!(invalidate_access_token("test-token-id").await.is_ok());
    }

    #[tokio::test]
    async fn test_get_user_by_id() {
        // Test with known demo user ID
        let admin_id = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let result = get_user_by_id(admin_id).await;
        assert!(result.is_ok());

        let user = result.unwrap();
        assert_eq!(user.id, admin_id);
        assert_eq!(user.role, UserRole::Admin);

        // Test with unknown user ID
        let unknown_id = Uuid::new_v4();
        let result = get_user_by_id(unknown_id).await;
        assert!(result.is_err());
    }
}