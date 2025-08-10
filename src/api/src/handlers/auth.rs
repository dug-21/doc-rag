use axum::{
    extract::{Request, State},
    response::Json,
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn, error};

use crate::{
    config::ApiConfig,
    models::{LoginRequest, AuthResponse, UserInfo, UserRole},
    middleware::auth::{generate_jwt_token, generate_refresh_token, AuthExtension},
    validation::{validate_login_request, validate_email, validate_password_strength},
    Result, ApiError,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogoutRequest {
    pub refresh_token: Option<String>,
}

/// User login endpoint
pub async fn login(
    State(config): State<Arc<ApiConfig>>,
    Json(request): Json<LoginRequest>,
) -> Result<Json<AuthResponse>> {
    info!("Login attempt for email: {}", request.email);
    
    // Validate the login request
    validate_login_request(&request)?;
    
    // Additional email validation
    validate_email(&request.email)?;
    
    // In a real implementation, this would:
    // 1. Hash the password and compare with stored hash
    // 2. Check user exists and is active
    // 3. Update last login timestamp
    // 4. Generate tokens with proper user context
    
    // For now, we'll simulate a successful login
    // In production, replace this with actual user authentication
    let user = authenticate_user(&request.email, &request.password, &config).await?;
    
    // Generate JWT access token
    let access_token = generate_jwt_token(
        user.id,
        user.email.clone(),
        user.role.to_string(),
        &config.security.jwt_secret,
        config.security.jwt_expiration_hours,
    )?;
    
    // Generate refresh token
    let refresh_token = generate_refresh_token();
    
    // In production, store the refresh token in database with expiration
    // store_refresh_token(&user.id, &refresh_token, expiration_time).await?;
    
    let auth_response = AuthResponse {
        access_token,
        refresh_token,
        expires_in: config.security.jwt_expiration_hours * 3600, // Convert to seconds
        user,
    };
    
    info!("Login successful for email: {}", request.email);
    
    Ok(Json(auth_response))
}

/// Refresh access token using refresh token
pub async fn refresh_token(
    State(config): State<Arc<ApiConfig>>,
    Json(request): Json<RefreshTokenRequest>,
) -> Result<Json<AuthResponse>> {
    info!("Token refresh requested");
    
    // In a real implementation, this would:
    // 1. Validate the refresh token
    // 2. Check if it's not expired and not revoked
    // 3. Get user information from the refresh token
    // 4. Generate new access token
    // 5. Optionally rotate the refresh token
    
    // For now, simulate successful refresh with dummy user
    let user = get_user_from_refresh_token(&request.refresh_token).await?;
    
    // Generate new JWT access token
    let access_token = generate_jwt_token(
        user.id,
        user.email.clone(),
        user.role.to_string(),
        &config.security.jwt_secret,
        config.security.jwt_expiration_hours,
    )?;
    
    // Generate new refresh token (token rotation)
    let new_refresh_token = generate_refresh_token();
    
    // In production:
    // revoke_refresh_token(&request.refresh_token).await?;
    // store_refresh_token(&user.id, &new_refresh_token, expiration_time).await?;
    
    let auth_response = AuthResponse {
        access_token,
        refresh_token: new_refresh_token,
        expires_in: config.security.jwt_expiration_hours * 3600,
        user,
    };
    
    info!("Token refresh successful");
    
    Ok(Json(auth_response))
}

/// User logout endpoint
pub async fn logout(
    request: Request,
    Json(logout_request): Json<LogoutRequest>,
) -> Result<StatusCode> {
    let auth_context = request.require_auth_context()?;
    info!("Logout requested for user: {}", auth_context.email);
    
    // In a real implementation, this would:
    // 1. Revoke the current access token (add to blacklist)
    // 2. If refresh_token is provided, revoke it from database
    // 3. Clear any server-side sessions
    // 4. Log the logout event for security audit
    
    if let Some(refresh_token) = logout_request.refresh_token {
        // revoke_refresh_token(&refresh_token).await?;
        info!("Refresh token revoked for user: {}", auth_context.email);
    }
    
    // In production, add access token to blacklist
    // blacklist_access_token(&auth_context.token_id).await?;
    
    info!("Logout successful for user: {}", auth_context.email);
    
    Ok(StatusCode::NO_CONTENT)
}

/// Get current user information
pub async fn user_info(
    request: Request,
) -> Result<Json<UserInfo>> {
    let auth_context = request.require_auth_context()?;
    info!("User info requested for: {}", auth_context.email);
    
    // In a real implementation, this would fetch fresh user data from database
    let user = get_user_by_id(&auth_context.user_id).await?;
    
    Ok(Json(user))
}

// Helper functions - In production, these would interact with your database

async fn authenticate_user(email: &str, password: &str, config: &ApiConfig) -> Result<UserInfo> {
    // In production, this would:
    // 1. Query database for user by email
    // 2. Verify password hash using argon2 or similar
    // 3. Check if user account is active
    // 4. Return user information
    
    // For demo purposes, we'll use a simple check
    if email == "admin@example.com" && password == "admin123" {
        Ok(UserInfo {
            id: uuid::Uuid::new_v4(),
            email: email.to_string(),
            name: "Admin User".to_string(),
            role: UserRole::Admin,
            created_at: chrono::Utc::now(),
            last_login: Some(chrono::Utc::now()),
        })
    } else if email.contains("@") && password.len() >= config.security.password_min_length {
        Ok(UserInfo {
            id: uuid::Uuid::new_v4(),
            email: email.to_string(),
            name: extract_name_from_email(email),
            role: UserRole::User,
            created_at: chrono::Utc::now(),
            last_login: Some(chrono::Utc::now()),
        })
    } else {
        Err(ApiError::Unauthorized("Invalid credentials".to_string()))
    }
}

async fn get_user_from_refresh_token(refresh_token: &str) -> Result<UserInfo> {
    // In production, this would:
    // 1. Look up the refresh token in database
    // 2. Check if it's valid and not expired
    // 3. Return associated user information
    
    // For demo purposes, return a dummy user
    if refresh_token.len() >= 32 {
        Ok(UserInfo {
            id: uuid::Uuid::new_v4(),
            email: "user@example.com".to_string(),
            name: "Demo User".to_string(),
            role: UserRole::User,
            created_at: chrono::Utc::now(),
            last_login: Some(chrono::Utc::now()),
        })
    } else {
        Err(ApiError::Unauthorized("Invalid refresh token".to_string()))
    }
}

async fn get_user_by_id(user_id: &uuid::Uuid) -> Result<UserInfo> {
    // In production, this would query the database for user by ID
    
    Ok(UserInfo {
        id: *user_id,
        email: "user@example.com".to_string(),
        name: "Demo User".to_string(),
        role: UserRole::User,
        created_at: chrono::Utc::now(),
        last_login: Some(chrono::Utc::now()),
    })
}

fn extract_name_from_email(email: &str) -> String {
    email.split('@')
        .next()
        .unwrap_or("User")
        .replace('.', " ")
        .replace('_', " ")
        .split(' ')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars.as_str().to_lowercase()).collect(),
            }
        })
        .collect::<Vec<String>>()
        .join(" ")
}

impl std::fmt::Display for UserRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UserRole::User => write!(f, "user"),
            UserRole::Admin => write!(f, "admin"),
            UserRole::System => write!(f, "system"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiConfig;

    #[tokio::test]
    async fn test_authenticate_user_valid_credentials() {
        let config = ApiConfig::default();
        let result = authenticate_user("admin@example.com", "admin123", &config).await;
        assert!(result.is_ok());
        
        let user = result.unwrap();
        assert_eq!(user.email, "admin@example.com");
        assert!(matches!(user.role, UserRole::Admin));
    }

    #[tokio::test]
    async fn test_authenticate_user_invalid_credentials() {
        let config = ApiConfig::default();
        let result = authenticate_user("user@example.com", "wrong", &config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_get_user_from_refresh_token_valid() {
        let result = get_user_from_refresh_token("a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_user_from_refresh_token_invalid() {
        let result = get_user_from_refresh_token("short").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_name_from_email() {
        assert_eq!(extract_name_from_email("john.doe@example.com"), "John Doe");
        assert_eq!(extract_name_from_email("jane_smith@example.com"), "Jane Smith");
        assert_eq!(extract_name_from_email("user@example.com"), "User");
    }

    #[test]
    fn test_user_role_display() {
        assert_eq!(UserRole::User.to_string(), "user");
        assert_eq!(UserRole::Admin.to_string(), "admin");
        assert_eq!(UserRole::System.to_string(), "system");
    }

    #[tokio::test]
    async fn test_login_validation() {
        let config = Arc::new(ApiConfig::default());
        
        let valid_request = LoginRequest {
            email: "test@example.com".to_string(),
            password: "password123".to_string(),
        };
        
        let result = login(State(config.clone()), Json(valid_request)).await;
        assert!(result.is_ok());
        
        let invalid_request = LoginRequest {
            email: "invalid-email".to_string(),
            password: "short".to_string(),
        };
        
        let result = login(State(config), Json(invalid_request)).await;
        assert!(result.is_err());
    }
}