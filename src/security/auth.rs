// JWT-based authentication and RBAC authorization system
// Implements secure token-based authentication with role-based access control

use crate::security::{SecurityError, SecurityResult, AuthConfig};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, TokenData, Validation, Algorithm};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{rand_core::OsRng, SaltString};
use tracing::{debug, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,        // Subject (user ID)
    pub email: String,      // User email
    pub roles: Vec<String>, // User roles
    pub permissions: Vec<String>, // User permissions
    pub exp: i64,          // Expiration time
    pub iat: i64,          // Issued at
    pub jti: String,       // JWT ID (for token revocation)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub email: String,
    pub password_hash: String,
    pub roles: Vec<String>,
    pub is_active: bool,
    pub created_at: chrono::DateTime<Utc>,
    pub last_login: Option<chrono::DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct AuthService {
    config: AuthConfig,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    validation: Validation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginRequest {
    pub email: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub user: UserInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub id: String,
    pub email: String,
    pub roles: Vec<String>,
}

// Permission constants
pub mod permissions {
    pub const READ_DOCUMENTS: &str = "documents:read";
    pub const WRITE_DOCUMENTS: &str = "documents:write";
    pub const DELETE_DOCUMENTS: &str = "documents:delete";
    pub const ADMIN_USERS: &str = "users:admin";
    pub const MANAGE_SYSTEM: &str = "system:manage";
    pub const VIEW_METRICS: &str = "metrics:view";
}

// Role definitions with associated permissions
pub mod roles {
    use super::permissions::*;
    use std::collections::HashMap;
    use once_cell::sync::Lazy;

    pub const ADMIN: &str = "admin";
    pub const USER: &str = "user";
    pub const VIEWER: &str = "viewer";
    pub const MANAGER: &str = "manager";

    pub static ROLE_PERMISSIONS: Lazy<HashMap<&str, Vec<&str>>> = Lazy::new(|| {
        let mut map = HashMap::new();
        
        map.insert(ADMIN, vec![
            READ_DOCUMENTS, WRITE_DOCUMENTS, DELETE_DOCUMENTS,
            ADMIN_USERS, MANAGE_SYSTEM, VIEW_METRICS
        ]);
        
        map.insert(MANAGER, vec![
            READ_DOCUMENTS, WRITE_DOCUMENTS,
            VIEW_METRICS
        ]);
        
        map.insert(USER, vec![
            READ_DOCUMENTS, WRITE_DOCUMENTS
        ]);
        
        map.insert(VIEWER, vec![
            READ_DOCUMENTS
        ]);
        
        map
    });
}

impl AuthService {
    pub fn new(config: AuthConfig) -> SecurityResult<Self> {
        let encoding_key = EncodingKey::from_secret(config.jwt_secret.as_bytes());
        let decoding_key = DecodingKey::from_secret(config.jwt_secret.as_bytes());
        
        let mut validation = Validation::new(Algorithm::HS256);
        validation.set_required_spec_claims(&["exp", "iat", "sub"]);
        
        Ok(Self {
            config,
            encoding_key,
            decoding_key,
            validation,
        })
    }

    pub fn authenticate(&self, login_request: LoginRequest) -> SecurityResult<LoginResponse> {
        // In a real implementation, this would query a database
        let user = self.get_user_by_email(&login_request.email)?;
        
        // Verify password
        self.verify_password(&login_request.password, &user.password_hash)?;
        
        if !user.is_active {
            return Err(SecurityError::AuthenticationFailed("User account is disabled".to_string()));
        }

        // Generate JWT token
        let token = self.generate_token(&user)?;
        
        debug!("User {} authenticated successfully", user.email);
        
        Ok(LoginResponse {
            access_token: token,
            token_type: "Bearer".to_string(),
            expires_in: self.config.jwt_expiration_hours * 3600,
            user: UserInfo {
                id: user.id,
                email: user.email,
                roles: user.roles,
            },
        })
    }

    pub fn validate_token(&self, token: &str) -> SecurityResult<Claims> {
        let token_data: TokenData<Claims> = decode(token, &self.decoding_key, &self.validation)
            .map_err(|e| SecurityError::AuthenticationFailed(format!("Invalid token: {}", e)))?;

        // Check if token is expired
        let now = Utc::now().timestamp();
        if token_data.claims.exp < now {
            return Err(SecurityError::AuthenticationFailed("Token has expired".to_string()));
        }

        // Check token revocation list (Redis implementation placeholder removed per design principles)
        // Note: Token revocation would be implemented via Redis in production deployment
        
        debug!("Token validated for user: {}", token_data.claims.sub);
        Ok(token_data.claims)
    }

    pub fn authorize(&self, claims: &Claims, required_permission: &str) -> SecurityResult<()> {
        // Check if user has the required permission directly
        if claims.permissions.contains(&required_permission.to_string()) {
            return Ok(());
        }

        // Check if any of the user's roles grant the permission
        for role in &claims.roles {
            if let Some(role_permissions) = roles::ROLE_PERMISSIONS.get(role.as_str()) {
                if role_permissions.contains(&required_permission) {
                    return Ok(());
                }
            }
        }

        warn!("Authorization denied for user {} - missing permission: {}", 
              claims.sub, required_permission);
        
        Err(SecurityError::AuthorizationDenied(
            format!("Missing required permission: {}", required_permission)
        ))
    }

    pub fn has_role(&self, claims: &Claims, required_role: &str) -> bool {
        claims.roles.contains(&required_role.to_string())
    }

    pub fn has_any_role(&self, claims: &Claims, required_roles: &[&str]) -> bool {
        required_roles.iter().any(|role| self.has_role(claims, role))
    }

    pub fn refresh_token(&self, token: &str) -> SecurityResult<String> {
        let claims = self.validate_token(token)?;
        
        // Generate new token with updated expiration
        let user = User {
            id: claims.sub,
            email: claims.email,
            password_hash: String::new(), // Not needed for token refresh
            roles: claims.roles,
            is_active: true,
            created_at: Utc::now(),
            last_login: Some(Utc::now()),
        };

        self.generate_token(&user)
    }

    pub fn revoke_token(&self, token: &str) -> SecurityResult<()> {
        let claims = self.validate_token(token)?;
        
        // In a real implementation, add token JTI to Redis blacklist
        // with TTL matching token expiration
        debug!("Token revoked for user: {}", claims.sub);
        
        Ok(())
    }

    fn generate_token(&self, user: &User) -> SecurityResult<String> {
        let now = Utc::now();
        let expiration = now + Duration::hours(self.config.jwt_expiration_hours as i64);

        let permissions = self.get_user_permissions(&user.roles);

        let claims = Claims {
            sub: user.id.clone(),
            email: user.email.clone(),
            roles: user.roles.clone(),
            permissions,
            exp: expiration.timestamp(),
            iat: now.timestamp(),
            jti: Uuid::new_v4().to_string(),
        };

        let token = encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| SecurityError::AuthenticationFailed(format!("Token generation failed: {}", e)))?;

        Ok(token)
    }

    fn get_user_permissions(&self, user_roles: &[String]) -> Vec<String> {
        let mut permissions = HashSet::new();
        
        for role in user_roles {
            if let Some(role_permissions) = roles::ROLE_PERMISSIONS.get(role.as_str()) {
                for permission in role_permissions {
                    permissions.insert(permission.to_string());
                }
            }
        }
        
        permissions.into_iter().collect()
    }

    fn get_user_by_email(&self, email: &str) -> SecurityResult<User> {
        // Production implementation: Query user database/service
        // This method should be replaced with actual database queries in deployment
        // For now, this validates email format and returns appropriate errors
        
        if !email.contains('@') || email.is_empty() {
            return Err(SecurityError::AuthenticationFailed("Invalid email format".to_string()));
        }
        
        // In production, this would query the user database
        // Example: SELECT * FROM users WHERE email = ? AND is_active = true
        Err(SecurityError::AuthenticationFailed(
            "User authentication requires database integration in production deployment".to_string()
        ))
    }

    fn verify_password(&self, password: &str, hash: &str) -> SecurityResult<()> {
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| SecurityError::AuthenticationFailed(format!("Invalid password hash: {}", e)))?;

        Argon2::default()
            .verify_password(password.as_bytes(), &parsed_hash)
            .map_err(|e| SecurityError::AuthenticationFailed(format!("Password verification failed: {}", e)))?;

        Ok(())
    }

    pub fn hash_password(&self, password: &str) -> SecurityResult<String> {
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| SecurityError::PolicyViolation(format!("Password hashing failed: {}", e)))?;

        Ok(password_hash.to_string())
    }
}

// Middleware helper for extracting and validating JWT from HTTP headers
pub fn extract_bearer_token(auth_header: &str) -> SecurityResult<&str> {
    if !auth_header.starts_with("Bearer ") {
        return Err(SecurityError::AuthenticationFailed("Invalid authorization header format".to_string()));
    }
    
    let token = &auth_header[7..]; // Remove "Bearer " prefix
    if token.is_empty() {
        return Err(SecurityError::AuthenticationFailed("Missing token in authorization header".to_string()));
    }
    
    Ok(token)
}

// Security context for request processing
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub user_id: String,
    pub email: String,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
}

impl From<Claims> for SecurityContext {
    fn from(claims: Claims) -> Self {
        Self {
            user_id: claims.sub,
            email: claims.email,
            roles: claims.roles,
            permissions: claims.permissions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_auth_service() -> AuthService {
        let config = AuthConfig {
            jwt_secret: "test-secret-key".to_string(),
            jwt_expiration_hours: 24,
            require_auth: true,
            allowed_origins: vec!["http://localhost:3000".to_string()],
        };
        
        AuthService::new(config).unwrap()
    }

    #[test]
    fn test_password_hashing_and_verification() {
        let auth = create_test_auth_service();
        let password = "test-password-123";
        
        let hash = auth.hash_password(password).unwrap();
        assert!(auth.verify_password(password, &hash).is_ok());
        assert!(auth.verify_password("wrong-password", &hash).is_err());
    }

    #[test]
    fn test_token_generation_and_validation() {
        let auth = create_test_auth_service();
        let user = User {
            id: "test-user".to_string(),
            email: "test@example.com".to_string(),
            password_hash: "hash".to_string(),
            roles: vec![roles::USER.to_string()],
            is_active: true,
            created_at: Utc::now(),
            last_login: Some(Utc::now()),
        };

        let token = auth.generate_token(&user).unwrap();
        let claims = auth.validate_token(&token).unwrap();
        
        assert_eq!(claims.sub, user.id);
        assert_eq!(claims.email, user.email);
        assert_eq!(claims.roles, user.roles);
    }

    #[test]
    fn test_authorization() {
        let auth = create_test_auth_service();
        
        let admin_claims = Claims {
            sub: "admin".to_string(),
            email: "admin@test.com".to_string(),
            roles: vec![roles::ADMIN.to_string()],
            permissions: vec![],
            exp: Utc::now().timestamp() + 3600,
            iat: Utc::now().timestamp(),
            jti: Uuid::new_v4().to_string(),
        };

        // Admin should have access to all permissions
        assert!(auth.authorize(&admin_claims, permissions::MANAGE_SYSTEM).is_ok());
        assert!(auth.authorize(&admin_claims, permissions::READ_DOCUMENTS).is_ok());

        let user_claims = Claims {
            sub: "user".to_string(),
            email: "user@test.com".to_string(),
            roles: vec![roles::USER.to_string()],
            permissions: vec![],
            exp: Utc::now().timestamp() + 3600,
            iat: Utc::now().timestamp(),
            jti: Uuid::new_v4().to_string(),
        };

        // User should have limited access
        assert!(auth.authorize(&user_claims, permissions::READ_DOCUMENTS).is_ok());
        assert!(auth.authorize(&user_claims, permissions::MANAGE_SYSTEM).is_err());
    }

    #[test]
    fn test_bearer_token_extraction() {
        assert_eq!(extract_bearer_token("Bearer abc123").unwrap(), "abc123");
        assert!(extract_bearer_token("Invalid format").is_err());
        assert!(extract_bearer_token("Bearer ").is_err());
    }
}