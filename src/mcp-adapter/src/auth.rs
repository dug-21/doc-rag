//! Complete OAuth2/JWT authentication with token refresh and validation

use chrono::{DateTime, Duration, Utc};
use jsonwebtoken::{decode, decode_header, DecodingKey, Validation, Algorithm, Header};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;
use url::Url;
use base64::{Engine as _, engine::general_purpose};
use sha2::{Sha256, Digest};
use rand::{thread_rng, Rng};

use crate::{McpError, Result};

/// OAuth2/JWT credentials for authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    pub client_id: String,
    pub client_secret: String,
    pub username: Option<String>,
    pub password: Option<String>,
    pub grant_type: GrantType,
    pub scope: Vec<String>,
    /// Optional authorization code for authorization code flow
    pub authorization_code: Option<String>,
    /// Optional redirect URI for authorization code flow
    pub redirect_uri: Option<String>,
    /// Optional code verifier for PKCE
    pub code_verifier: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GrantType {
    ClientCredentials,
    Password,
    AuthorizationCode,
    RefreshToken,
    /// Device authorization flow
    DeviceCode,
    /// JWT Bearer token grant
    JwtBearer,
}

impl std::fmt::Display for GrantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrantType::ClientCredentials => write!(f, "client_credentials"),
            GrantType::Password => write!(f, "password"),
            GrantType::AuthorizationCode => write!(f, "authorization_code"),
            GrantType::RefreshToken => write!(f, "refresh_token"),
            GrantType::DeviceCode => write!(f, "urn:ietf:params:oauth:grant-type:device_code"),
            GrantType::JwtBearer => write!(f, "urn:ietf:params:oauth:grant-type:jwt-bearer"),
        }
    }
}

/// JWT/OAuth2 authentication token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub access_token: String,
    pub token_type: String,
    pub expires_at: DateTime<Utc>,
    pub refresh_token: Option<String>,
    pub scope: Vec<String>,
    pub claims: Option<TokenClaims>,
    /// ID token for OpenID Connect
    pub id_token: Option<String>,
    /// Token creation timestamp
    pub issued_at: DateTime<Utc>,
    /// Last refresh timestamp
    pub last_refresh: Option<DateTime<Utc>>,
    /// Number of refresh attempts
    pub refresh_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    pub sub: String,
    pub iss: String,
    pub aud: Vec<String>,
    pub exp: i64,
    pub iat: i64,
    pub nbf: Option<i64>,
    pub jti: Option<String>,
    #[serde(flatten)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// OAuth2 token response from server
#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    token_type: String,
    expires_in: i64,
    refresh_token: Option<String>,
    scope: Option<String>,
    id_token: Option<String>,
}

/// OAuth2 error response
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: String,
    error_description: Option<String>,
    error_uri: Option<String>,
}

/// Device authorization response
#[derive(Debug, Deserialize)]
struct DeviceAuthResponse {
    device_code: String,
    user_code: String,
    verification_uri: String,
    verification_uri_complete: Option<String>,
    expires_in: i64,
    interval: Option<i64>,
}

/// PKCE (Proof Key for Code Exchange) helper
#[derive(Debug, Clone)]
pub struct PkceChallenge {
    pub code_verifier: String,
    pub code_challenge: String,
    pub code_challenge_method: String,
}

impl PkceChallenge {
    /// Generate a new PKCE challenge
    pub fn new() -> Self {
        let code_verifier = generate_code_verifier();
        let code_challenge = generate_code_challenge(&code_verifier);
        
        Self {
            code_verifier,
            code_challenge,
            code_challenge_method: "S256".to_string(),
        }
    }
}

/// Generate a cryptographically random code verifier
fn generate_code_verifier() -> String {
    let mut rng = thread_rng();
    let bytes: Vec<u8> = (0..96).map(|_| rng.gen_range(65..=90)).collect(); // A-Z
    String::from_utf8(bytes).expect("Valid ASCII")
}

/// Generate SHA256 code challenge from verifier
fn generate_code_challenge(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    general_purpose::URL_SAFE_NO_PAD.encode(digest)
}

impl AuthToken {
    /// Check if token needs refresh based on threshold
    pub fn needs_refresh(&self, threshold_secs: i64) -> bool {
        let now = Utc::now();
        let threshold_time = self.expires_at - Duration::seconds(threshold_secs);
        now >= threshold_time
    }

    /// Check if token is currently valid
    pub fn is_valid(&self) -> bool {
        Utc::now() < self.expires_at
    }

    /// Get remaining lifetime in seconds
    pub fn remaining_lifetime(&self) -> i64 {
        (self.expires_at - Utc::now()).num_seconds().max(0)
    }

    /// Check if token is close to expiry (within 5 minutes)
    pub fn is_expiring_soon(&self) -> bool {
        self.needs_refresh(300)
    }

    /// Check if refresh token is available
    pub fn can_refresh(&self) -> bool {
        self.refresh_token.is_some() && self.refresh_count < 10 // Limit refresh attempts
    }

    /// Update token with refresh data
    pub fn update_from_refresh(&mut self, new_token: AuthToken) {
        self.access_token = new_token.access_token;
        self.expires_at = new_token.expires_at;
        self.last_refresh = Some(Utc::now());
        self.refresh_count += 1;
        
        // Keep new refresh token if provided, otherwise keep the old one
        if let Some(new_refresh) = new_token.refresh_token {
            self.refresh_token = Some(new_refresh);
        }
        
        // Update other fields
        if let Some(new_id_token) = new_token.id_token {
            self.id_token = Some(new_id_token);
        }
        
        self.scope = new_token.scope;
        self.claims = new_token.claims;
    }

    /// Parse JWT claims from access token
    pub fn parse_claims(&mut self, validation_key: Option<&str>) -> Result<()> {
        // Try to decode header first to determine if this is a JWT
        let header = match decode_header(&self.access_token) {
            Ok(h) => h,
            Err(_) => {
                debug!("Token is not a JWT, treating as opaque token");
                return Ok(());
            }
        };

        let validation = if let Some(key) = validation_key {
            let mut validation = Validation::new(header.alg);
            // Configure validation parameters
            validation.validate_exp = true;
            validation.validate_nbf = true;
            validation.leeway = 60; // Allow 60 seconds clock skew
            validation
        } else {
            let mut validation = Validation::new(Algorithm::HS256);
            validation.insecure_disable_signature_validation();
            validation.validate_exp = false; // Don't validate expiration if no key
            validation
        };

        let decoding_key = validation_key
            .map(|k| {
                // Try different key formats
                if k.starts_with("-----BEGIN") {
                    // PEM format
                    match header.alg {
                        Algorithm::RS256 | Algorithm::RS384 | Algorithm::RS512 => {
                            DecodingKey::from_rsa_pem(k.as_bytes())
                                .unwrap_or_else(|_| DecodingKey::from_secret(k.as_bytes()))
                        },
                        Algorithm::ES256 | Algorithm::ES384 => {
                            DecodingKey::from_ec_pem(k.as_bytes())
                                .unwrap_or_else(|_| DecodingKey::from_secret(k.as_bytes()))
                        },
                        _ => DecodingKey::from_secret(k.as_bytes()),
                    }
                } else {
                    // Plain secret
                    DecodingKey::from_secret(k.as_bytes())
                }
            })
            .unwrap_or_else(|| DecodingKey::from_secret(b"dummy"));

        match decode::<TokenClaims>(&self.access_token, &decoding_key, &validation) {
            Ok(token_data) => {
                self.claims = Some(token_data.claims);
                debug!("Successfully parsed JWT claims");
                Ok(())
            }
            Err(e) => {
                debug!("Failed to parse JWT claims: {}. Token might be opaque or validation key incorrect.", e);
                // Don't fail completely - token might still be valid as opaque token
                Ok(())
            }
        }
    }

    /// Validate token claims (audience, issuer, etc.)
    pub fn validate_claims(&self, expected_audience: Option<&str>, expected_issuer: Option<&str>) -> Result<bool> {
        if let Some(claims) = &self.claims {
            // Check expiration
            let now = Utc::now().timestamp();
            if claims.exp <= now {
                return Ok(false);
            }

            // Check not before
            if let Some(nbf) = claims.nbf {
                if now < nbf {
                    return Ok(false);
                }
            }

            // Check audience
            if let Some(expected_aud) = expected_audience {
                if !claims.aud.contains(&expected_aud.to_string()) {
                    return Ok(false);
                }
            }

            // Check issuer
            if let Some(expected_iss) = expected_issuer {
                if claims.iss != expected_iss {
                    return Ok(false);
                }
            }

            Ok(true)
        } else {
            // No claims to validate - assume opaque token is valid if not expired
            Ok(self.is_valid())
        }
    }
}

/// Handles OAuth2/JWT authentication flows with comprehensive support
#[derive(Debug)]
pub struct AuthHandler {
    client_id: String,
    /// Cached token for reuse
    cached_token: Arc<RwLock<Option<AuthToken>>>,
    /// Token validation key
    validation_key: Option<String>,
    /// OAuth2 discovery document cache
    discovery_cache: Arc<RwLock<Option<OAuthDiscovery>>>,
    /// Maximum retry attempts for auth requests
    max_retries: usize,
}

/// OAuth2/OpenID Connect discovery document
#[derive(Debug, Clone, Deserialize)]
struct OAuthDiscovery {
    issuer: String,
    authorization_endpoint: String,
    token_endpoint: String,
    userinfo_endpoint: Option<String>,
    jwks_uri: Option<String>,
    scopes_supported: Option<Vec<String>>,
    response_types_supported: Vec<String>,
    grant_types_supported: Option<Vec<String>>,
    token_endpoint_auth_methods_supported: Option<Vec<String>>,
}

impl AuthHandler {
    pub fn new() -> Self {
        Self {
            client_id: Uuid::new_v4().to_string(),
            cached_token: Arc::new(RwLock::new(None)),
            validation_key: None,
            discovery_cache: Arc::new(RwLock::new(None)),
            max_retries: 3,
        }
    }

    /// Create a new AuthHandler with custom client ID and validation key
    pub fn with_config(client_id: String, validation_key: Option<String>) -> Self {
        Self {
            client_id,
            cached_token: Arc::new(RwLock::new(None)),
            validation_key,
            discovery_cache: Arc::new(RwLock::new(None)),
            max_retries: 3,
        }
    }

    /// Discover OAuth2/OpenID Connect endpoints
    pub async fn discover_endpoints(&self, client: &Client, base_url: &str) -> Result<OAuthDiscovery> {
        let discovery_url = if base_url.contains(".well-known") {
            base_url.to_string()
        } else {
            format!(
                "{}/.well-known/openid_configuration", 
                base_url.trim_end_matches('/')
            )
        };

        debug!("Fetching OAuth discovery from: {}", discovery_url);

        let mut retry_count = 0;
        loop {
            let response = client.get(&discovery_url).send().await;
            
            match response {
                Ok(resp) if resp.status().is_success() => {
                    let discovery: OAuthDiscovery = resp.json().await
                        .map_err(|e| McpError::AuthenticationFailed(format!("Invalid discovery document: {}", e)))?;
                    
                    // Cache the discovery document
                    *self.discovery_cache.write().await = Some(discovery.clone());
                    info!("OAuth discovery successful for issuer: {}", discovery.issuer);
                    return Ok(discovery);
                }
                Ok(resp) => {
                    let status = resp.status();
                    let error_text = resp.text().await.unwrap_or_default();
                    
                    if retry_count < self.max_retries {
                        retry_count += 1;
                        warn!("Discovery attempt {} failed with status {}, retrying...", retry_count, status);
                        tokio::time::sleep(tokio::time::Duration::from_millis(1000 * retry_count as u64)).await;
                        continue;
                    } else {
                        return Err(McpError::AuthenticationFailed(format!(
                            "Discovery failed with status {}: {}", status, error_text
                        )));
                    }
                }
                Err(e) => {
                    if retry_count < self.max_retries {
                        retry_count += 1;
                        warn!("Discovery attempt {} failed with error: {}, retrying...", retry_count, e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(1000 * retry_count as u64)).await;
                        continue;
                    } else {
                        return Err(McpError::NetworkError(e));
                    }
                }
            }
        }
    }

    /// Get cached token if valid
    pub async fn get_cached_token(&self) -> Option<AuthToken> {
        let token_guard = self.cached_token.read().await;
        token_guard.as_ref().filter(|t| t.is_valid()).cloned()
    }

    /// Cache a token
    pub async fn cache_token(&self, token: AuthToken) {
        *self.cached_token.write().await = Some(token);
    }

    /// Clear cached token
    pub async fn clear_cache(&self) {
        *self.cached_token.write().await = None;
    }

    /// Authenticate using OAuth2 flow with comprehensive error handling
    pub async fn authenticate(
        &self,
        client: &Client,
        endpoint: &str,
        credentials: Credentials,
    ) -> Result<AuthToken> {
        // Check for cached valid token first
        if let Some(cached) = self.get_cached_token().await {
            debug!("Using cached valid token");
            return Ok(cached);
        }

        // Try to discover endpoints first
        let token_url = match self.discover_endpoints(client, endpoint).await {
            Ok(discovery) => discovery.token_endpoint,
            Err(_) => {
                debug!("Discovery failed, using fallback endpoint");
                format!("{}/oauth/token", endpoint.trim_end_matches('/'))
            }
        };
        
        let mut params = vec![
            ("grant_type", credentials.grant_type.to_string()),
            ("client_id", credentials.client_id.clone()),
        ];

        if !credentials.scope.is_empty() {
            params.push(("scope", credentials.scope.join(" ")));
        }

        // Add grant-type specific parameters
        match credentials.grant_type {
            GrantType::Password => {
                if let (Some(username), Some(password)) = (&credentials.username, &credentials.password) {
                    params.push(("username", username.clone()));
                    params.push(("password", password.clone()));
                } else {
                    return Err(McpError::AuthenticationFailed(
                        "Username and password required for password grant".to_string(),
                    ));
                }
            }
            GrantType::ClientCredentials => {
                params.push(("client_secret", credentials.client_secret.clone()));
            }
            GrantType::AuthorizationCode => {
                if let Some(code) = &credentials.authorization_code {
                    params.push(("code", code.clone()));
                    params.push(("client_secret", credentials.client_secret.clone()));
                    
                    if let Some(redirect_uri) = &credentials.redirect_uri {
                        params.push(("redirect_uri", redirect_uri.clone()));
                    }
                    
                    // Add PKCE verifier if present
                    if let Some(verifier) = &credentials.code_verifier {
                        params.push(("code_verifier", verifier.clone()));
                    }
                } else {
                    return Err(McpError::AuthenticationFailed(
                        "Authorization code required for authorization code grant".to_string(),
                    ));
                }
            }
            GrantType::DeviceCode => {
                return self.authenticate_device_flow(client, &token_url, credentials).await;
            }
            GrantType::JwtBearer => {
                return self.authenticate_jwt_bearer(client, &token_url, credentials).await;
            }
            GrantType::RefreshToken => {
                return Err(McpError::AuthenticationFailed(
                    "Use refresh_token method for refresh token grant".to_string(),
                ));
            }
        }

        self.execute_token_request(client, &token_url, params).await
    }

    /// Execute the actual token request with retries
    async fn execute_token_request(
        &self, 
        client: &Client, 
        token_url: &str, 
        params: Vec<(&str, String)>
    ) -> Result<AuthToken> {
        debug!("Requesting OAuth token from {}", token_url);
        
        let mut retry_count = 0;
        loop {
            let response = client
                .post(token_url)
                .header("Content-Type", "application/x-www-form-urlencoded")
                .header("Accept", "application/json")
                .form(&params)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    let status = resp.status();
                    
                    if status.is_success() {
                        return self.parse_token_response(resp).await;
                    } else if status == StatusCode::TOO_MANY_REQUESTS {
                        if retry_count < self.max_retries {
                            retry_count += 1;
                            let retry_after = resp.headers()
                                .get("retry-after")
                                .and_then(|h| h.to_str().ok())
                                .and_then(|s| s.parse::<u64>().ok())
                                .unwrap_or(1);
                            
                            warn!("Rate limited, retrying after {} seconds", retry_after);
                            tokio::time::sleep(tokio::time::Duration::from_secs(retry_after)).await;
                            continue;
                        } else {
                            return Err(McpError::RateLimitExceeded);
                        }
                    } else {
                        let error_text = resp.text().await.unwrap_or_default();
                        // Try to parse as OAuth error response
                        if let Ok(oauth_error) = serde_json::from_str::<ErrorResponse>(&error_text) {
                            return Err(McpError::AuthenticationFailed(format!(
                                "OAuth error: {} - {}",
                                oauth_error.error,
                                oauth_error.error_description.unwrap_or_default()
                            )));
                        } else {
                            return Err(McpError::AuthenticationFailed(format!(
                                "Authentication failed with status {}: {}",
                                status, error_text
                            )));
                        }
                    }
                }
                Err(e) => {
                    if retry_count < self.max_retries && e.is_timeout() {
                        retry_count += 1;
                        warn!("Request timeout, retrying ({}/{})", retry_count, self.max_retries);
                        tokio::time::sleep(tokio::time::Duration::from_millis(1000 * retry_count as u64)).await;
                        continue;
                    } else {
                        return Err(McpError::NetworkError(e));
                    }
                }
            }
        }
    }

    /// Parse token response and create AuthToken
    async fn parse_token_response(&self, response: reqwest::Response) -> Result<AuthToken> {
        let token_response: TokenResponse = response.json().await
            .map_err(|e| McpError::AuthenticationFailed(format!("Failed to parse token response: {}", e)))?;

        let expires_at = Utc::now() + Duration::seconds(token_response.expires_in);
        let scope = token_response.scope
            .map(|s| s.split_whitespace().map(String::from).collect())
            .unwrap_or_default();

        let mut token = AuthToken {
            access_token: token_response.access_token,
            token_type: token_response.token_type,
            expires_at,
            refresh_token: token_response.refresh_token,
            scope,
            claims: None,
            id_token: token_response.id_token,
            issued_at: Utc::now(),
            last_refresh: None,
            refresh_count: 0,
        };

        // Attempt to parse JWT claims
        let _ = token.parse_claims(self.validation_key.as_deref());

        // Cache the token
        self.cache_token(token.clone()).await;
        
        info!("Authentication successful, token expires at {}", token.expires_at);
        Ok(token)
    }

    /// Device authorization flow
    async fn authenticate_device_flow(
        &self,
        client: &Client,
        _token_url: &str,
        credentials: Credentials,
    ) -> Result<AuthToken> {
        // This is a simplified implementation - real device flow requires user interaction
        Err(McpError::AuthenticationFailed(
            "Device authorization flow not fully implemented - requires user interaction".to_string(),
        ))
    }

    /// JWT Bearer token flow
    async fn authenticate_jwt_bearer(
        &self,
        client: &Client,
        token_url: &str,
        credentials: Credentials,
    ) -> Result<AuthToken> {
        // Create JWT assertion
        let assertion = self.create_jwt_assertion(&credentials)?;
        
        let params = vec![
            ("grant_type", credentials.grant_type.to_string()),
            ("assertion", assertion),
        ];

        self.execute_token_request(client, token_url, params).await
    }

    /// Create JWT assertion for JWT Bearer flow
    fn create_jwt_assertion(&self, credentials: &Credentials) -> Result<String> {
        // This is a simplified implementation - real JWT assertion creation
        // would require proper signing with private keys
        Err(McpError::AuthenticationFailed(
            "JWT Bearer flow requires proper JWT signing implementation".to_string(),
        ))
    }

    /// Refresh an existing token using refresh token with comprehensive error handling
    pub async fn refresh_token(
        &self,
        client: &Client,
        endpoint: &str,
        current_token: &mut AuthToken,
    ) -> Result<AuthToken> {
        let refresh_token = current_token.refresh_token.as_ref()
            .ok_or_else(|| McpError::AuthenticationFailed("No refresh token available".to_string()))?;

        if current_token.refresh_count >= 10 {
            return Err(McpError::AuthenticationFailed(
                "Maximum refresh attempts exceeded".to_string(),
            ));
        }

        // Try to discover endpoints first
        let token_url = match self.discovery_cache.read().await.as_ref() {
            Some(discovery) => discovery.token_endpoint.clone(),
            None => match self.discover_endpoints(client, endpoint).await {
                Ok(discovery) => discovery.token_endpoint,
                Err(_) => format!("{}/oauth/token", endpoint.trim_end_matches('/'))
            }
        };
        
        let params = vec![
            ("grant_type", "refresh_token".to_string()),
            ("refresh_token", refresh_token.clone()),
            ("client_id", self.client_id.clone()),
        ];

        debug!("Refreshing OAuth token (attempt {})", current_token.refresh_count + 1);
        
        match self.execute_token_request(client, &token_url, params).await {
            Ok(mut new_token) => {
                // Update the current token with new data
                current_token.update_from_refresh(new_token.clone());
                
                // Update cache
                self.cache_token(current_token.clone()).await;
                
                info!("Token refresh successful, new expiry: {}", current_token.expires_at);
                Ok(current_token.clone())
            }
            Err(e) => {
                // If refresh fails, clear cache and return error
                self.clear_cache().await;
                error!("Token refresh failed: {}", e);
                Err(e)
            }
        }
    }

    /// Automatically refresh token if needed
    pub async fn ensure_valid_token(
        &self,
        client: &Client,
        endpoint: &str,
        token: &mut AuthToken,
        refresh_threshold_secs: i64,
    ) -> Result<()> {
        if token.needs_refresh(refresh_threshold_secs) {
            if token.can_refresh() {
                self.refresh_token(client, endpoint, token).await?;
            } else {
                return Err(McpError::AuthenticationFailed(
                    "Token expired and cannot be refreshed".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Comprehensive token validation
    pub async fn validate_token(
        &self, 
        client: &Client, 
        token: &AuthToken, 
        expected_audience: Option<&str>,
        expected_issuer: Option<&str>
    ) -> Result<bool> {
        // Check basic validity
        if !token.is_valid() {
            debug!("Token is expired");
            return Ok(false);
        }

        // Validate claims if present
        if let Err(_) = token.validate_claims(expected_audience, expected_issuer) {
            debug!("Token claims validation failed");
            return Ok(false);
        }

        // For JWT tokens with signature validation
        if let Some(validation_key) = &self.validation_key {
            if let Some(_claims) = &token.claims {
                let header = match decode_header(&token.access_token) {
                    Ok(h) => h,
                    Err(e) => {
                        debug!("Failed to decode JWT header: {}", e);
                        return Ok(false);
                    }
                };
                
                let mut validation = Validation::new(header.alg);
                validation.validate_exp = true;
                validation.validate_nbf = true;
                validation.leeway = 60;
                
                if let Some(aud) = expected_audience {
                    validation.set_audience(&[aud]);
                }
                if let Some(iss) = expected_issuer {
                    validation.set_issuer(&[iss]);
                }
                
                let decoding_key = if validation_key.starts_with("-----BEGIN") {
                    match header.alg {
                        Algorithm::RS256 | Algorithm::RS384 | Algorithm::RS512 => {
                            DecodingKey::from_rsa_pem(validation_key.as_bytes())
                                .unwrap_or_else(|_| DecodingKey::from_secret(validation_key.as_bytes()))
                        },
                        Algorithm::ES256 | Algorithm::ES384 => {
                            DecodingKey::from_ec_pem(validation_key.as_bytes())
                                .unwrap_or_else(|_| DecodingKey::from_secret(validation_key.as_bytes()))
                        },
                        _ => DecodingKey::from_secret(validation_key.as_bytes()),
                    }
                } else {
                    DecodingKey::from_secret(validation_key.as_bytes())
                };
                
                match decode::<TokenClaims>(&token.access_token, &decoding_key, &validation) {
                    Ok(_) => {
                        debug!("JWT signature validation successful");
                        Ok(true)
                    }
                    Err(e) => {
                        error!("JWT signature validation failed: {}", e);
                        Ok(false)
                    }
                }
            } else {
                // Opaque token - can only validate expiration
                Ok(true)
            }
        } else {
            // No validation key - basic validation only
            Ok(true)
        }
    }

    /// Revoke a token (if endpoint supports it)
    pub async fn revoke_token(
        &self,
        client: &Client,
        endpoint: &str,
        token: &str,
        token_type_hint: Option<&str>,
    ) -> Result<()> {
        let revoke_url = format!("{}/oauth/revoke", endpoint.trim_end_matches('/'));
        
        let mut params = vec![
            ("token", token.to_string()),
            ("client_id", self.client_id.clone()),
        ];
        
        if let Some(hint) = token_type_hint {
            params.push(("token_type_hint", hint.to_string()));
        }

        debug!("Revoking token");
        
        let response = client
            .post(&revoke_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await
            .map_err(McpError::NetworkError)?;

        if response.status().is_success() {
            info!("Token revoked successfully");
            self.clear_cache().await;
            Ok(())
        } else {
            let error_text = response.text().await.unwrap_or_default();
            Err(McpError::AuthenticationFailed(format!(
                "Token revocation failed: {}", error_text
            )))
        }
    }

    /// Generate authorization URL for authorization code flow
    pub async fn get_authorization_url(
        &self,
        client: &Client,
        endpoint: &str,
        redirect_uri: &str,
        scopes: &[String],
        state: Option<&str>,
        use_pkce: bool,
    ) -> Result<(String, Option<PkceChallenge>)> {
        // Try to discover authorization endpoint
        let auth_endpoint = match self.discover_endpoints(client, endpoint).await {
            Ok(discovery) => discovery.authorization_endpoint,
            Err(_) => format!("{}/oauth/authorize", endpoint.trim_end_matches('/'))
        };

        let mut url = Url::parse(&auth_endpoint)
            .map_err(|e| McpError::InvalidUrl(e))?;

        {
            let mut query = url.query_pairs_mut();
            query.append_pair("response_type", "code");
            query.append_pair("client_id", &self.client_id);
            query.append_pair("redirect_uri", redirect_uri);
            
            if !scopes.is_empty() {
                query.append_pair("scope", &scopes.join(" "));
            }
            
            if let Some(state_val) = state {
                query.append_pair("state", state_val);
            }
        }

        let pkce_challenge = if use_pkce {
            let challenge = PkceChallenge::new();
            url.query_pairs_mut()
                .append_pair("code_challenge", &challenge.code_challenge)
                .append_pair("code_challenge_method", &challenge.code_challenge_method);
            Some(challenge)
        } else {
            None
        };

        Ok((url.to_string(), pkce_challenge))
    }
}

impl Default for AuthHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use tokio_test;

    #[test]
    fn test_token_expiry_check() {
        let token = AuthToken {
            access_token: "test".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Utc::now() + Duration::seconds(300),
            refresh_token: None,
            scope: vec![],
            claims: None,
            id_token: None,
            issued_at: Utc::now(),
            last_refresh: None,
            refresh_count: 0,
        };

        assert!(token.is_valid());
        assert!(token.needs_refresh(600)); // Should need refresh if threshold > remaining time
        assert!(!token.needs_refresh(200)); // Should not need refresh if threshold < remaining time
    }

    #[test]
    fn test_grant_type_display() {
        assert_eq!(GrantType::ClientCredentials.to_string(), "client_credentials");
        assert_eq!(GrantType::Password.to_string(), "password");
        assert_eq!(GrantType::AuthorizationCode.to_string(), "authorization_code");
        assert_eq!(GrantType::RefreshToken.to_string(), "refresh_token");
        assert_eq!(GrantType::DeviceCode.to_string(), "urn:ietf:params:oauth:grant-type:device_code");
        assert_eq!(GrantType::JwtBearer.to_string(), "urn:ietf:params:oauth:grant-type:jwt-bearer");
    }

    #[test]
    fn test_pkce_challenge() {
        let challenge = PkceChallenge::new();
        assert!(!challenge.code_verifier.is_empty());
        assert!(!challenge.code_challenge.is_empty());
        assert_eq!(challenge.code_challenge_method, "S256");
        
        // Verify code challenge is different from verifier
        assert_ne!(challenge.code_verifier, challenge.code_challenge);
    }

    #[test]
    fn test_token_refresh_count() {
        let mut token = AuthToken {
            access_token: "test".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Utc::now() + Duration::seconds(300),
            refresh_token: Some("refresh".to_string()),
            scope: vec![],
            claims: None,
            id_token: None,
            issued_at: Utc::now(),
            last_refresh: None,
            refresh_count: 0,
        };

        assert!(token.can_refresh());
        
        // Simulate multiple refreshes
        for i in 1..=10 {
            let new_token = AuthToken {
                access_token: format!("new_token_{}", i),
                token_type: "Bearer".to_string(),
                expires_at: Utc::now() + Duration::seconds(300),
                refresh_token: Some("new_refresh".to_string()),
                scope: vec![],
                claims: None,
                id_token: None,
                issued_at: Utc::now(),
                last_refresh: None,
                refresh_count: 0,
            };
            
            token.update_from_refresh(new_token);
        }
        
        assert_eq!(token.refresh_count, 10);
        assert!(!token.can_refresh()); // Should not allow more refreshes
    }

    #[tokio::test]
    async fn test_auth_handler_creation() {
        let handler = AuthHandler::new();
        assert!(!handler.client_id.is_empty());
        assert!(handler.validation_key.is_none());
        
        let custom_handler = AuthHandler::with_config(
            "custom_client".to_string(),
            Some("validation_key".to_string()),
        );
        assert_eq!(custom_handler.client_id, "custom_client");
        assert!(custom_handler.validation_key.is_some());
    }

    #[tokio::test]
    async fn test_token_caching() {
        let handler = AuthHandler::new();
        
        // Should return None initially
        assert!(handler.get_cached_token().await.is_none());
        
        let token = AuthToken {
            access_token: "test".to_string(),
            token_type: "Bearer".to_string(),
            expires_at: Utc::now() + Duration::seconds(300),
            refresh_token: None,
            scope: vec![],
            claims: None,
            id_token: None,
            issued_at: Utc::now(),
            last_refresh: None,
            refresh_count: 0,
        };
        
        handler.cache_token(token.clone()).await;
        let cached = handler.get_cached_token().await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().access_token, "test");
        
        handler.clear_cache().await;
        assert!(handler.get_cached_token().await.is_none());
    }
}