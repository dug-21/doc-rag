// Rate limiting implementation using Redis backend
// Implements token bucket algorithm with sliding window

use crate::security::{SecurityError, SecurityResult, RateLimitConfig};
use redis::{Client, Commands, Connection};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};
use tracing::{debug, warn, error};

pub struct RateLimiter {
    config: RateLimitConfig,
    redis_client: Client,
}

#[derive(Debug, Clone)]
pub struct RateLimitInfo {
    pub remaining: u32,
    pub reset_time: u64,
    pub limit: u32,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> SecurityResult<Self> {
        let redis_client = Client::open(config.redis_url.as_str())
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to connect to Redis: {}", e)))?;
        
        Ok(Self {
            config,
            redis_client,
        })
    }

    pub async fn check_rate_limit(&self, identifier: &str) -> SecurityResult<RateLimitInfo> {
        if !self.config.enabled {
            return Ok(RateLimitInfo {
                remaining: self.config.requests_per_minute,
                reset_time: 0,
                limit: self.config.requests_per_minute,
            });
        }

        let mut conn = self.get_connection().await?;
        let current_time = self.current_time_seconds();
        let window_start = current_time - 60; // 1 minute window
        
        let key = format!("rate_limit:{}", identifier);
        let requests = self.count_requests_in_window(&mut conn, &key, window_start, current_time).await?;
        
        if requests >= self.config.requests_per_minute {
            let reset_time = current_time + 60;
            warn!("Rate limit exceeded for identifier: {}", identifier);
            return Err(SecurityError::RateLimitExceeded(
                format!("Rate limit of {} requests per minute exceeded", self.config.requests_per_minute)
            ));
        }

        // Record this request
        self.record_request(&mut conn, &key, current_time).await?;
        
        let remaining = self.config.requests_per_minute - requests - 1;
        let reset_time = current_time + (60 - (current_time % 60));
        
        debug!("Rate limit check passed for {}: {} remaining", identifier, remaining);
        
        Ok(RateLimitInfo {
            remaining,
            reset_time,
            limit: self.config.requests_per_minute,
        })
    }

    pub async fn check_burst_limit(&self, identifier: &str) -> SecurityResult<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut conn = self.get_connection().await?;
        let current_time = self.current_time_seconds();
        let burst_window = current_time - 10; // 10 second burst window
        
        let key = format!("burst_limit:{}", identifier);
        let requests = self.count_requests_in_window(&mut conn, &key, burst_window, current_time).await?;
        
        if requests >= self.config.burst_size {
            warn!("Burst limit exceeded for identifier: {}", identifier);
            return Err(SecurityError::RateLimitExceeded(
                format!("Burst limit of {} requests per 10 seconds exceeded", self.config.burst_size)
            ));
        }

        // Record this request for burst tracking
        self.record_request(&mut conn, &key, current_time).await?;
        
        Ok(())
    }

    pub async fn whitelist_identifier(&self, identifier: &str, duration_seconds: u64) -> SecurityResult<()> {
        let mut conn = self.get_connection().await?;
        let key = format!("whitelist:{}", identifier);
        let _: () = conn.set_ex(&key, "1", duration_seconds as usize)
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to whitelist identifier: {}", e)))?;
        
        debug!("Whitelisted identifier {} for {} seconds", identifier, duration_seconds);
        Ok(())
    }

    pub async fn blacklist_identifier(&self, identifier: &str, duration_seconds: u64) -> SecurityResult<()> {
        let mut conn = self.get_connection().await?;
        let key = format!("blacklist:{}", identifier);
        let _: () = conn.set_ex(&key, "1", duration_seconds as usize)
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to blacklist identifier: {}", e)))?;
        
        warn!("Blacklisted identifier {} for {} seconds", identifier, duration_seconds);
        Ok(())
    }

    pub async fn is_blacklisted(&self, identifier: &str) -> SecurityResult<bool> {
        let mut conn = self.get_connection().await?;
        let key = format!("blacklist:{}", identifier);
        let exists: bool = conn.exists(&key)
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to check blacklist: {}", e)))?;
        
        if exists {
            warn!("Blocked request from blacklisted identifier: {}", identifier);
        }
        
        Ok(exists)
    }

    pub async fn is_whitelisted(&self, identifier: &str) -> SecurityResult<bool> {
        let mut conn = self.get_connection().await?;
        let key = format!("whitelist:{}", identifier);
        let exists: bool = conn.exists(&key)
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to check whitelist: {}", e)))?;
        
        Ok(exists)
    }

    async fn get_connection(&self) -> SecurityResult<Connection> {
        self.redis_client.get_connection()
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to get Redis connection: {}", e)))
    }

    async fn count_requests_in_window(&self, conn: &mut Connection, key: &str, start: u64, end: u64) -> SecurityResult<u32> {
        let count: u32 = conn.zcount(key, start, end)
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to count requests: {}", e)))?;
        
        Ok(count)
    }

    async fn record_request(&self, conn: &mut Connection, key: &str, timestamp: u64) -> SecurityResult<()> {
        // Add request with timestamp as both score and member
        let _: () = conn.zadd(key, timestamp, timestamp)
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to record request: {}", e)))?;
        
        // Clean up old entries (older than 5 minutes)
        let cleanup_time = timestamp - 300;
        let _: u32 = conn.zremrangebyscore(key, 0, cleanup_time)
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to cleanup old requests: {}", e)))?;
        
        // Set TTL on the key to prevent memory leaks
        let _: () = conn.expire(key, 3600) // 1 hour
            .map_err(|e| SecurityError::PolicyViolation(format!("Failed to set TTL: {}", e)))?;
        
        Ok(())
    }

    fn current_time_seconds(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
}

// Middleware for extracting client identifiers
pub fn extract_client_identifier(headers: &std::collections::HashMap<String, String>, ip: &str) -> String {
    // Try to get API key first
    if let Some(api_key) = headers.get("x-api-key") {
        return format!("api:{}", api_key);
    }
    
    // Try to get user ID from JWT
    if let Some(auth_header) = headers.get("authorization") {
        if let Some(user_id) = extract_user_id_from_jwt(auth_header) {
            return format!("user:{}", user_id);
        }
    }
    
    // Fall back to IP address
    format!("ip:{}", ip)
}

fn extract_user_id_from_jwt(auth_header: &str) -> Option<String> {
    use base64::{Engine as _, engine::general_purpose};
    use serde_json::Value;
    
    if !auth_header.starts_with("Bearer ") {
        return None;
    }
    
    let token = auth_header.strip_prefix("Bearer ")?;
    
    // Split JWT into parts (header.payload.signature)
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    
    // Decode the payload (second part)
    let payload_part = parts[1];
    
    // Add padding if necessary for base64 decoding
    let padded_payload = match payload_part.len() % 4 {
        0 => payload_part.to_string(),
        2 => format!("{}==", payload_part),
        3 => format!("{}=", payload_part),
        _ => return None,
    };
    
    // Decode base64url (JWT uses base64url encoding)
    let decoded_bytes = general_purpose::URL_SAFE_NO_PAD
        .decode(&padded_payload)
        .ok()?;
    
    // Parse JSON payload
    let payload_str = String::from_utf8(decoded_bytes).ok()?;
    let payload_json: Value = serde_json::from_str(&payload_str).ok()?;
    
    // Extract user ID from various common JWT fields
    if let Some(user_id) = payload_json.get("sub") {
        if let Some(user_id_str) = user_id.as_str() {
            return Some(user_id_str.to_string());
        }
    }
    
    if let Some(user_id) = payload_json.get("user_id") {
        if let Some(user_id_str) = user_id.as_str() {
            return Some(user_id_str.to_string());
        }
    }
    
    if let Some(user_id) = payload_json.get("uid") {
        if let Some(user_id_str) = user_id.as_str() {
            return Some(user_id_str.to_string());
        }
    }
    
    if let Some(email) = payload_json.get("email") {
        if let Some(email_str) = email.as_str() {
            return Some(email_str.to_string());
        }
    }
    
    None
}

// Rate limiting metrics for monitoring
pub struct RateLimitMetrics {
    pub total_requests: u64,
    pub blocked_requests: u64,
    pub unique_identifiers: u64,
}

impl RateLimiter {
    pub async fn get_metrics(&self) -> SecurityResult<RateLimitMetrics> {
        let mut conn = self.get_connection().await?;
        
        // Get total request count from a global counter
        let total: u64 = conn.get("rate_limit:total_requests").unwrap_or(0);
        let blocked: u64 = conn.get("rate_limit:blocked_requests").unwrap_or(0);
        
        // Count unique identifiers (this is expensive, consider sampling in production)
        let keys: Vec<String> = conn.keys("rate_limit:*").unwrap_or_default();
        let unique_identifiers = keys.len() as u64;
        
        Ok(RateLimitMetrics {
            total_requests: total,
            blocked_requests: blocked,
            unique_identifiers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a running Redis instance
    // In CI/CD, use testcontainers or similar for integration tests
    
    #[tokio::test]
    #[ignore] // Remove this to run with Redis
    async fn test_rate_limiting() {
        let config = RateLimitConfig {
            enabled: true,
            requests_per_minute: 5,
            burst_size: 2,
            redis_url: "redis://localhost:6379".to_string(),
        };
        
        let limiter = RateLimiter::new(config).unwrap();
        let identifier = "test_user";
        
        // Should allow initial requests
        for i in 0..5 {
            let result = limiter.check_rate_limit(identifier).await;
            assert!(result.is_ok(), "Request {} should be allowed", i);
        }
        
        // Should block the 6th request
        let result = limiter.check_rate_limit(identifier).await;
        assert!(result.is_err(), "6th request should be blocked");
    }
}