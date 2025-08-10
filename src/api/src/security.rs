use crate::config::SecurityConfig;
use std::time::Duration;

// Placeholder for rate limiting - would use tower-governor in production
pub struct RateLimitLayer;

impl RateLimitLayer {
    pub fn new(config: &SecurityConfig) -> Self {
        // Placeholder implementation - in production would use proper rate limiting
        Self
    }
}

// Extension to SecurityConfig for rate limiting
impl SecurityConfig {
    pub fn burst_size(&self) -> Option<u32> {
        Some(self.rate_limit_requests / 4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SecurityConfig;

    #[test]
    fn test_rate_limit_layer_creation() {
        let config = SecurityConfig {
            jwt_secret: "test-secret-key-must-be-at-least-32-characters-long".to_string(),
            jwt_expiration_hours: 24,
            password_min_length: 8,
            rate_limit_requests: 100,
            rate_limit_window_secs: 60,
            session_timeout_hours: 24,
            enable_cors: true,
            enable_csrf: true,
        };

        // This should not panic
        let _layer = RateLimitLayer::new(&config);
    }
}