//! # Circuit Breaker Implementation
//!
//! Fault tolerance pattern implementation with automatic recovery,
//! failure detection, and performance monitoring.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Circuit breaker states
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitBreakerState {
    /// Circuit is closed - requests pass through
    Closed,
    /// Circuit is open - requests are rejected
    Open,
    /// Circuit is half-open - testing if service recovered
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failure threshold to open circuit
    pub failure_threshold: usize,
    /// Success threshold to close circuit from half-open
    pub success_threshold: usize,
    /// Timeout before trying half-open from open
    pub timeout: Duration,
    /// Maximum number of calls in half-open state
    pub half_open_max_calls: usize,
    /// Sliding window size for failure rate calculation
    pub sliding_window_size: usize,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(30),
            half_open_max_calls: 3,
            sliding_window_size: 10,
        }
    }
}

/// Circuit breaker statistics
#[derive(Debug, Default, Clone)]
pub struct CircuitBreakerStats {
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Rejected requests (when circuit open)
    pub rejected_requests: u64,
    /// Current failure rate
    pub failure_rate: f64,
    /// Time in current state
    pub time_in_state: Duration,
    /// Last state change timestamp
    pub last_state_change: Option<Instant>,
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    /// Circuit breaker name/identifier
    name: String,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Current state
    state: Arc<RwLock<CircuitBreakerState>>,
    /// Statistics
    stats: Arc<RwLock<CircuitBreakerStats>>,
    /// Failure count in current window
    failure_count: Arc<RwLock<usize>>,
    /// Success count in half-open state
    success_count: Arc<RwLock<usize>>,
    /// Last failure time
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    /// Request results sliding window
    request_window: Arc<RwLock<Vec<bool>>>, // true = success, false = failure
}

impl CircuitBreaker {
    /// Create new circuit breaker
    pub fn new(name: String, config: CircuitBreakerConfig) -> Self {
        Self {
            name,
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
            stats: Arc::new(RwLock::new(CircuitBreakerStats::default())),
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            request_window: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Execute operation with circuit breaker protection
    pub async fn call<F, T, E>(&self, operation: F) -> Result<T, CircuitBreakerError<E>>
    where
        F: std::future::Future<Output = Result<T, E>>,
    {
        // Check if request should be allowed
        if !self.allow_request().await {
            self.record_rejected_request().await;
            return Err(CircuitBreakerError::CircuitOpen {
                name: self.name.clone(),
            });
        }
        
        // Execute operation
        let start_time = Instant::now();
        let result = operation.await;
        let duration = start_time.elapsed();
        
        // Record result
        match &result {
            Ok(_) => self.record_success(duration).await,
            Err(_) => self.record_failure(duration).await,
        }
        
        result.map_err(CircuitBreakerError::OperationFailed)
    }
    
    /// Check if request should be allowed
    async fn allow_request(&self) -> bool {
        let state = self.state.read().await;
        
        match *state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                drop(state);
                self.should_attempt_reset().await
            }
            CircuitBreakerState::HalfOpen => {
                drop(state);
                self.should_allow_half_open_request().await
            }
        }
    }
    
    /// Check if circuit should attempt reset from open state
    async fn should_attempt_reset(&self) -> bool {
        let last_failure = self.last_failure_time.read().await;
        
        match *last_failure {
            Some(last_time) => {
                if last_time.elapsed() >= self.config.timeout {
                    drop(last_failure);
                    self.transition_to_half_open().await;
                    true
                } else {
                    false
                }
            }
            None => {
                drop(last_failure);
                self.transition_to_half_open().await;
                true
            }
        }
    }
    
    /// Check if request should be allowed in half-open state
    async fn should_allow_half_open_request(&self) -> bool {
        let stats = self.stats.read().await;
        let current_requests = stats.total_requests;
        drop(stats);
        
        // Allow limited number of requests in half-open state
        current_requests % self.config.half_open_max_calls as u64 == 0
    }
    
    /// Record successful operation
    async fn record_success(&self, _duration: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.successful_requests += 1;
        
        // Update request window
        let mut window = self.request_window.write().await;
        window.push(true);
        if window.len() > self.config.sliding_window_size {
            window.remove(0);
        }
        
        // Update failure rate
        let failures = window.iter().filter(|&&result| !result).count();
        stats.failure_rate = failures as f64 / window.len() as f64;
        
        drop(stats);
        drop(window);
        
        let state = self.state.read().await;
        match *state {
            CircuitBreakerState::HalfOpen => {
                drop(state);
                let mut success_count = self.success_count.write().await;
                *success_count += 1;
                
                if *success_count >= self.config.success_threshold {
                    drop(success_count);
                    self.transition_to_closed().await;
                }
            }
            _ => {}
        }
    }
    
    /// Record failed operation
    async fn record_failure(&self, _duration: Duration) {
        let mut stats = self.stats.write().await;
        stats.total_requests += 1;
        stats.failed_requests += 1;
        
        // Update request window
        let mut window = self.request_window.write().await;
        window.push(false);
        if window.len() > self.config.sliding_window_size {
            window.remove(0);
        }
        
        // Update failure rate
        let failures = window.iter().filter(|&&result| !result).count();
        stats.failure_rate = failures as f64 / window.len() as f64;
        
        drop(stats);
        drop(window);
        
        let mut failure_count = self.failure_count.write().await;
        *failure_count += 1;
        
        let mut last_failure_time = self.last_failure_time.write().await;
        *last_failure_time = Some(Instant::now());
        
        let current_failures = *failure_count;
        drop(failure_count);
        drop(last_failure_time);
        
        let state = self.state.read().await;
        match *state {
            CircuitBreakerState::Closed => {
                if current_failures >= self.config.failure_threshold {
                    drop(state);
                    self.transition_to_open().await;
                }
            }
            CircuitBreakerState::HalfOpen => {
                drop(state);
                self.transition_to_open().await;
            }
            CircuitBreakerState::Open => {}
        }
    }
    
    /// Record rejected request
    async fn record_rejected_request(&self) {
        let mut stats = self.stats.write().await;
        stats.rejected_requests += 1;
    }
    
    /// Transition to closed state
    async fn transition_to_closed(&self) {
        info!("Circuit breaker '{}' transitioning to CLOSED", self.name);
        
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::Closed;
        
        // Reset counters
        let mut failure_count = self.failure_count.write().await;
        *failure_count = 0;
        
        let mut success_count = self.success_count.write().await;
        *success_count = 0;
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.last_state_change = Some(Instant::now());
    }
    
    /// Transition to open state
    async fn transition_to_open(&self) {
        warn!("Circuit breaker '{}' transitioning to OPEN", self.name);
        
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::Open;
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.last_state_change = Some(Instant::now());
    }
    
    /// Transition to half-open state
    async fn transition_to_half_open(&self) {
        info!("Circuit breaker '{}' transitioning to HALF_OPEN", self.name);
        
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::HalfOpen;
        
        // Reset success counter
        let mut success_count = self.success_count.write().await;
        *success_count = 0;
        
        // Update stats
        let mut stats = self.stats.write().await;
        stats.last_state_change = Some(Instant::now());
    }
    
    /// Get current state
    pub async fn state(&self) -> CircuitBreakerState {
        self.state.read().await.clone()
    }
    
    /// Get current statistics
    pub async fn stats(&self) -> CircuitBreakerStats {
        let mut stats = self.stats.read().await.clone();
        
        // Update time in current state
        if let Some(last_change) = stats.last_state_change {
            stats.time_in_state = last_change.elapsed();
        }
        
        stats
    }
    
    /// Force circuit breaker to specific state (for testing/admin)
    pub async fn force_state(&self, new_state: CircuitBreakerState) {
        warn!("Forcing circuit breaker '{}' to state: {:?}", self.name, new_state);
        
        let mut state = self.state.write().await;
        *state = new_state;
        
        let mut stats = self.stats.write().await;
        stats.last_state_change = Some(Instant::now());
    }
    
    /// Reset circuit breaker statistics
    pub async fn reset_stats(&self) {
        info!("Resetting statistics for circuit breaker '{}'", self.name);
        
        let mut stats = self.stats.write().await;
        *stats = CircuitBreakerStats::default();
        
        let mut window = self.request_window.write().await;
        window.clear();
        
        let mut failure_count = self.failure_count.write().await;
        *failure_count = 0;
        
        let mut success_count = self.success_count.write().await;
        *success_count = 0;
        
        let mut last_failure_time = self.last_failure_time.write().await;
        *last_failure_time = None;
    }
}

/// Circuit breaker error types
#[derive(thiserror::Error, Debug)]
pub enum CircuitBreakerError<E> {
    /// Circuit breaker is open
    #[error("Circuit breaker '{name}' is open - request rejected")]
    CircuitOpen { name: String },
    
    /// Operation failed
    #[error("Operation failed: {0}")]
    OperationFailed(E),
}

/// Circuit breaker registry for managing multiple circuit breakers
pub struct CircuitBreakerRegistry {
    breakers: Arc<RwLock<std::collections::HashMap<String, Arc<CircuitBreaker>>>>,
}

impl CircuitBreakerRegistry {
    /// Create new circuit breaker registry
    pub fn new() -> Self {
        Self {
            breakers: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }
    
    /// Get or create circuit breaker
    pub async fn get_or_create(&self, name: &str, config: Option<CircuitBreakerConfig>) -> Arc<CircuitBreaker> {
        let breakers = self.breakers.read().await;
        
        if let Some(breaker) = breakers.get(name) {
            return breaker.clone();
        }
        
        drop(breakers);
        
        let mut breakers = self.breakers.write().await;
        
        // Double-check pattern
        if let Some(breaker) = breakers.get(name) {
            return breaker.clone();
        }
        
        let breaker = Arc::new(CircuitBreaker::new(
            name.to_string(),
            config.unwrap_or_default(),
        ));
        
        breakers.insert(name.to_string(), breaker.clone());
        
        info!("Created new circuit breaker: {}", name);
        breaker
    }
    
    /// Get circuit breaker by name
    pub async fn get(&self, name: &str) -> Option<Arc<CircuitBreaker>> {
        let breakers = self.breakers.read().await;
        breakers.get(name).cloned()
    }
    
    /// Remove circuit breaker
    pub async fn remove(&self, name: &str) -> bool {
        let mut breakers = self.breakers.write().await;
        breakers.remove(name).is_some()
    }
    
    /// Get all circuit breaker names
    pub async fn list_names(&self) -> Vec<String> {
        let breakers = self.breakers.read().await;
        breakers.keys().cloned().collect()
    }
    
    /// Get statistics for all circuit breakers
    pub async fn get_all_stats(&self) -> std::collections::HashMap<String, (CircuitBreakerState, CircuitBreakerStats)> {
        let breakers = self.breakers.read().await;
        let mut stats = std::collections::HashMap::new();
        
        for (name, breaker) in breakers.iter() {
            let state = breaker.state().await;
            let breaker_stats = breaker.stats().await;
            stats.insert(name.clone(), (state, breaker_stats));
        }
        
        stats
    }
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    async fn failing_operation() -> Result<(), &'static str> {
        Err("Operation failed")
    }
    
    async fn succeeding_operation() -> Result<String, &'static str> {
        Ok("Success".to_string())
    }
    
    #[tokio::test]
    async fn test_circuit_breaker_closed_to_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test".to_string(), config);
        
        // Initially closed
        assert_eq!(breaker.state().await, CircuitBreakerState::Closed);
        
        // Cause failures to open circuit
        for _ in 0..3 {
            let _ = breaker.call(failing_operation()).await;
        }
        
        // Should be open now
        assert_eq!(breaker.state().await, CircuitBreakerState::Open);
        
        // Further requests should be rejected
        let result = breaker.call(succeeding_operation()).await;
        assert!(matches!(result, Err(CircuitBreakerError::CircuitOpen { .. })));
    }
    
    #[tokio::test]
    async fn test_circuit_breaker_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
            ..Default::default()
        };
        let breaker = CircuitBreaker::new("test".to_string(), config);
        
        // Cause failures to open circuit
        for _ in 0..2 {
            let _ = breaker.call(failing_operation()).await;
        }
        
        assert_eq!(breaker.state().await, CircuitBreakerState::Open);
        
        // Wait for timeout
        sleep(Duration::from_millis(150)).await;
        
        // Next request should transition to half-open
        let result = breaker.call(succeeding_operation()).await;
        assert!(result.is_ok());
        assert_eq!(breaker.state().await, CircuitBreakerState::HalfOpen);
        
        // Another success should close the circuit
        let result = breaker.call(succeeding_operation()).await;
        assert!(result.is_ok());
        assert_eq!(breaker.state().await, CircuitBreakerState::Closed);
    }
    
    #[tokio::test]
    async fn test_circuit_breaker_registry() {
        let registry = CircuitBreakerRegistry::new();
        
        let breaker1 = registry.get_or_create("test1", None).await;
        let breaker2 = registry.get_or_create("test2", None).await;
        
        assert_ne!(breaker1.name, breaker2.name);
        
        let names = registry.list_names().await;
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"test1".to_string()));
        assert!(names.contains(&"test2".to_string()));
        
        // Getting same breaker should return same instance
        let breaker1_again = registry.get_or_create("test1", None).await;
        assert!(Arc::ptr_eq(&breaker1, &breaker1_again));
    }
    
    #[tokio::test]
    async fn test_circuit_breaker_stats() {
        let breaker = CircuitBreaker::new("test".to_string(), CircuitBreakerConfig::default());
        
        // Record some operations
        let _ = breaker.call(succeeding_operation()).await;
        let _ = breaker.call(failing_operation()).await;
        let _ = breaker.call(succeeding_operation()).await;
        
        let stats = breaker.stats().await;
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_requests, 2);
        assert_eq!(stats.failed_requests, 1);
        assert_eq!(stats.failure_rate, 1.0 / 3.0);
    }
    
    #[tokio::test]
    async fn test_force_state() {
        let breaker = CircuitBreaker::new("test".to_string(), CircuitBreakerConfig::default());
        
        assert_eq!(breaker.state().await, CircuitBreakerState::Closed);
        
        breaker.force_state(CircuitBreakerState::Open).await;
        assert_eq!(breaker.state().await, CircuitBreakerState::Open);
        
        // Request should be rejected
        let result = breaker.call(succeeding_operation()).await;
        assert!(matches!(result, Err(CircuitBreakerError::CircuitOpen { .. })));
    }
}
