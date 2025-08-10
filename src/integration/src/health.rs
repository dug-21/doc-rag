//! # Health Monitoring System
//!
//! Comprehensive health monitoring for all system components with
//! automatic recovery, alerting, and performance tracking.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{info, warn, error, instrument};
use uuid::Uuid;

use crate::{
    Result, IntegrationError, ServiceDiscovery, ComponentHealthStatus,
    HealthStatus, ComponentHealth, SystemHealth,
};

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Check interval
    pub interval: Duration,
    /// Request timeout
    pub timeout: Duration,
    /// Unhealthy threshold (consecutive failures)
    pub unhealthy_threshold: usize,
    /// Recovery threshold (consecutive successes)
    pub recovery_threshold: usize,
    /// Enable automatic recovery attempts
    pub auto_recovery: bool,
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    /// Component name
    pub component: String,
    /// Check status
    pub status: ComponentHealthStatus,
    /// Response time
    pub response_time: Duration,
    /// Error message if unhealthy
    pub error: Option<String>,
    /// Check timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Health event types
#[derive(Debug, Clone)]
pub enum HealthEvent {
    /// Component became healthy
    ComponentHealthy {
        component: String,
        previous_status: ComponentHealthStatus,
    },
    /// Component became unhealthy
    ComponentUnhealthy {
        component: String,
        error: String,
        consecutive_failures: usize,
    },
    /// Component recovered
    ComponentRecovered {
        component: String,
        downtime: Duration,
    },
    /// System health changed
    SystemHealthChanged {
        old_status: HealthStatus,
        new_status: HealthStatus,
        affected_components: Vec<String>,
    },
    /// Performance degradation detected
    PerformanceDegraded {
        component: String,
        metric: String,
        threshold: f64,
        current: f64,
    },
}

/// Component health tracker
#[derive(Debug, Clone)]
struct ComponentHealthTracker {
    /// Component name
    name: String,
    /// Current status
    current_status: ComponentHealthStatus,
    /// Last successful check
    last_success: Option<chrono::DateTime<chrono::Utc>>,
    /// Last failure
    last_failure: Option<chrono::DateTime<chrono::Utc>>,
    /// Consecutive failures
    consecutive_failures: usize,
    /// Consecutive successes
    consecutive_successes: usize,
    /// Average response time
    avg_response_time: Duration,
    /// Total checks performed
    total_checks: u64,
    /// Success rate
    success_rate: f64,
    /// Health check configuration
    config: HealthCheckConfig,
}

/// Main health monitor
pub struct HealthMonitor {
    /// Monitor ID
    id: Uuid,
    /// Configuration
    config: Arc<crate::IntegrationConfig>,
    /// Service discovery
    service_discovery: Arc<ServiceDiscovery>,
    /// Component trackers
    trackers: Arc<RwLock<HashMap<String, ComponentHealthTracker>>>,
    /// Health event channel
    event_tx: mpsc::UnboundedSender<HealthEvent>,
    event_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<HealthEvent>>>>,
    /// System health cache
    system_health_cache: Arc<RwLock<SystemHealth>>,
    /// Health check configuration
    health_config: HealthCheckConfig,
}

impl HealthMonitor {
    /// Create new health monitor
    pub async fn new(
        config: Arc<crate::IntegrationConfig>,
        service_discovery: Arc<ServiceDiscovery>,
    ) -> Result<Self> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        
        let health_config = HealthCheckConfig {
            interval: Duration::from_secs(config.health_check_interval_secs.unwrap_or(30)),
            timeout: Duration::from_secs(10),
            unhealthy_threshold: 3,
            recovery_threshold: 2,
            auto_recovery: true,
        };
        
        let system_health = SystemHealth {
            system_id: Uuid::new_v4(),
            status: HealthStatus::Starting,
            components: HashMap::new(),
            uptime: Duration::from_secs(0),
            timestamp: chrono::Utc::now(),
        };
        
        Ok(Self {
            id: Uuid::new_v4(),
            config,
            service_discovery,
            trackers: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            event_rx: Arc::new(RwLock::new(Some(event_rx))),
            system_health_cache: Arc::new(RwLock::new(system_health)),
            health_config,
        })
    }
    
    /// Initialize health monitor
    #[instrument(skip(self))]
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing Health Monitor: {}", self.id);
        
        // Initialize component trackers
        self.initialize_component_trackers().await?;
        
        info!("Health Monitor initialized successfully");
        Ok(())
    }
    
    /// Start health monitor
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting Health Monitor...");
        
        // Start event processing
        let event_rx = self.event_rx.write().await.take()
            .ok_or_else(|| IntegrationError::Internal("Event receiver already taken".to_string()))?;
        
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.process_health_events(event_rx).await;
        });
        
        // Start health checking
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.run_health_checks().await;
        });
        
        // Start system health aggregation
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.aggregate_system_health().await;
        });
        
        // Update system status to healthy
        {
            let mut system_health = self.system_health_cache.write().await;
            system_health.status = HealthStatus::Healthy;
        }
        
        info!("Health Monitor started successfully");
        Ok(())
    }
    
    /// Stop health monitor
    #[instrument(skip(self))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Health Monitor...");
        
        // Update system status to stopping
        {
            let mut system_health = self.system_health_cache.write().await;
            system_health.status = HealthStatus::Stopping;
        }
        
        info!("Health Monitor stopped successfully");
        Ok(())
    }
    
    /// Get system health
    pub async fn system_health(&self) -> SystemHealth {
        self.system_health_cache.read().await.clone()
    }
    
    /// Get component health
    pub async fn component_health(&self, component: &str) -> Option<ComponentHealth> {
        let trackers = self.trackers.read().await;
        trackers.get(component).map(|tracker| ComponentHealth {
            name: tracker.name.clone(),
            status: self.status_from_component_status(&tracker.current_status),
            latency_ms: tracker.avg_response_time.as_millis() as u64,
            error: None,
            last_success: tracker.last_success,
        })
    }
    
    /// Register component for health monitoring
    pub async fn register_component(&self, name: &str) -> Result<()> {
        info!("Registering component for health monitoring: {}", name);
        
        let tracker = ComponentHealthTracker {
            name: name.to_string(),
            current_status: ComponentHealthStatus::Unknown,
            last_success: None,
            last_failure: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
            avg_response_time: Duration::from_millis(0),
            total_checks: 0,
            success_rate: 0.0,
            config: self.health_config.clone(),
        };
        
        {
            let mut trackers = self.trackers.write().await;
            trackers.insert(name.to_string(), tracker);
        }
        
        info!("Component registered for health monitoring: {}", name);
        Ok(())
    }
    
    /// Perform health check for a specific component
    pub async fn check_component_health(&self, component: &str) -> Result<HealthCheckResult> {
        let start = Instant::now();
        
        // Get service endpoint from service discovery
        let endpoint = self.service_discovery.get_service_endpoint(component).await
            .ok_or_else(|| IntegrationError::ComponentNotFound(component.to_string()))?;
        
        // Perform health check HTTP request
        let client = reqwest::Client::new();
        let health_url = format!("{}/health", endpoint);
        
        let result = tokio::time::timeout(
            self.health_config.timeout,
            client.get(&health_url).send()
        ).await;
        
        let response_time = start.elapsed();
        let timestamp = chrono::Utc::now();
        
        let check_result = match result {
            Ok(Ok(response)) if response.status().is_success() => {
                HealthCheckResult {
                    component: component.to_string(),
                    status: ComponentHealthStatus::Healthy,
                    response_time,
                    error: None,
                    timestamp,
                    metadata: HashMap::new(),
                }
            }
            Ok(Ok(response)) => {
                let status_code = response.status();
                HealthCheckResult {
                    component: component.to_string(),
                    status: ComponentHealthStatus::Unhealthy,
                    response_time,
                    error: Some(format!("HTTP {}", status_code)),
                    timestamp,
                    metadata: HashMap::new(),
                }
            }
            Ok(Err(e)) => {
                HealthCheckResult {
                    component: component.to_string(),
                    status: ComponentHealthStatus::Unhealthy,
                    response_time,
                    error: Some(e.to_string()),
                    timestamp,
                    metadata: HashMap::new(),
                }
            }
            Err(_) => {
                HealthCheckResult {
                    component: component.to_string(),
                    status: ComponentHealthStatus::Unhealthy,
                    response_time,
                    error: Some("Health check timeout".to_string()),
                    timestamp,
                    metadata: HashMap::new(),
                }
            }
        };
        
        // Update tracker
        self.update_component_tracker(&check_result).await;
        
        Ok(check_result)
    }
    
    /// Update component tracker with check result
    async fn update_component_tracker(&self, result: &HealthCheckResult) {
        let mut trackers = self.trackers.write().await;
        
        if let Some(tracker) = trackers.get_mut(&result.component) {
            let old_status = tracker.current_status.clone();
            tracker.total_checks += 1;
            
            match result.status {
                ComponentHealthStatus::Healthy => {
                    tracker.last_success = Some(result.timestamp);
                    tracker.consecutive_successes += 1;
                    tracker.consecutive_failures = 0;
                    
                    // Check if component recovered
                    if old_status != ComponentHealthStatus::Healthy {
                        if tracker.consecutive_successes >= tracker.config.recovery_threshold {
                            tracker.current_status = ComponentHealthStatus::Healthy;
                            
                            let downtime = tracker.last_failure
                                .map(|failure| result.timestamp.signed_duration_since(failure))
                                .and_then(|d| d.to_std().ok())
                                .unwrap_or_else(|| Duration::from_secs(0));
                            
                            if let Err(e) = self.event_tx.send(HealthEvent::ComponentRecovered {
                                component: result.component.clone(),
                                downtime,
                            }) {
                                warn!("Failed to send recovery event: {}", e);
                            }
                        }
                    } else {
                        tracker.current_status = ComponentHealthStatus::Healthy;
                    }
                }
                _ => {
                    tracker.last_failure = Some(result.timestamp);
                    tracker.consecutive_failures += 1;
                    tracker.consecutive_successes = 0;
                    
                    // Check if component became unhealthy
                    if tracker.consecutive_failures >= tracker.config.unhealthy_threshold {
                        if old_status != ComponentHealthStatus::Unhealthy {
                            tracker.current_status = ComponentHealthStatus::Unhealthy;
                            
                            if let Err(e) = self.event_tx.send(HealthEvent::ComponentUnhealthy {
                                component: result.component.clone(),
                                error: result.error.clone().unwrap_or_else(|| "Unknown error".to_string()),
                                consecutive_failures: tracker.consecutive_failures,
                            }) {
                                warn!("Failed to send unhealthy event: {}", e);
                            }
                        }
                    }
                }
            }
            
            // Update average response time
            let total_time = tracker.avg_response_time.as_millis() as f64 * (tracker.total_checks - 1) as f64;
            tracker.avg_response_time = Duration::from_millis(
                ((total_time + result.response_time.as_millis() as f64) / tracker.total_checks as f64) as u64
            );
            
            // Update success rate
            let successes = tracker.total_checks - tracker.consecutive_failures as u64;
            tracker.success_rate = successes as f64 / tracker.total_checks as f64;
        }
    }
    
    /// Initialize component trackers for known components
    async fn initialize_component_trackers(&self) -> Result<()> {
        let components = vec![
            "mcp-adapter",
            "chunker", 
            "embedder",
            "storage",
            "query-processor",
            "response-generator",
        ];
        
        for component in components {
            self.register_component(component).await?;
        }
        
        Ok(())
    }
    
    /// Run continuous health checks
    async fn run_health_checks(&self) {
        let mut interval = interval(self.health_config.interval);
        
        info!("Starting continuous health checks...");
        
        loop {
            interval.tick().await;
            
            let component_names: Vec<String> = {
                let trackers = self.trackers.read().await;
                trackers.keys().cloned().collect()
            };
            
            // Perform health checks concurrently
            let mut check_tasks = Vec::new();
            
            for component in component_names {
                let monitor = self.clone();
                let comp = component.clone();
                
                let task = tokio::spawn(async move {
                    if let Err(e) = monitor.check_component_health(&comp).await {
                        error!("Health check failed for {}: {}", comp, e);
                    }
                });
                
                check_tasks.push(task);
            }
            
            // Wait for all checks to complete
            for task in check_tasks {
                if let Err(e) = task.await {
                    error!("Health check task failed: {}", e);
                }
            }
        }
    }
    
    /// Process health events
    async fn process_health_events(&self, mut event_rx: mpsc::UnboundedReceiver<HealthEvent>) {
        info!("Starting health event processing...");
        
        while let Some(event) = event_rx.recv().await {
            match event {
                HealthEvent::ComponentHealthy { component, previous_status } => {
                    info!("Component {} became healthy (was {:?})", component, previous_status);
                }
                HealthEvent::ComponentUnhealthy { component, error, consecutive_failures } => {
                    error!("Component {} is unhealthy after {} failures: {}", 
                        component, consecutive_failures, error);
                }
                HealthEvent::ComponentRecovered { component, downtime } => {
                    info!("Component {} recovered after {:?} downtime", component, downtime);
                }
                HealthEvent::SystemHealthChanged { old_status, new_status, affected_components } => {
                    info!("System health changed from {:?} to {:?}, affected: {:?}", 
                        old_status, new_status, affected_components);
                }
                HealthEvent::PerformanceDegraded { component, metric, threshold, current } => {
                    warn!("Performance degraded for {}: {} = {} (threshold: {})", 
                        component, metric, current, threshold);
                }
            }
        }
        
        info!("Health event processing stopped");
    }
    
    /// Aggregate system health from component health
    async fn aggregate_system_health(&self) {
        let mut interval = interval(Duration::from_secs(10));
        let start_time = Instant::now();
        
        loop {
            interval.tick().await;
            
            let mut component_health = HashMap::new();
            let mut healthy_count = 0;
            let mut unhealthy_count = 0;
            let mut degraded_count = 0;
            
            // Collect component health
            {
                let trackers = self.trackers.read().await;
                for (name, tracker) in trackers.iter() {
                    let health = ComponentHealth {
                        name: tracker.name.clone(),
                        status: self.status_from_component_status(&tracker.current_status),
                        latency_ms: tracker.avg_response_time.as_millis() as u64,
                        error: None,
                        last_success: tracker.last_success,
                    };
                    
                    match health.status {
                        HealthStatus::Healthy => healthy_count += 1,
                        HealthStatus::Degraded => degraded_count += 1,
                        HealthStatus::Unhealthy => unhealthy_count += 1,
                        _ => {}
                    }
                    
                    component_health.insert(name.clone(), health);
                }
            }
            
            // Determine overall system status
            let old_status = {
                let system_health = self.system_health_cache.read().await;
                system_health.status.clone()
            };
            
            let new_status = if unhealthy_count > 0 {
                HealthStatus::Unhealthy
            } else if degraded_count > 0 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };
            
            // Update system health
            {
                let mut system_health = self.system_health_cache.write().await;
                system_health.status = new_status.clone();
                system_health.components = component_health;
                system_health.uptime = start_time.elapsed();
                system_health.timestamp = chrono::Utc::now();
            }
            
            // Send event if status changed
            if old_status != new_status {
                let affected_components: Vec<String> = {
                    let trackers = self.trackers.read().await;
                    trackers.keys().cloned().collect()
                };
                
                if let Err(e) = self.event_tx.send(HealthEvent::SystemHealthChanged {
                    old_status,
                    new_status,
                    affected_components,
                }) {
                    warn!("Failed to send system health change event: {}", e);
                }
            }
        }
    }
    
    /// Convert component health status to system health status
    fn status_from_component_status(&self, status: &ComponentHealthStatus) -> HealthStatus {
        match status {
            ComponentHealthStatus::Healthy => HealthStatus::Healthy,
            ComponentHealthStatus::Degraded => HealthStatus::Degraded,
            ComponentHealthStatus::Unhealthy => HealthStatus::Unhealthy,
            ComponentHealthStatus::Starting => HealthStatus::Starting,
            ComponentHealthStatus::Stopping => HealthStatus::Stopping,
            ComponentHealthStatus::Unknown => HealthStatus::Unhealthy,
        }
    }
}

impl Clone for HealthMonitor {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            config: self.config.clone(),
            service_discovery: self.service_discovery.clone(),
            trackers: self.trackers.clone(),
            event_tx: self.event_tx.clone(),
            event_rx: Arc::new(RwLock::new(None)), // Clone doesn't get receiver
            system_health_cache: self.system_health_cache.clone(),
            health_config: self.health_config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntegrationConfig;
    
    #[tokio::test]
    async fn test_health_monitor_creation() {
        let config = Arc::new(IntegrationConfig::default());
        let service_discovery = Arc::new(ServiceDiscovery::new(config.clone()).await.unwrap());
        
        let monitor = HealthMonitor::new(config, service_discovery).await;
        assert!(monitor.is_ok());
    }
    
    #[tokio::test]
    async fn test_component_registration() {
        let config = Arc::new(IntegrationConfig::default());
        let service_discovery = Arc::new(ServiceDiscovery::new(config.clone()).await.unwrap());
        let monitor = HealthMonitor::new(config, service_discovery).await.unwrap();
        
        monitor.register_component("test-component").await.unwrap();
        
        let health = monitor.component_health("test-component").await;
        assert!(health.is_some());
    }
    
    #[tokio::test]
    async fn test_system_health() {
        let config = Arc::new(IntegrationConfig::default());
        let service_discovery = Arc::new(ServiceDiscovery::new(config.clone()).await.unwrap());
        let monitor = HealthMonitor::new(config, service_discovery).await.unwrap();
        
        let system_health = monitor.system_health().await;
        assert_eq!(system_health.status, HealthStatus::Starting);
    }
}
