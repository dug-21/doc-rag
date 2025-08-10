//! # Service Discovery
//!
//! Service registry and discovery system for dynamic service location
//! and health-aware load balancing.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use uuid::Uuid;

use crate::Result;

/// Service registration information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ServiceRegistration {
    /// Service ID
    pub id: Uuid,
    /// Service name
    pub name: String,
    /// Service endpoint URL
    pub endpoint: String,
    /// Service metadata
    pub metadata: HashMap<String, String>,
    /// Health check endpoint
    pub health_endpoint: Option<String>,
    /// Service tags
    pub tags: Vec<String>,
    /// Registration timestamp
    pub registered_at: chrono::DateTime<chrono::Utc>,
    /// Last heartbeat timestamp
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    /// Service TTL (time to live)
    pub ttl: Duration,
    /// Service status
    pub status: ServiceStatus,
}

/// Service status enumeration
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ServiceStatus {
    /// Service is healthy and available
    Healthy,
    /// Service is degraded but functional
    Degraded,
    /// Service is unhealthy
    Unhealthy,
    /// Service is starting up
    Starting,
    /// Service is shutting down
    Stopping,
    /// Service status is unknown
    Unknown,
}

/// Service discovery configuration
#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfig {
    /// Default service TTL
    pub default_ttl: Duration,
    /// Cleanup interval for expired services
    pub cleanup_interval: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Enable distributed mode (for clustering)
    pub enable_distributed: bool,
    /// Gossip port for distributed mode
    pub gossip_port: Option<u16>,
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(30),
            cleanup_interval: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(30),
            enable_distributed: false,
            gossip_port: None,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin selection
    RoundRobin,
    /// Random selection
    Random,
    /// Least connections
    LeastConnections,
    /// Health-aware selection
    HealthAware,
    /// Weighted selection
    Weighted,
}

/// Service instance with load balancing metadata
#[derive(Debug, Clone)]
struct ServiceInstance {
    registration: ServiceRegistration,
    connections: u64,
    weight: f64,
    last_selected: Option<Instant>,
    response_times: Vec<Duration>,
}

impl ServiceInstance {
    fn new(registration: ServiceRegistration) -> Self {
        Self {
            registration,
            connections: 0,
            weight: 1.0,
            last_selected: None,
            response_times: Vec::new(),
        }
    }
    
    fn average_response_time(&self) -> Duration {
        if self.response_times.is_empty() {
            Duration::from_millis(100) // Default
        } else {
            let total: Duration = self.response_times.iter().sum();
            total / self.response_times.len() as u32
        }
    }
}

/// Service discovery system
pub struct ServiceDiscovery {
    /// System ID
    id: Uuid,
    /// Configuration
    config: Arc<crate::IntegrationConfig>,
    /// Service discovery configuration
    discovery_config: ServiceDiscoveryConfig,
    /// Service registry
    services: Arc<RwLock<HashMap<String, Vec<ServiceInstance>>>>,
    /// Service lookup index by ID
    service_index: Arc<RwLock<HashMap<Uuid, String>>>,
    /// Round-robin counters
    round_robin_counters: Arc<RwLock<HashMap<String, usize>>>,
}

impl ServiceDiscovery {
    /// Create new service discovery system
    pub async fn new(config: Arc<crate::IntegrationConfig>) -> Result<Self> {
        let discovery_config = ServiceDiscoveryConfig::default();
        
        Ok(Self {
            id: Uuid::new_v4(),
            config,
            discovery_config,
            services: Arc::new(RwLock::new(HashMap::new())),
            service_index: Arc::new(RwLock::new(HashMap::new())),
            round_robin_counters: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Initialize service discovery
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing Service Discovery: {}", self.id);
        
        // Register built-in services
        self.register_builtin_services().await?;
        
        info!("Service Discovery initialized successfully");
        Ok(())
    }
    
    /// Start service discovery
    pub async fn start(&self) -> Result<()> {
        info!("Starting Service Discovery...");
        
        // Start cleanup task
        let discovery = self.clone();
        tokio::spawn(async move {
            discovery.cleanup_expired_services().await;
        });
        
        // Start health checking
        let discovery = self.clone();
        tokio::spawn(async move {
            discovery.health_check_services().await;
        });
        
        info!("Service Discovery started successfully");
        Ok(())
    }
    
    /// Stop service discovery
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Service Discovery...");
        
        // Deregister all services
        let service_names: Vec<String> = {
            let services = self.services.read().await;
            services.keys().cloned().collect()
        };
        
        for service_name in service_names {
            if let Err(e) = self.deregister_all_instances(&service_name).await {
                warn!("Failed to deregister service {}: {}", service_name, e);
            }
        }
        
        info!("Service Discovery stopped successfully");
        Ok(())
    }
    
    /// Register a service
    pub async fn register_service(
        &self, 
        name: &str, 
        endpoint: &str,
    ) -> Result<Uuid> {
        let registration = ServiceRegistration {
            id: Uuid::new_v4(),
            name: name.to_string(),
            endpoint: endpoint.to_string(),
            metadata: HashMap::new(),
            health_endpoint: Some(format!("{}/health", endpoint)),
            tags: Vec::new(),
            registered_at: chrono::Utc::now(),
            last_heartbeat: chrono::Utc::now(),
            ttl: self.discovery_config.default_ttl,
            status: ServiceStatus::Starting,
        };
        
        self.register_service_with_details(registration).await
    }
    
    /// Register service with full details
    pub async fn register_service_with_details(
        &self,
        registration: ServiceRegistration,
    ) -> Result<Uuid> {
        info!("Registering service: {} at {}", registration.name, registration.endpoint);
        
        let service_id = registration.id;
        let service_name = registration.name.clone();
        let instance = ServiceInstance::new(registration);
        
        // Add to services
        {
            let mut services = self.services.write().await;
            services
                .entry(service_name.clone())
                .or_insert_with(Vec::new)
                .push(instance);
        }
        
        // Add to index
        {
            let mut index = self.service_index.write().await;
            index.insert(service_id, service_name.clone());
        }
        
        info!("Service registered successfully: {} ({})", service_name, service_id);
        Ok(service_id)
    }
    
    /// Deregister a service by ID
    pub async fn deregister_service(&self, service_id: Uuid) -> Result<()> {
        let service_name = {
            let mut index = self.service_index.write().await;
            index.remove(&service_id)
        };
        
        if let Some(name) = service_name {
            let mut services = self.services.write().await;
            if let Some(instances) = services.get_mut(&name) {
                instances.retain(|instance| instance.registration.id != service_id);
                
                if instances.is_empty() {
                    services.remove(&name);
                }
            }
            
            info!("Service deregistered: {} ({})", name, service_id);
        }
        
        Ok(())
    }
    
    /// Deregister all instances of a service
    pub async fn deregister_all_instances(&self, service_name: &str) -> Result<()> {
        info!("Deregistering all instances of service: {}", service_name);
        
        let service_ids: Vec<Uuid> = {
            let services = self.services.read().await;
            if let Some(instances) = services.get(service_name) {
                instances.iter().map(|i| i.registration.id).collect()
            } else {
                return Ok(());
            }
        };
        
        for service_id in service_ids {
            self.deregister_service(service_id).await?;
        }
        
        Ok(())
    }
    
    /// Discover services by name
    pub async fn discover_services(&self, service_name: &str) -> Vec<ServiceRegistration> {
        let services = self.services.read().await;
        
        if let Some(instances) = services.get(service_name) {
            instances.iter()
                .filter(|instance| instance.registration.status == ServiceStatus::Healthy)
                .map(|instance| instance.registration.clone())
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get service endpoint using load balancing
    pub async fn get_service_endpoint(
        &self, 
        service_name: &str,
    ) -> Option<String> {
        self.get_service_endpoint_with_strategy(service_name, LoadBalancingStrategy::HealthAware).await
    }
    
    /// Get service endpoint with specific load balancing strategy
    pub async fn get_service_endpoint_with_strategy(
        &self, 
        service_name: &str, 
        strategy: LoadBalancingStrategy,
    ) -> Option<String> {
        let services = self.services.read().await;
        let instances = match services.get(service_name) {
            Some(instances) => instances,
            None => return None,
        };
        
        let healthy_instances: Vec<&ServiceInstance> = instances.iter()
            .filter(|instance| instance.registration.status == ServiceStatus::Healthy)
            .collect();
        
        if healthy_instances.is_empty() {
            warn!("No healthy instances found for service: {}", service_name);
            return None;
        }
        
        let selected = match strategy {
            LoadBalancingStrategy::RoundRobin => {
                self.select_round_robin(service_name, &healthy_instances).await
            }
            LoadBalancingStrategy::Random => {
                self.select_random(&healthy_instances)
            }
            LoadBalancingStrategy::LeastConnections => {
                self.select_least_connections(&healthy_instances)
            }
            LoadBalancingStrategy::HealthAware => {
                self.select_health_aware(&healthy_instances)
            }
            LoadBalancingStrategy::Weighted => {
                self.select_weighted(&healthy_instances)
            }
        };
        
        selected.map(|instance| instance.registration.endpoint.clone())
    }
    
    /// Update service heartbeat
    pub async fn heartbeat(&self, service_id: Uuid) -> Result<()> {
        let service_name = {
            let index = self.service_index.read().await;
            index.get(&service_id).cloned()
        };
        
        if let Some(name) = service_name {
            let mut services = self.services.write().await;
            if let Some(instances) = services.get_mut(&name) {
                if let Some(instance) = instances.iter_mut().find(|i| i.registration.id == service_id) {
                    instance.registration.last_heartbeat = chrono::Utc::now();
                    instance.registration.status = ServiceStatus::Healthy;
                }
            }
        }
        
        Ok(())
    }
    
    /// Update service status
    pub async fn update_service_status(&self, service_id: Uuid, status: ServiceStatus) -> Result<()> {
        let service_name = {
            let index = self.service_index.read().await;
            index.get(&service_id).cloned()
        };
        
        if let Some(name) = service_name {
            let mut services = self.services.write().await;
            if let Some(instances) = services.get_mut(&name) {
                if let Some(instance) = instances.iter_mut().find(|i| i.registration.id == service_id) {
                    let old_status = instance.registration.status.clone();
                    instance.registration.status = status.clone();
                    
                    if old_status != status {
                        info!("Service {} status changed: {:?} -> {:?}", name, old_status, status);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Get all registered services
    pub async fn list_services(&self) -> HashMap<String, Vec<ServiceRegistration>> {
        let services = self.services.read().await;
        
        services.iter()
            .map(|(name, instances)| {
                let registrations = instances.iter()
                    .map(|instance| instance.registration.clone())
                    .collect();
                (name.clone(), registrations)
            })
            .collect()
    }
    
    /// Get service health status
    pub async fn get_service_health(&self, service_name: &str) -> Vec<(Uuid, ServiceStatus)> {
        let services = self.services.read().await;
        
        if let Some(instances) = services.get(service_name) {
            instances.iter()
                .map(|instance| (instance.registration.id, instance.registration.status.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Register built-in services
    async fn register_builtin_services(&self) -> Result<()> {
        let services = [
            ("mcp-adapter", &self.config.mcp_adapter_endpoint),
            ("chunker", &self.config.chunker_endpoint),
            ("embedder", &self.config.embedder_endpoint),
            ("storage", &self.config.storage_endpoint),
            ("query-processor", &self.config.query_processor_endpoint),
            ("response-generator", &self.config.response_generator_endpoint),
        ];
        
        for (name, endpoint) in services {
            self.register_service(name, endpoint).await?;
        }
        
        Ok(())
    }
    
    /// Round-robin selection
    async fn select_round_robin<'a>(
        &self,
        service_name: &str,
        instances: &'a [&'a ServiceInstance],
    ) -> Option<&'a ServiceInstance> {
        let mut counters = self.round_robin_counters.write().await;
        let counter = counters.entry(service_name.to_string()).or_insert(0);
        
        let selected = instances.get(*counter % instances.len());
        *counter = (*counter + 1) % instances.len();
        
        selected.copied()
    }
    
    /// Random selection
    fn select_random<'a>(&self, instances: &'a [&'a ServiceInstance]) -> Option<&'a ServiceInstance> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..instances.len());
        instances.get(index).copied()
    }
    
    /// Least connections selection
    fn select_least_connections<'a>(&self, instances: &'a [&'a ServiceInstance]) -> Option<&'a ServiceInstance> {
        instances.iter()
            .min_by_key(|instance| instance.connections)
            .copied()
    }
    
    /// Health-aware selection (combines response time and health)
    fn select_health_aware<'a>(&self, instances: &'a [&'a ServiceInstance]) -> Option<&'a ServiceInstance> {
        instances.iter()
            .min_by(|a, b| {
                let a_score = self.calculate_health_score(a);
                let b_score = self.calculate_health_score(b);
                a_score.partial_cmp(&b_score).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }
    
    /// Weighted selection
    fn select_weighted<'a>(&self, instances: &'a [&'a ServiceInstance]) -> Option<&'a ServiceInstance> {
        use rand::Rng;
        
        let total_weight: f64 = instances.iter().map(|i| i.weight).sum();
        if total_weight == 0.0 {
            return self.select_random(instances);
        }
        
        let mut rng = rand::thread_rng();
        let mut random_weight = rng.gen::<f64>() * total_weight;
        
        for instance in instances {
            random_weight -= instance.weight;
            if random_weight <= 0.0 {
                return Some(instance);
            }
        }
        
        instances.last().copied()
    }
    
    /// Calculate health score for health-aware load balancing
    fn calculate_health_score(&self, instance: &ServiceInstance) -> f64 {
        let response_time_ms = instance.average_response_time().as_millis() as f64;
        let connection_factor = 1.0 + (instance.connections as f64 * 0.1);
        
        // Lower score is better
        response_time_ms * connection_factor
    }
    
    /// Cleanup expired services
    async fn cleanup_expired_services(&self) {
        let mut interval = tokio::time::interval(self.discovery_config.cleanup_interval);
        
        loop {
            interval.tick().await;
            
            let now = chrono::Utc::now();
            let expired_services: Vec<Uuid> = {
                let services = self.services.read().await;
                let mut expired = Vec::new();
                
                for instances in services.values() {
                    for instance in instances {
                        let ttl_chrono = chrono::Duration::from_std(instance.registration.ttl).unwrap();
                        if now.signed_duration_since(instance.registration.last_heartbeat) > ttl_chrono {
                            expired.push(instance.registration.id);
                        }
                    }
                }
                
                expired
            };
            
            if !expired_services.is_empty() {
                info!("Cleaning up {} expired services", expired_services.len());
                
                for service_id in expired_services {
                    if let Err(e) = self.deregister_service(service_id).await {
                        error!("Failed to cleanup expired service {}: {}", service_id, e);
                    }
                }
            }
        }
    }
    
    /// Health check services
    async fn health_check_services(&self) {
        let mut interval = tokio::time::interval(self.discovery_config.health_check_interval);
        
        loop {
            interval.tick().await;
            
            let services_to_check: Vec<(Uuid, String)> = {
                let services = self.services.read().await;
                let mut to_check = Vec::new();
                
                for instances in services.values() {
                    for instance in instances {
                        if let Some(health_endpoint) = &instance.registration.health_endpoint {
                            to_check.push((instance.registration.id, health_endpoint.clone()));
                        }
                    }
                }
                
                to_check
            };
            
            for (service_id, health_endpoint) in services_to_check {
                let discovery = self.clone();
                tokio::spawn(async move {
                    let status = discovery.check_service_health(&health_endpoint).await;
                    if let Err(e) = discovery.update_service_status(service_id, status).await {
                        error!("Failed to update service status for {}: {}", service_id, e);
                    }
                });
            }
        }
    }
    
    /// Check individual service health
    async fn check_service_health(&self, health_endpoint: &str) -> ServiceStatus {
        let client = reqwest::Client::new();
        
        match tokio::time::timeout(
            Duration::from_secs(5),
            client.get(health_endpoint).send()
        ).await {
            Ok(Ok(response)) if response.status().is_success() => ServiceStatus::Healthy,
            Ok(Ok(_)) => ServiceStatus::Degraded,
            Ok(Err(_)) | Err(_) => ServiceStatus::Unhealthy,
        }
    }
}

impl Clone for ServiceDiscovery {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            config: self.config.clone(),
            discovery_config: self.discovery_config.clone(),
            services: self.services.clone(),
            service_index: self.service_index.clone(),
            round_robin_counters: self.round_robin_counters.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntegrationConfig;
    
    #[tokio::test]
    async fn test_service_registration() {
        let config = Arc::new(IntegrationConfig::default());
        let discovery = ServiceDiscovery::new(config).await.unwrap();
        
        let service_id = discovery.register_service("test-service", "http://localhost:8080").await.unwrap();
        
        let services = discovery.discover_services("test-service").await;
        assert_eq!(services.len(), 1);
        assert_eq!(services[0].id, service_id);
        assert_eq!(services[0].endpoint, "http://localhost:8080");
    }
    
    #[tokio::test]
    async fn test_service_deregistration() {
        let config = Arc::new(IntegrationConfig::default());
        let discovery = ServiceDiscovery::new(config).await.unwrap();
        
        let service_id = discovery.register_service("test-service", "http://localhost:8080").await.unwrap();
        
        let services = discovery.discover_services("test-service").await;
        assert_eq!(services.len(), 1);
        
        discovery.deregister_service(service_id).await.unwrap();
        
        let services = discovery.discover_services("test-service").await;
        assert_eq!(services.len(), 0);
    }
    
    #[tokio::test]
    async fn test_load_balancing() {
        let config = Arc::new(IntegrationConfig::default());
        let discovery = ServiceDiscovery::new(config).await.unwrap();
        
        // Register multiple instances
        discovery.register_service("test-service", "http://localhost:8080").await.unwrap();
        discovery.register_service("test-service", "http://localhost:8081").await.unwrap();
        discovery.register_service("test-service", "http://localhost:8082").await.unwrap();
        
        // Test round-robin
        let mut endpoints = Vec::new();
        for _ in 0..6 {
            if let Some(endpoint) = discovery.get_service_endpoint_with_strategy(
                "test-service", 
                LoadBalancingStrategy::RoundRobin
            ).await {
                endpoints.push(endpoint);
            }
        }
        
        // Should cycle through all instances
        assert_eq!(endpoints.len(), 6);
        assert_eq!(endpoints[0], endpoints[3]);
        assert_eq!(endpoints[1], endpoints[4]);
        assert_eq!(endpoints[2], endpoints[5]);
    }
    
    #[tokio::test]
    async fn test_heartbeat() {
        let config = Arc::new(IntegrationConfig::default());
        let discovery = ServiceDiscovery::new(config).await.unwrap();
        
        let service_id = discovery.register_service("test-service", "http://localhost:8080").await.unwrap();
        
        // Initial heartbeat time
        let services = discovery.discover_services("test-service").await;
        let initial_heartbeat = services[0].last_heartbeat;
        
        // Wait a bit and send heartbeat
        tokio::time::sleep(Duration::from_millis(10)).await;
        discovery.heartbeat(service_id).await.unwrap();
        
        // Check heartbeat was updated
        let services = discovery.discover_services("test-service").await;
        assert!(services[0].last_heartbeat > initial_heartbeat);
    }
}
