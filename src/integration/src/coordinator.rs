//! # Integration Coordinator
//!
//! Central coordination service that manages all system components and their interactions.
//! Implements service discovery, health monitoring, and inter-component communication.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, mpsc};
use tokio::time::{interval, Instant};
use tracing::{info, warn, error, instrument};
use uuid::Uuid;

use crate::{Result, IntegrationError, IntegrationConfig, ServiceDiscovery, MessageBus};

/// Component registration information
#[derive(Debug, Clone)]
pub struct ComponentRegistration {
    /// Component ID
    pub id: Uuid,
    /// Component name
    pub name: String,
    /// Component type
    pub component_type: ComponentType,
    /// Health check endpoint
    pub health_endpoint: String,
    /// Service endpoint
    pub service_endpoint: String,
    /// Registration timestamp
    pub registered_at: chrono::DateTime<chrono::Utc>,
    /// Last health check
    pub last_health_check: Option<chrono::DateTime<chrono::Utc>>,
    /// Current health status
    pub health_status: ComponentHealthStatus,
    /// Component configuration
    pub config: ComponentConfig,
}

/// Component types in the system
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ComponentType {
    /// MCP Protocol Adapter
    McpAdapter,
    /// Document Chunker
    Chunker,
    /// Embedding Generator
    Embedder,
    /// Vector Storage
    Storage,
    /// Query Processor
    QueryProcessor,
    /// Response Generator
    ResponseGenerator,
}

/// Component health status
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComponentHealthStatus {
    /// Component is healthy
    Healthy,
    /// Component is degraded
    Degraded,
    /// Component is unhealthy
    Unhealthy,
    /// Component is starting up
    Starting,
    /// Component is shutting down
    Stopping,
    /// Component is unknown
    Unknown,
}

/// Component configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentConfig {
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    /// Request timeout
    pub request_timeout_ms: u64,
    /// Retry attempts
    pub retry_attempts: usize,
    /// Circuit breaker threshold
    pub circuit_breaker_threshold: usize,
    /// Custom settings
    pub settings: HashMap<String, String>,
}

/// Coordination event types
#[derive(Debug, Clone)]
pub enum CoordinationEvent {
    /// Component registered
    ComponentRegistered {
        component: ComponentRegistration,
    },
    /// Component deregistered
    ComponentDeregistered {
        component_id: Uuid,
    },
    /// Component health changed
    ComponentHealthChanged {
        component_id: Uuid,
        old_status: ComponentHealthStatus,
        new_status: ComponentHealthStatus,
    },
    /// System health changed
    SystemHealthChanged {
        old_status: crate::HealthStatus,
        new_status: crate::HealthStatus,
    },
    /// Circuit breaker state changed
    CircuitBreakerChanged {
        component_id: Uuid,
        state: String,
    },
}

/// Main integration coordinator
pub struct IntegrationCoordinator {
    /// Coordinator ID
    id: Uuid,
    /// Configuration
    config: Arc<IntegrationConfig>,
    /// Service discovery
    service_discovery: Arc<ServiceDiscovery>,
    /// Message bus
    message_bus: Arc<MessageBus>,
    /// Registered components
    components: Arc<RwLock<HashMap<Uuid, ComponentRegistration>>>,
    /// Event channel
    event_tx: mpsc::UnboundedSender<CoordinationEvent>,
    event_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<CoordinationEvent>>>>,
    /// Component clients
    component_clients: Arc<RwLock<HashMap<ComponentType, Arc<dyn ComponentClient>>>>,
    /// Health check interval
    health_check_interval: Duration,
}

/// Component client trait for inter-component communication
#[async_trait::async_trait]
pub trait ComponentClient: Send + Sync {
    /// Get component health
    async fn health(&self) -> Result<ComponentHealthStatus>;
    
    /// Process request
    async fn process(&self, request: ComponentRequest) -> Result<ComponentResponse>;
    
    /// Get component configuration
    async fn config(&self) -> Result<ComponentConfig>;
    
    /// Shutdown component
    async fn shutdown(&self) -> Result<()>;
}

/// Generic component request
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentRequest {
    /// Request ID
    pub id: Uuid,
    /// Request type
    pub request_type: String,
    /// Request payload
    pub payload: serde_json::Value,
    /// Request timeout
    pub timeout_ms: Option<u64>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
}

/// Generic component response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentResponse {
    /// Request ID
    pub request_id: Uuid,
    /// Response status
    pub status: ResponseStatus,
    /// Response payload
    pub payload: serde_json::Value,
    /// Processing time
    pub processing_time_ms: u64,
    /// Response metadata
    pub metadata: HashMap<String, String>,
}

/// Response status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ResponseStatus {
    /// Success
    Success,
    /// Partial success
    PartialSuccess,
    /// Error
    Error,
    /// Timeout
    Timeout,
}

impl IntegrationCoordinator {
    /// Create new integration coordinator
    pub async fn new(
        config: Arc<IntegrationConfig>,
        service_discovery: Arc<ServiceDiscovery>,
        message_bus: Arc<MessageBus>,
    ) -> Result<Self> {
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        
        Ok(Self {
            id: Uuid::new_v4(),
            config: config.clone(),
            service_discovery,
            message_bus,
            components: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            event_rx: Arc::new(RwLock::new(Some(event_rx))),
            component_clients: Arc::new(RwLock::new(HashMap::new())),
            health_check_interval: Duration::from_secs(
                config.health_check_interval_secs.unwrap_or(30)
            ),
        })
    }
    
    /// Initialize coordinator
    #[instrument(skip(self))]
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing Integration Coordinator: {}", self.id);
        
        // Initialize component clients
        self.initialize_component_clients().await?;
        
        // Register system components
        self.register_system_components().await?;
        
        info!("Integration Coordinator initialized successfully");
        Ok(())
    }
    
    /// Start coordinator
    #[instrument(skip(self))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting Integration Coordinator...");
        
        // Start event processing
        let event_rx = self.event_rx.write().await.take()
            .ok_or_else(|| IntegrationError::Internal("Event receiver already taken".to_string()))?;
        
        let coordinator = self.clone();
        tokio::spawn(async move {
            coordinator.process_events(event_rx).await;
        });
        
        // Start health monitoring
        let coordinator = self.clone();
        tokio::spawn(async move {
            coordinator.monitor_component_health().await;
        });
        
        info!("Integration Coordinator started successfully");
        Ok(())
    }
    
    /// Stop coordinator
    #[instrument(skip(self))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Integration Coordinator...");
        
        // Shutdown all components
        let components = self.components.read().await;
        for (component_id, registration) in components.iter() {
            if let Some(client) = self.get_component_client(&registration.component_type).await {
                if let Err(e) = client.shutdown().await {
                    warn!("Failed to shutdown component {}: {}", component_id, e);
                }
            }
        }
        
        info!("Integration Coordinator stopped successfully");
        Ok(())
    }
    
    /// Register a component
    #[instrument(skip(self, registration))]
    pub async fn register_component(&self, mut registration: ComponentRegistration) -> Result<()> {
        info!("Registering component: {} ({})", registration.name, registration.component_type);
        
        registration.registered_at = chrono::Utc::now();
        registration.health_status = ComponentHealthStatus::Starting;
        
        let component_id = registration.id;
        
        // Store registration
        {
            let mut components = self.components.write().await;
            components.insert(component_id, registration.clone());
        }
        
        // Register with service discovery
        self.service_discovery.register_service(
            &registration.name,
            &registration.service_endpoint,
        ).await?;
        
        // Notify about registration
        if let Err(e) = self.event_tx.send(CoordinationEvent::ComponentRegistered { component: registration }) {
            warn!("Failed to send component registration event: {}", e);
        }
        
        info!("Component registered successfully: {}", component_id);
        Ok(())
    }
    
    /// Deregister a component
    #[instrument(skip(self))]
    pub async fn deregister_component(&self, component_id: Uuid) -> Result<()> {
        info!("Deregistering component: {}", component_id);
        
        // Remove from registry
        let registration = {
            let mut components = self.components.write().await;
            components.remove(&component_id)
        };
        
        if let Some(reg) = registration {
            // Deregister from service discovery
            self.service_discovery.deregister_service(reg.id).await?;
            
            // Notify about deregistration
            if let Err(e) = self.event_tx.send(CoordinationEvent::ComponentDeregistered { component_id }) {
                warn!("Failed to send component deregistration event: {}", e);
            }
        }
        
        info!("Component deregistered successfully: {}", component_id);
        Ok(())
    }
    
    /// Get component by type
    pub async fn get_component(&self, component_type: &ComponentType) -> Option<ComponentRegistration> {
        let components = self.components.read().await;
        components.values()
            .find(|reg| &reg.component_type == component_type)
            .cloned()
    }
    
    /// Get all components
    pub async fn get_all_components(&self) -> HashMap<Uuid, ComponentRegistration> {
        self.components.read().await.clone()
    }
    
    /// Get component client
    async fn get_component_client(&self, component_type: &ComponentType) -> Option<Arc<dyn ComponentClient>> {
        let clients = self.component_clients.read().await;
        clients.get(component_type).cloned()
    }
    
    /// Process coordination events
    async fn process_events(&self, mut event_rx: mpsc::UnboundedReceiver<CoordinationEvent>) {
        info!("Starting event processing...");
        
        while let Some(event) = event_rx.recv().await {
            match event {
                CoordinationEvent::ComponentRegistered { component } => {
                    info!("Component registered: {} ({})", component.name, component.component_type);
                    // Additional registration logic here
                }
                CoordinationEvent::ComponentDeregistered { component_id } => {
                    info!("Component deregistered: {}", component_id);
                    // Additional deregistration logic here
                }
                CoordinationEvent::ComponentHealthChanged { component_id, old_status, new_status } => {
                    info!("Component {} health changed: {:?} -> {:?}", component_id, old_status, new_status);
                    // Handle health changes
                }
                CoordinationEvent::SystemHealthChanged { old_status, new_status } => {
                    info!("System health changed: {:?} -> {:?}", old_status, new_status);
                    // Handle system health changes
                }
                CoordinationEvent::CircuitBreakerChanged { component_id, state } => {
                    info!("Component {} circuit breaker changed: {}", component_id, state);
                    // Handle circuit breaker changes
                }
            }
        }
        
        info!("Event processing stopped");
    }
    
    /// Monitor component health
    async fn monitor_component_health(&self) {
        let mut interval = interval(self.health_check_interval);
        
        info!("Starting component health monitoring...");
        
        loop {
            interval.tick().await;
            
            let components: Vec<_> = {
                let components = self.components.read().await;
                components.values().cloned().collect()
            };
            
            for registration in components {
                if let Some(client) = self.get_component_client(&registration.component_type).await {
                    let start = Instant::now();
                    match client.health().await {
                        Ok(status) => {
                            let duration = start.elapsed();
                            
                            if status != registration.health_status {
                                // Health status changed
                                if let Err(e) = self.event_tx.send(CoordinationEvent::ComponentHealthChanged {
                                    component_id: registration.id,
                                    old_status: registration.health_status.clone(),
                                    new_status: status.clone(),
                                }) {
                                    warn!("Failed to send health change event: {}", e);
                                }
                                
                                // Update stored status
                                let mut components = self.components.write().await;
                                if let Some(reg) = components.get_mut(&registration.id) {
                                    reg.health_status = status;
                                    reg.last_health_check = Some(chrono::Utc::now());
                                }
                            }
                            
                            if duration > Duration::from_millis(5000) {
                                warn!("Slow health check for {}: {:?}", registration.name, duration);
                            }
                        }
                        Err(e) => {
                            error!("Health check failed for {}: {}", registration.name, e);
                            
                            // Mark as unhealthy
                            if registration.health_status != ComponentHealthStatus::Unhealthy {
                                if let Err(e) = self.event_tx.send(CoordinationEvent::ComponentHealthChanged {
                                    component_id: registration.id,
                                    old_status: registration.health_status.clone(),
                                    new_status: ComponentHealthStatus::Unhealthy,
                                }) {
                                    warn!("Failed to send health change event: {}", e);
                                }
                                
                                let mut components = self.components.write().await;
                                if let Some(reg) = components.get_mut(&registration.id) {
                                    reg.health_status = ComponentHealthStatus::Unhealthy;
                                    reg.last_health_check = Some(chrono::Utc::now());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Initialize component clients
    async fn initialize_component_clients(&self) -> Result<()> {
        info!("Initializing component clients...");
        
        let mut clients = self.component_clients.write().await;
        
        // Initialize MCP Adapter client
        clients.insert(ComponentType::McpAdapter, Arc::new(McpAdapterClient::new(&self.config).await?));
        
        // Initialize other component clients
        clients.insert(ComponentType::Chunker, Arc::new(ChunkerClient::new(&self.config).await?));
        clients.insert(ComponentType::Embedder, Arc::new(EmbedderClient::new(&self.config).await?));
        clients.insert(ComponentType::Storage, Arc::new(StorageClient::new(&self.config).await?));
        clients.insert(ComponentType::QueryProcessor, Arc::new(QueryProcessorClient::new(&self.config).await?));
        clients.insert(ComponentType::ResponseGenerator, Arc::new(ResponseGeneratorClient::new(&self.config).await?));
        
        info!("Component clients initialized successfully");
        Ok(())
    }
    
    /// Register system components
    async fn register_system_components(&self) -> Result<()> {
        info!("Registering system components...");
        
        let components = vec![
            (ComponentType::McpAdapter, "mcp-adapter", "http://localhost:8001"),
            (ComponentType::Chunker, "chunker", "http://localhost:8002"),
            (ComponentType::Embedder, "embedder", "http://localhost:8003"),
            (ComponentType::Storage, "storage", "http://localhost:8004"),
            (ComponentType::QueryProcessor, "query-processor", "http://localhost:8005"),
            (ComponentType::ResponseGenerator, "response-generator", "http://localhost:8006"),
        ];
        
        for (component_type, name, endpoint) in components {
            let registration = ComponentRegistration {
                id: Uuid::new_v4(),
                name: name.to_string(),
                component_type,
                health_endpoint: format!("{}/health", endpoint),
                service_endpoint: endpoint.to_string(),
                registered_at: chrono::Utc::now(),
                last_health_check: None,
                health_status: ComponentHealthStatus::Starting,
                config: ComponentConfig {
                    max_concurrent_requests: 100,
                    request_timeout_ms: 30000,
                    retry_attempts: 3,
                    circuit_breaker_threshold: 5,
                    settings: HashMap::new(),
                },
            };
            
            self.register_component(registration).await?;
        }
        
        info!("System components registered successfully");
        Ok(())
    }
}

impl Clone for IntegrationCoordinator {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            config: self.config.clone(),
            service_discovery: self.service_discovery.clone(),
            message_bus: self.message_bus.clone(),
            components: self.components.clone(),
            event_tx: self.event_tx.clone(),
            event_rx: Arc::new(RwLock::new(None)), // Clone doesn't get receiver
            component_clients: self.component_clients.clone(),
            health_check_interval: self.health_check_interval,
        }
    }
}

// Component client implementations would be defined here
// For brevity, showing skeleton implementations

struct McpAdapterClient;
struct ChunkerClient;
struct EmbedderClient;
struct StorageClient;
struct QueryProcessorClient;
struct ResponseGeneratorClient;

// Implement skeleton clients
macro_rules! impl_component_client {
    ($client:ty, $name:expr) => {
        impl $client {
            async fn new(_config: &IntegrationConfig) -> Result<Self> {
                Ok(Self)
            }
        }
        
        #[async_trait::async_trait]
        impl ComponentClient for $client {
            async fn health(&self) -> Result<ComponentHealthStatus> {
                // Implementation would make HTTP call to component
                Ok(ComponentHealthStatus::Healthy)
            }
            
            async fn process(&self, _request: ComponentRequest) -> Result<ComponentResponse> {
                // Implementation would forward request to component
                Ok(ComponentResponse {
                    request_id: Uuid::new_v4(),
                    status: ResponseStatus::Success,
                    payload: serde_json::json!({}),
                    processing_time_ms: 0,
                    metadata: HashMap::new(),
                })
            }
            
            async fn config(&self) -> Result<ComponentConfig> {
                Ok(ComponentConfig {
                    max_concurrent_requests: 100,
                    request_timeout_ms: 30000,
                    retry_attempts: 3,
                    circuit_breaker_threshold: 5,
                    settings: HashMap::new(),
                })
            }
            
            async fn shutdown(&self) -> Result<()> {
                info!("Shutting down {} client", $name);
                Ok(())
            }
        }
    };
}

impl_component_client!(McpAdapterClient, "MCP Adapter");
impl_component_client!(ChunkerClient, "Chunker");
impl_component_client!(EmbedderClient, "Embedder");
impl_component_client!(StorageClient, "Storage");
impl_component_client!(QueryProcessorClient, "Query Processor");
impl_component_client!(ResponseGeneratorClient, "Response Generator");

// Display implementations for better debugging
impl std::fmt::Display for ComponentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComponentType::McpAdapter => write!(f, "MCP-Adapter"),
            ComponentType::Chunker => write!(f, "Chunker"),
            ComponentType::Embedder => write!(f, "Embedder"),
            ComponentType::Storage => write!(f, "Storage"),
            ComponentType::QueryProcessor => write!(f, "Query-Processor"),
            ComponentType::ResponseGenerator => write!(f, "Response-Generator"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntegrationConfig;
    
    #[tokio::test]
    async fn test_component_registration() {
        let config = Arc::new(IntegrationConfig::default());
        let service_discovery = Arc::new(ServiceDiscovery::new(config.clone()).await.unwrap());
        let message_bus = Arc::new(MessageBus::new(config.clone()).await.unwrap());
        
        let coordinator = IntegrationCoordinator::new(
            config,
            service_discovery,
            message_bus,
        ).await.unwrap();
        
        let registration = ComponentRegistration {
            id: Uuid::new_v4(),
            name: "test-component".to_string(),
            component_type: ComponentType::Chunker,
            health_endpoint: "http://localhost:8080/health".to_string(),
            service_endpoint: "http://localhost:8080".to_string(),
            registered_at: chrono::Utc::now(),
            last_health_check: None,
            health_status: ComponentHealthStatus::Starting,
            config: ComponentConfig {
                max_concurrent_requests: 100,
                request_timeout_ms: 30000,
                retry_attempts: 3,
                circuit_breaker_threshold: 5,
                settings: HashMap::new(),
            },
        };
        
        let component_id = registration.id;
        coordinator.register_component(registration).await.unwrap();
        
        let retrieved = coordinator.get_component(&ComponentType::Chunker).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, component_id);
    }
}
