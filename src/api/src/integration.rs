//! Integration Module for ruv-FANN, DAA, and FACT Systems
//! 
//! This module provides consolidated initialization and management of:
//! - ruv-FANN: Neural networks for boundary detection and classification
//! - DAA (Decentralized Autonomous Agents): Orchestrator for distributed systems
//! - FACT: Intelligent caching system
//!
//! The integration follows the Domain Wrapper Pattern for clean separation of concerns.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

use crate::config::ApiConfig;
use crate::errors::ApiError;

/// Configuration for integrated systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    pub ruv_fann: RuvFannConfig,
    pub daa: DaaConfig,
    pub fact: FactConfig,
    pub enable_ruv_fann: bool,
    pub enable_daa: bool,
    pub enable_fact: bool,
}

/// ruv-FANN Neural Network Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuvFannConfig {
    pub max_neurons: usize,
    pub learning_rate: f64,
    pub training_threshold: f64,
    pub max_iterations: u32,
    pub hidden_layers: Vec<usize>,
    pub activation_function: String,
}

/// DAA Orchestrator Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaaConfig {
    pub max_agents: usize,
    pub coordination_timeout_secs: u64,
    pub consensus_threshold: f64,
    pub auto_scaling: bool,
    pub load_balancing_strategy: String,
}

/// FACT Cache Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactConfig {
    pub cache_size_mb: usize,
    pub ttl_seconds: u64,
    pub eviction_policy: String,
    pub compression_enabled: bool,
    pub persistence_enabled: bool,
    pub backup_interval_secs: u64,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            ruv_fann: RuvFannConfig {
                max_neurons: 1000,
                learning_rate: 0.7,
                training_threshold: 0.001,
                max_iterations: 500000,
                hidden_layers: vec![100, 50, 25],
                activation_function: "FANN_SIGMOID_SYMMETRIC".to_string(),
            },
            daa: DaaConfig {
                max_agents: 10,
                coordination_timeout_secs: 30,
                consensus_threshold: 0.67,
                auto_scaling: true,
                load_balancing_strategy: "round_robin".to_string(),
            },
            fact: FactConfig {
                cache_size_mb: 256,
                ttl_seconds: 3600,
                eviction_policy: "lru".to_string(),
                compression_enabled: true,
                persistence_enabled: true,
                backup_interval_secs: 300,
            },
            enable_ruv_fann: true,
            enable_daa: true,
            enable_fact: true,
        }
    }
}

/// Health status for integrated systems
#[derive(Debug, Clone, Serialize)]
pub struct SystemHealth {
    pub ruv_fann: ComponentHealth,
    pub daa: ComponentHealth,
    pub fact: ComponentHealth,
    pub overall_status: HealthStatus,
    pub last_check: chrono::DateTime<chrono::Utc>,
}

/// Individual component health status
#[derive(Debug, Clone, Serialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub message: String,
    pub metrics: serde_json::Value,
}

/// System health status enumeration
#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Metrics for the integrated systems
#[derive(Debug, Clone, Serialize)]
pub struct IntegrationMetrics {
    pub ruv_fann_metrics: RuvFannMetrics,
    pub daa_metrics: DaaMetrics,
    pub fact_metrics: FactMetrics,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuvFannMetrics {
    pub active_networks: usize,
    pub total_neurons: usize,
    pub training_accuracy: f64,
    pub prediction_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct DaaMetrics {
    pub active_agents: usize,
    pub task_queue_length: usize,
    pub consensus_rate: f64,
    pub coordination_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FactMetrics {
    pub cache_hit_rate: f64,
    pub memory_usage_mb: usize,
    pub evictions_per_minute: f64,
    pub persistence_lag_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: usize,
    pub request_latency_p99_ms: f64,
    pub throughput_rps: f64,
}

/// Main integration manager for all systems
pub struct IntegrationManager {
    pub config: IntegrationConfig,
    pub ruv_fann_manager: Option<Arc<RwLock<RuvFannManager>>>,
    pub daa_manager: Option<Arc<RwLock<DaaManager>>>,
    pub fact_manager: Option<Arc<RwLock<FactManager>>>,
    pub health_status: Arc<RwLock<SystemHealth>>,
    pub metrics: Arc<RwLock<IntegrationMetrics>>,
}

/// ruv-FANN Neural Network Manager
pub struct RuvFannManager {
    pub config: RuvFannConfig,
    pub networks: std::collections::HashMap<String, ruv_fann::Network>,
    pub training_data: Vec<(Vec<f64>, Vec<f64>)>,
}

/// DAA Orchestrator Manager
pub struct DaaManager {
    pub config: DaaConfig,
    pub orchestrator: Option<daa_orchestrator::Orchestrator>,
    pub agents: std::collections::HashMap<Uuid, daa_orchestrator::Agent>,
}

/// FACT Cache Manager
pub struct FactManager {
    pub config: FactConfig,
    pub cache: Option<fact::Cache>,
    pub stats: fact::CacheStats,
}

#[async_trait]
pub trait SystemManager {
    async fn initialize(&mut self) -> Result<()>;
    async fn start(&mut self) -> Result<()>;
    async fn stop(&mut self) -> Result<()>;
    async fn health_check(&self) -> ComponentHealth;
    fn get_metrics(&self) -> serde_json::Value;
}

impl IntegrationManager {
    /// Create a new integration manager from API configuration
    pub fn from_config(_api_config: &ApiConfig) -> Self {
        let config = IntegrationConfig::default();
        
        Self {
            config,
            ruv_fann_manager: None,
            daa_manager: None,
            fact_manager: None,
            health_status: Arc::new(RwLock::new(SystemHealth {
                ruv_fann: ComponentHealth {
                    status: HealthStatus::Unknown,
                    message: "Not initialized".to_string(),
                    metrics: serde_json::json!({}),
                },
                daa: ComponentHealth {
                    status: HealthStatus::Unknown,
                    message: "Not initialized".to_string(),
                    metrics: serde_json::json!({}),
                },
                fact: ComponentHealth {
                    status: HealthStatus::Unknown,
                    message: "Not initialized".to_string(),
                    metrics: serde_json::json!({}),
                },
                overall_status: HealthStatus::Unknown,
                last_check: chrono::Utc::now(),
            })),
            metrics: Arc::new(RwLock::new(IntegrationMetrics {
                ruv_fann_metrics: RuvFannMetrics {
                    active_networks: 0,
                    total_neurons: 0,
                    training_accuracy: 0.0,
                    prediction_latency_ms: 0.0,
                },
                daa_metrics: DaaMetrics {
                    active_agents: 0,
                    task_queue_length: 0,
                    consensus_rate: 0.0,
                    coordination_latency_ms: 0.0,
                },
                fact_metrics: FactMetrics {
                    cache_hit_rate: 0.0,
                    memory_usage_mb: 0,
                    evictions_per_minute: 0.0,
                    persistence_lag_ms: 0.0,
                },
                performance_metrics: PerformanceMetrics {
                    cpu_usage_percent: 0.0,
                    memory_usage_mb: 0,
                    request_latency_p99_ms: 0.0,
                    throughput_rps: 0.0,
                },
            })),
        }
    }

    /// Initialize all enabled systems
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing integrated systems...");

        if self.config.enable_ruv_fann {
            info!("Initializing ruv-FANN neural networks...");
            let mut ruv_fann_manager = RuvFannManager::new(self.config.ruv_fann.clone());
            ruv_fann_manager.initialize().await?;
            self.ruv_fann_manager = Some(Arc::new(RwLock::new(ruv_fann_manager)));
            info!("ruv-FANN initialized successfully");
        }

        if self.config.enable_daa {
            info!("Initializing DAA orchestrator...");
            let mut daa_manager = DaaManager::new(self.config.daa.clone());
            daa_manager.initialize().await?;
            self.daa_manager = Some(Arc::new(RwLock::new(daa_manager)));
            info!("DAA orchestrator initialized successfully");
        }

        if self.config.enable_fact {
            info!("Initializing FACT cache...");
            let mut fact_manager = FactManager::new(self.config.fact.clone());
            fact_manager.initialize().await?;
            self.fact_manager = Some(Arc::new(RwLock::new(fact_manager)));
            info!("FACT cache initialized successfully");
        }

        info!("All integrated systems initialized successfully");
        Ok(())
    }

    /// Start all initialized systems
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting integrated systems...");

        if let Some(ruv_fann_manager) = &self.ruv_fann_manager {
            ruv_fann_manager.write().await.start().await?;
        }

        if let Some(daa_manager) = &self.daa_manager {
            daa_manager.write().await.start().await?;
        }

        if let Some(fact_manager) = &self.fact_manager {
            fact_manager.write().await.start().await?;
        }

        info!("All integrated systems started successfully");
        Ok(())
    }

    /// Stop all running systems
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping integrated systems...");

        if let Some(fact_manager) = &self.fact_manager {
            fact_manager.write().await.stop().await?;
        }

        if let Some(daa_manager) = &self.daa_manager {
            daa_manager.write().await.stop().await?;
        }

        if let Some(ruv_fann_manager) = &self.ruv_fann_manager {
            ruv_fann_manager.write().await.stop().await?;
        }

        info!("All integrated systems stopped successfully");
        Ok(())
    }

    /// Get overall system health status
    pub async fn get_health(&self) -> SystemHealth {
        let health = self.health_status.read().await.clone();
        health
    }

    /// Update system health status
    pub async fn update_health(&self) -> Result<()> {
        let mut health = self.health_status.write().await;
        
        let ruv_fann_health = if let Some(manager) = &self.ruv_fann_manager {
            manager.read().await.health_check().await
        } else {
            ComponentHealth {
                status: HealthStatus::Unknown,
                message: "Not initialized".to_string(),
                metrics: serde_json::json!({}),
            }
        };

        let daa_health = if let Some(manager) = &self.daa_manager {
            manager.read().await.health_check().await
        } else {
            ComponentHealth {
                status: HealthStatus::Unknown,
                message: "Not initialized".to_string(),
                metrics: serde_json::json!({}),
            }
        };

        let fact_health = if let Some(manager) = &self.fact_manager {
            manager.read().await.health_check().await
        } else {
            ComponentHealth {
                status: HealthStatus::Unknown,
                message: "Not initialized".to_string(),
                metrics: serde_json::json!({}),
            }
        };

        // Determine overall health status
        let overall_status = match (
            &ruv_fann_health.status,
            &daa_health.status,
            &fact_health.status,
        ) {
            (HealthStatus::Healthy, HealthStatus::Healthy, HealthStatus::Healthy) => {
                HealthStatus::Healthy
            }
            (HealthStatus::Unhealthy, _, _)
            | (_, HealthStatus::Unhealthy, _)
            | (_, _, HealthStatus::Unhealthy) => HealthStatus::Unhealthy,
            _ => HealthStatus::Degraded,
        };

        *health = SystemHealth {
            ruv_fann: ruv_fann_health,
            daa: daa_health,
            fact: fact_health,
            overall_status,
            last_check: chrono::Utc::now(),
        };

        Ok(())
    }

    /// Get comprehensive system metrics
    pub async fn get_metrics(&self) -> IntegrationMetrics {
        self.metrics.read().await.clone()
    }

    /// Helper function to create a neural network for boundary detection
    pub async fn create_boundary_detection_network(&self, input_size: usize) -> Result<String> {
        if let Some(ruv_fann_manager) = &self.ruv_fann_manager {
            let network_id = Uuid::new_v4().to_string();
            ruv_fann_manager
                .write()
                .await
                .create_network(&network_id, input_size)
                .await?;
            Ok(network_id)
        } else {
            Err(ApiError::ServiceUnavailable("ruv-FANN not initialized".to_string()).into())
        }
    }

    /// Helper function to spawn DAA agent
    pub async fn spawn_daa_agent(&self, agent_type: String, capabilities: Vec<String>) -> Result<Uuid> {
        if let Some(daa_manager) = &self.daa_manager {
            daa_manager
                .write()
                .await
                .spawn_agent(agent_type, capabilities)
                .await
        } else {
            Err(ApiError::ServiceUnavailable("DAA not initialized".to_string()).into())
        }
    }

    /// Helper function to cache data using FACT
    pub async fn cache_data<T: Serialize>(&self, key: &str, data: &T, ttl: Option<u64>) -> Result<()> {
        if let Some(fact_manager) = &self.fact_manager {
            fact_manager
                .write()
                .await
                .cache_data(key, data, ttl)
                .await
        } else {
            Err(ApiError::ServiceUnavailable("FACT cache not initialized".to_string()).into())
        }
    }

    /// Helper function to retrieve cached data using FACT
    pub async fn get_cached_data<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        if let Some(fact_manager) = &self.fact_manager {
            fact_manager.read().await.get_cached_data(key).await
        } else {
            Err(ApiError::ServiceUnavailable("FACT cache not initialized".to_string()).into())
        }
    }
}

impl RuvFannManager {
    pub fn new(config: RuvFannConfig) -> Self {
        Self {
            config,
            networks: std::collections::HashMap::new(),
            training_data: Vec::new(),
        }
    }

    pub async fn create_network(&mut self, network_id: &str, input_size: usize) -> Result<()> {
        debug!("Creating ruv-FANN network with ID: {}", network_id);
        
        let mut layer_sizes = vec![input_size];
        layer_sizes.extend(&self.config.hidden_layers);
        layer_sizes.push(1); // Output layer for boundary detection

        let network = ruv_fann::Network::new(&layer_sizes)
            .context("Failed to create ruv-FANN network")?;
        
        self.networks.insert(network_id.to_string(), network);
        
        info!("Created ruv-FANN network: {}", network_id);
        Ok(())
    }
}

#[async_trait]
impl SystemManager for RuvFannManager {
    async fn initialize(&mut self) -> Result<()> {
        debug!("Initializing ruv-FANN manager");
        // Initialization logic for ruv-FANN
        Ok(())
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting ruv-FANN manager");
        // Start logic for ruv-FANN
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping ruv-FANN manager");
        // Stop logic for ruv-FANN
        Ok(())
    }

    async fn health_check(&self) -> ComponentHealth {
        ComponentHealth {
            status: HealthStatus::Healthy,
            message: format!("Active networks: {}", self.networks.len()),
            metrics: serde_json::json!({
                "networks": self.networks.len(),
                "training_data_size": self.training_data.len()
            }),
        }
    }

    fn get_metrics(&self) -> serde_json::Value {
        serde_json::json!({
            "active_networks": self.networks.len(),
            "training_data_size": self.training_data.len()
        })
    }
}

impl DaaManager {
    pub fn new(config: DaaConfig) -> Self {
        Self {
            config,
            orchestrator: None,
            agents: std::collections::HashMap::new(),
        }
    }

    pub async fn spawn_agent(&mut self, agent_type: String, capabilities: Vec<String>) -> Result<Uuid> {
        debug!("Spawning DAA agent of type: {}", agent_type);
        
        let agent_id = Uuid::new_v4();
        
        // Create mock agent for demonstration
        // In real implementation, this would use daa_orchestrator::Agent
        let agent = daa_orchestrator::Agent::new(
            agent_id,
            agent_type.clone(),
            capabilities,
        ).context("Failed to create DAA agent")?;
        
        self.agents.insert(agent_id, agent);
        
        info!("Spawned DAA agent: {} ({})", agent_id, agent_type);
        Ok(agent_id)
    }
}

#[async_trait]
impl SystemManager for DaaManager {
    async fn initialize(&mut self) -> Result<()> {
        debug!("Initializing DAA manager");
        
        self.orchestrator = Some(
            daa_orchestrator::Orchestrator::new(self.config.max_agents)
                .context("Failed to create DAA orchestrator")?
        );
        
        Ok(())
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting DAA manager");
        
        if let Some(orchestrator) = &mut self.orchestrator {
            orchestrator.start().await
                .context("Failed to start DAA orchestrator")?;
        }
        
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping DAA manager");
        
        if let Some(orchestrator) = &mut self.orchestrator {
            orchestrator.stop().await
                .context("Failed to stop DAA orchestrator")?;
        }
        
        Ok(())
    }

    async fn health_check(&self) -> ComponentHealth {
        ComponentHealth {
            status: HealthStatus::Healthy,
            message: format!("Active agents: {}", self.agents.len()),
            metrics: serde_json::json!({
                "agents": self.agents.len(),
                "max_agents": self.config.max_agents
            }),
        }
    }

    fn get_metrics(&self) -> serde_json::Value {
        serde_json::json!({
            "active_agents": self.agents.len(),
            "max_agents": self.config.max_agents
        })
    }
}

impl FactManager {
    pub fn new(config: FactConfig) -> Self {
        Self {
            config,
            cache: None,
            stats: fact::CacheStats::default(),
        }
    }

    pub async fn cache_data<T: Serialize>(&mut self, key: &str, data: &T, ttl: Option<u64>) -> Result<()> {
        if let Some(cache) = &mut self.cache {
            let serialized = serde_json::to_vec(data)
                .context("Failed to serialize cache data")?;
            
            cache.set(key.to_string(), serialized, ttl.unwrap_or(self.config.ttl_seconds))
                .await
                .context("Failed to cache data")?;
            
            debug!("Cached data for key: {}", key);
        }
        
        Ok(())
    }

    pub async fn get_cached_data<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        if let Some(cache) = &self.cache {
            if let Some(data) = cache.get(key).await
                .context("Failed to retrieve cached data")? {
                let deserialized: T = serde_json::from_slice(&data)
                    .context("Failed to deserialize cached data")?;
                
                debug!("Retrieved cached data for key: {}", key);
                return Ok(Some(deserialized));
            }
        }
        
        Ok(None)
    }
}

#[async_trait]
impl SystemManager for FactManager {
    async fn initialize(&mut self) -> Result<()> {
        debug!("Initializing FACT cache manager");
        
        let cache_config = fact::Config {
            max_size_mb: self.config.cache_size_mb,
            ttl_seconds: self.config.ttl_seconds,
            eviction_policy: self.config.eviction_policy.clone(),
            compression_enabled: self.config.compression_enabled,
            persistence_enabled: self.config.persistence_enabled,
        };
        
        self.cache = Some(
            fact::Cache::new(cache_config)
                .await
                .context("Failed to create FACT cache")?
        );
        
        Ok(())
    }

    async fn start(&mut self) -> Result<()> {
        info!("Starting FACT cache manager");
        
        if let Some(cache) = &mut self.cache {
            cache.start_background_tasks()
                .await
                .context("Failed to start FACT background tasks")?;
        }
        
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        info!("Stopping FACT cache manager");
        
        if let Some(cache) = &mut self.cache {
            cache.flush_to_disk()
                .await
                .context("Failed to flush FACT cache to disk")?;
            
            cache.stop_background_tasks()
                .await
                .context("Failed to stop FACT background tasks")?;
        }
        
        Ok(())
    }

    async fn health_check(&self) -> ComponentHealth {
        ComponentHealth {
            status: HealthStatus::Healthy,
            message: "FACT cache operational".to_string(),
            metrics: serde_json::json!({
                "cache_size_mb": self.config.cache_size_mb,
                "hit_rate": self.stats.hit_rate
            }),
        }
    }

    fn get_metrics(&self) -> serde_json::Value {
        serde_json::json!({
            "cache_size_mb": self.config.cache_size_mb,
            "ttl_seconds": self.config.ttl_seconds,
            "compression_enabled": self.config.compression_enabled,
            "hit_rate": self.stats.hit_rate
        })
    }
}

// Mock implementations for external dependencies
// In a real implementation, these would be replaced with actual crate types

// Mock ruv_fann module
mod ruv_fann {
    use anyhow::Result;

    pub struct Network {
        layers: Vec<usize>,
    }

    impl Network {
        pub fn new(layers: &[usize]) -> Result<Self> {
            Ok(Self {
                layers: layers.to_vec(),
            })
        }
    }
}

// Mock daa_orchestrator module
mod daa_orchestrator {
    use anyhow::Result;
    use uuid::Uuid;

    pub struct Orchestrator {
        max_agents: usize,
    }

    impl Orchestrator {
        pub fn new(max_agents: usize) -> Result<Self> {
            Ok(Self { max_agents })
        }

        pub async fn start(&mut self) -> Result<()> {
            Ok(())
        }

        pub async fn stop(&mut self) -> Result<()> {
            Ok(())
        }
    }

    pub struct Agent {
        id: Uuid,
        agent_type: String,
        capabilities: Vec<String>,
    }

    impl Agent {
        pub fn new(id: Uuid, agent_type: String, capabilities: Vec<String>) -> Result<Self> {
            Ok(Self {
                id,
                agent_type,
                capabilities,
            })
        }
    }
}

// Mock fact module
mod fact {
    use anyhow::Result;

    #[derive(Clone)]
    pub struct Config {
        pub max_size_mb: usize,
        pub ttl_seconds: u64,
        pub eviction_policy: String,
        pub compression_enabled: bool,
        pub persistence_enabled: bool,
    }

    pub struct Cache {
        config: Config,
    }

    impl Cache {
        pub async fn new(config: Config) -> Result<Self> {
            Ok(Self { config })
        }

        pub async fn set(&mut self, _key: String, _data: Vec<u8>, _ttl: u64) -> Result<()> {
            Ok(())
        }

        pub async fn get(&self, _key: &str) -> Result<Option<Vec<u8>>> {
            Ok(None)
        }

        pub async fn start_background_tasks(&mut self) -> Result<()> {
            Ok(())
        }

        pub async fn stop_background_tasks(&mut self) -> Result<()> {
            Ok(())
        }

        pub async fn flush_to_disk(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[derive(Default, Clone)]
    pub struct CacheStats {
        pub hit_rate: f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ApiConfig;

    #[tokio::test]
    async fn test_integration_manager_creation() {
        let api_config = ApiConfig::default();
        let manager = IntegrationManager::from_config(&api_config);
        
        assert!(manager.ruv_fann_manager.is_none());
        assert!(manager.daa_manager.is_none());
        assert!(manager.fact_manager.is_none());
    }

    #[tokio::test]
    async fn test_integration_manager_initialization() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        // This should initialize all components
        let result = manager.initialize().await;
        assert!(result.is_ok());
        
        assert!(manager.ruv_fann_manager.is_some());
        assert!(manager.daa_manager.is_some());
        assert!(manager.fact_manager.is_some());
    }

    #[tokio::test]
    async fn test_system_health_check() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        manager.initialize().await.unwrap();
        manager.update_health().await.unwrap();
        
        let health = manager.get_health().await;
        assert_eq!(health.overall_status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_helper_functions() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        manager.initialize().await.unwrap();
        manager.start().await.unwrap();
        
        // Test boundary detection network creation
        let network_id = manager.create_boundary_detection_network(10).await;
        assert!(network_id.is_ok());
        
        // Test DAA agent spawning
        let agent_id = manager.spawn_daa_agent(
            "test_agent".to_string(),
            vec!["capability1".to_string(), "capability2".to_string()]
        ).await;
        assert!(agent_id.is_ok());
        
        // Test FACT caching
        let cache_result = manager.cache_data("test_key", &"test_value", None).await;
        assert!(cache_result.is_ok());
        
        let retrieved: Result<Option<String>> = manager.get_cached_data("test_key").await;
        assert!(retrieved.is_ok());
        
        manager.stop().await.unwrap();
    }

    #[test]
    fn test_default_configs() {
        let config = IntegrationConfig::default();
        
        assert!(config.enable_ruv_fann);
        assert!(config.enable_daa);
        assert!(config.enable_fact);
        
        assert_eq!(config.ruv_fann.max_neurons, 1000);
        assert_eq!(config.daa.max_agents, 10);
        assert_eq!(config.fact.cache_size_mb, 256);
    }
}