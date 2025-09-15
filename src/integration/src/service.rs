//! # Service Lifecycle Management
//!
//! Provides robust service initialization, health monitoring, and graceful shutdown
//! for containerized environments with proper signal handling and dependency management.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::signal;
use tracing::{info, warn, error, instrument};
use uuid::Uuid;

use crate::{Result, IntegrationError, IntegrationConfig};
use crate::system::SystemIntegration;
use crate::temp_types::SystemStatus;

/// Service state enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceState {
    /// Service is initializing
    Initializing,
    /// Service is starting up
    Starting,
    /// Service is healthy and running
    Running,
    /// Service is stopping gracefully
    Stopping,
    /// Service has stopped
    Stopped,
    /// Service encountered an error
    Failed(String),
}

/// Service lifecycle manager
pub struct ServiceManager {
    /// Service ID
    id: Uuid,
    /// Configuration
    config: Arc<IntegrationConfig>,
    /// Current service state
    state: Arc<RwLock<ServiceState>>,
    /// System integration instance
    system: Option<SystemIntegration>,
    /// Shutdown signal sender
    shutdown_tx: Option<oneshot::Sender<()>>,
    /// Service start time
    start_time: Option<Instant>,
}

impl ServiceManager {
    /// Create new service manager
    pub fn new(config: IntegrationConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            config: Arc::new(config),
            state: Arc::new(RwLock::new(ServiceState::Initializing)),
            system: None,
            shutdown_tx: None,
            start_time: None,
        }
    }
    
    /// Initialize and start the service
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> Result<()> {
        info!("🚀 Starting Neurosymbolic RAG Service {}", self.id);
        
        // Update state to starting
        {
            let mut state = self.state.write().await;
            *state = ServiceState::Starting;
        }
        
        // Pre-flight checks
        self.run_preflight_checks().await?;
        
        // Initialize system integration
        let system = SystemIntegration::new((*self.config).clone()).await
            .map_err(|e| {
                self.set_failed_state(format!("Failed to create system integration: {}", e));
                e
            })?;
        
        // Start the system
        system.start().await
            .map_err(|e| {
                self.set_failed_state(format!("Failed to start system: {}", e));
                e
            })?;
        
        self.system = Some(system);
        self.start_time = Some(Instant::now());
        
        // Update state to running
        {
            let mut state = self.state.write().await;
            *state = ServiceState::Running;
        }
        
        info!("✅ Neurosymbolic RAG Service started successfully");
        
        // Setup signal handlers
        self.setup_signal_handlers().await?;
        
        Ok(())
    }
    
    /// Wait for service to complete (blocks until shutdown)
    pub async fn wait_for_completion(&mut self) -> Result<()> {
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        self.shutdown_tx = Some(shutdown_tx);
        
        // Wait for shutdown signal
        if let Err(_) = shutdown_rx.await {
            warn!("Shutdown receiver channel closed unexpectedly");
        }
        
        self.shutdown().await
    }
    
    /// Initiate graceful shutdown
    #[instrument(skip(self))]
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("📤 Initiating graceful shutdown...");
        
        // Update state to stopping
        {
            let mut state = self.state.write().await;
            *state = ServiceState::Stopping;
        }
        
        // Stop the system if it exists
        if let Some(system) = &self.system {
            if let Err(e) = system.stop().await {
                error!("Error during system shutdown: {}", e);
            }
        }
        
        // Update state to stopped
        {
            let mut state = self.state.write().await;
            *state = ServiceState::Stopped;
        }
        
        if let Some(start_time) = self.start_time {
            let uptime = start_time.elapsed();
            info!("✅ Service stopped gracefully after {:?} uptime", uptime);
        } else {
            info!("✅ Service stopped gracefully");
        }
        
        Ok(())
    }
    
    /// Get current service state
    pub async fn get_state(&self) -> ServiceState {
        self.state.read().await.clone()
    }
    
    /// Get service uptime
    pub fn get_uptime(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }
    
    /// Check if service is healthy
    pub async fn is_healthy(&self) -> bool {
        matches!(self.get_state().await, ServiceState::Running)
    }
    
    /// Run pre-flight checks before service startup
    async fn run_preflight_checks(&self) -> Result<()> {
        info!("🔍 Running pre-flight checks...");
        
        // Check configuration validity
        self.config.validate()
            .map_err(|e| IntegrationError::ConfigurationError(e.to_string()))?;
        
        // Check if we're in a Docker environment and wait for services
        if self.config.environment == "docker" || std::env::var("DOCKER_ENV").is_ok() {
            info!("🐳 Docker environment detected, waiting for dependencies...");
            self.config.wait_for_services().await
                .map_err(|e| IntegrationError::ConfigurationError(e.to_string()))?;
        }
        
        // Validate required directories exist
        self.check_required_directories().await?;
        
        // Check system resources
        self.check_system_resources().await?;
        
        info!("✅ Pre-flight checks completed successfully");
        Ok(())
    }
    
    /// Check required directories exist and are writable
    async fn check_required_directories(&self) -> Result<()> {
        let required_dirs = vec![
            "/app/logs",
            "/app/models", 
            "/app/documents",
        ];
        
        for dir in required_dirs {
            if let Err(e) = tokio::fs::create_dir_all(dir).await {
                return Err(IntegrationError::ConfigurationError(
                    format!("Failed to create directory {}: {}", dir, e)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Check system resources are adequate
    async fn check_system_resources(&self) -> Result<()> {
        // Check available memory (simplified check)
        match sys_info::mem_info() {
            Ok(mem) => {
                let available_mb = mem.avail / 1024;
                if available_mb < 512 {
                    warn!("⚠️ Low available memory: {} MB", available_mb);
                }
                info!("💾 Available memory: {} MB", available_mb);
            }
            Err(e) => {
                warn!("Failed to get memory info: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Setup signal handlers for graceful shutdown
    async fn setup_signal_handlers(&mut self) -> Result<()> {
        let state = self.state.clone();
        let shutdown_tx = self.shutdown_tx.take();
        
        tokio::spawn(async move {
            Self::wait_for_shutdown_signal().await;
            
            // Update state to stopping
            {
                let mut service_state = state.write().await;
                *service_state = ServiceState::Stopping;
            }
            
            // Signal shutdown
            if let Some(tx) = shutdown_tx {
                let _ = tx.send(());
            }
        });
        
        Ok(())
    }
    
    /// Wait for shutdown signals
    async fn wait_for_shutdown_signal() {
        let ctrl_c = async {
            signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("failed to install TERM signal handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                info!("Received SIGINT (Ctrl+C), initiating shutdown");
            },
            _ = terminate => {
                info!("Received SIGTERM, initiating shutdown");
            },
        }
    }
    
    /// Set service state to failed with error message
    fn set_failed_state(&self, error: String) {
        // This is a synchronous operation for immediate error handling
        // In a real implementation, you might use a blocking wait or different approach
        tokio::spawn({
            let state = self.state.clone();
            async move {
                let mut service_state = state.write().await;
                *service_state = ServiceState::Failed(error);
            }
        });
    }
}

impl Drop for ServiceManager {
    fn drop(&mut self) {
        info!("Service manager {} is being dropped", self.id);
    }
}

/// Main service entry point for Docker containers
pub async fn run_service() -> Result<()> {
    // Load configuration
    let config = IntegrationConfig::from_env_with_docker_detection()
        .map_err(|e| IntegrationError::ConfigurationError(e.to_string()))?;
    
    // Create and start service manager
    let mut service_manager = ServiceManager::new(config);
    
    // Start the service
    service_manager.start().await?;
    
    // Wait for completion (blocks until shutdown signal)
    service_manager.wait_for_completion().await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_service_manager_creation() {
        let config = IntegrationConfig::default();
        let manager = ServiceManager::new(config);
        
        let state = manager.get_state().await;
        assert_eq!(state, ServiceState::Initializing);
    }
    
    #[tokio::test]
    async fn test_service_state_transitions() {
        let config = IntegrationConfig::default();
        let manager = ServiceManager::new(config);
        
        // Initial state
        assert_eq!(manager.get_state().await, ServiceState::Initializing);
        
        // Update to running
        {
            let mut state = manager.state.write().await;
            *state = ServiceState::Running;
        }
        
        assert_eq!(manager.get_state().await, ServiceState::Running);
        assert!(manager.is_healthy().await);
    }
}