//! # Integration Server Binary
//!
//! Main server binary for the Doc-RAG Integration System.
//! Provides unified API gateway and system orchestration.

use tracing::{info, error, warn};

#[cfg(feature = "tracing")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[cfg(unix)]
use tokio::signal;

use integration::{
    IntegrationConfig, SYSTEM_VERSION as VERSION,
    Result, service::ServiceManager,
};

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| integration::IntegrationError::Internal(format!("Failed to create tokio runtime: {}", e)))?;

    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    // Initialize tracing
    initialize_tracing().await?;
    
    info!("🚀 Starting Doc-RAG Neurosymbolic Integration System v{}", VERSION);
    
    // Load configuration with Docker detection
    let config = load_configuration().await?;
    info!("Configuration loaded for environment: {}", config.environment);
    info!("Service will bind to: {}", config.gateway_bind_address
        .map(|addr| addr.to_string())
        .unwrap_or_else(|| "0.0.0.0:8080".to_string()));
    
    // Create service manager with robust lifecycle management
    let mut service_manager = ServiceManager::new(config);
    
    // Start the service with comprehensive initialization
    if let Err(e) = service_manager.start().await {
        error!("❌ Failed to start service: {}", e);
        return Err(e);
    }
    
    info!("✅ Service started successfully");
    info!("   Health endpoint: http://0.0.0.0:8080/health");
    info!("   Metrics endpoint: http://0.0.0.0:9090/metrics");
    info!("   API endpoint: http://0.0.0.0:8080/api/v1/query");
    
    // Wait for service completion (handles shutdown signals)
    if let Err(e) = service_manager.wait_for_completion().await {
        error!("Service error: {}", e);
        return Err(e);
    }
    
    info!("🎉 Service completed successfully");
    Ok(())
}

/// Initialize tracing with appropriate configuration
async fn initialize_tracing() -> Result<()> {
    #[cfg(feature = "tracing")]
    {
        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "integration=info,tower_http=debug,axum=debug".into()),
            )
            .with(tracing_subscriber::fmt::layer().json());

        // Add OpenTelemetry layer if Jaeger endpoint is configured
        let subscriber = {
            if let Ok(_jaeger_endpoint) = std::env::var("JAEGER_ENDPOINT") {
                // Simplified tracing without Jaeger for compatibility
                warn!("Jaeger tracing configured but disabled for compatibility");
                subscriber
            } else {
                subscriber
            }
        };

        subscriber.try_init()
            .map_err(|e| integration::IntegrationError::Internal(format!("Failed to initialize tracing: {}", e)))?;
    }

    #[cfg(not(feature = "tracing"))]
    {
        // Initialize basic tracing without subscriber features
        tracing_subscriber::fmt::init();
    }

    Ok(())
}

/// Load configuration from environment variables and config files with Docker support
async fn load_configuration() -> Result<IntegrationConfig> {
    // Try to load from config file first
    let config = if let Ok(config_path) = std::env::var("CONFIG_FILE") {
        info!("Loading configuration from file: {}", config_path);
        IntegrationConfig::from_file(&config_path)
            .map_err(|e| integration::IntegrationError::ConfigurationError(e.to_string()))?
    } else {
        // Load from environment variables with Docker detection
        info!("Loading configuration from environment variables with Docker detection");
        IntegrationConfig::from_env_with_docker_detection()
            .map_err(|e| integration::IntegrationError::ConfigurationError(e.to_string()))?
    };
    
    // Wait for database services to be ready in containerized environments
    if config.environment == "docker" || std::env::var("DOCKER_ENV").is_ok() {
        info!("Waiting for database services to be ready...");
        config.wait_for_services().await
            .map_err(|e| integration::IntegrationError::ConfigurationError(e.to_string()))?;
    }
    
    // Validate configuration
    config.validate()
        .map_err(|e| integration::IntegrationError::ConfigurationError(e.to_string()))?;
    
    info!("Configuration loaded and validated successfully");
    Ok(config)
}

/// Wait for shutdown signal (SIGINT, SIGTERM)
async fn wait_for_shutdown() {
    let ctrl_c = async {
        #[cfg(unix)]
        {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        }
        #[cfg(not(unix))]
        {
            // Windows signal handling fallback
            std::future::pending::<()>().await;
        }
    };
    
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    
    #[cfg(unix)]
    tokio::select! {
        _ = ctrl_c => {
            info!("Received SIGINT (Ctrl+C)");
        },
        _ = terminate => {
            info!("Received SIGTERM");
        },
    }

    #[cfg(not(unix))]
    ctrl_c.await;
}
