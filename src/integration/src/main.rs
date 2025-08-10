//! # Integration Server Binary
//!
//! Main server binary for the Doc-RAG Integration System.
//! Provides unified API gateway and system orchestration.

use std::sync::Arc;
use tokio::signal;
use tracing::{info, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use integration::{
    SystemIntegration, IntegrationConfig,
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    initialize_tracing().await?;
    
    info!("Starting Doc-RAG Integration System v{}", integration::VERSION);
    
    // Load configuration
    let config = load_configuration().await?;
    info!("Configuration loaded for environment: {}", config.environment);
    
    // Create and start system
    let system = SystemIntegration::new(config).await?;
    info!("System initialized with ID: {}", system.id());
    
    // Start system components
    if let Err(e) = system.start().await {
        error!("Failed to start system: {}", e);
        return Err(e);
    }
    
    info!("🚀 Doc-RAG Integration System started successfully");
    info!("   System ID: {}", system.id());
    info!("   Health endpoint: http://{}:{}/health", 
        system.id(), // Placeholder for actual endpoint
        "8000" // Default port
    );
    
    // Wait for shutdown signal
    wait_for_shutdown().await;
    
    info!("📤 Shutdown signal received, stopping system...");
    
    // Graceful shutdown
    if let Err(e) = system.stop().await {
        error!("Error during shutdown: {}", e);
    } else {
        info!("✅ System stopped gracefully");
    }
    
    Ok(())
}

/// Initialize tracing with appropriate configuration
async fn initialize_tracing() -> Result<()> {
    let subscriber = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "integration=info,tower_http=debug,axum=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().json());
    
    // Add OpenTelemetry layer if Jaeger endpoint is configured
    #[cfg(feature = "tracing")]
    let subscriber = {
        if let Ok(jaeger_endpoint) = std::env::var("JAEGER_ENDPOINT") {
            use tracing_opentelemetry::OpenTelemetryLayer;
            use opentelemetry_jaeger::JaegerTraceExporter;
            
            let tracer = opentelemetry_jaeger::new_agent_pipeline()
                .with_endpoint(&jaeger_endpoint)
                .with_service_name("doc-rag-integration")
                .install_batch(opentelemetry::runtime::Tokio)
                .map_err(|e| integration::IntegrationError::TracingError(e.to_string()))?;
            
            subscriber.with(OpenTelemetryLayer::new(tracer))
        } else {
            subscriber
        }
    };
    
    subscriber.try_init()
        .map_err(|e| integration::IntegrationError::Internal(format!("Failed to initialize tracing: {}", e)))?;
    
    Ok(())
}

/// Load configuration from environment variables and config files
async fn load_configuration() -> Result<IntegrationConfig> {
    // Try to load from config file first
    let config = if let Ok(config_path) = std::env::var("CONFIG_FILE") {
        info!("Loading configuration from file: {}", config_path);
        IntegrationConfig::from_file(&config_path)?
    } else {
        // Load from environment variables
        info!("Loading configuration from environment variables");
        IntegrationConfig::from_env()?
    };
    
    // Validate configuration
    config.validate()
        .map_err(|e| integration::IntegrationError::ConfigurationError(e.to_string()))?;
    
    info!("Configuration validated successfully");
    Ok(config)
}

/// Wait for shutdown signal (SIGINT, SIGTERM)
async fn wait_for_shutdown() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    
    tokio::select! {
        _ = ctrl_c => {
            info!("Received SIGINT (Ctrl+C)");
        },
        _ = terminate => {
            info!("Received SIGTERM");
        },
    }
}
