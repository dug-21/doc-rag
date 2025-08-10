use anyhow::Result;
use api::{
    config::ApiConfig,
    server::ApiServer,
    tracing::init_tracing,
};
use clap::Parser;
use std::sync::Arc;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "config/api.toml")]
    config: String,

    /// Server port
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Bind address
    #[arg(short, long, default_value = "0.0.0.0")]
    bind: String,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Enable development mode
    #[arg(long)]
    dev: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize tracing
    init_tracing(&args.log_level)?;

    info!("Starting Doc-RAG API Gateway v{}", env!("CARGO_PKG_VERSION"));
    info!("Loading configuration from: {}", args.config);

    // Load configuration
    let mut config = ApiConfig::from_file(&args.config).await?;
    
    // Override with command line arguments
    config.server.port = args.port;
    config.server.bind_address = args.bind.clone();
    
    if args.dev {
        config.enable_development_features();
        info!("Development mode enabled");
    }

    info!(
        "API Gateway configured - bind: {}, port: {}, workers: {}",
        config.server.bind_address, 
        config.server.port,
        config.server.worker_threads
    );

    // Create and start the server
    let server = ApiServer::new(Arc::new(config)).await?;
    
    info!("🚀 Doc-RAG API Gateway starting on {}:{}", args.bind, args.port);
    info!("📊 Metrics available at /metrics");
    info!("🔍 Health check at /health");
    info!("📖 API documentation at /docs");
    
    // Graceful shutdown handling
    let shutdown_signal = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
        info!("Shutdown signal received");
    };

    // Run server with graceful shutdown
    server.run_with_graceful_shutdown(shutdown_signal).await?;

    info!("Doc-RAG API Gateway shut down successfully");
    Ok(())
}