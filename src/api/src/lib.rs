pub mod config;
pub mod server;
pub mod routes;
pub mod handlers;
pub mod enhanced_handlers;  // ruv-FANN enhanced handlers
pub mod middleware;
pub mod clients;
pub mod models;
pub mod errors;
pub mod metrics;
pub mod tracing;
pub mod security;
pub mod validation;
pub mod pipeline;  // Phase 2 pipeline with mandatory dependencies
pub mod integration;  // Phase 1 Integration: ruv-FANN, DAA, and FACT

#[cfg(test)]
pub mod integration_test;  // Standalone integration tests

pub use config::ApiConfig;
pub use server::ApiServer;
pub use errors::ApiError;
pub use models::*;
pub use integration::{IntegrationManager, IntegrationConfig, SystemHealth};

// Re-export commonly used types
pub type Result<T> = std::result::Result<T, ApiError>;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

// API versioning
pub const API_VERSION: &str = "v1";
pub const API_PREFIX: &str = "/api/v1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert_eq!(NAME, "api");
    }

    #[test]
    fn test_api_constants() {
        assert_eq!(API_VERSION, "v1");
        assert_eq!(API_PREFIX, "/api/v1");
    }
}