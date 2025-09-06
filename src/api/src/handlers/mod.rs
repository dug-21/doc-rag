// Enhanced handlers with ruv-FANN integration
pub mod documents;
pub mod queries;
pub mod health;
pub mod metrics;
pub mod auth;
pub mod files;
pub mod admin;

// Re-export common handler utilities
pub use documents::*;
pub use queries::*;
pub use health::*;
pub use metrics::*;
pub use auth::*;
pub use files::*;
pub use admin::*;

// Re-export ruv-FANN enhanced handlers from enhanced_handlers module
pub use crate::enhanced_handlers::{
    handle_query, handle_upload, handle_system_dependencies,
    initialize_ruv_fann, initialize_fact_cache, initialize_daa_orchestrator,
    QueryRequest, QueryResponse, UploadResponse, SystemDependencies
};