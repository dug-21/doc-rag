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