pub mod client;
pub mod schema;

pub use client::Neo4jClient;
pub use schema::SchemaManager;

use crate::GraphConfig;
use serde::{Deserialize, Serialize};

/// Neo4j-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neo4jConfig {
    pub base: GraphConfig,
    pub database: String,
    pub routing: bool,
    pub encrypted: bool,
    pub trust: String,
    pub user_agent: String,
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        Self {
            base: GraphConfig::default(),
            database: "neo4j".to_string(),
            routing: false,
            encrypted: false,
            trust: "TRUST_ALL_CERTIFICATES".to_string(),
            user_agent: "doc-rag-graph/1.0".to_string(),
        }
    }
}

impl From<GraphConfig> for Neo4jConfig {
    fn from(config: GraphConfig) -> Self {
        Self {
            base: config,
            ..Default::default()
        }
    }
}