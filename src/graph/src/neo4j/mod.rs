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
    pub cache_config: CacheConfig,
}

/// Cache configuration for Neo4j queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub max_size: usize,
    pub ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 1000,
            ttl_seconds: 300, // 5 minutes
        }
    }
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
            cache_config: CacheConfig::default(),
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

