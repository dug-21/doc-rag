pub mod models;
pub mod neo4j;

pub use models::*;
pub use neo4j::{Neo4jClient, Neo4jConfig};

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
// use std::collections::HashMap; // Unused for now
// use uuid::Uuid; // Unused for now

/// Main graph database trait for requirement relationships
#[async_trait]
pub trait GraphDatabase: Send + Sync {
    /// Create a document hierarchy in the graph
    async fn create_document_hierarchy(
        &self,
        document: &ProcessedDocument,
    ) -> Result<DocumentGraph>;

    /// Create a requirement node
    async fn create_requirement_node(&self, requirement: &Requirement) -> Result<RequirementNode>;

    /// Create a typed relationship between nodes
    async fn create_relationship(
        &self,
        from_id: &str,
        to_id: &str,
        relationship_type: RelationshipType,
    ) -> Result<RelationshipEdge>;

    /// Traverse requirement relationships up to max_depth
    async fn traverse_requirements(
        &self,
        start_id: &str,
        max_depth: usize,
        relationship_types: Vec<RelationshipType>,
    ) -> Result<TraversalResult>;

    /// Find requirements by properties
    async fn find_requirements(&self, filter: RequirementFilter) -> Result<Vec<Requirement>>;

    /// Get performance metrics
    async fn get_performance_metrics(&self) -> Result<GraphPerformanceMetrics>;

    /// Health check
    async fn health_check(&self) -> Result<bool>;
}

/// Performance metrics for graph operations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphPerformanceMetrics {
    pub total_queries: u64,
    pub average_query_time_ms: f64,
    pub total_nodes: u64,
    pub total_relationships: u64,
    pub cache_hit_ratio: f64,
    pub last_updated: DateTime<Utc>,
}

/// Filter for finding requirements
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RequirementFilter {
    ByDomain(String),
    ByType(String),
    ByPriority(String),
}

/// Error types for graph operations
#[derive(thiserror::Error, Debug)]
pub enum GraphError {
    #[error("Connection error: {0}")]
    Connection(#[from] neo4rs::Error),

    #[error("Query error: {message}")]
    Query { message: String },

    #[error("Schema validation error: {0}")]
    SchemaValidation(String),

    #[error("Performance threshold exceeded: {operation} took {actual_ms}ms (limit: {limit_ms}ms)")]
    PerformanceThreshold {
        operation: String,
        actual_ms: u64,
        limit_ms: u64,
    },

    #[error("Node not found: {id}")]
    NodeNotFound { id: String },

    #[error("Relationship not found between {from_id} and {to_id}")]
    RelationshipNotFound { from_id: String, to_id: String },

    #[error("Invalid graph operation: {0}")]
    InvalidOperation(String),
}

// Add From implementations for neo4rs errors
impl From<neo4rs::DeError> for GraphError {
    fn from(err: neo4rs::DeError) -> Self {
        GraphError::Query { message: format!("Deserialization error: {}", err) }
    }
}

// Note: anyhow already implements From<E> for any error that implements std::error::Error

pub type GraphResult<T> = std::result::Result<T, GraphError>;

/// Graph database configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub max_connections: usize,
    pub connection_timeout_ms: u64,
    pub query_timeout_ms: u64,
    pub enable_metrics: bool,
    pub enable_cache: bool,
    pub cache_ttl_seconds: u64,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            uri: "bolt://localhost:7687".to_string(),
            username: "neo4j".to_string(),
            password: "neo4j_password".to_string(),
            max_connections: 16,
            connection_timeout_ms: 5000,
            query_timeout_ms: 30000,
            enable_metrics: true,
            enable_cache: true,
            cache_ttl_seconds: 3600,
        }
    }
}