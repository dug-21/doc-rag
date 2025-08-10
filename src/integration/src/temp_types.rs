//! Temporary type definitions for compilation testing
//! These will be replaced with actual component types once dependencies are resolved

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// Temporary query types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    pub id: Uuid,
    pub query: String,
    pub filters: Option<HashMap<String, String>>,
    pub format: Option<ResponseFormat>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub request_id: Uuid,
    pub response: String,
    pub confidence: f64,
    pub citations: Vec<Citation>,
    pub processing_time_ms: u64,
    pub component_times: HashMap<String, u64>,
    pub format: ResponseFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub id: Uuid,
    pub source: String,
    pub reference: String,
    pub relevance: f64,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResponseFormat {
    Json,
    Text,
    Markdown,
}

impl Default for ResponseFormat {
    fn default() -> Self {
        Self::Json
    }
}

// Temporary component client types
pub struct McpAdapterClient;
pub struct ChunkerClient;
pub struct EmbedderClient;
pub struct StorageClient;
pub struct QueryProcessorClient;
pub struct ResponseGeneratorClient;

impl McpAdapterClient {
    pub fn new(_endpoint: &str) -> Self { Self }
    pub async fn health_check(&self) -> bool { true }
}

impl ChunkerClient {
    pub fn new(_endpoint: &str) -> Self { Self }
    pub async fn health_check(&self) -> bool { true }
}

impl EmbedderClient {
    pub fn new(_endpoint: &str) -> Self { Self }
    pub async fn health_check(&self) -> bool { true }
}

impl StorageClient {
    pub fn new(_endpoint: &str) -> Self { Self }
    pub async fn health_check(&self) -> bool { true }
}

impl QueryProcessorClient {
    pub fn new(_endpoint: &str) -> Self { Self }
    pub async fn health_check(&self) -> bool { true }
}

impl ResponseGeneratorClient {
    pub fn new(_endpoint: &str) -> Self { Self }
    pub async fn health_check(&self) -> bool { true }
}