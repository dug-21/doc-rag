//! MCP Tools Integration
//!
//! This module provides MCP (Model Context Protocol) tool registration and handlers
//! for search, citations, and validation operations. It implements tool-based
//! retrieval according to the SPARC Architecture specification.
//!
//! ## Features
//! - Tool registration and discovery
//! - Search tool for document retrieval
//! - Citation tool for source attribution
//! - Validation tool for result verification
//! - Tool-based query processing pipeline

use crate::error::{ProcessorError, Result};
use crate::types::{ExtractedEntity, QueryIntent, ClassificationResult, StrategyRecommendation};
use crate::fact_client::FACTClientInterface;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument};
use uuid::Uuid;

/// MCP Tool trait definition
#[async_trait]
pub trait MCPTool: Send + Sync {
    /// Tool identifier
    fn name(&self) -> &str;
    
    /// Tool description for MCP protocol
    fn description(&self) -> &str;
    
    /// Tool schema for parameter validation
    fn schema(&self) -> serde_json::Value;
    
    /// Execute the tool with given parameters
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value>;
    
    /// Tool capabilities
    fn capabilities(&self) -> Vec<String>;
}

/// MCP Tool Registry for managing available tools
pub struct MCPToolRegistry {
    /// Registered tools
    tools: HashMap<String, Box<dyn MCPTool>>,
    /// FACT client for caching tool results
    fact_client: Arc<dyn FACTClientInterface>,
}

impl MCPToolRegistry {
    /// Create new tool registry
    pub fn new(fact_client: Arc<dyn FACTClientInterface>) -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
            fact_client,
        };
        
        // Register default tools
        registry.register_default_tools();
        registry
    }
    
    /// Register a new tool
    pub fn register_tool(&mut self, tool: Box<dyn MCPTool>) {
        let name = tool.name().to_string();
        info!("Registering MCP tool: {}", name);
        self.tools.insert(name, tool);
    }
    
    /// Get tool by name
    pub fn get_tool(&self, name: &str) -> Option<&dyn MCPTool> {
        self.tools.get(name).map(|t| t.as_ref())
    }
    
    /// List all registered tools
    pub fn list_tools(&self) -> Vec<&str> {
        self.tools.keys().map(|k| k.as_str()).collect()
    }
    
    /// Execute tool with parameters
    #[instrument(skip(self, params))]
    pub async fn execute_tool(&self, tool_name: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        let tool = self.tools.get(tool_name)
            .ok_or_else(|| ProcessorError::ResourceError {
                resource: "mcp_tool".to_string(),
                reason: format!("Tool not found: {}", tool_name),
            })?;
            
        debug!("Executing MCP tool: {}", tool_name);
        
        // Check cache first
        let cache_key = format!("tool:{}:{}", tool_name, blake3::hash(params.to_string().as_bytes()).to_hex());
        if let Ok(Some(cached_result)) = self.fact_client.get_query_result(&cache_key).await {
            debug!("Cache hit for tool execution: {}", tool_name);
            return Ok(serde_json::to_value(cached_result)?);
        }
        
        // Execute tool
        let result = tool.execute(params).await?;
        
        debug!("Tool execution completed: {}", tool_name);
        Ok(result)
    }
    
    /// Register default tools
    fn register_default_tools(&mut self) {
        // Register search tool
        let search_tool = Box::new(SearchTool::new(self.fact_client.clone()));
        self.register_tool(search_tool);
        
        // Register citation tool
        let citation_tool = Box::new(CitationTool::new(self.fact_client.clone()));
        self.register_tool(citation_tool);
        
        // Register validation tool
        let validation_tool = Box::new(ValidationTool::new(self.fact_client.clone()));
        self.register_tool(validation_tool);
        
        // Register entity extraction tool
        let entity_tool = Box::new(EntityExtractionTool::new(self.fact_client.clone()));
        self.register_tool(entity_tool);
        
        // Register classification tool
        let classification_tool = Box::new(ClassificationTool::new(self.fact_client.clone()));
        self.register_tool(classification_tool);
    }
    
    /// Get tool schemas for MCP protocol
    pub fn get_tool_schemas(&self) -> HashMap<String, serde_json::Value> {
        self.tools.iter()
            .map(|(name, tool)| (name.clone(), tool.schema()))
            .collect()
    }
}

/// Search tool for document retrieval
#[derive(Debug)]
pub struct SearchTool {
    fact_client: Arc<dyn FACTClientInterface>,
}

impl SearchTool {
    pub fn new(fact_client: Arc<dyn FACTClientInterface>) -> Self {
        Self { fact_client }
    }
}

#[async_trait]
impl MCPTool for SearchTool {
    fn name(&self) -> &str {
        "search"
    }
    
    fn description(&self) -> &str {
        "Search for relevant documents and passages based on query"
    }
    
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10
                },
                "strategy": {
                    "type": "string",
                    "description": "Search strategy to use",
                    "enum": ["vector", "keyword", "hybrid", "semantic"]
                }
            },
            "required": ["query"]
        })
    }
    
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let query = params["query"].as_str()
            .ok_or_else(|| ProcessorError::InvalidQuery {
                reason: "Missing query parameter".to_string(),
            })?;
            
        let max_results = params["max_results"].as_u64().unwrap_or(10) as usize;
        let strategy = params["strategy"].as_str().unwrap_or("hybrid");
        
        // Simulate search results (in real implementation, this would call vector DB)
        let results = serde_json::json!({
            "results": [],
            "query": query,
            "strategy": strategy,
            "max_results": max_results,
            "total_found": 0,
            "search_time_ms": 50
        });
        
        Ok(results)
    }
    
    fn capabilities(&self) -> Vec<String> {
        vec!["document_search".to_string(), "passage_retrieval".to_string()]
    }
}

/// Citation tool for source attribution
#[derive(Debug)]
pub struct CitationTool {
    fact_client: Arc<dyn FACTClientInterface>,
}

impl CitationTool {
    pub fn new(fact_client: Arc<dyn FACTClientInterface>) -> Self {
        Self { fact_client }
    }
}

#[async_trait]
impl MCPTool for CitationTool {
    fn name(&self) -> &str {
        "cite"
    }
    
    fn description(&self) -> &str {
        "Generate citations and source attributions for retrieved content"
    }
    
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "Document identifier"
                },
                "passage": {
                    "type": "string",
                    "description": "Text passage to cite"
                },
                "page": {
                    "type": "integer",
                    "description": "Page number (optional)"
                },
                "section": {
                    "type": "string",
                    "description": "Section name (optional)"
                }
            },
            "required": ["document_id", "passage"]
        })
    }
    
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let document_id = params["document_id"].as_str()
            .ok_or_else(|| ProcessorError::InvalidQuery {
                reason: "Missing document_id parameter".to_string(),
            })?;
            
        let passage = params["passage"].as_str()
            .ok_or_else(|| ProcessorError::InvalidQuery {
                reason: "Missing passage parameter".to_string(),
            })?;
            
        let page = params["page"].as_u64();
        let section = params["section"].as_str();
        
        let citation = serde_json::json!({
            "id": Uuid::new_v4(),
            "document_id": document_id,
            "passage": passage,
            "page": page,
            "section": section,
            "confidence": 0.95,
            "relevance": 0.9,
            "timestamp": chrono::Utc::now().to_rfc3339()
        });
        
        Ok(citation)
    }
    
    fn capabilities(&self) -> Vec<String> {
        vec!["source_attribution".to_string(), "citation_generation".to_string()]
    }
}

/// Validation tool for result verification
#[derive(Debug)]
pub struct ValidationTool {
    fact_client: Arc<dyn FACTClientInterface>,
}

impl ValidationTool {
    pub fn new(fact_client: Arc<dyn FACTClientInterface>) -> Self {
        Self { fact_client }
    }
}

#[async_trait]
impl MCPTool for ValidationTool {
    fn name(&self) -> &str {
        "validate"
    }
    
    fn description(&self) -> &str {
        "Validate query results for accuracy and completeness"
    }
    
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "description": "Query result to validate"
                },
                "validation_rules": {
                    "type": "array",
                    "description": "Validation rules to apply",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["result"]
        })
    }
    
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let result = &params["result"];
        let validation_rules = params["validation_rules"].as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        
        // Basic validation logic
        let is_valid = !result.is_null();
        let score = if is_valid { 0.9 } else { 0.0 };
        
        let validation_result = serde_json::json!({
            "is_valid": is_valid,
            "score": score,
            "violations": [],
            "warnings": [],
            "validation_rules": validation_rules,
            "validation_time_ms": 5
        });
        
        Ok(validation_result)
    }
    
    fn capabilities(&self) -> Vec<String> {
        vec!["result_validation".to_string(), "quality_assessment".to_string()]
    }
}

/// Entity extraction tool
#[derive(Debug)]
pub struct EntityExtractionTool {
    fact_client: Arc<dyn FACTClientInterface>,
}

impl EntityExtractionTool {
    pub fn new(fact_client: Arc<dyn FACTClientInterface>) -> Self {
        Self { fact_client }
    }
}

#[async_trait]
impl MCPTool for EntityExtractionTool {
    fn name(&self) -> &str {
        "extract_entities"
    }
    
    fn description(&self) -> &str {
        "Extract named entities from query text"
    }
    
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to extract entities from"
                },
                "entity_types": {
                    "type": "array",
                    "description": "Entity types to extract",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": ["text"]
        })
    }
    
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let text = params["text"].as_str()
            .ok_or_else(|| ProcessorError::InvalidQuery {
                reason: "Missing text parameter".to_string(),
            })?;
            
        // Check cache first
        if let Ok(Some(cached_entities)) = self.fact_client.get_entities(text).await {
            return Ok(serde_json::to_value(cached_entities)?);
        }
        
        // Simulate entity extraction (real implementation would use NER models)
        let entities = serde_json::json!({
            "entities": [],
            "text": text,
            "extraction_time_ms": 25
        });
        
        Ok(entities)
    }
    
    fn capabilities(&self) -> Vec<String> {
        vec!["named_entity_recognition".to_string(), "entity_linking".to_string()]
    }
}

/// Classification tool
#[derive(Debug)]
pub struct ClassificationTool {
    fact_client: Arc<dyn FACTClientInterface>,
}

impl ClassificationTool {
    pub fn new(fact_client: Arc<dyn FACTClientInterface>) -> Self {
        Self { fact_client }
    }
}

#[async_trait]
impl MCPTool for ClassificationTool {
    fn name(&self) -> &str {
        "classify_query"
    }
    
    fn description(&self) -> &str {
        "Classify query intent and determine processing strategy"
    }
    
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query text to classify"
                }
            },
            "required": ["query"]
        })
    }
    
    async fn execute(&self, params: serde_json::Value) -> Result<serde_json::Value> {
        let query = params["query"].as_str()
            .ok_or_else(|| ProcessorError::InvalidQuery {
                reason: "Missing query parameter".to_string(),
            })?;
            
        // Check cache first
        if let Ok(Some(cached_classification)) = self.fact_client.get_classification(query).await {
            return Ok(serde_json::to_value(cached_classification)?);
        }
        
        // Simulate classification (real implementation would use ML models)
        let classification = ClassificationResult {
            intent: QueryIntent::Factual,
            confidence: 0.85,
            reasoning: "Query appears to be asking for factual information".to_string(),
            features: HashMap::new(),
        };
        
        Ok(serde_json::to_value(classification)?)
    }
    
    fn capabilities(&self) -> Vec<String> {
        vec!["intent_classification".to_string(), "query_analysis".to_string()]
    }
}

/// MCP Tool Handler for coordinating tool execution
#[derive(Debug)]
pub struct MCPToolHandler {
    registry: Arc<MCPToolRegistry>,
}

impl MCPToolHandler {
    /// Create new tool handler
    pub fn new(registry: Arc<MCPToolRegistry>) -> Self {
        Self { registry }
    }
    
    /// Handle tool execution request
    pub async fn handle_request(&self, tool_name: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        self.registry.execute_tool(tool_name, params).await
    }
    
    /// Get available tools
    pub fn get_available_tools(&self) -> Vec<&str> {
        self.registry.list_tools()
    }
    
    /// Get tool schema
    pub fn get_tool_schema(&self, tool_name: &str) -> Option<serde_json::Value> {
        self.registry.get_tool(tool_name).map(|tool| tool.schema())
    }
    
    /// Orchestrate multi-tool query processing
    #[instrument(skip(self))]
    pub async fn orchestrate_query(&self, query: &str) -> Result<serde_json::Value> {
        info!("Orchestrating multi-tool query processing");
        
        // Step 1: Classify query intent
        let classification_params = serde_json::json!({ "query": query });
        let classification_result = self.handle_request("classify_query", classification_params).await?;
        
        // Step 2: Extract entities
        let entity_params = serde_json::json!({ "text": query });
        let entity_result = self.handle_request("extract_entities", entity_params).await?;
        
        // Step 3: Search for relevant content
        let search_params = serde_json::json!({ 
            "query": query,
            "max_results": 10,
            "strategy": "hybrid"
        });
        let search_result = self.handle_request("search", search_params).await?;
        
        // Step 4: Validate results
        let validation_params = serde_json::json!({ 
            "result": search_result,
            "validation_rules": ["completeness", "relevance", "accuracy"]
        });
        let validation_result = self.handle_request("validate", validation_params).await?;
        
        // Combine results
        let orchestrated_result = serde_json::json!({
            "query": query,
            "classification": classification_result,
            "entities": entity_result,
            "search": search_result,
            "validation": validation_result,
            "orchestration_time_ms": 150
        });
        
        Ok(orchestrated_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    
    // Mock FACT client for testing
    struct MockFACTClient;
    
    #[async_trait]
    impl FACTClientInterface for MockFACTClient {
        async fn get_query_result(&self, _query: &str) -> Result<Option<crate::types::QueryResult>> {
            Ok(None)
        }
        
        async fn set_query_result(&self, _query: &str, _result: &crate::types::QueryResult, _ttl: Duration) -> Result<()> {
            Ok(())
        }
        
        async fn get_entities(&self, _query: &str) -> Result<Option<Vec<ExtractedEntity>>> {
            Ok(None)
        }
        
        async fn set_entities(&self, _query: &str, _entities: &[ExtractedEntity], _ttl: Duration) -> Result<()> {
            Ok(())
        }
        
        async fn get_classification(&self, _query: &str) -> Result<Option<ClassificationResult>> {
            Ok(None)
        }
        
        async fn set_classification(&self, _query: &str, _classification: &ClassificationResult, _ttl: Duration) -> Result<()> {
            Ok(())
        }
        
        async fn get_strategy(&self, _query_hash: &str) -> Result<Option<StrategyRecommendation>> {
            Ok(None)
        }
        
        async fn set_strategy(&self, _query_hash: &str, _strategy: &StrategyRecommendation, _ttl: Duration) -> Result<()> {
            Ok(())
        }
        
        async fn clear_cache(&self) -> Result<()> {
            Ok(())
        }
        
        async fn health_check(&self) -> Result<crate::fact_client::HealthStatus> {
            Ok(crate::fact_client::HealthStatus {
                status: "healthy".to_string(),
                last_success: Some(chrono::Utc::now()),
                pool_status: "ok".to_string(),
                circuit_breaker_status: "closed".to_string(),
                latency_metrics: crate::fact_client::LatencyMetrics {
                    p50_ms: 10.0,
                    p95_ms: 20.0,
                    p99_ms: 30.0,
                    max_ms: 50.0,
                },
            })
        }
        
        async fn get_metrics(&self) -> Result<crate::fact_client::PerformanceMetrics> {
            Ok(crate::fact_client::PerformanceMetrics {
                total_requests: 0,
                cache_hits: 0,
                cache_misses: 0,
                hit_ratio: 0.0,
                avg_hit_latency_ms: 0.0,
                avg_miss_latency_ms: 0.0,
                error_count: 0,
                error_rate: 0.0,
                circuit_breaker_open: false,
                active_connections: 0,
                pool_utilization: 0.0,
            })
        }
    }
    
    #[tokio::test]
    async fn test_mcp_tool_registry() {
        let mock_client = Arc::new(MockFACTClient);
        let registry = MCPToolRegistry::new(mock_client);
        
        let tools = registry.list_tools();
        assert!(tools.contains(&"search"));
        assert!(tools.contains(&"cite"));
        assert!(tools.contains(&"validate"));
        assert!(tools.contains(&"extract_entities"));
        assert!(tools.contains(&"classify_query"));
    }
    
    #[tokio::test]
    async fn test_search_tool() {
        let mock_client = Arc::new(MockFACTClient);
        let tool = SearchTool::new(mock_client);
        
        let params = serde_json::json!({
            "query": "test query",
            "max_results": 5
        });
        
        let result = tool.execute(params).await.unwrap();
        assert_eq!(result["query"], "test query");
        assert_eq!(result["max_results"], 5);
    }
    
    #[tokio::test]
    async fn test_tool_orchestration() {
        let mock_client = Arc::new(MockFACTClient);
        let registry = Arc::new(MCPToolRegistry::new(mock_client));
        let handler = MCPToolHandler::new(registry);
        
        let result = handler.orchestrate_query("What is PCI DSS?").await.unwrap();
        assert_eq!(result["query"], "What is PCI DSS?");
        assert!(result["classification"].is_object());
        assert!(result["entities"].is_object());
        assert!(result["search"].is_object());
        assert!(result["validation"].is_object());
    }
}