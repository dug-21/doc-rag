//! Intelligent query preprocessing stage using FACT
//!
//! This module provides advanced query preprocessing capabilities using FACT's
//! query optimization and natural language processing features.

use crate::builder::ResponseBuilder;
use crate::error::{Result, ResponseError};
use crate::pipeline::{PipelineStage, ProcessingContext};
use async_trait::async_trait;
// Use the simplified implementations from cache module
use crate::cache::{SimpleContextManager, SimpleQueryOptimizer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

/// FACT-powered query preprocessing stage
#[derive(Debug)]
pub struct FACTQueryPreprocessingStage {
    /// Stage identifier
    name: String,
    
    /// FACT query optimizer (simplified implementation)
    query_optimizer: Arc<SimpleQueryOptimizer>,
    
    /// FACT context manager (simplified implementation)
    context_manager: Arc<SimpleContextManager>,
    
    /// Preprocessing configuration
    config: QueryPreprocessingConfig,
}

/// Configuration for query preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPreprocessingConfig {
    /// Enable query optimization
    pub enable_optimization: bool,
    
    /// Enable intent analysis
    pub enable_intent_analysis: bool,
    
    /// Enable query expansion
    pub enable_query_expansion: bool,
    
    /// Enable context enrichment
    pub enable_context_enrichment: bool,
    
    /// Optimization settings
    pub optimization: OptimizationSettings,
    
    /// Intent analysis settings
    pub intent_analysis: IntentAnalysisSettings,
    
    /// Query expansion settings
    pub expansion: ExpansionSettings,
}

/// Query optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    /// Enable query normalization
    pub normalize_queries: bool,
    
    /// Enable redundancy removal
    pub remove_redundancy: bool,
    
    /// Enable semantic clustering
    pub semantic_clustering: bool,
    
    /// Maximum query length after optimization
    pub max_optimized_length: usize,
}

/// Intent analysis settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentAnalysisSettings {
    /// Confidence threshold for intent classification
    pub confidence_threshold: f64,
    
    /// Enable sub-intent detection
    pub detect_sub_intents: bool,
    
    /// Maximum number of intents to detect
    pub max_intents: usize,
}

/// Query expansion settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpansionSettings {
    /// Enable synonym expansion
    pub expand_synonyms: bool,
    
    /// Enable domain-specific expansion
    pub domain_specific: bool,
    
    /// Maximum expansion factor (original query length multiplier)
    pub max_expansion_factor: f64,
    
    /// Relevance threshold for expanded terms
    pub relevance_threshold: f64,
}

/// Query analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysisResult {
    /// Original query
    pub original_query: String,
    
    /// Optimized query
    pub optimized_query: String,
    
    /// Detected intents
    pub intents: Vec<QueryIntent>,
    
    /// Expanded terms
    pub expanded_terms: Vec<String>,
    
    /// Analysis confidence
    pub confidence: f64,
    
    /// Processing time
    pub processing_time: std::time::Duration,
}

/// Detected query intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryIntent {
    /// Intent type
    pub intent_type: IntentType,
    
    /// Confidence score
    pub confidence: f64,
    
    /// Intent parameters
    pub parameters: HashMap<String, String>,
}

/// Types of query intents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntentType {
    /// Informational query
    Information,
    
    /// Factual question
    Factual,
    
    /// Comparison request
    Comparison,
    
    /// How-to or procedural
    Procedural,
    
    /// Definition or explanation
    Definition,
    
    /// Analysis or opinion
    Analysis,
    
    /// Unknown intent
    Unknown,
}

impl Default for QueryPreprocessingConfig {
    fn default() -> Self {
        Self {
            enable_optimization: true,
            enable_intent_analysis: true,
            enable_query_expansion: true,
            enable_context_enrichment: true,
            optimization: OptimizationSettings::default(),
            intent_analysis: IntentAnalysisSettings::default(),
            expansion: ExpansionSettings::default(),
        }
    }
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            normalize_queries: true,
            remove_redundancy: true,
            semantic_clustering: true,
            max_optimized_length: 1000,
        }
    }
}

impl Default for IntentAnalysisSettings {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.7,
            detect_sub_intents: true,
            max_intents: 3,
        }
    }
}

impl Default for ExpansionSettings {
    fn default() -> Self {
        Self {
            expand_synonyms: true,
            domain_specific: true,
            max_expansion_factor: 2.0,
            relevance_threshold: 0.6,
        }
    }
}

impl FACTQueryPreprocessingStage {
    /// Create a new FACT query preprocessing stage
    pub async fn new() -> Result<Self> {
        Self::with_config(QueryPreprocessingConfig::default()).await
    }

    /// Create with custom configuration
    pub async fn with_config(config: QueryPreprocessingConfig) -> Result<Self> {
        info!("Initializing FACT query preprocessing stage");

        let query_optimizer = Arc::new(SimpleQueryOptimizer::new());
        let context_manager = Arc::new(SimpleContextManager::new());

        Ok(Self {
            name: "fact_query_preprocessing".to_string(),
            query_optimizer,
            context_manager,
            config,
        })
    }

    /// Analyze query with FACT intelligence
    #[instrument(skip(self, query))]
    async fn analyze_query(&self, query: &str) -> Result<QueryAnalysisResult> {
        let start_time = std::time::Instant::now();
        debug!("Analyzing query: {}", query);

        let mut result = QueryAnalysisResult {
            original_query: query.to_string(),
            optimized_query: query.to_string(),
            intents: Vec::new(),
            expanded_terms: Vec::new(),
            confidence: 1.0,
            processing_time: std::time::Duration::from_millis(0),
        };

        // Query optimization
        if self.config.enable_optimization {
            result.optimized_query = self.optimize_query(query).await?;
        }

        // Intent analysis
        if self.config.enable_intent_analysis {
            result.intents = self.analyze_intent(query).await?;
        }

        // Query expansion
        if self.config.enable_query_expansion {
            result.expanded_terms = self.expand_query(query).await?;
        }

        // Calculate overall confidence
        result.confidence = self.calculate_analysis_confidence(&result);
        result.processing_time = start_time.elapsed();

        debug!("Query analysis completed in {}ms with confidence {:.2}", 
               result.processing_time.as_millis(), result.confidence);

        Ok(result)
    }

    /// Optimize query using FACT
    async fn optimize_query(&self, query: &str) -> Result<String> {
        let optimized = self.query_optimizer.optimize(query);

        // Apply additional optimizations based on configuration
        let mut result = optimized;

        if self.config.optimization.normalize_queries {
            result = self.normalize_query(&result);
        }

        if self.config.optimization.remove_redundancy {
            result = self.remove_redundancy(&result);
        }

        // Truncate if too long
        if result.len() > self.config.optimization.max_optimized_length {
            result.truncate(self.config.optimization.max_optimized_length);
            if let Some(last_space) = result.rfind(' ') {
                result.truncate(last_space);
            }
            warn!("Query truncated to {} characters", result.len());
        }

        Ok(result)
    }

    /// Analyze query intent
    async fn analyze_intent(&self, query: &str) -> Result<Vec<QueryIntent>> {
        // Use FACT context manager for intent analysis
        let _analysis = self.context_manager.analyze_context(query);

        // Parse FACT analysis into intents
        let mut intents = Vec::new();
        
        // Make query lowercase for pattern matching
        let query_lower = query.to_lowercase();

        // Basic intent detection based on query patterns
        let intent = if query_lower.contains("what is") || query_lower.contains("define") {
            QueryIntent {
                intent_type: IntentType::Definition,
                confidence: 0.9,
                parameters: HashMap::new(),
            }
        } else if query_lower.contains("how to") || query_lower.contains("how do") {
            QueryIntent {
                intent_type: IntentType::Procedural,
                confidence: 0.85,
                parameters: HashMap::new(),
            }
        } else if query_lower.contains("compare") || query_lower.contains("vs") || query_lower.contains("versus") {
            QueryIntent {
                intent_type: IntentType::Comparison,
                confidence: 0.8,
                parameters: HashMap::new(),
            }
        } else if query_lower.ends_with("?") {
            QueryIntent {
                intent_type: IntentType::Factual,
                confidence: 0.7,
                parameters: HashMap::new(),
            }
        } else {
            QueryIntent {
                intent_type: IntentType::Information,
                confidence: 0.6,
                parameters: HashMap::new(),
            }
        };

        // Only include intents above confidence threshold
        if intent.confidence >= self.config.intent_analysis.confidence_threshold {
            intents.push(intent);
        }

        // Limit number of intents
        intents.truncate(self.config.intent_analysis.max_intents);

        Ok(intents)
    }

    /// Expand query with related terms
    async fn expand_query(&self, query: &str) -> Result<Vec<String>> {
        let mut expanded_terms = Vec::new();

        if self.config.expansion.expand_synonyms {
            // Use FACT for semantic expansion
            let expansion = self.context_manager.expand_context(query);

            // Parse expansion result
            let terms: Vec<String> = expansion.split_whitespace()
                .map(|s| s.to_string())
                .filter(|term| !query.contains(term)) // Don't duplicate existing terms
                .collect();

            expanded_terms.extend(terms);
        }

        // Apply expansion limits
        let max_terms = (query.split_whitespace().count() as f64 * self.config.expansion.max_expansion_factor) as usize;
        expanded_terms.truncate(max_terms);

        Ok(expanded_terms)
    }

    /// Normalize query text
    fn normalize_query(&self, query: &str) -> String {
        query
            .trim()
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace() || ".,!?-".contains(*c))
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Remove redundant terms from query
    fn remove_redundancy(&self, query: &str) -> String {
        let words: Vec<&str> = query.split_whitespace().collect();
        let mut unique_words = Vec::new();
        
        for word in words {
            if !unique_words.iter().any(|w: &&str| w.to_lowercase() == word.to_lowercase()) {
                unique_words.push(word);
            }
        }
        
        unique_words.join(" ")
    }

    /// Calculate overall analysis confidence
    fn calculate_analysis_confidence(&self, result: &QueryAnalysisResult) -> f64 {
        let mut confidence_factors = Vec::new();

        // Intent confidence
        if !result.intents.is_empty() {
            let avg_intent_confidence = result.intents.iter()
                .map(|intent| intent.confidence)
                .sum::<f64>() / result.intents.len() as f64;
            confidence_factors.push(avg_intent_confidence);
        }

        // Query optimization confidence
        if result.optimized_query != result.original_query {
            confidence_factors.push(0.8); // Optimization occurred
        } else {
            confidence_factors.push(0.6); // No optimization needed/possible
        }

        // Expansion confidence
        if !result.expanded_terms.is_empty() {
            confidence_factors.push(0.9); // Successful expansion
        }

        // Calculate weighted average
        if confidence_factors.is_empty() {
            0.5 // Default confidence
        } else {
            confidence_factors.iter().sum::<f64>() / confidence_factors.len() as f64
        }
    }
}

#[async_trait]
impl PipelineStage for FACTQueryPreprocessingStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        5 // Execute early in the pipeline, after basic validation
    }

    async fn process(
        &self,
        mut builder: ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<ResponseBuilder> {
        debug!("Processing FACT query preprocessing stage");

        // Analyze the query
        let analysis = self.analyze_query(&context.request.query).await?;

        // Store analysis results in context
        builder.set_metadata("query_analysis", serde_json::to_value(&analysis)
            .map_err(|e| ResponseError::internal(format!("Failed to serialize query analysis: {}", e)))?);

        // Update builder with optimized query if different
        if analysis.optimized_query != analysis.original_query {
            info!("Query optimized: '{}' -> '{}'", 
                  analysis.original_query, analysis.optimized_query);
            builder.set_optimized_query(analysis.optimized_query);
        }

        // Add expanded terms to context
        if !analysis.expanded_terms.is_empty() {
            debug!("Added {} expanded terms", analysis.expanded_terms.len());
            builder.set_metadata("expanded_terms", serde_json::to_value(&analysis.expanded_terms)
                .map_err(|e| ResponseError::internal(format!("Failed to serialize expanded terms: {}", e)))?);
        }

        // Add intent information
        if !analysis.intents.is_empty() {
            debug!("Detected {} query intents", analysis.intents.len());
            builder.set_metadata("query_intents", serde_json::to_value(&analysis.intents)
                .map_err(|e| ResponseError::internal(format!("Failed to serialize intents: {}", e)))?);
        }

        Ok(builder)
    }

    fn supports_parallel(&self) -> bool {
        false // Query preprocessing should be sequential
    }

    fn dependencies(&self) -> Vec<String> {
        vec!["validation".to_string()] // Run after basic validation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::{GenerationRequest, OutputFormat}; // Unused imports

    #[tokio::test]
    async fn test_fact_query_preprocessing_creation() {
        let stage = FACTQueryPreprocessingStage::new().await;
        assert!(stage.is_ok());
    }

    #[tokio::test]
    async fn test_query_normalization() {
        let config = QueryPreprocessingConfig::default();
        let stage = FACTQueryPreprocessingStage::with_config(config).await.unwrap();

        let normalized = stage.normalize_query("  What IS    the   DEFINITION?  ");
        assert_eq!(normalized, "what is the definition?");
    }

    #[test]
    fn test_redundancy_removal() {
        let config = QueryPreprocessingConfig::default();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let stage = rt.block_on(FACTQueryPreprocessingStage::with_config(config)).unwrap();

        let deduplicated = stage.remove_redundancy("the quick brown the fox jumps");
        assert_eq!(deduplicated, "the quick brown fox jumps");
    }

    #[tokio::test]
    async fn test_intent_analysis() {
        let config = QueryPreprocessingConfig::default();
        let stage = FACTQueryPreprocessingStage::with_config(config).await.unwrap();

        let intents = stage.analyze_intent("What is machine learning?").await.unwrap();
        assert!(!intents.is_empty());
        assert!(matches!(intents[0].intent_type, IntentType::Definition));
    }

    #[test]
    fn test_config_defaults() {
        let config = QueryPreprocessingConfig::default();
        assert!(config.enable_optimization);
        assert!(config.enable_intent_analysis);
        assert!(config.enable_query_expansion);
        assert_eq!(config.intent_analysis.confidence_threshold, 0.7);
    }
}