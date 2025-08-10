//! # Query Processor
//!
//! High-accuracy query processing component for RAG system with semantic understanding,
//! entity extraction, intent classification, and Byzantine fault-tolerant consensus.
//!
//! ## Architecture Overview
//!
//! The Query Processor implements a multi-stage pipeline:
//! 1. **Query Analysis** - Semantic parsing and structure analysis
//! 2. **Entity Extraction** - Named entities, key terms, and concepts
//! 3. **Intent Classification** - Query type determination (factual, comparison, etc.)
//! 4. **Strategy Selection** - Optimal search strategy based on query characteristics
//! 5. **Consensus Validation** - Byzantine fault-tolerant result validation
//!
//! ## Key Features
//!
//! - **99% Accuracy Target** - Multi-layer validation and consensus mechanisms
//! - **Sub-2s Response Time** - Optimized processing pipeline
//! - **Byzantine Fault Tolerance** - Consensus with 66% threshold
//! - **Complete Citation Tracking** - Full source attribution
//! - **Multi-language Support** - Unicode-aware text processing
//!
//! ## Usage Example
//!
//! ```rust
//! use query_processor::{QueryProcessor, ProcessorConfig, Query};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ProcessorConfig::default();
//!     let processor = QueryProcessor::new(config).await?;
//!     
//!     let query = Query::new("What are the encryption requirements for stored payment card data?");
//!     let result = processor.process(query).await?;
//!     
//!     println!("Intent: {:?}", result.intent);
//!     println!("Entities: {:?}", result.entities);
//!     println!("Strategy: {:?}", result.strategy);
//!     
//!     Ok(())
//! }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::module_name_repetitions)]

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, instrument};
use uuid::Uuid;

pub mod analyzer;
pub mod classifier;
pub mod config;
pub mod consensus;
pub mod entities;
pub mod error;
pub mod extractor;
pub mod metrics;
pub mod query;
pub mod strategy;
pub mod types;
pub mod validation;

pub use analyzer::QueryAnalyzer;
pub use classifier::{IntentClassifier, QueryIntent};
pub use config::ProcessorConfig;
pub use consensus::ConsensusEngine;
pub use entities::EntityExtractor;
pub use error::{ProcessorError, Result};
pub use extractor::KeyTermExtractor;
pub use metrics::ProcessorMetrics;
pub use query::{ProcessedQuery, Query, QueryMetadata};
pub use strategy::{SearchStrategy, StrategySelector};
pub use types::*;
pub use validation::ValidationEngine;

/// Main Query Processor that coordinates all processing stages
#[derive(Clone)]
pub struct QueryProcessor {
    /// Unique processor instance ID
    id: Uuid,
    /// Configuration settings
    config: Arc<ProcessorConfig>,
    /// Query analyzer for semantic understanding
    analyzer: Arc<QueryAnalyzer>,
    /// Entity extraction component
    entity_extractor: Arc<EntityExtractor>,
    /// Key term extraction component  
    term_extractor: Arc<KeyTermExtractor>,
    /// Intent classification component
    intent_classifier: Arc<IntentClassifier>,
    /// Strategy selection component
    strategy_selector: Arc<StrategySelector>,
    /// Consensus validation engine
    consensus_engine: Arc<ConsensusEngine>,
    /// Validation engine
    validation_engine: Arc<ValidationEngine>,
    /// Performance metrics
    metrics: Arc<RwLock<ProcessorMetrics>>,
}

impl QueryProcessor {
    /// Create a new Query Processor instance
    #[instrument(skip(config))]
    pub async fn new(config: ProcessorConfig) -> Result<Self> {
        info!("Initializing Query Processor with config: {:?}", config);
        
        let config = Arc::new(config);
        let analyzer = Arc::new(QueryAnalyzer::new(config.clone()).await?);
        let entity_extractor = Arc::new(EntityExtractor::new(config.clone()).await?);
        let term_extractor = Arc::new(KeyTermExtractor::new(config.clone()).await?);
        let intent_classifier = Arc::new(IntentClassifier::new(config.clone()).await?);
        let strategy_selector = Arc::new(StrategySelector::new(config.clone()).await?);
        let consensus_engine = Arc::new(ConsensusEngine::new(config.clone()).await?);
        let validation_engine = Arc::new(ValidationEngine::new(config.clone()).await?);
        let metrics = Arc::new(RwLock::new(ProcessorMetrics::new()));

        Ok(Self {
            id: Uuid::new_v4(),
            config,
            analyzer,
            entity_extractor,
            term_extractor,
            intent_classifier,
            strategy_selector,
            consensus_engine,
            validation_engine,
            metrics,
        })
    }

    /// Process a query through the complete pipeline
    #[instrument(skip(self, query))]
    pub async fn process(&self, query: Query) -> Result<ProcessedQuery> {
        let start = std::time::Instant::now();
        
        info!("Processing query: {}", query.id());
        
        // Stage 1: Semantic Analysis
        let analysis = self.analyzer.analyze(&query).await?;
        
        // Stage 2: Entity Extraction
        let entities = self.entity_extractor.extract(&query, &analysis).await?;
        
        // Stage 3: Key Term Extraction
        let key_terms = self.term_extractor.extract(&query, &analysis).await?;
        
        // Stage 4: Intent Classification
        let intent = self.intent_classifier.classify(&query, &analysis).await?;
        
        // Stage 5: Strategy Selection
        let strategy = self.strategy_selector
            .select(&query, &analysis, &intent, &entities)
            .await?;
        
        // Stage 6: Build processed query
        let mut processed = ProcessedQuery::new(query, analysis, entities, key_terms, intent, strategy);
        
        // Stage 7: Consensus Validation (if enabled)
        if self.config.enable_consensus {
            processed = self.consensus_engine.validate(processed).await?;
        }
        
        // Stage 8: Final Validation
        processed = self.validation_engine.validate(processed).await?;
        
        // Update metrics
        let duration = start.elapsed();
        {
            let mut metrics = self.metrics.write().await;
            metrics.record_processing(duration, &processed);
        }
        
        info!(
            "Query processing completed in {:?}ms: {}",
            duration.as_millis(),
            processed.query.id()
        );
        
        Ok(processed)
    }

    /// Process multiple queries concurrently
    #[instrument(skip(self, queries))]
    pub async fn process_batch(&self, queries: Vec<Query>) -> Result<Vec<ProcessedQuery>> {
        info!("Processing batch of {} queries", queries.len());
        
        let tasks: Vec<_> = queries
            .into_iter()
            .map(|query| {
                let processor = self.clone();
                tokio::spawn(async move { processor.process(query).await })
            })
            .collect();
        
        let mut results = Vec::new();
        for task in tasks {
            match task.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(ProcessorError::ProcessingFailed(e.to_string())),
            }
        }
        
        Ok(results)
    }

    /// Get processor ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get current metrics
    pub async fn metrics(&self) -> ProcessorMetrics {
        self.metrics.read().await.clone()
    }

    /// Get processor health status
    pub async fn health(&self) -> HealthStatus {
        let metrics = self.metrics.read().await;
        
        HealthStatus {
            processor_id: self.id,
            uptime: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default(),
            total_processed: metrics.total_processed,
            success_rate: metrics.success_rate(),
            average_latency: metrics.average_latency(),
            status: if metrics.success_rate() > 0.95 {
                "healthy".to_string()
            } else {
                "degraded".to_string()
            },
        }
    }

    /// Shutdown the processor gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down Query Processor: {}", self.id);
        
        // Export final metrics
        let metrics = self.metrics.read().await;
        info!("Final metrics: {:?}", *metrics);
        
        Ok(())
    }
}

/// Health status for the query processor
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    /// Processor instance ID
    pub processor_id: Uuid,
    /// Uptime duration
    pub uptime: std::time::Duration,
    /// Total queries processed
    pub total_processed: u64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Average processing latency
    pub average_latency: std::time::Duration,
    /// Current status
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_processor_creation() {
        let config = ProcessorConfig::default();
        let processor = QueryProcessor::new(config).await;
        assert!(processor.is_ok());
    }
    
    #[tokio::test]
    async fn test_basic_query_processing() {
        let config = ProcessorConfig::default();
        let processor = QueryProcessor::new(config).await.unwrap();
        
        let query = Query::new("What are the PCI DSS encryption requirements?");
        let result = processor.process(query).await;
        
        assert!(result.is_ok());
        let processed = result.unwrap();
        assert!(!processed.entities.is_empty());
        assert!(matches!(processed.intent, QueryIntent::Factual));
    }
    
    #[tokio::test]
    async fn test_batch_processing() {
        let config = ProcessorConfig::default();
        let processor = QueryProcessor::new(config).await.unwrap();
        
        let queries = vec![
            Query::new("What is PCI DSS?"),
            Query::new("Compare PCI DSS 3.2.1 and 4.0"),
            Query::new("Summarize encryption requirements"),
        ];
        
        let results = processor.process_batch(queries).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 3);
    }
    
    #[tokio::test]
    async fn test_health_status() {
        let config = ProcessorConfig::default();
        let processor = QueryProcessor::new(config).await.unwrap();
        
        let health = processor.health().await;
        assert_eq!(health.processor_id, processor.id());
        assert_eq!(health.total_processed, 0);
    }
}