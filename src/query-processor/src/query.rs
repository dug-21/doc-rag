//! Query data structures and processing
//!
//! This module defines the core Query and ProcessedQuery structures that flow
//! through the processing pipeline, along with metadata and validation.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::types::{
    ConsensusResult, ExtractedEntity, IntentClassification, KeyTerm, SemanticAnalysis,
    StrategySelection,
};
use crate::error::{ProcessorError, Result};

/// Input query to be processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    /// Unique query identifier
    id: Uuid,
    /// Original query text
    text: String,
    /// Query metadata
    metadata: QueryMetadata,
    /// Creation timestamp
    created_at: DateTime<Utc>,
    /// Additional context or parameters
    context: HashMap<String, serde_json::Value>,
}

/// Query metadata for tracking and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetadata {
    /// Source of the query (user, system, test, etc.)
    pub source: String,
    /// User ID or session ID
    pub user_id: Option<String>,
    /// Session ID for tracking related queries
    pub session_id: Option<String>,
    /// Query priority level
    pub priority: QueryPriority,
    /// Expected response time requirement
    pub expected_response_time: Option<std::time::Duration>,
    /// Required accuracy level
    pub required_accuracy: Option<f64>,
    /// Language hint (if known)
    pub language_hint: Option<String>,
    /// Domain-specific hints
    pub domain_hints: Vec<String>,
    /// Custom attributes
    pub attributes: HashMap<String, String>,
}

/// Query priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum QueryPriority {
    /// Low priority - can be processed with delay
    Low,
    /// Normal priority - standard processing
    Normal,
    /// High priority - expedited processing
    High,
    /// Critical priority - immediate processing required
    Critical,
}

impl Default for QueryPriority {
    fn default() -> Self {
        QueryPriority::Normal
    }
}

/// Processed query with all analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedQuery {
    /// Original query
    pub query: Query,
    /// Semantic analysis results
    pub analysis: SemanticAnalysis,
    /// Extracted entities
    pub entities: Vec<ExtractedEntity>,
    /// Key terms
    pub key_terms: Vec<KeyTerm>,
    /// Intent classification
    pub intent: IntentClassification,
    /// Selected search strategy
    pub strategy: StrategySelection,
    /// Consensus validation result (if enabled)
    pub consensus: Option<ConsensusResult>,
    /// Processing metadata
    pub processing_metadata: ProcessingMetadata,
    /// Processing timestamp
    pub processed_at: DateTime<Utc>,
}

/// Processing metadata and performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Total processing time
    pub total_duration: std::time::Duration,
    /// Time spent in each stage
    pub stage_durations: HashMap<String, std::time::Duration>,
    /// Processing statistics
    pub statistics: ProcessingStatistics,
    /// Validation results
    pub validation_results: Vec<ValidationResult>,
    /// Warnings encountered during processing
    pub warnings: Vec<String>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    /// Number of entities extracted
    pub entity_count: usize,
    /// Number of key terms extracted
    pub key_term_count: usize,
    /// Overall processing confidence
    pub overall_confidence: f64,
    /// Quality score
    pub quality_score: f64,
    /// Complexity assessment
    pub complexity_score: f64,
    /// Resource usage
    pub resource_usage: ResourceUsage,
}

/// Resource usage during processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Peak memory usage in bytes
    pub peak_memory: u64,
    /// CPU time used
    pub cpu_time: std::time::Duration,
    /// Number of API calls made
    pub api_calls: u32,
    /// Cache hits
    pub cache_hits: u32,
    /// Cache misses
    pub cache_misses: u32,
}

/// Validation result for a processing stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Stage that was validated
    pub stage: String,
    /// Validation status
    pub status: ValidationStatus,
    /// Validation message
    pub message: String,
    /// Validation score (0.0 to 1.0)
    pub score: f64,
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
}

/// Validation status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationStatus {
    /// Validation passed
    Passed,
    /// Validation passed with warnings
    PassedWithWarnings,
    /// Validation failed
    Failed,
    /// Validation was skipped
    Skipped,
}

/// Performance metrics for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Throughput (queries per second)
    pub throughput: f64,
    /// Latency percentiles
    pub latency_percentiles: LatencyPercentiles,
    /// Error rate
    pub error_rate: f64,
    /// Success rate
    pub success_rate: f64,
    /// Resource efficiency score
    pub efficiency_score: f64,
}

/// Latency percentile measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    /// 50th percentile (median)
    pub p50: std::time::Duration,
    /// 95th percentile
    pub p95: std::time::Duration,
    /// 99th percentile
    pub p99: std::time::Duration,
    /// 99.9th percentile
    pub p999: std::time::Duration,
}

impl Query {
    /// Create a new query with default metadata
    pub fn new(text: impl Into<String>) -> Self {
        let text = text.into();
        
        // Basic validation
        if text.trim().is_empty() {
            panic!("Query text cannot be empty");
        }
        
        Self {
            id: Uuid::new_v4(),
            text,
            metadata: QueryMetadata::default(),
            created_at: Utc::now(),
            context: HashMap::new(),
        }
    }

    /// Create a new query with custom metadata
    pub fn with_metadata(text: impl Into<String>, metadata: QueryMetadata) -> Self {
        let text = text.into();
        
        if text.trim().is_empty() {
            panic!("Query text cannot be empty");
        }
        
        Self {
            id: Uuid::new_v4(),
            text,
            metadata,
            created_at: Utc::now(),
            context: HashMap::new(),
        }
    }

    /// Get the query ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the query text
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get query metadata
    pub fn metadata(&self) -> &QueryMetadata {
        &self.metadata
    }

    /// Get creation timestamp
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Get context value
    pub fn get_context(&self, key: &str) -> Option<&serde_json::Value> {
        self.context.get(key)
    }

    /// Set context value
    pub fn set_context(&mut self, key: String, value: serde_json::Value) {
        self.context.insert(key, value);
    }

    /// Add domain hint
    pub fn add_domain_hint(&mut self, hint: String) {
        self.metadata.domain_hints.push(hint);
    }

    /// Set priority
    pub fn set_priority(&mut self, priority: QueryPriority) {
        self.metadata.priority = priority;
    }

    /// Set user ID
    pub fn set_user_id(&mut self, user_id: String) {
        self.metadata.user_id = Some(user_id);
    }

    /// Set session ID
    pub fn set_session_id(&mut self, session_id: String) {
        self.metadata.session_id = Some(session_id);
    }

    /// Validate the query
    pub fn validate(&self) -> Result<()> {
        // Check text length
        if self.text.trim().is_empty() {
            return Err(ProcessorError::InvalidQuery {
                reason: "Query text cannot be empty".to_string(),
            });
        }

        if self.text.len() > 10_000 {
            return Err(ProcessorError::InvalidQuery {
                reason: "Query text is too long (max 10,000 characters)".to_string(),
            });
        }

        // Check for suspicious content
        if self.contains_suspicious_patterns() {
            return Err(ProcessorError::InvalidQuery {
                reason: "Query contains suspicious patterns".to_string(),
            });
        }

        // Validate metadata
        self.metadata.validate()?;

        Ok(())
    }

    /// Check for suspicious patterns in query text
    fn contains_suspicious_patterns(&self) -> bool {
        let text = self.text.to_lowercase();
        
        // Check for common injection patterns
        let suspicious_patterns = [
            "script>", "<iframe", "javascript:", "data:",
            "drop table", "delete from", "insert into",
            "union select", "exec(", "eval(",
        ];
        
        suspicious_patterns.iter().any(|pattern| text.contains(pattern))
    }

    /// Get query complexity estimation
    pub fn complexity_estimate(&self) -> f64 {
        let text = &self.text;
        let mut complexity = 0.0;
        
        // Base complexity from length
        complexity += (text.len() as f64 / 100.0).min(5.0);
        
        // Question words increase complexity
        let question_words = ["what", "why", "how", "when", "where", "which", "who"];
        for word in question_words {
            if text.to_lowercase().contains(word) {
                complexity += 0.5;
            }
        }
        
        // Comparison words increase complexity
        let comparison_words = ["compare", "difference", "versus", "vs", "between"];
        for word in comparison_words {
            if text.to_lowercase().contains(word) {
                complexity += 1.0;
            }
        }
        
        // Technical terms increase complexity
        let technical_indicators = ["requirement", "compliance", "security", "audit", "risk"];
        for term in technical_indicators {
            if text.to_lowercase().contains(term) {
                complexity += 0.3;
            }
        }
        
        // Multiple sentences increase complexity
        let sentence_count = text.split('.').count();
        if sentence_count > 1 {
            complexity += (sentence_count - 1) as f64 * 0.5;
        }
        
        complexity.min(10.0) // Cap at 10.0
    }
}

impl Default for QueryMetadata {
    fn default() -> Self {
        Self {
            source: "unknown".to_string(),
            user_id: None,
            session_id: None,
            priority: QueryPriority::Normal,
            expected_response_time: None,
            required_accuracy: None,
            language_hint: None,
            domain_hints: Vec::new(),
            attributes: HashMap::new(),
        }
    }
}

impl QueryMetadata {
    /// Create new metadata with source
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            ..Default::default()
        }
    }

    /// Set expected response time
    pub fn with_response_time(mut self, duration: std::time::Duration) -> Self {
        self.expected_response_time = Some(duration);
        self
    }

    /// Set required accuracy
    pub fn with_accuracy(mut self, accuracy: f64) -> Self {
        self.required_accuracy = Some(accuracy.clamp(0.0, 1.0));
        self
    }

    /// Set language hint
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language_hint = Some(language.into());
        self
    }

    /// Add domain hint
    pub fn with_domain_hint(mut self, hint: impl Into<String>) -> Self {
        self.domain_hints.push(hint.into());
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: QueryPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Validate metadata
    pub fn validate(&self) -> Result<()> {
        // Validate accuracy requirement
        if let Some(accuracy) = self.required_accuracy {
            if !(0.0..=1.0).contains(&accuracy) {
                return Err(ProcessorError::InvalidQuery {
                    reason: "Required accuracy must be between 0.0 and 1.0".to_string(),
                });
            }
        }

        // Validate response time requirement
        if let Some(duration) = self.expected_response_time {
            if duration > std::time::Duration::from_secs(300) {
                return Err(ProcessorError::InvalidQuery {
                    reason: "Expected response time cannot exceed 5 minutes".to_string(),
                });
            }
        }

        Ok(())
    }
}

impl ProcessedQuery {
    /// Create a new processed query
    pub fn new(
        query: Query,
        analysis: SemanticAnalysis,
        entities: Vec<ExtractedEntity>,
        key_terms: Vec<KeyTerm>,
        intent: IntentClassification,
        strategy: StrategySelection,
    ) -> Self {
        let processing_metadata = ProcessingMetadata {
            total_duration: std::time::Duration::ZERO,
            stage_durations: HashMap::new(),
            statistics: ProcessingStatistics::default(),
            validation_results: Vec::new(),
            warnings: Vec::new(),
            performance_metrics: PerformanceMetrics::default(),
        };

        Self {
            query,
            analysis,
            entities,
            key_terms,
            intent,
            strategy,
            consensus: None,
            processing_metadata,
            processed_at: Utc::now(),
        }
    }

    /// Get the query ID
    pub fn id(&self) -> Uuid {
        self.query.id()
    }

    /// Get the original query text
    pub fn text(&self) -> &str {
        self.query.text()
    }

    /// Add processing duration for a stage
    pub fn add_stage_duration(&mut self, stage: String, duration: std::time::Duration) {
        self.processing_metadata.stage_durations.insert(stage, duration);
        self.update_total_duration();
    }

    /// Update total processing duration
    fn update_total_duration(&mut self) {
        self.processing_metadata.total_duration = self
            .processing_metadata
            .stage_durations
            .values()
            .sum();
    }

    /// Add validation result
    pub fn add_validation_result(&mut self, result: ValidationResult) {
        self.processing_metadata.validation_results.push(result);
    }

    /// Add warning message
    pub fn add_warning(&mut self, warning: String) {
        self.processing_metadata.warnings.push(warning);
    }

    /// Set consensus result
    pub fn set_consensus(&mut self, consensus: ConsensusResult) {
        self.consensus = Some(consensus);
    }

    /// Get overall processing confidence
    pub fn overall_confidence(&self) -> f64 {
        let mut confidence_sum = 0.0;
        let mut confidence_count = 0;

        // Intent confidence
        confidence_sum += self.intent.primary_confidence;
        confidence_count += 1;

        // Strategy confidence
        confidence_sum += self.strategy.primary_confidence;
        confidence_count += 1;

        // Entity confidences (average)
        if !self.entities.is_empty() {
            let entity_confidence: f64 = self
                .entities
                .iter()
                .map(|e| e.entity.confidence)
                .sum::<f64>()
                / self.entities.len() as f64;
            confidence_sum += entity_confidence;
            confidence_count += 1;
        }

        // Consensus confidence (if available)
        if let Some(ref consensus) = self.consensus {
            confidence_sum += consensus.agreement_ratio;
            confidence_count += 1;
        }

        if confidence_count > 0 {
            confidence_sum / confidence_count as f64
        } else {
            0.0
        }
    }

    /// Get processing quality score
    pub fn quality_score(&self) -> f64 {
        self.processing_metadata.statistics.quality_score
    }

    /// Check if processing meets quality thresholds
    pub fn meets_quality_threshold(&self, threshold: f64) -> bool {
        self.overall_confidence() >= threshold && self.quality_score() >= threshold
    }

    /// Get summary information
    pub fn summary(&self) -> ProcessingSummary {
        ProcessingSummary {
            query_id: self.query.id(),
            query_text: self.query.text().to_string(),
            intent: self.intent.primary_intent.clone(),
            entity_count: self.entities.len(),
            key_term_count: self.key_terms.len(),
            strategy: self.strategy.primary_strategy.clone(),
            confidence: self.overall_confidence(),
            quality_score: self.quality_score(),
            processing_duration: self.processing_metadata.total_duration,
            warnings: self.processing_metadata.warnings.len(),
            consensus_reached: self.consensus.as_ref().map(|c| c.consensus_reached),
        }
    }
}

/// Summary of processing results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingSummary {
    /// Query ID
    pub query_id: Uuid,
    /// Original query text
    pub query_text: String,
    /// Classified intent
    pub intent: crate::types::QueryIntent,
    /// Number of entities extracted
    pub entity_count: usize,
    /// Number of key terms extracted
    pub key_term_count: usize,
    /// Selected strategy
    pub strategy: crate::types::SearchStrategy,
    /// Overall confidence
    pub confidence: f64,
    /// Quality score
    pub quality_score: f64,
    /// Processing duration
    pub processing_duration: std::time::Duration,
    /// Number of warnings
    pub warnings: usize,
    /// Whether consensus was reached
    pub consensus_reached: Option<bool>,
}

impl Default for ProcessingStatistics {
    fn default() -> Self {
        Self {
            entity_count: 0,
            key_term_count: 0,
            overall_confidence: 0.0,
            quality_score: 0.0,
            complexity_score: 0.0,
            resource_usage: ResourceUsage::default(),
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            peak_memory: 0,
            cpu_time: std::time::Duration::ZERO,
            api_calls: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency_percentiles: LatencyPercentiles::default(),
            error_rate: 0.0,
            success_rate: 1.0,
            efficiency_score: 0.0,
        }
    }
}

impl Default for LatencyPercentiles {
    fn default() -> Self {
        Self {
            p50: std::time::Duration::ZERO,
            p95: std::time::Duration::ZERO,
            p99: std::time::Duration::ZERO,
            p999: std::time::Duration::ZERO,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_creation() {
        let query = Query::new("What is PCI DSS?");
        assert_eq!(query.text(), "What is PCI DSS?");
        assert!(query.validate().is_ok());
    }

    #[test]
    fn test_empty_query_validation() {
        let result = std::panic::catch_unwind(|| Query::new(""));
        assert!(result.is_err());
    }

    #[test]
    fn test_query_complexity() {
        let simple_query = Query::new("What is PCI DSS?");
        let complex_query = Query::new("Compare the encryption requirements between PCI DSS 3.2.1 and 4.0, specifically focusing on key management and vulnerability assessment procedures.");
        
        assert!(complex_query.complexity_estimate() > simple_query.complexity_estimate());
    }

    #[test]
    fn test_suspicious_content_detection() {
        let malicious_query = Query::new("What is PCI DSS? <script>alert('xss')</script>");
        assert!(malicious_query.validate().is_err());
    }

    #[test]
    fn test_metadata_validation() {
        let mut metadata = QueryMetadata::default();
        metadata.required_accuracy = Some(1.5); // Invalid accuracy
        
        let query = Query::with_metadata("Test query", metadata);
        assert!(query.validate().is_err());
    }

    #[test]
    fn test_processed_query_confidence() {
        use crate::types::*;
        
        let query = Query::new("Test query");
        let analysis = SemanticAnalysis {
            syntactic_features: SyntacticFeatures {
                pos_tags: vec![],
                named_entities: vec![],
                noun_phrases: vec![],
                verb_phrases: vec![],
                question_words: vec![],
            },
            semantic_features: SemanticFeatures {
                semantic_roles: vec![],
                coreferences: vec![],
                sentiment: None,
                similarity_vectors: vec![],
            },
            dependencies: vec![],
            topics: vec![],
            confidence: 0.8,
        };
        
        let intent = IntentClassification {
            primary_intent: QueryIntent::Factual,
            primary_confidence: 0.9,
            secondary_intents: vec![],
            metadata: ClassificationMetadata {
                classifier: "test".to_string(),
                classified_at: Utc::now(),
                features: vec![],
                alternatives: vec![],
            },
        };
        
        let strategy = StrategySelection {
            primary_strategy: SearchStrategy::VectorSimilarity,
            primary_confidence: 0.85,
            fallback_strategies: vec![],
            reasoning: "Test reasoning".to_string(),
            predictions: PerformancePrediction {
                accuracy: 0.9,
                latency: std::time::Duration::from_millis(100),
                recall: 0.85,
                precision: 0.90,
                confidence: 0.8,
            },
            metadata: SelectionMetadata {
                algorithm: "test".to_string(),
                selected_at: Utc::now(),
                factors: vec![],
                characteristics: HashMap::new(),
            },
        };
        
        let processed = ProcessedQuery::new(
            query,
            analysis,
            vec![],
            vec![],
            intent,
            strategy,
        );
        
        let confidence = processed.overall_confidence();
        assert!(confidence > 0.0 && confidence <= 1.0);
    }
}