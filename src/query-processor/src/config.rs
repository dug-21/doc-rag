//! Configuration management for the Query Processor
//!
//! This module provides comprehensive configuration options for all processor components,
//! including validation, defaults, and environment variable support.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use validator::Validate;

/// Main configuration for the Query Processor
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ProcessorConfig {
    /// Analyzer configuration
    #[validate(nested)]
    pub analyzer: AnalyzerConfig,
    
    /// Entity extraction configuration
    #[validate(nested)]
    pub entity_extractor: EntityExtractorConfig,
    
    /// Key term extraction configuration
    #[validate(nested)]
    pub term_extractor: TermExtractorConfig,
    
    /// Intent classification configuration
    #[validate(nested)]
    pub intent_classifier: IntentClassifierConfig,
    
    /// Strategy selection configuration
    #[validate(nested)]
    pub strategy_selector: StrategyConfig,
    
    /// Consensus engine configuration
    #[validate(nested)]
    pub consensus: ConsensusConfig,
    
    /// Validation engine configuration
    #[validate(nested)]
    pub validation: ValidationConfig,
    
    /// Performance settings
    #[validate(nested)]
    pub performance: PerformanceConfig,
    
    /// Observability settings
    #[validate(nested)]
    pub observability: ObservabilityConfig,
    
    /// Whether to enable Byzantine consensus validation
    pub enable_consensus: bool,
    
    /// Whether to enable citation tracking
    pub enable_citations: bool,
    
    /// Whether to enable neural processing features
    pub enable_neural: bool,
}

/// Query analyzer configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct AnalyzerConfig {
    /// Maximum query length to process
    #[validate(range(min = 1, max = 10000))]
    pub max_query_length: usize,
    
    /// Language detection confidence threshold
    #[validate(range(min = 0.0, max = 1.0))]
    pub language_confidence_threshold: f64,
    
    /// Text preprocessing options
    pub preprocessing: PreprocessingConfig,
    
    /// Semantic analysis options
    pub semantic: SemanticAnalysisConfig,
}

/// Text preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PreprocessingConfig {
    /// Enable Unicode normalization
    pub normalize_unicode: bool,
    
    /// Enable case folding
    pub case_folding: bool,
    
    /// Enable whitespace normalization
    pub normalize_whitespace: bool,
    
    /// Enable punctuation handling
    pub handle_punctuation: bool,
}

/// Semantic analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct SemanticAnalysisConfig {
    /// Enable dependency parsing
    pub enable_dependency_parsing: bool,
    
    /// Enable sentiment analysis
    pub enable_sentiment_analysis: bool,
    
    /// Enable topic modeling
    pub enable_topic_modeling: bool,
    
    /// Minimum confidence for semantic features
    #[validate(range(min = 0.0, max = 1.0))]
    pub confidence_threshold: f64,
}

/// Entity extractor configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct EntityExtractorConfig {
    /// Enable named entity recognition
    pub enable_ner: bool,
    
    /// Enable compliance-specific entity extraction
    pub enable_compliance_entities: bool,
    
    /// Enable technical term extraction
    pub enable_technical_terms: bool,
    
    /// Minimum confidence for entity extraction
    #[validate(range(min = 0.0, max = 1.0))]
    pub confidence_threshold: f64,
    
    /// Maximum number of entities to extract
    #[validate(range(min = 1, max = 1000))]
    pub max_entities: usize,
    
    /// Entity types to recognize
    pub entity_types: Vec<String>,
}

/// Key term extractor configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TermExtractorConfig {
    /// Enable TF-IDF scoring
    pub enable_tfidf: bool,
    
    /// Enable N-gram extraction
    pub enable_ngrams: bool,
    
    /// N-gram sizes to extract (1=unigrams, 2=bigrams, etc.)
    #[validate(length(min = 1, max = 5))]
    pub ngram_sizes: Vec<usize>,
    
    /// Maximum number of key terms
    #[validate(range(min = 1, max = 100))]
    pub max_terms: usize,
    
    /// Minimum term frequency
    #[validate(range(min = 1, max = 100))]
    pub min_frequency: usize,
    
    /// Stop words to filter
    pub stop_words: Vec<String>,
}

/// Intent classification configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct IntentClassifierConfig {
    /// Classification model to use
    pub model_type: ClassificationModel,
    
    /// Confidence threshold for classification
    #[validate(range(min = 0.0, max = 1.0))]
    pub confidence_threshold: f64,
    
    /// Enable multi-label classification
    pub enable_multi_label: bool,
    
    /// Maximum number of intents per query
    #[validate(range(min = 1, max = 10))]
    pub max_intents: usize,
    
    /// Training data configuration (if applicable)
    pub training: Option<TrainingConfig>,
}

/// Classification model types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassificationModel {
    /// Rule-based classifier using patterns
    RuleBased,
    /// Neural network classifier (ruv-FANN)
    Neural,
    /// Ensemble of multiple models
    Ensemble,
    /// External API-based classification
    External,
}

/// Training configuration for ML models
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct TrainingConfig {
    /// Training data file path
    pub training_data_path: String,
    
    /// Validation split ratio
    #[validate(range(min = 0.1, max = 0.5))]
    pub validation_split: f64,
    
    /// Number of training epochs
    #[validate(range(min = 1, max = 1000))]
    pub epochs: usize,
    
    /// Learning rate
    #[validate(range(min = 0.001, max = 0.1))]
    pub learning_rate: f64,
}

/// Strategy selection configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct StrategyConfig {
    /// Default search strategy
    pub default_strategy: DefaultStrategy,
    
    /// Enable adaptive strategy selection
    pub enable_adaptive: bool,
    
    /// Strategy scoring weights
    pub weights: StrategyWeights,
    
    /// Performance thresholds for strategy switching
    pub thresholds: StrategyThresholds,
}

/// Default search strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefaultStrategy {
    /// Vector similarity search
    Vector,
    /// Keyword-based search
    Keyword,
    /// Hybrid search combining multiple approaches
    Hybrid,
    /// Graph-based semantic search
    Semantic,
}

/// Weights for strategy scoring
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct StrategyWeights {
    /// Weight for query complexity
    #[validate(range(min = 0.0, max = 1.0))]
    pub complexity: f64,
    
    /// Weight for entity density
    #[validate(range(min = 0.0, max = 1.0))]
    pub entity_density: f64,
    
    /// Weight for intent confidence
    #[validate(range(min = 0.0, max = 1.0))]
    pub intent_confidence: f64,
    
    /// Weight for historical performance
    #[validate(range(min = 0.0, max = 1.0))]
    pub historical_performance: f64,
}

/// Performance thresholds for strategy selection
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct StrategyThresholds {
    /// Minimum accuracy threshold
    #[validate(range(min = 0.0, max = 1.0))]
    pub accuracy_threshold: f64,
    
    /// Maximum latency threshold
    pub latency_threshold: Duration,
    
    /// Minimum recall threshold
    #[validate(range(min = 0.0, max = 1.0))]
    pub recall_threshold: f64,
    
    /// Minimum precision threshold
    #[validate(range(min = 0.0, max = 1.0))]
    pub precision_threshold: f64,
}

/// Consensus engine configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ConsensusConfig {
    /// Consensus algorithm to use
    pub algorithm: ConsensusAlgorithm,
    
    /// Agreement threshold (Byzantine fault tolerance = 0.66)
    #[validate(range(min = 0.5, max = 1.0))]
    pub agreement_threshold: f64,
    
    /// Maximum timeout for consensus
    pub consensus_timeout: Duration,
    
    /// Number of validator agents
    #[validate(range(min = 1, max = 100))]
    pub validator_count: usize,
    
    /// Minimum number of responses required
    #[validate(range(min = 1, max = 100))]
    pub minimum_responses: usize,
}

/// Consensus algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsensusAlgorithm {
    /// Byzantine fault-tolerant consensus
    Byzantine,
    /// Simple majority voting
    Majority,
    /// Weighted voting based on confidence
    Weighted,
    /// RAFT consensus algorithm
    Raft,
}

/// Validation engine configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ValidationConfig {
    /// Enable input validation
    pub enable_input_validation: bool,
    
    /// Enable output validation
    pub enable_output_validation: bool,
    
    /// Enable consistency checking
    pub enable_consistency_checking: bool,
    
    /// Enable performance validation
    pub enable_performance_validation: bool,
    
    /// Validation rules configuration
    pub rules: ValidationRules,
}

/// Validation rules configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ValidationRules {
    /// Maximum processing time allowed
    pub max_processing_time: Duration,
    
    /// Minimum confidence score required
    #[validate(range(min = 0.0, max = 1.0))]
    pub min_confidence: f64,
    
    /// Required fields in output
    pub required_fields: Vec<String>,
    
    /// Validation constraints
    pub constraints: Vec<ValidationConstraint>,
}

/// Validation constraint definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConstraint {
    /// Field name to validate
    pub field: String,
    
    /// Constraint type
    pub constraint_type: ConstraintType,
    
    /// Constraint value
    pub value: serde_json::Value,
}

/// Types of validation constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintType {
    /// Minimum value constraint
    MinValue,
    /// Maximum value constraint
    MaxValue,
    /// Required field constraint
    Required,
    /// Pattern matching constraint
    Pattern,
    /// Custom validation function
    Custom,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct PerformanceConfig {
    /// Maximum concurrent queries
    #[validate(range(min = 1, max = 10000))]
    pub max_concurrent: usize,
    
    /// Query timeout
    pub query_timeout: Duration,
    
    /// Enable caching
    pub enable_caching: bool,
    
    /// Cache configuration
    pub cache: CacheConfig,
    
    /// Memory limits
    pub memory: MemoryConfig,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CacheConfig {
    /// Cache size (number of entries)
    #[validate(range(min = 100, max = 1000000))]
    pub size: usize,
    
    /// Cache TTL
    pub ttl: Duration,
    
    /// Enable cache compression
    pub enable_compression: bool,
    
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// Time To Live
    Ttl,
    /// Random eviction
    Random,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MemoryConfig {
    /// Maximum memory usage in bytes
    #[validate(range(min = 1048576, max = 17179869184))] // 1MB to 16GB
    pub max_memory: u64,
    
    /// Memory monitoring interval
    pub monitoring_interval: Duration,
    
    /// Enable garbage collection tuning
    pub enable_gc_tuning: bool,
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ObservabilityConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    
    /// Enable distributed tracing
    pub enable_tracing: bool,
    
    /// Enable structured logging
    pub enable_logging: bool,
    
    /// Metrics configuration
    pub metrics: MetricsConfig,
    
    /// Tracing configuration
    pub tracing: TracingConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Metrics export endpoint
    pub endpoint: String,
    
    /// Export interval
    pub export_interval: Duration,
    
    /// Enable high-cardinality metrics
    pub enable_high_cardinality: bool,
    
    /// Custom metrics to track
    pub custom_metrics: Vec<String>,
}

/// Distributed tracing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Tracing service endpoint
    pub endpoint: String,
    
    /// Service name for tracing
    pub service_name: String,
    
    /// Sampling ratio (0.0 to 1.0)
    pub sampling_ratio: f64,
    
    /// Enable trace correlation
    pub enable_correlation: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    
    /// Log format (json, text)
    pub format: String,
    
    /// Enable structured logging
    pub structured: bool,
    
    /// Log output destination
    pub output: String,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            analyzer: AnalyzerConfig::default(),
            entity_extractor: EntityExtractorConfig::default(),
            term_extractor: TermExtractorConfig::default(),
            intent_classifier: IntentClassifierConfig::default(),
            strategy_selector: StrategyConfig::default(),
            consensus: ConsensusConfig::default(),
            validation: ValidationConfig::default(),
            performance: PerformanceConfig::default(),
            observability: ObservabilityConfig::default(),
            enable_consensus: true,
            enable_citations: true,
            enable_neural: true,
        }
    }
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            max_query_length: 5000,
            language_confidence_threshold: 0.8,
            preprocessing: PreprocessingConfig::default(),
            semantic: SemanticAnalysisConfig::default(),
        }
    }
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            normalize_unicode: true,
            case_folding: true,
            normalize_whitespace: true,
            handle_punctuation: true,
        }
    }
}

impl Default for SemanticAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_dependency_parsing: true,
            enable_sentiment_analysis: false,
            enable_topic_modeling: true,
            confidence_threshold: 0.7,
        }
    }
}

impl Default for EntityExtractorConfig {
    fn default() -> Self {
        Self {
            enable_ner: true,
            enable_compliance_entities: true,
            enable_technical_terms: true,
            confidence_threshold: 0.8,
            max_entities: 50,
            entity_types: vec![
                "STANDARD".to_string(),
                "REQUIREMENT".to_string(),
                "CONTROL".to_string(),
                "TECHNICAL_TERM".to_string(),
                "ORGANIZATION".to_string(),
                "DATE".to_string(),
                "VERSION".to_string(),
            ],
        }
    }
}

impl Default for TermExtractorConfig {
    fn default() -> Self {
        Self {
            enable_tfidf: true,
            enable_ngrams: true,
            ngram_sizes: vec![1, 2, 3],
            max_terms: 20,
            min_frequency: 2,
            stop_words: vec![
                "the".to_string(), "a".to_string(), "an".to_string(),
                "and".to_string(), "or".to_string(), "but".to_string(),
                "in".to_string(), "on".to_string(), "at".to_string(),
                "to".to_string(), "for".to_string(), "of".to_string(),
                "with".to_string(), "by".to_string(), "is".to_string(),
                "are".to_string(), "was".to_string(), "were".to_string(),
            ],
        }
    }
}

impl Default for IntentClassifierConfig {
    fn default() -> Self {
        Self {
            model_type: ClassificationModel::RuleBased,
            confidence_threshold: 0.8,
            enable_multi_label: false,
            max_intents: 3,
            training: None,
        }
    }
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            default_strategy: DefaultStrategy::Hybrid,
            enable_adaptive: true,
            weights: StrategyWeights::default(),
            thresholds: StrategyThresholds::default(),
        }
    }
}

impl Default for StrategyWeights {
    fn default() -> Self {
        Self {
            complexity: 0.3,
            entity_density: 0.2,
            intent_confidence: 0.3,
            historical_performance: 0.2,
        }
    }
}

impl Default for StrategyThresholds {
    fn default() -> Self {
        Self {
            accuracy_threshold: 0.95,
            latency_threshold: Duration::from_secs(2),
            recall_threshold: 0.90,
            precision_threshold: 0.95,
        }
    }
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            algorithm: ConsensusAlgorithm::Byzantine,
            agreement_threshold: 0.66, // Byzantine fault tolerance
            consensus_timeout: Duration::from_millis(500),
            validator_count: 5,
            minimum_responses: 3,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_input_validation: true,
            enable_output_validation: true,
            enable_consistency_checking: true,
            enable_performance_validation: true,
            rules: ValidationRules::default(),
        }
    }
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            max_processing_time: Duration::from_secs(5),
            min_confidence: 0.8,
            required_fields: vec![
                "intent".to_string(),
                "entities".to_string(),
                "strategy".to_string(),
            ],
            constraints: vec![],
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 100,
            query_timeout: Duration::from_secs(5),
            enable_caching: true,
            cache: CacheConfig::default(),
            memory: MemoryConfig::default(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            size: 10000,
            ttl: Duration::from_secs(300), // 5 minutes
            enable_compression: true,
            eviction_policy: EvictionPolicy::Lru,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory: 2_147_483_648, // 2GB
            monitoring_interval: Duration::from_secs(30),
            enable_gc_tuning: true,
        }
    }
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            enable_logging: true,
            metrics: MetricsConfig::default(),
            tracing: TracingConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:9090/metrics".to_string(),
            export_interval: Duration::from_secs(15),
            enable_high_cardinality: false,
            custom_metrics: vec![],
        }
    }
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:14268/api/traces".to_string(),
            service_name: "query-processor".to_string(),
            sampling_ratio: 0.1,
            enable_correlation: true,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "json".to_string(),
            structured: true,
            output: "stdout".to_string(),
        }
    }
}

impl ProcessorConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }
    
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let config = config::Config::builder()
            .add_source(config::Environment::with_prefix("QUERY_PROCESSOR"))
            .build()?;
            
        let config: Self = config.try_deserialize()?;
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate_config(&self) -> Result<(), validator::ValidationErrors> {
        self.validate()
    }
    
    /// Get configuration summary for logging
    pub fn summary(&self) -> String {
        format!(
            "QueryProcessor Config: consensus={}, citations={}, neural={}, max_concurrent={}",
            self.enable_consensus,
            self.enable_citations, 
            self.enable_neural,
            self.performance.max_concurrent
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProcessorConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.enable_consensus);
        assert!(config.enable_citations);
        assert_eq!(config.consensus.agreement_threshold, 0.66);
    }

    #[test]
    fn test_config_serialization() {
        let config = ProcessorConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: ProcessorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.enable_consensus, deserialized.enable_consensus);
    }

    #[test]
    fn test_validation() {
        let mut config = ProcessorConfig::default();
        config.consensus.agreement_threshold = 1.5; // Invalid value
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_summary() {
        let config = ProcessorConfig::default();
        let summary = config.summary();
        assert!(summary.contains("consensus=true"));
        assert!(summary.contains("neural=true"));
    }
}