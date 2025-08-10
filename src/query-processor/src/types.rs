//! Type definitions for the Query Processor
//!
//! This module contains all the core data structures and types used throughout
//! the query processing pipeline, including semantic analysis results, entities,
//! search strategies, and consensus validation structures.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::time::Duration;

/// Language detection result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Language {
    /// ISO 639-1 language code
    pub code: String,
    /// Human-readable language name
    pub name: String,
    /// Detection confidence (0.0 to 1.0)
    pub confidence: f64,
}

impl Language {
    pub fn new(code: String, name: String, confidence: f64) -> Self {
        Self { code, name, confidence }
    }
}

/// Text preprocessing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessedText {
    /// Original text
    pub original: String,
    /// Normalized text
    pub normalized: String,
    /// Text tokens
    pub tokens: Vec<String>,
    /// Language detection result
    pub language: Option<Language>,
    /// Text statistics
    pub statistics: TextStatistics,
}

/// Text statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStatistics {
    /// Total character count
    pub char_count: usize,
    /// Total word count
    pub word_count: usize,
    /// Total sentence count
    pub sentence_count: usize,
    /// Average word length
    pub avg_word_length: f64,
    /// Text complexity score (0.0 to 1.0)
    pub complexity_score: f64,
    /// Reading level estimation
    pub reading_level: f64,
    /// Keyword density
    pub keyword_density: HashMap<String, f64>,
}

/// Complete semantic analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    /// Syntactic features
    pub syntactic_features: SyntacticFeatures,
    /// Semantic features
    pub semantic_features: SemanticFeatures,
    /// Dependency relations
    pub dependencies: Vec<Dependency>,
    /// Topic information
    pub topics: Vec<Topic>,
    /// Overall confidence score
    pub confidence: f64,
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Processing duration
    pub processing_time: Duration,
}

impl SemanticAnalysis {
    pub fn new(
        syntactic_features: SyntacticFeatures,
        semantic_features: SemanticFeatures,
        dependencies: Vec<Dependency>,
        topics: Vec<Topic>,
        confidence: f64,
        processing_time: Duration,
    ) -> Self {
        Self {
            syntactic_features,
            semantic_features,
            dependencies,
            topics,
            confidence,
            timestamp: Utc::now(),
            processing_time,
        }
    }
}

/// Syntactic features of the text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntacticFeatures {
    /// Part-of-speech tags
    pub pos_tags: Vec<PosTag>,
    /// Named entities
    pub named_entities: Vec<NamedEntity>,
    /// Noun phrases
    pub noun_phrases: Vec<Phrase>,
    /// Verb phrases
    pub verb_phrases: Vec<Phrase>,
    /// Question words/phrases
    pub question_words: Vec<String>,
}

/// Part-of-speech tag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PosTag {
    /// Token text
    pub token: String,
    /// POS tag
    pub tag: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Confidence score
    pub confidence: f64,
}

/// Named entity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NamedEntity {
    /// Unique entity ID
    pub id: Uuid,
    /// Entity text
    pub text: String,
    /// Entity type (PERSON, ORG, STANDARD, etc.)
    pub entity_type: String,
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Confidence score
    pub confidence: f64,
    /// Normalized form (for matching)
    pub normalized: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Extraction method used
    pub extraction_method: EntityExtractionMethod,
}

impl NamedEntity {
    pub fn new(
        text: String,
        entity_type: String,
        start: usize,
        end: usize,
        confidence: f64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            normalized: text.to_lowercase(),
            text,
            entity_type,
            start,
            end,
            confidence,
            metadata: HashMap::new(),
            extraction_method: EntityExtractionMethod::RuleBased,
        }
    }
}

/// Entity extraction methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntityExtractionMethod {
    RuleBased,
    Neural,
    Hybrid,
    External,
}

/// Text phrase (noun phrase, verb phrase, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phrase {
    /// Phrase text
    pub text: String,
    /// Phrase type
    pub phrase_type: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Head word
    pub head: Option<String>,
    /// Modifiers
    pub modifiers: Vec<String>,
}

/// Semantic features of the text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFeatures {
    /// Semantic roles
    pub semantic_roles: Vec<SemanticRole>,
    /// Coreference chains
    pub coreferences: Vec<CoreferenceChain>,
    /// Sentiment analysis result
    pub sentiment: Option<Sentiment>,
    /// Semantic similarity vectors
    pub similarity_vectors: Vec<f32>,
}

/// Semantic role structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRole {
    /// Role ID
    pub id: Uuid,
    /// Predicate (usually a verb)
    pub predicate: String,
    /// Arguments (subjects, objects, etc.)
    pub arguments: Vec<Argument>,
    /// Confidence score
    pub confidence: f64,
    /// Semantic frame
    pub frame: Option<String>,
    /// Frame elements
    pub frame_elements: HashMap<String, String>,
}

impl SemanticRole {
    pub fn new(predicate: String, arguments: Vec<Argument>, confidence: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            predicate,
            arguments,
            confidence,
            frame: None,
            frame_elements: HashMap::new(),
        }
    }
}

/// Semantic argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    /// Argument text
    pub text: String,
    /// Argument role (ARG0, ARG1, etc.)
    pub role: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
}

/// Coreference chain for entity resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreferenceChain {
    /// Chain ID
    pub id: String,
    /// Mentions in the chain
    pub mentions: Vec<Mention>,
    /// Representative mention
    pub representative: Option<String>,
}

/// Coreference mention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mention {
    /// Mention text
    pub text: String,
    /// Start position
    pub start: usize,
    /// End position
    pub end: usize,
    /// Mention type
    pub mention_type: String,
}

/// Sentiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sentiment {
    /// Sentiment label (positive, negative, neutral)
    pub label: String,
    /// Sentiment score (-1.0 to 1.0)
    pub score: f64,
    /// Confidence in classification
    pub confidence: f64,
    /// Detailed emotion scores
    pub emotions: HashMap<String, f64>,
    /// Subjectivity score (0.0 objective to 1.0 subjective)
    pub subjectivity: f64,
}

impl Default for Sentiment {
    fn default() -> Self {
        Self {
            label: "neutral".to_string(),
            score: 0.0,
            confidence: 0.0,
            emotions: HashMap::new(),
            subjectivity: 0.5,
        }
    }
}

/// Dependency relationship between words
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    /// Head word
    pub head: String,
    /// Dependent word
    pub dependent: String,
    /// Dependency relation type
    pub relation: String,
    /// Head position
    pub head_pos: usize,
    /// Dependent position
    pub dependent_pos: usize,
}

/// Topic modeling result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    /// Topic ID
    pub id: String,
    /// Topic probability
    pub probability: f64,
    /// Top words for this topic
    pub words: Vec<TopicWord>,
    /// Topic label (if available)
    pub label: Option<String>,
}

/// Word in a topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicWord {
    /// Word text
    pub word: String,
    /// Probability in topic
    pub probability: f64,
}

/// Query intent types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QueryIntent {
    /// Factual information request
    Factual,
    /// Comparison between items/concepts
    Comparison,
    /// Request for analytical insights
    Analytical,
    /// Request for summary/overview
    Summary,
    /// Procedural/how-to request
    Procedural,
    /// Definition request
    Definition,
    /// Causal relationship inquiry
    Causal,
    /// Temporal/historical inquiry
    Temporal,
    /// Compliance/regulatory inquiry
    Compliance,
    /// Multi-intent query
    Multi(Vec<QueryIntent>),
    /// Unknown/unclear intent
    Unknown,
}

impl QueryIntent {
    pub fn confidence_weight(&self) -> f64 {
        match self {
            QueryIntent::Factual => 0.9,
            QueryIntent::Comparison => 0.8,
            QueryIntent::Analytical => 0.7,
            QueryIntent::Summary => 0.8,
            QueryIntent::Procedural => 0.8,
            QueryIntent::Definition => 0.9,
            QueryIntent::Causal => 0.7,
            QueryIntent::Temporal => 0.8,
            QueryIntent::Compliance => 0.9,
            QueryIntent::Multi(_) => 0.6,
            QueryIntent::Unknown => 0.3,
        }
    }
    
    pub fn is_complex(&self) -> bool {
        matches!(self, 
            QueryIntent::Analytical | 
            QueryIntent::Comparison | 
            QueryIntent::Causal |
            QueryIntent::Multi(_)
        )
    }
}

/// Intent classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentClassification {
    /// Primary intent
    pub primary_intent: QueryIntent,
    /// Secondary intents (if any)
    pub secondary_intents: Vec<QueryIntent>,
    /// Classification confidence
    pub confidence: f64,
    /// Intent probabilities
    pub probabilities: HashMap<QueryIntent, f64>,
    /// Classification method used
    pub method: ClassificationMethod,
    /// Features used for classification
    pub features: Vec<String>,
}

/// Classification methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClassificationMethod {
    RuleBased,
    Neural,
    Ensemble,
    Hybrid,
}

/// Key term with importance scoring
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KeyTerm {
    /// Term ID
    pub id: Uuid,
    /// Term text
    pub term: String,
    /// Normalized form
    pub normalized: String,
    /// TF-IDF score
    pub tfidf_score: f64,
    /// Frequency in query
    pub frequency: usize,
    /// Term category
    pub category: TermCategory,
    /// Position information
    pub positions: Vec<(usize, usize)>,
    /// Semantic importance score
    pub importance: f64,
    /// N-gram size (1 for unigram, 2 for bigram, etc.)
    pub ngram_size: usize,
}

/// Term categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TermCategory {
    TechnicalTerm,
    ComplianceKeyword,
    BusinessTerm,
    GeneralTerm,
    StopWord,
}

/// Search strategy types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SearchStrategy {
    /// Vector similarity search
    Vector {
        model: String,
        similarity_threshold: f64,
        max_results: usize,
    },
    /// Keyword-based search
    Keyword {
        algorithm: KeywordAlgorithm,
        boost_factors: HashMap<String, f64>,
        fuzzy_matching: bool,
    },
    /// Hybrid search combining multiple approaches
    Hybrid {
        strategies: Vec<SearchStrategy>,
        weights: HashMap<String, f64>,
        combination_method: CombinationMethod,
    },
    /// Semantic search using knowledge graphs
    Semantic {
        graph_traversal: GraphTraversal,
        relationship_types: Vec<String>,
        depth_limit: usize,
    },
    /// Adaptive strategy that changes based on query characteristics
    Adaptive {
        base_strategy: Box<SearchStrategy>,
        adaptation_rules: Vec<AdaptationRule>,
        learning_enabled: bool,
    },
}

/// Keyword search algorithms
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum KeywordAlgorithm {
    BM25,
    TfIdf,
    Boolean,
    Fuzzy,
}

/// Strategy combination methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CombinationMethod {
    WeightedSum,
    RankFusion,
    Cascade,
    Voting,
}

/// Graph traversal methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GraphTraversal {
    BreadthFirst,
    DepthFirst,
    PageRank,
    RandomWalk,
}

/// Adaptation rule for dynamic strategy selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AdaptationRule {
    /// Rule name
    pub name: String,
    /// Condition to trigger adaptation
    pub condition: AdaptationCondition,
    /// Action to take
    pub action: AdaptationAction,
    /// Rule priority
    pub priority: u8,
}

/// Adaptation conditions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AdaptationCondition {
    QueryComplexity(f64),
    EntityCount(usize),
    IntentConfidence(f64),
    HistoricalPerformance(f64),
    ResponseTime(Duration),
}

/// Adaptation actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AdaptationAction {
    SwitchStrategy(SearchStrategy),
    ModifyParameters(HashMap<String, f64>),
    AddFallback(SearchStrategy),
    IncreaseTimeout(Duration),
}

/// Strategy selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySelection {
    /// Selected strategy
    pub strategy: SearchStrategy,
    /// Selection confidence
    pub confidence: f64,
    /// Reasoning for selection
    pub reasoning: String,
    /// Expected performance metrics
    pub expected_metrics: PerformanceMetrics,
    /// Fallback strategies
    pub fallbacks: Vec<SearchStrategy>,
}

/// Performance metrics prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Expected accuracy
    pub expected_accuracy: f64,
    /// Expected response time
    pub expected_response_time: Duration,
    /// Expected recall
    pub expected_recall: f64,
    /// Expected precision
    pub expected_precision: f64,
    /// Resource utilization estimate
    pub resource_usage: ResourceUsage,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Network I/O in bytes
    pub network_io: u64,
    /// Disk I/O in bytes
    pub disk_io: u64,
}

/// Consensus validation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusType {
    Byzantine,
    Majority,
    Weighted,
    Raft,
    Proof,
}

/// Consensus participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusParticipant {
    /// Participant ID
    pub id: Uuid,
    /// Participant name/identifier
    pub name: String,
    /// Voting weight
    pub weight: f64,
    /// Reputation score
    pub reputation: f64,
    /// Response time history
    pub avg_response_time: Duration,
    /// Reliability score
    pub reliability: f64,
}

/// Consensus vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVote {
    /// Voter ID
    pub voter_id: Uuid,
    /// Vote value/decision
    pub vote: ConsensusDecision,
    /// Confidence in vote
    pub confidence: f64,
    /// Vote timestamp
    pub timestamp: DateTime<Utc>,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Consensus decision types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConsensusDecision {
    Approve,
    Reject,
    Abstain,
    Conditional(String),
}

/// Consensus result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    /// Final decision
    pub decision: ConsensusDecision,
    /// Agreement percentage
    pub agreement: f64,
    /// Number of participants
    pub participant_count: usize,
    /// Vote distribution
    pub vote_distribution: HashMap<ConsensusDecision, usize>,
    /// Time to reach consensus
    pub consensus_time: Duration,
    /// Byzantine fault tolerance achieved
    pub byzantine_tolerance: bool,
}

/// Validation rule types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationRule {
    Required,
    MinValue(f64),
    MaxValue(f64),
    Range(f64, f64),
    Pattern(String),
    Custom(String),
    Dependency(String),
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall validation success
    pub is_valid: bool,
    /// Validation score (0.0 to 1.0)
    pub score: f64,
    /// Rule violations
    pub violations: Vec<ValidationViolation>,
    /// Warnings (non-blocking)
    pub warnings: Vec<ValidationWarning>,
    /// Validation time
    pub validation_time: Duration,
}

/// Validation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationViolation {
    /// Rule that was violated
    pub rule: ValidationRule,
    /// Field that failed validation
    pub field: String,
    /// Description of the violation
    pub message: String,
    /// Severity level
    pub severity: ViolationSeverity,
}

/// Validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Affected field
    pub field: String,
    /// Suggested action
    pub suggestion: Option<String>,
}

/// Violation severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Cache entry for query results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry<T> {
    /// Cache key
    pub key: String,
    /// Cached value
    pub value: T,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last access timestamp
    pub last_accessed: DateTime<Utc>,
    /// Access count
    pub access_count: u64,
    /// Time to live
    pub ttl: Duration,
    /// Entry size in bytes
    pub size_bytes: u64,
    /// Compression used
    pub compressed: bool,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStatistics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Hit rate percentage
    pub hit_rate: f64,
    /// Total entries
    pub total_entries: u64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Average access time
    pub avg_access_time: Duration,
    /// Eviction count
    pub evictions: u64,
}

/// Processing stage metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageMetrics {
    /// Stage name
    pub stage_name: String,
    /// Processing time
    pub processing_time: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Error count
    pub error_count: u64,
    /// Memory usage peak
    pub peak_memory: u64,
    /// CPU utilization
    pub cpu_usage: f64,
}

/// Query complexity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityScore {
    /// Overall complexity (0.0 to 1.0)
    pub overall: f64,
    /// Syntactic complexity
    pub syntactic: f64,
    /// Semantic complexity
    pub semantic: f64,
    /// Entity complexity
    pub entity: f64,
    /// Intent complexity
    pub intent: f64,
    /// Factors contributing to complexity
    pub factors: Vec<ComplexityFactor>,
}

/// Complexity contributing factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityFactor {
    /// Factor name
    pub name: String,
    /// Factor contribution (0.0 to 1.0)
    pub contribution: f64,
    /// Factor description
    pub description: String,
}

/// Citation information for query results
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Citation {
    /// Citation ID
    pub id: Uuid,
    /// Source document ID
    pub document_id: String,
    /// Source chunk/section ID
    pub chunk_id: String,
    /// Citation text/excerpt
    pub text: String,
    /// Relevance score
    pub relevance: f64,
    /// Confidence in citation accuracy
    pub confidence: f64,
    /// Page number (if applicable)
    pub page_number: Option<u32>,
    /// Section/chapter reference
    pub section: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Neural network model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralModel {
    /// Model name/identifier
    pub name: String,
    /// Model version
    pub version: String,
    /// Model type
    pub model_type: NeuralModelType,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Output dimensions
    pub output_dims: Vec<usize>,
    /// Model parameters count
    pub parameter_count: u64,
    /// Model accuracy metrics
    pub accuracy_metrics: HashMap<String, f64>,
}

/// Neural model types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NeuralModelType {
    Classifier,
    Embedder,
    Transformer,
    Custom(String),
}

/// Distributed processing node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingNode {
    /// Node ID
    pub id: Uuid,
    /// Node address/endpoint
    pub address: String,
    /// Node capabilities
    pub capabilities: Vec<String>,
    /// Current load (0.0 to 1.0)
    pub load: f64,
    /// Health status
    pub health: NodeHealth,
    /// Last heartbeat
    pub last_heartbeat: DateTime<Utc>,
    /// Processing statistics
    pub statistics: NodeStatistics,
}

/// Node health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Offline,
}

/// Node processing statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeStatistics {
    /// Total queries processed
    pub queries_processed: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Error rate
    pub error_rate: f64,
    /// Current queue depth
    pub queue_depth: usize,
    /// Resource utilization
    pub resource_usage: ResourceUsage,
}

/// Extracted entity with metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExtractedEntity {
    /// Base named entity
    #[serde(flatten)]
    pub entity: NamedEntity,
    /// Entity category
    pub category: EntityCategory,
    /// Extraction metadata
    pub metadata: EntityMetadata,
    /// Relationships to other entities
    pub relationships: Vec<EntityRelationship>,
}

/// Entity categories for classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EntityCategory {
    /// Compliance standard (PCI DSS, HIPAA, etc.)
    Standard,
    /// Specific requirement within a standard
    Requirement,
    /// Security control
    Control,
    /// Technical term or concept
    TechnicalTerm,
    /// Organization or company
    Organization,
    /// Date or time reference
    Date,
    /// Version number
    Version,
    /// Person name
    Person,
    /// Location
    Location,
    /// Process or procedure
    Process,
    /// System or technology
    System,
    /// Risk or threat
    Risk,
    /// Compliance objective
    Objective,
    /// Audit element
    AuditElement,
    /// Unknown category
    Unknown,
}

/// Entity extraction metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMetadata {
    /// Extraction method used
    pub extraction_method: String,
    /// Extraction timestamp
    pub extracted_at: DateTime<Utc>,
    /// Context window around the entity
    pub context: String,
    /// Normalization applied
    pub normalization: Option<String>,
    /// Additional properties
    pub properties: HashMap<String, serde_json::Value>,
}

/// Relationship between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRelationship {
    /// Related entity ID
    pub entity_id: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Relationship strength/confidence
    pub strength: f64,
    /// Context of the relationship
    pub context: Option<String>,
}

/// Types of relationships between entities
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RelationshipType {
    /// Parent-child relationship
    ParentChild,
    /// References or mentions
    References,
    /// Part of or belongs to
    PartOf,
    /// Implements or fulfills
    Implements,
    /// Related or associated with
    RelatedTo,
    /// Conflicts with
    ConflictsWith,
    /// Supersedes or replaces
    Supersedes,
    /// Depends on
    DependsOn,
    /// Similar to
    SimilarTo,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            network_io: 0,
            disk_io: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_intent_properties() {
        assert_eq!(QueryIntent::Factual.confidence_weight(), 0.9);
        assert!(QueryIntent::Analytical.is_complex());
        assert!(!QueryIntent::Factual.is_complex());
    }

    #[test]
    fn test_named_entity_creation() {
        let entity = NamedEntity::new(
            "PCI DSS".to_string(),
            "STANDARD".to_string(),
            0,
            7,
            0.95,
        );

        assert_eq!(entity.text, "PCI DSS");
        assert_eq!(entity.entity_type, "STANDARD");
        assert_eq!(entity.normalized, "pci dss");
        assert!(entity.id != Uuid::nil());
    }

    #[test]
    fn test_consensus_result() {
        let result = ConsensusResult {
            decision: ConsensusDecision::Approve,
            agreement: 0.8,
            participant_count: 5,
            vote_distribution: HashMap::new(),
            consensus_time: Duration::from_millis(500),
            byzantine_tolerance: true,
        };

        assert_eq!(result.decision, ConsensusDecision::Approve);
        assert!(result.byzantine_tolerance);
    }

    #[test]
    fn test_serialization() {
        let intent = QueryIntent::Compliance;
        let json = serde_json::to_string(&intent).unwrap();
        let deserialized: QueryIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(intent, deserialized);
    }
}