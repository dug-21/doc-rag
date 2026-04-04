// Neurosymbolic Response Integration Interface Definitions
// Supporting code for NEUROSYMBOLIC_RESPONSE_INTEGRATION_SPEC.md

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio_stream::Stream;
use uuid::Uuid;

// ============================================================================
// CORE RESPONSE GENERATION TRAITS
// ============================================================================

/// Core response generation trait for neurosymbolic processor integration
#[async_trait]
pub trait NeurosymbolicResponseGenerator: Send + Sync {
    type Request;
    type Response;
    type Error;

    /// Generate complete response from symbolic reasoning results
    async fn generate_response(
        &self,
        request: Self::Request,
    ) -> Result<Self::Response, Self::Error>;

    /// Generate streaming response for large outputs
    async fn generate_streaming(
        &self,
        request: Self::Request,
    ) -> Result<impl Stream<Item = Result<ResponseChunk, Self::Error>>, Self::Error>;

    /// Validate response generation request
    async fn validate_request(&self, request: &Self::Request) -> Result<(), Self::Error>;

    /// Get performance metrics for the last generation
    async fn get_metrics(&self) -> ResponseGenerationMetrics;
}

/// Template engine trait for deterministic response generation
#[async_trait]
pub trait TemplateEngine: Send + Sync {
    /// Render template with extracted variables
    async fn render_template(
        &self,
        template: &ResponseTemplate,
        variables: &HashMap<String, VariableValue>,
        context: &RenderContext,
    ) -> Result<String>;

    /// Select optimal template based on classification
    async fn select_template(
        &self,
        classification: &ClassificationResult,
        complexity: ComplexityLevel,
    ) -> Result<ResponseTemplate>;

    /// Validate template structure and requirements
    fn validate_template(&self, template: &ResponseTemplate) -> Result<()>;

    /// Pre-compile template for performance optimization
    async fn compile_template(&self, template: &ResponseTemplate) -> Result<CompiledTemplate>;
}

/// Symbolic-to-response mapping interface
pub trait SymbolicToResponseMapper {
    /// Extract response variables from symbolic reasoning results
    fn extract_variables(
        &self,
        results: &[QueryResult],
        template: &ResponseTemplate,
    ) -> Result<HashMap<String, VariableValue>>;

    /// Map proof steps to response evidence
    fn map_proof_evidence(
        &self,
        proof_chain: &[ProofStep],
    ) -> Result<Vec<EvidenceItem>>;

    /// Calculate response confidence from multiple sources
    fn calculate_response_confidence(
        &self,
        neural_confidence: f64,
        symbolic_results: &[QueryResult],
        template_match_score: f64,
    ) -> f64;

    /// Extract citations from symbolic results
    fn extract_citations(
        &self,
        results: &[QueryResult],
        proof_chain: &[ProofStep],
    ) -> Result<Vec<EnhancedCitation>>;
}

// ============================================================================
// REQUEST/RESPONSE TYPES
// ============================================================================

/// Request for neurosymbolic response generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeurosymbolicResponseRequest {
    /// Unique request identifier
    pub id: Uuid,
    /// Original query information
    pub query: NeurosymbolicQuery,
    /// Results from symbolic reasoning
    pub symbolic_results: Vec<QueryResult>,
    /// Neural classification output
    pub classification: ClassificationResult,
    /// Generated proof chain
    pub proof_chain: Option<Vec<ProofStep>>,
    /// Response generation configuration
    pub generation_config: ResponseGenerationConfig,
    /// Performance constraints
    pub constraints: PerformanceConstraints,
}

/// Complete neurosymbolic response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeurosymbolicResponse {
    /// Request identifier
    pub request_id: Uuid,
    /// Original query
    pub query: String,
    /// Query classification
    pub classification: String,
    /// Generated response content
    pub content: String,
    /// Response format
    pub format: OutputFormat,
    /// Overall confidence score (0.0-1.0)
    pub confidence_score: f64,
    /// Source citations with enhanced metadata
    pub citations: Vec<EnhancedCitation>,
    /// Proof chain references
    pub proof_references: Vec<ProofReference>,
    /// Response generation metadata
    pub metadata: ResponseMetadata,
    /// Performance metrics
    pub metrics: ResponseGenerationMetrics,
    /// Processing warnings
    pub warnings: Vec<String>,
}

/// Response chunk for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseChunk {
    /// Chunk content
    pub content: String,
    /// Chunk type
    pub chunk_type: ResponseChunkType,
    /// Position in complete response
    pub position: usize,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Confidence score for this chunk
    pub confidence: Option<f64>,
    /// Associated metadata (only in final chunk)
    pub metadata: Option<ResponseGenerationMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseChunkType {
    Header,
    Content,
    Citation,
    ProofStep,
    Metadata,
    Final,
    Error,
}

// ============================================================================
// TEMPLATE SYSTEM
// ============================================================================

/// Response template types based on query classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemplateType {
    /// Requirement-specific queries
    RequirementQuery {
        requirement_type: RequirementType,
        query_intent: QueryIntent,
    },
    /// Compliance verification queries
    ComplianceQuery {
        compliance_type: ComplianceType,
        scope: ComplianceScope,
    },
    /// Entity relationship queries
    RelationshipQuery {
        relationship_type: RelationshipType,
        entity_types: Vec<EntityType>,
    },
    /// Factual and definition queries
    FactualQuery {
        fact_type: FactType,
        complexity_level: ComplexityLevel,
    },
    /// Analysis and comparison queries
    AnalyticalQuery {
        analysis_type: AnalysisType,
        comparison_scope: ComparisonScope,
    },
}

/// Structured response template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTemplate {
    /// Unique template identifier
    pub id: Uuid,
    /// Human-readable template name
    pub name: String,
    /// Template classification type
    pub template_type: TemplateType,
    /// Content structure definition
    pub content_structure: ContentStructure,
    /// Variable placeholder definitions
    pub variable_slots: Vec<VariableSlot>,
    /// Citation requirements for this template
    pub citation_requirements: CitationRequirements,
    /// Validation rules for generated content
    pub validation_rules: Vec<ValidationRule>,
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
}

/// Compiled template for performance optimization
#[derive(Debug, Clone)]
pub struct CompiledTemplate {
    /// Source template reference
    pub template_id: Uuid,
    /// Pre-compiled content sections
    pub compiled_sections: Vec<CompiledSection>,
    /// Variable extraction plan
    pub variable_plan: VariableExtractionPlan,
    /// Cached metadata
    pub cache_metadata: CacheMetadata,
}

/// Variable slot in template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableSlot {
    /// Variable name
    pub name: String,
    /// Variable data type
    pub variable_type: VariableType,
    /// Source of variable data
    pub source: VariableSource,
    /// Validation requirements
    pub validation: VariableValidation,
    /// Optional fallback value
    pub fallback: Option<String>,
    /// Whether variable is required
    pub required: bool,
}

/// Source of variable data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableSource {
    /// Extract from symbolic reasoning result
    SymbolicResult { predicate: String, field: String },
    /// Extract from proof step
    ProofStep { step_number: Option<usize>, field: String },
    /// Extract from neural classification
    Classification { field: String },
    /// Computed from expression
    Computed { expression: String },
    /// Static value
    Static { value: String },
}

/// Variable value with type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableValue {
    String(String),
    Number(f64),
    Boolean(bool),
    List(Vec<VariableValue>),
    Object(HashMap<String, VariableValue>),
}

// ============================================================================
// CITATION AND EVIDENCE SYSTEM
// ============================================================================

/// Enhanced citation with neurosymbolic metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedCitation {
    /// Citation identifier
    pub id: Uuid,
    /// Source document information
    pub source: SourceDocument,
    /// Specific reference (section, paragraph, etc.)
    pub reference: String,
    /// Citation text excerpt
    pub excerpt: String,
    /// Relevance score to query
    pub relevance_score: f64,
    /// Confidence in citation accuracy
    pub confidence_score: f64,
    /// Type of citation
    pub citation_type: CitationType,
    /// Supporting evidence from proof chain
    pub proof_evidence: Vec<ProofEvidence>,
    /// Position in response text
    pub text_positions: Vec<TextRange>,
}

/// Source document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDocument {
    /// Document identifier
    pub id: String,
    /// Document title
    pub title: String,
    /// Document type (standard, regulation, policy, etc.)
    pub document_type: DocumentType,
    /// Publication date
    pub publication_date: Option<chrono::DateTime<chrono::Utc>>,
    /// Document version
    pub version: Option<String>,
    /// Authority or organization
    pub authority: Option<String>,
}

/// Citation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CitationType {
    DirectQuote,
    Paraphrase,
    Reference,
    SupportingEvidence,
    Contradiction,
    Clarification,
}

/// Proof evidence supporting citation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofEvidence {
    /// Proof step reference
    pub step_id: usize,
    /// Rule that was applied
    pub rule_applied: String,
    /// Evidence strength
    pub evidence_strength: f64,
    /// Evidence type
    pub evidence_type: EvidenceType,
}

/// Text range for citation positioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRange {
    /// Start position in text
    pub start: usize,
    /// End position in text
    pub end: usize,
    /// Associated confidence score
    pub confidence: f64,
}

// ============================================================================
// CONFIDENCE AND VALIDATION SYSTEM
// ============================================================================

/// Response metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Overall confidence score
    pub confidence_score: f64,
    /// Per-section confidence scores
    pub section_confidence: Vec<SectionConfidence>,
    /// Source quality assessment
    pub source_quality: Vec<SourceQuality>,
    /// Alternative interpretations
    pub alternatives: Vec<AlternativeInterpretation>,
    /// Identified uncertainty areas
    pub uncertainty_areas: Vec<UncertaintyArea>,
    /// Response completeness score
    pub completeness_score: f64,
    /// Template match quality
    pub template_match_score: f64,
}

/// Per-section confidence scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionConfidence {
    /// Section identifier
    pub section_id: String,
    /// Section type
    pub section_type: SectionType,
    /// Confidence score
    pub confidence: f64,
    /// Contributing factors
    pub factors: Vec<ConfidenceFactor>,
    /// Supporting evidence count
    pub evidence_count: usize,
}

/// Uncertainty area identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyArea {
    /// Location in response text
    pub location: TextRange,
    /// Type of uncertainty
    pub uncertainty_type: UncertaintyType,
    /// Confidence range
    pub confidence_range: (f64, f64),
    /// Alternative interpretations
    pub alternatives: Vec<String>,
    /// Recommended action
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyType {
    InsufficientEvidence,
    ConflictingEvidence,
    AmbiguousInterpretation,
    OutOfScope,
    RequiresHumanReview,
    TemporalRelevance,
}

// ============================================================================
// PERFORMANCE AND OPTIMIZATION
// ============================================================================

/// Performance constraints for response generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConstraints {
    /// Maximum total processing time
    pub max_processing_time: Duration,
    /// Maximum response generation time
    pub max_generation_time: Duration,
    /// Memory usage limit
    pub max_memory_usage: Option<usize>,
    /// Maximum response length
    pub max_response_length: Option<usize>,
    /// Require streaming for large responses
    pub force_streaming_threshold: Option<usize>,
}

impl Default for PerformanceConstraints {
    fn default() -> Self {
        Self {
            max_processing_time: Duration::from_millis(1000), // CONSTRAINT-006
            max_generation_time: Duration::from_millis(500),
            max_memory_usage: Some(100 * 1024 * 1024), // 100MB
            max_response_length: Some(10000), // 10k characters
            force_streaming_threshold: Some(5000), // 5k characters
        }
    }
}

/// Response generation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseGenerationMetrics {
    /// Total generation time
    pub total_duration: Duration,
    /// Template selection time
    pub template_selection_duration: Duration,
    /// Variable extraction time
    pub variable_extraction_duration: Duration,
    /// Citation processing time
    pub citation_processing_duration: Duration,
    /// Content rendering time
    pub content_rendering_duration: Duration,
    /// Validation time
    pub validation_duration: Duration,
    /// Number of templates evaluated
    pub templates_evaluated: usize,
    /// Number of variables extracted
    pub variables_extracted: usize,
    /// Number of citations processed
    pub citations_processed: usize,
    /// Memory usage peak
    pub peak_memory_usage: usize,
    /// Response length in characters
    pub response_length: usize,
}

/// Template performance profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// Expected generation time range
    pub expected_generation_time: (Duration, Duration),
    /// Memory usage estimate
    pub memory_usage_estimate: usize,
    /// Complexity score
    pub complexity_score: f64,
    /// Cache efficiency rating
    pub cache_efficiency: f64,
}

// ============================================================================
// SUPPORTING ENUMS AND TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequirementType {
    Must,
    Shall,
    Should,
    May,
    Recommended,
    Optional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryIntent {
    Lookup,
    Verification,
    Comparison,
    Analysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceType {
    PCI,
    ISO27001,
    SOC2,
    GDPR,
    HIPAA,
    NIST,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceScope {
    Full,
    Partial(Vec<String>),
    Section(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    Dependency,
    Inheritance,
    Association,
    Composition,
    Implementation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Requirement,
    Control,
    Process,
    System,
    Data,
    Role,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactType {
    Definition,
    Specification,
    Procedure,
    Example,
    Exception,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    HighlyComplex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisType {
    Comparison,
    Gap,
    Risk,
    Impact,
    Trend,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonScope {
    TwoEntity,
    MultiEntity,
    Temporal,
    CrossDomain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Markdown,
    Html,
    PlainText,
    Structured,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableType {
    Text,
    Number,
    Boolean,
    List,
    Object,
    Citation,
    Date,
    Url,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentType {
    Standard,
    Regulation,
    Policy,
    Procedure,
    Guideline,
    Specification,
    Reference,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceType {
    Direct,
    Indirect,
    Circumstantial,
    Expert,
    Statistical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SectionType {
    Summary,
    Analysis,
    Requirements,
    Procedures,
    Examples,
    Citations,
    Conclusions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfidenceFactor {
    SourceQuality(f64),
    EvidenceStrength(f64),
    ConsistencyScore(f64),
    TemplateMatch(f64),
    ValidationPassed(f64),
}

// ============================================================================
// IMPLEMENTATION STUBS FOR COMPILATION
// ============================================================================

// These would be implemented in the actual integration

#[derive(Debug, Clone)]
pub struct ContentStructure {
    pub sections: Vec<ContentSection>,
}

#[derive(Debug, Clone)]
pub struct ContentSection {
    pub section_type: SectionType,
    pub template: String,
    pub variables: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CitationRequirements {
    pub minimum_citations: usize,
    pub required_types: Vec<CitationType>,
    pub quality_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct ValidationRule {
    pub rule_type: String,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct VariableValidation {
    pub validation_type: String,
    pub constraints: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct CompiledSection {
    pub template: String,
    pub variables: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct VariableExtractionPlan {
    pub steps: Vec<ExtractionStep>,
}

#[derive(Debug, Clone)]
pub struct ExtractionStep {
    pub variable: String,
    pub source: VariableSource,
}

#[derive(Debug, Clone)]
pub struct CacheMetadata {
    pub cache_key: String,
    pub ttl: Duration,
}

#[derive(Debug, Clone)]
pub struct RenderContext {
    pub query: String,
    pub classification: String,
}

#[derive(Debug, Clone)]
pub struct EvidenceItem {
    pub evidence_type: EvidenceType,
    pub content: String,
    pub strength: f64,
}

#[derive(Debug, Clone)]
pub struct SourceQuality {
    pub source_id: String,
    pub quality_score: f64,
    pub factors: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AlternativeInterpretation {
    pub interpretation: String,
    pub confidence: f64,
    pub supporting_evidence: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ProofReference {
    pub step_number: usize,
    pub rule_applied: String,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ResponseGenerationConfig {
    pub format: OutputFormat,
    pub include_citations: bool,
    pub include_proof_chain: bool,
    pub confidence_threshold: f64,
    pub max_length: Option<usize>,
}

impl Default for ResponseGenerationConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            include_citations: true,
            include_proof_chain: true,
            confidence_threshold: 0.8,
            max_length: None,
        }
    }
}

// Placeholder types that would reference actual neurosymbolic types
pub type QueryResult = crate::symbolic::types::QueryResult;
pub type ProofStep = crate::symbolic::types::ProofStep;
pub type ClassificationResult = crate::neural::ClassificationResult;
pub type NeurosymbolicQuery = crate::symbolic::neurosymbolic::NeurosymbolicQuery;

// Result type alias
pub type Result<T> = std::result::Result<T, ResponseGenerationError>;

// Error type
#[derive(Debug, thiserror::Error)]
pub enum ResponseGenerationError {
    #[error("Template not found: {0}")]
    TemplateNotFound(String),
    #[error("Variable extraction failed: {0}")]
    VariableExtractionFailed(String),
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    #[error("Performance constraint violated: {0}")]
    PerformanceConstraintViolated(String),
}