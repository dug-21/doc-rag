//! Proof Chain Integration for Template Variable Substitution
//! 
//! Connects symbolic reasoning proof chains with template engine variable substitution
//! for deterministic response generation with complete audit trails.

use crate::{Result, ResponseError};
use crate::template_engine::{
    VariableType, ProofChainData, ProofElementType, SubstitutionSource, 
    VariableSubstitution, CalculationType
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tracing::{debug, info, instrument, warn};

/// Proof chain integration manager
#[derive(Debug, Clone)]
pub struct ProofChainIntegrationManager {
    /// Integration configuration
    config: ProofChainIntegrationConfig,
    /// Symbolic reasoning client
    symbolic_client: SymbolicReasoningClient,
    /// Proof chain processor
    proof_processor: ProofChainProcessor,
    /// Variable resolver
    variable_resolver: VariableResolver,
    /// Confidence calculator
    confidence_calculator: ConfidenceCalculator,
}

/// Configuration for proof chain integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofChainIntegrationConfig {
    /// Minimum confidence threshold for proof elements
    pub min_confidence_threshold: f64,
    /// Maximum proof chain depth to traverse
    pub max_proof_depth: usize,
    /// Enable proof chain validation
    pub enable_proof_validation: bool,
    /// Proof element cache TTL in seconds
    pub proof_cache_ttl: u64,
    /// Maximum variables per proof chain
    pub max_variables_per_chain: usize,
    /// Enable confidence propagation
    pub enable_confidence_propagation: bool,
}

impl Default for ProofChainIntegrationConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.7,
            max_proof_depth: 10,
            enable_proof_validation: true,
            proof_cache_ttl: 3600,
            max_variables_per_chain: 50,
            enable_confidence_propagation: true,
        }
    }
}

/// Symbolic reasoning client interface
#[derive(Debug, Clone)]
pub struct SymbolicReasoningClient {
    /// Client configuration
    config: SymbolicClientConfig,
    /// Connection pool
    connection_pool: ConnectionPool,
}

/// Proof chain processor for variable extraction
#[derive(Debug, Clone)]
pub struct ProofChainProcessor {
    /// Processing configuration
    config: ProcessorConfig,
    /// Element extractors
    extractors: HashMap<ProofElementType, Box<dyn ElementExtractor>>,
    /// Validation engine
    validator: ProofValidationEngine,
}

/// Variable resolver for proof chain elements
#[derive(Debug, Clone)]
pub struct VariableResolver {
    /// Resolution rules
    resolution_rules: Vec<ResolutionRule>,
    /// Type mappers
    type_mappers: HashMap<ProofElementType, VariableTypeMapper>,
    /// Format converters
    format_converters: HashMap<String, FormatConverter>,
}

/// Confidence calculator for proof chains
#[derive(Debug, Clone)]
pub struct ConfidenceCalculator {
    /// Calculation methods
    methods: HashMap<CalculationType, ConfidenceMethod>,
    /// Propagation rules
    propagation_rules: Vec<PropagationRule>,
    /// Decay factors
    decay_factors: HashMap<ProofElementType, f64>,
}

/// Proof chain query request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofChainQuery {
    /// Query identifier
    pub id: Uuid,
    /// Query text
    pub query: String,
    /// Required proof elements
    pub required_elements: Vec<ProofElementType>,
    /// Confidence threshold
    pub confidence_threshold: f64,
    /// Maximum depth
    pub max_depth: usize,
    /// Variable requirements
    pub variable_requirements: Vec<VariableRequirement>,
}

/// Variable requirement from proof chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableRequirement {
    /// Variable name in template
    pub variable_name: String,
    /// Required proof element type
    pub required_element_type: ProofElementType,
    /// Minimum confidence required
    pub min_confidence: f64,
    /// Extraction rules
    pub extraction_rules: Vec<ExtractionRule>,
    /// Validation rules
    pub validation_rules: Vec<ValidationRule>,
}

/// Proof chain response with variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofChainResponse {
    /// Response identifier
    pub id: Uuid,
    /// Query that generated this response
    pub query: ProofChainQuery,
    /// Proof chain elements
    pub proof_elements: Vec<ProofElement>,
    /// Extracted variables
    pub extracted_variables: Vec<ExtractedVariable>,
    /// Overall confidence
    pub overall_confidence: f64,
    /// Proof validation result
    pub validation_result: ProofValidationResult,
    /// Processing metrics
    pub metrics: ProofProcessingMetrics,
    /// Response timestamp
    pub created_at: DateTime<Utc>,
}

/// Proof element with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofElement {
    /// Element identifier
    pub id: Uuid,
    /// Element type
    pub element_type: ProofElementType,
    /// Element content
    pub content: String,
    /// Element confidence
    pub confidence: f64,
    /// Source reference
    pub source_reference: SourceReference,
    /// Parent elements (premises)
    pub parent_elements: Vec<Uuid>,
    /// Child elements (conclusions)
    pub child_elements: Vec<Uuid>,
    /// Rule applied
    pub rule_applied: Option<String>,
    /// Conditions satisfied
    pub conditions: Vec<String>,
    /// Element metadata
    pub metadata: HashMap<String, String>,
    /// Extraction timestamp
    pub extracted_at: DateTime<Utc>,
}

/// Extracted variable from proof chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedVariable {
    /// Variable name
    pub name: String,
    /// Variable value
    pub value: String,
    /// Variable type
    pub variable_type: VariableType,
    /// Source proof element
    pub source_element: Uuid,
    /// Extraction confidence
    pub confidence: f64,
    /// Extraction method
    pub extraction_method: ExtractionMethod,
    /// Validation status
    pub validation_status: VariableValidationStatus,
    /// Supporting elements
    pub supporting_elements: Vec<Uuid>,
    /// Extracted timestamp
    pub extracted_at: DateTime<Utc>,
}

/// Source reference for proof elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReference {
    /// Source document ID
    pub document_id: String,
    /// Section reference
    pub section: Option<String>,
    /// Page number
    pub page: Option<u32>,
    /// Paragraph reference
    pub paragraph: Option<String>,
    /// Character range
    pub char_range: Option<(usize, usize)>,
    /// Source confidence
    pub source_confidence: f64,
    /// Source type
    pub source_type: SourceType,
}

/// Source types for proof elements
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SourceType {
    /// Regulatory document
    Regulatory,
    /// Standard specification
    Standard,
    /// Policy document
    Policy,
    /// Technical specification
    Technical,
    /// Knowledge base entry
    KnowledgeBase,
    /// Derived inference
    Inference,
}

/// Extraction methods for variables
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtractionMethod {
    /// Direct text extraction
    DirectExtraction,
    /// Pattern-based extraction
    PatternBased,
    /// Rule-based extraction
    RuleBased,
    /// Semantic extraction
    Semantic,
    /// Calculated from multiple sources
    Calculated,
    /// Template-driven extraction
    TemplateDriven,
}

/// Variable validation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariableValidationStatus {
    /// Valid and ready for substitution
    Valid,
    /// Valid with warnings
    ValidWithWarnings(Vec<String>),
    /// Invalid - cannot be used
    Invalid(Vec<String>),
    /// Requires manual review
    RequiresReview(String),
}

/// Proof validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofValidationResult {
    /// Overall validation success
    pub is_valid: bool,
    /// Validation score
    pub validation_score: f64,
    /// Chain completeness
    pub chain_complete: bool,
    /// Circular dependency check
    pub has_circular_dependencies: bool,
    /// Missing premises
    pub missing_premises: Vec<String>,
    /// Validation errors
    pub validation_errors: Vec<ProofValidationError>,
    /// Validation warnings
    pub validation_warnings: Vec<ProofValidationWarning>,
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
}

/// Proof processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofProcessingMetrics {
    /// Total processing time
    pub total_processing_time: std::time::Duration,
    /// Symbolic query time
    pub symbolic_query_time: std::time::Duration,
    /// Variable extraction time
    pub variable_extraction_time: std::time::Duration,
    /// Validation time
    pub validation_time: std::time::Duration,
    /// Elements processed
    pub elements_processed: usize,
    /// Variables extracted
    pub variables_extracted: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Cache misses
    pub cache_misses: usize,
}

// Supporting types and trait definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicClientConfig;

#[derive(Debug, Clone)]
pub struct ConnectionPool;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig;

/// Helper trait for cloning ElementExtractor trait objects
pub trait CloneElementExtractor {
    fn clone_box(&self) -> Box<dyn ElementExtractor>;
}

impl<T> CloneElementExtractor for T
where
    T: 'static + ElementExtractor + Clone,
{
    fn clone_box(&self) -> Box<dyn ElementExtractor> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ElementExtractor> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

pub trait ElementExtractor: Send + Sync + std::fmt::Debug + CloneElementExtractor {
    fn extract(&self, element: &ProofElement) -> Result<Vec<ExtractedVariable>>;
}

#[derive(Debug, Clone)]
pub struct ProofValidationEngine;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionRule;

#[derive(Debug, Clone)]
pub struct VariableTypeMapper;

#[derive(Debug, Clone)]
pub struct FormatConverter;

#[derive(Debug, Clone)]
pub struct ConfidenceMethod;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropagationRule;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionRule;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofValidationError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofValidationWarning;

impl ProofChainIntegrationManager {
    /// Create new proof chain integration manager
    pub fn new(config: ProofChainIntegrationConfig) -> Self {
        let symbolic_client = SymbolicReasoningClient::new(SymbolicClientConfig);
        let proof_processor = ProofChainProcessor::new(ProcessorConfig);
        let variable_resolver = VariableResolver::new();
        let confidence_calculator = ConfidenceCalculator::new();
        
        Self {
            config,
            symbolic_client,
            proof_processor,
            variable_resolver,
            confidence_calculator,
        }
    }
    
    /// Extract variables from proof chain for template substitution
    #[instrument(skip(self, query))]
    pub async fn extract_variables_from_proof_chain(
        &self,
        query: ProofChainQuery,
    ) -> Result<Vec<VariableSubstitution>> {
        let start_time = std::time::Instant::now();
        
        info!("Extracting variables from proof chain for query: {}", query.id);
        
        // Stage 1: Query symbolic reasoning system
        let symbolic_start = std::time::Instant::now();
        let proof_response = self.symbolic_client.query_proof_chain(&query).await?;
        let symbolic_query_time = symbolic_start.elapsed();
        
        // Stage 2: Process proof elements
        let processing_start = std::time::Instant::now();
        let processed_elements = self.proof_processor
            .process_elements(&proof_response.proof_elements)
            .await?;
        let processing_time = processing_start.elapsed();
        
        // Stage 3: Extract variables from elements
        let extraction_start = std::time::Instant::now();
        let extracted_variables = self.extract_variables_from_elements(&processed_elements, &query).await?;
        let extraction_time = extraction_start.elapsed();
        
        // Stage 4: Resolve variables to template format
        let resolution_start = std::time::Instant::now();
        let resolved_variables = self.variable_resolver
            .resolve_variables(&extracted_variables, &query.variable_requirements)
            .await?;
        let resolution_time = resolution_start.elapsed();
        
        // Stage 5: Calculate confidence scores
        let confidence_start = std::time::Instant::now();
        let confidence_scores = self.confidence_calculator
            .calculate_variable_confidence(&resolved_variables, &processed_elements)
            .await?;
        let confidence_time = confidence_start.elapsed();
        
        // Stage 6: Convert to variable substitutions
        let substitutions = self.create_variable_substitutions(
            &resolved_variables,
            &confidence_scores,
            &processed_elements,
        ).await?;
        
        let total_time = start_time.elapsed();
        
        // Validate confidence thresholds
        let original_count = substitutions.len();
        let valid_substitutions: Vec<_> = substitutions.into_iter()
            .filter(|sub| sub.confidence >= self.config.min_confidence_threshold)
            .collect();
        
        if valid_substitutions.len() < original_count {
            warn!(
                "Filtered {} substitutions below confidence threshold {}",
                original_count - valid_substitutions.len(),
                self.config.min_confidence_threshold
            );
        }
        
        info!(
            "Extracted {} variables from proof chain in {}ms",
            valid_substitutions.len(),
            total_time.as_millis()
        );
        
        Ok(valid_substitutions)
    }
    
    /// Query proof chain for specific variable requirements
    pub async fn query_for_variables(
        &self,
        query_text: &str,
        variable_requirements: Vec<VariableRequirement>,
    ) -> Result<ProofChainResponse> {
        let query = ProofChainQuery {
            id: Uuid::new_v4(),
            query: query_text.to_string(),
            required_elements: variable_requirements.iter()
                .map(|req| req.required_element_type.clone())
                .collect(),
            confidence_threshold: self.config.min_confidence_threshold,
            max_depth: self.config.max_proof_depth,
            variable_requirements,
        };
        
        self.symbolic_client.query_proof_chain(&query).await
    }
    
    /// Validate proof chain completeness for variables
    pub async fn validate_proof_chain_completeness(
        &self,
        proof_elements: &[ProofElement],
        required_variables: &[VariableRequirement],
    ) -> Result<ProofValidationResult> {
        // Check if all required variables can be satisfied
        let mut missing_premises = Vec::new();
        let mut validation_errors = Vec::new();
        let mut validation_warnings = Vec::new();
        
        for requirement in required_variables {
            let satisfied = proof_elements.iter().any(|element| {
                element.element_type == requirement.required_element_type &&
                element.confidence >= requirement.min_confidence
            });
            
            if !satisfied {
                missing_premises.push(format!(
                    "Missing {} element for variable {}",
                    format!("{:?}", requirement.required_element_type),
                    requirement.variable_name
                ));
            }
        }
        
        // Check for circular dependencies
        let has_circular_dependencies = self.detect_circular_dependencies(proof_elements);
        
        let is_valid = missing_premises.is_empty() && !has_circular_dependencies;
        let validation_score = if is_valid { 1.0 } else { 0.5 };
        
        Ok(ProofValidationResult {
            is_valid,
            validation_score,
            chain_complete: missing_premises.is_empty(),
            has_circular_dependencies,
            missing_premises,
            validation_errors,
            validation_warnings,
            validated_at: Utc::now(),
        })
    }
    
    /// Extract variables from processed proof elements
    async fn extract_variables_from_elements(
        &self,
        elements: &[ProofElement],
        query: &ProofChainQuery,
    ) -> Result<Vec<ExtractedVariable>> {
        let mut extracted_variables = Vec::new();
        
        for element in elements {
            // Extract variables based on element type and requirements
            for requirement in &query.variable_requirements {
                if element.element_type == requirement.required_element_type {
                    let variable = self.extract_variable_from_element(element, requirement).await?;
                    if let Some(var) = variable {
                        extracted_variables.push(var);
                    }
                }
            }
        }
        
        Ok(extracted_variables)
    }
    
    /// Extract a specific variable from a proof element
    async fn extract_variable_from_element(
        &self,
        element: &ProofElement,
        requirement: &VariableRequirement,
    ) -> Result<Option<ExtractedVariable>> {
        // Apply extraction rules to extract variable value
        let value = self.apply_extraction_rules(&element.content, &requirement.extraction_rules)?;
        
        if let Some(extracted_value) = value {
            let variable = ExtractedVariable {
                name: requirement.variable_name.clone(),
                value: extracted_value,
                variable_type: VariableType::ProofChainElement {
                    element_type: element.element_type.clone(),
                    confidence_threshold: requirement.min_confidence,
                },
                source_element: element.id,
                confidence: element.confidence,
                extraction_method: ExtractionMethod::RuleBased,
                validation_status: VariableValidationStatus::Valid,
                supporting_elements: vec![element.id],
                extracted_at: Utc::now(),
            };
            
            Ok(Some(variable))
        } else {
            Ok(None)
        }
    }
    
    /// Apply extraction rules to content
    fn apply_extraction_rules(
        &self,
        content: &str,
        _rules: &[ExtractionRule],
    ) -> Result<Option<String>> {
        // Simplified extraction - in practice would apply complex rules
        Ok(Some(content.to_string()))
    }
    
    /// Create variable substitutions from resolved variables
    async fn create_variable_substitutions(
        &self,
        variables: &[ExtractedVariable],
        confidence_scores: &HashMap<String, f64>,
        elements: &[ProofElement],
    ) -> Result<Vec<VariableSubstitution>> {
        let mut substitutions = Vec::new();
        
        for variable in variables {
            let confidence = confidence_scores.get(&variable.name).copied()
                .unwrap_or(variable.confidence);
            
            let source_element = elements.iter()
                .find(|e| e.id == variable.source_element);
            
            let source = if let Some(element) = source_element {
                SubstitutionSource::ProofChain {
                    proof_step_id: element.id.to_string(),
                    element_type: element.element_type.clone(),
                }
            } else {
                SubstitutionSource::ManualOverride {
                    override_reason: "Element not found".to_string(),
                }
            };
            
            substitutions.push(VariableSubstitution {
                variable_name: variable.name.clone(),
                placeholder: format!("{{{}}}", variable.name),
                substituted_value: variable.value.clone(),
                source,
                confidence,
                substituted_at: Utc::now(),
            });
        }
        
        Ok(substitutions)
    }
    
    /// Detect circular dependencies in proof chain
    fn detect_circular_dependencies(&self, elements: &[ProofElement]) -> bool {
        // Simplified cycle detection - would implement proper graph traversal
        false
    }
}

// Implementation of support components

impl SymbolicReasoningClient {
    fn new(_config: SymbolicClientConfig) -> Self {
        Self {
            config: SymbolicClientConfig,
            connection_pool: ConnectionPool,
        }
    }
    
    async fn query_proof_chain(&self, query: &ProofChainQuery) -> Result<ProofChainResponse> {
        // Mock implementation - would connect to actual symbolic reasoning system
        let proof_elements = vec![
            ProofElement {
                id: Uuid::new_v4(),
                element_type: ProofElementType::Premise,
                content: "PCI DSS requires encryption of stored payment card data".to_string(),
                confidence: 0.95,
                source_reference: SourceReference {
                    document_id: "pci_dss_v4.0".to_string(),
                    section: Some("3.4".to_string()),
                    page: Some(45),
                    paragraph: Some("3.4.1".to_string()),
                    char_range: Some((1200, 1350)),
                    source_confidence: 0.98,
                    source_type: SourceType::Standard,
                },
                parent_elements: vec![],
                child_elements: vec![],
                rule_applied: Some("requirement_extraction".to_string()),
                conditions: vec!["payment_card_data_present".to_string()],
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("requirement_type".to_string(), "MUST".to_string());
                    meta.insert("compliance_level".to_string(), "1".to_string());
                    meta
                },
                extracted_at: Utc::now(),
            }
        ];
        
        Ok(ProofChainResponse {
            id: Uuid::new_v4(),
            query: query.clone(),
            proof_elements,
            extracted_variables: vec![],
            overall_confidence: 0.9,
            validation_result: ProofValidationResult {
                is_valid: true,
                validation_score: 0.95,
                chain_complete: true,
                has_circular_dependencies: false,
                missing_premises: vec![],
                validation_errors: vec![],
                validation_warnings: vec![],
                validated_at: Utc::now(),
            },
            metrics: ProofProcessingMetrics {
                total_processing_time: std::time::Duration::from_millis(150),
                symbolic_query_time: std::time::Duration::from_millis(50),
                variable_extraction_time: std::time::Duration::from_millis(30),
                validation_time: std::time::Duration::from_millis(20),
                elements_processed: 1,
                variables_extracted: 1,
                cache_hits: 0,
                cache_misses: 1,
            },
            created_at: Utc::now(),
        })
    }
}

impl ProofChainProcessor {
    fn new(_config: ProcessorConfig) -> Self {
        Self {
            config: ProcessorConfig,
            extractors: HashMap::new(),
            validator: ProofValidationEngine,
        }
    }
    
    async fn process_elements(&self, elements: &[ProofElement]) -> Result<Vec<ProofElement>> {
        // Process and validate elements
        Ok(elements.to_vec())
    }
}

impl VariableResolver {
    fn new() -> Self {
        Self {
            resolution_rules: vec![],
            type_mappers: HashMap::new(),
            format_converters: HashMap::new(),
        }
    }
    
    async fn resolve_variables(
        &self,
        variables: &[ExtractedVariable],
        _requirements: &[VariableRequirement],
    ) -> Result<Vec<ExtractedVariable>> {
        // Resolve variables to template format
        Ok(variables.to_vec())
    }
}

impl ConfidenceCalculator {
    fn new() -> Self {
        Self {
            methods: HashMap::new(),
            propagation_rules: vec![],
            decay_factors: HashMap::new(),
        }
    }
    
    async fn calculate_variable_confidence(
        &self,
        variables: &[ExtractedVariable],
        _elements: &[ProofElement],
    ) -> Result<HashMap<String, f64>> {
        let mut confidence_scores = HashMap::new();
        
        for variable in variables {
            confidence_scores.insert(variable.name.clone(), variable.confidence);
        }
        
        Ok(confidence_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proof_chain_integration() {
        let config = ProofChainIntegrationConfig::default();
        let manager = ProofChainIntegrationManager::new(config);
        
        let variable_requirements = vec![
            VariableRequirement {
                variable_name: "REQUIREMENT_TEXT".to_string(),
                required_element_type: ProofElementType::Premise,
                min_confidence: 0.7,
                extraction_rules: vec![],
                validation_rules: vec![],
            }
        ];
        
        let query = ProofChainQuery {
            id: Uuid::new_v4(),
            query: "What are the encryption requirements?".to_string(),
            required_elements: vec![ProofElementType::Premise],
            confidence_threshold: 0.7,
            max_depth: 5,
            variable_requirements,
        };
        
        let result = manager.extract_variables_from_proof_chain(query).await;
        assert!(result.is_ok());
        
        let substitutions = result.unwrap();
        assert!(!substitutions.is_empty());
        assert!(substitutions[0].confidence >= 0.7);
    }

    #[tokio::test]
    async fn test_variable_extraction() {
        let config = ProofChainIntegrationConfig::default();
        let manager = ProofChainIntegrationManager::new(config);
        
        let proof_elements = vec![
            ProofElement {
                id: Uuid::new_v4(),
                element_type: ProofElementType::Premise,
                content: "Test requirement content".to_string(),
                confidence: 0.9,
                source_reference: SourceReference {
                    document_id: "test_doc".to_string(),
                    section: None,
                    page: None,
                    paragraph: None,
                    char_range: None,
                    source_confidence: 0.9,
                    source_type: SourceType::Standard,
                },
                parent_elements: vec![],
                child_elements: vec![],
                rule_applied: None,
                conditions: vec![],
                metadata: HashMap::new(),
                extracted_at: Utc::now(),
            }
        ];
        
        let variable_requirements = vec![
            VariableRequirement {
                variable_name: "TEST_VAR".to_string(),
                required_element_type: ProofElementType::Premise,
                min_confidence: 0.8,
                extraction_rules: vec![],
                validation_rules: vec![],
            }
        ];
        
        let result = manager.validate_proof_chain_completeness(&proof_elements, &variable_requirements).await;
        assert!(result.is_ok());
        
        let validation = result.unwrap();
        assert!(validation.is_valid);
        assert!(validation.chain_complete);
    }
}