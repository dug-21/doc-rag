//! Validation engine for comprehensive query processing validation
//!
//! This module provides multi-layer validation including syntax validation,
//! semantic validation, and factual validation with quality assurance.

use async_trait::async_trait;
use chrono::Utc;
// use std::collections::HashMap; // Unused
use std::sync::Arc;
use std::time::Instant; // removed unused Duration
use tracing::{info, instrument}; // removed unused warn
// use uuid::Uuid; // Unused

use crate::config::ValidationConfig;
use crate::error::{ProcessorError, Result};
use crate::query::{ProcessedQuery, ValidationResult, ValidationStatus};
use crate::types::*;

/// Multi-layer validation engine
pub struct ValidationEngine {
    config: Arc<ValidationConfig>,
    syntax_validator: SyntaxValidator,
    semantic_validator: SemanticValidator,
    factual_validator: FactualValidator,
    performance_validator: PerformanceValidator,
    consistency_checker: ConsistencyChecker,
}

impl ValidationEngine {
    /// Create a new validation engine
    #[instrument(skip(config))]
    pub async fn new(config: Arc<ValidationConfig>) -> Result<Self> {
        info!("Initializing Validation Engine");
        
        let syntax_validator = SyntaxValidator::new(config.clone()).await?;
        let semantic_validator = SemanticValidator::new(config.clone()).await?;
        let factual_validator = FactualValidator::new(config.clone()).await?;
        let performance_validator = PerformanceValidator::new(config.clone()).await?;
        let consistency_checker = ConsistencyChecker::new(config.clone()).await?;
        
        Ok(Self {
            config,
            syntax_validator,
            semantic_validator,
            factual_validator,
            performance_validator,
            consistency_checker,
        })
    }

    /// Validate processed query through all validation layers
    #[instrument(skip(self, processed_query))]
    pub async fn validate(&self, mut processed_query: ProcessedQuery) -> Result<ProcessedQuery> {
        info!("Starting comprehensive validation for query: {}", processed_query.id());
        
        let start_time = Instant::now();
        let mut validation_results = Vec::new();
        let mut warnings = Vec::new();
        
        // Input validation
        if self.config.enable_input_validation {
            match self.validate_input(&processed_query).await {
                Ok(result) => validation_results.push(result),
                Err(e) => {
                    warnings.push(format!("Input validation warning: {}", e));
                }
            }
        }
        
        // Syntax validation
        let syntax_result = self.syntax_validator.validate(&processed_query).await?;
        validation_results.push(syntax_result);
        
        // Semantic validation
        let semantic_result = self.semantic_validator.validate(&processed_query).await?;
        validation_results.push(semantic_result);
        
        // Factual validation
        let factual_result = self.factual_validator.validate(&processed_query).await?;
        validation_results.push(factual_result);
        
        // Performance validation
        if self.config.enable_performance_validation {
            let performance_result = self.performance_validator.validate(&processed_query).await?;
            validation_results.push(performance_result);
        }
        
        // Consistency checking
        if self.config.enable_consistency_checking {
            let consistency_result = self.consistency_checker.validate(&processed_query).await?;
            validation_results.push(consistency_result);
        }
        
        // Output validation
        if self.config.enable_output_validation {
            let output_result = self.validate_output(&processed_query).await?;
            validation_results.push(output_result);
        }
        
        // Add validation results to processed query
        for result in validation_results {
            processed_query.add_validation_result(result);
        }
        
        // Add warnings
        for warning in warnings {
            processed_query.add_warning(warning);
        }
        
        let duration = start_time.elapsed();
        processed_query.add_stage_duration("validation".to_string(), duration);
        
        // Check overall validation status
        let overall_status = self.determine_overall_status(&processed_query);
        
        if overall_status == ValidationStatus::Failed {
            return Err(ProcessorError::ValidationFailed {
                field: "validation".to_string(),
                reason: "Overall validation failed".to_string(),
            });
        }
        
        info!(
            "Validation completed in {:?}ms with status: {:?}",
            duration.as_millis(),
            overall_status
        );
        
        Ok(processed_query)
    }

    /// Validate input requirements
    async fn validate_input(&self, query: &ProcessedQuery) -> Result<ValidationResult> {
        let mut issues = Vec::new();
        
        // Check required fields
        for required_field in &self.config.rules.required_fields {
            match required_field.as_str() {
                "intent" => {
                    if query.intent.confidence < self.config.rules.min_confidence {
                        issues.push(format!(
                            "Intent confidence {} below minimum {}",
                            query.intent.confidence,
                            self.config.rules.min_confidence
                        ));
                    }
                }
                "entities" => {
                    if query.entities.is_empty() {
                        issues.push("No entities extracted".to_string());
                    }
                }
                "strategy" => {
                    if query.strategy.confidence < self.config.rules.min_confidence {
                        issues.push(format!(
                            "Strategy confidence {} below minimum {}",
                            query.strategy.confidence,
                            self.config.rules.min_confidence
                        ));
                    }
                }
                _ => {}
            }
        }
        
        let status = if issues.is_empty() {
            ValidationStatus::Passed
        } else if issues.len() <= 2 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Failed
        };
        
        let message = if issues.is_empty() {
            "Input validation passed".to_string()
        } else {
            format!("Input validation issues: {}", issues.join("; "))
        };
        
        Ok(ValidationResult {
            stage: "input".to_string(),
            status,
            message,
            score: if issues.is_empty() { 1.0 } else { 0.5 },
            validated_at: Utc::now(),
        })
    }

    /// Validate output completeness and quality
    async fn validate_output(&self, query: &ProcessedQuery) -> Result<ValidationResult> {
        let mut score = 1.0;
        let mut issues = Vec::new();
        
        // Check overall confidence
        let overall_confidence = query.overall_confidence();
        if overall_confidence < self.config.rules.min_confidence {
            score -= 0.3;
            issues.push(format!(
                "Overall confidence {} below threshold {}",
                overall_confidence,
                self.config.rules.min_confidence
            ));
        }
        
        // Check processing completeness
        if query.entities.is_empty() && query.key_terms.is_empty() {
            score -= 0.2;
            issues.push("No entities or key terms extracted".to_string());
        }
        
        // Check for warnings
        if query.processing_metadata.warnings.len() > 3 {
            score -= 0.1;
            issues.push(format!(
                "High number of warnings: {}",
                query.processing_metadata.warnings.len()
            ));
        }
        
        let status = if score >= 0.8 {
            ValidationStatus::Passed
        } else if score >= 0.5 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Failed
        };
        
        let message = if issues.is_empty() {
            "Output validation passed".to_string()
        } else {
            format!("Output validation issues: {}", issues.join("; "))
        };
        
        Ok(ValidationResult {
            stage: "output".to_string(),
            status,
            message,
            score: f64::min(1.0, f64::max(0.0, score)),
            validated_at: Utc::now(),
        })
    }

    /// Determine overall validation status
    fn determine_overall_status(&self, query: &ProcessedQuery) -> ValidationStatus {
        let results = &query.processing_metadata.validation_results;
        
        if results.is_empty() {
            return ValidationStatus::Skipped;
        }
        
        let failed_count = results.iter().filter(|r| r.status == ValidationStatus::Failed).count();
        let warning_count = results.iter().filter(|r| r.status == ValidationStatus::PassedWithWarnings).count();
        
        if failed_count > 0 {
            ValidationStatus::Failed
        } else if warning_count > 0 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Passed
        }
    }
}

/// Syntax validation layer
pub struct SyntaxValidator {
    _config: Arc<ValidationConfig>,
}

impl SyntaxValidator {
    pub async fn new(config: Arc<ValidationConfig>) -> Result<Self> {
        Ok(Self { _config: config })
    }
}

#[async_trait]
impl Validator for SyntaxValidator {
    async fn validate(&self, query: &ProcessedQuery) -> Result<ValidationResult> {
        let mut score = 1.0;
        let mut issues = Vec::new();
        
        // Check query text syntax
        let text = query.text();
        
        // Check for empty or very short queries
        if text.trim().is_empty() {
            score = 0.0;
            issues.push("Empty query text".to_string());
        } else if text.len() < 3 {
            score -= 0.5;
            issues.push("Query too short".to_string());
        }
        
        // Check for suspicious patterns
        if text.contains("<script>") || text.contains("javascript:") {
            score = 0.0;
            issues.push("Suspicious content detected".to_string());
        }
        
        // Check query structure
        if query.analysis.syntactic_features.pos_tags.is_empty() {
            score -= 0.2;
            issues.push("No POS tags found".to_string());
        }
        
        // Check Unicode handling
        let char_count = text.chars().count();
        let byte_count = text.len();
        if byte_count > char_count * 4 {
            score -= 0.1;
            issues.push("Potential Unicode encoding issues".to_string());
        }
        
        let status = if score >= 0.8 {
            ValidationStatus::Passed
        } else if score >= 0.5 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Failed
        };
        
        let message = if issues.is_empty() {
            "Syntax validation passed".to_string()
        } else {
            format!("Syntax issues: {}", issues.join("; "))
        };
        
        Ok(ValidationResult {
            stage: "syntax".to_string(),
            status,
            message,
            score,
            validated_at: Utc::now(),
        })
    }
}

/// Semantic validation layer
pub struct SemanticValidator {
    _config: Arc<ValidationConfig>,
}

impl SemanticValidator {
    pub async fn new(config: Arc<ValidationConfig>) -> Result<Self> {
        Ok(Self { _config: config })
    }
}

#[async_trait]
impl Validator for SemanticValidator {
    async fn validate(&self, query: &ProcessedQuery) -> Result<ValidationResult> {
        let mut score = 1.0;
        let mut issues = Vec::new();
        
        // Check semantic analysis quality
        if query.analysis.confidence < 0.7 {
            score -= 0.3;
            issues.push(format!(
                "Low semantic analysis confidence: {}",
                query.analysis.confidence
            ));
        }
        
        // Check intent classification quality
        if query.intent.confidence < 0.8 {
            score -= 0.2;
            issues.push(format!(
                "Low intent classification confidence: {}",
                query.intent.confidence
            ));
        }
        
        // Check entity extraction quality
        let avg_entity_confidence = if query.entities.is_empty() {
            0.5 // Neutral if no entities
        } else {
            query.entities.iter().map(|e| e.entity.confidence).sum::<f64>() / query.entities.len() as f64
        };
        
        if avg_entity_confidence < 0.7 {
            score -= 0.2;
            issues.push(format!(
                "Low average entity confidence: {}",
                avg_entity_confidence
            ));
        }
        
        // Check for semantic inconsistencies
        if query.intent.primary_intent == QueryIntent::Comparison && query.entities.len() < 2 {
            score -= 0.1;
            issues.push("Comparison query with insufficient entities".to_string());
        }
        
        let status = if score >= 0.8 {
            ValidationStatus::Passed
        } else if score >= 0.5 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Failed
        };
        
        let message = if issues.is_empty() {
            "Semantic validation passed".to_string()
        } else {
            format!("Semantic issues: {}", issues.join("; "))
        };
        
        Ok(ValidationResult {
            stage: "semantic".to_string(),
            status,
            message,
            score,
            validated_at: Utc::now(),
        })
    }
}

/// Factual validation layer
pub struct FactualValidator {
    _config: Arc<ValidationConfig>,
}

impl FactualValidator {
    pub async fn new(config: Arc<ValidationConfig>) -> Result<Self> {
        Ok(Self { _config: config })
    }
}

#[async_trait]
impl Validator for FactualValidator {
    async fn validate(&self, query: &ProcessedQuery) -> Result<ValidationResult> {
        let mut score = 1.0;
        let mut issues = Vec::new();
        
        // Check for factual consistency in entity extraction
        for entity in &query.entities {
            // Validate entity types
            match entity.category {
                EntityCategory::Standard => {
                    if !entity.entity.text.to_lowercase().contains("pci") 
                        && !entity.entity.text.to_lowercase().contains("hipaa")
                        && !entity.entity.text.to_lowercase().contains("sox")
                        && !entity.entity.text.to_lowercase().contains("gdpr") {
                        score -= 0.1;
                        issues.push(format!(
                            "Questionable standard classification: {}",
                            entity.entity.text
                        ));
                    }
                }
                EntityCategory::Version => {
                    // Check version format
                    if !entity.entity.text.chars().any(|c| c.is_ascii_digit()) {
                        score -= 0.05;
                        issues.push(format!(
                            "Version without numbers: {}",
                            entity.entity.text
                        ));
                    }
                }
                _ => {}
            }
        }
        
        // Check strategy-intent alignment
        let strategy_intent_alignment = self.check_strategy_intent_alignment(
            &query.intent.primary_intent,
            &query.strategy.strategy
        );
        
        if !strategy_intent_alignment {
            score -= 0.2;
            issues.push("Strategy-intent misalignment detected".to_string());
        }
        
        let status = if score >= 0.8 {
            ValidationStatus::Passed
        } else if score >= 0.5 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Failed
        };
        
        let message = if issues.is_empty() {
            "Factual validation passed".to_string()
        } else {
            format!("Factual issues: {}", issues.join("; "))
        };
        
        Ok(ValidationResult {
            stage: "factual".to_string(),
            status,
            message,
            score,
            validated_at: Utc::now(),
        })
    }
}

impl FactualValidator {
    /// Check if strategy aligns with intent
    fn check_strategy_intent_alignment(&self, intent: &QueryIntent, strategy: &SearchStrategy) -> bool {
        match (intent, strategy) {
            (QueryIntent::Factual, SearchStrategy::KeywordSearch) => true,
            (QueryIntent::Factual, SearchStrategy::ExactMatch) => true,
            (QueryIntent::Comparison, SearchStrategy::SemanticSearch) => true,
            (QueryIntent::Comparison, SearchStrategy::HybridSearch) => true,
            (QueryIntent::Summary, SearchStrategy::VectorSimilarity) => true,
            (QueryIntent::ComplianceCheck, SearchStrategy::HybridSearch) => true,
            (QueryIntent::ComplianceCheck, SearchStrategy::SemanticSearch) => true,
            _ => false, // Conservative approach - mark as misaligned if not explicitly aligned
        }
    }
}

/// Performance validation layer
pub struct PerformanceValidator {
    config: Arc<ValidationConfig>,
}

impl PerformanceValidator {
    pub async fn new(config: Arc<ValidationConfig>) -> Result<Self> {
        Ok(Self { config })
    }
}

#[async_trait]
impl Validator for PerformanceValidator {
    async fn validate(&self, query: &ProcessedQuery) -> Result<ValidationResult> {
        let mut score = 1.0;
        let mut issues = Vec::new();
        
        // Check processing time
        let total_duration = query.processing_metadata.total_duration;
        if total_duration > self.config.rules.max_processing_time {
            score -= 0.3;
            issues.push(format!(
                "Processing time {:?} exceeds maximum {:?}",
                total_duration,
                self.config.rules.max_processing_time
            ));
        }
        
        // Check memory usage (if available)
        let peak_memory = query.processing_metadata.statistics.resource_usage.peak_memory;
        if peak_memory > 100_000_000 { // 100MB
            score -= 0.1;
            issues.push(format!(
                "High memory usage: {} bytes",
                peak_memory
            ));
        }
        
        // Check API call efficiency
        let api_calls = query.processing_metadata.statistics.resource_usage.api_calls;
        if api_calls > 10 {
            score -= 0.1;
            issues.push(format!(
                "High API call count: {}",
                api_calls
            ));
        }
        
        let status = if score >= 0.8 {
            ValidationStatus::Passed
        } else if score >= 0.5 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Failed
        };
        
        let message = if issues.is_empty() {
            "Performance validation passed".to_string()
        } else {
            format!("Performance issues: {}", issues.join("; "))
        };
        
        Ok(ValidationResult {
            stage: "performance".to_string(),
            status,
            message,
            score,
            validated_at: Utc::now(),
        })
    }
}

/// Consistency checker
pub struct ConsistencyChecker {
    _config: Arc<ValidationConfig>,
}

impl ConsistencyChecker {
    pub async fn new(config: Arc<ValidationConfig>) -> Result<Self> {
        Ok(Self { _config: config })
    }
}

#[async_trait]
impl Validator for ConsistencyChecker {
    async fn validate(&self, query: &ProcessedQuery) -> Result<ValidationResult> {
        let mut score = 1.0;
        let mut issues = Vec::new();
        
        // Check entity-term consistency
        let entity_terms: Vec<String> = query.entities.iter()
            .map(|e| e.entity.text.to_lowercase())
            .collect();
        
        let key_terms: Vec<String> = query.key_terms.iter()
            .map(|t| t.term.to_lowercase())
            .collect();
        
        let overlap_count = entity_terms.iter()
            .filter(|entity| key_terms.iter().any(|term| term.contains(*entity) || entity.contains(term)))
            .count();
        
        if !entity_terms.is_empty() && !key_terms.is_empty() && overlap_count == 0 {
            score -= 0.2;
            issues.push("No overlap between entities and key terms".to_string());
        }
        
        // Check intent-analysis consistency
        if query.intent.primary_intent == QueryIntent::Unknown && query.analysis.confidence > 0.8 {
            score -= 0.3;
            issues.push("High analysis confidence but unknown intent".to_string());
        }
        
        // Check strategy-performance consistency
        let predicted_latency_secs = query.strategy.predictions.latency;
        let actual_duration_secs = query.processing_metadata.total_duration.as_secs_f64();
        
        // Allow 50% variance
        if actual_duration_secs > predicted_latency_secs + predicted_latency_secs / 2.0 {
            score -= 0.1;
            issues.push(format!(
                "Actual duration {:.2}s significantly exceeds prediction {:.2}s",
                actual_duration_secs,
                predicted_latency_secs
            ));
        }
        
        let status = if score >= 0.8 {
            ValidationStatus::Passed
        } else if score >= 0.5 {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Failed
        };
        
        let message = if issues.is_empty() {
            "Consistency validation passed".to_string()
        } else {
            format!("Consistency issues: {}", issues.join("; "))
        };
        
        Ok(ValidationResult {
            stage: "consistency".to_string(),
            status,
            message,
            score,
            validated_at: Utc::now(),
        })
    }
}

/// Base validator trait
#[async_trait]
trait Validator {
    async fn validate(&self, query: &ProcessedQuery) -> Result<ValidationResult>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ValidationConfig;
    use crate::query::Query;
    use std::collections::HashMap;
    use uuid::Uuid;
    use std::time::Duration;

    fn create_test_processed_query() -> ProcessedQuery {
        let query = Query::new("What are PCI DSS encryption requirements?");
        let analysis = SemanticAnalysis::new(
            SyntacticFeatures {
                pos_tags: vec![],
                named_entities: vec![
                    NamedEntity::new("PCI DSS".to_string(), "STANDARD".to_string(), 9, 16, 0.95),
                ],
                noun_phrases: vec![],
                verb_phrases: vec![],
                question_words: vec!["what".to_string()],
            },
            SemanticFeatures {
                semantic_roles: vec![],
                coreferences: vec![],
                sentiment: None,
                similarity_vectors: vec![],
            },
            vec![], // dependencies
            vec![], // topics
            0.8,    // confidence
            std::time::Duration::from_millis(10), // processing_time
        );
        let entities = vec![
            ExtractedEntity {
                entity: NamedEntity::new("PCI DSS".to_string(), "STANDARD".to_string(), 9, 16, 0.95),
                category: EntityCategory::Standard,
                metadata: EntityMetadata {
                    extraction_method: "test".to_string(),
                    extracted_at: Utc::now(),
                    context: "".to_string(),
                    normalization: None,
                    properties: HashMap::new(),
                },
                relationships: vec![],
            },
        ];
        let key_terms = vec![
            KeyTerm {
                id: Uuid::new_v4(),
                term: "encryption".to_string(),
                normalized: "encryption".to_string(),
                tfidf_score: 0.8,
                frequency: 1,
                category: TermCategory::Technical,
                positions: vec![(28, 38)],
                importance: 0.9,
                contexts: vec![],
                ngram_size: 1,
            },
        ];
        let intent = IntentClassification {
            primary_intent: QueryIntent::Factual,
            confidence: 0.9,
            secondary_intents: vec![],
            probabilities: HashMap::new(),
            method: ClassificationMethod::RuleBased,
            features: vec![],
        };
        let strategy = StrategySelection {
            strategy: SearchStrategy::KeywordSearch,
            confidence: 0.85,
            fallbacks: vec![],
            reasoning: "Test reasoning".to_string(),
            expected_metrics: PerformanceMetrics {
                expected_accuracy: 0.9,
                expected_response_time: Duration::from_millis(100),
                expected_recall: 0.85,
                expected_precision: 0.90,
                resource_usage: ResourceUsage {
                    cpu_usage: 0.5,
                    memory_usage: 1000,
                    network_io: 0,
                    disk_io: 0,
                    memory: 1000,
                    cpu: 0.5,
                    api_calls: 1,
                    cache_hits: 0,
                    cache_misses: 1,
                    peak_memory: 2000,
                },
            },
            predictions: StrategyPredictions {
                latency: 0.1,
                accuracy: 0.85,
                resource_usage: 0.5,
            },
        };
        
        ProcessedQuery::new(query.unwrap(), analysis, entities, key_terms, intent, strategy)
    }

    #[tokio::test]
    async fn test_validation_engine_creation() {
        let config = Arc::new(ValidationConfig::default());
        let engine = ValidationEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_syntax_validator() {
        let config = Arc::new(ValidationConfig::default());
        let validator = SyntaxValidator::new(config).await.unwrap();
        let query = create_test_processed_query();
        
        let result = validator.validate(&query).await;
        assert!(result.is_ok());
        
        let validation = result.unwrap();
        assert_eq!(validation.stage, "syntax");
        assert_eq!(validation.status, ValidationStatus::Passed);
    }

    #[tokio::test]
    async fn test_semantic_validator() {
        let config = Arc::new(ValidationConfig::default());
        let validator = SemanticValidator::new(config).await.unwrap();
        let query = create_test_processed_query();
        
        let result = validator.validate(&query).await;
        assert!(result.is_ok());
        
        let validation = result.unwrap();
        assert_eq!(validation.stage, "semantic");
        assert!(validation.score > 0.5);
    }

    #[tokio::test]
    async fn test_factual_validator() {
        let config = Arc::new(ValidationConfig::default());
        let validator = FactualValidator::new(config).await.unwrap();
        let query = create_test_processed_query();
        
        let result = validator.validate(&query).await;
        assert!(result.is_ok());
        
        let validation = result.unwrap();
        assert_eq!(validation.stage, "factual");
    }

    #[tokio::test]
    async fn test_strategy_intent_alignment() {
        let config = Arc::new(ValidationConfig::default());
        let validator = FactualValidator::new(config).await.unwrap();
        
        // Test aligned combination
        assert!(validator.check_strategy_intent_alignment(
            &QueryIntent::Factual, 
            &SearchStrategy::KeywordSearch
        ));
        
        // Test misaligned combination (example)
        assert!(!validator.check_strategy_intent_alignment(
            &QueryIntent::Factual,
            &SearchStrategy::NeuralSearch
        ));
    }

    #[tokio::test]
    async fn test_complete_validation() {
        let config = Arc::new(ValidationConfig::default());
        let engine = ValidationEngine::new(config).await.unwrap();
        let query = create_test_processed_query();
        
        let result = engine.validate(query).await;
        assert!(result.is_ok());
        
        let validated = result.unwrap();
        assert!(!validated.processing_metadata.validation_results.is_empty());
        
        // Should have multiple validation stages
        let stages: Vec<String> = validated.processing_metadata.validation_results
            .iter()
            .map(|r| r.stage.clone())
            .collect();
        
        assert!(stages.contains(&"syntax".to_string()));
        assert!(stages.contains(&"semantic".to_string()));
        assert!(stages.contains(&"factual".to_string()));
    }
}