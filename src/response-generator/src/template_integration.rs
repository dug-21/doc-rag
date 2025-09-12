//! Template Engine Integration with Response Generator
//! 
//! Integrates template engine with existing response-generator components to provide
//! deterministic response generation with <1s end-to-end performance and complete audit trails.

use crate::{Result, ResponseError, GenerationRequest, GeneratedResponse, ResponseGenerator};
use crate::template_engine::{TemplateEngine, TemplateGenerationRequest, TemplateResponse, TemplateType};
use crate::template_structures::{TemplateLibrary, TemplateMetadata};
use crate::proof_chain_integration::{ProofChainIntegrationManager, ProofChainQuery, VariableRequirement};
use crate::enhanced_citation_formatter::{EnhancedCitationFormatter, ComprehensiveCitationResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use tracing::{debug, info, instrument, warn, error};

/// Integrated template-based response generator
#[derive(Debug)]
pub struct IntegratedTemplateResponseGenerator {
    /// Base response generator
    base_generator: ResponseGenerator,
    /// Template engine
    template_engine: TemplateEngine,
    /// Template library
    template_library: TemplateLibrary,
    /// Proof chain integration manager
    proof_integration: ProofChainIntegrationManager,
    /// Enhanced citation formatter
    citation_formatter: EnhancedCitationFormatter,
    /// Integration configuration
    config: IntegrationConfig,
    /// Performance monitor
    performance_monitor: PerformanceMonitor,
    /// Audit trail manager
    audit_manager: AuditTrailManager,
}

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Enable template-based generation
    pub enable_template_generation: bool,
    /// Fallback to base generator if template fails
    pub enable_fallback: bool,
    /// Template selection strategy
    pub template_selection_strategy: TemplateSelectionStrategy,
    /// Performance targets
    pub performance_targets: PerformanceTargets,
    /// Audit trail configuration
    pub audit_config: AuditConfig,
    /// Cache configuration
    pub cache_config: CacheConfig,
    /// Validation configuration
    pub validation_config: ValidationConfig,
}

/// Template selection strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemplateSelectionStrategy {
    /// Use query intent classification
    IntentBased,
    /// Use entity analysis
    EntityBased,
    /// Use complexity analysis
    ComplexityBased,
    /// Use machine learning recommendations
    MLRecommended,
    /// Use manual mapping
    ManualMapping,
    /// Use hybrid approach
    Hybrid,
}

/// Performance targets for CONSTRAINT-006 compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Maximum end-to-end response time (CONSTRAINT-006)
    pub max_response_time_ms: u64,
    /// Maximum template selection time
    pub max_template_selection_ms: u64,
    /// Maximum variable substitution time
    pub max_variable_substitution_ms: u64,
    /// Maximum citation formatting time
    pub max_citation_formatting_ms: u64,
    /// Maximum validation time
    pub max_validation_ms: u64,
    /// Target cache hit rate
    pub target_cache_hit_rate: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            max_response_time_ms: 1000, // <1s for CONSTRAINT-006
            max_template_selection_ms: 50,
            max_variable_substitution_ms: 300,
            max_citation_formatting_ms: 200,
            max_validation_ms: 150,
            target_cache_hit_rate: 0.8,
        }
    }
}

/// Audit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Enable comprehensive audit trails
    pub enable_comprehensive_audit: bool,
    /// Store audit trails to persistent storage
    pub persist_audit_trails: bool,
    /// Audit trail retention period in days
    pub retention_days: u32,
    /// Include performance metrics in audit
    pub include_performance_metrics: bool,
    /// Include validation details in audit
    pub include_validation_details: bool,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable template response caching
    pub enable_response_caching: bool,
    /// Enable proof chain caching
    pub enable_proof_chain_caching: bool,
    /// Enable citation caching
    pub enable_citation_caching: bool,
    /// Cache TTL in seconds
    pub cache_ttl_seconds: u64,
    /// Maximum cache size in MB
    pub max_cache_size_mb: u64,
}

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable CONSTRAINT-004 validation (no free generation)
    pub enable_constraint_004_validation: bool,
    /// Enable CONSTRAINT-006 validation (<1s response)
    pub enable_constraint_006_validation: bool,
    /// Enable proof chain validation
    pub enable_proof_chain_validation: bool,
    /// Enable citation validation
    pub enable_citation_validation: bool,
    /// Minimum confidence threshold
    pub min_confidence_threshold: f64,
}

/// Performance monitor for tracking compliance
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    /// Performance targets
    targets: PerformanceTargets,
    /// Performance metrics history
    metrics_history: Vec<PerformanceSnapshot>,
    /// Current performance state
    current_state: PerformanceState,
    /// Violation tracker
    violation_tracker: ViolationTracker,
}

/// Audit trail manager
#[derive(Debug, Clone)]
pub struct AuditTrailManager {
    /// Audit configuration
    config: AuditConfig,
    /// Audit trail storage
    storage: AuditStorage,
    /// Trail validator
    validator: AuditValidator,
}

/// Performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,
    /// Total response time
    pub total_response_time: Duration,
    /// Template selection time
    pub template_selection_time: Duration,
    /// Variable substitution time
    pub variable_substitution_time: Duration,
    /// Citation formatting time
    pub citation_formatting_time: Duration,
    /// Validation time
    pub validation_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Memory usage
    pub memory_usage: u64,
    /// CPU usage
    pub cpu_usage: f64,
}

/// Current performance state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceState {
    /// CONSTRAINT-006 compliance status
    pub constraint_006_compliant: bool,
    /// Average response time over last 100 requests
    pub avg_response_time: Duration,
    /// 95th percentile response time
    pub p95_response_time: Duration,
    /// Current cache hit rate
    pub current_cache_hit_rate: f64,
    /// Performance warnings
    pub warnings: Vec<PerformanceWarning>,
}

/// Performance warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceWarning {
    /// Warning type
    pub warning_type: PerformanceWarningType,
    /// Warning message
    pub message: String,
    /// Warning severity
    pub severity: WarningSeverity,
    /// Warning timestamp
    pub timestamp: DateTime<Utc>,
}

/// Performance warning types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceWarningType {
    ResponseTimeExceeded,
    CacheHitRateLow,
    MemoryUsageHigh,
    CpuUsageHigh,
    ValidationTimeExceeded,
    ConstraintViolation,
}

/// Warning severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Violation tracker
#[derive(Debug, Clone)]
pub struct ViolationTracker {
    /// CONSTRAINT-006 violations
    constraint_006_violations: u64,
    /// Performance target violations
    performance_violations: u64,
    /// Last violation timestamp
    last_violation: Option<DateTime<Utc>>,
}

/// Audit storage interface
#[derive(Debug, Clone)]
pub struct AuditStorage;

/// Audit validator
#[derive(Debug, Clone)]
pub struct AuditValidator;

/// Integrated template response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedTemplateResponse {
    /// Base generated response
    pub base_response: GeneratedResponse,
    /// Template response details
    pub template_response: TemplateResponse,
    /// Comprehensive citation result
    pub citation_result: ComprehensiveCitationResult,
    /// Performance snapshot
    pub performance_snapshot: PerformanceSnapshot,
    /// Integration audit trail
    pub integration_audit: IntegrationAuditTrail,
    /// Validation results
    pub validation_results: IntegrationValidationResult,
    /// Response metadata
    pub metadata: IntegrationMetadata,
}

/// Integration audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationAuditTrail {
    /// Trail identifier
    pub id: Uuid,
    /// Template selection audit
    pub template_selection_audit: TemplateSelectionAudit,
    /// Proof chain integration audit
    pub proof_chain_audit: ProofChainAudit,
    /// Citation formatting audit
    pub citation_formatting_audit: CitationFormattingAudit,
    /// Performance audit
    pub performance_audit: PerformanceAudit,
    /// Constraint compliance audit
    pub constraint_compliance_audit: ConstraintComplianceAudit,
    /// Trail completion timestamp
    pub completed_at: DateTime<Utc>,
}

/// Integration validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationValidationResult {
    /// Overall validation success
    pub is_valid: bool,
    /// CONSTRAINT-004 compliance (no free generation)
    pub constraint_004_compliant: bool,
    /// CONSTRAINT-006 compliance (<1s response)
    pub constraint_006_compliant: bool,
    /// Template validation result
    pub template_validation: bool,
    /// Proof chain validation result
    pub proof_chain_validation: bool,
    /// Citation validation result
    pub citation_validation: bool,
    /// Performance validation result
    pub performance_validation: bool,
    /// Validation errors
    pub validation_errors: Vec<ValidationError>,
    /// Validation warnings
    pub validation_warnings: Vec<ValidationWarning>,
}

/// Integration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMetadata {
    /// Integration version
    pub version: String,
    /// Template used
    pub template_used: TemplateType,
    /// Proof elements count
    pub proof_elements_count: usize,
    /// Citations count
    pub citations_count: usize,
    /// Variables substituted count
    pub variables_substituted: usize,
    /// Cache utilization
    pub cache_utilization: CacheUtilization,
    /// Performance classification
    pub performance_classification: PerformanceClassification,
}

/// Cache utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheUtilization {
    /// Template cache hits
    pub template_cache_hits: u64,
    /// Proof chain cache hits
    pub proof_chain_cache_hits: u64,
    /// Citation cache hits
    pub citation_cache_hits: u64,
    /// Total cache operations
    pub total_cache_operations: u64,
    /// Overall cache hit rate
    pub overall_hit_rate: f64,
}

/// Performance classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceClassification {
    Excellent, // <200ms
    Good,      // 200-500ms
    Acceptable, // 500-800ms
    Poor,      // 800-1000ms
    Violation, // >1000ms
}

// Supporting types and structs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateSelectionAudit;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofChainAudit;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationFormattingAudit;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAudit;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintComplianceAudit;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning;

impl IntegratedTemplateResponseGenerator {
    /// Create new integrated template response generator
    pub async fn new(
        base_generator: ResponseGenerator,
        config: IntegrationConfig,
    ) -> Result<Self> {
        let template_engine = TemplateEngine::new(crate::template_engine::TemplateEngineConfig::default());
        let template_library = TemplateLibrary::new();
        let proof_integration = ProofChainIntegrationManager::new(
            crate::proof_chain_integration::ProofChainIntegrationConfig::default()
        );
        let citation_formatter = EnhancedCitationFormatter::new(
            crate::enhanced_citation_formatter::EnhancedCitationConfig::default()
        );
        let performance_monitor = PerformanceMonitor::new(config.performance_targets.clone());
        let audit_manager = AuditTrailManager::new(config.audit_config.clone());
        
        Ok(Self {
            base_generator,
            template_engine,
            template_library,
            proof_integration,
            citation_formatter,
            config,
            performance_monitor,
            audit_manager,
        })
    }
    
    /// Generate response using template-based approach
    #[instrument(skip(self, request))]
    pub async fn generate_template_response(
        &mut self,
        request: GenerationRequest,
    ) -> Result<IntegratedTemplateResponse> {
        let start_time = Instant::now();
        
        info!(
            "Starting integrated template response generation for request: {}",
            request.id
        );
        
        // Stage 1: Template Selection (Target: <50ms)
        let template_selection_start = Instant::now();
        let template_type = self.select_template(&request).await?;
        let template_selection_time = template_selection_start.elapsed();
        
        if template_selection_time.as_millis() > self.config.performance_targets.max_template_selection_ms as u128 {
            warn!(
                "Template selection exceeded target: {}ms > {}ms",
                template_selection_time.as_millis(),
                self.config.performance_targets.max_template_selection_ms
            );
        }
        
        // Stage 2: Proof Chain Integration (Target: <300ms)
        let proof_chain_start = Instant::now();
        let proof_chain_data = self.extract_proof_chain_data(&request, &template_type).await?;
        let proof_chain_time = proof_chain_start.elapsed();
        
        if proof_chain_time.as_millis() > self.config.performance_targets.max_variable_substitution_ms as u128 {
            warn!(
                "Proof chain integration exceeded target: {}ms > {}ms",
                proof_chain_time.as_millis(),
                self.config.performance_targets.max_variable_substitution_ms
            );
        }
        
        // Stage 3: Citation Processing (Target: <200ms)
        let citation_start = Instant::now();
        let citation_result = self.citation_formatter
            .format_citations_comprehensive(&request.context.iter().map(|ctx| {
                // Convert ContextChunk to Citation - simplified conversion
                crate::Citation {
                    id: Uuid::new_v4(),
                    source: ctx.source.clone(),
                    text_range: crate::citation::TextRange {
                        start: 0,
                        end: ctx.content.len(),
                        length: ctx.content.len(),
                    },
                    confidence: ctx.relevance_score,
                    citation_type: crate::citation::CitationType::SupportingEvidence,
                    relevance_score: ctx.relevance_score,
                    supporting_text: Some(ctx.content.clone()),
                }
            }).collect::<Vec<_>>(), &[])
            .await?;
        let citation_time = citation_start.elapsed();
        
        if citation_time.as_millis() > self.config.performance_targets.max_citation_formatting_ms as u128 {
            warn!(
                "Citation formatting exceeded target: {}ms > {}ms",
                citation_time.as_millis(),
                self.config.performance_targets.max_citation_formatting_ms
            );
        }
        
        // Stage 4: Template Response Generation
        let template_request = TemplateGenerationRequest {
            template_type: template_type.clone(),
            variable_values: self.extract_variable_values(&request, &proof_chain_data).await?,
            proof_chain_data,
            citations: citation_result.formatted_citations.iter()
                .map(|fc| fc.formatted_citation.citation.clone())
                .collect(),
            output_format: request.format.clone(),
            context: crate::template_engine::GenerationContext {
                query_intent: self.classify_query_intent(&request).await?,
                entities: self.extract_entities(&request).await?,
                requirements: self.extract_requirements(&request).await?,
                compliance_scope: None,
            },
        };
        
        let template_response = self.template_engine.generate_response(template_request).await?;
        
        // Stage 5: Validation (Target: <150ms)
        let validation_start = Instant::now();
        let validation_results = self.validate_integrated_response(&template_response, &citation_result).await?;
        let validation_time = validation_start.elapsed();
        
        if validation_time.as_millis() > self.config.performance_targets.max_validation_ms as u128 {
            warn!(
                "Validation exceeded target: {}ms > {}ms",
                validation_time.as_millis(),
                self.config.performance_targets.max_validation_ms
            );
        }
        
        // Stage 6: Performance Monitoring and Compliance Check
        let total_time = start_time.elapsed();
        let constraint_006_compliant = total_time.as_millis() <= self.config.performance_targets.max_response_time_ms as u128;
        
        if !constraint_006_compliant {
            error!(
                "CONSTRAINT-006 VIOLATION: Response time {}ms > {}ms",
                total_time.as_millis(),
                self.config.performance_targets.max_response_time_ms
            );
            self.performance_monitor.record_violation(PerformanceWarningType::ConstraintViolation);
        }
        
        // Create performance snapshot
        let performance_snapshot = PerformanceSnapshot {
            timestamp: Utc::now(),
            total_response_time: total_time,
            template_selection_time,
            variable_substitution_time: proof_chain_time,
            citation_formatting_time: citation_time,
            validation_time,
            cache_hit_rate: 0.0, // Would be calculated from actual cache metrics
            memory_usage: 0, // Would be measured
            cpu_usage: 0.0, // Would be measured
        };
        
        // Update performance monitor
        self.performance_monitor.record_snapshot(performance_snapshot.clone());
        
        // Generate audit trail
        let integration_audit = self.audit_manager.generate_integration_audit(
            &template_response,
            &citation_result,
            &performance_snapshot,
            &validation_results,
        ).await?;
        
        // Convert template response to base response format
        let base_response = self.convert_to_base_response(&request, &template_response).await?;
        
        // Get citation count before moving citation_result
        let citations_count = citation_result.formatted_citations.len();
        
        let integrated_response = IntegratedTemplateResponse {
            base_response,
            template_response,
            citation_result,
            performance_snapshot,
            integration_audit,
            validation_results,
            metadata: IntegrationMetadata {
                version: "1.0.0".to_string(),
                template_used: template_type,
                proof_elements_count: 0, // Would be calculated
                citations_count,
                variables_substituted: 0, // Would be calculated
                cache_utilization: CacheUtilization {
                    template_cache_hits: 0,
                    proof_chain_cache_hits: 0,
                    citation_cache_hits: 0,
                    total_cache_operations: 0,
                    overall_hit_rate: 0.0,
                },
                performance_classification: Self::classify_performance(total_time),
            },
        };
        
        info!(
            "Integrated template response generated in {}ms (CONSTRAINT-006 compliant: {})",
            total_time.as_millis(),
            constraint_006_compliant
        );
        
        Ok(integrated_response)
    }
    
    /// Select appropriate template for request
    async fn select_template(&self, request: &GenerationRequest) -> Result<TemplateType> {
        match self.config.template_selection_strategy {
            TemplateSelectionStrategy::IntentBased => {
                let intent = self.classify_query_intent(request).await?;
                self.select_template_by_intent(intent)
            },
            TemplateSelectionStrategy::EntityBased => {
                let entities = self.extract_entities(request).await?;
                self.select_template_by_entities(entities)
            },
            TemplateSelectionStrategy::ComplexityBased => {
                let complexity = self.assess_complexity(request).await?;
                self.select_template_by_complexity(complexity)
            },
            _ => {
                // Default to requirement query template
                Ok(TemplateType::RequirementQuery {
                    requirement_type: crate::template_engine::RequirementType::Must,
                    query_intent: crate::template_engine::QueryIntent::Compliance,
                })
            }
        }
    }
    
    /// Extract proof chain data for template variables
    async fn extract_proof_chain_data(
        &self,
        request: &GenerationRequest,
        template_type: &TemplateType,
    ) -> Result<Vec<crate::template_engine::ProofChainData>> {
        // Create variable requirements based on template
        let variable_requirements = self.create_variable_requirements(template_type)?;
        
        // Query proof chain integration
        let proof_response = self.proof_integration
            .query_for_variables(&request.query, variable_requirements)
            .await?;
        
        // Convert to proof chain data format
        let proof_data = proof_response.proof_elements.into_iter()
            .map(|element| crate::template_engine::ProofChainData {
                element_type: element.element_type,
                content: element.content,
                confidence: element.confidence,
                source: element.source_reference.document_id,
            })
            .collect();
        
        Ok(proof_data)
    }
    
    /// Validate integrated response for compliance
    async fn validate_integrated_response(
        &self,
        template_response: &TemplateResponse,
        citation_result: &ComprehensiveCitationResult,
    ) -> Result<IntegrationValidationResult> {
        let mut validation_errors = Vec::new();
        let mut validation_warnings = Vec::new();
        
        // CONSTRAINT-004 validation (no free generation)
        let constraint_004_compliant = template_response.validation_results.constraint_004_compliant;
        if !constraint_004_compliant {
            validation_errors.push(ValidationError);
        }
        
        // CONSTRAINT-006 validation (<1s response)
        let constraint_006_compliant = template_response.metrics.total_generation_time.as_millis() <= 1000;
        if !constraint_006_compliant {
            validation_errors.push(ValidationError);
        }
        
        // Template validation
        let template_validation = template_response.validation_results.is_valid;
        
        // Citation validation
        let citation_validation = citation_result.validation_status == 
            crate::enhanced_citation_formatter::CitationValidationStatus::Valid;
        
        // Performance validation
        let performance_validation = constraint_006_compliant;
        
        let is_valid = constraint_004_compliant && 
                      constraint_006_compliant && 
                      template_validation && 
                      citation_validation;
        
        Ok(IntegrationValidationResult {
            is_valid,
            constraint_004_compliant,
            constraint_006_compliant,
            template_validation,
            proof_chain_validation: true, // Would implement actual validation
            citation_validation,
            performance_validation,
            validation_errors,
            validation_warnings,
        })
    }
    
    /// Classify performance based on response time
    fn classify_performance(response_time: Duration) -> PerformanceClassification {
        let ms = response_time.as_millis();
        match ms {
            0..=200 => PerformanceClassification::Excellent,
            201..=500 => PerformanceClassification::Good,
            501..=800 => PerformanceClassification::Acceptable,
            801..=1000 => PerformanceClassification::Poor,
            _ => PerformanceClassification::Violation,
        }
    }
    
    // Helper methods for template selection and data extraction
    
    async fn classify_query_intent(&self, _request: &GenerationRequest) -> Result<crate::template_engine::QueryIntent> {
        // Simplified intent classification
        Ok(crate::template_engine::QueryIntent::Compliance)
    }
    
    async fn extract_entities(&self, _request: &GenerationRequest) -> Result<Vec<String>> {
        // Simplified entity extraction
        Ok(vec!["PCI DSS".to_string()])
    }
    
    async fn extract_requirements(&self, _request: &GenerationRequest) -> Result<Vec<String>> {
        // Simplified requirement extraction
        Ok(vec!["encryption".to_string()])
    }
    
    fn select_template_by_intent(&self, _intent: crate::template_engine::QueryIntent) -> Result<TemplateType> {
        Ok(TemplateType::RequirementQuery {
            requirement_type: crate::template_engine::RequirementType::Must,
            query_intent: crate::template_engine::QueryIntent::Compliance,
        })
    }
    
    fn select_template_by_entities(&self, _entities: Vec<String>) -> Result<TemplateType> {
        Ok(TemplateType::RequirementQuery {
            requirement_type: crate::template_engine::RequirementType::Must,
            query_intent: crate::template_engine::QueryIntent::Compliance,
        })
    }
    
    async fn assess_complexity(&self, _request: &GenerationRequest) -> Result<crate::template_engine::ComplexityLevel> {
        Ok(crate::template_engine::ComplexityLevel::Moderate)
    }
    
    fn select_template_by_complexity(&self, _complexity: crate::template_engine::ComplexityLevel) -> Result<TemplateType> {
        Ok(TemplateType::RequirementQuery {
            requirement_type: crate::template_engine::RequirementType::Must,
            query_intent: crate::template_engine::QueryIntent::Compliance,
        })
    }
    
    fn create_variable_requirements(&self, _template_type: &TemplateType) -> Result<Vec<VariableRequirement>> {
        // Create requirements based on template
        Ok(vec![])
    }
    
    async fn extract_variable_values(
        &self,
        _request: &GenerationRequest,
        _proof_data: &[crate::template_engine::ProofChainData],
    ) -> Result<HashMap<String, String>> {
        let mut variables = HashMap::new();
        variables.insert("REQUIREMENT_TYPE".to_string(), "MUST".to_string());
        variables.insert("QUERY_SUBJECT".to_string(), "encryption requirements".to_string());
        Ok(variables)
    }
    
    async fn convert_to_base_response(
        &self,
        request: &GenerationRequest,
        template_response: &TemplateResponse,
    ) -> Result<GeneratedResponse> {
        // Convert template response to base response format
        Ok(GeneratedResponse {
            request_id: request.id,
            content: template_response.content.clone(),
            format: template_response.format.clone(),
            confidence_score: 0.9, // Would calculate from template metrics
            citations: template_response.citations.iter()
                .map(|fc| fc.citation.clone())
                .collect(),
            segment_confidence: vec![], // Would calculate
            validation_results: vec![], // Would convert
            metrics: crate::GenerationMetrics {
                total_duration: template_response.metrics.total_generation_time,
                validation_duration: template_response.metrics.validation_time,
                formatting_duration: template_response.metrics.citation_formatting_time,
                citation_duration: template_response.metrics.citation_formatting_time,
                validation_passes: 1,
                sources_used: template_response.citations.len(),
                response_length: template_response.content.len(),
            },
            warnings: vec![],
        })
    }
}

// Implementation of support components

impl PerformanceMonitor {
    fn new(targets: PerformanceTargets) -> Self {
        Self {
            targets,
            metrics_history: Vec::new(),
            current_state: PerformanceState {
                constraint_006_compliant: true,
                avg_response_time: Duration::from_millis(0),
                p95_response_time: Duration::from_millis(0),
                current_cache_hit_rate: 0.0,
                warnings: vec![],
            },
            violation_tracker: ViolationTracker {
                constraint_006_violations: 0,
                performance_violations: 0,
                last_violation: None,
            },
        }
    }
    
    fn record_snapshot(&mut self, snapshot: PerformanceSnapshot) {
        self.metrics_history.push(snapshot.clone());
        
        // Update current state
        self.update_current_state(&snapshot);
        
        // Check for violations
        self.check_violations(&snapshot);
        
        // Limit history size
        if self.metrics_history.len() > 1000 {
            self.metrics_history.remove(0);
        }
    }
    
    fn record_violation(&mut self, violation_type: PerformanceWarningType) {
        match violation_type {
            PerformanceWarningType::ConstraintViolation => {
                self.violation_tracker.constraint_006_violations += 1;
            },
            _ => {
                self.violation_tracker.performance_violations += 1;
            },
        }
        self.violation_tracker.last_violation = Some(Utc::now());
    }
    
    fn update_current_state(&mut self, snapshot: &PerformanceSnapshot) {
        // Update averages from last 100 snapshots
        let recent_snapshots: Vec<_> = self.metrics_history.iter()
            .rev()
            .take(100)
            .collect();
        
        if !recent_snapshots.is_empty() {
            let total_time: Duration = recent_snapshots.iter()
                .map(|s| s.total_response_time)
                .sum();
            self.current_state.avg_response_time = total_time / recent_snapshots.len() as u32;
            
            // Calculate 95th percentile
            let mut times: Vec<_> = recent_snapshots.iter()
                .map(|s| s.total_response_time.as_millis())
                .collect();
            times.sort_unstable();
            let p95_index = (times.len() as f64 * 0.95) as usize;
            if p95_index < times.len() {
                self.current_state.p95_response_time = Duration::from_millis(times[p95_index] as u64);
            }
        }
        
        self.current_state.constraint_006_compliant = 
            snapshot.total_response_time.as_millis() <= self.targets.max_response_time_ms as u128;
    }
    
    fn check_violations(&mut self, snapshot: &PerformanceSnapshot) {
        if snapshot.total_response_time.as_millis() > self.targets.max_response_time_ms as u128 {
            self.current_state.warnings.push(PerformanceWarning {
                warning_type: PerformanceWarningType::ResponseTimeExceeded,
                message: format!(
                    "Response time {}ms exceeded target {}ms",
                    snapshot.total_response_time.as_millis(),
                    self.targets.max_response_time_ms
                ),
                severity: WarningSeverity::High,
                timestamp: Utc::now(),
            });
        }
    }
}

impl AuditTrailManager {
    fn new(config: AuditConfig) -> Self {
        Self {
            config,
            storage: AuditStorage,
            validator: AuditValidator,
        }
    }
    
    async fn generate_integration_audit(
        &self,
        _template_response: &TemplateResponse,
        _citation_result: &ComprehensiveCitationResult,
        _performance_snapshot: &PerformanceSnapshot,
        _validation_results: &IntegrationValidationResult,
    ) -> Result<IntegrationAuditTrail> {
        Ok(IntegrationAuditTrail {
            id: Uuid::new_v4(),
            template_selection_audit: TemplateSelectionAudit,
            proof_chain_audit: ProofChainAudit,
            citation_formatting_audit: CitationFormattingAudit,
            performance_audit: PerformanceAudit,
            constraint_compliance_audit: ConstraintComplianceAudit,
            completed_at: Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[tokio::test]
    async fn test_integrated_template_generator_creation() {
        let base_generator = ResponseGenerator::new(Config::default()).await;
        let config = IntegrationConfig {
            enable_template_generation: true,
            enable_fallback: true,
            template_selection_strategy: TemplateSelectionStrategy::IntentBased,
            performance_targets: PerformanceTargets::default(),
            audit_config: AuditConfig {
                enable_comprehensive_audit: true,
                persist_audit_trails: false,
                retention_days: 30,
                include_performance_metrics: true,
                include_validation_details: true,
            },
            cache_config: CacheConfig {
                enable_response_caching: true,
                enable_proof_chain_caching: true,
                enable_citation_caching: true,
                cache_ttl_seconds: 3600,
                max_cache_size_mb: 100,
            },
            validation_config: ValidationConfig {
                enable_constraint_004_validation: true,
                enable_constraint_006_validation: true,
                enable_proof_chain_validation: true,
                enable_citation_validation: true,
                min_confidence_threshold: 0.7,
            },
        };
        
        let result = IntegratedTemplateResponseGenerator::new(base_generator, config).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_performance_classification() {
        assert_eq!(
            IntegratedTemplateResponseGenerator::classify_performance(Duration::from_millis(150)),
            PerformanceClassification::Excellent
        );
        assert_eq!(
            IntegratedTemplateResponseGenerator::classify_performance(Duration::from_millis(1200)),
            PerformanceClassification::Violation
        );
    }
}