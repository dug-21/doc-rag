//! Template Response Structures for Different Query Types
//! 
//! Defines comprehensive template structures for requirement, compliance, and relationship queries
//! ensuring deterministic response generation with complete audit trails.

use crate::template_engine::{
    TemplateType, ResponseTemplate, ContentStructure, SectionTemplate, TemplateVariable,
    VariableType, RequirementType, QueryIntent, ComplianceType, ComplianceScope,
    RelationshipType, EntityType, FactType, ComplexityLevel, AnalysisType, ComparisonScope,
    ProofElement, CitationRequirements, TemplateValidationRule, ProofElementType,
    CalculationType, ComplianceStatusType, ValidationRuleType, ValidationSeverity
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Template library manager
#[derive(Debug, Clone)]
pub struct TemplateLibrary {
    /// All available templates
    templates: HashMap<TemplateType, ResponseTemplate>,
    /// Template metadata
    metadata: HashMap<TemplateType, TemplateMetadata>,
    /// Template usage statistics
    usage_stats: HashMap<TemplateType, TemplateUsageStats>,
    /// Template validation rules
    validation_rules: HashMap<TemplateType, Vec<TemplateValidationRule>>,
}

/// Template metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMetadata {
    /// Template identifier
    pub id: Uuid,
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template version
    pub version: String,
    /// Template author
    pub author: String,
    /// Creation date
    pub created_at: DateTime<Utc>,
    /// Last modification date
    pub last_modified: DateTime<Utc>,
    /// Template tags
    pub tags: Vec<String>,
    /// Complexity level
    pub complexity_level: ComplexityLevel,
    /// Usage recommendations
    pub usage_recommendations: Vec<String>,
    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,
}

/// Template usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateUsageStats {
    /// Total usage count
    pub usage_count: u64,
    /// Average response time
    pub avg_response_time: std::time::Duration,
    /// Success rate
    pub success_rate: f64,
    /// User satisfaction score
    pub satisfaction_score: f64,
    /// Last used date
    pub last_used: Option<DateTime<Utc>>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
}

/// Performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Expected generation time
    pub expected_generation_time: std::time::Duration,
    /// Memory usage estimate
    pub memory_usage_estimate: u64,
    /// CPU usage estimate
    pub cpu_usage_estimate: f64,
    /// Accuracy estimate
    pub accuracy_estimate: f64,
    /// Completeness estimate
    pub completeness_estimate: f64,
}

/// Performance metrics for templates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average processing time
    pub avg_processing_time: std::time::Duration,
    /// 95th percentile processing time
    pub p95_processing_time: std::time::Duration,
    /// Memory usage peak
    pub peak_memory_usage: u64,
    /// Error rate
    pub error_rate: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

impl TemplateLibrary {
    /// Create new template library with default templates
    pub fn new() -> Self {
        let mut library = Self {
            templates: HashMap::new(),
            metadata: HashMap::new(),
            usage_stats: HashMap::new(),
            validation_rules: HashMap::new(),
        };
        
        library.load_default_templates();
        library
    }
    
    /// Load all default templates
    fn load_default_templates(&mut self) {
        // Load requirement query templates
        self.load_requirement_templates();
        
        // Load compliance query templates
        self.load_compliance_templates();
        
        // Load relationship query templates
        self.load_relationship_templates();
        
        // Load factual query templates
        self.load_factual_templates();
        
        // Load analytical query templates
        self.load_analytical_templates();
    }
    
    /// Load requirement query templates
    fn load_requirement_templates(&mut self) {
        // MUST requirement template
        let must_template = ResponseTemplate {
            id: Uuid::new_v4(),
            name: "MUST Requirement Template".to_string(),
            template_type: TemplateType::RequirementQuery {
                requirement_type: RequirementType::Must,
                query_intent: QueryIntent::Compliance,
            },
            content_structure: ContentStructure {
                introduction: SectionTemplate {
                    name: "Introduction".to_string(),
                    content_template: "## {REQUIREMENT_TYPE} Requirement Analysis\n\nBased on symbolic reasoning analysis of {STANDARD_NAME}, the following mandatory requirement has been identified:\n\n**Requirement ID:** {REQUIREMENT_ID}\n**Section:** {SECTION_REFERENCE}\n**Confidence:** {CONFIDENCE_SCORE}".to_string(),
                    variables: vec!["REQUIREMENT_TYPE".to_string(), "STANDARD_NAME".to_string(), "REQUIREMENT_ID".to_string(), "SECTION_REFERENCE".to_string(), "CONFIDENCE_SCORE".to_string()],
                    required_elements: vec!["REQUIREMENT_TYPE".to_string(), "REQUIREMENT_ID".to_string()],
                    order: 1,
                },
                main_sections: vec![
                    SectionTemplate {
                        name: "Requirement Statement".to_string(),
                        content_template: "### Requirement Statement\n\n{REQUIREMENT_TEXT}\n\n**Applicability:** {APPLICABILITY_CONDITIONS}\n**Compliance Level:** {COMPLIANCE_LEVEL}\n**Implementation Priority:** {PRIORITY_LEVEL}".to_string(),
                        variables: vec!["REQUIREMENT_TEXT".to_string(), "APPLICABILITY_CONDITIONS".to_string(), "COMPLIANCE_LEVEL".to_string(), "PRIORITY_LEVEL".to_string()],
                        required_elements: vec!["REQUIREMENT_TEXT".to_string()],
                        order: 2,
                    },
                    SectionTemplate {
                        name: "Implementation Guidance".to_string(),
                        content_template: "### Implementation Guidance\n\n{IMPLEMENTATION_GUIDANCE}\n\n**Key Controls:**\n{CONTROL_REQUIREMENTS}\n\n**Technical Specifications:**\n{TECHNICAL_SPECIFICATIONS}".to_string(),
                        variables: vec!["IMPLEMENTATION_GUIDANCE".to_string(), "CONTROL_REQUIREMENTS".to_string(), "TECHNICAL_SPECIFICATIONS".to_string()],
                        required_elements: vec!["IMPLEMENTATION_GUIDANCE".to_string()],
                        order: 3,
                    },
                    SectionTemplate {
                        name: "Compliance Assessment".to_string(),
                        content_template: "### Compliance Assessment\n\n**Current Status:** {COMPLIANCE_STATUS}\n**Gap Analysis:** {GAP_ANALYSIS}\n**Remediation Steps:** {REMEDIATION_STEPS}\n\n**Assessment Confidence:** {ASSESSMENT_CONFIDENCE}".to_string(),
                        variables: vec!["COMPLIANCE_STATUS".to_string(), "GAP_ANALYSIS".to_string(), "REMEDIATION_STEPS".to_string(), "ASSESSMENT_CONFIDENCE".to_string()],
                        required_elements: vec!["COMPLIANCE_STATUS".to_string()],
                        order: 4,
                    },
                ],
                citations_section: SectionTemplate {
                    name: "Sources and References".to_string(),
                    content_template: "### Sources and References\n\n{FORMATTED_CITATIONS}\n\n**Proof Chain References:**\n{PROOF_CHAIN_REFERENCES}".to_string(),
                    variables: vec!["FORMATTED_CITATIONS".to_string(), "PROOF_CHAIN_REFERENCES".to_string()],
                    required_elements: vec!["FORMATTED_CITATIONS".to_string()],
                    order: 5,
                },
                conclusion: SectionTemplate {
                    name: "Summary".to_string(),
                    content_template: "### Summary\n\nThis analysis provides a deterministic assessment of the {REQUIREMENT_TYPE} requirement with {OVERALL_CONFIDENCE} confidence based on symbolic reasoning chains. All implementation guidance is derived from authoritative sources with complete audit trails.\n\n**Key Takeaways:**\n{KEY_TAKEAWAYS}".to_string(),
                    variables: vec!["REQUIREMENT_TYPE".to_string(), "OVERALL_CONFIDENCE".to_string(), "KEY_TAKEAWAYS".to_string()],
                    required_elements: vec!["OVERALL_CONFIDENCE".to_string()],
                    order: 6,
                },
                audit_trail_section: SectionTemplate {
                    name: "Audit Trail".to_string(),
                    content_template: "---\n**Audit Trail**\n- Generated: {GENERATION_TIMESTAMP}\n- Processing Time: {PROCESSING_TIME_MS}ms\n- Template: {TEMPLATE_NAME}\n- Proof Elements: {PROOF_ELEMENTS_COUNT}\n- Citations: {CITATIONS_COUNT}\n- Validation Status: {VALIDATION_STATUS}".to_string(),
                    variables: vec!["GENERATION_TIMESTAMP".to_string(), "PROCESSING_TIME_MS".to_string(), "TEMPLATE_NAME".to_string(), "PROOF_ELEMENTS_COUNT".to_string(), "CITATIONS_COUNT".to_string(), "VALIDATION_STATUS".to_string()],
                    required_elements: vec!["GENERATION_TIMESTAMP".to_string()],
                    order: 7,
                },
            },
            variables: vec![
                TemplateVariable {
                    name: "REQUIREMENT_TYPE".to_string(),
                    variable_type: VariableType::RequirementReference {
                        requirement_type: RequirementType::Must,
                        include_conditions: true,
                    },
                    required: true,
                    default_value: Some("MUST".to_string()),
                    validation: None,
                    description: "Type of requirement (MUST/SHOULD/MAY)".to_string(),
                },
                TemplateVariable {
                    name: "REQUIREMENT_TEXT".to_string(),
                    variable_type: VariableType::ProofChainElement {
                        element_type: ProofElementType::Premise,
                        confidence_threshold: 0.8,
                    },
                    required: true,
                    default_value: None,
                    validation: None,
                    description: "Full text of the requirement from proof chain".to_string(),
                },
                TemplateVariable {
                    name: "COMPLIANCE_STATUS".to_string(),
                    variable_type: VariableType::ComplianceStatus {
                        status_type: ComplianceStatusType::Compliant,
                        include_reasoning: true,
                    },
                    required: true,
                    default_value: Some("Unknown".to_string()),
                    validation: None,
                    description: "Current compliance status assessment".to_string(),
                },
                TemplateVariable {
                    name: "CONFIDENCE_SCORE".to_string(),
                    variable_type: VariableType::CalculatedValue {
                        calculation_type: CalculationType::Confidence,
                        source_elements: vec!["proof_chain".to_string()],
                    },
                    required: true,
                    default_value: Some("0.0".to_string()),
                    validation: None,
                    description: "Overall confidence from proof chain analysis".to_string(),
                },
                TemplateVariable {
                    name: "FORMATTED_CITATIONS".to_string(),
                    variable_type: VariableType::CitationReference {
                        citation_format: crate::template_engine::CitationFormat::Academic,
                        include_metadata: true,
                    },
                    required: true,
                    default_value: None,
                    validation: None,
                    description: "Formatted citations with audit trail".to_string(),
                },
            ],
            required_proof_elements: vec![
                ProofElement {
                    id: "proof-1".to_string(),
                    element_type: ProofElementType::Premise,
                    content: "Requirement premise".to_string(),
                    confidence: 0.95,
                    source_references: vec![],
                    created_at: Utc::now(),
                }
            ],
            citation_requirements: CitationRequirements {
                minimum_citations: 2,
                required_source_types: vec!["regulatory".to_string(), "standard".to_string()],
                minimum_confidence: 0.85,
                require_page_numbers: true,
                require_access_dates: true,
            },
            validation_rules: vec![
                TemplateValidationRule {
                    rule_id: "rule-1".to_string(),
                    rule_type: ValidationRuleType::Required,
                    condition: "must_have_citations".to_string(),
                    error_message: "Citations required for compliance template".to_string(),
                    severity: ValidationSeverity::Error,
                }
            ],
            created_at: Utc::now(),
            last_modified: Utc::now(),
        };
        
        let template_type = must_template.template_type.clone();
        self.templates.insert(template_type.clone(), must_template);
        
        // Add metadata
        self.metadata.insert(template_type.clone(), TemplateMetadata {
            id: Uuid::new_v4(),
            name: "MUST Requirement Template".to_string(),
            description: "Template for mandatory requirements analysis with compliance assessment".to_string(),
            version: "1.0.0".to_string(),
            author: "Template Engine System".to_string(),
            created_at: Utc::now(),
            last_modified: Utc::now(),
            tags: vec!["requirement".to_string(), "compliance".to_string(), "mandatory".to_string()],
            complexity_level: ComplexityLevel::Moderate,
            usage_recommendations: vec![
                "Use for analyzing mandatory compliance requirements".to_string(),
                "Requires high-confidence proof chain elements".to_string(),
                "Best for regulatory and standard requirements".to_string(),
            ],
            performance_characteristics: PerformanceCharacteristics {
                expected_generation_time: std::time::Duration::from_millis(500),
                memory_usage_estimate: 1024 * 1024, // 1MB
                cpu_usage_estimate: 0.3,
                accuracy_estimate: 0.95,
                completeness_estimate: 0.92,
            },
        });
        
        // Initialize usage stats
        self.usage_stats.insert(template_type, TemplateUsageStats {
            usage_count: 0,
            avg_response_time: std::time::Duration::from_millis(0),
            success_rate: 1.0,
            satisfaction_score: 0.0,
            last_used: None,
            performance_metrics: PerformanceMetrics {
                avg_processing_time: std::time::Duration::from_millis(0),
                p95_processing_time: std::time::Duration::from_millis(0),
                peak_memory_usage: 0,
                error_rate: 0.0,
                cache_hit_rate: 0.0,
            },
        });
    }
    
    /// Load compliance query templates
    fn load_compliance_templates(&mut self) {
        let compliance_template = ResponseTemplate {
            id: Uuid::new_v4(),
            name: "Regulatory Compliance Template".to_string(),
            template_type: TemplateType::ComplianceQuery {
                compliance_type: ComplianceType::Regulatory,
                scope: ComplianceScope::Full,
            },
            content_structure: ContentStructure {
                introduction: SectionTemplate {
                    name: "Compliance Overview".to_string(),
                    content_template: "## {REGULATION_NAME} Compliance Analysis\n\n**Scope:** {COMPLIANCE_SCOPE}\n**Assessment Date:** {ASSESSMENT_DATE}\n**Overall Status:** {OVERALL_COMPLIANCE_STATUS}\n**Confidence Level:** {CONFIDENCE_LEVEL}".to_string(),
                    variables: vec!["REGULATION_NAME".to_string(), "COMPLIANCE_SCOPE".to_string(), "ASSESSMENT_DATE".to_string(), "OVERALL_COMPLIANCE_STATUS".to_string(), "CONFIDENCE_LEVEL".to_string()],
                    required_elements: vec!["REGULATION_NAME".to_string(), "OVERALL_COMPLIANCE_STATUS".to_string()],
                    order: 1,
                },
                main_sections: vec![
                    SectionTemplate {
                        name: "Compliance Requirements".to_string(),
                        content_template: "### Regulatory Requirements\n\n{REQUIREMENTS_ANALYSIS}\n\n**Critical Requirements:**\n{CRITICAL_REQUIREMENTS}\n\n**Standard Requirements:**\n{STANDARD_REQUIREMENTS}".to_string(),
                        variables: vec!["REQUIREMENTS_ANALYSIS".to_string(), "CRITICAL_REQUIREMENTS".to_string(), "STANDARD_REQUIREMENTS".to_string()],
                        required_elements: vec!["REQUIREMENTS_ANALYSIS".to_string()],
                        order: 2,
                    },
                    SectionTemplate {
                        name: "Gap Analysis".to_string(),
                        content_template: "### Compliance Gap Analysis\n\n**Identified Gaps:**\n{COMPLIANCE_GAPS}\n\n**Risk Assessment:**\n{RISK_ASSESSMENT}\n\n**Priority Actions:**\n{PRIORITY_ACTIONS}".to_string(),
                        variables: vec!["COMPLIANCE_GAPS".to_string(), "RISK_ASSESSMENT".to_string(), "PRIORITY_ACTIONS".to_string()],
                        required_elements: vec!["COMPLIANCE_GAPS".to_string()],
                        order: 3,
                    },
                    SectionTemplate {
                        name: "Implementation Roadmap".to_string(),
                        content_template: "### Implementation Roadmap\n\n**Phase 1 (Immediate):**\n{PHASE_1_ACTIONS}\n\n**Phase 2 (Short-term):**\n{PHASE_2_ACTIONS}\n\n**Phase 3 (Long-term):**\n{PHASE_3_ACTIONS}".to_string(),
                        variables: vec!["PHASE_1_ACTIONS".to_string(), "PHASE_2_ACTIONS".to_string(), "PHASE_3_ACTIONS".to_string()],
                        required_elements: vec!["PHASE_1_ACTIONS".to_string()],
                        order: 4,
                    },
                ],
                citations_section: SectionTemplate {
                    name: "Regulatory References".to_string(),
                    content_template: "### Regulatory References\n\n{FORMATTED_CITATIONS}\n\n**Regulatory Sources:**\n{REGULATORY_SOURCES}".to_string(),
                    variables: vec!["FORMATTED_CITATIONS".to_string(), "REGULATORY_SOURCES".to_string()],
                    required_elements: vec!["FORMATTED_CITATIONS".to_string()],
                    order: 5,
                },
                conclusion: SectionTemplate {
                    name: "Compliance Summary".to_string(),
                    content_template: "### Compliance Summary\n\nThis compliance analysis provides a comprehensive assessment with {OVERALL_CONFIDENCE} confidence based on regulatory interpretation and organizational context.\n\n**Next Steps:**\n{NEXT_STEPS}".to_string(),
                    variables: vec!["OVERALL_CONFIDENCE".to_string(), "NEXT_STEPS".to_string()],
                    required_elements: vec!["OVERALL_CONFIDENCE".to_string()],
                    order: 6,
                },
                audit_trail_section: SectionTemplate {
                    name: "Assessment Audit Trail".to_string(),
                    content_template: "---\n**Assessment Audit Trail**\n- Analysis Generated: {GENERATION_TIMESTAMP}\n- Regulatory Framework: {REGULATORY_FRAMEWORK}\n- Assessment Method: {ASSESSMENT_METHOD}\n- Data Sources: {DATA_SOURCES_COUNT}\n- Validation: {VALIDATION_STATUS}".to_string(),
                    variables: vec!["GENERATION_TIMESTAMP".to_string(), "REGULATORY_FRAMEWORK".to_string(), "ASSESSMENT_METHOD".to_string(), "DATA_SOURCES_COUNT".to_string(), "VALIDATION_STATUS".to_string()],
                    required_elements: vec!["GENERATION_TIMESTAMP".to_string()],
                    order: 7,
                },
            },
            variables: vec![
                TemplateVariable {
                    name: "REGULATION_NAME".to_string(),
                    variable_type: VariableType::EntityReference {
                        entity_type: EntityType::Standard,
                        include_relationships: false,
                    },
                    required: true,
                    default_value: None,
                    validation: None,
                    description: "Name of the regulation being assessed".to_string(),
                },
                TemplateVariable {
                    name: "OVERALL_COMPLIANCE_STATUS".to_string(),
                    variable_type: VariableType::ComplianceStatus {
                        status_type: ComplianceStatusType::Partial,
                        include_reasoning: true,
                    },
                    required: true,
                    default_value: Some("Unknown".to_string()),
                    validation: None,
                    description: "Overall compliance status assessment".to_string(),
                },
                TemplateVariable {
                    name: "COMPLIANCE_GAPS".to_string(),
                    variable_type: VariableType::CalculatedValue {
                        calculation_type: CalculationType::Gap,
                        source_elements: vec!["requirements".to_string(), "current_state".to_string()],
                    },
                    required: true,
                    default_value: None,
                    validation: None,
                    description: "Identified compliance gaps from analysis".to_string(),
                },
            ],
            required_proof_elements: vec![
                ProofElement {
                    id: "compliance-proof-1".to_string(),
                    element_type: ProofElementType::Evidence,
                    content: "Compliance evidence".to_string(),
                    confidence: 0.92,
                    source_references: vec![],
                    created_at: Utc::now(),
                }
            ],
            citation_requirements: CitationRequirements {
                minimum_citations: 3,
                required_source_types: vec!["compliance".to_string(), "audit".to_string()],
                minimum_confidence: 0.90,
                require_page_numbers: true,
                require_access_dates: true,
            },
            validation_rules: vec![
                TemplateValidationRule {
                    rule_id: "compliance-rule-1".to_string(),
                    rule_type: ValidationRuleType::Required,
                    condition: "compliance_status_required".to_string(),
                    error_message: "Compliance status required for audit template".to_string(),
                    severity: ValidationSeverity::Critical,
                }
            ],
            created_at: Utc::now(),
            last_modified: Utc::now(),
        };
        
        let template_type = compliance_template.template_type.clone();
        self.templates.insert(template_type.clone(), compliance_template);
        
        // Add metadata for compliance template
        self.metadata.insert(template_type.clone(), TemplateMetadata {
            id: Uuid::new_v4(),
            name: "Regulatory Compliance Template".to_string(),
            description: "Comprehensive compliance assessment template for regulatory frameworks".to_string(),
            version: "1.0.0".to_string(),
            author: "Template Engine System".to_string(),
            created_at: Utc::now(),
            last_modified: Utc::now(),
            tags: vec!["compliance".to_string(), "regulatory".to_string(), "assessment".to_string()],
            complexity_level: ComplexityLevel::Complex,
            usage_recommendations: vec![
                "Use for full regulatory compliance assessments".to_string(),
                "Requires comprehensive organizational data".to_string(),
                "Best for formal compliance reporting".to_string(),
            ],
            performance_characteristics: PerformanceCharacteristics {
                expected_generation_time: std::time::Duration::from_millis(800),
                memory_usage_estimate: 2 * 1024 * 1024, // 2MB
                cpu_usage_estimate: 0.5,
                accuracy_estimate: 0.92,
                completeness_estimate: 0.95,
            },
        });
        
        self.usage_stats.insert(template_type, TemplateUsageStats {
            usage_count: 0,
            avg_response_time: std::time::Duration::from_millis(0),
            success_rate: 1.0,
            satisfaction_score: 0.0,
            last_used: None,
            performance_metrics: PerformanceMetrics {
                avg_processing_time: std::time::Duration::from_millis(0),
                p95_processing_time: std::time::Duration::from_millis(0),
                peak_memory_usage: 0,
                error_rate: 0.0,
                cache_hit_rate: 0.0,
            },
        });
    }
    
    /// Load relationship query templates
    fn load_relationship_templates(&mut self) {
        let relationship_template = ResponseTemplate {
            id: Uuid::new_v4(),
            name: "Entity Relationship Template".to_string(),
            template_type: TemplateType::RelationshipQuery {
                relationship_type: RelationshipType::References,
                entity_types: vec![EntityType::Standard, EntityType::Requirement],
            },
            content_structure: ContentStructure {
                introduction: SectionTemplate {
                    name: "Relationship Analysis".to_string(),
                    content_template: "## {RELATIONSHIP_TYPE} Analysis: {PRIMARY_ENTITY} ↔ {SECONDARY_ENTITY}\n\n**Relationship Strength:** {RELATIONSHIP_STRENGTH}\n**Analysis Confidence:** {ANALYSIS_CONFIDENCE}".to_string(),
                    variables: vec!["RELATIONSHIP_TYPE".to_string(), "PRIMARY_ENTITY".to_string(), "SECONDARY_ENTITY".to_string(), "RELATIONSHIP_STRENGTH".to_string(), "ANALYSIS_CONFIDENCE".to_string()],
                    required_elements: vec!["RELATIONSHIP_TYPE".to_string(), "PRIMARY_ENTITY".to_string()],
                    order: 1,
                },
                main_sections: vec![
                    SectionTemplate {
                        name: "Direct Relationships".to_string(),
                        content_template: "### Direct Relationships\n\n{DIRECT_RELATIONSHIPS}\n\n**Key Connections:**\n{KEY_CONNECTIONS}".to_string(),
                        variables: vec!["DIRECT_RELATIONSHIPS".to_string(), "KEY_CONNECTIONS".to_string()],
                        required_elements: vec!["DIRECT_RELATIONSHIPS".to_string()],
                        order: 2,
                    },
                    SectionTemplate {
                        name: "Indirect Relationships".to_string(),
                        content_template: "### Indirect Relationships\n\n{INDIRECT_RELATIONSHIPS}\n\n**Relationship Chain:**\n{RELATIONSHIP_CHAIN}".to_string(),
                        variables: vec!["INDIRECT_RELATIONSHIPS".to_string(), "RELATIONSHIP_CHAIN".to_string()],
                        required_elements: vec!["INDIRECT_RELATIONSHIPS".to_string()],
                        order: 3,
                    },
                ],
                citations_section: SectionTemplate {
                    name: "Relationship Sources".to_string(),
                    content_template: "### Relationship Sources\n\n{FORMATTED_CITATIONS}".to_string(),
                    variables: vec!["FORMATTED_CITATIONS".to_string()],
                    required_elements: vec!["FORMATTED_CITATIONS".to_string()],
                    order: 4,
                },
                conclusion: SectionTemplate {
                    name: "Relationship Summary".to_string(),
                    content_template: "### Relationship Summary\n\nAnalysis completed with {OVERALL_CONFIDENCE} confidence. Relationships identified through symbolic reasoning and knowledge graph traversal.".to_string(),
                    variables: vec!["OVERALL_CONFIDENCE".to_string()],
                    required_elements: vec!["OVERALL_CONFIDENCE".to_string()],
                    order: 5,
                },
                audit_trail_section: SectionTemplate {
                    name: "Analysis Audit Trail".to_string(),
                    content_template: "---\n**Relationship Analysis Audit Trail**\n- Generated: {GENERATION_TIMESTAMP}\n- Method: Graph Traversal + Symbolic Reasoning\n- Depth: {ANALYSIS_DEPTH}\n- Entities Analyzed: {ENTITIES_COUNT}".to_string(),
                    variables: vec!["GENERATION_TIMESTAMP".to_string(), "ANALYSIS_DEPTH".to_string(), "ENTITIES_COUNT".to_string()],
                    required_elements: vec!["GENERATION_TIMESTAMP".to_string()],
                    order: 6,
                },
            },
            variables: vec![
                TemplateVariable {
                    name: "PRIMARY_ENTITY".to_string(),
                    variable_type: VariableType::EntityReference {
                        entity_type: EntityType::Standard,
                        include_relationships: true,
                    },
                    required: true,
                    default_value: None,
                    validation: None,
                    description: "Primary entity in the relationship".to_string(),
                },
                TemplateVariable {
                    name: "RELATIONSHIP_TYPE".to_string(),
                    variable_type: VariableType::Text,
                    required: true,
                    default_value: Some("References".to_string()),
                    validation: None,
                    description: "Type of relationship being analyzed".to_string(),
                },
                TemplateVariable {
                    name: "DIRECT_RELATIONSHIPS".to_string(),
                    variable_type: VariableType::CalculatedValue {
                        calculation_type: CalculationType::Coverage,
                        source_elements: vec!["graph_traversal".to_string()],
                    },
                    required: true,
                    default_value: None,
                    validation: None,
                    description: "Direct relationships identified from graph analysis".to_string(),
                },
            ],
            required_proof_elements: vec![
                ProofElement {
                    id: "relationship-proof-1".to_string(),
                    element_type: ProofElementType::Rule,
                    content: "Relationship rule".to_string(),
                    confidence: 0.88,
                    source_references: vec![],
                    created_at: Utc::now(),
                }
            ],
            citation_requirements: CitationRequirements {
                minimum_citations: 1,
                required_source_types: vec!["technical".to_string()],
                minimum_confidence: 0.80,
                require_page_numbers: false,
                require_access_dates: true,
            },
            validation_rules: vec![
                TemplateValidationRule {
                    rule_id: "relationship-rule-1".to_string(),
                    rule_type: ValidationRuleType::Format,
                    condition: "valid_relationship_format".to_string(),
                    error_message: "Relationship must follow graph format".to_string(),
                    severity: ValidationSeverity::Warning,
                }
            ],
            created_at: Utc::now(),
            last_modified: Utc::now(),
        };
        
        let template_type = relationship_template.template_type.clone();
        self.templates.insert(template_type.clone(), relationship_template);
        
        // Add metadata and stats...
        self.metadata.insert(template_type.clone(), TemplateMetadata {
            id: Uuid::new_v4(),
            name: "Entity Relationship Template".to_string(),
            description: "Template for analyzing relationships between entities".to_string(),
            version: "1.0.0".to_string(),
            author: "Template Engine System".to_string(),
            created_at: Utc::now(),
            last_modified: Utc::now(),
            tags: vec!["relationship".to_string(), "entity".to_string(), "graph".to_string()],
            complexity_level: ComplexityLevel::Moderate,
            usage_recommendations: vec![
                "Use for entity relationship discovery".to_string(),
                "Requires knowledge graph data".to_string(),
                "Best for understanding entity connections".to_string(),
            ],
            performance_characteristics: PerformanceCharacteristics {
                expected_generation_time: std::time::Duration::from_millis(600),
                memory_usage_estimate: 1536 * 1024, // 1.5MB
                cpu_usage_estimate: 0.4,
                accuracy_estimate: 0.88,
                completeness_estimate: 0.85,
            },
        });
        
        self.usage_stats.insert(template_type, TemplateUsageStats {
            usage_count: 0,
            avg_response_time: std::time::Duration::from_millis(0),
            success_rate: 1.0,
            satisfaction_score: 0.0,
            last_used: None,
            performance_metrics: PerformanceMetrics {
                avg_processing_time: std::time::Duration::from_millis(0),
                p95_processing_time: std::time::Duration::from_millis(0),
                peak_memory_usage: 0,
                error_rate: 0.0,
                cache_hit_rate: 0.0,
            },
        });
    }
    
    /// Load factual query templates
    fn load_factual_templates(&mut self) {
        // Implementation for factual templates...
    }
    
    /// Load analytical query templates
    fn load_analytical_templates(&mut self) {
        // Implementation for analytical templates...
    }
    
    /// Get template by type
    pub fn get_template(&self, template_type: &TemplateType) -> Option<&ResponseTemplate> {
        self.templates.get(template_type)
    }
    
    /// Get template metadata
    pub fn get_metadata(&self, template_type: &TemplateType) -> Option<&TemplateMetadata> {
        self.metadata.get(template_type)
    }
    
    /// Get template usage statistics
    pub fn get_usage_stats(&self, template_type: &TemplateType) -> Option<&TemplateUsageStats> {
        self.usage_stats.get(template_type)
    }
    
    /// Update usage statistics
    pub fn update_usage_stats(
        &mut self,
        template_type: &TemplateType,
        response_time: std::time::Duration,
        success: bool,
    ) {
        if let Some(stats) = self.usage_stats.get_mut(template_type) {
            stats.usage_count += 1;
            stats.last_used = Some(Utc::now());
            
            // Update average response time
            let total_time = stats.avg_response_time.as_millis() as u64 * (stats.usage_count - 1);
            let new_total = total_time + response_time.as_millis() as u64;
            stats.avg_response_time = std::time::Duration::from_millis(new_total / stats.usage_count);
            
            // Update success rate
            let total_successes = (stats.success_rate * (stats.usage_count - 1) as f64).round() as u64;
            let new_successes = total_successes + if success { 1 } else { 0 };
            stats.success_rate = new_successes as f64 / stats.usage_count as f64;
        }
    }
    
    /// List all available template types
    pub fn list_template_types(&self) -> Vec<TemplateType> {
        self.templates.keys().cloned().collect()
    }
    
    /// Get template recommendations based on query characteristics
    pub fn recommend_template(
        &self,
        query_intent: &QueryIntent,
        entities: &[String],
        complexity: ComplexityLevel,
    ) -> Vec<TemplateType> {
        let mut recommendations = Vec::new();
        
        for (template_type, metadata) in &self.metadata {
            // Score template based on compatibility
            let mut score = 0.0;
            
            // Intent matching
            match (query_intent, template_type) {
                (QueryIntent::Compliance, TemplateType::RequirementQuery { .. }) => score += 0.8,
                (QueryIntent::Compliance, TemplateType::ComplianceQuery { .. }) => score += 1.0,
                (QueryIntent::Relationship, TemplateType::RelationshipQuery { .. }) => score += 1.0,
                (QueryIntent::Factual, TemplateType::FactualQuery { .. }) => score += 1.0,
                (QueryIntent::Analytical, TemplateType::AnalyticalQuery { .. }) => score += 1.0,
                _ => score += 0.1,
            }
            
            // Complexity matching
            if metadata.complexity_level == complexity {
                score += 0.2;
            }
            
            // Entity count consideration
            if entities.len() > 5 && matches!(template_type, TemplateType::RelationshipQuery { .. }) {
                score += 0.3;
            }
            
            if score >= 0.5 {
                recommendations.push(template_type.clone());
            }
        }
        
        // Sort by usage statistics (prefer well-tested templates)
        recommendations.sort_by(|a, b| {
            let stats_a = self.usage_stats.get(a).map(|s| s.success_rate).unwrap_or(0.0);
            let stats_b = self.usage_stats.get(b).map(|s| s.success_rate).unwrap_or(0.0);
            stats_b.partial_cmp(&stats_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        recommendations
    }
}

impl Default for TemplateLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_library_creation() {
        let library = TemplateLibrary::new();
        assert!(!library.templates.is_empty());
        assert!(!library.metadata.is_empty());
        assert!(!library.usage_stats.is_empty());
    }

    #[test]
    fn test_template_recommendation() {
        let library = TemplateLibrary::new();
        
        let recommendations = library.recommend_template(
            &QueryIntent::Compliance,
            &vec!["PCI DSS".to_string()],
            ComplexityLevel::Moderate,
        );
        
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|t| matches!(t, TemplateType::RequirementQuery { .. })));
    }

    #[test]
    fn test_usage_stats_update() {
        let mut library = TemplateLibrary::new();
        let template_type = library.list_template_types()[0].clone();
        
        library.update_usage_stats(
            &template_type,
            std::time::Duration::from_millis(500),
            true,
        );
        
        let stats = library.get_usage_stats(&template_type).unwrap();
        assert_eq!(stats.usage_count, 1);
        assert_eq!(stats.success_rate, 1.0);
    }
}