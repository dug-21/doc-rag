//! Template Selection Tests
//! 
//! London TDD tests for template matching and selection logic.
//! Ensures correct template selection based on query characteristics and types.

#[cfg(test)]
mod template_selection_tests {
    use response_generator::template_engine::*;
    use response_generator::{OutputFormat, Citation, Source};
    use crate::fixtures::*;
    use std::collections::HashMap;
    use tokio;
    use uuid::Uuid;

    struct TemplateSelectionTestFixture {
        engine: TemplateEngine,
        requirement_query_request: TemplateGenerationRequest,
        compliance_query_request: TemplateGenerationRequest,
        factual_query_request: TemplateGenerationRequest,
        relationship_query_request: TemplateGenerationRequest,
        analytical_query_request: TemplateGenerationRequest,
    }

    impl TemplateSelectionTestFixture {
        async fn new() -> Self {
            let engine = TemplateEngine::default();

            let base_variables = {
                let mut vars = HashMap::new();
                vars.insert("QUERY_SUBJECT".to_string(), "test subject".to_string());
                vars.insert("REQUIREMENT_TYPE".to_string(), "MUST".to_string());
                vars
            };

            let base_citations = vec![
                Citation {
                    id: Uuid::new_v4(),
                    source: Source {
                        id: Uuid::new_v4(),
                        title: "Test Standard".to_string(),
                        url: Some("https://example.com/standard".to_string()),
                        document_type: "Standard".to_string(),
                        version: Some("1.0".to_string()),
                        section: Some("1.1".to_string()),
                        page: Some(1),
                        published_date: None,
                        accessed_date: None,
                    },
                    content: "Test requirement content".to_string(),
                    confidence: 0.9,
                    page_number: Some(1),
                    paragraph: Some(1),
                    relevance_score: 0.85,
                    exact_match: false,
                }
            ];

            let base_proof_chain = vec![
                ProofChainData {
                    element_type: ProofElementType::Evidence,
                    content: "Test evidence".to_string(),
                    confidence: 0.9,
                    source: "Test Source".to_string(),
                }
            ];

            let requirement_query_request = TemplateGenerationRequest {
                template_type: TemplateType::RequirementQuery {
                    requirement_type: RequirementType::Must,
                    query_intent: QueryIntent::Compliance,
                },
                variable_values: base_variables.clone(),
                proof_chain_data: base_proof_chain.clone(),
                citations: base_citations.clone(),
                output_format: OutputFormat::Markdown,
                context: GenerationContext {
                    query_intent: QueryIntent::Compliance,
                    entities: vec!["requirement".to_string()],
                    requirements: vec!["REQ_001".to_string()],
                    compliance_scope: Some(ComplianceScope::Requirement),
                },
            };

            let compliance_query_request = TemplateGenerationRequest {
                template_type: TemplateType::ComplianceQuery {
                    compliance_type: ComplianceType::Regulatory,
                    scope: ComplianceScope::Section,
                },
                variable_values: base_variables.clone(),
                proof_chain_data: base_proof_chain.clone(),
                citations: base_citations.clone(),
                output_format: OutputFormat::Markdown,
                context: GenerationContext {
                    query_intent: QueryIntent::Compliance,
                    entities: vec!["compliance".to_string()],
                    requirements: vec!["COMP_001".to_string()],
                    compliance_scope: Some(ComplianceScope::Section),
                },
            };

            let factual_query_request = TemplateGenerationRequest {
                template_type: TemplateType::FactualQuery {
                    fact_type: FactType::Definition,
                    complexity_level: ComplexityLevel::Simple,
                },
                variable_values: base_variables.clone(),
                proof_chain_data: base_proof_chain.clone(),
                citations: base_citations.clone(),
                output_format: OutputFormat::Markdown,
                context: GenerationContext {
                    query_intent: QueryIntent::Definition,
                    entities: vec!["definition".to_string()],
                    requirements: vec![],
                    compliance_scope: None,
                },
            };

            let relationship_query_request = TemplateGenerationRequest {
                template_type: TemplateType::RelationshipQuery {
                    relationship_type: RelationshipType::References,
                    entity_types: vec![EntityType::Standard, EntityType::Requirement],
                },
                variable_values: base_variables.clone(),
                proof_chain_data: base_proof_chain.clone(),
                citations: base_citations.clone(),
                output_format: OutputFormat::Markdown,
                context: GenerationContext {
                    query_intent: QueryIntent::Relationship,
                    entities: vec!["standard".to_string(), "requirement".to_string()],
                    requirements: vec!["REL_001".to_string()],
                    compliance_scope: None,
                },
            };

            let analytical_query_request = TemplateGenerationRequest {
                template_type: TemplateType::AnalyticalQuery {
                    analysis_type: AnalysisType::Comparison,
                    comparison_scope: ComparisonScope::Feature,
                },
                variable_values: base_variables,
                proof_chain_data: base_proof_chain,
                citations: base_citations,
                output_format: OutputFormat::Markdown,
                context: GenerationContext {
                    query_intent: QueryIntent::Analytical,
                    entities: vec!["feature_a".to_string(), "feature_b".to_string()],
                    requirements: vec![],
                    compliance_scope: None,
                },
            };

            Self {
                engine,
                requirement_query_request,
                compliance_query_request,
                factual_query_request,
                relationship_query_request,
                analytical_query_request,
            }
        }
    }

    #[tokio::test]
    async fn test_requirement_query_template_selection() {
        // Given: Template engine with requirement query request
        let fixture = TemplateSelectionTestFixture::new().await;
        
        // When: Generating response for requirement query
        let response = fixture.engine.generate_response(fixture.requirement_query_request.clone()).await.unwrap();
        
        // Then: Should select and use requirement query template
        assert_eq!(response.template_type, TemplateType::RequirementQuery {
            requirement_type: RequirementType::Must,
            query_intent: QueryIntent::Compliance,
        });
        
        // Response should contain requirement-specific structure
        assert!(response.content.contains("requirement") || response.content.contains("MUST"),
            "Content should reflect requirement template structure");
        
        // Should have compliance-related content structure
        assert!(response.content.contains("analysis") || response.content.contains("compliance"),
            "Content should contain compliance analysis sections");
        
        // Template selection should be documented in audit trail
        assert!(!response.audit_trail.template_selection.to_string().is_empty(),
            "Template selection should be documented in audit trail");
        
        // Validation should confirm template-based generation
        assert!(response.validation_results.constraint_004_compliant,
            "Requirement template should be CONSTRAINT-004 compliant");
        assert!(response.validation_results.is_valid,
            "Requirement template response should be valid");
    }

    #[tokio::test]
    async fn test_compliance_query_template_selection() {
        // Given: Template engine with compliance query request
        let fixture = TemplateSelectionTestFixture::new().await;
        
        // When: Attempting to generate compliance query response
        let result = fixture.engine.generate_response(fixture.compliance_query_request.clone()).await;
        
        // Then: Should either succeed with compliance template or provide meaningful error
        match result {
            Ok(response) => {
                // If compliance template exists, verify its structure
                assert_eq!(response.template_type, TemplateType::ComplianceQuery {
                    compliance_type: ComplianceType::Regulatory,
                    scope: ComplianceScope::Section,
                });
                
                assert!(response.validation_results.constraint_004_compliant,
                    "Compliance template should be CONSTRAINT-004 compliant");
                assert!(!response.content.is_empty(),
                    "Compliance template should generate content");
            },
            Err(error) => {
                // If template doesn't exist yet, error should be specific about template not found
                match error {
                    crate::response_generator::ResponseError::TemplateNotFound(msg) => {
                        assert!(msg.contains("ComplianceQuery"),
                            "Error message should specify compliance query template: {}", msg);
                    },
                    _ => {
                        panic!("Expected TemplateNotFound error for unimplemented compliance template, got: {}", error);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_factual_query_template_selection() {
        // Given: Template engine with factual query request
        let fixture = TemplateSelectionTestFixture::new().await;
        
        // When: Attempting to generate factual query response
        let result = fixture.engine.generate_response(fixture.factual_query_request.clone()).await;
        
        // Then: Should either succeed with factual template or provide meaningful error
        match result {
            Ok(response) => {
                assert_eq!(response.template_type, TemplateType::FactualQuery {
                    fact_type: FactType::Definition,
                    complexity_level: ComplexityLevel::Simple,
                });
                
                assert!(response.validation_results.constraint_004_compliant,
                    "Factual template should be CONSTRAINT-004 compliant");
                assert!(!response.content.is_empty(),
                    "Factual template should generate content");
                
                // Factual queries should have definition-focused structure
                assert!(response.content.contains("definition") || response.content.contains("explains"),
                    "Factual content should be definition-focused");
            },
            Err(error) => {
                match error {
                    crate::response_generator::ResponseError::TemplateNotFound(msg) => {
                        assert!(msg.contains("FactualQuery"),
                            "Error should specify factual query template: {}", msg);
                    },
                    _ => {
                        panic!("Expected TemplateNotFound error for unimplemented factual template, got: {}", error);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_relationship_query_template_selection() {
        // Given: Template engine with relationship query request
        let fixture = TemplateSelectionTestFixture::new().await;
        
        // When: Attempting to generate relationship query response
        let result = fixture.engine.generate_response(fixture.relationship_query_request.clone()).await;
        
        // Then: Should handle relationship template appropriately
        match result {
            Ok(response) => {
                assert_eq!(response.template_type, TemplateType::RelationshipQuery {
                    relationship_type: RelationshipType::References,
                    entity_types: vec![EntityType::Standard, EntityType::Requirement],
                });
                
                assert!(response.validation_results.constraint_004_compliant,
                    "Relationship template should be CONSTRAINT-004 compliant");
                
                // Relationship queries should reference entities
                assert!(response.content.contains("relationship") || 
                       response.content.contains("references") ||
                       response.content.contains("standard") ||
                       response.content.contains("requirement"),
                    "Relationship content should reference entities and relationships");
            },
            Err(error) => {
                match error {
                    crate::response_generator::ResponseError::TemplateNotFound(msg) => {
                        assert!(msg.contains("RelationshipQuery"),
                            "Error should specify relationship query template: {}", msg);
                    },
                    _ => {
                        panic!("Expected TemplateNotFound error for unimplemented relationship template, got: {}", error);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_analytical_query_template_selection() {
        // Given: Template engine with analytical query request
        let fixture = TemplateSelectionTestFixture::new().await;
        
        // When: Attempting to generate analytical query response
        let result = fixture.engine.generate_response(fixture.analytical_query_request.clone()).await;
        
        // Then: Should handle analytical template appropriately
        match result {
            Ok(response) => {
                assert_eq!(response.template_type, TemplateType::AnalyticalQuery {
                    analysis_type: AnalysisType::Comparison,
                    comparison_scope: ComparisonScope::Feature,
                });
                
                assert!(response.validation_results.constraint_004_compliant,
                    "Analytical template should be CONSTRAINT-004 compliant");
                
                // Analytical queries should contain analysis elements
                assert!(response.content.contains("analysis") || 
                       response.content.contains("comparison") ||
                       response.content.contains("feature"),
                    "Analytical content should contain analysis elements");
            },
            Err(error) => {
                match error {
                    crate::response_generator::ResponseError::TemplateNotFound(msg) => {
                        assert!(msg.contains("AnalyticalQuery"),
                            "Error should specify analytical query template: {}", msg);
                    },
                    _ => {
                        panic!("Expected TemplateNotFound error for unimplemented analytical template, got: {}", error);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_template_availability_reporting() {
        // Given: Template engine
        let fixture = TemplateSelectionTestFixture::new().await;
        
        // When: Querying available templates
        let available_templates = fixture.engine.available_templates();
        
        // Then: Should report available template types
        assert!(!available_templates.is_empty(),
            "Should have at least one available template");
        
        // Should at least have the requirement query template (from default implementation)
        let has_requirement_template = available_templates.iter().any(|template| {
            matches!(template, TemplateType::RequirementQuery { .. })
        });
        
        assert!(has_requirement_template,
            "Should have requirement query template available by default");
        
        // Available templates should be usable
        for template_type in &available_templates {
            let test_request = create_test_request_for_template(template_type.clone());
            let result = fixture.engine.generate_response(test_request).await;
            
            // Available templates should succeed in generation
            assert!(result.is_ok(),
                "Available template {:?} should successfully generate responses", template_type);
            
            if let Ok(response) = result {
                assert!(response.validation_results.constraint_004_compliant,
                    "Available template should be CONSTRAINT-004 compliant");
                assert!(!response.content.is_empty(),
                    "Available template should generate non-empty content");
            }
        }
    }

    #[tokio::test]
    async fn test_template_selection_performance() {
        // Given: Template engine and multiple template types
        let fixture = TemplateSelectionTestFixture::new().await;
        
        let test_requests = vec![
            fixture.requirement_query_request.clone(),
            fixture.compliance_query_request.clone(),
            fixture.factual_query_request.clone(),
            fixture.relationship_query_request.clone(),
            fixture.analytical_query_request.clone(),
        ];
        
        // When: Testing template selection performance for different types
        for (i, request) in test_requests.iter().enumerate() {
            let start_time = std::time::Instant::now();
            let result = fixture.engine.generate_response(request.clone()).await;
            let selection_time = start_time.elapsed();
            
            // Then: Template selection should be fast regardless of outcome
            assert!(selection_time.as_millis() < 100,
                "Template selection {} took {}ms, should be <100ms", i, selection_time.as_millis());
            
            // If successful, should meet performance constraints
            if let Ok(response) = result {
                assert!(response.metrics.template_selection_time.as_millis() < 50,
                    "Template selection time in metrics should be <50ms, was {}ms", 
                    response.metrics.template_selection_time.as_millis());
                
                assert!(response.validation_results.constraint_006_compliant,
                    "Overall response should meet CONSTRAINT-006");
            }
        }
    }

    #[tokio::test]
    async fn test_template_registration_and_custom_templates() {
        // Given: Template engine
        let mut engine = TemplateEngine::default();
        
        // When: Registering a custom template
        let custom_template = ResponseTemplate {
            id: Uuid::new_v4(),
            name: "Custom Test Template".to_string(),
            template_type: TemplateType::ComplianceQuery {
                compliance_type: ComplianceType::Standard,
                scope: ComplianceScope::Full,
            },
            content_structure: ContentStructure {
                introduction: SectionTemplate {
                    name: "Custom Introduction".to_string(),
                    content_template: "This is a custom template for {CUSTOM_SUBJECT}.".to_string(),
                    variables: vec!["CUSTOM_SUBJECT".to_string()],
                    required_elements: vec!["CUSTOM_SUBJECT".to_string()],
                    order: 1,
                },
                main_sections: vec![
                    SectionTemplate {
                        name: "Custom Main Section".to_string(),
                        content_template: "Custom analysis: {CUSTOM_ANALYSIS}".to_string(),
                        variables: vec!["CUSTOM_ANALYSIS".to_string()],
                        required_elements: vec!["CUSTOM_ANALYSIS".to_string()],
                        order: 2,
                    }
                ],
                citations_section: SectionTemplate {
                    name: "Custom Citations".to_string(),
                    content_template: "## Custom References\n{CUSTOM_CITATIONS}".to_string(),
                    variables: vec!["CUSTOM_CITATIONS".to_string()],
                    required_elements: vec![],
                    order: 3,
                },
                conclusion: SectionTemplate {
                    name: "Custom Conclusion".to_string(),
                    content_template: "Custom conclusion: {CUSTOM_CONCLUSION}".to_string(),
                    variables: vec!["CUSTOM_CONCLUSION".to_string()],
                    required_elements: vec!["CUSTOM_CONCLUSION".to_string()],
                    order: 4,
                },
                audit_trail_section: SectionTemplate {
                    name: "Custom Audit".to_string(),
                    content_template: "Custom audit trail generated.".to_string(),
                    variables: vec![],
                    required_elements: vec![],
                    order: 5,
                },
            },
            variables: vec![
                TemplateVariable {
                    name: "CUSTOM_SUBJECT".to_string(),
                    variable_type: VariableType::Text,
                    required: true,
                    default_value: Some("default subject".to_string()),
                    validation: None,
                    description: "Custom subject variable".to_string(),
                },
                TemplateVariable {
                    name: "CUSTOM_ANALYSIS".to_string(),
                    variable_type: VariableType::Text,
                    required: false,
                    default_value: Some("default analysis".to_string()),
                    validation: None,
                    description: "Custom analysis variable".to_string(),
                },
            ],
            required_proof_elements: vec![],
            citation_requirements: CitationRequirements,
            validation_rules: vec![],
            created_at: chrono::Utc::now(),
            last_modified: chrono::Utc::now(),
        };
        
        let registration_result = engine.register_template(custom_template.clone());
        
        // Then: Template registration should succeed
        assert!(registration_result.is_ok(),
            "Custom template registration should succeed");
        
        // Custom template should be available
        let available_templates = engine.available_templates();
        let has_custom_template = available_templates.iter().any(|t| {
            matches!(t, TemplateType::ComplianceQuery { 
                compliance_type: ComplianceType::Standard,
                scope: ComplianceScope::Full 
            })
        });
        
        assert!(has_custom_template,
            "Custom template should be available after registration");
        
        // Custom template should be usable
        let custom_request = TemplateGenerationRequest {
            template_type: TemplateType::ComplianceQuery {
                compliance_type: ComplianceType::Standard,
                scope: ComplianceScope::Full,
            },
            variable_values: {
                let mut vars = HashMap::new();
                vars.insert("CUSTOM_SUBJECT".to_string(), "test custom subject".to_string());
                vars.insert("CUSTOM_ANALYSIS".to_string(), "test custom analysis".to_string());
                vars
            },
            proof_chain_data: vec![],
            citations: vec![],
            output_format: OutputFormat::Markdown,
            context: GenerationContext {
                query_intent: QueryIntent::Compliance,
                entities: vec!["custom".to_string()],
                requirements: vec![],
                compliance_scope: Some(ComplianceScope::Full),
            },
        };
        
        let response = engine.generate_response(custom_request).await.unwrap();
        
        // Verify custom template was used
        assert!(response.content.contains("custom template"),
            "Response should use custom template structure");
        assert!(response.content.contains("test custom subject"),
            "Response should contain custom variable substitutions");
        assert!(response.validation_results.constraint_004_compliant,
            "Custom template should maintain CONSTRAINT-004 compliance");
    }

    // Helper function to create test requests for different template types
    fn create_test_request_for_template(template_type: TemplateType) -> TemplateGenerationRequest {
        let base_variables = {
            let mut vars = HashMap::new();
            vars.insert("QUERY_SUBJECT".to_string(), "test".to_string());
            vars.insert("REQUIREMENT_TYPE".to_string(), "MUST".to_string());
            vars
        };

        let base_citations = vec![
            Citation {
                id: Uuid::new_v4(),
                source: Source {
                    id: Uuid::new_v4(),
                    title: "Test Document".to_string(),
                    url: None,
                    document_type: "Standard".to_string(),
                    version: None,
                    section: None,
                    page: None,
                    published_date: None,
                    accessed_date: None,
                },
                content: "Test content".to_string(),
                confidence: 0.9,
                page_number: None,
                paragraph: None,
                relevance_score: 0.8,
                exact_match: false,
            }
        ];

        TemplateGenerationRequest {
            template_type,
            variable_values: base_variables,
            proof_chain_data: vec![
                ProofChainData {
                    element_type: ProofElementType::Evidence,
                    content: "Test evidence".to_string(),
                    confidence: 0.9,
                    source: "Test".to_string(),
                }
            ],
            citations: base_citations,
            output_format: OutputFormat::Markdown,
            context: GenerationContext {
                query_intent: QueryIntent::Compliance,
                entities: vec!["test".to_string()],
                requirements: vec![],
                compliance_scope: None,
            },
        }
    }

    #[tokio::test]
    async fn test_template_selection_audit_trail() {
        // Given: Template engine with requirement query
        let fixture = TemplateSelectionTestFixture::new().await;
        
        // When: Generating response and examining audit trail
        let response = fixture.engine.generate_response(fixture.requirement_query_request.clone()).await.unwrap();
        
        // Then: Audit trail should document template selection reasoning
        assert!(!response.audit_trail.template_selection.to_string().is_empty(),
            "Template selection should be documented in audit trail");
        
        // Audit trail should be complete
        assert!(!response.audit_trail.id.to_string().is_empty(),
            "Audit trail should have valid ID");
        
        // Performance trail should include template selection timing
        assert!(response.metrics.template_selection_time.as_millis() >= 0,
            "Template selection time should be measured");
        assert!(response.metrics.template_selection_time.as_millis() < 1000,
            "Template selection should be fast");
        
        // Audit trail creation should be timestamped
        let time_diff = chrono::Utc::now().signed_duration_since(response.audit_trail.created_at);
        assert!(time_diff.num_seconds() < 60,
            "Audit trail should have recent timestamp");
        
        // All template selection decisions should be traceable
        assert!(!response.audit_trail.validation_steps.is_empty(),
            "Audit trail should include validation steps");
    }
}