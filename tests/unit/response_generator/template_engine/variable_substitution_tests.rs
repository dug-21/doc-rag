//! Variable substitution tests for proof chain integration
//!
//! Tests behavior verification for variable substitution from proof chains,
//! citations, and context with confidence scoring and audit trail validation.

use crate::fixtures::*;
use std::time::Duration;
use std::collections::HashMap;
use mockall::predicate::*;

#[cfg(test)]
mod variable_substitution_tests {
    use super::*;

    #[tokio::test]
    async fn test_proof_chain_variable_substitution() {
        // Given: Template engine with proof chain variables
        let mut fixture = TemplateEngineTestFixture::new().await;
        
        let proof_chain_request = MockTemplateRequest {
            template_type: "ProofChainTemplate".to_string(),
            variable_values: HashMap::new(), // Force proof chain substitution
            proof_chain_data: vec![
                "encryption_rule: AES-256 required".to_string(),
                "storage_policy: Encrypted at rest".to_string(),
                "transmission_security: TLS 1.3 minimum".to_string(),
            ],
            citations: Vec::new(),
            context: HashMap::new(),
        };

        // Configure proof chain substitution
        fixture.engine
            .expect_substitute_variables()
            .with(eq("ProofChainTemplate"), eq(proof_chain_request.proof_chain_data.clone()))
            .times(1)
            .returning(|_, proof_chain| {
                let substitutions = proof_chain.iter().enumerate().map(|(i, element)| {
                    MockSubstitution {
                        variable: format!("proof_element_{}", i),
                        value: element.clone(),
                        source: "ProofChain".to_string(),
                        confidence: 0.9 - (i as f64 * 0.1), // Decreasing confidence
                    }
                }).collect();
                Ok(substitutions)
            });

        fixture.engine
            .expect_generate_response()
            .with(eq(proof_chain_request.clone()))
            .times(1)
            .returning(|request| {
                let proof_substitutions = request.proof_chain_data.iter().enumerate().map(|(i, element)| {
                    MockSubstitution {
                        variable: format!("proof_element_{}", i),
                        value: element.clone(),
                        source: "ProofChain".to_string(),
                        confidence: 0.9 - (i as f64 * 0.1),
                    }
                }).collect();

                Ok(MockTemplateResponse {
                    content: "Response with proof chain substitutions".to_string(),
                    constraint_004_compliant: true,
                    constraint_006_compliant: true,
                    generation_time: Duration::from_millis(320),
                    validation_results: MockValidationResults {
                        is_valid: true,
                        constraint_004_compliant: true,
                        constraint_006_compliant: true,
                    },
                    substitutions: proof_substitutions,
                    audit_trail: MockAuditTrail {
                        substitution_trail: vec![
                            "Proof chain substitution: proof_element_0 -> encryption_rule: AES-256 required".to_string(),
                            "Proof chain substitution: proof_element_1 -> storage_policy: Encrypted at rest".to_string(),
                            "Proof chain substitution: proof_element_2 -> transmission_security: TLS 1.3 minimum".to_string(),
                        ],
                        performance_trail: MockPerformanceTrail {
                            substitution_time: Duration::from_millis(25),
                            validation_time: Duration::from_millis(10),
                            total_time: Duration::from_millis(320),
                        },
                    },
                })
            });

        // When: Substituting variables from proof chain
        let substitutions = fixture.engine
            .substitute_variables("ProofChainTemplate", &proof_chain_request.proof_chain_data)
            .await
            .unwrap();

        let response = fixture.engine
            .generate_response(proof_chain_request)
            .await
            .unwrap();

        // Then: Variables should be substituted from proof chain
        assert!(!substitutions.is_empty(), "Should have proof chain substitutions");
        assert_eq!(substitutions.len(), 3, "Should substitute all proof chain elements");
        
        for substitution in &substitutions {
            assert!(substitution.source.contains("ProofChain"),
                    "Substitution source should indicate proof chain origin");
            assert!(substitution.confidence >= 0.5,
                    "Substitution confidence {:.3} should be >= 0.5", substitution.confidence);
        }

        // And: Response should contain proof chain substitutions
        let proof_substitutions: Vec<_> = response.substitutions.iter()
            .filter(|s| s.source.contains("ProofChain"))
            .collect();
        assert!(!proof_substitutions.is_empty(), "Response should have proof chain substitutions");

        // And: Audit trail should track all substitutions
        assert_eq!(response.audit_trail.substitution_trail.len(), response.substitutions.len(),
                   "Audit trail should track all substitutions");
        assert!(response.audit_trail.performance_trail.substitution_time.as_millis() < 50,
                "Substitution time should be < 50ms");
    }

    #[tokio::test]
    async fn test_citation_variable_substitution() {
        // Given: Template engine requiring citation variables
        let mut fixture = TemplateEngineTestFixture::new().await;
        
        let citation_request = MockTemplateRequest {
            template_type: "CitationTemplate".to_string(),
            variable_values: HashMap::new(),
            proof_chain_data: Vec::new(),
            citations: vec![
                "PCI DSS 3.2.1 Section 3.4.1".to_string(),
                "NIST 800-53 SC-8 (Transmission Confidentiality)".to_string(),
                "ISO 27001:2013 A.10.1.1".to_string(),
            ],
            context: HashMap::new(),
        };

        fixture.engine
            .expect_generate_response()
            .with(eq(citation_request.clone()))
            .times(1)
            .returning(|request| {
                let citation_substitutions = request.citations.iter().enumerate().map(|(i, citation)| {
                    MockSubstitution {
                        variable: format!("citation_{}", i),
                        value: citation.clone(),
                        source: "CitationSource".to_string(),
                        confidence: 0.85 + (i as f64 * 0.05), // Increasing confidence
                    }
                }).collect();

                Ok(MockTemplateResponse {
                    content: "Response with formatted citations and references".to_string(),
                    constraint_004_compliant: true,
                    constraint_006_compliant: true,
                    generation_time: Duration::from_millis(280),
                    validation_results: MockValidationResults {
                        is_valid: true,
                        constraint_004_compliant: true,
                        constraint_006_compliant: true,
                    },
                    substitutions: citation_substitutions,
                    audit_trail: MockAuditTrail {
                        substitution_trail: request.citations.iter().enumerate().map(|(i, citation)| {
                            format!("Citation substitution: citation_{} -> {}", i, citation)
                        }).collect(),
                        performance_trail: MockPerformanceTrail {
                            substitution_time: Duration::from_millis(18),
                            validation_time: Duration::from_millis(12),
                            total_time: Duration::from_millis(280),
                        },
                    },
                })
            });

        // When: Generating response with citations
        let response = fixture.engine
            .generate_response(citation_request.clone())
            .await
            .unwrap();

        // Then: Citations should be properly formatted and substituted
        assert!(!response.substitutions.is_empty(), "Should have citation substitutions");
        
        let citation_substitutions: Vec<_> = response.substitutions.iter()
            .filter(|s| s.source.contains("CitationSource"))
            .collect();
        assert_eq!(citation_substitutions.len(), 3, "Should have all citation substitutions");

        for substitution in &citation_substitutions {
            assert!(substitution.confidence >= 0.7,
                    "Citation confidence {:.3} should be >= 0.7", substitution.confidence);
            assert!(!substitution.value.is_empty(),
                    "Citation value should not be empty");
        }

        // And: Citation variables should be properly tracked in audit trail
        let citation_audit_events: Vec<_> = response.audit_trail.substitution_trail.iter()
            .filter(|event| event.contains("Citation substitution"))
            .collect();
        assert_eq!(citation_audit_events.len(), 3,
                   "Should audit all citation substitutions");
    }

    #[tokio::test]
    async fn test_complex_variable_substitution_performance() {
        // Given: Template engine with complex variable substitution scenario
        let mut fixture = TemplateEngineTestFixture::new().await;
        let complex_template = fixture.complex_variable_template.clone();

        fixture.engine
            .expect_substitute_variables()
            .with(eq("ComplexTemplate"), eq(complex_template.proof_chain_data.clone()))
            .times(1)
            .returning(|_, proof_chain| {
                // Simulate 50 variable substitutions
                let substitutions = (0..50).map(|i| MockSubstitution {
                    variable: format!("var_{}", i),
                    value: format!("value_{}", i),
                    source: "ProofChain".to_string(),
                    confidence: 0.8 + (i as f64 % 20.0) / 100.0, // Vary confidence
                }).collect();
                Ok(substitutions)
            });

        fixture.engine
            .expect_generate_response()
            .with(eq(complex_template.clone()))
            .times(1)
            .returning(|_| {
                let substitutions = (0..50).map(|i| MockSubstitution {
                    variable: format!("var_{}", i),
                    value: format!("value_{}", i),
                    source: "ProofChain".to_string(),
                    confidence: 0.8 + (i as f64 % 20.0) / 100.0,
                }).collect();

                Ok(MockTemplateResponse {
                    content: "Complex response with 50 variable substitutions".to_string(),
                    constraint_004_compliant: true,
                    constraint_006_compliant: true,
                    generation_time: Duration::from_millis(450),
                    validation_results: MockValidationResults {
                        is_valid: true,
                        constraint_004_compliant: true,
                        constraint_006_compliant: true,
                    },
                    substitutions,
                    audit_trail: MockAuditTrail {
                        substitution_trail: (0..50).map(|i| format!("Variable substitution: var_{} -> value_{}", i, i)).collect(),
                        performance_trail: MockPerformanceTrail {
                            substitution_time: Duration::from_millis(75), // Realistic time for 50 substitutions
                            validation_time: Duration::from_millis(15),
                            total_time: Duration::from_millis(450),
                        },
                    },
                })
            });

        // When: Performing complex variable substitution
        let substitutions = fixture.engine
            .substitute_variables("ComplexTemplate", &complex_template.proof_chain_data)
            .await
            .unwrap();

        let response = fixture.engine
            .generate_response(complex_template)
            .await
            .unwrap();

        // Then: Should complete complex substitution efficiently
        assert_eq!(substitutions.len(), 50, "Should substitute all 50 variables");
        assert!(response.audit_trail.performance_trail.substitution_time.as_millis() < 100,
                "Complex substitution took {}ms, should be < 100ms",
                response.audit_trail.performance_trail.substitution_time.as_millis());

        // And: All substitutions should have confidence scores
        for substitution in &substitutions {
            assert!(substitution.confidence > 0.0,
                    "Substitution {} should have confidence > 0", substitution.variable);
            assert!(substitution.confidence <= 1.0,
                    "Substitution {} confidence should be <= 1.0", substitution.variable);
        }

        // And: Overall response should still meet performance constraint
        assert!(response.generation_time.as_millis() < 1000,
                "Total generation time {}ms should meet <1s constraint",
                response.generation_time.as_millis());
    }

    #[tokio::test]
    async fn test_variable_substitution_conflict_resolution() {
        // Given: Template engine with conflicting variable sources
        let mut fixture = TemplateEngineTestFixture::new().await;
        
        let conflict_request = MockTemplateRequest {
            template_type: "ConflictTemplate".to_string(),
            variable_values: {
                let mut vars = HashMap::new();
                vars.insert("standard".to_string(), "ISO 27001".to_string());
                vars.insert("requirement".to_string(), "Encryption mandatory".to_string());
                vars
            },
            proof_chain_data: vec![
                "standard: PCI DSS".to_string(), // Conflicts with variable_values
                "requirement: Encryption recommended".to_string(), // Conflicts with variable_values
                "additional_context: Risk assessment".to_string(),
            ],
            citations: vec!["NIST 800-53".to_string()],
            context: HashMap::new(),
        };

        fixture.engine
            .expect_generate_response()
            .with(eq(conflict_request.clone()))
            .times(1)
            .returning(|request| {
                // Simulate conflict resolution: variable_values take precedence
                let mut substitutions = Vec::new();
                
                // Variable values (higher priority)
                for (var, value) in &request.variable_values {
                    substitutions.push(MockSubstitution {
                        variable: var.clone(),
                        value: value.clone(),
                        source: "VariableValues".to_string(),
                        confidence: 1.0, // Highest confidence for explicit values
                    });
                }
                
                // Proof chain (non-conflicting)
                substitutions.push(MockSubstitution {
                    variable: "additional_context".to_string(),
                    value: "Risk assessment".to_string(),
                    source: "ProofChain".to_string(),
                    confidence: 0.8,
                });

                Ok(MockTemplateResponse {
                    content: "Response with resolved variable conflicts".to_string(),
                    constraint_004_compliant: true,
                    constraint_006_compliant: true,
                    generation_time: Duration::from_millis(290),
                    validation_results: MockValidationResults {
                        is_valid: true,
                        constraint_004_compliant: true,
                        constraint_006_compliant: true,
                    },
                    substitutions,
                    audit_trail: MockAuditTrail {
                        substitution_trail: vec![
                            "Variable conflict detected: standard".to_string(),
                            "Resolution: VariableValues precedence over ProofChain".to_string(),
                            "Variable conflict detected: requirement".to_string(),
                            "Resolution: VariableValues precedence over ProofChain".to_string(),
                            "Substitution: standard -> ISO 27001 (VariableValues)".to_string(),
                            "Substitution: requirement -> Encryption mandatory (VariableValues)".to_string(),
                            "Substitution: additional_context -> Risk assessment (ProofChain)".to_string(),
                        ],
                        performance_trail: MockPerformanceTrail {
                            substitution_time: Duration::from_millis(35),
                            validation_time: Duration::from_millis(8),
                            total_time: Duration::from_millis(290),
                        },
                    },
                })
            });

        // When: Processing request with variable conflicts
        let response = fixture.engine
            .generate_response(conflict_request.clone())
            .await
            .unwrap();

        // Then: Should resolve conflicts with precedence rules
        let standard_substitution = response.substitutions.iter()
            .find(|s| s.variable == "standard")
            .expect("Should have standard substitution");
        assert_eq!(standard_substitution.value, "ISO 27001",
                   "Should use VariableValues over ProofChain for conflicts");
        assert_eq!(standard_substitution.source, "VariableValues",
                   "Should indicate source resolution");
        assert_eq!(standard_substitution.confidence, 1.0,
                   "Variable values should have highest confidence");

        // And: Audit trail should document conflict resolution
        let conflict_events: Vec<_> = response.audit_trail.substitution_trail.iter()
            .filter(|event| event.contains("conflict"))
            .collect();
        assert!(!conflict_events.is_empty(), "Should document variable conflicts");
        
        let resolution_events: Vec<_> = response.audit_trail.substitution_trail.iter()
            .filter(|event| event.contains("Resolution"))
            .collect();
        assert!(!resolution_events.is_empty(), "Should document conflict resolutions");
    }

    #[tokio::test]
    async fn test_variable_substitution_validation() {
        // Given: Template engine with variable validation requirements
        let mut fixture = TemplateEngineTestFixture::new().await;
        
        let validation_scenarios = vec![
            (
                "valid_variables",
                MockTemplateRequest {
                    template_type: "ValidationTemplate".to_string(),
                    variable_values: {
                        let mut vars = HashMap::new();
                        vars.insert("compliance_level".to_string(), "Level 1".to_string());
                        vars.insert("data_type".to_string(), "Cardholder Data".to_string());
                        vars
                    },
                    proof_chain_data: vec!["encryption_strength: AES-256".to_string()],
                    citations: vec!["PCI DSS 3.2.1".to_string()],
                    context: HashMap::new(),
                },
                true,
                "Valid variables should pass validation"
            ),
            (
                "empty_variable_values",
                MockTemplateRequest {
                    template_type: "ValidationTemplate".to_string(),
                    variable_values: {
                        let mut vars = HashMap::new();
                        vars.insert("compliance_level".to_string(), "".to_string()); // Empty value
                        vars
                    },
                    proof_chain_data: Vec::new(),
                    citations: Vec::new(),
                    context: HashMap::new(),
                },
                false,
                "Empty variable values should fail validation"
            ),
            (
                "malformed_proof_chain",
                MockTemplateRequest {
                    template_type: "ValidationTemplate".to_string(),
                    variable_values: HashMap::new(),
                    proof_chain_data: vec!["malformed_data_without_colon".to_string()],
                    citations: Vec::new(),
                    context: HashMap::new(),
                },
                false,
                "Malformed proof chain data should fail validation"
            ),
        ];

        for (scenario_name, request, should_succeed, description) in validation_scenarios {
            if should_succeed {
                fixture.engine
                    .expect_generate_response()
                    .with(eq(request.clone()))
                    .times(1)
                    .returning(|_| Ok(MockTemplateResponse {
                        content: "Valid response".to_string(),
                        constraint_004_compliant: true,
                        constraint_006_compliant: true,
                        generation_time: Duration::from_millis(200),
                        validation_results: MockValidationResults {
                            is_valid: true,
                            constraint_004_compliant: true,
                            constraint_006_compliant: true,
                        },
                        substitutions: vec![MockSubstitution {
                            variable: "test_var".to_string(),
                            value: "test_value".to_string(),
                            source: "validation".to_string(),
                            confidence: 0.9,
                        }],
                        audit_trail: MockAuditTrail {
                            substitution_trail: vec!["Variable validation: PASSED".to_string()],
                            performance_trail: MockPerformanceTrail {
                                substitution_time: Duration::from_millis(10),
                                validation_time: Duration::from_millis(5),
                                total_time: Duration::from_millis(200),
                            },
                        },
                    }));
            } else {
                fixture.engine
                    .expect_generate_response()
                    .with(eq(request.clone()))
                    .times(1)
                    .returning(move |_| Err(format!("Variable validation failed for {}", scenario_name)));
            }

            // When: Processing variable validation scenario
            let result = fixture.engine
                .generate_response(request)
                .await;

            // Then: Should validate appropriately
            if should_succeed {
                assert!(result.is_ok(), "{}: {}", scenario_name, description);
                let response = result.unwrap();
                assert!(response.validation_results.is_valid,
                        "Response should be valid for {}", scenario_name);
            } else {
                assert!(result.is_err(), "{}: {}", scenario_name, description);
                let error = result.unwrap_err();
                assert!(error.contains("validation"),
                        "Error should mention validation for {}: {}", scenario_name, error);
            }
        }
    }

    #[tokio::test]
    async fn test_variable_substitution_confidence_scoring() {
        // Given: Template engine with confidence-based variable substitution
        let mut fixture = TemplateEngineTestFixture::new().await;
        
        let confidence_request = MockTemplateRequest {
            template_type: "ConfidenceTemplate".to_string(),
            variable_values: {
                let mut vars = HashMap::new();
                vars.insert("high_confidence_var".to_string(), "Explicit Value".to_string());
                vars
            },
            proof_chain_data: vec![
                "medium_confidence_var: Inferred from context".to_string(),
                "low_confidence_var: Ambiguous reference".to_string(),
            ],
            citations: vec!["Authoritative Source Citation".to_string()],
            context: {
                let mut ctx = HashMap::new();
                ctx.insert("derived_var".to_string(), "Context-derived value".to_string());
                ctx
            },
        };

        fixture.engine
            .expect_generate_response()
            .with(eq(confidence_request.clone()))
            .times(1)
            .returning(|_| {
                let substitutions = vec![
                    MockSubstitution {
                        variable: "high_confidence_var".to_string(),
                        value: "Explicit Value".to_string(),
                        source: "VariableValues".to_string(),
                        confidence: 1.0, // Explicit values have highest confidence
                    },
                    MockSubstitution {
                        variable: "citation_var".to_string(),
                        value: "Authoritative Source Citation".to_string(),
                        source: "CitationSource".to_string(),
                        confidence: 0.95, // Authoritative sources have high confidence
                    },
                    MockSubstitution {
                        variable: "medium_confidence_var".to_string(),
                        value: "Inferred from context".to_string(),
                        source: "ProofChain".to_string(),
                        confidence: 0.75, // Inferred values have medium confidence
                    },
                    MockSubstitution {
                        variable: "derived_var".to_string(),
                        value: "Context-derived value".to_string(),
                        source: "Context".to_string(),
                        confidence: 0.6, // Context-derived values have lower confidence
                    },
                    MockSubstitution {
                        variable: "low_confidence_var".to_string(),
                        value: "Ambiguous reference".to_string(),
                        source: "ProofChain".to_string(),
                        confidence: 0.4, // Ambiguous values have low confidence
                    },
                ];

                Ok(MockTemplateResponse {
                    content: "Response with confidence-scored substitutions".to_string(),
                    constraint_004_compliant: true,
                    constraint_006_compliant: true,
                    generation_time: Duration::from_millis(350),
                    validation_results: MockValidationResults {
                        is_valid: true,
                        constraint_004_compliant: true,
                        constraint_006_compliant: true,
                    },
                    substitutions,
                    audit_trail: MockAuditTrail {
                        substitution_trail: vec![
                            "Confidence scoring: high_confidence_var = 1.00".to_string(),
                            "Confidence scoring: citation_var = 0.95".to_string(),
                            "Confidence scoring: medium_confidence_var = 0.75".to_string(),
                            "Confidence scoring: derived_var = 0.60".to_string(),
                            "Confidence scoring: low_confidence_var = 0.40".to_string(),
                        ],
                        performance_trail: MockPerformanceTrail {
                            substitution_time: Duration::from_millis(40),
                            validation_time: Duration::from_millis(15),
                            total_time: Duration::from_millis(350),
                        },
                    },
                })
            });

        // When: Generating response with confidence scoring
        let response = fixture.engine
            .generate_response(confidence_request)
            .await
            .unwrap();

        // Then: Should provide appropriate confidence scores by source type
        let high_conf_substitutions: Vec<_> = response.substitutions.iter()
            .filter(|s| s.confidence >= 0.9)
            .collect();
        assert!(!high_conf_substitutions.is_empty(),
                "Should have high confidence substitutions");

        let medium_conf_substitutions: Vec<_> = response.substitutions.iter()
            .filter(|s| s.confidence >= 0.7 && s.confidence < 0.9)
            .collect();
        assert!(!medium_conf_substitutions.is_empty(),
                "Should have medium confidence substitutions");

        let low_conf_substitutions: Vec<_> = response.substitutions.iter()
            .filter(|s| s.confidence < 0.7)
            .collect();
        assert!(!low_conf_substitutions.is_empty(),
                "Should have low confidence substitutions");

        // And: Confidence scores should correlate with source reliability
        let explicit_value_sub = response.substitutions.iter()
            .find(|s| s.source == "VariableValues")
            .expect("Should have explicit value substitution");
        assert_eq!(explicit_value_sub.confidence, 1.0,
                   "Explicit values should have maximum confidence");

        let citation_sub = response.substitutions.iter()
            .find(|s| s.source == "CitationSource")
            .expect("Should have citation substitution");
        assert!(citation_sub.confidence >= 0.9,
                "Citations should have high confidence");

        // And: Audit trail should document confidence scoring
        let confidence_events: Vec<_> = response.audit_trail.substitution_trail.iter()
            .filter(|event| event.contains("Confidence scoring"))
            .collect();
        assert_eq!(confidence_events.len(), response.substitutions.len(),
                   "Should audit confidence scoring for all substitutions");
    }
}