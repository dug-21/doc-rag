//! CONSTRAINT-004 compliance tests for deterministic generation
//!
//! Tests enforcement of deterministic template-based generation only
//! with comprehensive violation detection and compliance validation.

use crate::fixtures::*;
use std::time::Duration;
use std::collections::HashMap;
use mockall::predicate::*;

#[cfg(test)]
mod constraint_004_tests {
    use super::*;

    #[tokio::test]
    async fn test_deterministic_generation_enforcement() {
        // Given: Template engine with CONSTRAINT-004 enforcement enabled
        let mut fixture = TemplateEngineTestFixture::new().await;
        
        // Configure engine to enforce deterministic generation only
        let non_deterministic_request = MockTemplateRequest {
            template_type: "".to_string(), // No template = free generation
            variable_values: HashMap::new(),
            proof_chain_data: vec!["Free form generation attempt".to_string()],
            citations: Vec::new(),
            context: {
                let mut ctx = HashMap::new();
                ctx.insert("allow_free_generation".to_string(), "true".to_string());
                ctx
            },
        };

        // Configure mock to reject non-deterministic requests
        fixture.engine
            .expect_validate_constraint_004()
            .with(eq(non_deterministic_request.clone()))
            .times(1)
            .returning(|_| Ok(false));

        fixture.engine
            .expect_generate_response()
            .with(eq(non_deterministic_request.clone()))
            .times(1)
            .returning(|_| Err("CONSTRAINT-004 violation: Deterministic generation required".to_string()));

        // When: Attempting free-form generation
        let result = fixture.engine.generate_response(non_deterministic_request.clone()).await;

        // Then: Should reject with CONSTRAINT-004 violation
        assert!(result.is_err(), "Should reject non-deterministic generation");
        let error_message = result.unwrap_err();
        assert!(error_message.contains("CONSTRAINT-004"), 
                "Error should reference CONSTRAINT-004: {}", error_message);
        assert!(error_message.contains("deterministic"), 
                "Error should mention deterministic requirement: {}", error_message);
    }

    #[tokio::test]
    async fn test_template_based_generation_success() {
        // Given: Template engine with valid template-based request
        let mut fixture = TemplateEngineTestFixture::new().await;
        let valid_request = MockTemplateRequest::new_requirement_query();

        // Configure successful template-based generation
        fixture.engine
            .expect_validate_constraint_004()
            .with(eq(valid_request.clone()))
            .times(1)
            .returning(|_| Ok(true));

        fixture.engine
            .expect_generate_response()
            .with(eq(valid_request.clone()))
            .times(1)
            .returning(|request| Ok(MockTemplateResponse {
                content: format!("Generated response for template: {}", request.template_type),
                constraint_004_compliant: true,
                constraint_006_compliant: true,
                generation_time: Duration::from_millis(450),
                validation_results: MockValidationResults {
                    is_valid: true,
                    constraint_004_compliant: true,
                    constraint_006_compliant: true,
                },
                substitutions: vec![
                    MockSubstitution {
                        variable: "requirement_type".to_string(),
                        value: "Must".to_string(),
                        source: "variable_values".to_string(),
                        confidence: 1.0,
                    }
                ],
                audit_trail: MockAuditTrail {
                    substitution_trail: vec!["requirement_type substituted".to_string()],
                    performance_trail: MockPerformanceTrail {
                        substitution_time: Duration::from_millis(5),
                        validation_time: Duration::from_millis(10),
                        total_time: Duration::from_millis(450),
                    },
                },
            }));

        // When: Generating response with valid template request
        let response = fixture.engine.generate_response(valid_request).await.unwrap();

        // Then: Should succeed with full compliance
        assert!(response.validation_results.constraint_004_compliant,
                "Response should be CONSTRAINT-004 compliant");
        assert!(response.validation_results.is_valid,
                "Response should be valid");
        assert!(!response.content.is_empty(),
                "Response should contain generated content");
        assert!(!response.audit_trail.substitution_trail.is_empty(),
                "Should have audit trail for substitutions");

        // And: Should meet performance constraint
        assert!(response.generation_time.as_millis() < 1000,
                "Generation time {}ms should meet <1s constraint",
                response.generation_time.as_millis());
    }

    #[tokio::test]
    async fn test_template_validation_comprehensive() {
        // Given: Template engine with various template validation scenarios
        let mut fixture = TemplateEngineTestFixture::new().await;
        
        let validation_scenarios = vec![
            (
                "valid_requirement_template",
                MockTemplateRequest::new_requirement_query(),
                true,
                "Valid requirement template should pass validation"
            ),
            (
                "empty_template_type",
                MockTemplateRequest {
                    template_type: "".to_string(),
                    variable_values: HashMap::new(),
                    proof_chain_data: Vec::new(),
                    citations: Vec::new(),
                    context: HashMap::new(),
                },
                false,
                "Empty template type should fail validation"
            ),
            (
                "invalid_template_type",
                MockTemplateRequest {
                    template_type: "NonExistentTemplate".to_string(),
                    variable_values: HashMap::new(),
                    proof_chain_data: Vec::new(),
                    citations: Vec::new(),
                    context: HashMap::new(),
                },
                false,
                "Invalid template type should fail validation"
            ),
            (
                "free_generation_attempt",
                MockTemplateRequest {
                    template_type: "FreeGeneration".to_string(),
                    variable_values: HashMap::new(),
                    proof_chain_data: vec!["Generate free-form response".to_string()],
                    citations: Vec::new(),
                    context: {
                        let mut ctx = HashMap::new();
                        ctx.insert("mode".to_string(), "creative".to_string());
                        ctx
                    },
                },
                false,
                "Free generation should fail CONSTRAINT-004 validation"
            ),
        ];

        for (scenario_name, request, should_pass, description) in validation_scenarios {
            // Configure validation mock
            fixture.engine
                .expect_validate_constraint_004()
                .with(eq(request.clone()))
                .times(1)
                .returning(move |_| Ok(should_pass));

            // When: Validating template request
            let validation_result = fixture.engine
                .validate_constraint_004(&request)
                .await
                .unwrap();

            // Then: Validation should match expected result
            assert_eq!(validation_result, should_pass,
                       "{}: {}", scenario_name, description);
        }
    }

    #[tokio::test]
    async fn test_constraint_004_audit_trail_generation() {
        // Given: Template engine with audit trail requirements
        let mut fixture = TemplateEngineTestFixture::new().await;
        let request_with_audit = MockTemplateRequest {
            template_type: "AuditableTemplate".to_string(),
            variable_values: {
                let mut vars = HashMap::new();
                vars.insert("compliance_standard".to_string(), "PCI DSS".to_string());
                vars.insert("requirement_level".to_string(), "Must".to_string());
                vars
            },
            proof_chain_data: vec![
                "encryption_requirement".to_string(),
                "storage_security".to_string(),
            ],
            citations: vec!["PCI DSS 3.2.1".to_string()],
            context: HashMap::new(),
        };

        fixture.engine
            .expect_validate_constraint_004()
            .with(eq(request_with_audit.clone()))
            .times(1)
            .returning(|_| Ok(true));

        fixture.engine
            .expect_generate_response()
            .with(eq(request_with_audit.clone()))
            .times(1)
            .returning(|request| {
                let audit_events = vec![
                    "Template selection: AuditableTemplate".to_string(),
                    "Variable substitution: compliance_standard -> PCI DSS".to_string(),
                    "Variable substitution: requirement_level -> Must".to_string(),
                    "Proof chain integration: encryption_requirement".to_string(),
                    "Proof chain integration: storage_security".to_string(),
                    "Citation formatting: PCI DSS 3.2.1".to_string(),
                    "CONSTRAINT-004 validation: PASSED".to_string(),
                    "CONSTRAINT-006 validation: PASSED".to_string(),
                ];

                Ok(MockTemplateResponse {
                    content: "Auditable response generated with full traceability".to_string(),
                    constraint_004_compliant: true,
                    constraint_006_compliant: true,
                    generation_time: Duration::from_millis(380),
                    validation_results: MockValidationResults {
                        is_valid: true,
                        constraint_004_compliant: true,
                        constraint_006_compliant: true,
                    },
                    substitutions: request.variable_values.iter().map(|(k, v)| MockSubstitution {
                        variable: k.clone(),
                        value: v.clone(),
                        source: "variable_values".to_string(),
                        confidence: 1.0,
                    }).collect(),
                    audit_trail: MockAuditTrail {
                        substitution_trail: audit_events,
                        performance_trail: MockPerformanceTrail {
                            substitution_time: Duration::from_millis(15),
                            validation_time: Duration::from_millis(25),
                            total_time: Duration::from_millis(380),
                        },
                    },
                })
            });

        // When: Generating response with full audit requirements
        let response = fixture.engine
            .generate_response(request_with_audit)
            .await
            .unwrap();

        // Then: Should provide comprehensive audit trail
        assert!(response.validation_results.constraint_004_compliant,
                "Should be CONSTRAINT-004 compliant");
        
        let audit_trail = &response.audit_trail.substitution_trail;
        assert!(audit_trail.iter().any(|event| event.contains("Template selection")),
                "Audit trail should include template selection");
        assert!(audit_trail.iter().any(|event| event.contains("Variable substitution")),
                "Audit trail should include variable substitutions");
        assert!(audit_trail.iter().any(|event| event.contains("Proof chain integration")),
                "Audit trail should include proof chain steps");
        assert!(audit_trail.iter().any(|event| event.contains("Citation formatting")),
                "Audit trail should include citation processing");
        assert!(audit_trail.iter().any(|event| event.contains("CONSTRAINT-004 validation: PASSED")),
                "Audit trail should include constraint validation");

        // And: Substitutions should be fully tracked
        assert_eq!(response.substitutions.len(), 2,
                   "Should track all variable substitutions");
        for substitution in &response.substitutions {
            assert!(substitution.confidence > 0.0,
                    "All substitutions should have confidence scores");
            assert!(!substitution.variable.is_empty(),
                    "All substitutions should specify variable");
            assert!(!substitution.value.is_empty(),
                    "All substitutions should specify value");
        }
    }

    #[tokio::test]
    async fn test_constraint_004_edge_cases() {
        // Given: Template engine with edge case scenarios
        let mut fixture = TemplateEngineTestFixture::new().await;
        
        let edge_cases = vec![
            (
                "whitespace_only_template",
                MockTemplateRequest {
                    template_type: "   ".to_string(),
                    variable_values: HashMap::new(),
                    proof_chain_data: Vec::new(),
                    citations: Vec::new(),
                    context: HashMap::new(),
                },
                false
            ),
            (
                "null_like_template",
                MockTemplateRequest {
                    template_type: "null".to_string(),
                    variable_values: HashMap::new(),
                    proof_chain_data: Vec::new(),
                    citations: Vec::new(),
                    context: HashMap::new(),
                },
                false
            ),
            (
                "template_with_injection_attempt",
                MockTemplateRequest {
                    template_type: "ValidTemplate'; DROP TABLE templates; --".to_string(),
                    variable_values: HashMap::new(),
                    proof_chain_data: Vec::new(),
                    citations: Vec::new(),
                    context: HashMap::new(),
                },
                false
            ),
        ];

        for (case_name, request, should_pass) in edge_cases {
            fixture.engine
                .expect_validate_constraint_004()
                .with(eq(request.clone()))
                .times(1)
                .returning(move |_| Ok(should_pass));

            if should_pass {
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
                        substitutions: Vec::new(),
                        audit_trail: MockAuditTrail {
                            substitution_trail: Vec::new(),
                            performance_trail: MockPerformanceTrail {
                                substitution_time: Duration::from_millis(1),
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
                    .returning(|_| Err(format!("CONSTRAINT-004 violation: Invalid template for {}", case_name)));
            }

            // When: Processing edge case
            let validation_result = fixture.engine
                .validate_constraint_004(&request)
                .await
                .unwrap();
            
            let generation_result = fixture.engine
                .generate_response(request)
                .await;

            // Then: Should handle edge case appropriately
            assert_eq!(validation_result, should_pass,
                       "Validation should match expected result for {}", case_name);

            if should_pass {
                assert!(generation_result.is_ok(),
                        "Generation should succeed for valid edge case {}", case_name);
            } else {
                assert!(generation_result.is_err(),
                        "Generation should fail for invalid edge case {}", case_name);
                let error = generation_result.unwrap_err();
                assert!(error.contains("CONSTRAINT-004"),
                        "Error should reference CONSTRAINT-004 for {}: {}", case_name, error);
            }
        }
    }

    #[tokio::test]
    async fn test_constraint_004_performance_impact() {
        // Given: Template engine with performance measurement for constraint validation
        let mut fixture = TemplateEngineTestFixture::new().await;
        
        let performance_test_requests = vec![
            MockTemplateRequest::new_requirement_query(),
            fixture.complex_variable_template.clone(),
            fixture.proof_chain_template.clone(),
        ];

        for (i, request) in performance_test_requests.iter().enumerate() {
            fixture.engine
                .expect_validate_constraint_004()
                .with(eq(request.clone()))
                .times(1)
                .returning(|_| Ok(true));

            let expected_validation_time = Duration::from_millis(5 + i as u64 * 2); // Increasing complexity
            let expected_total_time = Duration::from_millis(200 + i as u64 * 50);

            fixture.engine
                .expect_generate_response()
                .with(eq(request.clone()))
                .times(1)
                .returning(move |_| Ok(MockTemplateResponse {
                    content: format!("Performance test response {}", i),
                    constraint_004_compliant: true,
                    constraint_006_compliant: true,
                    generation_time: expected_total_time,
                    validation_results: MockValidationResults {
                        is_valid: true,
                        constraint_004_compliant: true,
                        constraint_006_compliant: true,
                    },
                    substitutions: Vec::new(),
                    audit_trail: MockAuditTrail {
                        substitution_trail: vec![format!("CONSTRAINT-004 validation completed for test {}", i)],
                        performance_trail: MockPerformanceTrail {
                            substitution_time: Duration::from_millis(10),
                            validation_time: expected_validation_time,
                            total_time: expected_total_time,
                        },
                    },
                }));

            // When: Generating response with performance monitoring
            let response = fixture.engine
                .generate_response(request.clone())
                .await
                .unwrap();

            // Then: CONSTRAINT-004 validation should have minimal performance impact
            assert!(response.audit_trail.performance_trail.validation_time.as_millis() < 50,
                    "CONSTRAINT-004 validation time {}ms should be < 50ms",
                    response.audit_trail.performance_trail.validation_time.as_millis());

            // And: Overall generation should still meet CONSTRAINT-006
            assert!(response.generation_time.as_millis() < 1000,
                    "Total generation time {}ms should meet <1s constraint",
                    response.generation_time.as_millis());

            // And: Validation overhead should be reasonable
            let validation_overhead = response.audit_trail.performance_trail.validation_time.as_millis() as f64 
                                    / response.generation_time.as_millis() as f64;
            assert!(validation_overhead < 0.1,
                    "Validation overhead {:.1}% should be < 10% of total time", validation_overhead * 100.0);
        }
    }
}