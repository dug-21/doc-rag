//! Standalone validation tests for symbolic reasoning components
//! These tests validate the symbolic-first architecture without integration dependencies

#[cfg(test)]
mod tests {
    use crate::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_symbolic_first_architecture() {
        // Test that symbolic reasoning is PRIMARY, not fallback
        let mut engine = DatalogEngine::new();

        // Add test rules to validate symbolic reasoning
        let rule = crate::types::RequirementRule {
            id: "test-symbolic-rule".to_string(),
            requirement_type: "encryption_requirement".to_string(),
            conditions: vec!["contains_cardholder_data".to_string()],
            section: "3.2.1".to_string(),
            confidence: 0.95,
        };

        engine.load_requirements(&[rule]).unwrap();

        // Test symbolic reasoning response time meets CONSTRAINT-001
        let start = Instant::now();
        let results = engine.query("Does cardholder data require encryption?").await.unwrap();
        let elapsed = start.elapsed();

        // CONSTRAINT-001: <100ms symbolic reasoning
        assert!(elapsed.as_millis() < 100, "Symbolic reasoning took {:?}, exceeds 100ms", elapsed);
        assert!(!results.is_empty(), "Symbolic reasoning must return results");

        // Verify proof chains are generated for audit compliance
        assert!(!results[0].proof_steps.is_empty(), "Proof chains required for compliance");
    }

    #[tokio::test]
    async fn test_neural_classification_only_constraint() {
        let mut classifier = NeuralClassifierSystem::new();
        classifier.initialize().await.unwrap();

        // Test CONSTRAINT-003: Neural networks are CLASSIFICATION ONLY (not generation)
        let result = classifier.classify_query("What are encryption requirements?").await.unwrap();

        // CONSTRAINT-003: <10ms neural inference
        assert!(result.inference_time_ms < 10, "Neural inference took {}ms, exceeds 10ms", result.inference_time_ms);

        // Verify neural network returns ONLY classification labels, not generated text
        assert!(matches!(
            result.classification.as_str(),
            "RequirementLookup" | "ComplianceCheck" | "RelationshipQuery" | "ComplexReasoning" | "GeneralQuery"
        ), "Neural network must return only classification labels, not generated content");

        // Classification should NOT contain natural language generation
        assert!(!result.classification.contains(" "), "Classification should be enum label, not sentence");
        assert!(result.classification.len() < 50, "Classification should be label, not generated text");
    }

    #[tokio::test]
    async fn test_neurosymbolic_processing_architecture() {
        let processor = NeurosymbolicProcessor::new().await.unwrap();

        let query = crate::neurosymbolic_processor::NeurosymbolicQuery {
            query: "What encryption is required for PCI compliance?".to_string(),
            confidence_threshold: 0.8,
            max_results: 10,
            use_proof_chains: true,
        };

        let start = Instant::now();
        let result = processor.process_query(query).await.unwrap();
        let elapsed = start.elapsed();

        // CONSTRAINT-006: Total processing <1s
        assert!(elapsed.as_millis() < 1000, "Total processing took {:?}, exceeds 1s", elapsed);

        // Verify symbolic-first architecture
        assert!(!result.symbolic_results.is_empty() || result.classification.contains("General"),
                "Symbolic reasoning should provide primary results");

        // CONSTRAINT-004: Template-based generation (no hallucination)
        assert!(result.response.contains("analysis") || result.response.contains("requirements") ||
                result.response.contains("compliance"),
                "Response should use templates, not free generation");

        // Verify proof chains for audit compliance
        if result.proof_chain.is_some() {
            let proof = result.proof_chain.unwrap();
            assert!(!proof.is_empty(), "Proof chains must contain steps for compliance");
        }
    }

    #[tokio::test]
    async fn test_datalog_performance_constraint() {
        let mut engine = DatalogEngine::new();

        // Add multiple rules to test performance
        for i in 0..10 {
            let rule = crate::types::RequirementRule {
                id: format!("perf-rule-{}", i),
                requirement_type: format!("requirement_type_{}", i),
                conditions: vec![format!("condition_{}", i)],
                section: format!("Section {}", i),
                confidence: 0.9,
            };
            engine.load_requirements(&[rule]).unwrap();
        }

        // Test multiple queries to ensure consistent performance
        for _ in 0..5 {
            let start = Instant::now();
            let _results = engine.query("test performance query").await.unwrap();
            let elapsed = start.elapsed();

            // CONSTRAINT-001: Each query must be <100ms
            assert!(elapsed.as_millis() < 100,
                   "Datalog query took {:?}, exceeds 100ms constraint", elapsed);
        }
    }

    #[tokio::test]
    async fn test_template_prevention_of_hallucination() {
        let processor = NeurosymbolicProcessor::new().await.unwrap();

        // Test different query types to validate template usage
        let test_queries = vec![
            ("What are the requirements?", "RequirementLookup"),
            ("Is the system compliant?", "ComplianceCheck"),
            ("General information request", "GeneralQuery"),
        ];

        for (query_text, expected_type) in test_queries {
            let query = crate::neurosymbolic_processor::NeurosymbolicQuery {
                query: query_text.to_string(),
                confidence_threshold: 0.8,
                max_results: 5,
                use_proof_chains: false,
            };

            let result = processor.process_query(query).await.unwrap();

            // CONSTRAINT-004: Template-based responses prevent hallucination
            match expected_type {
                "RequirementLookup" => {
                    assert!(result.response.contains("Requirements") || result.response.contains("analysis"),
                           "RequirementLookup should use requirement template");
                },
                "ComplianceCheck" => {
                    assert!(result.response.contains("Compliance") || result.response.contains("Status"),
                           "ComplianceCheck should use compliance template");
                },
                "GeneralQuery" => {
                    assert!(result.response.contains("Results") || result.response.contains("Query"),
                           "GeneralQuery should use general template");
                },
                _ => {}
            }

            // Responses should be structured, not free-form generated text
            assert!(result.response.len() > 10, "Response should have content");
            assert!(result.response.len() < 1000, "Response should be concise, not generated essay");
        }
    }

    #[tokio::test]
    async fn test_integration_readiness() {
        // Verify all components are ready for DAA integration when circular deps resolved

        // Test DatalogEngine basic functionality
        let engine = DatalogEngine::new();
        assert!(engine.query("test").await.is_ok(), "DatalogEngine should be functional");

        // Test NeuralClassifier basic functionality
        let mut classifier = NeuralClassifierSystem::new();
        assert!(classifier.initialize().await.is_ok(), "NeuralClassifier should initialize");

        // Test NeurosymbolicProcessor basic functionality
        let processor = NeurosymbolicProcessor::new().await;
        assert!(processor.is_ok(), "NeurosymbolicProcessor should initialize");

        // All components compile and run without integration layer
        println!("✅ Symbolic engine components are ready for DAA integration");
    }
}