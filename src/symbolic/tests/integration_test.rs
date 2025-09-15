//! Integration tests for symbolic reasoning engine
//! Tests CONSTRAINT-001 compliance and symbolic-first neurosymbolic approach

use std::time::Instant;
use tokio;
use symbolic::{
    DatalogEngine,
    NeuralClassifier, NeurosymbolicProcessor, NeurosymbolicQuery,
    prolog::{PrologEngine, PrologQuery}
};
use symbolic::types::RequirementRule;
use symbolic::prolog::engine::{PrologFact, PrologRule};

#[tokio::test]
async fn test_datalog_performance_constraint() {
    let mut engine = DatalogEngine::new();

    // Load compliance requirements
    let requirement_rules = vec![
        RequirementRule {
            id: "pci_dss_3_2_1".to_string(),
            requirement_type: "encryption_requirement".to_string(),
            conditions: vec!["cardholder_data".to_string(), "stored_at_rest".to_string()],
            section: "3.2.1".to_string(),
            confidence: 0.98,
        },
        RequirementRule {
            id: "pci_dss_4_1".to_string(),
            requirement_type: "network_security".to_string(),
            conditions: vec!["transmission_protection".to_string()],
            section: "4.1".to_string(),
            confidence: 0.95,
        },
    ];

    engine.load_requirements(&requirement_rules).unwrap();

    // Test CONSTRAINT-001: <100ms query response time
    let start = Instant::now();
    let results = engine.query("Does cardholder data require encryption?").await.unwrap();
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 100, "Datalog query exceeded 100ms constraint: {:?}", elapsed);
    assert!(!results.is_empty(), "Should find encryption requirements");
    assert!(results[0].confidence > 0.9, "High confidence expected for explicit requirements");

    // Verify proof chain generation
    assert!(!results[0].proof_steps.is_empty(), "Proof steps should be generated for explainability");
}

#[tokio::test]
async fn test_prolog_complex_reasoning() {
    let mut prolog_engine = PrologEngine::new();

    // Add compliance facts
    prolog_engine.add_fact(PrologFact {
        predicate: "sensitive_data".to_string(),
        terms: vec!["cardholder_data".to_string()],
        source: "pci_dss".to_string(),
    });

    prolog_engine.add_fact(PrologFact {
        predicate: "system_handles".to_string(),
        terms: vec!["payment_system".to_string(), "cardholder_data".to_string()],
        source: "system_analysis".to_string(),
    });

    // Add inference rule: requires_protection(System, Data) :- system_handles(System, Data), sensitive_data(Data)
    prolog_engine.add_rule(PrologRule {
        head: "requires_protection(System, Data)".to_string(),
        body: vec![
            "system_handles(System, Data)".to_string(),
            "sensitive_data(Data)".to_string(),
        ],
        rule_id: "protection_rule".to_string(),
    });

    // Test complex reasoning query
    let query = PrologQuery {
        goal: "requires_protection(payment_system, cardholder_data)".to_string(),
        variables: vec![],
        timeout_ms: 100,
    };

    let start = Instant::now();
    let result = prolog_engine.query(query).await.unwrap();
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 100, "Prolog query exceeded 100ms constraint: {:?}", elapsed);
    assert!(result.success, "Should successfully prove protection requirement");
    assert!(!result.proof_steps.is_empty(), "Should generate complete proof chain");
    assert!(result.confidence > 0.8, "High confidence expected for logical proofs");
}

#[tokio::test]
async fn test_symbolic_first_neurosymbolic_approach() {
    let processor = NeurosymbolicProcessor::new().await.unwrap();

    // Load test requirements for symbolic reasoning
    let requirements = vec![
        RequirementRule {
            id: "encryption_rule".to_string(),
            requirement_type: "encryption_requirement".to_string(),
            conditions: vec!["contains_pii".to_string()],
            section: "Security Controls".to_string(),
            confidence: 0.95,
        },
        RequirementRule {
            id: "access_control_rule".to_string(),
            requirement_type: "access_control".to_string(),
            conditions: vec!["user_authentication".to_string()],
            section: "Access Management".to_string(),
            confidence: 0.92,
        },
    ];

    processor.load_requirements(&requirements).await.unwrap();

    // Test symbolic-first approach with compliance query
    let query = NeurosymbolicQuery {
        query: "What encryption requirements apply to PII data?".to_string(),
        confidence_threshold: 0.8,
        max_results: 10,
        use_proof_chains: true,
    };

    let start = Instant::now();
    let result = processor.process_query(query).await.unwrap();
    let elapsed = start.elapsed();

    // Verify performance constraints
    assert!(elapsed.as_millis() < 1000, "Total processing exceeded 1s target: {:?}", elapsed);
    assert!(result.processing_time_ms < 1000, "CONSTRAINT-006: Processing should be under 1s");

    // Verify symbolic-first approach
    assert_eq!(result.classification, "RequirementLookup");
    assert!(result.confidence > 0.7, "Should have high confidence from symbolic reasoning");
    assert!(!result.symbolic_results.is_empty(), "Should find symbolic results first");
    assert!(result.proof_chain.is_some(), "Should generate proof chain for explainability");

    // Verify response quality
    assert!(!result.response.is_empty(), "Should generate meaningful response");
    assert!(result.response.contains("encryption") || result.response.contains("Requirements"),
            "Response should be relevant to the query");
}

#[tokio::test]
async fn test_neural_classification_performance() {
    let mut classifier = NeuralClassifier::new();
    classifier.initialize().await.unwrap();

    let test_queries = vec![
        "What are the encryption requirements for cardholder data?",
        "Is the system compliant with PCI DSS?",
        "How are access controls related to user authentication?",
        "What security measures are required?",
    ];

    for query in test_queries {
        let start = Instant::now();
        let result = classifier.classify_query(query).await.unwrap();
        let elapsed = start.elapsed();

        // CONSTRAINT-003: Neural classification must be <10ms
        assert!(elapsed.as_millis() < 10, "Neural classification exceeded 10ms: {:?} for query: {}", elapsed, query);
        assert!(result.inference_time_ms < 10, "Reported inference time should be <10ms");
        assert!(result.confidence > 0.0, "Should have some confidence in classification");
    }
}

#[tokio::test]
async fn test_end_to_end_symbolic_reasoning() {
    // Test complete symbolic reasoning pipeline
    let processor = NeurosymbolicProcessor::new().await.unwrap();

    // Load comprehensive compliance requirements
    let requirements = vec![
        RequirementRule {
            id: "pci_3_4".to_string(),
            requirement_type: "encryption_transit".to_string(),
            conditions: vec!["pan_transmission".to_string(), "open_network".to_string()],
            section: "3.4".to_string(),
            confidence: 1.0,
        },
        RequirementRule {
            id: "pci_8_1".to_string(),
            requirement_type: "user_identification".to_string(),
            conditions: vec!["system_access".to_string()],
            section: "8.1".to_string(),
            confidence: 0.98,
        },
    ];

    processor.load_requirements(&requirements).await.unwrap();

    let queries = vec![
        ("What encryption is required for PAN transmission?", "RequirementLookup"),
        ("Is the system compliant with user identification requirements?", "ComplianceCheck"),
        ("How do encryption requirements relate to network security?", "RelationshipQuery"),
    ];

    for (query_text, expected_classification) in queries {
        let query = NeurosymbolicQuery {
            query: query_text.to_string(),
            confidence_threshold: 0.7,
            max_results: 5,
            use_proof_chains: true,
        };

        let result = processor.process_query(query).await.unwrap();

        // Verify symbolic-first processing
        assert_eq!(result.classification, expected_classification);
        assert!(result.proof_chain.is_some(), "Should generate proof chains");
        assert!(!result.sources.is_empty() || result.symbolic_results.is_empty(),
                "Should extract sources when symbolic results exist");

        // Performance verification
        assert!(result.processing_time_ms < 1000, "End-to-end processing should be fast");

        // Quality verification
        assert!(result.confidence > 0.5, "Should maintain reasonable confidence");
        assert!(!result.response.is_empty(), "Should generate meaningful response");
    }
}

#[tokio::test]
async fn test_proof_chain_completeness() {
    let mut datalog_engine = DatalogEngine::new();

    // Add chained requirements for proof testing
    let rules = vec![
        RequirementRule {
            id: "base_requirement".to_string(),
            requirement_type: "data_protection".to_string(),
            conditions: vec!["sensitive_data".to_string()],
            section: "Base".to_string(),
            confidence: 1.0,
        },
        RequirementRule {
            id: "derived_requirement".to_string(),
            requirement_type: "encryption_requirement".to_string(),
            conditions: vec!["data_protection".to_string(), "at_rest".to_string()],
            section: "Derived".to_string(),
            confidence: 0.95,
        },
    ];

    datalog_engine.load_requirements(&rules).unwrap();

    let results = datalog_engine.query("encryption requirement for sensitive data at rest").await.unwrap();

    // Verify complete proof chain
    assert!(!results.is_empty(), "Should find relevant results");

    for result in &results {
        assert!(!result.proof_steps.is_empty(), "Each result should have proof steps");

        for proof_step in &result.proof_steps {
            assert!(!proof_step.rule_applied.is_empty(), "Proof step should reference rule");
            assert!(!proof_step.conclusion.is_empty(), "Should have conclusion");
            assert!(!proof_step.premises.is_empty() || proof_step.conclusion.contains("requirement"),
                    "Should have premises or be a conclusion");
        }
    }
}

#[tokio::test]
async fn test_performance_under_load() {
    let processor = NeurosymbolicProcessor::new().await.unwrap();

    // Load requirements
    let requirements: Vec<RequirementRule> = (0..50).map(|i| {
        RequirementRule {
            id: format!("req_{}", i),
            requirement_type: if i % 3 == 0 { "security".to_string() } else { "compliance".to_string() },
            conditions: vec![format!("condition_{}", i)],
            section: format!("Section {}", i),
            confidence: 0.8 + (i as f64 * 0.01) % 0.2,
        }
    }).collect();

    processor.load_requirements(&requirements).await.unwrap();

    // Test multiple concurrent queries
    let queries: Vec<NeurosymbolicQuery> = (0..20).map(|i| {
        NeurosymbolicQuery {
            query: format!("What are the requirements for condition_{}?", i),
            confidence_threshold: 0.7,
            max_results: 10,
            use_proof_chains: i % 2 == 0, // Vary proof chain usage
        }
    }).collect();

    let start = Instant::now();

    // Process queries sequentially (concurrent would need different test design)
    for query in queries {
        let result = processor.process_query(query).await.unwrap();

        // Each query should meet performance constraints
        assert!(result.processing_time_ms < 1000, "Individual query exceeded time limit");
    }

    let total_elapsed = start.elapsed();

    // Total processing should be reasonable for 20 queries
    assert!(total_elapsed.as_millis() < 10000, "Total processing time excessive: {:?}", total_elapsed);

    // Check metrics
    let metrics = processor.get_metrics().await;
    assert!(metrics.queries_processed >= 20, "Should have processed all queries");
    assert!(metrics.avg_processing_time_ms < 500.0, "Average processing time should be reasonable");
}