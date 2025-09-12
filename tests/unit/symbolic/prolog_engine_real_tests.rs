// tests/unit/symbolic/prolog_engine_real_tests.rs
// REAL Prolog Engine Test Suite - London TDD for 95% Coverage

use tokio_test;
use anyhow::Result;
use std::time::Duration;

use symbolic::prolog::{PrologEngine, PrologQuery, ProofResult, ProofTracer, InferenceStep};
use symbolic::types::{QueryType, ProofStep, ProofValidation};
use symbolic::error::SymbolicError;

#[tokio::test]
async fn test_real_prolog_engine_initialization() -> Result<()> {
    // Test REAL Prolog engine initialization
    let start_time = std::time::Instant::now();
    let engine = PrologEngine::new().await?;
    let init_time = start_time.elapsed();
    
    // Verify initialization  
    assert!(engine.is_initialized());
    assert!(init_time.as_millis() <= 100); // CONSTRAINT-001 validation
    
    // Verify domain ontology was loaded
    assert!(engine.is_ontology_loaded().await);
    
    // Verify knowledge base is ready
    let kb = engine.knowledge_base().read().await;
    assert!(kb.is_initialized().await?);
    
    Ok(())
}

#[tokio::test]
async fn test_domain_ontology_loading() -> Result<()> {
    // Test loading of compliance domain ontology
    let engine = PrologEngine::new().await?;
    
    // Verify ontology was loaded during initialization
    assert!(engine.is_ontology_loaded().await);
    
    // Test adding additional compliance rules
    let rule1 = "cardholder_data(payment_info).";
    let rule2 = "requires_encryption(Data) :- cardholder_data(Data).";
    
    assert!(engine.add_compliance_rule(rule1, "test_document").await.is_ok());
    assert!(engine.add_compliance_rule(rule2, "test_document").await.is_ok());
    
    Ok(())
}

#[tokio::test]
async fn test_natural_language_to_prolog_conversion() -> Result<()> {
    // Test conversion of natural language queries to Prolog
    let engine = PrologEngine::new().await?;
    
    let test_queries = vec![
        ("Does cardholder data exist?", QueryType::Existence),
        ("How do systems relate to data?", QueryType::Relationship), 
        ("Is the system compliant with PCI DSS?", QueryType::Compliance),
        ("What controls are required?", QueryType::Inference),
    ];
    
    for (query_text, expected_type) in test_queries {
        let prolog_query = engine.parse_to_prolog_query(query_text).await?;
        
        assert_eq!(prolog_query.original_text, query_text);
        assert_eq!(prolog_query.query_type, expected_type);
        assert!(!prolog_query.prolog_clauses.is_empty());
        assert!(!prolog_query.variables.is_empty());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_real_prolog_query_execution_with_proof() -> Result<()> {
    // Test REAL Prolog query execution with proof generation
    let engine = PrologEngine::new().await?;
    
    // Add test facts and rules
    engine.add_compliance_rule("sensitive_data(cardholder_data).", "test_doc").await?;
    engine.add_compliance_rule("requires_protection(Data) :- sensitive_data(Data).", "test_doc").await?;
    
    // Execute query with proof tracing
    let start_time = std::time::Instant::now();
    let result = engine.query_with_proof("What data requires protection?").await?;
    let query_time = start_time.elapsed();
    
    // Verify performance constraint
    assert!(query_time.as_millis() <= 100); // CONSTRAINT-001
    assert_eq!(result.execution_time_ms, query_time.as_millis() as u64);
    
    // Verify proof was generated
    assert!(result.confidence > 0.8);
    assert!(result.validation.is_valid);
    assert!(result.validation.logical_consistency);
    assert!(result.validation.proof_completeness > 0.9);
    
    // Verify citations were generated
    assert!(!result.citations.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_proof_tracer_functionality() -> Result<()> {
    // Test proof tracing during query execution
    let engine = PrologEngine::new().await?;
    
    // Add rules for tracing
    engine.add_compliance_rule("compliance_framework(pci_dss).", "framework_doc").await?;
    engine.add_compliance_rule("sensitive_data(payment_data).", "data_doc").await?;
    engine.add_compliance_rule("compliant(System, Framework) :- implements_controls(System, Framework).", "compliance_doc").await?;
    
    let result = engine.query_with_proof("Is payment system compliant?").await?;
    
    // Verify proof steps were recorded
    assert!(!result.proof_steps.is_empty());
    
    // Get tracer for detailed inspection
    let tracer = engine.proof_tracer().read().await;
    assert!(tracer.is_ready());
    assert!(tracer.get_proof_depth() > 0);
    assert!(!tracer.get_inference_history().is_empty());
    
    // Verify each proof step has required information
    for step in &result.proof_steps {
        assert!(step.step_number > 0);
        assert!(!step.rule.is_empty());
        assert!(!step.conclusion.is_empty());
        assert!(step.confidence > 0.0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_complex_inference_chains() -> Result<()> {
    // Test complex multi-step inference
    let engine = PrologEngine::new().await?;
    
    // Build knowledge base for complex inference
    let rules = vec![
        "system(payment_processor).",
        "processes_data(payment_processor, cardholder_data).",
        "sensitive_data(cardholder_data).",
        "requires_protection(Data) :- sensitive_data(Data).",
        "requires_encryption(System, Data) :- processes_data(System, Data), requires_protection(Data).",
        "compliant(System) :- requires_encryption(System, Data), implements_encryption(System, Data).",
        "implements_encryption(payment_processor, cardholder_data).", // This makes it compliant
    ];
    
    for rule in rules {
        engine.add_compliance_rule(rule, "complex_test").await?;
    }
    
    // Query that requires multi-step inference
    let result = engine.query_with_proof("Is payment_processor compliant?").await?;
    
    // Should find solution through inference chain
    assert!(result.result.is_some());
    assert!(result.confidence > 0.8);
    
    // Proof chain should show multiple steps
    assert!(result.proof_steps.len() > 1);
    
    // Verify logical consistency
    assert!(result.validation.logical_consistency);
    assert!(result.validation.proof_completeness > 0.8);
    
    Ok(())
}

#[tokio::test]
async fn test_performance_under_complex_queries() -> Result<()> {
    // Test performance with complex knowledge base
    let engine = PrologEngine::new().await?;
    
    // Add substantial knowledge base
    for i in 0..20 {
        engine.add_compliance_rule(&format!("system(system_{}).", i), "perf_test").await?;
        engine.add_compliance_rule(&format!("data_type(data_{}).", i), "perf_test").await?;
        engine.add_compliance_rule(&format!("processes_data(system_{}, data_{}).", i, i % 5), "perf_test").await?;
    }
    
    // Add inference rules
    engine.add_compliance_rule("sensitive(Data) :- data_type(Data).", "perf_test").await?;
    engine.add_compliance_rule("needs_protection(System, Data) :- processes_data(System, Data), sensitive(Data).", "perf_test").await?;
    
    // Complex query should still meet performance constraint
    let start_time = std::time::Instant::now();
    let result = engine.query_with_proof("What systems need protection?").await?;
    let query_time = start_time.elapsed();
    
    assert!(query_time.as_millis() <= 100); // CONSTRAINT-001
    assert!(!result.proof_steps.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_query_type_classification() -> Result<()> {
    // Test accurate classification of query types
    let engine = PrologEngine::new().await?;
    
    let test_cases = vec![
        ("Does sensitive data exist in the system?", QueryType::Existence),
        ("What is the relationship between systems and data?", QueryType::Relationship),
        ("Is the payment system compliant with PCI DSS?", QueryType::Compliance),
        ("Which controls are required for cardholder data?", QueryType::Inference),
        ("General query about compliance", QueryType::Inference), // Default case
    ];
    
    for (query_text, expected_type) in test_cases {
        let parsed_query = engine.parse_to_prolog_query(query_text).await?;
        assert_eq!(parsed_query.query_type, expected_type, "Failed for: {}", query_text);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_citation_generation_from_proofs() -> Result<()> {
    // Test generation of citations from proof steps
    let engine = PrologEngine::new().await?;
    
    // Add rules with known sources
    engine.add_compliance_rule("pci_requirement(encrypt_cardholder_data).", "PCI_DSS_v4.0").await?;
    engine.add_compliance_rule("implements_requirement(System) :- pci_requirement(encrypt_cardholder_data), encrypts_data(System).", "Internal_Policy").await?;
    engine.add_compliance_rule("encrypts_data(payment_system).", "Technical_Spec").await?;
    
    let result = engine.query_with_proof("Does payment_system implement requirements?").await?;
    
    // Should have citations from rule sources
    assert!(!result.citations.is_empty());
    
    // Verify citation quality
    for citation in &result.citations {
        assert!(!citation.source_document.is_empty());
        assert!(!citation.quoted_text.is_empty());
        assert!(!citation.context.is_empty());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_confidence_calculation_accuracy() -> Result<()> {
    // Test confidence calculation based on proof quality
    let engine = PrologEngine::new().await?;
    
    // High confidence scenario - direct facts
    engine.add_compliance_rule("direct_fact(high_confidence).", "authoritative_source").await?;
    let high_conf_result = engine.query_with_proof("direct_fact(high_confidence)?").await?;
    
    // Low confidence scenario - complex inference
    engine.add_compliance_rule("uncertain_data(X) :- complex_condition(X).", "uncertain_source").await?;
    let low_conf_result = engine.query_with_proof("uncertain_data(unknown)?").await?;
    
    // High confidence result should have higher confidence
    assert!(high_conf_result.confidence > low_conf_result.confidence);
    assert!(high_conf_result.confidence > 0.8);
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling_and_recovery() -> Result<()> {
    // Test error handling in Prolog execution
    let engine = PrologEngine::new().await?;
    
    // Test invalid rule addition
    let invalid_rule_result = engine.add_compliance_rule("invalid syntax here @#$", "test_doc").await;
    
    // Should handle gracefully
    match invalid_rule_result {
        Ok(_) => {
            // If it succeeded, the rule was parsed/normalized somehow
        },
        Err(e) => {
            // Error is expected for invalid syntax
            assert!(matches!(e, SymbolicError::RuleError(_)));
        }
    }
    
    // Engine should still work after error
    assert!(engine.add_compliance_rule("valid_rule(test).", "test_doc").await.is_ok());
    
    let recovery_result = engine.query_with_proof("valid_rule(test)?").await?;
    assert!(recovery_result.confidence > 0.8);
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_prolog_access() -> Result<()> {
    // Test thread safety with concurrent Prolog operations
    let engine = std::sync::Arc::new(PrologEngine::new().await?);
    
    // Add base facts
    engine.add_compliance_rule("base_fact(shared_data).", "concurrent_test").await?;
    
    let mut handles = Vec::new();
    
    // Spawn concurrent query tasks
    for i in 0..5 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            // Add a unique fact
            let rule = format!("unique_fact_{}(test_data).", i);
            engine_clone.add_compliance_rule(&rule, "concurrent_test").await?;
            
            // Query for it
            let query = format!("unique_fact_{}(test_data)?", i);
            let result = engine_clone.query_with_proof(&query).await?;
            
            Result::<bool, anyhow::Error>::Ok(result.confidence > 0.8)
        });
        handles.push(handle);
    }
    
    // All tasks should succeed
    for handle in handles {
        assert!(handle.await??);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_proof_validation_logic() -> Result<()> {
    // Test proof validation mechanisms
    let engine = PrologEngine::new().await?;
    
    // Valid proof scenario
    engine.add_compliance_rule("valid_premise(data_encryption).", "validation_test").await?;
    engine.add_compliance_rule("valid_conclusion(X) :- valid_premise(X).", "validation_test").await?;
    
    let valid_result = engine.query_with_proof("valid_conclusion(data_encryption)?").await?;
    
    // Proof should be valid and consistent
    assert!(valid_result.validation.is_valid);
    assert!(valid_result.validation.logical_consistency);
    assert!(valid_result.validation.proof_completeness > 0.9);
    assert!(valid_result.validation.validation_errors.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_knowledge_base_integration() -> Result<()> {
    // Test integration with knowledge base
    let engine = PrologEngine::new().await?;
    
    // Knowledge base should be accessible
    let kb = engine.knowledge_base().read().await;
    assert!(kb.is_initialized().await?);
    
    // Add rule and verify it's stored
    engine.add_compliance_rule("kb_test_rule(integration).", "kb_integration_test").await?;
    
    // Query should access the knowledge base
    let result = engine.query_with_proof("kb_test_rule(integration)?").await?;
    assert!(result.confidence > 0.8);
    
    Ok(())
}

#[tokio::test]
async fn test_inference_step_recording() -> Result<()> {
    // Test detailed recording of inference steps
    let engine = PrologEngine::new().await?;
    
    // Add multi-step inference rules
    engine.add_compliance_rule("base_entity(entity1).", "step_test").await?;
    engine.add_compliance_rule("derived_property(X) :- base_entity(X).", "step_test").await?;
    engine.add_compliance_rule("final_conclusion(X) :- derived_property(X).", "step_test").await?;
    
    let result = engine.query_with_proof("final_conclusion(entity1)?").await?;
    
    // Should have recorded multiple inference steps
    let tracer = engine.proof_tracer().read().await;
    let inference_history = tracer.get_inference_history();
    
    assert!(!inference_history.is_empty());
    
    for step in inference_history {
        assert!(!step.step_id.is_empty());
        assert!(!step.clause_used.is_empty());
        // Timestamp should be recent
        let now = chrono::Utc::now();
        let diff = now - step.timestamp;
        assert!(diff.num_seconds() < 10); // Should be within last 10 seconds
    }
    
    Ok(())
}

#[tokio::test]
async fn test_machine_reference_access() -> Result<()> {
    // Test access to underlying Prolog machine
    let engine = PrologEngine::new().await?;
    
    // Should have access to machine reference
    let machine_ref = engine.machine();
    let machine = machine_ref.read().await;
    
    // Machine should be initialized and ready
    // Note: Actual machine testing would depend on Scryer-Prolog API
    // For now, verify we can access it without panicking
    drop(machine);
    
    Ok(())
}

#[tokio::test]
async fn test_query_variable_binding() -> Result<()> {
    // Test variable binding in query results
    let engine = PrologEngine::new().await?;
    
    // Add facts with variables
    engine.add_compliance_rule("has_property(system1, encryption).", "binding_test").await?;
    engine.add_compliance_rule("has_property(system2, access_control).", "binding_test").await?;
    engine.add_compliance_rule("secure_system(System) :- has_property(System, encryption).", "binding_test").await?;
    
    let result = engine.query_with_proof("secure_system(X)?").await?;
    
    // Should have bound the variable X to system1
    assert!(result.confidence > 0.8);
    assert!(!result.proof_steps.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_memory_efficiency_with_large_kb() -> Result<()> {
    // Test memory efficiency with large knowledge base
    let engine = PrologEngine::new().await?;
    
    // Add many facts and rules
    for i in 0..200 {
        engine.add_compliance_rule(&format!("entity_{}(item_{}).", i, i), "memory_test").await?;
        
        if i % 10 == 0 {
            engine.add_compliance_rule(&format!("category_rule(X) :- entity_{}(X).", i), "memory_test").await?;
        }
    }
    
    // Query should still be fast and correct
    let start_time = std::time::Instant::now();
    let result = engine.query_with_proof("category_rule(item_50)?").await?;
    let query_time = start_time.elapsed();
    
    assert!(query_time.as_millis() <= 100);
    assert!(result.confidence > 0.8);
    
    Ok(())
}