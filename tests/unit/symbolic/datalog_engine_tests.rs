// tests/unit/symbolic/datalog_engine_tests.rs
// London TDD test suite for DatalogEngine - CONSTRAINT-001 validation

use std::time::{Duration, Instant};
use tokio_test;
use criterion::black_box;
use anyhow::Result;

// Mock imports - will be implemented after tests pass
use symbolic::datalog::{DatalogEngine, DatalogRule, QueryResult, ParsedRequirement};
use symbolic::types::{RequirementType, Priority};

#[tokio::test]
async fn test_datalog_engine_initialization() -> Result<()> {
    // London TDD: Test the initialization of DatalogEngine
    let engine = DatalogEngine::new().await?;
    
    assert!(engine.is_initialized());
    assert_eq!(engine.rule_count(), 0);
    assert!(engine.performance_metrics().is_some());
    
    Ok(())
}

#[tokio::test]
async fn test_requirement_compilation_mandatory_rule() -> Result<()> {
    // Test MUST/SHALL requirement compilation
    let requirement_text = "Cardholder data MUST be encrypted when stored at rest";
    
    let rule = DatalogEngine::compile_requirement_to_rule(requirement_text).await?;
    
    assert_eq!(rule.rule_type, RequirementType::Must);
    assert!(rule.text.contains("requires_encryption"));
    assert!(rule.text.contains("cardholder_data"));
    assert!(rule.text.contains("stored_at_rest"));
    assert!(rule.text.ends_with('.'));
    
    // Validate rule syntax
    assert!(DatalogEngine::validate_rule_syntax(&rule.text).await?);
    
    Ok(())
}

#[tokio::test]
async fn test_requirement_compilation_recommended_rule() -> Result<()> {
    // Test SHOULD requirement compilation
    let requirement_text = "Organizations SHOULD implement multi-factor authentication";
    
    let rule = DatalogEngine::compile_requirement_to_rule(requirement_text).await?;
    
    assert_eq!(rule.rule_type, RequirementType::Should);
    assert!(rule.text.contains("recommended_"));
    assert!(rule.text.contains("multi_factor_authentication"));
    
    Ok(())
}

#[tokio::test]
async fn test_requirement_compilation_optional_rule() -> Result<()> {
    // Test MAY requirement compilation
    let requirement_text = "Systems MAY use hardware security modules for key storage";
    
    let rule = DatalogEngine::compile_requirement_to_rule(requirement_text).await?;
    
    assert_eq!(rule.rule_type, RequirementType::May);
    assert!(rule.text.contains("optional_"));
    assert!(rule.text.contains("hardware_security_modules"));
    
    Ok(())
}

#[tokio::test]
async fn test_complex_requirement_with_conditions() -> Result<()> {
    // Test complex requirement with multiple conditions
    let requirement_text = "Access to cardholder data MUST be restricted to authorized personnel only when conducting legitimate business activities";
    
    let rule = DatalogEngine::compile_requirement_to_rule(requirement_text).await?;
    
    assert!(rule.text.contains("authorized_personnel"));
    assert!(rule.text.contains("legitimate_business_activities"));
    assert!(rule.dependencies.len() > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_rule_addition_success() -> Result<()> {
    // Test successful rule addition to engine
    let mut engine = DatalogEngine::new().await?;
    let requirement_text = "All systems MUST maintain audit logs";
    let rule = DatalogEngine::compile_requirement_to_rule(requirement_text).await?;
    
    engine.add_rule(rule.clone()).await?;
    
    assert_eq!(engine.rule_count(), 1);
    assert!(engine.rule_cache().contains_key(&rule.id));
    
    Ok(())
}

#[tokio::test]
async fn test_rule_conflict_detection() -> Result<()> {
    // Test conflicting rule detection
    let mut engine = DatalogEngine::new().await?;
    
    let rule1 = DatalogRule {
        id: "test_rule".to_string(),
        text: "secure_data(X) :- encrypted(X).".to_string(),
        source_requirement: "Data must be encrypted".to_string(),
        rule_type: RequirementType::Must,
        created_at: chrono::Utc::now(),
        dependencies: vec![],
    };
    
    let conflicting_rule = DatalogRule {
        id: "test_rule".to_string(), // Same ID
        text: "secure_data(X) :- not_encrypted(X).".to_string(), // Conflicting logic
        source_requirement: "Data must not be encrypted".to_string(),
        rule_type: RequirementType::Must,
        created_at: chrono::Utc::now(),
        dependencies: vec![],
    };
    
    engine.add_rule(rule1).await?;
    
    // This should fail due to conflict
    let result = engine.add_rule(conflicting_rule).await;
    assert!(result.is_err());
    assert_eq!(engine.rule_count(), 1);
    
    Ok(())
}

#[tokio::test]
async fn test_query_execution_performance_constraint() -> Result<()> {
    // CONSTRAINT-001: MUST achieve <100ms logic query response time
    let mut engine = DatalogEngine::new().await?;
    
    // Add sample rules
    let rules = vec![
        "encrypted(cardholder_data).",
        "stored_at_rest(cardholder_data).",
        "requires_encryption(X) :- cardholder_data(X), stored_at_rest(X).",
        "compliant(System, pci_dss) :- implements_encryption(System).",
    ];
    
    for rule_text in rules {
        let rule = DatalogRule {
            id: format!("rule_{}", uuid::Uuid::new_v4()),
            text: rule_text.to_string(),
            source_requirement: "Test requirement".to_string(),
            rule_type: RequirementType::Must,
            created_at: chrono::Utc::now(),
            dependencies: vec![],
        };
        engine.add_rule(rule).await?;
    }
    
    let query = "requires_encryption(cardholder_data)?";
    
    let start_time = Instant::now();
    let result = engine.query(query).await?;
    let execution_time = start_time.elapsed();
    
    // CRITICAL CONSTRAINT: Must be under 100ms
    assert!(execution_time < Duration::from_millis(100), 
            "Query execution took {}ms, exceeds 100ms limit", 
            execution_time.as_millis());
    
    assert!(!result.results.is_empty());
    assert!(result.confidence > 0.95); // Symbolic reasoning should have high confidence
    assert!(!result.proof_chain.is_empty()); // Must have complete proof chain
    
    Ok(())
}

#[tokio::test]
async fn test_proof_chain_generation() -> Result<()> {
    // CONSTRAINT-001: MUST generate complete proof chains for all answers
    let mut engine = DatalogEngine::new().await?;
    
    // Add rules that create an inference chain
    let rules = vec![
        "cardholder_data(credit_card_number).",
        "stored_at_rest(credit_card_number).",
        "requires_encryption(X) :- cardholder_data(X), stored_at_rest(X).",
        "security_violation(X) :- requires_encryption(X), not_encrypted(X).",
    ];
    
    for (i, rule_text) in rules.iter().enumerate() {
        let rule = DatalogRule {
            id: format!("proof_rule_{}", i),
            text: rule_text.to_string(),
            source_requirement: format!("Test requirement {}", i),
            rule_type: RequirementType::Must,
            created_at: chrono::Utc::now(),
            dependencies: vec![],
        };
        engine.add_rule(rule).await?;
    }
    
    let query = "requires_encryption(credit_card_number)?";
    let result = engine.query(query).await?;
    
    // Validate complete proof chain
    assert!(!result.proof_chain.is_empty());
    assert!(result.proof_chain.len() >= 2); // Should have multiple inference steps
    
    // Validate proof chain completeness
    let proof_steps = &result.proof_chain;
    assert!(proof_steps.iter().any(|step| step.rule.contains("cardholder_data")));
    assert!(proof_steps.iter().any(|step| step.rule.contains("stored_at_rest")));
    assert!(proof_steps.iter().any(|step| step.rule.contains("requires_encryption")));
    
    // Each step should have source citations
    for step in proof_steps {
        assert!(!step.source_section.is_empty());
        assert!(!step.conditions.is_empty() || step.rule.ends_with('.'));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_query_caching_mechanism() -> Result<()> {
    // Test query result caching for performance optimization
    let mut engine = DatalogEngine::new().await?;
    
    // Add a rule
    let rule = DatalogRule {
        id: "cache_test_rule".to_string(),
        text: "test_predicate(X) :- input_data(X).".to_string(),
        source_requirement: "Cache test requirement".to_string(),
        rule_type: RequirementType::Must,
        created_at: chrono::Utc::now(),
        dependencies: vec![],
    };
    engine.add_rule(rule).await?;
    
    let query = "test_predicate(some_data)?";
    
    // First query - should be computed
    let start_time = Instant::now();
    let result1 = engine.query(query).await?;
    let first_query_time = start_time.elapsed();
    
    // Second identical query - should be cached
    let start_time = Instant::now();
    let result2 = engine.query(query).await?;
    let second_query_time = start_time.elapsed();
    
    // Cache hit should be significantly faster
    assert!(second_query_time < first_query_time);
    assert!(second_query_time < Duration::from_millis(10)); // Cached queries should be very fast
    
    // Results should be identical
    assert_eq!(result1.results.len(), result2.results.len());
    assert_eq!(result1.confidence, result2.confidence);
    
    // Verify cache hit metrics were updated
    let metrics = engine.performance_metrics().read().await;
    assert!(metrics.cache_hit_count > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_natural_language_requirement_parsing() -> Result<()> {
    // Test parsing of natural language requirements into structured components
    let requirement_text = "All payment card industry systems must implement strong cryptographic protocols to protect sensitive authentication data during transmission over open networks";
    
    let parsed = DatalogEngine::parse_requirement_structure(requirement_text).await?;
    
    assert_eq!(parsed.requirement_type, RequirementType::Must);
    assert!(parsed.entities.iter().any(|e| e.name.contains("payment_card_industry")));
    assert!(parsed.actions.iter().any(|a| a.verb.contains("implement")));
    assert!(parsed.conditions.iter().any(|c| c.text.contains("transmission")));
    assert!(parsed.confidence > 0.8);
    
    Ok(())
}

#[tokio::test]
async fn test_cross_reference_extraction() -> Result<()> {
    // Test extraction of cross-references from requirements
    let requirement_text = "Systems must comply with section 3.2.1 and reference requirement REQ-4.1.2 for encryption standards";
    
    let parsed = DatalogEngine::parse_requirement_structure(requirement_text).await?;
    
    assert!(parsed.cross_references.len() >= 2);
    assert!(parsed.cross_references.iter().any(|cr| cr.reference.contains("3.2.1")));
    assert!(parsed.cross_references.iter().any(|cr| cr.reference.contains("REQ-4.1.2")));
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling_invalid_syntax() -> Result<()> {
    // Test proper error handling for invalid rule syntax
    let mut engine = DatalogEngine::new().await?;
    
    let invalid_rule = DatalogRule {
        id: "invalid_rule".to_string(),
        text: "invalid syntax without proper structure".to_string(), // Invalid Datalog syntax
        source_requirement: "Invalid requirement".to_string(),
        rule_type: RequirementType::Must,
        created_at: chrono::Utc::now(),
        dependencies: vec![],
    };
    
    let result = engine.add_rule(invalid_rule).await;
    assert!(result.is_err());
    
    // Engine should remain in valid state
    assert_eq!(engine.rule_count(), 0);
    assert!(engine.is_initialized());
    
    Ok(())
}

#[tokio::test]
async fn test_performance_metrics_tracking() -> Result<()> {
    // Test that performance metrics are properly tracked
    let mut engine = DatalogEngine::new().await?;
    
    // Add some rules and execute queries
    let rule = DatalogRule {
        id: "metrics_test_rule".to_string(),
        text: "metrics_test(X) :- test_condition(X).".to_string(),
        source_requirement: "Metrics test requirement".to_string(),
        rule_type: RequirementType::Must,
        created_at: chrono::Utc::now(),
        dependencies: vec![],
    };
    engine.add_rule(rule).await?;
    
    let _result = engine.query("metrics_test(data)?").await?;
    
    let metrics = engine.performance_metrics().read().await;
    assert!(metrics.total_queries > 0);
    assert!(metrics.average_query_time_ms > 0.0);
    assert!(metrics.total_rules_added > 0);
    
    Ok(())
}

// Benchmark tests for performance validation
#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{Criterion, BenchmarkId};
    
    #[tokio::test]
    async fn benchmark_query_performance_under_load() -> Result<()> {
        // Stress test to ensure <100ms performance under load
        let mut engine = DatalogEngine::new().await?;
        
        // Add 100 rules to simulate realistic load
        for i in 0..100 {
            let rule = DatalogRule {
                id: format!("load_test_rule_{}", i),
                text: format!("rule_{}(X) :- condition_{}(X).", i, i),
                source_requirement: format!("Load test requirement {}", i),
                rule_type: RequirementType::Must,
                created_at: chrono::Utc::now(),
                dependencies: vec![],
            };
            engine.add_rule(rule).await?;
        }
        
        // Execute multiple queries concurrently
        let mut handles = vec![];
        for i in 0..50 {
            let query = format!("rule_{}(test_data)?", i % 100);
            let engine_clone = engine.clone(); // Assuming Arc wrapper
            
            handles.push(tokio::spawn(async move {
                let start_time = Instant::now();
                let _result = engine_clone.query(&query).await?;
                let duration = start_time.elapsed();
                
                // Each individual query must still be under 100ms
                assert!(duration < Duration::from_millis(100));
                Ok::<(), anyhow::Error>(())
            }));
        }
        
        // Wait for all queries to complete
        for handle in handles {
            handle.await??;
        }
        
        Ok(())
    }
}

// Edge case tests
#[tokio::test]
async fn test_empty_requirement_handling() -> Result<()> {
    let result = DatalogEngine::compile_requirement_to_rule("").await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_malformed_query_handling() -> Result<()> {
    let engine = DatalogEngine::new().await?;
    
    let result = engine.query("malformed query without proper syntax").await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_unicode_requirement_handling() -> Result<()> {
    let requirement_text = "Les données de carte de crédit DOIVENT être chiffrées au repos";
    
    let rule = DatalogEngine::compile_requirement_to_rule(requirement_text).await?;
    assert!(!rule.text.is_empty());
    assert!(rule.rule_type == RequirementType::Must);
    
    Ok(())
}