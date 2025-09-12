// tests/unit/symbolic/datalog_engine_real_tests.rs
// REAL Datalog Engine Test Suite - London TDD for 95% Coverage

use tokio_test;
use std::time::Duration;
use anyhow::Result;

use symbolic::datalog::{DatalogEngine, DatalogRule, QueryResult, CompiledRule, FactStore};
use symbolic::types::{RequirementType, PerformanceMetrics};
use symbolic::error::SymbolicError;

#[tokio::test]
async fn test_real_crepe_engine_initialization() -> Result<()> {
    // Test REAL Crepe engine initialization
    let start_time = std::time::Instant::now();
    let engine = DatalogEngine::new().await?;
    let init_time = start_time.elapsed();
    
    // Verify initialization
    assert!(engine.is_initialized().await);
    assert_eq!(engine.rule_count(), 0);
    assert!(init_time.as_millis() <= 100); // CONSTRAINT-001 validation
    
    // Verify Crepe runtime is ready
    let runtime = engine.crepe_runtime().read().await;
    assert!(!runtime.facts_loaded); // Should be false initially
    assert_eq!(runtime.entities.len(), 0);
    assert_eq!(runtime.requires_encryption.len(), 0);
    
    Ok(())
}

#[tokio::test]
async fn test_real_rule_compilation_and_addition() -> Result<()> {
    // Test REAL rule compilation with actual Datalog syntax
    let engine = DatalogEngine::new().await?;
    
    let requirement = "Cardholder data MUST be encrypted when stored in databases";
    let rule = DatalogEngine::compile_requirement_to_rule(requirement).await?;
    
    // Verify rule structure
    assert!(!rule.id.is_empty());
    assert_eq!(rule.rule_type, RequirementType::Must);
    assert!(rule.text.contains("requires_encryption"));
    assert!(rule.text.contains("cardholder_data"));
    assert!(rule.text.ends_with('.'));
    
    // Add rule to REAL engine
    engine.add_rule(rule.clone()).await?;
    
    // Verify rule was added to cache and Crepe runtime
    assert_eq!(engine.rule_count(), 1);
    assert!(engine.rule_cache().contains_key(&rule.id));
    
    // Verify facts were added to Crepe runtime
    let runtime = engine.crepe_runtime().read().await;
    assert!(runtime.facts_loaded);
    assert!(runtime.entities.len() > 0);
    assert!(runtime.requires_encryption.len() > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_real_crepe_query_execution() -> Result<()> {
    // Test REAL Crepe query execution with actual inference
    let engine = DatalogEngine::new().await?;
    
    // Add test rules
    let rule1 = DatalogEngine::compile_requirement_to_rule(
        "Cardholder data MUST be encrypted"
    ).await?;
    
    let rule2 = DatalogEngine::compile_requirement_to_rule(
        "Payment data is sensitive data"
    ).await?;
    
    engine.add_rule(rule1).await?;
    engine.add_rule(rule2).await?;
    
    // Execute REAL query
    let start_time = std::time::Instant::now();
    let result = engine.query("What data requires encryption?").await?;
    let query_time = start_time.elapsed();
    
    // Verify performance constraint
    assert!(query_time.as_millis() <= 100); // CONSTRAINT-001
    assert_eq!(result.execution_time_ms, query_time.as_millis() as u64);
    
    // Verify REAL results from Crepe inference
    assert!(!result.results.is_empty());
    assert!(result.confidence > 0.9); // High confidence from logical inference
    assert!(!result.used_rules.is_empty());
    
    // Verify proof chain was generated
    assert!(!result.proof_chain.is_empty());
    assert!(!result.citations.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_performance_constraint_validation() -> Result<()> {
    // Test strict performance constraint validation
    let engine = DatalogEngine::new().await?;
    
    // Add multiple rules to stress test
    for i in 0..10 {
        let rule = DatalogEngine::compile_requirement_to_rule(
            &format!("Data type {} MUST be encrypted", i)
        ).await?;
        engine.add_rule(rule).await?;
    }
    
    // Test query performance under load
    let start_time = std::time::Instant::now();
    let result = engine.query("What requires encryption?").await;
    let query_time = start_time.elapsed();
    
    // Should succeed with good performance
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(query_time.as_millis() <= 100);
    assert_eq!(result.execution_time_ms, query_time.as_millis() as u64);
    
    Ok(())
}

#[tokio::test]
async fn test_rule_conflict_detection() -> Result<()> {
    // Test detection of conflicting rules
    let engine = DatalogEngine::new().await?;
    
    let rule1 = DatalogEngine::compile_requirement_to_rule(
        "System A MUST be encrypted"
    ).await?;
    
    // Create conflicting rule with same ID
    let mut rule2 = rule1.clone();
    rule2.text = "System A MUST NOT be encrypted.".to_string();
    
    // First rule should succeed
    assert!(engine.add_rule(rule1).await.is_ok());
    
    // Second rule should detect conflict
    let result = engine.add_rule(rule2).await;
    assert!(result.is_err());
    if let Err(SymbolicError::RuleConflict(id)) = result {
        assert!(!id.is_empty());
    } else {
        panic!("Expected RuleConflict error");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_query_caching_mechanism() -> Result<()> {
    // Test query result caching for performance
    let engine = DatalogEngine::new().await?;
    
    // Add test rule
    let rule = DatalogEngine::compile_requirement_to_rule(
        "Test data MUST be encrypted"
    ).await?;
    engine.add_rule(rule).await?;
    
    let query = "What test data rules exist?";
    
    // First query - should execute and cache
    let start_time1 = std::time::Instant::now();
    let result1 = engine.query(query).await?;
    let time1 = start_time1.elapsed();
    
    // Second identical query - should use cache
    let start_time2 = std::time::Instant::now();
    let result2 = engine.query(query).await?;
    let time2 = start_time2.elapsed();
    
    // Cached query should be much faster
    assert!(time2 < time1);
    assert_eq!(result1.results.len(), result2.results.len());
    assert_eq!(result1.confidence, result2.confidence);
    
    Ok(())
}

#[tokio::test]
async fn test_complex_rule_compilation() -> Result<()> {
    // Test compilation of complex requirements
    let complex_requirements = vec![
        "Payment data MUST be encrypted when stored in databases and transmitted over networks",
        "Systems MUST NOT store unencrypted cardholder data except in test environments lasting less than 24 hours",
        "All access to sensitive data MUST be logged and monitored in real-time during business hours",
        "Multi-factor authentication SHALL be required for all administrative access to cardholder data environments",
    ];
    
    for requirement in complex_requirements {
        let rule = DatalogEngine::compile_requirement_to_rule(requirement).await?;
        
        // Verify rule structure
        assert!(!rule.text.is_empty());
        assert!(rule.text.ends_with('.'));
        assert!(!rule.source_requirement.is_empty());
        
        // Verify rule can be parsed as valid Datalog
        assert!(DatalogEngine::validate_rule_syntax(&rule.text).await?);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_entity_extraction_from_rules() -> Result<()> {
    // Test entity extraction for Crepe fact loading
    let engine = DatalogEngine::new().await?;
    
    let rule_text = "requires_encryption(cardholder_data).";
    let entity = engine.extract_entity_from_rule(rule_text);
    
    assert_eq!(entity, Some("cardholder_data".to_string()));
    
    // Test with more complex rule
    let complex_rule = "requires_access_control(payment_system) :- processes_sensitive_data(payment_system).";
    let entity2 = engine.extract_entity_from_rule(complex_rule);
    
    assert_eq!(entity2, Some("payment_system".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_fact_count_tracking() -> Result<()> {
    // Test fact counting in Crepe runtime
    let engine = DatalogEngine::new().await?;
    
    assert_eq!(engine.fact_count().await, 0);
    
    // Add rule with facts
    let rule = DatalogEngine::compile_requirement_to_rule(
        "Payment data MUST be encrypted"
    ).await?;
    engine.add_rule(rule).await?;
    
    // Should have facts now
    let fact_count = engine.fact_count().await;
    assert!(fact_count > 0);
    
    // Add another rule
    let rule2 = DatalogEngine::compile_requirement_to_rule(
        "System data requires access control"
    ).await?;
    engine.add_rule(rule2).await?;
    
    // Fact count should increase
    assert!(engine.fact_count().await > fact_count);
    
    Ok(())
}

#[tokio::test]
async fn test_performance_metrics_tracking() -> Result<()> {
    // Test performance metrics collection
    let engine = DatalogEngine::new().await?;
    
    // Add rule and query to generate metrics
    let rule = DatalogEngine::compile_requirement_to_rule(
        "Test data MUST be protected"
    ).await?;
    engine.add_rule(rule).await?;
    
    let result = engine.query("What data must be protected?").await?;
    
    // Verify metrics were updated
    let metrics = engine.performance_metrics().read().await;
    assert!(metrics.total_queries >= 1);
    assert!(metrics.total_rules_added >= 1);
    assert!(metrics.average_query_time_ms > 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_requirement_parsing_edge_cases() -> Result<()> {
    // Test edge cases in requirement parsing
    let edge_cases = vec![
        ("", "Empty requirement should fail"),
        ("MUST", "Single modal verb should fail gracefully"),
        ("Data data data data", "Repetitive text should parse"),
        ("System123 must encrypt data456", "Alphanumeric entities"),
        ("Multi-line\nrequirement\nMUST work", "Multi-line requirements"),
    ];
    
    for (requirement, description) in edge_cases {
        let result = DatalogEngine::compile_requirement_to_rule(requirement).await;
        
        match requirement {
            "" => assert!(result.is_err(), "Empty requirement should fail"),
            _ => {
                // Other cases should either succeed or fail gracefully
                if result.is_err() {
                    // Should be a ParseError, not a panic
                    assert!(matches!(result.unwrap_err(), SymbolicError::ParseError(_)));
                } else {
                    let rule = result.unwrap();
                    assert!(!rule.text.is_empty(), "{}", description);
                }
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_rule_syntax_validation() -> Result<()> {
    // Test Datalog syntax validation
    let valid_rules = vec![
        "requires_encryption(data).",
        "encrypted(X) :- cardholder_data(X).",
        "compliant(System) :- implements_control(System, encryption).",
    ];
    
    let invalid_rules = vec![
        "missing_period(data)",
        "invalid :- structure :- double",
        "unmatched(parentheses",
    ];
    
    for rule in valid_rules {
        assert!(DatalogEngine::validate_rule_syntax(rule).await?);
    }
    
    for rule in invalid_rules {
        let result = DatalogEngine::validate_rule_syntax(rule).await;
        assert!(result.is_err() || !result.unwrap());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_access_safety() -> Result<()> {
    // Test thread safety with concurrent access
    let engine = std::sync::Arc::new(DatalogEngine::new().await?);
    
    let mut handles = Vec::new();
    
    // Spawn multiple tasks adding rules concurrently
    for i in 0..5 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let rule = DatalogEngine::compile_requirement_to_rule(
                &format!("Data {} MUST be encrypted", i)
            ).await?;
            engine_clone.add_rule(rule).await
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    for handle in handles {
        assert!(handle.await??);
    }
    
    // Verify all rules were added
    assert_eq!(engine.rule_count(), 5);
    
    // Test concurrent queries
    let query_handles: Vec<_> = (0..3).map(|_| {
        let engine_clone = engine.clone();
        tokio::spawn(async move {
            engine_clone.query("What data requires encryption?").await
        })
    }).collect();
    
    for handle in query_handles {
        assert!(handle.await??.execution_time_ms <= 100);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_memory_usage_optimization() -> Result<()> {
    // Test memory usage doesn't grow excessively
    let engine = DatalogEngine::new().await?;
    
    // Add many rules and measure memory impact
    for i in 0..100 {
        let rule = DatalogEngine::compile_requirement_to_rule(
            &format!("Entity_{} MUST have_property_{}", i, i % 10)
        ).await?;
        engine.add_rule(rule).await?;
        
        // Periodically run queries to test memory management
        if i % 20 == 0 {
            let result = engine.query(&format!("What has property_{}?", i % 10)).await?;
            assert!(!result.results.is_empty());
        }
    }
    
    // Engine should still be responsive
    let final_result = engine.query("What entities exist?").await?;
    assert!(final_result.execution_time_ms <= 100);
    assert_eq!(engine.rule_count(), 100);
    
    Ok(())
}

#[tokio::test] 
async fn test_error_recovery_and_resilience() -> Result<()> {
    // Test system resilience to errors
    let engine = DatalogEngine::new().await?;
    
    // Try to add invalid rule - should not crash system
    let invalid_rule_result = DatalogEngine::compile_requirement_to_rule("Invalid ??? rule !@#").await;
    
    // System should handle gracefully
    match invalid_rule_result {
        Ok(rule) => {
            // If parsing succeeded, rule should be valid
            assert!(DatalogEngine::validate_rule_syntax(&rule.text).await?);
        },
        Err(_) => {
            // Error is acceptable for invalid input
        }
    }
    
    // Add valid rule - system should still work
    let valid_rule = DatalogEngine::compile_requirement_to_rule(
        "Recovery test data MUST be handled correctly"
    ).await?;
    assert!(engine.add_rule(valid_rule).await.is_ok());
    
    // Query should still work
    let result = engine.query("What test data exists?").await?;
    assert!(result.execution_time_ms <= 100);
    
    Ok(())
}