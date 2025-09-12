// tests/integration/symbolic_integration_tests.rs
// REAL Engine Integration Test Suite - End-to-End Testing

use tokio_test;
use std::time::Instant;
use anyhow::Result;

use symbolic::datalog::{DatalogEngine};
use symbolic::prolog::{PrologEngine};
use symbolic::logic_parser::{LogicParser};
use symbolic::types::{RequirementType, Citation, ProofValidation};

#[tokio::test]
async fn test_complete_requirement_processing_pipeline() -> Result<()> {
    // Test complete pipeline: Natural Language -> Logic -> Datalog -> Query -> Results
    
    let parser = LogicParser::new().await?;
    let datalog_engine = DatalogEngine::new().await?;
    let prolog_engine = PrologEngine::new().await?;
    
    let requirements = vec![
        "Payment card data MUST be encrypted when stored in databases",
        "Access to sensitive systems SHALL be restricted to authorized personnel only", 
        "All financial transactions MUST be logged and monitored for audit purposes",
        "Encryption keys SHOULD be managed using hardware security modules",
        "Customer personal information MAY be anonymized for analytics",
    ];
    
    let mut all_rules = Vec::new();
    
    // Process each requirement through the complete pipeline
    for (i, requirement) in requirements.iter().enumerate() {
        println!("Processing requirement {}: {}", i + 1, requirement);
        
        // Step 1: Parse natural language to logic structure
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        
        // Verify parsing quality
        assert!(parsed.confidence > 0.7);
        assert!(!parsed.subject.is_empty());
        assert!(!parsed.predicate.is_empty());
        assert!(matches!(parsed.requirement_type, 
            RequirementType::Must | RequirementType::Should | RequirementType::May));
        
        println!("  Parsed: subject={}, predicate={}, confidence={:.2}", 
                parsed.subject, parsed.predicate, parsed.confidence);
        
        // Step 2: Convert to Datalog rule
        let datalog_rule = parsed.to_datalog_rule()?;
        println!("  Datalog: {}", datalog_rule);
        
        // Step 3: Compile and add to engine
        let compiled_rule = DatalogEngine::compile_requirement_to_rule(requirement).await?;
        datalog_engine.add_rule(compiled_rule.clone()).await?;
        all_rules.push(compiled_rule);
        
        // Step 4: Add corresponding Prolog facts for cross-validation
        let prolog_fact = format!("requirement_{}(requirement_text({})).", 
                                 i, requirement.replace("\"", "'"));
        prolog_engine.add_compliance_rule(&prolog_fact, "integration_test").await?;
        
        println!("  Added rule {} to both engines", i + 1);
    }
    
    // Step 5: Test cross-engine queries and validation
    
    // Query Datalog engine
    let datalog_results = datalog_engine.query("What data must be encrypted?").await?;
    assert!(!datalog_results.results.is_empty());
    assert!(datalog_results.execution_time_ms <= 100);
    assert!(datalog_results.confidence > 0.8);
    
    println!("Datalog query returned {} results with confidence {:.2}", 
            datalog_results.results.len(), datalog_results.confidence);
    
    // Query Prolog engine for inference
    let prolog_results = prolog_engine.query_with_proof("requirement_0(X)?").await?;
    assert!(prolog_results.confidence > 0.8);
    assert!(prolog_results.validation.is_valid);
    
    println!("Prolog inference confidence: {:.2}, proof steps: {}", 
            prolog_results.confidence, prolog_results.proof_steps.len());
    
    // Step 6: Cross-validate results between engines
    assert_eq!(all_rules.len(), requirements.len());
    assert!(datalog_engine.rule_count() == requirements.len());
    
    println!("Integration pipeline completed successfully!");
    
    Ok(())
}

#[tokio::test]
async fn test_real_world_compliance_scenario() -> Result<()> {
    // Test realistic PCI DSS compliance scenario
    
    let parser = LogicParser::new().await?;
    let datalog_engine = DatalogEngine::new().await?;
    let prolog_engine = PrologEngine::new().await?;
    
    // Real PCI DSS requirements
    let pci_requirements = vec![
        "Install and maintain a firewall configuration to protect cardholder data",
        "Do not use vendor-supplied defaults for system passwords and other security parameters", 
        "Protect stored cardholder data with strong cryptography",
        "Encrypt transmission of cardholder data across open, public networks",
        "Use and regularly update anti-virus software or programs",
        "Develop and maintain secure systems and applications",
        "Restrict access to cardholder data by business need-to-know",
        "Assign a unique ID to each person with computer access",
        "Restrict physical access to cardholder data",
        "Track and monitor all access to network resources and cardholder data",
        "Regularly test security systems and processes",
        "Maintain a policy that addresses information security for all personnel"
    ];
    
    let scenario_start = Instant::now();
    
    // Process all requirements
    let mut processed_rules = Vec::new();
    for (i, requirement) in pci_requirements.iter().enumerate() {
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        let compiled = DatalogEngine::compile_requirement_to_rule(requirement).await?;
        
        datalog_engine.add_rule(compiled.clone()).await?;
        processed_rules.push((parsed, compiled));
        
        // Add to Prolog knowledge base
        let prolog_rule = format!("pci_requirement_{}('{}').", i + 1, 
                                 requirement.replace("'", "''"));
        prolog_engine.add_compliance_rule(&prolog_rule, "PCI_DSS_v4.0").await?;
    }
    
    let processing_time = scenario_start.elapsed();
    println!("Processed {} PCI requirements in {}ms", 
            pci_requirements.len(), processing_time.as_millis());
    
    // Test compliance queries
    let compliance_queries = vec![
        "What data protection measures are required?",
        "Which requirements involve cardholder data?",
        "What access controls must be implemented?",
        "Which systems require encryption?",
    ];
    
    for query in compliance_queries {
        let query_start = Instant::now();
        let result = datalog_engine.query(query).await?;
        let query_time = query_start.elapsed();
        
        assert!(query_time.as_millis() <= 100);
        assert!(!result.results.is_empty());
        assert!(result.confidence > 0.7);
        
        println!("Query '{}' returned {} results in {}ms", 
                query, result.results.len(), query_time.as_millis());
    }
    
    // Test cross-requirement inference with Prolog
    let inference_queries = vec![
        "pci_requirement_3(X)?", // Encryption requirement
        "pci_requirement_7(X)?", // Access control requirement
    ];
    
    for query in inference_queries {
        let inference_result = prolog_engine.query_with_proof(query).await?;
        
        assert!(inference_result.confidence > 0.8);
        assert!(inference_result.validation.is_valid);
        assert!(!inference_result.citations.is_empty());
        
        println!("Prolog inference for '{}': confidence {:.2}, {} citations", 
                query, inference_result.confidence, inference_result.citations.len());
    }
    
    println!("Real-world compliance scenario completed successfully!");
    
    Ok(())
}

#[tokio::test]
async fn test_complex_conditional_logic_processing() -> Result<()> {
    // Test complex conditional requirements with both engines
    
    let parser = LogicParser::new().await?;
    let datalog_engine = DatalogEngine::new().await?;
    let prolog_engine = PrologEngine::new().await?;
    
    let conditional_requirements = vec![
        "If cardholder data is stored electronically, then it must be encrypted using AES-256 or equivalent strong cryptography",
        "When processing payment transactions, systems must validate card data and maintain transaction logs unless the transaction value is less than $10",
        "Unless otherwise specified in section 4.1.2, all authentication data must be rendered unreadable after authorization",
        "Systems shall implement multi-factor authentication for remote access, except for system components that are not connected to the cardholder data environment",
    ];
    
    for (i, requirement) in conditional_requirements.iter().enumerate() {
        println!("Processing conditional requirement {}: {}", i + 1, requirement);
        
        // Parse complex conditional logic
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        
        // Should detect conditional structure
        assert!(parsed.confidence > 0.6); // May be lower due to complexity
        assert!(!parsed.subject.is_empty());
        
        // Convert to engines
        let compiled = DatalogEngine::compile_requirement_to_rule(requirement).await?;
        datalog_engine.add_rule(compiled).await?;
        
        // Add conditional logic to Prolog
        let prolog_conditional = format!(
            "conditional_requirement_{}(Condition, Action) :- condition_{}(Condition), action_{}(Action).",
            i, i, i
        );
        prolog_engine.add_compliance_rule(&prolog_conditional, "conditional_test").await?;
        
        // Add specific condition and action facts
        prolog_engine.add_compliance_rule(
            &format!("condition_{}(data_storage_electronic).", i), "conditional_test"
        ).await?;
        prolog_engine.add_compliance_rule(
            &format!("action_{}(encrypt_with_aes256).", i), "conditional_test"
        ).await?;
        
        println!("  Processed conditional logic for requirement {}", i + 1);
    }
    
    // Test conditional queries
    let result = datalog_engine.query("What happens when data is stored electronically?").await?;
    assert!(result.execution_time_ms <= 100);
    
    let prolog_result = prolog_engine.query_with_proof("conditional_requirement_0(data_storage_electronic, Action)?").await?;
    assert!(prolog_result.confidence > 0.8);
    assert!(!prolog_result.proof_steps.is_empty());
    
    println!("Complex conditional logic processing completed!");
    
    Ok(())
}

#[tokio::test]
async fn test_multi_framework_compliance_integration() -> Result<()> {
    // Test integration across multiple compliance frameworks
    
    let parser = LogicParser::new().await?;
    let datalog_engine = DatalogEngine::new().await?;
    let prolog_engine = PrologEngine::new().await?;
    
    // Multi-framework requirements
    let frameworks = vec![
        ("PCI_DSS", vec![
            "Cardholder data must be encrypted at rest",
            "Payment processing systems must implement access controls",
        ]),
        ("ISO_27001", vec![
            "Information security policies must be established and maintained", 
            "Access to information systems must be controlled and monitored",
        ]),
        ("SOC2", vec![
            "Logical access controls must prevent unauthorized access to data",
            "System availability must be monitored and maintained",
        ]),
        ("NIST", vec![
            "Cryptographic mechanisms must protect data confidentiality",
            "Audit logs must capture security-relevant events",
        ]),
    ];
    
    let mut framework_rules = std::collections::HashMap::new();
    
    // Process each framework
    for (framework_name, requirements) in frameworks {
        println!("Processing {} framework", framework_name);
        let mut rules_for_framework = Vec::new();
        
        for (i, requirement) in requirements.iter().enumerate() {
            // Parse requirement
            let parsed = parser.parse_requirement_to_logic(requirement).await?;
            assert!(parsed.confidence > 0.7);
            
            // Add to Datalog engine
            let compiled = DatalogEngine::compile_requirement_to_rule(requirement).await?;
            datalog_engine.add_rule(compiled.clone()).await?;
            rules_for_framework.push(compiled);
            
            // Add framework-specific Prolog facts
            let framework_fact = format!("framework_requirement({}, {}, '{}').", 
                                        framework_name.to_lowercase(), i + 1, requirement);
            prolog_engine.add_compliance_rule(&framework_fact, framework_name).await?;
            
            // Add framework taxonomy
            let taxonomy_rule = format!("belongs_to_framework({}, {}).", 
                                       format!("req_{}_{}", framework_name.to_lowercase(), i + 1),
                                       framework_name.to_lowercase());
            prolog_engine.add_compliance_rule(&taxonomy_rule, framework_name).await?;
        }
        
        framework_rules.insert(framework_name.to_string(), rules_for_framework);
        println!("  Added {} requirements for {}", requirements.len(), framework_name);
    }
    
    // Test cross-framework queries
    let cross_framework_queries = vec![
        "What encryption requirements exist across frameworks?",
        "Which frameworks address access control?", 
        "What monitoring requirements are specified?",
    ];
    
    for query in cross_framework_queries {
        let result = datalog_engine.query(query).await?;
        assert!(result.execution_time_ms <= 100);
        assert!(!result.results.is_empty());
        
        println!("Cross-framework query '{}' found {} results", 
                query, result.results.len());
    }
    
    // Test framework-specific Prolog inference
    let framework_inference = prolog_engine.query_with_proof(
        "framework_requirement(pci_dss, 1, Requirement)?"
    ).await?;
    
    assert!(framework_inference.confidence > 0.8);
    assert!(!framework_inference.citations.is_empty());
    
    // Test compliance gap analysis
    let gap_analysis = prolog_engine.query_with_proof(
        "belongs_to_framework(Req, Framework)?"
    ).await?;
    
    assert!(gap_analysis.validation.is_valid);
    assert!(!gap_analysis.proof_steps.is_empty());
    
    println!("Multi-framework compliance integration completed!");
    
    Ok(())
}

#[tokio::test]
async fn test_performance_under_integrated_load() -> Result<()> {
    // Test performance when all components work together under load
    
    let parser = LogicParser::new().await?;
    let datalog_engine = DatalogEngine::new().await?;
    let prolog_engine = PrologEngine::new().await?;
    
    let load_start = Instant::now();
    
    // Generate substantial integrated load
    let requirement_count = 100;
    let query_count = 50;
    
    // Add requirements
    for i in 0..requirement_count {
        let requirement = format!(
            "Entity_{} MUST implement security_control_{} when processing data_type_{} in environment_{}",
            i, i % 10, i % 5, i % 3
        );
        
        // Full pipeline processing
        let parsed = parser.parse_requirement_to_logic(&requirement).await?;
        let compiled = DatalogEngine::compile_requirement_to_rule(&requirement).await?;
        
        datalog_engine.add_rule(compiled).await?;
        
        let prolog_fact = format!("load_requirement_{}(entity_{}, control_{}, data_{}).", 
                                 i, i, i % 10, i % 5);
        prolog_engine.add_compliance_rule(&prolog_fact, "load_test").await?;
        
        // Periodic performance check
        if i % 20 == 0 && i > 0 {
            let check_start = Instant::now();
            let check_result = datalog_engine.query(&format!("What controls exist for data_{}?", i % 5)).await?;
            let check_time = check_start.elapsed();
            
            assert!(check_time.as_millis() <= 100, "Performance degraded at rule {}", i);
            assert!(check_result.confidence > 0.8);
            
            println!("Performance check at rule {}: {}ms", i, check_time.as_millis());
        }
    }
    
    let loading_time = load_start.elapsed();
    println!("Loaded {} requirements in {}ms", requirement_count, loading_time.as_millis());
    
    // Execute queries under load
    let query_start = Instant::now();
    let mut query_results = Vec::new();
    
    for i in 0..query_count {
        let queries = vec![
            format!("What entities use control_{}?", i % 10),
            format!("Which data_types require protection in environment_{}?", i % 3),
            format!("What security controls are implemented?"),
        ];
        
        for query in queries {
            let result = datalog_engine.query(&query).await?;
            assert!(result.execution_time_ms <= 100);
            query_results.push(result);
        }
        
        // Prolog queries
        if i % 10 == 0 {
            let prolog_query = format!("load_requirement_{}(Entity, Control, Data)?", i);
            let prolog_result = prolog_engine.query_with_proof(&prolog_query).await?;
            assert!(prolog_result.confidence > 0.8);
        }
    }
    
    let query_time = query_start.elapsed();
    let total_queries = query_count * 3 + (query_count / 10);
    
    println!("Executed {} queries in {}ms", total_queries, query_time.as_millis());
    println!("Average query time: {}ms", query_time.as_millis() / total_queries as u128);
    
    // Verify all results meet quality standards
    for result in &query_results {
        assert!(result.confidence > 0.8);
        assert!(!result.citations.is_empty() || !result.results.is_empty());
    }
    
    // Final system health check
    let health_check_start = Instant::now();
    let final_datalog_query = datalog_engine.query("What is the current system state?").await?;
    let final_prolog_query = prolog_engine.query_with_proof("load_requirement_50(X, Y, Z)?").await?;
    let health_check_time = health_check_start.elapsed();
    
    assert!(health_check_time.as_millis() <= 100);
    assert!(final_datalog_query.confidence > 0.8);
    assert!(final_prolog_query.confidence > 0.8);
    
    println!("Integration load test completed successfully!");
    println!("Total runtime: {}ms", load_start.elapsed().as_millis());
    println!("System remains healthy and responsive under load");
    
    Ok(())
}

#[tokio::test]
async fn test_proof_chain_validation_across_engines() -> Result<()> {
    // Test proof chain generation and validation across both engines
    
    let datalog_engine = DatalogEngine::new().await?;
    let prolog_engine = PrologEngine::new().await?;
    
    // Build knowledge base for proof validation
    let knowledge_rules = vec![
        "sensitive_data(cardholder_information)",
        "sensitive_data(authentication_data)", 
        "requires_protection(Data) :- sensitive_data(Data)",
        "encryption_required(Data) :- requires_protection(Data), stored_electronically(Data)",
        "compliant_system(System) :- encryption_required(Data), processes_data(System, Data), implements_encryption(System)",
    ];
    
    for rule in &knowledge_rules {
        let compiled = DatalogEngine::compile_requirement_to_rule(rule).await?;
        datalog_engine.add_rule(compiled).await?;
        
        prolog_engine.add_compliance_rule(rule, "proof_validation").await?;
    }
    
    // Add specific facts
    let facts = vec![
        "stored_electronically(cardholder_information)",
        "processes_data(payment_system, cardholder_information)",
        "implements_encryption(payment_system)",
    ];
    
    for fact in &facts {
        let compiled = DatalogEngine::compile_requirement_to_rule(fact).await?;
        datalog_engine.add_rule(compiled).await?;
        
        prolog_engine.add_compliance_rule(fact, "proof_validation").await?;
    }
    
    // Test proof generation in Datalog
    let datalog_result = datalog_engine.query("What systems are compliant?").await?;
    assert!(datalog_result.confidence > 0.8);
    assert!(!datalog_result.proof_chain.is_empty());
    assert!(!datalog_result.citations.is_empty());
    
    // Test proof generation in Prolog
    let prolog_result = prolog_engine.query_with_proof("compliant_system(payment_system)?").await?;
    assert!(prolog_result.confidence > 0.8);
    assert!(!prolog_result.proof_steps.is_empty());
    assert!(prolog_result.validation.is_valid);
    assert!(prolog_result.validation.logical_consistency);
    
    // Cross-validate proof chains
    println!("Datalog proof chain has {} steps", datalog_result.proof_chain.len());
    println!("Prolog proof chain has {} steps", prolog_result.proof_steps.len());
    
    // Both should reach similar conclusions
    assert!(datalog_result.confidence > 0.8);
    assert!(prolog_result.confidence > 0.8);
    
    // Validate proof quality
    for step in &datalog_result.proof_chain {
        assert!(step.step_number > 0);
        assert!(!step.rule.is_empty());
        assert!(step.confidence > 0.0);
    }
    
    for step in &prolog_result.proof_steps {
        assert!(step.step_number > 0);
        assert!(!step.rule.is_empty());
        assert!(step.confidence > 0.0);
    }
    
    println!("Proof chain validation completed successfully!");
    
    Ok(())
}

#[tokio::test]
async fn test_error_recovery_in_integrated_system() -> Result<()> {
    // Test system resilience and error recovery when components interact
    
    let parser = LogicParser::new().await?;
    let datalog_engine = DatalogEngine::new().await?;
    let prolog_engine = PrologEngine::new().await?;
    
    // Test invalid input handling
    let invalid_inputs = vec![
        "",
        "??Invalid requirement syntax!!",
        "MUST MUST MUST without structure",
        "Random words that make no sense in compliance context",
    ];
    
    for invalid_input in invalid_inputs {
        // System should handle gracefully without crashing
        let parse_result = parser.parse_requirement_to_logic(invalid_input).await;
        
        match parse_result {
            Ok(parsed) => {
                // If parsing succeeded, confidence should be very low
                assert!(parsed.confidence < 0.5);
            },
            Err(_) => {
                // Error is acceptable for invalid input
                println!("Handled invalid input gracefully: {}", invalid_input);
            }
        }
        
        // Try to compile anyway - should handle gracefully
        let compile_result = DatalogEngine::compile_requirement_to_rule(invalid_input).await;
        match compile_result {
            Ok(rule) => {
                // If compilation succeeded, try to add it
                let add_result = datalog_engine.add_rule(rule).await;
                // May succeed or fail, but shouldn't crash
            },
            Err(_) => {
                println!("Handled invalid compilation gracefully");
            }
        }
    }
    
    // System should still work after error conditions
    let recovery_requirement = "Recovery test: Data MUST be handled correctly";
    let recovery_parsed = parser.parse_requirement_to_logic(recovery_requirement).await?;
    let recovery_compiled = DatalogEngine::compile_requirement_to_rule(recovery_requirement).await?;
    
    assert!(datalog_engine.add_rule(recovery_compiled).await.is_ok());
    assert!(prolog_engine.add_compliance_rule("recovery_test(successful).", "recovery").await.is_ok());
    
    // Queries should still work
    let recovery_query = datalog_engine.query("What recovery tests exist?").await?;
    assert!(recovery_query.execution_time_ms <= 100);
    
    let prolog_recovery = prolog_engine.query_with_proof("recovery_test(successful)?").await?;
    assert!(prolog_recovery.confidence > 0.8);
    
    println!("Error recovery testing completed successfully!");
    
    Ok(())
}