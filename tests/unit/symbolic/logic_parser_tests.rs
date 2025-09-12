// tests/unit/symbolic/logic_parser_tests.rs
// London TDD test suite for LogicParser - Natural language to logic conversion

use std::collections::HashMap;
use tokio_test;
use anyhow::Result;

// Mock imports - will be implemented after tests pass
use symbolic::logic_parser::{LogicParser, ParsedLogic, LogicElement, LogicOperator};
use symbolic::types::{RequirementType, Entity, Action, Condition, CrossReference};

#[tokio::test]
async fn test_logic_parser_initialization() -> Result<()> {
    // London TDD: Test initialization of LogicParser with domain knowledge
    let parser = LogicParser::new().await?;
    
    assert!(parser.is_initialized());
    assert!(parser.domain_ontology().is_loaded());
    assert!(parser.linguistic_patterns().len() > 0);
    assert!(parser.entity_recognizer().is_ready());
    
    Ok(())
}

#[tokio::test]
async fn test_simple_requirement_parsing() -> Result<()> {
    // Test parsing of simple mandatory requirement
    let parser = LogicParser::new().await?;
    
    let requirement = "Cardholder data MUST be encrypted";
    let parsed = parser.parse_requirement_to_logic(requirement).await?;
    
    assert_eq!(parsed.requirement_type, RequirementType::Must);
    assert_eq!(parsed.subject, "cardholder_data");
    assert_eq!(parsed.predicate, "requires_encryption");
    assert!(parsed.conditions.is_empty());
    assert!(parsed.confidence > 0.9);
    
    // Generated logic should be valid Datalog
    let datalog_rule = parsed.to_datalog_rule()?;
    assert_eq!(datalog_rule, "requires_encryption(cardholder_data).");
    
    Ok(())
}

#[tokio::test]
async fn test_conditional_requirement_parsing() -> Result<()> {
    // Test parsing of requirement with conditions
    let parser = LogicParser::new().await?;
    
    let requirement = "Payment data MUST be encrypted when stored in databases";
    let parsed = parser.parse_requirement_to_logic(requirement).await?;
    
    assert_eq!(parsed.subject, "payment_data");
    assert_eq!(parsed.predicate, "requires_encryption");
    assert_eq!(parsed.conditions.len(), 1);
    assert_eq!(parsed.conditions[0].condition_type, "storage_location");
    assert_eq!(parsed.conditions[0].value, "databases");
    
    let datalog_rule = parsed.to_datalog_rule()?;
    assert_eq!(datalog_rule, "requires_encryption(payment_data) :- stored_in_databases(payment_data).");
    
    Ok(())
}

#[tokio::test]
async fn test_multi_condition_requirement_parsing() -> Result<()> {
    // Test complex requirement with multiple conditions
    let parser = LogicParser::new().await?;
    
    let requirement = "Access to sensitive data MUST be restricted to authorized personnel during business hours when conducting legitimate activities";
    let parsed = parser.parse_requirement_to_logic(requirement).await?;
    
    assert_eq!(parsed.subject, "sensitive_data");
    assert_eq!(parsed.predicate, "requires_access_restriction");
    assert_eq!(parsed.conditions.len(), 3);
    
    // Verify conditions are extracted
    let condition_types: Vec<&str> = parsed.conditions.iter()
        .map(|c| c.condition_type.as_str())
        .collect();
    assert!(condition_types.contains(&"personnel_authorization"));
    assert!(condition_types.contains(&"time_restriction"));
    assert!(condition_types.contains(&"activity_legitimacy"));
    
    let datalog_rule = parsed.to_datalog_rule()?;
    assert!(datalog_rule.contains("authorized_personnel"));
    assert!(datalog_rule.contains("business_hours"));
    assert!(datalog_rule.contains("legitimate_activities"));
    
    Ok(())
}

#[tokio::test]
async fn test_negation_handling() -> Result<()> {
    // Test parsing of requirements with negations
    let parser = LogicParser::new().await?;
    
    let requirement = "Systems MUST NOT store unencrypted cardholder data";
    let parsed = parser.parse_requirement_to_logic(requirement).await?;
    
    assert_eq!(parsed.predicate, "prohibits_storage");
    assert!(parsed.negation_present);
    assert_eq!(parsed.object, "unencrypted_cardholder_data");
    
    let datalog_rule = parsed.to_datalog_rule()?;
    assert!(datalog_rule.contains("not") || datalog_rule.contains("prohibits"));
    
    Ok(())
}

#[tokio::test]
async fn test_exception_clause_parsing() -> Result<()> {
    // Test parsing of requirements with exception clauses
    let parser = LogicParser::new().await?;
    
    let requirement = "All data MUST be encrypted except for test environments lasting less than 24 hours";
    let parsed = parser.parse_requirement_to_logic(requirement).await?;
    
    assert_eq!(parsed.subject, "all_data");
    assert_eq!(parsed.predicate, "requires_encryption");
    assert_eq!(parsed.exceptions.len(), 1);
    
    let exception = &parsed.exceptions[0];
    assert!(exception.condition.contains("test_environment"));
    assert!(exception.condition.contains("duration_less_than"));
    
    let datalog_rule = parsed.to_datalog_rule()?;
    assert!(datalog_rule.contains("not exception_applies"));
    
    Ok(())
}

#[tokio::test]
async fn test_cross_reference_extraction() -> Result<()> {
    // Test extraction of cross-references from requirements
    let parser = LogicParser::new().await?;
    
    let requirement = "Systems must comply with section 3.2.1 and reference requirement REQ-4.1.2 for additional guidance per appendix A.3";
    let parsed = parser.parse_requirement_to_logic(requirement).await?;
    
    assert_eq!(parsed.cross_references.len(), 3);
    
    let references: Vec<&str> = parsed.cross_references.iter()
        .map(|cr| cr.reference.as_str())
        .collect();
    assert!(references.contains(&"3.2.1"));
    assert!(references.contains(&"REQ-4.1.2"));
    assert!(references.contains(&"A.3"));
    
    Ok(())
}

#[tokio::test]
async fn test_entity_recognition_accuracy() -> Result<()> {
    // Test accurate recognition of domain entities
    let parser = LogicParser::new().await?;
    
    let requirement = "Payment Card Industry systems must implement strong cryptographic protocols to protect Primary Account Numbers during network transmission";
    let parsed = parser.parse_requirement_to_logic(requirement).await?;
    
    let entities = parsed.extract_entities();
    
    // Should recognize domain-specific entities
    assert!(entities.iter().any(|e| e.entity_type == "system_type" && e.name.contains("payment_card_industry")));
    assert!(entities.iter().any(|e| e.entity_type == "data_type" && e.name.contains("primary_account_numbers")));
    assert!(entities.iter().any(|e| e.entity_type == "security_control" && e.name.contains("cryptographic_protocols")));
    assert!(entities.iter().any(|e| e.entity_type == "context" && e.name.contains("network_transmission")));
    
    Ok(())
}

#[tokio::test]
async fn test_action_verb_identification() -> Result<()> {
    // Test identification and normalization of action verbs
    let parser = LogicParser::new().await?;
    
    let requirements = vec![
        "Systems must implement security controls",
        "Organizations shall establish access policies", 
        "Entities should maintain audit logs",
        "Applications may utilize hardware tokens",
    ];
    
    for requirement in requirements {
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        let actions = parsed.extract_actions();
        
        assert!(!actions.is_empty());
        
        // Actions should be normalized to standard predicates
        let action_predicates: Vec<&str> = actions.iter()
            .map(|a| a.predicate.as_str())
            .collect();
        
        assert!(action_predicates.iter().any(|p| 
            ["requires_implementation", "requires_establishment", "requires_maintenance", "allows_utilization"]
            .contains(p)
        ));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_modal_verb_requirement_classification() -> Result<()> {
    // Test classification of requirement strength based on modal verbs
    let parser = LogicParser::new().await?;
    
    let test_cases = vec![
        ("Data MUST be encrypted", RequirementType::Must),
        ("Systems SHALL implement controls", RequirementType::Must),
        ("Organizations SHOULD conduct reviews", RequirementType::Should),
        ("Users MAY choose authentication methods", RequirementType::May),
        ("Encryption is REQUIRED for all data", RequirementType::Must),
        ("Regular backups are RECOMMENDED", RequirementType::Should),
    ];
    
    for (requirement, expected_type) in test_cases {
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        assert_eq!(parsed.requirement_type, expected_type);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_complex_sentence_structure_parsing() -> Result<()> {
    // Test parsing of complex sentence structures
    let parser = LogicParser::new().await?;
    
    let complex_requirement = "In order to ensure compliance with payment card industry data security standards, all merchants that store, process, or transmit cardholder data must implement and maintain a secure network architecture with properly configured firewalls and routers that restrict access to cardholder data environments";
    
    let parsed = parser.parse_requirement_to_logic(complex_requirement).await?;
    
    // Should extract key components despite complexity
    assert!(parsed.subjects.contains(&"merchants".to_string()));
    assert!(parsed.predicate.contains("implement"));
    assert!(parsed.conditions.iter().any(|c| c.condition_type == "data_operations"));
    assert!(parsed.conditions.iter().any(|c| c.value.contains("store_process_transmit")));
    
    // Should generate valid logic despite complexity
    let datalog_rule = parsed.to_datalog_rule()?;
    assert!(datalog_rule.len() > 50); // Should be a substantial rule
    assert!(datalog_rule.contains(":-")); // Should have conditions
    
    Ok(())
}

#[tokio::test]
async fn test_domain_specific_terminology_handling() -> Result<()> {
    // Test handling of domain-specific compliance terminology
    let parser = LogicParser::new().await?;
    
    let requirements = vec![
        "CHD must be protected with strong cryptographic algorithms",
        "PAN truncation should mask all but the last 4 digits",
        "HSMs are required for key generation and management",
        "QSAs must validate PCI DSS compliance annually",
    ];
    
    for requirement in requirements {
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        
        // Should expand acronyms and use standardized terms
        let datalog_rule = parsed.to_datalog_rule()?;
        
        if requirement.contains("CHD") {
            assert!(datalog_rule.contains("cardholder_data"));
        }
        if requirement.contains("PAN") {
            assert!(datalog_rule.contains("primary_account_number"));
        }
        if requirement.contains("HSM") {
            assert!(datalog_rule.contains("hardware_security_module"));
        }
        if requirement.contains("QSA") {
            assert!(datalog_rule.contains("qualified_security_assessor"));
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_quantifier_handling() -> Result<()> {
    // Test handling of quantifiers (all, some, any, none)
    let parser = LogicParser::new().await?;
    
    let test_cases = vec![
        ("ALL systems must be encrypted", "forall"),
        ("SOME data may be exempted", "exists"),
        ("ANY unauthorized access is prohibited", "forall"),
        ("NO unencrypted data shall be stored", "not exists"),
    ];
    
    for (requirement, expected_quantifier) in test_cases {
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        
        assert!(parsed.quantifiers.iter().any(|q| q.quantifier_type == expected_quantifier));
        
        let datalog_rule = parsed.to_datalog_rule()?;
        // Quantifiers should influence rule structure
        assert!(datalog_rule.len() > 20); // Should generate substantial rule
    }
    
    Ok(())
}

#[tokio::test]
async fn test_temporal_constraint_parsing() -> Result<()> {
    // Test parsing of temporal constraints
    let parser = LogicParser::new().await?;
    
    let requirement = "Audit logs must be retained for at least 12 months and reviewed monthly during business operations";
    let parsed = parser.parse_requirement_to_logic(requirement).await?;
    
    assert!(parsed.temporal_constraints.len() >= 2);
    
    let constraints: Vec<&str> = parsed.temporal_constraints.iter()
        .map(|tc| tc.constraint_type.as_str())
        .collect();
    assert!(constraints.contains(&"retention_period"));
    assert!(constraints.contains(&"review_frequency"));
    
    let datalog_rule = parsed.to_datalog_rule()?;
    assert!(datalog_rule.contains("retention_months(12)"));
    assert!(datalog_rule.contains("review_frequency(monthly)"));
    
    Ok(())
}

#[tokio::test]
async fn test_conditional_logic_parsing() -> Result<()> {
    // Test parsing of if-then-else conditional logic
    let parser = LogicParser::new().await?;
    
    let requirement = "If cardholder data is stored electronically, then it must be encrypted, otherwise it should be physically secured";
    let parsed = parser.parse_requirement_to_logic(requirement).await?;
    
    assert!(parsed.has_conditional_logic);
    assert_eq!(parsed.conditional_structure.condition_type, "if_then_else");
    
    let if_condition = &parsed.conditional_structure.if_condition;
    assert!(if_condition.contains("stored_electronically"));
    
    let then_clause = &parsed.conditional_structure.then_clause;
    assert!(then_clause.contains("requires_encryption"));
    
    let else_clause = &parsed.conditional_structure.else_clause;
    assert!(else_clause.as_ref().unwrap().contains("physical_security"));
    
    Ok(())
}

#[tokio::test]
async fn test_performance_under_load() -> Result<()> {
    // Test parsing performance with multiple requirements
    let parser = LogicParser::new().await?;
    
    let requirements = vec![
        "Systems must encrypt all sensitive data using AES-256 encryption",
        "Access controls shall be implemented for all cardholder data environments", 
        "Regular vulnerability scanning should be performed on all network components",
        "Incident response procedures must be documented and tested annually",
        "Physical access to secure areas should be restricted to authorized personnel only",
    ];
    
    let start_time = std::time::Instant::now();
    
    let mut parsed_results = Vec::new();
    for requirement in requirements {
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        parsed_results.push(parsed);
    }
    
    let total_time = start_time.elapsed();
    
    // Should parse multiple requirements quickly
    assert!(total_time < std::time::Duration::from_millis(500));
    
    // All results should be valid
    for parsed in parsed_results {
        assert!(parsed.confidence > 0.8);
        assert!(!parsed.subject.is_empty());
        assert!(!parsed.predicate.is_empty());
        
        let datalog_rule = parsed.to_datalog_rule()?;
        assert!(datalog_rule.len() > 10);
        assert!(datalog_rule.ends_with('.'));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_ambiguity_resolution() -> Result<()> {
    // Test resolution of ambiguous requirements
    let parser = LogicParser::new().await?;
    
    let ambiguous_requirement = "The system must be secure and reliable";
    let parsed = parser.parse_requirement_to_logic(ambiguous_requirement).await?;
    
    // Parser should flag ambiguity and provide multiple interpretations
    assert!(parsed.ambiguity_detected);
    assert!(parsed.alternative_interpretations.len() > 1);
    assert!(parsed.confidence < 0.8); // Lower confidence due to ambiguity
    
    // Each interpretation should be valid
    for interpretation in &parsed.alternative_interpretations {
        assert!(!interpretation.subject.is_empty());
        assert!(!interpretation.predicate.is_empty());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_multilingual_requirement_parsing() -> Result<()> {
    // Test parsing of non-English requirements (if supported)
    let parser = LogicParser::new().await?;
    
    let french_requirement = "Les données de carte de crédit DOIVENT être chiffrées au repos";
    
    let result = parser.parse_requirement_to_logic(french_requirement).await;
    
    if result.is_ok() {
        let parsed = result?;
        // Should handle non-English requirements
        assert!(parsed.subject.contains("cardholder_data") || parsed.subject.contains("donnees"));
        assert!(parsed.predicate.contains("encryption") || parsed.predicate.contains("chiffrement"));
    } else {
        // If multilingual support not implemented, should gracefully handle
        assert!(result.is_err());
    }
    
    Ok(())
}

// Edge cases and error handling
#[tokio::test]
async fn test_empty_requirement_handling() -> Result<()> {
    let parser = LogicParser::new().await?;
    
    let result = parser.parse_requirement_to_logic("").await;
    assert!(result.is_err());
    
    Ok(())
}

#[tokio::test]
async fn test_malformed_requirement_handling() -> Result<()> {
    let parser = LogicParser::new().await?;
    
    let malformed_requirements = vec![
        ";;;;;;;;",
        "MUST MUST MUST",
        "Random words without structure meaning",
        "123 456 789",
    ];
    
    for malformed in malformed_requirements {
        let result = parser.parse_requirement_to_logic(malformed).await;
        
        if result.is_ok() {
            let parsed = result.unwrap();
            // Should have very low confidence
            assert!(parsed.confidence < 0.3);
        } else {
            // Or should gracefully error
            assert!(result.is_err());
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_extremely_long_requirement_handling() -> Result<()> {
    let parser = LogicParser::new().await?;
    
    // Create extremely long requirement (1000+ words)
    let long_requirement = "Systems ".repeat(500) + "must be encrypted.";
    
    let result = parser.parse_requirement_to_logic(&long_requirement).await;
    
    // Should either handle gracefully or error appropriately
    if result.is_ok() {
        let parsed = result.unwrap();
        assert!(!parsed.subject.is_empty());
        assert!(!parsed.predicate.is_empty());
    } else {
        // Should fail gracefully without crashing
        assert!(result.is_err());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_special_characters_handling() -> Result<()> {
    let parser = LogicParser::new().await?;
    
    let requirement_with_special_chars = "Data must be encrypted with AES-256 (Advanced Encryption Standard) @ 256-bit key length & secure protocols!";
    let parsed = parser.parse_requirement_to_logic(requirement_with_special_chars).await?;
    
    // Should handle special characters gracefully
    assert!(!parsed.subject.is_empty());
    assert!(parsed.predicate.contains("encryption"));
    
    let datalog_rule = parsed.to_datalog_rule()?;
    // Special characters should be normalized or removed
    assert!(!datalog_rule.contains("@"));
    assert!(!datalog_rule.contains("&"));
    assert!(!datalog_rule.contains("!"));
    
    Ok(())
}