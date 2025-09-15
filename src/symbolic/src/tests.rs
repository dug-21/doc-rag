//! Basic tests for symbolic reasoning components

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use crate::{DatalogEngine, TypesRequirementRule as RequirementRule};

#[tokio::test]
async fn test_basic_datalog_functionality() -> Result<()> {
    let mut datalog = DatalogEngine::new();

    // Create a test requirement
    let requirement = RequirementRule {
        id: "test_rule".to_string(),
        requirement_type: "encryption_requirement".to_string(),
        conditions: vec!["cardholder_data".to_string()],
        section: "3.2.1".to_string(),
        confidence: 0.95,
    };

    // Test basic requirement loading
    datalog.load_requirements(&[requirement])?;

    // Test query processing
    let query = "What encryption is required for cardholder data?";
    let results = datalog.query(query).await?;

    assert!(!results.is_empty(), "Should find at least one result");

    Ok(())
}

#[tokio::test]
async fn test_complex_compliance_query() -> Result<()> {
    let mut datalog = DatalogEngine::new();

    let requirements = vec![
        RequirementRule {
            id: "pci_3_4".to_string(),
            requirement_type: "encryption".to_string(),
            conditions: vec!["transmission".to_string(), "cardholder_data".to_string()],
            section: "3.4".to_string(),
            confidence: 1.0,
        },
        RequirementRule {
            id: "pci_8_1".to_string(),
            requirement_type: "access_control".to_string(),
            conditions: vec!["user_identification".to_string()],
            section: "8.1".to_string(),
            confidence: 0.98,
        },
    ];

    datalog.load_requirements(&requirements)?;

    let query = "Which systems are compliant with all PCI DSS requirements for data protection?";
    let results = datalog.query(query).await?;

    // Basic validation
    for result in &results {
        assert!(result.confidence > 0.0, "Results should have positive confidence");
        assert!(!result.predicate.is_empty(), "Results should have predicate");
    }

    Ok(())
}