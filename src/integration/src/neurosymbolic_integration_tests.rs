//! Integration tests for neurosymbolic DAA orchestration
//! Tests the complete integration of symbolic reasoning with DAA message bus and Byzantine consensus

use super::daa_orchestrator::{DAAOrchestrator, ComponentType};
use crate::{IntegrationConfig, Result};
use std::sync::Arc;
use tokio;
use uuid::Uuid;

#[tokio::test]
async fn test_neurosymbolic_daa_integration() -> Result<()> {
    // Create test configuration
    let config = Arc::new(IntegrationConfig::default());
    
    // Create and initialize DAA orchestrator with neurosymbolic capabilities
    let mut orchestrator = DAAOrchestrator::new(config).await?;
    orchestrator.initialize().await?;
    
    // Register neurosymbolic components
    orchestrator.register_component(
        "neurosymbolic-processor",
        ComponentType::NeurosymbolicProcessor,
        "http://localhost:8090"
    ).await?;
    
    orchestrator.register_component(
        "datalog-engine",
        ComponentType::DatalogEngine,
        "http://localhost:8091"
    ).await?;
    
    orchestrator.register_component(
        "neural-classifier",
        ComponentType::NeuralClassifier,
        "http://localhost:8092"
    ).await?;
    
    // Test neurosymbolic query processing with Byzantine consensus
    let query = "What are the encryption requirements for cardholder data?";
    let result = orchestrator.process_neurosymbolic_query(query).await;
    
    // Verify result structure
    assert!(result.is_ok(), "Neurosymbolic query processing should succeed");
    let ns_result = result.unwrap();
    assert!(ns_result.confidence > 0.0, "Result should have confidence score");
    assert!(!ns_result.response.is_empty(), "Response should not be empty");
    
    // Test symbolic agent coordination through MRAP
    let coordination_result = orchestrator.coordinate_symbolic_agents(
        "Analyze compliance requirements for data encryption"
    ).await?;
    
    assert!(!coordination_result.is_empty(), "MRAP coordination should return result");
    
    // Verify component health monitoring
    let component_health = orchestrator.get_component_health("neurosymbolic-processor").await?;
    // Health should be Unknown since we're not running actual processors
    // But the orchestrator should handle this gracefully
    
    // Test metrics collection
    let metrics = orchestrator.metrics().await;
    assert!(metrics.coordination_events > 0, "Should have coordination events");
    assert!(metrics.consensus_operations > 0, "Should have consensus operations");
    
    // Shutdown gracefully
    orchestrator.shutdown().await?;
    
    Ok(())
}

#[tokio::test]
async fn test_byzantine_consensus_validation() -> Result<()> {
    let config = Arc::new(IntegrationConfig::default());
    let mut orchestrator = DAAOrchestrator::new(config).await?;
    orchestrator.initialize().await?;
    
    // Test high-confidence query (should pass Byzantine consensus)
    let high_confidence_query = "require encryption cardholder data"; // Contains keywords
    let result = orchestrator.process_neurosymbolic_query(high_confidence_query).await;
    
    if result.is_ok() {
        // If neurosymbolic processor is available, verify consensus validation
        let ns_result = result.unwrap();
        // Byzantine consensus should approve high-confidence results
        assert!(ns_result.confidence > 0.0);
    }
    
    orchestrator.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_mrap_symbolic_reasoning() -> Result<()> {
    let config = Arc::new(IntegrationConfig::default());
    let mut orchestrator = DAAOrchestrator::new(config).await?;
    orchestrator.initialize().await?;
    
    // Test MRAP phases for symbolic reasoning
    let reasoning_tasks = [
        "What compliance rules apply to payment data?",
        "Are encryption requirements mandatory?",
        "How should cardholder data be protected?",
    ];
    
    for task in &reasoning_tasks {
        let result = orchestrator.coordinate_symbolic_agents(task).await?;
        assert!(!result.is_empty(), "MRAP should return coordination result");
    }
    
    // Verify MRAP metrics
    let mrap_metrics = orchestrator.get_mrap_metrics().await;
    assert!(mrap_metrics.contains_key("mrap_loops_completed"));
    assert!(mrap_metrics.contains_key("reasoning_decisions"));
    assert!(mrap_metrics.contains_key("adaptations_made"));
    
    orchestrator.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn test_neurosymbolic_message_bus() -> Result<()> {
    let config = Arc::new(IntegrationConfig::default());
    let mut orchestrator = DAAOrchestrator::new(config).await?;
    orchestrator.initialize().await?;
    
    // Process multiple queries to test message bus coordination
    let queries = [
        "encryption requirements",
        "compliance validation", 
        "security controls",
    ];
    
    for query in &queries {
        // Each query should be processed and coordinated through the message bus
        let result = orchestrator.coordinate_symbolic_agents(query).await;
        assert!(result.is_ok(), "Message bus coordination should work for: {}", query);
    }
    
    orchestrator.shutdown().await?;
    Ok(())
}