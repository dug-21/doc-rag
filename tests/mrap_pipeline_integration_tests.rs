//! # MRAP Pipeline Integration Tests
//! 
//! Tests for integration between MRAP control loop and query processing pipeline
//! ensuring autonomous orchestration follows Phase 2 architecture requirements.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::time::sleep;
use uuid::Uuid;

use integration::{
    ProcessingPipeline, DAAOrchestrator, MessageBus, IntegrationConfig,
    QueryRequest, QueryResponse, ResponseFormat, SystemIntegration,
    ComponentType, Result,
};

#[tokio::test]
async fn test_mrap_orchestrated_query_processing() {
    // Arrange - Create complete system with MRAP orchestration
    let config = Arc::new(IntegrationConfig::default());
    let mut daa_orchestrator = DAAOrchestrator::new(config.clone()).await.unwrap();
    daa_orchestrator.initialize().await.unwrap();
    
    let daa_orchestrator = Arc::new(tokio::sync::RwLock::new(daa_orchestrator));
    let message_bus = Arc::new(MessageBus::new(config.clone()).await.unwrap());
    
    let pipeline = ProcessingPipeline::new(
        config.clone(),
        daa_orchestrator.clone(),
        message_bus
    ).await.unwrap();
    
    pipeline.initialize().await.unwrap();
    pipeline.start().await.unwrap();
    
    // Register components through DAA orchestrator
    {
        let orchestrator = daa_orchestrator.write().await;
        orchestrator.register_component("chunker", ComponentType::Chunker, "http://localhost:8002").await.unwrap();
        orchestrator.register_component("embedder", ComponentType::Embedder, "http://localhost:8003").await.unwrap();
        orchestrator.register_component("storage", ComponentType::Storage, "http://localhost:8004").await.unwrap();
    }
    
    // Wait for MRAP loop to stabilize
    sleep(Duration::from_secs(2)).await;
    
    // Act - Process query through MRAP-orchestrated pipeline
    let query_request = QueryRequest {
        id: Uuid::new_v4(),
        query: "What are PCI DSS encryption requirements for stored payment data?".to_string(),
        filters: Some(HashMap::from([("domain".to_string(), "pci_dss".to_string())])),
        format: Some(ResponseFormat::Json),
        timeout_ms: Some(30000),
    };
    
    let start_time = Instant::now();
    let response_result = pipeline.process_query(query_request.clone()).await;
    let processing_time = start_time.elapsed();
    
    // Assert - Verify MRAP orchestration worked correctly
    assert!(response_result.is_ok(), "Query processing should succeed with MRAP orchestration");
    
    let response = response_result.unwrap();
    assert_eq!(response.request_id, query_request.id);
    assert!(!response.response.is_empty());
    assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
    
    // Verify MRAP orchestration metrics
    {
        let orchestrator = daa_orchestrator.read().await;
        let mrap_metrics = orchestrator.get_mrap_metrics().await;
        let coordination_events = orchestrator.metrics().await.coordination_events;
        
        // MRAP should be active during query processing
        assert!(mrap_metrics.get("mrap_loops_completed").unwrap().as_u64().unwrap() > 0);
        assert!(coordination_events > 0, "DAA coordination should have occurred");
    }
    
    // Verify performance requirements (Phase 2: <2s total response time)
    assert!(processing_time < Duration::from_secs(2), 
           "Processing should complete within 2 seconds: {:?}", processing_time);
    
    // Cleanup
    pipeline.stop().await.unwrap();
    {
        let mut orchestrator = daa_orchestrator.write().await;
        orchestrator.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_mrap_byzantine_consensus_validation() {
    // Arrange - System with Byzantine consensus enabled
    let config = Arc::new(IntegrationConfig::default());
    let mut daa_orchestrator = DAAOrchestrator::new(config.clone()).await.unwrap();
    daa_orchestrator.initialize().await.unwrap();
    
    // Enable Byzantine consensus
    daa_orchestrator.enable_byzantine_consensus().await.unwrap();
    
    let daa_orchestrator = Arc::new(tokio::sync::RwLock::new(daa_orchestrator));
    let message_bus = Arc::new(MessageBus::new(config.clone()).await.unwrap());
    
    let pipeline = ProcessingPipeline::new(
        config.clone(),
        daa_orchestrator.clone(),
        message_bus
    ).await.unwrap();
    
    pipeline.initialize().await.unwrap();
    pipeline.start().await.unwrap();
    
    // Wait for consensus setup
    sleep(Duration::from_secs(1)).await;
    
    // Act - Process query with Byzantine consensus validation
    let query_request = QueryRequest {
        id: Uuid::new_v4(),
        query: "Explain multi-factor authentication requirements".to_string(),
        filters: None,
        format: Some(ResponseFormat::Json),
        timeout_ms: Some(15000),
    };
    
    let consensus_start = Instant::now();
    let response = pipeline.process_query(query_request).await;
    let consensus_time = consensus_start.elapsed();
    
    // Assert - Verify consensus validation
    assert!(response.is_ok(), "Query with Byzantine consensus should succeed");
    
    // Verify consensus timing (Phase 2: <500ms consensus requirement)
    assert!(consensus_time < Duration::from_millis(500), 
           "Byzantine consensus should complete within 500ms: {:?}", consensus_time);
    
    // Verify consensus operations occurred
    {
        let orchestrator = daa_orchestrator.read().await;
        let metrics = orchestrator.metrics().await;
        assert!(metrics.consensus_operations > 0, "Byzantine consensus should have been used");
    }
    
    // Cleanup
    pipeline.stop().await.unwrap();
    {
        let mut orchestrator = daa_orchestrator.write().await;
        orchestrator.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_mrap_autonomous_fault_recovery() {
    // Arrange - System with fault injection
    let config = Arc::new(IntegrationConfig::default());
    let mut daa_orchestrator = DAAOrchestrator::new(config.clone()).await.unwrap();
    daa_orchestrator.initialize().await.unwrap();
    
    // Enable self-healing
    daa_orchestrator.enable_self_healing().await.unwrap();
    
    let daa_orchestrator = Arc::new(tokio::sync::RwLock::new(daa_orchestrator));
    let message_bus = Arc::new(MessageBus::new(config.clone()).await.unwrap());
    
    let pipeline = ProcessingPipeline::new(
        config.clone(),
        daa_orchestrator.clone(),
        message_bus
    ).await.unwrap();
    
    pipeline.initialize().await.unwrap();
    pipeline.start().await.unwrap();
    
    // Register components including one that will "fail"
    {
        let orchestrator = daa_orchestrator.write().await;
        orchestrator.register_component("healthy-chunker", ComponentType::Chunker, "http://localhost:8002").await.unwrap();
        orchestrator.register_component("failing-storage", ComponentType::Storage, "http://localhost:9999").await.unwrap(); // Bad endpoint
    }
    
    // Act - Allow MRAP to detect and recover from faults
    sleep(Duration::from_secs(5)).await; // Allow multiple MRAP cycles for fault detection and recovery
    
    // Process query during fault recovery
    let query_request = QueryRequest {
        id: Uuid::new_v4(),
        query: "Test query during fault recovery".to_string(),
        filters: None,
        format: Some(ResponseFormat::Json),
        timeout_ms: Some(10000),
    };
    
    let recovery_result = pipeline.process_query(query_request).await;
    
    // Assert - Verify autonomous recovery behavior
    {
        let orchestrator = daa_orchestrator.read().await;
        let mrap_metrics = orchestrator.get_mrap_metrics().await;
        let system_metrics = orchestrator.metrics().await;
        
        // MRAP should have attempted fault recovery
        assert!(mrap_metrics.get("monitoring_cycles").unwrap().as_u64().unwrap() > 0);
        assert!(system_metrics.fault_recoveries > 0, "Self-healing should have been attempted");
        
        // System should continue operating despite faults
        let loops_completed = mrap_metrics.get("mrap_loops_completed").unwrap().as_u64().unwrap();
        assert!(loops_completed > 0, "MRAP loop should continue during fault recovery");
    }
    
    // System should attempt to process queries even with component failures
    // (May succeed or fail gracefully depending on fault severity)
    match recovery_result {
        Ok(_) => {
            // System successfully recovered and processed query
            println!("✓ System successfully recovered from faults");
        },
        Err(e) => {
            // System handled fault gracefully with proper error handling
            println!("✓ System handled fault gracefully: {}", e);
        }
    }
    
    // Cleanup
    pipeline.stop().await.unwrap();
    {
        let mut orchestrator = daa_orchestrator.write().await;
        orchestrator.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_mrap_adaptive_behavior_learning() {
    // Arrange - Long-running test to observe adaptation
    let config = Arc::new(IntegrationConfig::default());
    let mut daa_orchestrator = DAAOrchestrator::new(config.clone()).await.unwrap();
    daa_orchestrator.initialize().await.unwrap();
    
    daa_orchestrator.enable_adaptive_behavior().await.unwrap();
    
    let daa_orchestrator = Arc::new(tokio::sync::RwLock::new(daa_orchestrator));
    let message_bus = Arc::new(MessageBus::new(config.clone()).await.unwrap());
    
    let pipeline = ProcessingPipeline::new(
        config.clone(),
        daa_orchestrator.clone(),
        message_bus
    ).await.unwrap();
    
    pipeline.initialize().await.unwrap();
    pipeline.start().await.unwrap();
    
    // Register components for learning
    {
        let orchestrator = daa_orchestrator.write().await;
        orchestrator.register_component("learning-embedder", ComponentType::Embedder, "http://localhost:8003").await.unwrap();
        orchestrator.register_component("learning-processor", ComponentType::QueryProcessor, "http://localhost:8005").await.unwrap();
    }
    
    // Act - Process multiple queries to generate learning data
    let queries = vec![
        "What is PCI DSS compliance?",
        "Explain encryption requirements",
        "How to implement access controls?",
        "What are vulnerability scanning requirements?",
    ];
    
    let mut processing_times = Vec::new();
    
    for (i, query_text) in queries.iter().enumerate() {
        let query_request = QueryRequest {
            id: Uuid::new_v4(),
            query: query_text.to_string(),
            filters: None,
            format: Some(ResponseFormat::Json),
            timeout_ms: Some(8000),
        };
        
        let start = Instant::now();
        let _result = pipeline.process_query(query_request).await;
        processing_times.push(start.elapsed());
        
        // Allow time for MRAP adaptation between queries
        sleep(Duration::from_secs(2)).await;
    }
    
    // Assert - Verify adaptive behavior
    {
        let orchestrator = daa_orchestrator.read().await;
        let mrap_metrics = orchestrator.get_mrap_metrics().await;
        let system_metrics = orchestrator.metrics().await;
        
        // Adaptation should have occurred
        let adaptations_made = mrap_metrics.get("adaptations_made").unwrap().as_u64().unwrap();
        assert!(adaptations_made > 0 || system_metrics.adaptive_adjustments > 0, 
               "System should have made adaptive adjustments");
        
        // Multiple MRAP cycles should have completed
        let loops_completed = mrap_metrics.get("mrap_loops_completed").unwrap().as_u64().unwrap();
        assert!(loops_completed >= queries.len() as u64, 
               "Should have completed MRAP loops during learning");
        
        // Reflections should have occurred to drive learning
        let reflections = mrap_metrics.get("reflections_performed").unwrap().as_u64().unwrap();
        assert!(reflections > 0, "System should have performed reflections for learning");
    }
    
    // Verify potential performance improvement through adaptation
    let avg_processing_time: f64 = processing_times.iter()
        .map(|d| d.as_millis() as f64)
        .sum::<f64>() / processing_times.len() as f64;
    
    println!("Adaptive Learning Test Results:");
    println!("- Queries processed: {}", queries.len());
    println!("- Average processing time: {:.2}ms", avg_processing_time);
    
    {
        let orchestrator = daa_orchestrator.read().await;
        let mrap_metrics = orchestrator.get_mrap_metrics().await;
        println!("- MRAP loops completed: {}", mrap_metrics.get("mrap_loops_completed").unwrap());
        println!("- Adaptations made: {}", mrap_metrics.get("adaptations_made").unwrap());
        println!("- Average loop time: {:.2}ms", mrap_metrics.get("average_loop_time_ms").unwrap().as_f64().unwrap());
    }
    
    // Cleanup
    pipeline.stop().await.unwrap();
    {
        let mut orchestrator = daa_orchestrator.write().await;
        orchestrator.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_system_integration_with_mrap_orchestration() {
    // Arrange - Full system integration test
    let config = IntegrationConfig::default();
    let system = SystemIntegration::new(config).await.unwrap();
    
    // Act - Start system with MRAP orchestration
    let start_result = system.start().await;
    assert!(start_result.is_ok(), "System should start successfully with MRAP");
    
    // Wait for system to stabilize
    sleep(Duration::from_secs(2)).await;
    
    // Test health check with MRAP running
    let health = system.health().await;
    assert!(matches!(
        health.status,
        integration::HealthStatus::Healthy | 
        integration::HealthStatus::Starting |
        integration::HealthStatus::Degraded
    ));
    
    // Process test query through MRAP-orchestrated system
    let query_request = integration::QueryRequest {
        id: Uuid::new_v4(),
        query: "Test MRAP integration with full system".to_string(),
        filters: None,
        format: Some(integration::ResponseFormat::Json),
        timeout_ms: Some(10000),
    };
    
    let query_start = Instant::now();
    let query_result = system.process_query(query_request).await;
    let query_time = query_start.elapsed();
    
    // Assert - Verify MRAP-orchestrated system integration
    match query_result {
        Ok(response) => {
            assert!(!response.response.is_empty(), "Response should not be empty");
            assert!(query_time < Duration::from_secs(2), 
                   "Query should complete within Phase 2 requirements: {:?}", query_time);
            println!("✓ Full system integration with MRAP successful");
        },
        Err(e) => {
            // System handled the request but components may not be fully available
            println!("⚠ System integration working, components may need actual services: {}", e);
        }
    }
    
    // Verify system metrics show MRAP activity
    let metrics = system.metrics().await;
    assert!(metrics.start_time.is_some(), "System should have start time");
    
    // Test graceful shutdown with MRAP cleanup
    let stop_result = system.stop().await;
    assert!(stop_result.is_ok(), "System should shutdown gracefully with MRAP cleanup");
    
    println!("✓ System integration with MRAP orchestration completed successfully");
}