//! Integration tests for the Query Processor component
//! Tests complete workflows and component interactions

use query_processor::*;
use std::collections::HashMap;
use tokio_test;
use uuid::Uuid;

/// Test complete query processing workflow
#[tokio::test]
async fn test_complete_query_processing() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    let query = ProcessingRequest::builder()
        .query("What are the main features of Rust programming language?")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let result = processor.process(query).await.unwrap();
    
    // Verify processing result structure
    assert!(!result.processed_query.is_empty());
    assert!(result.confidence_score > 0.0);
    assert!(result.confidence_score <= 1.0);
    assert!(!result.entities.is_empty());
    assert!(result.intent.is_some());
    assert!(result.search_strategy.is_some());
    
    // Verify Byzantine consensus validation
    assert!(result.consensus_result.passed);
    assert!(result.consensus_result.confidence >= 0.66); // BFT threshold
    
    // Verify multi-layer validation
    assert!(!result.validation_results.is_empty());
    assert!(result.validation_results.iter().all(|v| v.confidence > 0.0));
}

/// Test query processing with various intents
#[tokio::test]
async fn test_different_query_intents() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    let test_cases = vec![
        ("What is machine learning?", QueryIntent::Factual),
        ("Compare Python and Java programming languages", QueryIntent::Comparison),
        ("Summarize the benefits of cloud computing", QueryIntent::Summary),
        ("How to deploy a web application?", QueryIntent::Procedural),
    ];
    
    for (query_text, expected_intent) in test_cases {
        let query = ProcessingRequest::builder()
            .query(query_text)
            .query_id(Uuid::new_v4())
            .build()
            .unwrap();
        
        let result = processor.process(query).await.unwrap();
        
        assert_eq!(result.intent.unwrap(), expected_intent);
        assert!(result.confidence_score > 0.5);
    }
}

/// Test query processing performance
#[tokio::test]
async fn test_processing_performance() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    let query = ProcessingRequest::builder()
        .query("Explain artificial intelligence and machine learning")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let start_time = std::time::Instant::now();
    let result = processor.process(query).await.unwrap();
    let processing_time = start_time.elapsed();
    
    // Verify performance target <100ms
    assert!(processing_time.as_millis() < 100, 
           "Processing took {}ms, exceeding 100ms target", processing_time.as_millis());
    
    // Verify result quality
    assert!(result.confidence_score > 0.7);
    assert!(!result.entities.is_empty());
}

/// Test complex query with multiple entities
#[tokio::test]
async fn test_complex_entity_extraction() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    let query = ProcessingRequest::builder()
        .query("How does GDPR compliance affect data processing in European Union member states?")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let result = processor.process(query).await.unwrap();
    
    // Should extract multiple entities
    assert!(result.entities.len() >= 3);
    
    // Should identify compliance terms
    let has_compliance_entity = result.entities.iter()
        .any(|e| e.entity_type == EntityType::ComplianceTerm);
    assert!(has_compliance_entity);
    
    // Should identify geographic entities
    let has_location_entity = result.entities.iter()
        .any(|e| e.entity_type == EntityType::Location);
    assert!(has_location_entity);
}

/// Test validation engine with various scenarios
#[tokio::test]
async fn test_validation_scenarios() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    // Test well-formed query
    let good_query = ProcessingRequest::builder()
        .query("What are the key principles of software architecture?")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let good_result = processor.process(good_query).await.unwrap();
    assert!(good_result.validation_results.iter().all(|v| v.passed));
    
    // Test potentially problematic query
    let edge_case_query = ProcessingRequest::builder()
        .query("???") // Minimal query
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let edge_result = processor.process(edge_case_query).await.unwrap();
    
    // Should still process but with warnings
    assert!(edge_result.confidence_score < 0.5);
    assert!(!edge_result.warnings.is_empty());
}

/// Test concurrent processing
#[tokio::test]
async fn test_concurrent_processing() {
    let config = Config::default();
    let processor = std::sync::Arc::new(QueryProcessor::new(config).await.unwrap());
    
    let queries = vec![
        "What is cloud computing?",
        "Explain microservices architecture",
        "How does blockchain work?",
        "What are design patterns?",
        "Describe REST API principles",
    ];
    
    let handles: Vec<_> = queries.into_iter().enumerate().map(|(i, query_text)| {
        let processor = processor.clone();
        tokio::spawn(async move {
            let query = ProcessingRequest::builder()
                .query(query_text)
                .query_id(Uuid::new_v4())
                .build()
                .unwrap();
            
            (i, processor.process(query).await)
        })
    }).collect();
    
    let results = futures::future::join_all(handles).await;
    
    // All queries should complete successfully
    for (i, result) in results {
        let (query_id, processing_result) = result.unwrap();
        assert_eq!(query_id, i);
        
        let processing_result = processing_result.unwrap();
        assert!(processing_result.confidence_score > 0.0);
        assert!(!processing_result.processed_query.is_empty());
    }
}

/// Test strategy selection for different query types
#[tokio::test]
async fn test_strategy_selection() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    // Vector similarity strategy for semantic queries
    let semantic_query = ProcessingRequest::builder()
        .query("Find similar concepts to artificial intelligence")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let semantic_result = processor.process(semantic_query).await.unwrap();
    assert_eq!(semantic_result.search_strategy.unwrap(), SearchStrategy::VectorSimilarity);
    
    // Keyword strategy for specific term queries
    let keyword_query = ProcessingRequest::builder()
        .query("Python programming language syntax")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let keyword_result = processor.process(keyword_query).await.unwrap();
    assert!(matches!(keyword_result.search_strategy.unwrap(), 
                    SearchStrategy::Keyword | SearchStrategy::Hybrid));
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    // Test empty query handling
    let empty_query = ProcessingRequest::builder()
        .query("")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let result = processor.process(empty_query).await.unwrap();
    assert!(!result.warnings.is_empty());
    assert!(result.confidence_score < 0.5);
    
    // Test very long query handling
    let long_query = "a".repeat(10000);
    let oversized_query = ProcessingRequest::builder()
        .query(long_query)
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let long_result = processor.process(oversized_query).await.unwrap();
    assert!(long_result.processed_query.len() <= 5000); // Should be truncated
}

/// Test metrics collection
#[tokio::test]
async fn test_metrics_collection() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    // Process multiple queries to generate metrics
    for i in 0..5 {
        let query = ProcessingRequest::builder()
            .query(format!("Test query number {}", i))
            .query_id(Uuid::new_v4())
            .build()
            .unwrap();
        
        processor.process(query).await.unwrap();
    }
    
    let metrics = processor.get_metrics();
    assert!(metrics.total_processed >= 5);
    assert!(metrics.successful_processes >= 5);
    assert!(metrics.latency_stats.avg_latency.as_millis() > 0);
    assert!(metrics.throughput_stats.requests_per_second > 0.0);
}

/// Test configuration loading and validation
#[tokio::test]
async fn test_configuration_handling() {
    // Test default configuration
    let default_config = Config::default();
    assert_eq!(default_config.max_query_length, 5000);
    assert_eq!(default_config.processing_timeout.as_millis(), 100);
    
    // Test configuration validation
    let mut invalid_config = Config::default();
    invalid_config.consensus.byzantine_threshold = 1.5; // Invalid threshold > 1.0
    
    let validation_result = invalid_config.validate();
    assert!(validation_result.is_err());
}

/// Test Unicode and international query handling
#[tokio::test]
async fn test_unicode_queries() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    let unicode_queries = vec![
        "¿Qué es la inteligencia artificial?", // Spanish
        "Was ist maschinelles Lernen?",        // German
        "什么是云计算？",                      // Chinese
        "Что такое блокчейн?",                // Russian
    ];
    
    for query_text in unicode_queries {
        let query = ProcessingRequest::builder()
            .query(query_text)
            .query_id(Uuid::new_v4())
            .build()
            .unwrap();
        
        let result = processor.process(query).await.unwrap();
        
        // Should handle Unicode gracefully
        assert!(!result.processed_query.is_empty());
        assert!(result.confidence_score > 0.0);
    }
}

/// Integration test with mock external services
#[tokio::test]
async fn test_with_external_dependencies() {
    // This would test integration with actual storage, embedding services, etc.
    // For now, we test the internal coordination
    
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    let query = ProcessingRequest::builder()
        .query("How do distributed systems maintain consistency?")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let result = processor.process(query).await.unwrap();
    
    // Verify all components worked together
    assert!(result.consensus_result.passed);
    assert!(!result.entities.is_empty());
    assert!(result.intent.is_some());
    assert!(result.search_strategy.is_some());
    assert!(result.confidence_score > 0.6);
}

/// Stress test with high load
#[tokio::test]
async fn test_high_load_processing() {
    let config = Config::default();
    let processor = std::sync::Arc::new(QueryProcessor::new(config).await.unwrap());
    
    let num_concurrent_queries = 20;
    let queries_per_task = 5;
    
    let handles: Vec<_> = (0..num_concurrent_queries).map(|task_id| {
        let processor = processor.clone();
        tokio::spawn(async move {
            let mut successes = 0;
            let start_time = std::time::Instant::now();
            
            for i in 0..queries_per_task {
                let query = ProcessingRequest::builder()
                    .query(format!("Load test query {} from task {}", i, task_id))
                    .query_id(Uuid::new_v4())
                    .build()
                    .unwrap();
                
                if processor.process(query).await.is_ok() {
                    successes += 1;
                }
            }
            
            let duration = start_time.elapsed();
            (task_id, successes, duration)
        })
    }).collect();
    
    let results = futures::future::join_all(handles).await;
    
    let total_successes: usize = results.iter()
        .map(|r| r.as_ref().unwrap().1)
        .sum();
    
    let total_expected = num_concurrent_queries * queries_per_task;
    let success_rate = total_successes as f64 / total_expected as f64;
    
    // Should handle high load with reasonable success rate
    assert!(success_rate > 0.8, "Success rate {:.2}% too low under load", success_rate * 100.0);
}