//! Integration tests for the Query Processor component
//! Tests complete workflows and component interactions with MRAP control loop and Byzantine consensus

use query_processor::{
    Config, QueryProcessor, ProcessingRequest, QueryIntent, EntityType,
    SearchStrategy, ConsensusResult, ValidationStatus
};
use std::time::Instant;
use uuid::Uuid;
use std::collections::HashMap;

// Import futures for concurrent operations
use futures::future::join_all;

/// Test complete query processing workflow with MRAP control loop integration
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
    
    // Verify MRAP control loop components
    assert!(!result.query.text().is_empty());
    assert!(result.processing_metadata.statistics.overall_confidence > 0.0);
    assert!(result.processing_metadata.statistics.overall_confidence <= 1.0);
    assert!(!result.entities.is_empty());
    assert!(result.intent.primary_intent != QueryIntent::Unknown);
    assert!(result.strategy.strategy != SearchStrategy::ExactMatch);
    
    // Verify Byzantine consensus validation with 66% threshold (critical requirement)
    if let Some(consensus) = &result.consensus {
        match consensus {
            ConsensusResult::QueryProcessing { result: query_result } => {
                // Ensure Byzantine 66% threshold is met
                assert!(query_result.confidence >= 0.66, 
                        "Byzantine consensus requires >=66% confidence, got {:.2}%", 
                        query_result.confidence * 100.0);
                
                // Verify Byzantine consensus metadata is present
                assert!(query_result.metadata.contains_key("consensus_type"));
                assert_eq!(query_result.metadata.get("consensus_type"), Some(&"byzantine".to_string()));
                assert!(query_result.metadata.contains_key("threshold"));
            },
            _ => { /* Other consensus types */ }
        }
    }
    
    // Verify DAA multi-layer validation
    assert!(!result.processing_metadata.validation_results.is_empty());
    assert!(result.processing_metadata.validation_results.iter().all(|v| v.score > 0.0));
    
    // Verify MRAP metrics collection
    assert!(result.processing_metadata.total_duration.as_millis() > 0);
    assert!(!result.processing_metadata.stage_durations.is_empty());
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
        
        assert_eq!(result.intent.primary_intent, expected_intent);
        assert!(result.processing_metadata.statistics.overall_confidence > 0.5);
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
    assert!(result.processing_metadata.statistics.overall_confidence > 0.7);
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
        .any(|e| e.category == EntityType::Standard);
    assert!(has_compliance_entity);
    
    // Should identify geographic entities
    let has_location_entity = result.entities.iter()
        .any(|e| e.category == EntityType::Location);
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
    assert!(good_result.processing_metadata.validation_results.iter().all(|v| v.status == ValidationStatus::Passed));
    
    // Test potentially problematic query
    let edge_case_query = ProcessingRequest::builder()
        .query("???") // Minimal query
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let edge_result = processor.process(edge_case_query).await.unwrap();
    
    // Should still process but with warnings
    assert!(edge_result.processing_metadata.statistics.overall_confidence < 0.5);
    assert!(!edge_result.processing_metadata.warnings.is_empty());
}

/// Test concurrent processing with MRAP coordination
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
    
    let query_count = queries.len();
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
    
    let results = join_all(handles).await;
    
    // Verify MRAP concurrent processing coordination
    for result in results {
        let (query_id, processing_result) = result.unwrap();
        assert!(query_id < query_count);
        
        let processing_result = processing_result.unwrap();
        assert!(processing_result.processing_metadata.statistics.overall_confidence > 0.0);
        assert!(!processing_result.query.text().is_empty());
        
        // Verify Byzantine consensus worked under concurrent load (66% threshold)
        if let Some(consensus) = &processing_result.consensus {
            match consensus {
                ConsensusResult::QueryProcessing { result: query_result } => {
                    assert!(query_result.confidence >= 0.66, 
                            "Concurrent Byzantine consensus failed: {:.2}% < 66%", 
                            query_result.confidence * 100.0);
                },
                _ => { /* Other consensus types */ }
            }
        }
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
    assert_eq!(semantic_result.strategy.strategy, SearchStrategy::VectorSimilarity);
    
    // Keyword strategy for specific term queries
    let keyword_query = ProcessingRequest::builder()
        .query("Python programming language syntax")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let keyword_result = processor.process(keyword_query).await.unwrap();
    assert!(matches!(keyword_result.strategy.strategy, 
                    SearchStrategy::Keyword { .. } | SearchStrategy::Hybrid { .. }));
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
    assert!(!result.processing_metadata.warnings.is_empty());
    assert!(result.processing_metadata.statistics.overall_confidence < 0.5);
    
    // Test very long query handling
    let long_query = "a".repeat(10000);
    let oversized_query = ProcessingRequest::builder()
        .query(long_query)
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let long_result = processor.process(oversized_query).await.unwrap();
    assert!(long_result.query.text().len() <= 5000); // Should be truncated
}

/// Test MRAP metrics collection and analysis
#[tokio::test]
async fn test_metrics_collection() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    // Process multiple queries to generate MRAP metrics
    for i in 0..5 {
        let query = ProcessingRequest::builder()
            .query(format!("Test query number {}", i))
            .query_id(Uuid::new_v4())
            .build()
            .unwrap();
        
        processor.process(query).await.unwrap();
    }
    
    let metrics = processor.metrics().await;
    assert!(metrics.total_processed >= 5);
    assert!(metrics.successful_processes >= 5);
    assert!(metrics.latency_stats.average().as_millis() > 0);
    assert!(metrics.throughput_stats.current_qps() > 0.0);
    
    // Verify MRAP control loop metrics - these are embedded in processing stats
    assert!(metrics.accuracy_stats.average_confidence() > 0.0);
    assert!(!metrics.intent_distribution.is_empty());
}

/// Test configuration loading and validation
#[tokio::test]
async fn test_configuration_handling() {
    // Test default configuration
    let default_config = Config::default();
    assert_eq!(default_config.analyzer.max_query_length, 5000);
    // Note: processing_timeout might be in a different location
    
    // Test configuration validation
    let mut invalid_config = Config::default();
    invalid_config.consensus.agreement_threshold = 1.5; // Invalid threshold > 1.0
    
    // Configuration validation would be done at processing time
    let validation_result = QueryProcessor::new(invalid_config).await;
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
        assert!(!result.query.text().is_empty());
        assert!(result.processing_metadata.statistics.overall_confidence > 0.0);
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
    if let Some(consensus) = &result.consensus {
        match consensus {
            ConsensusResult::QueryProcessing { result: query_result } => {
                assert!(query_result.confidence > 0.6);
            },
            _ => { /* Other consensus types */ }
        }
    }
    assert!(!result.entities.is_empty());
    assert!(result.intent.primary_intent != QueryIntent::Unknown);
    assert!(result.strategy.strategy != SearchStrategy::ExactMatch);
    assert!(result.processing_metadata.statistics.overall_confidence > 0.6);
}

/// Stress test MRAP control loop with high load and Byzantine fault tolerance
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
            let mut consensus_successes = 0;
            let start_time = Instant::now();
            
            for i in 0..queries_per_task {
                let query = ProcessingRequest::builder()
                    .query(format!("Load test query {} from task {}", i, task_id))
                    .query_id(Uuid::new_v4())
                    .build()
                    .unwrap();
                
                if let Ok(result) = processor.process(query).await {
                    successes += 1;
                    if let Some(consensus) = &result.consensus {
                        match consensus {
                            ConsensusResult::QueryProcessing { result: query_result } => {
                                // Byzantine consensus requires 66% threshold
                                if query_result.confidence >= 0.66 {
                                    consensus_successes += 1;
                                }
                            },
                            _ => { /* Other consensus types */ }
                        }
                    }
                }
            }
            
            let duration = start_time.elapsed();
            (task_id, successes, consensus_successes, duration)
        })
    }).collect();
    
    let results = join_all(handles).await;
    
    let (total_successes, total_consensus_successes): (usize, usize) = results.iter()
        .map(|r| (r.as_ref().unwrap().1, r.as_ref().unwrap().2))
        .fold((0, 0), |(acc_s, acc_c), (s, c)| (acc_s + s, acc_c + c));
    
    let total_expected = num_concurrent_queries * queries_per_task;
    let success_rate = total_successes as f64 / total_expected as f64;
    let consensus_rate = total_consensus_successes as f64 / total_successes as f64;
    
    // Verify MRAP handles high load with Byzantine consensus (66% threshold)
    assert!(success_rate > 0.8, "Success rate {:.2}% too low under load", success_rate * 100.0);
    assert!(consensus_rate > 0.9, "Byzantine consensus rate {:.2}% too low under load (66% threshold)", consensus_rate * 100.0);
}

/// Test DAA error handling and recovery mechanisms
#[tokio::test]
async fn test_daa_error_handling() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    // Test with malformed query - should be detected and handled
    let malformed_query = ProcessingRequest::builder()
        .query("<script>alert('xss')</script>")
        .query_id(Uuid::new_v4())
        .build()
        .unwrap();
    
    let result = processor.process(malformed_query).await;
    // Should handle malicious content gracefully
    assert!(result.is_err() || result.unwrap().processing_metadata.warnings.len() > 0);
}

/// Test MRAP control loop adaptation under different query types
#[tokio::test]
async fn test_mrap_adaptation() {
    let config = Config::default();
    let processor = QueryProcessor::new(config).await.unwrap();
    
    let query_types = vec![
        ("Simple factual query", QueryIntent::Factual),
        ("Complex analytical question requiring deep analysis and multiple data sources", QueryIntent::Analytical),
        ("Compare PCI DSS requirements across different versions", QueryIntent::Comparison),
    ];
    
    for (query_text, expected_intent) in query_types {
        let query = ProcessingRequest::builder()
            .query(query_text)
            .query_id(Uuid::new_v4())
            .build()
            .unwrap();
        
        let result = processor.process(query).await.unwrap();
        
        // Verify MRAP adapts strategy based on query complexity
        assert_eq!(result.intent.primary_intent, expected_intent);
        
        if expected_intent == QueryIntent::Analytical {
            // Complex queries should use more sophisticated strategies
            assert!(matches!(result.strategy.strategy, 
                SearchStrategy::Hybrid { .. } | 
                SearchStrategy::Semantic { .. } |
                SearchStrategy::Adaptive { .. }
            ));
        }
    }
}

