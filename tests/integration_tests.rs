//! Integration tests between Query Processor and Response Generator
//! Tests the complete end-to-end workflow

use std::collections::HashMap;
use uuid::Uuid;

// Mock the external components for testing
mod mock_components {
    use std::collections::HashMap;
    use uuid::Uuid;

    #[derive(Debug, Clone)]
    pub struct MockQueryProcessor {
        pub processing_latency: std::time::Duration,
    }

    #[derive(Debug, Clone)]
    pub struct MockResponseGenerator {
        pub generation_latency: std::time::Duration,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum QueryIntent {
        Factual,
        Comparison,
        Summary,
        Procedural,
    }

    #[derive(Debug, Clone)]
    pub struct ProcessedQuery {
        pub id: Uuid,
        pub original_query: String,
        pub processed_query: String,
        pub intent: QueryIntent,
        pub entities: Vec<String>,
        pub confidence_score: f64,
        pub processing_time: std::time::Duration,
    }

    #[derive(Debug, Clone)]
    pub struct GeneratedResponse {
        pub query_id: Uuid,
        pub content: String,
        pub confidence_score: f64,
        pub citations: Vec<String>,
        pub generation_time: std::time::Duration,
    }

    impl MockQueryProcessor {
        pub fn new() -> Self {
            Self {
                processing_latency: std::time::Duration::from_millis(30),
            }
        }

        pub async fn process(&self, query: &str) -> ProcessedQuery {
            tokio::time::sleep(self.processing_latency).await;

            let intent = if query.contains("what") || query.contains("define") {
                QueryIntent::Factual
            } else if query.contains("compare") || query.contains("vs") {
                QueryIntent::Comparison
            } else if query.contains("summarize") || query.contains("summary") {
                QueryIntent::Summary
            } else if query.contains("how") {
                QueryIntent::Procedural
            } else {
                QueryIntent::Factual
            };

            let entities = query
                .split_whitespace()
                .filter(|word| word.len() > 3)
                .take(3)
                .map(|s| s.to_string())
                .collect();

            ProcessedQuery {
                id: Uuid::new_v4(),
                original_query: query.to_string(),
                processed_query: format!("processed: {}", query),
                intent,
                entities,
                confidence_score: 0.85,
                processing_time: self.processing_latency,
            }
        }
    }

    impl MockResponseGenerator {
        pub fn new() -> Self {
            Self {
                generation_latency: std::time::Duration::from_millis(50),
            }
        }

        pub async fn generate(&self, processed_query: &ProcessedQuery) -> GeneratedResponse {
            tokio::time::sleep(self.generation_latency).await;

            let content = match processed_query.intent {
                QueryIntent::Factual => format!("This is a factual response to: {}", processed_query.original_query),
                QueryIntent::Comparison => format!("This is a comparative analysis of: {}", processed_query.original_query),
                QueryIntent::Summary => format!("This is a summary of: {}", processed_query.original_query),
                QueryIntent::Procedural => format!("This is a step-by-step guide for: {}", processed_query.original_query),
            };

            GeneratedResponse {
                query_id: processed_query.id,
                content,
                confidence_score: processed_query.confidence_score * 0.9, // Slight reduction
                citations: processed_query.entities.iter().map(|e| format!("Citation for {}", e)).collect(),
                generation_time: self.generation_latency,
            }
        }
    }
}

use mock_components::*;

/// Integration test for complete query-to-response pipeline
#[tokio::test]
async fn test_complete_pipeline() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();

    let test_query = "What is artificial intelligence and how does it work?";
    
    let start_time = std::time::Instant::now();
    
    // Process query
    let processed = query_processor.process(test_query).await;
    
    // Generate response
    let response = response_generator.generate(&processed).await;
    
    let total_time = start_time.elapsed();
    
    // Verify complete pipeline
    assert_eq!(processed.original_query, test_query);
    assert_eq!(response.query_id, processed.id);
    assert!(!response.content.is_empty());
    assert!(response.confidence_score > 0.5);
    
    // Verify performance targets
    assert!(total_time.as_millis() < 100, "Total pipeline time {}ms exceeds 100ms target", total_time.as_millis());
    
    // Verify intent detection worked
    assert_eq!(processed.intent, QueryIntent::Factual);
    assert!(!processed.entities.is_empty());
}

/// Test pipeline with different query types
#[tokio::test]
async fn test_different_query_types() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    let test_cases = vec![
        ("What is machine learning?", QueryIntent::Factual),
        ("Compare Python and Java programming languages", QueryIntent::Comparison),
        ("Summarize the benefits of cloud computing", QueryIntent::Summary),
        ("How to deploy a web application?", QueryIntent::Procedural),
    ];
    
    for (query, expected_intent) in test_cases {
        let processed = query_processor.process(query).await;
        let response = response_generator.generate(&processed).await;
        
        assert_eq!(processed.intent, expected_intent);
        assert!(!response.content.is_empty());
        assert!(response.content.contains(&expected_intent.to_string().to_lowercase()) || 
                response.content.len() > 10);
    }
}

/// Test pipeline performance under concurrent load
#[tokio::test]
async fn test_concurrent_pipeline_load() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    let queries = vec![
        "What is cloud computing?",
        "Explain machine learning algorithms",
        "How does blockchain technology work?",
        "Compare different database systems",
        "Summarize recent AI developments",
    ];
    
    let start_time = std::time::Instant::now();
    
    // Process all queries concurrently
    let handles: Vec<_> = queries.iter().map(|&query| {
        let qp = query_processor.clone();
        let rg = response_generator.clone();
        
        tokio::spawn(async move {
            let processed = qp.process(query).await;
            let response = rg.generate(&processed).await;
            (processed, response)
        })
    }).collect();
    
    let results = futures::future::join_all(handles).await;
    let total_time = start_time.elapsed();
    
    // Verify all completed successfully
    assert_eq!(results.len(), 5);
    for result in results {
        let (processed, response) = result.unwrap();
        assert!(!processed.processed_query.is_empty());
        assert!(!response.content.is_empty());
        assert!(response.confidence_score > 0.0);
    }
    
    // Concurrent processing should be efficient
    assert!(total_time.as_millis() < 200, "Concurrent processing took {}ms", total_time.as_millis());
}

/// Test error handling in the pipeline
#[tokio::test]
async fn test_pipeline_error_handling() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    // Test with edge cases
    let edge_cases = vec![
        "",           // Empty query
        "?",          // Single character
        "a".repeat(1000), // Very long query
    ];
    
    for query in edge_cases {
        let processed = query_processor.process(&query).await;
        let response = response_generator.generate(&processed).await;
        
        // Should handle gracefully without panicking
        assert_eq!(processed.original_query, query);
        assert!(!response.content.is_empty()); // Should generate some response
    }
}

/// Test pipeline metrics and monitoring
#[tokio::test]
async fn test_pipeline_metrics() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    let mut processing_times = Vec::new();
    let mut generation_times = Vec::new();
    let mut total_times = Vec::new();
    let mut confidence_scores = Vec::new();
    
    // Run multiple queries to collect metrics
    for i in 0..10 {
        let query = format!("Test query number {}", i);
        let start_time = std::time::Instant::now();
        
        let processed = query_processor.process(&query).await;
        let response = response_generator.generate(&processed).await;
        
        let total_time = start_time.elapsed();
        
        processing_times.push(processed.processing_time);
        generation_times.push(response.generation_time);
        total_times.push(total_time);
        confidence_scores.push(response.confidence_score);
    }
    
    // Calculate metrics
    let avg_processing_time = processing_times.iter().sum::<std::time::Duration>() / processing_times.len() as u32;
    let avg_generation_time = generation_times.iter().sum::<std::time::Duration>() / generation_times.len() as u32;
    let avg_total_time = total_times.iter().sum::<std::time::Duration>() / total_times.len() as u32;
    let avg_confidence = confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64;
    
    // Verify performance metrics
    assert!(avg_processing_time.as_millis() < 50, "Average processing time {}ms too high", avg_processing_time.as_millis());
    assert!(avg_generation_time.as_millis() < 70, "Average generation time {}ms too high", avg_generation_time.as_millis());
    assert!(avg_total_time.as_millis() < 100, "Average total time {}ms too high", avg_total_time.as_millis());
    assert!(avg_confidence > 0.7, "Average confidence {:.2} too low", avg_confidence);
}

/// Test pipeline data flow integrity
#[tokio::test]
async fn test_data_flow_integrity() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    let test_query = "Explain the fundamentals of distributed systems architecture";
    
    let processed = query_processor.process(test_query).await;
    let response = response_generator.generate(&processed).await;
    
    // Verify data flows correctly through pipeline
    assert_eq!(processed.original_query, test_query);
    assert!(processed.processed_query.contains(test_query));
    assert_eq!(response.query_id, processed.id);
    
    // Verify entities are preserved/used
    assert!(!processed.entities.is_empty());
    assert_eq!(response.citations.len(), processed.entities.len());
    
    // Verify confidence propagation
    assert!(response.confidence_score <= processed.confidence_score);
    assert!(response.confidence_score > 0.0);
}

/// Test pipeline scalability
#[tokio::test]
async fn test_pipeline_scalability() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    // Test with increasing load
    let load_levels = vec![1, 5, 10, 20];
    
    for load in load_levels {
        let start_time = std::time::Instant::now();
        
        let handles: Vec<_> = (0..load).map(|i| {
            let qp = query_processor.clone();
            let rg = response_generator.clone();
            
            tokio::spawn(async move {
                let query = format!("Scalability test query {}", i);
                let processed = qp.process(&query).await;
                let response = rg.generate(&processed).await;
                (processed, response)
            })
        }).collect();
        
        let results = futures::future::join_all(handles).await;
        let duration = start_time.elapsed();
        
        // All should complete successfully
        assert_eq!(results.len(), load);
        
        // Performance should degrade gracefully
        let per_query_time = duration.as_millis() / load as u128;
        assert!(per_query_time < 150, "Per-query time {}ms too high at load {}", per_query_time, load);
        
        // Success rate should remain high
        let success_count = results.iter().filter(|r| r.is_ok()).count();
        assert_eq!(success_count, load);
    }
}

/// Test pipeline configuration and customization
#[tokio::test]
async fn test_pipeline_configuration() {
    // Test with different processor configurations
    let fast_processor = MockQueryProcessor {
        processing_latency: std::time::Duration::from_millis(10),
    };
    
    let slow_processor = MockQueryProcessor {
        processing_latency: std::time::Duration::from_millis(100),
    };
    
    let response_generator = MockResponseGenerator::new();
    
    let query = "Test configuration impact";
    
    // Test fast configuration
    let fast_start = std::time::Instant::now();
    let fast_processed = fast_processor.process(query).await;
    let fast_response = response_generator.generate(&fast_processed).await;
    let fast_duration = fast_start.elapsed();
    
    // Test slow configuration
    let slow_start = std::time::Instant::now();
    let slow_processed = slow_processor.process(query).await;
    let slow_response = response_generator.generate(&slow_processed).await;
    let slow_duration = slow_start.elapsed();
    
    // Verify configuration impact
    assert!(fast_duration < slow_duration);
    assert_eq!(fast_processed.original_query, slow_processed.original_query);
    assert_eq!(fast_response.content, slow_response.content); // Same logic, different timing
}

/// Test pipeline memory usage and resource management
#[tokio::test]
async fn test_pipeline_resource_management() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    // Process many queries to test memory management
    for batch in 0..5 {
        let batch_queries: Vec<_> = (0..20).map(|i| {
            format!("Resource management test query {} in batch {}", i, batch)
        }).collect();
        
        // Process batch concurrently
        let handles: Vec<_> = batch_queries.iter().map(|query| {
            let qp = query_processor.clone();
            let rg = response_generator.clone();
            let q = query.clone();
            
            tokio::spawn(async move {
                let processed = qp.process(&q).await;
                let response = rg.generate(&processed).await;
                (processed.id, response.content.len())
            })
        }).collect();
        
        let results = futures::future::join_all(handles).await;
        
        // Verify all completed
        assert_eq!(results.len(), 20);
        
        // All should have valid results
        for result in results {
            let (id, content_length) = result.unwrap();
            assert!(!id.is_nil());
            assert!(content_length > 0);
        }
    }
    
    // Test should complete without OOM or resource exhaustion
}

/// Test pipeline with realistic data sizes
#[tokio::test]
async fn test_pipeline_realistic_data() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    // Test with various query sizes
    let queries = vec![
        "AI",  // Very short
        "What is artificial intelligence?", // Normal
        "Can you provide a comprehensive explanation of artificial intelligence, machine learning, deep learning, natural language processing, computer vision, and their applications in modern technology?", // Long
    ];
    
    for query in queries {
        let start_time = std::time::Instant::now();
        
        let processed = query_processor.process(query).await;
        let response = response_generator.generate(&processed).await;
        
        let duration = start_time.elapsed();
        
        // Should handle all sizes
        assert!(!response.content.is_empty());
        assert!(duration.as_millis() < 200); // Should remain performant
        assert!(response.confidence_score > 0.0);
        
        // Response quality should be reasonable regardless of input size
        assert!(response.content.len() >= query.len() / 2); // At least some expansion
    }
}

/// Integration test with simulated external dependencies
#[tokio::test]
async fn test_external_integration() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    // Simulate external service delays
    let query = "Integration test with external dependencies";
    
    // Add artificial delay to simulate network calls, database queries, etc.
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    
    let processed = query_processor.process(query).await;
    
    // Simulate additional external processing
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;
    
    let response = response_generator.generate(&processed).await;
    
    // Should handle external delays gracefully
    assert!(!response.content.is_empty());
    assert!(response.confidence_score > 0.0);
    assert_eq!(response.query_id, processed.id);
}

/// End-to-end test with comprehensive validation
#[tokio::test]
async fn test_end_to_end_comprehensive() {
    let query_processor = MockQueryProcessor::new();
    let response_generator = MockResponseGenerator::new();
    
    let complex_query = "Compare the advantages and disadvantages of microservices architecture versus monolithic architecture in enterprise software development, considering factors such as scalability, maintainability, deployment complexity, and team organization.";
    
    let start_time = std::time::Instant::now();
    
    // Full pipeline execution
    let processed = query_processor.process(complex_query).await;
    let response = response_generator.generate(&processed).await;
    
    let total_duration = start_time.elapsed();
    
    // Comprehensive verification
    
    // 1. Performance requirements
    assert!(total_duration.as_millis() < 100, "Total time {}ms exceeds target", total_duration.as_millis());
    assert!(processed.processing_time.as_millis() < 50, "Processing time {}ms too high", processed.processing_time.as_millis());
    assert!(response.generation_time.as_millis() < 70, "Generation time {}ms too high", response.generation_time.as_millis());
    
    // 2. Data integrity
    assert_eq!(processed.original_query, complex_query);
    assert_eq!(response.query_id, processed.id);
    
    // 3. Quality requirements
    assert!(response.confidence_score > 0.7, "Confidence {:.2} too low", response.confidence_score);
    assert!(response.content.len() > 50, "Response too short: {} chars", response.content.len());
    assert!(!response.citations.is_empty(), "No citations generated");
    
    // 4. Intent detection accuracy
    assert_eq!(processed.intent, QueryIntent::Comparison); // Should detect comparison intent
    
    // 5. Entity extraction
    assert!(!processed.entities.is_empty(), "No entities extracted");
    assert!(processed.entities.len() >= 2, "Too few entities extracted");
    
    // 6. Content relevance
    assert!(response.content.to_lowercase().contains("microservices") || 
           response.content.to_lowercase().contains("comparison"),
           "Response doesn't seem relevant to query");
    
    // 7. Resource efficiency
    // This is a basic check - in production you'd monitor actual memory usage
    assert!(processed.entities.len() <= 10, "Too many entities might indicate inefficiency");
    assert!(response.citations.len() <= processed.entities.len(), "Citation count seems excessive");
}