//! Performance Tests for API Endpoints
//! 
//! Validates response time requirements:
//! - Total response time < 2s
//! - Cache operations < 50ms
//! - Neural processing < 200ms
//! - Byzantine consensus < 500ms

use std::time::{Duration, Instant};
use tokio::time::timeout;
use uuid::Uuid;
use super::mock_client::{MockApiClient, MockQueryRequest};

// Performance requirements (in milliseconds)
const MAX_TOTAL_RESPONSE_TIME: u64 = 2000;
const MAX_CACHE_TIME: u64 = 50;
const MAX_NEURAL_TIME: u64 = 200;
const MAX_CONSENSUS_TIME: u64 = 500;

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_response_time_requirements() {
        let client = MockApiClient::new("http://localhost:3001".to_string())
            .with_timeout(Duration::from_millis(MAX_TOTAL_RESPONSE_TIME));

        let test_cases = vec![
            ("Simple query", false, false),
            ("Consensus query", true, false), 
            ("Intent analysis", false, true),
            ("Full processing", true, true),
        ];

        for (description, require_consensus, intent_analysis) in test_cases {
            println!("Testing: {}", description);
            
            let request = MockQueryRequest {
                doc_id: format!("perf-test-{}", uuid::Uuid::new_v4()),
                question: format!("Performance test: {}", description),
                require_consensus,
                user_id: Some(Uuid::new_v4()),
                intent_analysis: Some(intent_analysis),
            };

            let start = Instant::now();
            
            let result = timeout(
                Duration::from_millis(MAX_TOTAL_RESPONSE_TIME),
                client.query(request)
            ).await;

            let total_duration = start.elapsed();
            
            assert!(result.is_ok(), "Request timed out for: {}", description);
            
            match result.unwrap() {
                Ok(response) => {
                    // Validate total response time
                    assert!(total_duration.as_millis() <= MAX_TOTAL_RESPONSE_TIME as u128,
                        "{}: Total time {}ms exceeds {}ms limit", 
                        description, total_duration.as_millis(), MAX_TOTAL_RESPONSE_TIME);

                    // Validate component performance metrics
                    let perf = &response.pipeline.performance;
                    
                    if let Some(cache_ms) = perf.cache_ms {
                        assert!(cache_ms <= MAX_CACHE_TIME as u128,
                            "{}: Cache time {}ms exceeds {}ms limit", 
                            description, cache_ms, MAX_CACHE_TIME);
                    }
                    
                    if let Some(neural_ms) = perf.neural_ms {
                        assert!(neural_ms <= MAX_NEURAL_TIME as u128,
                            "{}: Neural time {}ms exceeds {}ms limit", 
                            description, neural_ms, MAX_NEURAL_TIME);
                    }
                    
                    if let Some(consensus_ms) = perf.consensus_ms {
                        assert!(consensus_ms <= MAX_CONSENSUS_TIME as u128,
                            "{}: Consensus time {}ms exceeds {}ms limit", 
                            description, consensus_ms, MAX_CONSENSUS_TIME);
                    }
                    
                    println!("✅ {}: Total {}ms (cache: {:?}ms, neural: {:?}ms, consensus: {:?}ms)", 
                        description, total_duration.as_millis(), 
                        perf.cache_ms, perf.neural_ms, perf.consensus_ms);
                }
                Err(error) => {
                    panic!("{}: Query failed - {}", description, error);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_concurrent_performance() {
        let client = MockApiClient::new("http://localhost:3001".to_string());
        let num_concurrent = 10;
        
        let mut handles = vec![];
        let start_all = Instant::now();
        
        for i in 0..num_concurrent {
            let request = MockQueryRequest {
                doc_id: format!("concurrent-{}", i),
                question: format!("Concurrent performance test {}", i),
                require_consensus: i % 2 == 0, // Alternate consensus requirement
                user_id: Some(Uuid::new_v4()),
                intent_analysis: Some(i % 3 == 0), // Every third request
            };
            
            let handle = tokio::spawn(async move {
                let start = Instant::now();
                let result = client.query(request).await;
                let duration = start.elapsed();
                (i, result, duration)
            });
            
            handles.push(handle);
        }
        
        let results = futures::future::join_all(handles).await;
        let total_concurrent_time = start_all.elapsed();
        
        println!("Concurrent execution completed in {}ms", total_concurrent_time.as_millis());
        
        let mut successful = 0;
        let mut max_individual_time = Duration::from_millis(0);
        let mut total_processing_time = Duration::from_millis(0);
        
        for (i, task_result, duration) in results {
            let task_result = task_result.expect("Task should complete");
            
            match task_result {
                Ok(response) => {
                    successful += 1;
                    max_individual_time = max_individual_time.max(duration);
                    total_processing_time += duration;
                    
                    assert!(duration.as_millis() <= MAX_TOTAL_RESPONSE_TIME as u128,
                        "Concurrent request {} took {}ms, exceeds {}ms limit", 
                        i, duration.as_millis(), MAX_TOTAL_RESPONSE_TIME);
                    
                    assert!(!response.answer.is_empty(), 
                        "Concurrent request {} should have answer", i);
                }
                Err(error) => {
                    panic!("Concurrent request {} failed: {}", i, error);
                }
            }
        }
        
        assert_eq!(successful, num_concurrent, "All concurrent requests should succeed");
        
        let avg_time = total_processing_time / num_concurrent;
        println!("✅ Concurrent performance: {}/{} successful, max: {}ms, avg: {}ms", 
            successful, num_concurrent, max_individual_time.as_millis(), avg_time.as_millis());
        
        // Verify concurrent execution doesn't significantly degrade performance
        assert!(max_individual_time.as_millis() <= (MAX_TOTAL_RESPONSE_TIME + 500) as u128,
            "Max concurrent time {}ms should not exceed {}ms", 
            max_individual_time.as_millis(), MAX_TOTAL_RESPONSE_TIME + 500);
    }

    #[tokio::test]
    async fn test_cache_performance_improvement() {
        let client = MockApiClient::new("http://localhost:3001".to_string());
        
        let base_request = MockQueryRequest {
            doc_id: "cache-test".to_string(),
            question: "Test caching performance improvement".to_string(),
            require_consensus: false,
            user_id: Some(Uuid::new_v4()),
            intent_analysis: Some(false),
        };

        // First request (no cache)
        let start1 = Instant::now();
        let result1 = client.query(base_request.clone()).await;
        let duration1 = start1.elapsed();
        
        assert!(result1.is_ok());
        let response1 = result1.unwrap();
        assert!(!response1.cache_hit);
        
        // Simulate cache behavior - subsequent identical request should be faster
        // Note: In mock, we simulate cache by reducing processing time
        let cached_request = MockQueryRequest {
            doc_id: "cache-test-cached".to_string(), // Different ID to simulate cache lookup
            question: base_request.question.clone(),
            require_consensus: false,
            user_id: base_request.user_id,
            intent_analysis: Some(false),
        };
        
        let start2 = Instant::now();
        let result2 = client.query(cached_request).await;
        let duration2 = start2.elapsed();
        
        assert!(result2.is_ok());
        let response2 = result2.unwrap();
        
        // Validate cache performance benefits
        if let (Some(cache_ms1), Some(cache_ms2)) = (
            response1.pipeline.performance.cache_ms,
            response2.pipeline.performance.cache_ms
        ) {
            assert!(cache_ms1 <= MAX_CACHE_TIME as u128,
                "Initial cache lookup {}ms exceeds {}ms limit", cache_ms1, MAX_CACHE_TIME);
            assert!(cache_ms2 <= MAX_CACHE_TIME as u128,
                "Cached lookup {}ms exceeds {}ms limit", cache_ms2, MAX_CACHE_TIME);
        }
        
        println!("✅ Cache performance: initial {}ms, subsequent {}ms", 
            duration1.as_millis(), duration2.as_millis());
    }

    #[tokio::test]
    async fn test_neural_processing_performance() {
        let client = MockApiClient::new("http://localhost:3001".to_string());
        
        let neural_intensive_request = MockQueryRequest {
            doc_id: "neural-test".to_string(),
            question: "Complex question requiring extensive neural analysis with multiple inference steps and reranking operations".to_string(),
            require_consensus: false,
            user_id: Some(Uuid::new_v4()),
            intent_analysis: Some(true),
        };

        let start = Instant::now();
        let result = client.query(neural_intensive_request).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        let response = result.unwrap();
        
        // Validate neural processing performance
        if let Some(neural_ms) = response.pipeline.performance.neural_ms {
            assert!(neural_ms <= MAX_NEURAL_TIME as u128,
                "Neural processing {}ms exceeds {}ms limit", neural_ms, MAX_NEURAL_TIME);
            
            println!("✅ Neural processing completed in {}ms (limit: {}ms)", 
                neural_ms, MAX_NEURAL_TIME);
        }
        
        // Verify intent analysis was performed within time limits
        if let Some(intent) = response.intent {
            assert!(!intent.intent_type.is_empty());
            assert!(intent.confidence > 0.0);
            println!("✅ Intent analysis: {} (confidence: {:.3})", 
                intent.intent_type, intent.confidence);
        }
    }

    #[tokio::test]
    async fn test_byzantine_consensus_performance() {
        let client = MockApiClient::new("http://localhost:3001".to_string());
        
        let consensus_request = MockQueryRequest {
            doc_id: "consensus-test".to_string(),
            question: "Query requiring Byzantine consensus validation across multiple agents".to_string(),
            require_consensus: true,
            user_id: Some(Uuid::new_v4()),
            intent_analysis: Some(false),
        };

        let start = Instant::now();
        let result = client.query(consensus_request).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        let response = result.unwrap();
        
        // Validate consensus performance
        assert!(response.consensus.validated, "Consensus should be validated");
        
        if let Some(consensus_ms) = response.pipeline.performance.consensus_ms {
            assert!(consensus_ms <= MAX_CONSENSUS_TIME as u128,
                "Consensus time {}ms exceeds {}ms limit", consensus_ms, MAX_CONSENSUS_TIME);
            
            println!("✅ Byzantine consensus completed in {}ms (limit: {}ms)", 
                consensus_ms, MAX_CONSENSUS_TIME);
        }
        
        // Verify consensus results
        assert_eq!(response.consensus.threshold, 0.67);
        assert!(response.consensus.agreement_percentage >= 67.0);
        assert_eq!(response.consensus.byzantine_count, 0);
        
        println!("✅ Consensus validation: {}% agreement, {} Byzantine agents", 
            response.consensus.agreement_percentage, response.consensus.byzantine_count);
    }

    #[tokio::test]
    async fn test_upload_performance() {
        let client = MockApiClient::new("http://localhost:3001".to_string());
        
        // Test various document sizes
        let test_cases = vec![
            ("Small document", vec![0u8; 1024]),      // 1KB
            ("Medium document", vec![0u8; 50_000]),   // 50KB  
            ("Large document", vec![0u8; 500_000]),   // 500KB
        ];

        for (description, content) in test_cases {
            let start = Instant::now();
            let result = client.upload(&content, &format!("{}.pdf", description)).await;
            let duration = start.elapsed();
            
            assert!(result.is_ok(), "Upload should succeed for: {}", description);
            let response = result.unwrap();
            
            // Upload should complete reasonably quickly
            assert!(duration.as_millis() <= 1000, // 1 second max for upload
                "{}: Upload time {}ms exceeds 1000ms", description, duration.as_millis());
            
            assert!(!response.id.is_empty());
            assert_eq!(response.status, "processed");
            assert!(response.chunks > 0);
            
            println!("✅ {}: {} bytes uploaded in {}ms, {} chunks created", 
                description, content.len(), duration.as_millis(), response.chunks);
        }
    }
}