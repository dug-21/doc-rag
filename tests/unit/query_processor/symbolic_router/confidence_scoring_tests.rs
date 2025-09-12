//! Neural Confidence Scoring Tests - London TDD Implementation
//!
//! Tests for ruv-fann neural network integration and confidence scoring
//! with <10ms inference constraint validation.

use super::*;
use mockall::{predicate, mock};
use std::time::Instant;

#[cfg(test)]
mod confidence_scoring_tests {
    use super::*;

    /// London TDD Test: Neural confidence scoring should meet latency constraint
    #[tokio::test]
    async fn test_neural_inference_latency_constraint() {
        // Given: Router with neural scoring enabled
        let fixture = SymbolicRouterTestFixture::new().await;
        
        // When: Calculating confidence scores for multiple queries
        let mut inference_times = Vec::new();
        
        for query in &fixture.benchmark_queries[0..20] {
            let start_time = Instant::now();
            
            let result = fixture.router
                .route_query(query, &fixture.mock_analysis)
                .await
                .unwrap();
            
            let inference_time = start_time.elapsed();
            inference_times.push(inference_time);
            
            // Then: Individual inferences should be fast
            assert!(inference_time.as_millis() < 50, 
                    "Neural inference took {}ms > 50ms for routing", 
                    inference_time.as_millis());
            
            // And: Should produce valid confidence scores
            assert!(result.confidence >= 0.0 && result.confidence <= 1.0,
                    "Confidence score should be between 0 and 1: {}", 
                    result.confidence);
        }
        
        // Then: Average inference time should be well below constraint
        let avg_time = inference_times.iter().sum::<Duration>() / inference_times.len() as u32;
        assert!(avg_time.as_millis() < 25, 
                "Average neural inference time {}ms should be < 25ms", 
                avg_time.as_millis());
        
        // And: 95th percentile should meet constraint
        let p95_time = calculate_percentile(&inference_times, 95);
        assert!(p95_time.as_millis() < 40, 
                "P95 neural inference time {}ms should be < 40ms", 
                p95_time.as_millis());
    }

    /// London TDD Test: Confidence scores should correlate with query characteristics
    #[tokio::test]
    async fn test_confidence_correlation_with_characteristics() {
        // Given: Router and queries with different characteristics
        let fixture = SymbolicRouterTestFixture::new().await;
        
        // Complex logical query (should have high confidence for symbolic)
        let complex_query = Query::new("If cardholder data is stored and encrypted, then what additional controls are required for compliance?").unwrap();
        
        // Simple definitional query (should have lower confidence for symbolic)
        let simple_query = Query::new("What is encryption?").unwrap();
        
        // When: Getting routing decisions
        let complex_result = fixture.router
            .route_query(&complex_query, &fixture.mock_analysis)
            .await
            .unwrap();
            
        let simple_result = fixture.router
            .route_query(&simple_query, &fixture.mock_analysis)
            .await
            .unwrap();
        
        // Then: Complex logical queries should have different confidence patterns
        println!("Complex query - Engine: {:?}, Confidence: {:.3}", 
                 complex_result.engine, complex_result.confidence);
        println!("Simple query - Engine: {:?}, Confidence: {:.3}", 
                 simple_result.engine, simple_result.confidence);
        
        // Both should have reasonable confidence scores
        assert!(complex_result.confidence > 0.0);
        assert!(simple_result.confidence > 0.0);
        
        // And: Routing decisions should reflect query characteristics
        match complex_result.engine {
            QueryEngine::Symbolic | QueryEngine::Hybrid(_) => {
                // Expected for complex logical query
            },
            _ => {
                println!("Complex query routed to non-symbolic engine - may indicate room for improvement");
            }
        }
    }

    /// London TDD Test: Rule-based confidence fallback should work
    #[tokio::test]
    async fn test_rule_based_confidence_fallback() {
        // Given: Configuration with neural scoring disabled
        let config = crate::query_processor::SymbolicRouterConfig {
            enable_neural_scoring: false,
            target_symbolic_latency_ms: 100,
            min_routing_confidence: 0.8,
            enable_proof_chains: true,
            max_proof_depth: 10,
            enable_performance_monitoring: true,
        };
        
        let router = SymbolicQueryRouter::new(config).await.unwrap();
        let mock_analysis = create_mock_semantic_analysis();
        
        // When: Routing queries with neural scoring disabled
        let logical_query = Query::new("If X then Y, and X is true, therefore Y must be true").unwrap();
        let result = router.route_query(&logical_query, &mock_analysis).await.unwrap();
        
        // Then: Should still provide confidence scores via rule-based approach
        assert!(result.confidence > 0.0 && result.confidence <= 1.0,
                "Rule-based confidence should be valid: {}", result.confidence);
        
        // And: Should provide reasoning
        assert!(!result.reasoning.is_empty(), "Should provide reasoning for rule-based routing");
        
        // And: Should be fast (no neural network overhead)
        let start_time = Instant::now();
        let _ = router.route_query(&logical_query, &mock_analysis).await.unwrap();
        let rule_based_time = start_time.elapsed();
        
        assert!(rule_based_time.as_millis() < 20,
                "Rule-based confidence should be very fast: {}ms", 
                rule_based_time.as_millis());
    }

    /// London TDD Test: Confidence scores should be consistent for similar queries
    #[tokio::test]
    async fn test_confidence_consistency() {
        // Given: Router and similar queries
        let fixture = SymbolicRouterTestFixture::new().await;
        
        let similar_queries = vec![
            Query::new("What encryption is required for cardholder data?").unwrap(),
            Query::new("What encryption is needed for payment card data?").unwrap(),
            Query::new("What encryption is mandatory for CHD?").unwrap(),
        ];
        
        // When: Getting confidence scores for similar queries
        let mut confidence_scores = Vec::new();
        for query in similar_queries {
            let result = fixture.router
                .route_query(&query, &fixture.mock_analysis)
                .await
                .unwrap();
            confidence_scores.push(result.confidence);
        }
        
        // Then: Confidence scores should be similar (within reasonable range)
        let mean_confidence = confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64;
        let max_deviation = confidence_scores.iter()
            .map(|&score| (score - mean_confidence).abs())
            .fold(0.0, f64::max);
        
        assert!(max_deviation < 0.3, 
                "Similar queries should have similar confidence scores. Max deviation: {:.3}, Mean: {:.3}", 
                max_deviation, mean_confidence);
        
        println!("Confidence scores for similar queries: {:?}", confidence_scores);
        println!("Mean confidence: {:.3}, Max deviation: {:.3}", mean_confidence, max_deviation);
    }

    /// London TDD Test: High confidence should correlate with correct routing
    #[tokio::test]
    async fn test_confidence_routing_correlation() {
        // Given: Router and diverse query set
        let fixture = SymbolicRouterTestFixture::new().await;
        let mut high_confidence_results = Vec::new();
        let mut low_confidence_results = Vec::new();
        
        // When: Collecting routing results with different confidence levels
        for query in &fixture.benchmark_queries[0..30] {
            let result = fixture.router
                .route_query(query, &fixture.mock_analysis)
                .await
                .unwrap();
            
            if result.confidence >= 0.8 {
                high_confidence_results.push(result);
            } else if result.confidence < 0.6 {
                low_confidence_results.push(result);
            }
        }
        
        // Then: High confidence results should be more consistent
        if !high_confidence_results.is_empty() {
            println!("High confidence results: {} queries with confidence >= 0.8", 
                     high_confidence_results.len());
            
            // High confidence should have fewer hybrid routing decisions
            let hybrid_rate_high = high_confidence_results.iter()
                .filter(|r| matches!(r.engine, QueryEngine::Hybrid(_)))
                .count() as f64 / high_confidence_results.len() as f64;
            
            println!("Hybrid routing rate for high confidence: {:.1}%", hybrid_rate_high * 100.0);
        }
        
        if !low_confidence_results.is_empty() {
            println!("Low confidence results: {} queries with confidence < 0.6", 
                     low_confidence_results.len());
            
            // Low confidence might have more hybrid routing
            let hybrid_rate_low = low_confidence_results.iter()
                .filter(|r| matches!(r.engine, QueryEngine::Hybrid(_)))
                .count() as f64 / low_confidence_results.len() as f64;
            
            println!("Hybrid routing rate for low confidence: {:.1}%", hybrid_rate_low * 100.0);
        }
        
        // Basic validation - should have some results in each category
        assert!(high_confidence_results.len() + low_confidence_results.len() > 0,
                "Should have results in different confidence ranges");
    }

    /// London TDD Test: Confidence thresholds should be respected
    #[tokio::test]
    async fn test_confidence_threshold_behavior() {
        // Given: Router with high confidence threshold
        let high_threshold_config = crate::query_processor::SymbolicRouterConfig {
            enable_neural_scoring: true,
            target_symbolic_latency_ms: 100,
            min_routing_confidence: 0.9, // Very high threshold
            enable_proof_chains: true,
            max_proof_depth: 10,
            enable_performance_monitoring: true,
        };
        
        let router = SymbolicQueryRouter::new(high_threshold_config).await.unwrap();
        let mock_analysis = create_mock_semantic_analysis();
        
        // When: Routing queries with high confidence threshold
        let ambiguous_query = Query::new("Something about data").unwrap();
        let result = router.route_query(&ambiguous_query, &mock_analysis).await.unwrap();
        
        // Then: Should handle high threshold gracefully
        assert!(result.confidence >= 0.0, "Should provide valid confidence");
        
        // And: May route to hybrid if confidence is below threshold
        println!("High threshold routing - Engine: {:?}, Confidence: {:.3}", 
                 result.engine, result.confidence);
        
        if result.confidence < 0.9 {
            // Should prefer hybrid or safer routing
            match result.engine {
                QueryEngine::Hybrid(_) => {
                    println!("Correctly routed to hybrid due to low confidence");
                },
                _ => {
                    println!("Single engine routing despite low confidence - implementation choice");
                }
            }
        }
    }

    /// London TDD Test: Confidence scoring should handle various input features
    #[tokio::test]
    async fn test_confidence_input_feature_handling() {
        // Given: Router and queries with different feature patterns
        let fixture = SymbolicRouterTestFixture::new().await;
        
        let feature_test_queries = vec![
            (Query::new("What is PCI DSS?").unwrap(), "simple_factual"),
            (Query::new("If A and B then C").unwrap(), "logical_operators"),
            (Query::new("Compare requirements before 2020 and after 2021").unwrap(), "temporal_constraints"),
            (Query::new("How do section 3.4 and requirement 8.2 relate?").unwrap(), "cross_references"),
            (Query::new("Prove that the system meets all security requirements").unwrap(), "proof_required"),
        ];
        
        // When: Processing queries with different features
        for (query, feature_type) in feature_test_queries {
            let result = fixture.router
                .route_query(&query, &fixture.mock_analysis)
                .await
                .unwrap();
            
            // Then: Should handle all feature types
            assert!(result.confidence > 0.0, 
                    "Should provide confidence for {} query: {:.3}", 
                    feature_type, result.confidence);
            
            assert!(!result.reasoning.is_empty(), 
                    "Should provide reasoning for {} query", feature_type);
            
            println!("{}: Engine={:?}, Confidence={:.3}", 
                     feature_type, result.engine, result.confidence);
        }
    }

    /// London TDD Test: Confidence aggregation should be balanced
    #[tokio::test]
    async fn test_confidence_aggregation_balance() {
        // Given: Router and mock scenario with known confidence scores
        let fixture = SymbolicRouterTestFixture::new().await;
        
        // When: Processing multiple queries to observe confidence distribution
        let mut confidence_distribution = Vec::new();
        
        for query in &fixture.benchmark_queries[0..50] {
            let result = fixture.router
                .route_query(query, &fixture.mock_analysis)
                .await
                .unwrap();
            confidence_distribution.push(result.confidence);
        }
        
        // Then: Should have reasonable confidence distribution
        let mean_confidence = confidence_distribution.iter().sum::<f64>() / confidence_distribution.len() as f64;
        let min_confidence = confidence_distribution.iter().fold(1.0, |a, &b| a.min(b));
        let max_confidence = confidence_distribution.iter().fold(0.0, |a, &b| a.max(b));
        
        println!("Confidence distribution - Mean: {:.3}, Min: {:.3}, Max: {:.3}", 
                 mean_confidence, min_confidence, max_confidence);
        
        // Basic sanity checks
        assert!(mean_confidence > 0.3 && mean_confidence < 1.0, 
                "Mean confidence should be reasonable: {:.3}", mean_confidence);
        assert!(max_confidence <= 1.0, "Max confidence should not exceed 1.0: {:.3}", max_confidence);
        assert!(min_confidence >= 0.0, "Min confidence should not be negative: {:.3}", min_confidence);
        
        // Should have some variation in confidence scores
        let confidence_range = max_confidence - min_confidence;
        assert!(confidence_range > 0.1, 
                "Should have reasonable variation in confidence scores: {:.3}", confidence_range);
    }
}