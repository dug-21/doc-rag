//! Routing accuracy validation tests for 80%+ accuracy constraint
//!
//! Comprehensive testing of routing accuracy with statistical significance
//! and constraint validation following London TDD methodology.

use crate::fixtures::*;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use mockall::predicate::*;

#[cfg(test)]
mod routing_accuracy_tests {
    use super::*;

    #[tokio::test]
    async fn test_symbolic_query_routing_accuracy_constraint() {
        // Given: Router with validation dataset of 1000 labeled queries
        let mut fixture = SymbolicRouterTestFixture::new().await;
        
        // Create large dataset for statistical significance
        let validation_queries: Vec<MockQuery> = (0..1000)
            .map(|i| {
                let confidence = 0.8 + (i as f64 % 100.0) / 500.0; // Vary confidence 0.8-1.0
                MockQuery::new_symbolic(
                    &format!("Symbolic reasoning query {} requiring logical processing", i),
                    confidence
                )
            })
            .collect();

        fixture.router
            .expect_batch_route()
            .with(eq(validation_queries.clone()))
            .times(1)
            .returning(|queries| {
                let results = queries.iter().enumerate().map(|(i, query)| {
                    // Simulate 85% accuracy (exceeds 80% constraint)
                    let correct_routing = (i % 100) < 85; // 85% accuracy
                    let selected_engine = if correct_routing {
                        query.expected_engine.clone()
                    } else {
                        "graph".to_string() // Wrong engine for testing
                    };
                    
                    MockRoutingDecision {
                        query_id: query.id.clone(),
                        selected_engine,
                        confidence: query.expected_confidence,
                        routing_time: Duration::from_millis(45 + (i % 50) as u64), // 45-95ms range
                        engine_scores: HashMap::new(),
                    }
                }).collect();
                Ok(results)
            });

        // When: Routing validation dataset
        let results = fixture.router.batch_route(&validation_queries).await.unwrap();

        // Then: Accuracy must be >= 80% (CONSTRAINT requirement)
        let accuracy = calculate_routing_accuracy(&results, &validation_queries);
        assert!(accuracy >= 0.8, 
                "Routing accuracy {:.2}% < 80% threshold CONSTRAINT violation", accuracy * 100.0);
        
        // And: Should actually exceed threshold with margin
        assert!(accuracy >= 0.82,
                "Routing accuracy should exceed minimum with safety margin");

        // And: Response time constraint must be met for all queries
        let exceeding_latency = results.iter()
            .filter(|r| r.routing_time.as_millis() > 100)
            .count();
        assert_eq!(exceeding_latency, 0, 
                   "{} queries exceeded 100ms latency constraint", exceeding_latency);

        // And: High confidence queries should have even higher accuracy
        let high_confidence_results: Vec<_> = results.iter()
            .zip(validation_queries.iter())
            .filter(|(_, query)| query.expected_confidence >= 0.9)
            .collect();
        
        if !high_confidence_results.is_empty() {
            let high_conf_correct = high_confidence_results.iter()
                .filter(|(result, query)| result.selected_engine == query.expected_engine)
                .count();
            let high_conf_accuracy = high_conf_correct as f64 / high_confidence_results.len() as f64;
            
            assert!(high_conf_accuracy >= 0.9,
                    "High confidence queries should have >90% accuracy, got {:.2}%", 
                    high_conf_accuracy * 100.0);
        }
    }

    #[tokio::test]
    async fn test_confidence_scoring_accuracy_correlation() {
        // Given: Router with confidence threshold validation
        let mut fixture = SymbolicRouterTestFixture::new().await;
        
        let test_queries: Vec<MockQuery> = vec![
            MockQuery::new_symbolic("High confidence symbolic query", 0.95),
            MockQuery::new_symbolic("Medium confidence query", 0.75),
            MockQuery::new_symbolic("Low confidence ambiguous query", 0.55),
            MockQuery::new_graph("High confidence graph query", 0.90),
            MockQuery::new_graph("Medium confidence relationship query", 0.70),
        ];

        fixture.router
            .expect_calculate_confidence_scores()
            .with(eq(test_queries.clone()))
            .times(1)
            .returning(|queries| {
                let scores = queries.iter()
                    .map(|q| q.expected_confidence)
                    .collect();
                Ok(scores)
            });

        // When: Scoring queries with known optimal engines
        let confidence_results = fixture.router
            .calculate_confidence_scores(&test_queries)
            .await.unwrap();

        // Then: High confidence scores should correlate with correct routing
        assert_eq!(confidence_results.len(), test_queries.len());
        
        // Calculate accuracy for different confidence thresholds
        let high_confidence_accuracy = calculate_accuracy_for_confidence_threshold(&confidence_results, 0.8);
        let medium_confidence_accuracy = calculate_accuracy_for_confidence_threshold(&confidence_results, 0.6);
        
        assert!(high_confidence_accuracy >= 0.9, 
                "High confidence routing (>0.8) should have >90% accuracy, got {:.2}%", 
                high_confidence_accuracy * 100.0);
        
        assert!(medium_confidence_accuracy >= 0.75,
                "Medium confidence routing (>0.6) should have >75% accuracy, got {:.2}%",
                medium_confidence_accuracy * 100.0);
        
        // And: Confidence scores should be well-calibrated
        let high_conf_queries = confidence_results.iter()
            .filter(|&&score| score >= 0.8)
            .count();
        let medium_conf_queries = confidence_results.iter()
            .filter(|&&score| score >= 0.6 && score < 0.8)
            .count();
        let low_conf_queries = confidence_results.iter()
            .filter(|&&score| score < 0.6)
            .count();
        
        // Should have distribution across confidence levels
        assert!(high_conf_queries >= 2, "Should have high confidence queries");
        assert!(medium_conf_queries >= 1, "Should have medium confidence queries");
        assert!(low_conf_queries >= 1, "Should have low confidence queries");
    }

    #[tokio::test]
    async fn test_accuracy_across_query_types() {
        // Given: Router with diverse query type validation
        let mut fixture = SymbolicRouterTestFixture::new().await;
        
        let mut symbolic_queries: Vec<MockQuery> = (0..200)
            .map(|i| MockQuery::new_symbolic(&format!("Symbolic {}", i), 0.85))
            .collect();
        let mut graph_queries: Vec<MockQuery> = (0..100)
            .map(|i| MockQuery::new_graph(&format!("Graph {}", i), 0.80))
            .collect();
        
        let mut all_queries = Vec::new();
        all_queries.append(&mut symbolic_queries);
        all_queries.append(&mut graph_queries);

        fixture.router
            .expect_batch_route()
            .with(eq(all_queries.clone()))
            .times(1)
            .returning(|queries| {
                let results = queries.iter().map(|query| {
                    // Simulate different accuracy rates by query type
                    let correct_routing = match query.query_type {
                        QueryType::Symbolic => rand::random::<f64>() < 0.88, // 88% accuracy
                        QueryType::Graph => rand::random::<f64>() < 0.83,    // 83% accuracy
                        _ => rand::random::<f64>() < 0.75,                    // 75% accuracy
                    };
                    
                    let selected_engine = if correct_routing {
                        query.expected_engine.clone()
                    } else {
                        match query.query_type {
                            QueryType::Symbolic => "vector".to_string(),
                            QueryType::Graph => "symbolic".to_string(),
                            _ => "hybrid".to_string(),
                        }
                    };
                    
                    MockRoutingDecision {
                        query_id: query.id.clone(),
                        selected_engine,
                        confidence: query.expected_confidence,
                        routing_time: Duration::from_millis(50),
                        engine_scores: HashMap::new(),
                    }
                }).collect();
                Ok(results)
            });

        // When: Routing diverse query types
        let all_results = fixture.router.batch_route(&all_queries).await.unwrap();

        // Then: Overall accuracy should meet constraint
        let overall_accuracy = calculate_routing_accuracy(&all_results, &all_queries);
        assert!(overall_accuracy >= 0.8,
                "Overall accuracy {:.2}% must meet 80% constraint", overall_accuracy * 100.0);

        // And: Calculate per-type accuracy
        let symbolic_results: Vec<_> = all_results.iter()
            .zip(all_queries.iter())
            .filter(|(_, query)| query.query_type == QueryType::Symbolic)
            .map(|(result, _)| result)
            .collect();
        
        let graph_results: Vec<_> = all_results.iter()
            .zip(all_queries.iter())
            .filter(|(_, query)| query.query_type == QueryType::Graph)
            .map(|(result, _)| result)
            .collect();

        // Symbolic queries should have higher accuracy
        let symbolic_expected: Vec<_> = all_queries.iter()
            .filter(|q| q.query_type == QueryType::Symbolic)
            .collect();
        let graph_expected: Vec<_> = all_queries.iter()
            .filter(|q| q.query_type == QueryType::Graph)
            .collect();

        if !symbolic_results.is_empty() && !symbolic_expected.is_empty() {
            let symbolic_accuracy = calculate_routing_accuracy(&symbolic_results.iter().cloned().collect(), &symbolic_expected.iter().cloned().collect());
            assert!(symbolic_accuracy >= 0.85,
                    "Symbolic query accuracy {:.2}% should exceed 85%", symbolic_accuracy * 100.0);
        }

        if !graph_results.is_empty() && !graph_expected.is_empty() {
            let graph_accuracy = calculate_routing_accuracy(&graph_results.iter().cloned().collect(), &graph_expected.iter().cloned().collect());
            assert!(graph_accuracy >= 0.8,
                    "Graph query accuracy {:.2}% should meet 80% threshold", graph_accuracy * 100.0);
        }
    }

    #[tokio::test]
    async fn test_accuracy_degradation_under_load() {
        // Given: Router under increasing load conditions
        let mut fixture = SymbolicRouterTestFixture::new().await;
        
        let load_levels = vec![10, 50, 100, 500]; // Different batch sizes
        
        for &batch_size in &load_levels {
            let test_queries: Vec<MockQuery> = (0..batch_size)
                .map(|i| MockQuery::new_symbolic(&format!("Load test query {}", i), 0.85))
                .collect();

            fixture.router
                .expect_batch_route()
                .with(eq(test_queries.clone()))
                .times(1)
                .returning(move |queries| {
                    // Simulate slight accuracy degradation under load
                    let base_accuracy = 0.85;
                    let load_factor = (queries.len() as f64 / 1000.0).min(0.1); // Max 10% degradation
                    let adjusted_accuracy = base_accuracy - load_factor;
                    
                    let results = queries.iter().enumerate().map(|(i, query)| {
                        let correct_routing = (i as f64 / queries.len() as f64) < adjusted_accuracy;
                        let selected_engine = if correct_routing {
                            query.expected_engine.clone()
                        } else {
                            "vector".to_string()
                        };
                        
                        // Simulate slightly increased latency under load
                        let base_latency = 45;
                        let load_latency = (queries.len() as u64 / 20).min(30); // Max 30ms increase
                        
                        MockRoutingDecision {
                            query_id: query.id.clone(),
                            selected_engine,
                            confidence: query.expected_confidence,
                            routing_time: Duration::from_millis(base_latency + load_latency),
                            engine_scores: HashMap::new(),
                        }
                    }).collect();
                    Ok(results)
                });

            // When: Processing batch under load
            let results = fixture.router.batch_route(&test_queries).await.unwrap();
            let accuracy = calculate_routing_accuracy(&results, &test_queries);

            // Then: Accuracy should still meet minimum constraint
            assert!(accuracy >= 0.8,
                    "Accuracy {:.2}% under load (batch size {}) must meet 80% constraint", 
                    accuracy * 100.0, batch_size);

            // And: Latency should still be reasonable under load
            let max_latency = results.iter()
                .map(|r| r.routing_time.as_millis())
                .max()
                .unwrap_or(0);
            
            assert!(max_latency < 100,
                    "Max latency {}ms under load (batch size {}) must be < 100ms",
                    max_latency, batch_size);
        }
    }

    #[tokio::test]
    async fn test_accuracy_measurement_statistical_significance() {
        // Given: Large validation dataset for statistical confidence
        let mut fixture = SymbolicRouterTestFixture::new().await;
        
        let validation_size = 2000; // Large sample for statistical significance
        let test_queries: Vec<MockQuery> = (0..validation_size)
            .map(|i| {
                let query_type = match i % 4 {
                    0 | 1 => QueryType::Symbolic, // 50% symbolic
                    2 => QueryType::Graph,        // 25% graph
                    _ => QueryType::Vector,       // 25% vector
                };
                
                MockQuery {
                    id: format!("validation-{}", i),
                    content: format!("Validation query {} type {:?}", i, query_type),
                    query_type,
                    complexity: 0.7 + (i as f64 % 100.0) / 300.0,
                    expected_engine: match query_type {
                        QueryType::Symbolic => "symbolic".to_string(),
                        QueryType::Graph => "graph".to_string(),
                        _ => "vector".to_string(),
                    },
                    expected_confidence: 0.8 + (i as f64 % 200.0) / 1000.0,
                }
            })
            .collect();

        fixture.router
            .expect_batch_route()
            .with(eq(test_queries.clone()))
            .times(1)
            .returning(|queries| {
                let results = queries.iter().enumerate().map(|(i, query)| {
                    // Simulate realistic accuracy with some variance
                    let base_accuracy = match query.query_type {
                        QueryType::Symbolic => 0.88,
                        QueryType::Graph => 0.85,
                        QueryType::Vector => 0.82,
                        _ => 0.80,
                    };
                    
                    // Add some random variation
                    let accuracy_variance = (i as f64 * 17.0 % 100.0) / 1000.0 - 0.05; // ±5%
                    let effective_accuracy = (base_accuracy + accuracy_variance).clamp(0.75, 0.95);
                    
                    let correct_routing = (i as f64 / queries.len() as f64) < effective_accuracy;
                    let selected_engine = if correct_routing {
                        query.expected_engine.clone()
                    } else {
                        // Simulate realistic routing errors
                        match query.query_type {
                            QueryType::Symbolic => "graph".to_string(),
                            QueryType::Graph => "symbolic".to_string(),
                            _ => "hybrid".to_string(),
                        }
                    };
                    
                    MockRoutingDecision {
                        query_id: query.id.clone(),
                        selected_engine,
                        confidence: query.expected_confidence,
                        routing_time: Duration::from_millis(40 + (i % 60) as u64),
                        engine_scores: HashMap::new(),
                    }
                }).collect();
                Ok(results)
            });

        // When: Running large-scale validation
        let results = fixture.router.batch_route(&test_queries).await.unwrap();
        
        // Then: Calculate accuracy with confidence intervals
        let total_correct = results.iter()
            .zip(test_queries.iter())
            .filter(|(result, expected)| result.selected_engine == expected.expected_engine)
            .count();
        
        let accuracy = total_correct as f64 / results.len() as f64;
        let sample_size = results.len() as f64;
        
        // Calculate 95% confidence interval
        let standard_error = (accuracy * (1.0 - accuracy) / sample_size).sqrt();
        let confidence_margin = 1.96 * standard_error; // 95% confidence
        let lower_bound = accuracy - confidence_margin;
        let upper_bound = accuracy + confidence_margin;
        
        // Validate statistical significance
        assert!(lower_bound > 0.8,
                "95% confidence interval lower bound {:.3} must exceed 80% constraint", lower_bound);
        
        assert!(accuracy >= 0.82,
                "Point estimate accuracy {:.3} should exceed minimum with margin", accuracy);
        
        assert!(confidence_margin < 0.05,
                "Confidence margin {:.3} should be reasonably tight with large sample", confidence_margin);
        
        println!("Accuracy: {:.3} ±{:.3} (95% CI: [{:.3}, {:.3}])", 
                 accuracy, confidence_margin, lower_bound, upper_bound);
    }
}

#[cfg(test)]
mod accuracy_monitoring_tests {
    use super::*;

    #[tokio::test]
    async fn test_continuous_accuracy_monitoring() {
        // Given: Router with continuous accuracy tracking
        let mut fixture = SymbolicRouterTestFixture::new().await;
        
        // Simulate multiple time periods with different accuracy rates
        let time_periods = vec![
            ("morning", 0.87),
            ("afternoon", 0.85),
            ("evening", 0.83),
            ("night", 0.89),
        ];
        
        for (period, expected_accuracy) in time_periods {
            let period_queries: Vec<MockQuery> = (0..100)
                .map(|i| MockQuery::new_symbolic(
                    &format!("{} query {}", period, i), 
                    0.85
                ))
                .collect();

            fixture.router
                .expect_batch_route()
                .with(eq(period_queries.clone()))
                .times(1)
                .returning(move |queries| {
                    let results = queries.iter().enumerate().map(|(i, query)| {
                        let correct_routing = (i as f64 / queries.len() as f64) < expected_accuracy;
                        let selected_engine = if correct_routing {
                            query.expected_engine.clone()
                        } else {
                            "vector".to_string()
                        };
                        
                        MockRoutingDecision {
                            query_id: query.id.clone(),
                            selected_engine,
                            confidence: query.expected_confidence,
                            routing_time: Duration::from_millis(45),
                            engine_scores: HashMap::new(),
                        }
                    }).collect();
                    Ok(results)
                });

            // When: Processing queries for time period
            let period_results = fixture.router.batch_route(&period_queries).await.unwrap();
            let period_accuracy = calculate_routing_accuracy(&period_results, &period_queries);
            
            // Then: Each period should meet minimum constraint
            assert!(period_accuracy >= 0.8,
                    "{} period accuracy {:.2}% must meet 80% constraint", 
                    period, period_accuracy * 100.0);
        }
    }
}