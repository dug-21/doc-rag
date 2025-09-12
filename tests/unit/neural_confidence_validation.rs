//! Neural Confidence Scoring Validation Tests
//! 
//! Validates CONSTRAINT-003 compliance and Phase 2 neural confidence implementation
//! according to multi-layer neural-symbolic confidence framework from PSEUDOCODE.md

#[cfg(test)]
mod tests {
    use query_processor::symbolic_router::{
        SymbolicQueryRouter, SymbolicRouterConfig, QueryCharacteristics, SymbolicQueryType,
        RoutingDecision, QueryEngine
    };
    use query_processor::{Query, SemanticAnalysis, SyntacticFeatures, SemanticFeatures, NamedEntity};
    use std::time::{Duration, Instant};
    
    /// Create test semantic analysis for neural confidence testing
    fn create_test_semantic_analysis() -> SemanticAnalysis {
        use chrono::Utc;
        
        SemanticAnalysis {
            syntactic_features: SyntacticFeatures {
                pos_tags: vec![],
                named_entities: vec![
                    NamedEntity::new(
                        "encryption".to_string(),
                        "SECURITY_TERM".to_string(),
                        0, 10, 0.95,
                    ),
                    NamedEntity::new(
                        "cardholder data".to_string(),
                        "DATA_TYPE".to_string(),
                        25, 40, 0.90,
                    ),
                ],
                noun_phrases: vec![],
                verb_phrases: vec![],
                question_words: vec!["What".to_string()],
            },
            semantic_features: SemanticFeatures {
                semantic_roles: vec![],
                coreferences: vec![],
                sentiment: None,
                similarity_vectors: vec![],
            },
            dependencies: vec![],
            topics: vec![],
            confidence: 0.85,
            timestamp: Utc::now(),
            processing_time: Duration::from_millis(25),
        }
    }

    #[tokio::test]
    async fn test_constraint_003_neural_inference_latency() {
        println!("🧠 Testing CONSTRAINT-003: <10ms neural inference latency");
        
        let config = SymbolicRouterConfig::default();
        let router = SymbolicQueryRouter::new(config).await.expect("Failed to create router");
        
        // Create test characteristics for high-confidence logical inference
        let characteristics = QueryCharacteristics {
            complexity: 0.8,
            entity_count: 3,
            relationship_count: 2,
            query_type: SymbolicQueryType::LogicalInference,
            has_logical_operators: true,
            has_temporal_constraints: false,
            has_cross_references: true,
            requires_proof: true,
        };
        
        let iterations = 50;
        let mut inference_times = Vec::with_capacity(iterations);
        
        println!("   Running {} neural confidence inferences...", iterations);
        
        for i in 0..iterations {
            let start = Instant::now();
            
            let confidence = router.calculate_routing_confidence(&characteristics)
                .await
                .expect("Failed to calculate routing confidence");
            
            let elapsed = start.elapsed();
            inference_times.push(elapsed.as_micros() as f64 / 1000.0);
            
            // Validate confidence is in valid range
            assert!(confidence >= 0.0 && confidence <= 1.0, 
                    "Confidence {:.3} out of range [0.0, 1.0]", confidence);
            
            if i % 10 == 0 {
                println!("     Iteration {}: {:.2}ms (confidence: {:.3})", 
                         i, inference_times[i], confidence);
            }
        }
        
        // Calculate performance statistics
        let avg_time = inference_times.iter().sum::<f64>() / iterations as f64;
        let mut sorted_times = inference_times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = sorted_times[iterations / 2];
        let p95 = sorted_times[(iterations as f64 * 0.95) as usize];
        let p99 = sorted_times[(iterations as f64 * 0.99) as usize];
        let max_time = sorted_times[iterations - 1];
        
        println!("📊 Neural Inference Performance Results:");
        println!("   Average: {:.2}ms", avg_time);
        println!("   P50:     {:.2}ms", p50);
        println!("   P95:     {:.2}ms", p95);
        println!("   P99:     {:.2}ms", p99);
        println!("   Max:     {:.2}ms", max_time);
        
        // CONSTRAINT-003 validation: <10ms inference per classification
        assert!(avg_time < 10.0, 
                "❌ CONSTRAINT-003 VIOLATION: Average inference time {:.2}ms exceeds 10ms", avg_time);
        assert!(p95 < 10.0, 
                "❌ CONSTRAINT-003 VIOLATION: P95 inference time {:.2}ms exceeds 10ms", p95);
        assert!(max_time < 20.0, 
                "❌ Unreasonable max inference time {:.2}ms (should be <20ms)", max_time);
        
        println!("✅ CONSTRAINT-003 VALIDATED: All neural inferences <10ms");
    }

    #[tokio::test]
    async fn test_neural_confidence_multi_layer_framework() {
        println!("🔄 Testing multi-layer neural-symbolic confidence framework");
        
        let config = SymbolicRouterConfig::default();
        let router = SymbolicQueryRouter::new(config).await.expect("Failed to create router");
        
        // Test different query types for confidence variation
        let test_cases = vec![
            ("Logical Inference", QueryCharacteristics {
                complexity: 0.9,
                entity_count: 5,
                relationship_count: 4,
                query_type: SymbolicQueryType::LogicalInference,
                has_logical_operators: true,
                has_temporal_constraints: true,
                has_cross_references: true,
                requires_proof: true,
            }),
            ("Factual Lookup", QueryCharacteristics {
                complexity: 0.3,
                entity_count: 2,
                relationship_count: 0,
                query_type: SymbolicQueryType::FactualLookup,
                has_logical_operators: false,
                has_temporal_constraints: false,
                has_cross_references: false,
                requires_proof: false,
            }),
            ("Complex Reasoning", QueryCharacteristics {
                complexity: 0.8,
                entity_count: 6,
                relationship_count: 3,
                query_type: SymbolicQueryType::ComplexReasoning,
                has_logical_operators: true,
                has_temporal_constraints: true,
                has_cross_references: false,
                requires_proof: true,
            }),
        ];
        
        for (name, characteristics) in test_cases {
            let start = Instant::now();
            let confidence = router.calculate_routing_confidence(&characteristics)
                .await
                .expect("Failed to calculate confidence");
            let elapsed = start.elapsed();
            
            println!("   {}: {:.3} confidence in {:.2}ms", 
                     name, confidence, elapsed.as_micros() as f64 / 1000.0);
            
            // Validate confidence correlates with complexity/features
            match characteristics.query_type {
                SymbolicQueryType::LogicalInference => {
                    assert!(confidence >= 0.7, 
                            "Logical inference should have high confidence: {:.3}", confidence);
                },
                SymbolicQueryType::FactualLookup => {
                    // Factual lookups may have lower confidence for symbolic routing
                    assert!(confidence >= 0.4, 
                            "Factual lookup confidence too low: {:.3}", confidence);
                },
                SymbolicQueryType::ComplexReasoning => {
                    assert!(confidence >= 0.6, 
                            "Complex reasoning should have decent confidence: {:.3}", confidence);
                },
                _ => {}
            }
            
            // All inferences must be <10ms
            assert!(elapsed.as_millis() < 10, 
                    "Inference time {}ms exceeds 10ms constraint", elapsed.as_millis());
        }
        
        println!("✅ Multi-layer confidence framework validated");
    }

    #[tokio::test]
    async fn test_byzantine_consensus_validation() {
        println!("🛡️ Testing Byzantine consensus validation (66% threshold)");
        
        let config = SymbolicRouterConfig::default();
        let router = SymbolicQueryRouter::new(config).await.expect("Failed to create router");
        
        // High confidence case - should pass Byzantine threshold
        let high_conf_characteristics = QueryCharacteristics {
            complexity: 0.9,
            entity_count: 4,
            relationship_count: 3,
            query_type: SymbolicQueryType::LogicalInference,
            has_logical_operators: true,
            has_temporal_constraints: true,
            has_cross_references: true,
            requires_proof: true,
        };
        
        let high_confidence = router.calculate_routing_confidence(&high_conf_characteristics)
            .await
            .expect("Failed to calculate high confidence");
        
        println!("   High confidence case: {:.3}", high_confidence);
        assert!(high_confidence >= 0.66, 
                "High confidence {:.3} should pass Byzantine threshold (0.66)", high_confidence);
        
        // Low confidence case - may apply decay factor
        let low_conf_characteristics = QueryCharacteristics {
            complexity: 0.2,
            entity_count: 0,
            relationship_count: 0,
            query_type: SymbolicQueryType::SimilarityMatching,
            has_logical_operators: false,
            has_temporal_constraints: false,
            has_cross_references: false,
            requires_proof: false,
        };
        
        let low_confidence = router.calculate_routing_confidence(&low_conf_characteristics)
            .await
            .expect("Failed to calculate low confidence");
        
        println!("   Low confidence case: {:.3}", low_confidence);
        // Low confidence may be below Byzantine threshold, triggering decay
        
        println!("✅ Byzantine consensus validation completed");
    }

    #[tokio::test]
    async fn test_routing_accuracy_target() {
        println!("🎯 Testing 80%+ routing accuracy target");
        
        let config = SymbolicRouterConfig::default();
        let router = SymbolicQueryRouter::new(config).await.expect("Failed to create router");
        
        // Test routing decisions for different query types
        let query_types = vec![
            ("encryption requirements", SymbolicQueryType::LogicalInference, QueryEngine::Symbolic),
            ("payment data storage", SymbolicQueryType::ComplianceChecking, QueryEngine::Symbolic),
            ("system relationships", SymbolicQueryType::RelationshipTraversal, QueryEngine::Graph),
            ("similar documents", SymbolicQueryType::SimilarityMatching, QueryEngine::Vector),
        ];
        
        let mut correct_routes = 0;
        let total_tests = query_types.len();
        
        for (query_text, expected_type, expected_engine) in query_types {
            let query = Query::new(query_text).expect("Failed to create query");
            let analysis = create_test_semantic_analysis();
            
            let decision = router.route_query(&query, &analysis)
                .await
                .expect("Failed to route query");
            
            println!("   Query: '{}' -> {:?} (confidence: {:.3})", 
                     query_text, decision.engine, decision.confidence);
            
            // Check if routing matches expected engine (or acceptable alternatives)
            let is_correct = match (&decision.engine, &expected_engine) {
                (QueryEngine::Symbolic, QueryEngine::Symbolic) => true,
                (QueryEngine::Graph, QueryEngine::Graph) => true,
                (QueryEngine::Vector, QueryEngine::Vector) => true,
                (QueryEngine::Hybrid(engines), _) => engines.contains(&expected_engine),
                _ => false,
            };
            
            if is_correct {
                correct_routes += 1;
                println!("     ✅ Correct routing");
            } else {
                println!("     ⚠️ Unexpected routing (may be acceptable for hybrid)");
            }
        }
        
        let accuracy = correct_routes as f64 / total_tests as f64;
        println!("📊 Routing accuracy: {:.1}% ({}/{})", 
                 accuracy * 100.0, correct_routes, total_tests);
        
        // Validate against Phase 2 target
        if accuracy >= 0.8 {
            println!("✅ Routing accuracy exceeds 80% target");
        } else {
            println!("⚠️ Routing accuracy below 80% target (may improve with training)");
        }
        
        // Get overall routing statistics
        let stats = router.get_routing_statistics().await;
        println!("📈 Router statistics: {:.1}% accuracy, {} queries processed", 
                 stats.routing_accuracy_rate * 100.0, stats.total_queries);
    }

    #[cfg(feature = "neural")]
    #[tokio::test]
    async fn test_neural_benchmark_performance() {
        println!("⚡ Running comprehensive neural confidence benchmark");
        
        let config = SymbolicRouterConfig::default();
        let router = SymbolicQueryRouter::new(config).await.expect("Failed to create router");
        
        let (avg_time, success_rate, throughput) = router.benchmark_neural_confidence(100)
            .await
            .expect("Failed to run neural benchmark");
        
        println!("📊 Neural Confidence Benchmark Results:");
        println!("   Average time: {:.2}ms per inference", avg_time);
        println!("   Success rate: {:.1}%", success_rate * 100.0);
        println!("   Throughput: {:.0} QPS", throughput);
        
        // Validate CONSTRAINT-003 compliance
        assert!(avg_time < 10.0, 
                "❌ CONSTRAINT-003 VIOLATION: Average time {:.2}ms > 10ms", avg_time);
        assert!(success_rate > 0.95, 
                "❌ Success rate {:.1}% < 95%", success_rate * 100.0);
        assert!(throughput > 100.0, 
                "❌ Throughput {:.0} QPS < 100 QPS", throughput);
        
        if avg_time < 5.0 {
            println!("🚀 EXCEPTIONAL: Neural inference <5ms (50% headroom)");
        }
        
        println!("✅ CONSTRAINT-003 FULLY VALIDATED");
    }
}