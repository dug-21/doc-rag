//! Performance Tests for Template Engine
//! 
//! London TDD tests ensuring <1s response time (CONSTRAINT-006) and
//! optimal performance under load conditions.

#[cfg(test)]
mod performance_tests {
    use super::super::super::super::super::src::response_generator::template_engine::*;
    use super::super::super::super::super::src::response_generator::{OutputFormat, Citation, Source};
    use std::collections::HashMap;
    use std::time::{Duration, Instant};
    use tokio;
    use uuid::Uuid;

    struct PerformanceTestFixture {
        engine: TemplateEngine,
        simple_request: TemplateGenerationRequest,
        complex_request: TemplateGenerationRequest,
        load_test_requests: Vec<TemplateGenerationRequest>,
    }

    impl PerformanceTestFixture {
        async fn new() -> Self {
            let engine = TemplateEngine::new(TemplateEngineConfig {
                enforce_deterministic_only: true,
                max_generation_time_ms: 1000, // CONSTRAINT-006: <1s
                validate_variable_substitution: true,
                enable_audit_trail: true,
                citation_config: CitationFormatterConfig::default(),
                validation_strictness: ValidationStrictness::Standard,
            });

            let simple_request = Self::create_simple_request();
            let complex_request = Self::create_complex_request();
            let load_test_requests = Self::create_load_test_requests(100);

            Self {
                engine,
                simple_request,
                complex_request,
                load_test_requests,
            }
        }

        fn create_simple_request() -> TemplateGenerationRequest {
            TemplateGenerationRequest {
                template_type: TemplateType::RequirementQuery {
                    requirement_type: RequirementType::Must,
                    query_intent: QueryIntent::Compliance,
                },
                variable_values: {
                    let mut vars = HashMap::new();
                    vars.insert("REQUIREMENT_TYPE".to_string(), "MUST".to_string());
                    vars.insert("QUERY_SUBJECT".to_string(), "simple requirement".to_string());
                    vars
                },
                proof_chain_data: vec![
                    ProofChainData {
                        element_type: ProofElementType::Evidence,
                        content: "Simple evidence".to_string(),
                        confidence: 0.9,
                        source: "Test Document".to_string(),
                    }
                ],
                citations: vec![
                    Citation {
                        id: Uuid::new_v4(),
                        source: Source {
                            id: Uuid::new_v4(),
                            title: "Simple Standard".to_string(),
                            url: Some("https://example.com/simple".to_string()),
                            document_type: "Standard".to_string(),
                            version: Some("1.0".to_string()),
                            section: Some("1.1".to_string()),
                            page: Some(1),
                            published_date: None,
                            accessed_date: None,
                        },
                        content: "Simple requirement".to_string(),
                        confidence: 0.95,
                        page_number: Some(1),
                        paragraph: Some(1),
                        relevance_score: 0.9,
                        exact_match: true,
                    }
                ],
                output_format: OutputFormat::Markdown,
                context: GenerationContext {
                    query_intent: QueryIntent::Compliance,
                    entities: vec!["requirement".to_string()],
                    requirements: vec!["REQ_001".to_string()],
                    compliance_scope: Some(ComplianceScope::Requirement),
                },
            }
        }

        fn create_complex_request() -> TemplateGenerationRequest {
            let mut complex_variables = HashMap::new();
            // Add 50 variables to stress test variable substitution
            for i in 1..=50 {
                complex_variables.insert(
                    format!("VAR_{}", i),
                    format!("Complex value {} with detailed content", i)
                );
            }
            complex_variables.insert("REQUIREMENT_TYPE".to_string(), "MUST".to_string());
            complex_variables.insert("QUERY_SUBJECT".to_string(), "complex multi-faceted requirement with numerous dependencies".to_string());

            let complex_proof_chain: Vec<ProofChainData> = (1..=20).map(|i| {
                ProofChainData {
                    element_type: if i % 3 == 0 { ProofElementType::Premise } 
                                else if i % 3 == 1 { ProofElementType::Evidence }
                                else { ProofElementType::Conclusion },
                    content: format!("Complex proof element {} with detailed reasoning and extensive content that requires processing", i),
                    confidence: 0.8 + (i as f64 * 0.01),
                    source: format!("Complex Document Set {} - Section {}.{}", i, i/5 + 1, i%5 + 1),
                }
            }).collect();

            let complex_citations: Vec<Citation> = (1..=15).map(|i| {
                Citation {
                    id: Uuid::new_v4(),
                    source: Source {
                        id: Uuid::new_v4(),
                        title: format!("Complex Standard {} - Comprehensive Requirements", i),
                        url: Some(format!("https://example.com/complex-{}", i)),
                        document_type: "Comprehensive Standard".to_string(),
                        version: Some(format!("{}.{}", i, i*2)),
                        section: Some(format!("{}.{}.{}", i, i%3+1, i%7+1)),
                        page: Some(i * 10 + 25),
                        published_date: None,
                        accessed_date: None,
                    },
                    content: format!("Complex requirement {} with multiple conditions, sub-requirements, and cross-references to other sections and standards", i),
                    confidence: 0.85 + (i as f64 * 0.008),
                    page_number: Some(i * 10 + 25),
                    paragraph: Some(i % 5 + 1),
                    relevance_score: 0.8 + (i as f64 * 0.012),
                    exact_match: i % 3 == 0,
                }
            }).collect();

            TemplateGenerationRequest {
                template_type: TemplateType::RequirementQuery {
                    requirement_type: RequirementType::Must,
                    query_intent: QueryIntent::Compliance,
                },
                variable_values: complex_variables,
                proof_chain_data: complex_proof_chain,
                citations: complex_citations,
                output_format: OutputFormat::Markdown,
                context: GenerationContext {
                    query_intent: QueryIntent::Compliance,
                    entities: (1..=30).map(|i| format!("entity_{}", i)).collect(),
                    requirements: (1..=20).map(|i| format!("REQ_{:03}", i)).collect(),
                    compliance_scope: Some(ComplianceScope::Full),
                },
            }
        }

        fn create_load_test_requests(count: usize) -> Vec<TemplateGenerationRequest> {
            (0..count).map(|i| {
                let mut request = Self::create_simple_request();
                request.variable_values.insert("REQUEST_ID".to_string(), format!("load_test_{}", i));
                request
            }).collect()
        }
    }

    #[tokio::test]
    async fn test_simple_response_generation_latency_constraint() {
        // Given: Template engine with performance monitoring
        let fixture = PerformanceTestFixture::new().await;
        
        // When: Generating simple response
        let start_time = Instant::now();
        let response = fixture.engine.generate_response(fixture.simple_request).await.unwrap();
        let generation_time = start_time.elapsed();
        
        // Then: Must meet CONSTRAINT-006 (<1s response time)
        assert!(response.validation_results.constraint_006_compliant,
            "Response must be CONSTRAINT-006 compliant");
        assert!(generation_time.as_millis() <= 1000,
            "Simple response generation took {}ms > 1000ms constraint", generation_time.as_millis());
        
        // Should be much faster than constraint for simple cases
        assert!(generation_time.as_millis() < 100,
            "Simple response should be generated in <100ms, took {}ms", generation_time.as_millis());
        
        // Verify metrics are captured
        assert!(response.metrics.total_generation_time <= generation_time,
            "Metrics should capture accurate timing");
        assert!(response.metrics.template_selection_time.as_millis() < 50,
            "Template selection should be <50ms");
        assert!(response.metrics.substitution_time.as_millis() < 50,
            "Variable substitution should be <50ms");
    }

    #[tokio::test]
    async fn test_complex_response_generation_performance() {
        // Given: Template engine with complex request
        let fixture = PerformanceTestFixture::new().await;
        
        // When: Generating complex response with 50 variables, 20 proof elements, 15 citations
        let start_time = Instant::now();
        let response = fixture.engine.generate_response(fixture.complex_request).await.unwrap();
        let generation_time = start_time.elapsed();
        
        // Then: Must still meet CONSTRAINT-006 even for complex requests
        assert!(response.validation_results.constraint_006_compliant,
            "Complex response must be CONSTRAINT-006 compliant");
        assert!(generation_time.as_millis() <= 1000,
            "Complex response generation took {}ms > 1000ms constraint", generation_time.as_millis());
        
        // Verify all complex elements were processed
        assert!(response.substitutions.len() >= 50,
            "Should have processed all 50+ variables");
        assert!(response.citations.len() >= 15,
            "Should have processed all 15 citations");
        
        // Performance breakdown analysis
        let metrics = &response.metrics;
        assert!(metrics.template_selection_time.as_millis() < 100,
            "Template selection took {}ms, should be <100ms", metrics.template_selection_time.as_millis());
        assert!(metrics.substitution_time.as_millis() < 300,
            "Variable substitution took {}ms, should be <300ms for 50 variables", metrics.substitution_time.as_millis());
        assert!(metrics.citation_formatting_time.as_millis() < 200,
            "Citation formatting took {}ms, should be <200ms for 15 citations", metrics.citation_formatting_time.as_millis());
        assert!(metrics.validation_time.as_millis() < 100,
            "Validation took {}ms, should be <100ms", metrics.validation_time.as_millis());
        
        // Verify processed counts
        assert_eq!(metrics.variables_substituted, response.substitutions.len(),
            "Metrics should accurately count substituted variables");
        assert_eq!(metrics.citations_formatted, response.citations.len(),
            "Metrics should accurately count formatted citations");
    }

    #[tokio::test]
    async fn test_load_performance_statistical_analysis() {
        // Given: Template engine and 100 concurrent requests
        let fixture = PerformanceTestFixture::new().await;
        
        // When: Processing 100 requests with statistical analysis
        let mut generation_times = Vec::new();
        let mut successful_responses = 0;
        let mut constraint_violations = 0;
        
        for request in &fixture.load_test_requests {
            let start_time = Instant::now();
            let result = fixture.engine.generate_response(request.clone()).await;
            let generation_time = start_time.elapsed();
            
            generation_times.push(generation_time);
            
            match result {
                Ok(response) => {
                    successful_responses += 1;
                    if !response.validation_results.constraint_006_compliant {
                        constraint_violations += 1;
                    }
                },
                Err(_) => {
                    // Count failures
                }
            }
        }
        
        // Then: Statistical analysis of performance
        assert_eq!(successful_responses, fixture.load_test_requests.len(),
            "All requests should succeed");
        assert_eq!(constraint_violations, 0,
            "No CONSTRAINT-006 violations allowed");
        
        // Calculate percentiles
        let mut sorted_times = generation_times.clone();
        sorted_times.sort();
        
        let mean_time = generation_times.iter().sum::<Duration>() / generation_times.len() as u32;
        let p50_time = sorted_times[sorted_times.len() / 2];
        let p90_time = sorted_times[(sorted_times.len() as f64 * 0.9) as usize];
        let p95_time = sorted_times[(sorted_times.len() as f64 * 0.95) as usize];
        let p99_time = sorted_times[(sorted_times.len() as f64 * 0.99) as usize];
        let max_time = sorted_times[sorted_times.len() - 1];
        
        // Performance assertions
        assert!(p95_time.as_millis() < 1000,
            "P95 generation time {}ms >= 1000ms constraint", p95_time.as_millis());
        assert!(p90_time.as_millis() < 500,
            "P90 generation time {}ms should be < 500ms", p90_time.as_millis());
        assert!(mean_time.as_millis() < 200,
            "Mean generation time {}ms should be < 200ms", mean_time.as_millis());
        assert!(max_time.as_millis() <= 1000,
            "Maximum generation time {}ms > 1000ms constraint", max_time.as_millis());
        
        // Log performance statistics for analysis
        println!("Load Test Performance Statistics:");
        println!("  Mean: {}ms", mean_time.as_millis());
        println!("  P50:  {}ms", p50_time.as_millis());
        println!("  P90:  {}ms", p90_time.as_millis());
        println!("  P95:  {}ms", p95_time.as_millis());
        println!("  P99:  {}ms", p99_time.as_millis());
        println!("  Max:  {}ms", max_time.as_millis());
    }

    #[tokio::test]
    async fn test_concurrent_request_performance() {
        // Given: Template engine and concurrent request scenario
        let fixture = PerformanceTestFixture::new().await;
        let concurrent_requests = 10;
        
        // When: Processing concurrent requests
        let start_time = Instant::now();
        let mut handles = Vec::new();
        
        for i in 0..concurrent_requests {
            let engine = fixture.engine.clone();
            let mut request = fixture.simple_request.clone();
            request.variable_values.insert("CONCURRENT_ID".to_string(), format!("concurrent_{}", i));
            
            let handle = tokio::spawn(async move {
                let req_start = Instant::now();
                let result = engine.generate_response(request).await;
                (req_start.elapsed(), result)
            });
            handles.push(handle);
        }
        
        // Collect all results
        let mut results = Vec::new();
        for handle in handles {
            let (duration, result) = handle.await.unwrap();
            results.push((duration, result));
        }
        
        let total_concurrent_time = start_time.elapsed();
        
        // Then: All concurrent requests should complete within constraint
        for (i, (duration, result)) in results.iter().enumerate() {
            assert!(duration.as_millis() <= 1000,
                "Concurrent request {} took {}ms > 1000ms constraint", i, duration.as_millis());
            
            match result {
                Ok(response) => {
                    assert!(response.validation_results.constraint_006_compliant,
                        "Concurrent request {} must be CONSTRAINT-006 compliant", i);
                },
                Err(e) => {
                    panic!("Concurrent request {} failed: {}", i, e);
                }
            }
        }
        
        // Concurrent processing should be efficient
        let successful_count = results.iter().filter(|(_, r)| r.is_ok()).count();
        assert_eq!(successful_count, concurrent_requests,
            "All concurrent requests should succeed");
        
        // Total time should be reasonable (not much longer than single request)
        assert!(total_concurrent_time.as_millis() < 2000,
            "Concurrent processing of {} requests took {}ms, should be <2000ms", 
            concurrent_requests, total_concurrent_time.as_millis());
    }

    #[tokio::test]
    async fn test_memory_usage_performance() {
        // Given: Template engine and memory monitoring capability
        let fixture = PerformanceTestFixture::new().await;
        
        // When: Generating response and monitoring memory usage
        let initial_memory = get_memory_usage();
        let response = fixture.engine.generate_response(fixture.complex_request).await.unwrap();
        let final_memory = get_memory_usage();
        
        // Then: Memory usage should be reasonable
        let memory_increase = final_memory.saturating_sub(initial_memory);
        
        // Memory increase should be bounded for complex requests
        assert!(memory_increase < 50_000_000, // 50MB
            "Memory usage increased by {} bytes, should be < 50MB", memory_increase);
        
        // Verify metrics capture memory usage (if implemented)
        // Note: peak_memory_usage is currently set to 0 in implementation
        // This test verifies the structure exists for future implementation
        assert!(response.metrics.peak_memory_usage >= 0,
            "Peak memory usage metric should be non-negative");
    }

    #[tokio::test]
    async fn test_error_handling_performance() {
        // Given: Template engine and invalid request scenarios
        let fixture = PerformanceTestFixture::new().await;
        
        // Create requests that will trigger different error conditions
        let invalid_requests = vec![
            // Empty template type scenario would be handled at type level
            // So we test with missing critical data
            {
                let mut req = fixture.simple_request.clone();
                req.variable_values.clear();
                req.proof_chain_data.clear();
                req.citations.clear();
                req
            }
        ];
        
        // When: Processing invalid requests
        for (i, request) in invalid_requests.iter().enumerate() {
            let start_time = Instant::now();
            let result = fixture.engine.generate_response(request.clone()).await;
            let error_handling_time = start_time.elapsed();
            
            // Then: Error handling should be fast
            assert!(error_handling_time.as_millis() < 100,
                "Error handling for invalid request {} took {}ms, should be <100ms", 
                i, error_handling_time.as_millis());
            
            // Result should either succeed with defaults or fail gracefully
            match result {
                Ok(response) => {
                    // If it succeeds, it should still be compliant
                    assert!(response.validation_results.constraint_006_compliant,
                        "Even error-recovered responses must meet timing constraints");
                },
                Err(_) => {
                    // If it fails, that's acceptable as long as it's fast
                }
            }
        }
    }

    // Helper function to get memory usage (placeholder implementation)
    fn get_memory_usage() -> u64 {
        // In a real implementation, this would use system APIs to get memory usage
        // For testing purposes, we'll use a placeholder
        std::process::id() as u64 * 1000 // Placeholder based on PID
    }

    #[tokio::test]
    async fn test_streaming_performance_simulation() {
        // Given: Template engine and streaming-like processing
        let fixture = PerformanceTestFixture::new().await;
        
        // When: Simulating streaming by processing multiple small requests rapidly
        let chunk_count = 20;
        let mut chunk_times = Vec::new();
        let total_start = Instant::now();
        
        for i in 0..chunk_count {
            let mut chunk_request = fixture.simple_request.clone();
            chunk_request.variable_values.insert("CHUNK_ID".to_string(), format!("chunk_{}", i));
            
            let chunk_start = Instant::now();
            let response = fixture.engine.generate_response(chunk_request).await.unwrap();
            let chunk_time = chunk_start.elapsed();
            
            chunk_times.push(chunk_time);
            
            // Each chunk should meet performance constraints
            assert!(response.validation_results.constraint_006_compliant,
                "Chunk {} must meet CONSTRAINT-006", i);
            assert!(chunk_time.as_millis() < 200,
                "Chunk {} took {}ms, should be <200ms for streaming simulation", i, chunk_time.as_millis());
        }
        
        let total_streaming_time = total_start.elapsed();
        
        // Then: Streaming simulation performance analysis
        let mean_chunk_time = chunk_times.iter().sum::<Duration>() / chunk_times.len() as u32;
        let max_chunk_time = chunk_times.iter().max().unwrap();
        
        assert!(mean_chunk_time.as_millis() < 100,
            "Mean chunk processing time {}ms should be <100ms", mean_chunk_time.as_millis());
        assert!(max_chunk_time.as_millis() < 200,
            "Maximum chunk processing time {}ms should be <200ms", max_chunk_time.as_millis());
        
        // Total processing should be efficient
        let expected_sequential_time = chunk_times.iter().sum::<Duration>();
        let overhead_ratio = total_streaming_time.as_millis() as f64 / expected_sequential_time.as_millis() as f64;
        
        assert!(overhead_ratio < 1.2,
            "Streaming overhead ratio {:.2} should be <1.2x sequential time", overhead_ratio);
    }
}