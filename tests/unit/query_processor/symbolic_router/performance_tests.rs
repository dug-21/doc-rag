//! Performance Tests - London TDD Implementation
//!
//! Comprehensive performance validation for symbolic router with
//! <100ms routing constraint and statistical analysis.

use super::*;
use std::time::Instant;
use tokio::time::timeout;
use std::sync::Arc;
use tokio::sync::Semaphore;

#[cfg(test)]
mod performance_tests {
    use super::*;

    /// London TDD Test: Symbolic routing latency constraint (<100ms)
    #[tokio::test]
    async fn test_symbolic_routing_latency_constraint() {
        // Given: Router with performance monitoring
        let fixture = SymbolicRouterTestFixture::new().await;
        
        // When: Processing 100 queries for statistical significance
        let mut latencies = Vec::new();
        let start_total = Instant::now();
        
        for query in &fixture.benchmark_queries {
            let start_time = Instant::now();
            let result = fixture.router
                .route_query(query, &fixture.mock_analysis)
                .await
                .unwrap();
            let latency = start_time.elapsed();
            
            latencies.push(latency);
            
            // Individual query constraint validation
            assert!(latency.as_millis() < 100, 
                    "Query routing took {}ms > 100ms constraint", 
                    latency.as_millis());
            
            // Verify result completeness
            assert!(result.confidence > 0.0);
            assert!(!result.reasoning.is_empty());
        }
        
        let total_time = start_total.elapsed();
        
        // Then: Statistical analysis of performance
        let mean_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
        let p95_latency = calculate_percentile(&latencies, 95);
        let p99_latency = calculate_percentile(&latencies, 99);
        
        println!("Performance Statistics:");
        println!("  Total queries: {}", latencies.len());
        println!("  Total time: {}ms", total_time.as_millis());
        println!("  Mean latency: {}ms", mean_latency.as_millis());
        println!("  P95 latency: {}ms", p95_latency.as_millis());
        println!("  P99 latency: {}ms", p99_latency.as_millis());
        
        // Performance constraint validation
        assert!(p95_latency.as_millis() < 100, 
                "P95 latency {}ms > 100ms constraint", p95_latency.as_millis());
        assert!(mean_latency.as_millis() < 50, 
                "Mean latency {}ms should be well below constraint", mean_latency.as_millis());
        
        // Throughput validation
        let throughput = latencies.len() as f64 / total_time.as_secs_f64();
        assert!(throughput > 10.0, 
                "Should achieve reasonable throughput: {:.1} queries/sec", throughput);
    }

    /// London TDD Test: Concurrent routing performance
    #[tokio::test]
    async fn test_concurrent_routing_performance() {
        // Given: Router with concurrent access
        let fixture = Arc::new(SymbolicRouterTestFixture::new().await);
        let concurrency_limit = 10;
        let queries_per_task = 10;
        let semaphore = Arc::new(Semaphore::new(concurrency_limit));
        
        // When: Processing queries concurrently
        let mut tasks = Vec::new();
        let start_time = Instant::now();
        
        for task_id in 0..concurrency_limit {
            let fixture_clone = fixture.clone();
            let semaphore_clone = semaphore.clone();
            
            let task = tokio::spawn(async move {
                let _permit = semaphore_clone.acquire().await.unwrap();
                let mut task_latencies = Vec::new();
                
                for i in 0..queries_per_task {
                    let query_index = (task_id * queries_per_task + i) % fixture_clone.benchmark_queries.len();
                    let query = &fixture_clone.benchmark_queries[query_index];
                    
                    let start_query = Instant::now();
                    let result = fixture_clone.router
                        .route_query(query, &fixture_clone.mock_analysis)
                        .await
                        .unwrap();
                    let query_latency = start_query.elapsed();
                    
                    task_latencies.push(query_latency);
                    
                    // Validate individual query performance under concurrency
                    assert!(query_latency.as_millis() < 200, // Relaxed under concurrency
                            "Concurrent query took {}ms > 200ms", query_latency.as_millis());
                    assert!(result.confidence > 0.0);
                }
                
                task_latencies
            });
            
            tasks.push(task);
        }
        
        // Then: Collect results from all concurrent tasks
        let mut all_latencies = Vec::new();
        for task in tasks {
            let task_latencies = task.await.unwrap();
            all_latencies.extend(task_latencies);
        }
        
        let total_concurrent_time = start_time.elapsed();
        
        // Performance analysis under concurrency
        let mean_concurrent_latency = all_latencies.iter().sum::<Duration>() / all_latencies.len() as u32;
        let p95_concurrent_latency = calculate_percentile(&all_latencies, 95);
        let concurrent_throughput = all_latencies.len() as f64 / total_concurrent_time.as_secs_f64();
        
        println!("Concurrent Performance Statistics:");
        println!("  Concurrent tasks: {}", concurrency_limit);
        println!("  Total queries: {}", all_latencies.len());
        println!("  Total time: {}ms", total_concurrent_time.as_millis());
        println!("  Mean concurrent latency: {}ms", mean_concurrent_latency.as_millis());
        println!("  P95 concurrent latency: {}ms", p95_concurrent_latency.as_millis());
        println!("  Concurrent throughput: {:.1} queries/sec", concurrent_throughput);
        
        // Concurrent performance validation
        assert!(p95_concurrent_latency.as_millis() < 300, // Relaxed for concurrency
                "P95 concurrent latency {}ms should be reasonable", p95_concurrent_latency.as_millis());
        assert!(concurrent_throughput > 20.0,
                "Concurrent throughput should be reasonable: {:.1} queries/sec", concurrent_throughput);
    }

    /// London TDD Test: Memory usage should be reasonable
    #[tokio::test]
    async fn test_memory_usage_stability() {
        // Given: Router for memory testing
        let fixture = SymbolicRouterTestFixture::new().await;
        
        // When: Processing many queries to test memory stability
        let iterations = 200;
        let mut results = Vec::new();
        
        for i in 0..iterations {
            let query_index = i % fixture.benchmark_queries.len();
            let query = &fixture.benchmark_queries[query_index];
            
            let result = fixture.router
                .route_query(query, &fixture.mock_analysis)
                .await
                .unwrap();
            
            // Store only essential data to avoid memory buildup in test
            results.push((result.confidence, result.engine.clone()));
            
            // Periodic memory pressure simulation
            if i % 50 == 0 {
                // Force garbage collection opportunity
                tokio::task::yield_now().await;
            }
        }
        
        // Then: Should complete without memory issues
        assert_eq!(results.len(), iterations);
        
        // Validate results are still reasonable after many iterations
        let final_confidence_sum: f64 = results.iter().map(|(c, _)| c).sum();
        let avg_confidence = final_confidence_sum / results.len() as f64;
        
        assert!(avg_confidence > 0.3 && avg_confidence < 1.0,
                "Average confidence should remain reasonable after {} iterations: {:.3}",
                iterations, avg_confidence);
        
        println!("Memory stability test completed: {} iterations, avg confidence: {:.3}",
                 iterations, avg_confidence);
    }

    /// London TDD Test: Cold start performance
    #[tokio::test]
    async fn test_cold_start_performance() {
        // Given: Fresh router instances for cold start testing
        let config = crate::query_processor::SymbolicRouterConfig::default();
        
        // When: Measuring first query performance on new instances
        let mut cold_start_times = Vec::new();
        
        for _ in 0..5 {
            // Create new router instance
            let start_init = Instant::now();
            let router = SymbolicQueryRouter::new(config.clone()).await.unwrap();
            let init_time = start_init.elapsed();
            
            // First query (cold start)
            let query = Query::new("What are the encryption requirements?").unwrap();
            let analysis = create_mock_semantic_analysis();
            
            let start_query = Instant::now();
            let result = router.route_query(&query, &analysis).await.unwrap();
            let query_time = start_query.elapsed();
            
            cold_start_times.push((init_time, query_time));
            
            // Validate cold start results
            assert!(result.confidence > 0.0);
            assert!(!result.reasoning.is_empty());
            
            println!("Cold start #{}: Init={}ms, FirstQuery={}ms", 
                     cold_start_times.len(), init_time.as_millis(), query_time.as_millis());
        }
        
        // Then: Analyze cold start performance
        let avg_init_time = cold_start_times.iter().map(|(i, _)| *i).sum::<Duration>() / cold_start_times.len() as u32;
        let avg_first_query_time = cold_start_times.iter().map(|(_, q)| *q).sum::<Duration>() / cold_start_times.len() as u32;
        
        println!("Cold Start Performance:");
        println!("  Average init time: {}ms", avg_init_time.as_millis());
        println!("  Average first query time: {}ms", avg_first_query_time.as_millis());
        
        // Cold start performance constraints
        assert!(avg_init_time.as_millis() < 1000,
                "Router initialization should be fast: {}ms", avg_init_time.as_millis());
        assert!(avg_first_query_time.as_millis() < 200,
                "First query should meet relaxed constraint: {}ms", avg_first_query_time.as_millis());
    }

    /// London TDD Test: Performance with different query complexities
    #[tokio::test]
    async fn test_performance_by_query_complexity() {
        // Given: Router and queries of different complexities
        let fixture = SymbolicRouterTestFixture::new().await;
        
        let simple_queries = vec![
            Query::new("What is PCI?").unwrap(),
            Query::new("Define encryption").unwrap(),
            Query::new("What is CHD?").unwrap(),
        ];
        
        let complex_queries = vec![
            Query::new("If cardholder data is stored in environment A and accessed by system B, then what encryption and access control requirements apply according to PCI DSS 3.2.1 section 3.4 and related requirements?").unwrap(),
            Query::new("Compare and contrast the encryption requirements between PCI DSS 4.0 and 3.2.1, considering the implications for multi-tenant cloud environments with cross-border data transfers").unwrap(),
        ];
        
        // When: Measuring performance by complexity
        let mut simple_times = Vec::new();
        let mut complex_times = Vec::new();
        
        // Test simple queries
        for query in &simple_queries {
            let start_time = Instant::now();
            let result = fixture.router.route_query(query, &fixture.mock_analysis).await.unwrap();
            let query_time = start_time.elapsed();
            
            simple_times.push(query_time);
            assert!(result.confidence > 0.0);
        }
        
        // Test complex queries
        for query in &complex_queries {
            let start_time = Instant::now();
            let result = fixture.router.route_query(query, &fixture.mock_analysis).await.unwrap();
            let query_time = start_time.elapsed();
            
            complex_times.push(query_time);
            assert!(result.confidence > 0.0);
        }
        
        // Then: Analyze performance by complexity
        let avg_simple_time = simple_times.iter().sum::<Duration>() / simple_times.len() as u32;
        let avg_complex_time = complex_times.iter().sum::<Duration>() / complex_times.len() as u32;
        
        println!("Performance by Complexity:");
        println!("  Simple queries: {}ms average", avg_simple_time.as_millis());
        println!("  Complex queries: {}ms average", avg_complex_time.as_millis());
        
        // Performance expectations
        assert!(avg_simple_time.as_millis() < 50,
                "Simple queries should be very fast: {}ms", avg_simple_time.as_millis());
        assert!(avg_complex_time.as_millis() < 100,
                "Complex queries should still meet constraint: {}ms", avg_complex_time.as_millis());
        
        // Complex queries may take longer, but should still be reasonable
        let complexity_ratio = avg_complex_time.as_secs_f64() / avg_simple_time.as_secs_f64();
        assert!(complexity_ratio < 5.0,
                "Complexity should not cause excessive slowdown: {:.1}x", complexity_ratio);
    }

    /// London TDD Test: Performance regression detection
    #[tokio::test]
    async fn test_performance_regression_detection() {
        // Given: Baseline performance measurements
        let fixture = SymbolicRouterTestFixture::new().await;
        let regression_test_queries = &fixture.benchmark_queries[0..20];
        
        // When: Running baseline measurements
        let mut baseline_times = Vec::new();
        
        for query in regression_test_queries {
            let start_time = Instant::now();
            let _ = fixture.router.route_query(query, &fixture.mock_analysis).await.unwrap();
            let query_time = start_time.elapsed();
            baseline_times.push(query_time);
        }
        
        let baseline_p95 = calculate_percentile(&baseline_times, 95);
        
        // Simulate second run (checking for consistency)
        let mut second_run_times = Vec::new();
        
        for query in regression_test_queries {
            let start_time = Instant::now();
            let _ = fixture.router.route_query(query, &fixture.mock_analysis).await.unwrap();
            let query_time = start_time.elapsed();
            second_run_times.push(query_time);
        }
        
        let second_run_p95 = calculate_percentile(&second_run_times, 95);
        
        // Then: Performance should be consistent
        let performance_ratio = second_run_p95.as_secs_f64() / baseline_p95.as_secs_f64();
        
        println!("Performance Consistency:");
        println!("  Baseline P95: {}ms", baseline_p95.as_millis());
        println!("  Second run P95: {}ms", second_run_p95.as_millis());
        println!("  Performance ratio: {:.2}x", performance_ratio);
        
        // Performance should be stable (within 50% variance)
        assert!(performance_ratio < 1.5 && performance_ratio > 0.5,
                "Performance should be consistent between runs: {:.2}x variance", 
                performance_ratio);
        
        // Both runs should meet constraints
        assert!(baseline_p95.as_millis() < 100,
                "Baseline P95 should meet constraint: {}ms", baseline_p95.as_millis());
        assert!(second_run_p95.as_millis() < 100,
                "Second run P95 should meet constraint: {}ms", second_run_p95.as_millis());
    }

    /// London TDD Test: Timeout handling
    #[tokio::test]
    async fn test_timeout_handling() {
        // Given: Router with timeout testing
        let fixture = SymbolicRouterTestFixture::new().await;
        let test_query = &fixture.benchmark_queries[0];
        
        // When: Setting aggressive timeout
        let timeout_duration = Duration::from_millis(500);
        
        let result = timeout(timeout_duration, 
            fixture.router.route_query(test_query, &fixture.mock_analysis)
        ).await;
        
        // Then: Should complete within timeout
        match result {
            Ok(Ok(routing_result)) => {
                // Successful completion
                assert!(routing_result.confidence > 0.0);
                println!("Query completed successfully within {}ms timeout", 
                         timeout_duration.as_millis());
            },
            Ok(Err(e)) => {
                // Query error (but not timeout)
                panic!("Query failed with error: {:?}", e);
            },
            Err(_) => {
                // Timeout occurred
                panic!("Query timed out after {}ms - should complete faster", 
                       timeout_duration.as_millis());
            }
        }
    }
}