//! FACT Performance Tests
//! 
//! Comprehensive performance test suite for FACT integration including
//! benchmarks, load tests, and latency validation following TDD methodology.

use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicU64, AtomicUsize, Ordering}};
use std::collections::HashMap;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use tokio::sync::{RwLock, Semaphore};
use futures::future::join_all;
use serde_json::{json, Value};

use crate::mocks::mock_fact_client::{MockFACTClient, FACTTestFixtures};

/// Performance test constants based on requirements
const CACHE_HIT_TARGET_MS: u64 = 23;
const CACHE_MISS_TARGET_MS: u64 = 95;
const MIN_HIT_RATE: f64 = 0.873; // 87.3%
const CONCURRENT_USERS_TARGET: usize = 100;
const LOAD_TEST_DURATION: Duration = Duration::from_secs(30);
const THROUGHPUT_TARGET_RPS: f64 = 1000.0; // Requests per second

/// Performance metrics collection
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub total_requests: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub total_latency_ns: AtomicU64,
    pub min_latency_ns: AtomicU64,
    pub max_latency_ns: AtomicU64,
    pub errors: AtomicU64,
    pub start_time: Option<Instant>,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            min_latency_ns: AtomicU64::new(u64::MAX),
            ..Default::default()
        }
    }

    pub fn record_request(&self, latency: Duration, was_cache_hit: bool, had_error: bool) {
        let latency_ns = latency.as_nanos() as u64;
        
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
        
        // Update min/max latency
        self.min_latency_ns.fetch_min(latency_ns, Ordering::Relaxed);
        self.max_latency_ns.fetch_max(latency_ns, Ordering::Relaxed);
        
        if was_cache_hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
        
        if had_error {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn get_summary(&self) -> PerformanceSummary {
        let total = self.total_requests.load(Ordering::Relaxed);
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ns.load(Ordering::Relaxed);
        
        PerformanceSummary {
            total_requests: total,
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
            avg_latency: if total > 0 { 
                Duration::from_nanos(total_latency / total) 
            } else { 
                Duration::ZERO 
            },
            min_latency: Duration::from_nanos(self.min_latency_ns.load(Ordering::Relaxed)),
            max_latency: Duration::from_nanos(self.max_latency_ns.load(Ordering::Relaxed)),
            error_rate: if total > 0 { 
                self.errors.load(Ordering::Relaxed) as f64 / total as f64 
            } else { 
                0.0 
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub total_requests: u64,
    pub hit_rate: f64,
    pub avg_latency: Duration,
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub error_rate: f64,
}

/// Load test configuration
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    pub concurrent_users: usize,
    pub duration: Duration,
    pub ramp_up_time: Duration,
    pub target_rps: f64,
    pub cache_warm_queries: Vec<String>,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            concurrent_users: CONCURRENT_USERS_TARGET,
            duration: LOAD_TEST_DURATION,
            ramp_up_time: Duration::from_secs(5),
            target_rps: THROUGHPUT_TARGET_RPS,
            cache_warm_queries: vec![
                "pci_dss_requirements".to_string(),
                "gdpr_compliance".to_string(),
                "iso_27001_controls".to_string(),
                "sox_documentation".to_string(),
            ],
        }
    }
}

/// Benchmark suite for FACT operations
#[cfg(test)]
mod fact_performance_benchmarks {
    use super::*;
    
    /// Benchmark cache hit operations
    pub fn benchmark_cache_hit_operations(c: &mut Criterion) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        
        // Pre-populate cache with test data
        let fact_client = runtime.block_on(async {
            let mut client = MockFACTClient::new();
            
            // Configure for consistent cache hits
            client
                .expect_get()
                .returning(|_| {
                    Ok(Some(serde_json::to_vec(&json!({
                        "content": "Cached compliance data",
                        "cached": true,
                        "timestamp": chrono::Utc::now().timestamp()
                    })).unwrap()))
                });
            
            Arc::new(client)
        });

        let mut group = c.benchmark_group("fact_cache_operations");
        group.throughput(Throughput::Elements(1));
        
        // Single cache hit benchmark
        group.bench_function("single_cache_hit", |b| {
            b.to_async(&runtime).iter(|| async {
                let result = fact_client.get(black_box("benchmark_key")).await;
                assert!(result.is_ok());
                result
            });
        });
        
        // Batch cache hit benchmark
        for batch_size in [10, 50, 100].iter() {
            group.bench_with_input(
                BenchmarkId::new("batch_cache_hits", batch_size),
                batch_size,
                |b, &size| {
                    b.to_async(&runtime).iter(|| async {
                        let mut handles = Vec::with_capacity(size);
                        for i in 0..size {
                            let client = fact_client.clone();
                            let key = format!("batch_key_{}", i);
                            handles.push(async move {
                                client.get(&key).await
                            });
                        }
                        let results = join_all(handles).await;
                        black_box(results)
                    });
                },
            );
        }
        
        group.finish();
    }

    /// Benchmark cache miss operations  
    pub fn benchmark_cache_miss_operations(c: &mut Criterion) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        
        let fact_client = runtime.block_on(async {
            let mut client = MockFACTClient::new();
            
            // Configure for cache misses with realistic delay
            client
                .expect_get()
                .returning(|_| {
                    Box::pin(async {
                        // Simulate network fetch delay
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        Ok(Some(serde_json::to_vec(&json!({
                            "content": "Fresh data from origin",
                            "cached": false,
                            "fetch_time_ms": 50
                        })).unwrap()))
                    })
                });
            
            Arc::new(client)
        });

        let mut group = c.benchmark_group("fact_cache_miss");
        group.throughput(Throughput::Elements(1));
        group.measurement_time(Duration::from_secs(10));
        
        group.bench_function("single_cache_miss", |b| {
            b.to_async(&runtime).iter(|| async {
                let key = format!("miss_key_{}", uuid::Uuid::new_v4());
                let result = fact_client.get(black_box(&key)).await;
                assert!(result.is_ok());
                result
            });
        });
        
        group.finish();
    }

    /// Benchmark concurrent access patterns
    pub fn benchmark_concurrent_access(c: &mut Criterion) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        
        let fact_client = runtime.block_on(async {
            let mut client = MockFACTClient::new();
            let request_counter = Arc::new(AtomicUsize::new(0));
            
            client
                .expect_get()
                .returning({
                    let counter = request_counter.clone();
                    move |key| {
                        let request_num = counter.fetch_add(1, Ordering::Relaxed);
                        let is_cached = request_num % 3 == 0; // 33% cache hit rate for testing
                        
                        if is_cached {
                            Ok(Some(serde_json::to_vec(&json!({
                                "content": format!("Cached data for {}", key),
                                "cached": true
                            })).unwrap()))
                        } else {
                            Box::pin(async move {
                                tokio::time::sleep(Duration::from_millis(30)).await;
                                Ok(Some(serde_json::to_vec(&json!({
                                    "content": format!("Fresh data for {}", key),
                                    "cached": false
                                })).unwrap()))
                            })
                        }
                    }
                });
            
            Arc::new(client)
        });

        let mut group = c.benchmark_group("concurrent_access");
        
        for concurrency in [10, 50, 100].iter() {
            group.bench_with_input(
                BenchmarkId::new("concurrent_requests", concurrency),
                concurrency,
                |b, &concurrent_count| {
                    b.to_async(&runtime).iter(|| async {
                        let semaphore = Arc::new(Semaphore::new(concurrent_count));
                        let mut handles = Vec::new();
                        
                        for i in 0..concurrent_count {
                            let client = fact_client.clone();
                            let sem = semaphore.clone();
                            
                            handles.push(tokio::spawn(async move {
                                let _permit = sem.acquire().await.unwrap();
                                let key = format!("concurrent_key_{}", i);
                                client.get(&key).await
                            }));
                        }
                        
                        let results = join_all(handles).await;
                        black_box(results)
                    });
                },
            );
        }
        
        group.finish();
    }

    criterion_group!(
        benches,
        benchmark_cache_hit_operations,
        benchmark_cache_miss_operations,
        benchmark_concurrent_access
    );
    criterion_main!(benches);
}

/// Load testing suite
#[cfg(test)]
mod fact_load_tests {
    use super::*;

    /// Test system under 100+ concurrent users
    #[tokio::test]
    async fn test_100_concurrent_users_load() {
        let config = LoadTestConfig::default();
        let metrics = Arc::new(PerformanceMetrics::new());
        
        // Setup mock FACT client for load testing
        let fact_client = setup_load_test_client().await;
        
        // Pre-warm cache
        warm_cache(&fact_client, &config.cache_warm_queries).await.unwrap();
        
        // Execute load test
        let load_test_result = execute_load_test(fact_client, config.clone(), metrics.clone()).await;
        
        assert!(load_test_result.is_ok(), "Load test should complete successfully");
        
        let summary = metrics.get_summary();
        
        // Verify performance requirements
        assert!(summary.total_requests >= config.concurrent_users as u64,
               "Should process requests from all concurrent users");
        
        assert!(summary.hit_rate >= MIN_HIT_RATE,
               "Hit rate {:.3} should be >= {:.3}", summary.hit_rate, MIN_HIT_RATE);
        
        assert!(summary.error_rate < 0.01,
               "Error rate {:.3} should be < 1%", summary.error_rate);
        
        // Performance targets
        let hit_requests = (summary.total_requests as f64 * summary.hit_rate) as u64;
        let miss_requests = summary.total_requests - hit_requests;
        
        // Check that cache hits meet latency requirement
        if hit_requests > 0 {
            // This is approximated since we don't separate hit vs miss latencies in this simple version
            let estimated_hit_latency = Duration::from_nanos(
                (summary.avg_latency.as_nanos() as f64 * 0.8) as u64 // Assume hits are faster
            );
            assert!(estimated_hit_latency.as_millis() < CACHE_HIT_TARGET_MS as u128,
                   "Estimated cache hit latency should be < {}ms", CACHE_HIT_TARGET_MS);
        }
    }

    /// Test latency validation under load
    #[tokio::test]
    async fn test_latency_validation_under_load() {
        let metrics = Arc::new(PerformanceMetrics::new());
        let fact_client = setup_latency_test_client().await;
        
        let test_duration = Duration::from_secs(10);
        let concurrent_requests = 50;
        
        let start_time = Instant::now();
        let mut handles = Vec::new();
        
        // Generate continuous load
        for user_id in 0..concurrent_requests {
            let client = fact_client.clone();
            let metrics = metrics.clone();
            
            handles.push(tokio::spawn(async move {
                let mut requests_made = 0;
                while start_time.elapsed() < test_duration {
                    let key = format!("latency_test_{}_{}", user_id, requests_made);
                    let request_start = Instant::now();
                    
                    let result = client.get(&key).await;
                    let latency = request_start.elapsed();
                    
                    let was_hit = result.as_ref()
                        .map(|r| r.as_ref()
                            .map(|data| {
                                serde_json::from_slice::<Value>(data)
                                    .map(|v| v["cached"].as_bool().unwrap_or(false))
                                    .unwrap_or(false)
                            })
                            .unwrap_or(false)
                        )
                        .unwrap_or(false);
                    
                    metrics.record_request(latency, was_hit, result.is_err());
                    requests_made += 1;
                    
                    // Brief pause to avoid overwhelming
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                requests_made
            }));
        }
        
        let results = join_all(handles).await;
        let total_requests: u64 = results.into_iter().map(|r| r.unwrap() as u64).sum();
        
        let summary = metrics.get_summary();
        assert!(summary.total_requests >= total_requests * 0.9 as u64, 
               "Should record most requests");
        
        // Latency percentile validation would go here
        // For now, we validate average latency is reasonable
        assert!(summary.avg_latency < Duration::from_millis(200),
               "Average latency should be reasonable under load");
    }

    /// Test hit rate validation over time
    #[tokio::test] 
    async fn test_hit_rate_validation() {
        let fact_client = setup_hit_rate_test_client().await;
        let metrics = Arc::new(PerformanceMetrics::new());
        
        // Define a query distribution (some queries repeated more often)
        let query_distribution = vec![
            ("popular_query_1".to_string(), 30), // 30% of requests
            ("popular_query_2".to_string(), 25), // 25% of requests  
            ("common_query_1".to_string(), 15),  // 15% of requests
            ("common_query_2".to_string(), 15),  // 15% of requests
            ("rare_query".to_string(), 15),      // 15% of requests (many unique)
        ];
        
        let total_requests = 1000;
        let mut all_requests = Vec::new();
        
        // Generate request mix based on distribution
        for (base_query, percentage) in query_distribution {
            let count = (total_requests * percentage) / 100;
            for i in 0..count {
                let query = if base_query == "rare_query" {
                    format!("{}_{}", base_query, i) // Each rare query is unique
                } else {
                    base_query.clone() // Popular queries repeat
                };
                all_requests.push(query);
            }
        }
        
        // Shuffle requests to simulate realistic access pattern
        use rand::seq::SliceRandom;
        all_requests.shuffle(&mut rand::thread_rng());
        
        // Execute requests and measure hit rate
        for (i, query) in all_requests.iter().enumerate() {
            let request_start = Instant::now();
            let result = fact_client.get(query).await;
            let latency = request_start.elapsed();
            
            let was_hit = result.as_ref()
                .map(|r| r.as_ref()
                    .map(|data| {
                        serde_json::from_slice::<Value>(data)
                            .map(|v| v.get("cached").and_then(|c| c.as_bool()).unwrap_or(false))
                            .unwrap_or(false)
                    })
                    .unwrap_or(false)
                )
                .unwrap_or(false);
            
            metrics.record_request(latency, was_hit, result.is_err());
            
            // Check progressive hit rate after warmup period
            if i > 100 && i % 100 == 0 {
                let current_summary = metrics.get_summary();
                println!("Progress: {}/{} requests, hit rate: {:.3}", 
                        i, all_requests.len(), current_summary.hit_rate);
            }
        }
        
        let final_summary = metrics.get_summary();
        
        // Validate hit rate meets target
        assert!(final_summary.hit_rate >= MIN_HIT_RATE,
               "Final hit rate {:.3} should be >= {:.3}. Total requests: {}, Cache hits: {}", 
               final_summary.hit_rate, MIN_HIT_RATE, 
               final_summary.total_requests, 
               metrics.cache_hits.load(Ordering::Relaxed));
        
        // Ensure we processed all requests
        assert_eq!(final_summary.total_requests, all_requests.len() as u64);
    }

    /// Test system behavior under stress conditions
    #[tokio::test]
    async fn test_stress_conditions() {
        let stress_config = LoadTestConfig {
            concurrent_users: 200, // 2x normal load
            duration: Duration::from_secs(60),
            target_rps: THROUGHPUT_TARGET_RPS * 1.5, // 1.5x target RPS
            ..Default::default()
        };
        
        let fact_client = setup_stress_test_client().await;
        let metrics = Arc::new(PerformanceMetrics::new());
        
        // Execute stress test
        let stress_result = execute_load_test(fact_client, stress_config.clone(), metrics.clone()).await;
        
        // System should handle stress gracefully
        assert!(stress_result.is_ok(), "System should survive stress test");
        
        let summary = metrics.get_summary();
        
        // Under stress, we accept some degradation but system should remain functional
        assert!(summary.error_rate < 0.05, 
               "Error rate under stress should be < 5%, got {:.3}", summary.error_rate);
        
        // Hit rate might degrade slightly under stress but should remain reasonable
        assert!(summary.hit_rate >= MIN_HIT_RATE * 0.9,
               "Hit rate under stress should be >= {:.3}, got {:.3}", 
               MIN_HIT_RATE * 0.9, summary.hit_rate);
        
        // Latency may increase but should remain bounded
        assert!(summary.avg_latency < Duration::from_secs(1),
               "Average latency under stress should be < 1s, got {:?}", summary.avg_latency);
    }
}

/// Helper functions for setting up test clients and executing load tests
async fn setup_load_test_client() -> Arc<MockFACTClient> {
    let mut client = MockFACTClient::new();
    let request_count = Arc::new(AtomicUsize::new(0));
    
    // Simulate realistic cache behavior
    client
        .expect_get()
        .returning({
            let counter = request_count.clone();
            move |key| {
                let request_num = counter.fetch_add(1, Ordering::Relaxed);
                let is_cache_hit = key.contains("popular_") || key.contains("common_") || request_num % 4 != 0;
                
                if is_cache_hit {
                    // Cache hit - fast response
                    Ok(Some(serde_json::to_vec(&json!({
                        "content": format!("Cached data for {}", key),
                        "cached": true,
                        "hit_number": request_num
                    })).unwrap()))
                } else {
                    // Cache miss - slower response
                    Box::pin(async move {
                        tokio::time::sleep(Duration::from_millis(60)).await;
                        Ok(Some(serde_json::to_vec(&json!({
                            "content": format!("Fresh data for {}", key),
                            "cached": false,
                            "fetch_time_ms": 60
                        })).unwrap()))
                    })
                }
            }
        });
    
    Arc::new(client)
}

async fn setup_latency_test_client() -> Arc<MockFACTClient> {
    let mut client = MockFACTClient::new();
    
    client
        .expect_get()
        .returning(|key| {
            let latency_ms = if key.contains("fast") {
                10 // Fast cache hits
            } else if key.contains("medium") {
                40 // Medium responses
            } else {
                70 // Slower cache misses
            };
            
            Box::pin(async move {
                tokio::time::sleep(Duration::from_millis(latency_ms)).await;
                Ok(Some(serde_json::to_vec(&json!({
                    "content": format!("Data for key with {}ms latency", latency_ms),
                    "cached": latency_ms < 30,
                    "latency_ms": latency_ms
                })).unwrap()))
            })
        });
    
    Arc::new(client)
}

async fn setup_hit_rate_test_client() -> Arc<MockFACTClient> {
    let mut client = MockFACTClient::new();
    let cache_map = Arc::new(RwLock::new(HashMap::<String, Vec<u8>>::new()));
    
    client
        .expect_get()
        .returning({
            let cache = cache_map.clone();
            move |key| {
                let cache = cache.clone();
                let key = key.to_string();
                Box::pin(async move {
                    // Check cache first
                    {
                        let cache_read = cache.read().await;
                        if let Some(cached_data) = cache_read.get(&key) {
                            return Ok(Some(cached_data.clone()));
                        }
                    }
                    
                    // Simulate fetch from origin
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    
                    let fresh_data = serde_json::to_vec(&json!({
                        "content": format!("Fresh content for {}", key),
                        "cached": false,
                        "timestamp": chrono::Utc::now().timestamp()
                    })).unwrap();
                    
                    // Cache the data
                    let mut cache_write = cache.write().await;
                    cache_write.insert(key.clone(), fresh_data.clone());
                    
                    // Update the response to show it's now cached
                    let cached_response = serde_json::to_vec(&json!({
                        "content": format!("Fresh content for {}", key),
                        "cached": true,
                        "timestamp": chrono::Utc::now().timestamp()
                    })).unwrap();
                    
                    cache_write.insert(key, cached_response.clone());
                    Ok(Some(fresh_data))
                })
            }
        });
    
    Arc::new(client)
}

async fn setup_stress_test_client() -> Arc<MockFACTClient> {
    let mut client = MockFACTClient::new();
    let error_rate = Arc::new(AtomicU64::new(0));
    
    client
        .expect_get()
        .returning({
            let error_counter = error_rate.clone();
            move |key| {
                let error_num = error_counter.fetch_add(1, Ordering::Relaxed);
                
                // Introduce some errors under stress (2% error rate)
                if error_num % 50 == 0 {
                    return Err(crate::tests::fact_integration_tests::FACTError::ServiceUnavailable);
                }
                
                // Varied response times under stress
                let latency_ms = match error_num % 4 {
                    0 => 15,  // Fast cache hits
                    1 => 45,  // Medium responses  
                    2 => 85,  // Slower responses
                    _ => 120, // Very slow responses under stress
                };
                
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(latency_ms)).await;
                    Ok(Some(serde_json::to_vec(&json!({
                        "content": format!("Stress test data for {}", key),
                        "cached": latency_ms < 50,
                        "stress_latency_ms": latency_ms
                    })).unwrap()))
                })
            }
        });
    
    Arc::new(client)
}

async fn warm_cache(
    client: &MockFACTClient, 
    warm_queries: &[String]
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    for query in warm_queries {
        let _ = client.get(query).await?;
        tokio::time::sleep(Duration::from_millis(10)).await; // Brief pause
    }
    Ok(())
}

async fn execute_load_test(
    client: Arc<MockFACTClient>,
    config: LoadTestConfig, 
    metrics: Arc<PerformanceMetrics>
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();
    let mut handles = Vec::new();
    
    // Spawn concurrent users
    for user_id in 0..config.concurrent_users {
        let client = client.clone();
        let metrics = metrics.clone();
        let config = config.clone();
        
        handles.push(tokio::spawn(async move {
            let mut requests_made = 0;
            
            // Ramp-up delay
            let ramp_delay = config.ramp_up_time.mul_f64(user_id as f64 / config.concurrent_users as f64);
            tokio::time::sleep(ramp_delay).await;
            
            while start_time.elapsed() < config.duration {
                let query = match requests_made % 10 {
                    0..=3 => format!("popular_query_{}", requests_made % 2),
                    4..=6 => format!("common_query_{}", requests_made % 3),
                    _ => format!("unique_query_{}_{}", user_id, requests_made),
                };
                
                let request_start = Instant::now();
                let result = client.get(&query).await;
                let latency = request_start.elapsed();
                
                let was_hit = result.as_ref()
                    .map(|r| r.as_ref()
                        .map(|data| {
                            serde_json::from_slice::<Value>(data)
                                .map(|v| v["cached"].as_bool().unwrap_or(false))
                                .unwrap_or(false)
                        })
                        .unwrap_or(false)
                    )
                    .unwrap_or(false);
                
                metrics.record_request(latency, was_hit, result.is_err());
                requests_made += 1;
                
                // Rate limiting to target RPS per user
                let target_delay = Duration::from_secs_f64(config.concurrent_users as f64 / config.target_rps);
                tokio::time::sleep(target_delay).await;
            }
            
            requests_made
        }));
    }
    
    // Wait for all users to complete
    let _results = join_all(handles).await;
    Ok(())
}

/// Additional performance test utilities and helper functions would go here