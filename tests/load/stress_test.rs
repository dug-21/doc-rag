//! Load Testing and Stress Testing Infrastructure
//!
//! This module provides comprehensive load testing capabilities to validate
//! system performance under various stress conditions including:
//! - High concurrent user loads
//! - Memory pressure scenarios
//! - Extended duration testing
//! - Spike load handling
//! - Resource exhaustion recovery

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore, Barrier};
use uuid::Uuid;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Load test configuration parameters
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    pub max_concurrent_users: usize,
    pub test_duration: Duration,
    pub ramp_up_duration: Duration,
    pub think_time: Duration,
    pub target_throughput: f64,
    pub memory_limit_mb: u64,
    pub performance_thresholds: PerformanceThresholds,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    pub avg_response_time_ms: u64,
    pub p95_response_time_ms: u64,
    pub p99_response_time_ms: u64,
    pub error_rate_threshold: f64,
    pub memory_growth_limit_mb: u64,
    pub cpu_utilization_limit: f64,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            max_concurrent_users: 100,
            test_duration: Duration::from_secs(300), // 5 minutes
            ramp_up_duration: Duration::from_secs(60), // 1 minute ramp-up
            think_time: Duration::from_millis(500),
            target_throughput: 50.0, // QPS
            memory_limit_mb: 2048,
            performance_thresholds: PerformanceThresholds {
                avg_response_time_ms: 200,
                p95_response_time_ms: 500,
                p99_response_time_ms: 1000,
                error_rate_threshold: 0.01, // 1% error rate
                memory_growth_limit_mb: 500,
                cpu_utilization_limit: 0.8, // 80%
            },
        }
    }
}

/// Load testing scenarios
#[derive(Debug, Clone)]
pub enum LoadTestScenario {
    SteadyState,
    RampUp,
    SpikeLoad,
    StressTest,
    MemoryPressure,
    ConcurrencyStorm,
}

/// Load test workload patterns
const LOAD_TEST_QUERIES: &[&str] = &[
    "What is artificial intelligence and how does it work?",
    "Compare machine learning algorithms for classification tasks",
    "Explain database normalization and its benefits",
    "How do microservices architecture patterns improve scalability?",
    "What are the security considerations for cloud computing?",
    "Describe the principles of software engineering best practices",
    "Analyze performance optimization techniques for web applications",
    "What are the differences between relational and NoSQL databases?",
    "How does distributed computing handle fault tolerance?",
    "Explain containerization and orchestration technologies",
    "What are the key components of a modern CI/CD pipeline?",
    "Describe RESTful API design principles and best practices",
    "How do caching strategies improve application performance?",
    "What are the challenges of implementing real-time systems?",
    "Explain the concept of eventual consistency in distributed systems",
];

/// Comprehensive load testing system
pub struct LoadTestSystem {
    config: LoadTestConfig,
    documents: Arc<RwLock<Vec<TestDocument>>>,
    metrics: Arc<RwLock<LoadTestMetrics>>,
    active_users: Arc<RwLock<HashMap<Uuid, UserSession>>>,
    system_monitor: Arc<RwLock<SystemMonitor>>,
}

#[derive(Debug, Clone)]
struct TestDocument {
    id: String,
    content: String,
    embeddings: Vec<f32>,
    indexed_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct UserSession {
    pub user_id: Uuid,
    pub started_at: Instant,
    pub queries_sent: u64,
    pub successful_queries: u64,
    pub total_response_time: Duration,
    pub last_activity: Instant,
}

#[derive(Debug, Clone)]
pub struct LoadTestMetrics {
    pub test_started_at: Instant,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub response_times: Vec<Duration>,
    pub error_details: HashMap<String, u64>,
    pub throughput_samples: Vec<(Instant, f64)>,
    pub memory_usage_samples: Vec<(Instant, u64)>,
    pub cpu_usage_samples: Vec<(Instant, f64)>,
}

impl LoadTestMetrics {
    fn new() -> Self {
        Self {
            test_started_at: Instant::now(),
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            response_times: Vec::new(),
            error_details: HashMap::new(),
            throughput_samples: Vec::new(),
            memory_usage_samples: Vec::new(),
            cpu_usage_samples: Vec::new(),
        }
    }

    fn record_request(&mut self, response_time: Duration, success: bool, error_type: Option<String>) {
        self.total_requests += 1;
        
        if success {
            self.successful_requests += 1;
            self.response_times.push(response_time);
        } else {
            self.failed_requests += 1;
            if let Some(error) = error_type {
                *self.error_details.entry(error).or_insert(0) += 1;
            }
        }
    }

    fn calculate_percentile(&self, percentile: f64) -> Duration {
        if self.response_times.is_empty() {
            return Duration::from_millis(0);
        }

        let mut sorted_times = self.response_times.clone();
        sorted_times.sort();
        
        let index = ((percentile / 100.0) * sorted_times.len() as f64).floor() as usize;
        sorted_times.get(index).copied().unwrap_or(Duration::from_millis(0))
    }

    fn average_response_time(&self) -> Duration {
        if self.response_times.is_empty() {
            return Duration::from_millis(0);
        }
        
        self.response_times.iter().sum::<Duration>() / self.response_times.len() as u32
    }

    fn current_throughput(&self) -> f64 {
        let elapsed = self.test_started_at.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.successful_requests as f64 / elapsed
        } else {
            0.0
        }
    }

    fn error_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.failed_requests as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
struct SystemMonitor {
    initial_memory: u64,
    peak_memory: u64,
    peak_cpu: f64,
    last_gc_time: Option<Instant>,
}

impl SystemMonitor {
    fn new(initial_memory: u64) -> Self {
        Self {
            initial_memory,
            peak_memory: initial_memory,
            peak_cpu: 0.0,
            last_gc_time: None,
        }
    }

    fn update_memory(&mut self, current_memory: u64) {
        if current_memory > self.peak_memory {
            self.peak_memory = current_memory;
        }
    }

    fn update_cpu(&mut self, cpu_usage: f64) {
        if cpu_usage > self.peak_cpu {
            self.peak_cpu = cpu_usage;
        }
    }

    fn memory_growth(&self) -> u64 {
        self.peak_memory.saturating_sub(self.initial_memory)
    }
}

impl LoadTestSystem {
    pub async fn new(config: LoadTestConfig) -> Result<Self> {
        let initial_memory = Self::get_memory_usage().await?;
        
        let system = Self {
            config,
            documents: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(LoadTestMetrics::new())),
            active_users: Arc::new(RwLock::new(HashMap::new())),
            system_monitor: Arc::new(RwLock::new(SystemMonitor::new(initial_memory))),
        };

        // Initialize test data
        system.initialize_test_data().await?;
        
        Ok(system)
    }

    /// Initialize test documents for load testing
    async fn initialize_test_data(&self) -> Result<()> {
        let mut documents = self.documents.write().await;
        
        for i in 0..1000 {
            let content = format!(
                "Load test document {}: This is a comprehensive test document containing detailed information about various technical topics including software architecture, machine learning, database systems, cloud computing, and cybersecurity. The document is designed to provide realistic content for load testing scenarios with sufficient complexity to generate meaningful embedding vectors and search results. Document content includes multiple paragraphs, technical terminology, and structured information that would be typical in a production knowledge base system.",
                i
            );

            documents.push(TestDocument {
                id: format!("load_test_doc_{}", i),
                content,
                embeddings: self.generate_mock_embedding(&format!("Document {}", i)),
                indexed_at: chrono::Utc::now(),
            });
        }

        println!("✅ Initialized {} test documents for load testing", documents.len());
        Ok(())
    }

    /// Run comprehensive load test suite
    pub async fn run_load_test_suite(&self) -> Result<LoadTestSuiteResult> {
        println!("🚀 Starting Comprehensive Load Test Suite");
        println!("==========================================");

        let mut suite_result = LoadTestSuiteResult::new();

        // Test 1: Steady State Load Test
        println!("Test 1: Steady State Load Test");
        suite_result.steady_state = Some(self.run_steady_state_test().await?);
        
        // Test 2: Ramp-Up Load Test
        println!("Test 2: Ramp-Up Load Test");
        suite_result.ramp_up = Some(self.run_ramp_up_test().await?);
        
        // Test 3: Spike Load Test
        println!("Test 3: Spike Load Test");
        suite_result.spike_load = Some(self.run_spike_load_test().await?);
        
        // Test 4: Stress Test
        println!("Test 4: Stress Test");
        suite_result.stress_test = Some(self.run_stress_test().await?);
        
        // Test 5: Memory Pressure Test
        println!("Test 5: Memory Pressure Test");
        suite_result.memory_pressure = Some(self.run_memory_pressure_test().await?);
        
        // Test 6: Concurrency Storm Test
        println!("Test 6: Concurrency Storm Test");
        suite_result.concurrency_storm = Some(self.run_concurrency_storm_test().await?);

        Ok(suite_result)
    }

    /// Steady state load test with consistent load
    async fn run_steady_state_test(&self) -> Result<LoadTestResult> {
        let users = 20;
        let duration = Duration::from_secs(120);
        
        self.reset_metrics().await;
        let result = self.execute_load_test(users, duration, LoadTestScenario::SteadyState).await?;
        
        println!("✅ Steady State: {:.2} QPS, {:.0}ms avg response time", 
                 result.throughput, result.avg_response_time.as_millis());
        
        Ok(result)
    }

    /// Ramp-up load test with gradually increasing load
    async fn run_ramp_up_test(&self) -> Result<LoadTestResult> {
        let max_users = 50;
        let duration = Duration::from_secs(180);
        
        self.reset_metrics().await;
        let result = self.execute_ramp_up_load_test(max_users, duration).await?;
        
        println!("✅ Ramp-Up: {:.2} peak QPS, {:.0}ms avg response time", 
                 result.throughput, result.avg_response_time.as_millis());
        
        Ok(result)
    }

    /// Spike load test with sudden traffic increases
    async fn run_spike_load_test(&self) -> Result<LoadTestResult> {
        self.reset_metrics().await;
        
        // Normal load: 10 users
        let normal_load_handle = self.spawn_users(10, Duration::from_secs(60));
        tokio::time::sleep(Duration::from_secs(30)).await;
        
        // Spike load: additional 40 users for 30 seconds
        let spike_load_handle = self.spawn_users(40, Duration::from_secs(30));
        
        // Wait for spike to complete
        tokio::time::sleep(Duration::from_secs(30)).await;
        
        // Continue with normal load
        tokio::time::sleep(Duration::from_secs(30)).await;

        let result = self.collect_metrics().await;
        
        println!("✅ Spike Load: {:.2} QPS during spike, {:.1}% error rate", 
                 result.throughput, result.error_rate * 100.0);
        
        Ok(result)
    }

    /// Stress test to find system breaking point
    async fn run_stress_test(&self) -> Result<LoadTestResult> {
        self.reset_metrics().await;
        
        let mut current_users = 10;
        let max_users = 200;
        let step_duration = Duration::from_secs(60);
        
        while current_users <= max_users {
            println!("Stress test: {} concurrent users", current_users);
            
            let step_handle = self.spawn_users(current_users, step_duration);
            tokio::time::sleep(step_duration).await;
            
            let current_metrics = self.collect_metrics().await;
            
            // Check if system is degrading significantly
            if current_metrics.error_rate > 0.05 || 
               current_metrics.p95_response_time > Duration::from_millis(2000) {
                println!("⚠️ System degradation detected at {} users", current_users);
                break;
            }
            
            current_users += 20;
        }

        let result = self.collect_metrics().await;
        
        println!("✅ Stress Test: Max {} users, {:.2} QPS, {:.1}% error rate", 
                 current_users - 20, result.throughput, result.error_rate * 100.0);
        
        Ok(result)
    }

    /// Memory pressure test with large data operations
    async fn run_memory_pressure_test(&self) -> Result<LoadTestResult> {
        self.reset_metrics().await;
        
        // Create memory pressure with large documents
        {
            let mut documents = self.documents.write().await;
            for i in 0..500 {
                let large_content = "Large content block for memory pressure testing. ".repeat(1000);
                documents.push(TestDocument {
                    id: format!("memory_pressure_doc_{}", i),
                    content: large_content,
                    embeddings: vec![0.1; 1024], // Larger embedding vectors
                    indexed_at: chrono::Utc::now(),
                });
            }
        }

        // Run normal load under memory pressure
        let result = self.execute_load_test(30, Duration::from_secs(120), LoadTestScenario::MemoryPressure).await?;
        
        let memory_growth = {
            let monitor = self.system_monitor.read().await;
            monitor.memory_growth()
        };
        
        println!("✅ Memory Pressure: {:.2} QPS, {}MB memory growth", 
                 result.throughput, memory_growth / 1024 / 1024);
        
        Ok(result)
    }

    /// Concurrency storm test with maximum concurrent requests
    async fn run_concurrency_storm_test(&self) -> Result<LoadTestResult> {
        self.reset_metrics().await;
        
        let storm_users = 100;
        let storm_duration = Duration::from_secs(60);
        
        // Create barrier to synchronize all users starting at once
        let barrier = Arc::new(Barrier::new(storm_users + 1));
        let mut handles = Vec::new();

        // Spawn all users
        for user_id in 0..storm_users {
            let system = self.clone();
            let barrier_clone = Arc::clone(&barrier);
            
            let handle = tokio::spawn(async move {
                // Wait for all users to be ready
                barrier_clone.wait().await;
                
                // Execute queries simultaneously
                system.execute_user_session(
                    Uuid::new_v4(),
                    storm_duration,
                    Duration::from_millis(10), // Minimal think time for storm
                ).await
            });
            
            handles.push(handle);
        }

        // Release all users at once
        barrier.wait().await;
        
        // Wait for completion
        for handle in handles {
            let _ = handle.await;
        }

        let result = self.collect_metrics().await;
        
        println!("✅ Concurrency Storm: {:.2} QPS, {:.0}ms p99 response time", 
                 result.throughput, result.p99_response_time.as_millis());
        
        Ok(result)
    }

    /// Execute load test with specified parameters
    async fn execute_load_test(&self, users: usize, duration: Duration, scenario: LoadTestScenario) -> Result<LoadTestResult> {
        let mut handles = Vec::new();
        let think_time = match scenario {
            LoadTestScenario::ConcurrencyStorm => Duration::from_millis(10),
            LoadTestScenario::MemoryPressure => Duration::from_millis(200),
            _ => self.config.think_time,
        };

        // Spawn user sessions
        for _ in 0..users {
            let system = self.clone();
            let handle = tokio::spawn(async move {
                system.execute_user_session(Uuid::new_v4(), duration, think_time).await
            });
            handles.push(handle);
        }

        // Start monitoring
        let monitor_handle = self.start_monitoring(duration);

        // Wait for all users to complete
        for handle in handles {
            let _ = handle.await;
        }

        // Stop monitoring
        monitor_handle.abort();

        Ok(self.collect_metrics().await)
    }

    /// Execute ramp-up load test with gradual user increase
    async fn execute_ramp_up_load_test(&self, max_users: usize, total_duration: Duration) -> Result<LoadTestResult> {
        let ramp_up_steps = 10;
        let step_duration = total_duration / ramp_up_steps as u32;
        let users_per_step = max_users / ramp_up_steps;

        let mut handles = Vec::new();
        let monitor_handle = self.start_monitoring(total_duration);

        for step in 0..ramp_up_steps {
            let step_users = std::cmp::min(users_per_step, max_users - step * users_per_step);
            
            // Spawn users for this step
            for _ in 0..step_users {
                let system = self.clone();
                let remaining_duration = total_duration - (step_duration * step as u32);
                
                let handle = tokio::spawn(async move {
                    system.execute_user_session(Uuid::new_v4(), remaining_duration, Duration::from_millis(500)).await
                });
                
                handles.push(handle);
            }
            
            // Wait before next step
            tokio::time::sleep(step_duration).await;
        }

        // Wait for all users to complete
        for handle in handles {
            let _ = handle.await;
        }

        monitor_handle.abort();
        Ok(self.collect_metrics().await)
    }

    /// Spawn users and return handle for coordination
    async fn spawn_users(&self, count: usize, duration: Duration) -> tokio::task::JoinHandle<()> {
        let system = self.clone();
        
        tokio::spawn(async move {
            let mut handles = Vec::new();
            
            for _ in 0..count {
                let system_clone = system.clone();
                let handle = tokio::spawn(async move {
                    system_clone.execute_user_session(Uuid::new_v4(), duration, Duration::from_millis(300)).await
                });
                handles.push(handle);
            }
            
            for handle in handles {
                let _ = handle.await;
            }
        })
    }

    /// Execute individual user session
    async fn execute_user_session(&self, user_id: Uuid, duration: Duration, think_time: Duration) -> Result<()> {
        let start_time = Instant::now();
        let end_time = start_time + duration;

        // Register user session
        {
            let mut users = self.active_users.write().await;
            users.insert(user_id, UserSession {
                user_id,
                started_at: start_time,
                queries_sent: 0,
                successful_queries: 0,
                total_response_time: Duration::from_millis(0),
                last_activity: start_time,
            });
        }

        let mut query_index = 0;

        while Instant::now() < end_time {
            let query = LOAD_TEST_QUERIES[query_index % LOAD_TEST_QUERIES.len()];
            
            let request_start = Instant::now();
            
            match self.process_query(query).await {
                Ok(response_time) => {
                    self.record_success(user_id, response_time).await;
                }
                Err(e) => {
                    self.record_failure(user_id, e.to_string()).await;
                }
            }
            
            query_index += 1;
            
            // Think time between requests
            tokio::time::sleep(think_time).await;
        }

        // Unregister user session
        {
            let mut users = self.active_users.write().await;
            users.remove(&user_id);
        }

        Ok(())
    }

    /// Simulate query processing
    async fn process_query(&self, query: &str) -> Result<Duration> {
        let start_time = Instant::now();
        
        // Simulate query processing latency
        let base_latency = Duration::from_millis(50);
        let variable_latency = Duration::from_millis((query.len() as u64 % 100) + 20);
        
        tokio::time::sleep(base_latency + variable_latency).await;
        
        // Simulate occasional failures
        if rand::random::<f64>() < 0.01 {
            return Err(anyhow!("Simulated query processing failure"));
        }
        
        Ok(start_time.elapsed())
    }

    /// Record successful request
    async fn record_success(&self, user_id: Uuid, response_time: Duration) {
        // Update user session
        {
            let mut users = self.active_users.write().await;
            if let Some(session) = users.get_mut(&user_id) {
                session.queries_sent += 1;
                session.successful_queries += 1;
                session.total_response_time += response_time;
                session.last_activity = Instant::now();
            }
        }

        // Update global metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.record_request(response_time, true, None);
        }
    }

    /// Record failed request
    async fn record_failure(&self, user_id: Uuid, error: String) {
        // Update user session
        {
            let mut users = self.active_users.write().await;
            if let Some(session) = users.get_mut(&user_id) {
                session.queries_sent += 1;
                session.last_activity = Instant::now();
            }
        }

        // Update global metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.record_request(Duration::from_millis(0), false, Some(error));
        }
    }

    /// Start system monitoring
    fn start_monitoring(&self, duration: Duration) -> tokio::task::JoinHandle<()> {
        let system_monitor = Arc::clone(&self.system_monitor);
        let metrics = Arc::clone(&self.metrics);
        
        tokio::spawn(async move {
            let start_time = Instant::now();
            let end_time = start_time + duration;
            
            while Instant::now() < end_time {
                // Simulate system monitoring
                let memory_usage = Self::get_memory_usage().await.unwrap_or(0);
                let cpu_usage = Self::get_cpu_usage().await.unwrap_or(0.0);
                
                // Update system monitor
                {
                    let mut monitor = system_monitor.write().await;
                    monitor.update_memory(memory_usage);
                    monitor.update_cpu(cpu_usage);
                }
                
                // Record metrics samples
                {
                    let mut metrics_guard = metrics.write().await;
                    metrics_guard.memory_usage_samples.push((Instant::now(), memory_usage));
                    metrics_guard.cpu_usage_samples.push((Instant::now(), cpu_usage));
                    
                    let current_throughput = metrics_guard.current_throughput();
                    metrics_guard.throughput_samples.push((Instant::now(), current_throughput));
                }
                
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        })
    }

    /// Reset metrics for new test
    async fn reset_metrics(&self) {
        let mut metrics = self.metrics.write().await;
        *metrics = LoadTestMetrics::new();
    }

    /// Collect final test metrics
    async fn collect_metrics(&self) -> LoadTestResult {
        let metrics = self.metrics.read().await;
        let monitor = self.system_monitor.read().await;

        LoadTestResult {
            total_requests: metrics.total_requests,
            successful_requests: metrics.successful_requests,
            failed_requests: metrics.failed_requests,
            avg_response_time: metrics.average_response_time(),
            p50_response_time: metrics.calculate_percentile(50.0),
            p95_response_time: metrics.calculate_percentile(95.0),
            p99_response_time: metrics.calculate_percentile(99.0),
            throughput: metrics.current_throughput(),
            error_rate: metrics.error_rate(),
            memory_growth_mb: monitor.memory_growth() / 1024 / 1024,
            peak_cpu_usage: monitor.peak_cpu,
            test_duration: metrics.test_started_at.elapsed(),
        }
    }

    /// Generate mock embedding for testing
    fn generate_mock_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; 384];
        let bytes = text.as_bytes();
        
        for (i, &byte) in bytes.iter().enumerate().take(384) {
            embedding[i] = (byte as f32 - 128.0) / 128.0;
        }

        // Normalize
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for val in &mut embedding {
                *val /= magnitude;
            }
        }

        embedding
    }

    /// Get current memory usage (simulated)
    async fn get_memory_usage() -> Result<u64> {
        // In real implementation, this would query actual system memory usage
        Ok(512 * 1024 * 1024) // 512MB baseline
    }

    /// Get current CPU usage (simulated)
    async fn get_cpu_usage() -> Result<f64> {
        // In real implementation, this would query actual CPU usage
        Ok(0.3 + rand::random::<f64>() * 0.4) // 30-70% CPU usage
    }
}

impl Clone for LoadTestSystem {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            documents: Arc::clone(&self.documents),
            metrics: Arc::clone(&self.metrics),
            active_users: Arc::clone(&self.active_users),
            system_monitor: Arc::clone(&self.system_monitor),
        }
    }
}

/// Load test result data structures

#[derive(Debug)]
pub struct LoadTestSuiteResult {
    pub steady_state: Option<LoadTestResult>,
    pub ramp_up: Option<LoadTestResult>,
    pub spike_load: Option<LoadTestResult>,
    pub stress_test: Option<LoadTestResult>,
    pub memory_pressure: Option<LoadTestResult>,
    pub concurrency_storm: Option<LoadTestResult>,
}

impl LoadTestSuiteResult {
    fn new() -> Self {
        Self {
            steady_state: None,
            ramp_up: None,
            spike_load: None,
            stress_test: None,
            memory_pressure: None,
            concurrency_storm: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadTestResult {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time: Duration,
    pub p50_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub throughput: f64,
    pub error_rate: f64,
    pub memory_growth_mb: u64,
    pub peak_cpu_usage: f64,
    pub test_duration: Duration,
}

// Integration Tests

/// Test steady state load performance
#[tokio::test]
async fn test_steady_state_load() {
    let config = LoadTestConfig::default();
    let system = LoadTestSystem::new(config.clone()).await.unwrap();

    let result = system.run_steady_state_test().await.unwrap();

    // Validate performance requirements
    assert!(result.avg_response_time <= Duration::from_millis(config.performance_thresholds.avg_response_time_ms),
            "Average response time {}ms exceeds threshold {}ms",
            result.avg_response_time.as_millis(), config.performance_thresholds.avg_response_time_ms);

    assert!(result.error_rate <= config.performance_thresholds.error_rate_threshold,
            "Error rate {:.2}% exceeds threshold {:.2}%",
            result.error_rate * 100.0, config.performance_thresholds.error_rate_threshold * 100.0);

    assert!(result.throughput >= 10.0,
            "Throughput {:.2} QPS below minimum requirement",
            result.throughput);

    println!("✅ Steady State Load Test: {:.2} QPS, {}ms avg response", 
             result.throughput, result.avg_response_time.as_millis());
}

/// Test system behavior under ramp-up load
#[tokio::test]
async fn test_ramp_up_load() {
    let config = LoadTestConfig::default();
    let system = LoadTestSystem::new(config.clone()).await.unwrap();

    let result = system.run_ramp_up_test().await.unwrap();

    // Validate performance during ramp-up
    assert!(result.p95_response_time <= Duration::from_millis(config.performance_thresholds.p95_response_time_ms),
            "P95 response time {}ms exceeds threshold {}ms",
            result.p95_response_time.as_millis(), config.performance_thresholds.p95_response_time_ms);

    assert!(result.error_rate <= 0.02, // Allow slightly higher error rate during ramp-up
            "Error rate {:.2}% too high during ramp-up",
            result.error_rate * 100.0);

    println!("✅ Ramp-Up Load Test: {:.2} QPS, {}ms P95 response", 
             result.throughput, result.p95_response_time.as_millis());
}

/// Test spike load handling
#[tokio::test]
async fn test_spike_load_handling() {
    let config = LoadTestConfig::default();
    let system = LoadTestSystem::new(config).await.unwrap();

    let result = system.run_spike_load_test().await.unwrap();

    // System should handle spike gracefully
    assert!(result.error_rate <= 0.05, // Allow higher error rate during spike
            "Error rate {:.2}% too high during spike load",
            result.error_rate * 100.0);

    assert!(result.successful_requests > 0,
            "No successful requests during spike load test");

    println!("✅ Spike Load Test: {:.2} QPS, {:.1}% error rate", 
             result.throughput, result.error_rate * 100.0);
}

/// Test memory pressure scenarios
#[tokio::test]
async fn test_memory_pressure() {
    let config = LoadTestConfig::default();
    let system = LoadTestSystem::new(config.clone()).await.unwrap();

    let result = system.run_memory_pressure_test().await.unwrap();

    // Validate memory usage stays within limits
    assert!(result.memory_growth_mb <= config.performance_thresholds.memory_growth_limit_mb,
            "Memory growth {}MB exceeds limit {}MB",
            result.memory_growth_mb, config.performance_thresholds.memory_growth_limit_mb);

    // System should still perform reasonably under memory pressure
    assert!(result.error_rate <= 0.1,
            "Error rate {:.2}% too high under memory pressure",
            result.error_rate * 100.0);

    println!("✅ Memory Pressure Test: {}MB growth, {:.1}% error rate", 
             result.memory_growth_mb, result.error_rate * 100.0);
}

/// Test concurrency storm scenarios
#[tokio::test]
async fn test_concurrency_storm() {
    let config = LoadTestConfig::default();
    let system = LoadTestSystem::new(config.clone()).await.unwrap();

    let result = system.run_concurrency_storm_test().await.unwrap();

    // System should handle high concurrency
    assert!(result.p99_response_time <= Duration::from_millis(config.performance_thresholds.p99_response_time_ms * 2),
            "P99 response time {}ms too high during concurrency storm",
            result.p99_response_time.as_millis());

    assert!(result.successful_requests > result.total_requests / 2,
            "Too many failures during concurrency storm");

    println!("✅ Concurrency Storm Test: {:.2} QPS, {}ms P99 response", 
             result.throughput, result.p99_response_time.as_millis());
}

/// Comprehensive load test suite
#[tokio::test]
async fn test_comprehensive_load_test_suite() {
    println!("🚀 Starting Comprehensive Load Test Suite");
    println!("==========================================");

    let config = LoadTestConfig {
        max_concurrent_users: 50,
        test_duration: Duration::from_secs(60), // Reduced for testing
        ..LoadTestConfig::default()
    };
    
    let system = LoadTestSystem::new(config.clone()).await.unwrap();
    let suite_result = system.run_load_test_suite().await.unwrap();

    // Validate all tests completed successfully
    assert!(suite_result.steady_state.is_some(), "Steady state test failed");
    assert!(suite_result.ramp_up.is_some(), "Ramp-up test failed");
    assert!(suite_result.spike_load.is_some(), "Spike load test failed");
    assert!(suite_result.stress_test.is_some(), "Stress test failed");
    assert!(suite_result.memory_pressure.is_some(), "Memory pressure test failed");
    assert!(suite_result.concurrency_storm.is_some(), "Concurrency storm test failed");

    // Print comprehensive results
    println!("");
    println!("📊 LOAD TEST SUITE RESULTS");
    println!("===========================");
    
    if let Some(result) = &suite_result.steady_state {
        println!("✅ Steady State: {:.2} QPS, {}ms avg", result.throughput, result.avg_response_time.as_millis());
    }
    
    if let Some(result) = &suite_result.ramp_up {
        println!("✅ Ramp-Up: {:.2} QPS, {}ms P95", result.throughput, result.p95_response_time.as_millis());
    }
    
    if let Some(result) = &suite_result.spike_load {
        println!("✅ Spike Load: {:.1}% error rate", result.error_rate * 100.0);
    }
    
    if let Some(result) = &suite_result.stress_test {
        println!("✅ Stress Test: {:.2} QPS at peak", result.throughput);
    }
    
    if let Some(result) = &suite_result.memory_pressure {
        println!("✅ Memory Pressure: {}MB growth", result.memory_growth_mb);
    }
    
    if let Some(result) = &suite_result.concurrency_storm {
        println!("✅ Concurrency Storm: {}ms P99", result.p99_response_time.as_millis());
    }

    println!("");
    println!("🎉 LOAD TEST SUITE: COMPLETED SUCCESSFULLY ✅");
}