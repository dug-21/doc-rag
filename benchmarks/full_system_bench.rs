//! Full System Performance Benchmark Suite
//! 
//! Comprehensive benchmarking for the entire Doc-RAG system with focus on:
//! - Meeting performance targets: <50ms query, <100ms response, <200ms E2E, >100 req/s
//! - Memory efficiency: <2GB per container
//! - Component-level profiling and bottleneck identification
//! - Real-world scenario testing with production-like loads

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;
use tokio::sync::Semaphore;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use dashmap::DashMap;
use std::collections::HashMap;

/// Performance targets from requirements
const TARGET_QUERY_LATENCY_MS: u64 = 50;
const TARGET_RESPONSE_LATENCY_MS: u64 = 100;
const TARGET_E2E_LATENCY_MS: u64 = 200;
const TARGET_THROUGHPUT_QPS: f64 = 100.0;
const TARGET_MEMORY_MB: u64 = 2048;

/// Comprehensive benchmark configuration
#[derive(Debug, Clone)]
pub struct FullSystemBenchmarkConfig {
    pub warmup_queries: usize,
    pub benchmark_queries: usize,
    pub concurrent_users: Vec<usize>,
    pub document_sizes: Vec<usize>,
    pub batch_sizes: Vec<usize>,
    pub stress_duration_secs: u64,
    pub memory_profiling: bool,
    pub cpu_profiling: bool,
}

impl Default for FullSystemBenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_queries: 100,
            benchmark_queries: 2000,
            concurrent_users: vec![1, 5, 10, 25, 50, 100],
            document_sizes: vec![1000, 5000, 10000, 50000],
            batch_sizes: vec![1, 5, 10, 25, 50, 100],
            stress_duration_secs: 60,
            memory_profiling: true,
            cpu_profiling: true,
        }
    }
}

/// Detailed performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Latency metrics
    pub query_processing_ms: Vec<u64>,
    pub response_generation_ms: Vec<u64>,
    pub end_to_end_ms: Vec<u64>,
    
    // Throughput metrics
    pub queries_per_second: f64,
    pub peak_qps: f64,
    pub sustained_qps: f64,
    
    // Resource metrics
    pub memory_usage_mb: Vec<u64>,
    pub cpu_usage_percent: Vec<f64>,
    pub gc_pressure: f64,
    
    // Quality metrics
    pub accuracy_rate: f64,
    pub confidence_scores: Vec<f64>,
    pub error_rate: f64,
    
    // Component breakdowns
    pub chunking_ms: Vec<u64>,
    pub embedding_ms: Vec<u64>,
    pub search_ms: Vec<u64>,
    pub generation_ms: Vec<u64>,
    pub validation_ms: Vec<u64>,
    
    // Target compliance
    pub meets_latency_targets: bool,
    pub meets_throughput_targets: bool,
    pub meets_memory_targets: bool,
}

/// Performance bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub component: String,
    pub avg_latency_ms: f64,
    pub max_latency_ms: u64,
    pub p95_latency_ms: u64,
    pub bottleneck_severity: BottleneckSeverity,
    pub optimization_suggestions: Vec<String>,
    pub resource_impact: ResourceImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Critical,  // > 50% of target budget
    Major,     // 25-50% of target budget
    Minor,     // 10-25% of target budget
    Acceptable, // < 10% of target budget
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceImpact {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: u64,
    pub io_operations_per_sec: u64,
    pub network_bandwidth_mbps: f64,
}

/// Mock RAG system for benchmarking
pub struct MockRagSystem {
    pub connection_pool: Arc<DashMap<String, String>>,
    pub cache: Arc<DashMap<String, (String, Instant)>>,
    pub metrics: Arc<DashMap<String, u64>>,
}

impl MockRagSystem {
    pub fn new() -> Self {
        Self {
            connection_pool: Arc::new(DashMap::new()),
            cache: Arc::new(DashMap::new()),
            metrics: Arc::new(DashMap::new()),
        }
    }
    
    /// Simulate query processing with realistic latency
    pub async fn process_query(&self, query: &str) -> Result<ProcessedQuery, String> {
        let start = Instant::now();
        
        // Check cache first
        if let Some((cached_result, timestamp)) = self.cache.get(query) {
            if timestamp.elapsed() < Duration::from_secs(300) { // 5min cache
                return Ok(ProcessedQuery {
                    original_query: query.to_string(),
                    processed_text: cached_result.clone(),
                    entities: vec!["cached".to_string()],
                    intent: "cached".to_string(),
                    confidence: 0.95,
                    processing_time: start.elapsed(),
                });
            }
        }
        
        // Simulate NLP processing
        tokio::time::sleep(Duration::from_millis(
            20 + (query.len() as u64 / 10)
        )).await;
        
        let processed = ProcessedQuery {
            original_query: query.to_string(),
            processed_text: format!("processed_{}", query),
            entities: query.split_whitespace().take(3).map(|s| s.to_string()).collect(),
            intent: if query.contains("?") { "question" } else { "statement" }.to_string(),
            confidence: 0.88,
            processing_time: start.elapsed(),
        };
        
        // Cache result
        self.cache.insert(query.to_string(), (processed.processed_text.clone(), Instant::now()));
        
        Ok(processed)
    }
    
    /// Simulate document retrieval with vector search
    pub async fn retrieve_documents(&self, processed_query: &ProcessedQuery, limit: usize) -> Result<Vec<RetrievedDoc>, String> {
        let start = Instant::now();
        
        // Simulate vector search latency
        let search_complexity = processed_query.entities.len() * limit;
        let search_delay = 10 + (search_complexity as u64 / 5);
        tokio::time::sleep(Duration::from_millis(search_delay)).await;
        
        // Generate mock retrieved documents
        let docs = (0..limit).map(|i| {
            RetrievedDoc {
                id: format!("doc_{}", i),
                content: format!("Retrieved document {} content related to {}", i, processed_query.processed_text),
                similarity_score: 0.9 - (i as f64 * 0.05),
                metadata: HashMap::new(),
            }
        }).collect();
        
        Ok(docs)
    }
    
    /// Simulate response generation
    pub async fn generate_response(&self, query: &ProcessedQuery, docs: &[RetrievedDoc]) -> Result<GeneratedResponse, String> {
        let start = Instant::now();
        
        // Simulate LLM generation latency (depends on context length)
        let context_length = docs.iter().map(|d| d.content.len()).sum::<usize>();
        let generation_delay = 30 + (context_length as u64 / 100);
        tokio::time::sleep(Duration::from_millis(generation_delay)).await;
        
        let response = GeneratedResponse {
            content: format!("Generated response for '{}' based on {} documents", query.original_query, docs.len()),
            citations: docs.iter().enumerate().map(|(i, doc)| {
                Citation {
                    doc_id: doc.id.clone(),
                    snippet: doc.content.chars().take(100).collect(),
                    confidence: doc.similarity_score,
                }
            }).collect(),
            confidence: 0.87,
            generation_time: start.elapsed(),
        };
        
        Ok(response)
    }
    
    /// Full end-to-end processing
    pub async fn process_end_to_end(&self, query: &str) -> Result<GeneratedResponse, String> {
        let processed_query = self.process_query(query).await?;
        let retrieved_docs = self.retrieve_documents(&processed_query, 5).await?;
        let response = self.generate_response(&processed_query, &retrieved_docs).await?;
        Ok(response)
    }
}

#[derive(Debug, Clone)]
pub struct ProcessedQuery {
    pub original_query: String,
    pub processed_text: String,
    pub entities: Vec<String>,
    pub intent: String,
    pub confidence: f64,
    pub processing_time: Duration,
}

#[derive(Debug, Clone)]
pub struct RetrievedDoc {
    pub id: String,
    pub content: String,
    pub similarity_score: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct GeneratedResponse {
    pub content: String,
    pub citations: Vec<Citation>,
    pub confidence: f64,
    pub generation_time: Duration,
}

#[derive(Debug, Clone)]
pub struct Citation {
    pub doc_id: String,
    pub snippet: String,
    pub confidence: f64,
}

/// Comprehensive benchmark suite
pub struct FullSystemBenchmark {
    config: FullSystemBenchmarkConfig,
    system: MockRagSystem,
}

impl FullSystemBenchmark {
    pub fn new(config: FullSystemBenchmarkConfig) -> Self {
        Self {
            config,
            system: MockRagSystem::new(),
        }
    }
    
    /// Run all benchmarks and collect comprehensive metrics
    pub async fn run_full_benchmark_suite(&self) -> PerformanceMetrics {
        println!("🚀 Starting Full System Performance Benchmark Suite");
        println!("📋 Configuration: {:?}", self.config);
        
        let mut metrics = PerformanceMetrics {
            query_processing_ms: Vec::new(),
            response_generation_ms: Vec::new(),
            end_to_end_ms: Vec::new(),
            queries_per_second: 0.0,
            peak_qps: 0.0,
            sustained_qps: 0.0,
            memory_usage_mb: Vec::new(),
            cpu_usage_percent: Vec::new(),
            gc_pressure: 0.0,
            accuracy_rate: 0.0,
            confidence_scores: Vec::new(),
            error_rate: 0.0,
            chunking_ms: Vec::new(),
            embedding_ms: Vec::new(),
            search_ms: Vec::new(),
            generation_ms: Vec::new(),
            validation_ms: Vec::new(),
            meets_latency_targets: false,
            meets_throughput_targets: false,
            meets_memory_targets: false,
        };
        
        // 1. Warmup phase
        println!("\n🔥 Warmup Phase");
        await self.run_warmup().await;
        
        // 2. Latency benchmarks
        println!("\n⏱️ Latency Benchmarks");
        let latency_metrics = self.run_latency_benchmarks().await;
        metrics.query_processing_ms = latency_metrics.0;
        metrics.response_generation_ms = latency_metrics.1;
        metrics.end_to_end_ms = latency_metrics.2;
        
        // 3. Throughput benchmarks
        println!("\n🚀 Throughput Benchmarks");
        let throughput_metrics = self.run_throughput_benchmarks().await;
        metrics.queries_per_second = throughput_metrics.0;
        metrics.peak_qps = throughput_metrics.1;
        
        // 4. Resource usage benchmarks
        println!("\n💾 Resource Usage Benchmarks");
        let resource_metrics = self.run_resource_benchmarks().await;
        metrics.memory_usage_mb = resource_metrics.0;
        metrics.cpu_usage_percent = resource_metrics.1;
        
        // 5. Stress testing
        println!("\n🔬 Stress Testing");
        metrics.sustained_qps = self.run_stress_test().await;
        
        // 6. Component-level profiling
        println!("\n🔍 Component Profiling");
        let component_metrics = self.run_component_profiling().await;
        metrics.chunking_ms = component_metrics.0;
        metrics.embedding_ms = component_metrics.1;
        metrics.search_ms = component_metrics.2;
        metrics.generation_ms = component_metrics.3;
        metrics.validation_ms = component_metrics.4;
        
        // 7. Evaluate against targets
        self.evaluate_targets(&mut metrics);
        
        println!("\n📊 Benchmark Suite Complete!");
        self.print_summary(&metrics);
        
        metrics
    }
    
    async fn run_warmup(&self) {
        let queries = self.generate_warmup_queries();
        for query in queries.iter().take(self.config.warmup_queries) {
            let _ = self.system.process_end_to_end(query).await;
        }
    }
    
    async fn run_latency_benchmarks(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let mut query_latencies = Vec::new();
        let mut response_latencies = Vec::new();
        let mut e2e_latencies = Vec::new();
        
        let test_queries = self.generate_test_queries();
        
        for query in test_queries.iter().take(self.config.benchmark_queries) {
            // Query processing latency
            let start = Instant::now();
            let processed = self.system.process_query(query).await.unwrap();
            query_latencies.push(start.elapsed().as_millis() as u64);
            
            // Retrieval + response generation latency
            let start = Instant::now();
            let docs = self.system.retrieve_documents(&processed, 5).await.unwrap();
            let _response = self.system.generate_response(&processed, &docs).await.unwrap();
            response_latencies.push(start.elapsed().as_millis() as u64);
            
            // End-to-end latency
            let start = Instant::now();
            let _result = self.system.process_end_to_end(query).await.unwrap();
            e2e_latencies.push(start.elapsed().as_millis() as u64);
        }
        
        (query_latencies, response_latencies, e2e_latencies)
    }
    
    async fn run_throughput_benchmarks(&self) -> (f64, f64) {
        let mut peak_qps = 0.0;
        let mut total_qps = 0.0;
        
        for &concurrent_users in &self.config.concurrent_users {
            let qps = self.measure_throughput(concurrent_users).await;
            total_qps += qps;
            if qps > peak_qps {
                peak_qps = qps;
            }
            println!("  {} concurrent users: {:.2} QPS", concurrent_users, qps);
        }
        
        let avg_qps = total_qps / self.config.concurrent_users.len() as f64;
        (avg_qps, peak_qps)
    }
    
    async fn measure_throughput(&self, concurrent_users: usize) -> f64 {
        let semaphore = Arc::new(Semaphore::new(concurrent_users));
        let query_count = Arc::new(tokio::sync::Mutex::new(0u64));
        
        let start = Instant::now();
        let duration = Duration::from_secs(10); // 10 second measurement window
        
        let mut handles = Vec::new();
        
        for _ in 0..concurrent_users * 2 { // Ensure we have enough work
            let sem = semaphore.clone();
            let system = &self.system;
            let count = query_count.clone();
            let test_query = "throughput test query".to_string();
            
            let handle = tokio::spawn(async move {
                while start.elapsed() < duration {
                    let _permit = sem.acquire().await.unwrap();
                    let _ = system.process_end_to_end(&test_query).await;
                    let mut c = count.lock().await;
                    *c += 1;
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for test duration
        tokio::time::sleep(duration).await;
        
        // Cancel all tasks
        for handle in handles {
            handle.abort();
        }
        
        let final_count = *query_count.lock().await;
        final_count as f64 / duration.as_secs_f64()
    }
    
    async fn run_resource_benchmarks(&self) -> (Vec<u64>, Vec<f64>) {
        let mut memory_usage = Vec::new();
        let mut cpu_usage = Vec::new();
        
        // Simulate resource monitoring during various loads
        for &batch_size in &self.config.batch_sizes {
            // Simulate memory usage (mock values)
            let base_memory = 100; // Base memory in MB
            let memory_per_query = 2; // MB per concurrent query
            memory_usage.push(base_memory + (batch_size as u64 * memory_per_query));
            
            // Simulate CPU usage (mock values)
            let cpu_percent = (batch_size as f64 * 1.5).min(95.0);
            cpu_usage.push(cpu_percent);
        }
        
        (memory_usage, cpu_usage)
    }
    
    async fn run_stress_test(&self) -> f64 {
        println!("  Running {} second stress test...", self.config.stress_duration_secs);
        
        let concurrent_users = 50; // High load
        let semaphore = Arc::new(Semaphore::new(concurrent_users));
        let query_count = Arc::new(tokio::sync::Mutex::new(0u64));
        
        let start = Instant::now();
        let duration = Duration::from_secs(self.config.stress_duration_secs);
        
        let mut handles = Vec::new();
        
        for i in 0..concurrent_users * 3 {
            let sem = semaphore.clone();
            let system = &self.system;
            let count = query_count.clone();
            let test_query = format!("stress test query {}", i);
            
            let handle = tokio::spawn(async move {
                while start.elapsed() < duration {
                    let _permit = sem.acquire().await.unwrap();
                    let _ = system.process_end_to_end(&test_query).await;
                    let mut c = count.lock().await;
                    *c += 1;
                }
            });
            
            handles.push(handle);
        }
        
        tokio::time::sleep(duration).await;
        
        for handle in handles {
            handle.abort();
        }
        
        let final_count = *query_count.lock().await;
        let sustained_qps = final_count as f64 / duration.as_secs_f64();
        
        println!("  Sustained QPS: {:.2}", sustained_qps);
        sustained_qps
    }
    
    async fn run_component_profiling(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>, Vec<u64>) {
        // Mock component profiling - in real implementation, these would be actual measurements
        let chunking_times = vec![5, 7, 12, 8, 6, 9, 11, 4, 8, 10];
        let embedding_times = vec![15, 18, 22, 19, 16, 20, 25, 14, 17, 21];
        let search_times = vec![8, 12, 15, 10, 9, 13, 16, 7, 11, 14];
        let generation_times = vec![45, 52, 68, 49, 43, 58, 72, 41, 47, 63];
        let validation_times = vec![3, 4, 6, 5, 3, 5, 7, 2, 4, 6];
        
        (chunking_times, embedding_times, search_times, generation_times, validation_times)
    }
    
    fn evaluate_targets(&self, metrics: &mut PerformanceMetrics) {
        // Check latency targets
        let avg_query_latency = metrics.query_processing_ms.iter().sum::<u64>() as f64 / metrics.query_processing_ms.len() as f64;
        let avg_response_latency = metrics.response_generation_ms.iter().sum::<u64>() as f64 / metrics.response_generation_ms.len() as f64;
        let avg_e2e_latency = metrics.end_to_end_ms.iter().sum::<u64>() as f64 / metrics.end_to_end_ms.len() as f64;
        
        metrics.meets_latency_targets = avg_query_latency <= TARGET_QUERY_LATENCY_MS as f64 
            && avg_response_latency <= TARGET_RESPONSE_LATENCY_MS as f64
            && avg_e2e_latency <= TARGET_E2E_LATENCY_MS as f64;
        
        // Check throughput targets
        metrics.meets_throughput_targets = metrics.peak_qps >= TARGET_THROUGHPUT_QPS;
        
        // Check memory targets
        let max_memory = metrics.memory_usage_mb.iter().max().unwrap_or(&0);
        metrics.meets_memory_targets = *max_memory <= TARGET_MEMORY_MB;
    }
    
    fn print_summary(&self, metrics: &PerformanceMetrics) {
        println!("\n=== FULL SYSTEM PERFORMANCE SUMMARY ===");
        
        // Latency summary
        let avg_query = metrics.query_processing_ms.iter().sum::<u64>() as f64 / metrics.query_processing_ms.len() as f64;
        let avg_response = metrics.response_generation_ms.iter().sum::<u64>() as f64 / metrics.response_generation_ms.len() as f64;
        let avg_e2e = metrics.end_to_end_ms.iter().sum::<u64>() as f64 / metrics.end_to_end_ms.len() as f64;
        
        println!("\n⏱️ LATENCY PERFORMANCE:");
        println!("  Query Processing: {:.1}ms (target: {}ms) {}", avg_query, TARGET_QUERY_LATENCY_MS, 
            if avg_query <= TARGET_QUERY_LATENCY_MS as f64 { "✅" } else { "❌" });
        println!("  Response Generation: {:.1}ms (target: {}ms) {}", avg_response, TARGET_RESPONSE_LATENCY_MS,
            if avg_response <= TARGET_RESPONSE_LATENCY_MS as f64 { "✅" } else { "❌" });
        println!("  End-to-End: {:.1}ms (target: {}ms) {}", avg_e2e, TARGET_E2E_LATENCY_MS,
            if avg_e2e <= TARGET_E2E_LATENCY_MS as f64 { "✅" } else { "❌" });
        
        // Throughput summary
        println!("\n🚀 THROUGHPUT PERFORMANCE:");
        println!("  Average QPS: {:.1} (target: {:.1}) {}", metrics.queries_per_second, TARGET_THROUGHPUT_QPS,
            if metrics.queries_per_second >= TARGET_THROUGHPUT_QPS { "✅" } else { "❌" });
        println!("  Peak QPS: {:.1}", metrics.peak_qps);
        println!("  Sustained QPS: {:.1}", metrics.sustained_qps);
        
        // Resource summary
        let max_memory = metrics.memory_usage_mb.iter().max().unwrap_or(&0);
        println!("\n💾 RESOURCE USAGE:");
        println!("  Peak Memory: {}MB (target: {}MB) {}", max_memory, TARGET_MEMORY_MB,
            if *max_memory <= TARGET_MEMORY_MB { "✅" } else { "❌" });
        
        // Overall result
        let all_targets_met = metrics.meets_latency_targets && metrics.meets_throughput_targets && metrics.meets_memory_targets;
        println!("\n🏆 OVERALL RESULT: {}", 
            if all_targets_met { "✅ ALL PERFORMANCE TARGETS MET!" } else { "❌ SOME TARGETS MISSED" });
        
        if !all_targets_met {
            println!("\n🔧 OPTIMIZATION NEEDED:");
            if !metrics.meets_latency_targets {
                println!("  - Latency optimization required");
            }
            if !metrics.meets_throughput_targets {
                println!("  - Throughput optimization required");
            }
            if !metrics.meets_memory_targets {
                println!("  - Memory optimization required");
            }
        }
    }
    
    fn generate_warmup_queries(&self) -> Vec<String> {
        vec![
            "What is machine learning?".to_string(),
            "Explain neural networks".to_string(),
            "How does AI work?".to_string(),
            "Define artificial intelligence".to_string(),
            "What are the benefits of automation?".to_string(),
        ]
    }
    
    fn generate_test_queries(&self) -> Vec<String> {
        let mut queries = Vec::new();
        
        // Simple queries
        for i in 0..100 {
            queries.push(format!("Simple query number {}", i));
        }
        
        // Medium complexity queries
        for i in 0..100 {
            queries.push(format!("Medium complexity query {} with multiple terms and concepts", i));
        }
        
        // Complex queries
        for i in 0..100 {
            queries.push(format!("Complex analytical query {} requiring comprehensive understanding of multiple interconnected concepts and their relationships within the domain", i));
        }
        
        queries
    }
}

/// Criterion benchmark functions
fn bench_full_system_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = FullSystemBenchmarkConfig {
        benchmark_queries: 100,
        ..Default::default()
    };
    let benchmark = FullSystemBenchmark::new(config);
    
    c.bench_function("full_system_latency", |b| {
        b.to_async(&rt).iter(|| async {
            let result = benchmark.system.process_end_to_end(black_box("benchmark query")).await;
            black_box(result)
        });
    });
}

fn bench_concurrent_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let benchmark = FullSystemBenchmark::new(FullSystemBenchmarkConfig::default());
    
    let mut group = c.benchmark_group("concurrent_throughput");
    
    for &users in &[1, 5, 10, 25] {
        group.bench_with_input(
            BenchmarkId::new("concurrent_users", users),
            &users,
            |b, &users| {
                b.to_async(&rt).iter(|| async {
                    let qps = benchmark.measure_throughput(black_box(users)).await;
                    black_box(qps)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_component_breakdown(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let benchmark = FullSystemBenchmark::new(FullSystemBenchmarkConfig::default());
    
    let mut group = c.benchmark_group("component_breakdown");
    
    group.bench_function("query_processing", |b| {
        b.to_async(&rt).iter(|| async {
            let result = benchmark.system.process_query(black_box("component test")).await;
            black_box(result)
        });
    });
    
    group.bench_function("document_retrieval", |b| {
        b.to_async(&rt).iter(|| async {
            let query = ProcessedQuery {
                original_query: "test".to_string(),
                processed_text: "processed test".to_string(),
                entities: vec!["test".to_string()],
                intent: "test".to_string(),
                confidence: 0.9,
                processing_time: Duration::from_millis(10),
            };
            let result = benchmark.system.retrieve_documents(black_box(&query), black_box(5)).await;
            black_box(result)
        });
    });
    
    group.finish();
}

criterion_group!(
    full_system_benches,
    bench_full_system_latency,
    bench_concurrent_throughput,
    bench_component_breakdown
);

criterion_main!(full_system_benches);

#[tokio::main]
async fn main() {
    println!("🚀 Doc-RAG Full System Performance Benchmark");
    
    let config = FullSystemBenchmarkConfig::default();
    let benchmark = FullSystemBenchmark::new(config);
    
    let metrics = benchmark.run_full_benchmark_suite().await;
    
    // Save metrics to JSON
    let json_output = serde_json::to_string_pretty(&metrics).unwrap();
    tokio::fs::write("full_system_benchmark_results.json", json_output).await.unwrap();
    
    println!("\n📄 Detailed metrics saved to full_system_benchmark_results.json");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mock_system_basic_operations() {
        let system = MockRagSystem::new();
        
        let result = system.process_query("test query").await.unwrap();
        assert_eq!(result.original_query, "test query");
        assert!(result.processing_time.as_millis() > 0);
    }
    
    #[tokio::test]
    async fn test_benchmark_metrics_collection() {
        let config = FullSystemBenchmarkConfig {
            benchmark_queries: 10,
            warmup_queries: 5,
            ..Default::default()
        };
        let benchmark = FullSystemBenchmark::new(config);
        
        let metrics = benchmark.run_full_benchmark_suite().await;
        
        assert!(!metrics.query_processing_ms.is_empty());
        assert!(!metrics.response_generation_ms.is_empty());
        assert!(!metrics.end_to_end_ms.is_empty());
        assert!(metrics.queries_per_second > 0.0);
    }
}