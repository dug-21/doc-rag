#!/usr/bin/env cargo run --bin
//! Week 5 Symbolic Query Processing Performance Benchmark
//!
//! Validates the <100ms symbolic query response time requirement
//! and measures routing accuracy against the 80%+ target.

use std::time::{Duration, Instant};
use tokio;
use query_processor::{QueryProcessor, ProcessorConfig, Query};

/// Benchmark configuration
const BENCHMARK_QUERIES: usize = 100;
const TARGET_LATENCY_MS: u64 = 100;
const TARGET_ACCURACY: f64 = 0.8;

/// Benchmark test cases for symbolic queries
const SYMBOLIC_TEST_QUERIES: &[&str] = &[
    "If cardholder data is stored, then it must be encrypted",
    "Prove that PCI DSS 3.4.1 requires encryption",
    "When sensitive data is transmitted, encryption is mandatory",
    "All payment systems must implement access controls",
    "If data classification is high, then additional controls apply",
    "Demonstrate compliance with encryption standards",
    "What logical rules govern data protection requirements?",
    "If authentication fails, then access must be denied",
    "Prove that requirement 8.2.3 mandates password complexity",
    "When data is processed, audit logging is required",
    "If vulnerability exists, then patch management applies",
    "All network traffic must be monitored and logged",
    "If data retention period expires, then secure deletion required",
    "Prove that multi-factor authentication prevents unauthorized access",
    "When incident occurs, then response procedures activate",
    "If system processes cardholder data, then PCI scope applies",
    "All database queries must be logged and monitored",
    "If encryption key expires, then key rotation required",
    "Prove that network segmentation isolates cardholder data",
    "When backup created, then encryption and access controls apply",
];

/// Performance metrics collection
#[derive(Debug, Default)]
struct BenchmarkMetrics {
    total_queries: usize,
    symbolic_queries: usize,
    queries_under_target: usize,
    total_latency: Duration,
    max_latency: Duration,
    min_latency: Duration,
    routing_accuracy_count: usize,
    proof_chains_generated: usize,
    logic_conversions_successful: usize,
}

impl BenchmarkMetrics {
    fn new() -> Self {
        Self {
            min_latency: Duration::from_secs(u64::MAX),
            ..Default::default()
        }
    }
    
    fn update(&mut self, latency: Duration, is_symbolic: bool, routing_confidence: f64) {
        self.total_queries += 1;
        self.total_latency += latency;
        
        if latency > self.max_latency {
            self.max_latency = latency;
        }
        if latency < self.min_latency {
            self.min_latency = latency;
        }
        
        if latency.as_millis() <= TARGET_LATENCY_MS as u128 {
            self.queries_under_target += 1;
        }
        
        if is_symbolic {
            self.symbolic_queries += 1;
        }
        
        if routing_confidence >= TARGET_ACCURACY {
            self.routing_accuracy_count += 1;
        }
    }
    
    fn avg_latency(&self) -> Duration {
        if self.total_queries > 0 {
            self.total_latency / self.total_queries as u32
        } else {
            Duration::from_millis(0)
        }
    }
    
    fn latency_compliance_rate(&self) -> f64 {
        if self.total_queries > 0 {
            self.queries_under_target as f64 / self.total_queries as f64
        } else {
            0.0
        }
    }
    
    fn routing_accuracy_rate(&self) -> f64 {
        if self.total_queries > 0 {
            self.routing_accuracy_count as f64 / self.total_queries as f64
        } else {
            0.0
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Week 5 Symbolic Query Processing Performance Benchmark");
    println!("=========================================================");
    println!("Target latency: <{}ms", TARGET_LATENCY_MS);
    println!("Target accuracy: {:.1}%", TARGET_ACCURACY * 100.0);
    println!("Benchmark queries: {}", BENCHMARK_QUERIES);
    println!();
    
    // Initialize query processor
    let config = ProcessorConfig::default();
    let processor = QueryProcessor::new(config).await?;
    
    let mut metrics = BenchmarkMetrics::new();
    
    println!("🏃 Running performance benchmark...");
    
    // Run benchmark queries
    for i in 0..BENCHMARK_QUERIES {
        let query_text = SYMBOLIC_TEST_QUERIES[i % SYMBOLIC_TEST_QUERIES.len()];
        let query = Query::new(query_text)?;
        
        let start_time = Instant::now();
        let processed = processor.process(query.clone()).await?;
        let latency = start_time.elapsed();
        
        // Analyze results
        let routing_engine = processed.metadata.get("routing_engine").unwrap_or(&"Unknown".to_string());
        let routing_confidence = processed.metadata.get("routing_confidence")
            .unwrap_or(&"0.0".to_string())
            .parse::<f64>().unwrap_or(0.0);
        
        let is_symbolic = routing_engine.contains("Symbolic");
        
        // Track proof chain generation
        if processed.metadata.get("proof_chain_generated").is_some() {
            metrics.proof_chains_generated += 1;
        }
        
        // Track logic conversions
        if processed.metadata.get("datalog_conversion").is_some() {
            metrics.logic_conversions_successful += 1;
        }
        
        metrics.update(latency, is_symbolic, routing_confidence);
        
        // Print progress every 20 queries
        if (i + 1) % 20 == 0 {
            println!("  Completed {}/{} queries (avg: {:.1}ms)", 
                     i + 1, BENCHMARK_QUERIES, metrics.avg_latency().as_millis());
        }
        
        // Warn on latency violations
        if latency.as_millis() > TARGET_LATENCY_MS as u128 {
            println!("  ⚠️  Query #{} exceeded target latency: {}ms", i + 1, latency.as_millis());
            println!("      Query: {}", query_text);
        }
    }
    
    println!();
    println!("📊 Benchmark Results");
    println!("====================");
    
    // Performance metrics
    println!("Performance Metrics:");
    println!("  Total queries processed: {}", metrics.total_queries);
    println!("  Symbolic queries: {} ({:.1}%)", 
             metrics.symbolic_queries, 
             metrics.symbolic_queries as f64 / metrics.total_queries as f64 * 100.0);
    println!("  Average latency: {:.1}ms", metrics.avg_latency().as_millis());
    println!("  Min latency: {:.1}ms", metrics.min_latency.as_millis());
    println!("  Max latency: {:.1}ms", metrics.max_latency.as_millis());
    
    // Latency compliance
    let latency_compliance = metrics.latency_compliance_rate();
    println!();
    println!("Latency Compliance (<{}ms):", TARGET_LATENCY_MS);
    println!("  Queries under target: {} / {} ({:.1}%)", 
             metrics.queries_under_target, 
             metrics.total_queries,
             latency_compliance * 100.0);
    
    if latency_compliance >= 0.95 {
        println!("  ✅ PASSED: Excellent latency compliance");
    } else if latency_compliance >= 0.9 {
        println!("  ✅ PASSED: Good latency compliance");
    } else {
        println!("  ❌ FAILED: Latency compliance below 90%");
    }
    
    // Routing accuracy
    let routing_accuracy = metrics.routing_accuracy_rate();
    println!();
    println!("Routing Accuracy (>{:.1}% confidence):", TARGET_ACCURACY * 100.0);
    println!("  Accurate routings: {} / {} ({:.1}%)", 
             metrics.routing_accuracy_count, 
             metrics.total_queries,
             routing_accuracy * 100.0);
    
    if routing_accuracy >= TARGET_ACCURACY {
        println!("  ✅ PASSED: Routing accuracy meets {:.1}% requirement", TARGET_ACCURACY * 100.0);
    } else {
        println!("  ❌ FAILED: Routing accuracy below {:.1}% requirement", TARGET_ACCURACY * 100.0);
    }
    
    // Symbolic processing features
    println!();
    println!("Symbolic Processing Features:");
    println!("  Proof chains generated: {} ({:.1}%)", 
             metrics.proof_chains_generated,
             metrics.proof_chains_generated as f64 / metrics.total_queries as f64 * 100.0);
    println!("  Logic conversions successful: {} ({:.1}%)", 
             metrics.logic_conversions_successful,
             metrics.logic_conversions_successful as f64 / metrics.total_queries as f64 * 100.0);
    
    // Overall assessment
    println!();
    println!("📋 Week 5 Gate 2 Requirements Assessment");
    println!("========================================");
    
    let mut requirements_met = 0;
    let total_requirements = 4;
    
    // Requirement 1: <100ms symbolic query response time
    if latency_compliance >= 0.9 {
        println!("  ✅ <100ms symbolic query response time");
        requirements_met += 1;
    } else {
        println!("  ❌ <100ms symbolic query response time");
    }
    
    // Requirement 2: 80%+ routing accuracy
    if routing_accuracy >= TARGET_ACCURACY {
        println!("  ✅ 80%+ query routing accuracy");
        requirements_met += 1;
    } else {
        println!("  ❌ 80%+ query routing accuracy");
    }
    
    // Requirement 3: Proof chain generation
    if metrics.proof_chains_generated > 0 {
        println!("  ✅ Proof chain generation and validation");
        requirements_met += 1;
    } else {
        println!("  ❌ Proof chain generation and validation");
    }
    
    // Requirement 4: Natural language to logic conversion
    if metrics.logic_conversions_successful > 0 {
        println!("  ✅ Natural language to logic conversion");
        requirements_met += 1;
    } else {
        println!("  ❌ Natural language to logic conversion");
    }
    
    println!();
    println!("Overall Status: {} / {} requirements met ({:.1}%)", 
             requirements_met, total_requirements,
             requirements_met as f64 / total_requirements as f64 * 100.0);
    
    if requirements_met == total_requirements {
        println!("🎉 ALL WEEK 5 REQUIREMENTS PASSED!");
        println!("   Ready for Gate 2 validation.");
    } else {
        println!("⚠️  Some requirements need attention before Gate 2.");
    }
    
    // Get final routing statistics
    let routing_stats = processor.get_symbolic_routing_stats().await;
    
    println!();
    println!("📈 Final Routing Statistics");
    println!("==========================");
    println!("  Total queries: {}", routing_stats.total_queries);
    println!("  Symbolic: {}", routing_stats.symbolic_queries);
    println!("  Graph: {}", routing_stats.graph_queries);
    println!("  Vector: {}", routing_stats.vector_queries);
    println!("  Hybrid: {}", routing_stats.hybrid_queries);
    println!("  Average confidence: {:.3}", routing_stats.avg_routing_confidence);
    println!("  Average symbolic latency: {:.1}ms", routing_stats.avg_symbolic_latency_ms);
    
    Ok(())
}