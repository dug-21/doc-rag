// tests/performance/symbolic_engine_performance_tests.rs
// Performance Test Suite for REAL Engine Integration - CONSTRAINT-001 Validation

use std::time::{Duration, Instant};
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;

use symbolic::datalog::{DatalogEngine, DatalogRule};
use symbolic::prolog::{PrologEngine};
use symbolic::logic_parser::{LogicParser};
use symbolic::error::Result;

/// CONSTRAINT-001: ALL operations MUST complete within 100ms
const PERFORMANCE_CONSTRAINT_MS: u128 = 100;
const STRICT_CONSTRAINT_MS: u128 = 50; // For critical operations

#[tokio::test]
async fn test_datalog_engine_initialization_performance() -> Result<()> {
    // Test REAL Crepe engine initialization speed
    let iterations = 10;
    let mut init_times = Vec::new();
    
    for _ in 0..iterations {
        let start_time = Instant::now();
        let _engine = DatalogEngine::new().await?;
        let init_time = start_time.elapsed();
        init_times.push(init_time.as_millis());
    }
    
    let avg_time = init_times.iter().sum::<u128>() / iterations as u128;
    let max_time = *init_times.iter().max().unwrap();
    let min_time = *init_times.iter().min().unwrap();
    
    println!("Datalog Engine Init Performance:");
    println!("  Average: {}ms", avg_time);
    println!("  Max: {}ms", max_time);
    println!("  Min: {}ms", min_time);
    
    // CONSTRAINT-001 validation
    assert!(avg_time <= PERFORMANCE_CONSTRAINT_MS, 
        "Average init time {}ms exceeds constraint {}ms", avg_time, PERFORMANCE_CONSTRAINT_MS);
    assert!(max_time <= PERFORMANCE_CONSTRAINT_MS,
        "Max init time {}ms exceeds constraint {}ms", max_time, PERFORMANCE_CONSTRAINT_MS);
    
    Ok(())
}

#[tokio::test]
async fn test_prolog_engine_initialization_performance() -> Result<()> {
    // Test REAL Scryer-Prolog engine initialization speed
    let iterations = 10;
    let mut init_times = Vec::new();
    
    for _ in 0..iterations {
        let start_time = Instant::now();
        let _engine = PrologEngine::new().await?;
        let init_time = start_time.elapsed();
        init_times.push(init_time.as_millis());
    }
    
    let avg_time = init_times.iter().sum::<u128>() / iterations as u128;
    let max_time = *init_times.iter().max().unwrap();
    let min_time = *init_times.iter().min().unwrap();
    
    println!("Prolog Engine Init Performance:");
    println!("  Average: {}ms", avg_time);
    println!("  Max: {}ms", max_time);
    println!("  Min: {}ms", min_time);
    
    // CONSTRAINT-001 validation
    assert!(avg_time <= PERFORMANCE_CONSTRAINT_MS);
    assert!(max_time <= PERFORMANCE_CONSTRAINT_MS);
    
    Ok(())
}

#[tokio::test]
async fn test_logic_parser_initialization_performance() -> Result<()> {
    // Test REAL Logic Parser initialization speed
    let iterations = 10;
    let mut init_times = Vec::new();
    
    for _ in 0..iterations {
        let start_time = Instant::now();
        let _parser = LogicParser::new().await?;
        let init_time = start_time.elapsed();
        init_times.push(init_time.as_millis());
    }
    
    let avg_time = init_times.iter().sum::<u128>() / iterations as u128;
    let max_time = *init_times.iter().max().unwrap();
    let min_time = *init_times.iter().min().unwrap();
    
    println!("Logic Parser Init Performance:");
    println!("  Average: {}ms", avg_time);
    println!("  Max: {}ms", max_time);
    println!("  Min: {}ms", min_time);
    
    // CONSTRAINT-001 validation
    assert!(avg_time <= PERFORMANCE_CONSTRAINT_MS);
    assert!(max_time <= PERFORMANCE_CONSTRAINT_MS);
    
    Ok(())
}

#[tokio::test]
async fn test_single_rule_compilation_performance() -> Result<()> {
    // Test single rule compilation speed
    let requirements = vec![
        "Cardholder data MUST be encrypted",
        "Payment systems MUST implement access controls", 
        "Sensitive data SHALL be protected during transmission",
        "Authentication data SHOULD use strong cryptography",
        "Audit logs MAY be stored for extended periods",
    ];
    
    let mut compile_times = Vec::new();
    
    for requirement in requirements {
        let start_time = Instant::now();
        let _rule = DatalogEngine::compile_requirement_to_rule(requirement).await?;
        let compile_time = start_time.elapsed();
        compile_times.push(compile_time.as_millis());
        
        println!("Compiled '{}' in {}ms", requirement, compile_time.as_millis());
    }
    
    let avg_time = compile_times.iter().sum::<u128>() / compile_times.len() as u128;
    let max_time = *compile_times.iter().max().unwrap();
    
    println!("Rule Compilation Performance:");
    println!("  Average: {}ms", avg_time);
    println!("  Max: {}ms", max_time);
    
    // Should be very fast for individual rules
    assert!(avg_time <= STRICT_CONSTRAINT_MS);
    assert!(max_time <= PERFORMANCE_CONSTRAINT_MS);
    
    Ok(())
}

#[tokio::test]
async fn test_batch_rule_addition_performance() -> Result<()> {
    // Test performance of adding multiple rules
    let engine = DatalogEngine::new().await?;
    let rule_counts = vec![10, 50, 100, 200];
    
    for count in rule_counts {
        let start_time = Instant::now();
        
        for i in 0..count {
            let rule = DatalogEngine::compile_requirement_to_rule(
                &format!("Entity_{} MUST have_property_{}", i, i % 10)
            ).await?;
            engine.add_rule(rule).await?;
        }
        
        let batch_time = start_time.elapsed();
        let avg_per_rule = batch_time.as_millis() / count as u128;
        
        println!("Added {} rules in {}ms ({}ms per rule)", 
                count, batch_time.as_millis(), avg_per_rule);
        
        // Total batch should meet constraint
        assert!(batch_time.as_millis() <= PERFORMANCE_CONSTRAINT_MS * (count as u128 / 10).max(1));
        
        // Individual rule addition should be fast
        assert!(avg_per_rule <= 10); // Very fast individual additions
    }
    
    Ok(())
}

#[tokio::test]
async fn test_simple_query_performance() -> Result<()> {
    // Test performance of simple queries
    let engine = DatalogEngine::new().await?;
    
    // Add test rules
    for i in 0..20 {
        let rule = DatalogEngine::compile_requirement_to_rule(
            &format!("Data_{} MUST be encrypted", i)
        ).await?;
        engine.add_rule(rule).await?;
    }
    
    let queries = vec![
        "What data must be encrypted?",
        "Which entities require protection?",
        "What compliance rules exist?",
        "Which data types are sensitive?",
    ];
    
    let mut query_times = Vec::new();
    
    for query in queries {
        let start_time = Instant::now();
        let _result = engine.query(query).await?;
        let query_time = start_time.elapsed();
        query_times.push(query_time.as_millis());
        
        println!("Query '{}' took {}ms", query, query_time.as_millis());
        
        // Each query MUST meet constraint
        assert!(query_time.as_millis() <= PERFORMANCE_CONSTRAINT_MS);
    }
    
    let avg_query_time = query_times.iter().sum::<u128>() / query_times.len() as u128;
    let max_query_time = *query_times.iter().max().unwrap();
    
    println!("Query Performance Summary:");
    println!("  Average: {}ms", avg_query_time);
    println!("  Max: {}ms", max_query_time);
    
    assert!(avg_query_time <= STRICT_CONSTRAINT_MS); // Should be very fast
    assert!(max_query_time <= PERFORMANCE_CONSTRAINT_MS);
    
    Ok(())
}

#[tokio::test]
async fn test_complex_inference_performance() -> Result<()> {
    // Test performance of complex inference queries
    let engine = PrologEngine::new().await?;
    
    // Build complex knowledge base
    let base_facts = 50;
    let inference_rules = 20;
    
    let setup_start = Instant::now();
    
    // Add base facts
    for i in 0..base_facts {
        engine.add_compliance_rule(
            &format!("base_entity(entity_{}).", i), 
            "performance_test"
        ).await?;
        
        engine.add_compliance_rule(
            &format!("has_property(entity_{}, property_{}).", i, i % 10), 
            "performance_test"
        ).await?;
    }
    
    // Add inference rules
    for i in 0..inference_rules {
        engine.add_compliance_rule(
            &format!("derived_property_{}(X) :- has_property(X, property_{}).", i, i % 10),
            "performance_test"
        ).await?;
        
        engine.add_compliance_rule(
            &format!("complex_conclusion_{}(X) :- derived_property_{}(X), base_entity(X).", i, i),
            "performance_test"
        ).await?;
    }
    
    let setup_time = setup_start.elapsed();
    println!("Knowledge base setup took {}ms", setup_time.as_millis());
    
    // Test complex queries
    let complex_queries = vec![
        "complex_conclusion_5(entity_15)?",
        "derived_property_3(X)?", 
        "has_property(X, property_7)?",
        "base_entity(entity_25)?",
    ];
    
    let mut inference_times = Vec::new();
    
    for query in complex_queries {
        let start_time = Instant::now();
        let result = engine.query_with_proof(query).await?;
        let inference_time = start_time.elapsed();
        inference_times.push(inference_time.as_millis());
        
        println!("Complex query '{}' took {}ms (confidence: {:.2})", 
                query, inference_time.as_millis(), result.confidence);
        
        // CONSTRAINT-001: Each query must complete within 100ms
        assert!(inference_time.as_millis() <= PERFORMANCE_CONSTRAINT_MS,
            "Query '{}' took {}ms, exceeding constraint", query, inference_time.as_millis());
    }
    
    let avg_inference_time = inference_times.iter().sum::<u128>() / inference_times.len() as u128;
    let max_inference_time = *inference_times.iter().max().unwrap();
    
    println!("Complex Inference Performance:");
    println!("  Average: {}ms", avg_inference_time);
    println!("  Max: {}ms", max_inference_time);
    
    assert!(max_inference_time <= PERFORMANCE_CONSTRAINT_MS);
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_query_performance() -> Result<()> {
    // Test performance under concurrent load
    let engine = std::sync::Arc::new(DatalogEngine::new().await?);
    
    // Setup knowledge base
    for i in 0..30 {
        let rule = DatalogEngine::compile_requirement_to_rule(
            &format!("Concurrent_data_{} MUST be processed", i)
        ).await?;
        engine.add_rule(rule).await?;
    }
    
    let concurrent_queries = 10;
    let mut handles = Vec::new();
    
    let start_time = Instant::now();
    
    // Launch concurrent queries
    for i in 0..concurrent_queries {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let query = format!("What data_{} must be processed?", i % 5);
            let query_start = Instant::now();
            let result = engine_clone.query(&query).await?;
            let query_time = query_start.elapsed();
            
            anyhow::Result::<(u128, f64)>::Ok((query_time.as_millis(), result.confidence))
        });
        handles.push(handle);
    }
    
    // Collect results
    let mut query_times = Vec::new();
    let mut confidences = Vec::new();
    
    for handle in handles {
        let (query_time, confidence) = handle.await??;
        query_times.push(query_time);
        confidences.push(confidence);
        
        // Each concurrent query must meet constraint
        assert!(query_time <= PERFORMANCE_CONSTRAINT_MS);
    }
    
    let total_time = start_time.elapsed();
    let avg_query_time = query_times.iter().sum::<u128>() / query_times.len() as u128;
    let max_query_time = *query_times.iter().max().unwrap();
    let avg_confidence = confidences.iter().sum::<f64>() / confidences.len() as f64;
    
    println!("Concurrent Query Performance:");
    println!("  Total time: {}ms", total_time.as_millis());
    println!("  Avg query time: {}ms", avg_query_time);
    println!("  Max query time: {}ms", max_query_time);
    println!("  Avg confidence: {:.2}", avg_confidence);
    
    // All concurrent operations should complete quickly
    assert!(max_query_time <= PERFORMANCE_CONSTRAINT_MS);
    assert!(avg_confidence > 0.8);
    
    Ok(())
}

#[tokio::test]
async fn test_memory_usage_under_load() -> Result<()> {
    // Test memory usage doesn't grow excessively under load
    let engine = DatalogEngine::new().await?;
    
    let initial_fact_count = engine.fact_count().await;
    
    // Add substantial load
    let load_iterations = 500;
    let batch_size = 10;
    
    let load_start = Instant::now();
    
    for batch in 0..(load_iterations / batch_size) {
        let batch_start = Instant::now();
        
        // Add batch of rules
        for i in 0..batch_size {
            let idx = batch * batch_size + i;
            let rule = DatalogEngine::compile_requirement_to_rule(
                &format!("Load_entity_{} MUST have_load_property_{}", idx, idx % 20)
            ).await?;
            engine.add_rule(rule).await?;
        }
        
        let batch_time = batch_start.elapsed();
        
        // Periodically test queries to ensure performance doesn't degrade
        if batch % 5 == 0 {
            let query_start = Instant::now();
            let result = engine.query(&format!("What has load_property_{}?", batch % 20)).await?;
            let query_time = query_start.elapsed();
            
            assert!(query_time.as_millis() <= PERFORMANCE_CONSTRAINT_MS,
                "Query performance degraded after {} rules", batch * batch_size);
            assert!(result.confidence > 0.8);
        }
        
        println!("Batch {} ({} rules) completed in {}ms", 
                batch, batch * batch_size, batch_time.as_millis());
    }
    
    let total_load_time = load_start.elapsed();
    let final_fact_count = engine.fact_count().await;
    let final_rule_count = engine.rule_count();
    
    println!("Memory Load Test Results:");
    println!("  Total load time: {}ms", total_load_time.as_millis());
    println!("  Rules added: {}", final_rule_count);
    println!("  Facts added: {}", final_fact_count - initial_fact_count);
    println!("  Avg time per rule: {}ms", total_load_time.as_millis() / load_iterations as u128);
    
    // Final performance check
    let final_query_start = Instant::now();
    let final_result = engine.query("What entities have load properties?").await?;
    let final_query_time = final_query_start.elapsed();
    
    println!("  Final query time: {}ms", final_query_time.as_millis());
    println!("  Final query confidence: {:.2}", final_result.confidence);
    
    // System should still be responsive after heavy load
    assert!(final_query_time.as_millis() <= PERFORMANCE_CONSTRAINT_MS);
    assert!(final_result.confidence > 0.8);
    assert!(!final_result.results.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_logic_parsing_performance() -> Result<()> {
    // Test REAL logic parsing performance
    let parser = LogicParser::new().await?;
    
    let test_requirements = vec![
        "Simple data MUST be encrypted",
        "Payment Card Industry systems MUST implement strong cryptographic protocols to protect Primary Account Numbers during network transmission across all environments",
        "Access to cardholder data environments MUST be restricted to authorized personnel with legitimate business need, authenticated through multi-factor authentication, during designated business hours, with all access logged and monitored in real-time",
        "If cardholder data is stored electronically, then it must be encrypted using strong cryptography with proper key management, otherwise it should be physically secured and access controlled",
        "All merchants that store, process, or transmit cardholder data must implement and maintain a secure network architecture with properly configured firewalls and routers that restrict access to cardholder data environments per section 1.2.1 and requirement REQ-1.1.1",
    ];
    
    let mut parsing_times = Vec::new();
    
    for (i, requirement) in test_requirements.iter().enumerate() {
        let start_time = Instant::now();
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        let parsing_time = start_time.elapsed();
        parsing_times.push(parsing_time.as_millis());
        
        println!("Parsed requirement {} in {}ms (confidence: {:.2})", 
                i + 1, parsing_time.as_millis(), parsed.confidence);
        
        // Each parsing operation must meet constraint
        assert!(parsing_time.as_millis() <= STRICT_CONSTRAINT_MS,
            "Parsing took {}ms, exceeding constraint", parsing_time.as_millis());
        
        // Verify parsing quality
        assert!(parsed.confidence > 0.7);
        assert!(!parsed.subject.is_empty());
        assert!(!parsed.predicate.is_empty());
    }
    
    let avg_parsing_time = parsing_times.iter().sum::<u128>() / parsing_times.len() as u128;
    let max_parsing_time = *parsing_times.iter().max().unwrap();
    
    println!("Logic Parsing Performance Summary:");
    println!("  Average: {}ms", avg_parsing_time);
    println!("  Max: {}ms", max_parsing_time);
    
    assert!(avg_parsing_time <= STRICT_CONSTRAINT_MS);
    assert!(max_parsing_time <= STRICT_CONSTRAINT_MS);
    
    Ok(())
}

#[tokio::test]
async fn test_end_to_end_pipeline_performance() -> Result<()> {
    // Test complete pipeline: Parse -> Compile -> Add -> Query
    let parser = LogicParser::new().await?;
    let engine = DatalogEngine::new().await?;
    
    let requirements = vec![
        "Customer data MUST be encrypted at rest",
        "Payment processing systems SHALL implement access controls", 
        "Audit logs SHOULD be maintained for compliance verification",
    ];
    
    let pipeline_start = Instant::now();
    
    for (i, requirement) in requirements.iter().enumerate() {
        let step_start = Instant::now();
        
        // Step 1: Parse logic
        let parsed = parser.parse_requirement_to_logic(requirement).await?;
        let parse_time = step_start.elapsed();
        
        // Step 2: Compile to Datalog
        let rule = DatalogEngine::compile_requirement_to_rule(requirement).await?;
        let compile_time = step_start.elapsed() - parse_time;
        
        // Step 3: Add to engine
        engine.add_rule(rule).await?;
        let add_time = step_start.elapsed() - parse_time - compile_time;
        
        // Step 4: Query for verification
        let query = format!("What rules relate to step {}?", i + 1);
        let result = engine.query(&query).await?;
        let total_step_time = step_start.elapsed();
        
        println!("Pipeline step {} completed in {}ms:", i + 1, total_step_time.as_millis());
        println!("  Parse: {}ms", parse_time.as_millis());
        println!("  Compile: {}ms", compile_time.as_millis()); 
        println!("  Add: {}ms", add_time.as_millis());
        println!("  Query: {}ms", result.execution_time_ms);
        println!("  Confidence: {:.2}", result.confidence);
        
        // Each step must meet performance constraints
        assert!(total_step_time.as_millis() <= PERFORMANCE_CONSTRAINT_MS);
        assert!(result.execution_time_ms <= PERFORMANCE_CONSTRAINT_MS as u64);
        assert!(result.confidence > 0.8);
    }
    
    let total_pipeline_time = pipeline_start.elapsed();
    println!("Complete pipeline took {}ms", total_pipeline_time.as_millis());
    
    // Entire pipeline should be efficient
    assert!(total_pipeline_time.as_millis() <= PERFORMANCE_CONSTRAINT_MS * 2); // Allow some overhead
    
    Ok(())
}

// Criterion benchmarks for detailed performance profiling
fn datalog_engine_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("datalog_engine_init", |b| {
        b.to_async(&rt).iter(|| async {
            let _engine = DatalogEngine::new().await.unwrap();
        });
    });
    
    let engine = rt.block_on(DatalogEngine::new()).unwrap();
    
    c.bench_function("simple_rule_compilation", |b| {
        b.to_async(&rt).iter(|| async {
            let _rule = DatalogEngine::compile_requirement_to_rule("Test data MUST be encrypted").await.unwrap();
        });
    });
    
    // Add a few rules for query testing
    rt.block_on(async {
        for i in 0..10 {
            let rule = DatalogEngine::compile_requirement_to_rule(
                &format!("Benchmark_data_{} MUST be processed", i)
            ).await.unwrap();
            engine.add_rule(rule).await.unwrap();
        }
    });
    
    c.bench_function("simple_datalog_query", |b| {
        b.to_async(&rt).iter(|| async {
            let _result = engine.query("What benchmark data exists?").await.unwrap();
        });
    });
}

fn prolog_engine_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("prolog_engine_init", |b| {
        b.to_async(&rt).iter(|| async {
            let _engine = PrologEngine::new().await.unwrap();
        });
    });
    
    let engine = rt.block_on(PrologEngine::new()).unwrap();
    
    // Add benchmark facts
    rt.block_on(async {
        engine.add_compliance_rule("benchmark_fact(test_entity).", "benchmark").await.unwrap();
        engine.add_compliance_rule("benchmark_rule(X) :- benchmark_fact(X).", "benchmark").await.unwrap();
    });
    
    c.bench_function("simple_prolog_query", |b| {
        b.to_async(&rt).iter(|| async {
            let _result = engine.query_with_proof("benchmark_rule(test_entity)?").await.unwrap();
        });
    });
}

fn logic_parser_benchmarks(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("logic_parser_init", |b| {
        b.to_async(&rt).iter(|| async {
            let _parser = LogicParser::new().await.unwrap();
        });
    });
    
    let parser = rt.block_on(LogicParser::new()).unwrap();
    
    c.bench_function("simple_requirement_parsing", |b| {
        b.to_async(&rt).iter(|| async {
            let _parsed = parser.parse_requirement_to_logic("Simple data MUST be encrypted").await.unwrap();
        });
    });
    
    c.bench_function("complex_requirement_parsing", |b| {
        b.to_async(&rt).iter(|| async {
            let _parsed = parser.parse_requirement_to_logic(
                "Payment Card Industry systems MUST implement strong cryptographic protocols to protect Primary Account Numbers during network transmission across all environments"
            ).await.unwrap();
        });
    });
}

criterion_group!(benches, datalog_engine_benchmarks, prolog_engine_benchmarks, logic_parser_benchmarks);
criterion_main!(benches);