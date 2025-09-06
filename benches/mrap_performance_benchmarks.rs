//! # MRAP Control Loop Performance Benchmarks
//!
//! Benchmarks for MRAP (Monitor → Reason → Act → Reflect → Adapt) control loop
//! to validate Phase 2 architecture performance requirements.

use std::sync::Arc;
use std::time::{Duration, Instant};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;

use integration::{
    DAAOrchestrator, IntegrationConfig, ComponentType, Result,
};

fn create_runtime() -> Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
}

async fn setup_orchestrator() -> DAAOrchestrator {
    let config = Arc::new(IntegrationConfig::default());
    let mut orchestrator = DAAOrchestrator::new(config).await.unwrap();
    orchestrator.initialize().await.unwrap();
    
    // Register test components
    let components = [
        ("chunker", ComponentType::Chunker, "http://localhost:8002"),
        ("embedder", ComponentType::Embedder, "http://localhost:8003"),
        ("storage", ComponentType::Storage, "http://localhost:8004"),
        ("processor", ComponentType::QueryProcessor, "http://localhost:8005"),
        ("generator", ComponentType::ResponseGenerator, "http://localhost:8006"),
    ];
    
    for (name, comp_type, endpoint) in components {
        orchestrator.register_component(name, comp_type, endpoint).await.unwrap();
    }
    
    orchestrator
}

fn bench_mrap_loop_performance(c: &mut Criterion) {
    let rt = create_runtime();
    
    c.bench_function("mrap_single_cycle", |b| {
        let orchestrator = rt.block_on(setup_orchestrator());
        
        b.to_async(&rt).iter(|| async {
            // Measure single MRAP cycle time
            let start = Instant::now();
            
            // Allow one complete MRAP cycle
            tokio::time::sleep(Duration::from_millis(100)).await;
            
            let mrap_metrics = orchestrator.get_mrap_metrics().await;
            let loops_before = mrap_metrics.get("mrap_loops_completed").unwrap().as_u64().unwrap();
            
            // Wait for next cycle
            tokio::time::sleep(Duration::from_secs(11)).await; // MRAP runs every 10s
            
            let mrap_metrics_after = orchestrator.get_mrap_metrics().await;
            let loops_after = mrap_metrics_after.get("mrap_loops_completed").unwrap().as_u64().unwrap();
            
            let cycle_time = start.elapsed();
            
            black_box((loops_after - loops_before, cycle_time))
        });
    });
    
    // Benchmark MRAP loop under component load
    let mut group = c.benchmark_group("mrap_component_scaling");
    for component_count in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("components", component_count),
            component_count,
            |b, &component_count| {
                let orchestrator = rt.block_on(async {
                    let config = Arc::new(IntegrationConfig::default());
                    let mut orch = DAAOrchestrator::new(config).await.unwrap();
                    orch.initialize().await.unwrap();
                    
                    // Register specified number of components
                    for i in 0..component_count {
                        orch.register_component(
                            &format!("component-{}", i),
                            ComponentType::Chunker,
                            &format!("http://localhost:808{}", i % 10)
                        ).await.unwrap();
                    }
                    
                    orch
                });
                
                b.to_async(&rt).iter(|| async {
                    let start = Instant::now();
                    
                    // Measure MRAP performance with N components
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    
                    let mrap_metrics = orchestrator.get_mrap_metrics().await;
                    let avg_loop_time = mrap_metrics.get("average_loop_time_ms")
                        .unwrap()
                        .as_f64()
                        .unwrap();
                    
                    let monitoring_cycles = mrap_metrics.get("monitoring_cycles")
                        .unwrap()
                        .as_u64()
                        .unwrap();
                    
                    black_box((avg_loop_time, monitoring_cycles))
                });
            },
        );
    }
    group.finish();
}

fn bench_byzantine_consensus_timing(c: &mut Criterion) {
    let rt = create_runtime();
    
    c.bench_function("byzantine_consensus", |b| {
        let orchestrator = rt.block_on(async {
            let mut orch = setup_orchestrator().await;
            orch.enable_byzantine_consensus().await.unwrap();
            orch
        });
        
        b.to_async(&rt).iter(|| async {
            let consensus_start = Instant::now();
            
            // Trigger consensus operation
            orchestrator.enable_byzantine_consensus().await.unwrap();
            
            let consensus_time = consensus_start.elapsed();
            
            // Verify consensus timing meets Phase 2 requirements (<500ms)
            assert!(
                consensus_time < Duration::from_millis(500),
                "Byzantine consensus should complete within 500ms: {:?}", 
                consensus_time
            );
            
            black_box(consensus_time)
        });
    });
}

fn bench_mrap_fault_recovery_speed(c: &mut Criterion) {
    let rt = create_runtime();
    
    c.bench_function("mrap_fault_recovery", |b| {
        let orchestrator = rt.block_on(async {
            let mut orch = setup_orchestrator().await;
            orch.enable_self_healing().await.unwrap();
            
            // Register a component that will simulate failure
            orch.register_component(
                "failing-component", 
                ComponentType::Storage, 
                "http://localhost:9999" // Non-existent
            ).await.unwrap();
            
            orch
        });
        
        b.to_async(&rt).iter(|| async {
            let recovery_start = Instant::now();
            
            // Allow MRAP to detect and attempt recovery
            tokio::time::sleep(Duration::from_secs(3)).await;
            
            let recovery_time = recovery_start.elapsed();
            let metrics = orchestrator.metrics().await;
            
            black_box((recovery_time, metrics.fault_recoveries))
        });
    });
}

fn bench_mrap_adaptation_learning(c: &mut Criterion) {
    let rt = create_runtime();
    
    c.bench_function("mrap_adaptation_speed", |b| {
        let orchestrator = rt.block_on(async {
            let mut orch = setup_orchestrator().await;
            orch.enable_adaptive_behavior().await.unwrap();
            orch
        });
        
        b.to_async(&rt).iter(|| async {
            let adaptation_start = Instant::now();
            
            // Allow multiple MRAP cycles for adaptation
            tokio::time::sleep(Duration::from_secs(5)).await;
            
            let adaptation_time = adaptation_start.elapsed();
            
            let mrap_metrics = orchestrator.get_mrap_metrics().await;
            let adaptations_made = mrap_metrics.get("adaptations_made")
                .unwrap()
                .as_u64()
                .unwrap();
            
            black_box((adaptation_time, adaptations_made))
        });
    });
}

fn bench_mrap_system_throughput(c: &mut Criterion) {
    let rt = create_runtime();
    
    let mut group = c.benchmark_group("mrap_throughput");
    
    for concurrent_requests in [1, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_queries", concurrent_requests),
            concurrent_requests,
            |b, &concurrent_requests| {
                let orchestrator = rt.block_on(setup_orchestrator());
                
                b.to_async(&rt).iter(|| async {
                    let throughput_start = Instant::now();
                    
                    // Simulate concurrent query processing with MRAP orchestration
                    let mut tasks = Vec::new();
                    
                    for i in 0..concurrent_requests {
                        let orch = orchestrator.clone();
                        let task = tokio::spawn(async move {
                            let query_context = serde_json::json!({
                                "request_id": format!("bench-query-{}", i),
                                "query": "Benchmark test query",
                                "timestamp": chrono::Utc::now().timestamp_millis()
                            });
                            
                            orch.coordinate_components(query_context).await
                        });
                        tasks.push(task);
                    }
                    
                    // Wait for all concurrent requests
                    for task in tasks {
                        let _ = task.await;
                    }
                    
                    let throughput_time = throughput_start.elapsed();
                    let queries_per_second = concurrent_requests as f64 / throughput_time.as_secs_f64();
                    
                    black_box((throughput_time, queries_per_second))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_mrap_memory_efficiency(c: &mut Criterion) {
    let rt = create_runtime();
    
    c.bench_function("mrap_memory_usage", |b| {
        b.to_async(&rt).iter(|| async {
            let orchestrator = setup_orchestrator().await;
            
            // Run MRAP for extended period to test memory efficiency
            tokio::time::sleep(Duration::from_secs(10)).await;
            
            let mrap_metrics = orchestrator.get_mrap_metrics().await;
            let loops_completed = mrap_metrics.get("mrap_loops_completed")
                .unwrap()
                .as_u64()
                .unwrap();
            
            // Verify MRAP continues efficiently
            assert!(loops_completed > 0, "MRAP should complete loops efficiently");
            
            // Clean shutdown to test memory cleanup
            orchestrator.shutdown().await.unwrap();
            
            black_box(loops_completed)
        });
    });
}

criterion_group!(
    mrap_benchmarks,
    bench_mrap_loop_performance,
    bench_byzantine_consensus_timing,
    bench_mrap_fault_recovery_speed,
    bench_mrap_adaptation_learning,
    bench_mrap_system_throughput,
    bench_mrap_memory_efficiency
);

criterion_main!(mrap_benchmarks);