//! Concurrent message processing benchmarks for MCP adapter

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use mcp_adapter::*;
use tokio::runtime::Runtime;
use std::sync::Arc;
use std::time::Duration;

fn bench_concurrent_message_sending(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let adapter = Arc::new(McpAdapter::new());
    
    let mut group = c.benchmark_group("concurrent_message_sending");
    
    for concurrency in [1, 5, 10, 20, 50].iter() {
        group.throughput(Throughput::Elements(*concurrency as u64));
        group.bench_with_input(
            BenchmarkId::new("send_messages_concurrent", concurrency),
            concurrency,
            |b, &concurrency| {
                b.to_async(&rt).iter(|| async {
                    let messages: Vec<Message> = (0..concurrency)
                        .map(|i| {
                            Message::request(serde_json::json!({
                                "query_id": i,
                                "content": format!("Benchmark query number {}", i),
                                "timestamp": std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap()
                                    .as_millis()
                            }))
                        })
                        .collect();
                    
                    // Since we don't have send_messages_concurrent, simulate concurrent processing
                    let mut results = Vec::new();
                    for message in black_box(messages) {
                        let result = adapter.send_message(message).await;
                        results.push(result);
                    }
                    
                    black_box(results)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_queue_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("queue_concurrent_operations");
    
    for num_producers in [1, 5, 10].iter() {
        for num_consumers in [1, 5, 10].iter() {
            let benchmark_name = format!("{}p_{}c", num_producers, num_consumers);
            group.throughput(Throughput::Elements((num_producers + num_consumers) as u64 * 100));
            
            group.bench_function(&benchmark_name, |b| {
                b.to_async(&rt).iter(|| async {
                    let queue = Arc::new(MessageQueue::new(20000));
                    let mut handles = Vec::new();
                    
                    // Producer tasks
                    for producer_id in 0..*num_producers {
                        let queue_clone = queue.clone();
                        let handle = tokio::spawn(async move {
                            for i in 0..100 {
                                let message = Message::request(serde_json::json!({
                                    "producer": producer_id,
                                    "item": i,
                                    "data": format!("Message from producer {} item {}", producer_id, i)
                                }));
                                queue_clone.enqueue(message).await.unwrap();
                            }
                        });
                        handles.push(handle);
                    }
                    
                    // Consumer tasks
                    for consumer_id in 0..*num_consumers {
                        let queue_clone = queue.clone();
                        let handle = tokio::spawn(async move {
                            for _ in 0..100 {
                                while let Some(message) = queue_clone.dequeue().await {
                                    black_box(message);
                                    break;
                                }
                                // Small delay to prevent busy waiting
                                tokio::time::sleep(Duration::from_nanos(1)).await;
                            }
                        });
                        handles.push(handle);
                    }
                    
                    // Wait for all tasks to complete
                    for handle in handles {
                        handle.await.unwrap();
                    }
                    
                    black_box(queue)
                });
            });
        }
    }
    
    group.finish();
}

fn bench_high_throughput_message_processing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("high_throughput");
    group.sample_size(10); // Fewer samples for high-throughput tests
    
    for message_count in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*message_count as u64));
        group.bench_with_input(
            BenchmarkId::new("process_messages", message_count),
            message_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let queue = MessageQueue::new(count * 2); // Larger queue size
                    
                    // Pre-generate all messages
                    let messages: Vec<Message> = (0..count)
                        .map(|i| {
                            let priority = match i % 4 {
                                0 => MessagePriority::Low,
                                1 => MessagePriority::Normal,
                                2 => MessagePriority::High,
                                3 => MessagePriority::Critical,
                                _ => MessagePriority::Normal,
                            };
                            
                            match priority {
                                MessagePriority::Critical => Message::error("CRITICAL", "critical error")
                                    .with_priority(MessagePriority::Critical),
                                MessagePriority::High => Message::request(serde_json::json!({"high_id": i}))
                                    .with_priority(MessagePriority::High),
                                _ => Message::request(serde_json::json!({"normal_id": i})),
                            }
                        })
                        .collect();
                    
                    // Enqueue all messages
                    for message in messages {
                        queue.enqueue(black_box(message)).await.unwrap();
                    }
                    
                    // Dequeue all messages
                    for _ in 0..count {
                        let message = queue.dequeue().await.unwrap();
                        black_box(message);
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_intensive_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("large_message_handling", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = MessageQueue::new(10000);
            
            // Create messages with large payloads
            for i in 0..100 {
                let large_payload = serde_json::json!({
                    "id": i,
                    "large_text": "A".repeat(10000), // 10KB of text
                    "metadata": {
                        "processing_instructions": "B".repeat(5000), // 5KB more
                        "additional_data": (0..1000).collect::<Vec<i32>>(), // Array of 1000 integers
                    }
                });
                
                let message = Message::request(large_payload);
                
                queue.enqueue(black_box(message)).await.unwrap();
            }
            
            // Process all messages
            for _ in 0..100 {
                let message = queue.dequeue().await.unwrap();
                black_box(message);
            }
            
            black_box(queue)
        });
    });
}

fn bench_adapter_under_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("adapter_stress_test", |b| {
        b.to_async(&rt).iter(|| async {
            let adapter = McpAdapter::new();
            
            // Create many concurrent operations
            let mut handles = Vec::new();
            
            // Authentication operations
            for i in 0..50 {
                let adapter_clone = adapter.clone();
                let handle = tokio::spawn(async move {
                    let creds = Credentials::ApiKey {
                        key: format!("stress-test-key-{}", i),
                        header_name: Some("X-API-Key".to_string()),
                    };
                    
                    let result = adapter_clone.authenticate(creds).await;
                    black_box(result)
                });
                handles.push(handle);
            }
            
            // Metrics collection operations
            for _ in 0..50 {
                let adapter_clone = adapter.clone();
                let handle = tokio::spawn(async move {
                    let stats = adapter_clone.queue_stats();
                    black_box(stats)
                });
                handles.push(handle);
            }
            
            // Health check operations
            for _ in 0..50 {
                let adapter_clone = adapter.clone();
                let handle = tokio::spawn(async move {
                    let connected = adapter_clone.is_connected();
                    black_box(connected)
                });
                handles.push(handle);
            }
            
            // Wait for all operations to complete
            for handle in handles {
                handle.await.unwrap();
            }
            
            // Clean shutdown
            adapter.shutdown().await.unwrap();
            black_box(adapter)
        });
    });
}

fn bench_priority_queue_under_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("priority_queue_stress", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = Arc::new(MessageQueue::new(20000));
            let mut handles = Vec::new();
            
            // Multiple producers with different priorities
            for producer in 0..10 {
                let queue_clone = queue.clone();
                let handle = tokio::spawn(async move {
                    for i in 0..200 {
                        let priority = match (producer + i) % 4 {
                            0 => MessagePriority::Critical,
                            1 => MessagePriority::High,
                            2 => MessagePriority::Normal,
                            _ => MessagePriority::Low,
                        };
                        
                        let message = match priority {
                            MessagePriority::Critical => Message::error("CRITICAL", "critical error")
                                .with_priority(MessagePriority::Critical),
                            MessagePriority::High => Message::request(serde_json::json!({"producer": producer, "item": i}))
                                .with_priority(MessagePriority::High),
                            _ => Message::request(serde_json::json!({"producer": producer, "item": i})),
                        };
                        
                        queue_clone.enqueue(message).await.unwrap();
                    }
                });
                handles.push(handle);
            }
            
            // Multiple consumers
            for consumer in 0..5 {
                let queue_clone = queue.clone();
                let handle = tokio::spawn(async move {
                    for _ in 0..400 { // 10 producers * 200 messages / 5 consumers
                        while let Some(message) = queue_clone.dequeue().await {
                            black_box(message);
                            break;
                        }
                        tokio::time::sleep(Duration::from_nanos(1)).await;
                    }
                });
                handles.push(handle);
            }
            
            // Wait for all tasks
            for handle in handles {
                handle.await.unwrap();
            }
            
            black_box(queue)
        });
    });
}

criterion_group!(
    concurrent_benches,
    bench_concurrent_message_sending,
    bench_queue_concurrent_operations
);

criterion_group!(
    throughput_benches,
    bench_high_throughput_message_processing,
    bench_memory_intensive_operations
);

criterion_group!(
    stress_benches,
    bench_adapter_under_load,
    bench_priority_queue_under_load
);

criterion_main!(concurrent_benches, throughput_benches, stress_benches);