//! Latency benchmarks for MCP adapter message processing

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use mcp_adapter::*;
use tokio::runtime::Runtime;
use std::time::Duration;

fn bench_message_creation(c: &mut Criterion) {
    c.bench_function("message_creation", |b| {
        b.iter(|| {
            let message = Message::request(black_box(serde_json::json!({"test": "benchmark data"})));
            black_box(message)
        })
    });
}

fn bench_message_serialization(c: &mut Criterion) {
    let message = Message::request(serde_json::json!({
        "document_id": "doc-12345",
        "content": "This is a test document for benchmarking serialization performance",
        "metadata": {
            "size": 1024,
            "type": "text/plain",
            "encoding": "utf-8"
        }
    }));

    c.bench_function("message_serialization", |b| {
        b.iter(|| {
            let serialized = serde_json::to_string(&black_box(&message)).unwrap();
            black_box(serialized)
        })
    });
}

fn bench_message_deserialization(c: &mut Criterion) {
    let message = Message::request(serde_json::json!({
        "document_id": "doc-12345",
        "content": "This is a test document for benchmarking deserialization performance",
        "metadata": {
            "size": 1024,
            "type": "text/plain",
            "encoding": "utf-8"
        }
    }));
    
    let serialized = serde_json::to_string(&message).unwrap();

    c.bench_function("message_deserialization", |b| {
        b.iter(|| {
            let deserialized: Message = serde_json::from_str(&black_box(&serialized)).unwrap();
            black_box(deserialized)
        })
    });
}

fn bench_queue_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("queue_operations");
    
    for message_count in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("enqueue", message_count),
            message_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let queue = MessageQueue::new(2000);
                    
                    for i in 0..count {
                        let message = Message::request(serde_json::json!({"id": i}));
                        queue.enqueue(black_box(message)).await.unwrap();
                    }
                    
                    black_box(queue)
                });
            },
        );
    }
    
    for message_count in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("dequeue", message_count),
            message_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let queue = MessageQueue::new(2000);
                    
                    // Pre-populate queue
                    for i in 0..count {
                        let message = Message::request(serde_json::json!({"id": i}));
                        queue.enqueue(message).await.unwrap();
                    }
                    
                    // Benchmark dequeue operations
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

fn bench_priority_queue_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("priority_queue_mixed_operations", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = MessageQueue::new(200);
            
            // Add messages with different priorities
            for i in 0..100 {
                let priority = match i % 4 {
                    0 => MessagePriority::Low,
                    1 => MessagePriority::Normal,
                    2 => MessagePriority::High,
                    3 => MessagePriority::Critical,
                    _ => MessagePriority::Normal,
                };
                
                let message = match priority {
                    MessagePriority::Low => Message::new(MessageType::Request).with_priority(MessagePriority::Low),
                    MessagePriority::Normal => Message::new(MessageType::Request).with_priority(MessagePriority::Normal),
                    MessagePriority::High => Message::new(MessageType::Request).with_priority(MessagePriority::High),
                    MessagePriority::Critical => Message::new(MessageType::Request).with_priority(MessagePriority::Critical),
                };
                
                queue.enqueue(black_box(message)).await.unwrap();
            }
            
            // Dequeue all messages (should come out in priority order)
            for _ in 0..100 {
                let message = queue.dequeue().await.unwrap();
                black_box(message);
            }
        });
    });
}

fn bench_authentication_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let adapter = McpAdapter::new();
    
    let mut group = c.benchmark_group("authentication");
    
    group.bench_function("api_key_auth", |b| {
        b.to_async(&rt).iter(|| async {
            let creds = Credentials::ApiKey {
                key: black_box("benchmark-api-key-12345".to_string()),
                header_name: Some("X-API-Key".to_string()),
            };
            
            let result = adapter.authenticate(black_box(creds)).await;
            black_box(result)
        });
    });
    
    group.bench_function("jwt_auth", |b| {
        b.to_async(&rt).iter(|| async {
            let creds = Credentials::Jwt {
                token: black_box("eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIiwiZXhwIjoxNjQwOTk1MjAwfQ.signature".to_string()),
            };
            
            let result = adapter.authenticate(black_box(creds)).await;
            black_box(result)
        });
    });
    
    group.bench_function("basic_auth", |b| {
        b.to_async(&rt).iter(|| async {
            let creds = Credentials::Basic {
                username: black_box("benchmark_user".to_string()),
                password: black_box("benchmark_pass".to_string()),
            };
            
            let result = adapter.authenticate(black_box(creds)).await;
            black_box(result)
        });
    });
    
    group.finish();
}

fn bench_adapter_metrics(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let adapter = McpAdapter::new();
    
    c.bench_function("get_queue_stats", |b| {
        b.to_async(&rt).iter(|| async {
            let stats = adapter.queue_stats();
            black_box(stats)
        });
    });
    
    c.bench_function("connection_check", |b| {
        b.to_async(&rt).iter(|| async {
            let connected = adapter.is_connected();
            black_box(connected)
        });
    });
}

fn bench_response_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("response_creation");
    
    group.bench_function("success_response", |b| {
        b.iter(|| {
            let response = Response::success(
                black_box(123),
                black_box(serde_json::json!({
                    "result": "operation completed successfully",
                    "data": {
                        "processed_items": 1000,
                        "processing_time_ms": 150,
                        "status": "ok"
                    }
                })),
            );
            black_box(response)
        })
    });
    
    group.bench_function("error_response", |b| {
        b.iter(|| {
            let error = ResponseError::new(
                "VALIDATION_ERROR".to_string(),
                "Invalid input parameters".to_string(),
            );
            let response = Response::error(
                black_box(456),
                black_box(error),
            );
            black_box(response)
        })
    });
    
    group.finish();
}

fn bench_message_with_headers_and_metadata(c: &mut Criterion) {
    c.bench_function("message_with_full_metadata", |b| {
        b.iter(|| {
            let message = Message::request(
                black_box(serde_json::json!({
                    "document_id": "doc-12345",
                    "content_length": 10240,
                    "format": "pdf"
                }))
            )
            .with_correlation_id(black_box(uuid::Uuid::new_v4()))
            .with_ttl(black_box(600000))
            .with_header(black_box("X-Processing-Priority".to_string()), black_box("high".to_string()))
            .with_header(black_box("X-Content-Type".to_string()), black_box("application/pdf".to_string()));
            
            black_box(message)
        })
    });
}

fn bench_concurrent_queue_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("concurrent_enqueue_dequeue", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = MessageQueue::new(200);
            let queue = std::sync::Arc::new(queue);
            
            // Spawn concurrent tasks
            let mut handles = Vec::new();
            
            // Enqueue tasks
            for i in 0..50 {
                let queue_clone = queue.clone();
                let handle = tokio::spawn(async move {
                    let message = Message::request(
                        serde_json::json!({"task": i})
                    );
                    queue_clone.enqueue(message).await.unwrap();
                });
                handles.push(handle);
            }
            
            // Dequeue tasks
            for _ in 0..50 {
                let queue_clone = queue.clone();
                let handle = tokio::spawn(async move {
                    // Wait a bit to ensure some messages are enqueued
                    tokio::time::sleep(Duration::from_micros(100)).await;
                    if let Some(message) = queue_clone.dequeue().await {
                        black_box(message);
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

criterion_group!(
    message_benches,
    bench_message_creation,
    bench_message_serialization,
    bench_message_deserialization,
    bench_message_with_headers_and_metadata
);

criterion_group!(
    queue_benches,
    bench_queue_operations,
    bench_priority_queue_performance,
    bench_concurrent_queue_operations
);

criterion_group!(
    adapter_benches,
    bench_authentication_performance,
    bench_adapter_metrics
);

criterion_group!(
    response_benches,
    bench_response_creation
);

criterion_main!(message_benches, queue_benches, adapter_benches, response_benches);