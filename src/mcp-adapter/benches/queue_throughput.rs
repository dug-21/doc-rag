use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use mcp_adapter::{
    message::{Message, MessageType, MessagePriority},
    queue::{MessageQueue, MultiQueue},
};
use std::sync::Arc;
use tokio::runtime::Runtime;

fn bench_queue_enqueue_dequeue(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("queue_basic_ops");
    
    for capacity in [100, 1000, 10000].iter() {
        let queue = MessageQueue::new(*capacity);
        
        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("enqueue_dequeue", capacity),
            &queue,
            |b, queue| {
                b.to_async(&rt).iter(|| async {
                    // Enqueue 100 messages
                    for i in 0..100 {
                        let msg = Message::request(serde_json::json!({"id": i}));
                        queue.enqueue(msg).await.unwrap();
                    }
                    
                    // Dequeue 100 messages
                    for _ in 0..100 {
                        let _ = queue.try_dequeue();
                    }
                    
                    black_box(queue.size())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_priority_queue_ordering(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("priority_queue");
    
    for message_count in [100, 1000, 5000].iter() {
        let queue = MessageQueue::new(*message_count * 2);
        
        group.throughput(Throughput::Elements(*message_count as u64));
        group.bench_with_input(
            BenchmarkId::new("priority_ordering", message_count),
            &queue,
            |b, queue| {
                b.to_async(&rt).iter(|| async {
                    // Enqueue messages with different priorities
                    for i in 0..*message_count {
                        let priority = match i % 4 {
                            0 => MessagePriority::Low,
                            1 => MessagePriority::Normal,
                            2 => MessagePriority::High,
                            3 => MessagePriority::Critical,
                            _ => MessagePriority::Normal,
                        };
                        
                        let msg = Message::request(serde_json::json!({"id": i}))
                            .with_priority(priority);
                        queue.enqueue(msg).await.unwrap();
                    }
                    
                    // Dequeue all messages (should come out in priority order)
                    let mut dequeued = Vec::new();
                    while let Some(msg) = queue.try_dequeue() {
                        dequeued.push(msg);
                    }
                    
                    black_box(dequeued.len())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_concurrent_queue_access(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("concurrent_queue_access");
    
    for concurrency in [10, 50, 100].iter() {
        let queue = Arc::new(MessageQueue::new(10000));
        
        group.throughput(Throughput::Elements(*concurrency as u64 * 100));
        group.bench_with_input(
            BenchmarkId::new("concurrent_producers_consumers", concurrency),
            &queue,
            |b, queue| {
                b.to_async(&rt).iter(|| async {
                    let queue = Arc::clone(queue);
                    
                    // Spawn concurrent producers
                    let producers: Vec<_> = (0..*concurrency)
                        .map(|i| {
                            let queue = Arc::clone(&queue);
                            tokio::spawn(async move {
                                for j in 0..100 {
                                    let msg = Message::request(serde_json::json!({
                                        "producer": i,
                                        "message": j
                                    }));
                                    let _ = queue.enqueue(msg).await;
                                }
                            })
                        })
                        .collect();
                    
                    // Spawn concurrent consumers
                    let consumers: Vec<_> = (0..*concurrency)
                        .map(|_| {
                            let queue = Arc::clone(&queue);
                            tokio::spawn(async move {
                                let mut consumed = 0;
                                for _ in 0..100 {
                                    if queue.try_dequeue().is_some() {
                                        consumed += 1;
                                    }
                                    tokio::task::yield_now().await;
                                }
                                consumed
                            })
                        })
                        .collect();
                    
                    // Wait for all tasks to complete
                    let _: Vec<_> = futures::future::join_all(producers).await;
                    let consumption_counts: Vec<_> = futures::future::join_all(consumers).await;
                    
                    let total_consumed: i32 = consumption_counts.into_iter().map(|r| r.unwrap()).sum();
                    black_box(total_consumed)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("batch_operations");
    
    for batch_size in [10, 50, 100, 200].iter() {
        let queue = MessageQueue::new(10000);
        
        group.throughput(Throughput::Elements(*batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_dequeue", batch_size),
            &queue,
            |b, queue| {
                b.to_async(&rt).iter(|| async {
                    // Pre-fill queue with messages
                    for i in 0..(*batch_size * 2) {
                        let msg = Message::request(serde_json::json!({"id": i}));
                        queue.enqueue(msg).await.unwrap();
                    }
                    
                    // Benchmark batch dequeue
                    let batch = queue.dequeue_batch(*batch_size).await;
                    black_box(batch.len())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_queue_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("queue_memory");
    
    for message_size in ["small", "medium", "large"].iter() {
        let queue = MessageQueue::new(10000);
        
        group.bench_with_input(
            BenchmarkId::new("memory_efficiency", message_size),
            &queue,
            |b, queue| {
                b.to_async(&rt).iter(|| async {
                    let payload = match *message_size {
                        "small" => serde_json::json!({"data": "x".repeat(100)}),
                        "medium" => serde_json::json!({"data": "x".repeat(1000)}),
                        "large" => serde_json::json!({"data": "x".repeat(10000)}),
                        _ => serde_json::json!({}),
                    };
                    
                    // Fill queue with messages of different sizes
                    for i in 0..1000 {
                        let mut msg = Message::request(payload.clone());
                        msg.headers.insert("index".to_string(), i.to_string());
                        queue.enqueue(msg).await.unwrap();
                    }
                    
                    // Measure queue stats
                    let stats = queue.stats();
                    black_box(stats)
                });
            },
        );
    }
    
    group.finish();
}

fn bench_multi_queue_routing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("multi_queue_routing");
    
    for queue_count in [5, 10, 20].iter() {
        let mut multi_queue = MultiQueue::new(1000);
        
        // Create multiple named queues
        for i in 0..*queue_count {
            let queue_name = format!("queue_{}", i);
            let queue = Arc::new(MessageQueue::new(1000));
            multi_queue.add_queue(queue_name, queue);
        }
        
        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("route_messages", queue_count),
            &multi_queue,
            |b, multi_queue| {
                b.to_async(&rt).iter(|| async {
                    // Route 1000 messages to different queues
                    for i in 0..1000 {
                        let queue_name = format!("queue_{}", i % queue_count);
                        let mut msg = Message::request(serde_json::json!({"id": i}));
                        msg.headers.insert("queue_name".to_string(), queue_name);
                        
                        multi_queue.route_message(msg).await.unwrap();
                    }
                    
                    // Check queue stats
                    let stats = multi_queue.all_stats();
                    black_box(stats.len())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_queue_cleanup_expired(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    c.bench_function("cleanup_expired_messages", |b| {
        b.to_async(&rt).iter(|| async {
            let queue = MessageQueue::new(10000);
            
            // Add messages with different TTLs
            for i in 0..1000 {
                let mut msg = Message::request(serde_json::json!({"id": i}));
                // Some messages expire quickly, others don't
                if i % 3 == 0 {
                    msg = msg.with_ttl(1); // 1ms TTL - will expire quickly
                } else {
                    msg = msg.with_ttl(60000); // 60s TTL - won't expire
                }
                queue.enqueue(msg).await.unwrap();
            }
            
            // Wait for some messages to expire
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            
            // Benchmark cleanup operation
            let expired_count = queue.cleanup_expired().await;
            black_box(expired_count)
        });
    });
}

criterion_group!(
    queue_benches,
    bench_queue_enqueue_dequeue,
    bench_priority_queue_ordering,
    bench_concurrent_queue_access,
    bench_batch_operations,
    bench_queue_memory_usage,
    bench_multi_queue_routing,
    bench_queue_cleanup_expired
);

criterion_main!(queue_benches);