use crate::{Message, MessagePriority, Result, McpError};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, VecDeque};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::Instant;
use tracing::debug;

/// Priority wrapper for messages in the queue
#[derive(Debug)]
struct PriorityMessage {
    message: Message,
    enqueued_at: Instant,
}

impl PartialEq for PriorityMessage {
    fn eq(&self, other: &Self) -> bool {
        self.message.priority == other.message.priority
    }
}

impl Eq for PriorityMessage {}

impl PartialOrd for PriorityMessage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityMessage {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority messages come first, then by enqueue time (FIFO)
        match self.message.priority.cmp(&other.message.priority) {
            Ordering::Equal => other.enqueued_at.cmp(&self.enqueued_at), // FIFO for same priority
            other => other, // Higher priority first
        }
    }
}

/// Concurrent message queue with priority handling
pub struct MessageQueue {
    priority_queue: Arc<RwLock<BinaryHeap<PriorityMessage>>>,
    fifo_queue: Arc<RwLock<VecDeque<Message>>>,
    capacity: usize,
    current_size: AtomicUsize,
    total_enqueued: AtomicU64,
    total_dequeued: AtomicU64,
    total_dropped: AtomicU64,
    notify: Arc<Notify>,
    use_priority_queue: bool,
}

impl MessageQueue {
    /// Create a new message queue with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            priority_queue: Arc::new(RwLock::new(BinaryHeap::with_capacity(capacity))),
            fifo_queue: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
            capacity,
            current_size: AtomicUsize::new(0),
            total_enqueued: AtomicU64::new(0),
            total_dequeued: AtomicU64::new(0),
            total_dropped: AtomicU64::new(0),
            notify: Arc::new(Notify::new()),
            use_priority_queue: true,
        }
    }

    /// Create a FIFO-only queue (no priority handling)
    pub fn new_fifo(capacity: usize) -> Self {
        let mut queue = Self::new(capacity);
        queue.use_priority_queue = false;
        queue
    }

    /// Enqueue a message
    pub async fn enqueue(&self, message: Message) -> Result<()> {
        // Check if message is expired
        if message.is_expired() {
            self.total_dropped.fetch_add(1, AtomicOrdering::Relaxed);
            return Err(McpError::Internal("Message expired before enqueuing".to_string()));
        }

        // Check capacity
        let current_size = self.current_size.load(AtomicOrdering::Relaxed);
        if current_size >= self.capacity {
            // Try to drop expired messages first
            self.cleanup_expired().await;
            
            let size_after_cleanup = self.current_size.load(AtomicOrdering::Relaxed);
            if size_after_cleanup >= self.capacity {
                self.total_dropped.fetch_add(1, AtomicOrdering::Relaxed);
                return Err(McpError::Internal("Queue is full".to_string()));
            }
        }

        // Enqueue based on queue type
        if self.use_priority_queue {
            let priority_msg = PriorityMessage {
                message,
                enqueued_at: Instant::now(),
            };
            self.priority_queue.write().push(priority_msg);
        } else {
            self.fifo_queue.write().push_back(message);
        }

        self.current_size.fetch_add(1, AtomicOrdering::Relaxed);
        self.total_enqueued.fetch_add(1, AtomicOrdering::Relaxed);
        
        // Notify waiting consumers
        self.notify.notify_one();
        
        Ok(())
    }

    /// Dequeue a single message
    pub async fn dequeue(&self) -> Option<Message> {
        loop {
            // Try to get a message
            let message = if self.use_priority_queue {
                self.priority_queue.write().pop().map(|pm| pm.message)
            } else {
                self.fifo_queue.write().pop_front()
            };

            match message {
                Some(msg) => {
                    // Check if message is expired
                    if msg.is_expired() {
                        self.total_dropped.fetch_add(1, AtomicOrdering::Relaxed);
                        self.current_size.fetch_sub(1, AtomicOrdering::Relaxed);
                        continue; // Try next message
                    }

                    self.current_size.fetch_sub(1, AtomicOrdering::Relaxed);
                    self.total_dequeued.fetch_add(1, AtomicOrdering::Relaxed);
                    return Some(msg);
                }
                None => {
                    // Queue is empty, wait for notification
                    self.notify.notified().await;
                    // Return None if still empty after notification
                    if self.is_empty() {
                        return None;
                    }
                    // Continue loop to try again
                }
            }
        }
    }

    /// Dequeue multiple messages up to batch_size
    pub async fn dequeue_batch(&self, batch_size: usize) -> Vec<Message> {
        let mut batch = Vec::with_capacity(batch_size);
        
        // Get as many messages as possible without waiting
        for _ in 0..batch_size {
            let message = if self.use_priority_queue {
                self.priority_queue.write().pop().map(|pm| pm.message)
            } else {
                self.fifo_queue.write().pop_front()
            };

            match message {
                Some(msg) => {
                    if msg.is_expired() {
                        self.total_dropped.fetch_add(1, AtomicOrdering::Relaxed);
                        self.current_size.fetch_sub(1, AtomicOrdering::Relaxed);
                        continue;
                    }
                    
                    batch.push(msg);
                    self.current_size.fetch_sub(1, AtomicOrdering::Relaxed);
                    self.total_dequeued.fetch_add(1, AtomicOrdering::Relaxed);
                }
                None => break, // No more messages available
            }
        }

        batch
    }

    /// Try to dequeue a message without waiting
    pub fn try_dequeue(&self) -> Option<Message> {
        let message = if self.use_priority_queue {
            self.priority_queue.write().pop().map(|pm| pm.message)
        } else {
            self.fifo_queue.write().pop_front()
        };

        match message {
            Some(msg) => {
                if msg.is_expired() {
                    self.total_dropped.fetch_add(1, AtomicOrdering::Relaxed);
                    self.current_size.fetch_sub(1, AtomicOrdering::Relaxed);
                    return None;
                }

                self.current_size.fetch_sub(1, AtomicOrdering::Relaxed);
                self.total_dequeued.fetch_add(1, AtomicOrdering::Relaxed);
                Some(msg)
            }
            None => None,
        }
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.current_size.load(AtomicOrdering::Relaxed) == 0
    }

    /// Check if queue is full
    pub fn is_full(&self) -> bool {
        self.current_size.load(AtomicOrdering::Relaxed) >= self.capacity
    }

    /// Get current queue size
    pub fn size(&self) -> usize {
        self.current_size.load(AtomicOrdering::Relaxed)
    }

    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get queue statistics
    pub fn stats(&self) -> QueueStats {
        QueueStats {
            current_size: self.current_size.load(AtomicOrdering::Relaxed),
            capacity: self.capacity,
            total_enqueued: self.total_enqueued.load(AtomicOrdering::Relaxed),
            total_dequeued: self.total_dequeued.load(AtomicOrdering::Relaxed),
            total_dropped: self.total_dropped.load(AtomicOrdering::Relaxed),
            utilization_percent: (self.current_size.load(AtomicOrdering::Relaxed) as f64 / self.capacity as f64 * 100.0) as u8,
        }
    }

    /// Clear all messages from the queue
    pub fn clear(&self) {
        if self.use_priority_queue {
            self.priority_queue.write().clear();
        } else {
            self.fifo_queue.write().clear();
        }
        
        let cleared_count = self.current_size.swap(0, AtomicOrdering::Relaxed);
        self.total_dropped.fetch_add(cleared_count as u64, AtomicOrdering::Relaxed);
    }

    /// Cleanup expired messages
    pub async fn cleanup_expired(&self) -> usize {
        let mut expired_count = 0;

        if self.use_priority_queue {
            let mut queue = self.priority_queue.write();
            let original_size = queue.len();
            
            // Rebuild heap without expired messages
            let valid_messages: BinaryHeap<PriorityMessage> = queue
                .drain()
                .filter(|pm| {
                    if pm.message.is_expired() {
                        expired_count += 1;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            
            *queue = valid_messages;
        } else {
            let mut queue = self.fifo_queue.write();
            let original_size = queue.len();
            
            // Keep only non-expired messages
            let valid_messages: VecDeque<Message> = queue
                .drain(..)
                .filter(|msg| {
                    if msg.is_expired() {
                        expired_count += 1;
                        false
                    } else {
                        true
                    }
                })
                .collect();
            
            *queue = valid_messages;
        }

        // Update counters
        self.current_size.fetch_sub(expired_count, AtomicOrdering::Relaxed);
        self.total_dropped.fetch_add(expired_count as u64, AtomicOrdering::Relaxed);

        if expired_count > 0 {
            debug!("Cleaned up {} expired messages from queue", expired_count);
        }

        expired_count
    }

    /// Get messages by priority (for monitoring/debugging)
    pub fn get_priority_distribution(&self) -> PriorityDistribution {
        if !self.use_priority_queue {
            return PriorityDistribution::default();
        }

        let queue = self.priority_queue.read();
        let mut critical = 0;
        let mut high = 0;
        let mut normal = 0;
        let mut low = 0;

        for pm in queue.iter() {
            match pm.message.priority {
                MessagePriority::Critical => critical += 1,
                MessagePriority::High => high += 1,
                MessagePriority::Normal => normal += 1,
                MessagePriority::Low => low += 1,
            }
        }

        PriorityDistribution {
            critical,
            high,
            normal,
            low,
        }
    }

    /// Peek at the next message without dequeuing it
    pub fn peek(&self) -> Option<Message> {
        if self.use_priority_queue {
            self.priority_queue.read().peek().map(|pm| pm.message.clone())
        } else {
            self.fifo_queue.read().front().cloned()
        }
    }
}

/// Queue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStats {
    pub current_size: usize,
    pub capacity: usize,
    pub total_enqueued: u64,
    pub total_dequeued: u64,
    pub total_dropped: u64,
    pub utilization_percent: u8,
}

/// Priority distribution in queue
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PriorityDistribution {
    pub critical: usize,
    pub high: usize,
    pub normal: usize,
    pub low: usize,
}

/// Multi-queue manager for handling different message types
pub struct MultiQueue {
    queues: std::collections::HashMap<String, Arc<MessageQueue>>,
    default_queue: Arc<MessageQueue>,
}

impl MultiQueue {
    /// Create a new multi-queue manager
    pub fn new(default_capacity: usize) -> Self {
        Self {
            queues: std::collections::HashMap::new(),
            default_queue: Arc::new(MessageQueue::new(default_capacity)),
        }
    }

    /// Add a named queue
    pub fn add_queue(&mut self, name: String, queue: Arc<MessageQueue>) {
        self.queues.insert(name, queue);
    }

    /// Get or create a queue by name
    pub fn get_or_create_queue(&mut self, name: &str, capacity: usize) -> Arc<MessageQueue> {
        self.queues
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(MessageQueue::new(capacity)))
            .clone()
    }

    /// Route message to appropriate queue based on message type or headers
    pub async fn route_message(&self, message: Message) -> Result<()> {
        let queue_name = message.headers.get("queue_name")
            .map(|s| s.as_str())
            .unwrap_or("default");

        let queue = if queue_name == "default" {
            &self.default_queue
        } else {
            self.queues.get(queue_name).unwrap_or(&self.default_queue)
        };

        queue.enqueue(message).await
    }

    /// Get queue by name
    pub fn get_queue(&self, name: &str) -> Option<Arc<MessageQueue>> {
        if name == "default" {
            Some(self.default_queue.clone())
        } else {
            self.queues.get(name).cloned()
        }
    }

    /// Get all queue statistics
    pub fn all_stats(&self) -> std::collections::HashMap<String, QueueStats> {
        let mut stats = std::collections::HashMap::new();
        
        stats.insert("default".to_string(), self.default_queue.stats());
        
        for (name, queue) in &self.queues {
            stats.insert(name.clone(), queue.stats());
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Message, MessageType};
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_basic_enqueue_dequeue() {
        let queue = MessageQueue::new(10);
        let msg = Message::request(serde_json::json!({"test": "data"}));
        
        queue.enqueue(msg.clone()).await.unwrap();
        assert_eq!(queue.size(), 1);
        
        let dequeued = queue.try_dequeue().unwrap();
        assert_eq!(dequeued.payload, msg.payload);
        assert_eq!(queue.size(), 0);
    }

    #[tokio::test]
    async fn test_priority_ordering() {
        let queue = MessageQueue::new(10);
        
        let low_msg = Message::new(MessageType::Request).with_priority(MessagePriority::Low);
        let high_msg = Message::new(MessageType::Request).with_priority(MessagePriority::High);
        let critical_msg = Message::new(MessageType::Request).with_priority(MessagePriority::Critical);
        
        queue.enqueue(low_msg).await.unwrap();
        queue.enqueue(high_msg).await.unwrap();
        queue.enqueue(critical_msg).await.unwrap();
        
        // Should dequeue in priority order: Critical, High, Low
        let first = queue.try_dequeue().unwrap();
        assert_eq!(first.priority, MessagePriority::Critical);
        
        let second = queue.try_dequeue().unwrap();
        assert_eq!(second.priority, MessagePriority::High);
        
        let third = queue.try_dequeue().unwrap();
        assert_eq!(third.priority, MessagePriority::Low);
    }

    #[tokio::test]
    async fn test_capacity_limit() {
        let queue = MessageQueue::new(2);
        let msg1 = Message::new(MessageType::Request);
        let msg2 = Message::new(MessageType::Request);
        let msg3 = Message::new(MessageType::Request);
        
        assert!(queue.enqueue(msg1).await.is_ok());
        assert!(queue.enqueue(msg2).await.is_ok());
        assert!(queue.enqueue(msg3).await.is_err()); // Should fail when full
    }

    #[tokio::test]
    async fn test_expired_message_cleanup() {
        let queue = MessageQueue::new(10);
        let mut expired_msg = Message::new(MessageType::Request);
        expired_msg.timestamp = Some(chrono::Utc::now() - chrono::Duration::seconds(10));
        expired_msg.ttl_ms = Some(1000); // 1 second TTL, so it's expired
        
        let valid_msg = Message::new(MessageType::Request);
        
        queue.enqueue(expired_msg).await.expect_err("Should reject expired message");
        queue.enqueue(valid_msg.clone()).await.unwrap();
        
        let dequeued = queue.try_dequeue().unwrap();
        assert_eq!(dequeued.message_type, valid_msg.message_type);
    }

    #[tokio::test]
    async fn test_batch_dequeue() {
        let queue = MessageQueue::new(10);
        
        for i in 0..5 {
            let msg = Message::request(serde_json::json!({"id": i}));
            queue.enqueue(msg).await.unwrap();
        }
        
        let batch = queue.dequeue_batch(3).await;
        assert_eq!(batch.len(), 3);
        assert_eq!(queue.size(), 2);
    }

    #[tokio::test]
    async fn test_queue_stats() {
        let queue = MessageQueue::new(10);
        let msg = Message::new(MessageType::Request);
        
        queue.enqueue(msg).await.unwrap();
        let stats = queue.stats();
        
        assert_eq!(stats.current_size, 1);
        assert_eq!(stats.capacity, 10);
        assert_eq!(stats.total_enqueued, 1);
        assert_eq!(stats.utilization_percent, 10);
    }

    #[tokio::test]
    async fn test_fifo_queue() {
        let queue = MessageQueue::new_fifo(10);
        
        let msg1 = Message::request(serde_json::json!({"order": 1}));
        let msg2 = Message::request(serde_json::json!({"order": 2}));
        let msg3 = Message::request(serde_json::json!({"order": 3}));
        
        queue.enqueue(msg1).await.unwrap();
        queue.enqueue(msg2).await.unwrap();
        queue.enqueue(msg3).await.unwrap();
        
        // Should dequeue in FIFO order
        let first = queue.try_dequeue().unwrap();
        assert_eq!(first.payload["order"], 1);
        
        let second = queue.try_dequeue().unwrap();
        assert_eq!(second.payload["order"], 2);
        
        let third = queue.try_dequeue().unwrap();
        assert_eq!(third.payload["order"], 3);
    }

    #[tokio::test]
    async fn test_multi_queue() {
        let mut multi_queue = MultiQueue::new(10);
        
        let high_priority_queue = Arc::new(MessageQueue::new(5));
        multi_queue.add_queue("high_priority".to_string(), high_priority_queue.clone());
        
        let mut msg = Message::new(MessageType::Request);
        msg.headers.insert("queue_name".to_string(), "high_priority".to_string());
        
        multi_queue.route_message(msg).await.unwrap();
        
        assert_eq!(high_priority_queue.size(), 1);
        assert_eq!(multi_queue.default_queue.size(), 0);
    }
}