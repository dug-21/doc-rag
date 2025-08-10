//! # Message Bus
//!
//! Inter-component messaging system with pub/sub patterns,
//! guaranteed delivery, and dead letter handling.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, broadcast};
use tracing::{info, warn, error, debug};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

use crate::{Result, IntegrationError};

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MessagePriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

impl Default for MessagePriority {
    fn default() -> Self {
        MessagePriority::Normal
    }
}

/// Message delivery guarantee levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    /// Fire and forget
    AtMostOnce,
    /// At least once delivery (may duplicate)
    AtLeastOnce,
    /// Exactly once delivery (requires acknowledgment)
    ExactlyOnce,
}

/// Message envelope for inter-component communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Message ID
    pub id: Uuid,
    /// Source component
    pub source: String,
    /// Target component (None for broadcast)
    pub target: Option<String>,
    /// Message topic/type
    pub topic: String,
    /// Message payload
    pub payload: serde_json::Value,
    /// Message priority
    pub priority: MessagePriority,
    /// Delivery guarantee
    pub delivery_guarantee: DeliveryGuarantee,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Expiration timestamp
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    /// Retry count
    pub retry_count: u32,
    /// Maximum retries
    pub max_retries: u32,
    /// Correlation ID for request/response
    pub correlation_id: Option<Uuid>,
    /// Message headers
    pub headers: HashMap<String, String>,
}

impl Message {
    /// Create new message
    pub fn new<T: Serialize>(
        source: &str,
        topic: &str,
        payload: &T,
    ) -> Result<Self> {
        Ok(Self {
            id: Uuid::new_v4(),
            source: source.to_string(),
            target: None,
            topic: topic.to_string(),
            payload: serde_json::to_value(payload)?,
            priority: MessagePriority::default(),
            delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
            created_at: chrono::Utc::now(),
            expires_at: None,
            retry_count: 0,
            max_retries: 3,
            correlation_id: None,
            headers: HashMap::new(),
        })
    }
    
    /// Set target component
    pub fn to(mut self, target: &str) -> Self {
        self.target = Some(target.to_string());
        self
    }
    
    /// Set message priority
    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }
    
    /// Set delivery guarantee
    pub fn with_delivery_guarantee(mut self, guarantee: DeliveryGuarantee) -> Self {
        self.delivery_guarantee = guarantee;
        self
    }
    
    /// Set expiration
    pub fn expires_in(mut self, duration: Duration) -> Self {
        self.expires_at = Some(chrono::Utc::now() + chrono::Duration::from_std(duration).unwrap());
        self
    }
    
    /// Set correlation ID
    pub fn with_correlation_id(mut self, correlation_id: Uuid) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }
    
    /// Add header
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Check if message is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            chrono::Utc::now() > expires_at
        } else {
            false
        }
    }
    
    /// Check if message can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }
    
    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
    }
}

/// Message acknowledgment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAck {
    /// Message ID being acknowledged
    pub message_id: Uuid,
    /// Acknowledgment status
    pub status: AckStatus,
    /// Optional error message
    pub error: Option<String>,
    /// Processing time
    pub processing_time: Duration,
    /// Acknowledgment timestamp
    pub acked_at: chrono::DateTime<chrono::Utc>,
}

/// Acknowledgment status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AckStatus {
    /// Message processed successfully
    Success,
    /// Message processing failed, should retry
    Retry,
    /// Message processing failed, should not retry
    Reject,
}

/// Message handler trait
#[async_trait::async_trait]
pub trait MessageHandler: Send + Sync {
    /// Handle incoming message
    async fn handle_message(&self, message: Message) -> MessageAck;
    
    /// Get topics this handler subscribes to
    fn subscribed_topics(&self) -> Vec<String>;
    
    /// Handler name for identification
    fn name(&self) -> &str;
}

/// Message bus configuration
#[derive(Debug, Clone)]
pub struct MessageBusConfig {
    /// Maximum queue size per topic
    pub max_queue_size: usize,
    /// Message retention duration
    pub message_retention: Duration,
    /// Dead letter queue size
    pub dead_letter_queue_size: usize,
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Batch processing size
    pub batch_size: usize,
    /// Batch processing timeout
    pub batch_timeout: Duration,
}

impl Default for MessageBusConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            message_retention: Duration::from_secs(3600), // 1 hour
            dead_letter_queue_size: 1000,
            enable_metrics: true,
            batch_size: 100,
            batch_timeout: Duration::from_millis(100),
        }
    }
}

/// Message bus metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MessageBusMetrics {
    /// Messages sent
    pub messages_sent: u64,
    /// Messages received
    pub messages_received: u64,
    /// Messages acknowledged
    pub messages_acked: u64,
    /// Messages in dead letter queue
    pub messages_dead_lettered: u64,
    /// Messages expired
    pub messages_expired: u64,
    /// Messages retried
    pub messages_retried: u64,
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Metrics by topic
    pub topic_metrics: HashMap<String, TopicMetrics>,
}

/// Topic-specific metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TopicMetrics {
    /// Messages in topic
    pub message_count: u64,
    /// Subscribers count
    pub subscriber_count: usize,
    /// Average message size
    pub avg_message_size: usize,
    /// Processing rate (messages/second)
    pub processing_rate: f64,
}

/// Message queue for a topic
struct TopicQueue {
    /// Topic name
    topic: String,
    /// Message queue
    messages: mpsc::UnboundedSender<Message>,
    /// Message receiver
    receiver: Arc<RwLock<Option<mpsc::UnboundedReceiver<Message>>>>,
    /// Broadcast channel for multiple subscribers
    broadcast_tx: broadcast::Sender<Message>,
    /// Subscribers
    subscribers: Vec<String>,
}

impl TopicQueue {
    fn new(topic: String) -> Self {
        let (messages, receiver) = mpsc::unbounded_channel();
        let (broadcast_tx, _) = broadcast::channel(1000);
        
        Self {
            topic,
            messages,
            receiver: Arc::new(RwLock::new(Some(receiver))),
            broadcast_tx,
            subscribers: Vec::new(),
        }
    }
    
    async fn send_message(&self, message: Message) -> Result<()> {
        // Send to queue
        self.messages.send(message.clone())
            .map_err(|e| IntegrationError::MessageBusError(format!("Failed to send message: {}", e)))?;
        
        // Broadcast to subscribers
        if self.broadcast_tx.send(message).is_err() {
            debug!("No active subscribers for topic: {}", self.topic);
        }
        
        Ok(())
    }
    
    fn subscribe(&self) -> broadcast::Receiver<Message> {
        self.broadcast_tx.subscribe()
    }
}

/// Main message bus implementation
pub struct MessageBus {
    /// Bus ID
    id: Uuid,
    /// Configuration
    config: Arc<crate::IntegrationConfig>,
    /// Message bus configuration
    bus_config: MessageBusConfig,
    /// Topic queues
    topics: Arc<RwLock<HashMap<String, Arc<TopicQueue>>>>,
    /// Message handlers
    handlers: Arc<RwLock<HashMap<String, Arc<dyn MessageHandler>>>>,
    /// Dead letter queue
    dead_letter_queue: Arc<RwLock<Vec<(Message, String)>>>, // (message, reason)
    /// Message bus metrics
    metrics: Arc<RwLock<MessageBusMetrics>>,
    /// Pending acknowledgments
    pending_acks: Arc<RwLock<HashMap<Uuid, (Message, Instant)>>>,
}

impl MessageBus {
    /// Create new message bus
    pub async fn new(config: Arc<crate::IntegrationConfig>) -> Result<Self> {
        let bus_config = MessageBusConfig::default();
        
        Ok(Self {
            id: Uuid::new_v4(),
            config,
            bus_config,
            topics: Arc::new(RwLock::new(HashMap::new())),
            handlers: Arc::new(RwLock::new(HashMap::new())),
            dead_letter_queue: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(MessageBusMetrics::default())),
            pending_acks: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Initialize message bus
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing Message Bus: {}", self.id);
        
        // Create default topics
        let default_topics = [
            "component.health",
            "component.metrics",
            "system.events",
            "pipeline.events",
            "error.events",
        ];
        
        for topic in default_topics {
            self.create_topic(topic).await?;
        }
        
        info!("Message Bus initialized successfully");
        Ok(())
    }
    
    /// Start message bus
    pub async fn start(&self) -> Result<()> {
        info!("Starting Message Bus...");
        
        // Start message processing
        let bus = self.clone();
        tokio::spawn(async move {
            bus.process_messages().await;
        });
        
        // Start cleanup tasks
        let bus = self.clone();
        tokio::spawn(async move {
            bus.cleanup_expired_messages().await;
        });
        
        let bus = self.clone();
        tokio::spawn(async move {
            bus.process_pending_acks().await;
        });
        
        // Start metrics collection
        if self.bus_config.enable_metrics {
            let bus = self.clone();
            tokio::spawn(async move {
                bus.collect_metrics().await;
            });
        }
        
        info!("Message Bus started successfully");
        Ok(())
    }
    
    /// Stop message bus
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Message Bus...");
        
        // Process remaining messages in dead letter queue
        let dead_letters = {
            let dlq = self.dead_letter_queue.read().await;
            dlq.len()
        };
        
        if dead_letters > 0 {
            warn!("Message Bus stopping with {} messages in dead letter queue", dead_letters);
        }
        
        info!("Message Bus stopped successfully");
        Ok(())
    }
    
    /// Create a new topic
    pub async fn create_topic(&self, topic_name: &str) -> Result<()> {
        let mut topics = self.topics.write().await;
        
        if !topics.contains_key(topic_name) {
            let topic_queue = Arc::new(TopicQueue::new(topic_name.to_string()));
            topics.insert(topic_name.to_string(), topic_queue);
            
            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.topic_metrics.insert(topic_name.to_string(), TopicMetrics::default());
            
            info!("Created topic: {}", topic_name);
        }
        
        Ok(())
    }
    
    /// Publish message to topic
    pub async fn publish(&self, message: Message) -> Result<()> {
        let topic_name = message.topic.clone();
        
        // Ensure topic exists
        if !self.topic_exists(&topic_name).await {
            self.create_topic(&topic_name).await?;
        }
        
        // Check if message is expired
        if message.is_expired() {
            self.handle_expired_message(message).await;
            return Ok(());
        }
        
        // Get topic queue
        let topics = self.topics.read().await;
        if let Some(topic_queue) = topics.get(&topic_name) {
            // Add to pending acks if required
            if message.delivery_guarantee == DeliveryGuarantee::ExactlyOnce {
                let mut pending_acks = self.pending_acks.write().await;
                pending_acks.insert(message.id, (message.clone(), Instant::now()));
            }
            
            // Send message
            topic_queue.send_message(message.clone()).await?;
            
            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.messages_sent += 1;
            
            if let Some(topic_metrics) = metrics.topic_metrics.get_mut(&topic_name) {
                topic_metrics.message_count += 1;
                topic_metrics.avg_message_size = 
                    (topic_metrics.avg_message_size + message.payload.to_string().len()) / 2;
            }
            
            debug!("Published message {} to topic {}", message.id, topic_name);
        }
        
        Ok(())
    }
    
    /// Subscribe to topic with handler
    pub async fn subscribe<H: MessageHandler + 'static>(
        &self,
        handler: H,
    ) -> Result<()> {
        let handler_name = handler.name().to_string();
        let topics = handler.subscribed_topics();
        
        info!("Subscribing handler '{}' to topics: {:?}", handler_name, topics);
        
        // Store handler
        let handler_arc = Arc::new(handler);
        {
            let mut handlers = self.handlers.write().await;
            handlers.insert(handler_name.clone(), handler_arc.clone());
        }
        
        // Subscribe to each topic
        for topic_name in topics {
            self.subscribe_to_topic(&topic_name, handler_arc.clone()).await?;
        }
        
        Ok(())
    }
    
    /// Subscribe handler to specific topic
    async fn subscribe_to_topic(
        &self,
        topic_name: &str,
        handler: Arc<dyn MessageHandler>,
    ) -> Result<()> {
        // Ensure topic exists
        if !self.topic_exists(topic_name).await {
            self.create_topic(topic_name).await?;
        }
        
        // Get topic queue and subscribe
        let topics = self.topics.read().await;
        if let Some(topic_queue) = topics.get(topic_name) {
            let mut receiver = topic_queue.subscribe();
            let bus = self.clone();
            let _topic = topic_name.to_string();
            
            tokio::spawn(async move {
                while let Ok(message) = receiver.recv().await {
                    let start_time = Instant::now();
                    let ack = handler.handle_message(message.clone()).await;
                    let processing_time = start_time.elapsed();
                    
                    // Update acknowledgment with processing time
                    let mut full_ack = ack;
                    full_ack.processing_time = processing_time;
                    
                    if let Err(e) = bus.handle_acknowledgment(full_ack).await {
                        error!("Failed to handle acknowledgment: {}", e);
                    }
                }
            });
            
            // Update topic metrics
            let mut metrics = self.metrics.write().await;
            if let Some(topic_metrics) = metrics.topic_metrics.get_mut(topic_name) {
                topic_metrics.subscriber_count += 1;
            }
        }
        
        Ok(())
    }
    
    /// Handle message acknowledgment
    async fn handle_acknowledgment(&self, ack: MessageAck) -> Result<()> {
        // Remove from pending acks
        let pending_message = {
            let mut pending_acks = self.pending_acks.write().await;
            pending_acks.remove(&ack.message_id)
        };
        
        match ack.status {
            AckStatus::Success => {
                let mut metrics = self.metrics.write().await;
                metrics.messages_acked += 1;
                
                // Update average processing time
                let total_time = metrics.avg_processing_time.as_millis() as f64 * (metrics.messages_acked - 1) as f64;
                metrics.avg_processing_time = Duration::from_millis(
                    ((total_time + ack.processing_time.as_millis() as f64) / metrics.messages_acked as f64) as u64
                );
            }
            AckStatus::Retry => {
                if let Some((mut message, _)) = pending_message {
                    if message.can_retry() {
                        message.increment_retry();
                        
                        // Republish with delay
                        let bus = self.clone();
                        tokio::spawn(async move {
                            tokio::time::sleep(Duration::from_secs(2_u64.pow(message.retry_count))).await;
                            if let Err(e) = bus.publish(message.clone()).await {
                                error!("Failed to retry message {}: {}", message.id, e);
                                bus.send_to_dead_letter(message, "Retry failed".to_string()).await;
                            }
                        });
                        
                        let mut metrics = self.metrics.write().await;
                        metrics.messages_retried += 1;
                    } else {
                        self.send_to_dead_letter(message, "Max retries exceeded".to_string()).await;
                    }
                }
            }
            AckStatus::Reject => {
                if let Some((message, _)) = pending_message {
                    let reason = ack.error.unwrap_or_else(|| "Message rejected by handler".to_string());
                    self.send_to_dead_letter(message, reason).await;
                }
            }
        }
        
        Ok(())
    }
    
    /// Send message to dead letter queue
    async fn send_to_dead_letter(&self, message: Message, reason: String) {
        let mut dlq = self.dead_letter_queue.write().await;
        
        if dlq.len() >= self.bus_config.dead_letter_queue_size {
            // Remove oldest message
            dlq.remove(0);
        }
        
        dlq.push((message.clone(), reason.clone()));
        
        let mut metrics = self.metrics.write().await;
        metrics.messages_dead_lettered += 1;
        
        warn!("Message {} sent to dead letter queue: {}", message.id, reason);
    }
    
    /// Check if topic exists
    async fn topic_exists(&self, topic_name: &str) -> bool {
        let topics = self.topics.read().await;
        topics.contains_key(topic_name)
    }
    
    /// Handle expired message
    async fn handle_expired_message(&self, message: Message) {
        let mut metrics = self.metrics.write().await;
        metrics.messages_expired += 1;
        
        warn!("Message {} expired", message.id);
        
        drop(metrics);
        self.send_to_dead_letter(message, "Message expired".to_string()).await;
    }
    
    /// Process messages (main processing loop)
    async fn process_messages(&self) {
        // Main message processing is handled by individual topic subscriptions
        // This could be extended for additional processing logic
        info!("Message processing started");
    }
    
    /// Cleanup expired messages and acknowledgments
    async fn cleanup_expired_messages(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            // Clean up expired pending acknowledgments
            let expired_acks: Vec<Uuid> = {
                let pending_acks = self.pending_acks.read().await;
                let now = Instant::now();
                
                pending_acks.iter()
                    .filter(|(_, (_, timestamp))| now.duration_since(*timestamp) > Duration::from_secs(300))
                    .map(|(&id, _)| id)
                    .collect()
            };
            
            if !expired_acks.is_empty() {
                let mut pending_acks = self.pending_acks.write().await;
                for ack_id in expired_acks {
                    if let Some((message, _)) = pending_acks.remove(&ack_id) {
                        drop(pending_acks);
                        self.send_to_dead_letter(message, "Acknowledgment timeout".to_string()).await;
                        pending_acks = self.pending_acks.write().await;
                    }
                }
            }
        }
    }
    
    /// Process pending acknowledgments
    async fn process_pending_acks(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            let pending_count = self.pending_acks.read().await.len();
            if pending_count > 0 {
                debug!("Pending acknowledgments: {}", pending_count);
            }
        }
    }
    
    /// Collect metrics
    async fn collect_metrics(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            let metrics = self.metrics.read().await;
            info!("Message Bus Metrics: sent={}, received={}, acked={}, dead_lettered={}",
                metrics.messages_sent,
                metrics.messages_received,
                metrics.messages_acked,
                metrics.messages_dead_lettered
            );
        }
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> MessageBusMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get dead letter queue contents
    pub async fn get_dead_letter_queue(&self) -> Vec<(Message, String)> {
        self.dead_letter_queue.read().await.clone()
    }
    
    /// Replay message from dead letter queue
    pub async fn replay_dead_letter(&self, message_id: Uuid) -> Result<()> {
        let message_to_replay = {
            let mut dlq = self.dead_letter_queue.write().await;
            let pos = dlq.iter().position(|(msg, _)| msg.id == message_id);
            
            if let Some(index) = pos {
                Some(dlq.remove(index).0)
            } else {
                None
            }
        };
        
        if let Some(mut message) = message_to_replay {
            message.retry_count = 0; // Reset retry count
            message.created_at = chrono::Utc::now(); // Update timestamp
            
            self.publish(message).await?;
            info!("Replayed message {} from dead letter queue", message_id);
        } else {
            return Err(IntegrationError::MessageBusError(
                format!("Message {} not found in dead letter queue", message_id)
            ));
        }
        
        Ok(())
    }
}

impl Clone for MessageBus {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            config: self.config.clone(),
            bus_config: self.bus_config.clone(),
            topics: self.topics.clone(),
            handlers: self.handlers.clone(),
            dead_letter_queue: self.dead_letter_queue.clone(),
            metrics: self.metrics.clone(),
            pending_acks: self.pending_acks.clone(),
        }
    }
}

/// Example message handler for testing
#[cfg(test)]
struct TestMessageHandler {
    name: String,
    topics: Vec<String>,
}

#[cfg(test)]
#[async_trait::async_trait]
impl MessageHandler for TestMessageHandler {
    async fn handle_message(&self, message: Message) -> MessageAck {
        // Simulate message processing
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        MessageAck {
            message_id: message.id,
            status: AckStatus::Success,
            error: None,
            processing_time: Duration::from_millis(10),
            acked_at: chrono::Utc::now(),
        }
    }
    
    fn subscribed_topics(&self) -> Vec<String> {
        self.topics.clone()
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntegrationConfig;
    
    #[tokio::test]
    async fn test_message_bus_creation() {
        let config = Arc::new(IntegrationConfig::default());
        let bus = MessageBus::new(config).await.unwrap();
        
        bus.initialize().await.unwrap();
        assert!(bus.topic_exists("component.health").await);
    }
    
    #[tokio::test]
    async fn test_topic_creation() {
        let config = Arc::new(IntegrationConfig::default());
        let bus = MessageBus::new(config).await.unwrap();
        
        bus.create_topic("test.topic").await.unwrap();
        assert!(bus.topic_exists("test.topic").await);
    }
    
    #[tokio::test]
    async fn test_message_publishing() {
        let config = Arc::new(IntegrationConfig::default());
        let bus = MessageBus::new(config).await.unwrap();
        
        bus.initialize().await.unwrap();
        bus.start().await.unwrap();
        
        let message = Message::new("test-component", "test.topic", &"test payload").unwrap();
        let result = bus.publish(message).await;
        
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_message_subscription() {
        let config = Arc::new(IntegrationConfig::default());
        let bus = MessageBus::new(config).await.unwrap();
        
        bus.initialize().await.unwrap();
        bus.start().await.unwrap();
        
        let handler = TestMessageHandler {
            name: "test-handler".to_string(),
            topics: vec!["test.topic".to_string()],
        };
        
        let result = bus.subscribe(handler).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_message_expiration() {
        let config = Arc::new(IntegrationConfig::default());
        let bus = MessageBus::new(config).await.unwrap();
        
        let mut message = Message::new("test-component", "test.topic", &"test payload").unwrap();
        message.expires_at = Some(chrono::Utc::now() - chrono::Duration::seconds(1));
        
        assert!(message.is_expired());
    }
    
    #[tokio::test]
    async fn test_message_retry() {
        let mut message = Message::new("test-component", "test.topic", &"test payload").unwrap();
        message.max_retries = 3;
        
        assert!(message.can_retry());
        
        message.increment_retry();
        assert_eq!(message.retry_count, 1);
        assert!(message.can_retry());
        
        message.retry_count = 3;
        assert!(!message.can_retry());
    }
    
    #[tokio::test]
    async fn test_dead_letter_queue() {
        let config = Arc::new(IntegrationConfig::default());
        let bus = MessageBus::new(config).await.unwrap();
        
        let message = Message::new("test-component", "test.topic", &"test payload").unwrap();
        let message_id = message.id;
        
        bus.send_to_dead_letter(message, "Test reason".to_string()).await;
        
        let dlq = bus.get_dead_letter_queue().await;
        assert_eq!(dlq.len(), 1);
        assert_eq!(dlq[0].0.id, message_id);
        assert_eq!(dlq[0].1, "Test reason");
    }
}
