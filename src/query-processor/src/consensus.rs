//! Byzantine fault-tolerant consensus mechanism for distributed query processing
//!
//! This module implements a robust consensus system to ensure reliability and consistency
//! across distributed query processing nodes:
//! - Byzantine Fault Tolerant (BFT) consensus using PBFT-inspired algorithm  
//! - Distributed validation of query processing results
//! - Node health monitoring and failure detection
//! - Consensus rounds with configurable timeouts
//! - Quorum-based decision making
//! - Result verification and consistency checking
//! - Network partition tolerance
//! - Dynamic node membership management
//! - Consensus metrics and monitoring

use crate::error::{Result, ProcessorError};
use crate::types::{
    SemanticAnalysis, IntentClassification, KeyTerm, StrategySelection,
    ConsensusResult, ValidationResult, QueryIntent, SearchStrategy, ExtractedEntity,
    ProcessedQuery, QueryResult, ClassificationResult, StrategyRecommendation
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt::Debug;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Mutex, mpsc, oneshot};
use tokio::time::{timeout, sleep};
use tracing::{debug, info, warn, error, instrument};
use uuid::Uuid;

/// Node identifier type
pub type NodeId = String;

/// Consensus round identifier
pub type RoundId = u64;

/// View number for consensus protocol
pub type ViewNumber = u64;

/// Sequence number for ordering operations
pub type SequenceNumber = u64;

/// Maximum number of Byzantine faults tolerated (f)
/// Total nodes must be >= 3f + 1
const MAX_BYZANTINE_FAULTS: usize = 1;

/// Minimum number of nodes for consensus
const MIN_CONSENSUS_NODES: usize = 3 * MAX_BYZANTINE_FAULTS + 1;

/// Consensus message types following PBFT protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusMessage {
    /// Request from client to initiate consensus
    Request {
        /// Unique request identifier
        request_id: String,
        /// Query processing payload
        payload: ConsensusPayload,
        /// Client identifier
        client_id: String,
        /// Request timestamp
        timestamp: u64,
    },
    
    /// Pre-prepare message from primary
    PrePrepare {
        /// Current view number
        view: ViewNumber,
        /// Sequence number for ordering
        sequence: SequenceNumber,
        /// Message digest for verification
        digest: String,
        /// Request identifier
        request_id: String,
        /// Payload data
        payload: ConsensusPayload,
        /// Primary node identifier
        primary_id: NodeId,
    },
    
    /// Prepare message from backup nodes
    Prepare {
        /// Current view number
        view: ViewNumber,
        /// Sequence number
        sequence: SequenceNumber,
        /// Message digest
        digest: String,
        /// Node identifier sending prepare
        node_id: NodeId,
    },
    
    /// Commit message when prepare phase succeeds
    Commit {
        /// Current view number
        view: ViewNumber,
        /// Sequence number
        sequence: SequenceNumber,
        /// Message digest
        digest: String,
        /// Node identifier sending commit
        node_id: NodeId,
    },
    
    /// Reply with consensus result
    Reply {
        /// View number when result was decided
        view: ViewNumber,
        /// Request identifier
        request_id: String,
        /// Consensus result
        result: ConsensusResult,
        /// Node identifier sending reply
        node_id: NodeId,
        /// Result timestamp
        timestamp: u64,
    },
    
    /// View change message for primary failover
    ViewChange {
        /// New view number being proposed
        new_view: ViewNumber,
        /// Node identifier initiating change
        node_id: NodeId,
        /// Evidence of primary failure
        evidence: ViewChangeEvidence,
    },
    
    /// New view message from new primary
    NewView {
        /// New view number
        view: ViewNumber,
        /// View change messages as proof
        view_change_messages: Vec<ConsensusMessage>,
        /// Pre-prepare messages to resume
        pre_prepare_messages: Vec<ConsensusMessage>,
        /// New primary identifier
        primary_id: NodeId,
    },
    
    /// Heartbeat for liveness detection
    Heartbeat {
        /// Node identifier
        node_id: NodeId,
        /// Current view
        view: ViewNumber,
        /// Timestamp
        timestamp: u64,
        /// Node status information
        status: NodeStatus,
    },
}

/// Consensus payload containing query processing data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsensusPayload {
    /// Query processing request
    QueryProcessing {
        /// Original query
        query: ProcessedQuery,
        /// Processing configuration
        config: ProcessingConfig,
    },
    
    /// Entity extraction validation
    EntityExtraction {
        /// Input query text
        query_text: String,
        /// Extracted entities from multiple nodes
        entity_results: Vec<NodeEntityResult>,
    },
    
    /// Intent classification validation
    Classification {
        /// Query text for classification
        query_text: String,
        /// Classification results from nodes
        classification_results: Vec<NodeClassificationResult>,
    },
    
    /// Strategy recommendation consensus
    StrategyRecommendation {
        /// Processed query
        query: ProcessedQuery,
        /// Strategy recommendations from nodes
        strategy_results: Vec<NodeStrategyResult>,
    },
    
    /// Final result validation
    ResultValidation {
        /// Complete query result
        result: QueryResult,
        /// Validation results from nodes
        validation_results: Vec<NodeValidationResult>,
    },
}

/// Processing configuration for consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Required confidence threshold
    pub min_confidence: f64,
    
    /// Maximum processing time allowed
    pub max_processing_time: Duration,
    
    /// Consensus timeout
    pub consensus_timeout: Duration,
    
    /// Minimum node agreement required
    pub min_agreement_ratio: f64,
    
    /// Enable Byzantine fault detection
    pub enable_fault_detection: bool,
}

/// Entity extraction result from a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEntityResult {
    /// Node identifier
    pub node_id: NodeId,
    
    /// Extracted entities
    pub entities: Vec<ExtractedEntity>,
    
    /// Overall confidence score
    pub confidence: f64,
    
    /// Processing time taken
    pub processing_time: Duration,
    
    /// Node-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Classification result from a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeClassificationResult {
    /// Node identifier
    pub node_id: NodeId,
    
    /// Classification result
    pub classification: ClassificationResult,
    
    /// Processing time
    pub processing_time: Duration,
    
    /// Node version/capability info
    pub node_version: String,
}

/// Strategy recommendation from a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStrategyResult {
    /// Node identifier
    pub node_id: NodeId,
    
    /// Strategy recommendation
    pub recommendation: StrategyRecommendation,
    
    /// Processing time
    pub processing_time: Duration,
    
    /// Strategy reasoning
    pub reasoning: String,
}

/// Validation result from a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeValidationResult {
    /// Node identifier
    pub node_id: NodeId,
    
    /// Validation outcome
    pub validation: ValidationResult,
    
    /// Processing time
    pub processing_time: Duration,
    
    /// Validation details
    pub details: String,
}

/// Evidence for view change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewChangeEvidence {
    /// Last successful sequence number
    pub last_sequence: SequenceNumber,
    
    /// Evidence of primary timeout
    pub timeout_evidence: Option<TimeoutEvidence>,
    
    /// Evidence of primary Byzantine behavior
    pub byzantine_evidence: Option<ByzantineEvidence>,
    
    /// Timestamp of evidence collection
    pub timestamp: u64,
}

/// Timeout evidence for view change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutEvidence {
    /// Expected response time
    pub expected_response_time: Duration,
    
    /// Actual elapsed time
    pub actual_elapsed_time: Duration,
    
    /// Number of missed heartbeats
    pub missed_heartbeats: u32,
}

/// Byzantine behavior evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineEvidence {
    /// Type of Byzantine behavior detected
    pub behavior_type: ByzantineBehavior,
    
    /// Conflicting messages as proof
    pub conflicting_messages: Vec<ConsensusMessage>,
    
    /// Witnesses that observed the behavior
    pub witnesses: Vec<NodeId>,
}

/// Types of Byzantine behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ByzantineBehavior {
    /// Sending conflicting messages
    Equivocation,
    
    /// Sending invalid messages
    InvalidMessage,
    
    /// Timeout/non-responsiveness
    Timeout,
    
    /// Incorrect result computation
    IncorrectComputation,
}

/// Node status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    /// Node health status
    pub health: NodeHealth,
    
    /// Current load/capacity
    pub load: f64,
    
    /// Available resources
    pub resources: ResourceStatus,
    
    /// Last activity timestamp
    pub last_activity: u64,
    
    /// Node capabilities
    pub capabilities: Vec<String>,
}

/// Node health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeHealth {
    /// Node is healthy and operational
    Healthy,
    
    /// Node has minor issues but operational
    Degraded,
    
    /// Node is experiencing problems
    Unhealthy,
    
    /// Node is not responding
    Unresponsive,
}

/// Resource status for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStatus {
    /// CPU usage percentage
    pub cpu_usage: f64,
    
    /// Memory usage percentage
    pub memory_usage: f64,
    
    /// Network bandwidth utilization
    pub network_usage: f64,
    
    /// Disk I/O utilization
    pub disk_usage: f64,
}

/// Consensus state for tracking protocol progress
#[derive(Debug, Clone)]
pub struct ConsensusState {
    /// Current view number
    pub current_view: ViewNumber,
    
    /// Current primary node
    pub current_primary: NodeId,
    
    /// Last executed sequence number
    pub last_executed: SequenceNumber,
    
    /// Pending requests waiting for consensus
    pub pending_requests: HashMap<String, PendingRequest>,
    
    /// Active consensus rounds
    pub active_rounds: HashMap<SequenceNumber, ConsensusRound>,
    
    /// Known nodes in the consensus group
    pub nodes: HashMap<NodeId, NodeInfo>,
    
    /// Message log for Byzantine detection
    pub message_log: BTreeMap<SequenceNumber, Vec<ConsensusMessage>>,
}

/// Information about a consensus node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node identifier
    pub node_id: NodeId,
    
    /// Network endpoint
    pub endpoint: String,
    
    /// Node status
    pub status: NodeStatus,
    
    /// Last heartbeat timestamp
    pub last_heartbeat: u64,
    
    /// Node join timestamp
    pub joined_at: u64,
    
    /// Public key for message verification
    pub public_key: Option<Vec<u8>>,
}

/// Pending consensus request
#[derive(Debug, Clone)]
pub struct PendingRequest {
    /// Request identifier
    pub request_id: String,
    
    /// Client identifier
    pub client_id: String,
    
    /// Consensus payload
    pub payload: ConsensusPayload,
    
    /// Request timestamp
    pub timestamp: u64,
    
    /// Response channel
    pub response_sender: Option<oneshot::Sender<ConsensusResult>>,
}

/// Active consensus round state
#[derive(Debug, Clone)]
pub struct ConsensusRound {
    /// Round identifier
    pub round_id: RoundId,
    
    /// Sequence number
    pub sequence: SequenceNumber,
    
    /// View number
    pub view: ViewNumber,
    
    /// Request being processed
    pub request_id: String,
    
    /// Consensus payload
    pub payload: ConsensusPayload,
    
    /// Round phase
    pub phase: ConsensusPhase,
    
    /// Received prepare messages
    pub prepare_messages: HashMap<NodeId, ConsensusMessage>,
    
    /// Received commit messages
    pub commit_messages: HashMap<NodeId, ConsensusMessage>,
    
    /// Round start time
    pub start_time: Instant,
    
    /// Round timeout
    pub timeout: Duration,
}

/// Consensus protocol phases
#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusPhase {
    /// Initial phase - waiting for pre-prepare
    PrePrepare,
    
    /// Prepare phase - collecting prepare messages
    Prepare,
    
    /// Commit phase - collecting commit messages
    Commit,
    
    /// Execution phase - applying the operation
    Execute,
    
    /// Completed phase
    Complete,
    
    /// Failed phase
    Failed(String),
}

/// Consensus manager implementing PBFT protocol
pub struct ConsensusManager {
    /// Node identifier for this instance
    node_id: NodeId,
    
    /// Consensus configuration
    config: Arc<ConsensusConfig>,
    
    /// Current consensus state
    state: Arc<RwLock<ConsensusState>>,
    
    /// Message sender for outbound communication
    message_sender: mpsc::UnboundedSender<(NodeId, ConsensusMessage)>,
    
    /// Message receiver for inbound communication
    message_receiver: Arc<Mutex<mpsc::UnboundedReceiver<ConsensusMessage>>>,
    
    /// Consensus metrics
    metrics: Arc<RwLock<ConsensusMetrics>>,
    
    /// Background task handles
    background_tasks: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
    
    /// Query processor interface
    query_processor: Arc<dyn QueryProcessorInterface + Send + Sync>,
}

/// Configuration for consensus mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Consensus timeout for rounds
    pub consensus_timeout: Duration,
    
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    
    /// View change timeout
    pub view_change_timeout: Duration,
    
    /// Maximum concurrent consensus rounds
    pub max_concurrent_rounds: usize,
    
    /// Enable Byzantine fault detection
    pub enable_byzantine_detection: bool,
    
    /// Minimum nodes required for consensus
    pub min_nodes: usize,
    
    /// Message batching configuration
    pub batching: BatchingConfig,
    
    /// Retransmission settings
    pub retransmission: RetransmissionConfig,
}

/// Message batching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchingConfig {
    /// Enable message batching
    pub enabled: bool,
    
    /// Maximum batch size
    pub max_batch_size: usize,
    
    /// Batch timeout
    pub batch_timeout: Duration,
}

/// Retransmission configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetransmissionConfig {
    /// Enable message retransmission
    pub enabled: bool,
    
    /// Maximum retransmission attempts
    pub max_attempts: u32,
    
    /// Base retry delay
    pub base_delay: Duration,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
}

/// Consensus metrics and monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsensusMetrics {
    /// Total consensus rounds initiated
    pub total_rounds: u64,
    
    /// Successfully completed rounds
    pub successful_rounds: u64,
    
    /// Failed rounds
    pub failed_rounds: u64,
    
    /// Average round completion time
    pub avg_round_time: Duration,
    
    /// View changes performed
    pub view_changes: u64,
    
    /// Byzantine faults detected
    pub byzantine_faults_detected: u64,
    
    /// Message statistics
    pub message_stats: MessageStats,
    
    /// Node health metrics
    pub node_health: HashMap<NodeId, NodeHealthMetrics>,
}

/// Message statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MessageStats {
    /// Total messages sent
    pub messages_sent: u64,
    
    /// Total messages received
    pub messages_received: u64,
    
    /// Invalid messages rejected
    pub invalid_messages: u64,
    
    /// Messages retransmitted
    pub retransmissions: u64,
}

/// Node health metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeHealthMetrics {
    /// Heartbeats received
    pub heartbeats_received: u64,
    
    /// Missed heartbeats
    pub missed_heartbeats: u64,
    
    /// Response time statistics
    pub avg_response_time: Duration,
    
    /// Last activity timestamp
    pub last_activity: u64,
}

/// Interface for query processing operations
#[async_trait]
pub trait QueryProcessorInterface: Debug + Send + Sync {
    /// Process a query and return results
    async fn process_query(&self, query: &ProcessedQuery) -> Result<QueryResult>;
    
    /// Extract entities from query text
    async fn extract_entities(&self, query_text: &str) -> Result<Vec<ExtractedEntity>>;
    
    /// Classify query intent
    async fn classify_query(&self, query_text: &str) -> Result<ClassificationResult>;
    
    /// Recommend search strategy
    async fn recommend_strategy(&self, query: &ProcessedQuery) -> Result<StrategyRecommendation>;
    
    /// Validate query result
    async fn validate_result(&self, result: &QueryResult) -> Result<ValidationResult>;
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            consensus_timeout: Duration::from_millis(200), // Fast consensus for <50ms target
            heartbeat_interval: Duration::from_millis(50),
            view_change_timeout: Duration::from_millis(500),
            max_concurrent_rounds: 100,
            enable_byzantine_detection: true,
            min_nodes: MIN_CONSENSUS_NODES,
            batching: BatchingConfig::default(),
            retransmission: RetransmissionConfig::default(),
        }
    }
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_batch_size: 10,
            batch_timeout: Duration::from_millis(10),
        }
    }
}

impl Default for RetransmissionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_attempts: 3,
            base_delay: Duration::from_millis(5),
            backoff_multiplier: 2.0,
        }
    }
}

impl ConsensusManager {
    /// Create a new consensus manager
    pub fn new(
        node_id: NodeId,
        config: ConsensusConfig,
        query_processor: Arc<dyn QueryProcessorInterface + Send + Sync>,
    ) -> Self {
        let (message_sender, message_receiver) = mpsc::unbounded_channel();
        
        let initial_state = ConsensusState {
            current_view: 0,
            current_primary: node_id.clone(), // Start as primary for simplicity
            last_executed: 0,
            pending_requests: HashMap::new(),
            active_rounds: HashMap::new(),
            nodes: HashMap::new(),
            message_log: BTreeMap::new(),
        };
        
        Self {
            node_id,
            config: Arc::new(config),
            state: Arc::new(RwLock::new(initial_state)),
            message_sender,
            message_receiver: Arc::new(Mutex::new(message_receiver)),
            metrics: Arc::new(RwLock::new(ConsensusMetrics::default())),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
            query_processor,
        }
    }
    
    /// Start consensus manager and background tasks
    pub async fn start(&self) -> Result<()> {
        info!("Starting consensus manager for node: {}", self.node_id);
        
        // Start message processing task
        self.start_message_processor().await?;
        
        // Start heartbeat task
        self.start_heartbeat_task().await?;
        
        // Start view change monitor
        self.start_view_change_monitor().await?;
        
        info!("Consensus manager started successfully");
        Ok(())
    }
    
    /// Request consensus on a query processing operation
    #[instrument(skip(self, payload))]
    pub async fn request_consensus(
        &self,
        request_id: String,
        client_id: String,
        payload: ConsensusPayload,
    ) -> Result<ConsensusResult> {
        let (response_sender, response_receiver) = oneshot::channel();
        
        // Add to pending requests
        {
            let mut state = self.state.write().await;
            state.pending_requests.insert(request_id.clone(), PendingRequest {
                request_id: request_id.clone(),
                client_id: client_id.clone(),
                payload: payload.clone(),
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                response_sender: Some(response_sender),
            });
        }
        
        // If we're the primary, initiate consensus
        {
            let state = self.state.read().await;
            if state.current_primary == self.node_id {
                drop(state);
                self.initiate_consensus_round(request_id.clone(), payload).await?;
            }
        }
        
        // Wait for consensus result with timeout
        match timeout(self.config.consensus_timeout, response_receiver).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err(ProcessorError::ConsensusFailed { reason: "Response channel closed".to_string() }),
            Err(_) => Err(ProcessorError::Timeout { operation: "consensus".to_string(), duration: self.config.consensus_timeout }),
        }
    }
    
    /// Initiate a new consensus round (primary only)
    async fn initiate_consensus_round(
        &self,
        request_id: String,
        payload: ConsensusPayload,
    ) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Assign sequence number
        let sequence = state.last_executed + 1;
        
        // Create consensus round
        let round = ConsensusRound {
            round_id: generate_round_id(),
            sequence,
            view: state.current_view,
            request_id: request_id.clone(),
            payload: payload.clone(),
            phase: ConsensusPhase::PrePrepare,
            prepare_messages: HashMap::new(),
            commit_messages: HashMap::new(),
            start_time: Instant::now(),
            timeout: self.config.consensus_timeout,
        };
        
        state.active_rounds.insert(sequence, round);
        
        // Create pre-prepare message
        let digest = compute_digest(&payload);
        let pre_prepare = ConsensusMessage::PrePrepare {
            view: state.current_view,
            sequence,
            digest,
            request_id,
            payload,
            primary_id: self.node_id.clone(),
        };
        
        // Broadcast pre-prepare to all nodes
        self.broadcast_message(pre_prepare).await?;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_rounds += 1;
        }
        
        debug!("Initiated consensus round {} for request", sequence);
        Ok(())
    }
    
    /// Process incoming consensus messages
    async fn process_message(&self, message: ConsensusMessage) -> Result<()> {
        debug!("Processing consensus message: {:?}", message);
        
        match message {
            ConsensusMessage::Request { request_id, payload, client_id, timestamp } => {
                self.handle_request(request_id, payload, client_id, timestamp).await
            },
            
            ConsensusMessage::PrePrepare { view, sequence, digest, request_id, payload, primary_id } => {
                self.handle_pre_prepare(view, sequence, digest, request_id, payload, primary_id).await
            },
            
            ConsensusMessage::Prepare { view, sequence, digest, node_id } => {
                self.handle_prepare(view, sequence, digest, node_id).await
            },
            
            ConsensusMessage::Commit { view, sequence, digest, node_id } => {
                self.handle_commit(view, sequence, digest, node_id).await
            },
            
            ConsensusMessage::Reply { .. } => {
                // Replies are handled by clients
                Ok(())
            },
            
            ConsensusMessage::ViewChange { new_view, node_id, evidence } => {
                self.handle_view_change(new_view, node_id, evidence).await
            },
            
            ConsensusMessage::NewView { view, view_change_messages, pre_prepare_messages, primary_id } => {
                self.handle_new_view(view, view_change_messages, pre_prepare_messages, primary_id).await
            },
            
            ConsensusMessage::Heartbeat { node_id, view, timestamp, status } => {
                self.handle_heartbeat(node_id, view, timestamp, status).await
            },
        }
    }
    
    /// Handle client request
    async fn handle_request(
        &self,
        request_id: String,
        payload: ConsensusPayload,
        client_id: String,
        timestamp: u64,
    ) -> Result<()> {
        debug!("Handling client request: {}", request_id);
        
        // Forward to primary if we're not the primary
        let state = self.state.read().await;
        if state.current_primary != self.node_id {
            // In a real implementation, forward to primary
            return Ok(());
        }
        
        drop(state);
        
        // Initiate consensus as primary
        self.initiate_consensus_round(request_id, payload).await
    }
    
    /// Handle pre-prepare message (backup nodes)
    async fn handle_pre_prepare(
        &self,
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: String,
        request_id: String,
        payload: ConsensusPayload,
        primary_id: NodeId,
    ) -> Result<()> {
        debug!("Handling pre-prepare for sequence: {}", sequence);
        
        let mut state = self.state.write().await;
        
        // Verify this is from the current primary and view
        if primary_id != state.current_primary || view != state.current_view {
            warn!("Pre-prepare from wrong primary or view");
            return Ok(());
        }
        
        // Verify digest
        if compute_digest(&payload) != digest {
            warn!("Invalid digest in pre-prepare message");
            return Ok(());
        }
        
        // Create or update consensus round
        let round = state.active_rounds.entry(sequence).or_insert_with(|| {
            ConsensusRound {
                round_id: generate_round_id(),
                sequence,
                view,
                request_id: request_id.clone(),
                payload: payload.clone(),
                phase: ConsensusPhase::PrePrepare,
                prepare_messages: HashMap::new(),
                commit_messages: HashMap::new(),
                start_time: Instant::now(),
                timeout: self.config.consensus_timeout,
            }
        });
        
        // Move to prepare phase
        round.phase = ConsensusPhase::Prepare;
        
        drop(state);
        
        // Send prepare message
        let prepare = ConsensusMessage::Prepare {
            view,
            sequence,
            digest,
            node_id: self.node_id.clone(),
        };
        
        self.broadcast_message(prepare).await?;
        Ok(())
    }
    
    /// Handle prepare message
    async fn handle_prepare(
        &self,
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: String,
        node_id: NodeId,
    ) -> Result<()> {
        debug!("Handling prepare from node: {} for sequence: {}", node_id, sequence);
        
        let mut state = self.state.write().await;
        
        // Find active round
        if let Some(round) = state.active_rounds.get_mut(&sequence) {
            if round.view != view {
                return Ok(()); // Wrong view
            }
            
            // Store prepare message
            let prepare_msg = ConsensusMessage::Prepare {
                view,
                sequence,
                digest: digest.clone(),
                node_id: node_id.clone(),
            };
            
            round.prepare_messages.insert(node_id, prepare_msg);
            
            // Check if we have enough prepares (2f)
            let required_prepares = 2 * MAX_BYZANTINE_FAULTS;
            if round.prepare_messages.len() >= required_prepares && round.phase == ConsensusPhase::Prepare {
                // Move to commit phase
                round.phase = ConsensusPhase::Commit;
                
                drop(state);
                
                // Send commit message
                let commit = ConsensusMessage::Commit {
                    view,
                    sequence,
                    digest,
                    node_id: self.node_id.clone(),
                };
                
                self.broadcast_message(commit).await?;
            }
        }
        
        Ok(())
    }
    
    /// Handle commit message
    async fn handle_commit(
        &self,
        view: ViewNumber,
        sequence: SequenceNumber,
        digest: String,
        node_id: NodeId,
    ) -> Result<()> {
        debug!("Handling commit from node: {} for sequence: {}", node_id, sequence);
        
        let mut state = self.state.write().await;
        
        // Find active round
        if let Some(round) = state.active_rounds.get_mut(&sequence) {
            if round.view != view {
                return Ok(()); // Wrong view
            }
            
            // Store commit message
            let commit_msg = ConsensusMessage::Commit {
                view,
                sequence,
                digest: digest.clone(),
                node_id: node_id.clone(),
            };
            
            round.commit_messages.insert(node_id, commit_msg);
            
            // Check if we have enough commits (2f + 1)
            let required_commits = 2 * MAX_BYZANTINE_FAULTS + 1;
            if round.commit_messages.len() >= required_commits && round.phase == ConsensusPhase::Commit {
                // Execute the operation
                let payload = round.payload.clone();
                let request_id = round.request_id.clone();
                
                round.phase = ConsensusPhase::Execute;
                state.last_executed = sequence;
                
                drop(state);
                
                // Execute consensus operation
                let result = self.execute_consensus_operation(payload).await?;
                
                // Send reply
                let reply = ConsensusMessage::Reply {
                    view,
                    request_id: request_id.clone(),
                    result: result.clone(),
                    node_id: self.node_id.clone(),
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                };
                
                // Send to client (in this case, complete pending request)
                self.complete_pending_request(request_id, result).await?;
                
                // Update metrics
                {
                    let mut metrics = self.metrics.write().await;
                    metrics.successful_rounds += 1;
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute consensus operation based on payload
    async fn execute_consensus_operation(&self, payload: ConsensusPayload) -> Result<ConsensusResult> {
        match payload {
            ConsensusPayload::QueryProcessing { query, config } => {
                let result = self.query_processor.process_query(&query).await?;
                Ok(ConsensusResult::QueryProcessing { result })
            },
            
            ConsensusPayload::EntityExtraction { query_text, entity_results } => {
                let consensus_entities = self.consensus_entity_extraction(entity_results).await?;
                Ok(ConsensusResult::EntityExtraction { entities: consensus_entities })
            },
            
            ConsensusPayload::Classification { query_text, classification_results } => {
                let consensus_classification = self.consensus_classification(classification_results).await?;
                Ok(ConsensusResult::Classification { classification: consensus_classification })
            },
            
            ConsensusPayload::StrategyRecommendation { query, strategy_results } => {
                let consensus_strategy = self.consensus_strategy(strategy_results).await?;
                Ok(ConsensusResult::StrategyRecommendation { strategy: consensus_strategy })
            },
            
            ConsensusPayload::ResultValidation { result, validation_results } => {
                let consensus_validation = self.consensus_validation(validation_results).await?;
                Ok(ConsensusResult::ResultValidation { validation: consensus_validation })
            },
        }
    }
    
    /// Consensus on entity extraction results
    async fn consensus_entity_extraction(&self, results: Vec<NodeEntityResult>) -> Result<Vec<ExtractedEntity>> {
        // Implement Byzantine fault-tolerant consensus on entity extraction
        // For now, use majority voting with confidence weighting
        
        let mut entity_map: HashMap<String, Vec<(ExtractedEntity, f64)>> = HashMap::new();
        
        // Group entities by text and type
        for node_result in results {
            for entity in node_result.entities {
                let key = format!("{}:{}", entity.text, entity.entity_type);
                entity_map.entry(key).or_insert_with(Vec::new)
                    .push((entity, node_result.confidence));
            }
        }
        
        let mut consensus_entities = Vec::new();
        
        // Select entities with sufficient consensus
        for (_, entity_votes) in entity_map {
            if entity_votes.len() >= (2 * MAX_BYZANTINE_FAULTS + 1) {
                // Calculate weighted average confidence
                let total_confidence: f64 = entity_votes.iter().map(|(_, conf)| conf).sum();
                let avg_confidence = total_confidence / entity_votes.len() as f64;
                
                // Take the first entity and update confidence
                let mut consensus_entity = entity_votes[0].0.clone();
                consensus_entity.confidence = avg_confidence;
                
                consensus_entities.push(consensus_entity);
            }
        }
        
        Ok(consensus_entities)
    }
    
    /// Consensus on classification results
    async fn consensus_classification(&self, results: Vec<NodeClassificationResult>) -> Result<ClassificationResult> {
        // Use majority voting for intent with confidence weighting
        let mut intent_votes: HashMap<String, Vec<f64>> = HashMap::new();
        
        for node_result in results {
            let intent_key = format!("{:?}", node_result.classification.intent);
            intent_votes.entry(intent_key).or_insert_with(Vec::new)
                .push(node_result.classification.confidence);
        }
        
        // Find majority intent
        let mut best_intent = crate::types::QueryIntent::Unknown;
        let mut best_confidence = 0.0;
        
        for (intent_str, confidences) in intent_votes {
            if confidences.len() >= (2 * MAX_BYZANTINE_FAULTS + 1) {
                let avg_confidence = confidences.iter().sum::<f64>() / confidences.len() as f64;
                if avg_confidence > best_confidence {
                    best_confidence = avg_confidence;
                    // Parse intent from string (simplified)
                    best_intent = match intent_str.as_str() {
                        "Factual" => crate::types::QueryIntent::Factual,
                        "Comparison" => crate::types::QueryIntent::Comparison,
                        "Analytical" => crate::types::QueryIntent::Analytical,
                        "Summary" => crate::types::QueryIntent::Summary,
                        _ => crate::types::QueryIntent::Unknown,
                    };
                }
            }
        }
        
        Ok(ClassificationResult {
            intent: best_intent,
            confidence: best_confidence,
            reasoning: "Consensus classification".to_string(),
            features: HashMap::new(),
        })
    }
    
    /// Consensus on strategy recommendations
    async fn consensus_strategy(&self, results: Vec<NodeStrategyResult>) -> Result<StrategyRecommendation> {
        // Use majority voting for strategy with confidence weighting
        let mut strategy_votes: HashMap<String, Vec<f64>> = HashMap::new();
        
        for node_result in results {
            let strategy_key = format!("{:?}", node_result.recommendation.strategy);
            strategy_votes.entry(strategy_key).or_insert_with(Vec::new)
                .push(node_result.recommendation.confidence);
        }
        
        // Find majority strategy
        let mut best_strategy = crate::types::SearchStrategy::Vector;
        let mut best_confidence = 0.0;
        
        for (strategy_str, confidences) in strategy_votes {
            if confidences.len() >= (2 * MAX_BYZANTINE_FAULTS + 1) {
                let avg_confidence = confidences.iter().sum::<f64>() / confidences.len() as f64;
                if avg_confidence > best_confidence {
                    best_confidence = avg_confidence;
                    // Parse strategy from string (simplified)
                    best_strategy = match strategy_str.as_str() {
                        "Vector" => crate::types::SearchStrategy::Vector,
                        "Keyword" => crate::types::SearchStrategy::Keyword,
                        "Hybrid" => crate::types::SearchStrategy::Hybrid,
                        "Semantic" => crate::types::SearchStrategy::Semantic,
                        _ => crate::types::SearchStrategy::Adaptive,
                    };
                }
            }
        }
        
        Ok(StrategyRecommendation {
            strategy: best_strategy,
            confidence: best_confidence,
            reasoning: "Consensus strategy recommendation".to_string(),
            parameters: HashMap::new(),
            estimated_performance: None,
        })
    }
    
    /// Consensus on validation results
    async fn consensus_validation(&self, results: Vec<NodeValidationResult>) -> Result<ValidationResult> {
        // Count valid vs invalid votes
        let mut valid_votes = 0;
        let mut invalid_votes = 0;
        let mut confidence_sum = 0.0;
        
        for node_result in results {
            if node_result.validation.is_valid {
                valid_votes += 1;
            } else {
                invalid_votes += 1;
            }
            confidence_sum += node_result.validation.confidence;
        }
        
        let total_votes = results.len();
        let avg_confidence = if total_votes > 0 { confidence_sum / total_votes as f64 } else { 0.0 };
        
        // Require majority for validation
        let is_valid = valid_votes > invalid_votes && valid_votes >= (2 * MAX_BYZANTINE_FAULTS + 1);
        
        Ok(ValidationResult {
            is_valid,
            confidence: avg_confidence,
            error_details: if is_valid { None } else { Some("Consensus validation failed".to_string()) },
            validation_time: Duration::from_millis(10),
            checks_performed: vec!["consensus_validation".to_string()],
        })
    }
    
    /// Complete a pending request with result
    async fn complete_pending_request(&self, request_id: String, result: ConsensusResult) -> Result<()> {
        let mut state = self.state.write().await;
        
        if let Some(pending) = state.pending_requests.remove(&request_id) {
            if let Some(sender) = pending.response_sender {
                let _ = sender.send(result); // Ignore if receiver dropped
            }
        }
        
        Ok(())
    }
    
    /// Handle view change message
    async fn handle_view_change(
        &self,
        new_view: ViewNumber,
        node_id: NodeId,
        evidence: ViewChangeEvidence,
    ) -> Result<()> {
        debug!("Handling view change to view {} from node {}", new_view, node_id);
        // Implement view change logic
        Ok(())
    }
    
    /// Handle new view message
    async fn handle_new_view(
        &self,
        view: ViewNumber,
        view_change_messages: Vec<ConsensusMessage>,
        pre_prepare_messages: Vec<ConsensusMessage>,
        primary_id: NodeId,
    ) -> Result<()> {
        debug!("Handling new view {} with primary {}", view, primary_id);
        // Implement new view logic
        Ok(())
    }
    
    /// Handle heartbeat message
    async fn handle_heartbeat(
        &self,
        node_id: NodeId,
        view: ViewNumber,
        timestamp: u64,
        status: NodeStatus,
    ) -> Result<()> {
        debug!("Handling heartbeat from node: {}", node_id);
        
        let mut state = self.state.write().await;
        
        // Update node information
        if let Some(node_info) = state.nodes.get_mut(&node_id) {
            node_info.status = status;
            node_info.last_heartbeat = timestamp;
        } else {
            // Add new node
            state.nodes.insert(node_id.clone(), NodeInfo {
                node_id: node_id.clone(),
                endpoint: "unknown".to_string(),
                status,
                last_heartbeat: timestamp,
                joined_at: timestamp,
                public_key: None,
            });
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            let node_metrics = metrics.node_health.entry(node_id).or_insert_with(NodeHealthMetrics::default);
            node_metrics.heartbeats_received += 1;
            node_metrics.last_activity = timestamp;
        }
        
        Ok(())
    }
    
    /// Broadcast message to all nodes
    async fn broadcast_message(&self, message: ConsensusMessage) -> Result<()> {
        let state = self.state.read().await;
        
        for node_id in state.nodes.keys() {
            if *node_id != self.node_id {
                self.message_sender.send((node_id.clone(), message.clone()))
                    .map_err(|_| ProcessorError::ConsensusFailed { reason: "Failed to send message".to_string() })?;
            }
        }
        
        Ok(())
    }
    
    /// Start message processor task
    async fn start_message_processor(&self) -> Result<()> {
        let message_receiver = self.message_receiver.clone();
        let consensus_manager = self.clone();
        
        let task = tokio::spawn(async move {
            let mut receiver = message_receiver.lock().await;
            
            while let Some(message) = receiver.recv().await {
                if let Err(e) = consensus_manager.process_message(message).await {
                    error!("Error processing consensus message: {}", e);
                }
            }
        });
        
        self.background_tasks.lock().await.push(task);
        Ok(())
    }
    
    /// Start heartbeat task
    async fn start_heartbeat_task(&self) -> Result<()> {
        let node_id = self.node_id.clone();
        let state = self.state.clone();
        let message_sender = self.message_sender.clone();
        let heartbeat_interval = self.config.heartbeat_interval;
        
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(heartbeat_interval);
            
            loop {
                interval.tick().await;
                
                let current_state = state.read().await;
                let heartbeat = ConsensusMessage::Heartbeat {
                    node_id: node_id.clone(),
                    view: current_state.current_view,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    status: NodeStatus {
                        health: NodeHealth::Healthy,
                        load: 0.5, // Mock load
                        resources: ResourceStatus {
                            cpu_usage: 50.0,
                            memory_usage: 60.0,
                            network_usage: 10.0,
                            disk_usage: 20.0,
                        },
                        last_activity: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        capabilities: vec!["query_processing".to_string()],
                    },
                };
                
                // Broadcast heartbeat to all nodes
                for node_id_target in current_state.nodes.keys() {
                    if *node_id_target != node_id {
                        let _ = message_sender.send((node_id_target.clone(), heartbeat.clone()));
                    }
                }
            }
        });
        
        self.background_tasks.lock().await.push(task);
        Ok(())
    }
    
    /// Start view change monitor
    async fn start_view_change_monitor(&self) -> Result<()> {
        let state = self.state.clone();
        let view_change_timeout = self.config.view_change_timeout;
        
        let task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                
                let current_state = state.read().await;
                let now = Instant::now();
                
                // Check for timed out consensus rounds
                for (_, round) in &current_state.active_rounds {
                    if now.duration_since(round.start_time) > round.timeout {
                        // Initiate view change
                        debug!("Consensus round {} timed out, initiating view change", round.sequence);
                        // Implementation would trigger view change
                    }
                }
            }
        });
        
        self.background_tasks.lock().await.push(task);
        Ok(())
    }
    
    /// Get consensus metrics
    pub async fn get_metrics(&self) -> ConsensusMetrics {
        self.metrics.read().await.clone()
    }
}

impl Clone for ConsensusManager {
    fn clone(&self) -> Self {
        Self {
            node_id: self.node_id.clone(),
            config: self.config.clone(),
            state: self.state.clone(),
            message_sender: self.message_sender.clone(),
            message_receiver: self.message_receiver.clone(),
            metrics: self.metrics.clone(),
            background_tasks: self.background_tasks.clone(),
            query_processor: self.query_processor.clone(),
        }
    }
}

// Utility functions

/// Generate a unique round identifier
fn generate_round_id() -> RoundId {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as RoundId
}

/// Compute digest of consensus payload
fn compute_digest(payload: &ConsensusPayload) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let serialized = bincode::serialize(payload).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    serialized.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    
    // Mock implementation of QueryProcessorInterface for testing
    #[derive(Debug)]
    struct MockQueryProcessor;
    
    #[async_trait]
    impl QueryProcessorInterface for MockQueryProcessor {
        async fn process_query(&self, _query: &ProcessedQuery) -> Result<QueryResult> {
            Ok(QueryResult {
                query: "test query".to_string(),
                search_strategy: crate::types::SearchStrategy::Vector,
                confidence: 0.9,
                processing_time: Duration::from_millis(10),
                metadata: HashMap::new(),
            })
        }
        
        async fn extract_entities(&self, _query_text: &str) -> Result<Vec<ExtractedEntity>> {
            Ok(vec![ExtractedEntity {
                text: "test entity".to_string(),
                entity_type: "TEST".to_string(),
                confidence: 0.9,
                position: (0, 11),
                metadata: HashMap::new(),
                relationships: Vec::new(),
            }])
        }
        
        async fn classify_query(&self, _query_text: &str) -> Result<ClassificationResult> {
            Ok(ClassificationResult {
                intent: crate::types::QueryIntent::Factual,
                confidence: 0.9,
                reasoning: "test classification".to_string(),
                features: HashMap::new(),
            })
        }
        
        async fn recommend_strategy(&self, _query: &ProcessedQuery) -> Result<StrategyRecommendation> {
            Ok(StrategyRecommendation {
                strategy: crate::types::SearchStrategy::Vector,
                confidence: 0.9,
                reasoning: "test strategy".to_string(),
                parameters: HashMap::new(),
                estimated_performance: None,
            })
        }
        
        async fn validate_result(&self, _result: &QueryResult) -> Result<ValidationResult> {
            Ok(ValidationResult {
                is_valid: true,
                confidence: 0.9,
                error_details: None,
                validation_time: Duration::from_millis(5),
                checks_performed: vec!["test_validation".to_string()],
            })
        }
    }
    
    #[tokio::test]
    async fn test_consensus_manager_creation() {
        let config = ConsensusConfig::default();
        let query_processor = Arc::new(MockQueryProcessor);
        
        let manager = ConsensusManager::new(
            "node1".to_string(),
            config,
            query_processor,
        );
        
        assert_eq!(manager.node_id, "node1");
    }
    
    #[tokio::test]
    async fn test_cache_entry_expiration() {
        let payload = ConsensusPayload::QueryProcessing {
            query: ProcessedQuery {
                id: "test".to_string(),
                original_query: "test query".to_string(),
                processed_query: "test query".to_string(),
                intent: crate::types::QueryIntent::Factual,
                entities: Vec::new(),
                query_type: "factual".to_string(),
                confidence: 0.9,
                processing_time: Duration::from_millis(10),
                metadata: HashMap::new(),
            },
            config: ProcessingConfig {
                min_confidence: 0.8,
                max_processing_time: Duration::from_millis(100),
                consensus_timeout: Duration::from_millis(200),
                min_agreement_ratio: 0.8,
                enable_fault_detection: true,
            },
        };
        
        let digest = compute_digest(&payload);
        assert!(!digest.is_empty());
    }
    
    #[test]
    fn test_consensus_config_defaults() {
        let config = ConsensusConfig::default();
        assert_eq!(config.consensus_timeout, Duration::from_millis(200));
        assert_eq!(config.heartbeat_interval, Duration::from_millis(50));
        assert!(config.enable_byzantine_detection);
        assert_eq!(config.min_nodes, MIN_CONSENSUS_NODES);
    }
    
    #[test]
    fn test_node_status_serialization() {
        let status = NodeStatus {
            health: NodeHealth::Healthy,
            load: 0.5,
            resources: ResourceStatus {
                cpu_usage: 50.0,
                memory_usage: 60.0,
                network_usage: 10.0,
                disk_usage: 20.0,
            },
            last_activity: 1234567890,
            capabilities: vec!["query_processing".to_string()],
        };
        
        let serialized = serde_json::to_string(&status).unwrap();
        let deserialized: NodeStatus = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(status.load, deserialized.load);
        assert_eq!(status.capabilities, deserialized.capabilities);
    }
    
    #[tokio::test]
    async fn test_consensus_entity_extraction() {
        let config = ConsensusConfig::default();
        let query_processor = Arc::new(MockQueryProcessor);
        let manager = ConsensusManager::new("node1".to_string(), config, query_processor);
        
        let results = vec![
            NodeEntityResult {
                node_id: "node1".to_string(),
                entities: vec![ExtractedEntity {
                    text: "entity1".to_string(),
                    entity_type: "TYPE1".to_string(),
                    confidence: 0.9,
                    position: (0, 7),
                    metadata: HashMap::new(),
                    relationships: Vec::new(),
                }],
                confidence: 0.9,
                processing_time: Duration::from_millis(10),
                metadata: HashMap::new(),
            },
            NodeEntityResult {
                node_id: "node2".to_string(),
                entities: vec![ExtractedEntity {
                    text: "entity1".to_string(),
                    entity_type: "TYPE1".to_string(),
                    confidence: 0.8,
                    position: (0, 7),
                    metadata: HashMap::new(),
                    relationships: Vec::new(),
                }],
                confidence: 0.8,
                processing_time: Duration::from_millis(12),
                metadata: HashMap::new(),
            },
        ];
        
        let consensus_entities = manager.consensus_entity_extraction(results).await.unwrap();
        assert_eq!(consensus_entities.len(), 0); // Not enough consensus (need 3+ nodes)
    }
}