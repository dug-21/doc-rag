//! Mock implementations for API integration tests
//!
//! Comprehensive mocks that comply with architecture requirements:
//! - Query → DAA → FACT → ruv-FANN → Consensus → Response
//! - <2s end-to-end response time
//! - All systems properly mocked for testing

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;
use tokio::sync::{RwLock, Mutex};
use serde_json::Value;

use response_generator::{
    Citation, CitationQualityMetrics, FACTCitationProvider, Result as ResponseResult,
    ResponseError, TextRange, Source
};

/// Mock FACT Citation Provider that implements the required trait
#[derive(Debug)]
pub struct MockFACTCitationProvider {
    pub cached_citations: Arc<RwLock<HashMap<String, Vec<Citation>>>>,
    pub call_log: Arc<Mutex<Vec<String>>>,
}

impl MockFACTCitationProvider {
    pub fn new() -> Self {
        Self {
            cached_citations: Arc::new(RwLock::new(HashMap::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn add_mock_citations(&self, key: &str, citations: Vec<Citation>) {
        self.cached_citations.write().await.insert(key.to_string(), citations);
    }

    pub async fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().await.clone()
    }
}

#[async_trait::async_trait]
impl FACTCitationProvider for MockFACTCitationProvider {
    async fn get_cached_citations(&self, key: &str) -> ResponseResult<Option<Vec<Citation>>> {
        self.call_log.lock().await.push(format!("get_cached_citations: {}", key));
        
        let cache = self.cached_citations.read().await;
        Ok(cache.get(key).cloned())
    }

    async fn store_citations(&self, key: &str, citations: &[Citation]) -> ResponseResult<()> {
        self.call_log.lock().await.push(format!("store_citations: {} citations for {}", citations.len(), key));
        
        let mut cache = self.cached_citations.write().await;
        cache.insert(key.to_string(), citations.to_vec());
        Ok(())
    }

    async fn validate_citation_quality(&self, citation: &Citation) -> ResponseResult<CitationQualityMetrics> {
        self.call_log.lock().await.push(format!("validate_citation_quality: {}", citation.id));
        
        // Mock quality validation based on citation properties
        let quality_score = if citation.relevance_score > 0.8 
            && citation.source.metadata.contains_key("peer_reviewed") 
        {
            0.9
        } else if citation.relevance_score > 0.6 {
            0.7
        } else {
            0.4
        };

        Ok(CitationQualityMetrics {
            overall_quality_score: quality_score,
            source_authority_score: 0.8,
            relevance_score: citation.relevance_score,
            completeness_score: if citation.supporting_text.is_some() { 0.9 } else { 0.5 },
            recency_score: 0.8,
            peer_review_factor: if citation.source.metadata.contains_key("peer_reviewed") { 0.2 } else { 0.0 },
            impact_factor: 0.15,
            citation_count_factor: 0.1,
            author_credibility_factor: 0.1,
            passed_quality_threshold: quality_score > 0.8,
        })
    }

    async fn deduplicate_citations(&self, citations: Vec<Citation>) -> ResponseResult<Vec<Citation>> {
        self.call_log.lock().await.push(format!("deduplicate_citations: {} input citations", citations.len()));
        
        let mut unique_citations = Vec::new();
        let mut seen_sources = std::collections::HashSet::new();

        for citation in citations {
            let source_key = format!("{}:{}", citation.source.title, citation.text_range.start);
            if !seen_sources.contains(&source_key) {
                seen_sources.insert(source_key);
                unique_citations.push(citation);
            }
        }

        Ok(unique_citations)
    }

    async fn optimize_citation_chain(&self, chain: &response_generator::CitationChain) -> ResponseResult<response_generator::CitationChain> {
        self.call_log.lock().await.push(format!("optimize_citation_chain: {} levels", chain.levels));
        
        // Mock optimization - return the same chain
        Ok(chain.clone())
    }
}

/// Mock DAA Orchestrator for testing
#[derive(Debug, Clone)]
pub struct MockDAAOrchestrator {
    pub decisions: Arc<RwLock<HashMap<String, bool>>>,
    pub registered_components: Arc<RwLock<Vec<String>>>,
    pub call_log: Arc<Mutex<Vec<String>>>,
}

impl MockDAAOrchestrator {
    pub fn new() -> Self {
        Self {
            decisions: Arc::new(RwLock::new(HashMap::new())),
            registered_components: Arc::new(RwLock::new(Vec::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn set_consensus_decision(&self, proposal: &str, decision: bool) {
        self.decisions.write().await.insert(proposal.to_string(), decision);
    }

    pub async fn consensus_decision(&self, proposal: &str) -> bool {
        self.call_log.lock().await.push(format!("consensus_decision: {}", proposal));
        
        let decisions = self.decisions.read().await;
        *decisions.get(proposal).unwrap_or(&true) // Default to approval
    }

    pub async fn register_component(&self, name: &str) {
        self.call_log.lock().await.push(format!("register_component: {}", name));
        self.registered_components.write().await.push(name.to_string());
    }

    pub async fn get_registered_components(&self) -> Vec<String> {
        self.registered_components.read().await.clone()
    }

    pub async fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().await.clone()
    }
}

/// Mock ruv-FANN Neural Network provider
#[derive(Debug, Clone)]
pub struct MockRuvFANNProvider {
    pub processing_times: Arc<RwLock<Vec<Duration>>>,
    pub call_log: Arc<Mutex<Vec<String>>>,
}

impl MockRuvFANNProvider {
    pub fn new() -> Self {
        Self {
            processing_times: Arc::new(RwLock::new(Vec::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn process_neural_query(&self, query: &str) -> MockNeuralResult {
        let start = Instant::now();
        self.call_log.lock().await.push(format!("process_neural_query: {}", query));

        // Simulate neural processing time (< 200ms for architecture compliance)
        tokio::time::sleep(Duration::from_millis(150)).await;

        let processing_time = start.elapsed();
        self.processing_times.write().await.push(processing_time);

        MockNeuralResult {
            confidence: 0.87,
            embeddings: vec![0.1, 0.2, 0.3, 0.4, 0.5], // Mock embedding vector
            semantic_tags: vec!["document".to_string(), "processing".to_string()],
            processing_time,
        }
    }

    pub async fn get_average_processing_time(&self) -> Option<Duration> {
        let times = self.processing_times.read().await;
        if times.is_empty() {
            None
        } else {
            let total_millis: u128 = times.iter().map(|d| d.as_millis()).sum();
            Some(Duration::from_millis((total_millis / times.len() as u128) as u64))
        }
    }

    pub async fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().await.clone()
    }
}

#[derive(Debug, Clone)]
pub struct MockNeuralResult {
    pub confidence: f64,
    pub embeddings: Vec<f32>,
    pub semantic_tags: Vec<String>,
    pub processing_time: Duration,
}

/// Mock Byzantine Consensus system
#[derive(Debug, Clone)]
pub struct MockByzantineConsensus {
    pub nodes: Arc<RwLock<Vec<String>>>,
    pub proposals: Arc<RwLock<HashMap<String, MockProposal>>>,
    pub call_log: Arc<Mutex<Vec<String>>>,
    pub consensus_threshold: f64, // 66% for Byzantine fault tolerance
}

impl MockByzantineConsensus {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(vec![
                "node_1".to_string(),
                "node_2".to_string(),
                "node_3".to_string(),
                "node_4".to_string(),
            ])),
            proposals: Arc::new(RwLock::new(HashMap::new())),
            call_log: Arc::new(Mutex::new(Vec::new())),
            consensus_threshold: 0.66, // 66% threshold for Byzantine consensus
        }
    }

    pub async fn submit_proposal(&self, id: &str, data: Value) -> bool {
        self.call_log.lock().await.push(format!("submit_proposal: {}", id));

        let proposal = MockProposal {
            id: id.to_string(),
            data,
            votes: HashMap::new(),
            created_at: Instant::now(),
        };

        // Simulate voting process
        let nodes = self.nodes.read().await;
        let mut votes = HashMap::new();

        // Simulate node votes (most approve)
        for (i, node) in nodes.iter().enumerate() {
            let vote = i < (nodes.len() as f64 * 0.8) as usize; // 80% approval rate
            votes.insert(node.clone(), vote);
        }

        let proposal_with_votes = MockProposal {
            votes,
            ..proposal
        };

        let consensus_reached = self.calculate_consensus(&proposal_with_votes).await;
        
        self.proposals.write().await.insert(id.to_string(), proposal_with_votes);
        
        consensus_reached
    }

    async fn calculate_consensus(&self, proposal: &MockProposal) -> bool {
        let total_nodes = proposal.votes.len();
        let approvals = proposal.votes.values().filter(|&&vote| vote).count();
        
        let approval_rate = approvals as f64 / total_nodes as f64;
        approval_rate >= self.consensus_threshold
    }

    pub async fn get_proposal_status(&self, id: &str) -> Option<MockProposalStatus> {
        let proposals = self.proposals.read().await;
        if let Some(proposal) = proposals.get(id) {
            let total_votes = proposal.votes.len();
            let approvals = proposal.votes.values().filter(|&&vote| vote).count();
            let consensus_reached = self.calculate_consensus(proposal).await;

            Some(MockProposalStatus {
                id: proposal.id.clone(),
                total_votes,
                approvals,
                consensus_reached,
                processing_time: proposal.created_at.elapsed(),
            })
        } else {
            None
        }
    }

    pub async fn get_call_log(&self) -> Vec<String> {
        self.call_log.lock().await.clone()
    }
}

#[derive(Debug, Clone)]
pub struct MockProposal {
    pub id: String,
    pub data: Value,
    pub votes: HashMap<String, bool>,
    pub created_at: Instant,
}

#[derive(Debug, Clone)]
pub struct MockProposalStatus {
    pub id: String,
    pub total_votes: usize,
    pub approvals: usize,
    pub consensus_reached: bool,
    pub processing_time: Duration,
}

/// Mock Pipeline Integration for end-to-end testing
#[derive(Debug)]
pub struct MockPipelineIntegration {
    pub fact_provider: Arc<MockFACTCitationProvider>,
    pub daa_orchestrator: Arc<MockDAAOrchestrator>,
    pub ruv_fann: Arc<MockRuvFANNProvider>,
    pub byzantine_consensus: Arc<MockByzantineConsensus>,
    pub call_log: Arc<Mutex<Vec<String>>>,
}

impl MockPipelineIntegration {
    pub fn new() -> Self {
        Self {
            fact_provider: Arc::new(MockFACTCitationProvider::new()),
            daa_orchestrator: Arc::new(MockDAAOrchestrator::new()),
            ruv_fann: Arc::new(MockRuvFANNProvider::new()),
            byzantine_consensus: Arc::new(MockByzantineConsensus::new()),
            call_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Execute complete pipeline: Query → DAA → FACT → ruv-FANN → Consensus → Response
    pub async fn execute_pipeline(&self, query: &str) -> MockPipelineResult {
        let pipeline_start = Instant::now();
        self.call_log.lock().await.push(format!("execute_pipeline: {}", query));

        // Phase 1: DAA Orchestration
        let daa_start = Instant::now();
        let consensus_decision = self.daa_orchestrator.consensus_decision("process_query").await;
        let daa_time = daa_start.elapsed();

        if !consensus_decision {
            return MockPipelineResult {
                success: false,
                error: Some("DAA consensus rejected query processing".to_string()),
                total_time: pipeline_start.elapsed(),
                component_times: HashMap::new(),
            };
        }

        // Phase 2: FACT Cache Check
        let fact_start = Instant::now();
        let cached_citations = self.fact_provider.get_cached_citations(&format!("query_hash_{}", query.len())).await.unwrap();
        let fact_time = fact_start.elapsed();

        // Phase 3: ruv-FANN Neural Processing
        let neural_start = Instant::now();
        let neural_result = self.ruv_fann.process_neural_query(query).await;
        let neural_time = neural_start.elapsed();

        // Phase 4: Byzantine Consensus for Response Approval
        let consensus_start = Instant::now();
        let proposal_data = serde_json::json!({
            "query": query,
            "neural_confidence": neural_result.confidence,
            "cached_available": cached_citations.is_some()
        });
        let response_approved = self.byzantine_consensus.submit_proposal("response_approval", proposal_data).await;
        let consensus_time = consensus_start.elapsed();

        // Phase 5: Generate Final Response
        let response_start = Instant::now();
        let final_response = if response_approved {
            format!("Processed query: '{}' with confidence: {:.2}", query, neural_result.confidence)
        } else {
            "Query processing rejected by consensus".to_string()
        };
        let response_time = response_start.elapsed();

        let total_time = pipeline_start.elapsed();

        let mut component_times = HashMap::new();
        component_times.insert("daa_orchestration".to_string(), daa_time);
        component_times.insert("fact_cache".to_string(), fact_time);
        component_times.insert("ruv_fann_neural".to_string(), neural_time);
        component_times.insert("byzantine_consensus".to_string(), consensus_time);
        component_times.insert("response_generation".to_string(), response_time);

        MockPipelineResult {
            success: response_approved,
            response: Some(final_response),
            error: if response_approved { None } else { Some("Consensus rejected response".to_string()) },
            neural_confidence: Some(neural_result.confidence),
            citations_found: cached_citations.map(|c| c.len()).unwrap_or(0),
            total_time,
            component_times,
        }
    }

    pub async fn get_full_call_log(&self) -> HashMap<String, Vec<String>> {
        let mut logs = HashMap::new();
        logs.insert("pipeline".to_string(), self.call_log.lock().await.clone());
        logs.insert("fact".to_string(), self.fact_provider.get_call_log().await);
        logs.insert("daa".to_string(), self.daa_orchestrator.get_call_log().await);
        logs.insert("ruv_fann".to_string(), self.ruv_fann.get_call_log().await);
        logs.insert("byzantine".to_string(), self.byzantine_consensus.get_call_log().await);
        logs
    }
}

#[derive(Debug, Clone)]
pub struct MockPipelineResult {
    pub success: bool,
    pub response: Option<String>,
    pub error: Option<String>,
    pub neural_confidence: Option<f64>,
    pub citations_found: usize,
    pub total_time: Duration,
    pub component_times: HashMap<String, Duration>,
}

/// Helper function to create mock citations for testing
pub fn create_mock_citations() -> Vec<Citation> {
    vec![
        Citation {
            id: Uuid::new_v4(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Test Academic Paper".to_string(),
                url: Some("https://example.edu/paper.pdf".to_string()),
                document_type: "academic".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("peer_reviewed".to_string(), "true".to_string());
                    meta.insert("publication_year".to_string(), "2024".to_string());
                    meta
                },
            },
            text_range: TextRange { start: 0, end: 100, length: 100 },
            confidence: 0.9,
            citation_type: response_generator::CitationType::SupportingEvidence,
            relevance_score: 0.85,
            supporting_text: Some("This citation provides strong supporting evidence".to_string()),
        },
        Citation {
            id: Uuid::new_v4(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Technical Documentation".to_string(),
                url: Some("https://docs.example.com/tech-spec".to_string()),
                document_type: "documentation".to_string(),
                metadata: HashMap::new(),
            },
            text_range: TextRange { start: 150, end: 250, length: 100 },
            confidence: 0.8,
            citation_type: response_generator::CitationType::BackgroundContext,
            relevance_score: 0.75,
            supporting_text: Some("Background context for technical understanding".to_string()),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_fact_provider() {
        let provider = MockFACTCitationProvider::new();
        let citations = create_mock_citations();
        
        // Test caching
        provider.store_citations("test_key", &citations).await.unwrap();
        let retrieved = provider.get_cached_citations("test_key").await.unwrap();
        
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().len(), 2);
        
        // Verify call log
        let log = provider.get_call_log().await;
        assert!(log.contains(&"store_citations: 2 citations for test_key".to_string()));
    }

    #[tokio::test]
    async fn test_pipeline_integration_under_2s() {
        let pipeline = MockPipelineIntegration::new();
        
        // Test pipeline execution time
        let result = pipeline.execute_pipeline("What is machine learning?").await;
        
        // Verify <2s requirement
        assert!(result.total_time.as_millis() < 2000, 
                "Pipeline took {}ms, exceeds 2s requirement", 
                result.total_time.as_millis());
        
        assert!(result.success);
        assert!(result.neural_confidence.unwrap() > 0.8);
        
        // Verify all components were called
        let logs = pipeline.get_full_call_log().await;
        assert!(!logs["pipeline"].is_empty());
        assert!(!logs["daa"].is_empty());
        assert!(!logs["fact"].is_empty());
        assert!(!logs["ruv_fann"].is_empty());
        assert!(!logs["byzantine"].is_empty());
    }

    #[tokio::test]
    async fn test_byzantine_consensus_66_percent_threshold() {
        let consensus = MockByzantineConsensus::new();
        
        let proposal_data = serde_json::json!({"test": "proposal"});
        let approved = consensus.submit_proposal("test_proposal", proposal_data).await;
        
        // Should be approved with 80% mock approval rate (above 66% threshold)
        assert!(approved);
        
        let status = consensus.get_proposal_status("test_proposal").await.unwrap();
        assert!(status.consensus_reached);
        assert!(status.approvals as f64 / status.total_votes as f64 >= 0.66);
    }
}