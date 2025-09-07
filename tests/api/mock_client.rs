//! Mock HTTP Client for API Testing
//! 
//! Provides mock implementations for testing API endpoints without
//! requiring full system dependencies.

use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct MockApiClient {
    base_url: String,
    timeout: Duration,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MockQueryRequest {
    pub doc_id: String,
    pub question: String,
    pub require_consensus: bool,
    pub user_id: Option<Uuid>,
    pub intent_analysis: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MockQueryResponse {
    pub answer: String,
    pub citations: Vec<MockCitation>,
    pub confidence: f64,
    pub doc_id: String,
    pub question: String,
    pub processing_time_ms: u128,
    pub cache_hit: bool,
    pub pipeline: MockPipelineMetadata,
    pub consensus: MockConsensusResult,
    pub intent: Option<MockIntentResult>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MockCitation {
    pub source: String,
    pub page: u32,
    pub relevance: f64,
    pub text: String,
    pub author: Option<String>,
    pub year: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MockPipelineMetadata {
    pub pattern: String,
    pub steps: Vec<String>,
    pub mrap_executed: bool,
    pub performance: MockPerformanceMetrics,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MockPerformanceMetrics {
    pub cache_ms: Option<u128>,
    pub neural_ms: Option<u128>,
    pub consensus_ms: Option<u128>,
    pub total_ms: u128,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MockConsensusResult {
    pub validated: bool,
    pub threshold: f64,
    pub agreement_percentage: f64,
    pub byzantine_count: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MockIntentResult {
    pub intent_type: String,
    pub confidence: f64,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MockUploadResponse {
    pub id: String,
    pub status: String,
    pub message: String,
    pub chunks: usize,
    pub facts: usize,
    pub processor: String,
}

impl MockApiClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            timeout: Duration::from_secs(5),
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub async fn query(&self, request: MockQueryRequest) -> Result<MockQueryResponse, String> {
        let start = Instant::now();
        
        // Simulate processing delay based on request complexity
        let processing_delay = if request.require_consensus { 400 } else { 150 };
        tokio::time::sleep(Duration::from_millis(processing_delay)).await;
        
        let total_ms = start.elapsed().as_millis();
        
        // Generate mock response based on request
        let mock_response = MockQueryResponse {
            answer: self.generate_mock_answer(&request.question),
            citations: self.generate_mock_citations(&request.doc_id),
            confidence: 0.85,
            doc_id: request.doc_id.clone(),
            question: request.question.clone(),
            processing_time_ms: total_ms,
            cache_hit: false,
            pipeline: MockPipelineMetadata {
                pattern: "DAA→FACT→ruv-FANN→DAA→ruv-FANN→Byzantine→FACT".to_string(),
                steps: vec![
                    "DAA_MRAP_Monitor".to_string(),
                    "DAA_MRAP_Reason".to_string(),
                    "Cache_Check".to_string(),
                    "ruv-FANN_Intent_Analysis".to_string(),
                    "DAA_Multi_Agent_Processing".to_string(),
                    "ruv-FANN_Reranking".to_string(),
                    "DAA_Byzantine_Consensus".to_string(),
                    "Citation_Assembly".to_string(),
                ],
                mrap_executed: true,
                performance: MockPerformanceMetrics {
                    cache_ms: Some(25),
                    neural_ms: Some(150),
                    consensus_ms: if request.require_consensus { Some(350) } else { None },
                    total_ms,
                },
            },
            consensus: MockConsensusResult {
                validated: request.require_consensus,
                threshold: 0.67,
                agreement_percentage: if request.require_consensus { 85.0 } else { 0.0 },
                byzantine_count: 0,
            },
            intent: if request.intent_analysis.unwrap_or(false) {
                Some(MockIntentResult {
                    intent_type: self.classify_question_intent(&request.question),
                    confidence: 0.9,
                    parameters: serde_json::json!({"analyzed": true}),
                })
            } else {
                None
            },
        };

        Ok(mock_response)
    }

    pub async fn upload(&self, content: &[u8], filename: &str) -> Result<MockUploadResponse, String> {
        let start = Instant::now();
        
        // Simulate upload processing
        tokio::time::sleep(Duration::from_millis(200)).await;
        
        let chunks = (content.len() / 512).max(1); // Simulate chunking
        let facts = chunks * 2; // Simulate fact extraction
        
        Ok(MockUploadResponse {
            id: format!("mock_doc_{}", Uuid::new_v4()),
            status: "processed".to_string(),
            message: format!("Mock document {} processed successfully", filename),
            chunks,
            facts,
            processor: "ruv-fann-mock-v0.1.6".to_string(),
        })
    }

    pub async fn health_check(&self) -> Result<serde_json::Value, String> {
        Ok(serde_json::json!({
            "status": "ok",
            "service": "api-mock",
            "version": "1.0.0-test",
            "dependencies": {
                "neural": "active",
                "orchestration": "active", 
                "cache": "active"
            }
        }))
    }

    fn generate_mock_answer(&self, question: &str) -> String {
        if question.to_lowercase().contains("machine learning") {
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without explicit programming.".to_string()
        } else if question.to_lowercase().contains("neural") {
            "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.".to_string()
        } else if question.to_lowercase().contains("types") {
            "There are three main types: supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error).".to_string()
        } else {
            format!("Based on analysis of the document, here is the answer to '{}': The document provides relevant information processed through neural analysis and validated via consensus mechanisms.", question)
        }
    }

    fn generate_mock_citations(&self, doc_id: &str) -> Vec<MockCitation> {
        vec![
            MockCitation {
                source: format!("Document {}", doc_id),
                page: 1,
                relevance: 0.92,
                text: "This citation provides strong evidence for the main point discussed in the answer.".to_string(),
                author: Some("AI Researcher".to_string()),
                year: Some(2023),
            },
            MockCitation {
                source: format!("Document {}", doc_id),
                page: 2,
                relevance: 0.78,
                text: "Additional supporting evidence from the document that reinforces the primary claims.".to_string(),
                author: Some("Data Scientist".to_string()),
                year: Some(2023),
            },
        ]
    }

    fn classify_question_intent(&self, question: &str) -> String {
        let q_lower = question.to_lowercase();
        
        if q_lower.starts_with("what") {
            "factual_question"
        } else if q_lower.starts_with("define") || q_lower.contains("definition") {
            "definition_request"
        } else if q_lower.starts_with("explain") || q_lower.starts_with("how") {
            "explanation_request"
        } else if q_lower.contains("compare") || q_lower.contains("difference") {
            "comparison_request"
        } else {
            "general_query"
        }.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_client_query() {
        let client = MockApiClient::new("http://mock-api".to_string());
        
        let request = MockQueryRequest {
            doc_id: "test-doc".to_string(),
            question: "What is machine learning?".to_string(),
            require_consensus: true,
            user_id: Some(Uuid::new_v4()),
            intent_analysis: Some(true),
        };

        let result = client.query(request).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert!(!response.answer.is_empty());
        assert_eq!(response.citations.len(), 2);
        assert!(response.consensus.validated);
        assert!(response.intent.is_some());
    }

    #[tokio::test]
    async fn test_mock_client_upload() {
        let client = MockApiClient::new("http://mock-api".to_string());
        let content = b"Test document content for upload";
        
        let result = client.upload(content, "test.pdf").await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert!(!response.id.is_empty());
        assert_eq!(response.status, "processed");
        assert!(response.chunks > 0);
    }

    #[tokio::test]
    async fn test_intent_classification() {
        let client = MockApiClient::new("http://mock-api".to_string());
        
        let test_cases = vec![
            ("What is AI?", "factual_question"),
            ("Define neural networks", "definition_request"),
            ("Explain how learning works", "explanation_request"),
            ("Compare supervised and unsupervised", "comparison_request"),
            ("Tell me about data", "general_query"),
        ];

        for (question, expected) in test_cases {
            let intent = client.classify_question_intent(question);
            assert_eq!(intent, expected, "Failed for question: {}", question);
        }
    }

    #[tokio::test]
    async fn test_response_time_simulation() {
        let client = MockApiClient::new("http://mock-api".to_string());
        
        let consensus_request = MockQueryRequest {
            doc_id: "test".to_string(),
            question: "Test consensus timing".to_string(),
            require_consensus: true,
            user_id: Some(Uuid::new_v4()),
            intent_analysis: Some(false),
        };

        let start = Instant::now();
        let result = client.query(consensus_request).await;
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        let response = result.unwrap();
        
        // Consensus requests should take longer
        assert!(duration.as_millis() >= 400);
        assert!(response.pipeline.performance.consensus_ms.is_some());
        assert!(response.consensus.validated);
    }
}