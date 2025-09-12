//! Symbolic Query Router Test Module
//!
//! Comprehensive London TDD test suite for Phase 2 symbolic query routing
//! following test-first development methodology with behavior verification.
//!
//! Test Coverage:
//! - Query routing accuracy (80%+ requirement)
//! - Neural confidence scoring (<10ms constraint)
//! - Logic conversion (Datalog/Prolog)
//! - Proof chain generation and validation
//! - Performance constraints (<100ms routing)
//! - Integration with existing QueryProcessor

pub mod routing_decision_tests;
pub mod confidence_scoring_tests;
pub mod logic_conversion_tests;
pub mod proof_chain_tests;
pub mod performance_tests;
pub mod accuracy_validation_tests;
pub mod engine_selection_tests;
pub mod integration_tests;

// Test fixtures and utilities
pub mod test_fixtures;
pub mod mock_components;
pub mod test_utilities;

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::query_processor::{
    Query, QueryProcessor, SymbolicQueryRouter, RoutingDecision, QueryEngine,
    LogicConversion, ProofChain, QueryCharacteristics, SymbolicQueryType,
    SemanticAnalysis, SyntacticFeatures, SemanticFeatures, NamedEntity,
    ExtractedEntity, QueryIntent, SearchStrategy
};

/// Test fixture for symbolic router testing with mocked dependencies
#[derive(Clone)]
pub struct SymbolicRouterTestFixture {
    /// Router under test
    pub router: Arc<SymbolicQueryRouter>,
    /// Test queries for different engine types
    pub symbolic_queries: Vec<Query>,
    pub graph_queries: Vec<Query>, 
    pub vector_queries: Vec<Query>,
    pub hybrid_queries: Vec<Query>,
    /// Expected routing decisions for accuracy testing
    pub expected_routes: HashMap<String, QueryEngine>,
    /// Performance benchmark queries
    pub benchmark_queries: Vec<Query>,
    /// Mock analysis data
    pub mock_analysis: SemanticAnalysis,
}

impl SymbolicRouterTestFixture {
    /// Create new test fixture with comprehensive test data
    pub async fn new() -> Self {
        let config = crate::query_processor::SymbolicRouterConfig {
            enable_neural_scoring: true,
            target_symbolic_latency_ms: 100,
            min_routing_confidence: 0.8,
            enable_proof_chains: true,
            max_proof_depth: 10,
            enable_performance_monitoring: true,
        };
        
        let router = Arc::new(SymbolicQueryRouter::new(config).await.unwrap());
        
        Self {
            router,
            symbolic_queries: create_symbolic_test_queries(),
            graph_queries: create_graph_test_queries(),
            vector_queries: create_vector_test_queries(),
            hybrid_queries: create_hybrid_test_queries(),
            expected_routes: create_expected_routing_map(),
            benchmark_queries: create_benchmark_queries(),
            mock_analysis: create_mock_semantic_analysis(),
        }
    }
}

/// Create test queries that should route to symbolic engine
pub fn create_symbolic_test_queries() -> Vec<Query> {
    vec![
        Query::new("What encryption is required for cardholder data storage?").unwrap(),
        Query::new("Prove that PCI DSS requires encryption for stored CHD").unwrap(),
        Query::new("If cardholder data is stored, then what security controls are mandatory?").unwrap(),
        Query::new("Demonstrate compliance with PCI DSS requirement 3.4").unwrap(),
        Query::new("What are the logical implications of storing payment card information?").unwrap(),
    ]
}

/// Create test queries that should route to graph engine
pub fn create_graph_test_queries() -> Vec<Query> {
    vec![
        Query::new("What are the relationships between PCI DSS requirements?").unwrap(),
        Query::new("How do encryption requirements connect to access control?").unwrap(),
        Query::new("Show the dependency graph between security controls").unwrap(),
        Query::new("What requirements are related to network security?").unwrap(),
        Query::new("Map the connections between authentication and authorization").unwrap(),
    ]
}

/// Create test queries that should route to vector engine
pub fn create_vector_test_queries() -> Vec<Query> {
    vec![
        Query::new("What is PCI DSS?").unwrap(),
        Query::new("Define cardholder data").unwrap(),
        Query::new("Explain encryption algorithms").unwrap(),
        Query::new("What are the different types of security controls?").unwrap(),
        Query::new("Describe network segmentation").unwrap(),
    ]
}

/// Create test queries that should route to hybrid engine
pub fn create_hybrid_test_queries() -> Vec<Query> {
    vec![
        Query::new("Compare PCI DSS 3.2.1 and 4.0 encryption requirements with compliance implications").unwrap(),
        Query::new("Analyze the security architecture requirements and their logical dependencies").unwrap(),
        Query::new("What are the complex relationships between access control, encryption, and monitoring?").unwrap(),
        Query::new("Evaluate the compliance status and remediation requirements for multi-entity environments").unwrap(),
    ]
}

/// Create expected routing decisions for accuracy testing
pub fn create_expected_routing_map() -> HashMap<String, QueryEngine> {
    let mut map = HashMap::new();
    
    // Symbolic engine expectations
    map.insert("encryption required".to_string(), QueryEngine::Symbolic);
    map.insert("prove that".to_string(), QueryEngine::Symbolic);
    map.insert("if then".to_string(), QueryEngine::Symbolic);
    map.insert("demonstrate compliance".to_string(), QueryEngine::Symbolic);
    map.insert("logical implications".to_string(), QueryEngine::Symbolic);
    
    // Graph engine expectations  
    map.insert("relationships between".to_string(), QueryEngine::Graph);
    map.insert("connect to".to_string(), QueryEngine::Graph);
    map.insert("dependency graph".to_string(), QueryEngine::Graph);
    map.insert("related to".to_string(), QueryEngine::Graph);
    map.insert("connections between".to_string(), QueryEngine::Graph);
    
    // Vector engine expectations
    map.insert("what is".to_string(), QueryEngine::Vector);
    map.insert("define".to_string(), QueryEngine::Vector);
    map.insert("explain".to_string(), QueryEngine::Vector);
    map.insert("describe".to_string(), QueryEngine::Vector);
    
    // Hybrid engine expectations
    map.insert("compare and".to_string(), QueryEngine::Hybrid(vec![QueryEngine::Symbolic, QueryEngine::Graph]));
    map.insert("analyze architecture".to_string(), QueryEngine::Hybrid(vec![QueryEngine::Symbolic, QueryEngine::Graph, QueryEngine::Vector]));
    
    map
}

/// Create benchmark queries for performance testing
pub fn create_benchmark_queries() -> Vec<Query> {
    let mut queries = Vec::new();
    
    // Add various complexity levels for performance testing
    for i in 1..=100 {
        let query_text = format!("What are the PCI DSS encryption requirements for scenario {}?", i);
        queries.push(Query::new(&query_text).unwrap());
    }
    
    queries
}

/// Create mock semantic analysis for testing
pub fn create_mock_semantic_analysis() -> SemanticAnalysis {
    SemanticAnalysis::new(
        SyntacticFeatures {
            pos_tags: vec![],
            named_entities: vec![
                NamedEntity::new(
                    "PCI DSS".to_string(),
                    "STANDARD".to_string(),
                    0, 7, 0.95
                ),
                NamedEntity::new(
                    "encryption".to_string(),
                    "SECURITY_CONTROL".to_string(),
                    8, 18, 0.90
                ),
            ],
            noun_phrases: vec![],
            verb_phrases: vec![],
            question_words: vec!["What".to_string()],
        },
        SemanticFeatures {
            semantic_roles: vec![],
            coreferences: vec![],
            sentiment: None,
            similarity_vectors: vec![],
        },
        vec![], // dependencies
        vec![], // topics
        0.85, // confidence
        Duration::from_millis(25), // processing time
    )
}

/// Calculate routing accuracy for test validation
pub fn calculate_routing_accuracy(results: &[RoutingDecision], expected: &HashMap<String, QueryEngine>) -> f64 {
    let mut correct_decisions = 0;
    let total_decisions = results.len();
    
    for result in results {
        // Simple heuristic matching for test purposes
        for (pattern, expected_engine) in expected {
            if result.reasoning.to_lowercase().contains(pattern) {
                if std::mem::discriminant(&result.engine) == std::mem::discriminant(expected_engine) {
                    correct_decisions += 1;
                }
                break;
            }
        }
    }
    
    correct_decisions as f64 / total_decisions as f64
}

/// Calculate percentile for performance analysis
pub fn calculate_percentile(values: &[Duration], percentile: u8) -> Duration {
    let mut sorted_values: Vec<Duration> = values.to_vec();
    sorted_values.sort();
    
    let index = (percentile as f64 / 100.0 * (sorted_values.len() - 1) as f64) as usize;
    sorted_values[index]
}

/// Performance validation result structure
#[derive(Debug, Clone)]
pub struct RoutingPerformanceResult {
    pub mean_latency: Duration,
    pub p95_latency: Duration,
    pub p99_latency: Duration,
    pub mean_confidence: f64,
    pub passes_latency_constraint: bool,
    pub passes_accuracy_constraint: bool,
}

/// Comprehensive performance validation result
#[derive(Debug, Clone)]
pub struct PerformanceValidationResult {
    pub routing_performance: RoutingPerformanceResult,
    pub routing_accuracy: f64,
    pub template_performance: Duration,
    pub pipeline_performance: Duration,
    pub integration_performance: HashMap<String, Duration>,
    pub load_test_performance: LoadTestResult,
}

impl PerformanceValidationResult {
    pub fn new() -> Self {
        Self {
            routing_performance: RoutingPerformanceResult {
                mean_latency: Duration::from_millis(0),
                p95_latency: Duration::from_millis(0),
                p99_latency: Duration::from_millis(0),
                mean_confidence: 0.0,
                passes_latency_constraint: false,
                passes_accuracy_constraint: false,
            },
            routing_accuracy: 0.0,
            template_performance: Duration::from_millis(0),
            pipeline_performance: Duration::from_millis(0),
            integration_performance: HashMap::new(),
            load_test_performance: LoadTestResult::default(),
        }
    }
}

/// Load testing result structure
#[derive(Debug, Clone, Default)]
pub struct LoadTestResult {
    pub concurrent_users: usize,
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub average_response_time: Duration,
    pub throughput_per_second: f64,
}

#[cfg(test)]
mod basic_fixture_tests {
    use super::*;

    #[tokio::test]
    async fn test_fixture_creation() {
        // Given: Creating test fixture
        let fixture = SymbolicRouterTestFixture::new().await;
        
        // Then: Fixture should be properly initialized
        assert!(!fixture.symbolic_queries.is_empty());
        assert!(!fixture.graph_queries.is_empty());
        assert!(!fixture.vector_queries.is_empty());
        assert!(!fixture.hybrid_queries.is_empty());
        assert!(!fixture.expected_routes.is_empty());
        assert_eq!(fixture.benchmark_queries.len(), 100);
        assert!(fixture.mock_analysis.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_query_creation_utilities() {
        // Given: Creating test queries
        let symbolic_queries = create_symbolic_test_queries();
        let graph_queries = create_graph_test_queries();
        let vector_queries = create_vector_test_queries();
        let hybrid_queries = create_hybrid_test_queries();
        
        // Then: All query sets should be non-empty and valid
        assert_eq!(symbolic_queries.len(), 5);
        assert_eq!(graph_queries.len(), 5);
        assert_eq!(vector_queries.len(), 5);
        assert_eq!(hybrid_queries.len(), 4);
        
        // And: All queries should have valid text
        assert!(symbolic_queries.iter().all(|q| !q.text().is_empty()));
        assert!(graph_queries.iter().all(|q| !q.text().is_empty()));
        assert!(vector_queries.iter().all(|q| !q.text().is_empty()));
        assert!(hybrid_queries.iter().all(|q| !q.text().is_empty()));
    }

    #[tokio::test]
    async fn test_accuracy_calculation_utilities() {
        // Given: Mock routing decisions and expectations
        let decisions = vec![
            RoutingDecision {
                engine: QueryEngine::Symbolic,
                confidence: 0.9,
                reasoning: "encryption required analysis".to_string(),
                expected_performance: Default::default(),
                fallback_engines: vec![],
                timestamp: chrono::Utc::now(),
            }
        ];
        
        let expected = create_expected_routing_map();
        
        // When: Calculating accuracy
        let accuracy = calculate_routing_accuracy(&decisions, &expected);
        
        // Then: Should calculate reasonable accuracy
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[tokio::test]
    async fn test_percentile_calculation() {
        // Given: Sample durations
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];
        
        // When: Calculating percentiles
        let p50 = calculate_percentile(&durations, 50);
        let p95 = calculate_percentile(&durations, 95);
        
        // Then: Should return reasonable percentile values
        assert!(p50.as_millis() >= 20 && p50.as_millis() <= 40);
        assert!(p95.as_millis() >= 40 && p95.as_millis() <= 50);
    }
}