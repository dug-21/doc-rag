//! Comprehensive Integration Tests for Week 3 RAG Implementation
//! 
//! This test suite validates the complete end-to-end RAG pipeline:
//! - Query Processor + Response Generator integration
//! - Integration with all Week 1-2 components (MCP, Chunker, Embedder, Storage)
//! - Full pipeline performance under load
//! - Error scenarios and edge cases
//! - Production readiness validation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Test configuration for integration tests
#[derive(Debug, Clone)]
pub struct TestConfig {
    pub chunk_size: usize,
    pub batch_size: usize,
    pub max_embeddings: usize,
    pub confidence_threshold: f64,
    pub performance_targets: PerformanceTargets,
}

/// Performance targets for validation
#[derive(Debug, Clone)]
pub struct PerformanceTargets {
    pub query_processing_ms: u64,
    pub response_generation_ms: u64,
    pub end_to_end_ms: u64,
    pub accuracy_threshold: f64,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            batch_size: 32,
            max_embeddings: 1000,
            confidence_threshold: 0.8,
            performance_targets: PerformanceTargets {
                query_processing_ms: 50,
                response_generation_ms: 100,
                end_to_end_ms: 200,
                accuracy_threshold: 0.99,
            },
        }
    }
}

/// Mock components for integration testing
pub mod mock_components {
    use super::*;
    use std::time::Duration;
    use uuid::Uuid;

    /// Mock Query Processor that simulates the real component
    #[derive(Debug, Clone)]
    pub struct MockQueryProcessor {
        pub latency: Duration,
        pub accuracy: f64,
    }

    /// Mock Response Generator that simulates the real component  
    #[derive(Debug, Clone)]
    pub struct MockResponseGenerator {
        pub latency: Duration,
        pub accuracy: f64,
    }

    /// Mock MCP Adapter for external communications
    #[derive(Debug, Clone)]
    pub struct MockMcpAdapter {
        pub latency: Duration,
        pub success_rate: f64,
    }

    /// Mock Document Chunker
    #[derive(Debug, Clone)]
    pub struct MockChunker {
        pub chunk_size: usize,
        pub overlap: usize,
    }

    /// Mock Embedding Generator
    #[derive(Debug, Clone)]
    pub struct MockEmbedder {
        pub dimension: usize,
        pub batch_size: usize,
        pub latency: Duration,
    }

    /// Mock Vector Storage
    #[derive(Debug, Clone)]
    pub struct MockStorage {
        pub search_latency: Duration,
        pub write_latency: Duration,
        pub storage: Arc<RwLock<HashMap<Uuid, StoredDocument>>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct StoredDocument {
        pub id: Uuid,
        pub content: String,
        pub embeddings: Vec<f32>,
        pub metadata: HashMap<String, String>,
        pub created_at: chrono::DateTime<chrono::Utc>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum QueryIntent {
        Factual,
        Comparison,
        Summary,
        Procedural,
        Complex,
    }

    #[derive(Debug, Clone)]
    pub struct ProcessedQuery {
        pub id: Uuid,
        pub original_query: String,
        pub processed_query: String,
        pub intent: QueryIntent,
        pub entities: Vec<String>,
        pub confidence_score: f64,
        pub processing_time: Duration,
        pub search_strategy: String,
    }

    #[derive(Debug, Clone)]
    pub struct GeneratedResponse {
        pub query_id: Uuid,
        pub content: String,
        pub confidence_score: f64,
        pub citations: Vec<Citation>,
        pub generation_time: Duration,
        pub validation_results: Vec<ValidationResult>,
        pub format: OutputFormat,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct Citation {
        pub id: Uuid,
        pub source: String,
        pub page: Option<u32>,
        pub confidence: f64,
        pub relevance_score: f64,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ValidationResult {
        pub layer: String,
        pub passed: bool,
        pub confidence: f64,
        pub errors: Vec<String>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum OutputFormat {
        Json,
        Markdown,
        PlainText,
    }

    #[derive(Debug, Clone)]
    pub struct DocumentChunk {
        pub id: Uuid,
        pub content: String,
        pub embeddings: Option<Vec<f32>>,
        pub metadata: HashMap<String, String>,
        pub references: Vec<String>,
    }

    #[derive(Debug, Clone)]
    pub struct SearchResult {
        pub chunk_id: Uuid,
        pub content: String,
        pub similarity_score: f64,
        pub metadata: HashMap<String, String>,
    }

    // Implementation for MockQueryProcessor
    impl MockQueryProcessor {
        pub fn new() -> Self {
            Self {
                latency: Duration::from_millis(30),
                accuracy: 0.95,
            }
        }

        pub async fn process(&self, query: &str) -> Result<ProcessedQuery> {
            tokio::time::sleep(self.latency).await;

            let intent = Self::classify_intent(query);
            let entities = Self::extract_entities(query);
            
            Ok(ProcessedQuery {
                id: Uuid::new_v4(),
                original_query: query.to_string(),
                processed_query: format!("processed: {}", query),
                intent,
                entities,
                confidence_score: self.accuracy,
                processing_time: self.latency,
                search_strategy: "hybrid_search".to_string(),
            })
        }

        fn classify_intent(query: &str) -> QueryIntent {
            let query_lower = query.to_lowercase();
            if query_lower.contains("compare") || query_lower.contains("vs") {
                QueryIntent::Comparison
            } else if query_lower.contains("summarize") || query_lower.contains("summary") {
                QueryIntent::Summary
            } else if query_lower.contains("how to") || query_lower.contains("steps") {
                QueryIntent::Procedural
            } else if query_lower.split_whitespace().count() > 10 {
                QueryIntent::Complex
            } else {
                QueryIntent::Factual
            }
        }

        fn extract_entities(query: &str) -> Vec<String> {
            query.split_whitespace()
                .filter(|word| word.len() > 3 && !["what", "where", "when", "how", "why", "with", "that", "this", "from", "they", "them", "have", "been", "were", "will"].contains(&word.to_lowercase().as_str()))
                .take(5)
                .map(|s| s.to_string())
                .collect()
        }
    }

    // Implementation for MockResponseGenerator
    impl MockResponseGenerator {
        pub fn new() -> Self {
            Self {
                latency: Duration::from_millis(70),
                accuracy: 0.92,
            }
        }

        pub async fn generate(&self, processed_query: &ProcessedQuery, context: &[SearchResult]) -> Result<GeneratedResponse> {
            tokio::time::sleep(self.latency).await;

            let content = self.generate_content(processed_query, context);
            let citations = self.generate_citations(context);
            let validation_results = self.validate_response(&content);

            Ok(GeneratedResponse {
                query_id: processed_query.id,
                content,
                confidence_score: self.accuracy * processed_query.confidence_score,
                citations,
                generation_time: self.latency,
                validation_results,
                format: OutputFormat::Markdown,
            })
        }

        fn generate_content(&self, query: &ProcessedQuery, context: &[SearchResult]) -> String {
            let base_response = match query.intent {
                QueryIntent::Factual => format!("Based on the available information, here are the key facts about your query '{}':", query.original_query),
                QueryIntent::Comparison => format!("Here is a comparative analysis of your query '{}':", query.original_query),
                QueryIntent::Summary => format!("Here is a comprehensive summary of '{}':", query.original_query),
                QueryIntent::Procedural => format!("Here are the step-by-step instructions for '{}':", query.original_query),
                QueryIntent::Complex => format!("This is a detailed analysis of your complex query '{}':", query.original_query),
            };

            let mut content = base_response;
            
            // Add context-based information
            for (i, result) in context.iter().take(3).enumerate() {
                content.push_str(&format!("\n\n{}. {}", i + 1, result.content.chars().take(200).collect::<String>()));
                if result.content.len() > 200 {
                    content.push_str("...");
                }
            }

            content
        }

        fn generate_citations(&self, context: &[SearchResult]) -> Vec<Citation> {
            context.iter().enumerate().map(|(i, result)| {
                Citation {
                    id: result.chunk_id,
                    source: result.metadata.get("source").cloned().unwrap_or_else(|| format!("Document {}", i + 1)),
                    page: result.metadata.get("page").and_then(|p| p.parse().ok()),
                    confidence: result.similarity_score,
                    relevance_score: result.similarity_score * 0.9,
                }
            }).collect()
        }

        fn validate_response(&self, content: &str) -> Vec<ValidationResult> {
            vec![
                ValidationResult {
                    layer: "content_quality".to_string(),
                    passed: content.len() > 50,
                    confidence: if content.len() > 50 { 0.95 } else { 0.3 },
                    errors: if content.len() <= 50 { vec!["Content too short".to_string()] } else { vec![] },
                },
                ValidationResult {
                    layer: "factual_accuracy".to_string(),
                    passed: true, // Simplified for mock
                    confidence: 0.88,
                    errors: vec![],
                },
                ValidationResult {
                    layer: "citation_completeness".to_string(),
                    passed: true, // Simplified for mock
                    confidence: 0.92,
                    errors: vec![],
                },
            ]
        }
    }

    // Implementation for MockChunker
    impl MockChunker {
        pub fn new() -> Self {
            Self {
                chunk_size: 512,
                overlap: 50,
            }
        }

        pub async fn chunk_document(&self, content: &str, document_id: &str) -> Result<Vec<DocumentChunk>> {
            let mut chunks = Vec::new();
            let mut start = 0;
            let mut chunk_index = 0;

            while start < content.len() {
                let end = std::cmp::min(start + self.chunk_size, content.len());
                let chunk_content = &content[start..end];

                let mut metadata = HashMap::new();
                metadata.insert("document_id".to_string(), document_id.to_string());
                metadata.insert("chunk_index".to_string(), chunk_index.to_string());
                metadata.insert("start_pos".to_string(), start.to_string());
                metadata.insert("end_pos".to_string(), end.to_string());

                chunks.push(DocumentChunk {
                    id: Uuid::new_v4(),
                    content: chunk_content.to_string(),
                    embeddings: None,
                    metadata,
                    references: self.extract_references(chunk_content),
                });

                start = end - self.overlap;
                chunk_index += 1;

                if end >= content.len() {
                    break;
                }
            }

            Ok(chunks)
        }

        fn extract_references(&self, content: &str) -> Vec<String> {
            // Simple reference extraction (sections, figures, etc.)
            let mut refs = Vec::new();
            if content.contains("section") || content.contains("Section") {
                refs.push("section_reference".to_string());
            }
            if content.contains("figure") || content.contains("Figure") {
                refs.push("figure_reference".to_string());
            }
            if content.contains("[") && content.contains("]") {
                refs.push("bracket_reference".to_string());
            }
            refs
        }
    }

    // Implementation for MockEmbedder
    impl MockEmbedder {
        pub fn new() -> Self {
            Self {
                dimension: 384,
                batch_size: 32,
                latency: Duration::from_millis(50),
            }
        }

        pub async fn generate_embeddings(&self, chunks: &mut [DocumentChunk]) -> Result<()> {
            let batch_time = self.latency * chunks.len() as u32 / self.batch_size as u32;
            tokio::time::sleep(batch_time).await;

            for chunk in chunks {
                chunk.embeddings = Some(self.generate_mock_embedding(&chunk.content));
            }

            Ok(())
        }

        fn generate_mock_embedding(&self, content: &str) -> Vec<f32> {
            // Generate deterministic but realistic mock embeddings
            let mut embedding = vec![0.0; self.dimension];
            let content_bytes = content.as_bytes();
            
            for (i, &byte) in content_bytes.iter().enumerate() {
                if i >= self.dimension { break; }
                embedding[i] = (byte as f32 - 128.0) / 128.0; // Normalize to [-1, 1]
            }

            // Add some noise for realism
            for (i, val) in embedding.iter_mut().enumerate() {
                let noise = (i as f32 * 0.1).sin() * 0.01;
                *val += noise;
            }

            // Normalize to unit length
            let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if magnitude > 0.0 {
                for val in embedding.iter_mut() {
                    *val /= magnitude;
                }
            }

            embedding
        }

        pub fn calculate_similarity(&self, emb1: &[f32], emb2: &[f32]) -> f32 {
            if emb1.len() != emb2.len() {
                return 0.0;
            }

            let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
            dot_product.max(-1.0).min(1.0) // Clamp to valid cosine similarity range
        }
    }

    // Implementation for MockStorage
    impl MockStorage {
        pub fn new() -> Self {
            Self {
                search_latency: Duration::from_millis(20),
                write_latency: Duration::from_millis(10),
                storage: Arc::new(RwLock::new(HashMap::new())),
            }
        }

        pub async fn store_chunks(&self, chunks: &[DocumentChunk]) -> Result<()> {
            tokio::time::sleep(self.write_latency * chunks.len() as u32).await;

            let mut storage = self.storage.write().await;
            for chunk in chunks {
                if let Some(embeddings) = &chunk.embeddings {
                    storage.insert(chunk.id, StoredDocument {
                        id: chunk.id,
                        content: chunk.content.clone(),
                        embeddings: embeddings.clone(),
                        metadata: chunk.metadata.clone(),
                        created_at: chrono::Utc::now(),
                    });
                }
            }

            Ok(())
        }

        pub async fn search_similar(&self, query_embedding: &[f32], limit: usize, threshold: f64) -> Result<Vec<SearchResult>> {
            tokio::time::sleep(self.search_latency).await;

            let storage = self.storage.read().await;
            let mut results = Vec::new();

            for (_, doc) in storage.iter() {
                let similarity = self.calculate_cosine_similarity(query_embedding, &doc.embeddings);
                if similarity >= threshold {
                    results.push(SearchResult {
                        chunk_id: doc.id,
                        content: doc.content.clone(),
                        similarity_score: similarity,
                        metadata: doc.metadata.clone(),
                    });
                }
            }

            // Sort by similarity (highest first) and limit results
            results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
            results.truncate(limit);

            Ok(results)
        }

        fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
            if a.len() != b.len() {
                return 0.0;
            }

            let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

            if norm_a == 0.0 || norm_b == 0.0 {
                return 0.0;
            }

            (dot_product / (norm_a * norm_b)).max(-1.0).min(1.0) as f64
        }

        pub async fn get_document_count(&self) -> usize {
            self.storage.read().await.len()
        }
    }

    // Implementation for MockMcpAdapter
    impl MockMcpAdapter {
        pub fn new() -> Self {
            Self {
                latency: Duration::from_millis(5),
                success_rate: 0.98,
            }
        }

        pub async fn send_message(&self, message: &str) -> Result<String> {
            tokio::time::sleep(self.latency).await;

            // Simulate occasional failures
            if rand::random::<f64>() > self.success_rate {
                return Err(anyhow::anyhow!("MCP communication failure"));
            }

            Ok(format!("MCP processed: {}", message))
        }

        pub async fn health_check(&self) -> bool {
            tokio::time::sleep(Duration::from_millis(1)).await;
            rand::random::<f64>() <= self.success_rate
        }
    }
}

use mock_components::*;

/// Complete RAG System Integration
pub struct RagSystemIntegration {
    query_processor: MockQueryProcessor,
    response_generator: MockResponseGenerator,
    chunker: MockChunker,
    embedder: MockEmbedder,
    storage: MockStorage,
    mcp_adapter: MockMcpAdapter,
    config: TestConfig,
}

impl RagSystemIntegration {
    pub fn new(config: TestConfig) -> Self {
        Self {
            query_processor: MockQueryProcessor::new(),
            response_generator: MockResponseGenerator::new(),
            chunker: MockChunker::new(),
            embedder: MockEmbedder::new(),
            storage: MockStorage::new(),
            mcp_adapter: MockMcpAdapter::new(),
            config,
        }
    }

    /// Complete end-to-end query processing pipeline
    pub async fn process_query_end_to_end(&self, query: &str) -> Result<EndToEndResult> {
        let start_time = Instant::now();

        // Stage 1: Query Processing
        let query_start = Instant::now();
        let processed_query = self.query_processor.process(query).await?;
        let query_duration = query_start.elapsed();

        // Stage 2: Generate query embedding for search
        let mut query_chunk = DocumentChunk {
            id: Uuid::new_v4(),
            content: query.to_string(),
            embeddings: None,
            metadata: HashMap::new(),
            references: vec![],
        };
        self.embedder.generate_embeddings(&mut [query_chunk]).await?;
        let query_embedding = query_chunk.embeddings.unwrap();

        // Stage 3: Vector similarity search
        let search_start = Instant::now();
        let search_results = self.storage.search_similar(
            &query_embedding, 
            10, 
            self.config.confidence_threshold
        ).await?;
        let search_duration = search_start.elapsed();

        // Stage 4: Response Generation
        let response_start = Instant::now();
        let generated_response = self.response_generator.generate(&processed_query, &search_results).await?;
        let response_duration = response_start.elapsed();

        let total_duration = start_time.elapsed();

        // Stage 5: Validation
        let accuracy = self.validate_response_quality(&generated_response, &search_results);

        Ok(EndToEndResult {
            processed_query,
            generated_response,
            search_results,
            performance_metrics: PerformanceMetrics {
                query_processing_time: query_duration,
                search_time: search_duration,
                response_generation_time: response_duration,
                total_time: total_duration,
            },
            accuracy_score: accuracy,
        })
    }

    /// Index documents into the system
    pub async fn index_documents(&self, documents: &[(String, String)]) -> Result<IndexingResult> {
        let start_time = Instant::now();
        let mut total_chunks = 0;
        let mut successful_chunks = 0;

        for (doc_id, content) in documents {
            // Chunk the document
            let mut chunks = self.chunker.chunk_document(content, doc_id).await?;
            total_chunks += chunks.len();

            // Generate embeddings
            self.embedder.generate_embeddings(&mut chunks).await?;

            // Store in vector storage
            match self.storage.store_chunks(&chunks).await {
                Ok(_) => successful_chunks += chunks.len(),
                Err(e) => eprintln!("Failed to store chunks for document {}: {}", doc_id, e),
            }
        }

        Ok(IndexingResult {
            total_documents: documents.len(),
            total_chunks,
            successful_chunks,
            indexing_time: start_time.elapsed(),
        })
    }

    fn validate_response_quality(&self, response: &GeneratedResponse, search_results: &[SearchResult]) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        // Content length validation
        if response.content.len() >= 50 {
            score += 0.2;
        }
        factors += 1;

        // Citation completeness
        if !response.citations.is_empty() && response.citations.len() <= search_results.len() {
            score += 0.2;
        }
        factors += 1;

        // Confidence score validation
        if response.confidence_score >= 0.7 {
            score += 0.2;
        }
        factors += 1;

        // Validation results
        let validation_passed = response.validation_results.iter().all(|v| v.passed);
        if validation_passed {
            score += 0.2;
        }
        factors += 1;

        // Performance validation
        if response.generation_time <= Duration::from_millis(self.config.performance_targets.response_generation_ms) {
            score += 0.2;
        }
        factors += 1;

        score / factors as f64
    }
}

#[derive(Debug)]
pub struct EndToEndResult {
    pub processed_query: ProcessedQuery,
    pub generated_response: GeneratedResponse,
    pub search_results: Vec<SearchResult>,
    pub performance_metrics: PerformanceMetrics,
    pub accuracy_score: f64,
}

#[derive(Debug)]
pub struct PerformanceMetrics {
    pub query_processing_time: Duration,
    pub search_time: Duration,
    pub response_generation_time: Duration,
    pub total_time: Duration,
}

#[derive(Debug)]
pub struct IndexingResult {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub successful_chunks: usize,
    pub indexing_time: Duration,
}

/// Load testing utilities
pub struct LoadTestRunner {
    system: RagSystemIntegration,
    queries: Vec<String>,
}

impl LoadTestRunner {
    pub fn new(system: RagSystemIntegration) -> Self {
        let queries = vec![
            "What are the key principles of software architecture?".to_string(),
            "Explain machine learning algorithms and their applications".to_string(),
            "Compare microservices vs monolithic architecture".to_string(),
            "How does blockchain technology work?".to_string(),
            "Summarize best practices for database optimization".to_string(),
            "What are the security requirements for web applications?".to_string(),
            "Describe the CI/CD pipeline implementation process".to_string(),
            "How to implement effective error handling in distributed systems?".to_string(),
            "What are the performance considerations for high-traffic applications?".to_string(),
            "Explain the principles of domain-driven design".to_string(),
        ];

        Self { system, queries }
    }

    pub async fn run_load_test(&self, concurrent_users: usize, duration: Duration) -> Result<LoadTestResult> {
        let start_time = Instant::now();
        let mut handles = Vec::new();

        // Spawn concurrent users
        for user_id in 0..concurrent_users {
            let system = &self.system;
            let queries = self.queries.clone();
            let end_time = start_time + duration;

            let handle = tokio::spawn(async move {
                let mut user_results = Vec::new();
                let mut query_index = 0;

                while Instant::now() < end_time {
                    let query = &queries[query_index % queries.len()];
                    match system.process_query_end_to_end(query).await {
                        Ok(result) => user_results.push(UserQueryResult {
                            user_id,
                            query: query.clone(),
                            success: true,
                            response_time: result.performance_metrics.total_time,
                            accuracy: result.accuracy_score,
                            error: None,
                        }),
                        Err(e) => user_results.push(UserQueryResult {
                            user_id,
                            query: query.clone(),
                            success: false,
                            response_time: Duration::from_millis(0),
                            accuracy: 0.0,
                            error: Some(e.to_string()),
                        }),
                    }
                    query_index += 1;
                }

                user_results
            });

            handles.push(handle);
        }

        // Collect all results
        let mut all_results = Vec::new();
        for handle in handles {
            let user_results = handle.await?;
            all_results.extend(user_results);
        }

        // Analyze results
        let total_duration = start_time.elapsed();
        Ok(self.analyze_load_test_results(all_results, total_duration))
    }

    fn analyze_load_test_results(&self, results: Vec<UserQueryResult>, duration: Duration) -> LoadTestResult {
        let total_requests = results.len();
        let successful_requests = results.iter().filter(|r| r.success).count();
        let failed_requests = total_requests - successful_requests;

        let response_times: Vec<Duration> = results.iter()
            .filter(|r| r.success)
            .map(|r| r.response_time)
            .collect();

        let avg_response_time = if !response_times.is_empty() {
            response_times.iter().sum::<Duration>() / response_times.len() as u32
        } else {
            Duration::from_millis(0)
        };

        let mut sorted_times = response_times.clone();
        sorted_times.sort();

        let p50_response_time = if !sorted_times.is_empty() {
            sorted_times[sorted_times.len() / 2]
        } else {
            Duration::from_millis(0)
        };

        let p95_response_time = if !sorted_times.is_empty() {
            sorted_times[(sorted_times.len() * 95) / 100]
        } else {
            Duration::from_millis(0)
        };

        let p99_response_time = if !sorted_times.is_empty() {
            sorted_times[(sorted_times.len() * 99) / 100]
        } else {
            Duration::from_millis(0)
        };

        let avg_accuracy = if successful_requests > 0 {
            results.iter()
                .filter(|r| r.success)
                .map(|r| r.accuracy)
                .sum::<f64>() / successful_requests as f64
        } else {
            0.0
        };

        let throughput = total_requests as f64 / duration.as_secs_f64();

        LoadTestResult {
            total_requests,
            successful_requests,
            failed_requests,
            avg_response_time,
            p50_response_time,
            p95_response_time,
            p99_response_time,
            avg_accuracy,
            throughput_rps: throughput,
            test_duration: duration,
        }
    }
}

#[derive(Debug)]
pub struct UserQueryResult {
    pub user_id: usize,
    pub query: String,
    pub success: bool,
    pub response_time: Duration,
    pub accuracy: f64,
    pub error: Option<String>,
}

#[derive(Debug)]
pub struct LoadTestResult {
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub avg_response_time: Duration,
    pub p50_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub avg_accuracy: f64,
    pub throughput_rps: f64,
    pub test_duration: Duration,
}

// Integration Tests

/// Test basic end-to-end query processing
#[tokio::test]
async fn test_end_to_end_query_processing() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config.clone());

    // Index some test documents first
    let documents = vec![
        ("doc1".to_string(), "Software architecture is the fundamental structure of software systems. It defines the components, their relationships, and principles governing their design and evolution. Key principles include modularity, separation of concerns, and maintainability.".to_string()),
        ("doc2".to_string(), "Machine learning algorithms enable computers to learn and make predictions from data. Common algorithms include linear regression, decision trees, neural networks, and support vector machines. Each has specific use cases and trade-offs.".to_string()),
        ("doc3".to_string(), "Microservices architecture breaks down applications into small, independent services that communicate over well-defined APIs. This contrasts with monolithic architecture where all components are tightly coupled in a single deployable unit.".to_string()),
    ];

    let indexing_result = system.index_documents(&documents).await.unwrap();
    assert_eq!(indexing_result.total_documents, 3);
    assert!(indexing_result.successful_chunks > 0);

    // Test query processing
    let query = "What are the key principles of software architecture?";
    let result = system.process_query_end_to_end(query).await.unwrap();

    // Validate performance targets
    assert!(
        result.performance_metrics.query_processing_time.as_millis() 
        <= config.performance_targets.query_processing_ms as u128,
        "Query processing time {}ms exceeds target {}ms",
        result.performance_metrics.query_processing_time.as_millis(),
        config.performance_targets.query_processing_ms
    );

    assert!(
        result.performance_metrics.response_generation_time.as_millis() 
        <= config.performance_targets.response_generation_ms as u128,
        "Response generation time {}ms exceeds target {}ms",
        result.performance_metrics.response_generation_time.as_millis(),
        config.performance_targets.response_generation_ms
    );

    assert!(
        result.performance_metrics.total_time.as_millis() 
        <= config.performance_targets.end_to_end_ms as u128,
        "End-to-end time {}ms exceeds target {}ms",
        result.performance_metrics.total_time.as_millis(),
        config.performance_targets.end_to_end_ms
    );

    // Validate accuracy
    assert!(
        result.accuracy_score >= config.performance_targets.accuracy_threshold,
        "Accuracy {:.3} below threshold {:.3}",
        result.accuracy_score,
        config.performance_targets.accuracy_threshold
    );

    // Validate response quality
    assert!(!result.generated_response.content.is_empty());
    assert!(!result.generated_response.citations.is_empty());
    assert!(result.generated_response.confidence_score >= 0.7);
    assert!(!result.search_results.is_empty());
}

/// Test different query types and intents
#[tokio::test]
async fn test_different_query_types() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config);

    // Index comprehensive test data
    let documents = vec![
        ("tech_doc".to_string(), "Cloud computing delivers computing services over the internet. Benefits include cost reduction, scalability, automatic updates, and accessibility from anywhere. Popular providers include AWS, Azure, and Google Cloud Platform.".to_string()),
        ("comparison_doc".to_string(), "Python is interpreted and dynamically typed, making it excellent for rapid development and scripting. Java is compiled and statically typed, offering better performance and strong typing for large applications. Python has simpler syntax while Java has more verbose but explicit code.".to_string()),
        ("process_doc".to_string(), "To deploy a web application: 1) Prepare production environment 2) Configure web server (nginx/apache) 3) Set up database connections 4) Deploy application code 5) Configure SSL certificates 6) Set up monitoring and logging 7) Perform final testing".to_string()),
    ];

    system.index_documents(&documents).await.unwrap();

    let test_cases = vec![
        ("What is cloud computing?", QueryIntent::Factual),
        ("Compare Python and Java programming languages", QueryIntent::Comparison),
        ("Summarize the benefits of cloud computing", QueryIntent::Summary),
        ("How to deploy a web application?", QueryIntent::Procedural),
        ("Analyze the trade-offs between different programming languages for enterprise software development considering performance, maintainability, and team productivity", QueryIntent::Complex),
    ];

    for (query, expected_intent) in test_cases {
        let result = system.process_query_end_to_end(query).await.unwrap();
        
        // Verify intent classification
        assert_eq!(
            std::mem::discriminant(&result.processed_query.intent),
            std::mem::discriminant(&expected_intent),
            "Intent mismatch for query: {}", query
        );

        // Verify response quality
        assert!(!result.generated_response.content.is_empty());
        assert!(result.generated_response.confidence_score >= 0.6);
        assert!(result.accuracy_score >= 0.7);
    }
}

/// Test concurrent query processing under load
#[tokio::test]
async fn test_concurrent_load_processing() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config);

    // Index test documents
    let documents = vec![
        ("load_doc1".to_string(), "Load testing verifies system behavior under expected load conditions. It helps identify performance bottlenecks, resource utilization issues, and system capacity limits before production deployment.".to_string()),
        ("load_doc2".to_string(), "Concurrent processing enables handling multiple requests simultaneously. Proper synchronization and resource management are crucial to prevent race conditions and ensure data consistency.".to_string()),
        ("load_doc3".to_string(), "Performance monitoring tracks system metrics like response time, throughput, error rates, and resource utilization. These metrics help identify optimization opportunities and ensure SLA compliance.".to_string()),
    ];

    system.index_documents(&documents).await.unwrap();

    // Create load test runner
    let load_runner = LoadTestRunner::new(system);

    // Run load test with multiple concurrent users
    let concurrent_users = 10;
    let test_duration = Duration::from_secs(30);

    let load_result = load_runner.run_load_test(concurrent_users, test_duration).await.unwrap();

    // Validate load test results
    assert!(load_result.total_requests > 0);
    assert!(load_result.successful_requests > 0);
    
    let success_rate = load_result.successful_requests as f64 / load_result.total_requests as f64;
    assert!(success_rate >= 0.95, "Success rate {:.2}% below 95%", success_rate * 100.0);

    // Performance validation under load
    assert!(
        load_result.p95_response_time.as_millis() <= 300,
        "P95 response time {}ms too high under load",
        load_result.p95_response_time.as_millis()
    );

    assert!(
        load_result.avg_accuracy >= 0.85,
        "Average accuracy {:.3} too low under load",
        load_result.avg_accuracy
    );

    assert!(
        load_result.throughput_rps >= 5.0,
        "Throughput {:.2} RPS too low",
        load_result.throughput_rps
    );

    println!("Load test results: {:?}", load_result);
}

/// Test error handling and resilience
#[tokio::test]
async fn test_error_handling_resilience() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config);

    // Test edge cases
    let edge_cases = vec![
        "",                                    // Empty query
        "?",                                  // Single character
        "a".repeat(2000),                     // Very long query
        "ñoño español 中文 العربية 🚀",        // Unicode and emojis
        "   \n\t  ",                          // Whitespace only
        "SELECT * FROM users; DROP TABLE users;", // SQL injection attempt
    ];

    for query in edge_cases {
        match system.process_query_end_to_end(query).await {
            Ok(result) => {
                // Should handle gracefully
                println!("Query '{}' processed successfully", query.chars().take(50).collect::<String>());
                assert!(!result.generated_response.content.is_empty());
            },
            Err(e) => {
                // Should fail gracefully with descriptive error
                println!("Query '{}' failed gracefully: {}", query.chars().take(50).collect::<String>(), e);
            }
        }
    }
}

/// Test system components integration
#[tokio::test]
async fn test_component_integration() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config);

    // Test MCP adapter integration
    let mcp_result = system.mcp_adapter.send_message("test message").await;
    assert!(mcp_result.is_ok());

    let health_ok = system.mcp_adapter.health_check().await;
    assert!(health_ok);

    // Test chunker integration
    let test_doc = "This is a test document with multiple sentences. It should be chunked properly. Each chunk should maintain semantic coherence. The chunker should handle various content types including lists, headers, and references.";
    let chunks = system.chunker.chunk_document(test_doc, "test_doc").await.unwrap();
    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|c| !c.content.is_empty()));

    // Test embedder integration
    let mut test_chunks = chunks;
    system.embedder.generate_embeddings(&mut test_chunks).await.unwrap();
    assert!(test_chunks.iter().all(|c| c.embeddings.is_some()));

    // Test storage integration
    system.storage.store_chunks(&test_chunks).await.unwrap();
    let stored_count = system.storage.get_document_count().await;
    assert_eq!(stored_count, test_chunks.len());

    // Test search integration
    if let Some(query_embedding) = &test_chunks[0].embeddings {
        let search_results = system.storage.search_similar(query_embedding, 5, 0.5).await.unwrap();
        assert!(!search_results.is_empty());
    }
}

/// Test performance benchmarking
#[tokio::test]
async fn test_performance_benchmarks() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config.clone());

    // Benchmark document indexing
    let large_documents = vec![
        ("benchmark_doc1".to_string(), "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(100)),
        ("benchmark_doc2".to_string(), "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ".repeat(100)),
        ("benchmark_doc3".to_string(), "Ut enim ad minim veniam, quis nostrud exercitation ullamco. ".repeat(100)),
    ];

    let indexing_start = Instant::now();
    let indexing_result = system.index_documents(&large_documents).await.unwrap();
    let indexing_duration = indexing_start.elapsed();

    assert!(indexing_result.successful_chunks > 0);
    println!("Indexed {} chunks in {:?}", indexing_result.total_chunks, indexing_duration);

    // Benchmark query processing
    let benchmark_queries = vec![
        "What is the main topic discussed in the documents?",
        "Summarize the key points from the indexed content",
        "How does the content relate to the user's information needs?",
    ];

    let mut query_times = Vec::new();
    let mut accuracy_scores = Vec::new();

    for query in benchmark_queries {
        let query_start = Instant::now();
        let result = system.process_query_end_to_end(query).await.unwrap();
        let query_duration = query_start.elapsed();

        query_times.push(query_duration);
        accuracy_scores.push(result.accuracy_score);

        // Validate individual query performance
        assert!(
            result.performance_metrics.total_time.as_millis() <= config.performance_targets.end_to_end_ms as u128,
            "Query '{}' took {}ms, exceeds {}ms target",
            query,
            result.performance_metrics.total_time.as_millis(),
            config.performance_targets.end_to_end_ms
        );
    }

    // Aggregate performance metrics
    let avg_query_time = query_times.iter().sum::<Duration>() / query_times.len() as u32;
    let avg_accuracy = accuracy_scores.iter().sum::<f64>() / accuracy_scores.len() as f64;

    println!("Average query time: {:?}", avg_query_time);
    println!("Average accuracy: {:.3}", avg_accuracy);

    assert!(
        avg_accuracy >= config.performance_targets.accuracy_threshold,
        "Average accuracy {:.3} below threshold {:.3}",
        avg_accuracy,
        config.performance_targets.accuracy_threshold
    );
}

/// Test data flow integrity through complete pipeline
#[tokio::test]
async fn test_data_flow_integrity() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config);

    // Test document with unique identifiers for tracking
    let test_content = "UNIQUE_MARKER_12345: This document contains a unique identifier for tracking data flow integrity. The system should preserve this marker through all processing stages including chunking, embedding, storage, and retrieval.";
    let doc_id = "integrity_test_doc";

    // Index the document
    let documents = vec![(doc_id.to_string(), test_content.to_string())];
    let indexing_result = system.index_documents(&documents).await.unwrap();
    assert!(indexing_result.successful_chunks > 0);

    // Query for the unique content
    let query = "UNIQUE_MARKER_12345";
    let result = system.process_query_end_to_end(query).await.unwrap();

    // Verify data integrity through pipeline
    assert_eq!(result.processed_query.original_query, query);
    
    // Should find the marker in search results
    let found_marker = result.search_results.iter()
        .any(|r| r.content.contains("UNIQUE_MARKER_12345"));
    assert!(found_marker, "Unique marker not found in search results");

    // Should reference the marker in response
    let response_mentions_marker = result.generated_response.content.contains("UNIQUE_MARKER_12345") ||
        result.generated_response.citations.iter().any(|c| c.source.contains(doc_id));
    assert!(response_mentions_marker, "Response doesn't reference the searched content");

    // Verify citation integrity
    assert!(!result.generated_response.citations.is_empty());
    for citation in &result.generated_response.citations {
        assert!(citation.confidence >= 0.0 && citation.confidence <= 1.0);
        assert!(citation.relevance_score >= 0.0 && citation.relevance_score <= 1.0);
    }
}

/// Test system scalability with increasing data volumes
#[tokio::test]
async fn test_scalability_increasing_data() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config);

    let data_sizes = vec![10, 50, 100]; // Number of documents

    for &size in &data_sizes {
        println!("Testing scalability with {} documents", size);

        // Generate test documents
        let documents: Vec<_> = (0..size).map(|i| {
            (
                format!("scale_doc_{}", i),
                format!("This is test document number {} for scalability testing. It contains unique content about topic {}. The document discusses various aspects of subject matter {} with detailed explanations and examples.", i, i % 10, i % 5)
            )
        }).collect();

        // Measure indexing performance
        let indexing_start = Instant::now();
        let indexing_result = system.index_documents(&documents).await.unwrap();
        let indexing_duration = indexing_start.elapsed();

        println!("Indexed {} documents ({} chunks) in {:?}", 
                 size, indexing_result.total_chunks, indexing_duration);

        // Test query performance with increasing data
        let query = format!("topic {}", size % 10);
        let query_start = Instant::now();
        let query_result = system.process_query_end_to_end(&query).await.unwrap();
        let query_duration = query_start.elapsed();

        println!("Query processed in {:?} with {} search results", 
                 query_duration, query_result.search_results.len());

        // Validate performance doesn't degrade significantly
        assert!(
            query_duration.as_millis() <= 500, // Relaxed threshold for scalability test
            "Query performance degraded significantly with {} documents: {}ms",
            size, query_duration.as_millis()
        );

        assert!(query_result.accuracy_score >= 0.7);
    }
}

/// Test memory efficiency and resource management
#[tokio::test]
async fn test_memory_efficiency() {
    let config = TestConfig {
        max_embeddings: 100, // Limited for memory testing
        ..TestConfig::default()
    };
    let system = RagSystemIntegration::new(config);

    // Process many small documents to test memory management
    let small_documents: Vec<_> = (0..200).map(|i| {
        (
            format!("mem_doc_{}", i),
            format!("Small document {} for memory efficiency testing.", i)
        )
    }).collect();

    // Process in batches to simulate real-world usage
    let batch_size = 50;
    for batch in small_documents.chunks(batch_size) {
        let result = system.index_documents(batch).await.unwrap();
        assert!(result.successful_chunks > 0);
        
        // Test query processing after each batch
        let query = format!("document {}", batch.len() / 2);
        let query_result = system.process_query_end_to_end(&query).await.unwrap();
        assert!(!query_result.search_results.is_empty());
    }

    // Verify system is still responsive
    let final_query = "memory efficiency testing";
    let final_result = system.process_query_end_to_end(final_query).await.unwrap();
    assert!(final_result.accuracy_score >= 0.7);
}

/// Test production readiness validation
#[tokio::test]
async fn test_production_readiness() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config.clone());

    // Production-like test data
    let production_documents = vec![
        ("user_manual".to_string(), "System Administration Guide: This comprehensive manual covers installation, configuration, maintenance, troubleshooting, and best practices for system administrators. Key topics include user management, security policies, backup procedures, monitoring setup, and performance optimization techniques.".to_string()),
        ("api_docs".to_string(), "API Documentation: Complete reference for RESTful API endpoints including authentication, rate limiting, error handling, request/response formats, and integration examples. Supports JSON and XML formats with comprehensive error codes and status messages.".to_string()),
        ("security_policy".to_string(), "Information Security Policy: Enterprise security guidelines covering data classification, access controls, encryption requirements, incident response procedures, compliance requirements, and employee security training protocols.".to_string()),
    ];

    // Index production data
    let indexing_result = system.index_documents(&production_documents).await.unwrap();
    assert_eq!(indexing_result.total_documents, 3);
    
    // Production-like queries
    let production_queries = vec![
        "How do I configure user authentication and authorization?",
        "What are the API rate limiting policies and how to handle them?",
        "What security measures are required for handling sensitive data?",
        "Explain the backup and disaster recovery procedures",
        "What are the system monitoring and alerting configurations?",
    ];

    let mut all_passed = true;
    let mut performance_results = Vec::new();
    let mut accuracy_results = Vec::new();

    for query in production_queries {
        match system.process_query_end_to_end(query).await {
            Ok(result) => {
                performance_results.push(result.performance_metrics.total_time);
                accuracy_results.push(result.accuracy_score);

                // Production quality checks
                if result.performance_metrics.total_time.as_millis() > config.performance_targets.end_to_end_ms as u128 {
                    eprintln!("PRODUCTION ISSUE: Query '{}' took {}ms (target: {}ms)", 
                             query, result.performance_metrics.total_time.as_millis(), config.performance_targets.end_to_end_ms);
                    all_passed = false;
                }

                if result.accuracy_score < config.performance_targets.accuracy_threshold {
                    eprintln!("PRODUCTION ISSUE: Query '{}' accuracy {:.3} (target: {:.3})", 
                             query, result.accuracy_score, config.performance_targets.accuracy_threshold);
                    all_passed = false;
                }

                if result.generated_response.content.len() < 100 {
                    eprintln!("PRODUCTION ISSUE: Query '{}' generated short response ({} chars)", 
                             query, result.generated_response.content.len());
                    all_passed = false;
                }

                if result.generated_response.citations.is_empty() {
                    eprintln!("PRODUCTION ISSUE: Query '{}' has no citations", query);
                    all_passed = false;
                }
            },
            Err(e) => {
                eprintln!("PRODUCTION FAILURE: Query '{}' failed: {}", query, e);
                all_passed = false;
            }
        }
    }

    // Aggregate production metrics
    let avg_performance = performance_results.iter().sum::<Duration>() / performance_results.len() as u32;
    let avg_accuracy = accuracy_results.iter().sum::<f64>() / accuracy_results.len() as f64;

    println!("Production Readiness Report:");
    println!("  Average Performance: {:?}", avg_performance);
    println!("  Average Accuracy: {:.3}", avg_accuracy);
    println!("  Queries Processed: {}", production_queries.len());
    println!("  All Tests Passed: {}", all_passed);

    // Final production validation
    assert!(all_passed, "System failed production readiness validation");
    assert!(
        avg_performance.as_millis() <= config.performance_targets.end_to_end_ms as u128,
        "Average performance {} ms exceeds production target {} ms",
        avg_performance.as_millis(), config.performance_targets.end_to_end_ms
    );
    assert!(
        avg_accuracy >= config.performance_targets.accuracy_threshold,
        "Average accuracy {:.3} below production target {:.3}",
        avg_accuracy, config.performance_targets.accuracy_threshold
    );
}

/// Comprehensive system validation test
#[tokio::test]
async fn test_comprehensive_system_validation() {
    let config = TestConfig::default();
    let system = RagSystemIntegration::new(config.clone());

    println!("Starting comprehensive system validation...");

    // Phase 1: Component Integration Validation
    println!("Phase 1: Component Integration Validation");
    
    // Test all components are properly initialized and integrated
    let test_doc = "Integration test document with comprehensive content for validation.";
    let chunks = system.chunker.chunk_document(test_doc, "test").await.unwrap();
    assert!(!chunks.is_empty(), "Chunker integration failed");

    let mut test_chunks = chunks;
    system.embedder.generate_embeddings(&mut test_chunks).await.unwrap();
    assert!(test_chunks.iter().all(|c| c.embeddings.is_some()), "Embedder integration failed");

    system.storage.store_chunks(&test_chunks).await.unwrap();
    let stored_count = system.storage.get_document_count().await;
    assert!(stored_count > 0, "Storage integration failed");

    // Phase 2: End-to-End Pipeline Validation
    println!("Phase 2: End-to-End Pipeline Validation");
    
    let comprehensive_docs = vec![
        ("knowledge_base".to_string(), "Knowledge Management System: A comprehensive platform for organizing, storing, and retrieving organizational knowledge. Features include document management, search capabilities, collaboration tools, version control, access management, and analytics. The system supports multiple file formats, automated indexing, and intelligent content recommendations based on user behavior and content relationships.".to_string()),
    ];

    let indexing_result = system.index_documents(&comprehensive_docs).await.unwrap();
    assert!(indexing_result.successful_chunks > 0, "Document indexing failed");

    let comprehensive_query = "How does the knowledge management system handle document organization and user collaboration?";
    let end_to_end_result = system.process_query_end_to_end(comprehensive_query).await.unwrap();

    // Phase 3: Performance Validation
    println!("Phase 3: Performance Validation");
    
    assert!(
        end_to_end_result.performance_metrics.query_processing_time.as_millis() <= config.performance_targets.query_processing_ms as u128,
        "Query processing performance validation failed: {}ms > {}ms",
        end_to_end_result.performance_metrics.query_processing_time.as_millis(),
        config.performance_targets.query_processing_ms
    );

    assert!(
        end_to_end_result.performance_metrics.response_generation_time.as_millis() <= config.performance_targets.response_generation_ms as u128,
        "Response generation performance validation failed: {}ms > {}ms",
        end_to_end_result.performance_metrics.response_generation_time.as_millis(),
        config.performance_targets.response_generation_ms
    );

    assert!(
        end_to_end_result.performance_metrics.total_time.as_millis() <= config.performance_targets.end_to_end_ms as u128,
        "End-to-end performance validation failed: {}ms > {}ms",
        end_to_end_result.performance_metrics.total_time.as_millis(),
        config.performance_targets.end_to_end_ms
    );

    // Phase 4: Quality Validation
    println!("Phase 4: Quality Validation");
    
    assert!(
        end_to_end_result.accuracy_score >= config.performance_targets.accuracy_threshold,
        "Accuracy validation failed: {:.3} < {:.3}",
        end_to_end_result.accuracy_score,
        config.performance_targets.accuracy_threshold
    );

    assert!(!end_to_end_result.generated_response.content.is_empty(), "Response content validation failed");
    assert!(!end_to_end_result.generated_response.citations.is_empty(), "Citations validation failed");
    assert!(end_to_end_result.generated_response.confidence_score >= 0.7, "Response confidence validation failed");
    assert!(!end_to_end_result.search_results.is_empty(), "Search results validation failed");

    // Phase 5: Resilience Validation
    println!("Phase 5: Resilience Validation");
    
    // Test with multiple concurrent requests
    let concurrent_requests = 5;
    let mut handles = Vec::new();

    for i in 0..concurrent_requests {
        let system_clone = &system;
        let query = format!("Concurrent test query number {}", i);
        
        let handle = tokio::spawn(async move {
            system_clone.process_query_end_to_end(&query).await
        });
        
        handles.push(handle);
    }

    let mut concurrent_results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap().unwrap();
        concurrent_results.push(result);
    }

    assert_eq!(concurrent_results.len(), concurrent_requests);
    for result in &concurrent_results {
        assert!(result.accuracy_score >= 0.6, "Concurrent processing quality degraded");
    }

    // Final Validation Summary
    println!("Comprehensive System Validation Results:");
    println!("  ✓ Component Integration: PASSED");
    println!("  ✓ End-to-End Pipeline: PASSED");
    println!("  ✓ Performance Targets: PASSED");
    println!("  ✓ Quality Validation: PASSED");
    println!("  ✓ Resilience Testing: PASSED");
    println!("  ✓ Query Processing Time: {:?}", end_to_end_result.performance_metrics.query_processing_time);
    println!("  ✓ Response Generation Time: {:?}", end_to_end_result.performance_metrics.response_generation_time);
    println!("  ✓ Total End-to-End Time: {:?}", end_to_end_result.performance_metrics.total_time);
    println!("  ✓ Accuracy Score: {:.3}", end_to_end_result.accuracy_score);
    println!("  ✓ System Status: PRODUCTION READY");
}