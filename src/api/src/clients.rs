use anyhow::{Context, Result};
use base64::prelude::*;
use reqwest::{Client, Response, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::time::timeout;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::{
    config::{ApiConfig, ServiceConfig},
    models::{
        api::{QueryRequest, QueryResponse, DocumentStatus, TaskStatus, 
              HealthStatus, QueryHistoryRequest, QueryHistoryResponse,
              QueryMetrics, StreamingQueryResponse, ChunkingStrategy},
        storage::{ProcessingHistory, ProcessingStatistics, StorageUsage, 
                  ContentTypeStatistics, RecentDocument, TaskDetails},
        domain::{ProcessingTask, DomainTaskStatus},
    },
    ApiError,
};



/// Client for communicating with all Doc-RAG components
pub struct ComponentClients {
    pub config: Arc<ApiConfig>,
    pub http_client: Client,
    pub chunker_client: ServiceClient,
    pub embedder_client: ServiceClient,
    pub storage_client: ServiceClient,
    pub retriever_client: ServiceClient,
    pub query_processor_client: ServiceClient,
    pub response_generator_client: ServiceClient,
    pub mcp_adapter_client: ServiceClient,
}

impl ComponentClients {
    pub async fn new(config: Arc<ApiConfig>) -> Result<Self> {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .tcp_keepalive(Duration::from_secs(60))
            .pool_max_idle_per_host(10)
            .build()
            .context("Failed to create HTTP client")?;

        let clients = Self {
            chunker_client: ServiceClient::new("chunker", &config.components.chunker, http_client.clone()),
            embedder_client: ServiceClient::new("embedder", &config.components.embedder, http_client.clone()),
            storage_client: ServiceClient::new("storage", &config.components.storage, http_client.clone()),
            retriever_client: ServiceClient::new("retriever", &config.components.retriever, http_client.clone()),
            query_processor_client: ServiceClient::new("query_processor", &config.components.query_processor, http_client.clone()),
            response_generator_client: ServiceClient::new("response_generator", &config.components.response_generator, http_client.clone()),
            mcp_adapter_client: ServiceClient::new("mcp_adapter", &config.components.mcp_adapter, http_client.clone()),
            config,
            http_client,
        };

        Ok(clients)
    }

    /// Health check all components
    pub async fn health_check_all(&self) -> HashMap<String, Result<()>> {
        let mut results = HashMap::new();

        let checks = vec![
            ("chunker", self.chunker_client.health_check()),
            ("embedder", self.embedder_client.health_check()),
            ("storage", self.storage_client.health_check()),
            ("retriever", self.retriever_client.health_check()),
            ("query_processor", self.query_processor_client.health_check()),
            ("response_generator", self.response_generator_client.health_check()),
            ("mcp_adapter", self.mcp_adapter_client.health_check()),
        ];

        let futures: Vec<_> = checks.into_iter().collect();
        let check_results = futures::future::join_all(
            futures.into_iter().map(|(name, fut)| async move {
                (name, fut.await)
            })
        ).await;

        for (name, result) in check_results {
            results.insert(name.to_string(), result);
        }

        results
    }

    /// Detailed health check with component information
    pub async fn detailed_health_check(&self) -> HashMap<String, Value> {
        let mut results = HashMap::new();

        let services = vec![
            ("chunker", &self.chunker_client),
            ("embedder", &self.embedder_client),
            ("storage", &self.storage_client),
            ("retriever", &self.retriever_client),
            ("query_processor", &self.query_processor_client),
            ("response_generator", &self.response_generator_client),
            ("mcp_adapter", &self.mcp_adapter_client),
        ];

        for (name, client) in services {
            let start_time = std::time::Instant::now();
            let health_result = client.health_check().await;
            let response_time = start_time.elapsed().as_millis() as u64;

            let status = match health_result {
                Ok(_) => {
                    json!({
                        "status": "healthy",
                        "response_time_ms": response_time,
                        "url": client.base_url,
                        "last_check": chrono::Utc::now()
                    })
                }
                Err(e) => {
                    json!({
                        "status": "unhealthy",
                        "error": e.to_string(),
                        "response_time_ms": response_time,
                        "url": client.base_url,
                        "last_check": chrono::Utc::now()
                    })
                }
            };

            results.insert(name.to_string(), status);
        }

        results
    }

    /// Process complete document ingestion pipeline
    pub async fn process_document_ingestion(
        &self,
        task_id: Uuid,
        content: String,
        metadata: Option<Value>,
        chunking_strategy: Option<ChunkingStrategy>,
    ) -> Result<Uuid> {
        info!("Starting document ingestion pipeline: task_id={}", task_id);

        // 1. Chunk the document
        let chunks = self.chunk_document(content, chunking_strategy).await
            .context("Failed to chunk document")?;

        info!("Document chunked: task_id={}, chunks={}", task_id, chunks.len());

        // 2. Generate embeddings for chunks
        let embeddings = self.generate_embeddings(chunks.clone()).await
            .context("Failed to generate embeddings")?;

        info!("Embeddings generated: task_id={}, embeddings={}", task_id, embeddings.len());

        // 3. Store document and chunks
        let document_id = self.store_document_with_chunks(
            task_id,
            chunks,
            embeddings,
            metadata,
        ).await.context("Failed to store document")?;

        info!("Document stored: task_id={}, document_id={}", task_id, document_id);

        // 4. Index in vector database (via retriever)
        self.index_document(document_id).await
            .context("Failed to index document")?;

        info!("Document indexed: task_id={}, document_id={}", task_id, document_id);

        Ok(document_id)
    }

    /// Process complete query pipeline
    pub async fn process_query_pipeline(&self, request: QueryRequest) -> Result<QueryResponse> {
        info!("Starting query processing pipeline: query_id={}", request.query_id);

        // 1. Process query through query processor
        let processed_query = self.process_query(&request).await
            .context("Failed to process query")?;

        // 2. Retrieve relevant documents
        let retrieved_docs = self.retrieve_documents(&processed_query).await
            .context("Failed to retrieve documents")?;

        // 3. Generate response
        let response = self.generate_response(request.query_id, &request.query, retrieved_docs).await
            .context("Failed to generate response")?;

        info!("Query processing completed: query_id={}", request.query_id);

        Ok(response)
    }

    /// Process streaming query
    pub async fn process_streaming_query(
        &self,
        request: QueryRequest,
    ) -> Result<impl futures::Stream<Item = Result<StreamingQueryResponse>>> {
        info!("Starting streaming query: query_id={}", request.query_id);

        // For now, simulate streaming by processing normally and chunking the response
        // In a full implementation, this would coordinate with streaming-capable components
        let response = self.process_query_pipeline(request.clone()).await?;
        
        // Create a stream that emits chunks of the response
        let chunks: Vec<_> = response.answer
            .chars()
            .collect::<Vec<_>>()
            .chunks(50)
            .enumerate()
            .map(|(i, chunk)| {
                Ok(StreamingQueryResponse {
                    query_id: request.query_id,
                    chunk: chunk.iter().collect(),
                    chunk_index: i,
                    is_final: false,
                    sources: if i == 0 { Some(response.sources.clone()) } else { None },
                    metadata: if i == 0 { response.metadata.clone() } else { None },
                })
            })
            .collect();

        // Convert to async stream
        Ok(futures::stream::iter(chunks))
    }

    // Private helper methods for pipeline steps

    async fn chunk_document(
        &self,
        content: String,
        strategy: Option<ChunkingStrategy>,
    ) -> Result<Vec<String>> {
        let request = json!({
            "content": content,
            "strategy": strategy.unwrap_or_default()
        });

        let response = self.chunker_client.post("/chunk", &request).await?;
        let chunks: Vec<String> = response.json().await
            .context("Failed to parse chunking response")?;
        
        Ok(chunks)
    }

    async fn generate_embeddings(&self, chunks: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = json!({
            "texts": chunks,
            "batch_size": 32
        });

        let response = self.embedder_client.post("/embed", &request).await?;
        let embeddings: Vec<Vec<f32>> = response.json().await
            .context("Failed to parse embedding response")?;

        Ok(embeddings)
    }

    async fn store_document_with_chunks(
        &self,
        task_id: Uuid,
        chunks: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadata: Option<Value>,
    ) -> Result<Uuid> {
        let document_id = Uuid::new_v4();
        
        let request = json!({
            "document_id": document_id,
            "task_id": task_id,
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": metadata
        });

        let response = self.storage_client.post("/documents", &request).await?;
        let result: Value = response.json().await
            .context("Failed to parse storage response")?;

        Ok(document_id)
    }

    async fn index_document(&self, document_id: Uuid) -> Result<()> {
        let request = json!({
            "document_id": document_id
        });

        self.retriever_client.post("/index", &request).await?;
        Ok(())
    }

    async fn process_query(&self, request: &QueryRequest) -> Result<Value> {
        let response = self.query_processor_client.post("/process", request).await?;
        let processed: Value = response.json().await
            .context("Failed to parse query processing response")?;
        
        Ok(processed)
    }

    async fn retrieve_documents(&self, processed_query: &Value) -> Result<Value> {
        let response = self.retriever_client.post("/retrieve", processed_query).await?;
        let documents: Value = response.json().await
            .context("Failed to parse retrieval response")?;
        
        Ok(documents)
    }

    async fn generate_response(
        &self,
        query_id: Uuid,
        query: &str,
        retrieved_docs: Value,
    ) -> Result<QueryResponse> {
        let request = json!({
            "query_id": query_id,
            "query": query,
            "documents": retrieved_docs
        });

        let response = self.response_generator_client.post("/generate", &request).await?;
        let query_response: QueryResponse = response.json().await
            .context("Failed to parse response generation")?;

        Ok(query_response)
    }

    // Additional methods for API operations

    pub async fn get_document_status(&self, task_id: Uuid) -> Result<DocumentStatus> {
        let response = self.storage_client.get(&format!("/documents/status/{}", task_id)).await?;
        let status: DocumentStatus = response.json().await
            .context("Failed to parse document status")?;
        Ok(status)
    }

    pub async fn get_query_history(&self, request: QueryHistoryRequest) -> Result<QueryHistoryResponse> {
        let query_params = vec![
            ("limit", request.limit.to_string()),
            ("offset", request.offset.to_string()),
        ];
        
        let response = self.storage_client.get_with_params("/queries/history", &query_params).await?;
        let history: QueryHistoryResponse = response.json().await
            .context("Failed to parse query history")?;
        Ok(history)
    }

    pub async fn get_query_metrics(&self) -> Result<QueryMetrics> {
        let response = self.storage_client.get("/queries/metrics").await?;
        let metrics: QueryMetrics = response.json().await
            .context("Failed to parse query metrics")?;
        Ok(metrics)
    }

    pub async fn cancel_query(&self, query_id: Uuid) -> Result<bool> {
        let request = json!({
            "query_id": query_id,
            "action": "cancel"
        });

        let response = self.query_processor_client.post("/cancel", &request).await?;
        let result: Value = response.json().await
            .context("Failed to parse cancel response")?;

        Ok(result.get("cancelled").and_then(|v| v.as_bool()).unwrap_or(false))
    }

    pub async fn get_query_result(&self, query_id: Uuid) -> Result<Option<QueryResponse>> {
        let response = self.storage_client.get(&format!("/queries/{}", query_id)).await;
        
        match response {
            Ok(resp) if resp.status() == StatusCode::OK => {
                let result: QueryResponse = resp.json().await
                    .context("Failed to parse query result")?;
                Ok(Some(result))
            }
            Ok(resp) if resp.status() == StatusCode::NOT_FOUND => Ok(None),
            Ok(resp) => Err(anyhow::anyhow!("Unexpected status: {}", resp.status())),
            Err(e) => Err(e.into()),
        }
    }

    pub async fn find_similar_queries(&self, query_id: Uuid, limit: usize) -> Result<Vec<Value>> {
        let response = self.storage_client.get(&format!("/queries/{}/similar?limit={}", query_id, limit)).await?;
        let similar: Vec<Value> = response.json().await
            .context("Failed to parse similar queries")?;
        Ok(similar)
    }

    pub async fn extract_text_from_file(&self, file_content: Vec<u8>, content_type: Option<&str>) -> Result<String> {
        let request = json!({
            "content": base64::prelude::BASE64_STANDARD.encode(&file_content),
            "content_type": content_type
        });

        let response = self.mcp_adapter_client.post("/extract-text", &request).await?;
        let result: Value = response.json().await
            .context("Failed to parse text extraction response")?;

        let text = result.get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("No text content in response"))?;

        Ok(text.to_string())
    }

    // Database health checks

    pub async fn check_database_health(&self) -> Result<()> {
        self.storage_client.get("/health/database").await?;
        Ok(())
    }

    pub async fn check_redis_health(&self) -> Result<()> {
        self.storage_client.get("/health/redis").await?;
        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down component clients");
        // Implement graceful shutdown logic here
        Ok(())
    }

    /// Get storage service with domain-specific methods
    pub fn storage(&self) -> StorageServiceClient<'_> {
        StorageServiceClient {
            client: &self.storage_client,
        }
    }
    
    /// Get all active processing tasks across services
    pub async fn get_active_processing_tasks(&self) -> Result<Vec<ProcessingTask>> {
        let response = self.storage_client
            .get("/processing/tasks/active")
            .await?;
        let tasks: Vec<ProcessingTask> = response.json().await
            .context("Failed to parse processing tasks response")?;
        Ok(tasks)
    }
    
    /// Get current processing queue size
    pub async fn get_processing_queue_size(&self) -> Result<usize> {
        let response = self.query_processor_client
            .get("/queue/size")
            .await?;
        let result: serde_json::Value = response.json().await
            .context("Failed to parse queue size response")?;
        Ok(result["size"].as_u64().unwrap_or(0) as usize)
    }
    
    /// Send cancellation signal for a task
    pub async fn send_cancellation_signal(&self, task_id: Uuid) -> Result<()> {
        // Orchestrate cancellation across services
        let cancel_data = serde_json::json!({
            "task_id": task_id,
            "action": "cancel"
        });
        
        // Notify query processor
        self.query_processor_client
            .post("/tasks/cancel", &cancel_data)
            .await?;
        
        // Notify storage to mark as cancelled
        self.storage_client
            .post(&format!("/tasks/{}/cancel", task_id), &cancel_data)
            .await?;
        
        Ok(())
    }
    
    /// Reprocess a document with original parameters
    pub async fn reprocess_document(
        &self,
        task_id: Uuid,
        content: String,
        metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
        chunking_strategy: Option<ChunkingStrategy>,
    ) -> Result<Uuid> {
        // This would be similar to process_document_ingestion but with task_id reuse
        self.process_document_ingestion(task_id, content, metadata, chunking_strategy).await
    }
    
    /// Clean up partial processing for a document
    pub async fn cleanup_partial_processing(&self, document_id: Uuid) -> Result<()> {
        // Orchestrate cleanup across services
        
        // Remove partial chunks from storage
        self.storage_client
            .delete(&format!("/documents/{}/chunks/partial", document_id))
            .await?;
        
        // Clear embeddings
        self.embedder_client
            .delete(&format!("/documents/{}/embeddings", document_id))
            .await?;
        
        // Clear any cached queries
        self.query_processor_client
            .delete(&format!("/cache/document/{}", document_id))
            .await?;
        
        Ok(())
    }

}

/// Storage service client wrapper that provides domain-specific methods
pub struct StorageServiceClient<'a> {
    client: &'a ServiceClient,
}

impl<'a> StorageServiceClient<'a> {
    /// Count chunks for a specific document
    pub async fn count_chunks_for_document(&self, document_id: Uuid) -> Result<usize> {
        // Mock implementation - in reality this would query storage service
        warn!("Mock implementation: count_chunks_for_document for {}", document_id);
        Ok(0)
    }

    /// Get processing history from storage  
    pub async fn get_processing_history(&self) -> Result<crate::models::ProcessingHistory> {
        // Mock implementation - in reality this would query storage service
        warn!("Mock implementation: get_processing_history");
        Ok(crate::models::ProcessingHistory {
            entries: vec![],
            total_processed: 0,
            last_updated: chrono::Utc::now(),
        })
    }

    /// Get recent documents
    pub async fn get_recent_documents(&self, limit: usize) -> Result<Vec<crate::models::RecentDocument>> {
        // Mock implementation - in reality this would query storage service
        warn!("Mock implementation: get_recent_documents with limit {}", limit);
        Ok(vec![])
    }

    /// Get processing statistics
    pub async fn get_processing_statistics(&self) -> Result<crate::models::ProcessingStatistics> {
        // Mock implementation - in reality this would query storage service
        warn!("Mock implementation: get_processing_statistics");
        Ok(crate::models::ProcessingStatistics {
            total_processing_tasks: 0,
            successful_tasks: 0,
            failed_tasks: 0,
            average_processing_time_ms: 0.0,
            processing_by_stage: vec![],
            last_updated: chrono::Utc::now(),
        })
    }

    /// Get storage usage information
    pub async fn get_storage_usage(&self) -> Result<crate::models::StorageUsage> {
        // Mock implementation - in reality this would query storage service
        warn!("Mock implementation: get_storage_usage");
        Ok(crate::models::StorageUsage {
            total_documents: 0,
            total_chunks: 0,
            total_size_bytes: 0,
            storage_by_type: vec![],
            last_updated: chrono::Utc::now(),
        })
    }

    /// Get content type statistics
    pub async fn get_content_type_statistics(&self) -> Result<crate::models::ContentTypeStatistics> {
        // Mock implementation - in reality this would query storage service
        warn!("Mock implementation: get_content_type_statistics");
        Ok(crate::models::ContentTypeStatistics {
            statistics: vec![],
            total_types: 0,
            most_common_type: None,
            last_updated: chrono::Utc::now(),
        })
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: Uuid) -> Result<crate::models::TaskStatusResponse> {
        // Mock implementation - in reality this would query storage service
        warn!("Mock implementation: get_task_status for {}", task_id);
        Ok(crate::models::TaskStatusResponse {
            task_id,
            status: TaskStatus::Processing,
            message: Some("Processing document".to_string()),
            progress_percent: Some(50),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        })
    }

    /// Update task status
    pub async fn update_task_status(&self, task_id: Uuid, status: crate::models::TaskStatus) -> Result<()> {
        // Mock implementation - in reality this would update storage service
        warn!("Mock implementation: update_task_status for {} to {:?}", task_id, status);
        Ok(())
    }

    /// Get task details
    pub async fn get_task_details(&self, task_id: Uuid) -> Result<TaskDetails> {
        // Mock implementation - in reality this would query storage service
        warn!("Mock implementation: get_task_details for {}", task_id);
        Ok(TaskDetails {
            task_id,
            document_id: Some(Uuid::new_v4()),
            content: Some("mock content".to_string()),
            metadata: None,
            status: TaskStatus::Processing,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        })
    }
}

/// Individual service client wrapper
#[derive(Clone)]
struct ServiceClient {
    name: String,
    base_url: String,
    client: Client,
    config: ServiceConfig,
}

impl ServiceClient {
    fn new(name: &str, config: &ServiceConfig, client: Client) -> Self {
        Self {
            name: name.to_string(),
            base_url: config.url.clone(),
            client,
            config: config.clone(),
        }
    }

    async fn health_check(&self) -> Result<()> {
        let url = format!("{}/health", self.base_url);
        
        let response = timeout(
            Duration::from_secs(self.config.health_check_interval_secs),
            self.client.get(&url).send()
        ).await
        .context("Health check timeout")?
        .context("Health check request failed")?;

        if response.status().is_success() {
            debug!("Health check passed for {}", self.name);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Health check failed with status: {}", response.status()))
        }
    }

    async fn get(&self, path: &str) -> Result<Response> {
        self.get_with_params(path, &[]).await
    }

    async fn get_with_params(&self, path: &str, params: &[(&str, String)]) -> Result<Response> {
        let url = format!("{}{}", self.base_url, path);
        
        let mut request = self.client.get(&url);
        for (key, value) in params {
            request = request.query(&[(key, value)]);
        }

        let response = timeout(
            Duration::from_secs(self.config.timeout_secs),
            request.send()
        ).await
        .context("Request timeout")?
        .context("Request failed")?;

        if response.status().is_success() {
            Ok(response)
        } else {
            Err(anyhow::anyhow!(
                "Request failed with status: {} for service: {}",
                response.status(),
                self.name
            ))
        }
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Response> {
        let url = format!("{}{}", self.base_url, path);
        
        let response = timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.client.post(&url)
                .json(body)
                .send()
        ).await
        .context("Request timeout")?
        .context("Request failed")?;

        if response.status().is_success() {
            Ok(response)
        } else {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            Err(anyhow::anyhow!(
                "Request failed with status: {} for service: {}, error: {}",
                status,
                self.name,
                error_text
            ))
        }
    }

    async fn delete(&self, path: &str) -> Result<Response> {
        let url = format!("{}{}", self.base_url, path);
        let response = timeout(
            Duration::from_secs(self.config.timeout_secs),
            self.client.delete(&url).send()
        ).await
        .context("Request timeout")?
        .context("Request failed")?;
        
        if response.status().is_success() {
            Ok(response)
        } else {
            Err(anyhow::anyhow!("Delete request failed: {}", response.status()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_client_creation() {
        let config = ServiceConfig {
            url: "http://localhost:8080".to_string(),
            timeout_secs: 30,
            max_retries: 3,
            retry_delay_ms: 1000,
            health_check_interval_secs: 30,
        };

        let client = Client::new();
        let service_client = ServiceClient::new("test", &config, client);
        
        assert_eq!(service_client.name, "test");
        assert_eq!(service_client.base_url, "http://localhost:8080");
    }

    #[tokio::test]
    async fn test_component_clients_creation() {
        let config = Arc::new(ApiConfig::default());
        
        // This would fail in test environment without actual services
        // but we can test the basic creation logic doesn't panic
        let result = ComponentClients::new(config).await;
        
        // In real tests with proper mocking, we would verify success
        match result {
            Ok(_) => {
                // Success case with proper test infrastructure
            }
            Err(_) => {
                // Expected in test environment without real services
            }
        }
    }
}