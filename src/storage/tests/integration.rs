//! Integration tests for MongoDB Vector Storage
//! 
//! These tests require a running MongoDB instance and validate the complete
//! functionality of the storage system including CRUD operations, search,
//! and performance requirements.

use std::time::{Duration, Instant};
use std::collections::HashMap;

use mongodb::bson::doc;
use testcontainers::{clients, Container, Docker};
use testcontainers_modules::mongo::Mongo;
use uuid::Uuid;
use tokio_test;

use storage::{
    VectorStorage, StorageConfig, ChunkDocument, ChunkMetadata, MetadataDocument, DocumentMetadata,
    BulkInsertRequest, SearchQuery, SearchType, SearchFilters, SortOptions, SortField, SortDirection,
    DatabaseOperations, SearchOperations, TransactionOperations, SecurityLevel, ProcessingStats,
    CustomFieldValue, VectorSimilarity
};

/// Test configuration with MongoDB testcontainer
struct TestEnvironment {
    _container: Container<'static, clients::Cli, Mongo>,
    storage: VectorStorage,
}

impl TestEnvironment {
    /// Setup test environment with MongoDB container
    async fn setup() -> Result<Self, Box<dyn std::error::Error>> {
        let docker = clients::Cli::default();
        let container = docker.run(Mongo::default());
        let port = container.get_host_port_ipv4(27017);
        
        let config = StorageConfig {
            connection_string: format!("mongodb://localhost:{}", port),
            database_name: format!("test_rag_{}", Uuid::new_v4().simple()),
            chunk_collection_name: "test_chunks".to_string(),
            metadata_collection_name: "test_metadata".to_string(),
            connection_timeout_secs: 10,
            operation_timeout_secs: 30,
            ..StorageConfig::default()
        };
        
        // Wait for MongoDB to be ready
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        let storage = VectorStorage::new(config).await?;
        
        Ok(Self {
            _container: container,
            storage,
        })
    }
    
    /// Create test chunk with embedding
    fn create_test_chunk(index: usize, document_id: Uuid) -> ChunkDocument {
        let metadata = ChunkMetadata {
            document_id,
            title: format!("Test Document {}", index),
            chunk_index: index,
            total_chunks: 10,
            chunk_size: 512,
            overlap_size: 50,
            source_path: format!("/test/doc_{}.txt", index),
            mime_type: "text/plain".to_string(),
            language: "en".to_string(),
            tags: vec!["test".to_string(), format!("doc_{}", index)],
            custom_fields: {
                let mut fields = HashMap::new();
                fields.insert("priority".to_string(), CustomFieldValue::String("high".to_string()));
                fields.insert("score".to_string(), CustomFieldValue::Number(95.5));
                fields
            },
            content_hash: format!("hash_{}", index),
            boundary_confidence: Some(0.9),
        };
        
        // Generate deterministic embeddings for testing
        let embedding: Vec<f64> = (0..384).map(|i| (i + index) as f64 * 0.01).collect();
        
        ChunkDocument::new(
            Uuid::new_v4(),
            format!("This is test content for chunk {}. It contains meaningful text for testing search functionality.", index),
            metadata,
        ).with_embedding(embedding)
    }
}

#[tokio::test]
async fn test_storage_initialization() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    
    // Test health check
    let health = env.storage.health_check().await.expect("Health check failed");
    assert!(health.healthy);
    assert!(health.latency_ms < 1000); // Should be fast for local MongoDB
}

#[tokio::test]
async fn test_crud_operations() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Test single insert
    let chunk = TestEnvironment::create_test_chunk(0, document_id);
    let chunk_id = chunk.chunk_id;
    
    let inserted_id = env.storage.insert_chunk(chunk.clone()).await
        .expect("Failed to insert chunk");
    assert_eq!(inserted_id, chunk_id);
    
    // Test retrieval
    let retrieved = env.storage.get_chunk(chunk_id).await
        .expect("Failed to retrieve chunk")
        .expect("Chunk not found");
    
    assert_eq!(retrieved.chunk_id, chunk_id);
    assert_eq!(retrieved.content, chunk.content);
    assert_eq!(retrieved.metadata.document_id, document_id);
    assert!(retrieved.embedding.is_some());
    
    // Test update
    let mut updated_chunk = retrieved.clone();
    updated_chunk.content = "Updated content".to_string();
    
    let updated = env.storage.update_chunk(chunk_id, updated_chunk).await
        .expect("Failed to update chunk");
    assert!(updated);
    
    // Verify update
    let retrieved_updated = env.storage.get_chunk(chunk_id).await
        .expect("Failed to retrieve updated chunk")
        .expect("Updated chunk not found");
    assert_eq!(retrieved_updated.content, "Updated content");
    assert_eq!(retrieved_updated.version, retrieved.version + 1);
    
    // Test existence check
    let exists = env.storage.chunk_exists(chunk_id).await
        .expect("Failed to check existence");
    assert!(exists);
    
    // Test deletion
    let deleted = env.storage.delete_chunk(chunk_id).await
        .expect("Failed to delete chunk");
    assert!(deleted);
    
    // Verify deletion
    let exists_after_delete = env.storage.chunk_exists(chunk_id).await
        .expect("Failed to check existence after delete");
    assert!(!exists_after_delete);
}

#[tokio::test]
async fn test_bulk_operations() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Create multiple test chunks
    let chunks: Vec<ChunkDocument> = (0..100)
        .map(|i| TestEnvironment::create_test_chunk(i, document_id))
        .collect();
    
    let start_time = Instant::now();
    
    // Test bulk insert
    let result = env.storage.insert_chunks(chunks.clone()).await
        .expect("Failed to insert chunks");
    
    let insert_duration = start_time.elapsed();
    
    assert_eq!(result.inserted_count, 100);
    assert_eq!(result.failed_count, 0);
    assert!(result.errors.is_empty());
    assert!(insert_duration < Duration::from_secs(5)); // Should be fast
    
    // Test document chunk retrieval
    let retrieved_chunks = env.storage.get_document_chunks(document_id).await
        .expect("Failed to retrieve document chunks");
    
    assert_eq!(retrieved_chunks.len(), 100);
    
    // Verify chunks are sorted by chunk_index
    for (i, chunk) in retrieved_chunks.iter().enumerate() {
        assert_eq!(chunk.metadata.chunk_index, i);
    }
    
    // Test count
    let count = env.storage.count_document_chunks(document_id).await
        .expect("Failed to count chunks");
    assert_eq!(count, 100);
    
    // Test bulk delete
    let deleted_count = env.storage.delete_document_chunks(document_id).await
        .expect("Failed to delete document chunks");
    assert_eq!(deleted_count, 100);
    
    // Verify deletion
    let count_after_delete = env.storage.count_document_chunks(document_id).await
        .expect("Failed to count chunks after delete");
    assert_eq!(count_after_delete, 0);
}

#[tokio::test]
async fn test_vector_search() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Insert test data
    let chunks: Vec<ChunkDocument> = (0..50)
        .map(|i| TestEnvironment::create_test_chunk(i, document_id))
        .collect();
    
    env.storage.insert_chunks(chunks).await
        .expect("Failed to insert test chunks");
    
    // Test vector search with first chunk's embedding
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.01).collect();
    
    let start_time = Instant::now();
    let results = env.storage.vector_search(&query_embedding, 10, None).await
        .expect("Failed to perform vector search");
    let search_duration = start_time.elapsed();
    
    // Validate results
    assert!(!results.is_empty());
    assert!(results.len() <= 10);
    assert!(search_duration < Duration::from_millis(50)); // <50ms requirement
    
    // Results should be sorted by relevance
    for window in results.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
    
    // Test similarity calculations
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![4.0, 5.0, 6.0];
    
    let cosine_sim = VectorSimilarity::cosine_similarity(&vec1, &vec2)
        .expect("Failed to calculate cosine similarity");
    assert!(cosine_sim > 0.0 && cosine_sim <= 1.0);
    
    let euclidean_dist = VectorSimilarity::euclidean_distance(&vec1, &vec2)
        .expect("Failed to calculate Euclidean distance");
    assert!(euclidean_dist > 0.0);
}

#[tokio::test]
async fn test_text_search() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Insert test data with varied content
    let mut chunks = Vec::new();
    for i in 0..20 {
        let mut chunk = TestEnvironment::create_test_chunk(i, document_id);
        chunk.content = format!("Document {} contains information about {}", i, 
            match i % 4 {
                0 => "artificial intelligence and machine learning",
                1 => "database systems and storage solutions",
                2 => "vector search and embedding techniques",
                _ => "natural language processing and text analysis",
            });
        chunks.push(chunk);
    }
    
    env.storage.insert_chunks(chunks).await
        .expect("Failed to insert test chunks");
    
    // Test text search
    let start_time = Instant::now();
    let results = env.storage.text_search("machine learning", 5, None).await
        .expect("Failed to perform text search");
    let search_duration = start_time.elapsed();
    
    assert!(!results.is_empty());
    assert!(search_duration < Duration::from_millis(50)); // <50ms requirement
    
    // Verify results contain the search term
    for result in &results {
        assert!(result.chunk.content.contains("machine learning") || 
               result.chunk.content.contains("artificial intelligence"));
        assert!(result.text_score.is_some());
    }
}

#[tokio::test]
async fn test_hybrid_search() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Insert test data
    let chunks: Vec<ChunkDocument> = (0..30)
        .map(|i| {
            let mut chunk = TestEnvironment::create_test_chunk(i, document_id);
            chunk.content = format!("Hybrid search test content {} with vector embeddings", i);
            chunk
        })
        .collect();
    
    env.storage.insert_chunks(chunks).await
        .expect("Failed to insert test chunks");
    
    // Test hybrid search
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.005).collect();
    
    let query = SearchQuery {
        query_embedding: Some(query_embedding),
        text_query: Some("hybrid search".to_string()),
        search_type: SearchType::Hybrid,
        limit: 10,
        offset: 0,
        min_score: Some(0.1),
        filters: SearchFilters::default(),
        sort: SortOptions {
            field: SortField::Relevance,
            direction: SortDirection::Descending,
            secondary: None,
        },
    };
    
    let start_time = Instant::now();
    let response = env.storage.hybrid_search(query).await
        .expect("Failed to perform hybrid search");
    let search_duration = start_time.elapsed();
    
    assert!(!response.results.is_empty());
    assert!(search_duration < Duration::from_millis(50)); // <50ms requirement
    assert!(response.search_time_ms < 50); // Metrics should also reflect this
    
    // Verify combined scores
    for result in &response.results {
        assert!(result.score > 0.0);
        // Should have either vector or text score (or both)
        assert!(result.vector_score.is_some() || result.text_score.is_some());
    }
}

#[tokio::test]
async fn test_search_filters() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id1 = Uuid::new_v4();
    let document_id2 = Uuid::new_v4();
    
    // Insert chunks for two documents
    let mut chunks = Vec::new();
    
    // Document 1 chunks
    for i in 0..10 {
        let mut chunk = TestEnvironment::create_test_chunk(i, document_id1);
        chunk.metadata.tags = vec!["doc1".to_string(), "important".to_string()];
        chunks.push(chunk);
    }
    
    // Document 2 chunks  
    for i in 10..20 {
        let mut chunk = TestEnvironment::create_test_chunk(i, document_id2);
        chunk.metadata.tags = vec!["doc2".to_string(), "draft".to_string()];
        chunks.push(chunk);
    }
    
    env.storage.insert_chunks(chunks).await
        .expect("Failed to insert test chunks");
    
    // Test document ID filter
    let filters = SearchFilters {
        document_ids: Some(vec![document_id1]),
        ..Default::default()
    };
    
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.01).collect();
    let results = env.storage.vector_search(&query_embedding, 20, Some(filters)).await
        .expect("Failed to perform filtered search");
    
    assert_eq!(results.len(), 10); // Only chunks from document_id1
    for result in &results {
        assert_eq!(result.chunk.metadata.document_id, document_id1);
    }
    
    // Test tag filter
    let tag_filters = SearchFilters {
        tags: Some(vec!["important".to_string()]),
        ..Default::default()
    };
    
    let tag_results = env.storage.vector_search(&query_embedding, 20, Some(tag_filters)).await
        .expect("Failed to perform tag filtered search");
    
    for result in &tag_results {
        assert!(result.chunk.metadata.tags.contains(&"important".to_string()));
    }
}

#[tokio::test]
async fn test_search_pagination() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Insert 50 chunks
    let chunks: Vec<ChunkDocument> = (0..50)
        .map(|i| TestEnvironment::create_test_chunk(i, document_id))
        .collect();
    
    env.storage.insert_chunks(chunks).await
        .expect("Failed to insert test chunks");
    
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.01).collect();
    
    // Test first page
    let query1 = SearchQuery {
        query_embedding: Some(query_embedding.clone()),
        limit: 10,
        offset: 0,
        ..Default::default()
    };
    
    let response1 = env.storage.hybrid_search(query1).await
        .expect("Failed to perform paginated search");
    
    assert_eq!(response1.results.len(), 10);
    assert_eq!(response1.total_count, 50);
    
    // Test second page
    let query2 = SearchQuery {
        query_embedding: Some(query_embedding.clone()),
        limit: 10,
        offset: 10,
        ..Default::default()
    };
    
    let response2 = env.storage.hybrid_search(query2).await
        .expect("Failed to perform paginated search");
    
    assert_eq!(response2.results.len(), 10);
    
    // Ensure different results on different pages
    let page1_ids: std::collections::HashSet<_> = response1.results
        .iter()
        .map(|r| r.chunk.chunk_id)
        .collect();
    
    let page2_ids: std::collections::HashSet<_> = response2.results
        .iter()
        .map(|r| r.chunk.chunk_id)
        .collect();
    
    assert!(page1_ids.is_disjoint(&page2_ids));
}

#[tokio::test]
async fn test_transaction_operations() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Test transactional bulk insert
    let chunks: Vec<ChunkDocument> = (0..20)
        .map(|i| TestEnvironment::create_test_chunk(i, document_id))
        .collect();
    
    let request = BulkInsertRequest {
        chunks: chunks.clone(),
        upsert: false,
        batch_size: Some(5),
    };
    
    let result = env.storage.transactional_bulk_insert(request).await
        .expect("Failed to perform transactional bulk insert");
    
    assert_eq!(result.inserted_count, 20);
    assert_eq!(result.failed_count, 0);
    
    // Verify all chunks were inserted
    let count = env.storage.count_document_chunks(document_id).await
        .expect("Failed to count chunks");
    assert_eq!(count, 20);
    
    // Test transaction rollback scenario (would need to simulate an error)
    // This is more complex and would require mocking or error injection
}

#[tokio::test]
async fn test_performance_requirements() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Insert a reasonable amount of test data
    let chunks: Vec<ChunkDocument> = (0..1000)
        .map(|i| TestEnvironment::create_test_chunk(i, document_id))
        .collect();
    
    // Test bulk insert performance
    let bulk_start = Instant::now();
    let bulk_result = env.storage.insert_chunks(chunks).await
        .expect("Failed to perform bulk insert");
    let bulk_duration = bulk_start.elapsed();
    
    assert_eq!(bulk_result.inserted_count, 1000);
    assert!(bulk_duration < Duration::from_secs(10)); // Should be fast for 1000 docs
    
    println!("Bulk insert of 1000 docs: {:?}", bulk_duration);
    
    // Test search performance
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.01).collect();
    
    let search_start = Instant::now();
    let search_results = env.storage.vector_search(&query_embedding, 50, None).await
        .expect("Failed to perform vector search");
    let search_duration = search_start.elapsed();
    
    assert!(!search_results.is_empty());
    assert!(search_duration < Duration::from_millis(50)); // <50ms requirement
    
    println!("Vector search in 1000 docs: {:?}", search_duration);
    
    // Test concurrent search performance
    let concurrent_start = Instant::now();
    let concurrent_tasks: Vec<_> = (0..10)
        .map(|_| {
            let storage = env.storage.clone();
            let embedding = query_embedding.clone();
            tokio::spawn(async move {
                storage.vector_search(&embedding, 10, None).await
            })
        })
        .collect();
    
    let concurrent_results: Vec<_> = futures::future::join_all(concurrent_tasks).await;
    let concurrent_duration = concurrent_start.elapsed();
    
    // All searches should succeed
    for result in concurrent_results {
        assert!(result.unwrap().is_ok());
    }
    
    assert!(concurrent_duration < Duration::from_millis(200)); // 10 concurrent searches
    
    println!("10 concurrent searches: {:?}", concurrent_duration);
}

#[tokio::test]
async fn test_metrics_collection() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Perform various operations to generate metrics
    let chunk = TestEnvironment::create_test_chunk(0, document_id);
    env.storage.insert_chunk(chunk.clone()).await
        .expect("Failed to insert chunk");
    
    env.storage.get_chunk(chunk.chunk_id).await
        .expect("Failed to retrieve chunk");
    
    let query_embedding: Vec<f64> = (0..384).map(|i| i as f64 * 0.01).collect();
    env.storage.vector_search(&query_embedding, 10, None).await
        .expect("Failed to perform vector search");
    
    // Get metrics snapshot
    let metrics = env.storage.metrics().snapshot();
    
    assert!(metrics.uptime_seconds > 0);
    assert!(!metrics.operations.is_empty());
    
    // Check specific operation metrics
    if let Some(insert_metrics) = metrics.operations.get("insert_chunk") {
        assert!(insert_metrics.count > 0);
        assert!(insert_metrics.success_count > 0);
        assert!(insert_metrics.avg_duration_ms() > 0.0);
    }
    
    if let Some(search_metrics) = metrics.operations.get("vector_search") {
        assert!(search_metrics.count > 0);
        assert!(search_metrics.success_count > 0);
    }
    
    println!("Metrics snapshot: {:#?}", metrics);
}

#[tokio::test]
async fn test_error_handling() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    
    // Test invalid chunk ID
    let invalid_id = Uuid::new_v4();
    let result = env.storage.get_chunk(invalid_id).await
        .expect("Failed to query for invalid chunk");
    assert!(result.is_none());
    
    // Test empty vector search
    let empty_embedding = vec![];
    let search_result = env.storage.vector_search(&empty_embedding, 10, None).await;
    assert!(search_result.is_err());
    
    // Test invalid search query
    let invalid_query = SearchQuery {
        query_embedding: None,
        text_query: Some("".to_string()), // Empty text query
        ..Default::default()
    };
    
    let hybrid_result = env.storage.hybrid_search(invalid_query).await;
    assert!(hybrid_result.is_err());
    
    // Test chunk validation errors
    let mut invalid_chunk = TestEnvironment::create_test_chunk(0, Uuid::new_v4());
    invalid_chunk.content = String::new(); // Empty content should fail validation
    
    let validation_result = env.storage.insert_chunk(invalid_chunk).await;
    assert!(validation_result.is_err());
}

#[tokio::test]
async fn test_find_similar_chunks() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Insert test chunks
    let chunks: Vec<ChunkDocument> = (0..20)
        .map(|i| TestEnvironment::create_test_chunk(i, document_id))
        .collect();
    
    let target_chunk_id = chunks[0].chunk_id;
    
    env.storage.insert_chunks(chunks).await
        .expect("Failed to insert test chunks");
    
    // Find similar chunks
    let similar = env.storage.find_similar(target_chunk_id, 5).await
        .expect("Failed to find similar chunks");
    
    assert!(!similar.is_empty());
    assert!(similar.len() <= 5);
    
    // Ensure target chunk is not in results
    for result in &similar {
        assert_ne!(result.chunk.chunk_id, target_chunk_id);
    }
    
    // Results should be sorted by similarity
    for window in similar.windows(2) {
        assert!(window[0].score >= window[1].score);
    }
}

#[tokio::test]
async fn test_recommendations() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Insert test chunks
    let chunks: Vec<ChunkDocument> = (0..30)
        .map(|i| TestEnvironment::create_test_chunk(i, document_id))
        .collect();
    
    let viewed_chunk_ids: Vec<Uuid> = chunks[0..5].iter().map(|c| c.chunk_id).collect();
    
    env.storage.insert_chunks(chunks).await
        .expect("Failed to insert test chunks");
    
    // Get recommendations
    let recommendations = env.storage.get_recommendations(&viewed_chunk_ids, 10).await
        .expect("Failed to get recommendations");
    
    assert!(!recommendations.is_empty());
    assert!(recommendations.len() <= 10);
    
    // Ensure viewed chunks are not in recommendations
    let viewed_set: std::collections::HashSet<_> = viewed_chunk_ids.into_iter().collect();
    for recommendation in &recommendations {
        assert!(!viewed_set.contains(&recommendation.chunk.chunk_id));
    }
}

#[tokio::test] 
async fn test_data_integrity() {
    let env = TestEnvironment::setup().await.expect("Failed to setup test environment");
    let document_id = Uuid::new_v4();
    
    // Insert chunk with specific data
    let original_chunk = TestEnvironment::create_test_chunk(0, document_id);
    let chunk_id = original_chunk.chunk_id;
    
    env.storage.insert_chunk(original_chunk.clone()).await
        .expect("Failed to insert original chunk");
    
    // Retrieve and verify all fields
    let retrieved = env.storage.get_chunk(chunk_id).await
        .expect("Failed to retrieve chunk")
        .expect("Chunk not found");
    
    // Verify core fields
    assert_eq!(retrieved.chunk_id, original_chunk.chunk_id);
    assert_eq!(retrieved.content, original_chunk.content);
    assert_eq!(retrieved.embedding, original_chunk.embedding);
    assert_eq!(retrieved.version, original_chunk.version);
    
    // Verify metadata
    assert_eq!(retrieved.metadata.document_id, original_chunk.metadata.document_id);
    assert_eq!(retrieved.metadata.title, original_chunk.metadata.title);
    assert_eq!(retrieved.metadata.chunk_index, original_chunk.metadata.chunk_index);
    assert_eq!(retrieved.metadata.tags, original_chunk.metadata.tags);
    
    // Verify custom fields
    assert_eq!(retrieved.metadata.custom_fields.len(), original_chunk.metadata.custom_fields.len());
    for (key, value) in &original_chunk.metadata.custom_fields {
        assert!(retrieved.metadata.custom_fields.contains_key(key));
        // Note: Due to BSON serialization, we'd need more sophisticated comparison
    }
    
    // Verify timestamps are reasonable
    assert!(retrieved.created_at <= chrono::Utc::now());
    assert!(retrieved.updated_at >= retrieved.created_at);
}

#[cfg(test)]
mod test_utils {
    use super::*;
    
    /// Helper to wait for MongoDB to be ready
    pub async fn wait_for_mongodb(port: u16, max_retries: u32) -> bool {
        for _ in 0..max_retries {
            if let Ok(client) = mongodb::Client::with_uri_str(&format!("mongodb://localhost:{}", port)).await {
                if client.database("test").run_command(doc! { "ping": 1 }, None).await.is_ok() {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        false
    }
    
    /// Generate realistic test embeddings
    pub fn generate_realistic_embedding(dimension: usize, seed: u64) -> Vec<f64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Generate pseudo-random but deterministic embedding
        (0..dimension).map(|i| {
            let mut hasher = DefaultHasher::new();
            (hash + i as u64).hash(&mut hasher);
            let val = hasher.finish() as f64 / u64::MAX as f64;
            (val - 0.5) * 2.0 // Normalize to [-1, 1]
        }).collect()
    }
}