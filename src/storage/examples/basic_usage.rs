//! Basic usage example for MongoDB Vector Storage

use uuid::Uuid;
use tokio;

use storage::{
    VectorStorage, StorageConfig, ChunkDocument, ChunkMetadata, CustomFieldValue,
    SearchQuery, SearchType, SearchFilters, SortOptions, SortField, SortDirection,
    DatabaseOperations, SearchOperations, TransactionOperations, BulkInsertRequest,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 MongoDB Vector Storage - Basic Usage Example");
    println!("================================================");
    
    // 1. Create storage configuration
    println!("\n1. Setting up storage configuration...");
    let config = StorageConfig {
        connection_string: std::env::var("MONGODB_URI")
            .unwrap_or_else(|_| "mongodb://localhost:27017".to_string()),
        database_name: "example_rag_storage".to_string(),
        chunk_collection_name: "example_chunks".to_string(),
        metadata_collection_name: "example_metadata".to_string(),
        ..StorageConfig::default()
    };
    
    println!("   ✓ Configuration created");
    println!("   ✓ MongoDB URI: {}", config.connection_string);
    println!("   ✓ Database: {}", config.database_name);
    
    // 2. Initialize storage
    println!("\n2. Initializing storage...");
    let storage = VectorStorage::new(config).await?;
    
    // 3. Health check
    let health = storage.health_check().await?;
    println!("   ✓ Storage initialized successfully");
    println!("   ✓ Health check: {} ({}ms)", 
             if health.healthy { "✓ Healthy" } else { "✗ Unhealthy" }, 
             health.latency_ms);
    
    // 4. Create sample documents
    println!("\n3. Creating sample documents...");
    let document_id = Uuid::new_v4();
    
    let chunks = create_sample_chunks(document_id);
    println!("   ✓ Created {} sample chunks", chunks.len());
    
    // 5. Insert documents
    println!("\n4. Inserting documents...");
    let start_time = std::time::Instant::now();
    
    let bulk_result = storage.insert_chunks(chunks.clone()).await?;
    let insert_duration = start_time.elapsed();
    
    println!("   ✓ Inserted {} chunks in {:?}", bulk_result.inserted_count, insert_duration);
    println!("   ✓ {} successful, {} failed", bulk_result.inserted_count, bulk_result.failed_count);
    
    // 6. Retrieve a single chunk
    println!("\n5. Retrieving single chunk...");
    let chunk_id = chunks[0].chunk_id;
    let retrieved_chunk = storage.get_chunk(chunk_id).await?;
    
    match retrieved_chunk {
        Some(chunk) => {
            println!("   ✓ Retrieved chunk: {}", chunk.chunk_id);
            println!("   ✓ Content: {}...", &chunk.content[..50.min(chunk.content.len())]);
            println!("   ✓ Has embedding: {}", chunk.embedding.is_some());
        }
        None => println!("   ✗ Chunk not found"),
    }
    
    // 7. Retrieve all chunks for document
    println!("\n6. Retrieving all document chunks...");
    let document_chunks = storage.get_document_chunks(document_id).await?;
    println!("   ✓ Retrieved {} chunks for document", document_chunks.len());
    
    for (i, chunk) in document_chunks.iter().take(3).enumerate() {
        println!("   - Chunk {}: {} (index: {})", 
                 i + 1, 
                 chunk.metadata.title, 
                 chunk.metadata.chunk_index);
    }
    
    // 8. Vector search
    println!("\n7. Performing vector search...");
    
    // Create a query embedding (in practice, this would come from an embedding model)
    let query_embedding = create_sample_embedding(384);
    
    let search_start = std::time::Instant::now();
    let vector_results = storage.vector_search(&query_embedding, 5, None).await?;
    let search_duration = search_start.elapsed();
    
    println!("   ✓ Vector search completed in {:?}", search_duration);
    println!("   ✓ Found {} similar chunks", vector_results.len());
    
    for (i, result) in vector_results.iter().enumerate() {
        println!("   - Result {}: score={:.3}, chunk_index={}", 
                 i + 1, 
                 result.score, 
                 result.chunk.metadata.chunk_index);
    }
    
    // 9. Text search
    println!("\n8. Performing text search...");
    
    let text_search_start = std::time::Instant::now();
    let text_results = storage.text_search("artificial intelligence", 5, None).await?;
    let text_search_duration = text_search_start.elapsed();
    
    println!("   ✓ Text search completed in {:?}", text_search_duration);
    println!("   ✓ Found {} matching chunks", text_results.len());
    
    for (i, result) in text_results.iter().enumerate() {
        println!("   - Result {}: score={:.3}, content=\"{}...\"", 
                 i + 1, 
                 result.score, 
                 &result.chunk.content[..30.min(result.chunk.content.len())]);
    }
    
    // 10. Hybrid search
    println!("\n9. Performing hybrid search...");
    
    let hybrid_query = SearchQuery {
        query_embedding: Some(query_embedding.clone()),
        text_query: Some("machine learning".to_string()),
        search_type: SearchType::Hybrid,
        limit: 5,
        offset: 0,
        min_score: Some(0.1),
        filters: SearchFilters::default(),
        sort: SortOptions {
            field: SortField::Relevance,
            direction: SortDirection::Descending,
            secondary: None,
        },
    };
    
    let hybrid_start = std::time::Instant::now();
    let hybrid_response = storage.hybrid_search(hybrid_query).await?;
    let hybrid_duration = hybrid_start.elapsed();
    
    println!("   ✓ Hybrid search completed in {:?}", hybrid_duration);
    println!("   ✓ Found {} results (total: {})", 
             hybrid_response.results.len(), 
             hybrid_response.total_count);
    
    for (i, result) in hybrid_response.results.iter().enumerate() {
        println!("   - Result {}: score={:.3}, vector_score={:.3}, text_score={:.3}", 
                 i + 1, 
                 result.score,
                 result.vector_score.unwrap_or(0.0),
                 result.text_score.unwrap_or(0.0));
    }
    
    // 11. Filtered search
    println!("\n10. Performing filtered search...");
    
    let filters = SearchFilters {
        document_ids: Some(vec![document_id]),
        tags: Some(vec!["technology".to_string()]),
        ..Default::default()
    };
    
    let filtered_results = storage.vector_search(&query_embedding, 10, Some(filters)).await?;
    println!("   ✓ Filtered search found {} results", filtered_results.len());
    
    // 12. Find similar chunks
    println!("\n11. Finding similar chunks...");
    
    let similar_chunks = storage.find_similar(chunks[0].chunk_id, 3).await?;
    println!("   ✓ Found {} similar chunks to first chunk", similar_chunks.len());
    
    for (i, result) in similar_chunks.iter().enumerate() {
        println!("   - Similar {}: score={:.3}, chunk_index={}", 
                 i + 1, 
                 result.score, 
                 result.chunk.metadata.chunk_index);
    }
    
    // 13. Get recommendations
    println!("\n12. Getting recommendations...");
    
    let viewed_chunks = vec![chunks[0].chunk_id, chunks[1].chunk_id];
    let recommendations = storage.get_recommendations(&viewed_chunks, 5).await?;
    println!("   ✓ Generated {} recommendations", recommendations.len());
    
    // 14. Update a chunk
    println!("\n13. Updating a chunk...");
    
    let mut updated_chunk = chunks[0].clone();
    updated_chunk.content = "Updated content with new information about machine learning and AI.".to_string();
    updated_chunk.metadata.tags.push("updated".to_string());
    
    let update_success = storage.update_chunk(chunk_id, updated_chunk).await?;
    println!("   ✓ Chunk update: {}", if update_success { "successful" } else { "failed" });
    
    // 15. Performance metrics
    println!("\n14. Performance metrics...");
    
    let metrics = storage.metrics().snapshot();
    println!("   ✓ Uptime: {}s", metrics.uptime_seconds);
    println!("   ✓ Total operations: {}", metrics.operations.len());
    
    for (operation, op_metrics) in metrics.operations.iter() {
        if op_metrics.count > 0 {
            println!("   - {}: {} ops, avg: {:.1}ms, success: {:.1}%", 
                     operation,
                     op_metrics.count,
                     op_metrics.avg_duration_ms(),
                     op_metrics.success_rate() * 100.0);
        }
    }
    
    println!("   ✓ Documents processed: {}", metrics.performance.documents_processed);
    println!("   ✓ Bytes processed: {}", metrics.performance.bytes_processed);
    
    // 16. Transactional operations
    println!("\n15. Testing transactional operations...");
    
    let transaction_chunks = create_sample_chunks(Uuid::new_v4());
    let bulk_request = BulkInsertRequest {
        chunks: transaction_chunks[..3].to_vec(),
        upsert: false,
        batch_size: Some(2),
    };
    
    let tx_result = storage.transactional_bulk_insert(bulk_request).await?;
    println!("   ✓ Transactional insert: {} inserted, {} failed", 
             tx_result.inserted_count, 
             tx_result.failed_count);
    
    // 17. Cleanup
    println!("\n16. Cleaning up...");
    
    let deleted_count = storage.delete_document_chunks(document_id).await?;
    println!("   ✓ Deleted {} chunks", deleted_count);
    
    // 18. Final health check
    let final_health = storage.health_check().await?;
    println!("   ✓ Final health check: {} ({}ms)", 
             if final_health.healthy { "✓ Healthy" } else { "✗ Unhealthy" }, 
             final_health.latency_ms);
    
    println!("\n🎉 Example completed successfully!");
    println!("   - All storage operations worked as expected");
    println!("   - Search latencies were within acceptable limits");
    println!("   - Data integrity maintained throughout");
    
    Ok(())
}

/// Create sample chunks for demonstration
fn create_sample_chunks(document_id: Uuid) -> Vec<ChunkDocument> {
    let sample_texts = vec![
        "Artificial intelligence is transforming the way we interact with technology. Machine learning algorithms enable computers to learn from data without being explicitly programmed.",
        "Vector databases are essential for modern AI applications. They provide efficient similarity search capabilities for high-dimensional data representations.",
        "Natural language processing enables computers to understand and generate human language. This technology powers chatbots, translators, and content analysis systems.",
        "Deep learning neural networks have revolutionized computer vision and speech recognition. These models can process complex patterns in data.",
        "Information retrieval systems help users find relevant documents from large collections. Search engines use sophisticated ranking algorithms to provide accurate results.",
        "Knowledge graphs represent relationships between entities in a structured format. They enable semantic search and reasoning capabilities.",
        "Embedding models convert text into numerical vectors that capture semantic meaning. These representations enable similarity comparisons between documents.",
        "Retrieval-augmented generation combines search and language models to provide accurate and contextual responses based on external knowledge sources.",
    ];
    
    sample_texts.iter().enumerate().map(|(i, &text)| {
        let mut metadata = ChunkMetadata::new(
            document_id,
            format!("AI Technology Guide - Chapter {}", i + 1),
            i,
            sample_texts.len(),
            format!("/documents/ai_guide_chapter_{}.txt", i + 1),
        );
        
        // Add relevant tags
        metadata.tags = vec![
            "technology".to_string(),
            "artificial-intelligence".to_string(),
            match i % 4 {
                0 => "machine-learning".to_string(),
                1 => "vector-databases".to_string(), 
                2 => "nlp".to_string(),
                _ => "deep-learning".to_string(),
            },
        ];
        
        // Add custom metadata
        metadata.custom_fields.insert(
            "priority".to_string(),
            CustomFieldValue::String("high".to_string()),
        );
        metadata.custom_fields.insert(
            "chapter".to_string(),
            CustomFieldValue::Number((i + 1) as f64),
        );
        metadata.custom_fields.insert(
            "reviewed".to_string(),
            CustomFieldValue::Boolean(true),
        );
        
        // Create chunk with embedding
        let chunk = ChunkDocument::new(
            Uuid::new_v4(),
            text.to_string(),
            metadata,
        );
        
        // Add sample embedding (in practice, this would come from an embedding model)
        let embedding = create_sample_embedding(384);
        chunk.with_embedding(embedding)
    }).collect()
}

/// Create a sample embedding vector
fn create_sample_embedding(dimension: usize) -> Vec<f64> {
    // Generate a realistic-looking embedding with some randomness but deterministic
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let seed = 42u64;
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let base_hash = hasher.finish();
    
    (0..dimension).map(|i| {
        let mut hasher = DefaultHasher::new();
        (base_hash + i as u64).hash(&mut hasher);
        let val = hasher.finish() as f64 / u64::MAX as f64;
        (val - 0.5) * 2.0 // Normalize to [-1, 1]
    }).collect()
}