//! Vector search demonstration with realistic data and performance analysis

use std::time::{Duration, Instant};
use uuid::Uuid;
use tokio;

use storage::{
    VectorStorage, StorageConfig, ChunkDocument, ChunkMetadata, CustomFieldValue,
    SearchQuery, SearchType, SearchFilters,
    VectorSimilarity, DatabaseOperations, SearchOperations,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🔍 MongoDB Vector Storage - Vector Search Demo");
    println!("===============================================");
    
    // Setup storage
    let config = StorageConfig {
        connection_string: std::env::var("MONGODB_URI")
            .unwrap_or_else(|_| "mongodb://localhost:27017".to_string()),
        database_name: "vector_search_demo".to_string(),
        ..StorageConfig::for_testing()
    };
    
    let storage = VectorStorage::new(config).await?;
    
    // Create realistic dataset
    println!("\n📚 Creating realistic dataset...");
    let documents = create_realistic_documents(1000);
    println!("   ✓ Created {} documents with embeddings", documents.len());
    
    // Insert documents
    println!("\n💾 Inserting documents...");
    let insert_start = Instant::now();
    let bulk_result = storage.insert_chunks(documents.clone()).await?;
    let insert_duration = insert_start.elapsed();
    
    println!("   ✓ Inserted {} documents in {:?}", bulk_result.inserted_count, insert_duration);
    println!("   ✓ Average insert rate: {:.0} docs/sec", 
             documents.len() as f64 / insert_duration.as_secs_f64());
    
    // Demonstration 1: Basic Vector Search
    println!("\n🎯 Demo 1: Basic Vector Search");
    println!("==============================");
    
    demonstrate_basic_vector_search(&storage, &documents).await?;
    
    // Demonstration 2: Similarity Metrics Comparison
    println!("\n📊 Demo 2: Similarity Metrics Comparison");
    println!("========================================");
    
    demonstrate_similarity_metrics().await?;
    
    // Demonstration 3: Search Performance Analysis
    println!("\n⚡ Demo 3: Search Performance Analysis");
    println!("=====================================");
    
    demonstrate_search_performance(&storage).await?;
    
    // Demonstration 4: Filtered Search
    println!("\n🔍 Demo 4: Filtered Vector Search");
    println!("=================================");
    
    demonstrate_filtered_search(&storage, documents[0].metadata.document_id).await?;
    
    // Demonstration 5: Different K Values
    println!("\n📈 Demo 5: K-Nearest Neighbors Analysis");
    println!("======================================");
    
    demonstrate_k_values(&storage, &documents).await?;
    
    // Demonstration 6: Hybrid vs Pure Vector Search
    println!("\n🔄 Demo 6: Hybrid vs Pure Vector Search");
    println!("=======================================");
    
    demonstrate_hybrid_comparison(&storage, &documents).await?;
    
    // Demonstration 7: Search Accuracy Analysis
    println!("\n🎯 Demo 7: Search Accuracy Analysis");
    println!("===================================");
    
    demonstrate_search_accuracy(&storage, &documents).await?;
    
    // Demonstration 8: Concurrent Search Performance
    println!("\n⚡ Demo 8: Concurrent Search Performance");
    println!("=======================================");
    
    demonstrate_concurrent_search(&storage).await?;
    
    // Performance Summary
    println!("\n📋 Performance Summary");
    println!("=====================");
    
    let metrics = storage.metrics().snapshot();
    print_performance_summary(&metrics);
    
    // Cleanup
    println!("\n🧹 Cleaning up...");
    let unique_docs: Vec<Uuid> = documents.iter()
        .map(|d| d.metadata.document_id)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    
    let mut total_deleted = 0;
    for doc_id in unique_docs {
        let deleted = storage.delete_document_chunks(doc_id).await?;
        total_deleted += deleted;
    }
    
    println!("   ✓ Deleted {} total chunks", total_deleted);
    println!("\n🎉 Vector search demo completed successfully!");
    
    Ok(())
}

async fn demonstrate_basic_vector_search(
    storage: &VectorStorage, 
    documents: &[ChunkDocument]
) -> Result<(), Box<dyn std::error::Error>> {
    // Use the first document's embedding as query
    let query_embedding = documents[0].embedding.as_ref().unwrap();
    
    println!("   Query: Searching for documents similar to: \"{}...\"", 
             &documents[0].content[..50]);
    
    let search_start = Instant::now();
    let results = storage.vector_search(query_embedding, 5, None).await?;
    let search_duration = search_start.elapsed();
    
    println!("   ✓ Search completed in {:?} (<50ms requirement: {})", 
             search_duration,
             if search_duration < Duration::from_millis(50) { "✓ PASS" } else { "✗ FAIL" });
    
    println!("   ✓ Found {} similar documents:", results.len());
    
    for (i, result) in results.iter().enumerate() {
        let content_preview = &result.chunk.content[..50.min(result.chunk.content.len())];
        println!("     {}. Score: {:.4} | \"{}...\"", 
                 i + 1, 
                 result.score, 
                 content_preview);
    }
    
    // Verify the first result is the query document itself (should have highest similarity)
    if let Some(first_result) = results.first() {
        if first_result.chunk.chunk_id == documents[0].chunk_id {
            println!("   ✓ Self-similarity check passed (query doc is top result)");
        } else {
            println!("   ⚠ Self-similarity check: query doc not top result (similarity: {:.4})", 
                     first_result.score);
        }
    }
    
    Ok(())
}

async fn demonstrate_similarity_metrics() -> Result<(), Box<dyn std::error::Error>> {
    // Create test vectors for similarity comparison
    let vec1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let vec2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // 2x vec1
    let vec3 = vec![1.0, -1.0, 2.0, -2.0, 3.0]; // Different pattern
    let vec4 = vec![0.0, 0.0, 0.0, 0.0, 1.0]; // Sparse
    
    println!("   Comparing different similarity metrics:");
    println!("   Vector 1: {:?}", vec1);
    println!("   Vector 2: {:?} (2x Vector 1)", vec2);
    println!("   Vector 3: {:?} (different pattern)", vec3);
    println!("   Vector 4: {:?} (sparse)", vec4);
    
    let vectors = vec![
        ("Vec1 vs Vec2", &vec1, &vec2),
        ("Vec1 vs Vec3", &vec1, &vec3),
        ("Vec1 vs Vec4", &vec1, &vec4),
        ("Vec2 vs Vec3", &vec2, &vec3),
    ];
    
    println!("\n   Cosine Similarity Results:");
    for (name, v1, v2) in &vectors {
        let cosine_sim = VectorSimilarity::cosine_similarity(v1, v2)?;
        println!("     {}: {:.4}", name, cosine_sim);
    }
    
    println!("\n   Euclidean Distance Results:");
    for (name, v1, v2) in &vectors {
        let euclidean_dist = VectorSimilarity::euclidean_distance(v1, v2)?;
        let similarity = VectorSimilarity::distance_to_similarity(euclidean_dist, 20.0);
        println!("     {}: {:.4} (distance: {:.4})", name, similarity, euclidean_dist);
    }
    
    // Performance comparison
    let large_vec1: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
    let large_vec2: Vec<f64> = (0..1000).map(|i| (i * 2) as f64 * 0.001).collect();
    
    println!("\n   Performance test (1000-dimensional vectors):");
    
    let cosine_start = Instant::now();
    for _ in 0..1000 {
        VectorSimilarity::cosine_similarity(&large_vec1, &large_vec2)?;
    }
    let cosine_duration = cosine_start.elapsed();
    
    let euclidean_start = Instant::now();
    for _ in 0..1000 {
        VectorSimilarity::euclidean_distance(&large_vec1, &large_vec2)?;
    }
    let euclidean_duration = euclidean_start.elapsed();
    
    println!("     Cosine similarity: {:?} (1000 calculations)", cosine_duration);
    println!("     Euclidean distance: {:?} (1000 calculations)", euclidean_duration);
    
    Ok(())
}

async fn demonstrate_search_performance(
    storage: &VectorStorage
) -> Result<(), Box<dyn std::error::Error>> {
    let query_embedding = create_test_embedding(384, 12345);
    
    // Test different search sizes
    let k_values = vec![1, 5, 10, 20, 50];
    
    println!("   Testing search performance with different K values:");
    
    for k in k_values {
        let mut durations = Vec::new();
        
        // Run multiple searches for better statistics
        for _ in 0..10 {
            let start = Instant::now();
            let results = storage.vector_search(&query_embedding, k, None).await?;
            let duration = start.elapsed();
            durations.push(duration);
            
            // Verify we got results
            assert!(!results.is_empty(), "No results returned for k={}", k);
        }
        
        let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
        let min_duration = durations.iter().min().unwrap();
        let max_duration = durations.iter().max().unwrap();
        
        println!("     k={:2}: avg={:?}, min={:?}, max={:?} | {}",
                 k,
                 avg_duration,
                 min_duration,
                 max_duration,
                 if avg_duration < Duration::from_millis(50) { "✓" } else { "⚠" });
    }
    
    // Test search consistency
    println!("\n   Testing search result consistency:");
    let results1 = storage.vector_search(&query_embedding, 10, None).await?;
    let results2 = storage.vector_search(&query_embedding, 10, None).await?;
    
    if results1.len() == results2.len() {
        let mut consistent = true;
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            if r1.chunk.chunk_id != r2.chunk.chunk_id {
                consistent = false;
                break;
            }
        }
        
        if consistent {
            println!("     ✓ Search results are consistent across calls");
        } else {
            println!("     ⚠ Search results vary between identical calls");
        }
    } else {
        println!("     ⚠ Different number of results returned: {} vs {}", 
                 results1.len(), results2.len());
    }
    
    Ok(())
}

async fn demonstrate_filtered_search(
    storage: &VectorStorage,
    document_id: Uuid,
) -> Result<(), Box<dyn std::error::Error>> {
    let query_embedding = create_test_embedding(384, 54321);
    
    // Unfiltered search baseline
    let unfiltered_start = Instant::now();
    let unfiltered_results = storage.vector_search(&query_embedding, 20, None).await?;
    let unfiltered_duration = unfiltered_start.elapsed();
    
    println!("   Unfiltered search: {} results in {:?}", 
             unfiltered_results.len(), unfiltered_duration);
    
    // Document ID filter
    let doc_filters = SearchFilters {
        document_ids: Some(vec![document_id]),
        ..Default::default()
    };
    
    let filtered_start = Instant::now();
    let filtered_results = storage.vector_search(&query_embedding, 20, Some(doc_filters)).await?;
    let filtered_duration = filtered_start.elapsed();
    
    println!("   Document filtered: {} results in {:?}", 
             filtered_results.len(), filtered_duration);
    
    // Tag filter
    let tag_filters = SearchFilters {
        tags: Some(vec!["technology".to_string()]),
        ..Default::default()
    };
    
    let tag_start = Instant::now();
    let tag_results = storage.vector_search(&query_embedding, 20, Some(tag_filters)).await?;
    let tag_duration = tag_start.elapsed();
    
    println!("   Tag filtered: {} results in {:?}", 
             tag_results.len(), tag_duration);
    
    // Combined filters
    let combined_filters = SearchFilters {
        document_ids: Some(vec![document_id]),
        tags: Some(vec!["technology".to_string()]),
        ..Default::default()
    };
    
    let combined_start = Instant::now();
    let combined_results = storage.vector_search(&query_embedding, 20, Some(combined_filters)).await?;
    let combined_duration = combined_start.elapsed();
    
    println!("   Combined filtered: {} results in {:?}", 
             combined_results.len(), combined_duration);
    
    // Analyze filtering effectiveness
    println!("\n   Filter effectiveness:");
    println!("     Baseline: {} results", unfiltered_results.len());
    println!("     Document filter: {} results ({:.1}% reduction)", 
             filtered_results.len(),
             100.0 * (1.0 - filtered_results.len() as f64 / unfiltered_results.len() as f64));
    println!("     Tag filter: {} results ({:.1}% reduction)", 
             tag_results.len(),
             100.0 * (1.0 - tag_results.len() as f64 / unfiltered_results.len() as f64));
    println!("     Combined filter: {} results ({:.1}% reduction)", 
             combined_results.len(),
             100.0 * (1.0 - combined_results.len() as f64 / unfiltered_results.len() as f64));
    
    Ok(())
}

async fn demonstrate_k_values(
    storage: &VectorStorage,
    documents: &[ChunkDocument],
) -> Result<(), Box<dyn std::error::Error>> {
    let query_embedding = documents[10].embedding.as_ref().unwrap();
    let k_values = vec![1, 3, 5, 10, 20, 50];
    
    println!("   Analyzing K-nearest neighbors for different K values:");
    
    for k in k_values {
        let start = Instant::now();
        let results = storage.vector_search(query_embedding, k, None).await?;
        let duration = start.elapsed();
        
        if results.is_empty() {
            println!("     k={:2}: No results found", k);
            continue;
        }
        
        let avg_score = results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32;
        let min_score = results.iter().map(|r| r.score).fold(f32::INFINITY, f32::min);
        let max_score = results.iter().map(|r| r.score).fold(f32::NEG_INFINITY, f32::max);
        
        println!("     k={:2}: {} results, avg_score={:.4}, range=[{:.4}, {:.4}], time={:?}",
                 k, results.len(), avg_score, min_score, max_score, duration);
        
        // Check for score degradation with larger K
        if k > 1 {
            let score_range = max_score - min_score;
            if score_range > 0.5 {
                println!("           ⚠ Large score range suggests diverse results");
            }
        }
    }
    
    // Analyze score distribution for k=20
    let detailed_results = storage.vector_search(query_embedding, 20, None).await?;
    
    if detailed_results.len() >= 10 {
        println!("\n   Score distribution analysis (k=20):");
        
        let scores: Vec<f32> = detailed_results.iter().map(|r| r.score).collect();
        let percentiles = calculate_percentiles(&scores);
        
        println!("     Min:  {:.4}", scores.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
        println!("     P25:  {:.4}", percentiles.0);
        println!("     P50:  {:.4}", percentiles.1);
        println!("     P75:  {:.4}", percentiles.2);
        println!("     Max:  {:.4}", scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    }
    
    Ok(())
}

async fn demonstrate_hybrid_comparison(
    storage: &VectorStorage,
    _documents: &[ChunkDocument],
) -> Result<(), Box<dyn std::error::Error>> {
    let query_embedding = create_test_embedding(384, 99999);
    let text_query = "artificial intelligence machine learning";
    
    // Pure vector search
    let vector_start = Instant::now();
    let vector_results = storage.vector_search(&query_embedding, 10, None).await?;
    let vector_duration = vector_start.elapsed();
    
    // Pure text search
    let text_start = Instant::now();
    let text_results = storage.text_search(text_query, 10, None).await?;
    let text_duration = text_start.elapsed();
    
    // Hybrid search
    let hybrid_query = SearchQuery {
        query_embedding: Some(query_embedding.clone()),
        text_query: Some(text_query.to_string()),
        search_type: SearchType::Hybrid,
        limit: 10,
        ..Default::default()
    };
    
    let hybrid_start = Instant::now();
    let hybrid_response = storage.hybrid_search(hybrid_query).await?;
    let hybrid_duration = hybrid_start.elapsed();
    
    println!("   Search method comparison:");
    println!("     Vector search:  {} results in {:?}", vector_results.len(), vector_duration);
    println!("     Text search:    {} results in {:?}", text_results.len(), text_duration);
    println!("     Hybrid search:  {} results in {:?}", hybrid_response.results.len(), hybrid_duration);
    
    // Analyze result overlap
    let vector_ids: std::collections::HashSet<_> = vector_results.iter().map(|r| r.chunk.chunk_id).collect();
    let text_ids: std::collections::HashSet<_> = text_results.iter().map(|r| r.chunk.chunk_id).collect();
    let hybrid_ids: std::collections::HashSet<_> = hybrid_response.results.iter().map(|r| r.chunk.chunk_id).collect();
    
    let vector_text_overlap = vector_ids.intersection(&text_ids).count();
    let vector_hybrid_overlap = vector_ids.intersection(&hybrid_ids).count();
    let text_hybrid_overlap = text_ids.intersection(&hybrid_ids).count();
    
    println!("\n   Result overlap analysis:");
    println!("     Vector ∩ Text:   {} results", vector_text_overlap);
    println!("     Vector ∩ Hybrid: {} results", vector_hybrid_overlap);
    println!("     Text ∩ Hybrid:   {} results", text_hybrid_overlap);
    
    // Analyze hybrid scores
    if !hybrid_response.results.is_empty() {
        println!("\n   Hybrid score analysis:");
        let mut vector_only = 0;
        let mut text_only = 0;
        let mut both = 0;
        
        for result in &hybrid_response.results {
            match (result.vector_score.is_some(), result.text_score.is_some()) {
                (true, true) => both += 1,
                (true, false) => vector_only += 1,
                (false, true) => text_only += 1,
                (false, false) => {} // Shouldn't happen
            }
        }
        
        println!("     Results with both scores: {}", both);
        println!("     Results with vector only: {}", vector_only);
        println!("     Results with text only:   {}", text_only);
    }
    
    Ok(())
}

async fn demonstrate_search_accuracy(
    storage: &VectorStorage,
    documents: &[ChunkDocument],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing search accuracy with known similar documents:");
    
    // Test self-similarity (should be highest)
    for i in [0, 10, 50, 100].iter().take_while(|&&i| i < documents.len()) {
        let query_doc = &documents[*i];
        let query_embedding = query_doc.embedding.as_ref().unwrap();
        
        let results = storage.vector_search(query_embedding, 5, None).await?;
        
        if let Some(top_result) = results.first() {
            let is_self = top_result.chunk.chunk_id == query_doc.chunk_id;
            let self_score = if is_self { top_result.score } else { 
                results.iter().find(|r| r.chunk.chunk_id == query_doc.chunk_id).map(|r| r.score).unwrap_or(0.0)
            };
            
            println!("     Doc {}: self-similarity={:.4}, top_match={}, accuracy={}",
                     i,
                     self_score,
                     if is_self { "SELF" } else { "OTHER" },
                     if is_self { "✓" } else { "✗" });
        }
    }
    
    // Test semantic clustering
    println!("\n   Testing semantic clustering by topic:");
    
    let tech_query = create_test_embedding(384, 11111);
    let tech_results = storage.vector_search(&tech_query, 10, Some(SearchFilters {
        tags: Some(vec!["technology".to_string()]),
        ..Default::default()
    })).await?;
    
    if !tech_results.is_empty() {
        let avg_tech_score = tech_results.iter().map(|r| r.score).sum::<f32>() / tech_results.len() as f32;
        println!("     Technology cluster: {} docs, avg_score={:.4}", 
                 tech_results.len(), avg_tech_score);
    }
    
    // Test score distribution quality
    let general_results = storage.vector_search(&tech_query, 50, None).await?;
    
    if general_results.len() >= 10 {
        let scores: Vec<f32> = general_results.iter().map(|r| r.score).collect();
        let score_std = calculate_std_dev(&scores);
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let min_score = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let score_range = max_score - min_score;
        
        println!("     Score distribution: std_dev={:.4}, range={:.4}", score_std, score_range);
        
        if score_std > 0.1 {
            println!("       ✓ Good score discrimination");
        } else {
            println!("       ⚠ Low score discrimination");
        }
    }
    
    Ok(())
}

async fn demonstrate_concurrent_search(
    storage: &VectorStorage,
) -> Result<(), Box<dyn std::error::Error>> {
    let concurrency_levels = vec![1, 2, 4, 8];
    
    for &concurrency in &concurrency_levels {
        println!("   Testing {} concurrent searches:", concurrency);
        
        let start = Instant::now();
        
        let tasks: Vec<_> = (0..concurrency).map(|i| {
            let storage = storage.clone();
            let query_embedding = create_test_embedding(384, 20000 + i);
            
            tokio::spawn(async move {
                let search_start = Instant::now();
                let results = storage.vector_search(&query_embedding, 10, None).await.unwrap();
                let search_duration = search_start.elapsed();
                (results.len(), search_duration)
            })
        }).collect();
        
        let results = futures::future::join_all(tasks).await;
        let total_duration = start.elapsed();
        
        let mut individual_durations = Vec::new();
        let mut total_results = 0;
        
        for result in results {
            let (result_count, duration) = result.unwrap();
            individual_durations.push(duration);
            total_results += result_count;
        }
        
        let avg_individual = individual_durations.iter().sum::<Duration>() / individual_durations.len() as u32;
        let max_individual = individual_durations.iter().max().unwrap();
        
        println!("     Total time: {:?}, avg individual: {:?}, max individual: {:?}",
                 total_duration, avg_individual, max_individual);
        println!("     Total results: {}, throughput: {:.1} searches/sec",
                 total_results,
                 concurrency as f64 / total_duration.as_secs_f64());
        
        // Check if concurrent performance is reasonable
        if *max_individual > Duration::from_millis(100) {
            println!("       ⚠ Some searches took longer than 100ms under concurrent load");
        } else {
            println!("       ✓ All searches completed within acceptable time");
        }
    }
    
    Ok(())
}

fn print_performance_summary(metrics: &storage::MetricsSnapshot) {
    println!("   System uptime: {}s", metrics.uptime_seconds);
    println!("   Total documents processed: {}", metrics.performance.documents_processed);
    println!("   Total bytes processed: {}", metrics.performance.bytes_processed);
    
    if metrics.performance.avg_throughput_dps > 0.0 {
        println!("   Average throughput: {:.1} docs/sec, {:.1} MB/sec",
                 metrics.performance.avg_throughput_dps,
                 metrics.performance.avg_throughput_bps / 1_000_000.0);
    }
    
    println!("\n   Operation performance:");
    for (op_name, op_metrics) in &metrics.operations {
        if op_metrics.count > 0 {
            println!("     {}: {} ops, avg: {:.1}ms, success: {:.1}%",
                     op_name,
                     op_metrics.count,
                     op_metrics.avg_duration_ms(),
                     op_metrics.success_rate() * 100.0);
            
            if op_metrics.recent_durations.len() > 5 {
                println!("       p95: {:.1}ms, p99: {:.1}ms",
                         op_metrics.p95_duration_ms(),
                         op_metrics.p99_duration_ms());
            }
        }
    }
    
    if metrics.errors.error_rate > 0.0 {
        println!("\n   Error summary:");
        println!("     Error rate: {:.3}%", metrics.errors.error_rate * 100.0);
        println!("     Errors per hour: {:.1}", metrics.errors.errors_per_hour);
    } else {
        println!("\n   ✓ No errors detected");
    }
}

fn create_realistic_documents(count: usize) -> Vec<ChunkDocument> {
    let domains = vec![
        ("artificial-intelligence", "Artificial intelligence and machine learning technologies are revolutionizing various industries."),
        ("database-systems", "Modern database systems provide scalable and efficient data storage solutions."),
        ("natural-language-processing", "Natural language processing enables computers to understand and generate human language."),
        ("computer-vision", "Computer vision algorithms can analyze and interpret visual information from images and videos."),
        ("distributed-systems", "Distributed systems architecture enables building scalable and resilient applications."),
        ("cybersecurity", "Cybersecurity measures protect digital assets from various threats and vulnerabilities."),
        ("cloud-computing", "Cloud computing platforms provide on-demand access to computing resources."),
        ("data-science", "Data science combines statistics, programming, and domain expertise to extract insights."),
        ("software-engineering", "Software engineering practices ensure reliable and maintainable code development."),
        ("web-development", "Web development frameworks enable creating interactive and responsive web applications."),
    ];
    
    let mut documents = Vec::new();
    let mut doc_id_counter = 0;
    
    for i in 0..count {
        let (domain, base_text) = &domains[i % domains.len()];
        
        // Create document ID for every 100 chunks
        if i % 100 == 0 {
            doc_id_counter += 1;
        }
        let document_id = Uuid::new_v4();
        
        let mut metadata = ChunkMetadata::new(
            document_id,
            format!("{} - Document {}", domain, doc_id_counter),
            i % 100,
            100,
            format!("/documents/{}/doc_{}.txt", domain, doc_id_counter),
        );
        
        metadata.tags = vec![
            "technology".to_string(),
            domain.to_string(),
            format!("batch_{}", i / 100),
        ];
        
        metadata.custom_fields.insert(
            "domain".to_string(),
            CustomFieldValue::String(domain.to_string()),
        );
        metadata.custom_fields.insert(
            "importance".to_string(),
            CustomFieldValue::Number((i % 10) as f64 / 10.0),
        );
        
        let content = format!("{} This document contains detailed information about {} and related concepts. Document ID: {}, Chunk: {}. Additional context and relevant details are provided to create realistic content length.",
                             base_text, domain, doc_id_counter, i);
        
        let embedding = create_test_embedding(384, i as u64);
        
        let chunk = ChunkDocument::new(
            Uuid::new_v4(),
            content,
            metadata,
        ).with_embedding(embedding);
        
        documents.push(chunk);
    }
    
    documents
}

fn create_test_embedding(dimension: usize, seed: u64) -> Vec<f64> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let base_hash = hasher.finish();
    
    (0..dimension).map(|i| {
        let mut hasher = DefaultHasher::new();
        (base_hash + i as u64).hash(&mut hasher);
        let val = hasher.finish() as f64 / u64::MAX as f64;
        (val - 0.5) * 2.0
    }).collect()
}

fn calculate_percentiles(scores: &[f32]) -> (f32, f32, f32) {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let len = sorted.len();
    let p25_idx = (len as f32 * 0.25) as usize;
    let p50_idx = (len as f32 * 0.50) as usize;
    let p75_idx = (len as f32 * 0.75) as usize;
    
    (
        sorted[p25_idx.min(len - 1)],
        sorted[p50_idx.min(len - 1)],
        sorted[p75_idx.min(len - 1)],
    )
}

fn calculate_std_dev(scores: &[f32]) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let variance = scores.iter()
        .map(|score| (score - mean).powi(2))
        .sum::<f32>() / scores.len() as f32;
    
    variance.sqrt()
}