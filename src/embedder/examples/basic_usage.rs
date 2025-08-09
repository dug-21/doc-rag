//! Basic usage example for the embedding generator
//!
//! This example demonstrates how to use the embedding generator for
//! basic text embedding tasks.

use embedder::{EmbeddingGenerator, EmbedderConfig, ModelType, Device, Chunk, ChunkMetadata};
use std::collections::HashMap;
use uuid::Uuid;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();
    
    println!("🚀 Embedding Generator Basic Usage Example");
    println!("==========================================\n");
    
    // Create configuration
    let config = EmbedderConfig::new()
        .with_model_type(ModelType::AllMiniLmL6V2)
        .with_batch_size(8)
        .with_device(Device::Cpu)
        .with_normalize(true)
        .with_cache_size(1000);
    
    println!("📋 Configuration:");
    println!("  Model: {}", config.model_type);
    println!("  Batch size: {}", config.batch_size);
    println!("  Device: {}", config.device);
    println!("  Normalize: {}", config.normalize);
    println!("  Cache size: {}\n", config.cache_size);
    
    // Initialize embedding generator
    println!("🔄 Initializing embedding generator...");
    let generator = match EmbeddingGenerator::new(config).await {
        Ok(gen) => {
            println!("✅ Generator initialized successfully!");
            println!("   Embedding dimension: {}", gen.get_dimension());
            gen
        }
        Err(e) => {
            println!("❌ Failed to initialize generator: {}", e);
            println!("💡 Make sure you have the required model files in /models/");
            println!("   You can download all-MiniLM-L6-v2 from Hugging Face:");
            println!("   https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2");
            return Ok(());
        }
    };
    
    // Create sample text chunks
    println!("\n📝 Creating sample text chunks...");
    let sample_texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a powerful tool for data analysis.",
        "Natural language processing enables computers to understand human language.",
        "Embeddings capture semantic meaning in high-dimensional vectors.",
        "Rust is a systems programming language focused on safety and performance.",
    ];
    
    let chunks: Vec<Chunk> = sample_texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| Chunk {
            id: Uuid::new_v4(),
            content: text.to_string(),
            metadata: ChunkMetadata {
                source: format!("example_doc_{}", i / 2 + 1),
                page: Some((i / 2 + 1) as u32),
                section: Some(format!("Section {}", i + 1)),
                created_at: chrono::Utc::now(),
                properties: HashMap::from([
                    ("category".to_string(), "example".to_string()),
                    ("language".to_string(), "en".to_string()),
                ]),
            },
            embeddings: None,
            references: Vec::new(),
        })
        .collect();
    
    println!("   Created {} chunks", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("   {}. \"{}\"", i + 1, chunk.content);
    }
    
    // Generate embeddings
    println!("\n🔍 Generating embeddings...");
    let start_time = std::time::Instant::now();
    
    let embedded_chunks = generator.generate_embeddings(chunks).await?;
    
    let duration = start_time.elapsed();
    println!("✅ Generated {} embeddings in {:?}", embedded_chunks.len(), duration);
    println!("   Average time per embedding: {:.2}ms", 
             duration.as_millis() as f64 / embedded_chunks.len() as f64);
    
    // Display embedding information
    println!("\n📊 Embedding Information:");
    for (i, embedded_chunk) in embedded_chunks.iter().enumerate() {
        println!("   Chunk {}: {} dimensions", i + 1, embedded_chunk.embeddings.len());
        
        // Show first few dimensions
        let preview: Vec<String> = embedded_chunk.embeddings.iter()
            .take(5)
            .map(|f| format!("{:.4}", f))
            .collect();
        println!("     Preview: [{}...]", preview.join(", "));
        
        // Validate embedding
        match generator.validate_embedding(&embedded_chunk.embeddings) {
            Ok(()) => println!("     ✅ Valid embedding"),
            Err(e) => println!("     ❌ Invalid embedding: {}", e),
        }
    }
    
    // Calculate similarities between chunks
    println!("\n🔗 Calculating similarities between chunks:");
    for i in 0..embedded_chunks.len() {
        for j in (i + 1)..embedded_chunks.len() {
            let similarity = generator.calculate_similarity(
                &embedded_chunks[i].embeddings,
                &embedded_chunks[j].embeddings,
            )?;
            
            let chunk_i_preview = if embedded_chunks[i].chunk.content.len() > 30 {
                format!("{}...", &embedded_chunks[i].chunk.content[..30])
            } else {
                embedded_chunks[i].chunk.content.clone()
            };
            
            let chunk_j_preview = if embedded_chunks[j].chunk.content.len() > 30 {
                format!("{}...", &embedded_chunks[j].chunk.content[..30])
            } else {
                embedded_chunks[j].chunk.content.clone()
            };
            
            println!("   Chunk {} ↔ Chunk {}: {:.4}", i + 1, j + 1, similarity);
            println!("     \"{}\"", chunk_i_preview);
            println!("     \"{}\"", chunk_j_preview);
        }
    }
    
    // Find most similar pair
    let mut max_similarity = -2.0;
    let mut most_similar_pair = (0, 0);
    
    for i in 0..embedded_chunks.len() {
        for j in (i + 1)..embedded_chunks.len() {
            let similarity = generator.calculate_similarity(
                &embedded_chunks[i].embeddings,
                &embedded_chunks[j].embeddings,
            )?;
            
            if similarity > max_similarity {
                max_similarity = similarity;
                most_similar_pair = (i, j);
            }
        }
    }
    
    println!("\n🏆 Most similar pair (similarity: {:.4}):", max_similarity);
    println!("   \"{}\"", embedded_chunks[most_similar_pair.0].chunk.content);
    println!("   \"{}\"", embedded_chunks[most_similar_pair.1].chunk.content);
    
    // Show generator statistics
    println!("\n📈 Generator Statistics:");
    let stats = generator.get_stats().await;
    println!("   Total embeddings generated: {}", stats.total_embeddings);
    println!("   Total batches processed: {}", stats.total_batches);
    println!("   Cache hits: {}", stats.cache_hits);
    println!("   Cache misses: {}", stats.cache_misses);
    println!("   Average batch time: {:.2}ms", stats.avg_batch_time_ms);
    println!("   Total processing time: {:.2}ms", stats.total_time_ms);
    
    // Show cache statistics
    println!("\n💾 Cache Statistics:");
    let cache_stats = generator.get_cache_stats().await;
    println!("   Cached entries: {}", cache_stats.total_entries);
    println!("   Cache capacity: {}", cache_stats.max_capacity);
    println!("   Memory usage: {} bytes", cache_stats.memory_usage_bytes);
    println!("   Hit rate: {:.1}%", cache_stats.hit_rate * 100.0);
    
    // Demonstrate cache effectiveness by processing same texts again
    println!("\n🔄 Demonstrating cache effectiveness...");
    println!("   Processing same chunks again (should hit cache):");
    
    let start_time = std::time::Instant::now();
    let cached_chunks = generator.generate_embeddings(
        embedded_chunks.iter().map(|ec| ec.chunk.clone()).collect()
    ).await?;
    let cached_duration = start_time.elapsed();
    
    println!("   ✅ Processed {} chunks in {:?} (cached)", cached_chunks.len(), cached_duration);
    println!("   Speedup: {:.1}x faster due to caching", 
             duration.as_millis() as f64 / cached_duration.as_millis() as f64);
    
    // Final statistics
    let final_stats = generator.get_stats().await;
    println!("\n📊 Final Statistics:");
    println!("   Cache hits: {} (+{})", final_stats.cache_hits, final_stats.cache_hits - stats.cache_hits);
    println!("   Cache hit rate: {:.1}%", 
             final_stats.cache_hits as f64 / (final_stats.cache_hits + final_stats.cache_misses) as f64 * 100.0);
    
    println!("\n✨ Example completed successfully!");
    
    Ok(())
}