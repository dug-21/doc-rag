//! Batch processing example for the embedding generator
//!
//! This example demonstrates advanced batch processing capabilities,
//! including adaptive batching and performance monitoring.

use embedder::{
    EmbeddingGenerator, EmbedderConfig, ModelType, Device, Chunk, ChunkMetadata,
    BatchProcessor, AdaptiveBatchConfig,
};
use std::collections::HashMap;
use uuid::Uuid;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    println!("🔄 Embedding Generator Batch Processing Example");
    println!("===============================================\n");
    
    // Create high-performance configuration
    let config = EmbedderConfig::new()
        .with_model_type(ModelType::AllMiniLmL6V2)
        .with_batch_size(32)
        .high_performance()
        .with_device(Device::Cpu);
    
    println!("⚙️ Configuration:");
    println!("   Model: {}", config.model_type);
    println!("   Batch size: {}", config.batch_size);
    println!("   High performance mode: enabled");
    println!("   FP16: {}", config.optimization.use_fp16);
    println!("   Memory optimization: {}", config.optimization.memory_optimization);
    
    // Initialize generator
    println!("\n🚀 Initializing embedding generator...");
    let generator = match EmbeddingGenerator::new(config).await {
        Ok(gen) => {
            println!("   ✅ Generator initialized successfully!");
            gen
        }
        Err(e) => {
            println!("   ❌ Failed to initialize: {}", e);
            println!("   💡 This example requires model files. Set SKIP_MODEL_TESTS=1 to skip.");
            return Ok(());
        }
    };
    
    // Generate a large number of test chunks
    println!("\n📝 Generating test data...");
    let chunk_count = 1000;
    let chunks = generate_test_chunks(chunk_count);
    println!("   Created {} test chunks", chunks.len());
    
    // Demonstrate different batch processing strategies
    println!("\n🎯 Testing Different Batch Processing Strategies");
    println!("================================================");
    
    // 1. Standard batch processing
    println!("\n1️⃣ Standard Batch Processing:");
    let standard_start = std::time::Instant::now();
    let standard_results = generator.generate_embeddings(chunks.clone()).await?;
    let standard_duration = standard_start.elapsed();
    
    println!("   ✅ Processed {} embeddings in {:?}", standard_results.len(), standard_duration);
    println!("   📊 Rate: {:.1} embeddings/sec", 
             standard_results.len() as f64 / standard_duration.as_secs_f64());
    
    // 2. Manual batch processor demonstration
    println!("\n2️⃣ Manual Batch Processing:");
    let processor = BatchProcessor::new(16); // Smaller batches
    let text_chunks: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    
    let batches = processor.create_batches(&text_chunks);
    println!("   Created {} batches from {} texts", batches.len(), text_chunks.len());
    
    let manual_start = std::time::Instant::now();
    let mut all_embeddings = Vec::new();
    
    for (i, batch) in batches.iter().enumerate() {
        println!("   Processing batch {}/{} ({} texts)...", i + 1, batches.len(), batch.len());
        
        let batch_chunks: Vec<Chunk> = batch.iter()
            .map(|text| create_chunk_from_text(text))
            .collect();
        
        let batch_embeddings = generator.generate_embeddings(batch_chunks).await?;
        all_embeddings.extend(batch_embeddings);
        
        processor.mark_batch_completed(batch.len());
        
        // Show progress
        let progress = processor.get_progress();
        println!("     Progress: {:.1}% ({}/{} items)", 
                progress.completion_percentage, 
                progress.processed_items, 
                progress.total_items);
    }
    
    let manual_duration = manual_start.elapsed();
    println!("   ✅ Manual batching completed in {:?}", manual_duration);
    
    // 3. Adaptive batch processing
    println!("\n3️⃣ Adaptive Batch Processing:");
    let adaptive_config = AdaptiveBatchConfig {
        min_batch_size: 4,
        max_batch_size: 64,
        target_memory_mb: 256,
        avg_text_length_chars: estimate_avg_text_length(&text_chunks),
        embedding_dimension: generator.get_dimension(),
    };
    
    println!("   Adaptive config:");
    println!("     Min batch size: {}", adaptive_config.min_batch_size);
    println!("     Max batch size: {}", adaptive_config.max_batch_size);
    println!("     Target memory: {}MB", adaptive_config.target_memory_mb);
    println!("     Avg text length: {} chars", adaptive_config.avg_text_length_chars);
    
    let adaptive_processor = BatchProcessor::new(32);
    let adaptive_batches = adaptive_processor.create_adaptive_batches(&text_chunks, &adaptive_config);
    
    println!("   Created {} adaptive batches", adaptive_batches.len());
    
    let adaptive_start = std::time::Instant::now();
    let mut adaptive_embeddings = Vec::new();
    
    for (i, batch) in adaptive_batches.iter().enumerate() {
        let batch_chunks: Vec<Chunk> = batch.iter()
            .map(|text| create_chunk_from_text(text))
            .collect();
        
        let batch_embeddings = generator.generate_embeddings(batch_chunks).await?;
        adaptive_embeddings.extend(batch_embeddings);
        
        if i % 10 == 0 || i == adaptive_batches.len() - 1 {
            println!("     Processed {}/{} batches", i + 1, adaptive_batches.len());
        }
    }
    
    let adaptive_duration = adaptive_start.elapsed();
    println!("   ✅ Adaptive batching completed in {:?}", adaptive_duration);
    
    // 4. Memory-constrained processing
    println!("\n4️⃣ Memory-Constrained Processing:");
    let memory_processor = BatchProcessor::with_memory_limit(64, 128); // 128MB limit
    let memory_batches = memory_processor.create_batches(&text_chunks);
    
    println!("   Memory-constrained batches: {}", memory_batches.len());
    println!("   Average batch size: {:.1}", 
             text_chunks.len() as f64 / memory_batches.len() as f64);
    
    let memory_start = std::time::Instant::now();
    let mut memory_embeddings = Vec::new();
    
    for batch in memory_batches.iter() {
        let batch_chunks: Vec<Chunk> = batch.iter()
            .map(|text| create_chunk_from_text(text))
            .collect();
        
        let batch_embeddings = generator.generate_embeddings(batch_chunks).await?;
        memory_embeddings.extend(batch_embeddings);
    }
    
    let memory_duration = memory_start.elapsed();
    println!("   ✅ Memory-constrained processing completed in {:?}", memory_duration);
    
    // Performance comparison
    println!("\n📊 Performance Comparison");
    println!("========================");
    
    let strategies = [
        ("Standard", standard_duration, standard_results.len()),
        ("Manual", manual_duration, all_embeddings.len()),
        ("Adaptive", adaptive_duration, adaptive_embeddings.len()),
        ("Memory-constrained", memory_duration, memory_embeddings.len()),
    ];
    
    println!("   Strategy              | Duration    | Rate (emb/sec) | Throughput");
    println!("   ---------------------|-------------|----------------|------------");
    
    for (name, duration, count) in &strategies {
        let rate = *count as f64 / duration.as_secs_f64();
        let throughput_pct = (rate / (standard_results.len() as f64 / standard_duration.as_secs_f64())) * 100.0;
        println!("   {:<20} | {:>9.3}s | {:>12.1} | {:>9.1}%", 
                name, duration.as_secs_f64(), rate, throughput_pct);
    }
    
    // Memory usage analysis
    println!("\n💾 Memory Usage Analysis");
    println!("=======================");
    
    let embedding_memory = generator.calculate_memory_usage(chunk_count);
    println!("   Embedding storage: {:.2} MB", embedding_memory as f64 / 1_048_576.0);
    
    let cache_stats = generator.get_cache_stats().await;
    println!("   Cache memory: {:.2} MB", cache_stats.memory_usage_bytes as f64 / 1_048_576.0);
    
    // Batch size optimization analysis
    println!("\n⚡ Batch Size Optimization");
    println!("=========================");
    
    let batch_sizes = [1, 4, 8, 16, 32, 64, 128];
    println!("   Testing different batch sizes with 100 chunks...");
    
    let test_chunks = chunks.iter().take(100).cloned().collect::<Vec<_>>();
    
    for &batch_size in &batch_sizes {
        let opt_config = EmbedderConfig::new()
            .with_model_type(ModelType::AllMiniLmL6V2)
            .with_batch_size(batch_size)
            .with_device(Device::Cpu);
        
        // Create a new generator for each batch size test
        if let Ok(test_generator) = EmbeddingGenerator::new(opt_config).await {
            let start = std::time::Instant::now();
            let results = test_generator.generate_embeddings(test_chunks.clone()).await?;
            let duration = start.elapsed();
            
            let rate = results.len() as f64 / duration.as_secs_f64();
            println!("     Batch size {:<3}: {:>8.1} emb/sec ({:>6.3}s)", 
                    batch_size, rate, duration.as_secs_f64());
        }
    }
    
    // Concurrent processing demonstration
    println!("\n🚀 Concurrent Processing");
    println!("========================");
    
    let concurrent_chunks: Vec<Vec<Chunk>> = chunks
        .chunks(chunks.len() / 4)
        .map(|chunk_slice| chunk_slice.to_vec())
        .collect();
    
    println!("   Processing {} chunks across {} concurrent tasks...", 
             chunks.len(), concurrent_chunks.len());
    
    let concurrent_start = std::time::Instant::now();
    
    let generator_arc = std::sync::Arc::new(generator);
    let mut handles = Vec::new();
    
    for (i, chunk_batch) in concurrent_chunks.into_iter().enumerate() {
        let gen = generator_arc.clone();
        let handle = tokio::spawn(async move {
            println!("     Task {} processing {} chunks...", i + 1, chunk_batch.len());
            let results = gen.generate_embeddings(chunk_batch).await;
            (i, results)
        });
        handles.push(handle);
    }
    
    let concurrent_results = futures::future::try_join_all(handles).await?;
    let concurrent_duration = concurrent_start.elapsed();
    
    let total_concurrent_embeddings: usize = concurrent_results.iter()
        .map(|(_, result)| result.as_ref().unwrap().len())
        .sum();
    
    println!("   ✅ Concurrent processing completed:");
    println!("     Total embeddings: {}", total_concurrent_embeddings);
    println!("     Duration: {:?}", concurrent_duration);
    println!("     Rate: {:.1} emb/sec", 
             total_concurrent_embeddings as f64 / concurrent_duration.as_secs_f64());
    
    // Final statistics
    println!("\n📈 Final Statistics");
    println!("==================");
    
    let final_stats = generator_arc.get_stats().await;
    println!("   Total embeddings processed: {}", final_stats.total_embeddings);
    println!("   Total batches: {}", final_stats.total_batches);
    println!("   Cache hit rate: {:.1}%", 
             final_stats.cache_hits as f64 / (final_stats.cache_hits + final_stats.cache_misses) as f64 * 100.0);
    println!("   Average batch time: {:.2}ms", final_stats.avg_batch_time_ms);
    
    // Performance target check
    let target_rate = 1000.0; // 1000 chunks/sec target
    let best_rate = standard_results.len() as f64 / standard_duration.as_secs_f64();
    
    println!("\n🎯 Performance Target Analysis");
    println!("==============================");
    println!("   Target rate: {:.0} embeddings/sec", target_rate);
    println!("   Best achieved rate: {:.1} embeddings/sec", best_rate);
    
    if best_rate >= target_rate {
        println!("   ✅ Target achieved! ({:.1}x target)", best_rate / target_rate);
    } else {
        println!("   ⚠️  Target not met ({:.1}% of target)", (best_rate / target_rate) * 100.0);
        println!("   💡 Consider optimizations:");
        println!("      - Increase batch size");
        println!("      - Use GPU acceleration");
        println!("      - Enable FP16 precision");
        println!("      - Optimize model format (ONNX)");
    }
    
    println!("\n✨ Batch processing example completed!");
    
    Ok(())
}

// Helper functions

fn generate_test_chunks(count: usize) -> Vec<Chunk> {
    let sample_texts = vec![
        "The quick brown fox jumps over the lazy dog in the forest.",
        "Machine learning algorithms can analyze vast amounts of data quickly.",
        "Natural language processing enables computers to understand human speech.",
        "Deep learning neural networks require significant computational resources.",
        "Artificial intelligence is transforming industries across the globe.",
        "Data science combines statistics, programming, and domain expertise.",
        "Computer vision systems can recognize and classify objects in images.",
        "Robotics engineering integrates mechanical, electrical, and software systems.",
        "Cloud computing provides scalable infrastructure for modern applications.",
        "Cybersecurity protects digital systems from malicious attacks and threats.",
    ];
    
    (0..count)
        .map(|i| {
            let text = &sample_texts[i % sample_texts.len()];
            let extended_text = format!("{} Document {} contains additional context and information for testing purposes.", text, i + 1);
            
            Chunk {
                id: Uuid::new_v4(),
                content: extended_text,
                metadata: ChunkMetadata {
                    source: format!("batch_test_doc_{}", i / 50 + 1),
                    page: Some((i / 50 + 1) as u32),
                    section: Some(format!("Section {}", i % 10 + 1)),
                    created_at: chrono::Utc::now(),
                    properties: HashMap::from([
                        ("batch_id".to_string(), format!("batch_{}", i / 100)),
                        ("test_type".to_string(), "performance".to_string()),
                    ]),
                },
                embeddings: None,
                references: Vec::new(),
            }
        })
        .collect()
}

fn create_chunk_from_text(text: &str) -> Chunk {
    Chunk {
        id: Uuid::new_v4(),
        content: text.to_string(),
        metadata: ChunkMetadata {
            source: "batch_processing_example".to_string(),
            page: Some(1),
            section: Some("generated".to_string()),
            created_at: chrono::Utc::now(),
            properties: HashMap::new(),
        },
        embeddings: None,
        references: Vec::new(),
    }
}

fn estimate_avg_text_length(texts: &[String]) -> usize {
    if texts.is_empty() {
        return 500; // Default estimate
    }
    
    texts.iter().map(|t| t.len()).sum::<usize>() / texts.len()
}