//! Model comparison example
//!
//! This example demonstrates how to compare different embedding models
//! and their performance characteristics.

use embedder::{EmbeddingGenerator, EmbedderConfig, ModelType, Device};
use std::time::Duration;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    
    println!("🔍 Embedding Model Comparison Example");
    println!("=====================================\n");
    
    // Define models to compare
    let models = vec![
        ModelType::AllMiniLmL6V2,
        ModelType::BertBaseUncased,
        ModelType::SentenceT5Base,
    ];
    
    // Test texts for comparison
    let test_texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models require large amounts of training data.",
        "Computer vision systems can analyze and interpret visual information.",
    ];
    
    println!("📝 Test texts ({} samples):", test_texts.len());
    for (i, text) in test_texts.iter().enumerate() {
        println!("   {}. {}", i + 1, text);
    }
    println!();
    
    let mut model_results = Vec::new();
    
    for model_type in &models {
        println!("🔧 Testing model: {}", model_type);
        println!("   Dimension: {}", model_type.dimension());
        println!("   Max length: {}", model_type.default_max_length());
        println!("   Supports ONNX: {}", model_type.supports_onnx());
        
        let config = EmbedderConfig::new()
            .with_model_type(*model_type)
            .with_batch_size(8)
            .with_device(Device::Cpu);
        
        match EmbeddingGenerator::new(config).await {
            Ok(generator) => {
                println!("   ✅ Model loaded successfully");
                
                // Create chunks from test texts
                let chunks = test_texts.iter()
                    .map(|&text| embedder::Chunk {
                        id: uuid::Uuid::new_v4(),
                        content: text.to_string(),
                        metadata: embedder::ChunkMetadata {
                            source: "comparison_test".to_string(),
                            page: Some(1),
                            section: None,
                            created_at: chrono::Utc::now(),
                            properties: std::collections::HashMap::new(),
                        },
                        embeddings: None,
                        references: Vec::new(),
                    })
                    .collect();
                
                // Measure embedding generation time
                let start_time = std::time::Instant::now();
                let embedded_chunks = generator.generate_embeddings(chunks).await?;
                let generation_duration = start_time.elapsed();
                
                // Calculate embeddings per second
                let embeddings_per_sec = embedded_chunks.len() as f64 / generation_duration.as_secs_f64();
                
                println!("   ⏱️  Generation time: {:?}", generation_duration);
                println!("   📊 Rate: {:.1} embeddings/sec", embeddings_per_sec);
                
                // Test similarity calculations
                let similarity_start = std::time::Instant::now();
                let mut similarities = Vec::new();
                
                for i in 0..embedded_chunks.len() {
                    for j in (i + 1)..embedded_chunks.len() {
                        let sim = generator.calculate_similarity(
                            &embedded_chunks[i].embeddings,
                            &embedded_chunks[j].embeddings,
                        )?;
                        similarities.push(sim);
                    }
                }
                
                let similarity_duration = similarity_start.elapsed();
                let similarities_per_sec = similarities.len() as f64 / similarity_duration.as_secs_f64();
                
                println!("   🔗 Similarity calculations: {} pairs in {:?}", 
                        similarities.len(), similarity_duration);
                println!("   📈 Similarity rate: {:.1} calculations/sec", similarities_per_sec);
                
                // Analyze embedding characteristics
                let embedding_stats = analyze_embeddings(&embedded_chunks);
                println!("   📐 Embedding analysis:");
                println!("      Mean magnitude: {:.4}", embedding_stats.mean_magnitude);
                println!("      Std deviation: {:.4}", embedding_stats.std_deviation);
                println!("      Min value: {:.4}", embedding_stats.min_value);
                println!("      Max value: {:.4}", embedding_stats.max_value);
                
                // Test memory usage
                let memory_usage = generator.calculate_memory_usage(1000);
                println!("   💾 Memory for 1000 embeddings: {:.2} MB", 
                        memory_usage as f64 / 1_048_576.0);
                
                // Get generator stats
                let stats = generator.get_stats().await;
                println!("   📊 Generator stats:");
                println!("      Model load time: {:.2}ms", stats.model_load_time_ms);
                println!("      Average batch time: {:.2}ms", stats.avg_batch_time_ms);
                
                model_results.push(ModelResult {
                    model_type: *model_type,
                    dimension: model_type.dimension(),
                    generation_rate: embeddings_per_sec,
                    similarity_rate: similarities_per_sec,
                    model_load_time: stats.model_load_time_ms,
                    memory_usage,
                    embedding_stats,
                    similarities: similarities.clone(),
                    embeddings: embedded_chunks.iter()
                        .map(|ec| ec.embeddings.clone())
                        .collect(),
                });
                
                println!("   ✅ Model evaluation complete\n");
            }
            Err(e) => {
                println!("   ❌ Failed to load model: {}", e);
                println!("   💡 Model files may not be available\n");
            }
        }
    }
    
    if model_results.is_empty() {
        println!("⚠️  No models were successfully loaded for comparison.");
        println!("💡 Make sure you have model files in the appropriate directories:");
        println!("   - /models/all-MiniLM-L6-v2/");
        println!("   - /models/bert-base-uncased/");
        println!("   - /models/sentence-t5-base/");
        return Ok(());
    }
    
    // Generate comparison report
    println!("📊 Model Comparison Report");
    println!("==========================\n");
    
    // Performance comparison table
    println!("Performance Metrics:");
    println!("┌─────────────────────────┬──────────┬─────────────┬───────────────┬─────────────────┬─────────────┐");
    println!("│ Model                   │ Dimension│ Gen Rate    │ Similarity    │ Model Load      │ Memory      │");
    println!("│                         │          │ (emb/sec)   │ (calc/sec)    │ (ms)            │ (MB/1k)     │");
    println!("├─────────────────────────┼──────────┼─────────────┼───────────────┼─────────────────┼─────────────┤");
    
    for result in &model_results {
        println!("│ {:<23} │ {:>8} │ {:>11.1} │ {:>13.1} │ {:>15.2} │ {:>11.2} │",
                result.model_type.name(),
                result.dimension,
                result.generation_rate,
                result.similarity_rate,
                result.model_load_time,
                result.memory_usage as f64 / 1_048_576.0);
    }
    println!("└─────────────────────────┴──────────┴─────────────┴───────────────┴─────────────────┴─────────────┘\n");
    
    // Quality comparison (semantic similarity analysis)
    println!("Semantic Quality Analysis:");
    println!("─────────────────────────");
    
    if model_results.len() >= 2 {
        for i in 0..model_results.len() {
            for j in (i + 1)..model_results.len() {
                let model1 = &model_results[i];
                let model2 = &model_results[j];
                
                println!("\n{} vs {}:", model1.model_type.name(), model2.model_type.name());
                
                // Compare similarity rankings
                let ranking_correlation = calculate_ranking_correlation(&model1.similarities, &model2.similarities);
                println!("   Similarity ranking correlation: {:.3}", ranking_correlation);
                
                // Compare embedding spaces (if same dimension)
                if model1.dimension == model2.dimension {
                    let space_similarity = calculate_embedding_space_similarity(&model1.embeddings, &model2.embeddings);
                    println!("   Embedding space similarity: {:.3}", space_similarity);
                }
            }
        }
    }
    
    // Recommendations
    println!("\n🎯 Recommendations:");
    println!("──────────────────");
    
    // Find best performing model for different criteria
    let fastest_gen = model_results.iter()
        .max_by(|a, b| a.generation_rate.partial_cmp(&b.generation_rate).unwrap())
        .unwrap();
    
    let fastest_sim = model_results.iter()
        .max_by(|a, b| a.similarity_rate.partial_cmp(&b.similarity_rate).unwrap())
        .unwrap();
    
    let most_memory_efficient = model_results.iter()
        .min_by(|a, b| a.memory_usage.cmp(&b.memory_usage))
        .unwrap();
    
    let fastest_load = model_results.iter()
        .min_by(|a, b| a.model_load_time.partial_cmp(&b.model_load_time).unwrap())
        .unwrap();
    
    println!("   🏃 Fastest generation: {} ({:.1} emb/sec)", 
            fastest_gen.model_type.name(), fastest_gen.generation_rate);
    println!("   ⚡ Fastest similarity: {} ({:.1} calc/sec)", 
            fastest_sim.model_type.name(), fastest_sim.similarity_rate);
    println!("   💾 Most memory efficient: {} ({:.2} MB/1k)", 
            most_memory_efficient.model_type.name(), 
            most_memory_efficient.memory_usage as f64 / 1_048_576.0);
    println!("   🚀 Fastest load: {} ({:.2}ms)", 
            fastest_load.model_type.name(), fastest_load.model_load_time);
    
    // Usage recommendations
    println!("\n   Use cases:");
    for result in &model_results {
        match result.model_type {
            ModelType::AllMiniLmL6V2 => {
                println!("   • {}: Best balance of speed and quality for general use", 
                        result.model_type.name());
            }
            ModelType::BertBaseUncased => {
                println!("   • {}: Higher quality embeddings, good for accuracy-critical tasks", 
                        result.model_type.name());
            }
            ModelType::SentenceT5Base => {
                println!("   • {}: Advanced semantic understanding, best for complex queries", 
                        result.model_type.name());
            }
            _ => {
                println!("   • {}: Custom model with {} dimensions", 
                        result.model_type.name(), result.dimension);
            }
        }
    }
    
    // Performance targets analysis
    println!("\n🎯 Performance Target Analysis:");
    println!("───────────────────────────────");
    
    let target_rate = 1000.0; // 1000 embeddings/sec
    println!("   Target: {:.0} embeddings/sec", target_rate);
    
    for result in &model_results {
        let percentage = (result.generation_rate / target_rate) * 100.0;
        let status = if result.generation_rate >= target_rate { "✅" } else { "⚠️" };
        
        println!("   {} {}: {:.1} emb/sec ({:.1}% of target)", 
                status, result.model_type.name(), result.generation_rate, percentage);
    }
    
    println!("\n✨ Model comparison completed!");
    
    Ok(())
}

// Supporting structures and functions

#[derive(Debug)]
struct ModelResult {
    model_type: ModelType,
    dimension: usize,
    generation_rate: f64,
    similarity_rate: f64,
    model_load_time: f64,
    memory_usage: usize,
    embedding_stats: EmbeddingStats,
    similarities: Vec<f32>,
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug)]
struct EmbeddingStats {
    mean_magnitude: f32,
    std_deviation: f32,
    min_value: f32,
    max_value: f32,
}

fn analyze_embeddings(embedded_chunks: &[embedder::EmbeddedChunk]) -> EmbeddingStats {
    let all_values: Vec<f32> = embedded_chunks.iter()
        .flat_map(|ec| ec.embeddings.iter())
        .copied()
        .collect();
    
    if all_values.is_empty() {
        return EmbeddingStats {
            mean_magnitude: 0.0,
            std_deviation: 0.0,
            min_value: 0.0,
            max_value: 0.0,
        };
    }
    
    let min_value = all_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_value = all_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Calculate magnitudes
    let magnitudes: Vec<f32> = embedded_chunks.iter()
        .map(|ec| {
            let sum_of_squares: f32 = ec.embeddings.iter().map(|&x| x * x).sum();
            sum_of_squares.sqrt()
        })
        .collect();
    
    let mean_magnitude = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
    
    let variance = magnitudes.iter()
        .map(|&mag| {
            let diff = mag - mean_magnitude;
            diff * diff
        })
        .sum::<f32>() / magnitudes.len() as f32;
    
    let std_deviation = variance.sqrt();
    
    EmbeddingStats {
        mean_magnitude,
        std_deviation,
        min_value,
        max_value,
    }
}

fn calculate_ranking_correlation(sims1: &[f32], sims2: &[f32]) -> f32 {
    if sims1.len() != sims2.len() || sims1.is_empty() {
        return 0.0;
    }
    
    // Simple Pearson correlation coefficient
    let n = sims1.len() as f32;
    let mean1 = sims1.iter().sum::<f32>() / n;
    let mean2 = sims2.iter().sum::<f32>() / n;
    
    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;
    
    for i in 0..sims1.len() {
        let diff1 = sims1[i] - mean1;
        let diff2 = sims2[i] - mean2;
        
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }
    
    let denominator = (sum_sq1 * sum_sq2).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn calculate_embedding_space_similarity(embs1: &[Vec<f32>], embs2: &[Vec<f32>]) -> f32 {
    if embs1.len() != embs2.len() || embs1.is_empty() {
        return 0.0;
    }
    
    // Calculate average cosine similarity between corresponding embeddings
    let mut total_similarity = 0.0;
    let mut valid_pairs = 0;
    
    for (e1, e2) in embs1.iter().zip(embs2.iter()) {
        if let Ok(sim) = embedder::similarity::cosine_similarity(e1, e2) {
            total_similarity += sim;
            valid_pairs += 1;
        }
    }
    
    if valid_pairs > 0 {
        total_similarity / valid_pairs as f32
    } else {
        0.0
    }
}