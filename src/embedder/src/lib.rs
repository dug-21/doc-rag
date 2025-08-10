//! High-performance embedding generator for RAG system
//! 
//! This crate provides a complete embedding generation system with:
//! - Multiple model backends (ONNX via ORT, Candle native)
//! - Batch processing with configurable batch sizes
//! - Memory-efficient tensor operations
//! - Cosine similarity calculations
//! - Model caching and management
//! - Performance benchmarking
//!
//! # Example
//! ```no_run
//! use embedder::{EmbeddingGenerator, EmbedderConfig, ModelType};
//! use tokio;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = EmbedderConfig {
//!         model_type: ModelType::AllMiniLmL6V2,
//!         batch_size: 32,
//!         max_length: 512,
//!         device: embedder::Device::Cpu,
//!         normalize: true,
//!     };
//!     
//!     let generator = EmbeddingGenerator::new(config).await?;
//!     
//!     let texts = vec!["Hello world", "How are you?"];
//!     let embeddings = generator.generate_embeddings(&texts).await?;
//!     
//!     println!("Generated {} embeddings", embeddings.len());
//!     Ok(())
//! }
//! ```

pub mod models;
pub mod config;
pub mod error;
pub mod similarity;
pub mod batch;
pub mod cache;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, instrument};
use anyhow::Result;
use uuid::Uuid;

// Re-export key types
pub use config::*;
pub use error::*;
pub use models::*;
pub use similarity::*;
pub use batch::*;
pub use cache::*;

/// Represents a text chunk with optional metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Chunk {
    pub id: Uuid,
    pub content: String,
    pub metadata: ChunkMetadata,
    pub embeddings: Option<Vec<f32>>,
    pub references: Vec<ChunkReference>,
}

/// Metadata associated with a chunk
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkMetadata {
    pub source: String,
    pub page: Option<u32>,
    pub section: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub properties: HashMap<String, String>,
}

/// Reference to another chunk
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChunkReference {
    pub chunk_id: Uuid,
    pub reference_type: String,
    pub confidence: f32,
}

/// A chunk with generated embeddings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddedChunk {
    pub chunk: Chunk,
    pub embeddings: Vec<f32>,
    pub model_version: String,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Core embedding generator with multi-model support and caching
pub struct EmbeddingGenerator {
    config: EmbedderConfig,
    model_manager: Arc<RwLock<ModelManager>>,
    cache: Arc<EmbeddingCache>,
    batch_processor: BatchProcessor,
    stats: Arc<RwLock<GeneratorStats>>,
}

/// Statistics for the embedding generator
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeneratorStats {
    pub total_embeddings: u64,
    pub total_batches: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_batch_time_ms: f64,
    pub total_time_ms: f64,
    pub model_load_time_ms: f64,
}

impl EmbeddingGenerator {
    /// Create a new embedding generator with the specified configuration
    #[instrument(skip(config))]
    pub async fn new(config: EmbedderConfig) -> Result<Self> {
        info!("Initializing embedding generator with config: {:?}", config);
        
        let model_manager = Arc::new(RwLock::new(ModelManager::new()));
        let cache = Arc::new(EmbeddingCache::new(config.cache_size));
        let batch_processor = BatchProcessor::new(config.batch_size);
        let stats = Arc::new(RwLock::new(GeneratorStats::default()));
        
        let generator = Self {
            config: config.clone(),
            model_manager,
            cache,
            batch_processor,
            stats,
        };
        
        // Pre-load the primary model
        let start_time = std::time::Instant::now();
        generator.ensure_model_loaded().await?;
        
        {
            let mut stats = generator.stats.write().await;
            stats.model_load_time_ms = start_time.elapsed().as_millis() as f64;
        }
        
        info!("Embedding generator initialized successfully");
        Ok(generator)
    }
    
    /// Generate embeddings for a batch of text chunks
    #[instrument(skip(self, chunks), fields(chunk_count = chunks.len()))]
    pub async fn generate_embeddings(&self, chunks: Vec<Chunk>) -> Result<Vec<EmbeddedChunk>> {
        let start_time = std::time::Instant::now();
        
        info!("Generating embeddings for {} chunks", chunks.len());
        
        // Extract texts and check cache
        let mut results = Vec::with_capacity(chunks.len());
        let mut texts_to_process = Vec::new();
        let mut indices_to_process = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            if let Some(embeddings) = self.cache.get(&chunk.content).await {
                debug!("Cache hit for chunk {}", chunk.id);
                let mut stats = self.stats.write().await;
                stats.cache_hits += 1;
                
                results.push(Some(EmbeddedChunk {
                    chunk: chunk.clone(),
                    embeddings,
                    model_version: self.config.model_type.to_string(),
                    generated_at: chrono::Utc::now(),
                }));
            } else {
                debug!("Cache miss for chunk {}", chunk.id);
                let mut stats = self.stats.write().await;
                stats.cache_misses += 1;
                
                texts_to_process.push(chunk.content.clone());
                indices_to_process.push(i);
                results.push(None);
            }
        }
        
        // Process uncached texts in batches
        if !texts_to_process.is_empty() {
            let embeddings = self.generate_embeddings_batch(&texts_to_process).await?;
            
            // Store in cache and update results
            for (embedding, &result_idx) in embeddings.iter().zip(indices_to_process.iter()) {
                let chunk = &chunks[result_idx];
                
                // Cache the embedding
                self.cache.put(chunk.content.clone(), embedding.clone()).await;
                
                results[result_idx] = Some(EmbeddedChunk {
                    chunk: chunk.clone(),
                    embeddings: embedding.clone(),
                    model_version: self.config.model_type.to_string(),
                    generated_at: chrono::Utc::now(),
                });
            }
        }
        
        let final_results: Vec<EmbeddedChunk> = results.into_iter()
            .map(|opt| opt.unwrap())
            .collect();
        
        // Update statistics
        let elapsed = start_time.elapsed().as_millis() as f64;
        let mut stats = self.stats.write().await;
        stats.total_embeddings += final_results.len() as u64;
        stats.total_batches += 1;
        stats.total_time_ms += elapsed;
        stats.avg_batch_time_ms = stats.total_time_ms / stats.total_batches as f64;
        
        info!("Generated {} embeddings in {:.2}ms", final_results.len(), elapsed);
        
        Ok(final_results)
    }
    
    /// Generate embeddings for a batch of texts (internal method)
    #[instrument(skip(self, texts), fields(text_count = texts.len()))]
    async fn generate_embeddings_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.ensure_model_loaded().await?;
        
        let model_manager = self.model_manager.read().await;
        let model = model_manager.get_model(&self.config.model_type)?;
        
        // Process in configurable batch sizes
        let batches = self.batch_processor.create_batches(texts);
        let mut all_embeddings = Vec::new();
        
        for batch in batches {
            debug!("Processing batch of {} texts", batch.len());
            let batch_embeddings = model.encode_batch(&batch).await?;
            
            for embedding in batch_embeddings {
                let processed = if self.config.normalize {
                    self.normalize_embedding(&embedding)?
                } else {
                    embedding
                };
                all_embeddings.push(processed);
            }
        }
        
        Ok(all_embeddings)
    }
    
    /// Calculate cosine similarity between two embeddings
    pub fn calculate_similarity(&self, emb1: &[f32], emb2: &[f32]) -> Result<f32> {
        similarity::cosine_similarity(emb1, emb2)
    }
    
    /// Calculate cosine similarities between one embedding and a batch
    pub fn calculate_similarities(&self, query_embedding: &[f32], embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        similarity::batch_cosine_similarity(query_embedding, embeddings)
    }
    
    /// Normalize an embedding vector to unit length
    fn normalize_embedding(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        similarity::normalize_l2(embedding)
    }
    
    /// Ensure the primary model is loaded
    async fn ensure_model_loaded(&self) -> Result<()> {
        let model_manager = self.model_manager.read().await;
        if !model_manager.is_model_loaded(&self.config.model_type) {
            drop(model_manager);
            let mut model_manager = self.model_manager.write().await;
            model_manager.load_model(&self.config.model_type, &self.config).await?;
        }
        Ok(())
    }
    
    /// Get generator statistics
    pub async fn get_stats(&self) -> GeneratorStats {
        self.stats.read().await.clone()
    }
    
    /// Clear the embedding cache
    pub async fn clear_cache(&self) {
        self.cache.clear().await;
    }
    
    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> CacheStats {
        self.cache.get_stats().await
    }
    
    /// Preload multiple models for faster switching
    pub async fn preload_models(&self, model_types: &[ModelType]) -> Result<()> {
        let mut model_manager = self.model_manager.write().await;
        for model_type in model_types {
            if !model_manager.is_model_loaded(model_type) {
                info!("Preloading model: {:?}", model_type);
                model_manager.load_model(model_type, &self.config).await?;
            }
        }
        Ok(())
    }
    
    /// Switch to a different model
    pub async fn switch_model(&mut self, model_type: ModelType) -> Result<()> {
        info!("Switching to model: {:?}", model_type);
        self.config.model_type = model_type;
        self.ensure_model_loaded().await?;
        Ok(())
    }
    
    /// Get embedding dimension for the current model
    pub fn get_dimension(&self) -> usize {
        self.config.model_type.dimension()
    }
    
    /// Validate embedding dimensions
    pub fn validate_embedding(&self, embedding: &[f32]) -> Result<()> {
        let expected_dim = self.get_dimension();
        if embedding.len() != expected_dim {
            return Err(EmbedderError::DimensionMismatch {
                expected: expected_dim,
                actual: embedding.len(),
            }.into());
        }
        
        // Check for NaN or infinity values
        for (i, &value) in embedding.iter().enumerate() {
            if !value.is_finite() {
                return Err(EmbedderError::InvalidEmbedding {
                    message: format!("Non-finite value at index {}: {}", i, value),
                }.into());
            }
        }
        
        Ok(())
    }
    
    /// Calculate the memory usage of embeddings
    pub fn calculate_memory_usage(&self, num_embeddings: usize) -> usize {
        num_embeddings * self.get_dimension() * std::mem::size_of::<f32>()
    }
    
    /// Estimate processing time for a given number of texts
    pub async fn estimate_processing_time(&self, num_texts: usize) -> f64 {
        let stats = self.stats.read().await;
        if stats.total_embeddings == 0 {
            // Fallback estimate: 1ms per text
            return num_texts as f64;
        }
        
        let avg_time_per_embedding = stats.total_time_ms / stats.total_embeddings as f64;
        num_texts as f64 * avg_time_per_embedding
    }
}

impl Default for Chunk {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            content: String::new(),
            metadata: ChunkMetadata {
                source: String::new(),
                page: None,
                section: None,
                created_at: chrono::Utc::now(),
                properties: HashMap::new(),
            },
            embeddings: None,
            references: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    
    #[tokio::test]
    async fn test_embedding_generator_creation() {
        let config = EmbedderConfig::default();
        let result = EmbeddingGenerator::new(config).await;
        assert!(result.is_ok(), "Failed to create embedding generator: {:?}", result.err());
    }
    
    #[tokio::test]
    async fn test_chunk_default() {
        let chunk = Chunk::default();
        assert!(!chunk.content.is_empty() || chunk.content.is_empty()); // Just test it doesn't panic
        assert!(chunk.embeddings.is_none());
        assert!(chunk.references.is_empty());
    }
    
    #[tokio::test]
    async fn test_dimension_validation() {
        let config = EmbedderConfig::default();
        let generator = EmbeddingGenerator::new(config).await.unwrap();
        
        let correct_embedding = vec![0.5; generator.get_dimension()];
        assert!(generator.validate_embedding(&correct_embedding).is_ok());
        
        let incorrect_embedding = vec![0.5; generator.get_dimension() + 10];
        assert!(generator.validate_embedding(&incorrect_embedding).is_err());
        
        let nan_embedding = vec![f32::NAN; generator.get_dimension()];
        assert!(generator.validate_embedding(&nan_embedding).is_err());
    }
    
    #[tokio::test]
    async fn test_memory_calculation() {
        let config = EmbedderConfig::default();
        let generator = EmbeddingGenerator::new(config).await.unwrap();
        
        let memory = generator.calculate_memory_usage(1000);
        let expected = 1000 * generator.get_dimension() * std::mem::size_of::<f32>();
        assert_eq!(memory, expected);
    }
}