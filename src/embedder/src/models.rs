//! Model management for different embedding models
//!
//! This module handles loading, caching, and managing different embedding models
//! including ONNX models via ORT and Candle native models.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use anyhow::{Result, Context};
use tracing::{info, warn, debug, instrument};
use tokio::fs;
use candle_core::IndexOp;

use crate::{EmbedderConfig, EmbedderError, ModelType, Device};

/// Trait for embedding models
#[async_trait::async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Encode a single text into an embedding
    async fn encode(&self, text: &str) -> Result<Vec<f32>>;
    
    /// Encode a batch of texts into embeddings
    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    
    /// Get the embedding dimension
    fn dimension(&self) -> usize;
    
    /// Get the model name
    fn name(&self) -> String;
    
    /// Get the maximum sequence length
    fn max_length(&self) -> usize;
}

/// ONNX Runtime based embedding model
pub struct OnnxEmbeddingModel {
    #[cfg(feature = "ort")]
    session: ort::Session,
    #[cfg(not(feature = "ort"))]
    session: (),
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    dimension: usize,
    name: String,
    max_length: usize,
}

/// Candle based embedding model
pub struct CandleEmbeddingModel {
    model: candle_transformers::models::bert::BertModel,
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
    device: candle_core::Device,
    dimension: usize,
    name: String,
    max_length: usize,
}

/// Tokenizer trait for different tokenizer implementations
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<TokenizerOutput>;
    fn encode_batch(&self, texts: &[String]) -> Result<Vec<TokenizerOutput>>;
    fn get_vocab_size(&self) -> usize;
}

/// Output from tokenizer
#[derive(Debug, Clone)]
pub struct TokenizerOutput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Option<Vec<i64>>,
}

/// Simple BERT-like tokenizer implementation
pub struct BertTokenizer {
    vocab: HashMap<String, i64>,
    vocab_size: usize,
    max_length: usize,
    cls_token: String,
    sep_token: String,
    unk_token: String,
    pad_token: String,
}

impl BertTokenizer {
    pub fn new(vocab_path: &Path, max_length: usize) -> Result<Self> {
        let vocab_content = std::fs::read_to_string(vocab_path)
            .with_context(|| format!("Failed to read vocab file: {:?}", vocab_path))?;
        
        let mut vocab = HashMap::new();
        for (idx, line) in vocab_content.lines().enumerate() {
            vocab.insert(line.trim().to_string(), idx as i64);
        }
        
        let vocab_size = vocab.len();
        
        Ok(Self {
            vocab,
            vocab_size,
            max_length,
            cls_token: "[CLS]".to_string(),
            sep_token: "[SEP]".to_string(),
            unk_token: "[UNK]".to_string(),
            pad_token: "[PAD]".to_string(),
        })
    }
    
    fn tokenize(&self, text: &str) -> Vec<String> {
        // Simple whitespace and punctuation tokenization
        // In a real implementation, you'd use a proper tokenizer like tokenizers-rs
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        
        for ch in text.chars() {
            if ch.is_whitespace() || ch.is_ascii_punctuation() {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                if ch.is_ascii_punctuation() {
                    tokens.push(ch.to_string());
                }
            } else {
                current_token.push(ch);
            }
        }
        
        if !current_token.is_empty() {
            tokens.push(current_token);
        }
        
        tokens
    }
}

impl Tokenizer for BertTokenizer {
    fn encode(&self, text: &str) -> Result<TokenizerOutput> {
        let tokens = self.tokenize(text);
        let mut input_ids: Vec<i64> = Vec::new();
        let mut attention_mask = Vec::new();
        
        // Add [CLS] token
        input_ids.push(*self.vocab.get(&self.cls_token).unwrap_or(&0));
        attention_mask.push(1);
        
        // Add tokens (truncate if necessary)
        let max_tokens = self.max_length - 2; // Account for [CLS] and [SEP]
        for token in tokens.iter().take(max_tokens) {
            let id = *self.vocab.get(token).unwrap_or(
                self.vocab.get(&self.unk_token).unwrap_or(&0)
            );
            input_ids.push(id);
            attention_mask.push(1);
        }
        
        // Add [SEP] token
        input_ids.push(*self.vocab.get(&self.sep_token).unwrap_or(&1));
        attention_mask.push(1);
        
        // Pad to max_length
        let pad_id = *self.vocab.get(&self.pad_token).unwrap_or(&0);
        while input_ids.len() < self.max_length {
            input_ids.push(pad_id);
            attention_mask.push(0);
        }
        
        Ok(TokenizerOutput {
            input_ids,
            attention_mask,
            token_type_ids: Some(vec![0; self.max_length]),
        })
    }
    
    fn encode_batch(&self, texts: &[String]) -> Result<Vec<TokenizerOutput>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }
    
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for OnnxEmbeddingModel {
    async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let tokenized = self.tokenizer.encode(text)?;
        let batch = vec![tokenized];
        let embeddings = self.encode_tokenized_batch(&batch).await?;
        Ok(embeddings.into_iter().next().unwrap())
    }
    
    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let tokenized_batch = self.tokenizer.encode_batch(texts)?;
        self.encode_tokenized_batch(&tokenized_batch).await
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> String {
        self.name.clone()
    }
    
    fn max_length(&self) -> usize {
        self.max_length
    }
}

impl OnnxEmbeddingModel {
    #[instrument(skip(config))]
    pub async fn load(model_path: &Path, config: &EmbedderConfig) -> Result<Self> {
        info!("Loading ONNX model from: {:?}", model_path);
        
        // Initialize ONNX Runtime session with ORT 2.0 API
        #[cfg(feature = "ort")]
        let session = {
            ort::Session::builder()?
                .commit_from_file(model_path)?
        };
        
        #[cfg(not(feature = "ort"))]
        let _session = ();
        
        // Load tokenizer
        let tokenizer_path = model_path.parent()
            .ok_or_else(|| EmbedderError::ModelNotFound)?
            .join("vocab.txt");
        
        let tokenizer = Box::new(BertTokenizer::new(&tokenizer_path, config.max_length)?)
            as Box<dyn Tokenizer + Send + Sync>;
        
        // Get model metadata
        let dimension = match config.model_type {
            ModelType::AllMiniLmL6V2 => 384,
            ModelType::BertBaseUncased => 768,
            ModelType::SentenceT5Base => 768,
            ModelType::Custom { dimension, .. } => dimension,
        };
        
        let name = model_path.file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        
        Ok(Self {
            #[cfg(feature = "ort")]
            session,
            #[cfg(not(feature = "ort"))]
            session: _session,
            tokenizer,
            dimension,
            name,
            max_length: config.max_length,
        })
    }
    
    async fn encode_tokenized_batch(&self, batch: &[TokenizerOutput]) -> Result<Vec<Vec<f32>>> {
        // ORT inputs functionality - API changed in newer versions
        
        let batch_size = batch.len();
        let seq_len = batch[0].input_ids.len();
        
        // Prepare input tensors
        let mut input_ids: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        
        for tokenized in batch {
            input_ids.extend(&tokenized.input_ids);
            attention_mask.extend(&tokenized.attention_mask);
        }
        
        // Run inference
        #[cfg(feature = "ort")]
        let outputs = {
            self.session.run(ort::inputs![
                "input_ids" => ndarray::Array2::from_shape_vec(
                    (batch_size, seq_len),
                    input_ids
                )?.into_dyn(),
                "attention_mask" => ndarray::Array2::from_shape_vec(
                    (batch_size, seq_len),
                    attention_mask
                )?.into_dyn()
            ])?
        };
        
        #[cfg(not(feature = "ort"))]
        {
            // Return dummy results when ONNX is disabled
            let mut results = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                results.push(vec![0.0f32; self.dimension]);
            }
            return Ok(results);
        }
        
        // Extract embeddings (usually from the pooler output or mean pooling of last hidden states)
        #[cfg(feature = "ort")]
        let embeddings_tensor = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()?;
        
        #[cfg(feature = "ort")]
        let embeddings_array = embeddings_tensor.view();
        
        // Mean pooling over sequence dimension
        #[cfg(feature = "ort")]
        {
            let mut results = Vec::with_capacity(batch_size);
            for batch_idx in 0..batch_size {
                let mut embedding = vec![0.0f32; self.dimension];
                let mut valid_tokens = 0;
                
                for seq_idx in 0..seq_len {
                    // Only consider non-padded tokens
                    if batch[batch_idx].attention_mask[seq_idx] == 1 {
                        for dim in 0..self.dimension {
                            embedding[dim] += embeddings_array[[batch_idx, seq_idx, dim]];
                        }
                        valid_tokens += 1;
                    }
                }
                
                // Average
                if valid_tokens > 0 {
                    for dim in 0..self.dimension {
                        embedding[dim] /= valid_tokens as f32;
                    }
                }
                
                results.push(embedding);
            }
            return Ok(results);
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for CandleEmbeddingModel {
    async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let batch = vec![text.to_string()];
        let embeddings = self.encode_batch(&batch).await?;
        Ok(embeddings.into_iter().next().unwrap())
    }
    
    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let tokenized_batch = self.tokenizer.encode_batch(texts)?;
        self.encode_tokenized_batch(&tokenized_batch).await
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> String {
        self.name.clone()
    }
    
    fn max_length(&self) -> usize {
        self.max_length
    }
}

impl CandleEmbeddingModel {
    #[instrument(skip(config))]
    pub async fn load(model_path: &Path, config: &EmbedderConfig) -> Result<Self> {
        info!("Loading Candle model from: {:?}", model_path);
        
        let device = match config.device {
            Device::Cpu => candle_core::Device::Cpu,
            Device::Cuda => {
                candle_core::Device::new_cuda(0)
                    .unwrap_or_else(|_| {
                        warn!("CUDA not available, falling back to CPU");
                        candle_core::Device::Cpu
                    })
            }
        };
        
        // Load model configuration
        let config_path = model_path.join("config.json");
        let model_config: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(&config_path).await
                .with_context(|| format!("Failed to read config: {:?}", config_path))?
        )?;
        
        // Extract model parameters
        let vocab_size = model_config["vocab_size"].as_u64().unwrap_or(30522) as usize;
        let hidden_size = model_config["hidden_size"].as_u64().unwrap_or(768) as usize;
        let num_hidden_layers = model_config["num_hidden_layers"].as_u64().unwrap_or(12) as usize;
        let num_attention_heads = model_config["num_attention_heads"].as_u64().unwrap_or(12) as usize;
        let intermediate_size = model_config["intermediate_size"].as_u64().unwrap_or(3072) as usize;
        let max_position_embeddings = model_config["max_position_embeddings"].as_u64().unwrap_or(512) as usize;
        
        // Create BERT configuration
        let bert_config = candle_transformers::models::bert::Config {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            intermediate_size,
            hidden_act: candle_transformers::models::bert::HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            // attention_probs_dropout_prob removed in newer candle versions
            max_position_embeddings,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: candle_transformers::models::bert::PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: None,
        };
        
        // Load model weights
        let weights_path = model_path.join("pytorch_model.bin");
        let vb = if weights_path.exists() {
            candle_nn::VarBuilder::from_pth(&weights_path, candle_core::DType::F32, &device)?
        } else {
            // Try safetensors format
            let safetensors_path = model_path.join("model.safetensors");
            let tensors = candle_core::safetensors::load(&safetensors_path, &device)?;
            candle_nn::VarBuilder::from_tensors(tensors, candle_core::DType::F32, &device)
        };
        
        // Initialize BERT model
        let model = candle_transformers::models::bert::BertModel::load(vb, &bert_config)?;
        
        // Load tokenizer
        let tokenizer_path = model_path.join("vocab.txt");
        let tokenizer = Box::new(BertTokenizer::new(&tokenizer_path, config.max_length)?)
            as Box<dyn Tokenizer + Send + Sync>;
        
        let name = model_path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        
        Ok(Self {
            model,
            tokenizer,
            device,
            dimension: hidden_size,
            name,
            max_length: config.max_length,
        })
    }
    
    async fn encode_tokenized_batch(&self, batch: &[TokenizerOutput]) -> Result<Vec<Vec<f32>>> {
        use candle_core::Tensor;
        
        let batch_size = batch.len();
        let seq_len = batch[0].input_ids.len();
        
        // Prepare input tensors
        let mut input_ids_data = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_data = Vec::with_capacity(batch_size * seq_len);
        
        for tokenized in batch {
            input_ids_data.extend(tokenized.input_ids.iter().map(|&x| x as u32));
            attention_mask_data.extend(tokenized.attention_mask.iter().map(|&x| x as u32));
        }
        
        let input_ids = Tensor::from_vec(input_ids_data, (batch_size, seq_len), &self.device)?;
        let attention_mask = Tensor::from_vec(attention_mask_data, (batch_size, seq_len), &self.device)?;
        
        // Forward pass
        let hidden_states = self.model.forward(&input_ids, &attention_mask, None)?;
        
        // Mean pooling
        let mut results = Vec::with_capacity(batch_size);
        
        for batch_idx in 0..batch_size {
            let hidden = hidden_states.i(batch_idx)?;
            let mask = attention_mask.i(batch_idx)?;
            
            // Apply attention mask and compute mean
            let masked_hidden = hidden.broadcast_mul(&mask.unsqueeze(1)?)?;
            let sum_hidden = masked_hidden.sum(0)?;
            let mask_sum = mask.sum_all()?;
            
            let pooled = sum_hidden.div(&mask_sum)?;
            let embedding: Vec<f32> = pooled.to_vec1()?;
            
            results.push(embedding);
        }
        
        Ok(results)
    }
}

/// Model manager for loading and caching embedding models
pub struct ModelManager {
    models: HashMap<ModelType, Arc<dyn EmbeddingModel>>,
    model_paths: HashMap<ModelType, PathBuf>,
}

impl ModelManager {
    pub fn new() -> Self {
        let mut model_paths = HashMap::new();
        
        // Default model paths
        model_paths.insert(ModelType::AllMiniLmL6V2, 
            PathBuf::from("/models/all-MiniLM-L6-v2"));
        model_paths.insert(ModelType::BertBaseUncased, 
            PathBuf::from("/models/bert-base-uncased"));
        model_paths.insert(ModelType::SentenceT5Base, 
            PathBuf::from("/models/sentence-t5-base"));
        
        Self {
            models: HashMap::new(),
            model_paths,
        }
    }
    
    #[instrument(skip(self, config))]
    pub async fn load_model(&mut self, model_type: &ModelType, config: &EmbedderConfig) -> Result<()> {
        if self.models.contains_key(model_type) {
            debug!("Model {:?} already loaded", model_type);
            return Ok(());
        }
        
        let model_path = self.get_model_path(model_type)?;
        info!("Loading model {:?} from {:?}", model_type, model_path);
        
        let model: Arc<dyn EmbeddingModel> = if model_path.join("model.onnx").exists() {
            Arc::new(OnnxEmbeddingModel::load(&model_path.join("model.onnx"), config).await?)
        } else {
            Arc::new(CandleEmbeddingModel::load(&model_path, config).await?)
        };
        
        self.models.insert(model_type.clone(), model);
        info!("Successfully loaded model {:?}", model_type);
        
        Ok(())
    }
    
    pub fn get_model(&self, model_type: &ModelType) -> Result<Arc<dyn EmbeddingModel>> {
        self.models.get(model_type)
            .cloned()
            .ok_or_else(|| EmbedderError::ModelNotLoaded {
                model_type: format!("{:?}", model_type)
            }.into())
    }
    
    pub fn is_model_loaded(&self, model_type: &ModelType) -> bool {
        self.models.contains_key(model_type)
    }
    
    pub fn set_model_path(&mut self, model_type: ModelType, path: PathBuf) {
        self.model_paths.insert(model_type, path);
    }
    
    fn get_model_path(&self, model_type: &ModelType) -> Result<PathBuf> {
        match model_type {
            ModelType::Custom { path, .. } => Ok(path.clone()),
            _ => self.model_paths.get(model_type)
                .cloned()
                .ok_or_else(|| EmbedderError::ModelNotFound.into())
        }
    }
    
    pub fn unload_model(&mut self, model_type: &ModelType) {
        self.models.remove(model_type);
    }
    
    pub fn clear_cache(&mut self) {
        self.models.clear();
    }
    
    pub fn loaded_models(&self) -> Vec<ModelType> {
        self.models.keys().cloned().collect()
    }
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::io::Write;
    
    #[tokio::test]
    async fn test_model_manager_creation() {
        let manager = ModelManager::new();
        assert!(!manager.is_model_loaded(&ModelType::AllMiniLmL6V2));
        assert!(manager.loaded_models().is_empty());
    }
    
    #[tokio::test] 
    async fn test_tokenizer_basic() {
        let temp_dir = tempdir().unwrap();
        let vocab_path = temp_dir.path().join("vocab.txt");
        let mut vocab_file = std::fs::File::create(&vocab_path).unwrap();
        
        writeln!(vocab_file, "[PAD]").unwrap();
        writeln!(vocab_file, "[UNK]").unwrap();
        writeln!(vocab_file, "[CLS]").unwrap();
        writeln!(vocab_file, "[SEP]").unwrap();
        writeln!(vocab_file, "hello").unwrap();
        writeln!(vocab_file, "world").unwrap();
        
        let tokenizer = BertTokenizer::new(&vocab_path, 128).unwrap();
        let output = tokenizer.encode("hello world").unwrap();
        
        assert_eq!(output.input_ids.len(), 128);
        assert_eq!(output.attention_mask.len(), 128);
        assert!(output.input_ids[0] == 2); // [CLS] token
        assert!(output.attention_mask[0] == 1);
    }
    
    #[tokio::test]
    async fn test_model_path_management() {
        let mut manager = ModelManager::new();
        let custom_path = PathBuf::from("/custom/model/path");
        
        manager.set_model_path(ModelType::AllMiniLmL6V2, custom_path.clone());
        let retrieved_path = manager.get_model_path(&ModelType::AllMiniLmL6V2).unwrap();
        assert_eq!(retrieved_path, custom_path);
    }
}