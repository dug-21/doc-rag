//! ruv-FANN based embedding models
//!
//! This module implements embedding generation using ruv-FANN neural networks
//! for CONSTRAINT-003 compliance: <10ms inference, no external LLM dependencies.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use anyhow::{Result, Context};
use tracing::{info, warn, debug, instrument};
use tokio::fs;
use hashbrown::HashMap as FastHashMap;

#[cfg(feature = "neural")]
use ruv_fann::{Network, TrainingData};
use crate::{EmbedderConfig, EmbedderError, ModelType};

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

/// ruv-FANN based embedding model
pub struct RuvFannEmbeddingModel {
    network: Mutex<Network<f32>>,
    feature_extractor: TextFeatureExtractor,
    dimension: usize,
    name: String,
    max_length: usize,
}

/// Text feature extractor for converting text to neural network input
pub struct TextFeatureExtractor {
    _vocab: FastHashMap<String, f32>,
    word_weights: FastHashMap<String, f32>,
    feature_count: usize,
}

impl TextFeatureExtractor {
    pub fn new(feature_count: usize) -> Self {
        let mut extractor = Self {
            _vocab: FastHashMap::new(),
            word_weights: FastHashMap::new(),
            feature_count,
        };

        extractor.initialize_default_features();
        extractor
    }

    fn initialize_default_features(&mut self) {
        // Initialize common technical terms with weights
        let technical_terms = [
            ("requirement", 1.0), ("must", 1.2), ("shall", 1.2), ("should", 0.8),
            ("compliance", 1.0), ("security", 1.0), ("encryption", 1.0), ("data", 0.9),
            ("access", 0.8), ("control", 0.8), ("audit", 0.9), ("log", 0.7),
            ("authentication", 1.0), ("authorization", 1.0), ("configuration", 0.8),
            ("system", 0.7), ("network", 0.8), ("policy", 0.9), ("procedure", 0.8),
            ("standard", 1.0), ("guideline", 0.8), ("framework", 0.9), ("document", 0.6),
            ("section", 0.7), ("subsection", 0.6), ("clause", 0.7), ("appendix", 0.6),
            ("reference", 0.8), ("example", 0.6), ("note", 0.5), ("warning", 0.9),
        ];

        for (term, weight) in &technical_terms {
            self.word_weights.insert(term.to_string(), *weight);
        }
    }

    pub fn extract_features(&self, text: &str) -> Vec<f32> {
        let mut features = vec![0.0; self.feature_count];

        // Text preprocessing
        let text_lower = text.to_lowercase();
        let words: Vec<&str> = text_lower.split_whitespace().collect();

        // Basic statistical features (indices 0-19)
        features[0] = words.len() as f32;                    // Word count
        features[1] = text.chars().count() as f32;           // Character count
        features[2] = text_lower.matches('.').count() as f32; // Sentence indicators
        features[3] = text_lower.matches('?').count() as f32; // Question indicators
        features[4] = text_lower.matches(':').count() as f32; // List indicators
        features[5] = text_lower.matches('-').count() as f32; // Dash indicators
        features[6] = text_lower.matches('(').count() as f32; // Parentheses indicators
        features[7] = words.iter().filter(|w| w.chars().all(|c| c.is_ascii_uppercase() || !c.is_alphabetic())).count() as f32; // Uppercase words

        // Calculate average word length
        let avg_word_len = if !words.is_empty() {
            words.iter().map(|w| w.len()).sum::<usize>() as f32 / words.len() as f32
        } else {
            0.0
        };
        features[8] = avg_word_len;

        // Technical term density (indices 9-19)
        features[9] = if text_lower.contains("must") || text_lower.contains("shall") { 1.0 } else { 0.0 };
        features[10] = if text_lower.contains("should") || text_lower.contains("may") { 1.0 } else { 0.0 };
        features[11] = if text_lower.contains("requirement") { 1.0 } else { 0.0 };
        features[12] = if text_lower.contains("compliance") || text_lower.contains("conform") { 1.0 } else { 0.0 };
        features[13] = if text_lower.contains("security") || text_lower.contains("secure") { 1.0 } else { 0.0 };
        features[14] = if text_lower.contains("encryption") || text_lower.contains("encrypt") { 1.0 } else { 0.0 };
        features[15] = if text_lower.contains("access") || text_lower.contains("control") { 1.0 } else { 0.0 };
        features[16] = if text_lower.contains("audit") || text_lower.contains("log") { 1.0 } else { 0.0 };
        features[17] = if text_lower.contains("authentication") || text_lower.contains("authorization") { 1.0 } else { 0.0 };
        features[18] = if text_lower.contains("configuration") || text_lower.contains("setting") { 1.0 } else { 0.0 };
        features[19] = if text_lower.contains("policy") || text_lower.contains("procedure") { 1.0 } else { 0.0 };

        // Document structure features (indices 20-39)
        features[20] = if text_lower.starts_with("section") { 1.0 } else { 0.0 };
        features[21] = if text_lower.starts_with("subsection") { 1.0 } else { 0.0 };
        features[22] = if text_lower.starts_with("appendix") { 1.0 } else { 0.0 };
        features[23] = if text_lower.contains("see section") || text_lower.contains("see ") { 1.0 } else { 0.0 };
        features[24] = if text_lower.contains("refer to") || text_lower.contains("reference") { 1.0 } else { 0.0 };
        features[25] = if text_lower.contains("example") || text_lower.contains("e.g.") { 1.0 } else { 0.0 };
        features[26] = if text_lower.contains("note:") || text_lower.contains("note ") { 1.0 } else { 0.0 };
        features[27] = if text_lower.contains("warning") || text_lower.contains("caution") { 1.0 } else { 0.0 };
        features[28] = if text_lower.contains("figure") || text_lower.contains("table") { 1.0 } else { 0.0 };
        features[29] = if text_lower.contains("step") && (text_lower.contains("1") || text_lower.contains("first")) { 1.0 } else { 0.0 };

        // Numeric and pattern features (indices 30-49)
        features[30] = text.matches(char::is_numeric).count() as f32;
        features[31] = if text.contains('@') { 1.0 } else { 0.0 };
        features[32] = if text.contains("http") || text.contains("www") { 1.0 } else { 0.0 };
        features[33] = if text.contains('/') || text.contains('\\') { 1.0 } else { 0.0 };
        features[34] = text.matches('[').count() as f32;
        features[35] = text.matches('{').count() as f32;
        features[36] = if text.contains("=") || text.contains("!=") { 1.0 } else { 0.0 };
        features[37] = if text.contains("<") || text.contains(">") { 1.0 } else { 0.0 };
        features[38] = if text.contains("%") || text.contains("percent") { 1.0 } else { 0.0 };
        features[39] = if text.contains("$") || text.contains("cost") { 1.0 } else { 0.0 };

        // Weighted word features (indices 50-511)
        let start_idx = 50;
        let available_slots = self.feature_count - start_idx;

        for (i, word) in words.iter().enumerate() {
            if i >= available_slots { break; }

            let weight = self.word_weights.get(*word).copied().unwrap_or(0.1);
            features[start_idx + i] = weight;
        }

        // Normalize features to prevent overflow
        let max_val = features.iter().cloned().fold(0.0f32, f32::max);
        if max_val > 10.0 {
            for feature in &mut features {
                *feature = (*feature / max_val) * 10.0;
            }
        }

        features
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for RuvFannEmbeddingModel {
    async fn encode(&self, text: &str) -> Result<Vec<f32>> {
        let start = std::time::Instant::now();

        // Extract features from text
        let features = self.feature_extractor.extract_features(text);

        // Run neural network inference
        let mut network = self.network.lock().unwrap();
        let embedding = network.run(&features);

        let elapsed = start.elapsed();
        if elapsed.as_millis() > 10 {
            warn!("Inference time {}ms exceeds 10ms constraint", elapsed.as_millis());
        }

        Ok(embedding)
    }

    async fn encode_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let start = std::time::Instant::now();

        let mut results = Vec::with_capacity(texts.len());

        // Process each text individually to maintain <10ms constraint
        for text in texts {
            let embedding = self.encode(text).await?;
            results.push(embedding);
        }

        let elapsed = start.elapsed();
        debug!("Batch encoding of {} texts took {}ms", texts.len(), elapsed.as_millis());

        Ok(results)
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

impl RuvFannEmbeddingModel {
    #[instrument(skip(config))]
    pub async fn load(model_path: &Path, config: &EmbedderConfig) -> Result<Self> {
        info!("Loading ruv-FANN model from: {:?}", model_path);

        // Create neural network from model configuration
        let layers = config.model_type.layers();
        let mut network = Network::<f32>::new(layers);

        // Configure activation functions
        network.set_activation_function_hidden(config.model_type.hidden_activation());
        network.set_activation_function_output(config.model_type.output_activation());

        // Try to load existing weights
        let network_file = model_path.join("network.ruv");
        if network_file.exists() {
            debug!("Loading pre-trained weights from {:?}", network_file);
            let _weights_data = fs::read(&network_file).await
                .with_context(|| format!("Failed to read network file: {:?}", network_file))?;

            // Deserialize weights (simplified - in production you'd use proper serialization)
            // For now, we'll initialize with random weights
            info!("Using default initialization (weights file format not yet implemented)");
        } else {
            info!("No pre-trained weights found, using default initialization");
        }

        // Initialize feature extractor
        let input_size = layers[0];
        let feature_extractor = TextFeatureExtractor::new(input_size);

        let dimension = config.model_type.dimension();
        let name = config.model_type.name().to_string();

        Ok(Self {
            network: Mutex::new(network),
            feature_extractor,
            dimension,
            name,
            max_length: config.max_length,
        })
    }

    /// Save the trained model to disk
    pub async fn save(&self, model_path: &Path) -> Result<()> {
        tokio::fs::create_dir_all(model_path).await?;

        let _network_file = model_path.join("network.ruv");
        let config_file = model_path.join("config.json");

        // Save network weights (simplified serialization)
        let _network = self.network.lock().unwrap();

        // Create basic config for the model
        let config = serde_json::json!({
            "model_type": "ruv_fann",
            "dimension": self.dimension,
            "name": self.name,
            "max_length": self.max_length,
            "created_at": chrono::Utc::now().to_rfc3339()
        });

        tokio::fs::write(&config_file, serde_json::to_string_pretty(&config)?).await?;

        // Note: ruv-FANN serialization would go here in a complete implementation
        info!("Model configuration saved to {:?}", config_file);

        Ok(())
    }

    /// Train the model with provided data
    pub async fn train(&mut self, training_data: &[(String, Vec<f32>)], epochs: usize) -> Result<()> {
        info!("Training model with {} examples for {} epochs", training_data.len(), epochs);

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for (text, target) in training_data {
            let features = self.feature_extractor.extract_features(text);
            inputs.push(features);
            outputs.push(target.clone());
        }

        let train_data = TrainingData {
            inputs,
            outputs,
        };

        let mut network = self.network.lock().unwrap();

        for epoch in 0..epochs {
            let result = network.train(&train_data.inputs, &train_data.outputs, 0.01, 1);
            if epoch % 10 == 0 {
                debug!("Epoch {}: training result = {:?}", epoch, result);
            }
        }

        info!("Training completed");
        Ok(())
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

        // Default model paths for ruv-FANN models
        model_paths.insert(ModelType::default_fast(),
            PathBuf::from("./models/ruv-fann-fast"));
        model_paths.insert(ModelType::default_balanced(),
            PathBuf::from("./models/ruv-fann-balanced"));

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
        info!("Loading ruv-FANN model {:?} from {:?}", model_type, model_path);

        // Ensure model directory exists
        if !model_path.exists() {
            tokio::fs::create_dir_all(&model_path).await?;
        }

        let model: Arc<dyn EmbeddingModel> = Arc::new(
            RuvFannEmbeddingModel::load(&model_path, config).await?
        );

        self.models.insert(model_type.clone(), model);
        info!("Successfully loaded ruv-FANN model {:?}", model_type);

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

    #[tokio::test]
    async fn test_model_manager_creation() {
        let manager = ModelManager::new();
        assert!(!manager.is_model_loaded(&ModelType::default_fast()));
        assert!(manager.loaded_models().is_empty());
    }

    #[tokio::test]
    async fn test_feature_extractor() {
        let extractor = TextFeatureExtractor::new(512);
        let features = extractor.extract_features("This is a test requirement that must be implemented");

        assert_eq!(features.len(), 512);
        assert!(features[0] > 0.0); // Word count
        assert!(features[1] > 0.0); // Character count
        assert!(features[9] > 0.0); // Contains "must"
        assert!(features[11] > 0.0); // Contains "requirement"
    }

    #[tokio::test]
    async fn test_ruv_fann_model_creation() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();

        let config = EmbedderConfig::new();
        let model = RuvFannEmbeddingModel::load(model_path, &config).await;

        assert!(model.is_ok());
        let model = model.unwrap();
        assert_eq!(model.dimension(), 384);
        assert_eq!(model.name(), "ruv-fann-fast");
    }

    #[tokio::test]
    async fn test_embedding_inference() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();

        let config = EmbedderConfig::new();
        let model = RuvFannEmbeddingModel::load(model_path, &config).await.unwrap();

        let start = std::time::Instant::now();
        let embedding = model.encode("Test requirement").await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(embedding.len(), 384);
        assert!(elapsed.as_millis() <= 10, "Inference time {}ms exceeds 10ms constraint", elapsed.as_millis());
    }

    #[tokio::test]
    async fn test_batch_encoding() {
        let temp_dir = tempdir().unwrap();
        let model_path = temp_dir.path();

        let config = EmbedderConfig::new();
        let model = RuvFannEmbeddingModel::load(model_path, &config).await.unwrap();

        let texts = vec![
            "First requirement".to_string(),
            "Second requirement".to_string(),
            "Third requirement".to_string(),
        ];

        let embeddings = model.encode_batch(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in embeddings {
            assert_eq!(embedding.len(), 384);
        }
    }

    #[tokio::test]
    async fn test_model_path_management() {
        let mut manager = ModelManager::new();
        let custom_path = PathBuf::from("/custom/model/path");

        manager.set_model_path(ModelType::default_fast(), custom_path.clone());
        let retrieved_path = manager.get_model_path(&ModelType::default_fast()).unwrap();
        assert_eq!(retrieved_path, custom_path);
    }
}