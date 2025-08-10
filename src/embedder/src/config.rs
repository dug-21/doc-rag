//! Configuration for the embedding generator
//!
//! This module defines all configuration options for the embedding system,
//! including model types, device preferences, batch sizes, and performance tuning.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Configuration for the embedding generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedderConfig {
    /// Type of embedding model to use
    pub model_type: ModelType,
    
    /// Maximum batch size for processing
    pub batch_size: usize,
    
    /// Maximum sequence length for tokenization
    pub max_length: usize,
    
    /// Device to run inference on
    pub device: Device,
    
    /// Whether to normalize embeddings to unit length
    pub normalize: bool,
    
    /// Size of the embedding cache
    pub cache_size: usize,
    
    /// Number of threads for parallel processing
    pub num_threads: Option<usize>,
    
    /// Memory mapping for large models
    pub use_mmap: bool,
    
    /// Model download settings
    pub download_config: ModelDownloadConfig,
    
    /// Performance optimization settings
    pub optimization: OptimizationConfig,
}

/// Available embedding model types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// all-MiniLM-L6-v2: Fast, lightweight model (384 dimensions)
    AllMiniLmL6V2,
    
    /// BERT base uncased (768 dimensions) 
    BertBaseUncased,
    
    /// Sentence-T5 base (768 dimensions)
    SentenceT5Base,
    
    /// Custom model with specified path and dimensions
    Custom {
        name: String,
        path: PathBuf,
        dimension: usize,
    },
}

impl ModelType {
    /// Get the embedding dimension for this model type
    pub fn dimension(&self) -> usize {
        match self {
            ModelType::AllMiniLmL6V2 => 384,
            ModelType::BertBaseUncased => 768,
            ModelType::SentenceT5Base => 768,
            ModelType::Custom { dimension, .. } => *dimension,
        }
    }
    
    /// Get the model name/identifier
    pub fn name(&self) -> &str {
        match self {
            ModelType::AllMiniLmL6V2 => "all-MiniLM-L6-v2",
            ModelType::BertBaseUncased => "bert-base-uncased",
            ModelType::SentenceT5Base => "sentence-t5-base",
            ModelType::Custom { name, .. } => name,
        }
    }
    
    /// Get the Hugging Face model identifier
    pub fn hf_model_id(&self) -> Option<&str> {
        match self {
            ModelType::AllMiniLmL6V2 => Some("sentence-transformers/all-MiniLM-L6-v2"),
            ModelType::BertBaseUncased => Some("bert-base-uncased"),
            ModelType::SentenceT5Base => Some("sentence-transformers/sentence-t5-base"),
            ModelType::Custom { .. } => None,
        }
    }
    
    /// Get the default maximum sequence length
    pub fn default_max_length(&self) -> usize {
        match self {
            ModelType::AllMiniLmL6V2 => 512,
            ModelType::BertBaseUncased => 512,
            ModelType::SentenceT5Base => 512,
            ModelType::Custom { .. } => 512,
        }
    }
    
    /// Check if this model supports ONNX format
    pub fn supports_onnx(&self) -> bool {
        match self {
            ModelType::AllMiniLmL6V2 | ModelType::BertBaseUncased => true,
            ModelType::SentenceT5Base => false, // T5 is more complex
            ModelType::Custom { .. } => true,
        }
    }
    
    /// Get expected model files
    pub fn expected_files(&self) -> Vec<&'static str> {
        match self {
            ModelType::AllMiniLmL6V2 | ModelType::BertBaseUncased => {
                vec![
                    "config.json",
                    "pytorch_model.bin",
                    "tokenizer_config.json",
                    "vocab.txt",
                ]
            }
            ModelType::SentenceT5Base => {
                vec![
                    "config.json",
                    "pytorch_model.bin",
                    "tokenizer_config.json",
                    "spiece.model",
                ]
            }
            ModelType::Custom { .. } => {
                vec!["config.json", "vocab.txt"]
            }
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Device for running inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// CPU inference
    Cpu,
    /// CUDA GPU inference
    Cuda,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda => write!(f, "cuda"),
        }
    }
}

/// Configuration for model downloading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDownloadConfig {
    /// Base URL for model downloads (defaults to Hugging Face)
    pub base_url: String,
    
    /// Local cache directory for models
    pub cache_dir: PathBuf,
    
    /// Whether to automatically download missing models
    pub auto_download: bool,
    
    /// Connection timeout for downloads (seconds)
    pub timeout_secs: u64,
    
    /// Maximum number of download retries
    pub max_retries: u32,
    
    /// Use authentication token for private models
    pub auth_token: Option<String>,
}

impl Default for ModelDownloadConfig {
    fn default() -> Self {
        Self {
            base_url: "https://huggingface.co".to_string(),
            cache_dir: PathBuf::from("./models"),
            auto_download: true,
            timeout_secs: 300, // 5 minutes
            max_retries: 3,
            auth_token: None,
        }
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Use mixed precision (FP16) inference
    pub use_fp16: bool,
    
    /// Use quantization for model weights
    pub quantization: QuantizationType,
    
    /// Enable ONNX Runtime optimizations
    pub onnx_optimization: OnnxOptimizationLevel,
    
    /// Number of intra-op threads
    pub intra_op_threads: Option<usize>,
    
    /// Number of inter-op threads  
    pub inter_op_threads: Option<usize>,
    
    /// Enable memory optimization
    pub memory_optimization: bool,
    
    /// Batch size optimization strategy
    pub batch_strategy: BatchStrategy,
    
    /// Enable model compilation (where supported)
    pub enable_compilation: bool,
}

/// Quantization options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization
    None,
    /// 8-bit quantization
    Int8,
    /// 4-bit quantization (experimental)
    Int4,
}

/// ONNX optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnnxOptimizationLevel {
    /// Disable optimizations
    Disabled,
    /// Basic optimizations
    Basic,
    /// Extended optimizations  
    Extended,
    /// All optimizations
    All,
}

/// Batch processing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchStrategy {
    /// Fixed batch size
    Fixed,
    /// Dynamic batch size based on available memory
    Dynamic,
    /// Adaptive batch size based on input length distribution
    Adaptive,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            use_fp16: false,
            quantization: QuantizationType::None,
            onnx_optimization: OnnxOptimizationLevel::Extended,
            intra_op_threads: None,
            inter_op_threads: None,
            memory_optimization: true,
            batch_strategy: BatchStrategy::Fixed,
            enable_compilation: false,
        }
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::AllMiniLmL6V2,
            batch_size: 32,
            max_length: 512,
            device: Device::Cpu,
            normalize: true,
            cache_size: 10000,
            num_threads: None,
            use_mmap: true,
            download_config: ModelDownloadConfig::default(),
            optimization: OptimizationConfig::default(),
        }
    }
}

impl EmbedderConfig {
    /// Create a new configuration with sensible defaults
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the model type
    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.max_length = model_type.default_max_length();
        self.model_type = model_type;
        self
    }
    
    /// Set the batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }
    
    /// Set the maximum sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length.max(1);
        self
    }
    
    /// Set the device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    
    /// Enable or disable normalization
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
    
    /// Set the cache size
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }
    
    /// Set the number of threads
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }
    
    /// Configure for high performance
    pub fn high_performance(mut self) -> Self {
        self.optimization.use_fp16 = true;
        self.optimization.memory_optimization = true;
        self.optimization.batch_strategy = BatchStrategy::Dynamic;
        self.optimization.enable_compilation = true;
        self.batch_size = 64;
        self
    }
    
    /// Configure for low memory usage
    pub fn low_memory(mut self) -> Self {
        self.batch_size = 8;
        self.cache_size = 1000;
        self.optimization.memory_optimization = true;
        self.optimization.quantization = QuantizationType::Int8;
        self.use_mmap = false;
        self
    }
    
    /// Configure for CUDA if available
    pub fn with_cuda_if_available(mut self) -> Self {
        // In a real implementation, you'd check CUDA availability
        self.device = Device::Cuda;
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }
        
        if self.max_length == 0 {
            return Err("Max length must be greater than 0".to_string());
        }
        
        if self.cache_size == 0 {
            return Err("Cache size must be greater than 0".to_string());
        }
        
        if let Some(threads) = self.num_threads {
            if threads == 0 {
                return Err("Number of threads must be greater than 0".to_string());
            }
        }
        
        if self.download_config.timeout_secs == 0 {
            return Err("Download timeout must be greater than 0".to_string());
        }
        
        Ok(())
    }
    
    /// Get estimated memory usage in bytes
    pub fn estimated_memory_usage(&self, num_embeddings: usize) -> usize {
        let embedding_size = self.model_type.dimension() * std::mem::size_of::<f32>();
        let cache_size = self.cache_size * embedding_size;
        let batch_size = num_embeddings.min(self.batch_size) * embedding_size;
        
        // Add some overhead for model weights and intermediate tensors
        let model_overhead = match self.model_type {
            ModelType::AllMiniLmL6V2 => 90_000_000, // ~90MB
            ModelType::BertBaseUncased => 440_000_000, // ~440MB
            ModelType::SentenceT5Base => 220_000_000, // ~220MB
            ModelType::Custom { .. } => 100_000_000, // Estimate
        };
        
        cache_size + batch_size + model_overhead
    }
    
    /// Create configuration optimized for the given constraints
    pub fn optimize_for_constraints(
        mut self,
        max_memory_mb: Option<usize>,
        min_throughput_per_sec: Option<usize>,
        max_latency_ms: Option<usize>,
    ) -> Self {
        if let Some(max_memory) = max_memory_mb {
            let max_memory_bytes = max_memory * 1_048_576; // Convert MB to bytes
            
            // Adjust cache size and batch size based on memory constraints
            let embedding_size = self.model_type.dimension() * std::mem::size_of::<f32>();
            let max_cache_entries = max_memory_bytes / (embedding_size * 4); // Reserve 75% for model
            
            self.cache_size = self.cache_size.min(max_cache_entries);
            self.batch_size = self.batch_size.min(max_memory_bytes / (embedding_size * 100));
            
            if max_memory < 500 {
                self = self.low_memory();
            }
        }
        
        if let Some(min_throughput) = min_throughput_per_sec {
            // Increase batch size for higher throughput
            self.batch_size = (min_throughput / 10).max(self.batch_size);
            self.optimization.batch_strategy = BatchStrategy::Dynamic;
        }
        
        if let Some(max_latency) = max_latency_ms {
            if max_latency < 100 {
                // Optimize for low latency
                self.batch_size = 1;
                self.optimization.enable_compilation = true;
                self.optimization.memory_optimization = true;
            }
        }
        
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_creation() {
        let config = EmbedderConfig::new();
        assert_eq!(config.model_type, ModelType::AllMiniLmL6V2);
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize);
    }
    
    #[test]
    fn test_config_builder_pattern() {
        let config = EmbedderConfig::new()
            .with_model_type(ModelType::BertBaseUncased)
            .with_batch_size(64)
            .with_device(Device::Cuda)
            .with_normalize(false);
        
        assert_eq!(config.model_type, ModelType::BertBaseUncased);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.device, Device::Cuda);
        assert!(!config.normalize);
    }
    
    #[test]
    fn test_config_validation() {
        let config = EmbedderConfig::new();
        assert!(config.validate().is_ok());
        
        let invalid_config = EmbedderConfig {
            batch_size: 0,
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_model_type_properties() {
        let model = ModelType::AllMiniLmL6V2;
        assert_eq!(model.dimension(), 384);
        assert_eq!(model.name(), "all-MiniLM-L6-v2");
        assert!(model.supports_onnx());
        assert!(model.hf_model_id().is_some());
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = EmbedderConfig::new();
        let memory = config.estimated_memory_usage(1000);
        assert!(memory > 0);
    }
    
    #[test]
    fn test_optimization_configs() {
        let high_perf = EmbedderConfig::new().high_performance();
        assert!(high_perf.optimization.use_fp16);
        assert_eq!(high_perf.batch_size, 64);
        
        let low_mem = EmbedderConfig::new().low_memory();
        assert_eq!(low_mem.batch_size, 8);
        assert_eq!(low_mem.optimization.quantization, QuantizationType::Int8);
    }
    
    #[test]
    fn test_constraint_optimization() {
        let config = EmbedderConfig::new()
            .optimize_for_constraints(Some(256), Some(1000), Some(50));
        
        assert!(config.cache_size <= 10000); // Should be reduced for low memory
        assert_eq!(config.batch_size, 1); // Should be 1 for low latency
    }
}