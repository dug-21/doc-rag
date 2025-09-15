//! Configuration for the embedding generator
//!
//! This module defines all configuration options for the embedding system,
//! including model types, device preferences, batch sizes, and performance tuning.

use std::path::PathBuf;
use serde::{Deserialize, Serialize};
#[cfg(feature = "neural")]
use ruv_fann::ActivationFunction;

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
    
    /// Model storage settings
    pub storage_config: ModelStorageConfig,
    
    /// Performance optimization settings
    pub optimization: OptimizationConfig,
}

/// Available embedding model types (ruv-FANN compliant)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// Fast lightweight classification model (384 dimensions)
    #[cfg(feature = "neural")]
    RuvFannFast {
        layers: Vec<usize>,
        hidden_activation: ActivationFunction,
        output_activation: ActivationFunction,
        dimension: usize,
    },

    /// Balanced performance model (512 dimensions)
    #[cfg(feature = "neural")]
    RuvFannBalanced {
        layers: Vec<usize>,
        hidden_activation: ActivationFunction,
        output_activation: ActivationFunction,
        dimension: usize,
    },

    /// Custom ruv-FANN model with specified configuration
    #[cfg(feature = "neural")]
    Custom {
        name: String,
        path: PathBuf,
        layers: Vec<usize>,
        hidden_activation: ActivationFunction,
        output_activation: ActivationFunction,
        dimension: usize,
    },

    /// Fallback model type for when neural features are disabled
    #[cfg(not(feature = "neural"))]
    Fallback {
        dimension: usize,
    },
}

impl ModelType {
    /// Get the embedding dimension for this model type
    pub fn dimension(&self) -> usize {
        match self {
            #[cfg(feature = "neural")]
            ModelType::RuvFannFast { dimension, .. } => *dimension,
            #[cfg(feature = "neural")]
            ModelType::RuvFannBalanced { dimension, .. } => *dimension,
            #[cfg(feature = "neural")]
            ModelType::Custom { dimension, .. } => *dimension,
            #[cfg(not(feature = "neural"))]
            ModelType::Fallback { dimension } => *dimension,
        }
    }

    /// Get the model name/identifier
    pub fn name(&self) -> &str {
        match self {
            #[cfg(feature = "neural")]
            ModelType::RuvFannFast { .. } => "ruv-fann-fast",
            #[cfg(feature = "neural")]
            ModelType::RuvFannBalanced { .. } => "ruv-fann-balanced",
            #[cfg(feature = "neural")]
            ModelType::Custom { name, .. } => name,
            #[cfg(not(feature = "neural"))]
            ModelType::Fallback { .. } => "fallback",
        }
    }

    /// Get the network layers configuration
    #[cfg(feature = "neural")]
    pub fn layers(&self) -> &[usize] {
        match self {
            ModelType::RuvFannFast { layers, .. } => layers,
            ModelType::RuvFannBalanced { layers, .. } => layers,
            ModelType::Custom { layers, .. } => layers,
        }
    }

    /// Get the network layers configuration (fallback)
    #[cfg(not(feature = "neural"))]
    pub fn layers(&self) -> &[usize] {
        match self {
            ModelType::Fallback { .. } => &[128, 64],
        }
    }

    /// Get the hidden layer activation function
    #[cfg(feature = "neural")]
    pub fn hidden_activation(&self) -> ActivationFunction {
        match self {
            ModelType::RuvFannFast { hidden_activation, .. } => *hidden_activation,
            ModelType::RuvFannBalanced { hidden_activation, .. } => *hidden_activation,
            ModelType::Custom { hidden_activation, .. } => *hidden_activation,
        }
    }

    /// Get the output layer activation function
    #[cfg(feature = "neural")]
    pub fn output_activation(&self) -> ActivationFunction {
        match self {
            ModelType::RuvFannFast { output_activation, .. } => *output_activation,
            ModelType::RuvFannBalanced { output_activation, .. } => *output_activation,
            ModelType::Custom { output_activation, .. } => *output_activation,
        }
    }

    /// Get the default maximum sequence length for feature extraction
    pub fn default_max_length(&self) -> usize {
        512  // Standard feature extraction window
    }

    /// Get expected model files for ruv-FANN models
    pub fn expected_files(&self) -> Vec<&'static str> {
        vec!["network.ruv", "config.json"]
    }

    /// Create default fast model configuration
    #[cfg(feature = "neural")]
    pub fn default_fast() -> Self {
        Self::RuvFannFast {
            layers: vec![512, 256, 128, 384],  // text_features -> embedding
            hidden_activation: ActivationFunction::SigmoidSymmetric,
            output_activation: ActivationFunction::Linear,
            dimension: 384,
        }
    }

    /// Create default balanced model configuration
    #[cfg(feature = "neural")]
    pub fn default_balanced() -> Self {
        Self::RuvFannBalanced {
            layers: vec![512, 256, 128, 512],  // text_features -> embedding
            hidden_activation: ActivationFunction::SigmoidSymmetric,
            output_activation: ActivationFunction::Linear,
            dimension: 512,
        }
    }

    /// Create default fast model configuration (fallback)
    #[cfg(not(feature = "neural"))]
    pub fn default_fast() -> Self {
        Self::Fallback {
            dimension: 384,
        }
    }

    /// Create default balanced model configuration (fallback)
    #[cfg(not(feature = "neural"))]
    pub fn default_balanced() -> Self {
        Self::Fallback {
            dimension: 512,
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

/// Configuration for ruv-FANN model storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStorageConfig {
    /// Local directory for ruv-FANN models
    pub model_dir: PathBuf,

    /// Whether to automatically initialize missing models
    pub auto_initialize: bool,

    /// Model persistence format
    pub persistence_format: ModelPersistenceFormat,
}

/// Model persistence format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelPersistenceFormat {
    /// Binary ruv-FANN format
    Binary,
    /// JSON format for debugging
    Json,
}

impl Default for ModelStorageConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("./models"),
            auto_initialize: true,
            persistence_format: ModelPersistenceFormat::Binary,
        }
    }
}

/// Performance optimization configuration for ruv-FANN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Target inference time (must be <10ms per CONSTRAINT-003)
    pub target_inference_time_ms: u64,

    /// Enable parallel processing for batch operations
    pub enable_parallel_processing: bool,

    /// Number of threads for parallel processing
    pub num_threads: Option<usize>,

    /// Batch processing strategy
    pub batch_strategy: BatchStrategy,

    /// Enable feature extraction caching
    pub cache_features: bool,

    /// Learning rate for online training (if enabled)
    pub learning_rate: f32,
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
            target_inference_time_ms: 10, // CONSTRAINT-003: <10ms
            enable_parallel_processing: true,
            num_threads: None,
            batch_strategy: BatchStrategy::Fixed,
            cache_features: true,
            learning_rate: 0.001,
        }
    }
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::default_fast(),
            batch_size: 32,
            max_length: 512,
            device: Device::Cpu,
            normalize: true,
            cache_size: 10000,
            num_threads: None,
            use_mmap: true,
            storage_config: ModelStorageConfig::default(),
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
        self.optimization.enable_parallel_processing = true;
        self.optimization.batch_strategy = BatchStrategy::Dynamic;
        self.optimization.cache_features = true;
        self.batch_size = 64;
        self
    }

    /// Configure for low memory usage
    pub fn low_memory(mut self) -> Self {
        self.batch_size = 8;
        self.cache_size = 1000;
        self.optimization.cache_features = true;
        self.optimization.target_inference_time_ms = 5; // Faster for low memory
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
        
        if self.optimization.target_inference_time_ms == 0 {
            return Err("Target inference time must be greater than 0".to_string());
        }

        if self.optimization.target_inference_time_ms > 10 {
            return Err("Target inference time must be ≤10ms per CONSTRAINT-003".to_string());
        }
        
        Ok(())
    }
    
    /// Get estimated memory usage in bytes
    pub fn estimated_memory_usage(&self, num_embeddings: usize) -> usize {
        let embedding_size = self.model_type.dimension() * std::mem::size_of::<f32>();
        let cache_size = self.cache_size * embedding_size;
        let batch_size = num_embeddings.min(self.batch_size) * embedding_size;
        
        // Add overhead for ruv-FANN model weights (much smaller)
        let total_params: usize = self.model_type.layers().windows(2)
            .map(|pair| pair[0] * pair[1])
            .sum();
        let model_overhead = total_params * std::mem::size_of::<f32>(); // Parameters in f32
        
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
                self.optimization.enable_parallel_processing = false; // Single threaded for low latency
                self.optimization.cache_features = true;
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
        assert_eq!(config.model_type.name(), "ruv-fann-fast");
        assert_eq!(config.batch_size, 32);
        assert!(config.normalize);
    }
    
    #[test]
    fn test_config_builder_pattern() {
        let config = EmbedderConfig::new()
            .with_model_type(ModelType::default_balanced())
            .with_batch_size(64)
            .with_device(Device::Cuda)
            .with_normalize(false);
        
        assert_eq!(config.model_type.name(), "ruv-fann-balanced");
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
        let model = ModelType::default_fast();
        assert_eq!(model.dimension(), 384);
        assert_eq!(model.name(), "ruv-fann-fast");
        assert_eq!(model.layers(), &[512, 256, 128, 384]);
        assert_eq!(model.hidden_activation(), ActivationFunction::SigmoidSymmetric);
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
        assert!(high_perf.optimization.enable_parallel_processing);
        assert_eq!(high_perf.batch_size, 64);
        
        let low_mem = EmbedderConfig::new().low_memory();
        assert_eq!(low_mem.batch_size, 8);
        assert!(low_mem.optimization.cache_features);
    }
    
    #[test]
    fn test_constraint_optimization() {
        let config = EmbedderConfig::new()
            .optimize_for_constraints(Some(256), Some(1000), Some(50));
        
        assert!(config.cache_size <= 10000); // Should be reduced for low memory
        assert_eq!(config.batch_size, 1); // Should be 1 for low latency
    }
}