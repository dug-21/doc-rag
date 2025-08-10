//! Configuration management for the response generator system

// Removed unused import: use crate::cache::CacheManagerConfig;
use crate::error::{Result, ResponseError};
use crate::fact_accelerated::FACTConfig;
use crate::formatter::{FormatterConfig, OutputFormat};
use crate::query_preprocessing::QueryPreprocessingConfig;
use crate::validator::ValidationConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
// use toml;  // Available in Cargo.toml but may not be needed
use tokio::time::Duration;

/// Main configuration for the response generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Response generation settings
    pub generation: GenerationConfig,
    
    /// Validation system configuration
    pub validation: ValidationConfig,
    
    /// Formatter configuration
    pub formatter: FormatterConfig,
    
    /// Pipeline configuration
    pub pipeline_stages: Vec<String>,
    
    /// Performance settings
    pub performance: PerformanceConfig,
    
    /// Caching configuration
    pub cache: CacheConfig,
    
    /// FACT acceleration configuration
    pub fact: FACTConfig,
    
    /// Query preprocessing configuration
    pub query_preprocessing: QueryPreprocessingConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Maximum response length in characters
    pub max_response_length: usize,
    
    /// Default confidence threshold
    pub default_confidence_threshold: f64,
}

/// Configuration for response generation behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Default output format
    pub default_format: OutputFormat,
    
    /// Maximum number of context chunks to use
    pub max_context_chunks: usize,
    
    /// Minimum relevance score for context chunks
    pub min_relevance_score: f64,
    
    /// Enable response streaming
    pub enable_streaming: bool,
    
    /// Streaming chunk size
    pub stream_chunk_size: usize,
    
    /// Response quality settings
    pub quality: QualityConfig,
}

/// Quality-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityConfig {
    /// Target accuracy percentage (0.0-1.0)
    pub target_accuracy: f64,
    
    /// Enable content deduplication
    pub enable_deduplication: bool,
    
    /// Coherence threshold for content linking
    pub coherence_threshold: f64,
    
    /// Minimum content length (words)
    pub min_content_length: usize,
    
    /// Maximum content length (words)
    pub max_content_length: usize,
}

/// Performance-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum processing time per request
    pub max_processing_time: Duration,
    
    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,
    
    /// Enable request queuing
    pub enable_request_queue: bool,
    
    /// Queue size limit
    pub max_queue_size: usize,
    
    /// Timeout for individual operations
    pub operation_timeout: Duration,
    
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

/// Resource limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    
    /// Maximum CPU usage percentage
    pub max_cpu_percent: f64,
    
    /// Maximum disk usage in MB
    pub max_disk_mb: usize,
    
    /// Maximum network bandwidth in MB/s
    pub max_network_mbps: f64,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable response caching
    pub enable_response_cache: bool,
    
    /// Response cache size (number of entries)
    pub response_cache_size: usize,
    
    /// Response cache TTL
    pub response_cache_ttl: Duration,
    
    /// Enable context cache
    pub enable_context_cache: bool,
    
    /// Context cache size
    pub context_cache_size: usize,
    
    /// Context cache TTL
    pub context_cache_ttl: Duration,
    
    /// Cache storage backend
    pub cache_backend: CacheBackend,
}

/// Cache storage backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheBackend {
    /// In-memory cache
    Memory,
    
    /// Redis cache
    Redis { url: String },
    
    /// File-based cache
    File { directory: String },
    
    /// Custom cache implementation
    Custom { config: HashMap<String, String> },
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    
    /// Enable structured logging
    pub structured: bool,
    
    /// Log file path (optional)
    pub file_path: Option<String>,
    
    /// Maximum log file size in MB
    pub max_file_size_mb: usize,
    
    /// Number of log files to keep
    pub max_files: usize,
    
    /// Enable performance logging
    pub enable_performance_logs: bool,
    
    /// Enable metrics export
    pub enable_metrics_export: bool,
    
    /// Metrics export endpoint
    pub metrics_endpoint: Option<String>,
}

/// Configuration builder for fluent API
#[derive(Debug, Default)]
pub struct ConfigBuilder {
    generation: Option<GenerationConfig>,
    validation: Option<ValidationConfig>,
    formatter: Option<FormatterConfig>,
    pipeline_stages: Option<Vec<String>>,
    performance: Option<PerformanceConfig>,
    cache: Option<CacheConfig>,
    fact: Option<FACTConfig>,
    query_preprocessing: Option<QueryPreprocessingConfig>,
    logging: Option<LoggingConfig>,
    max_response_length: Option<usize>,
    default_confidence_threshold: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            generation: GenerationConfig::default(),
            validation: ValidationConfig::default(),
            formatter: FormatterConfig::default(),
            pipeline_stages: vec![
                "fact_query_preprocessing".to_string(),
                "context_preprocessing".to_string(),
                "content_generation".to_string(),
                "quality_enhancement".to_string(),
                "citation_processing".to_string(),
                "final_optimization".to_string(),
            ],
            performance: PerformanceConfig::default(),
            cache: CacheConfig::default(),
            fact: FACTConfig::default(),
            query_preprocessing: QueryPreprocessingConfig::default(),
            logging: LoggingConfig::default(),
            max_response_length: 4096,
            default_confidence_threshold: 0.7,
        }
    }
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            default_format: OutputFormat::Json,
            max_context_chunks: 10,
            min_relevance_score: 0.3,
            enable_streaming: true,
            stream_chunk_size: 256,
            quality: QualityConfig::default(),
        }
    }
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            target_accuracy: 0.99, // 99% accuracy target
            enable_deduplication: true,
            coherence_threshold: 0.7,
            min_content_length: 10,
            max_content_length: 1000,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_processing_time: Duration::from_millis(100),
            max_concurrent_requests: 100,
            enable_request_queue: true,
            max_queue_size: 1000,
            operation_timeout: Duration::from_millis(30),
            resource_limits: ResourceLimits::default(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024, // 1GB
            max_cpu_percent: 80.0,
            max_disk_mb: 10240, // 10GB
            max_network_mbps: 100.0,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enable_response_cache: true,
            response_cache_size: 10000,
            response_cache_ttl: Duration::from_secs(3600), // 1 hour
            enable_context_cache: true,
            context_cache_size: 50000,
            context_cache_ttl: Duration::from_secs(7200), // 2 hours
            cache_backend: CacheBackend::Memory,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            structured: true,
            file_path: None,
            max_file_size_mb: 100,
            max_files: 5,
            enable_performance_logs: true,
            enable_metrics_export: false,
            metrics_endpoint: None,
        }
    }
}

impl Config {
    /// Create a new configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration builder
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder::default()
    }

    /// Load configuration from file
    pub async fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let extension = path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("toml");
            
        let content = tokio::fs::read_to_string(path.as_ref()).await
            .map_err(|e| crate::error::ResponseError::config(format!("Failed to read config file: {}", e)))?;

        let result = match extension {
            "toml" => {
                // TOML parsing not implemented per design principle - use JSON or YAML instead
                Err(crate::error::ResponseError::config("TOML format not supported. Use JSON or YAML configuration files instead".to_string()))
            }
            "yaml" | "yml" => {
                serde_yaml::from_str(&content)
                    .map_err(|e| crate::error::ResponseError::config(format!("Failed to parse YAML config: {}", e)))
            }
            "json" => {
                serde_json::from_str(&content)
                    .map_err(|e| ResponseError::config(format!("Failed to parse JSON config: {}", e)))
            }
            _ => Err(ResponseError::config(format!("Unsupported config file format: {}", extension)))
        };
        
        result
    }

    /// Save configuration to file
    pub async fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let extension = path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("toml");

        let content = match extension {
            "toml" => "# TOML export not yet implemented".to_string(),
            "yaml" | "yml" => serde_yaml::to_string(self)
                .map_err(|e| ResponseError::config(format!("Failed to serialize YAML config: {}", e)))?,
            "json" => serde_json::to_string_pretty(self)
                .map_err(|e| ResponseError::config(format!("Failed to serialize JSON config: {}", e)))?,
            _ => return Err(ResponseError::config(format!("Unsupported config file format: {}", extension)))
        };

        tokio::fs::write(path, content).await
            .map_err(|e| ResponseError::config(format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Config::default();

        // Load basic settings from environment
        if let Ok(max_length) = std::env::var("RESPONSE_MAX_LENGTH") {
            config.max_response_length = max_length.parse()
                .map_err(|e| ResponseError::config(format!("Invalid RESPONSE_MAX_LENGTH: {}", e)))?;
        }

        if let Ok(confidence_threshold) = std::env::var("DEFAULT_CONFIDENCE_THRESHOLD") {
            config.default_confidence_threshold = confidence_threshold.parse()
                .map_err(|e| ResponseError::config(format!("Invalid DEFAULT_CONFIDENCE_THRESHOLD: {}", e)))?;
        }

        // Load performance settings
        if let Ok(max_time) = std::env::var("MAX_PROCESSING_TIME_MS") {
            let ms: u64 = max_time.parse()
                .map_err(|e| ResponseError::config(format!("Invalid MAX_PROCESSING_TIME_MS: {}", e)))?;
            config.performance.max_processing_time = Duration::from_millis(ms);
        }

        if let Ok(max_concurrent) = std::env::var("MAX_CONCURRENT_REQUESTS") {
            config.performance.max_concurrent_requests = max_concurrent.parse()
                .map_err(|e| ResponseError::config(format!("Invalid MAX_CONCURRENT_REQUESTS: {}", e)))?;
        }

        // Load cache settings
        if let Ok(enable_cache) = std::env::var("ENABLE_RESPONSE_CACHE") {
            config.cache.enable_response_cache = enable_cache.parse()
                .map_err(|e| ResponseError::config(format!("Invalid ENABLE_RESPONSE_CACHE: {}", e)))?;
        }

        if let Ok(cache_size) = std::env::var("RESPONSE_CACHE_SIZE") {
            config.cache.response_cache_size = cache_size.parse()
                .map_err(|e| ResponseError::config(format!("Invalid RESPONSE_CACHE_SIZE: {}", e)))?;
        }

        // Load Redis cache URL if configured
        if let Ok(redis_url) = std::env::var("REDIS_CACHE_URL") {
            config.cache.cache_backend = CacheBackend::Redis { url: redis_url };
        }

        // Load logging settings
        if let Ok(log_level) = std::env::var("LOG_LEVEL") {
            config.logging.level = log_level;
        }

        if let Ok(log_file) = std::env::var("LOG_FILE_PATH") {
            config.logging.file_path = Some(log_file);
        }

        if let Ok(metrics_endpoint) = std::env::var("METRICS_ENDPOINT") {
            config.logging.metrics_endpoint = Some(metrics_endpoint);
            config.logging.enable_metrics_export = true;
        }

        Ok(config)
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<()> {
        // Validate response length
        if self.max_response_length == 0 {
            return Err(ResponseError::config("max_response_length must be greater than 0"));
        }

        // Validate confidence threshold
        if self.default_confidence_threshold < 0.0 || self.default_confidence_threshold > 1.0 {
            return Err(ResponseError::config("default_confidence_threshold must be between 0.0 and 1.0"));
        }

        // Validate performance settings
        if self.performance.max_concurrent_requests == 0 {
            return Err(ResponseError::config("max_concurrent_requests must be greater than 0"));
        }

        if self.performance.max_queue_size == 0 && self.performance.enable_request_queue {
            return Err(ResponseError::config("max_queue_size must be greater than 0 when request queue is enabled"));
        }

        // Validate cache settings
        if self.cache.response_cache_size == 0 && self.cache.enable_response_cache {
            return Err(ResponseError::config("response_cache_size must be greater than 0 when caching is enabled"));
        }

        // Validate quality settings
        if self.generation.quality.target_accuracy < 0.0 || self.generation.quality.target_accuracy > 1.0 {
            return Err(ResponseError::config("target_accuracy must be between 0.0 and 1.0"));
        }

        if self.generation.quality.min_content_length > self.generation.quality.max_content_length {
            return Err(ResponseError::config("min_content_length must be less than or equal to max_content_length"));
        }

        // Validate resource limits
        if self.performance.resource_limits.max_cpu_percent < 0.0 || self.performance.resource_limits.max_cpu_percent > 100.0 {
            return Err(ResponseError::config("max_cpu_percent must be between 0.0 and 100.0"));
        }

        Ok(())
    }

    /// Merge with another configuration (other takes precedence)
    pub fn merge(mut self, other: Config) -> Self {
        // Merge primitive values
        self.max_response_length = other.max_response_length;
        self.default_confidence_threshold = other.default_confidence_threshold;
        
        // Merge complex structures
        self.generation = other.generation;
        self.validation = other.validation;
        self.formatter = other.formatter;
        self.pipeline_stages = other.pipeline_stages;
        self.performance = other.performance;
        self.cache = other.cache;
        self.fact = other.fact;
        self.query_preprocessing = other.query_preprocessing;
        self.logging = other.logging;
        
        self
    }
}

impl ConfigBuilder {
    /// Set generation configuration
    pub fn generation(mut self, generation: GenerationConfig) -> Self {
        self.generation = Some(generation);
        self
    }

    /// Set validation configuration
    pub fn validation(mut self, validation: ValidationConfig) -> Self {
        self.validation = Some(validation);
        self
    }

    /// Set formatter configuration
    pub fn formatter(mut self, formatter: FormatterConfig) -> Self {
        self.formatter = Some(formatter);
        self
    }

    /// Set pipeline stages
    pub fn pipeline_stages(mut self, stages: Vec<String>) -> Self {
        self.pipeline_stages = Some(stages);
        self
    }

    /// Set performance configuration
    pub fn performance(mut self, performance: PerformanceConfig) -> Self {
        self.performance = Some(performance);
        self
    }

    /// Set cache configuration
    pub fn cache(mut self, cache: CacheConfig) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Set logging configuration
    pub fn logging(mut self, logging: LoggingConfig) -> Self {
        self.logging = Some(logging);
        self
    }

    /// Set maximum response length
    pub fn max_response_length(mut self, length: usize) -> Self {
        self.max_response_length = Some(length);
        self
    }

    /// Set default confidence threshold
    pub fn default_confidence_threshold(mut self, threshold: f64) -> Self {
        self.default_confidence_threshold = Some(threshold);
        self
    }

    /// Build the configuration
    pub fn build(self) -> Config {
        let default_config = Config::default();
        
        Config {
            generation: self.generation.unwrap_or(default_config.generation),
            validation: self.validation.unwrap_or(default_config.validation),
            formatter: self.formatter.unwrap_or(default_config.formatter),
            pipeline_stages: self.pipeline_stages.unwrap_or(default_config.pipeline_stages),
            performance: self.performance.unwrap_or(default_config.performance),
            cache: self.cache.unwrap_or(default_config.cache),
            fact: self.fact.unwrap_or(default_config.fact),
            query_preprocessing: self.query_preprocessing.unwrap_or(default_config.query_preprocessing),
            logging: self.logging.unwrap_or(default_config.logging),
            max_response_length: self.max_response_length.unwrap_or(default_config.max_response_length),
            default_confidence_threshold: self.default_confidence_threshold.unwrap_or(default_config.default_confidence_threshold),
        }
    }
}

// Serde support for Duration in configuration files
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_millis().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.max_response_length, 4096);
        assert_eq!(config.default_confidence_threshold, 0.7);
    }

    #[test]
    fn test_config_builder() {
        let config = Config::builder()
            .max_response_length(2048)
            .default_confidence_threshold(0.8)
            .build();

        assert_eq!(config.max_response_length, 2048);
        assert_eq!(config.default_confidence_threshold, 0.8);
    }

    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        config.max_response_length = 0;
        
        let result = config.validate();
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_config_file_operations() {
        let config = Config::default();
        let temp_file = NamedTempFile::new().unwrap();
        
        // Change extension to JSON to avoid TOML error
        let json_path = temp_file.path().with_extension("json");
        
        // Test saving
        config.save_to_file(&json_path).await.unwrap();
        
        // Test loading
        let loaded_config = Config::from_file(&json_path).await.unwrap();
        assert_eq!(config.max_response_length, loaded_config.max_response_length);
        
        // Cleanup
        std::fs::remove_file(&json_path).ok();
    }

    #[test]
    fn test_config_merge() {
        let config1 = Config::builder()
            .max_response_length(1000)
            .build();

        let config2 = Config::builder()
            .max_response_length(2000)
            .default_confidence_threshold(0.9)
            .build();

        let merged = config1.merge(config2);
        assert_eq!(merged.max_response_length, 2000);
        assert_eq!(merged.default_confidence_threshold, 0.9);
    }

    #[test]
    fn test_env_config_loading() {
        std::env::set_var("RESPONSE_MAX_LENGTH", "8192");
        std::env::set_var("DEFAULT_CONFIDENCE_THRESHOLD", "0.85");
        
        let config = Config::from_env().unwrap();
        assert_eq!(config.max_response_length, 8192);
        assert_eq!(config.default_confidence_threshold, 0.85);
        
        // Cleanup
        std::env::remove_var("RESPONSE_MAX_LENGTH");
        std::env::remove_var("DEFAULT_CONFIDENCE_THRESHOLD");
    }
}