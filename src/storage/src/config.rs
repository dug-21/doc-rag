//! Configuration management for MongoDB Vector Storage

use std::time::Duration;
use serde::{Deserialize, Serialize};
use config::{Config, ConfigError, Environment, File};
use std::path::Path;

/// Storage configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// MongoDB connection string
    pub connection_string: String,
    
    /// Database name
    pub database_name: String,
    
    /// Collection name for chunks
    pub chunk_collection_name: String,
    
    /// Collection name for metadata
    pub metadata_collection_name: String,
    
    /// Maximum connection pool size
    pub max_pool_size: u32,
    
    /// Minimum connection pool size
    pub min_pool_size: u32,
    
    /// Connection timeout in seconds
    pub connection_timeout_secs: u64,
    
    /// Operation timeout in seconds
    pub operation_timeout_secs: u64,
    
    /// Socket timeout in seconds
    pub socket_timeout_secs: u64,
    
    /// Server selection timeout in seconds
    pub server_selection_timeout_secs: u64,
    
    /// Enable SSL/TLS
    pub enable_tls: bool,
    
    /// TLS certificate path
    pub tls_cert_path: Option<String>,
    
    /// TLS key path
    pub tls_key_path: Option<String>,
    
    /// CA certificate path
    pub ca_cert_path: Option<String>,
    
    /// Allow invalid certificates (for development)
    pub allow_invalid_certs: bool,
    
    /// Vector search configuration
    pub vector_search: VectorSearchConfig,
    
    /// Text search configuration
    pub text_search: TextSearchConfig,
    
    /// Performance tuning settings
    pub performance: PerformanceConfig,
    
    /// Monitoring and metrics settings
    pub monitoring: MonitoringConfig,
}

/// Vector search specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSearchConfig {
    /// Default embedding dimension
    pub embedding_dimension: usize,
    
    /// Default similarity metric
    pub similarity_metric: SimilarityMetric,
    
    /// Number of candidates to examine for KNN search
    pub num_candidates: usize,
    
    /// Minimum similarity score threshold
    pub min_similarity_threshold: f32,
    
    /// Maximum results per search
    pub max_results: usize,
    
    /// Enable approximate nearest neighbor search
    pub enable_approximate_search: bool,
    
    /// Quantization settings for vector compression
    pub quantization: Option<QuantizationConfig>,
}

/// Similarity metrics for vector comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimilarityMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

/// Vector quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization method
    pub method: QuantizationMethod,
    
    /// Number of bits per dimension
    pub bits_per_dimension: u8,
    
    /// Enable dynamic quantization
    pub dynamic: bool,
}

/// Quantization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantizationMethod {
    Uniform,
    ProductQuantization,
    ScalarQuantization,
}

/// Text search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSearchConfig {
    /// Default text search language
    pub default_language: String,
    
    /// Enable fuzzy matching
    pub enable_fuzzy_search: bool,
    
    /// Fuzzy search edit distance
    pub fuzzy_edit_distance: u32,
    
    /// Enable stemming
    pub enable_stemming: bool,
    
    /// Stop words to exclude
    pub stop_words: Vec<String>,
    
    /// Minimum word length for indexing
    pub min_word_length: usize,
    
    /// Maximum phrase length
    pub max_phrase_length: usize,
    
    /// Text index weights
    pub field_weights: TextFieldWeights,
}

/// Field weights for text search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFieldWeights {
    pub content: f32,
    pub title: f32,
    pub tags: f32,
    pub metadata: f32,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Batch size for bulk operations
    pub batch_size: usize,
    
    /// Maximum concurrent operations
    pub max_concurrency: usize,
    
    /// Enable connection pooling
    pub enable_connection_pooling: bool,
    
    /// Enable query result caching
    pub enable_query_caching: bool,
    
    /// Query cache TTL in seconds
    pub query_cache_ttl_secs: u64,
    
    /// Maximum query cache size (number of entries)
    pub query_cache_max_size: usize,
    
    /// Enable compression for data transfer
    pub enable_compression: bool,
    
    /// Compression level (1-9)
    pub compression_level: u8,
    
    /// Read preference for queries
    pub read_preference: ReadPreference,
    
    /// Write concern level
    pub write_concern: WriteConcern,
}

/// MongoDB read preference
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadPreference {
    Primary,
    PrimaryPreferred,
    Secondary,
    SecondaryPreferred,
    Nearest,
}

/// MongoDB write concern
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteConcern {
    Acknowledged,
    Unacknowledged,
    Journaled,
    Majority,
    Custom(String),
}

/// Monitoring and metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable performance metrics collection
    pub enable_metrics: bool,
    
    /// Metrics collection interval in seconds
    pub metrics_interval_secs: u64,
    
    /// Enable distributed tracing
    pub enable_tracing: bool,
    
    /// Tracing service endpoint
    pub tracing_endpoint: Option<String>,
    
    /// Enable health checks
    pub enable_health_checks: bool,
    
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,
    
    /// Log level
    pub log_level: LogLevel,
    
    /// Enable structured logging
    pub enable_structured_logging: bool,
    
    /// Log output format
    pub log_format: LogFormat,
}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Log output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Plain,
    Json,
    Compact,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            connection_string: "mongodb://localhost:27017".to_string(),
            database_name: "rag_storage".to_string(),
            chunk_collection_name: "chunks".to_string(),
            metadata_collection_name: "metadata".to_string(),
            max_pool_size: 10,
            min_pool_size: 1,
            connection_timeout_secs: 10,
            operation_timeout_secs: 30,
            socket_timeout_secs: 30,
            server_selection_timeout_secs: 10,
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
            ca_cert_path: None,
            allow_invalid_certs: false,
            vector_search: VectorSearchConfig::default(),
            text_search: TextSearchConfig::default(),
            performance: PerformanceConfig::default(),
            monitoring: MonitoringConfig::default(),
        }
    }
}

impl Default for VectorSearchConfig {
    fn default() -> Self {
        Self {
            embedding_dimension: 384, // Common dimension for sentence transformers
            similarity_metric: SimilarityMetric::Cosine,
            num_candidates: 100,
            min_similarity_threshold: 0.1,
            max_results: 100,
            enable_approximate_search: true,
            quantization: None,
        }
    }
}

impl Default for TextSearchConfig {
    fn default() -> Self {
        Self {
            default_language: "en".to_string(),
            enable_fuzzy_search: true,
            fuzzy_edit_distance: 2,
            enable_stemming: true,
            stop_words: vec![
                "the".to_string(), "a".to_string(), "an".to_string(),
                "and".to_string(), "or".to_string(), "but".to_string(),
                "in".to_string(), "on".to_string(), "at".to_string(),
                "to".to_string(), "for".to_string(), "of".to_string(),
                "with".to_string(), "by".to_string(), "is".to_string(),
                "are".to_string(), "was".to_string(), "were".to_string(),
            ],
            min_word_length: 2,
            max_phrase_length: 5,
            field_weights: TextFieldWeights::default(),
        }
    }
}

impl Default for TextFieldWeights {
    fn default() -> Self {
        Self {
            content: 10.0,
            title: 5.0,
            tags: 3.0,
            metadata: 1.0,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_concurrency: 10,
            enable_connection_pooling: true,
            enable_query_caching: true,
            query_cache_ttl_secs: 300, // 5 minutes
            query_cache_max_size: 1000,
            enable_compression: true,
            compression_level: 6,
            read_preference: ReadPreference::PrimaryPreferred,
            write_concern: WriteConcern::Acknowledged,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval_secs: 30,
            enable_tracing: true,
            tracing_endpoint: None,
            enable_health_checks: true,
            health_check_interval_secs: 60,
            log_level: LogLevel::Info,
            enable_structured_logging: true,
            log_format: LogFormat::Json,
        }
    }
}

impl StorageConfig {
    /// Load configuration from file and environment
    pub fn load() -> Result<Self, ConfigError> {
        let mut config_builder = Config::builder();
        
        // Start with defaults
        config_builder = config_builder
            .add_source(Config::try_from(&StorageConfig::default())?);
        
        // Add configuration files
        if Path::new("storage.toml").exists() {
            config_builder = config_builder
                .add_source(File::with_name("storage").format(config::FileFormat::Toml));
        }
        
        if Path::new("storage.yaml").exists() {
            config_builder = config_builder
                .add_source(File::with_name("storage").format(config::FileFormat::Yaml));
        }
        
        // Add environment variables with STORAGE_ prefix
        config_builder = config_builder
            .add_source(Environment::with_prefix("STORAGE").separator("__"));
        
        let config = config_builder.build()?;
        config.try_deserialize()
    }
    
    /// Load configuration from a specific file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let format = match path.extension().and_then(|ext| ext.to_str()) {
            Some("toml") => config::FileFormat::Toml,
            Some("yaml") | Some("yml") => config::FileFormat::Yaml,
            Some("json") => config::FileFormat::Json,
            _ => return Err(ConfigError::Message("Unsupported file format".to_string())),
        };
        
        Config::builder()
            .add_source(Config::try_from(&StorageConfig::default())?)
            .add_source(File::from(path).format(format))
            .build()?
            .try_deserialize()
    }
    
    /// Create configuration for testing
    pub fn for_testing() -> Self {
        Self {
            connection_string: "mongodb://localhost:27017".to_string(),
            database_name: format!("test_rag_storage_{}", uuid::Uuid::new_v4().simple()),
            chunk_collection_name: "test_chunks".to_string(),
            metadata_collection_name: "test_metadata".to_string(),
            connection_timeout_secs: 5,
            operation_timeout_secs: 10,
            monitoring: MonitoringConfig {
                log_level: LogLevel::Debug,
                ..Default::default()
            },
            ..Default::default()
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate connection string
        if self.connection_string.is_empty() {
            return Err(ConfigError::Message("Connection string cannot be empty".to_string()));
        }
        
        if !self.connection_string.starts_with("mongodb://") && !self.connection_string.starts_with("mongodb+srv://") {
            return Err(ConfigError::Message("Invalid MongoDB connection string format".to_string()));
        }
        
        // Validate database and collection names
        if self.database_name.is_empty() {
            return Err(ConfigError::Message("Database name cannot be empty".to_string()));
        }
        
        if self.chunk_collection_name.is_empty() {
            return Err(ConfigError::Message("Chunk collection name cannot be empty".to_string()));
        }
        
        if self.metadata_collection_name.is_empty() {
            return Err(ConfigError::Message("Metadata collection name cannot be empty".to_string()));
        }
        
        // Validate pool sizes
        if self.min_pool_size > self.max_pool_size {
            return Err(ConfigError::Message("Min pool size cannot be greater than max pool size".to_string()));
        }
        
        if self.max_pool_size == 0 {
            return Err(ConfigError::Message("Max pool size must be greater than 0".to_string()));
        }
        
        // Validate timeouts
        if self.connection_timeout_secs == 0 {
            return Err(ConfigError::Message("Connection timeout must be greater than 0".to_string()));
        }
        
        if self.operation_timeout_secs == 0 {
            return Err(ConfigError::Message("Operation timeout must be greater than 0".to_string()));
        }
        
        // Validate vector search config
        if self.vector_search.embedding_dimension == 0 {
            return Err(ConfigError::Message("Embedding dimension must be greater than 0".to_string()));
        }
        
        if self.vector_search.num_candidates == 0 {
            return Err(ConfigError::Message("Number of candidates must be greater than 0".to_string()));
        }
        
        if self.vector_search.max_results == 0 {
            return Err(ConfigError::Message("Max results must be greater than 0".to_string()));
        }
        
        // Validate performance config
        if self.performance.batch_size == 0 {
            return Err(ConfigError::Message("Batch size must be greater than 0".to_string()));
        }
        
        if self.performance.max_concurrency == 0 {
            return Err(ConfigError::Message("Max concurrency must be greater than 0".to_string()));
        }
        
        if self.performance.compression_level > 9 {
            return Err(ConfigError::Message("Compression level must be between 1 and 9".to_string()));
        }
        
        // Validate TLS configuration
        if self.enable_tls {
            if let Some(ref cert_path) = self.tls_cert_path {
                if !Path::new(cert_path).exists() {
                    return Err(ConfigError::Message(format!("TLS certificate file not found: {}", cert_path)));
                }
            }
            
            if let Some(ref key_path) = self.tls_key_path {
                if !Path::new(key_path).exists() {
                    return Err(ConfigError::Message(format!("TLS key file not found: {}", key_path)));
                }
            }
            
            if let Some(ref ca_path) = self.ca_cert_path {
                if !Path::new(ca_path).exists() {
                    return Err(ConfigError::Message(format!("CA certificate file not found: {}", ca_path)));
                }
            }
        }
        
        Ok(())
    }
    
    /// Get connection timeout as Duration
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_secs(self.connection_timeout_secs)
    }
    
    /// Get operation timeout as Duration
    pub fn operation_timeout(&self) -> Duration {
        Duration::from_secs(self.operation_timeout_secs)
    }
    
    /// Get socket timeout as Duration
    pub fn socket_timeout(&self) -> Duration {
        Duration::from_secs(self.socket_timeout_secs)
    }
    
    /// Get server selection timeout as Duration
    pub fn server_selection_timeout(&self) -> Duration {
        Duration::from_secs(self.server_selection_timeout_secs)
    }
    
    /// Get health check interval as Duration
    pub fn health_check_interval(&self) -> Duration {
        Duration::from_secs(self.monitoring.health_check_interval_secs)
    }
    
    /// Get metrics collection interval as Duration
    pub fn metrics_interval(&self) -> Duration {
        Duration::from_secs(self.monitoring.metrics_interval_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_default_config() {
        let config = StorageConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.database_name, "rag_storage");
        assert_eq!(config.chunk_collection_name, "chunks");
        assert_eq!(config.max_pool_size, 10);
        assert_eq!(config.vector_search.embedding_dimension, 384);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = StorageConfig::default();
        
        // Test empty database name
        config.database_name = String::new();
        assert!(config.validate().is_err());
        
        // Test invalid pool sizes
        config = StorageConfig::default();
        config.min_pool_size = 10;
        config.max_pool_size = 5;
        assert!(config.validate().is_err());
        
        // Test zero embedding dimension
        config = StorageConfig::default();
        config.vector_search.embedding_dimension = 0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_from_file() {
        let config_content = r#"
        connection_string = "mongodb://test:27017"
        database_name = "test_db"
        
        [vector_search]
        embedding_dimension = 512
        similarity_metric = "euclidean"
        
        [performance]
        batch_size = 200
        "#;
        
        let mut temp_file = NamedTempFile::new().unwrap();
        fs::write(temp_file.path(), config_content).unwrap();
        
        // Rename to have .toml extension
        let toml_path = temp_file.path().with_extension("toml");
        fs::rename(temp_file.path(), &toml_path).unwrap();
        
        let config = StorageConfig::load_from_file(&toml_path).unwrap();
        assert_eq!(config.connection_string, "mongodb://test:27017");
        assert_eq!(config.database_name, "test_db");
        assert_eq!(config.vector_search.embedding_dimension, 512);
        assert_eq!(config.performance.batch_size, 200);
        
        // Cleanup
        fs::remove_file(toml_path).ok();
    }
    
    #[test]
    fn test_testing_config() {
        let config = StorageConfig::for_testing();
        assert!(config.validate().is_ok());
        assert!(config.database_name.starts_with("test_rag_storage_"));
        assert_eq!(config.chunk_collection_name, "test_chunks");
        assert_eq!(config.connection_timeout_secs, 5);
    }
    
    #[test]
    fn test_duration_methods() {
        let config = StorageConfig::default();
        assert_eq!(config.connection_timeout(), Duration::from_secs(10));
        assert_eq!(config.operation_timeout(), Duration::from_secs(30));
        assert_eq!(config.health_check_interval(), Duration::from_secs(60));
    }
}