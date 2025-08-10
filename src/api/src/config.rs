use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokio::fs;
use url::Url;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub server: ServerConfig,
    pub database: DatabaseConfig,
    pub redis: RedisConfig,
    pub security: SecurityConfig,
    pub components: ComponentsConfig,
    pub observability: ObservabilityConfig,
    pub features: FeatureConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub bind_address: String,
    pub port: u16,
    pub worker_threads: usize,
    pub max_connections: usize,
    pub request_timeout_secs: u64,
    pub idle_timeout_secs: u64,
    pub max_request_size: usize,
    pub cors_origins: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connect_timeout_secs: u64,
    pub idle_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub max_connections: u32,
    pub connection_timeout_secs: u64,
    pub command_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub jwt_secret: String,
    pub jwt_expiration_hours: u64,
    pub password_min_length: usize,
    pub rate_limit_requests: u32,
    pub rate_limit_window_secs: u64,
    pub session_timeout_hours: u64,
    pub enable_cors: bool,
    pub enable_csrf: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentsConfig {
    pub chunker: ServiceConfig,
    pub embedder: ServiceConfig,
    pub storage: ServiceConfig,
    pub retriever: ServiceConfig,
    pub query_processor: ServiceConfig,
    pub response_generator: ServiceConfig,
    pub mcp_adapter: ServiceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub url: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub health_check_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub metrics_enabled: bool,
    pub metrics_port: u16,
    pub tracing_enabled: bool,
    pub jaeger_endpoint: Option<String>,
    pub log_level: String,
    pub request_logging: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub enable_streaming: bool,
    pub enable_batch_processing: bool,
    pub enable_websockets: bool,
    pub enable_file_uploads: bool,
    pub max_file_size_mb: usize,
    pub supported_formats: Vec<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                bind_address: "0.0.0.0".to_string(),
                port: 8080,
                worker_threads: num_cpus::get(),
                max_connections: 1000,
                request_timeout_secs: 30,
                idle_timeout_secs: 60,
                max_request_size: 16 * 1024 * 1024, // 16MB
                cors_origins: vec!["*".to_string()],
            },
            database: DatabaseConfig {
                url: "postgres://docrag:docrag_secret@localhost:5432/docrag".to_string(),
                max_connections: 100,
                min_connections: 10,
                connect_timeout_secs: 30,
                idle_timeout_secs: 600,
            },
            redis: RedisConfig {
                url: "redis://localhost:6379".to_string(),
                max_connections: 50,
                connection_timeout_secs: 5,
                command_timeout_secs: 10,
            },
            security: SecurityConfig {
                jwt_secret: "your-secret-key-change-in-production".to_string(),
                jwt_expiration_hours: 24,
                password_min_length: 8,
                rate_limit_requests: 100,
                rate_limit_window_secs: 60,
                session_timeout_hours: 24,
                enable_cors: true,
                enable_csrf: true,
            },
            components: ComponentsConfig {
                chunker: ServiceConfig {
                    url: "http://chunker:8080".to_string(),
                    timeout_secs: 30,
                    max_retries: 3,
                    retry_delay_ms: 1000,
                    health_check_interval_secs: 30,
                },
                embedder: ServiceConfig {
                    url: "http://embedder:8082".to_string(),
                    timeout_secs: 60,
                    max_retries: 3,
                    retry_delay_ms: 1000,
                    health_check_interval_secs: 30,
                },
                storage: ServiceConfig {
                    url: "http://storage:8083".to_string(),
                    timeout_secs: 30,
                    max_retries: 3,
                    retry_delay_ms: 1000,
                    health_check_interval_secs: 30,
                },
                retriever: ServiceConfig {
                    url: "http://retriever:8084".to_string(),
                    timeout_secs: 30,
                    max_retries: 3,
                    retry_delay_ms: 1000,
                    health_check_interval_secs: 30,
                },
                query_processor: ServiceConfig {
                    url: "http://query-processor:8084".to_string(),
                    timeout_secs: 30,
                    max_retries: 3,
                    retry_delay_ms: 1000,
                    health_check_interval_secs: 30,
                },
                response_generator: ServiceConfig {
                    url: "http://response-generator:8085".to_string(),
                    timeout_secs: 60,
                    max_retries: 3,
                    retry_delay_ms: 1000,
                    health_check_interval_secs: 30,
                },
                mcp_adapter: ServiceConfig {
                    url: "http://mcp-adapter:8081".to_string(),
                    timeout_secs: 30,
                    max_retries: 3,
                    retry_delay_ms: 1000,
                    health_check_interval_secs: 30,
                },
            },
            observability: ObservabilityConfig {
                metrics_enabled: true,
                metrics_port: 9090,
                tracing_enabled: true,
                jaeger_endpoint: Some("http://jaeger:14268/api/traces".to_string()),
                log_level: "info".to_string(),
                request_logging: true,
            },
            features: FeatureConfig {
                enable_streaming: true,
                enable_batch_processing: true,
                enable_websockets: true,
                enable_file_uploads: true,
                max_file_size_mb: 100,
                supported_formats: vec![
                    "application/pdf".to_string(),
                    "text/plain".to_string(),
                    "text/markdown".to_string(),
                    "application/json".to_string(),
                    "text/html".to_string(),
                ],
            },
        }
    }
}

impl ApiConfig {
    /// Load configuration from file
    pub async fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        
        let config: ApiConfig = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;
        
        config.validate()?;
        Ok(config)
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();

        // Override with environment variables
        if let Ok(db_url) = std::env::var("DATABASE_URL") {
            config.database.url = db_url;
        }
        
        if let Ok(redis_url) = std::env::var("REDIS_URL") {
            config.redis.url = redis_url;
        }

        if let Ok(jwt_secret) = std::env::var("JWT_SECRET") {
            config.security.jwt_secret = jwt_secret;
        }

        if let Ok(bind) = std::env::var("BIND_ADDRESS") {
            config.server.bind_address = bind;
        }

        if let Ok(port) = std::env::var("PORT") {
            config.server.port = port.parse()
                .with_context(|| "Invalid PORT environment variable")?;
        }

        if let Ok(log_level) = std::env::var("RUST_LOG") {
            config.observability.log_level = log_level;
        }

        // Component URLs from environment
        if let Ok(chunker_url) = std::env::var("CHUNKER_URL") {
            config.components.chunker.url = chunker_url;
        }

        if let Ok(embedder_url) = std::env::var("EMBEDDER_URL") {
            config.components.embedder.url = embedder_url;
        }

        if let Ok(storage_url) = std::env::var("STORAGE_URL") {
            config.components.storage.url = storage_url;
        }

        if let Ok(retriever_url) = std::env::var("RETRIEVER_URL") {
            config.components.retriever.url = retriever_url;
        }

        if let Ok(query_processor_url) = std::env::var("QUERY_PROCESSOR_URL") {
            config.components.query_processor.url = query_processor_url;
        }

        if let Ok(response_generator_url) = std::env::var("RESPONSE_GENERATOR_URL") {
            config.components.response_generator.url = response_generator_url;
        }

        if let Ok(mcp_adapter_url) = std::env::var("MCP_ADAPTER_URL") {
            config.components.mcp_adapter.url = mcp_adapter_url;
        }

        config.validate()?;
        Ok(config)
    }

    /// Enable development-specific features
    pub fn enable_development_features(&mut self) {
        self.observability.log_level = "debug".to_string();
        self.observability.request_logging = true;
        self.security.enable_cors = true;
        self.server.cors_origins = vec!["*".to_string()];
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate URLs
        for (name, url) in [
            ("database", &self.database.url),
            ("redis", &self.redis.url),
            ("chunker", &self.components.chunker.url),
            ("embedder", &self.components.embedder.url),
            ("storage", &self.components.storage.url),
            ("retriever", &self.components.retriever.url),
            ("query_processor", &self.components.query_processor.url),
            ("response_generator", &self.components.response_generator.url),
            ("mcp_adapter", &self.components.mcp_adapter.url),
        ] {
            Url::parse(url)
                .with_context(|| format!("Invalid {} URL: {}", name, url))?;
        }

        // Validate JWT secret
        if self.security.jwt_secret.len() < 32 {
            anyhow::bail!("JWT secret must be at least 32 characters long");
        }

        // Validate port ranges
        if self.server.port == 0 {
            anyhow::bail!("Server port must be greater than 0");
        }

        if self.observability.metrics_port == 0 {
            anyhow::bail!("Metrics port must be greater than 0");
        }

        // Validate timeouts
        if self.server.request_timeout_secs == 0 {
            anyhow::bail!("Request timeout must be greater than 0");
        }

        Ok(())
    }

    /// Get server socket address
    pub fn server_addr(&self) -> String {
        format!("{}:{}", self.server.bind_address, self.server.port)
    }

    /// Get metrics socket address
    pub fn metrics_addr(&self) -> String {
        format!("{}:{}", self.server.bind_address, self.observability.metrics_port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::fs::write;

    #[tokio::test]
    async fn test_config_default() {
        let config = ApiConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.server.bind_address, "0.0.0.0");
    }

    #[tokio::test]
    async fn test_config_from_file() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        
        let config_content = r#"
[server]
bind_address = "127.0.0.1"
port = 3000
worker_threads = 4
max_connections = 500
request_timeout_secs = 60
idle_timeout_secs = 120
max_request_size = 33554432
cors_origins = ["http://localhost:3000"]

[database]
url = "postgres://test:test@localhost:5432/test"
max_connections = 50
min_connections = 5
connect_timeout_secs = 30
idle_timeout_secs = 600

[redis]
url = "redis://localhost:6379"
max_connections = 25
connection_timeout_secs = 5
command_timeout_secs = 10

[security]
jwt_secret = "super-secret-key-for-testing-purposes-only"
jwt_expiration_hours = 48
password_min_length = 10
rate_limit_requests = 200
rate_limit_window_secs = 120
session_timeout_hours = 48
enable_cors = false
enable_csrf = false

[components.chunker]
url = "http://localhost:8080"
timeout_secs = 30
max_retries = 3
retry_delay_ms = 1000
health_check_interval_secs = 30

[components.embedder]
url = "http://localhost:8082"
timeout_secs = 60
max_retries = 3
retry_delay_ms = 1000
health_check_interval_secs = 30

[components.storage]
url = "http://localhost:8083"
timeout_secs = 30
max_retries = 3
retry_delay_ms = 1000
health_check_interval_secs = 30

[components.retriever]
url = "http://localhost:8084"
timeout_secs = 30
max_retries = 3
retry_delay_ms = 1000
health_check_interval_secs = 30

[components.query_processor]
url = "http://localhost:8085"
timeout_secs = 30
max_retries = 3
retry_delay_ms = 1000
health_check_interval_secs = 30

[components.response_generator]
url = "http://localhost:8086"
timeout_secs = 60
max_retries = 3
retry_delay_ms = 1000
health_check_interval_secs = 30

[components.mcp_adapter]
url = "http://localhost:8081"
timeout_secs = 30
max_retries = 3
retry_delay_ms = 1000
health_check_interval_secs = 30

[observability]
metrics_enabled = true
metrics_port = 9090
tracing_enabled = true
jaeger_endpoint = "http://localhost:14268/api/traces"
log_level = "debug"
request_logging = true

[features]
enable_streaming = true
enable_batch_processing = true
enable_websockets = true
enable_file_uploads = true
max_file_size_mb = 200
supported_formats = ["application/pdf", "text/plain"]
"#;

        write(&config_path, config_content).await.unwrap();
        
        let config = ApiConfig::from_file(&config_path).await.unwrap();
        assert_eq!(config.server.port, 3000);
        assert_eq!(config.server.bind_address, "127.0.0.1");
        assert_eq!(config.database.max_connections, 50);
    }

    #[test]
    fn test_config_validation() {
        let mut config = ApiConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid JWT secret
        config.security.jwt_secret = "short".to_string();
        assert!(config.validate().is_err());

        // Reset and test invalid URL
        config = ApiConfig::default();
        config.database.url = "invalid-url".to_string();
        assert!(config.validate().is_err());

        // Reset and test invalid port
        config = ApiConfig::default();
        config.server.port = 0;
        assert!(config.validate().is_err());
    }
}