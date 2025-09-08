//! Test Module for FACT Integration
//! 
//! Comprehensive test suite organization for FACT integration testing
//! including integration tests, performance tests, and mock implementations.

pub mod fact_integration_tests;
pub mod fact_performance_tests;
pub mod mocks;

// Re-export commonly used test utilities
pub use mocks::mock_fact_client::{
    MockFACTClient, FACTTestFixtures, RealisticFACTSimulator, 
    SimulatorConfig, MockFACTClientFactory
};

pub use fact_integration_tests::{
    FACTTestConfig, MockResponseBuilder, Citation, 
    FACTError, RetryPolicy, PerformanceMetrics
};

pub use fact_performance_tests::{
    PerformanceSummary, LoadTestConfig
};

/// Test utilities and helper functions
pub mod test_utils {
    use super::*;
    use std::time::Duration;
    use std::collections::HashMap;
    use serde_json::{json, Value};
    
    /// Create a standard test configuration for FACT testing
    pub fn create_standard_test_config() -> FACTTestConfig {
        FACTTestConfig {
            cache_ttl: Duration::from_secs(1800),
            retry_attempts: 3,
            timeout: Duration::from_secs(5),
            consensus_threshold: 0.66,
        }
    }
    
    /// Generate test data for performance benchmarks
    pub fn generate_benchmark_queries(count: usize) -> Vec<String> {
        (0..count).map(|i| {
            match i % 5 {
                0 => format!("pci_dss_requirement_{}", i % 12),
                1 => format!("gdpr_article_{}", i % 99),
                2 => format!("iso_27001_control_{}", i % 114),
                3 => format!("sox_section_{}", i % 11),
                _ => format!("compliance_query_{}", i),
            }
        }).collect()
    }
    
    /// Create mock response data with realistic compliance content
    pub fn create_compliance_response(query: &str) -> Value {
        let content = match query {
            q if q.contains("pci_dss") => json!({
                "regulation": "PCI DSS 4.0",
                "requirement": extract_requirement_number(q),
                "description": "Cardholder data protection requirements",
                "compliance_level": "mandatory",
                "last_updated": "2024-03-15"
            }),
            q if q.contains("gdpr") => json!({
                "regulation": "GDPR",
                "article": extract_article_number(q),
                "description": "Data protection and privacy requirements",
                "compliance_level": "mandatory",
                "last_updated": "2024-01-20"
            }),
            q if q.contains("iso_27001") => json!({
                "standard": "ISO/IEC 27001:2022",
                "control": extract_control_number(q),
                "description": "Information security management controls",
                "compliance_level": "recommended",
                "last_updated": "2024-02-10"
            }),
            _ => json!({
                "content": format!("Generic compliance content for {}", query),
                "category": "general",
                "confidence": 0.8
            }),
        };
        
        json!({
            "data": content,
            "metadata": {
                "query": query,
                "cached": true,
                "confidence": calculate_confidence_score(query),
                "citations": generate_citations_for_query(query),
                "timestamp": chrono::Utc::now().timestamp()
            }
        })
    }
    
    /// Calculate confidence score based on query complexity
    fn calculate_confidence_score(query: &str) -> f64 {
        let base_confidence = 0.85;
        let complexity_factor = if query.len() > 50 { -0.1 } else { 0.0 };
        let specificity_factor = if query.contains("section") || query.contains("article") { 0.1 } else { 0.0 };
        
        (base_confidence + complexity_factor + specificity_factor).clamp(0.5, 1.0)
    }
    
    /// Generate relevant citations for a query
    fn generate_citations_for_query(query: &str) -> Vec<Value> {
        match query {
            q if q.contains("pci_dss") => vec![json!({
                "source": "PCI Data Security Standard v4.0",
                "section": format!("Requirement {}", extract_requirement_number(q).unwrap_or("3.4".to_string())),
                "page": 47,
                "confidence": 0.98
            })],
            q if q.contains("gdpr") => vec![json!({
                "source": "General Data Protection Regulation (EU) 2016/679",
                "section": format!("Article {}", extract_article_number(q).unwrap_or("7".to_string())),
                "page": 23,
                "confidence": 0.95
            })],
            q if q.contains("iso_27001") => vec![json!({
                "source": "ISO/IEC 27001:2022",
                "section": format!("Annex A.{}", extract_control_number(q).unwrap_or("9.1".to_string())),
                "page": 89,
                "confidence": 0.92
            })],
            _ => vec![],
        }
    }
    
    /// Extract requirement number from PCI DSS query
    fn extract_requirement_number(query: &str) -> Option<String> {
        // Simple regex-like extraction for testing
        if let Some(start) = query.find("requirement_") {
            let number_part = &query[start + 12..];
            if let Some(end) = number_part.find(|c: char| !c.is_numeric() && c != '.') {
                Some(format!("3.{}", &number_part[..end]))
            } else {
                Some(format!("3.{}", number_part))
            }
        } else {
            None
        }
    }
    
    /// Extract article number from GDPR query  
    fn extract_article_number(query: &str) -> Option<String> {
        if let Some(start) = query.find("article_") {
            let number_part = &query[start + 8..];
            if let Some(end) = number_part.find(|c: char| !c.is_numeric()) {
                Some(number_part[..end].to_string())
            } else {
                Some(number_part.to_string())
            }
        } else {
            None
        }
    }
    
    /// Extract control number from ISO 27001 query
    fn extract_control_number(query: &str) -> Option<String> {
        if let Some(start) = query.find("control_") {
            let number_part = &query[start + 8..];
            if let Some(end) = number_part.find(|c: char| !c.is_numeric() && c != '.') {
                Some(number_part[..end].to_string())
            } else {
                Some(number_part.to_string())
            }
        } else {
            None
        }
    }
    
    /// Create a test environment with pre-configured mock clients
    pub async fn setup_test_environment() -> TestEnvironment {
        let cache_hit_client = MockFACTClientFactory::create_cache_hit_optimized();
        let cache_miss_client = MockFACTClientFactory::create_cache_miss_simulation();
        let error_prone_client = MockFACTClientFactory::create_error_prone();
        
        let realistic_simulator = RealisticFACTSimulator::new(SimulatorConfig::default());
        
        // Preload some test data
        let test_data = FACTTestFixtures::generate_compliance_data()
            .into_iter()
            .map(|(k, v)| (k, serde_json::to_vec(&v).unwrap()))
            .collect();
        
        realistic_simulator.preload_cache(test_data).await;
        
        TestEnvironment {
            cache_hit_client,
            cache_miss_client,
            error_prone_client,
            realistic_simulator: Box::new(realistic_simulator),
        }
    }
    
    /// Test environment container
    pub struct TestEnvironment {
        pub cache_hit_client: MockFACTClient,
        pub cache_miss_client: MockFACTClient,
        pub error_prone_client: MockFACTClient,
        pub realistic_simulator: Box<RealisticFACTSimulator>,
    }
    
    /// Performance test runner with configurable parameters
    pub struct PerformanceTestRunner {
        pub config: LoadTestConfig,
        pub warmup_queries: Vec<String>,
    }
    
    impl PerformanceTestRunner {
        pub fn new() -> Self {
            Self {
                config: LoadTestConfig::default(),
                warmup_queries: generate_benchmark_queries(20),
            }
        }
        
        pub fn with_config(mut self, config: LoadTestConfig) -> Self {
            self.config = config;
            self
        }
        
        pub async fn run_benchmark(&self, client: &dyn FACTClientTrait) -> PerformanceSummary {
            let metrics = PerformanceMetrics::new();
            
            // Warmup phase
            for query in &self.warmup_queries {
                let _ = client.get(query).await;
            }
            
            // Main test phase
            let test_queries = generate_benchmark_queries(self.config.concurrent_users * 10);
            let start_time = std::time::Instant::now();
            
            for query in &test_queries {
                let request_start = std::time::Instant::now();
                let result = client.get(query).await;
                let latency = request_start.elapsed();
                
                let was_hit = result.as_ref()
                    .map(|r| r.as_ref()
                        .map(|data| {
                            serde_json::from_slice::<Value>(data)
                                .map(|v| v["metadata"]["cached"].as_bool().unwrap_or(false))
                                .unwrap_or(false)
                        })
                        .unwrap_or(false)
                    )
                    .unwrap_or(false);
                
                metrics.record_request(latency, was_hit, result.is_err());
                
                if start_time.elapsed() > self.config.duration {
                    break;
                }
            }
            
            metrics.get_summary()
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use test_utils::*;
    
    #[tokio::test]
    async fn test_environment_setup() {
        let env = setup_test_environment().await;
        
        // Test each client type
        let cache_hit_result = env.cache_hit_client.get("test_key").await;
        assert!(cache_hit_result.is_ok());
        
        let realistic_result = env.realistic_simulator.get("test_query").await;
        assert!(realistic_result.is_ok());
        
        let health = env.realistic_simulator.health_check().await;
        assert!(health.is_ok());
        assert!(health.unwrap().healthy);
    }
    
    #[tokio::test]
    async fn test_compliance_response_generation() {
        let pci_response = create_compliance_response("pci_dss_requirement_3");
        assert!(pci_response["data"]["regulation"].as_str().unwrap().contains("PCI DSS"));
        
        let gdpr_response = create_compliance_response("gdpr_article_7");
        assert!(gdpr_response["data"]["regulation"].as_str().unwrap().contains("GDPR"));
        
        let iso_response = create_compliance_response("iso_27001_control_9");
        assert!(iso_response["data"]["standard"].as_str().unwrap().contains("ISO"));
    }
    
    #[tokio::test]
    async fn test_performance_runner() {
        let env = setup_test_environment().await;
        let runner = PerformanceTestRunner::new();
        
        let summary = runner.run_benchmark(env.realistic_simulator.as_ref()).await;
        
        assert!(summary.total_requests > 0);
        assert!(summary.hit_rate >= 0.0 && summary.hit_rate <= 1.0);
        assert!(summary.avg_latency > Duration::ZERO);
        assert!(summary.error_rate >= 0.0 && summary.error_rate <= 1.0);
    }
}

/// Constants for testing
pub mod test_constants {
    use std::time::Duration;
    
    pub const CACHE_HIT_TARGET_MS: u64 = 23;
    pub const CACHE_MISS_TARGET_MS: u64 = 95; 
    pub const MIN_HIT_RATE: f64 = 0.873;
    pub const BYZANTINE_THRESHOLD: f64 = 0.66;
    pub const CONCURRENT_USERS_TARGET: usize = 100;
    pub const DEFAULT_TEST_TIMEOUT: Duration = Duration::from_secs(30);
    
    pub const COMPLIANCE_STANDARDS: &[&str] = &[
        "PCI DSS",
        "GDPR", 
        "ISO 27001",
        "SOX",
        "HIPAA",
        "NIST",
        "CCPA",
        "SOC 2"
    ];
}