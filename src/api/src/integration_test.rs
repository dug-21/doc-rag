//! Integration Test File for ruv-FANN, DAA, and FACT Integration Module
//! 
//! This file demonstrates the functionality of the integration module
//! with a focused test that doesn't depend on the rest of the codebase.

#[cfg(test)]
mod integration_standalone_tests {
    use crate::integration::*;
    use crate::config::ApiConfig;
    
    #[tokio::test]
    async fn test_integration_config_defaults() {
        let config = IntegrationConfig::default();
        
        // Test ruv-FANN defaults
        assert_eq!(config.ruv_fann.max_neurons, 1000);
        assert_eq!(config.ruv_fann.learning_rate, 0.7);
        assert_eq!(config.ruv_fann.training_threshold, 0.001);
        assert_eq!(config.ruv_fann.max_iterations, 500000);
        assert_eq!(config.ruv_fann.hidden_layers, vec![100, 50, 25]);
        assert_eq!(config.ruv_fann.activation_function, "FANN_SIGMOID_SYMMETRIC");
        
        // Test DAA defaults
        assert_eq!(config.daa.max_agents, 10);
        assert_eq!(config.daa.coordination_timeout_secs, 30);
        assert_eq!(config.daa.consensus_threshold, 0.67);
        assert!(config.daa.auto_scaling);
        assert_eq!(config.daa.load_balancing_strategy, "round_robin");
        
        // Test FACT defaults
        assert_eq!(config.fact.cache_size_mb, 256);
        assert_eq!(config.fact.ttl_seconds, 3600);
        assert_eq!(config.fact.eviction_policy, "lru");
        assert!(config.fact.compression_enabled);
        assert!(config.fact.persistence_enabled);
        assert_eq!(config.fact.backup_interval_secs, 300);
        
        // Test feature flags
        assert!(config.enable_ruv_fann);
        assert!(config.enable_daa);
        assert!(config.enable_fact);
    }
    
    #[tokio::test]
    async fn test_integration_manager_creation() {
        let api_config = ApiConfig::default();
        let manager = IntegrationManager::from_config(&api_config);
        
        // Initially, no managers should be initialized
        assert!(manager.ruv_fann_manager.is_none());
        assert!(manager.daa_manager.is_none());
        assert!(manager.fact_manager.is_none());
        
        // Check default health status
        let health = manager.get_health().await;
        assert_eq!(health.ruv_fann.status, HealthStatus::Unknown);
        assert_eq!(health.daa.status, HealthStatus::Unknown);
        assert_eq!(health.fact.status, HealthStatus::Unknown);
        assert_eq!(health.overall_status, HealthStatus::Unknown);
    }
    
    #[tokio::test]
    async fn test_integration_manager_initialization() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        // Initialize all systems
        let result = manager.initialize().await;
        assert!(result.is_ok(), "Initialization failed: {:?}", result);
        
        // All managers should now be initialized
        assert!(manager.ruv_fann_manager.is_some());
        assert!(manager.daa_manager.is_some());
        assert!(manager.fact_manager.is_some());
    }
    
    #[tokio::test]
    async fn test_integration_manager_lifecycle() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        // Test full lifecycle: initialize -> start -> stop
        manager.initialize().await.expect("Failed to initialize");
        manager.start().await.expect("Failed to start");
        manager.stop().await.expect("Failed to stop");
    }
    
    #[tokio::test]
    async fn test_health_status_updates() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        manager.initialize().await.expect("Failed to initialize");
        
        // Update health status
        manager.update_health().await.expect("Failed to update health");
        
        let health = manager.get_health().await;
        assert_eq!(health.overall_status, HealthStatus::Healthy);
        assert_eq!(health.ruv_fann.status, HealthStatus::Healthy);
        assert_eq!(health.daa.status, HealthStatus::Healthy);
        assert_eq!(health.fact.status, HealthStatus::Healthy);
    }
    
    #[tokio::test]
    async fn test_ruv_fann_helper_functions() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        manager.initialize().await.expect("Failed to initialize");
        manager.start().await.expect("Failed to start");
        
        // Test boundary detection network creation
        let network_result = manager.create_boundary_detection_network(10).await;
        assert!(network_result.is_ok(), "Failed to create network: {:?}", network_result);
        
        let network_id = network_result.unwrap();
        assert!(!network_id.is_empty());
        
        manager.stop().await.expect("Failed to stop");
    }
    
    #[tokio::test]
    async fn test_daa_helper_functions() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        manager.initialize().await.expect("Failed to initialize");
        manager.start().await.expect("Failed to start");
        
        // Test DAA agent spawning
        let agent_result = manager.spawn_daa_agent(
            "test_agent".to_string(),
            vec!["capability1".to_string(), "capability2".to_string()]
        ).await;
        assert!(agent_result.is_ok(), "Failed to spawn agent: {:?}", agent_result);
        
        let agent_id = agent_result.unwrap();
        // UUID should be a valid UUID format
        assert_eq!(agent_id.to_string().len(), 36); // Standard UUID string length
        
        manager.stop().await.expect("Failed to stop");
    }
    
    #[tokio::test]
    async fn test_fact_helper_functions() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        manager.initialize().await.expect("Failed to initialize");
        manager.start().await.expect("Failed to start");
        
        // Test FACT caching
        let test_data = "test_value".to_string();
        let cache_result = manager.cache_data("test_key", &test_data, None).await;
        assert!(cache_result.is_ok(), "Failed to cache data: {:?}", cache_result);
        
        // Test data retrieval (will return None in mock implementation)
        let retrieve_result: Result<Option<String>, _> = manager.get_cached_data("test_key").await;
        assert!(retrieve_result.is_ok(), "Failed to retrieve data: {:?}", retrieve_result);
        
        manager.stop().await.expect("Failed to stop");
    }
    
    #[tokio::test]
    async fn test_metrics_collection() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        manager.initialize().await.expect("Failed to initialize");
        
        let metrics = manager.get_metrics().await;
        
        // Check that metrics structure is properly initialized
        assert_eq!(metrics.ruv_fann_metrics.active_networks, 0);
        assert_eq!(metrics.daa_metrics.active_agents, 0);
        assert_eq!(metrics.fact_metrics.cache_hit_rate, 0.0);
        assert_eq!(metrics.performance_metrics.cpu_usage_percent, 0.0);
    }
    
    #[tokio::test]
    async fn test_selective_system_initialization() {
        let api_config = ApiConfig::default();
        let mut manager = IntegrationManager::from_config(&api_config);
        
        // Test with individual systems disabled
        manager.config.enable_ruv_fann = false;
        manager.config.enable_daa = true;
        manager.config.enable_fact = true;
        
        manager.initialize().await.expect("Failed to initialize");
        
        // Only DAA and FACT should be initialized
        assert!(manager.ruv_fann_manager.is_none());
        assert!(manager.daa_manager.is_some());
        assert!(manager.fact_manager.is_some());
        
        // Test that disabled system helper functions return appropriate errors
        let network_result = manager.create_boundary_detection_network(10).await;
        assert!(network_result.is_err());
        
        // But enabled systems should work
        let agent_result = manager.spawn_daa_agent("test".to_string(), vec!["test".to_string()]).await;
        assert!(agent_result.is_ok());
        
        let cache_result = manager.cache_data("key", &"value", None).await;
        assert!(cache_result.is_ok());
    }
    
    #[test]
    fn test_health_status_enum_equality() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Degraded);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Unhealthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Unknown);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = IntegrationConfig::default();
        
        // Test that the config can be serialized and deserialized
        let serialized = serde_json::to_string(&config).expect("Failed to serialize config");
        assert!(!serialized.is_empty());
        
        let deserialized: IntegrationConfig = serde_json::from_str(&serialized)
            .expect("Failed to deserialize config");
        
        assert_eq!(deserialized.ruv_fann.max_neurons, config.ruv_fann.max_neurons);
        assert_eq!(deserialized.daa.max_agents, config.daa.max_agents);
        assert_eq!(deserialized.fact.cache_size_mb, config.fact.cache_size_mb);
    }
}