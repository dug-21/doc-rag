// Integration tests for Neo4j implementation validation

#[cfg(test)]
mod neo4j_validation_tests {
    use super::*;
    use crate::{GraphDatabase, Neo4jClient, Neo4jConfig, models::*};
    use std::time::Instant;

    /// Test Neo4j deployment and basic connectivity
    #[tokio::test] 
    async fn test_neo4j_deployment_validation() {
        let config = Neo4jConfig::default();
        
        // Test 1: Connection establishment
        let client_result = Neo4jClient::new(config).await;
        if client_result.is_err() {
            println!("Neo4j not available for testing: {:?}", client_result.unwrap_err());
            return; // Skip test if Neo4j is not available
        }
        
        let client = client_result.unwrap();
        
        // Test 2: Health check
        let health = client.health_check().await.unwrap();
        assert!(health, "Neo4j should be healthy");
        
        println!("✓ Neo4j deployment validated successfully");
    }

    /// Test performance constraint validation
    #[tokio::test]
    async fn test_performance_constraint_validation() {
        let config = Neo4jConfig::default();
        let client = match Neo4jClient::new(config).await {
            Ok(client) => client,
            Err(_) => {
                println!("Skipping performance test - Neo4j not available");
                return;
            }
        };
        
        // Create test data for performance validation
        let test_req = create_test_requirement("PERF-TEST-001", "Test performance requirement");
        let _node = client.create_requirement_node(&test_req).await.unwrap();
        
        // Test 3-hop traversal performance constraint
        let start_time = Instant::now();
        let result = client.traverse_requirements(
            &test_req.id,
            3,
            vec![RelationshipType::References],
        ).await;
        let execution_time = start_time.elapsed();
        
        assert!(result.is_ok(), "Traversal should succeed");
        assert!(
            execution_time.as_millis() < 200,
            "CONSTRAINT-002 VIOLATION: 3-hop traversal took {}ms, must be <200ms",
            execution_time.as_millis()
        );
        
        println!("✓ Performance constraint CONSTRAINT-002 validated: {}ms", execution_time.as_millis());
    }
    
    fn create_test_requirement(id: &str, text: &str) -> Requirement {
        Requirement {
            id: id.to_string(),
            text: text.to_string(),
            section: "TEST-SECTION".to_string(),
            requirement_type: RequirementType::Must,
            domain: "validation".to_string(),
            priority: Priority::High,
            cross_references: Vec::new(),
            created_at: chrono::Utc::now(),
        }
    }
}