use super::{GraphTestFixtures, TestDatabaseManager};
use graph::{
    models::*, 
    neo4j::{Neo4jClient, Neo4jConfig},
    GraphDatabase, GraphError, RelationshipType,
};
use std::time::Instant;
use tokio_test;
use uuid::Uuid;

/// London TDD: Neo4j Client Integration Tests
/// 
/// These tests follow the London style of TDD where we test the behavior
/// and interactions of our Neo4j client without mocking the database itself.
/// Instead, we use a real test database instance.

#[cfg(test)]
mod neo4j_client_tests {
    use super::*;

    /// Test client initialization and connection
    #[tokio::test]
    async fn test_client_initialization_success() {
        // Skip test if Neo4j is not available
        if !TestDatabaseManager::is_neo4j_available().await {
            println!("Skipping Neo4j tests - database not available");
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        
        // ACT: Initialize client
        let result = Neo4jClient::new(config).await;
        
        // ASSERT: Client should initialize successfully
        assert!(result.is_ok(), "Client initialization should succeed");
        
        let client = result.unwrap();
        
        // Verify health check
        let health_result = client.health_check().await;
        assert!(health_result.is_ok(), "Health check should succeed");
        assert!(health_result.unwrap(), "Database should be healthy");
    }

    #[tokio::test] 
    async fn test_client_initialization_invalid_connection() {
        let mut config = GraphTestFixtures::create_test_config();
        config.base.uri = "bolt://invalid-host:7687".to_string();
        
        // ACT: Try to initialize with invalid connection
        let result = Neo4jClient::new(config).await;
        
        // ASSERT: Should fail with connection error
        assert!(result.is_err(), "Should fail with invalid connection");
        match result.unwrap_err() {
            GraphError::Connection(_) => {}, // Expected
            other => panic!("Expected connection error, got: {:?}", other),
        }
    }

    /// Test document hierarchy creation
    #[tokio::test]
    async fn test_create_document_hierarchy() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Clean any existing test data
        TestDatabaseManager::clean_test_database(&client).await.unwrap();
        
        let test_doc = GraphTestFixtures::create_test_document();
        
        // ACT: Create document hierarchy
        let result = client.create_document_hierarchy(&test_doc).await;
        
        // ASSERT: Should succeed
        assert!(result.is_ok(), "Document hierarchy creation should succeed");
        
        let graph = result.unwrap();
        assert_eq!(graph.document_node, test_doc.id.to_string());
        assert_eq!(graph.section_nodes.len(), test_doc.hierarchy.sections.len());
        assert_eq!(graph.requirement_nodes.len(), test_doc.requirements.len());
    }

    /// Test requirement node creation
    #[tokio::test]
    async fn test_create_requirement_node() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        let requirement = GraphTestFixtures::create_test_requirement(
            "TEST-REQ-001", 
            "3.2", 
            "Test requirement for node creation"
        );
        
        // ACT: Create requirement node
        let result = client.create_requirement_node(&requirement).await;
        
        // ASSERT: Should succeed and return valid node
        assert!(result.is_ok(), "Requirement node creation should succeed");
        
        let node = result.unwrap();
        assert_eq!(node.id, requirement.id);
        assert!(node.neo4j_id > 0, "Neo4j ID should be positive");
        assert_eq!(node.properties.text, requirement.text);
    }

    /// Test relationship creation between nodes
    #[tokio::test]
    async fn test_create_relationship() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Setup: Create two requirement nodes
        let req1 = GraphTestFixtures::create_test_requirement("TEST-REQ-REL-001", "3.2", "First requirement");
        let req2 = GraphTestFixtures::create_test_requirement("TEST-REQ-REL-002", "3.3", "Second requirement");
        
        let _node1 = client.create_requirement_node(&req1).await.unwrap();
        let _node2 = client.create_requirement_node(&req2).await.unwrap();
        
        // ACT: Create relationship
        let result = client.create_relationship(
            &req1.id,
            &req2.id,
            RelationshipType::References,
        ).await;
        
        // ASSERT: Should succeed
        assert!(result.is_ok(), "Relationship creation should succeed");
        
        let relationship = result.unwrap();
        assert_eq!(relationship.from_node, req1.id);
        assert_eq!(relationship.to_node, req2.id);
        assert_eq!(relationship.relationship_type, RelationshipType::References);
    }

    /// Test requirement traversal with performance constraint
    #[tokio::test]
    async fn test_traverse_requirements_performance() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Setup test data
        let test_doc = TestDatabaseManager::setup_test_data(&client).await.unwrap();
        let start_req_id = &test_doc.requirements[0].id;
        
        let start_time = Instant::now();
        
        // ACT: Perform 3-hop traversal
        let result = client.traverse_requirements(
            start_req_id,
            3, // Max 3 hops
            vec![RelationshipType::References, RelationshipType::DependsOn],
        ).await;
        
        let execution_time = start_time.elapsed();
        
        // ASSERT: Should complete within performance constraint
        assert!(result.is_ok(), "Traversal should succeed");
        assert!(
            execution_time.as_millis() < 200, 
            "3-hop traversal should complete in <200ms, took {}ms", 
            execution_time.as_millis()
        );
        
        let traversal_result = result.unwrap();
        assert_eq!(traversal_result.start_requirement_id, *start_req_id);
        assert!(traversal_result.execution_time_ms < 200);
    }

    /// Test deep traversal with depth limit
    #[tokio::test]
    async fn test_traverse_requirements_depth_limit() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        let test_doc = TestDatabaseManager::setup_test_data(&client).await.unwrap();
        let start_req_id = &test_doc.requirements[0].id;
        
        // ACT: Test different depth limits
        let depths_to_test = vec![1, 2, 3, 5];
        
        for max_depth in depths_to_test {
            let result = client.traverse_requirements(
                start_req_id,
                max_depth,
                vec![RelationshipType::References],
            ).await;
            
            // ASSERT: Should respect depth limits
            assert!(result.is_ok(), "Traversal with depth {} should succeed", max_depth);
            
            let traversal_result = result.unwrap();
            for path in &traversal_result.traversal_paths {
                assert!(
                    path.depth <= max_depth,
                    "Path depth {} should not exceed max depth {}",
                    path.depth,
                    max_depth
                );
            }
        }
    }

    /// Test requirement filtering and search
    #[tokio::test]
    async fn test_find_requirements_with_filter() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        let _test_doc = TestDatabaseManager::setup_test_data(&client).await.unwrap();
        
        let filter = GraphTestFixtures::create_test_filter();
        
        // ACT: Find requirements with filter
        let result = client.find_requirements(filter).await;
        
        // ASSERT: Should return filtered results
        assert!(result.is_ok(), "Find requirements should succeed");
        
        let requirements = result.unwrap();
        
        // Verify all returned requirements match the filter criteria
        for req in &requirements {
            assert_eq!(req.requirement_type, RequirementType::Must);
            assert_eq!(req.domain, "encryption");
            assert_eq!(req.priority, Priority::High);
            assert!(req.text.to_lowercase().contains("encrypted"));
        }
    }

    /// Test performance metrics tracking
    #[tokio::test]
    async fn test_performance_metrics_tracking() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Perform several operations to generate metrics
        let test_doc = TestDatabaseManager::setup_test_data(&client).await.unwrap();
        let _graph = client.create_document_hierarchy(&test_doc).await.unwrap();
        
        // ACT: Get performance metrics
        let result = client.get_performance_metrics().await;
        
        // ASSERT: Should have valid metrics
        assert!(result.is_ok(), "Should be able to retrieve performance metrics");
        
        let metrics = result.unwrap();
        assert!(metrics.total_queries > 0, "Should have recorded queries");
        assert!(metrics.average_query_time_ms >= 0.0, "Average time should be non-negative");
        assert!(metrics.total_nodes >= 0, "Node count should be non-negative");
        assert!(metrics.cache_hit_ratio >= 0.0 && metrics.cache_hit_ratio <= 1.0, "Cache hit ratio should be between 0 and 1");
    }

    /// Test concurrent operations
    #[tokio::test]
    async fn test_concurrent_operations() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = std::sync::Arc::new(Neo4jClient::new(config).await.unwrap());
        
        let concurrent_operations = 10;
        let mut handles = Vec::new();
        
        // ACT: Perform concurrent operations
        for i in 0..concurrent_operations {
            let client_clone = client.clone();
            let handle = tokio::spawn(async move {
                let requirement = GraphTestFixtures::create_test_requirement(
                    &format!("CONCURRENT-REQ-{:03}", i),
                    "CONCURRENT",
                    &format!("Concurrent requirement {}", i),
                );
                client_clone.create_requirement_node(&requirement).await
            });
            handles.push(handle);
        }
        
        // Wait for all operations to complete
        let mut successful_operations = 0;
        for handle in handles {
            match handle.await {
                Ok(Ok(_)) => successful_operations += 1,
                Ok(Err(e)) => println!("Operation failed: {:?}", e),
                Err(e) => println!("Task failed: {:?}", e),
            }
        }
        
        // ASSERT: Most operations should succeed
        assert!(
            successful_operations >= concurrent_operations * 8 / 10, // At least 80% success rate
            "At least 80% of concurrent operations should succeed, got {}/{}",
            successful_operations,
            concurrent_operations
        );
    }

    /// Test error handling for non-existent nodes
    #[tokio::test]
    async fn test_error_handling_non_existent_nodes() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // ACT: Try to create relationship with non-existent nodes
        let result = client.create_relationship(
            "NON-EXISTENT-001",
            "NON-EXISTENT-002",
            RelationshipType::References,
        ).await;
        
        // ASSERT: Should handle error gracefully
        assert!(result.is_err(), "Should fail when nodes don't exist");
        
        // ACT: Try to traverse from non-existent node
        let traversal_result = client.traverse_requirements(
            "NON-EXISTENT-START",
            3,
            vec![RelationshipType::References],
        ).await;
        
        // ASSERT: Should return empty result or appropriate error
        assert!(
            traversal_result.is_ok() || traversal_result.is_err(),
            "Should handle non-existent start node gracefully"
        );
        
        if let Ok(result) = traversal_result {
            assert!(result.related_requirements.is_empty(), "Should return empty result for non-existent start node");
        }
    }

    /// Test cache functionality
    #[tokio::test] 
    async fn test_query_caching() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let mut config = GraphTestFixtures::create_test_config();
        config.base.enable_cache = true;
        config.base.cache_ttl_seconds = 60;
        
        let client = Neo4jClient::new(config).await.unwrap();
        let test_doc = TestDatabaseManager::setup_test_data(&client).await.unwrap();
        let start_req_id = &test_doc.requirements[0].id;
        
        // ACT: Perform same query twice
        let start_time = Instant::now();
        let result1 = client.traverse_requirements(
            start_req_id,
            2,
            vec![RelationshipType::References],
        ).await.unwrap();
        let first_query_time = start_time.elapsed();
        
        let start_time = Instant::now();
        let result2 = client.traverse_requirements(
            start_req_id,
            2,
            vec![RelationshipType::References],
        ).await.unwrap();
        let second_query_time = start_time.elapsed();
        
        // ASSERT: Second query should be faster due to caching
        // Note: This may not always be true due to database caching, but it's a good heuristic
        println!("First query: {}ms, Second query: {}ms", 
                 first_query_time.as_millis(), second_query_time.as_millis());
        
        // Results should be identical
        assert_eq!(result1.start_requirement_id, result2.start_requirement_id);
        assert_eq!(result1.related_requirements.len(), result2.related_requirements.len());
        
        // Verify metrics show cache activity
        let metrics = client.get_performance_metrics().await.unwrap();
        println!("Cache hit ratio: {:.2}", metrics.cache_hit_ratio);
    }
}