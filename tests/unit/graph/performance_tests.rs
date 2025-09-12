use super::{GraphTestFixtures, TestDatabaseManager};
use graph::{
    models::*, 
    neo4j::{Neo4jClient, Neo4jConfig},
    GraphDatabase, RelationshipType,
};
use std::time::{Duration, Instant};
use tokio_test;

/// Performance-focused tests for Neo4j graph database integration
/// These tests validate the CONSTRAINT-002 requirement: <200ms graph traversal for 3-hop queries

#[cfg(test)]
mod performance_tests {
    use super::*;

    /// Test performance of 3-hop traversal queries (CONSTRAINT-002)
    #[tokio::test]
    async fn test_three_hop_traversal_performance_constraint() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Setup: Create a more complex graph with multiple levels
        let test_doc = create_complex_test_document();
        let _graph = client.create_document_hierarchy(&test_doc).await.unwrap();
        
        // Create additional relationships for traversal
        setup_traversal_relationships(&client, &test_doc).await;
        
        let start_req_id = &test_doc.requirements[0].id;
        let iterations = 10;
        let mut execution_times = Vec::new();
        
        // ACT: Perform multiple 3-hop traversals to measure performance
        for i in 0..iterations {
            let start_time = Instant::now();
            
            let result = client.traverse_requirements(
                start_req_id,
                3, // Exactly 3 hops as per constraint
                vec![RelationshipType::References, RelationshipType::DependsOn],
            ).await;
            
            let execution_time = start_time.elapsed();
            execution_times.push(execution_time);
            
            assert!(result.is_ok(), "Traversal {} should succeed", i + 1);
            
            // CONSTRAINT-002: Individual query must be <200ms
            assert!(
                execution_time.as_millis() < 200,
                "3-hop traversal {} took {}ms, exceeds 200ms limit",
                i + 1,
                execution_time.as_millis()
            );
        }
        
        // Calculate performance statistics
        let total_time: Duration = execution_times.iter().sum();
        let average_time = total_time / iterations as u32;
        let min_time = execution_times.iter().min().unwrap();
        let max_time = execution_times.iter().max().unwrap();
        
        println!("3-hop traversal performance over {} iterations:", iterations);
        println!("  Average: {}ms", average_time.as_millis());
        println!("  Min: {}ms", min_time.as_millis());
        println!("  Max: {}ms", max_time.as_millis());
        
        // ASSERT: Performance requirements
        assert!(
            average_time.as_millis() < 150, // Leave buffer room
            "Average 3-hop traversal time {}ms should be well under 200ms limit",
            average_time.as_millis()
        );
        
        assert!(
            max_time.as_millis() < 200,
            "Maximum 3-hop traversal time {}ms should be under 200ms limit",
            max_time.as_millis()
        );
    }

    /// Test performance degradation with increasing graph size
    #[tokio::test]
    async fn test_performance_scalability() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Test with different graph sizes
        let graph_sizes = vec![10, 50, 100, 250];
        let mut performance_results = Vec::new();
        
        for size in graph_sizes {
            println!("Testing performance with {} requirements", size);
            
            // Setup: Create graph of specified size
            let requirements = GraphTestFixtures::create_bulk_requirements(size);
            let document_id = uuid::Uuid::new_v4().to_string();
            
            // Create all requirement nodes
            for req in &requirements {
                let _node = client.create_requirement_node(req).await.unwrap();
            }
            
            // Create relationships between every 5th requirement
            for i in (0..requirements.len()).step_by(5) {
                if i + 5 < requirements.len() {
                    let _rel = client.create_relationship(
                        &requirements[i].id,
                        &requirements[i + 5].id,
                        RelationshipType::References,
                    ).await.unwrap();
                }
            }
            
            // Measure traversal performance
            let start_time = Instant::now();
            let result = client.traverse_requirements(
                &requirements[0].id,
                3,
                vec![RelationshipType::References],
            ).await;
            let execution_time = start_time.elapsed();
            
            assert!(result.is_ok(), "Traversal with {} nodes should succeed", size);
            
            performance_results.push((size, execution_time));
            
            // Clean up for next iteration
            TestDatabaseManager::clean_test_database(&client).await.unwrap();
        }
        
        // Analyze performance scaling
        println!("Performance scaling results:");
        for (size, time) in &performance_results {
            println!("  {} nodes: {}ms", size, time.as_millis());
        }
        
        // ASSERT: Performance should not degrade dramatically
        // Even with 250 nodes, 3-hop traversal should be under 200ms
        let largest_graph_time = performance_results.last().unwrap().1;
        assert!(
            largest_graph_time.as_millis() < 200,
            "Large graph (250 nodes) traversal took {}ms, exceeds 200ms limit",
            largest_graph_time.as_millis()
        );
    }

    /// Test concurrent query performance
    #[tokio::test]
    async fn test_concurrent_query_performance() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = std::sync::Arc::new(Neo4jClient::new(config).await.unwrap());
        
        // Setup: Create test data
        let test_doc = TestDatabaseManager::setup_test_data(&client).await.unwrap();
        setup_traversal_relationships(&client, &test_doc).await;
        
        let concurrent_queries = 20;
        let start_req_id = test_doc.requirements[0].id.clone();
        
        // ACT: Perform concurrent traversal queries
        let mut handles = Vec::new();
        let overall_start = Instant::now();
        
        for i in 0..concurrent_queries {
            let client_clone = client.clone();
            let req_id = start_req_id.clone();
            
            let handle = tokio::spawn(async move {
                let start_time = Instant::now();
                let result = client_clone.traverse_requirements(
                    &req_id,
                    3,
                    vec![RelationshipType::References, RelationshipType::DependsOn],
                ).await;
                let execution_time = start_time.elapsed();
                (i, result, execution_time)
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            let (query_id, result, execution_time) = handle.await.unwrap();
            results.push((query_id, result, execution_time));
        }
        
        let total_time = overall_start.elapsed();
        
        // ASSERT: All queries should succeed and meet performance requirements
        let mut successful_queries = 0;
        let mut total_query_time = Duration::new(0, 0);
        let mut max_query_time = Duration::new(0, 0);
        
        for (query_id, result, execution_time) in results {
            assert!(result.is_ok(), "Concurrent query {} should succeed", query_id);
            
            successful_queries += 1;
            total_query_time += execution_time;
            max_query_time = max_query_time.max(execution_time);
            
            // Each individual query should still meet the 200ms constraint
            assert!(
                execution_time.as_millis() < 200,
                "Concurrent query {} took {}ms, exceeds 200ms limit",
                query_id,
                execution_time.as_millis()
            );
        }
        
        let average_query_time = total_query_time / concurrent_queries as u32;
        
        println!("Concurrent query performance:");
        println!("  {} queries completed in {}ms total", concurrent_queries, total_time.as_millis());
        println!("  Average query time: {}ms", average_query_time.as_millis());
        println!("  Max query time: {}ms", max_query_time.as_millis());
        println!("  Success rate: {}/{}", successful_queries, concurrent_queries);
        
        // ASSERT: Performance should not degrade significantly under concurrency
        assert_eq!(successful_queries, concurrent_queries, "All concurrent queries should succeed");
        assert!(
            average_query_time.as_millis() < 250, // Allow some overhead for concurrency
            "Average concurrent query time {}ms should be reasonable",
            average_query_time.as_millis()
        );
    }

    /// Test memory usage and connection pooling efficiency
    #[tokio::test]
    async fn test_connection_pool_performance() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Setup test data
        let requirements = GraphTestFixtures::create_bulk_requirements(50);
        for req in &requirements {
            let _node = client.create_requirement_node(req).await.unwrap();
        }
        
        // Test rapid sequential queries (should stress connection pool)
        let query_count = 100;
        let start_time = Instant::now();
        
        for i in 0..query_count {
            let req_index = i % requirements.len();
            let result = client.traverse_requirements(
                &requirements[req_index].id,
                2,
                vec![RelationshipType::References],
            ).await;
            
            assert!(result.is_ok(), "Sequential query {} should succeed", i);
        }
        
        let total_time = start_time.elapsed();
        let average_per_query = total_time / query_count as u32;
        
        println!("Connection pool performance:");
        println!("  {} sequential queries in {}ms", query_count, total_time.as_millis());
        println!("  Average per query: {}ms", average_per_query.as_millis());
        
        // ASSERT: Sequential queries should be efficient due to connection pooling
        assert!(
            average_per_query.as_millis() < 100,
            "Average query time with connection pooling should be under 100ms, got {}ms",
            average_per_query.as_millis()
        );
    }

    /// Test performance monitoring and metrics accuracy
    #[tokio::test]
    async fn test_performance_metrics_accuracy() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Perform known number of operations
        let test_doc = TestDatabaseManager::setup_test_data(&client).await.unwrap();
        let known_operations = 5;
        
        let metrics_before = client.get_performance_metrics().await.unwrap();
        
        // Perform exactly 5 traversal operations
        for i in 0..known_operations {
            let result = client.traverse_requirements(
                &test_doc.requirements[0].id,
                2,
                vec![RelationshipType::References],
            ).await;
            assert!(result.is_ok(), "Operation {} should succeed", i);
        }
        
        let metrics_after = client.get_performance_metrics().await.unwrap();
        
        // ASSERT: Metrics should accurately reflect operations performed
        let operations_diff = metrics_after.total_queries - metrics_before.total_queries;
        assert!(
            operations_diff >= known_operations as u64,
            "Metrics should show at least {} new operations, showed {}",
            known_operations,
            operations_diff
        );
        
        assert!(
            metrics_after.average_query_time_ms > 0.0,
            "Average query time should be positive: {}ms",
            metrics_after.average_query_time_ms
        );
        
        println!("Performance metrics accuracy:");
        println!("  Operations before: {}", metrics_before.total_queries);
        println!("  Operations after: {}", metrics_after.total_queries);
        println!("  Operations difference: {}", operations_diff);
        println!("  Average query time: {}ms", metrics_after.average_query_time_ms);
    }
}

/// Helper function to create complex test document for performance testing
fn create_complex_test_document() -> ProcessedDocument {
    let doc_id = uuid::Uuid::new_v4();
    let now = chrono::Utc::now();
    
    // Create more sections and requirements for complex traversal
    let sections = (1..=10).map(|i| {
        Section {
            id: format!("PERF-SEC-{:02}", i),
            number: format!("{}.0", i),
            title: format!("Performance Test Section {}", i),
            text: format!("Section {} content for performance testing", i),
            page_range: (i as u32 * 10, (i as u32 + 1) * 10),
            subsections: Vec::new(),
            parent_id: None,
            section_type: SectionType::Requirements,
        }
    }).collect();
    
    let requirements = (1..=20).map(|i| {
        let section_num = ((i - 1) / 2) + 1; // 2 requirements per section
        Requirement {
            id: format!("PERF-REQ-{:03}", i),
            text: format!("Performance test requirement {}", i),
            section: format!("PERF-SEC-{:02}", section_num),
            requirement_type: if i % 3 == 0 { RequirementType::May } 
                             else if i % 2 == 0 { RequirementType::Should } 
                             else { RequirementType::Must },
            domain: if i <= 7 { "encryption".to_string() } 
                    else if i <= 14 { "access_control".to_string() } 
                    else { "monitoring".to_string() },
            priority: if i <= 5 { Priority::Critical }
                     else if i <= 10 { Priority::High }
                     else if i <= 15 { Priority::Medium }
                     else { Priority::Low },
            cross_references: Vec::new(),
            created_at: now,
        }
    }).collect::<Vec<_>>();
    
    // Create cross-references to enable traversal
    let cross_references = (0..10).map(|i| {
        CrossReference {
            from_requirement: format!("PERF-REQ-{:03}", i + 1),
            to_requirement: format!("PERF-REQ-{:03}", i + 2),
            reference_type: if i % 2 == 0 { ReferenceType::Direct } else { ReferenceType::Implied },
            context: format!("Performance test reference {}", i),
            confidence: 0.9,
        }
    }).collect();

    ProcessedDocument {
        id: doc_id,
        title: "Performance Test Document".to_string(),
        version: "1.0".to_string(),
        doc_type: DocumentType::PciDss,
        hierarchy: DocumentHierarchy {
            sections,
            total_sections: 10,
            max_depth: 1,
        },
        requirements,
        cross_references,
        metadata: DocumentMetadata {
            title: "Performance Test Document".to_string(),
            version: "1.0".to_string(),
            publication_date: Some(now),
            author: Some("Performance Test Suite".to_string()),
            page_count: 100,
            word_count: 50000,
        },
        created_at: now,
    }
}

/// Setup additional relationships for traversal testing
async fn setup_traversal_relationships(client: &Neo4jClient, test_doc: &ProcessedDocument) {
    // Create a chain of relationships for multi-hop traversal
    for i in 0..test_doc.requirements.len() - 1 {
        if i + 1 < test_doc.requirements.len() {
            let _rel = client.create_relationship(
                &test_doc.requirements[i].id,
                &test_doc.requirements[i + 1].id,
                RelationshipType::References,
            ).await.unwrap();
        }
        
        // Create some dependency relationships
        if i + 2 < test_doc.requirements.len() && i % 3 == 0 {
            let _rel = client.create_relationship(
                &test_doc.requirements[i].id,
                &test_doc.requirements[i + 2].id,
                RelationshipType::DependsOn,
            ).await.unwrap();
        }
    }
}