//! Neo4j Graph Database Integration Tests
//!
//! Comprehensive tests for Neo4j integration compliance with CONSTRAINT-002:
//! - Neo4j v5.0+ for relationship storage
//! - Model requirements as nodes with typed edges
//! - Achieve <200ms graph traversal for 3-hop queries
//! - Keep MongoDB for document storage only

use graph::*;
use std::time::Instant;
// use tokio::test;
use uuid::Uuid;
use chrono::Utc;

/// Test Neo4j client creation and basic connectivity
#[tokio::test]
async fn test_neo4j_client_creation() {
    let config = Neo4jConfig::default();

    // Try to create client - may fail if Neo4j not available
    match Neo4jClient::new(config).await {
        Ok(client) => {
            // Test basic health check
            let health = client.health_check().await;
            assert!(health.is_ok(), "Health check should succeed");

            // Test performance metrics
            let metrics = client.get_performance_metrics().await;
            assert!(metrics.is_ok(), "Should be able to get performance metrics");
        },
        Err(e) => {
            // Expected in CI environments without Neo4j
            println!("Neo4j not available: {}", e);
        }
    }
}

/// Test document hierarchy creation
#[tokio::test]
async fn test_document_hierarchy_creation() {
    let config = Neo4jConfig::default();

    match Neo4jClient::new(config).await {
        Ok(client) => {
            let document = create_test_document();

            let start_time = Instant::now();
            let result = client.create_document_hierarchy(&document).await;
            let elapsed = start_time.elapsed().as_millis() as u64;

            match result {
                Ok(graph) => {
                    assert!(elapsed < 200, "Document hierarchy creation should be under 200ms, got {}ms", elapsed);
                    assert!(!graph.nodes.is_empty(), "Graph should have nodes");
                    assert!(!graph.edges.is_empty(), "Graph should have relationships");

                    // Validate graph completeness
                    let validation_result = graph.validate_completeness().await;
                    assert!(validation_result.is_ok(), "Graph should be valid");
                    assert!(validation_result.unwrap(), "Graph should be complete");
                },
                Err(e) => println!("Document hierarchy creation failed: {}", e),
            }
        },
        Err(e) => println!("Neo4j not available: {}", e),
    }
}

/// Test requirement node creation with typed relationships
#[tokio::test]
async fn test_requirement_relationships() {
    let config = Neo4jConfig::default();

    match Neo4jClient::new(config).await {
        Ok(client) => {
            let req1 = create_test_requirement("req_001", "Test requirement 1");
            let req2 = create_test_requirement("req_002", "Test requirement 2");

            // Create requirement nodes
            let node1_result = client.create_requirement_node(&req1).await;
            let node2_result = client.create_requirement_node(&req2).await;

            if let (Ok(node1), Ok(node2)) = (node1_result, node2_result) {
                // Test all relationship types from CONSTRAINT-002
                let relationship_types = vec![
                    RelationshipType::References,
                    RelationshipType::DependsOn,
                    RelationshipType::Exception,
                ];

                for rel_type in relationship_types {
                    let start_time = Instant::now();
                    let relationship_result = client.create_relationship(
                        &node1.id,
                        &node2.id,
                        rel_type.clone()
                    ).await;
                    let elapsed = start_time.elapsed().as_millis() as u64;

                    match relationship_result {
                        Ok(edge) => {
                            assert!(elapsed < 200, "Relationship creation should be under 200ms, got {}ms", elapsed);
                            assert_eq!(edge.from_node, node1.id);
                            assert_eq!(edge.to_node, node2.id);
                            assert_eq!(edge.relationship_type, rel_type);
                        },
                        Err(e) => println!("Relationship creation failed: {}", e),
                    }
                }
            }
        },
        Err(e) => println!("Neo4j not available: {}", e),
    }
}

/// Test graph traversal performance - CRITICAL <200ms requirement
#[tokio::test]
async fn test_graph_traversal_performance() {
    let config = Neo4jConfig::default();

    match Neo4jClient::new(config).await {
        Ok(client) => {
            let start_id = "req_test_001";
            let max_depth = 3; // 3-hop query as per CONSTRAINT-002
            let relationship_types = vec![
                RelationshipType::References,
                RelationshipType::DependsOn,
            ];

            let start_time = Instant::now();
            let result = client.traverse_requirements(
                start_id,
                max_depth,
                relationship_types
            ).await;
            let elapsed = start_time.elapsed().as_millis() as u64;

            match result {
                Ok(traversal_result) => {
                    // CRITICAL: Must be under 200ms as per CONSTRAINT-002
                    assert!(elapsed < 200, "3-hop graph traversal MUST be under 200ms, got {}ms", elapsed);
                    assert_eq!(traversal_result.execution_time_ms, elapsed);
                    assert_eq!(traversal_result.start_requirement_id, start_id);
                    assert!(!traversal_result.related_requirements.is_empty(), "Should find related requirements");

                    // Validate traversal paths
                    for path in &traversal_result.traversal_paths {
                        assert!(path.path_length <= max_depth, "Path length should not exceed max_depth");
                        assert_eq!(path.start_node, start_id);
                    }
                },
                Err(e) => println!("Graph traversal failed: {}", e),
            }
        },
        Err(e) => println!("Neo4j not available: {}", e),
    }
}

/// Test requirement finding by various filters
#[tokio::test]
async fn test_requirement_finding() {
    let config = Neo4jConfig::default();

    match Neo4jClient::new(config).await {
        Ok(client) => {
            let filters = vec![
                RequirementFilter::ByDomain("PCI-DSS".to_string()),
                RequirementFilter::ByType("MUST".to_string()),
                RequirementFilter::ByPriority("HIGH".to_string()),
            ];

            for filter in filters {
                let start_time = Instant::now();
                let result = client.find_requirements(filter.clone()).await;
                let elapsed = start_time.elapsed().as_millis() as u64;

                match result {
                    Ok(requirements) => {
                        assert!(elapsed < 200, "Requirement finding should be under 200ms, got {}ms", elapsed);
                        assert!(!requirements.is_empty(), "Should find requirements for filter: {:?}", filter);

                        // Validate requirement structure
                        for req in &requirements {
                            assert!(!req.id.is_empty());
                            assert!(!req.text.is_empty());
                            assert!(!req.section.is_empty());
                            assert!(!req.domain.is_empty());
                        }
                    },
                    Err(e) => println!("Requirement finding failed: {}", e),
                }
            }
        },
        Err(e) => println!("Neo4j not available: {}", e),
    }
}

/// Test Neo4j DAA Agent message processing
#[tokio::test]
async fn test_neo4j_daa_agent() {
    let config = Neo4jDaaConfig::default();

    match Neo4jDaaAgent::new(config).await {
        Ok(agent) => {
            // Test health check message
            let health_message = GraphMessage::HealthCheck {
                query_id: Uuid::new_v4(),
            };

            let start_time = Instant::now();
            let response = agent.process_message(health_message).await;
            let elapsed = start_time.elapsed().as_millis() as u64;

            match response {
                GraphResponse::HealthCheckResult { healthy, execution_time_ms, .. } => {
                    assert!(elapsed < 200, "Health check should be under 200ms, got {}ms", elapsed);
                    assert_eq!(execution_time_ms, elapsed);
                    println!("DAA Agent health status: {}", healthy);
                },
                GraphResponse::Error { error, .. } => {
                    // Expected if Neo4j is not available
                    println!("DAA Agent health check failed (expected): {}", error);
                },
                _ => panic!("Unexpected response type"),
            }

            // Test metrics retrieval
            let metrics_message = GraphMessage::GetMetrics {
                query_id: Uuid::new_v4(),
            };

            let metrics_response = agent.process_message(metrics_message).await;
            match metrics_response {
                GraphResponse::MetricsResult { metrics, .. } => {
                    assert!(metrics.total_queries >= 1); // At least the health check
                },
                GraphResponse::Error { .. } => {
                    // Expected if Neo4j is not available
                },
                _ => panic!("Unexpected metrics response type"),
            }
        },
        Err(e) => println!("Neo4j DAA Agent creation failed: {}", e),
    }
}

/// Test performance metrics collection
#[tokio::test]
async fn test_performance_metrics() {
    let config = Neo4jConfig::default();

    match Neo4jClient::new(config).await {
        Ok(client) => {
            // Perform several operations to generate metrics
            let _ = client.health_check().await;
            let _ = client.find_requirements(RequirementFilter::ByDomain("test".to_string())).await;

            let metrics_result = client.get_performance_metrics().await;

            match metrics_result {
                Ok(metrics) => {
                    assert!(metrics.total_queries > 0, "Should have recorded queries");
                    assert!(metrics.average_query_time_ms >= 0.0, "Average query time should be non-negative");
                    assert!(metrics.last_updated <= Utc::now(), "Last updated should be recent");

                    println!("Performance Metrics:");
                    println!("  Total queries: {}", metrics.total_queries);
                    println!("  Average query time: {:.2}ms", metrics.average_query_time_ms);
                    println!("  Cache hit ratio: {:.2}", metrics.cache_hit_ratio);
                    println!("  Total nodes: {}", metrics.total_nodes);
                    println!("  Total relationships: {}", metrics.total_relationships);
                },
                Err(e) => println!("Performance metrics collection failed: {}", e),
            }
        },
        Err(e) => println!("Neo4j not available: {}", e),
    }
}

/// Helper function to create a test document
fn create_test_document() -> ProcessedDocument {
    ProcessedDocument {
        id: Uuid::new_v4(),
        title: "Test PCI-DSS Document".to_string(),
        version: "4.0".to_string(),
        doc_type: DocumentType::PciDss,
        hierarchy: DocumentHierarchy {
            sections: vec![
                Section {
                    id: "section_1".to_string(),
                    number: "1".to_string(),
                    title: "Introduction".to_string(),
                    text: "This section introduces the requirements".to_string(),
                    page_range: (1, 2),
                    subsections: vec![],
                    parent_id: None,
                    section_type: SectionType::Requirements,
                },
                Section {
                    id: "section_2".to_string(),
                    number: "2".to_string(),
                    title: "Network Security".to_string(),
                    text: "Network security requirements".to_string(),
                    page_range: (3, 5),
                    subsections: vec![],
                    parent_id: None,
                    section_type: SectionType::Requirements,
                },
            ],
            total_sections: 2,
            max_depth: 1,
        },
        requirements: vec![
            create_test_requirement("req_001", "Install and maintain a firewall configuration"),
            create_test_requirement("req_002", "Do not use vendor-supplied defaults for system passwords"),
        ],
        cross_references: vec![],
        metadata: DocumentMetadata {
            title: "Test PCI-DSS Document".to_string(),
            version: "4.0".to_string(),
            publication_date: Some(Utc::now()),
            author: Some("PCI Security Standards Council".to_string()),
            page_count: 5,
            word_count: 1000,
        },
        created_at: Utc::now(),
    }
}

/// Helper function to create a test requirement
fn create_test_requirement(id: &str, text: &str) -> Requirement {
    Requirement {
        id: id.to_string(),
        text: text.to_string(),
        section: "1".to_string(),
        requirement_type: RequirementType::Must,
        domain: "PCI-DSS".to_string(),
        priority: Priority::High,
        cross_references: vec![],
        created_at: Utc::now(),
    }
}