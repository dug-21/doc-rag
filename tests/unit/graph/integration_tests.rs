use super::{GraphTestFixtures, TestDatabaseManager};
use graph::{
    models::*, 
    neo4j::{Neo4jClient, Neo4jConfig},
    GraphDatabase, RelationshipType,
};
use std::collections::HashMap;
use tokio_test;

/// Integration tests that validate complete workflow scenarios
/// These tests verify end-to-end functionality and real-world usage patterns

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Test complete document ingestion workflow
    #[tokio::test]
    async fn test_complete_document_ingestion_workflow() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Clean start
        TestDatabaseManager::clean_test_database(&client).await.unwrap();
        
        // SCENARIO: Complete PCI-DSS document processing
        let pci_document = create_realistic_pci_document();
        
        // ACT: Process complete document
        let graph_result = client.create_document_hierarchy(&pci_document).await;
        assert!(graph_result.is_ok(), "Document hierarchy creation should succeed");
        
        let document_graph = graph_result.unwrap();
        
        // ASSERT: Document structure is properly created
        assert_eq!(document_graph.document_node, pci_document.id.to_string());
        assert_eq!(document_graph.section_nodes.len(), pci_document.hierarchy.sections.len());
        assert_eq!(document_graph.requirement_nodes.len(), pci_document.requirements.len());
        
        // Verify all sections are properly linked
        for section in &pci_document.hierarchy.sections {
            let section_exists = document_graph.section_nodes
                .iter()
                .any(|node| node.id == section.id);
            assert!(section_exists, "Section {} should exist in graph", section.id);
        }
        
        // Verify all requirements are properly linked
        for requirement in &pci_document.requirements {
            let req_exists = document_graph.requirement_nodes
                .iter()
                .any(|node| node.id == requirement.id);
            assert!(req_exists, "Requirement {} should exist in graph", requirement.id);
        }
        
        // ASSERT: Cross-references are properly created
        let traversal_result = client.traverse_requirements(
            &pci_document.requirements[0].id,
            3,
            vec![RelationshipType::References, RelationshipType::DependsOn],
        ).await.unwrap();
        
        assert!(
            !traversal_result.related_requirements.is_empty(),
            "Should find related requirements through cross-references"
        );
    }

    /// Test realistic compliance query scenarios
    #[tokio::test]
    async fn test_compliance_query_scenarios() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Setup: Create realistic compliance document
        let pci_document = create_realistic_pci_document();
        let _graph = client.create_document_hierarchy(&pci_document).await.unwrap();
        
        // SCENARIO 1: Find all encryption requirements
        let encryption_filter = RequirementFilter {
            section: None,
            requirement_type: Some(RequirementType::Must),
            domain: Some("encryption".to_string()),
            priority: None,
            text_contains: None,
        };
        
        let encryption_reqs = client.find_requirements(encryption_filter).await.unwrap();
        assert!(!encryption_reqs.is_empty(), "Should find encryption requirements");
        
        for req in &encryption_reqs {
            assert_eq!(req.domain, "encryption");
            assert_eq!(req.requirement_type, RequirementType::Must);
        }
        
        // SCENARIO 2: Find all requirements in section 3.2
        let section_filter = RequirementFilter {
            section: Some("SEC-3.2".to_string()),
            requirement_type: None,
            domain: None,
            priority: None,
            text_contains: None,
        };
        
        let section_reqs = client.find_requirements(section_filter).await.unwrap();
        assert!(!section_reqs.is_empty(), "Should find requirements in section 3.2");
        
        for req in &section_reqs {
            assert_eq!(req.section, "SEC-3.2");
        }
        
        // SCENARIO 3: Find critical priority requirements
        let critical_filter = RequirementFilter {
            section: None,
            requirement_type: None,
            domain: None,
            priority: Some(Priority::Critical),
            text_contains: None,
        };
        
        let critical_reqs = client.find_requirements(critical_filter).await.unwrap();
        for req in &critical_reqs {
            assert_eq!(req.priority, Priority::Critical);
        }
    }

    /// Test requirement dependency analysis
    #[tokio::test]
    async fn test_requirement_dependency_analysis() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Setup: Create document with complex dependencies
        let doc_with_deps = create_document_with_dependencies();
        let _graph = client.create_document_hierarchy(&doc_with_deps).await.unwrap();
        
        // Create additional dependency relationships
        setup_complex_dependencies(&client, &doc_with_deps).await;
        
        // SCENARIO: Analyze impact of changing a foundational requirement
        let foundational_req_id = &doc_with_deps.requirements[0].id; // First requirement as foundation
        
        // Find all requirements that depend on this foundational requirement
        let dependency_analysis = client.traverse_requirements(
            foundational_req_id,
            5, // Deep traversal to find all dependents
            vec![RelationshipType::DependsOn, RelationshipType::References],
        ).await.unwrap();
        
        // ASSERT: Should find dependent requirements
        assert!(
            !dependency_analysis.related_requirements.is_empty(),
            "Foundational requirement should have dependents"
        );
        
        // Verify dependency chain integrity
        let mut dependency_levels = HashMap::new();
        for path in &dependency_analysis.traversal_paths {
            let level = dependency_levels.entry(path.end_id.clone()).or_insert(path.depth);
            *level = (*level).min(path.depth); // Take shortest path as primary dependency level
        }
        
        println!("Dependency analysis for {}:", foundational_req_id);
        for (req_id, level) in &dependency_levels {
            println!("  {} at level {}", req_id, level);
        }
        
        // ASSERT: Should have multi-level dependencies
        let max_level = dependency_levels.values().max().unwrap_or(&0);
        assert!(
            *max_level > 1,
            "Should have multi-level dependencies, max level: {}",
            max_level
        );
    }

    /// Test document comparison and diff analysis
    #[tokio::test] 
    async fn test_document_comparison_analysis() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // Setup: Create two versions of a document
        let doc_v1 = create_realistic_pci_document();
        let doc_v2 = create_updated_pci_document();
        
        // Process both documents
        let _graph_v1 = client.create_document_hierarchy(&doc_v1).await.unwrap();
        let _graph_v2 = client.create_document_hierarchy(&doc_v2).await.unwrap();
        
        // SCENARIO: Compare requirements between versions
        let v1_encryption_filter = RequirementFilter {
            section: None,
            requirement_type: None,
            domain: Some("encryption".to_string()),
            priority: None,
            text_contains: Some("v1".to_string()), // V1 specific requirements
        };
        
        let v2_encryption_filter = RequirementFilter {
            section: None,
            requirement_type: None,
            domain: Some("encryption".to_string()),
            priority: None,
            text_contains: Some("v2".to_string()), // V2 specific requirements
        };
        
        let v1_reqs = client.find_requirements(v1_encryption_filter).await.unwrap();
        let v2_reqs = client.find_requirements(v2_encryption_filter).await.unwrap();
        
        // ASSERT: Should find version-specific requirements
        assert!(!v1_reqs.is_empty(), "Should find v1 requirements");
        assert!(!v2_reqs.is_empty(), "Should find v2 requirements");
        
        println!("Document comparison:");
        println!("  V1 encryption requirements: {}", v1_reqs.len());
        println!("  V2 encryption requirements: {}", v2_reqs.len());
        
        // Verify version isolation
        for req in &v1_reqs {
            assert!(req.text.contains("v1"), "V1 requirement should contain v1 marker");
        }
        
        for req in &v2_reqs {
            assert!(req.text.contains("v2"), "V2 requirement should contain v2 marker");
        }
    }

    /// Test error recovery and data consistency
    #[tokio::test]
    async fn test_error_recovery_and_consistency() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // SCENARIO 1: Attempt to create duplicate requirement IDs
        let req1 = GraphTestFixtures::create_test_requirement("DUPLICATE-001", "3.2", "First requirement");
        let req2 = GraphTestFixtures::create_test_requirement("DUPLICATE-001", "3.3", "Second requirement with same ID");
        
        let result1 = client.create_requirement_node(&req1).await;
        let result2 = client.create_requirement_node(&req2).await;
        
        assert!(result1.is_ok(), "First requirement creation should succeed");
        assert!(result2.is_err(), "Duplicate requirement creation should fail");
        
        // SCENARIO 2: Test partial document creation failure handling
        let mut problematic_doc = GraphTestFixtures::create_test_document();
        
        // Create a requirement with invalid data that might cause issues
        problematic_doc.requirements.push(Requirement {
            id: "".to_string(), // Empty ID should cause constraint violation
            text: "Invalid requirement".to_string(),
            section: "SEC-3.2".to_string(),
            requirement_type: RequirementType::Must,
            domain: "test".to_string(),
            priority: Priority::Medium,
            cross_references: Vec::new(),
            created_at: chrono::Utc::now(),
        });
        
        let result = client.create_document_hierarchy(&problematic_doc).await;
        
        // Should handle the error gracefully
        if result.is_err() {
            println!("Expected error handled: {:?}", result.unwrap_err());
        }
        
        // SCENARIO 3: Verify database consistency after errors
        let health_result = client.health_check().await;
        assert!(health_result.is_ok() && health_result.unwrap(), "Database should remain healthy after errors");
        
        let metrics = client.get_performance_metrics().await.unwrap();
        assert!(metrics.total_queries > 0, "Metrics should still be tracking properly");
    }

    /// Test large-scale operations and bulk processing
    #[tokio::test]
    async fn test_large_scale_operations() {
        if !TestDatabaseManager::is_neo4j_available().await {
            return;
        }

        let config = GraphTestFixtures::create_test_config();
        let client = Neo4jClient::new(config).await.unwrap();
        
        // SCENARIO: Process large compliance framework
        let large_doc = create_large_compliance_document(500); // 500 requirements
        
        // Measure bulk creation performance
        let start_time = std::time::Instant::now();
        let graph_result = client.create_document_hierarchy(&large_doc).await;
        let creation_time = start_time.elapsed();
        
        assert!(graph_result.is_ok(), "Large document creation should succeed");
        
        let document_graph = graph_result.unwrap();
        assert_eq!(document_graph.requirement_nodes.len(), 500);
        
        println!("Large-scale operations performance:");
        println!("  Created 500 requirements in {}ms", creation_time.as_millis());
        println!("  Average per requirement: {}ms", creation_time.as_millis() / 500);
        
        // ASSERT: Should handle large scale efficiently
        assert!(
            creation_time.as_secs() < 30, // Should complete within 30 seconds
            "Large document creation took {}s, should be under 30s",
            creation_time.as_secs()
        );
        
        // Test bulk traversal operations
        let bulk_traversal_start = std::time::Instant::now();
        let traversal_result = client.traverse_requirements(
            &large_doc.requirements[0].id,
            3,
            vec![RelationshipType::References, RelationshipType::DependsOn],
        ).await.unwrap();
        let traversal_time = bulk_traversal_start.elapsed();
        
        println!("  Traversal in large graph: {}ms", traversal_time.as_millis());
        
        // Should still meet performance constraints even with large graph
        assert!(
            traversal_time.as_millis() < 200,
            "Traversal in large graph took {}ms, exceeds 200ms constraint",
            traversal_time.as_millis()
        );
    }
}

/// Create realistic PCI-DSS document for integration testing
fn create_realistic_pci_document() -> ProcessedDocument {
    let doc_id = uuid::Uuid::new_v4();
    let now = chrono::Utc::now();
    
    ProcessedDocument {
        id: doc_id,
        title: "PCI Data Security Standard v4.0".to_string(),
        version: "4.0".to_string(),
        doc_type: DocumentType::PciDss,
        hierarchy: DocumentHierarchy {
            sections: vec![
                create_pci_section("3.2", "Cardholder Data Protection"),
                create_pci_section("3.3", "Encryption Requirements"),
                create_pci_section("3.4", "Key Management"),
                create_pci_section("8.1", "Access Control Policies"),
                create_pci_section("8.2", "Authentication Requirements"),
            ],
            total_sections: 5,
            max_depth: 2,
        },
        requirements: vec![
            create_pci_requirement("PCI-3.2.1", "SEC-3.2", "Cardholder data must be encrypted at rest v1"),
            create_pci_requirement("PCI-3.3.1", "SEC-3.3", "Encryption keys must be managed securely v1"),
            create_pci_requirement("PCI-3.4.1", "SEC-3.4", "Key rotation must occur every 12 months v1"),
            create_pci_requirement("PCI-8.1.1", "SEC-8.1", "Access control policies must be documented v1"),
            create_pci_requirement("PCI-8.2.1", "SEC-8.2", "Multi-factor authentication is required v1"),
        ],
        cross_references: vec![
            CrossReference {
                from_requirement: "PCI-3.2.1".to_string(),
                to_requirement: "PCI-3.3.1".to_string(),
                reference_type: ReferenceType::Direct,
                context: "Encryption implementation dependency".to_string(),
                confidence: 0.95,
            },
            CrossReference {
                from_requirement: "PCI-3.3.1".to_string(),
                to_requirement: "PCI-3.4.1".to_string(),
                reference_type: ReferenceType::Direct,
                context: "Key management requirement".to_string(),
                confidence: 0.90,
            },
        ],
        metadata: DocumentMetadata {
            title: "PCI Data Security Standard v4.0".to_string(),
            version: "4.0".to_string(),
            publication_date: Some(now),
            author: Some("PCI Security Standards Council".to_string()),
            page_count: 150,
            word_count: 75000,
        },
        created_at: now,
    }
}

/// Create updated version of PCI document for comparison testing
fn create_updated_pci_document() -> ProcessedDocument {
    let doc_id = uuid::Uuid::new_v4();
    let now = chrono::Utc::now();
    
    ProcessedDocument {
        id: doc_id,
        title: "PCI Data Security Standard v4.1".to_string(),
        version: "4.1".to_string(),
        doc_type: DocumentType::PciDss,
        hierarchy: DocumentHierarchy {
            sections: vec![
                create_pci_section("3.2", "Cardholder Data Protection"),
                create_pci_section("3.3", "Enhanced Encryption Requirements"),
                create_pci_section("3.4", "Advanced Key Management"),
            ],
            total_sections: 3,
            max_depth: 2,
        },
        requirements: vec![
            create_pci_requirement("PCI-3.2.1-V2", "SEC-3.2", "Cardholder data must be encrypted at rest with AES-256 v2"),
            create_pci_requirement("PCI-3.3.1-V2", "SEC-3.3", "Encryption keys must be managed with HSM v2"),
            create_pci_requirement("PCI-3.4.1-V2", "SEC-3.4", "Key rotation must occur every 6 months v2"),
        ],
        cross_references: vec![
            CrossReference {
                from_requirement: "PCI-3.2.1-V2".to_string(),
                to_requirement: "PCI-3.3.1-V2".to_string(),
                reference_type: ReferenceType::Direct,
                context: "Enhanced encryption dependency".to_string(),
                confidence: 0.98,
            },
        ],
        metadata: DocumentMetadata {
            title: "PCI Data Security Standard v4.1".to_string(),
            version: "4.1".to_string(),
            publication_date: Some(now),
            author: Some("PCI Security Standards Council".to_string()),
            page_count: 160,
            word_count: 80000,
        },
        created_at: now,
    }
}

/// Create document with complex dependency relationships
fn create_document_with_dependencies() -> ProcessedDocument {
    let doc_id = uuid::Uuid::new_v4();
    let now = chrono::Utc::now();
    
    ProcessedDocument {
        id: doc_id,
        title: "Complex Dependencies Test Document".to_string(),
        version: "1.0".to_string(),
        doc_type: DocumentType::Nist,
        hierarchy: DocumentHierarchy {
            sections: vec![
                create_pci_section("1.0", "Foundational Requirements"),
                create_pci_section("2.0", "Dependent Requirements"),
                create_pci_section("3.0", "Implementation Requirements"),
            ],
            total_sections: 3,
            max_depth: 3,
        },
        requirements: vec![
            create_pci_requirement("DEP-1.0.1", "SEC-1.0", "Foundational security policy must be established"),
            create_pci_requirement("DEP-2.0.1", "SEC-2.0", "Access controls must implement security policy"),
            create_pci_requirement("DEP-2.0.2", "SEC-2.0", "Monitoring must align with security policy"),
            create_pci_requirement("DEP-3.0.1", "SEC-3.0", "Implementation must follow access controls"),
            create_pci_requirement("DEP-3.0.2", "SEC-3.0", "Implementation must enable monitoring"),
        ],
        cross_references: Vec::new(), // Will be created by setup_complex_dependencies
        metadata: DocumentMetadata {
            title: "Complex Dependencies Test Document".to_string(),
            version: "1.0".to_string(),
            publication_date: Some(now),
            author: Some("Test Suite".to_string()),
            page_count: 50,
            word_count: 25000,
        },
        created_at: now,
    }
}

/// Helper function to create PCI section
fn create_pci_section(number: &str, title: &str) -> Section {
    Section {
        id: format!("SEC-{}", number),
        number: number.to_string(),
        title: title.to_string(),
        text: format!("Section {} covers {}", number, title),
        page_range: (10, 20),
        subsections: Vec::new(),
        parent_id: None,
        section_type: SectionType::Requirements,
    }
}

/// Helper function to create PCI requirement
fn create_pci_requirement(id: &str, section: &str, text: &str) -> Requirement {
    Requirement {
        id: id.to_string(),
        text: text.to_string(),
        section: section.to_string(),
        requirement_type: RequirementType::Must,
        domain: "encryption".to_string(),
        priority: if text.contains("must") { Priority::Critical } else { Priority::High },
        cross_references: Vec::new(),
        created_at: chrono::Utc::now(),
    }
}

/// Setup complex dependency relationships
async fn setup_complex_dependencies(client: &Neo4jClient, doc: &ProcessedDocument) {
    // Create dependency chain: Foundation -> Dependent -> Implementation
    if doc.requirements.len() >= 5 {
        // Foundation to dependents
        let _rel1 = client.create_relationship(
            &doc.requirements[0].id, // Foundation
            &doc.requirements[1].id, // First dependent
            RelationshipType::DependsOn,
        ).await.unwrap();
        
        let _rel2 = client.create_relationship(
            &doc.requirements[0].id, // Foundation
            &doc.requirements[2].id, // Second dependent
            RelationshipType::DependsOn,
        ).await.unwrap();
        
        // Dependents to implementations
        let _rel3 = client.create_relationship(
            &doc.requirements[1].id, // First dependent
            &doc.requirements[3].id, // First implementation
            RelationshipType::DependsOn,
        ).await.unwrap();
        
        let _rel4 = client.create_relationship(
            &doc.requirements[2].id, // Second dependent
            &doc.requirements[4].id, // Second implementation
            RelationshipType::DependsOn,
        ).await.unwrap();
        
        // Cross-implementation reference
        let _rel5 = client.create_relationship(
            &doc.requirements[3].id, // First implementation
            &doc.requirements[4].id, // Second implementation
            RelationshipType::References,
        ).await.unwrap();
    }
}

/// Create large compliance document for scale testing
fn create_large_compliance_document(requirement_count: usize) -> ProcessedDocument {
    let doc_id = uuid::Uuid::new_v4();
    let now = chrono::Utc::now();
    
    let sections: Vec<Section> = (1..=50).map(|i| {
        Section {
            id: format!("LARGE-SEC-{:03}", i),
            number: format!("{}.0", i),
            title: format!("Large Scale Section {}", i),
            text: format!("Section {} for large scale testing", i),
            page_range: (i as u32 * 10, (i as u32 + 1) * 10),
            subsections: Vec::new(),
            parent_id: None,
            section_type: SectionType::Requirements,
        }
    }).collect();
    
    let requirements: Vec<Requirement> = (1..=requirement_count).map(|i| {
        let section_num = ((i - 1) / 10) + 1; // 10 requirements per section
        Requirement {
            id: format!("LARGE-REQ-{:06}", i),
            text: format!("Large scale requirement {} for comprehensive testing", i),
            section: format!("LARGE-SEC-{:03}", section_num),
            requirement_type: match i % 4 {
                0 => RequirementType::Must,
                1 => RequirementType::Should,
                2 => RequirementType::May,
                _ => RequirementType::Guideline,
            },
            domain: match i % 5 {
                0 => "encryption".to_string(),
                1 => "access_control".to_string(),
                2 => "monitoring".to_string(),
                3 => "network_security".to_string(),
                _ => "compliance".to_string(),
            },
            priority: match i % 4 {
                0 => Priority::Critical,
                1 => Priority::High,
                2 => Priority::Medium,
                _ => Priority::Low,
            },
            cross_references: Vec::new(),
            created_at: now,
        }
    }).collect();
    
    ProcessedDocument {
        id: doc_id,
        title: "Large Scale Compliance Framework".to_string(),
        version: "1.0".to_string(),
        doc_type: DocumentType::Nist,
        hierarchy: DocumentHierarchy {
            sections,
            total_sections: 50,
            max_depth: 1,
        },
        requirements,
        cross_references: Vec::new(),
        metadata: DocumentMetadata {
            title: "Large Scale Compliance Framework".to_string(),
            version: "1.0".to_string(),
            publication_date: Some(now),
            author: Some("Scale Test Suite".to_string()),
            page_count: 500,
            word_count: 250000,
        },
        created_at: now,
    }
}