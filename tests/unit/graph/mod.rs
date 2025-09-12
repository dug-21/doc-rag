pub mod neo4j_client_tests;
pub mod schema_tests;
pub mod performance_tests;
pub mod integration_tests;

use chrono::{DateTime, Utc};
use graph::{
    models::*,
    neo4j::{Neo4jClient, Neo4jConfig},
    GraphConfig, GraphDatabase,
};
use std::sync::Arc;
use uuid::Uuid;

/// Test utilities and fixtures for graph database testing
pub struct GraphTestFixtures;

impl GraphTestFixtures {
    /// Create test document for graph operations
    pub fn create_test_document() -> ProcessedDocument {
        let doc_id = Uuid::new_v4();
        let now = Utc::now();

        ProcessedDocument {
            id: doc_id,
            title: "PCI-DSS Compliance Standard".to_string(),
            version: "4.0".to_string(),
            doc_type: DocumentType::PciDss,
            hierarchy: DocumentHierarchy {
                sections: vec![
                    Self::create_test_section("3.2", "Data Protection Requirements"),
                    Self::create_test_section("3.3", "Encryption Requirements"),
                ],
                total_sections: 2,
                max_depth: 2,
            },
            requirements: vec![
                Self::create_test_requirement("REQ-3.2.1", "3.2", "Cardholder data must be encrypted at rest"),
                Self::create_test_requirement("REQ-3.3.1", "3.3", "Encryption keys must be managed securely"),
            ],
            cross_references: vec![
                CrossReference {
                    from_requirement: "REQ-3.2.1".to_string(),
                    to_requirement: "REQ-3.3.1".to_string(),
                    reference_type: ReferenceType::Direct,
                    context: "Encryption implementation".to_string(),
                    confidence: 0.95,
                },
            ],
            metadata: DocumentMetadata {
                title: "PCI-DSS Compliance Standard".to_string(),
                version: "4.0".to_string(),
                publication_date: Some(now),
                author: Some("PCI Security Standards Council".to_string()),
                page_count: 100,
                word_count: 50000,
            },
            created_at: now,
        }
    }

    /// Create test section
    pub fn create_test_section(number: &str, title: &str) -> Section {
        Section {
            id: format!("SEC-{}", number),
            number: number.to_string(),
            title: title.to_string(),
            text: format!("This is section {} about {}", number, title),
            page_range: (10, 15),
            subsections: Vec::new(),
            parent_id: None,
            section_type: SectionType::Requirements,
        }
    }

    /// Create test requirement
    pub fn create_test_requirement(id: &str, section: &str, text: &str) -> Requirement {
        Requirement {
            id: id.to_string(),
            text: text.to_string(),
            section: format!("SEC-{}", section),
            requirement_type: RequirementType::Must,
            domain: "encryption".to_string(),
            priority: Priority::High,
            cross_references: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Create test graph configuration
    pub fn create_test_config() -> Neo4jConfig {
        Neo4jConfig {
            base: GraphConfig {
                uri: "bolt://localhost:7687".to_string(),
                username: "neo4j".to_string(),
                password: "neo4j_password".to_string(),
                max_connections: 4,
                connection_timeout_ms: 5000,
                query_timeout_ms: 10000,
                enable_metrics: true,
                enable_cache: true,
                cache_ttl_seconds: 300,
            },
            database: "neo4j".to_string(),
            routing: false,
            encrypted: false,
            trust: "TRUST_ALL_CERTIFICATES".to_string(),
            user_agent: "graph-test/1.0".to_string(),
        }
    }

    /// Create multiple test requirements for performance testing
    pub fn create_bulk_requirements(count: usize) -> Vec<Requirement> {
        (0..count)
            .map(|i| {
                Self::create_test_requirement(
                    &format!("REQ-BULK-{:04}", i),
                    "BULK",
                    &format!("Bulk requirement number {}", i),
                )
            })
            .collect()
    }

    /// Create test filter for requirement searches
    pub fn create_test_filter() -> RequirementFilter {
        RequirementFilter {
            section: Some("SEC-3.2".to_string()),
            requirement_type: Some(RequirementType::Must),
            domain: Some("encryption".to_string()),
            priority: Some(Priority::High),
            text_contains: Some("encrypted".to_string()),
        }
    }
}

/// Test database connection utilities
pub struct TestDatabaseManager;

impl TestDatabaseManager {
    /// Check if test Neo4j instance is available
    pub async fn is_neo4j_available() -> bool {
        let config = GraphTestFixtures::create_test_config();
        match Neo4jClient::new(config).await {
            Ok(client) => client.health_check().await.unwrap_or(false),
            Err(_) => false,
        }
    }

    /// Clean test database
    pub async fn clean_test_database(client: &Neo4jClient) -> Result<(), Box<dyn std::error::Error>> {
        // Clean all test data
        let cleanup_queries = vec![
            "MATCH (n) WHERE n.id STARTS WITH 'TEST-' OR n.id STARTS WITH 'REQ-' OR n.id STARTS WITH 'SEC-' DETACH DELETE n",
            "MATCH (n:Document) WHERE n.title CONTAINS 'Test' DETACH DELETE n",
        ];

        for query_str in cleanup_queries {
            // Note: This is a simplified cleanup - in a real implementation,
            // we'd need to properly construct and execute the queries
        }

        Ok(())
    }

    /// Setup test data
    pub async fn setup_test_data(client: &Neo4jClient) -> Result<ProcessedDocument, Box<dyn std::error::Error>> {
        let test_doc = GraphTestFixtures::create_test_document();
        let _graph = client.create_document_hierarchy(&test_doc).await?;
        Ok(test_doc)
    }
}