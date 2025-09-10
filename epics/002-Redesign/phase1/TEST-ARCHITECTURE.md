# Phase 1 Test Architecture: Neurosymbolic Foundation
## London TDD Methodology - Test-First Development

**Document Version**: 1.0  
**Date**: January 10, 2025  
**Phase**: Neurosymbolic Foundation Testing Strategy  
**Methodology**: London TDD with SPARC Integration  

---

## 🎯 TESTING STRATEGY OVERVIEW

**Philosophy**: London TDD (Interaction-based testing) with comprehensive unit, integration, and end-to-end coverage for neurosymbolic components.

**Test-First Approach**: Write tests before implementation to drive design and ensure requirements are met.

**Coverage Target**: >90% code coverage with 100% critical path coverage.

---

## 🏗️ TEST ARCHITECTURE LAYERS

### Layer 1: Unit Tests (Isolated Component Testing)

#### UT-1: Symbolic Reasoning Engine Tests

**File**: `tests/unit/symbolic/datalog_engine_test.rs`
```rust
#[cfg(test)]
mod datalog_engine_tests {
    use crate::symbolic::datalog::DatalogEngine;

    #[tokio::test]
    async fn test_simple_rule_compilation() {
        // GIVEN: A simple requirement rule
        let rule = "requires_encryption(Data) :- cardholder_data(Data).";
        let engine = DatalogEngine::new().await.unwrap();
        
        // WHEN: Rule is compiled
        let result = engine.compile_rule(rule).await;
        
        // THEN: Rule compiles successfully
        assert!(result.is_ok());
        assert!(engine.has_rule("requires_encryption").await);
    }

    #[tokio::test]
    async fn test_query_execution_performance() {
        // GIVEN: An engine with compiled rules
        let mut engine = DatalogEngine::new().await.unwrap();
        engine.load_test_rules().await.unwrap();
        
        // WHEN: Query is executed
        let start = std::time::Instant::now();
        let result = engine.query("requires_encryption(X)").await.unwrap();
        let duration = start.elapsed();
        
        // THEN: Query executes within performance target
        assert!(duration.as_millis() < 100); // <100ms requirement
        assert!(!result.is_empty());
    }
}
```

**File**: `tests/unit/symbolic/prolog_engine_test.rs`
```rust
#[cfg(test)]
mod prolog_engine_tests {
    use crate::symbolic::prolog::PrologEngine;

    #[tokio::test] 
    async fn test_complex_inference() {
        // GIVEN: A Prolog engine with complex rules
        let mut engine = PrologEngine::new().await.unwrap();
        engine.load_compliance_rules().await.unwrap();
        
        // WHEN: Complex query is executed
        let query = "compliant(System, pci_dss) :- implements(System, encryption).";
        let result = engine.query(query).await.unwrap();
        
        // THEN: Inference produces proof chain
        assert!(result.has_proof_chain());
        assert!(result.confidence() > 0.95);
    }

    #[tokio::test]
    async fn test_proof_chain_generation() {
        // GIVEN: Engine with linked rules
        let engine = setup_test_prolog_engine().await;
        
        // WHEN: Query requires multi-step reasoning
        let result = engine.query_with_proof("requires_encryption(cardholder_data)").await;
        
        // THEN: Complete proof chain is generated
        assert!(result.proof_steps.len() > 1);
        assert!(result.all_steps_valid());
        assert!(result.citations_complete());
    }
}
```

#### UT-2: Graph Database Tests

**File**: `tests/unit/graph/neo4j_client_test.rs`
```rust
#[cfg(test)]
mod neo4j_client_tests {
    use crate::graph::neo4j_client::Neo4jClient;
    use crate::models::{Requirement, RelationshipType};

    #[tokio::test]
    async fn test_requirement_node_creation() {
        // GIVEN: Neo4j client connected to test database
        let client = Neo4jClient::new_test_client().await.unwrap();
        
        // WHEN: Requirement node is created
        let req = Requirement::new("REQ-1", "Test requirement", "3.2.1", "MUST");
        let result = client.create_requirement_node(&req).await;
        
        // THEN: Node is created successfully
        assert!(result.is_ok());
        let node_id = result.unwrap();
        assert!(client.node_exists(node_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_relationship_creation() {
        // GIVEN: Two requirement nodes
        let client = Neo4jClient::new_test_client().await.unwrap();
        let req1_id = client.create_test_requirement("REQ-1").await.unwrap();
        let req2_id = client.create_test_requirement("REQ-2").await.unwrap();
        
        // WHEN: Relationship is created
        let result = client.create_relationship(
            req1_id, req2_id, RelationshipType::References
        ).await;
        
        // THEN: Relationship exists in graph
        assert!(result.is_ok());
        assert!(client.relationship_exists(req1_id, req2_id, "REFERENCES").await.unwrap());
    }

    #[tokio::test]
    async fn test_graph_traversal_performance() {
        // GIVEN: Graph with 100+ nodes and relationships
        let client = setup_performance_test_graph().await;
        
        // WHEN: 3-hop traversal query is executed
        let start = std::time::Instant::now();
        let results = client.traverse_requirements("REQ-1", 3).await.unwrap();
        let duration = start.elapsed();
        
        // THEN: Query completes within performance target
        assert!(duration.as_millis() < 200); // <200ms requirement
        assert!(!results.is_empty());
    }
}
```

#### UT-3: Smart Ingestion Pipeline Tests

**File**: `tests/unit/ingestion/document_classifier_test.rs`
```rust
#[cfg(test)]
mod document_classifier_tests {
    use crate::ingestion::DocumentClassifier;
    use crate::models::DocumentType;

    #[tokio::test]
    async fn test_pci_dss_classification() {
        // GIVEN: Trained document classifier
        let classifier = DocumentClassifier::load_trained().await.unwrap();
        let pci_sample = load_test_document("pci_dss_sample.pdf").await;
        
        // WHEN: Document is classified
        let result = classifier.classify(&pci_sample).await.unwrap();
        
        // THEN: Document is correctly classified as PCI-DSS
        assert_eq!(result.document_type, DocumentType::PciDss);
        assert!(result.confidence > 0.90); // >90% accuracy requirement
    }

    #[tokio::test]
    async fn test_section_type_identification() {
        // GIVEN: Document with multiple section types
        let classifier = DocumentClassifier::load_trained().await.unwrap();
        let doc_sections = load_test_sections("mixed_sections.json").await;
        
        // WHEN: Sections are classified
        let results = classifier.classify_sections(&doc_sections).await.unwrap();
        
        // THEN: All section types are correctly identified
        assert!(results.iter().any(|r| r.section_type == SectionType::Requirements));
        assert!(results.iter().any(|r| r.section_type == SectionType::Definitions));
        assert!(results.iter().all(|r| r.confidence > 0.85));
    }
}
```

**File**: `tests/unit/ingestion/requirement_extractor_test.rs`
```rust
#[cfg(test)]
mod requirement_extractor_tests {
    use crate::ingestion::RequirementExtractor;

    #[tokio::test]
    async fn test_requirement_extraction_completeness() {
        // GIVEN: Section with known requirements
        let extractor = RequirementExtractor::new().await.unwrap();
        let section = load_test_section("pci_section_3_2.txt").await;
        let expected_count = 5; // Known number of requirements in test section
        
        // WHEN: Requirements are extracted
        let results = extractor.extract_requirements(&section).await.unwrap();
        
        // THEN: All requirements are extracted
        assert_eq!(results.len(), expected_count); // 100% requirement coverage
        assert!(results.iter().all(|r| r.is_valid()));
    }

    #[tokio::test]
    async fn test_cross_reference_identification() {
        // GIVEN: Section with cross-references
        let extractor = RequirementExtractor::new().await.unwrap();
        let section = load_test_section("section_with_refs.txt").await;
        
        // WHEN: Cross-references are identified
        let results = extractor.extract_cross_references(&section).await.unwrap();
        
        // THEN: All cross-references are captured
        assert!(results.contains_reference("3.2.1"));
        assert!(results.contains_reference("Appendix A"));
        assert!(results.accuracy() > 0.98); // >98% cross-reference accuracy
    }
}
```

### Layer 2: Integration Tests (Component Interaction Testing)

#### IT-1: Symbolic-Graph Integration

**File**: `tests/integration/symbolic_graph_integration_test.rs`
```rust
#[cfg(test)]
mod symbolic_graph_integration_tests {
    use crate::symbolic::DatalogEngine;
    use crate::graph::Neo4jClient;
    use crate::integration::SymbolicGraphBridge;

    #[tokio::test]
    async fn test_requirement_rule_to_graph_sync() {
        // GIVEN: Symbolic engine with rules and graph database
        let datalog = DatalogEngine::new().await.unwrap();
        let graph = Neo4jClient::new_test_client().await.unwrap();
        let bridge = SymbolicGraphBridge::new(datalog, graph).await.unwrap();
        
        // WHEN: Requirement rule is added to symbolic engine
        let rule = "requires_encryption(Data) :- cardholder_data(Data).";
        bridge.add_rule_with_graph_sync(rule).await.unwrap();
        
        // THEN: Corresponding nodes exist in graph
        assert!(bridge.graph().node_exists_for_rule("requires_encryption").await.unwrap());
        assert!(bridge.symbolic().has_rule("requires_encryption").await);
    }

    #[tokio::test] 
    async fn test_graph_relationship_to_logic_sync() {
        // GIVEN: Bridge with test data
        let bridge = setup_test_bridge().await;
        
        // WHEN: Graph relationship is created
        bridge.create_graph_relationship("REQ-1", "REQ-2", "DEPENDS_ON").await.unwrap();
        
        // THEN: Logic rule reflects relationship
        let query_result = bridge.symbolic().query("depends_on('REQ-1', 'REQ-2')").await.unwrap();
        assert!(!query_result.is_empty());
    }
}
```

#### IT-2: Ingestion-Storage Integration

**File**: `tests/integration/ingestion_storage_integration_test.rs`
```rust
#[cfg(test)]
mod ingestion_storage_integration_tests {
    use crate::ingestion::SmartIngestionPipeline;
    use crate::storage::StorageManager;

    #[tokio::test]
    async fn test_end_to_end_document_processing() {
        // GIVEN: Complete ingestion pipeline
        let pipeline = SmartIngestionPipeline::new_with_test_config().await.unwrap();
        let test_document = load_test_pdf("pci_dss_sample.pdf").await;
        
        // WHEN: Document is processed through pipeline
        let result = pipeline.process_document(test_document).await.unwrap();
        
        // THEN: All components are populated correctly
        assert!(result.datalog_rules.len() > 0);
        assert!(result.graph_nodes.len() > 0);
        assert!(result.classifications.len() > 0);
        assert!(result.cross_references.len() > 0);
    }

    #[tokio::test]
    async fn test_storage_persistence_integration() {
        // GIVEN: Processed document
        let pipeline = SmartIngestionPipeline::new_with_test_config().await.unwrap();
        let storage = StorageManager::new_test().await.unwrap();
        let doc = load_and_process_test_document().await;
        
        // WHEN: Results are persisted
        storage.persist_processed_document(&doc).await.unwrap();
        
        // THEN: Data is retrievable from all storage systems
        assert!(storage.mongodb().document_exists(doc.id).await.unwrap());
        assert!(storage.neo4j().graph_exists_for_document(doc.id).await.unwrap());
        assert!(storage.datalog().rules_exist_for_document(doc.id).await.unwrap());
    }
}
```

#### IT-3: DAA Orchestrator Integration

**File**: `tests/integration/daa_neurosymbolic_integration_test.rs`
```rust
#[cfg(test)]
mod daa_neurosymbolic_integration_tests {
    use crate::integration::SystemIntegration;
    use crate::daa_orchestrator::DAAOrchestrator;

    #[tokio::test]
    async fn test_neurosymbolic_components_registration() {
        // GIVEN: System integration with DAA orchestrator
        let system = SystemIntegration::new_test_config().await.unwrap();
        
        // WHEN: Neurosymbolic components are registered
        system.register_neurosymbolic_components().await.unwrap();
        
        // THEN: All components are registered with DAA
        let orchestrator = system.daa_orchestrator();
        assert!(orchestrator.has_component("datalog-engine").await);
        assert!(orchestrator.has_component("neo4j-client").await);
        assert!(orchestrator.has_component("smart-ingestion").await);
    }

    #[tokio::test]
    async fn test_mrap_symbolic_reasoning_integration() {
        // GIVEN: MRAP controller with symbolic reasoning capability
        let system = SystemIntegration::new_test_config().await.unwrap();
        let mrap = system.mrap_controller();
        
        // WHEN: MRAP loop processes symbolic query
        let query = "What encryption is required for cardholder data?";
        let result = mrap.execute_mrap_loop(query).await.unwrap();
        
        // THEN: Response includes symbolic reasoning results
        assert!(result.used_symbolic_reasoning());
        assert!(result.has_proof_chain());
        assert!(result.processing_time_ms < 1000); // <1s requirement
    }
}
```

### Layer 3: End-to-End Tests (System-Level Testing)

#### E2E-1: Complete Neurosymbolic Pipeline

**File**: `tests/e2e/neurosymbolic_pipeline_e2e_test.rs`
```rust
#[cfg(test)]
mod neurosymbolic_pipeline_e2e_tests {
    use crate::test_utils::TestDocumentLoader;
    use crate::models::QueryRequest;

    #[tokio::test]
    async fn test_complete_pci_dss_processing_pipeline() {
        // GIVEN: Complete system with test PCI-DSS document
        let system = setup_complete_test_system().await;
        let pci_document = TestDocumentLoader::load_pci_dss_v4().await;
        
        // WHEN: Document is ingested and processed
        system.ingest_document(pci_document).await.unwrap();
        
        // AND: Symbolic query is executed
        let query = QueryRequest::new("What are the encryption requirements for stored cardholder data?");
        let response = system.process_query(query).await.unwrap();
        
        // THEN: Response demonstrates full neurosymbolic capability
        assert!(response.used_symbolic_reasoning);
        assert!(response.proof_chain.is_some());
        assert!(response.citations.len() > 0);
        assert!(response.confidence > 0.95);
        assert!(response.processing_time_ms < 1000);
        
        // AND: Response contains accurate PCI-DSS information
        assert!(response.response.contains("3.2"));
        assert!(response.response.contains("cardholder data"));
        assert!(response.response.contains("encryption"));
    }

    #[tokio::test]
    async fn test_multi_document_cross_reference_resolution() {
        // GIVEN: System with multiple related standards
        let system = setup_complete_test_system().await;
        system.ingest_document(TestDocumentLoader::load_pci_dss_v4().await).unwrap();
        system.ingest_document(TestDocumentLoader::load_iso_27001().await).unwrap();
        
        // WHEN: Query involves cross-document reasoning
        let query = QueryRequest::new("How do PCI-DSS encryption requirements relate to ISO 27001 controls?");
        let response = system.process_query(query).await.unwrap();
        
        // THEN: System correctly identifies relationships across documents
        assert!(response.citations.iter().any(|c| c.source.contains("PCI-DSS")));
        assert!(response.citations.iter().any(|c| c.source.contains("ISO-27001")));
        assert!(response.has_cross_document_relationships());
    }
}
```

#### E2E-2: Performance and Scale Testing

**File**: `tests/e2e/performance_scale_e2e_test.rs`
```rust
#[cfg(test)]
mod performance_scale_e2e_tests {
    use tokio::time::Instant;

    #[tokio::test]
    async fn test_concurrent_query_performance() {
        // GIVEN: System loaded with test data
        let system = setup_performance_test_system().await;
        
        // WHEN: 100 concurrent queries are executed
        let queries: Vec<_> = (0..100)
            .map(|i| format!("Test query {}", i))
            .map(|q| QueryRequest::new(&q))
            .collect();
        
        let start = Instant::now();
        let results = futures::future::join_all(
            queries.into_iter().map(|q| system.process_query(q))
        ).await;
        let total_duration = start.elapsed();
        
        // THEN: All queries complete successfully within time limits
        assert!(results.iter().all(|r| r.is_ok()));
        assert!(total_duration.as_secs() < 30); // Reasonable concurrent performance
        
        let successful_results: Vec<_> = results.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
        let avg_response_time = successful_results.iter()
            .map(|r| r.processing_time_ms)
            .sum::<u64>() / successful_results.len() as u64;
        
        assert!(avg_response_time < 2000); // <2s average under load
    }

    #[tokio::test]
    async fn test_large_document_processing_scale() {
        // GIVEN: System and large document corpus
        let system = setup_complete_test_system().await;
        let large_documents = TestDocumentLoader::load_complete_standards_library().await;
        
        // WHEN: All documents are processed
        let start = Instant::now();
        for doc in large_documents {
            system.ingest_document(doc).await.unwrap();
        }
        let processing_duration = start.elapsed();
        
        // THEN: Processing completes within reasonable time
        let total_pages = system.get_total_processed_pages().await;
        let pages_per_second = total_pages as f64 / processing_duration.as_secs_f64();
        
        assert!(pages_per_second >= 2.0); // Meets 2-5 pages/second target
        assert!(pages_per_second <= 10.0); // Reasonable upper bound
    }
}
```

---

## 🧪 TEST DATA MANAGEMENT

### Test Data Sets

#### TD-1: Document Test Corpus
- **PCI-DSS v4.0**: Complete standard with all sections and requirements
- **ISO-27001**: Sample sections with requirements and cross-references  
- **SOC2 Type II**: Subset with control descriptions and relationships
- **NIST Framework**: Selected sections with implementation guidance
- **Synthetic Documents**: Generated test documents with known requirements

#### TD-2: Query Test Sets
- **Simple Symbolic Queries**: Direct requirement lookups
- **Complex Reasoning Queries**: Multi-step inference requirements
- **Cross-Reference Queries**: Relationship traversal requirements
- **Performance Test Queries**: High-volume concurrent query sets
- **Edge Case Queries**: Boundary conditions and error scenarios

#### TD-3: Expected Results
- **Ground Truth Answers**: Manually verified correct responses
- **Performance Baselines**: Established timing benchmarks
- **Accuracy Measurements**: Precision and recall calculations
- **Coverage Metrics**: Requirement extraction completeness validation

### Test Environment Management

#### TE-1: Database Test Isolation
```rust
// Test-specific database setup
pub async fn setup_test_neo4j() -> Neo4jClient {
    let test_db_name = format!("test_{}", uuid::Uuid::new_v4());
    Neo4jClient::new_with_database(&test_db_name).await.unwrap()
}

pub async fn cleanup_test_neo4j(client: Neo4jClient) {
    client.drop_database().await.unwrap();
}
```

#### TE-2: Symbolic Engine Test Isolation  
```rust
// Isolated symbolic engine for tests
pub async fn setup_test_datalog() -> DatalogEngine {
    let mut engine = DatalogEngine::new().await.unwrap();
    engine.load_test_rules_only().await.unwrap();
    engine
}
```

---

## 📊 TEST AUTOMATION & CI/CD

### Continuous Testing Pipeline

#### CT-1: Pre-commit Testing
- **Unit Tests**: All unit tests must pass before commit
- **Code Coverage**: Maintain >90% coverage on new code
- **Performance Regression**: Quick performance smoke tests
- **Code Quality**: Clippy lints and formatting checks

#### CT-2: Integration Testing
- **Component Integration**: Full integration test suite
- **Database Integration**: Tests with real database instances
- **End-to-End Smoke Tests**: Basic functionality verification
- **Performance Baselines**: Regression testing against benchmarks

#### CT-3: Nightly Testing
- **Full E2E Suite**: Complete end-to-end test execution
- **Performance Benchmarks**: Detailed performance analysis
- **Stress Testing**: High-load and concurrent user simulation
- **Memory and Resource Usage**: Resource consumption validation

### Test Reporting and Metrics

#### TR-1: Coverage Reports
- **Code Coverage**: Line, branch, and function coverage metrics
- **Test Coverage**: Requirement coverage mapping
- **Performance Coverage**: Benchmark coverage across components
- **Integration Coverage**: Inter-component interaction testing

#### TR-2: Quality Metrics
- **Test Execution Time**: Track test suite performance
- **Test Reliability**: Flaky test identification and resolution
- **Bug Detection Rate**: Tests catching regressions
- **Performance Regression Detection**: Automated performance analysis

---

## 🎯 PHASE 1 TESTING DELIVERABLES

### Primary Test Artifacts
1. **Unit Test Suite**: Complete unit tests for all neurosymbolic components
2. **Integration Test Suite**: Component interaction and system integration tests
3. **E2E Test Suite**: End-to-end system functionality and performance tests
4. **Test Data Corpus**: Comprehensive test document and query collections
5. **Performance Benchmarks**: Baseline metrics and regression tests
6. **Test Automation Pipeline**: CI/CD integration with automated testing

### Test Documentation
1. **Test Plan**: This document with detailed testing strategy
2. **Test Execution Reports**: Results from all test categories
3. **Performance Analysis**: Detailed performance benchmark results
4. **Coverage Reports**: Code and requirement coverage analysis
5. **Bug Reports**: Issues identified and resolution tracking
6. **Test Maintenance Guide**: Procedures for updating and maintaining tests

### Quality Gates
1. **Unit Test Gate**: >95% unit test pass rate required
2. **Integration Test Gate**: >90% integration test pass rate required
3. **Performance Gate**: All performance targets must be met
4. **Coverage Gate**: >90% code coverage required for new components
5. **Regression Gate**: No performance regressions allowed

---

*Test Architecture complete for Phase 1: Neurosymbolic Foundation*  
*Ready for implementation with test-first London TDD methodology*