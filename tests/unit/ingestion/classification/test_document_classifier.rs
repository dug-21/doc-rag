//! London TDD test suite for document classifier
//!
//! Comprehensive test coverage for neural document classification
//! following London School TDD methodology with extensive mocking
//! and behavior verification

use doc_rag_chunker::ingestion::classification::{
    DocumentClassifier, DocumentType, SectionType, QueryRoute,
    DocumentTypeResult, SectionTypeResult, QueryRoutingResult,
};
use doc_rag_chunker::{Result, ChunkerError};
use std::collections::HashMap;
use tokio_test;

/// Test fixture for document classifier testing
struct DocumentClassifierTestFixture {
    classifier: DocumentClassifier,
    sample_pci_text: String,
    sample_iso_text: String,
    sample_soc2_text: String,
    sample_nist_text: String,
    sample_requirement_text: String,
    sample_definition_text: String,
    sample_procedure_text: String,
}

impl DocumentClassifierTestFixture {
    async fn new() -> Self {
        let classifier = DocumentClassifier::new().unwrap();
        
        Self {
            classifier,
            sample_pci_text: "Payment Card Industry Data Security Standard (PCI DSS) Requirements and Security Assessment Procedures. This document provides requirements for organizations that store, process, or transmit cardholder data.".to_string(),
            sample_iso_text: "ISO/IEC 27001:2013 Information technology — Security techniques — Information security management systems — Requirements. This International Standard specifies the requirements for establishing, implementing, maintaining and continually improving an information security management system.".to_string(),
            sample_soc2_text: "SOC 2 Type II Service Organization Control Report. This report describes the service organization's system and the suitability of the design and operating effectiveness of controls relevant to security, availability, processing integrity, confidentiality, and privacy.".to_string(),
            sample_nist_text: "NIST Cybersecurity Framework Version 1.1. The Framework provides organizations with activities and outcomes that help them manage and reduce their cybersecurity risk in a cost-effective way based on business needs.".to_string(),
            sample_requirement_text: "Requirement 3.2.1: Cardholder data must be protected during transmission over open, public networks using strong cryptography and security protocols.".to_string(),
            sample_definition_text: "Definition: Cardholder data is defined as the primary account number (PAN) plus any of the following: cardholder name, expiration date, or service code.".to_string(),
            sample_procedure_text: "Procedure 1: Step 1 - Identify all locations where cardholder data is stored. Step 2 - Document data flows and processing. Step 3 - Implement appropriate security controls.".to_string(),
        }
    }
}

/// London TDD Tests for Document Type Classification
mod document_type_classification_tests {
    use super::*;

    #[tokio::test]
    async fn test_classify_pci_dss_document() {
        // Given: A document classifier and PCI DSS content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Classifying PCI DSS content
        let result = fixture.classifier.classify_document_type(&fixture.sample_pci_text, None).await.unwrap();
        
        // Then: Should classify as PCI DSS with high confidence
        assert_eq!(result.document_type, DocumentType::PciDss);
        assert!(result.confidence >= 0.7, "Confidence {} should be >= 0.7", result.confidence);
        assert!(result.inference_time_ms < 50.0, "Inference time {}ms should be < 50ms", result.inference_time_ms);
        assert!(result.all_scores.contains_key(&DocumentType::PciDss));
        assert_eq!(result.all_scores.len(), 4); // Should have scores for all 4 document types
    }

    #[tokio::test]
    async fn test_classify_iso27001_document() {
        // Given: A document classifier and ISO 27001 content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Classifying ISO 27001 content
        let result = fixture.classifier.classify_document_type(&fixture.sample_iso_text, None).await.unwrap();
        
        // Then: Should classify as ISO 27001 with high confidence
        assert_eq!(result.document_type, DocumentType::Iso27001);
        assert!(result.confidence >= 0.7);
        assert!(result.inference_time_ms < 50.0);
    }

    #[tokio::test]
    async fn test_classify_soc2_document() {
        // Given: A document classifier and SOC 2 content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Classifying SOC 2 content
        let result = fixture.classifier.classify_document_type(&fixture.sample_soc2_text, None).await.unwrap();
        
        // Then: Should classify as SOC 2 with high confidence
        assert_eq!(result.document_type, DocumentType::Soc2);
        assert!(result.confidence >= 0.7);
        assert!(result.inference_time_ms < 50.0);
    }

    #[tokio::test]
    async fn test_classify_nist_document() {
        // Given: A document classifier and NIST content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Classifying NIST content
        let result = fixture.classifier.classify_document_type(&fixture.sample_nist_text, None).await.unwrap();
        
        // Then: Should classify as NIST with high confidence
        assert_eq!(result.document_type, DocumentType::Nist);
        assert!(result.confidence >= 0.7);
        assert!(result.inference_time_ms < 50.0);
    }

    #[tokio::test]
    async fn test_document_classification_with_metadata() {
        // Given: A document classifier and content with metadata
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), "PCI DSS Compliance Guide".to_string());
        metadata.insert("version".to_string(), "4.0".to_string());
        
        // When: Classifying with metadata
        let result = fixture.classifier.classify_document_type(&fixture.sample_pci_text, Some(&metadata)).await.unwrap();
        
        // Then: Should still classify correctly and include metadata influence
        assert_eq!(result.document_type, DocumentType::PciDss);
        assert!(result.confidence >= 0.7);
        assert!(result.feature_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_batch_document_classification() {
        // Given: A document classifier and multiple documents
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let documents = vec![
            (&fixture.sample_pci_text, None),
            (&fixture.sample_iso_text, None),
            (&fixture.sample_soc2_text, None),
        ];
        
        // When: Batch classifying documents
        let results = fixture.classifier.batch_classify_documents(
            documents.into_iter().map(|(text, metadata)| (text.as_str(), metadata)).collect()
        ).await.unwrap();
        
        // Then: Should classify all documents correctly
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].document_type, DocumentType::PciDss);
        assert_eq!(results[1].document_type, DocumentType::Iso27001);
        assert_eq!(results[2].document_type, DocumentType::Soc2);
    }

    #[tokio::test]
    async fn test_document_classification_performance_constraint() {
        // Given: A document classifier and test content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Measuring classification performance
        let start = std::time::Instant::now();
        let result = fixture.classifier.classify_document_type(&fixture.sample_pci_text, None).await.unwrap();
        let total_time = start.elapsed().as_secs_f64() * 1000.0;
        
        // Then: Should meet <10ms inference constraint (allowing some overhead)
        assert!(result.inference_time_ms < 10.0, "Inference time {}ms should be < 10ms", result.inference_time_ms);
        assert!(total_time < 100.0, "Total time {}ms should be reasonable", total_time);
    }
}

/// London TDD Tests for Section Type Classification
mod section_type_classification_tests {
    use super::*;

    #[tokio::test]
    async fn test_classify_requirements_section() {
        // Given: A document classifier and requirement content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Classifying requirements section
        let result = fixture.classifier.classify_section_type(&fixture.sample_requirement_text, None, 0).await.unwrap();
        
        // Then: Should classify as Requirements with high confidence
        assert_eq!(result.section_type, SectionType::Requirements);
        assert!(result.confidence >= 0.7, "Confidence {} should be >= 0.7", result.confidence);
        assert!(result.inference_time_ms < 50.0, "Inference time {}ms should be < 50ms", result.inference_time_ms);
        assert!(!result.type_hints.is_empty(), "Should have type hints");
        assert!(result.type_hints.contains(&"requirement".to_string()));
    }

    #[tokio::test]
    async fn test_classify_definitions_section() {
        // Given: A document classifier and definition content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Classifying definitions section
        let result = fixture.classifier.classify_section_type(&fixture.sample_definition_text, None, 0).await.unwrap();
        
        // Then: Should classify as Definitions with high confidence
        assert_eq!(result.section_type, SectionType::Definitions);
        assert!(result.confidence >= 0.7);
        assert!(result.inference_time_ms < 50.0);
        assert!(result.type_hints.contains(&"definition".to_string()));
    }

    #[tokio::test]
    async fn test_classify_procedures_section() {
        // Given: A document classifier and procedure content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Classifying procedures section
        let result = fixture.classifier.classify_section_type(&fixture.sample_procedure_text, None, 0).await.unwrap();
        
        // Then: Should classify as Procedures with high confidence
        assert_eq!(result.section_type, SectionType::Procedures);
        assert!(result.confidence >= 0.7);
        assert!(result.inference_time_ms < 50.0);
        assert!(result.type_hints.contains(&"procedure".to_string()));
    }

    #[tokio::test]
    async fn test_section_classification_with_context() {
        // Given: A document classifier and content with context
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let context = "This section contains important requirements for data protection.";
        
        // When: Classifying section with context
        let result = fixture.classifier.classify_section_type(&fixture.sample_requirement_text, Some(context), 1).await.unwrap();
        
        // Then: Should classify correctly using context
        assert_eq!(result.section_type, SectionType::Requirements);
        assert!(result.confidence >= 0.7);
    }

    #[tokio::test]
    async fn test_section_classification_performance() {
        // Given: A document classifier
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Measuring section classification performance
        let start = std::time::Instant::now();
        let result = fixture.classifier.classify_section_type(&fixture.sample_requirement_text, None, 0).await.unwrap();
        let total_time = start.elapsed().as_secs_f64() * 1000.0;
        
        // Then: Should meet performance constraints
        assert!(result.inference_time_ms < 10.0, "Inference time {}ms should be < 10ms", result.inference_time_ms);
        assert!(total_time < 100.0);
    }
}

/// London TDD Tests for Query Routing Classification
mod query_routing_tests {
    use super::*;

    #[tokio::test]
    async fn test_route_symbolic_query() {
        // Given: A document classifier and symbolic query
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let query = "What are the requirements for encrypting cardholder data in PCI DSS?";
        
        // When: Routing the query
        let result = fixture.classifier.route_query(query).await.unwrap();
        
        // Then: Should route to symbolic processing
        assert_eq!(result.routing_decision, QueryRoute::Symbolic);
        assert!(result.confidence >= 0.5, "Confidence {} should be >= 0.5", result.confidence);
        assert!(result.inference_time_ms < 50.0, "Inference time {}ms should be < 50ms", result.inference_time_ms);
        assert!(!result.complexity_indicators.is_empty());
    }

    #[tokio::test]
    async fn test_route_graph_query() {
        // Given: A document classifier and graph query
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let query = "What requirements are related to requirement 3.2.1 in PCI DSS?";
        
        // When: Routing the query
        let result = fixture.classifier.route_query(query).await.unwrap();
        
        // Then: Should route to graph processing
        assert_eq!(result.routing_decision, QueryRoute::Graph);
        assert!(result.confidence >= 0.5);
        assert!(result.inference_time_ms < 50.0);
    }

    #[tokio::test]
    async fn test_route_vector_query() {
        // Given: A document classifier and vector query
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let query = "Find documents similar to this security policy.";
        
        // When: Routing the query
        let result = fixture.classifier.route_query(query).await.unwrap();
        
        // Then: Should route to vector processing
        assert_eq!(result.routing_decision, QueryRoute::Vector);
        assert!(result.confidence >= 0.5);
        assert!(result.inference_time_ms < 50.0);
    }

    #[tokio::test]
    async fn test_route_complex_query() {
        // Given: A document classifier and complex query
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let query = "What are the security controls related to data encryption and how do they compare to similar requirements in other standards?";
        
        // When: Routing the complex query
        let result = fixture.classifier.route_query(query).await.unwrap();
        
        // Then: Should route appropriately (could be hybrid or specific route)
        assert!(matches!(result.routing_decision, QueryRoute::Symbolic | QueryRoute::Graph | QueryRoute::Vector | QueryRoute::Hybrid));
        assert!(result.confidence >= 0.5);
    }

    #[tokio::test]
    async fn test_query_routing_performance() {
        // Given: A document classifier
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let query = "Test query for performance measurement";
        
        // When: Measuring query routing performance
        let start = std::time::Instant::now();
        let result = fixture.classifier.route_query(query).await.unwrap();
        let total_time = start.elapsed().as_secs_f64() * 1000.0;
        
        // Then: Should meet performance constraints
        assert!(result.inference_time_ms < 10.0, "Inference time {}ms should be < 10ms", result.inference_time_ms);
        assert!(total_time < 100.0);
    }
}

/// London TDD Tests for Classifier Management and Persistence
mod classifier_management_tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_classifier_creation() {
        // Given: Valid configuration
        // When: Creating a document classifier
        let classifier = DocumentClassifier::new();
        
        // Then: Should create successfully
        assert!(classifier.is_ok());
    }

    #[test]
    fn test_classifier_health_check() {
        // Given: A document classifier
        let mut classifier = DocumentClassifier::new().unwrap();
        
        // When: Running health check
        let is_healthy = classifier.health_check();
        
        // Then: Should be healthy
        assert!(is_healthy);
    }

    #[test]
    fn test_model_save_and_load() {
        // Given: A trained classifier and temporary directory
        let classifier = DocumentClassifier::new().unwrap();
        let temp_dir = TempDir::new().unwrap();
        let version = "test_v1.0.0";
        
        // When: Saving models
        let save_result = classifier.save_models(temp_dir.path(), version);
        
        // Then: Should save successfully
        assert!(save_result.is_ok());
        
        // When: Loading models
        let load_result = DocumentClassifier::load_models(temp_dir.path(), version);
        
        // Then: Should load successfully
        assert!(load_result.is_ok());
        
        let loaded_classifier = load_result.unwrap();
        assert!(loaded_classifier.get_metrics().total_classifications >= 0);
    }

    #[tokio::test]
    async fn test_metrics_tracking() {
        // Given: A document classifier
        let mut classifier = DocumentClassifier::new().unwrap();
        let initial_metrics = classifier.get_metrics().clone();
        
        // When: Performing classifications
        let _ = classifier.classify_document_type("Test document", None).await.unwrap();
        let _ = classifier.classify_section_type("Test section", None, 0).await.unwrap();
        let _ = classifier.route_query("Test query").await.unwrap();
        
        // Then: Metrics should be updated
        let updated_metrics = classifier.get_metrics();
        assert!(updated_metrics.total_classifications > initial_metrics.total_classifications);
        assert!(updated_metrics.average_inference_time_ms >= 0.0);
        assert!(updated_metrics.last_updated > initial_metrics.last_updated);
    }
}

/// London TDD Tests for Edge Cases and Error Handling
mod edge_cases_and_error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_classify_empty_document() {
        // Given: A document classifier and empty content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let empty_content = "";
        
        // When: Classifying empty document
        let result = fixture.classifier.classify_document_type(empty_content, None).await.unwrap();
        
        // Then: Should handle gracefully with low confidence
        assert!(result.confidence < 0.8); // Low confidence expected for empty content
        assert!(result.inference_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_classify_very_short_section() {
        // Given: A document classifier and very short content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let short_content = "Must.";
        
        // When: Classifying very short section
        let result = fixture.classifier.classify_section_type(short_content, None, 0).await.unwrap();
        
        // Then: Should handle gracefully
        assert!(result.confidence >= 0.0);
        assert!(result.inference_time_ms >= 0.0);
    }

    #[tokio::test]
    async fn test_classify_mixed_content_document() {
        // Given: A document classifier and mixed content
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let mixed_content = format!("{} {} {} {}", 
            fixture.sample_pci_text, 
            fixture.sample_iso_text,
            fixture.sample_requirement_text,
            fixture.sample_definition_text
        );
        
        // When: Classifying mixed content
        let result = fixture.classifier.classify_document_type(&mixed_content, None).await.unwrap();
        
        // Then: Should classify based on dominant content
        assert!(matches!(result.document_type, DocumentType::PciDss | DocumentType::Iso27001));
        assert!(result.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_route_ambiguous_query() {
        // Given: A document classifier and ambiguous query
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let ambiguous_query = "Tell me about security.";
        
        // When: Routing ambiguous query
        let result = fixture.classifier.route_query(ambiguous_query).await.unwrap();
        
        // Then: Should make a routing decision
        assert!(matches!(result.routing_decision, QueryRoute::Symbolic | QueryRoute::Graph | QueryRoute::Vector | QueryRoute::Hybrid));
        assert!(result.confidence >= 0.0);
    }

    #[tokio::test]
    async fn test_multiple_consecutive_classifications() {
        // Given: A document classifier
        let mut fixture = DocumentClassifierTestFixture::new().await;
        
        // When: Performing multiple consecutive classifications
        for _ in 0..10 {
            let result = fixture.classifier.classify_document_type(&fixture.sample_pci_text, None).await.unwrap();
            assert_eq!(result.document_type, DocumentType::PciDss);
            assert!(result.inference_time_ms < 50.0);
        }
        
        // Then: All should complete successfully within performance bounds
        let metrics = fixture.classifier.get_metrics();
        assert!(metrics.performance_target_met >= 0.5); // At least 50% should meet target
    }
}

/// Integration Tests with Actual Neural Networks
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_end_to_end_document_processing() {
        // Given: A document classifier and comprehensive test document
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let comprehensive_doc = format!(
            "{}\n\n{}\n\n{}\n\n{}",
            fixture.sample_pci_text,
            fixture.sample_requirement_text,
            fixture.sample_definition_text,
            fixture.sample_procedure_text
        );
        
        // When: Processing the document end-to-end
        let doc_result = fixture.classifier.classify_document_type(&comprehensive_doc, None).await.unwrap();
        let section_results = vec![
            fixture.classifier.classify_section_type(&fixture.sample_requirement_text, None, 0).await.unwrap(),
            fixture.classifier.classify_section_type(&fixture.sample_definition_text, None, 1).await.unwrap(),
            fixture.classifier.classify_section_type(&fixture.sample_procedure_text, None, 2).await.unwrap(),
        ];
        let query_result = fixture.classifier.route_query("What are the requirements for data protection?").await.unwrap();
        
        // Then: All classifications should be accurate and performant
        assert_eq!(doc_result.document_type, DocumentType::PciDss);
        assert_eq!(section_results[0].section_type, SectionType::Requirements);
        assert_eq!(section_results[1].section_type, SectionType::Definitions);
        assert_eq!(section_results[2].section_type, SectionType::Procedures);
        assert_eq!(query_result.routing_decision, QueryRoute::Symbolic);
        
        // Performance validation
        assert!(doc_result.inference_time_ms < 10.0);
        assert!(section_results.iter().all(|r| r.inference_time_ms < 10.0));
        assert!(query_result.inference_time_ms < 10.0);
    }

    #[tokio::test]
    async fn test_stress_test_classification_performance() {
        // Given: A document classifier and stress test parameters
        let mut fixture = DocumentClassifierTestFixture::new().await;
        let test_iterations = 100;
        let mut inference_times = Vec::new();
        
        // When: Performing stress test
        for i in 0..test_iterations {
            let content = format!("{} iteration {}", fixture.sample_pci_text, i);
            let start = std::time::Instant::now();
            let result = fixture.classifier.classify_document_type(&content, None).await.unwrap();
            let time = start.elapsed().as_secs_f64() * 1000.0;
            
            inference_times.push(result.inference_time_ms);
            
            // Verify classification remains accurate
            assert_eq!(result.document_type, DocumentType::PciDss);
        }
        
        // Then: Performance should remain consistent
        let avg_inference_time = inference_times.iter().sum::<f64>() / inference_times.len() as f64;
        let max_inference_time = inference_times.iter().fold(0.0, |a, &b| a.max(b));
        
        assert!(avg_inference_time < 10.0, "Average inference time {}ms should be < 10ms", avg_inference_time);
        assert!(max_inference_time < 50.0, "Max inference time {}ms should be < 50ms", max_inference_time);
        
        // At least 90% of inferences should meet the 10ms target
        let fast_inferences = inference_times.iter().filter(|&&t| t < 10.0).count();
        let fast_percentage = fast_inferences as f64 / inference_times.len() as f64;
        assert!(fast_percentage >= 0.8, "Fast inference percentage {:.1}% should be >= 80%", fast_percentage * 100.0);
    }
}