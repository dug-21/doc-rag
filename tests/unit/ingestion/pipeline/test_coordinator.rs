//! London TDD test suite for smart ingestion pipeline coordinator
//!
//! Comprehensive test coverage for the smart ingestion pipeline
//! following London School TDD methodology with behavior verification

use doc_rag_chunker::ingestion::pipeline::{
    SmartIngestionPipeline, PipelineConfig, ProcessedDocument, BatchProcessingResult,
    PipelineMetrics, ProcessingRecord,
};
use doc_rag_chunker::ingestion::classification::{DocumentType, SectionType, QueryRoute};
use doc_rag_chunker::Result;
use std::collections::HashMap;
use tokio_test;

/// Test fixture for smart ingestion pipeline testing
struct PipelineTestFixture {
    pipeline: SmartIngestionPipeline,
    config: PipelineConfig,
    sample_pci_document: String,
    sample_iso_document: String,
    sample_mixed_document: String,
    sample_requirements_section: String,
}

impl PipelineTestFixture {
    async fn new() -> Self {
        let config = PipelineConfig::default();
        let pipeline = SmartIngestionPipeline::new(config.clone()).await.unwrap();
        
        Self {
            pipeline,
            config,
            sample_pci_document: r#"
                Payment Card Industry Data Security Standard (PCI DSS)
                Requirements and Security Assessment Procedures
                
                # Requirements Section
                Requirement 3.2.1: Cardholder data must be protected during transmission 
                over open, public networks using strong cryptography and security protocols.
                
                Requirement 4.1: Use strong cryptography and security protocols to 
                safeguard sensitive cardholder data during transmission.
                
                # Definitions Section
                Definition: Cardholder data is defined as the primary account number (PAN) 
                plus any of the following: cardholder name, expiration date, or service code.
                
                # Procedures Section
                Procedure 1: 
                Step 1 - Identify all locations where cardholder data is stored
                Step 2 - Document data flows and processing methods
                Step 3 - Implement appropriate security controls and monitoring
            "#.to_string(),
            sample_iso_document: "ISO/IEC 27001:2013 Information Security Management Systems. This standard specifies requirements for establishing, implementing, maintaining and continually improving an information security management system.".to_string(),
            sample_mixed_document: "This document contains mixed compliance requirements from multiple standards including PCI DSS, ISO 27001, and SOC 2 Type II controls.".to_string(),
            sample_requirements_section: "Requirement 3.2.1: Organizations must implement multi-factor authentication for all user accounts accessing cardholder data environments.".to_string(),
        }
    }
    
    fn new_with_custom_config(config: PipelineConfig) -> impl std::future::Future<Output = Self> {
        async move {
            let pipeline = SmartIngestionPipeline::new(config.clone()).await.unwrap();
            
            Self {
                pipeline,
                config,
                sample_pci_document: "Sample PCI DSS document content".to_string(),
                sample_iso_document: "Sample ISO 27001 document content".to_string(),
                sample_mixed_document: "Mixed compliance content".to_string(),
                sample_requirements_section: "Sample requirements section".to_string(),
            }
        }
    }
}

/// London TDD Tests for Pipeline Creation and Initialization
mod pipeline_initialization_tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation_with_default_config() {
        // Given: Default pipeline configuration
        let config = PipelineConfig::default();
        
        // When: Creating pipeline with default config
        let result = SmartIngestionPipeline::new(config).await;
        
        // Then: Should create successfully
        assert!(result.is_ok());
        
        let pipeline = result.unwrap();
        let health = pipeline.health_check().await;
        assert!(health, "Pipeline should be healthy after creation");
    }

    #[tokio::test]
    async fn test_pipeline_creation_with_custom_config() {
        // Given: Custom pipeline configuration
        let config = PipelineConfig {
            enable_document_classification: true,
            enable_section_classification: true,
            enable_query_routing: false,
            max_processing_time_ms: 3000,
            batch_size: 5,
            confidence_threshold: 0.8,
            enable_performance_monitoring: true,
        };
        
        // When: Creating pipeline with custom config
        let result = SmartIngestionPipeline::new(config).await;
        
        // Then: Should create successfully with custom settings
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_pipeline_health_check() {
        // Given: A newly created pipeline
        let fixture = PipelineTestFixture::new().await;
        
        // When: Running health check
        let is_healthy = fixture.pipeline.health_check().await;
        
        // Then: Should be healthy
        assert!(is_healthy);
    }

    #[tokio::test]
    async fn test_initial_metrics() {
        // Given: A newly created pipeline
        let fixture = PipelineTestFixture::new().await;
        
        // When: Getting initial metrics
        let metrics = fixture.pipeline.get_metrics().await;
        
        // Then: Should have initialized metrics
        assert_eq!(metrics.total_documents, 0);
        assert_eq!(metrics.average_processing_time_ms, 0.0);
        assert!(metrics.document_classification_accuracy > 0.0);
        assert!(metrics.section_classification_accuracy > 0.0);
    }
}

/// London TDD Tests for Document Processing
mod document_processing_tests {
    use super::*;

    #[tokio::test]
    async fn test_process_pci_dss_document() {
        // Given: A pipeline and PCI DSS document
        let mut fixture = PipelineTestFixture::new().await;
        
        // When: Processing PCI DSS document
        let result = fixture.pipeline.process_document(&fixture.sample_pci_document, None).await;
        
        // Then: Should process successfully
        assert!(result.is_ok());
        let processed_doc = result.unwrap();
        
        // Verify document classification
        assert!(processed_doc.document_type_result.is_some());
        let doc_result = processed_doc.document_type_result.as_ref().unwrap();
        assert_eq!(doc_result.document_type, DocumentType::PciDss);
        assert!(doc_result.confidence >= 0.7);
        
        // Verify section classification
        assert!(!processed_doc.section_classifications.is_empty());
        
        // Should have detected multiple section types
        let section_types: std::collections::HashSet<_> = processed_doc.section_classifications
            .iter()
            .map(|s| &s.classification_result.section_type)
            .collect();
        assert!(section_types.len() > 1);
        
        // Verify boundaries were detected
        assert!(!processed_doc.boundaries.is_empty());
        
        // Verify quality metrics
        assert!(processed_doc.quality_metrics.overall_score > 0.0);
        assert!(processed_doc.quality_metrics.overall_score <= 1.0);
        
        // Verify processing timeline
        assert!(processed_doc.processing_timeline.total_processing_time_ms > 0.0);
        assert!(processed_doc.processing_timeline.document_classification_time_ms.is_some());
    }

    #[tokio::test]
    async fn test_process_document_with_metadata() {
        // Given: A pipeline, document, and metadata
        let mut fixture = PipelineTestFixture::new().await;
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), "PCI DSS Compliance Guide".to_string());
        metadata.insert("version".to_string(), "4.0".to_string());
        metadata.insert("author".to_string(), "PCI Security Standards Council".to_string());
        
        // When: Processing document with metadata
        let result = fixture.pipeline.process_document(&fixture.sample_pci_document, Some(metadata.clone())).await;
        
        // Then: Should process successfully and include metadata
        assert!(result.is_ok());
        let processed_doc = result.unwrap();
        
        assert_eq!(processed_doc.metadata, metadata);
        assert!(processed_doc.document_type_result.is_some());
    }

    #[tokio::test]
    async fn test_process_document_performance() {
        // Given: A pipeline and test document
        let mut fixture = PipelineTestFixture::new().await;
        
        // When: Processing document and measuring performance
        let start = std::time::Instant::now();
        let result = fixture.pipeline.process_document(&fixture.sample_pci_document, None).await;
        let total_time = start.elapsed().as_secs_f64() * 1000.0;
        
        // Then: Should meet performance constraints
        assert!(result.is_ok());
        let processed_doc = result.unwrap();
        
        // Total processing should be reasonable (allowing for test environment overhead)
        assert!(total_time < 5000.0, "Total processing time {}ms should be < 5000ms", total_time);
        
        // Individual inference times should meet 10ms constraint
        if let Some(doc_time) = processed_doc.processing_timeline.document_classification_time_ms {
            assert!(doc_time < 50.0, "Document classification time {}ms should be reasonable", doc_time);
        }
        
        assert!(processed_doc.processing_timeline.section_detection_time_ms < 1000.0);
        assert!(processed_doc.processing_timeline.section_classification_time_ms < 1000.0);
    }

    #[tokio::test]
    async fn test_process_empty_document() {
        // Given: A pipeline and empty document
        let mut fixture = PipelineTestFixture::new().await;
        let empty_content = "";
        
        // When: Processing empty document
        let result = fixture.pipeline.process_document(empty_content, None).await;
        
        // Then: Should handle gracefully
        assert!(result.is_ok());
        let processed_doc = result.unwrap();
        
        assert_eq!(processed_doc.content, "");
        assert!(processed_doc.quality_metrics.overall_score >= 0.0);
    }

    #[tokio::test]
    async fn test_process_document_updates_metrics() {
        // Given: A pipeline
        let mut fixture = PipelineTestFixture::new().await;
        let initial_metrics = fixture.pipeline.get_metrics().await;
        
        // When: Processing a document
        let _ = fixture.pipeline.process_document(&fixture.sample_pci_document, None).await.unwrap();
        
        // Then: Metrics should be updated
        let updated_metrics = fixture.pipeline.get_metrics().await;
        
        assert_eq!(updated_metrics.total_documents, initial_metrics.total_documents + 1);
        assert!(updated_metrics.average_processing_time_ms > 0.0);
        assert!(updated_metrics.last_updated > initial_metrics.last_updated);
    }
}

/// London TDD Tests for Batch Processing
mod batch_processing_tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_process_multiple_documents() {
        // Given: A pipeline and multiple documents
        let mut fixture = PipelineTestFixture::new().await;
        let documents = vec![
            (fixture.sample_pci_document.clone(), None),
            (fixture.sample_iso_document.clone(), None),
            (fixture.sample_mixed_document.clone(), None),
        ];
        
        // When: Batch processing documents
        let result = fixture.pipeline.batch_process_documents(documents).await;
        
        // Then: Should process all documents successfully
        assert!(result.is_ok());
        let batch_result = result.unwrap();
        
        assert_eq!(batch_result.successful_documents.len(), 3);
        assert_eq!(batch_result.failed_documents.len(), 0);
        
        // Verify batch metrics
        assert_eq!(batch_result.batch_metrics.total_documents, 3);
        assert_eq!(batch_result.batch_metrics.successful_count, 3);
        assert_eq!(batch_result.batch_metrics.failed_count, 0);
        assert!(batch_result.batch_metrics.average_time_per_document_ms > 0.0);
        assert!(batch_result.batch_metrics.throughput_docs_per_second > 0.0);
        
        // Verify document types are classified correctly
        let doc_types: Vec<_> = batch_result.successful_documents
            .iter()
            .filter_map(|d| d.document_type_result.as_ref())
            .map(|r| &r.document_type)
            .collect();
        
        assert!(doc_types.contains(&&DocumentType::PciDss));
        assert!(doc_types.contains(&&DocumentType::Iso27001));
    }

    #[tokio::test]
    async fn test_batch_process_performance() {
        // Given: A pipeline and batch of documents
        let mut fixture = PipelineTestFixture::new().await;
        let documents: Vec<_> = (0..10)
            .map(|i| (format!("Test document {} content", i), None))
            .collect();
        
        // When: Batch processing with performance measurement
        let start = std::time::Instant::now();
        let result = fixture.pipeline.batch_process_documents(documents).await;
        let batch_time = start.elapsed();
        
        // Then: Should achieve reasonable throughput
        assert!(result.is_ok());
        let batch_result = result.unwrap();
        
        assert_eq!(batch_result.batch_metrics.total_documents, 10);
        assert!(batch_result.batch_metrics.throughput_docs_per_second > 0.1);
        
        // Total batch time should be reasonable
        assert!(batch_time.as_secs() < 60, "Batch processing took too long: {:?}", batch_time);
    }

    #[tokio::test]
    async fn test_batch_process_with_custom_batch_size() {
        // Given: A pipeline with custom batch size
        let config = PipelineConfig {
            batch_size: 2,
            ..Default::default()
        };
        let mut fixture = PipelineTestFixture::new_with_custom_config(config).await;
        
        let documents = vec![
            ("Document 1".to_string(), None),
            ("Document 2".to_string(), None),
            ("Document 3".to_string(), None),
            ("Document 4".to_string(), None),
            ("Document 5".to_string(), None),
        ];
        
        // When: Batch processing with custom batch size
        let result = fixture.pipeline.batch_process_documents(documents).await;
        
        // Then: Should process all documents respecting batch size
        assert!(result.is_ok());
        let batch_result = result.unwrap();
        assert_eq!(batch_result.successful_documents.len(), 5);
    }

    #[tokio::test]
    async fn test_batch_process_empty_batch() {
        // Given: A pipeline and empty batch
        let mut fixture = PipelineTestFixture::new().await;
        let documents = vec![];
        
        // When: Processing empty batch
        let result = fixture.pipeline.batch_process_documents(documents).await;
        
        // Then: Should handle empty batch gracefully
        assert!(result.is_ok());
        let batch_result = result.unwrap();
        
        assert_eq!(batch_result.successful_documents.len(), 0);
        assert_eq!(batch_result.failed_documents.len(), 0);
        assert_eq!(batch_result.batch_metrics.total_documents, 0);
    }
}

/// London TDD Tests for Query Routing
mod query_routing_tests {
    use super::*;

    #[tokio::test]
    async fn test_route_symbolic_query() {
        // Given: A pipeline and symbolic query
        let fixture = PipelineTestFixture::new().await;
        let query = "What are the requirements for encrypting cardholder data?";
        
        // When: Routing the query
        let result = fixture.pipeline.route_query(query).await;
        
        // Then: Should route to symbolic processing
        assert!(result.is_ok());
        let routing_result = result.unwrap();
        
        assert_eq!(routing_result.routing_decision, QueryRoute::Symbolic);
        assert!(routing_result.confidence >= 0.5);
        assert!(routing_result.inference_time_ms < 50.0);
    }

    #[tokio::test]
    async fn test_route_graph_query() {
        // Given: A pipeline and graph query
        let fixture = PipelineTestFixture::new().await;
        let query = "What requirements are related to requirement 3.2.1?";
        
        // When: Routing the query
        let result = fixture.pipeline.route_query(query).await;
        
        // Then: Should route to graph processing
        assert!(result.is_ok());
        let routing_result = result.unwrap();
        
        assert_eq!(routing_result.routing_decision, QueryRoute::Graph);
        assert!(routing_result.confidence >= 0.5);
    }

    #[tokio::test]
    async fn test_route_vector_query() {
        // Given: A pipeline and vector query
        let fixture = PipelineTestFixture::new().await;
        let query = "Find documents similar to this security policy.";
        
        // When: Routing the query
        let result = fixture.pipeline.route_query(query).await;
        
        // Then: Should route to vector processing
        assert!(result.is_ok());
        let routing_result = result.unwrap();
        
        assert_eq!(routing_result.routing_decision, QueryRoute::Vector);
        assert!(routing_result.confidence >= 0.5);
    }

    #[tokio::test]
    async fn test_route_multiple_queries() {
        // Given: A pipeline and multiple queries
        let fixture = PipelineTestFixture::new().await;
        let queries = vec![
            "What are the compliance requirements?",
            "What controls are related to encryption?",
            "Find similar security policies.",
        ];
        
        // When: Routing multiple queries
        let mut results = Vec::new();
        for query in queries {
            let result = fixture.pipeline.route_query(query).await;
            assert!(result.is_ok());
            results.push(result.unwrap());
        }
        
        // Then: Should route appropriately
        assert_eq!(results.len(), 3);
        for result in results {
            assert!(matches!(result.routing_decision, 
                QueryRoute::Symbolic | QueryRoute::Graph | QueryRoute::Vector | QueryRoute::Hybrid));
            assert!(result.confidence >= 0.0);
        }
    }
}

/// London TDD Tests for Configuration Management
mod configuration_management_tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_with_disabled_features() {
        // Given: Configuration with disabled features
        let config = PipelineConfig {
            enable_document_classification: false,
            enable_section_classification: false,
            enable_query_routing: true,
            ..Default::default()
        };
        
        // When: Creating pipeline and processing document
        let mut pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        let result = pipeline.process_document("Test document", None).await;
        
        // Then: Should process with disabled features
        assert!(result.is_ok());
        let processed_doc = result.unwrap();
        
        assert!(processed_doc.document_type_result.is_none());
        assert!(processed_doc.section_classifications.is_empty());
    }

    #[tokio::test]
    async fn test_optimize_configuration() {
        // Given: A pipeline with some processing history
        let mut fixture = PipelineTestFixture::new().await;
        
        // Process some documents to generate metrics
        for i in 0..3 {
            let content = format!("Test document {}", i);
            let _ = fixture.pipeline.process_document(&content, None).await.unwrap();
        }
        
        // When: Optimizing configuration
        let result = fixture.pipeline.optimize_configuration().await;
        
        // Then: Should return optimized configuration
        assert!(result.is_ok());
        let optimized_config = result.unwrap();
        
        assert!(optimized_config.batch_size > 0);
        assert!(optimized_config.confidence_threshold > 0.0);
        assert!(optimized_config.max_processing_time_ms > 0);
    }

    #[tokio::test]
    async fn test_configuration_affects_processing() {
        // Given: Two pipelines with different configurations
        let fast_config = PipelineConfig {
            max_processing_time_ms: 1000,
            confidence_threshold: 0.5,
            ..Default::default()
        };
        
        let strict_config = PipelineConfig {
            max_processing_time_ms: 10000,
            confidence_threshold: 0.9,
            ..Default::default()
        };
        
        let mut fast_pipeline = SmartIngestionPipeline::new(fast_config).await.unwrap();
        let mut strict_pipeline = SmartIngestionPipeline::new(strict_config).await.unwrap();
        
        // When: Processing same document with both pipelines
        let content = "Test document for configuration comparison";
        let fast_result = fast_pipeline.process_document(content, None).await.unwrap();
        let strict_result = strict_pipeline.process_document(content, None).await.unwrap();
        
        // Then: Both should process successfully
        assert!(!fast_result.id.to_string().is_empty());
        assert!(!strict_result.id.to_string().is_empty());
        
        // Both should have quality metrics
        assert!(fast_result.quality_metrics.overall_score >= 0.0);
        assert!(strict_result.quality_metrics.overall_score >= 0.0);
    }
}

/// London TDD Tests for Metrics and Monitoring
mod metrics_and_monitoring_tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_tracking_over_multiple_documents() {
        // Given: A pipeline
        let mut fixture = PipelineTestFixture::new().await;
        let initial_metrics = fixture.pipeline.get_metrics().await;
        
        // When: Processing multiple documents
        let test_documents = vec![
            "PCI DSS document content",
            "ISO 27001 document content", 
            "SOC 2 document content",
        ];
        
        for content in test_documents {
            let _ = fixture.pipeline.process_document(content, None).await.unwrap();
        }
        
        // Then: Metrics should reflect all processing
        let final_metrics = fixture.pipeline.get_metrics().await;
        
        assert_eq!(final_metrics.total_documents, initial_metrics.total_documents + 3);
        assert!(final_metrics.average_processing_time_ms > 0.0);
        assert!(final_metrics.throughput_docs_per_second >= 0.0);
        assert!(final_metrics.last_updated > initial_metrics.last_updated);
    }

    #[tokio::test]
    async fn test_processing_history_tracking() {
        // Given: A pipeline
        let mut fixture = PipelineTestFixture::new().await;
        
        // When: Processing documents
        let document_count = 5;
        for i in 0..document_count {
            let content = format!("Test document {} for history tracking", i);
            let _ = fixture.pipeline.process_document(&content, None).await.unwrap();
        }
        
        // Then: History should be tracked
        let history = fixture.pipeline.get_processing_history(Some(10));
        
        assert_eq!(history.len(), document_count);
        
        // History should be in reverse chronological order (most recent first)
        for i in 1..history.len() {
            assert!(history[i-1].start_time >= history[i].start_time);
        }
        
        // All records should indicate success
        for record in history {
            assert!(record.success);
            assert!(record.duration_ms > 0.0);
            assert!(record.sections_processed >= 0);
        }
    }

    #[tokio::test]
    async fn test_performance_target_compliance() {
        // Given: A pipeline with strict performance config
        let config = PipelineConfig {
            max_processing_time_ms: 2000,
            enable_performance_monitoring: true,
            ..Default::default()
        };
        let mut pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        
        // When: Processing documents
        let short_content = "Short test document";
        let _ = pipeline.process_document(short_content, None).await.unwrap();
        
        // Then: Should track performance compliance
        let metrics = pipeline.get_metrics().await;
        assert!(metrics.performance_target_met >= 0.0);
        assert!(metrics.performance_target_met <= 1.0);
    }
}

/// London TDD Tests for Error Handling and Edge Cases  
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_process_malformed_document() {
        // Given: A pipeline and malformed document
        let mut fixture = PipelineTestFixture::new().await;
        let malformed_content = "\x00\x01\x02Invalid binary content\xFF\xFE";
        
        // When: Processing malformed document
        let result = fixture.pipeline.process_document(malformed_content, None).await;
        
        // Then: Should handle gracefully without crashing
        // Note: Depending on implementation, this might succeed with low quality
        // or might fail gracefully - both are acceptable
        match result {
            Ok(processed_doc) => {
                // If it succeeds, quality should be low
                assert!(processed_doc.quality_metrics.overall_score < 0.5);
            },
            Err(_) => {
                // Graceful failure is also acceptable
            }
        }
    }

    #[tokio::test]
    async fn test_process_extremely_long_document() {
        // Given: A pipeline and extremely long document
        let mut fixture = PipelineTestFixture::new().await;
        let long_content = "A ".repeat(100_000); // Very long document
        
        // When: Processing extremely long document
        let result = fixture.pipeline.process_document(&long_content, None).await;
        
        // Then: Should handle without crashing (though might be slow)
        assert!(result.is_ok());
        let processed_doc = result.unwrap();
        
        assert_eq!(processed_doc.content.len(), long_content.len());
        assert!(processed_doc.quality_metrics.overall_score >= 0.0);
    }

    #[tokio::test]
    async fn test_concurrent_processing() {
        // Given: Multiple pipelines processing simultaneously
        let config = PipelineConfig::default();
        let mut pipeline1 = SmartIngestionPipeline::new(config.clone()).await.unwrap();
        let mut pipeline2 = SmartIngestionPipeline::new(config).await.unwrap();
        
        // When: Processing documents concurrently
        let content1 = "Document 1 content for concurrent test";
        let content2 = "Document 2 content for concurrent test";
        
        let (result1, result2) = tokio::join!(
            pipeline1.process_document(content1, None),
            pipeline2.process_document(content2, None)
        );
        
        // Then: Both should process successfully
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        
        let doc1 = result1.unwrap();
        let doc2 = result2.unwrap();
        
        assert_ne!(doc1.id, doc2.id);
        assert_eq!(doc1.content, content1);
        assert_eq!(doc2.content, content2);
    }
}

/// Integration Tests
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_end_to_end_pipeline_processing() {
        // Given: A complete pipeline and comprehensive document
        let mut fixture = PipelineTestFixture::new().await;
        
        // When: Processing through complete pipeline
        let result = fixture.pipeline.process_document(&fixture.sample_pci_document, None).await;
        
        // Then: Should complete full processing successfully
        assert!(result.is_ok());
        let processed_doc = result.unwrap();
        
        // Verify all pipeline stages completed
        assert!(processed_doc.document_type_result.is_some());
        assert!(!processed_doc.section_classifications.is_empty());
        assert!(!processed_doc.boundaries.is_empty());
        
        // Verify quality metrics are comprehensive
        let quality = &processed_doc.quality_metrics;
        assert!(quality.overall_score > 0.0);
        assert!(quality.classification_confidence >= 0.0);
        assert!(quality.section_coverage >= 0.0);
        assert!(quality.boundary_accuracy >= 0.0);
        assert!(quality.performance_score >= 0.0);
        
        // Verify processing timeline is complete
        let timeline = &processed_doc.processing_timeline;
        assert!(timeline.document_classification_time_ms.is_some());
        assert!(timeline.section_detection_time_ms > 0.0);
        assert!(timeline.section_classification_time_ms >= 0.0);
        assert!(timeline.total_processing_time_ms > 0.0);
        assert!(timeline.end_time > timeline.start_time);
    }

    #[tokio::test]
    async fn test_pipeline_stress_test() {
        // Given: A pipeline and stress test parameters
        let mut fixture = PipelineTestFixture::new().await;
        let stress_document_count = 20;
        
        // When: Processing many documents rapidly
        let mut successful_count = 0;
        let start_time = std::time::Instant::now();
        
        for i in 0..stress_document_count {
            let content = format!("Stress test document {} with PCI DSS content", i);
            match fixture.pipeline.process_document(&content, None).await {
                Ok(_) => successful_count += 1,
                Err(e) => println!("Document {} failed: {}", i, e),
            }
        }
        
        let total_time = start_time.elapsed();
        
        // Then: Should handle stress successfully
        assert!(successful_count >= stress_document_count * 80 / 100, 
               "Should successfully process at least 80% of documents");
        
        let throughput = successful_count as f64 / total_time.as_secs_f64();
        assert!(throughput > 0.1, "Should maintain reasonable throughput: {} docs/sec", throughput);
        
        // Verify final metrics
        let metrics = fixture.pipeline.get_metrics().await;
        assert_eq!(metrics.total_documents, successful_count as u64);
        assert!(metrics.average_processing_time_ms > 0.0);
    }
}