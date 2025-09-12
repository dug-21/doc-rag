//! Integration demo for neural classification system
//!
//! Demonstrates Phase 1 neurosymbolic enhancements:
//! - Document type classification (>90% accuracy target)
//! - Section type classification (>95% accuracy target)
//! - Query routing (<10ms inference constraint)
//! - Smart ingestion pipeline coordination

use crate::ingestion::{
    DocumentClassifier, SmartIngestionPipeline, PipelineConfig,
    DocumentType, SectionType, QueryRoute
};
use crate::Result;
use std::collections::HashMap;
use tracing::info;

/// Demo runner for Phase 1 neural enhancements
pub struct Phase1Demo {
    pipeline: SmartIngestionPipeline,
    classifier: DocumentClassifier,
}

impl Phase1Demo {
    /// Initialize Phase 1 demo with neural components
    pub async fn new() -> Result<Self> {
        info!("🚀 Initializing Phase 1 Neurosymbolic Demo");
        
        let config = PipelineConfig {
            enable_document_classification: true,
            enable_section_classification: true,
            enable_query_routing: true,
            max_processing_time_ms: 5000,
            batch_size: 10,
            confidence_threshold: 0.7,
            enable_performance_monitoring: true,
        };
        
        let pipeline = SmartIngestionPipeline::new(config).await?;
        let classifier = DocumentClassifier::new()?;
        
        info!("✅ Phase 1 Demo initialized successfully");
        
        Ok(Self {
            pipeline,
            classifier,
        })
    }
    
    /// Run comprehensive demo showcasing all Phase 1 capabilities
    pub async fn run_comprehensive_demo(&mut self) -> Result<DemoResults> {
        info!("🎯 Running comprehensive Phase 1 demonstration");
        
        let mut results = DemoResults::new();
        
        // Demo 1: Document Type Classification
        results.document_classification = self.demo_document_classification().await?;
        
        // Demo 2: Section Type Classification
        results.section_classification = self.demo_section_classification().await?;
        
        // Demo 3: Query Routing Classification
        results.query_routing = self.demo_query_routing().await?;
        
        // Demo 4: Smart Ingestion Pipeline
        results.pipeline_processing = self.demo_pipeline_processing().await?;
        
        // Demo 5: Performance Validation
        results.performance_metrics = self.demo_performance_validation().await?;
        
        // Demo 6: Batch Processing
        results.batch_processing = self.demo_batch_processing().await?;
        
        info!("🎉 Comprehensive Phase 1 demonstration completed successfully");
        info!("📊 Results Summary:");
        info!("   - Document Classification: {} samples", results.document_classification.samples_tested);
        info!("   - Section Classification: {} samples", results.section_classification.samples_tested);
        info!("   - Query Routing: {} queries", results.query_routing.queries_tested);
        info!("   - Pipeline Processing: {} documents", results.pipeline_processing.documents_processed);
        info!("   - Performance: Avg inference {:.2}ms", results.performance_metrics.average_inference_time_ms);
        info!("   - Batch Processing: {:.1} docs/sec", results.batch_processing.throughput_docs_per_second);
        
        Ok(results)
    }
    
    /// Demo document type classification (PCI-DSS, ISO-27001, SOC2, NIST)
    async fn demo_document_classification(&mut self) -> Result<DocumentClassificationDemo> {
        info!("📋 Demonstrating document type classification");
        
        let test_documents = vec![
            ("Payment Card Industry Data Security Standard (PCI DSS) Requirements and Security Assessment Procedures. This document provides security requirements for organizations that store, process, or transmit cardholder data.", DocumentType::PciDss),
            ("ISO/IEC 27001:2013 Information Security Management Systems Requirements. This International Standard specifies requirements for establishing, implementing, maintaining and continually improving an information security management system.", DocumentType::Iso27001),
            ("SOC 2 Type II Service Organization Control Report describing the service organization's system and the suitability of design and operating effectiveness of controls relevant to security, availability, processing integrity, confidentiality, and privacy.", DocumentType::Soc2),
            ("NIST Cybersecurity Framework Version 1.1 providing organizations with activities and outcomes that help them manage and reduce cybersecurity risk in a cost-effective way based on business needs.", DocumentType::Nist),
        ];
        
        let mut results = Vec::new();
        let mut correct_classifications = 0;
        
        for (content, expected_type) in test_documents {
            let start_time = std::time::Instant::now();
            let result = self.classifier.classify_document_type(content, None).await?;
            let inference_time = start_time.elapsed().as_secs_f64() * 1000.0;
            
            let is_correct = result.document_type == expected_type;
            if is_correct {
                correct_classifications += 1;
            }
            
            info!("   📄 {} -> {:?} (confidence: {:.1}%, time: {:.2}ms) {}",
                  &content[..50],
                  result.document_type,
                  result.confidence * 100.0,
                  inference_time,
                  if is_correct { "✅" } else { "❌" }
            );
            
            results.push(result);
        }
        
        let accuracy = correct_classifications as f64 / results.len() as f64;
        let avg_confidence = results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;
        let avg_inference_time = results.iter().map(|r| r.inference_time_ms).sum::<f64>() / results.len() as f64;
        
        info!("📊 Document Classification Results:");
        info!("   - Accuracy: {:.1}% (target: >90%)", accuracy * 100.0);
        info!("   - Average Confidence: {:.1}%", avg_confidence * 100.0);
        info!("   - Average Inference Time: {:.2}ms (constraint: <10ms)", avg_inference_time);
        
        Ok(DocumentClassificationDemo {
            samples_tested: results.len(),
            accuracy,
            average_confidence: avg_confidence,
            average_inference_time_ms: avg_inference_time,
            constraint_compliance: avg_inference_time < 10.0,
            results,
        })
    }
    
    /// Demo section type classification (Requirements, Definitions, Procedures)
    async fn demo_section_classification(&mut self) -> Result<SectionClassificationDemo> {
        info!("📋 Demonstrating section type classification");
        
        let test_sections = vec![
            ("Requirement 3.2.1: Cardholder data must be protected during transmission over open, public networks using strong cryptography and security protocols.", SectionType::Requirements),
            ("Definition: Cardholder data is defined as the primary account number (PAN) plus any of the following: cardholder name, expiration date, or service code.", SectionType::Definitions),
            ("Procedure 1: Step 1 - Identify all locations where cardholder data is stored. Step 2 - Document data flows. Step 3 - Implement security controls.", SectionType::Procedures),
            ("Appendix A: This appendix provides additional guidance on implementing the requirements specified in this standard.", SectionType::Appendices),
            ("Example 1: The following example demonstrates proper implementation of network segmentation controls for cardholder data environments.", SectionType::Examples),
            ("References: 1. ISO/IEC 27001:2013, 2. NIST Cybersecurity Framework, 3. Payment Card Industry Security Standards Council", SectionType::References),
        ];
        
        let mut results = Vec::new();
        let mut correct_classifications = 0;
        
        for (content, expected_type) in test_sections {
            let start_time = std::time::Instant::now();
            let result = self.classifier.classify_section_type(content, None, 0).await?;
            let inference_time = start_time.elapsed().as_secs_f64() * 1000.0;
            
            let is_correct = result.section_type == expected_type;
            if is_correct {
                correct_classifications += 1;
            }
            
            info!("   📝 {} -> {:?} (confidence: {:.1}%, hints: {:?}, time: {:.2}ms) {}",
                  &content[..40],
                  result.section_type,
                  result.confidence * 100.0,
                  result.type_hints,
                  inference_time,
                  if is_correct { "✅" } else { "❌" }
            );
            
            results.push(result);
        }
        
        let accuracy = correct_classifications as f64 / results.len() as f64;
        let avg_confidence = results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;
        let avg_inference_time = results.iter().map(|r| r.inference_time_ms).sum::<f64>() / results.len() as f64;
        
        info!("📊 Section Classification Results:");
        info!("   - Accuracy: {:.1}% (target: >95%)", accuracy * 100.0);
        info!("   - Average Confidence: {:.1}%", avg_confidence * 100.0);
        info!("   - Average Inference Time: {:.2}ms (constraint: <10ms)", avg_inference_time);
        
        Ok(SectionClassificationDemo {
            samples_tested: results.len(),
            accuracy,
            average_confidence: avg_confidence,
            average_inference_time_ms: avg_inference_time,
            constraint_compliance: avg_inference_time < 10.0,
            results,
        })
    }
    
    /// Demo query routing classification (symbolic vs graph vs vector)
    async fn demo_query_routing(&mut self) -> Result<QueryRoutingDemo> {
        info!("🔀 Demonstrating query routing classification");
        
        let test_queries = vec![
            ("What are the requirements for encrypting cardholder data in PCI DSS?", QueryRoute::Symbolic),
            ("What security controls are related to requirement 3.2.1?", QueryRoute::Graph),
            ("Find documents similar to this security policy.", QueryRoute::Vector),
            ("What are the dependencies between encryption requirements and network controls?", QueryRoute::Graph),
            ("Is multi-factor authentication required for administrative access?", QueryRoute::Symbolic),
            ("Show me policies that are similar to our current data protection framework.", QueryRoute::Vector),
        ];
        
        let mut results = Vec::new();
        let mut correct_routings = 0;
        
        for (query, expected_route) in test_queries {
            let start_time = std::time::Instant::now();
            let result = self.classifier.route_query(query).await?;
            let inference_time = start_time.elapsed().as_secs_f64() * 1000.0;
            
            let is_correct = result.routing_decision == expected_route;
            if is_correct {
                correct_routings += 1;
            }
            
            info!("   🔍 {} -> {:?} (confidence: {:.1}%, time: {:.2}ms) {}",
                  query,
                  result.routing_decision,
                  result.confidence * 100.0,
                  inference_time,
                  if is_correct { "✅" } else { "❌" }
            );
            
            results.push(result);
        }
        
        let accuracy = correct_routings as f64 / results.len() as f64;
        let avg_confidence = results.iter().map(|r| r.confidence).sum::<f64>() / results.len() as f64;
        let avg_inference_time = results.iter().map(|r| r.inference_time_ms).sum::<f64>() / results.len() as f64;
        
        info!("📊 Query Routing Results:");
        info!("   - Routing Accuracy: {:.1}%", accuracy * 100.0);
        info!("   - Average Confidence: {:.1}%", avg_confidence * 100.0);
        info!("   - Average Inference Time: {:.2}ms (constraint: <10ms)", avg_inference_time);
        
        Ok(QueryRoutingDemo {
            queries_tested: results.len(),
            routing_accuracy: accuracy,
            average_confidence: avg_confidence,
            average_inference_time_ms: avg_inference_time,
            constraint_compliance: avg_inference_time < 10.0,
            results,
        })
    }
    
    /// Demo smart ingestion pipeline processing
    async fn demo_pipeline_processing(&mut self) -> Result<PipelineProcessingDemo> {
        info!("⚙️ Demonstrating smart ingestion pipeline");
        
        let test_document = r#"
            Payment Card Industry Data Security Standard (PCI DSS)
            Requirements and Security Assessment Procedures Version 4.0
            
            # Requirements Section
            Requirement 3.2.1: Cardholder data must be protected during transmission 
            over open, public networks using strong cryptography and security protocols.
            
            Requirement 4.1: Use strong cryptography and security protocols to 
            safeguard sensitive cardholder data during transmission over open networks.
            
            # Definitions Section
            Definition: Cardholder data is defined as the primary account number (PAN) 
            plus any of the following: cardholder name, expiration date, or service code.
            
            Definition: Sensitive authentication data includes full track data, 
            card verification codes/values, and PINs/PIN blocks.
            
            # Procedures Section
            Procedure 1: Identify all locations where cardholder data is stored
            Step 1 - Conduct comprehensive data discovery
            Step 2 - Document all data flows and processing methods  
            Step 3 - Implement appropriate security controls and monitoring
            
            Procedure 2: Implement network segmentation
            Step 1 - Design network architecture with appropriate segmentation
            Step 2 - Deploy and configure network security controls
            Step 3 - Validate segmentation effectiveness through testing
        "#;
        
        let mut metadata = HashMap::new();
        metadata.insert("title".to_string(), "PCI DSS Requirements v4.0".to_string());
        metadata.insert("version".to_string(), "4.0".to_string());
        metadata.insert("standard".to_string(), "PCI-DSS".to_string());
        
        let start_time = std::time::Instant::now();
        let processed_doc = self.pipeline.process_document(test_document, Some(metadata)).await?;
        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        info!("📄 Processed Document: {}", processed_doc.id);
        info!("   - Document Type: {:?} (confidence: {:.1}%)",
              processed_doc.document_type_result.as_ref().map(|r| &r.document_type),
              processed_doc.document_type_result.as_ref().map(|r| r.confidence * 100.0).unwrap_or(0.0)
        );
        info!("   - Sections Detected: {}", processed_doc.section_classifications.len());
        info!("   - Boundaries Found: {}", processed_doc.boundaries.len());
        info!("   - Overall Quality: {:.1}%", processed_doc.quality_metrics.overall_score * 100.0);
        info!("   - Total Processing Time: {:.2}ms", total_time);
        
        // Display section classifications
        for (i, section) in processed_doc.section_classifications.iter().enumerate() {
            info!("   📝 Section {}: {:?} (confidence: {:.1}%)",
                  i + 1,
                  section.classification_result.section_type,
                  section.classification_result.confidence * 100.0
            );
        }
        
        Ok(PipelineProcessingDemo {
            documents_processed: 1,
            total_processing_time_ms: total_time,
            average_quality_score: processed_doc.quality_metrics.overall_score,
            sections_classified: processed_doc.section_classifications.len(),
            boundaries_detected: processed_doc.boundaries.len(),
            processed_document: processed_doc,
        })
    }
    
    /// Demo performance validation against constraints
    async fn demo_performance_validation(&mut self) -> Result<PerformanceDemo> {
        info!("🎯 Demonstrating performance constraint validation");
        
        let test_content = "PCI DSS requirement 3.2.1: Organizations must encrypt cardholder data.";
        let iterations = 100;
        
        let mut doc_inference_times = Vec::new();
        let mut section_inference_times = Vec::new();
        let mut query_inference_times = Vec::new();
        
        // Test document classification performance
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = self.classifier.classify_document_type(test_content, None).await?;
            doc_inference_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        
        // Test section classification performance
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = self.classifier.classify_section_type(test_content, None, 0).await?;
            section_inference_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        
        // Test query routing performance
        for _ in 0..iterations {
            let start = std::time::Instant::now();
            let _ = self.classifier.route_query("What are the requirements?").await?;
            query_inference_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        
        let avg_doc_time = doc_inference_times.iter().sum::<f64>() / doc_inference_times.len() as f64;
        let avg_section_time = section_inference_times.iter().sum::<f64>() / section_inference_times.len() as f64;
        let avg_query_time = query_inference_times.iter().sum::<f64>() / query_inference_times.len() as f64;
        
        let max_doc_time = doc_inference_times.iter().fold(0.0f64, |a, &b| a.max(b));
        let max_section_time = section_inference_times.iter().fold(0.0f64, |a, &b| a.max(b));
        let max_query_time = query_inference_times.iter().fold(0.0f64, |a, &b| a.max(b));
        
        let doc_compliance = doc_inference_times.iter().filter(|&&t| t < 10.0).count() as f64 / iterations as f64;
        let section_compliance = section_inference_times.iter().filter(|&&t| t < 10.0).count() as f64 / iterations as f64;
        let query_compliance = query_inference_times.iter().filter(|&&t| t < 10.0).count() as f64 / iterations as f64;
        
        info!("📊 Performance Validation Results ({} iterations):", iterations);
        info!("   📄 Document Classification:");
        info!("     - Average: {:.2}ms, Max: {:.2}ms", avg_doc_time, max_doc_time);
        info!("     - <10ms Compliance: {:.1}%", doc_compliance * 100.0);
        info!("   📝 Section Classification:");
        info!("     - Average: {:.2}ms, Max: {:.2}ms", avg_section_time, max_section_time);
        info!("     - <10ms Compliance: {:.1}%", section_compliance * 100.0);
        info!("   🔀 Query Routing:");
        info!("     - Average: {:.2}ms, Max: {:.2}ms", avg_query_time, max_query_time);
        info!("     - <10ms Compliance: {:.1}%", query_compliance * 100.0);
        
        let overall_avg = (avg_doc_time + avg_section_time + avg_query_time) / 3.0;
        let overall_compliance = (doc_compliance + section_compliance + query_compliance) / 3.0;
        
        Ok(PerformanceDemo {
            iterations_tested: iterations,
            average_inference_time_ms: overall_avg,
            max_inference_time_ms: max_doc_time.max(max_section_time).max(max_query_time),
            constraint_compliance_percentage: overall_compliance * 100.0,
            document_classification_avg_ms: avg_doc_time,
            section_classification_avg_ms: avg_section_time,
            query_routing_avg_ms: avg_query_time,
        })
    }
    
    /// Demo batch processing capabilities
    async fn demo_batch_processing(&mut self) -> Result<BatchProcessingDemo> {
        info!("📦 Demonstrating batch processing capabilities");
        
        let batch_documents = vec![
            "PCI DSS compliance requirements for cardholder data protection".to_string(),
            "ISO 27001 information security management system controls".to_string(),
            "SOC 2 Type II service organization control report".to_string(),
            "NIST cybersecurity framework implementation guidelines".to_string(),
            "Additional PCI DSS requirements for network security controls".to_string(),
        ];
        
        let documents_with_metadata: Vec<_> = batch_documents.into_iter().map(|doc| (doc, None)).collect();
        
        let start_time = std::time::Instant::now();
        let batch_result = self.pipeline.batch_process_documents(documents_with_metadata).await?;
        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        info!("📊 Batch Processing Results:");
        info!("   - Total Documents: {}", batch_result.batch_metrics.total_documents);
        info!("   - Successful: {}", batch_result.batch_metrics.successful_count);
        info!("   - Failed: {}", batch_result.batch_metrics.failed_count);
        info!("   - Throughput: {:.1} docs/sec", batch_result.batch_metrics.throughput_docs_per_second);
        info!("   - Average Time per Doc: {:.2}ms", batch_result.batch_metrics.average_time_per_document_ms);
        
        // Show document type distribution
        let mut type_counts = std::collections::HashMap::new();
        for doc in &batch_result.successful_documents {
            if let Some(result) = &doc.document_type_result {
                *type_counts.entry(&result.document_type).or_insert(0) += 1;
            }
        }
        
        info!("   📋 Document Type Distribution:");
        for (doc_type, count) in type_counts {
            info!("     - {:?}: {}", doc_type, count);
        }
        
        Ok(BatchProcessingDemo {
            total_documents: batch_result.batch_metrics.total_documents,
            successful_documents: batch_result.batch_metrics.successful_count,
            failed_documents: batch_result.batch_metrics.failed_count,
            throughput_docs_per_second: batch_result.batch_metrics.throughput_docs_per_second,
            average_time_per_document_ms: batch_result.batch_metrics.average_time_per_document_ms,
            total_batch_time_ms: total_time,
        })
    }
}

/// Comprehensive demo results for Phase 1
#[derive(Debug)]
pub struct DemoResults {
    pub document_classification: DocumentClassificationDemo,
    pub section_classification: SectionClassificationDemo,
    pub query_routing: QueryRoutingDemo,
    pub pipeline_processing: PipelineProcessingDemo,
    pub performance_metrics: PerformanceDemo,
    pub batch_processing: BatchProcessingDemo,
}

#[derive(Debug)]
pub struct DocumentClassificationDemo {
    pub samples_tested: usize,
    pub accuracy: f64,
    pub average_confidence: f64,
    pub average_inference_time_ms: f64,
    pub constraint_compliance: bool,
    pub results: Vec<crate::ingestion::DocumentTypeResult>,
}

#[derive(Debug)]
pub struct SectionClassificationDemo {
    pub samples_tested: usize,
    pub accuracy: f64,
    pub average_confidence: f64,
    pub average_inference_time_ms: f64,
    pub constraint_compliance: bool,
    pub results: Vec<crate::ingestion::SectionTypeResult>,
}

#[derive(Debug)]
pub struct QueryRoutingDemo {
    pub queries_tested: usize,
    pub routing_accuracy: f64,
    pub average_confidence: f64,
    pub average_inference_time_ms: f64,
    pub constraint_compliance: bool,
    pub results: Vec<crate::ingestion::QueryRoutingResult>,
}

#[derive(Debug)]
pub struct PipelineProcessingDemo {
    pub documents_processed: usize,
    pub total_processing_time_ms: f64,
    pub average_quality_score: f64,
    pub sections_classified: usize,
    pub boundaries_detected: usize,
    pub processed_document: crate::ingestion::ProcessedDocument,
}

#[derive(Debug)]
pub struct PerformanceDemo {
    pub iterations_tested: usize,
    pub average_inference_time_ms: f64,
    pub max_inference_time_ms: f64,
    pub constraint_compliance_percentage: f64,
    pub document_classification_avg_ms: f64,
    pub section_classification_avg_ms: f64,
    pub query_routing_avg_ms: f64,
}

#[derive(Debug)]
pub struct BatchProcessingDemo {
    pub total_documents: usize,
    pub successful_documents: usize,
    pub failed_documents: usize,
    pub throughput_docs_per_second: f64,
    pub average_time_per_document_ms: f64,
    pub total_batch_time_ms: f64,
}

impl DemoResults {
    fn new() -> Self {
        Self {
            document_classification: DocumentClassificationDemo {
                samples_tested: 0,
                accuracy: 0.0,
                average_confidence: 0.0,
                average_inference_time_ms: 0.0,
                constraint_compliance: false,
                results: Vec::new(),
            },
            section_classification: SectionClassificationDemo {
                samples_tested: 0,
                accuracy: 0.0,
                average_confidence: 0.0,
                average_inference_time_ms: 0.0,
                constraint_compliance: false,
                results: Vec::new(),
            },
            query_routing: QueryRoutingDemo {
                queries_tested: 0,
                routing_accuracy: 0.0,
                average_confidence: 0.0,
                average_inference_time_ms: 0.0,
                constraint_compliance: false,
                results: Vec::new(),
            },
            pipeline_processing: PipelineProcessingDemo {
                documents_processed: 0,
                total_processing_time_ms: 0.0,
                average_quality_score: 0.0,
                sections_classified: 0,
                boundaries_detected: 0,
                processed_document: unsafe { std::mem::zeroed() }, // Will be properly initialized
            },
            performance_metrics: PerformanceDemo {
                iterations_tested: 0,
                average_inference_time_ms: 0.0,
                max_inference_time_ms: 0.0,
                constraint_compliance_percentage: 0.0,
                document_classification_avg_ms: 0.0,
                section_classification_avg_ms: 0.0,
                query_routing_avg_ms: 0.0,
            },
            batch_processing: BatchProcessingDemo {
                total_documents: 0,
                successful_documents: 0,
                failed_documents: 0,
                throughput_docs_per_second: 0.0,
                average_time_per_document_ms: 0.0,
                total_batch_time_ms: 0.0,
            },
        }
    }
}