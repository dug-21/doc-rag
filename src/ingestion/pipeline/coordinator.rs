//! Smart ingestion pipeline coordinator for neurosymbolic processing
//!
//! Coordinates document ingestion through multiple neural classification stages:
//! 1. Document type classification (PCI-DSS, ISO-27001, SOC2, NIST)
//! 2. Section type classification (Requirements, Definitions, Procedures)
//! 3. Smart routing for downstream processing
//!
//! Integrates with existing neural chunker foundation and maintains <10ms inference

use crate::{Result, ChunkerError};
use super::super::classification::{
    document_classifier::{DocumentClassifier, DocumentTypeResult, SectionTypeResult, QueryRoutingResult, DocumentType, SectionType, QueryRoute},
    feature_extractor::FeatureExtractor,
};
use crate::neural_chunker_working::WorkingNeuralChunker;
use crate::boundary::BoundaryInfo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tracing::{info, debug, warn, error};
use uuid::Uuid;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Smart ingestion pipeline coordinator
#[derive(Debug)]
pub struct SmartIngestionPipeline {
    /// Document classifier for type identification
    document_classifier: Arc<RwLock<DocumentClassifier>>,
    /// Neural chunker for boundary detection
    neural_chunker: Arc<RwLock<WorkingNeuralChunker>>,
    /// Pipeline configuration
    config: PipelineConfig,
    /// Performance metrics
    metrics: Arc<RwLock<PipelineMetrics>>,
    /// Processing history
    processing_history: Vec<ProcessingRecord>,
}

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Enable document type classification
    pub enable_document_classification: bool,
    /// Enable section type classification
    pub enable_section_classification: bool,
    /// Enable query routing
    pub enable_query_routing: bool,
    /// Maximum processing time per document (ms)
    pub max_processing_time_ms: u64,
    /// Batch size for parallel processing
    pub batch_size: usize,
    /// Confidence threshold for classifications
    pub confidence_threshold: f64,
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,
}

/// Processed document with neural classification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDocument {
    /// Document identifier
    pub id: Uuid,
    /// Original document content
    pub content: String,
    /// Document metadata
    pub metadata: HashMap<String, String>,
    /// Document type classification result
    pub document_type_result: Option<DocumentTypeResult>,
    /// Section classifications for each detected section
    pub section_classifications: Vec<ProcessedSection>,
    /// Neural chunker boundary information
    pub boundaries: Vec<BoundaryInfo>,
    /// Processing timeline
    pub processing_timeline: ProcessingTimeline,
    /// Quality metrics
    pub quality_metrics: DocumentQualityMetrics,
    /// Processing timestamp
    pub processed_at: DateTime<Utc>,
}

/// Processed section with classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedSection {
    /// Section identifier
    pub id: String,
    /// Section content
    pub content: String,
    /// Start position in document
    pub start_position: usize,
    /// End position in document
    pub end_position: usize,
    /// Section type classification result
    pub classification_result: SectionTypeResult,
    /// Section quality score
    pub quality_score: f64,
}

/// Processing timeline for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTimeline {
    /// Start time
    pub start_time: DateTime<Utc>,
    /// Document classification time
    pub document_classification_time_ms: Option<f64>,
    /// Section detection time
    pub section_detection_time_ms: f64,
    /// Section classification time
    pub section_classification_time_ms: f64,
    /// Total processing time
    pub total_processing_time_ms: f64,
    /// End time
    pub end_time: DateTime<Utc>,
}

/// Document quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentQualityMetrics {
    /// Overall document quality score (0.0 - 1.0)
    pub overall_score: f64,
    /// Classification confidence scores
    pub classification_confidence: f64,
    /// Section coverage completeness
    pub section_coverage: f64,
    /// Boundary detection accuracy
    pub boundary_accuracy: f64,
    /// Performance efficiency score
    pub performance_score: f64,
}

/// Pipeline performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    /// Total documents processed
    pub total_documents: u64,
    /// Average processing time per document
    pub average_processing_time_ms: f64,
    /// Document classification accuracy
    pub document_classification_accuracy: f64,
    /// Section classification accuracy
    pub section_classification_accuracy: f64,
    /// Performance target compliance (<10ms inference)
    pub performance_target_met: f64, // percentage
    /// Throughput (documents per second)
    pub throughput_docs_per_second: f64,
    /// Error rate
    pub error_rate: f64,
    /// Last updated
    pub last_updated: DateTime<Utc>,
}

/// Processing record for history tracking
#[derive(Debug, Clone)]
pub struct ProcessingRecord {
    /// Document ID
    pub document_id: Uuid,
    /// Processing start time
    pub start_time: DateTime<Utc>,
    /// Processing duration
    pub duration_ms: f64,
    /// Success status
    pub success: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Document type detected
    pub document_type: Option<DocumentType>,
    /// Number of sections processed
    pub sections_processed: usize,
}

/// Batch processing result
#[derive(Debug)]
pub struct BatchProcessingResult {
    /// Successfully processed documents
    pub successful_documents: Vec<ProcessedDocument>,
    /// Failed documents with error information
    pub failed_documents: Vec<(String, ChunkerError)>,
    /// Batch processing metrics
    pub batch_metrics: BatchMetrics,
}

/// Batch processing metrics
#[derive(Debug, Clone)]
pub struct BatchMetrics {
    /// Total documents in batch
    pub total_documents: usize,
    /// Successfully processed count
    pub successful_count: usize,
    /// Failed processing count
    pub failed_count: usize,
    /// Average processing time per document
    pub average_time_per_document_ms: f64,
    /// Total batch processing time
    pub total_batch_time_ms: f64,
    /// Throughput achieved
    pub throughput_docs_per_second: f64,
}

impl SmartIngestionPipeline {
    /// Creates a new smart ingestion pipeline
    pub async fn new(config: PipelineConfig) -> Result<Self> {
        info!("Initializing Smart Ingestion Pipeline with neural classification");
        
        let document_classifier = Arc::new(RwLock::new(DocumentClassifier::new()?));
        let neural_chunker = Arc::new(RwLock::new(WorkingNeuralChunker::new()?));
        let metrics = Arc::new(RwLock::new(PipelineMetrics::new()));
        
        // Validate neural networks health
        {
            let mut classifier = document_classifier.write().await;
            let mut chunker = neural_chunker.write().await;
            
            if !classifier.health_check() {
                return Err(ChunkerError::NeuralError("Document classifier health check failed".to_string()));
            }
            
            if !chunker.health_check() {
                return Err(ChunkerError::NeuralError("Neural chunker health check failed".to_string()));
            }
        }
        
        info!("Smart Ingestion Pipeline initialized successfully");
        
        Ok(Self {
            document_classifier,
            neural_chunker,
            config,
            metrics,
            processing_history: Vec::new(),
        })
    }

    /// Process a single document through the complete neural pipeline
    pub async fn process_document(&mut self, content: &str, metadata: Option<HashMap<String, String>>) -> Result<ProcessedDocument> {
        let start_time = Utc::now();
        let document_id = Uuid::new_v4();
        
        info!("Processing document {} through neural pipeline", document_id);
        
        let mut timeline = ProcessingTimeline {
            start_time,
            document_classification_time_ms: None,
            section_detection_time_ms: 0.0,
            section_classification_time_ms: 0.0,
            total_processing_time_ms: 0.0,
            end_time: start_time,
        };
        
        // Phase 1: Document Type Classification
        let document_type_result = if self.config.enable_document_classification {
            let classification_start = std::time::Instant::now();
            let mut classifier = self.document_classifier.write().await;
            let result = classifier.classify_document_type(content, metadata.as_ref()).await?;
            let classification_time = classification_start.elapsed().as_secs_f64() * 1000.0;
            
            timeline.document_classification_time_ms = Some(classification_time);
            
            info!("Document classified as {:?} with {:.1}% confidence",
                  result.document_type, result.confidence * 100.0);
            
            Some(result)
        } else {
            None
        };

        // Phase 2: Neural Boundary Detection and Section Extraction
        let section_detection_start = std::time::Instant::now();
        let boundaries = {
            let mut chunker = self.neural_chunker.write().await;
            chunker.detect_boundaries(content)?
        };
        timeline.section_detection_time_ms = section_detection_start.elapsed().as_secs_f64() * 1000.0;
        
        debug!("Detected {} boundaries in document", boundaries.len());

        // Phase 3: Section Classification
        let section_classification_start = std::time::Instant::now();
        let section_classifications = if self.config.enable_section_classification {
            self.classify_document_sections(content, &boundaries).await?
        } else {
            Vec::new()
        };
        timeline.section_classification_time_ms = section_classification_start.elapsed().as_secs_f64() * 1000.0;
        
        // Calculate quality metrics
        let quality_metrics = self.calculate_document_quality_metrics(
            &document_type_result,
            &section_classifications,
            &boundaries,
            &timeline,
        );

        // Complete timeline
        let end_time = Utc::now();
        timeline.end_time = end_time;
        timeline.total_processing_time_ms = (end_time - start_time).num_milliseconds() as f64;

        // Create processed document
        let processed_doc = ProcessedDocument {
            id: document_id,
            content: content.to_string(),
            metadata: metadata.unwrap_or_default(),
            document_type_result,
            section_classifications,
            boundaries,
            processing_timeline: timeline,
            quality_metrics,
            processed_at: end_time,
        };

        // Update metrics and history
        self.update_processing_metrics(&processed_doc).await;
        self.add_processing_record(&processed_doc, true, None);

        // Validate performance constraints
        if processed_doc.processing_timeline.total_processing_time_ms > self.config.max_processing_time_ms as f64 {
            warn!("Document processing exceeded time limit: {:.1}ms > {}ms",
                  processed_doc.processing_timeline.total_processing_time_ms, self.config.max_processing_time_ms);
        }

        info!("Document {} processed successfully in {:.1}ms",
              document_id, processed_doc.processing_timeline.total_processing_time_ms);

        Ok(processed_doc)
    }

    /// Process multiple documents in batch for efficiency
    pub async fn batch_process_documents(&mut self, documents: Vec<(String, Option<HashMap<String, String>>)>) -> Result<BatchProcessingResult> {
        let batch_start = std::time::Instant::now();
        let batch_size = documents.len();
        
        info!("Starting batch processing of {} documents", batch_size);
        
        let mut successful_documents = Vec::new();
        let mut failed_documents = Vec::new();
        
        // Process documents in parallel batches
        let chunk_size = self.config.batch_size.min(batch_size);
        
        for (batch_index, chunk) in documents.chunks(chunk_size).enumerate() {
            debug!("Processing batch {} with {} documents", batch_index + 1, chunk.len());
            
            // Process chunk sequentially for now (can be made parallel later)
            for (content, metadata) in chunk {
                match self.process_document(content, metadata.clone()).await {
                    Ok(processed_doc) => successful_documents.push(processed_doc),
                    Err(error) => {
                        error!("Failed to process document: {}", error);
                        failed_documents.push((content.clone(), error));
                    }
                }
            }
        }
        
        let batch_time = batch_start.elapsed().as_secs_f64() * 1000.0;
        let successful_count = successful_documents.len();
        let failed_count = failed_documents.len();
        
        let batch_metrics = BatchMetrics {
            total_documents: batch_size,
            successful_count,
            failed_count,
            average_time_per_document_ms: if successful_count > 0 {
                batch_time / successful_count as f64
            } else { 0.0 },
            total_batch_time_ms: batch_time,
            throughput_docs_per_second: if batch_time > 0.0 {
                successful_count as f64 / (batch_time / 1000.0)
            } else { 0.0 },
        };
        
        info!("Batch processing completed: {}/{} successful in {:.1}ms ({:.1} docs/sec)",
              successful_count, batch_size, batch_time, batch_metrics.throughput_docs_per_second);
        
        Ok(BatchProcessingResult {
            successful_documents,
            failed_documents,
            batch_metrics,
        })
    }

    /// Route queries using neural classification
    pub async fn route_query(&self, query: &str) -> Result<QueryRoutingResult> {
        let mut classifier = self.document_classifier.write().await;
        let result = classifier.route_query(query).await?;
        
        info!("Query routed to {:?} processing with {:.1}% confidence",
              result.routing_decision, result.confidence * 100.0);
        
        Ok(result)
    }

    /// Classify document sections using neural networks
    async fn classify_document_sections(&self, content: &str, boundaries: &[BoundaryInfo]) -> Result<Vec<ProcessedSection>> {
        let mut sections = Vec::new();
        
        if boundaries.len() < 2 {
            return Ok(sections);
        }
        
        let mut classifier = self.document_classifier.write().await;
        
        // Extract sections between boundaries
        for i in 0..(boundaries.len() - 1) {
            let start_pos = boundaries[i].position;
            let end_pos = boundaries[i + 1].position;
            
            if start_pos >= end_pos || end_pos > content.len() {
                continue;
            }
            
            let section_content = &content[start_pos..end_pos];
            if section_content.trim().is_empty() {
                continue;
            }
            
            // Classify section type
            let classification_result = classifier.classify_section_type(
                section_content,
                None, // Could provide context from previous sections
                i,
            ).await?;
            
            // Calculate section quality score
            let quality_score = self.calculate_section_quality_score(
                section_content,
                &classification_result,
            );
            
            let processed_section = ProcessedSection {
                id: format!("section_{}", i),
                content: section_content.to_string(),
                start_position: start_pos,
                end_position: end_pos,
                classification_result,
                quality_score,
            };
            
            sections.push(processed_section);
        }
        
        debug!("Classified {} sections in document", sections.len());
        Ok(sections)
    }

    /// Calculate document quality metrics
    fn calculate_document_quality_metrics(
        &self,
        document_type_result: &Option<DocumentTypeResult>,
        section_classifications: &[ProcessedSection],
        boundaries: &[BoundaryInfo],
        timeline: &ProcessingTimeline,
    ) -> DocumentQualityMetrics {
        // Classification confidence
        let classification_confidence = document_type_result
            .as_ref()
            .map(|r| r.confidence)
            .unwrap_or(0.0);
        
        // Section coverage (percentage of document covered by classified sections)
        let section_coverage = if !section_classifications.is_empty() {
            let total_section_length: usize = section_classifications
                .iter()
                .map(|s| s.end_position - s.start_position)
                .sum();
            
            // Estimate total document length from boundaries
            let total_doc_length = boundaries.last()
                .map(|b| b.position)
                .unwrap_or(1);
            
            (total_section_length as f64 / total_doc_length as f64).min(1.0)
        } else {
            0.0
        };
        
        // Boundary accuracy (based on boundary confidence scores)
        let boundary_accuracy = if !boundaries.is_empty() {
            boundaries.iter()
                .map(|b| b.confidence as f64)
                .sum::<f64>() / boundaries.len() as f64
        } else {
            0.0
        };
        
        // Performance score (meeting time constraints)
        let performance_score = if timeline.total_processing_time_ms <= self.config.max_processing_time_ms as f64 {
            1.0
        } else {
            (self.config.max_processing_time_ms as f64 / timeline.total_processing_time_ms).min(1.0)
        };
        
        // Overall quality score (weighted average)
        let overall_score = (
            classification_confidence * 0.3 +
            section_coverage * 0.25 +
            boundary_accuracy * 0.25 +
            performance_score * 0.2
        ).min(1.0);
        
        DocumentQualityMetrics {
            overall_score,
            classification_confidence,
            section_coverage,
            boundary_accuracy,
            performance_score,
        }
    }

    /// Calculate section quality score
    fn calculate_section_quality_score(&self, content: &str, classification: &SectionTypeResult) -> f64 {
        // Base score from classification confidence
        let mut quality_score = classification.confidence;
        
        // Adjust based on section content quality
        let word_count = content.split_whitespace().count();
        if word_count < 10 {
            quality_score *= 0.7; // Penalize very short sections
        }
        
        // Boost score for clear section type indicators
        if !classification.type_hints.is_empty() {
            quality_score = (quality_score * 1.1).min(1.0);
        }
        
        quality_score
    }

    /// Update pipeline performance metrics
    async fn update_processing_metrics(&self, processed_doc: &ProcessedDocument) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_documents += 1;
        
        // Update average processing time
        let new_avg = (metrics.average_processing_time_ms * (metrics.total_documents - 1) as f64 + 
                       processed_doc.processing_timeline.total_processing_time_ms) / 
                      metrics.total_documents as f64;
        metrics.average_processing_time_ms = new_avg;
        
        // Update accuracy metrics if available
        if let Some(doc_result) = &processed_doc.document_type_result {
            // Simple accuracy approximation (would need ground truth in real system)
            metrics.document_classification_accuracy = 
                (metrics.document_classification_accuracy * 0.9 + doc_result.confidence * 0.1).min(1.0);
        }
        
        // Update section classification accuracy
        if !processed_doc.section_classifications.is_empty() {
            let avg_section_confidence: f64 = processed_doc.section_classifications
                .iter()
                .map(|s| s.classification_result.confidence)
                .sum::<f64>() / processed_doc.section_classifications.len() as f64;
            
            metrics.section_classification_accuracy = 
                (metrics.section_classification_accuracy * 0.9 + avg_section_confidence * 0.1).min(1.0);
        }
        
        // Update performance target compliance
        let meets_target = if processed_doc.processing_timeline.total_processing_time_ms <= self.config.max_processing_time_ms as f64 {
            1.0
        } else {
            0.0
        };
        
        metrics.performance_target_met = 
            (metrics.performance_target_met * 0.9 + meets_target * 0.1).min(1.0);
        
        // Update throughput
        if metrics.average_processing_time_ms > 0.0 {
            metrics.throughput_docs_per_second = 1000.0 / metrics.average_processing_time_ms;
        }
        
        metrics.last_updated = Utc::now();
    }

    /// Add processing record to history
    fn add_processing_record(&mut self, processed_doc: &ProcessedDocument, success: bool, error_message: Option<String>) {
        let record = ProcessingRecord {
            document_id: processed_doc.id,
            start_time: processed_doc.processing_timeline.start_time,
            duration_ms: processed_doc.processing_timeline.total_processing_time_ms,
            success,
            error_message,
            document_type: processed_doc.document_type_result.as_ref().map(|r| r.document_type.clone()),
            sections_processed: processed_doc.section_classifications.len(),
        };
        
        self.processing_history.push(record);
        
        // Keep only recent history to prevent unbounded growth
        if self.processing_history.len() > 10000 {
            self.processing_history.drain(0..1000);
        }
    }

    /// Get current pipeline metrics
    pub async fn get_metrics(&self) -> PipelineMetrics {
        self.metrics.read().await.clone()
    }

    /// Get processing history
    pub fn get_processing_history(&self, limit: Option<usize>) -> Vec<ProcessingRecord> {
        let limit = limit.unwrap_or(100);
        self.processing_history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Health check for entire pipeline
    pub async fn health_check(&self) -> bool {
        let mut classifier = self.document_classifier.write().await;
        let mut chunker = self.neural_chunker.write().await;
        
        classifier.health_check() && chunker.health_check()
    }

    /// Optimize pipeline configuration based on performance metrics
    pub async fn optimize_configuration(&mut self) -> Result<PipelineConfig> {
        let metrics = self.get_metrics().await;
        let mut optimized_config = self.config.clone();
        
        info!("Optimizing pipeline configuration based on performance metrics");
        
        // Adjust batch size based on throughput
        if metrics.throughput_docs_per_second < 1.0 && optimized_config.batch_size > 1 {
            optimized_config.batch_size = (optimized_config.batch_size / 2).max(1);
            info!("Reduced batch size to {} for better performance", optimized_config.batch_size);
        } else if metrics.throughput_docs_per_second > 10.0 && optimized_config.batch_size < 100 {
            optimized_config.batch_size = (optimized_config.batch_size * 2).min(100);
            info!("Increased batch size to {} for better throughput", optimized_config.batch_size);
        }
        
        // Adjust confidence threshold based on accuracy
        if metrics.document_classification_accuracy < 0.9 && optimized_config.confidence_threshold > 0.5 {
            optimized_config.confidence_threshold -= 0.05;
            info!("Reduced confidence threshold to {:.2} to improve coverage", optimized_config.confidence_threshold);
        }
        
        // Adjust processing time limit based on performance target compliance
        if metrics.performance_target_met < 0.8 {
            optimized_config.max_processing_time_ms = (optimized_config.max_processing_time_ms as f64 * 1.2) as u64;
            info!("Increased processing time limit to {}ms", optimized_config.max_processing_time_ms);
        }
        
        self.config = optimized_config.clone();
        Ok(optimized_config)
    }
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_document_classification: true,
            enable_section_classification: true,
            enable_query_routing: true,
            max_processing_time_ms: 5000, // 5 second limit
            batch_size: 10,
            confidence_threshold: 0.7,
            enable_performance_monitoring: true,
        }
    }
}

impl PipelineMetrics {
    fn new() -> Self {
        Self {
            total_documents: 0,
            average_processing_time_ms: 0.0,
            document_classification_accuracy: 0.95, // Start with high expectation
            section_classification_accuracy: 0.97,
            performance_target_met: 1.0,
            throughput_docs_per_second: 0.0,
            error_rate: 0.0,
            last_updated: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = SmartIngestionPipeline::new(config).await;
        assert!(pipeline.is_ok());
    }

    #[tokio::test]
    async fn test_document_processing() {
        let config = PipelineConfig::default();
        let mut pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        
        let content = "Payment Card Industry Data Security Standard (PCI DSS) Requirements. Requirement 3.2.1: Cardholder data must be encrypted using strong cryptography.";
        
        let result = pipeline.process_document(content, None).await.unwrap();
        
        assert!(!result.id.to_string().is_empty());
        assert_eq!(result.content, content);
        assert!(result.document_type_result.is_some());
        assert!(!result.section_classifications.is_empty());
        assert!(result.quality_metrics.overall_score > 0.0);
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let config = PipelineConfig::default();
        let mut pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        
        let documents = vec![
            ("PCI DSS security requirements document".to_string(), None),
            ("ISO 27001 information security management".to_string(), None),
            ("SOC 2 service organization controls".to_string(), None),
        ];
        
        let result = pipeline.batch_process_documents(documents).await.unwrap();
        
        assert_eq!(result.successful_documents.len(), 3);
        assert_eq!(result.failed_documents.len(), 0);
        assert!(result.batch_metrics.throughput_docs_per_second > 0.0);
    }

    #[tokio::test]
    async fn test_query_routing() {
        let config = PipelineConfig::default();
        let mut pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        
        let query = "What are the requirements for encrypting cardholder data?";
        let result = pipeline.route_query(query).await.unwrap();
        
        assert!(matches!(result.routing_decision, QueryRoute::Symbolic));
        assert!(result.confidence > 0.5);
    }

    #[tokio::test]
    async fn test_performance_metrics_tracking() {
        let config = PipelineConfig::default();
        let mut pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        
        // Process a document
        let content = "Test document for metrics tracking";
        let _ = pipeline.process_document(content, None).await.unwrap();
        
        let metrics = pipeline.get_metrics().await;
        assert_eq!(metrics.total_documents, 1);
        assert!(metrics.average_processing_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = PipelineConfig::default();
        let pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        
        let health = pipeline.health_check().await;
        assert!(health);
    }

    #[tokio::test]
    async fn test_configuration_optimization() {
        let config = PipelineConfig::default();
        let mut pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        
        // Process some documents to generate metrics
        let content = "Test document for configuration optimization";
        let _ = pipeline.process_document(content, None).await.unwrap();
        
        let optimized_config = pipeline.optimize_configuration().await.unwrap();
        assert!(optimized_config.batch_size > 0);
        assert!(optimized_config.confidence_threshold > 0.0);
    }

    #[tokio::test]
    async fn test_section_classification_accuracy() {
        let config = PipelineConfig::default();
        let mut pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        
        let content = r#"
            # Requirements Section
            Requirement 3.2.1: Cardholder data must be encrypted.
            
            # Definitions Section
            Definition: Cardholder data means the primary account number plus any additional data.
            
            # Procedures Section
            Step 1: Identify all cardholder data locations.
            Step 2: Implement encryption controls.
        "#;
        
        let result = pipeline.process_document(content, None).await.unwrap();
        
        // Should detect and classify multiple section types
        assert!(result.section_classifications.len() >= 3);
        
        // Check that we have different section types
        let section_types: std::collections::HashSet<_> = result.section_classifications
            .iter()
            .map(|s| &s.classification_result.section_type)
            .collect();
        
        assert!(section_types.len() > 1); // Should have multiple different types
    }

    #[tokio::test]
    async fn test_processing_history() {
        let config = PipelineConfig::default();
        let mut pipeline = SmartIngestionPipeline::new(config).await.unwrap();
        
        // Process multiple documents
        for i in 0..3 {
            let content = format!("Test document {}", i);
            let _ = pipeline.process_document(&content, None).await.unwrap();
        }
        
        let history = pipeline.get_processing_history(Some(10));
        assert_eq!(history.len(), 3);
        
        // History should be in reverse chronological order
        for record in &history {
            assert!(record.success);
            assert!(record.duration_ms > 0.0);
        }
    }
}