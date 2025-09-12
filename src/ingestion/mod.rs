//! Document ingestion module with neural classification
//!
//! Provides intelligent document processing through:
//! - Neural document type classification (PCI-DSS, ISO-27001, SOC2, NIST)
//! - Section type classification (Requirements, Definitions, Procedures)
//! - Query routing for optimal processing (symbolic vs graph vs vector)
//! - Smart ingestion pipeline coordination
//!
//! CONSTRAINT-003: Uses ruv-fann v0.1.6 for classification ONLY with <10ms inference

pub mod classification;
pub mod pipeline;

pub use classification::{
    DocumentClassifier, DocumentTypeResult, SectionTypeResult, QueryRoutingResult,
    DocumentType, SectionType, QueryRoute, ClassificationMetrics,
    FeatureExtractor, DocumentFeatures, SectionFeatures, QueryFeatures,
};
pub use pipeline::{
    SmartIngestionPipeline, ProcessedDocument, ProcessedSection,
    PipelineConfig, PipelineMetrics, ProcessingTimeline,
    DocumentQualityMetrics, BatchProcessingResult,
};