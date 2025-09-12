//! Smart ingestion pipeline module
//!
//! Coordinates neural classification and document processing through
//! a comprehensive pipeline that integrates:
//! - Document type classification
//! - Section type classification  
//! - Neural boundary detection
//! - Quality metrics and performance monitoring
//!
//! Designed for high-throughput processing with <10ms inference constraints

pub mod coordinator;

pub use coordinator::{
    SmartIngestionPipeline, ProcessedDocument, ProcessedSection,
    PipelineConfig, PipelineMetrics, ProcessingTimeline,
    DocumentQualityMetrics, BatchProcessingResult, BatchMetrics,
    ProcessingRecord,
};