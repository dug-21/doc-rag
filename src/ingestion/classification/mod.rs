//! Neural classification module for document ingestion
//!
//! Provides comprehensive neural classification capabilities for:
//! - Document type classification (PCI-DSS, ISO-27001, SOC2, NIST)
//! - Section type classification (Requirements, Definitions, Procedures)
//! - Query routing classification (symbolic vs graph vs vector)
//!
//! Built on ruv-fann v0.1.6 with <10ms inference performance

pub mod document_classifier;
pub mod feature_extractor;

pub use document_classifier::{
    DocumentClassifier, DocumentTypeResult, SectionTypeResult, QueryRoutingResult,
    DocumentType, SectionType, QueryRoute, ClassificationMetrics, ModelMetadata
};
pub use feature_extractor::{
    FeatureExtractor, DocumentFeatures, SectionFeatures, QueryFeatures,
    KeywordPatterns, StructurePatterns, QueryPatterns, FeatureMetrics
};