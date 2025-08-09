//! Database schema definitions for MongoDB Vector Storage
//! 
//! This module defines the document structures used for storing chunks,
//! embeddings, and metadata in MongoDB with proper indexing support.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use bson::oid::ObjectId;
use mongodb::bson;
use std::collections::HashMap;

/// Main document structure for storing chunks with embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDocument {
    /// MongoDB ObjectId
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<ObjectId>,
    
    /// Unique chunk identifier
    pub chunk_id: Uuid,
    
    /// Text content of the chunk
    #[serde(with = "bson::serde_helpers::chrono_datetime_as_bson_datetime")]
    pub content: String,
    
    /// Vector embedding for similarity search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f64>>,
    
    /// Chunk metadata
    pub metadata: ChunkMetadata,
    
    /// References to other chunks
    #[serde(default)]
    pub references: Vec<ChunkReference>,
    
    /// Creation timestamp
    #[serde(with = "bson::serde_helpers::chrono_datetime_as_bson_datetime")]
    pub created_at: DateTime<Utc>,
    
    /// Last update timestamp
    #[serde(with = "bson::serde_helpers::chrono_datetime_as_bson_datetime")]
    pub updated_at: DateTime<Utc>,
    
    /// Document version for optimistic locking
    pub version: i64,
}

/// Metadata associated with a chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Original document identifier
    pub document_id: Uuid,
    
    /// Document title or name
    pub title: String,
    
    /// Chunk position within the document
    pub chunk_index: usize,
    
    /// Total number of chunks in the document
    pub total_chunks: usize,
    
    /// Chunk size in characters
    pub chunk_size: usize,
    
    /// Overlap with adjacent chunks
    pub overlap_size: usize,
    
    /// Source file path or URL
    pub source_path: String,
    
    /// MIME type of the source document
    pub mime_type: String,
    
    /// Language of the content
    #[serde(default = "default_language")]
    pub language: String,
    
    /// Document tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
    
    /// Custom metadata fields
    #[serde(default)]
    pub custom_fields: HashMap<String, CustomFieldValue>,
    
    /// Hash of the chunk content for deduplication
    pub content_hash: String,
    
    /// Confidence score for chunk boundary detection
    #[serde(default)]
    pub boundary_confidence: Option<f32>,
}

/// Custom field values that can be stored in metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CustomFieldValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<String>),
}

/// Reference to another chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkReference {
    /// Target chunk ID
    pub chunk_id: Uuid,
    
    /// Type of reference relationship
    pub reference_type: ReferenceType,
    
    /// Confidence score of the reference
    pub confidence: f32,
    
    /// Context around the reference
    pub context: String,
}

/// Types of references between chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceType {
    /// Sequential chunks from the same document
    Sequential,
    
    /// Chunks that mention similar topics
    Semantic,
    
    /// Cross-references between documents
    CrossDocument,
    
    /// Parent-child relationship
    Hierarchical,
    
    /// Custom reference type
    Custom(String),
}

/// Separate collection for storing metadata-only documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataDocument {
    /// MongoDB ObjectId
    #[serde(rename = "_id", skip_serializing_if = "Option::is_none")]
    pub id: Option<ObjectId>,
    
    /// Original document identifier
    pub document_id: Uuid,
    
    /// Document metadata
    pub metadata: DocumentMetadata,
    
    /// Processing statistics
    pub processing_stats: ProcessingStats,
    
    /// Creation timestamp
    #[serde(with = "bson::serde_helpers::chrono_datetime_as_bson_datetime")]
    pub created_at: DateTime<Utc>,
    
    /// Last update timestamp
    #[serde(with = "bson::serde_helpers::chrono_datetime_as_bson_datetime")]
    pub updated_at: DateTime<Utc>,
}

/// Document-level metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Document title
    pub title: String,
    
    /// Author information
    pub author: Option<String>,
    
    /// Creation date of the original document
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(with = "bson::serde_helpers::chrono_datetime_as_bson_datetime")]
    pub document_created_at: Option<DateTime<Utc>>,
    
    /// File size in bytes
    pub file_size: u64,
    
    /// Document format/type
    pub format: String,
    
    /// Document summary
    pub summary: Option<String>,
    
    /// Keywords extracted from the document
    #[serde(default)]
    pub keywords: Vec<String>,
    
    /// Document classification
    pub classification: Option<String>,
    
    /// Security level
    pub security_level: SecurityLevel,
}

/// Security classification levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SecurityLevel {
    Public,
    Internal,
    Confidential,
    Restricted,
    TopSecret,
}

/// Processing statistics for a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Total processing time in milliseconds
    pub processing_time_ms: u64,
    
    /// Number of chunks created
    pub chunk_count: usize,
    
    /// Average chunk size
    pub avg_chunk_size: usize,
    
    /// Embedding model used
    pub embedding_model: String,
    
    /// Embedding dimension
    pub embedding_dimension: usize,
    
    /// Processing errors encountered
    #[serde(default)]
    pub errors: Vec<ProcessingError>,
    
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Processing error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingError {
    /// Error type
    pub error_type: String,
    
    /// Error message
    pub message: String,
    
    /// Chunk index where error occurred
    pub chunk_index: Option<usize>,
    
    /// Timestamp of the error
    #[serde(with = "bson::serde_helpers::chrono_datetime_as_bson_datetime")]
    pub timestamp: DateTime<Utc>,
}

/// Quality metrics for processed documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Average boundary confidence score
    pub avg_boundary_confidence: f32,
    
    /// Semantic coherence score
    pub coherence_score: f32,
    
    /// Information density score
    pub information_density: f32,
    
    /// Overlap quality score
    pub overlap_quality: f32,
    
    /// Overall quality score (0-100)
    pub overall_quality: f32,
}

/// Bulk insert request for multiple chunks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkInsertRequest {
    /// Chunks to insert
    pub chunks: Vec<ChunkDocument>,
    
    /// Whether to update existing chunks
    pub upsert: bool,
    
    /// Batch size for processing
    pub batch_size: Option<usize>,
}

/// Bulk insert response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkInsertResponse {
    /// Number of successfully inserted documents
    pub inserted_count: u64,
    
    /// Number of documents that were updated
    pub modified_count: u64,
    
    /// Number of documents that failed to insert
    pub failed_count: u64,
    
    /// Errors encountered during insertion
    pub errors: Vec<BulkInsertError>,
    
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Bulk insert error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BulkInsertError {
    /// Index of the failed document
    pub document_index: usize,
    
    /// Error message
    pub error_message: String,
    
    /// Error code if available
    pub error_code: Option<i32>,
}

impl ChunkDocument {
    /// Create a new chunk document
    pub fn new(
        chunk_id: Uuid,
        content: String,
        metadata: ChunkMetadata,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: None,
            chunk_id,
            content,
            embedding: None,
            metadata,
            references: Vec::new(),
            created_at: now,
            updated_at: now,
            version: 1,
        }
    }
    
    /// Update the chunk with new embedding
    pub fn with_embedding(mut self, embedding: Vec<f64>) -> Self {
        self.embedding = Some(embedding);
        self.updated_at = Utc::now();
        self.version += 1;
        self
    }
    
    /// Add a reference to another chunk
    pub fn add_reference(&mut self, reference: ChunkReference) {
        self.references.push(reference);
        self.updated_at = Utc::now();
        self.version += 1;
    }
    
    /// Calculate content hash for deduplication
    pub fn calculate_content_hash(&self) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(self.content.as_bytes());
        format!("{:x}", hasher.finalize())
    }
    
    /// Validate the chunk document
    pub fn validate(&self) -> Result<(), String> {
        if self.content.is_empty() {
            return Err("Chunk content cannot be empty".to_string());
        }
        
        if self.metadata.chunk_size == 0 {
            return Err("Chunk size must be greater than 0".to_string());
        }
        
        if self.metadata.chunk_index >= self.metadata.total_chunks {
            return Err("Chunk index cannot be greater than or equal to total chunks".to_string());
        }
        
        if let Some(embedding) = &self.embedding {
            if embedding.is_empty() {
                return Err("Embedding cannot be empty if present".to_string());
            }
        }
        
        Ok(())
    }
}

impl ChunkMetadata {
    /// Create new chunk metadata
    pub fn new(
        document_id: Uuid,
        title: String,
        chunk_index: usize,
        total_chunks: usize,
        source_path: String,
    ) -> Self {
        Self {
            document_id,
            title,
            chunk_index,
            total_chunks,
            chunk_size: 0,
            overlap_size: 0,
            source_path,
            mime_type: "text/plain".to_string(),
            language: default_language(),
            tags: Vec::new(),
            custom_fields: HashMap::new(),
            content_hash: String::new(),
            boundary_confidence: None,
        }
    }
    
    /// Add a tag to the metadata
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }
    
    /// Set a custom field
    pub fn set_custom_field(&mut self, key: String, value: CustomFieldValue) {
        self.custom_fields.insert(key, value);
    }
    
    /// Get a custom field
    pub fn get_custom_field(&self, key: &str) -> Option<&CustomFieldValue> {
        self.custom_fields.get(key)
    }
}

impl MetadataDocument {
    /// Create a new metadata document
    pub fn new(document_id: Uuid, metadata: DocumentMetadata) -> Self {
        let now = Utc::now();
        Self {
            id: None,
            document_id,
            metadata,
            processing_stats: ProcessingStats::default(),
            created_at: now,
            updated_at: now,
        }
    }
}

impl Default for ProcessingStats {
    fn default() -> Self {
        Self {
            processing_time_ms: 0,
            chunk_count: 0,
            avg_chunk_size: 0,
            embedding_model: "unknown".to_string(),
            embedding_dimension: 0,
            errors: Vec::new(),
            quality_metrics: QualityMetrics::default(),
        }
    }
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            avg_boundary_confidence: 0.0,
            coherence_score: 0.0,
            information_density: 0.0,
            overlap_quality: 0.0,
            overall_quality: 0.0,
        }
    }
}

impl Default for SecurityLevel {
    fn default() -> Self {
        SecurityLevel::Internal
    }
}

fn default_language() -> String {
    "en".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_chunk_document_creation() {
        let metadata = ChunkMetadata::new(
            Uuid::new_v4(),
            "Test Document".to_string(),
            0,
            10,
            "/path/to/test.txt".to_string(),
        );
        
        let chunk = ChunkDocument::new(
            Uuid::new_v4(),
            "This is test content".to_string(),
            metadata,
        );
        
        assert!(chunk.validate().is_ok());
        assert_eq!(chunk.version, 1);
        assert!(chunk.embedding.is_none());
    }
    
    #[test]
    fn test_chunk_with_embedding() {
        let metadata = ChunkMetadata::new(
            Uuid::new_v4(),
            "Test Document".to_string(),
            0,
            10,
            "/path/to/test.txt".to_string(),
        );
        
        let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let chunk = ChunkDocument::new(
            Uuid::new_v4(),
            "This is test content".to_string(),
            metadata,
        ).with_embedding(embedding.clone());
        
        assert!(chunk.validate().is_ok());
        assert_eq!(chunk.embedding, Some(embedding));
        assert_eq!(chunk.version, 2);
    }
    
    #[test]
    fn test_custom_field_operations() {
        let mut metadata = ChunkMetadata::new(
            Uuid::new_v4(),
            "Test Document".to_string(),
            0,
            10,
            "/path/to/test.txt".to_string(),
        );
        
        metadata.set_custom_field(
            "priority".to_string(),
            CustomFieldValue::String("high".to_string()),
        );
        
        metadata.set_custom_field(
            "score".to_string(),
            CustomFieldValue::Number(95.5),
        );
        
        assert!(matches!(
            metadata.get_custom_field("priority"),
            Some(CustomFieldValue::String(s)) if s == "high"
        ));
        
        assert!(matches!(
            metadata.get_custom_field("score"),
            Some(CustomFieldValue::Number(n)) if *n == 95.5
        ));
    }
}