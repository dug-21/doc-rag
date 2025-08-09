//! Chunk data structures and metadata handling

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Document chunk with content and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique chunk identifier
    pub id: Uuid,
    /// Chunk content text
    pub content: String,
    /// Chunk metadata
    pub metadata: ChunkMetadata,
    /// Vector embeddings (populated by embedding service)
    pub embeddings: Option<Vec<f32>>,
    /// References to other chunks or documents
    pub references: Vec<ChunkReference>,
    /// Previous chunk in sequence
    pub prev_chunk_id: Option<Uuid>,
    /// Next chunk in sequence
    pub next_chunk_id: Option<Uuid>,
    /// Parent chunk (for hierarchical chunking)
    pub parent_chunk_id: Option<Uuid>,
    /// Child chunks (for hierarchical chunking)
    pub child_chunk_ids: Vec<Uuid>,
}

/// Comprehensive chunk metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Source document identifier
    pub document_id: String,
    /// Chunk index within document
    pub chunk_index: usize,
    /// Total number of chunks in document
    pub total_chunks: usize,
    /// Start offset in original document
    pub start_offset: usize,
    /// End offset in original document
    pub end_offset: usize,
    /// Section title or heading
    pub section_title: Option<String>,
    /// Section name (legacy field for compatibility)
    pub section: Option<String>,
    /// Subsection name
    pub subsection: Option<String>,
    /// Heading level (0-6 for HTML-style headings)
    pub heading_level: usize,
    /// Document position (0.0 to 1.0)
    pub document_position: f64,
    /// Character count
    pub character_count: usize,
    /// Word count
    pub word_count: usize,
    /// Paragraph count
    pub paragraph_count: usize,
    /// Line count
    pub line_count: usize,
    /// Page number in original document
    pub page_number: Option<u32>,
    /// Language of the content
    pub language: String,
    /// Content type (text/plain, text/markdown, etc.)
    pub content_type: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Semantic tags for categorization
    pub semantic_tags: Vec<String>,
    /// Quality score (0.0 to 1.0)
    pub quality_score: f64,
}

/// Reference to another chunk or document section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkReference {
    /// Type of reference
    pub reference_type: ReferenceType,
    /// Target chunk or document ID
    pub target_id: String,
    /// Reference context or description
    pub context: Option<String>,
    /// Confidence score for the reference
    pub confidence: f64,
}

/// Types of chunk references
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReferenceType {
    /// Reference to previous chunk for context
    PreviousChunk,
    /// Reference to next chunk for continuity
    NextChunk,
    /// Cross-reference to related content
    CrossReference,
    /// Reference to parent section or document
    Parent,
    /// Reference to child subsections
    Child,
    /// Citation or footnote reference
    Citation,
    /// Footnote reference
    Footnote,
    /// Table reference
    TableReference,
    /// Figure reference
    FigureReference,
    /// Section reference
    SectionReference,
    /// Table or figure reference (legacy)
    TableFigure,
    /// External document or URL reference
    External,
    /// External link
    ExternalLink,
}

impl Chunk {
    /// Creates a new chunk with basic metadata
    pub fn new(id: Uuid, content: String, document_id: String) -> Self {
        let word_count = content.split_whitespace().count();
        let character_count = content.len();
        let paragraph_count = content.split("\n\n").filter(|p| !p.trim().is_empty()).count().max(1);
        let line_count = content.lines().count();
        
        Self {
            id,
            content,
            metadata: ChunkMetadata {
                document_id,
                chunk_index: 0,
                total_chunks: 1,
                start_offset: 0,
                end_offset: 0,
                section_title: None,
                section: None,
                subsection: None,
                heading_level: 0,
                document_position: 0.0,
                character_count,
                word_count,
                paragraph_count,
                line_count,
                page_number: None,
                language: "en".to_string(),
                content_type: "text/plain".to_string(),
                created_at: Utc::now(),
                semantic_tags: Vec::new(),
                quality_score: 0.8,
            },
            embeddings: None,
            references: Vec::new(),
            prev_chunk_id: None,
            next_chunk_id: None,
            parent_chunk_id: None,
            child_chunk_ids: Vec::new(),
        }
    }

    /// Returns the content length in characters
    pub fn content_length(&self) -> usize {
        self.content.len()
    }

    /// Returns the content length in words (approximate)
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// Returns the content length in bytes
    pub fn byte_size(&self) -> usize {
        self.content.len()
    }

    /// Checks if chunk has embeddings
    pub fn has_embeddings(&self) -> bool {
        self.embeddings.is_some()
    }

    /// Returns embedding dimension if available
    pub fn embedding_dimension(&self) -> Option<usize> {
        self.embeddings.as_ref().map(|e| e.len())
    }

    /// Adds a reference to another chunk or document
    pub fn add_reference(&mut self, reference: ChunkReference) {
        self.references.push(reference);
    }

    /// Removes references of a specific type
    pub fn remove_references_of_type(&mut self, ref_type: ReferenceType) {
        self.references.retain(|r| r.reference_type != ref_type);
    }

    /// Gets all references of a specific type
    pub fn get_references_of_type(&self, ref_type: ReferenceType) -> Vec<&ChunkReference> {
        self.references.iter()
            .filter(|r| r.reference_type == ref_type)
            .collect()
    }

    /// Updates the quality score based on various factors
    pub fn update_quality_score(&mut self) {
        let mut score = 0.0;
        let mut factors = 0;

        // Content length factor (optimal range: 200-800 characters)
        let length_score = if self.content.len() < 200 {
            self.content.len() as f64 / 200.0 * 0.8
        } else if self.content.len() > 800 {
            0.8 + (1.0 - (self.content.len() as f64 - 800.0) / 1000.0).max(0.0) * 0.2
        } else {
            0.8 + (self.content.len() as f64 - 200.0) / 600.0 * 0.2
        };
        score += length_score;
        factors += 1;

        // Completeness factor (sentence endings, proper punctuation)
        let completeness_score = self.calculate_completeness_score();
        score += completeness_score;
        factors += 1;

        // Context factor (presence of references)
        let context_score = if self.references.is_empty() {
            0.6 // Standalone chunks get lower score
        } else {
            0.8 + (self.references.len().min(5) as f64 / 5.0) * 0.2
        };
        score += context_score;
        factors += 1;

        // Structure factor (section titles, proper formatting)
        let structure_score = self.calculate_structure_score();
        score += structure_score;
        factors += 1;

        self.metadata.quality_score = (score / factors as f64).min(1.0).max(0.0);
    }

    /// Calculates completeness score based on content analysis
    fn calculate_completeness_score(&self) -> f64 {
        let content = &self.content;
        let mut score = 0.0;

        // Check for sentence endings
        let sentence_endings = content.matches(&['.', '!', '?'][..]).count();
        let estimated_sentences = content.split_whitespace().count() / 15; // ~15 words per sentence
        if estimated_sentences > 0 {
            let sentence_ratio = sentence_endings as f64 / estimated_sentences as f64;
            score += sentence_ratio.min(1.0) * 0.4;
        } else {
            score += 0.2; // Partial credit for fragments
        }

        // Check for proper capitalization
        if content.chars().next().map_or(false, |c| c.is_uppercase()) {
            score += 0.2;
        }

        // Check for balanced quotes/parentheses
        let quote_balance = content.matches('"').count() % 2 == 0;
        let paren_balance = content.matches('(').count() == content.matches(')').count();
        if quote_balance && paren_balance {
            score += 0.2;
        }

        // Check for reasonable word distribution (not just repetition)
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.len() > 5 {
            let unique_words: std::collections::HashSet<&str> = words.iter().cloned().collect();
            let diversity = unique_words.len() as f64 / words.len() as f64;
            score += diversity * 0.2;
        }

        score.min(1.0)
    }

    /// Calculates structure score based on formatting and organization
    fn calculate_structure_score(&self) -> f64 {
        let mut score: f64 = 0.0;

        // Section title presence
        if self.metadata.section_title.is_some() {
            score += 0.3;
        }

        // Page number tracking
        if self.metadata.page_number.is_some() {
            score += 0.2;
        }

        // Semantic tags
        if !self.metadata.semantic_tags.is_empty() {
            score += 0.2;
        }

        // Proper content type
        if self.metadata.content_type != "text/plain" {
            score += 0.1;
        }

        // Language detection
        if self.metadata.language != "en" || self.is_language_detected() {
            score += 0.1;
        }

        // Structural elements in content (headers, lists, etc.)
        if self.has_structural_elements() {
            score += 0.1;
        }

        score.min(1.0)
    }

    /// Checks if language was properly detected (simplified)
    fn is_language_detected(&self) -> bool {
        // This is a simplified check - in production, use proper language detection
        let non_ascii_chars = self.content.chars().filter(|c| !c.is_ascii()).count();
        non_ascii_chars > 0 || self.metadata.language != "en"
    }

    /// Checks for structural elements in content
    fn has_structural_elements(&self) -> bool {
        let content = &self.content;
        
        // Headers (markdown-style)
        if content.contains('#') {
            return true;
        }

        // Lists
        if content.contains("- ") || content.contains("* ") {
            return true;
        }
        
        // Numbered lists (simple pattern)
        for line in content.lines() {
            let trimmed = line.trim();
            if let Some(first_char) = trimmed.chars().next() {
                if first_char.is_numeric() && trimmed.contains(". ") {
                    return true;
                }
            }
        }

        // Tables (simple detection)
        if content.contains('|') && content.matches('|').count() > 3 {
            return true;
        }

        // Code blocks
        if content.contains("```") || content.contains("    ") {
            return true;
        }

        false
    }

    /// Returns a summary of the chunk for debugging/logging
    pub fn summary(&self) -> String {
        let content_preview = if self.content.len() > 100 {
            format!("{}...", &self.content[..100])
        } else {
            self.content.clone()
        };

        format!(
            "Chunk {} ({}): {} chars, {} refs, quality: {:.2}, preview: \"{}\"",
            self.id,
            self.metadata.chunk_index,
            self.content.len(),
            self.references.len(),
            self.metadata.quality_score,
            content_preview.replace('\n', " ")
        )
    }

    /// Validates chunk data consistency
    pub fn validate(&self) -> Result<(), ChunkValidationError> {
        // Check ID
        if self.id.is_nil() {
            return Err(ChunkValidationError::InvalidId);
        }

        // Check content
        if self.content.is_empty() {
            return Err(ChunkValidationError::EmptyContent);
        }

        // Check metadata consistency
        if self.metadata.start_offset >= self.metadata.end_offset {
            return Err(ChunkValidationError::InvalidOffsets);
        }

        if self.metadata.quality_score < 0.0 || self.metadata.quality_score > 1.0 {
            return Err(ChunkValidationError::InvalidQualityScore);
        }

        // Check embeddings dimension consistency
        if let Some(ref embeddings) = self.embeddings {
            if embeddings.is_empty() {
                return Err(ChunkValidationError::EmptyEmbeddings);
            }
            
            // Check for NaN or infinite values
            for &value in embeddings {
                if !value.is_finite() {
                    return Err(ChunkValidationError::InvalidEmbeddings);
                }
            }
        }

        // Validate references
        for reference in &self.references {
            if reference.confidence < 0.0 || reference.confidence > 1.0 {
                return Err(ChunkValidationError::InvalidReferenceConfidence);
            }
        }

        Ok(())
    }
}

impl ChunkReference {
    /// Creates a new chunk reference
    pub fn new(reference_type: ReferenceType, target_id: String, confidence: f64) -> Self {
        Self {
            reference_type,
            target_id,
            context: None,
            confidence: confidence.min(1.0).max(0.0),
        }
    }

    /// Creates a reference with context
    pub fn with_context(
        reference_type: ReferenceType, 
        target_id: String, 
        context: String, 
        confidence: f64
    ) -> Self {
        Self {
            reference_type,
            target_id,
            context: Some(context),
            confidence: confidence.min(1.0).max(0.0),
        }
    }

    /// Checks if this is a high-confidence reference
    pub fn is_high_confidence(&self) -> bool {
        self.confidence >= 0.8
    }

    /// Checks if this is a contextual reference (has context information)
    pub fn is_contextual(&self) -> bool {
        self.context.is_some()
    }
}

impl ChunkMetadata {
    /// Creates metadata with minimal required fields
    pub fn minimal(document_id: String, chunk_index: usize) -> Self {
        Self {
            document_id,
            chunk_index,
            total_chunks: 1,
            start_offset: 0,
            end_offset: 0,
            section_title: None,
            section: None,
            subsection: None,
            heading_level: 0,
            document_position: 0.0,
            character_count: 0,
            word_count: 0,
            paragraph_count: 1,
            line_count: 1,
            page_number: None,
            language: "en".to_string(),
            content_type: "text/plain".to_string(),
            created_at: Utc::now(),
            semantic_tags: Vec::new(),
            quality_score: 0.5,
        }
    }

    /// Adds a semantic tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.semantic_tags.contains(&tag) {
            self.semantic_tags.push(tag);
        }
    }

    /// Removes a semantic tag
    pub fn remove_tag(&mut self, tag: &str) {
        self.semantic_tags.retain(|t| t != tag);
    }

    /// Checks if metadata has a specific tag
    pub fn has_tag(&self, tag: &str) -> bool {
        self.semantic_tags.contains(&tag.to_string())
    }

    /// Returns the age of the chunk in seconds
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.created_at).num_seconds()
    }

    /// Returns chunk position ratio within document (0.0 to 1.0)
    pub fn position_ratio(&self) -> f64 {
        if self.total_chunks <= 1 {
            0.0
        } else {
            self.chunk_index as f64 / (self.total_chunks - 1) as f64
        }
    }
}

/// Chunk validation errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum ChunkValidationError {
    #[error("Invalid or nil chunk ID")]
    InvalidId,
    #[error("Chunk content is empty")]
    EmptyContent,
    #[error("Invalid start/end offsets")]
    InvalidOffsets,
    #[error("Quality score must be between 0.0 and 1.0")]
    InvalidQualityScore,
    #[error("Embeddings vector is empty")]
    EmptyEmbeddings,
    #[error("Embeddings contain invalid values (NaN or infinite)")]
    InvalidEmbeddings,
    #[error("Reference confidence must be between 0.0 and 1.0")]
    InvalidReferenceConfidence,
    #[error("Validation failed for chunk {chunk_id}: {error}")]
    ValidationFailed {
        chunk_id: Uuid,
        error: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_creation() {
        let id = Uuid::new_v4();
        let chunk = Chunk::new(id, "Test content".to_string(), "doc-123".to_string());
        
        assert_eq!(chunk.id, id);
        assert_eq!(chunk.content, "Test content");
        assert_eq!(chunk.metadata.document_id, "doc-123");
        assert_eq!(chunk.content_length(), 12);
        assert_eq!(chunk.word_count(), 2);
    }

    #[test]
    fn test_chunk_validation() {
        let id = Uuid::new_v4();
        let mut chunk = Chunk::new(id, "Test content".to_string(), "doc-123".to_string());
        chunk.metadata.end_offset = 12;
        
        assert!(chunk.validate().is_ok());
        
        // Test invalid offsets
        chunk.metadata.start_offset = 20;
        chunk.metadata.end_offset = 10;
        assert!(chunk.validate().is_err());
    }

    #[test]
    fn test_quality_score_calculation() {
        let id = Uuid::new_v4();
        let mut chunk = Chunk::new(
            id, 
            "This is a well-formed sentence with proper punctuation. It has good structure and reasonable length.".to_string(), 
            "doc-123".to_string()
        );
        
        chunk.update_quality_score();
        assert!(chunk.metadata.quality_score >= 0.0);
        assert!(chunk.metadata.quality_score <= 1.0);
    }

    #[test]
    fn test_chunk_references() {
        let id = Uuid::new_v4();
        let mut chunk = Chunk::new(id, "Test content".to_string(), "doc-123".to_string());
        
        let reference = ChunkReference::new(
            ReferenceType::NextChunk,
            "chunk-456".to_string(),
            0.9
        );
        
        chunk.add_reference(reference);
        assert_eq!(chunk.references.len(), 1);
        
        let next_refs = chunk.get_references_of_type(ReferenceType::NextChunk);
        assert_eq!(next_refs.len(), 1);
        assert!(next_refs[0].is_high_confidence());
    }

    #[test]
    fn test_metadata_operations() {
        let mut metadata = ChunkMetadata::minimal("doc-123".to_string(), 0);
        
        metadata.add_tag("important".to_string());
        assert!(metadata.has_tag("important"));
        
        metadata.remove_tag("important");
        assert!(!metadata.has_tag("important"));
        
        assert_eq!(metadata.position_ratio(), 0.0);
    }

    #[test]
    fn test_chunk_summary() {
        let id = Uuid::new_v4();
        let chunk = Chunk::new(id, "Short content".to_string(), "doc-123".to_string());
        let summary = chunk.summary();
        
        assert!(summary.contains("Short content"));
        assert!(summary.contains(&id.to_string()));
        assert!(summary.contains("13 chars")); // Length of "Short content"
    }
}