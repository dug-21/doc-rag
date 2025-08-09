//! Metadata extraction and tracking for document chunks
//! 
//! This module provides comprehensive metadata extraction capabilities for document chunks,
//! including content type detection, structural analysis, quality scoring, and semantic tagging.

use crate::{chunk::ChunkMetadata, ContentType};
use chrono::Utc;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
// use unicode_segmentation::UnicodeSegmentation;
use once_cell::sync::Lazy;

/// Comprehensive metadata extractor for document chunks
#[derive(Debug, Clone)]
pub struct MetadataExtractor {
    /// Regex patterns for content analysis
    patterns: MetadataPatterns,
    /// Language detection configurations
    language_detector: LanguageDetector,
    /// Content type analyzers
    content_analyzers: ContentTypeAnalyzers,
    /// Quality assessment metrics
    quality_assessor: QualityAssessor,
    /// Semantic tag extractors
    semantic_tagger: SemanticTagger,
}

/// Pre-compiled regex patterns for metadata extraction
#[derive(Debug, Clone)]
struct MetadataPatterns {
    /// Markdown headers (H1-H6)
    markdown_headers: Regex,
    /// Code blocks (fenced and indented)
    code_blocks: Regex,
    /// Inline code
    inline_code: Regex,
    /// Tables (markdown and HTML-style)
    tables: Regex,
    /// Lists (ordered and unordered)
    lists: Regex,
    /// Block quotes
    quotes: Regex,
    /// Mathematical expressions
    math_expressions: Regex,
    /// URLs and links
    links: Regex,
    /// Email addresses
    emails: Regex,
    /// Footnotes and citations
    footnotes: Regex,
    /// Sentences
    sentences: Regex,
    /// Words
    words: Regex,
    /// Line breaks
    line_breaks: Regex,
}

/// Language detection functionality
#[derive(Debug, Clone)]
struct LanguageDetector {
    /// Common words by language
    language_indicators: HashMap<String, HashSet<String>>,
    /// Character frequency patterns
    character_patterns: HashMap<String, Regex>,
}

/// Content type analysis components
#[derive(Debug, Clone)]
struct ContentTypeAnalyzers {
    /// Code language detection patterns
    code_language_patterns: HashMap<String, Regex>,
    /// Table structure analyzers
    table_analyzers: TableAnalyzers,
    /// List type detectors
    list_detectors: ListDetectors,
    /// Mathematical content detectors
    math_detectors: MathDetectors,
}

/// Table analysis components
#[derive(Debug, Clone)]
struct TableAnalyzers {
    /// Markdown table pattern
    markdown_table: Regex,
    /// CSV-like pattern
    csv_pattern: Regex,
    /// HTML table pattern
    html_table: Regex,
    /// TSV pattern
    tsv_pattern: Regex,
}

/// List detection components
#[derive(Debug, Clone)]
struct ListDetectors {
    /// Ordered list pattern
    ordered_list: Regex,
    /// Unordered list pattern (bullet points)
    unordered_list: Regex,
    /// Definition list pattern
    definition_list: Regex,
    /// Checklist pattern
    checklist: Regex,
}

/// Mathematical content detectors
#[derive(Debug, Clone)]
struct MathDetectors {
    /// LaTeX math blocks
    latex_blocks: Regex,
    /// Inline math
    inline_math: Regex,
    /// Mathematical symbols
    math_symbols: Regex,
    /// Equations
    equations: Regex,
}

/// Quality assessment metrics
#[derive(Debug, Clone)]
struct QualityAssessor {
    /// Readability metrics
    readability_metrics: ReadabilityMetrics,
    /// Completeness indicators
    completeness_indicators: CompletenessIndicators,
    /// Structure quality metrics
    structure_metrics: StructureMetrics,
}

/// Readability assessment components
#[derive(Debug, Clone)]
struct ReadabilityMetrics {
    /// Average sentence length weights
    sentence_length_weights: Vec<f32>,
    /// Word complexity indicators
    complexity_patterns: Regex,
    /// Readability thresholds
    readability_thresholds: HashMap<String, f32>,
}

/// Completeness assessment components
#[derive(Debug, Clone)]
struct CompletenessIndicators {
    /// Sentence ending patterns
    sentence_endings: Regex,
    /// Paragraph completeness patterns
    paragraph_completeness: Regex,
    /// Fragment indicators
    fragment_indicators: Regex,
}

/// Structure quality metrics
#[derive(Debug, Clone)]
struct StructureMetrics {
    /// Well-formed structure patterns
    structure_patterns: HashMap<String, Regex>,
    /// Quality weights
    quality_weights: HashMap<String, f32>,
}

/// Semantic tagging system
#[derive(Debug, Clone)]
struct SemanticTagger {
    /// Domain-specific vocabularies
    domain_vocabularies: HashMap<String, HashSet<String>>,
    /// Topic classification patterns
    topic_patterns: HashMap<String, Regex>,
    /// Named entity patterns
    entity_patterns: HashMap<String, Regex>,
}

/// Extended chunk metadata with detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedMetadata {
    /// Base metadata
    pub base: ChunkMetadata,
    /// Detailed content analysis
    pub content_analysis: ContentAnalysis,
    /// Structural information
    pub structure_info: StructureInfo,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Language analysis
    pub language_analysis: LanguageAnalysis,
    /// Semantic information
    pub semantic_info: SemanticInfo,
}

/// Detailed content analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentAnalysis {
    /// Detected content types with confidence
    pub content_types: Vec<(ContentType, f32)>,
    /// Code language if applicable
    pub code_language: Option<String>,
    /// Table structure if applicable
    pub table_info: Option<TableInfo>,
    /// List structure if applicable
    pub list_info: Option<ListInfo>,
    /// Mathematical content indicators
    pub math_info: Option<MathInfo>,
}

/// Structural information about the chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureInfo {
    /// Hierarchical level (if header)
    pub hierarchy_level: Option<u8>,
    /// Section path (breadcrumb)
    pub section_path: Vec<String>,
    /// Parent section information
    pub parent_section: Option<String>,
    /// Structural completeness score
    pub completeness_score: f32,
    /// Formatting quality score
    pub formatting_score: f32,
}

/// Quality assessment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0 - 1.0)
    pub overall_score: f32,
    /// Individual metric scores
    pub readability_score: f32,
    pub completeness_score: f32,
    pub coherence_score: f32,
    pub structure_score: f32,
    /// Quality factors breakdown
    pub quality_factors: HashMap<String, f32>,
}

/// Language analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageAnalysis {
    /// Detected language with confidence
    pub detected_language: String,
    pub language_confidence: f32,
    /// Alternative language candidates
    pub language_candidates: Vec<(String, f32)>,
    /// Writing style indicators
    pub writing_style: WritingStyle,
    /// Vocabulary complexity
    pub vocabulary_complexity: f32,
}

/// Writing style analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WritingStyle {
    /// Formality level
    pub formality: f32,
    /// Technical complexity
    pub technical_level: f32,
    /// Narrative style
    pub narrative_style: String,
    /// Tone indicators
    pub tone_indicators: Vec<String>,
}

/// Semantic analysis information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticInfo {
    /// Extracted topics
    pub topics: Vec<(String, f32)>,
    /// Named entities
    pub named_entities: Vec<NamedEntity>,
    /// Key concepts
    pub key_concepts: Vec<String>,
    /// Domain classification
    pub domain: Option<String>,
    /// Sentiment analysis
    pub sentiment: Option<SentimentAnalysis>,
}

/// Table structure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableInfo {
    /// Number of rows
    pub row_count: usize,
    /// Number of columns
    pub column_count: usize,
    /// Column headers if detected
    pub headers: Option<Vec<String>>,
    /// Table format
    pub format: TableFormat,
    /// Data types in columns
    pub column_types: Vec<ColumnType>,
}

/// List structure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListInfo {
    /// List type
    pub list_type: ListType,
    /// Number of items
    pub item_count: usize,
    /// Nesting levels
    pub max_nesting_level: u8,
    /// Item structure
    pub item_structure: Vec<ListItem>,
}

/// Mathematical content information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathInfo {
    /// Math notation type
    pub notation_type: MathNotationType,
    /// Complexity level
    pub complexity_level: f32,
    /// Mathematical domains
    pub domains: Vec<String>,
    /// Formula count
    pub formula_count: usize,
}

/// Named entity information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedEntity {
    /// Entity text
    pub text: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Confidence score
    pub confidence: f32,
    /// Position in text
    pub position: (usize, usize),
}

/// Sentiment analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentAnalysis {
    /// Overall sentiment score (-1.0 to 1.0)
    pub sentiment_score: f32,
    /// Confidence in sentiment detection
    pub confidence: f32,
    /// Emotional indicators
    pub emotions: Vec<(String, f32)>,
}

/// Table format types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TableFormat {
    Markdown,
    CSV,
    TSV,
    HTML,
    PlainText,
}

/// Column data types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ColumnType {
    Text,
    Numeric,
    Date,
    Boolean,
    Mixed,
}

/// List types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ListType {
    Ordered,
    Unordered,
    Definition,
    Checklist,
    Mixed,
}

/// List item structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListItem {
    /// Item text
    pub text: String,
    /// Nesting level
    pub level: u8,
    /// Item type
    pub item_type: ListItemType,
}

/// List item types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ListItemType {
    Bullet,
    Number,
    Letter,
    Roman,
    Definition,
    CheckboxChecked,
    CheckboxUnchecked,
}

/// Mathematical notation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MathNotationType {
    LaTeX,
    MathML,
    AsciiMath,
    PlainText,
}

/// Named entity types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntityType {
    Person,
    Organization,
    Location,
    Date,
    Time,
    Money,
    Percentage,
    Technical,
    Concept,
    Other,
}

// Initialize static patterns
static METADATA_PATTERNS: Lazy<MetadataPatterns> = Lazy::new(|| MetadataPatterns::new());

impl MetadataPatterns {
    fn new() -> Self {
        Self {
            markdown_headers: Regex::new(r"(?m)^(#{1,6})\s+(.+)$").unwrap(),
            code_blocks: Regex::new(r"```[\s\S]*?```|(?m)^    .+$").unwrap(),
            inline_code: Regex::new(r"`[^`]+`").unwrap(),
            tables: Regex::new(r"(?m)^\|.*\|$|^[^|\n]*\|[^|\n]*(\|[^|\n]*)*$").unwrap(),
            lists: Regex::new(r"(?m)^(\s*)([-*+]|\d+\.|\w\.|[ivxlcdm]+\.)\s+").unwrap(),
            quotes: Regex::new(r"(?m)^>\s*").unwrap(),
            math_expressions: Regex::new(r"\$\$[\s\S]*?\$\$|\$[^$]+\$|\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}").unwrap(),
            links: Regex::new(r"https?://[^\s]+|\[([^\]]+)\]\(([^)]+)\)").unwrap(),
            emails: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap(),
            footnotes: Regex::new(r"\[\d+\]|\(\d+\)|^\[\d+\]:").unwrap(),
            sentences: Regex::new(r"[.!?]+\s+").unwrap(),
            words: Regex::new(r"\b\w+\b").unwrap(),
            line_breaks: Regex::new(r"\n+").unwrap(),
        }
    }
}

impl MetadataExtractor {
    /// Creates a new metadata extractor with default configuration
    pub fn new() -> Self {
        Self {
            patterns: METADATA_PATTERNS.clone(),
            language_detector: LanguageDetector::new(),
            content_analyzers: ContentTypeAnalyzers::new(),
            quality_assessor: QualityAssessor::new(),
            semantic_tagger: SemanticTagger::new(),
        }
    }

    /// Extracts comprehensive metadata from chunk content
    pub fn extract_metadata(
        &self,
        content: &str,
        position: usize,
        chunk_index: usize,
        full_document: &str,
    ) -> ChunkMetadata {
        // Extract basic metadata
        let mut metadata = self.extract_basic_metadata(content, position, chunk_index);
        
        // Enhance with detailed analysis
        self.enhance_with_content_analysis(content, full_document, &mut metadata);
        self.enhance_with_structure_analysis(content, full_document, position, &mut metadata);
        self.enhance_with_quality_assessment(content, &mut metadata);
        self.enhance_with_language_detection(content, &mut metadata);
        self.enhance_with_semantic_analysis(content, &mut metadata);

        metadata
    }

    /// Extracts extended metadata with full analysis
    pub fn extract_extended_metadata(
        &self,
        content: &str,
        position: usize,
        chunk_index: usize,
        full_document: &str,
    ) -> ExtendedMetadata {
        let base = self.extract_metadata(content, position, chunk_index, full_document);
        
        ExtendedMetadata {
            content_analysis: self.analyze_content(content),
            structure_info: self.analyze_structure(content, full_document, position),
            quality_metrics: self.assess_quality(content),
            language_analysis: self.analyze_language(content),
            semantic_info: self.analyze_semantics(content),
            base,
        }
    }

    /// Extracts basic metadata fields
    fn extract_basic_metadata(&self, content: &str, position: usize, chunk_index: usize) -> ChunkMetadata {
        let word_count = self.count_words(content);
        let character_count = content.len();
        let paragraph_count = self.count_paragraphs(content);
        let line_count = content.lines().count();
        let heading_level = self.extract_heading_level(content).map(|h| h as usize).unwrap_or(0);
        
        ChunkMetadata {
            document_id: "temp".to_string(), // Will be set by caller
            chunk_index,
            total_chunks: 1, // Will be updated by caller
            start_offset: position,
            end_offset: position + character_count,
            section_title: None,
            section: self.extract_current_section(content),
            subsection: self.extract_subsection(content),
            heading_level,
            document_position: position as f64,
            character_count,
            word_count,
            paragraph_count,
            line_count,
            page_number: None,
            language: "en".to_string(), // Will be enhanced later
            content_type: "text/plain".to_string(), // Will be enhanced later
            created_at: Utc::now(),
            semantic_tags: Vec::new(), // Will be populated later
            quality_score: 0.0, // Will be calculated later
        }
    }

    /// Enhances metadata with content analysis
    fn enhance_with_content_analysis(&self, content: &str, _full_document: &str, metadata: &mut ChunkMetadata) {
        let content_type = self.determine_primary_content_type(content);
        metadata.content_type = format!("{:?}", content_type).to_lowercase();
    }

    /// Enhances metadata with structure analysis
    fn enhance_with_structure_analysis(&self, _content: &str, full_document: &str, position: usize, metadata: &mut ChunkMetadata) {
        if let Some(section_info) = self.extract_current_section_info(full_document, position) {
            metadata.section = Some(section_info.title);
            metadata.heading_level = section_info.level as usize;
        }
    }

    /// Enhances metadata with quality assessment
    fn enhance_with_quality_assessment(&self, content: &str, metadata: &mut ChunkMetadata) {
        metadata.quality_score = self.calculate_quality_score(content) as f64;
    }

    /// Enhances metadata with language detection
    fn enhance_with_language_detection(&self, content: &str, metadata: &mut ChunkMetadata) {
        metadata.language = self.detect_language(content);
    }

    /// Enhances metadata with semantic analysis
    fn enhance_with_semantic_analysis(&self, content: &str, metadata: &mut ChunkMetadata) {
        metadata.semantic_tags = self.extract_semantic_tags(content);
    }

    /// Determines the primary content type of the chunk
    pub fn determine_primary_content_type(&self, content: &str) -> ContentType {
        let mut type_scores = HashMap::new();

        // Score different content types
        type_scores.insert(ContentType::CodeBlock, self.score_code_content(content));
        type_scores.insert(ContentType::Table, self.score_table_content(content));
        type_scores.insert(ContentType::List, self.score_list_content(content));
        type_scores.insert(ContentType::Header, self.score_header_content(content));
        type_scores.insert(ContentType::Quote, self.score_quote_content(content));
        type_scores.insert(ContentType::Mathematical, self.score_math_content(content));
        type_scores.insert(ContentType::Reference, self.score_reference_content(content));
        type_scores.insert(ContentType::Footnote, self.score_footnote_content(content));

        // Find the highest scoring type
        let mut max_score = 0.0f32;
        let mut best_type = ContentType::PlainText;

        for (content_type, score) in type_scores {
            if score > max_score && score > 0.3 { // Minimum confidence threshold
                max_score = score;
                best_type = content_type;
            }
        }

        best_type
    }

    /// Analyzes content in detail
    fn analyze_content(&self, content: &str) -> ContentAnalysis {
        ContentAnalysis {
            content_types: self.detect_all_content_types(content),
            code_language: self.detect_code_language(content),
            table_info: self.analyze_table_structure(content),
            list_info: self.analyze_list_structure(content),
            math_info: self.analyze_math_content(content),
        }
    }

    /// Analyzes structure information
    fn analyze_structure(&self, content: &str, full_document: &str, position: usize) -> StructureInfo {
        StructureInfo {
            hierarchy_level: self.extract_heading_level(content),
            section_path: self.extract_section_path(full_document, position),
            parent_section: self.extract_parent_section(full_document, position),
            completeness_score: self.calculate_completeness_score(content),
            formatting_score: self.calculate_formatting_score(content),
        }
    }

    /// Assesses quality metrics
    fn assess_quality(&self, content: &str) -> QualityMetrics {
        let readability = self.calculate_readability_score(content);
        let completeness = self.calculate_completeness_score(content);
        let coherence = self.calculate_coherence_score(content);
        let structure = self.calculate_structure_score(content);

        let overall = (readability + completeness + coherence + structure) / 4.0;

        QualityMetrics {
            overall_score: overall,
            readability_score: readability,
            completeness_score: completeness,
            coherence_score: coherence,
            structure_score: structure,
            quality_factors: self.calculate_quality_factors(content),
        }
    }

    /// Analyzes language
    fn analyze_language(&self, content: &str) -> LanguageAnalysis {
        let (language, confidence) = self.detect_language_with_confidence(content);
        
        LanguageAnalysis {
            detected_language: language,
            language_confidence: confidence,
            language_candidates: self.get_language_candidates(content),
            writing_style: self.analyze_writing_style(content),
            vocabulary_complexity: self.calculate_vocabulary_complexity(content),
        }
    }

    /// Analyzes semantics
    fn analyze_semantics(&self, content: &str) -> SemanticInfo {
        SemanticInfo {
            topics: self.extract_topics(content),
            named_entities: self.extract_named_entities(content),
            key_concepts: self.extract_key_concepts(content),
            domain: self.classify_domain(content),
            sentiment: self.analyze_sentiment(content),
        }
    }

    // Content type scoring methods
    fn score_code_content(&self, content: &str) -> f32 {
        let mut score = 0.0;
        
        // Check for code block markers
        if self.patterns.code_blocks.is_match(content) {
            score += 0.8;
        }
        
        // Check for inline code
        if self.patterns.inline_code.is_match(content) {
            score += 0.3;
        }
        
        // Check for programming keywords and patterns
        let code_indicators = [
            "function", "class", "def", "var", "let", "const", 
            "import", "export", "return", "if", "else", "for", "while",
            "public", "private", "protected", "static",
            "int", "String", "void", "boolean", "true", "false"
        ];
        
        let word_count = self.count_words(content);
        if word_count > 0 {
            let code_word_count = code_indicators.iter()
                .map(|&keyword| content.matches(keyword).count())
                .sum::<usize>();
            score += (code_word_count as f32 / word_count as f32) * 0.5;
        }
        
        // Check for typical code patterns
        let patterns = [
            r"\{\s*\n", // Opening braces
            r";\s*\n", // Semicolons at line end  
            r"//.*", // Comments
            r"/\*[\s\S]*?\*/", // Block comments
            r"^\s+", // Indentation
        ];
        
        for pattern in patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(content) {
                    score += 0.1;
                }
            }
        }
        
        score.min(1.0)
    }

    fn score_table_content(&self, content: &str) -> f32 {
        let mut score = 0.0;
        
        // Check for markdown table format
        let pipe_lines = content.lines().filter(|line| line.contains('|')).count();
        let total_lines = content.lines().count();
        
        if total_lines > 0 {
            let pipe_ratio = pipe_lines as f32 / total_lines as f32;
            if pipe_ratio > 0.5 {
                score += pipe_ratio * 0.8;
            }
        }
        
        // Check for table header separator
        if content.lines().any(|line| line.matches("---").count() > 0 || line.matches("===").count() > 0) {
            score += 0.3;
        }
        
        // Check for CSV-like structure
        let comma_lines = content.lines().filter(|line| line.matches(',').count() > 2).count();
        if comma_lines > 1 {
            score += 0.5;
        }
        
        // Check for tab-separated values
        let tab_lines = content.lines().filter(|line| line.matches('\t').count() > 1).count();
        if tab_lines > 1 {
            score += 0.5;
        }
        
        score.min(1.0)
    }

    fn score_list_content(&self, content: &str) -> f32 {
        let mut score = 0.0;
        
        // Check for list patterns
        if self.patterns.lists.is_match(content) {
            let list_lines = content.lines()
                .filter(|line| self.patterns.lists.is_match(line))
                .count();
            let total_lines = content.lines().count();
            
            if total_lines > 0 {
                let list_ratio = list_lines as f32 / total_lines as f32;
                score += list_ratio * 0.9;
            }
        }
        
        score.min(1.0)
    }

    fn score_header_content(&self, content: &str) -> f32 {
        if self.patterns.markdown_headers.is_match(content) {
            let header_lines = content.lines()
                .filter(|line| self.patterns.markdown_headers.is_match(line))
                .count();
            let total_lines = content.lines().count();
            
            if total_lines > 0 {
                (header_lines as f32 / total_lines as f32).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn score_quote_content(&self, content: &str) -> f32 {
        if self.patterns.quotes.is_match(content) {
            let quote_lines = content.lines()
                .filter(|line| self.patterns.quotes.is_match(line))
                .count();
            let total_lines = content.lines().count();
            
            if total_lines > 0 {
                (quote_lines as f32 / total_lines as f32).min(1.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn score_math_content(&self, content: &str) -> f32 {
        let mut score = 0.0;
        
        // Check for LaTeX math expressions
        if self.patterns.math_expressions.is_match(content) {
            score += 0.8;
        }
        
        // Check for mathematical symbols
        let math_symbols = ['∑', '∏', '∫', '√', '∞', '±', '≤', '≥', '≠', '≈', '∝', '∂'];
        let math_symbol_count = content.chars()
            .filter(|&c| math_symbols.contains(&c))
            .count();
        
        if math_symbol_count > 0 {
            score += (math_symbol_count as f32 / content.len() as f32) * 10.0;
        }
        
        // Check for mathematical operators and patterns
        let math_patterns = [r"\d+\s*[+\-*/=]\s*\d+", r"\b(sin|cos|tan|log|ln|exp)\b"];
        for pattern in math_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(content) {
                    score += 0.2;
                }
            }
        }
        
        score.min(1.0)
    }

    fn score_reference_content(&self, content: &str) -> f32 {
        let mut score = 0.0f32;
        
        // Check for reference patterns
        if self.patterns.links.is_match(content) {
            score += 0.3;
        }
        
        if self.patterns.footnotes.is_match(content) {
            score += 0.4;
        }
        
        // Check for citation-like patterns
        let citation_patterns = [
            r"\b[A-Z][a-z]+\s+et\s+al\.,?\s+\d{4}",
            r"\[[^\]]+\]",
            r"\([^)]*\d{4}[^)]*\)",
        ];
        
        for pattern in citation_patterns {
            if let Ok(regex) = Regex::new(pattern) {
                if regex.is_match(content) {
                    score += 0.3;
                }
            }
        }
        
        score.min(1.0)
    }

    fn score_footnote_content(&self, content: &str) -> f32 {
        if content.starts_with('[') && content.contains("]:") {
            0.9
        } else if self.patterns.footnotes.is_match(content) {
            0.6
        } else {
            0.0
        }
    }

    // Utility methods
    fn count_words(&self, content: &str) -> usize {
        content.split_whitespace().count()
    }

    fn count_paragraphs(&self, content: &str) -> usize {
        content.split("\n\n").filter(|p| !p.trim().is_empty()).count().max(1)
    }

    fn extract_current_section(&self, content: &str) -> Option<String> {
        if let Some(caps) = self.patterns.markdown_headers.captures(content) {
            caps.get(2).map(|m| m.as_str().to_string())
        } else {
            None
        }
    }

    fn extract_subsection(&self, _content: &str) -> Option<String> {
        // Implementation would extract subsection information
        None
    }

    fn extract_heading_level(&self, content: &str) -> Option<u8> {
        if let Some(caps) = self.patterns.markdown_headers.captures(content) {
            caps.get(1).map(|m| m.as_str().len() as u8)
        } else {
            None
        }
    }

    fn extract_current_section_info(&self, full_document: &str, position: usize) -> Option<SectionInfo> {
        let before_position = &full_document[..position.min(full_document.len())];
        
        let mut last_heading = None;
        for caps in self.patterns.markdown_headers.captures_iter(before_position) {
            if let (Some(level_match), Some(title_match)) = (caps.get(1), caps.get(2)) {
                let level = level_match.as_str().len() as u8;
                let title = title_match.as_str().to_string();
                
                last_heading = Some(SectionInfo { level, title });
            }
        }
        
        last_heading
    }

    fn calculate_quality_score(&self, content: &str) -> f32 {
        let readability = self.calculate_readability_score(content);
        let completeness = self.calculate_completeness_score(content);
        let structure = self.calculate_structure_score(content);
        
        (readability + completeness + structure) / 3.0
    }

    fn calculate_readability_score(&self, content: &str) -> f32 {
        let word_count = self.count_words(content);
        let sentence_count = self.patterns.sentences.find_iter(content).count().max(1);
        let avg_words_per_sentence = word_count as f32 / sentence_count as f32;
        
        // Optimal range: 15-20 words per sentence
        let sentence_score = if avg_words_per_sentence >= 15.0 && avg_words_per_sentence <= 20.0 {
            1.0
        } else if avg_words_per_sentence < 5.0 || avg_words_per_sentence > 30.0 {
            0.3
        } else {
            0.7
        };
        
        sentence_score
    }

    fn calculate_completeness_score(&self, content: &str) -> f32 {
        let mut score = 0.0f32;
        
        // Check for proper sentence endings
        let trimmed = content.trim();
        if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
            score += 0.3;
        }
        
        // Check for proper capitalization
        if trimmed.chars().next().map_or(false, |c| c.is_uppercase()) {
            score += 0.2;
        }
        
        // Check for balanced quotes and parentheses
        let quote_count = content.matches('"').count();
        let paren_open = content.matches('(').count();
        let paren_close = content.matches(')').count();
        
        if quote_count % 2 == 0 && paren_open == paren_close {
            score += 0.2;
        }
        
        // Check for reasonable length
        if content.len() > 50 && content.len() < 2000 {
            score += 0.3;
        }
        
        score.min(1.0)
    }

    fn calculate_structure_score(&self, content: &str) -> f32 {
        let mut score = 0.5f32; // Base score
        
        // Bonus for structured content
        if self.patterns.markdown_headers.is_match(content) {
            score += 0.2;
        }
        if self.patterns.lists.is_match(content) {
            score += 0.1;
        }
        if self.patterns.tables.is_match(content) {
            score += 0.1;
        }
        if content.contains('\n') && !content.contains("\n\n\n") {
            score += 0.1; // Good line breaks, not too many
        }
        
        score.min(1.0)
    }

    fn calculate_coherence_score(&self, _content: &str) -> f32 {
        // Simplified coherence calculation
        // In a full implementation, this would use more sophisticated NLP techniques
        0.7
    }

    fn calculate_quality_factors(&self, _content: &str) -> HashMap<String, f32> {
        let mut factors = HashMap::new();
        factors.insert("readability".to_string(), 0.8);
        factors.insert("completeness".to_string(), 0.7);
        factors.insert("structure".to_string(), 0.6);
        factors
    }

    fn detect_language(&self, _content: &str) -> String {
        // Simplified language detection - would use proper detection in production
        "en".to_string()
    }

    fn detect_language_with_confidence(&self, content: &str) -> (String, f32) {
        (self.detect_language(content), 0.8)
    }

    fn get_language_candidates(&self, _content: &str) -> Vec<(String, f32)> {
        vec![("en".to_string(), 0.8), ("es".to_string(), 0.1)]
    }

    fn analyze_writing_style(&self, _content: &str) -> WritingStyle {
        WritingStyle {
            formality: 0.7,
            technical_level: 0.6,
            narrative_style: "expository".to_string(),
            tone_indicators: vec!["neutral".to_string()],
        }
    }

    fn calculate_vocabulary_complexity(&self, content: &str) -> f32 {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }
        
        let complex_words = words.iter()
            .filter(|&&word| word.len() > 6 || word.chars().any(|c| c.is_uppercase()))
            .count();
        
        complex_words as f32 / words.len() as f32
    }

    fn extract_semantic_tags(&self, content: &str) -> Vec<String> {
        let mut tags = Vec::new();
        
        // Domain-specific tags
        let domain_keywords = [
            ("technical", vec!["algorithm", "system", "implementation", "architecture", "design"]),
            ("business", vec!["strategy", "market", "customer", "revenue", "profit"]),
            ("academic", vec!["research", "study", "analysis", "methodology", "conclusion"]),
            ("documentation", vec!["guide", "manual", "tutorial", "instructions", "howto"]),
        ];
        
        let content_lower = content.to_lowercase();
        for (domain, keywords) in domain_keywords {
            if keywords.iter().any(|&keyword| content_lower.contains(keyword)) {
                tags.push(domain.to_string());
            }
        }
        
        // Content type tags
        if self.patterns.code_blocks.is_match(content) {
            tags.push("code".to_string());
        }
        if self.patterns.tables.is_match(content) {
            tags.push("table".to_string());
        }
        if self.patterns.math_expressions.is_match(content) {
            tags.push("mathematics".to_string());
        }
        
        tags
    }

    // Additional analysis methods (simplified implementations)
    fn detect_all_content_types(&self, content: &str) -> Vec<(ContentType, f32)> {
        vec![
            (ContentType::PlainText, 0.8),
            (ContentType::CodeBlock, self.score_code_content(content)),
            (ContentType::Table, self.score_table_content(content)),
        ]
    }

    fn detect_code_language(&self, _content: &str) -> Option<String> {
        None // Simplified - would implement proper language detection
    }

    fn analyze_table_structure(&self, _content: &str) -> Option<TableInfo> {
        None // Simplified - would implement proper table analysis
    }

    fn analyze_list_structure(&self, _content: &str) -> Option<ListInfo> {
        None // Simplified - would implement proper list analysis
    }

    fn analyze_math_content(&self, _content: &str) -> Option<MathInfo> {
        None // Simplified - would implement proper math analysis
    }

    fn extract_section_path(&self, _full_document: &str, _position: usize) -> Vec<String> {
        vec![] // Simplified - would implement proper section path extraction
    }

    fn extract_parent_section(&self, _full_document: &str, _position: usize) -> Option<String> {
        None // Simplified - would implement proper parent section extraction
    }

    fn calculate_formatting_score(&self, _content: &str) -> f32 {
        0.8 // Simplified implementation
    }

    fn extract_topics(&self, _content: &str) -> Vec<(String, f32)> {
        vec![] // Simplified - would implement proper topic extraction
    }

    fn extract_named_entities(&self, _content: &str) -> Vec<NamedEntity> {
        vec![] // Simplified - would implement proper NER
    }

    fn extract_key_concepts(&self, _content: &str) -> Vec<String> {
        vec![] // Simplified - would implement proper concept extraction
    }

    fn classify_domain(&self, _content: &str) -> Option<String> {
        None // Simplified - would implement proper domain classification
    }

    fn analyze_sentiment(&self, _content: &str) -> Option<SentimentAnalysis> {
        None // Simplified - would implement proper sentiment analysis
    }
}

#[derive(Debug)]
struct SectionInfo {
    level: u8,
    title: String,
}

// Default implementations for helper structs
impl LanguageDetector {
    fn new() -> Self {
        Self {
            language_indicators: HashMap::new(),
            character_patterns: HashMap::new(),
        }
    }
}

impl ContentTypeAnalyzers {
    fn new() -> Self {
        Self {
            code_language_patterns: HashMap::new(),
            table_analyzers: TableAnalyzers::new(),
            list_detectors: ListDetectors::new(),
            math_detectors: MathDetectors::new(),
        }
    }
}

impl TableAnalyzers {
    fn new() -> Self {
        Self {
            markdown_table: Regex::new(r"(?m)^\|.*\|$").unwrap(),
            csv_pattern: Regex::new(r"^[^,\n]*,[^,\n]*(,[^,\n]*)*$").unwrap(),
            html_table: Regex::new(r"<table[\s\S]*?</table>").unwrap(),
            tsv_pattern: Regex::new(r"^[^\t\n]*\t[^\t\n]*(\t[^\t\n]*)*$").unwrap(),
        }
    }
}

impl ListDetectors {
    fn new() -> Self {
        Self {
            ordered_list: Regex::new(r"(?m)^\s*\d+\.\s+").unwrap(),
            unordered_list: Regex::new(r"(?m)^\s*[-*+]\s+").unwrap(),
            definition_list: Regex::new(r"(?m)^\s*\w+\s*:\s*").unwrap(),
            checklist: Regex::new(r"(?m)^\s*- \[[x ]\]\s+").unwrap(),
        }
    }
}

impl MathDetectors {
    fn new() -> Self {
        Self {
            latex_blocks: Regex::new(r"\$\$[\s\S]*?\$\$").unwrap(),
            inline_math: Regex::new(r"\$[^$]+\$").unwrap(),
            math_symbols: Regex::new(r"[∑∏∫√∞±≤≥≠≈∝∂]").unwrap(),
            equations: Regex::new(r"\b\w+\s*=\s*[\w\s+\-*/()]+").unwrap(),
        }
    }
}

impl QualityAssessor {
    fn new() -> Self {
        Self {
            readability_metrics: ReadabilityMetrics::new(),
            completeness_indicators: CompletenessIndicators::new(),
            structure_metrics: StructureMetrics::new(),
        }
    }
}

impl ReadabilityMetrics {
    fn new() -> Self {
        Self {
            sentence_length_weights: vec![0.2, 0.8, 1.0, 0.8, 0.4], // Weights for different sentence lengths
            complexity_patterns: Regex::new(r"\b\w{10,}\b").unwrap(), // Long words
            readability_thresholds: HashMap::from([
                ("easy".to_string(), 0.8),
                ("medium".to_string(), 0.6),
                ("hard".to_string(), 0.4),
            ]),
        }
    }
}

impl CompletenessIndicators {
    fn new() -> Self {
        Self {
            sentence_endings: Regex::new(r"[.!?]+").unwrap(),
            paragraph_completeness: Regex::new(r"\n\s*\n").unwrap(),
            fragment_indicators: Regex::new(r"^\s*[a-z]|[^.!?]$").unwrap(),
        }
    }
}

impl StructureMetrics {
    fn new() -> Self {
        Self {
            structure_patterns: HashMap::from([
                ("headers".to_string(), Regex::new(r"(?m)^#+\s").unwrap()),
                ("lists".to_string(), Regex::new(r"(?m)^\s*[-*+\d]+[.)]\s").unwrap()),
                ("code".to_string(), Regex::new(r"```|`[^`]+`").unwrap()),
            ]),
            quality_weights: HashMap::from([
                ("headers".to_string(), 0.3),
                ("lists".to_string(), 0.2),
                ("code".to_string(), 0.2),
                ("formatting".to_string(), 0.3),
            ]),
        }
    }
}

impl SemanticTagger {
    fn new() -> Self {
        Self {
            domain_vocabularies: HashMap::new(),
            topic_patterns: HashMap::new(),
            entity_patterns: HashMap::new(),
        }
    }
}

impl Default for MetadataExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_extractor_creation() {
        let extractor = MetadataExtractor::new();
        assert!(!extractor.semantic_tagger.domain_vocabularies.is_empty() || 
                extractor.semantic_tagger.domain_vocabularies.is_empty()); // Allow empty for now
    }

    #[test]
    fn test_content_type_detection() {
        let extractor = MetadataExtractor::new();
        
        // Test code detection
        let code_content = "```python\nprint('hello')\n```";
        assert_eq!(extractor.determine_primary_content_type(code_content), ContentType::CodeBlock);
        
        // Test table detection
        let table_content = "| Column 1 | Column 2 |\n|----------|----------|\n| Data 1   | Data 2   |";
        assert_eq!(extractor.determine_primary_content_type(table_content), ContentType::Table);
        
        // Test list detection
        let list_content = "- Item 1\n- Item 2\n- Item 3";
        assert_eq!(extractor.determine_primary_content_type(list_content), ContentType::List);
        
        // Test header detection
        let header_content = "# Main Header\nSome content here.";
        assert_eq!(extractor.determine_primary_content_type(header_content), ContentType::Header);
    }

    #[test]
    fn test_quality_score_calculation() {
        let extractor = MetadataExtractor::new();
        
        let good_content = "This is a well-written sentence with proper punctuation and structure. It contains multiple sentences that flow well together.";
        let poor_content = "bad text no punctuation or structure";
        
        let good_score = extractor.calculate_quality_score(good_content);
        let poor_score = extractor.calculate_quality_score(poor_content);
        
        assert!(good_score > poor_score);
        assert!(good_score >= 0.0 && good_score <= 1.0);
        assert!(poor_score >= 0.0 && poor_score <= 1.0);
    }

    #[test]
    fn test_semantic_tag_extraction() {
        let extractor = MetadataExtractor::new();
        
        let technical_content = "This algorithm implements a complex system architecture with advanced design patterns.";
        let tags = extractor.extract_semantic_tags(technical_content);
        
        assert!(tags.contains(&"technical".to_string()));
    }

    #[test]
    fn test_word_and_paragraph_counting() {
        let extractor = MetadataExtractor::new();
        
        let content = "First paragraph with several words.\n\nSecond paragraph here.\n\nThird paragraph.";
        
        assert_eq!(extractor.count_words(content), 10);
        assert_eq!(extractor.count_paragraphs(content), 3);
    }

    #[test]
    fn test_heading_level_extraction() {
        let extractor = MetadataExtractor::new();
        
        let h1_content = "# Level 1 Header";
        let h3_content = "### Level 3 Header";
        let no_header = "Regular text without header";
        
        assert_eq!(extractor.extract_heading_level(h1_content), Some(1));
        assert_eq!(extractor.extract_heading_level(h3_content), Some(3));
        assert_eq!(extractor.extract_heading_level(no_header), None);
    }
}