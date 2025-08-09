//! Cross-reference detection and preservation for document chunks
//! 
//! This module provides sophisticated cross-reference detection, tracking, and preservation
//! capabilities for maintaining document coherence and enabling intelligent chunk linking.

use crate::{ChunkReference, ReferenceType, ChunkerError};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use once_cell::sync::Lazy;

/// Comprehensive reference tracker for document chunks
#[derive(Debug, Clone)]
pub struct ReferenceTracker {
    /// Reference detection patterns
    patterns: ReferencePatterns,
    /// Reference resolution engine
    resolver: ReferenceResolver,
    /// Cross-document reference manager
    cross_doc_manager: CrossDocumentManager,
    /// Reference quality assessor
    quality_assessor: ReferenceQualityAssessor,
    /// Reference graph builder
    graph_builder: ReferenceGraphBuilder,
    /// Configuration settings
    config: ReferenceConfig,
}

/// Configuration for reference tracking
#[derive(Debug, Clone)]
pub struct ReferenceConfig {
    /// Minimum confidence threshold for reference detection
    pub min_confidence: f32,
    /// Maximum distance for local references
    pub max_local_distance: usize,
    /// Enable cross-document reference tracking
    pub enable_cross_document: bool,
    /// Context window size for reference resolution
    pub context_window: usize,
    /// Maximum references per chunk
    pub max_references_per_chunk: usize,
    /// Reference validation strictness
    pub validation_strictness: ValidationStrictness,
}

/// Reference validation strictness levels
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStrictness {
    Lenient,
    Moderate,
    Strict,
    VeryStrict,
}

/// Pre-compiled reference detection patterns
#[derive(Debug, Clone)]
struct ReferencePatterns {
    /// Cross-reference patterns
    cross_references: HashMap<String, Regex>,
    /// Citation patterns
    citations: HashMap<String, Regex>,
    /// Footnote patterns
    footnotes: HashMap<String, Regex>,
    /// Table and figure reference patterns
    table_figure_refs: HashMap<String, Regex>,
    /// Section reference patterns
    section_refs: HashMap<String, Regex>,
    /// External link patterns
    external_links: HashMap<String, Regex>,
    /// Internal link patterns
    internal_links: HashMap<String, Regex>,
    /// Bibliography patterns
    bibliography: HashMap<String, Regex>,
}

/// Reference resolution engine
#[derive(Debug, Clone)]
struct ReferenceResolver {
    /// Known document structure
    document_structure: DocumentStructure,
    /// Reference cache for quick lookup
    reference_cache: HashMap<String, ResolvedReference>,
    /// Forward reference tracker
    forward_references: HashMap<String, Vec<UnresolvedReference>>,
    /// Backward reference tracker  
    backward_references: HashMap<String, Vec<String>>,
}

/// Document structure information
#[derive(Debug, Clone)]
struct DocumentStructure {
    /// Section hierarchy
    sections: Vec<SectionInfo>,
    /// Table of contents
    toc: Vec<TocEntry>,
    /// Figure/table registry
    figures: HashMap<String, FigureInfo>,
    tables: HashMap<String, TableInfo>,
    /// Footnote registry
    footnotes: HashMap<String, FootnoteInfo>,
    /// Bibliography entries
    bibliography: HashMap<String, BibliographyEntry>,
}

/// Cross-document reference management
#[derive(Debug, Clone)]
struct CrossDocumentManager {
    /// Inter-document reference map
    document_refs: HashMap<String, Vec<ExternalReference>>,
    /// Document metadata
    document_metadata: HashMap<String, DocumentMetadata>,
    /// Reference validation cache
    validation_cache: HashMap<String, ValidationResult>,
}

/// Reference quality assessment
#[derive(Debug, Clone)]
struct ReferenceQualityAssessor {
    /// Quality metrics
    quality_metrics: QualityMetrics,
    /// Confidence calculators
    confidence_calculators: ConfidenceCalculators,
    /// Context analyzers
    context_analyzers: ContextAnalyzers,
}

/// Reference graph building
#[derive(Debug, Clone)]
struct ReferenceGraphBuilder {
    /// Reference graph
    graph: ReferenceGraph,
    /// Graph analysis tools
    analyzer: GraphAnalyzer,
    /// Path finding utilities
    path_finder: PathFinder,
}

/// Extended reference information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedReference {
    /// Base reference information
    pub base: ChunkReference,
    /// Reference metadata
    pub metadata: ReferenceMetadata,
    /// Resolution information
    pub resolution: Option<ReferenceResolution>,
    /// Quality assessment
    pub quality: ReferenceQuality,
    /// Graph position
    pub graph_info: GraphInfo,
}

/// Reference metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceMetadata {
    /// Reference source location
    pub source_location: LocationInfo,
    /// Target location (if resolved)
    pub target_location: Option<LocationInfo>,
    /// Reference category
    pub category: ReferenceCategory,
    /// Semantic context
    pub semantic_context: SemanticContext,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last validation timestamp
    pub last_validated: Option<chrono::DateTime<chrono::Utc>>,
}

/// Location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationInfo {
    /// Document identifier
    pub document_id: String,
    /// Chunk identifier
    pub chunk_id: Option<Uuid>,
    /// Character position
    pub char_position: usize,
    /// Line number
    pub line_number: usize,
    /// Column number
    pub column_number: usize,
    /// Context excerpt
    pub context: String,
}

/// Reference categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReferenceCategory {
    /// Structural references (sections, headings)
    Structural,
    /// Content references (figures, tables, equations)
    Content,
    /// Bibliographic references
    Bibliographic,
    /// Hyperlink references
    Hyperlink,
    /// Footnote references
    Footnote,
    /// Cross-document references
    CrossDocument,
    /// Contextual references (see also, related)
    Contextual,
}

/// Semantic context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContext {
    /// Context type
    pub context_type: ContextType,
    /// Surrounding topics
    pub topics: Vec<String>,
    /// Intent classification
    pub intent: ReferenceIntent,
    /// Importance score
    pub importance: f32,
}

/// Context types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ContextType {
    Definition,
    Example,
    Elaboration,
    Comparison,
    Citation,
    Navigation,
    Support,
}

/// Reference intents
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReferenceIntent {
    Informational,
    Navigational,
    Definitional,
    Comparative,
    Supportive,
    Clarifying,
}

/// Reference resolution information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceResolution {
    /// Resolution status
    pub status: ResolutionStatus,
    /// Resolved target
    pub target: Option<ResolvedTarget>,
    /// Resolution confidence
    pub confidence: f32,
    /// Resolution method used
    pub method: ResolutionMethod,
    /// Alternative candidates
    pub alternatives: Vec<ResolvedTarget>,
}

/// Resolution status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResolutionStatus {
    Resolved,
    PartiallyResolved,
    Unresolved,
    Ambiguous,
    Invalid,
    External,
}

/// Resolved target information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTarget {
    /// Target identifier
    pub target_id: String,
    /// Target type
    pub target_type: TargetType,
    /// Target location
    pub location: LocationInfo,
    /// Target title/description
    pub title: Option<String>,
    /// Match confidence
    pub match_confidence: f32,
}

/// Target types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TargetType {
    Section,
    Subsection,
    Paragraph,
    Figure,
    Table,
    Equation,
    Footnote,
    Bibliography,
    ExternalUrl,
    ExternalDocument,
}

/// Resolution methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResolutionMethod {
    ExactMatch,
    FuzzyMatch,
    SemanticMatch,
    StructuralInference,
    ContextualInference,
    Manual,
}

/// Reference quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceQuality {
    /// Overall quality score
    pub overall_score: f32,
    /// Individual quality metrics
    pub clarity_score: f32,
    pub relevance_score: f32,
    pub accuracy_score: f32,
    pub completeness_score: f32,
    /// Quality issues
    pub issues: Vec<QualityIssue>,
    /// Quality recommendations
    pub recommendations: Vec<String>,
}

/// Quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: IssueType,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Issue types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueType {
    BrokenLink,
    AmbiguousReference,
    MissingContext,
    InconsistentFormat,
    CircularReference,
    DeadReference,
    LowConfidence,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Graph position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphInfo {
    /// Node identifier in reference graph
    pub node_id: String,
    /// Incoming reference count
    pub in_degree: usize,
    /// Outgoing reference count
    pub out_degree: usize,
    /// Graph centrality measures
    pub centrality: CentralityMeasures,
    /// Connected components
    pub component_id: String,
}

/// Centrality measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityMeasures {
    /// Degree centrality
    pub degree: f32,
    /// Betweenness centrality
    pub betweenness: f32,
    /// Closeness centrality
    pub closeness: f32,
    /// PageRank score
    pub pagerank: f32,
}

// Static reference patterns
static REFERENCE_PATTERNS: Lazy<ReferencePatterns> = Lazy::new(|| ReferencePatterns::new());

impl Default for ReferenceConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            max_local_distance: 10000, // characters
            enable_cross_document: true,
            context_window: 200, // characters
            max_references_per_chunk: 20,
            validation_strictness: ValidationStrictness::Moderate,
        }
    }
}

impl ReferencePatterns {
    fn new() -> Self {
        let mut cross_references = HashMap::new();
        cross_references.insert("section".to_string(), 
            Regex::new(r"(?i)(?:see\s+)?(?:section|chapter|part)\s+(\d+(?:\.\d+)*|\w+)").unwrap());
        cross_references.insert("page".to_string(),
            Regex::new(r"(?i)(?:see\s+)?page\s+(\d+)").unwrap());
        cross_references.insert("appendix".to_string(),
            Regex::new(r"(?i)(?:see\s+)?appendix\s+([A-Z]\d*)").unwrap());

        let mut citations = HashMap::new();
        citations.insert("numbered".to_string(),
            Regex::new(r"\[(\d+(?:,\s*\d+)*(?:-\d+)?)\]").unwrap());
        citations.insert("author_year".to_string(),
            Regex::new(r"\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s+(\d{4})\)").unwrap());
        citations.insert("inline".to_string(),
            Regex::new(r"([A-Z][a-z]+(?:\s+et\s+al\.)?)\s+\((\d{4})\)").unwrap());

        let mut footnotes = HashMap::new();
        footnotes.insert("superscript".to_string(),
            Regex::new(r"\^(\d+)|\[(\d+)\]").unwrap());
        footnotes.insert("definition".to_string(),
            Regex::new(r"(?m)^\s*(\d+)\.\s+(.+)$").unwrap());

        let mut table_figure_refs = HashMap::new();
        table_figure_refs.insert("table".to_string(),
            Regex::new(r"(?i)(?:see\s+)?table\s+(\d+(?:\.\d+)*)").unwrap());
        table_figure_refs.insert("figure".to_string(),
            Regex::new(r"(?i)(?:see\s+)?(?:figure|fig\.?)\s+(\d+(?:\.\d+)*)").unwrap());
        table_figure_refs.insert("equation".to_string(),
            Regex::new(r"(?i)(?:see\s+)?(?:equation|eq\.?)\s+(\d+(?:\.\d+)*)").unwrap());

        let mut section_refs = HashMap::new();
        section_refs.insert("heading".to_string(),
            Regex::new(r"(?m)^#+\s+(.+)$").unwrap());
        section_refs.insert("numbered_section".to_string(),
            Regex::new(r"(?i)(\d+(?:\.\d+)*)\s+(.+)").unwrap());

        let mut external_links = HashMap::new();
        external_links.insert("url".to_string(),
            Regex::new(r"https?://[^\s\]]+").unwrap());
        external_links.insert("markdown_link".to_string(),
            Regex::new(r"\[([^\]]+)\]\(([^)]+)\)").unwrap());
        external_links.insert("email".to_string(),
            Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap());

        let mut internal_links = HashMap::new();
        internal_links.insert("anchor".to_string(),
            Regex::new(r"#([a-zA-Z][\w-]*)").unwrap());
        internal_links.insert("relative_link".to_string(),
            Regex::new(r"\[([^\]]+)\]\(\.?/([^)]+)\)").unwrap());

        let mut bibliography = HashMap::new();
        bibliography.insert("reference".to_string(),
            Regex::new(r"(?m)^\[(\w+)\]:\s*(.+)$").unwrap());
        bibliography.insert("bibtex".to_string(),
            Regex::new(r"@(\w+)\{([^,]+),").unwrap());

        Self {
            cross_references,
            citations,
            footnotes,
            table_figure_refs,
            section_refs,
            external_links,
            internal_links,
            bibliography,
        }
    }
}

impl ReferenceTracker {
    /// Creates a new reference tracker with default configuration
    pub fn new() -> Self {
        Self::with_config(ReferenceConfig::default())
    }

    /// Creates a reference tracker with custom configuration
    pub fn with_config(config: ReferenceConfig) -> Self {
        Self {
            patterns: REFERENCE_PATTERNS.clone(),
            resolver: ReferenceResolver::new(),
            cross_doc_manager: CrossDocumentManager::new(),
            quality_assessor: ReferenceQualityAssessor::new(),
            graph_builder: ReferenceGraphBuilder::new(),
            config,
        }
    }

    /// Extracts all references from chunk content
    pub fn extract_references(&self, content: &str) -> Vec<ChunkReference> {
        let mut references = Vec::new();

        // Extract different types of references
        references.extend(self.extract_cross_references(content));
        references.extend(self.extract_citations(content));
        references.extend(self.extract_footnotes(content));
        references.extend(self.extract_table_figure_refs(content));
        references.extend(self.extract_section_refs(content));
        references.extend(self.extract_external_links(content));
        references.extend(self.extract_internal_links(content));

        // Filter by confidence threshold
        references.retain(|r| r.confidence >= self.config.min_confidence as f64);

        // Limit number of references per chunk
        references.truncate(self.config.max_references_per_chunk);

        references
    }

    /// Extracts extended references with full metadata
    pub fn extract_extended_references(&self, content: &str, chunk_id: Uuid) -> Vec<ExtendedReference> {
        let base_references = self.extract_references(content);
        let mut extended_refs = Vec::new();

        for base_ref in base_references {
            let extended = ExtendedReference {
                metadata: self.build_reference_metadata(&base_ref, content, chunk_id),
                resolution: self.resolve_reference(&base_ref, content),
                quality: self.assess_reference_quality(&base_ref, content),
                graph_info: self.build_graph_info(&base_ref),
                base: base_ref,
            };
            extended_refs.push(extended);
        }

        extended_refs
    }

    /// Resolves references against document structure
    pub fn resolve_references(&mut self, references: &mut Vec<ChunkReference>, document_structure: &DocumentStructure) -> crate::Result<usize> {
        let mut resolved_count = 0;

        for reference in references.iter_mut() {
            if let Some(resolution) = self.attempt_resolution(reference, document_structure) {
                if let Some(target_id) = &resolution.target {
                    reference.target_id = target_id.target_id.clone();
                    resolved_count += 1;
                }
            }
        }

        Ok(resolved_count)
    }

    /// Builds reference graph for chunk relationships
    pub fn build_reference_graph(&mut self, chunks: &[crate::Chunk]) -> ReferenceGraph {
        let mut graph = ReferenceGraph::new();

        // Add nodes for each chunk
        for chunk in chunks {
            graph.add_node(chunk.id.to_string(), chunk.clone());
        }

        // Add edges for references
        for chunk in chunks {
            for reference in &chunk.references {
                if let Some(target_id) = &reference.target_id {
                    graph.add_edge(
                        chunk.id.to_string(),
                        target_id.to_string(),
                        reference.clone(),
                    );
                }
            }
        }

        // Analyze graph properties
        graph.analyze();
        graph
    }

    /// Validates reference integrity
    pub fn validate_references(&self, references: &[ChunkReference]) -> Vec<ReferenceValidationResult> {
        references.iter().map(|r| self.validate_single_reference(r)).collect()
    }

    /// Tracks cross-document references
    pub fn track_cross_document_references(&mut self, doc_id: &str, references: &[ChunkReference]) {
        self.cross_doc_manager.add_document_references(doc_id, references);
    }

    /// Updates document structure for reference resolution
    pub fn update_document_structure(&mut self, structure: DocumentStructure) {
        self.resolver.document_structure = structure;
        self.resolver.reference_cache.clear(); // Invalidate cache
    }

    // Internal extraction methods
    fn extract_cross_references(&self, content: &str) -> Vec<ChunkReference> {
        let mut references = Vec::new();

        for (ref_type, pattern) in &self.patterns.cross_references {
            for captures in pattern.captures_iter(content) {
                if let Some(target) = captures.get(1) {
                    let reference = ChunkReference {
                        reference_type: ReferenceType::CrossReference,
                        target_id: None,
                        target_text: target.as_str().to_string(),
                        context: self.extract_reference_context(content, captures.get(0).unwrap().start()),
                        confidence: self.calculate_cross_reference_confidence(ref_type, &captures),
                    };
                    references.push(reference);
                }
            }
        }

        references
    }

    fn extract_citations(&self, content: &str) -> Vec<ChunkReference> {
        let mut references = Vec::new();

        for (citation_type, pattern) in &self.patterns.citations {
            for captures in pattern.captures_iter(content) {
                let target_text = match citation_type.as_str() {
                    "numbered" => captures.get(1).map(|m| m.as_str()).unwrap_or("").to_string(),
                    "author_year" | "inline" => {
                        format!("{} ({})", 
                               captures.get(1).map(|m| m.as_str()).unwrap_or(""),
                               captures.get(2).map(|m| m.as_str()).unwrap_or(""))
                    }
                    _ => captures.get(0).map(|m| m.as_str()).unwrap_or("").to_string(),
                };

                let reference = ChunkReference {
                    reference_type: ReferenceType::Citation,
                    target_id: None,
                    target_text,
                    context: self.extract_reference_context(content, captures.get(0).unwrap().start()),
                    confidence: self.calculate_citation_confidence(citation_type, &captures),
                };
                references.push(reference);
            }
        }

        references
    }

    fn extract_footnotes(&self, content: &str) -> Vec<ChunkReference> {
        let mut references = Vec::new();

        for (footnote_type, pattern) in &self.patterns.footnotes {
            for captures in pattern.captures_iter(content) {
                let target_text = captures.get(1)
                    .or_else(|| captures.get(2))
                    .map(|m| m.as_str())
                    .unwrap_or("")
                    .to_string();

                let reference = ChunkReference {
                    reference_type: ReferenceType::Footnote,
                    target_id: None,
                    target_text,
                    context: self.extract_reference_context(content, captures.get(0).unwrap().start()),
                    confidence: self.calculate_footnote_confidence(footnote_type),
                };
                references.push(reference);
            }
        }

        references
    }

    fn extract_table_figure_refs(&self, content: &str) -> Vec<ChunkReference> {
        let mut references = Vec::new();

        for (ref_type, pattern) in &self.patterns.table_figure_refs {
            for captures in pattern.captures_iter(content) {
                if let Some(target) = captures.get(1) {
                    let reference_type = match ref_type.as_str() {
                        "table" => ReferenceType::TableReference,
                        "figure" => ReferenceType::FigureReference,
                        "equation" => ReferenceType::CrossReference,
                        _ => ReferenceType::CrossReference,
                    };

                    let reference = ChunkReference {
                        reference_type,
                        target_id: None,
                        target_text: target.as_str().to_string(),
                        context: self.extract_reference_context(content, captures.get(0).unwrap().start()),
                        confidence: self.calculate_table_figure_confidence(ref_type, &captures),
                    };
                    references.push(reference);
                }
            }
        }

        references
    }

    fn extract_section_refs(&self, content: &str) -> Vec<ChunkReference> {
        let mut references = Vec::new();

        // This is primarily for detecting when content IS a section/header
        // rather than referencing one
        for (ref_type, pattern) in &self.patterns.section_refs {
            for captures in pattern.captures_iter(content) {
                if ref_type == "heading" {
                    if let Some(title) = captures.get(1) {
                        let reference = ChunkReference {
                            reference_type: ReferenceType::SectionReference,
                            target_id: None,
                            target_text: title.as_str().to_string(),
                            context: self.extract_reference_context(content, captures.get(0).unwrap().start()),
                            confidence: 0.9, // High confidence for headers
                        };
                        references.push(reference);
                    }
                }
            }
        }

        references
    }

    fn extract_external_links(&self, content: &str) -> Vec<ChunkReference> {
        let mut references = Vec::new();

        for (link_type, pattern) in &self.patterns.external_links {
            for captures in pattern.captures_iter(content) {
                let (target_text, confidence) = match link_type.as_str() {
                    "url" => (captures.get(0).unwrap().as_str().to_string(), 0.95),
                    "markdown_link" => (captures.get(2).unwrap().as_str().to_string(), 0.9),
                    "email" => (captures.get(0).unwrap().as_str().to_string(), 0.85),
                    _ => (captures.get(0).unwrap().as_str().to_string(), 0.8),
                };

                let reference = ChunkReference {
                    reference_type: ReferenceType::ExternalLink,
                    target_id: None,
                    target_text,
                    context: self.extract_reference_context(content, captures.get(0).unwrap().start()),
                    confidence,
                };
                references.push(reference);
            }
        }

        references
    }

    fn extract_internal_links(&self, content: &str) -> Vec<ChunkReference> {
        let mut references = Vec::new();

        for (link_type, pattern) in &self.patterns.internal_links {
            for captures in pattern.captures_iter(content) {
                let target_text = match link_type.as_str() {
                    "anchor" => captures.get(1).unwrap().as_str().to_string(),
                    "relative_link" => captures.get(2).unwrap().as_str().to_string(),
                    _ => captures.get(0).unwrap().as_str().to_string(),
                };

                let reference = ChunkReference {
                    reference_type: ReferenceType::CrossReference,
                    target_id: None,
                    target_text,
                    context: self.extract_reference_context(content, captures.get(0).unwrap().start()),
                    confidence: 0.85,
                };
                references.push(reference);
            }
        }

        references
    }

    fn extract_reference_context(&self, content: &str, position: usize) -> String {
        let start = position.saturating_sub(self.config.context_window / 2);
        let end = (position + self.config.context_window / 2).min(content.len());
        
        content[start..end].to_string()
    }

    // Confidence calculation methods
    fn calculate_cross_reference_confidence(&self, ref_type: &str, captures: &regex::Captures) -> f32 {
        let mut confidence = 0.7f32; // Base confidence

        // Adjust based on reference type
        match ref_type {
            "section" => confidence += 0.15,
            "page" => confidence += 0.1,
            "appendix" => confidence += 0.2,
            _ => {}
        }

        // Check for explicit reference indicators
        let full_match = captures.get(0).unwrap().as_str();
        if full_match.to_lowercase().contains("see") {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }

    fn calculate_citation_confidence(&self, citation_type: &str, _captures: &regex::Captures) -> f32 {
        match citation_type {
            "numbered" => 0.95,
            "author_year" => 0.9,
            "inline" => 0.85,
            _ => 0.8,
        }
    }

    fn calculate_footnote_confidence(&self, footnote_type: &str) -> f32 {
        match footnote_type {
            "superscript" => 0.9,
            "definition" => 0.85,
            _ => 0.8,
        }
    }

    fn calculate_table_figure_confidence(&self, ref_type: &str, _captures: &regex::Captures) -> f32 {
        match ref_type {
            "table" => 0.9,
            "figure" => 0.9,
            "equation" => 0.85,
            _ => 0.8,
        }
    }

    // Reference processing methods
    fn build_reference_metadata(&self, reference: &ChunkReference, _content: &str, chunk_id: Uuid) -> ReferenceMetadata {
        ReferenceMetadata {
            source_location: LocationInfo {
                document_id: "current".to_string(), // Would be provided in real implementation
                chunk_id: Some(chunk_id),
                char_position: 0, // Would calculate from content
                line_number: 0,
                column_number: 0,
                context: reference.context.clone(),
            },
            target_location: None,
            category: self.classify_reference_category(reference),
            semantic_context: self.analyze_semantic_context(reference),
            created_at: chrono::Utc::now(),
            last_validated: None,
        }
    }

    fn resolve_reference(&self, _reference: &ChunkReference, _content: &str) -> Option<ReferenceResolution> {
        // Simplified implementation - would implement full resolution logic
        None
    }

    fn assess_reference_quality(&self, reference: &ChunkReference, _content: &str) -> ReferenceQuality {
        let clarity = if !reference.target_text.is_empty() { 0.8 } else { 0.3 };
        let relevance = reference.confidence;
        let accuracy = if reference.target_id.is_some() { 0.9 } else { 0.5 };
        let completeness = if !reference.context.is_empty() { 0.8 } else { 0.4 };

        ReferenceQuality {
            overall_score: (clarity + relevance + accuracy + completeness) / 4.0,
            clarity_score: clarity,
            relevance_score: relevance,
            accuracy_score: accuracy,
            completeness_score: completeness,
            issues: Vec::new(),
            recommendations: Vec::new(),
        }
    }

    fn build_graph_info(&self, _reference: &ChunkReference) -> GraphInfo {
        GraphInfo {
            node_id: Uuid::new_v4().to_string(),
            in_degree: 0,
            out_degree: 1,
            centrality: CentralityMeasures {
                degree: 0.0,
                betweenness: 0.0,
                closeness: 0.0,
                pagerank: 0.0,
            },
            component_id: "default".to_string(),
        }
    }

    fn classify_reference_category(&self, reference: &ChunkReference) -> ReferenceCategory {
        match reference.reference_type {
            ReferenceType::CrossReference => ReferenceCategory::Structural,
            ReferenceType::Citation => ReferenceCategory::Bibliographic,
            ReferenceType::Footnote => ReferenceCategory::Footnote,
            ReferenceType::TableReference | ReferenceType::FigureReference => ReferenceCategory::Content,
            ReferenceType::SectionReference => ReferenceCategory::Structural,
            ReferenceType::ExternalLink => ReferenceCategory::Hyperlink,
        }
    }

    fn analyze_semantic_context(&self, reference: &ChunkReference) -> SemanticContext {
        // Simplified semantic analysis
        let context_type = if reference.context.to_lowercase().contains("see") {
            ContextType::Navigation
        } else if reference.context.to_lowercase().contains("definition") {
            ContextType::Definition
        } else {
            ContextType::Support
        };

        SemanticContext {
            context_type,
            topics: Vec::new(), // Would extract from context
            intent: ReferenceIntent::Informational,
            importance: reference.confidence,
        }
    }

    fn attempt_resolution(&self, _reference: &ChunkReference, _structure: &DocumentStructure) -> Option<ReferenceResolution> {
        // Simplified resolution - would implement sophisticated matching
        None
    }

    fn validate_single_reference(&self, reference: &ChunkReference) -> ReferenceValidationResult {
        let mut issues = Vec::new();
        let mut is_valid = true;

        // Check target text is not empty
        if reference.target_text.is_empty() {
            issues.push(QualityIssue {
                issue_type: IssueType::MissingContext,
                severity: IssueSeverity::High,
                description: "Reference has empty target text".to_string(),
                suggestion: Some("Provide descriptive target text".to_string()),
            });
            is_valid = false;
        }

        // Check confidence level
        if reference.confidence < self.config.min_confidence {
            issues.push(QualityIssue {
                issue_type: IssueType::LowConfidence,
                severity: IssueSeverity::Medium,
                description: format!("Reference confidence {} below threshold {}", 
                                   reference.confidence, self.config.min_confidence),
                suggestion: Some("Review and improve reference quality".to_string()),
            });
        }

        ReferenceValidationResult {
            is_valid,
            issues,
            confidence: reference.confidence,
            recommendations: Vec::new(),
        }
    }
}

// Supporting types and implementations
#[derive(Debug, Clone)]
pub struct ReferenceValidationResult {
    pub is_valid: bool,
    pub issues: Vec<QualityIssue>,
    pub confidence: f32,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ReferenceGraph {
    nodes: HashMap<String, crate::Chunk>,
    edges: HashMap<String, Vec<(String, ChunkReference)>>,
    analyzed: bool,
}

impl ReferenceGraph {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            analyzed: false,
        }
    }

    fn add_node(&mut self, id: String, chunk: crate::Chunk) {
        self.nodes.insert(id, chunk);
        self.analyzed = false;
    }

    fn add_edge(&mut self, source: String, target: String, reference: ChunkReference) {
        self.edges.entry(source)
            .or_insert_with(Vec::new)
            .push((target, reference));
        self.analyzed = false;
    }

    fn analyze(&mut self) {
        // Would implement graph analysis algorithms
        self.analyzed = true;
    }

    pub fn get_connected_chunks(&self, chunk_id: &str) -> Vec<&crate::Chunk> {
        let mut connected = Vec::new();
        
        if let Some(edges) = self.edges.get(chunk_id) {
            for (target_id, _) in edges {
                if let Some(chunk) = self.nodes.get(target_id) {
                    connected.push(chunk);
                }
            }
        }

        connected
    }

    pub fn get_reference_count(&self, chunk_id: &str) -> usize {
        self.edges.get(chunk_id).map(|edges| edges.len()).unwrap_or(0)
    }
}

// Default implementations for helper types
#[derive(Debug, Clone)]
struct ResolvedReference;

#[derive(Debug, Clone)]  
struct UnresolvedReference;

#[derive(Debug, Clone)]
struct SectionInfo;

#[derive(Debug, Clone)]
struct TocEntry;

#[derive(Debug, Clone)]
struct FigureInfo;

#[derive(Debug, Clone)]
struct TableInfo;

#[derive(Debug, Clone)]
struct FootnoteInfo;

#[derive(Debug, Clone)]
struct BibliographyEntry;

#[derive(Debug, Clone)]
struct ExternalReference;

#[derive(Debug, Clone)]
struct DocumentMetadata;

#[derive(Debug, Clone)]
struct ValidationResult;

#[derive(Debug, Clone)]
struct QualityMetrics;

#[derive(Debug, Clone)]
struct ConfidenceCalculators;

#[derive(Debug, Clone)]
struct ContextAnalyzers;

#[derive(Debug, Clone)]
struct GraphAnalyzer;

#[derive(Debug, Clone)]
struct PathFinder;

impl ReferenceResolver {
    fn new() -> Self {
        Self {
            document_structure: DocumentStructure::new(),
            reference_cache: HashMap::new(),
            forward_references: HashMap::new(),
            backward_references: HashMap::new(),
        }
    }
}

impl DocumentStructure {
    fn new() -> Self {
        Self {
            sections: Vec::new(),
            toc: Vec::new(),
            figures: HashMap::new(),
            tables: HashMap::new(),
            footnotes: HashMap::new(),
            bibliography: HashMap::new(),
        }
    }
}

impl CrossDocumentManager {
    fn new() -> Self {
        Self {
            document_refs: HashMap::new(),
            document_metadata: HashMap::new(),
            validation_cache: HashMap::new(),
        }
    }

    fn add_document_references(&mut self, doc_id: &str, _references: &[ChunkReference]) {
        // Would implement cross-document reference tracking
        self.document_refs.insert(doc_id.to_string(), Vec::new());
    }
}

impl ReferenceQualityAssessor {
    fn new() -> Self {
        Self {
            quality_metrics: QualityMetrics,
            confidence_calculators: ConfidenceCalculators,
            context_analyzers: ContextAnalyzers,
        }
    }
}

impl ReferenceGraphBuilder {
    fn new() -> Self {
        Self {
            graph: ReferenceGraph::new(),
            analyzer: GraphAnalyzer,
            path_finder: PathFinder,
        }
    }
}

impl Default for ReferenceTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_tracker_creation() {
        let tracker = ReferenceTracker::new();
        assert_eq!(tracker.config.min_confidence, 0.7);
    }

    #[test]
    fn test_cross_reference_extraction() {
        let tracker = ReferenceTracker::new();
        let content = "Please see section 2.1 for more details. Also refer to page 42.";
        
        let references = tracker.extract_references(content);
        
        assert!(!references.is_empty());
        assert!(references.iter().any(|r| r.reference_type == ReferenceType::CrossReference));
        assert!(references.iter().any(|r| r.target_text.contains("2.1") || r.target_text.contains("42")));
    }

    #[test]
    fn test_citation_extraction() {
        let tracker = ReferenceTracker::new();
        let content = "This is supported by research [1, 2] and Smith et al. (2023).";
        
        let references = tracker.extract_references(content);
        
        let citations: Vec<_> = references.iter()
            .filter(|r| r.reference_type == ReferenceType::Citation)
            .collect();
        
        assert!(!citations.is_empty());
    }

    #[test]
    fn test_external_link_extraction() {
        let tracker = ReferenceTracker::new();
        let content = "Visit https://example.com for more information or [click here](https://test.com).";
        
        let references = tracker.extract_references(content);
        
        let links: Vec<_> = references.iter()
            .filter(|r| r.reference_type == ReferenceType::ExternalLink)
            .collect();
        
        assert!(!links.is_empty());
        assert!(links.iter().any(|r| r.target_text.contains("example.com")));
    }

    #[test]
    fn test_table_figure_reference_extraction() {
        let tracker = ReferenceTracker::new();
        let content = "As shown in Table 1 and Figure 2.1, the results are significant.";
        
        let references = tracker.extract_references(content);
        
        assert!(references.iter().any(|r| r.reference_type == ReferenceType::TableReference));
        assert!(references.iter().any(|r| r.reference_type == ReferenceType::FigureReference));
    }

    #[test]
    fn test_confidence_filtering() {
        let mut config = ReferenceConfig::default();
        config.min_confidence = 0.95; // Very high threshold
        
        let tracker = ReferenceTracker::with_config(config);
        let content = "Maybe see section 1 or something."; // Low confidence reference
        
        let references = tracker.extract_references(content);
        
        // Should filter out low confidence references
        assert!(references.is_empty() || references.iter().all(|r| r.confidence >= 0.95));
    }

    #[test]
    fn test_reference_context_extraction() {
        let tracker = ReferenceTracker::new();
        let content = "This is a longer paragraph with some context before see section 2.1 and some context after the reference.";
        
        let references = tracker.extract_references(content);
        
        if let Some(reference) = references.first() {
            assert!(!reference.context.is_empty());
            assert!(reference.context.contains("see section 2.1"));
        }
    }

    #[test]
    fn test_reference_graph_building() {
        let mut tracker = ReferenceTracker::new();
        let chunks = vec![]; // Would populate with test chunks
        
        let graph = tracker.build_reference_graph(&chunks);
        
        // Basic graph structure validation
        assert_eq!(graph.nodes.len(), chunks.len());
    }

    #[test]
    fn test_reference_validation() {
        let tracker = ReferenceTracker::new();
        let reference = crate::chunk::ChunkReference {
            reference_type: crate::chunk::ReferenceType::CrossReference,
            target_id: "section-1".to_string(),
            context: Some("See section 1 for details".to_string()),
            confidence: 0.8,
        };
        
        let validation = tracker.validate_single_reference(&reference);
        
        assert!(validation.is_valid);
        assert_eq!(validation.confidence, 0.8);
    }

    #[test]
    fn test_empty_target_validation() {
        let tracker = ReferenceTracker::new();
        let reference = crate::chunk::ChunkReference {
            reference_type: crate::chunk::ReferenceType::CrossReference,
            target_id: "".to_string(), // Empty target
            context: Some("Some context".to_string()),
            confidence: 0.8,
        };
        
        let validation = tracker.validate_single_reference(&reference);
        
        assert!(!validation.is_valid);
        assert!(!validation.issues.is_empty());
    }
}