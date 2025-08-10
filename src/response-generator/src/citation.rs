//! Citation tracking and source attribution system for response generation

use crate::error::{Result, ResponseError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

/// Citation tracker for managing source attribution
#[derive(Debug, Clone)]
pub struct CitationTracker {
    /// Configuration for citation processing
    config: CitationConfig,
    
    /// Cache of processed sources
    source_cache: HashMap<Uuid, Source>,
    
    /// Deduplication tracking
    deduplication_index: HashMap<String, Uuid>,
}

/// Source information with metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Source {
    /// Unique identifier for the source
    pub id: Uuid,
    
    /// Title of the source document
    pub title: String,
    
    /// URL or location of the source
    pub url: Option<String>,
    
    /// Type of document (pdf, webpage, text, etc.)
    pub document_type: String,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Citation with position and confidence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// Unique citation identifier
    pub id: Uuid,
    
    /// Source being cited
    pub source: Source,
    
    /// Position in response text where citation applies
    pub text_range: TextRange,
    
    /// Confidence in this citation's accuracy
    pub confidence: f64,
    
    /// Type of citation
    pub citation_type: CitationType,
    
    /// Relevance score to the cited text
    pub relevance_score: f64,
    
    /// Supporting quote or excerpt
    pub supporting_text: Option<String>,
}

/// Text range for citation positioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRange {
    /// Start character position
    pub start: usize,
    
    /// End character position
    pub end: usize,
    
    /// Length of cited text
    pub length: usize,
}

/// Types of citations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CitationType {
    /// Direct quote from source
    DirectQuote,
    
    /// Paraphrased content
    Paraphrase,
    
    /// Supporting evidence
    SupportingEvidence,
    
    /// Background context
    BackgroundContext,
    
    /// Statistical data
    Statistical,
    
    /// Reference for further reading
    Reference,
}

/// Source ranking system for prioritization
#[derive(Debug, Clone)]
pub struct SourceRanking {
    /// Credibility score (0.0-1.0)
    pub credibility: f64,
    
    /// Recency score (0.0-1.0)
    pub recency: f64,
    
    /// Authority score (0.0-1.0)
    pub authority: f64,
    
    /// Relevance score (0.0-1.0)
    pub relevance: f64,
    
    /// Overall ranking score
    pub overall_score: f64,
}

/// Configuration for citation processing
#[derive(Debug, Clone)]
pub struct CitationConfig {
    /// Maximum number of citations per response
    pub max_citations: usize,
    
    /// Minimum confidence threshold for citations
    pub min_confidence: f64,
    
    /// Enable source deduplication
    pub deduplicate_sources: bool,
    
    /// Citation style (APA, MLA, Chicago, etc.)
    pub citation_style: CitationStyle,
    
    /// Include page numbers when available
    pub include_page_numbers: bool,
    
    /// Maximum length for supporting text excerpts
    pub max_excerpt_length: usize,
}

/// Citation formatting styles
#[derive(Debug, Clone, PartialEq)]
pub enum CitationStyle {
    APA,
    MLA,
    Chicago,
    IEEE,
    Harvard,
    Custom(String),
}

impl Default for CitationConfig {
    fn default() -> Self {
        Self {
            max_citations: 10,
            min_confidence: 0.5,
            deduplicate_sources: true,
            citation_style: CitationStyle::APA,
            include_page_numbers: true,
            max_excerpt_length: 200,
        }
    }
}

impl CitationTracker {
    /// Create a new citation tracker
    pub fn new() -> Self {
        Self::with_config(CitationConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: CitationConfig) -> Self {
        Self {
            config,
            source_cache: HashMap::new(),
            deduplication_index: HashMap::new(),
        }
    }

    /// Process citations for a response
    #[instrument(skip(self, response, context_chunks))]
    pub async fn process_citations(
        &mut self,
        response: &crate::IntermediateResponse,
        context_chunks: &[crate::ContextChunk],
    ) -> Result<Vec<Citation>> {
        debug!("Processing citations for response with {} source references", response.source_references.len());

        // Build source map from context chunks
        self.build_source_cache(context_chunks).await?;

        // Extract citation opportunities from response text
        let citation_opportunities = self.extract_citation_opportunities(response).await?;
        let num_opportunities = citation_opportunities.len();

        // Match opportunities with sources
        let mut citations = self.match_citations_to_sources(citation_opportunities, response).await?;

        // Rank and filter citations
        citations = self.rank_and_filter_citations(citations).await?;

        // Deduplicate sources if enabled
        if self.config.deduplicate_sources {
            citations = self.deduplicate_citations(citations).await?;
        }

        // Validate citations
        citations = self.validate_citations(citations).await?;

        debug!("Generated {} citations from {} opportunities", citations.len(), num_opportunities);
        Ok(citations)
    }

    /// Add a source to the cache
    pub async fn add_source(&mut self, source: Source) -> Result<()> {
        // Create deduplication key
        let dedup_key = self.create_deduplication_key(&source);
        
        if let Some(existing_id) = self.deduplication_index.get(&dedup_key) {
            debug!("Source already exists with ID: {}", existing_id);
            return Ok(());
        }

        self.deduplication_index.insert(dedup_key, source.id);
        self.source_cache.insert(source.id, source);
        Ok(())
    }

    /// Get source by ID
    pub fn get_source(&self, id: &Uuid) -> Option<&Source> {
        self.source_cache.get(id)
    }

    /// Calculate source ranking
    #[instrument(skip(self, source))]
    pub async fn calculate_source_ranking(&self, source: &Source, query: &str) -> Result<SourceRanking> {
        // Calculate credibility based on source metadata
        let credibility = self.calculate_credibility(source).await?;
        
        // Calculate recency based on publication date
        let recency = self.calculate_recency(source).await?;
        
        // Calculate authority based on source type and metadata
        let authority = self.calculate_authority(source).await?;
        
        // Calculate relevance to query
        let relevance = self.calculate_source_relevance(source, query).await?;
        
        // Weighted overall score
        let overall_score = (credibility * 0.3) + (recency * 0.2) + (authority * 0.3) + (relevance * 0.2);

        Ok(SourceRanking {
            credibility,
            recency,
            authority,
            relevance,
            overall_score,
        })
    }

    /// Format citation according to configured style
    pub fn format_citation(&self, citation: &Citation) -> String {
        match self.config.citation_style {
            CitationStyle::APA => self.format_apa_citation(citation),
            CitationStyle::MLA => self.format_mla_citation(citation),
            CitationStyle::Chicago => self.format_chicago_citation(citation),
            CitationStyle::IEEE => self.format_ieee_citation(citation),
            CitationStyle::Harvard => self.format_harvard_citation(citation),
            CitationStyle::Custom(ref format) => self.format_custom_citation(citation, format),
        }
    }

    /// Generate bibliography from citations
    pub fn generate_bibliography(&self, citations: &[Citation]) -> Vec<String> {
        let mut bibliography = Vec::new();
        let mut seen_sources = HashSet::new();

        for citation in citations {
            if seen_sources.insert(citation.source.id) {
                bibliography.push(self.format_bibliography_entry(&citation.source));
            }
        }

        // Sort bibliography entries
        bibliography.sort();
        bibliography
    }

    /// Build source cache from context chunks
    async fn build_source_cache(&mut self, context_chunks: &[crate::ContextChunk]) -> Result<()> {
        for chunk in context_chunks {
            self.add_source(chunk.source.clone()).await?;
        }
        Ok(())
    }

    /// Extract citation opportunities from response text
    async fn extract_citation_opportunities(&self, response: &crate::IntermediateResponse) -> Result<Vec<CitationOpportunity>> {
        let mut opportunities = Vec::new();
        
        // Split text into sentences for citation analysis
        let sentences = self.split_into_sentences(&response.content);
        
        for (index, sentence) in sentences.iter().enumerate() {
            let opportunity = CitationOpportunity {
                text: sentence.clone(),
                position: self.calculate_sentence_position(&response.content, index),
                confidence_indicators: self.analyze_confidence_indicators(sentence).await?,
                citation_necessity: self.assess_citation_necessity(sentence).await?,
            };
            
            if opportunity.citation_necessity > 0.3 {
                opportunities.push(opportunity);
            }
        }

        Ok(opportunities)
    }

    /// Match citation opportunities with available sources
    async fn match_citations_to_sources(
        &self,
        opportunities: Vec<CitationOpportunity>,
        response: &crate::IntermediateResponse,
    ) -> Result<Vec<Citation>> {
        let mut citations = Vec::new();

        for opportunity in opportunities {
            let best_match = self.find_best_source_match(&opportunity, &response.source_references).await?;
            
            if let Some((source_id, relevance_score)) = best_match {
                if let Some(source) = self.source_cache.get(&source_id) {
                    let citation = Citation {
                        id: Uuid::new_v4(),
                        source: source.clone(),
                        text_range: TextRange {
                            start: opportunity.position,
                            end: opportunity.position + opportunity.text.len(),
                            length: opportunity.text.len(),
                        },
                        confidence: opportunity.confidence_indicators,
                        citation_type: self.determine_citation_type(&opportunity).await?,
                        relevance_score,
                        supporting_text: self.extract_supporting_text(source, &opportunity.text).await?,
                    };
                    
                    citations.push(citation);
                }
            }
        }

        Ok(citations)
    }

    /// Rank and filter citations based on quality and relevance
    async fn rank_and_filter_citations(&self, mut citations: Vec<Citation>) -> Result<Vec<Citation>> {
        // Sort by combined confidence and relevance score
        citations.sort_by(|a, b| {
            let score_a = (a.confidence + a.relevance_score) / 2.0;
            let score_b = (b.confidence + b.relevance_score) / 2.0;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Filter by minimum confidence
        citations.retain(|c| c.confidence >= self.config.min_confidence);

        // Limit to maximum citations
        citations.truncate(self.config.max_citations);

        Ok(citations)
    }

    /// Deduplicate citations from the same source
    async fn deduplicate_citations(&self, citations: Vec<Citation>) -> Result<Vec<Citation>> {
        let mut deduplicated = Vec::new();
        let mut source_citation_count: HashMap<Uuid, usize> = HashMap::new();

        for citation in citations {
            let count = source_citation_count.entry(citation.source.id).or_insert(0);
            
            // Allow maximum 2 citations per source for diversity
            if *count < 2 {
                deduplicated.push(citation);
                *count += 1;
            }
        }

        Ok(deduplicated)
    }

    /// Validate citation quality and accuracy
    async fn validate_citations(&self, citations: Vec<Citation>) -> Result<Vec<Citation>> {
        let mut validated = Vec::new();

        for citation in citations {
            // Validate text range
            if citation.text_range.start >= citation.text_range.end {
                warn!("Invalid text range in citation {}", citation.id);
                continue;
            }

            // Validate confidence score
            if citation.confidence < 0.0 || citation.confidence > 1.0 {
                warn!("Invalid confidence score in citation {}", citation.id);
                continue;
            }

            // Validate source information
            if citation.source.title.is_empty() {
                warn!("Empty source title in citation {}", citation.id);
                continue;
            }

            validated.push(citation);
        }

        Ok(validated)
    }

    /// Calculate source credibility
    async fn calculate_credibility(&self, source: &Source) -> Result<f64> {
        let mut credibility_factors = Vec::new();

        // Domain authority (if URL available)
        if let Some(url) = &source.url {
            let domain_credibility = self.assess_domain_credibility(url).await?;
            credibility_factors.push(domain_credibility);
        }

        // Source type credibility
        let type_credibility = match source.document_type.as_str() {
            "academic" => 0.9,
            "government" => 0.85,
            "news" => 0.7,
            "blog" => 0.5,
            "social" => 0.3,
            _ => 0.6,
        };
        credibility_factors.push(type_credibility);

        // Metadata-based credibility
        if let Some(peer_reviewed) = source.metadata.get("peer_reviewed") {
            if peer_reviewed == "true" {
                credibility_factors.push(0.95);
            }
        }

        if credibility_factors.is_empty() {
            Ok(0.5) // Default neutral credibility
        } else {
            Ok(credibility_factors.iter().sum::<f64>() / credibility_factors.len() as f64)
        }
    }

    /// Calculate source recency
    async fn calculate_recency(&self, source: &Source) -> Result<f64> {
        if let Some(publish_date) = source.metadata.get("publish_date") {
            // Parse date and calculate recency score
            if let Ok(date) = chrono::DateTime::parse_from_rfc3339(publish_date) {
                let now = chrono::Utc::now();
                let age_days = (now - date.with_timezone(&chrono::Utc)).num_days();
                
                // Exponential decay: recent sources get higher scores
                let recency = (-age_days as f64 / 365.0).exp().max(0.1);
                return Ok(recency.min(1.0));
            }
        }

        Ok(0.5) // Default neutral recency
    }

    /// Calculate source authority
    async fn calculate_authority(&self, source: &Source) -> Result<f64> {
        let mut authority_factors = Vec::new();

        // Author credentials
        if let Some(author) = source.metadata.get("author") {
            let author_authority = self.assess_author_authority(author).await?;
            authority_factors.push(author_authority);
        }

        // Publication authority
        if let Some(publication) = source.metadata.get("publication") {
            let pub_authority = self.assess_publication_authority(publication).await?;
            authority_factors.push(pub_authority);
        }

        // Citation count (academic authority)
        if let Some(citations) = source.metadata.get("citation_count") {
            if let Ok(count) = citations.parse::<u32>() {
                let citation_authority = (count as f64).log10() / 5.0; // Log scale
                authority_factors.push(citation_authority.min(1.0));
            }
        }

        if authority_factors.is_empty() {
            Ok(0.5) // Default neutral authority
        } else {
            Ok(authority_factors.iter().sum::<f64>() / authority_factors.len() as f64)
        }
    }

    /// Calculate source relevance to query
    async fn calculate_source_relevance(&self, source: &Source, query: &str) -> Result<f64> {
        let query_words: HashSet<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let title_words: HashSet<String> = source.title
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let intersection_count = query_words.intersection(&title_words).count();
        let relevance = intersection_count as f64 / query_words.len().max(1) as f64;

        Ok(relevance)
    }

    /// Create deduplication key for source
    fn create_deduplication_key(&self, source: &Source) -> String {
        if let Some(url) = &source.url {
            // Use URL as primary deduplication key
            url.clone()
        } else {
            // Use title + document type
            format!("{}:{}", source.title, source.document_type)
        }
    }

    /// Split text into sentences
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split('.')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// Calculate position of sentence in text
    fn calculate_sentence_position(&self, text: &str, sentence_index: usize) -> usize {
        let sentences: Vec<&str> = text.split('.').collect();
        
        sentences.iter()
            .take(sentence_index)
            .map(|s| s.len() + 1) // +1 for the period
            .sum()
    }

    /// Analyze confidence indicators in text
    async fn analyze_confidence_indicators(&self, sentence: &str) -> Result<f64> {
        let mut confidence: f64 = 0.5; // Base confidence

        // Look for confidence-boosting indicators
        let strong_indicators = ["studies show", "research indicates", "according to", "data shows"];
        let weak_indicators = ["might", "could", "possibly", "perhaps", "seems"];

        let sentence_lower = sentence.to_lowercase();

        for indicator in strong_indicators.iter() {
            if sentence_lower.contains(indicator) {
                confidence += 0.2;
            }
        }

        for indicator in weak_indicators.iter() {
            if sentence_lower.contains(indicator) {
                confidence -= 0.15;
            }
        }

        Ok(confidence.max(0.1).min(1.0))
    }

    /// Assess if a sentence needs citation
    async fn assess_citation_necessity(&self, sentence: &str) -> Result<f64> {
        let sentence_lower = sentence.to_lowercase();

        // High necessity indicators
        if sentence_lower.contains("statistics") ||
           sentence_lower.contains("study") ||
           sentence_lower.contains("research") ||
           sentence_lower.contains("data") ||
           sentence_lower.contains("according to") {
            return Ok(0.9);
        }

        // Medium necessity indicators
        if sentence_lower.contains("report") ||
           sentence_lower.contains("found") ||
           sentence_lower.contains("analysis") {
            return Ok(0.6);
        }

        // Low necessity (general statements)
        Ok(0.3)
    }

    /// Find best source match for citation opportunity
    async fn find_best_source_match(
        &self,
        opportunity: &CitationOpportunity,
        source_references: &[Uuid],
    ) -> Result<Option<(Uuid, f64)>> {
        let mut best_match = None;
        let mut best_score = 0.0;

        for source_id in source_references {
            if let Some(source) = self.source_cache.get(source_id) {
                let relevance_score = self.calculate_text_source_relevance(&opportunity.text, source).await?;
                
                if relevance_score > best_score {
                    best_score = relevance_score;
                    best_match = Some((*source_id, relevance_score));
                }
            }
        }

        Ok(best_match)
    }

    /// Calculate relevance between text and source
    async fn calculate_text_source_relevance(&self, text: &str, source: &Source) -> Result<f64> {
        let text_words: HashSet<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let source_words: HashSet<String> = source.title
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let intersection_count = text_words.intersection(&source_words).count();
        let union_count = text_words.union(&source_words).count();

        if union_count == 0 {
            Ok(0.0)
        } else {
            Ok(intersection_count as f64 / union_count as f64)
        }
    }

    /// Determine appropriate citation type
    async fn determine_citation_type(&self, opportunity: &CitationOpportunity) -> Result<CitationType> {
        let text_lower = opportunity.text.to_lowercase();

        if text_lower.contains("\"") || text_lower.contains("says") || text_lower.contains("states") {
            Ok(CitationType::DirectQuote)
        } else if text_lower.contains("according to") || text_lower.contains("based on") {
            Ok(CitationType::SupportingEvidence)
        } else if text_lower.contains("statistics") || text_lower.contains("data") || text_lower.contains("%") {
            Ok(CitationType::Statistical)
        } else {
            Ok(CitationType::Paraphrase)
        }
    }

    /// Extract supporting text from source
    async fn extract_supporting_text(&self, source: &Source, _cited_text: &str) -> Result<Option<String>> {
        // In a real implementation, this would search through the source content
        // For now, return a portion of the title as supporting context
        if source.title.len() > self.config.max_excerpt_length {
            Ok(Some(source.title[..self.config.max_excerpt_length].to_string() + "..."))
        } else {
            Ok(Some(source.title.clone()))
        }
    }

    /// Assess domain credibility
    async fn assess_domain_credibility(&self, url: &str) -> Result<f64> {
        // Simple domain credibility assessment
        if url.contains(".edu") {
            Ok(0.9)
        } else if url.contains(".gov") {
            Ok(0.85)
        } else if url.contains(".org") {
            Ok(0.7)
        } else if url.contains("wikipedia.org") {
            Ok(0.6)
        } else {
            Ok(0.5)
        }
    }

    /// Assess author authority
    async fn assess_author_authority(&self, _author: &str) -> Result<f64> {
        // Placeholder: would check author credentials, publications, etc.
        Ok(0.5)
    }

    /// Assess publication authority
    async fn assess_publication_authority(&self, _publication: &str) -> Result<f64> {
        // Placeholder: would check publication reputation, impact factor, etc.
        Ok(0.5)
    }

    // Citation formatting methods
    fn format_apa_citation(&self, citation: &Citation) -> String {
        let source = &citation.source;
        if let Some(url) = &source.url {
            format!("{}. Retrieved from {}", source.title, url)
        } else {
            source.title.clone()
        }
    }

    fn format_mla_citation(&self, citation: &Citation) -> String {
        let source = &citation.source;
        format!("\"{}\" Web.", source.title)
    }

    fn format_chicago_citation(&self, citation: &Citation) -> String {
        let source = &citation.source;
        if let Some(url) = &source.url {
            format!("\"{},\" accessed today, {}.", source.title, url)
        } else {
            format!("\"{}\"", source.title)
        }
    }

    fn format_ieee_citation(&self, citation: &Citation) -> String {
        format!("[{}]", citation.source.title)
    }

    fn format_harvard_citation(&self, citation: &Citation) -> String {
        format!("({})", citation.source.title)
    }

    fn format_custom_citation(&self, citation: &Citation, format: &str) -> String {
        // Custom format placeholder
        format.replace("{title}", &citation.source.title)
    }

    fn format_bibliography_entry(&self, source: &Source) -> String {
        if let Some(url) = &source.url {
            format!("{}. {}", source.title, url)
        } else {
            source.title.clone()
        }
    }
}

/// Citation opportunity identified in text
#[derive(Debug, Clone)]
struct CitationOpportunity {
    text: String,
    position: usize,
    confidence_indicators: f64,
    citation_necessity: f64,
}

impl Default for CitationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_creation() {
        let source = Source {
            id: Uuid::new_v4(),
            title: "Test Source".to_string(),
            url: Some("https://example.com".to_string()),
            document_type: "webpage".to_string(),
            metadata: HashMap::new(),
        };

        assert_eq!(source.title, "Test Source");
        assert_eq!(source.document_type, "webpage");
    }

    #[tokio::test]
    async fn test_citation_tracker_creation() {
        let tracker = CitationTracker::new();
        assert_eq!(tracker.config.max_citations, 10);
        assert_eq!(tracker.config.citation_style, CitationStyle::APA);
    }

    #[tokio::test]
    async fn test_source_credibility_calculation() {
        let tracker = CitationTracker::new();
        let source = Source {
            id: Uuid::new_v4(),
            title: "Academic Paper".to_string(),
            url: Some("https://university.edu/paper".to_string()),
            document_type: "academic".to_string(),
            metadata: HashMap::new(),
        };

        let credibility = tracker.calculate_credibility(&source).await.unwrap();
        assert!(credibility > 0.5); // Should be higher than default for academic + .edu
    }

    #[test]
    fn test_citation_formatting() {
        let tracker = CitationTracker::new();
        let source = Source {
            id: Uuid::new_v4(),
            title: "Test Article".to_string(),
            url: Some("https://example.com".to_string()),
            document_type: "webpage".to_string(),
            metadata: HashMap::new(),
        };

        let citation = Citation {
            id: Uuid::new_v4(),
            source,
            text_range: TextRange { start: 0, end: 10, length: 10 },
            confidence: 0.8,
            citation_type: CitationType::Paraphrase,
            relevance_score: 0.7,
            supporting_text: None,
        };

        let formatted = tracker.format_citation(&citation);
        assert!(formatted.contains("Test Article"));
    }
}