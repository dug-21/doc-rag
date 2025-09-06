//! Citation tracking and source attribution system for response generation

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, instrument, warn, info, error};
use uuid::Uuid;
use async_trait::async_trait;
use std::sync::Arc;

/// Citation tracker for managing source attribution with FACT integration
#[derive(Clone)]
pub struct CitationTracker {
    /// Configuration for citation processing
    config: CitationConfig,
    
    /// Cache of processed sources
    source_cache: HashMap<Uuid, Source>,
    
    /// Deduplication tracking
    deduplication_index: HashMap<String, Uuid>,
    
    /// FACT citation manager for caching and optimization
    fact_manager: Option<Arc<dyn FACTCitationProvider>>,
    
    /// Quality assurance system
    quality_assurance: CitationQualityAssurance,
    
    /// Coverage analyzer
    coverage_analyzer: CitationCoverageAnalyzer,
}

impl std::fmt::Debug for CitationTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CitationTracker")
            .field("config", &self.config)
            .field("source_cache", &self.source_cache)
            .field("deduplication_index", &self.deduplication_index)
            .field("fact_manager", &self.fact_manager.is_some())
            .field("quality_assurance", &self.quality_assurance)
            .field("coverage_analyzer", &self.coverage_analyzer)
            .finish()
    }
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
    
    /// Require 100% citation coverage
    pub require_100_percent_coverage: bool,
    
    /// Enable FACT integration for caching
    pub enable_fact_integration: bool,
    
    /// Citation quality threshold (0.0-1.0)
    pub citation_quality_threshold: f64,
    
    /// Maximum citations per paragraph
    pub max_citations_per_paragraph: usize,
    
    /// Require supporting text for direct quotes
    pub require_supporting_text_for_quotes: bool,
    
    /// Enable advanced deduplication
    pub enable_advanced_deduplication: bool,
    
    /// Enable quality assurance
    pub enable_quality_assurance: bool,
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
            require_100_percent_coverage: false,
            enable_fact_integration: false,
            citation_quality_threshold: 0.7,
            max_citations_per_paragraph: 3,
            require_supporting_text_for_quotes: true,
            enable_advanced_deduplication: true,
            enable_quality_assurance: true,
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
            fact_manager: None,
            quality_assurance: CitationQualityAssurance::new(),
            coverage_analyzer: CitationCoverageAnalyzer::new(),
        }
    }
    
    /// Ensure complete citation coverage for all claims in response
    pub async fn ensure_complete_citation_coverage(
        &mut self,
        response: &crate::IntermediateResponse,
    ) -> Result<CitationCoverageReport> {
        // Analyze the content for factual claims requiring citations
        let requirement_analysis = self.coverage_analyzer
            .analyze_citation_requirements(&response.content).await?;
        
        // Generate citations for identified claims
        let mut citations = Vec::new();
        let mut uncited_claims = Vec::new();
        
        for requirement in &requirement_analysis.required_citations {
            // Try to find supporting source for this claim
            if let Some(source_ref) = response.source_references.first() {
                if let Some(source) = self.source_cache.get(source_ref) {
                    let citation = Citation {
                        id: Uuid::new_v4(),
                        source: source.clone(),
                        text_range: requirement.text_range.clone(),
                        confidence: 0.8, // Default confidence
                        citation_type: CitationType::SupportingEvidence,
                        relevance_score: 0.8,
                        supporting_text: Some(requirement.claim_text.clone()),
                    };
                    citations.push(citation);
                } else {
                    // No supporting source found - mark as uncited
                    uncited_claims.push(UncitedClaim {
                        text: requirement.claim_text.clone(),
                        position: requirement.text_range.start,
                        claim_type: ClaimType::Factual,
                        confidence: 0.8,
                        citation_necessity: requirement.citation_necessity.clone(),
                    });
                }
            }
        }
        
        let coverage_percentage = if requirement_analysis.factual_claims_count == 0 {
            100.0
        } else {
            (citations.len() as f64 / requirement_analysis.factual_claims_count as f64 * 100.0).min(100.0)
        };
        
        Ok(CitationCoverageReport {
            coverage_percentage,
            factual_claims_detected: requirement_analysis.factual_claims_count,
            citations,
            uncited_claims,
            coverage_gaps: Vec::new(), // Simplified for now
            quality_score: 0.8, // Default quality score
        })
    }
    
    /// Advanced citation deduplication with semantic analysis
    pub async fn deduplicate_citations_advanced(&mut self, citations: Vec<Citation>) -> Result<Vec<Citation>> {
        if !self.config.enable_advanced_deduplication {
            return self.deduplicate_citations(citations).await;
        }
        
        let mut deduplicated = Vec::new();
        let mut processed_sources = HashSet::new();
        
        // Group citations by source
        let mut source_citations: HashMap<Uuid, Vec<Citation>> = HashMap::new();
        for citation in citations {
            source_citations
                .entry(citation.source.id)
                .or_insert_with(Vec::new)
                .push(citation);
        }
        
        // Process each source group
        for (source_id, mut source_cites) in source_citations {
            if processed_sources.contains(&source_id) {
                continue;
            }
            
            // Sort by confidence and relevance
            source_cites.sort_by(|a, b| {
                let score_a = (a.confidence + a.relevance_score) / 2.0;
                let score_b = (b.confidence + b.relevance_score) / 2.0;
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            // Keep best citations, merge similar ones
            let mut kept_citations = Vec::new();
            
            for citation in source_cites {
                let should_merge = kept_citations.iter().any(|kept: &Citation| {
                    self.should_merge_citations(kept, &citation)
                });
                
                if !should_merge && kept_citations.len() < 2 {
                    kept_citations.push(citation);
                }
            }
            
            deduplicated.extend(kept_citations);
            processed_sources.insert(source_id);
        }
        
        Ok(deduplicated)
    }
    
    /// Check if two citations should be merged
    fn should_merge_citations(&self, citation1: &Citation, citation2: &Citation) -> bool {
        // Same source and overlapping text ranges
        if citation1.source.id != citation2.source.id {
            return false;
        }
        
        let range1 = &citation1.text_range;
        let range2 = &citation2.text_range;
        
        // Check for overlap
        let overlap_start = range1.start.max(range2.start);
        let overlap_end = range1.end.min(range2.end);
        
        if overlap_start < overlap_end {
            let overlap_size = overlap_end - overlap_start;
            let min_size = range1.length.min(range2.length);
            
            // Merge if overlap is more than 50% of smaller citation
            overlap_size > min_size / 2
        } else {
            false
        }
    }
    
    /// Build citation chain for source attribution tracking
    pub async fn build_citation_chain(&self, citing_source: &Source, cited_source: &Source) -> Result<CitationChain> {
        let mut chain = CitationChain {
            primary_source: cited_source.clone(),
            citing_sources: vec![citing_source.clone()],
            levels: 2,
            attribution_complete: true,
            credibility_score: 0.0,
        };
        
        // Calculate credibility based on source types and metadata
        let primary_credibility = self.calculate_source_credibility(&chain.primary_source).await?;
        let citing_credibility = self.calculate_source_credibility(citing_source).await?;
        
        // Chain credibility is the minimum of all sources in the chain
        chain.credibility_score = primary_credibility.min(citing_credibility);
        
        // Verify attribution is complete by checking metadata
        if let Some(cites_field) = citing_source.metadata.get("cites") {
            chain.attribution_complete = cites_field.contains(&cited_source.id.to_string());
        } else {
            chain.attribution_complete = false;
        }
        
        Ok(chain)
    }
    
    /// Calculate source credibility score
    async fn calculate_source_credibility(&self, source: &Source) -> Result<f64> {
        let base_credibility = self.calculate_credibility(source).await?;
        
        // Additional credibility factors
        let mut credibility_factors = vec![base_credibility];
        
        // Check for peer review
        if source.metadata.get("peer_reviewed") == Some(&"true".to_string()) {
            credibility_factors.push(0.2);
        }
        
        // Check citation count
        if let Some(citation_count_str) = source.metadata.get("citation_count") {
            if let Ok(count) = citation_count_str.parse::<u32>() {
                let citation_factor = (count as f64).log10() / 10.0;
                credibility_factors.push(citation_factor.min(0.3));
            }
        }
        
        Ok(credibility_factors.iter().sum::<f64>() / credibility_factors.len() as f64)
    }
    
    /// Comprehensive citation validation
    pub async fn validate_citations_comprehensive(&self, citations: Vec<Citation>) -> Result<CitationValidationResult> {
        let mut valid_citations = 0;
        let mut invalid_citations = Vec::new();
        
        for citation in &citations {
            let mut failure_reasons = Vec::new();
            let mut recommendations = Vec::new();
            
            // Validate confidence threshold
            if citation.confidence < self.config.min_confidence {
                failure_reasons.push("Confidence below minimum threshold".to_string());
                recommendations.push(format!("Increase confidence above {}", self.config.min_confidence));
            }
            
            // Validate supporting text for direct quotes
            if self.config.require_supporting_text_for_quotes && 
               citation.citation_type == CitationType::DirectQuote && 
               citation.supporting_text.is_none() {
                failure_reasons.push("Missing supporting text for direct quote".to_string());
                recommendations.push("Add supporting text with exact quote from source".to_string());
            }
            
            // Validate source quality
            if citation.source.title.is_empty() {
                failure_reasons.push("Source title is empty".to_string());
                recommendations.push("Provide descriptive source title".to_string());
            }
            
            // Validate text range
            if citation.text_range.start >= citation.text_range.end {
                failure_reasons.push("Invalid text range".to_string());
                recommendations.push("Ensure start position is less than end position".to_string());
            }
            
            if failure_reasons.is_empty() {
                valid_citations += 1;
            } else {
                let severity = if failure_reasons.len() > 2 {
                    ValidationSeverity::Critical
                } else if failure_reasons.iter().any(|r| r.contains("confidence")) {
                    ValidationSeverity::Error
                } else {
                    ValidationSeverity::Warning
                };
                
                invalid_citations.push(CitationValidationFailure {
                    citation: citation.clone(),
                    failure_reasons,
                    severity,
                    recommendations,
                });
            }
        }
        
        let overall_validation_score = valid_citations as f64 / citations.len() as f64;
        let validation_passed = overall_validation_score >= 0.8;
        
        Ok(CitationValidationResult {
            total_citations: citations.len(),
            valid_citations,
            invalid_citations,
            overall_validation_score,
            validation_passed,
        })
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

    /// Assess author authority based on real metrics
    async fn assess_author_authority(&self, author: &str) -> Result<f64> {
        let mut authority_score = 0.0;
        let mut score_components = 0;
        
        // Check for academic credentials indicators
        let academic_indicators = [
            "Ph.D.", "PhD", "Dr.", "Professor", "Prof.", "M.D.", "MD", 
            "Ph.D", "D.Phil", "Sc.D", "Ed.D"
        ];
        
        for indicator in &academic_indicators {
            if author.contains(indicator) {
                authority_score += 0.3;
                score_components += 1;
                break;
            }
        }
        
        // Check for institutional affiliations
        let institution_indicators = [
            "University", "Institute", "College", "Academy", "Research Center",
            "Laboratory", "Foundation", "Corporation", "Ltd", "Inc"
        ];
        
        for indicator in &institution_indicators {
            if author.contains(indicator) {
                authority_score += 0.2;
                score_components += 1;
                break;
            }
        }
        
        // Assess based on name structure and formatting
        let parts: Vec<&str> = author.split_whitespace().collect();
        if parts.len() >= 2 {
            // Full name suggests more authoritative source
            authority_score += 0.1;
            score_components += 1;
            
            // Check for middle initials (common in academic writing)
            if parts.len() >= 3 && parts[1].len() <= 2 && parts[1].ends_with('.') {
                authority_score += 0.1;
                score_components += 1;
            }
        }
        
        // Check for multiple authors (collaboration often indicates authority)
        if author.contains(" and ") || author.contains(", ") {
            authority_score += 0.15;
            score_components += 1;
        }
        
        // Normalize score based on components found
        let final_score = if score_components > 0 {
            (authority_score / score_components as f64).max(0.1).min(1.0)
        } else {
            0.2 // Base score for any named author
        };
        
        Ok(final_score)
    }

    /// Assess publication authority based on real publication indicators
    async fn assess_publication_authority(&self, publication: &str) -> Result<f64> {
        let mut authority_score = 0.0;
        let mut score_components = 0;
        
        // High-authority publication types
        let high_authority_pubs = [
            "Nature", "Science", "Cell", "The Lancet", "NEJM", "PNAS",
            "Journal of", "IEEE", "ACM", "Proceedings of", "Annual Review"
        ];
        
        for pub_type in &high_authority_pubs {
            if publication.to_lowercase().contains(&pub_type.to_lowercase()) {
                authority_score += 0.8;
                score_components += 1;
                break;
            }
        }
        
        // Medium-authority publication indicators
        let medium_authority_indicators = [
            "International", "European", "American", "British", "Conference",
            "Symposium", "Workshop", "Review", "Letters", "Communications"
        ];
        
        for indicator in &medium_authority_indicators {
            if publication.to_lowercase().contains(&indicator.to_lowercase()) {
                authority_score += 0.6;
                score_components += 1;
                break;
            }
        }
        
        // Academic/research publication indicators
        let academic_indicators = [
            "University", "Press", "Academic", "Research", "Institute",
            "Laboratory", "Department", "School of"
        ];
        
        for indicator in &academic_indicators {
            if publication.to_lowercase().contains(&indicator.to_lowercase()) {
                authority_score += 0.7;
                score_components += 1;
                break;
            }
        }
        
        // Check for volume/issue numbers (indicates peer-reviewed journal)
        if publication.to_lowercase().contains("vol") || publication.to_lowercase().contains("issue") {
            authority_score += 0.4;
            score_components += 1;
        }
        
        // Check for DOI (indicates formal publication)
        if publication.to_lowercase().contains("doi") || publication.contains("10.") {
            authority_score += 0.3;
            score_components += 1;
        }
        
        // Check for recent years (more relevant)
        if publication.contains("2023") || publication.contains("2024") {
            authority_score += 0.2;
            score_components += 1;
        } else if publication.contains("202") { // Any 2020s year
            authority_score += 0.1;
            score_components += 1;
        }
        
        // Normalize score
        let final_score = if score_components > 0 {
            (authority_score / score_components as f64).max(0.1).min(1.0)
        } else {
            0.3 // Base score for any publication
        };
        
        Ok(final_score)
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

/// FACT Citation Provider trait for caching and optimization
#[async_trait]
pub trait FACTCitationProvider: Send + Sync + std::fmt::Debug {
    async fn get_cached_citations(&self, key: &str) -> Result<Option<Vec<Citation>>>;
    async fn store_citations(&self, key: &str, citations: &[Citation]) -> Result<()>;
    async fn validate_citation_quality(&self, citation: &Citation) -> Result<CitationQualityMetrics>;
    async fn deduplicate_citations(&self, citations: Vec<Citation>) -> Result<Vec<Citation>>;
    async fn optimize_citation_chain(&self, chain: &CitationChain) -> Result<CitationChain>;
}

/// Citation quality assurance system
#[derive(Debug, Clone)]
pub struct CitationQualityAssurance {
    config: CitationConfig,
    quality_calculator: CitationQualityCalculator,
}

/// Citation coverage analyzer for 100% attribution
#[derive(Debug, Clone)]
pub struct CitationCoverageAnalyzer {
    factual_claim_patterns: Vec<String>,
    statistical_patterns: Vec<String>,
    evidence_patterns: Vec<String>,
}

/// Citation chain for tracking source attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationChain {
    pub primary_source: Source,
    pub citing_sources: Vec<Source>,
    pub levels: usize,
    pub attribution_complete: bool,
    pub credibility_score: f64,
}

/// Citation quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationQualityMetrics {
    pub overall_quality_score: f64,
    pub source_authority_score: f64,
    pub recency_score: f64,
    pub relevance_score: f64,
    pub completeness_score: f64,
    pub peer_review_factor: f64,
    pub impact_factor: f64,
    pub citation_count_factor: f64,
    pub author_credibility_factor: f64,
    pub passed_quality_threshold: bool,
}

/// Citation validation result
#[derive(Debug, Clone)]
pub struct CitationValidationResult {
    pub total_citations: usize,
    pub valid_citations: usize,
    pub invalid_citations: Vec<CitationValidationFailure>,
    pub overall_validation_score: f64,
    pub validation_passed: bool,
}

/// Citation validation failure details
#[derive(Debug, Clone)]
pub struct CitationValidationFailure {
    pub citation: Citation,
    pub failure_reasons: Vec<String>,
    pub severity: ValidationSeverity,
    pub recommendations: Vec<String>,
}

/// Validation severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationSeverity {
    Warning,
    Error,
    Critical,
}

/// Citation coverage report
#[derive(Debug, Clone)]
pub struct CitationCoverageReport {
    pub coverage_percentage: f64,
    pub factual_claims_detected: usize,
    pub citations: Vec<Citation>,
    pub uncited_claims: Vec<UncitedClaim>,
    pub coverage_gaps: Vec<CoverageGap>,
    pub quality_score: f64,
}

/// Uncited factual claim
#[derive(Debug, Clone)]
pub struct UncitedClaim {
    pub text: String,
    pub position: usize,
    pub claim_type: ClaimType,
    pub confidence: f64,
    pub citation_necessity: CitationNecessity,
}

/// Coverage gap in citations
#[derive(Debug, Clone)]
pub struct CoverageGap {
    pub text_range: TextRange,
    pub gap_type: GapType,
    pub severity: ValidationSeverity,
    pub recommendations: Vec<String>,
}

/// Types of factual claims
#[derive(Debug, Clone, PartialEq)]
pub enum ClaimType {
    Statistical,
    Factual,
    Research,
    Expert,
    Historical,
    Technical,
}

/// Citation necessity levels
#[derive(Debug, Clone, PartialEq)]
pub enum CitationNecessity {
    Required,
    Recommended,
    Optional,
}

/// Types of coverage gaps
#[derive(Debug, Clone, PartialEq)]
pub enum GapType {
    MissingCitation,
    LowQualitySource,
    InsufficientEvidence,
    OutdatedSource,
    ConflictingEvidence,
}

/// Citation quality calculator
#[derive(Debug, Clone)]
pub struct CitationQualityCalculator {
    authority_weights: HashMap<String, f64>,
    recency_weights: HashMap<String, f64>,
    completeness_factors: Vec<String>,
}

/// Citation requirement analysis
#[derive(Debug, Clone)]
pub struct CitationRequirementAnalysis {
    pub factual_claims_count: usize,
    pub opinion_statements_count: usize,
    pub required_citations: Vec<CitationRequirement>,
    pub recommended_citations: Vec<CitationRequirement>,
}

/// Individual citation requirement
#[derive(Debug, Clone)]
pub struct CitationRequirement {
    pub claim_text: String,
    pub text_range: TextRange,
    pub citation_necessity: CitationNecessity,
    pub confidence_threshold: f64,
    pub recommended_source_types: Vec<String>,
}

/// Comprehensive citation system result
#[derive(Debug, Clone)]
pub struct ComprehensiveCitationResult {
    pub coverage_percentage: f64,
    pub quality_score: f64,
    pub final_citations: Vec<Citation>,
    pub fact_cache_utilized: bool,
    pub validation_passed: bool,
    pub quality_metrics: CitationQualityMetrics,
    pub coverage_report: CitationCoverageReport,
}

/// Comprehensive citation system integrating all components
pub struct ComprehensiveCitationSystem {
    tracker: CitationTracker,
    quality_assurance: CitationQualityAssurance,
    coverage_analyzer: CitationCoverageAnalyzer,
    fact_manager: Option<Arc<dyn FACTCitationProvider>>,
}

impl std::fmt::Debug for ComprehensiveCitationSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComprehensiveCitationSystem")
            .field("tracker", &self.tracker)
            .field("quality_assurance", &self.quality_assurance)
            .field("coverage_analyzer", &self.coverage_analyzer)
            .field("fact_manager", &self.fact_manager.is_some())
            .finish()
    }
}

/// FACT Citation Manager implementation
pub struct FACTCitationManager {
    provider: Box<dyn FACTCitationProvider>,
    cache_enabled: bool,
}

impl std::fmt::Debug for FACTCitationManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FACTCitationManager")
            .field("provider", &"<dyn FACTCitationProvider>")
            .field("cache_enabled", &self.cache_enabled)
            .finish()
    }
}

impl Default for CitationTracker {
    fn default() -> Self {
        Self::new()
    }
}

// Implementation of new citation system components
impl CitationQualityAssurance {
    pub fn new() -> Self {
        Self {
            config: CitationConfig::default(),
            quality_calculator: CitationQualityCalculator::new(),
        }
    }
    
    pub async fn validate_citation_quality(&mut self, citation: &Citation) -> Result<CitationQualityMetrics> {
        self.quality_calculator.calculate_comprehensive_quality(citation).await
    }
}

impl CitationCoverageAnalyzer {
    pub fn new() -> Self {
        Self {
            factual_claim_patterns: vec![
                "according to".to_string(),
                "studies show".to_string(),
                "research indicates".to_string(),
                "data shows".to_string(),
                "statistics reveal".to_string(),
                "\\d+%".to_string(), // Percentage patterns
                "\\d+\\.\\d+".to_string(), // Decimal numbers
            ],
            statistical_patterns: vec![
                "\\d+%".to_string(),
                "\\d+ percent".to_string(),
                "ratio of".to_string(),
                "on average".to_string(),
            ],
            evidence_patterns: vec![
                "evidence suggests".to_string(),
                "findings indicate".to_string(),
                "results show".to_string(),
                "analysis reveals".to_string(),
            ],
        }
    }
    
    pub async fn analyze_citation_requirements(&mut self, content: &str) -> Result<CitationRequirementAnalysis> {
        let mut factual_claims = 0;
        let mut opinion_statements = 0;
        let mut required_citations = Vec::new();
        let mut recommended_citations = Vec::new();
        
        // Simple sentence-based analysis
        let sentences: Vec<&str> = content.split('.').collect();
        
        for (i, sentence) in sentences.iter().enumerate() {
            let sentence = sentence.trim();
            if sentence.is_empty() { continue; }
            
            let start = content.find(sentence).unwrap_or(0);
            let end = start + sentence.len();
            
            // Check for factual claims
            let is_factual = self.contains_factual_indicators(sentence);
            let is_statistical = self.contains_statistical_indicators(sentence);
            let is_opinion = self.contains_opinion_indicators(sentence);
            
            if is_factual || is_statistical {
                factual_claims += 1;
                required_citations.push(CitationRequirement {
                    claim_text: sentence.to_string(),
                    text_range: TextRange { start, end, length: sentence.len() },
                    citation_necessity: CitationNecessity::Required,
                    confidence_threshold: if is_statistical { 0.9 } else { 0.8 },
                    recommended_source_types: vec!["academic".to_string(), "government".to_string()],
                });
            } else if is_opinion {
                opinion_statements += 1;
            }
        }
        
        Ok(CitationRequirementAnalysis {
            factual_claims_count: factual_claims,
            opinion_statements_count: opinion_statements,
            required_citations,
            recommended_citations,
        })
    }
    
    fn contains_factual_indicators(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        self.factual_claim_patterns.iter().any(|pattern| {
            if pattern.contains("\\") {
                // Regex pattern - simplified check for now
                text_lower.contains(&pattern.replace("\\d+", "[0-9]").replace("\\", ""))
            } else {
                text_lower.contains(pattern)
            }
        })
    }
    
    fn contains_statistical_indicators(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        self.statistical_patterns.iter().any(|pattern| {
            if pattern.contains("\\") {
                // Simplified regex check
                text_lower.contains("%") || text_lower.contains("percent")
            } else {
                text_lower.contains(pattern)
            }
        })
    }
    
    fn contains_opinion_indicators(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        text_lower.contains("i think") || 
        text_lower.contains("in my opinion") ||
        text_lower.contains("i believe") ||
        text_lower.contains("personally") ||
        text_lower.contains("it seems")
    }
}

impl CitationQualityCalculator {
    pub fn new() -> Self {
        let mut authority_weights = HashMap::new();
        authority_weights.insert("academic".to_string(), 0.9);
        authority_weights.insert("government".to_string(), 0.85);
        authority_weights.insert("news".to_string(), 0.7);
        authority_weights.insert("blog".to_string(), 0.4);
        
        let mut recency_weights = HashMap::new();
        recency_weights.insert("2024".to_string(), 1.0);
        recency_weights.insert("2023".to_string(), 0.9);
        recency_weights.insert("2022".to_string(), 0.8);
        
        Self {
            authority_weights,
            recency_weights,
            completeness_factors: vec![
                "title".to_string(),
                "author".to_string(),
                "publication_date".to_string(),
                "supporting_text".to_string(),
            ],
        }
    }
    
    pub async fn calculate_comprehensive_quality(&self, citation: &Citation) -> Result<CitationQualityMetrics> {
        // Source authority score
        let authority_score = self.authority_weights
            .get(&citation.source.document_type)
            .copied()
            .unwrap_or(0.5);
            
        // Recency score from metadata
        let recency_score = citation.source.metadata
            .get("publication_year")
            .and_then(|year| self.recency_weights.get(year))
            .copied()
            .unwrap_or(0.5);
            
        // Relevance score from citation
        let relevance_score = citation.relevance_score;
        
        // Completeness score based on available metadata
        let completeness_score = self.completeness_factors.iter()
            .map(|factor| {
                match factor.as_str() {
                    "title" => if !citation.source.title.is_empty() { 1.0 } else { 0.0 },
                    "author" => if citation.source.metadata.contains_key("author") { 1.0 } else { 0.0 },
                    "publication_date" => if citation.source.metadata.contains_key("publication_date") { 1.0 } else { 0.0 },
                    "supporting_text" => if citation.supporting_text.is_some() { 1.0 } else { 0.0 },
                    _ => 0.5,
                }
            })
            .fold(0.0, |acc, score| acc + score) / self.completeness_factors.len() as f64;
            
        // Additional quality factors
        let peer_review_factor = citation.source.metadata
            .get("peer_reviewed")
            .map(|pr| if pr == "true" { 0.2 } else { 0.0 })
            .unwrap_or(0.0);
            
        let impact_factor = citation.source.metadata
            .get("impact_factor")
            .and_then(|if_str| if_str.parse::<f64>().ok())
            .map(|if_val| (if_val / 10.0).min(0.2))
            .unwrap_or(0.0);
            
        let citation_count_factor = citation.source.metadata
            .get("citation_count")
            .and_then(|cc_str| cc_str.parse::<u32>().ok())
            .map(|cc| (cc as f64).ln() / 100.0)
            .unwrap_or(0.0)
            .min(0.2);
            
        let author_credibility_factor = citation.source.metadata
            .get("author_h_index")
            .and_then(|h_str| h_str.parse::<u32>().ok())
            .map(|h_index| (h_index as f64) / 100.0)
            .unwrap_or(0.0)
            .min(0.2);
            
        // Calculate overall quality score
        let overall_quality_score = (authority_score * 0.3) +
                                  (recency_score * 0.2) +
                                  (relevance_score * 0.2) +
                                  (completeness_score * 0.2) +
                                  peer_review_factor +
                                  impact_factor +
                                  citation_count_factor +
                                  author_credibility_factor;
                                  
        let passed_quality_threshold = overall_quality_score >= 0.7; // Default threshold
        
        Ok(CitationQualityMetrics {
            overall_quality_score,
            source_authority_score: authority_score,
            recency_score,
            relevance_score,
            completeness_score,
            peer_review_factor,
            impact_factor,
            citation_count_factor,
            author_credibility_factor,
            passed_quality_threshold,
        })
    }
}

impl CitationChain {
    pub fn is_valid(&self) -> bool {
        self.attribution_complete && self.credibility_score > 0.5
    }
}

impl FACTCitationManager {
    pub fn new(provider: Box<dyn FACTCitationProvider>) -> Self {
        Self {
            provider,
            cache_enabled: true,
        }
    }
    
    pub async fn get_or_generate_citations(&self, request: &crate::GenerationRequest, fallback_citations: Vec<Citation>) -> Result<Vec<Citation>> {
        if !self.cache_enabled {
            return Ok(fallback_citations);
        }
        
        let cache_key = format!("{:x}", md5::compute(&request.query));
        
        // Try to get cached citations
        if let Ok(Some(cached_citations)) = self.provider.get_cached_citations(&cache_key).await {
            info!("Retrieved {} cached citations for query", cached_citations.len());
            return Ok(cached_citations);
        }
        
        // Store generated citations for future use
        if let Err(e) = self.provider.store_citations(&cache_key, &fallback_citations).await {
            warn!("Failed to cache citations: {}", e);
        }
        
        Ok(fallback_citations)
    }
}

impl ComprehensiveCitationSystem {
    pub async fn new(config: CitationConfig) -> Result<Self> {
        let tracker = CitationTracker::with_config(config.clone());
        let quality_assurance = CitationQualityAssurance::new();
        let coverage_analyzer = CitationCoverageAnalyzer::new();
        
        Ok(Self {
            tracker,
            quality_assurance,
            coverage_analyzer,
            fact_manager: None, // Would be injected in real implementation
        })
    }
    
    pub async fn process_comprehensive_citations(
        &mut self,
        request: &crate::GenerationRequest,
        response: &crate::IntermediateResponse,
    ) -> Result<ComprehensiveCitationResult> {
        // Analyze citation requirements
        let requirement_analysis = self.coverage_analyzer
            .analyze_citation_requirements(&response.content).await?;
        
        // Generate citations
        let initial_citations = self.tracker
            .process_citations(response, &request.context).await?;
            
        // Quality assurance validation
        let mut validated_citations = Vec::new();
        let mut quality_metrics = CitationQualityMetrics {
            overall_quality_score: 0.0,
            source_authority_score: 0.0,
            recency_score: 0.0,
            relevance_score: 0.0,
            completeness_score: 0.0,
            peer_review_factor: 0.0,
            impact_factor: 0.0,
            citation_count_factor: 0.0,
            author_credibility_factor: 0.0,
            passed_quality_threshold: false,
        };
        
        for citation in initial_citations {
            let citation_quality = self.quality_assurance.validate_citation_quality(&citation).await?;
            if citation_quality.passed_quality_threshold {
                validated_citations.push(citation);
            }
            
            // Update overall quality metrics (simplified averaging)
            quality_metrics.overall_quality_score += citation_quality.overall_quality_score;
        }
        
        if !validated_citations.is_empty() {
            quality_metrics.overall_quality_score /= validated_citations.len() as f64;
            quality_metrics.passed_quality_threshold = quality_metrics.overall_quality_score >= 0.7;
        }
        
        // Calculate coverage
        let coverage_percentage = if requirement_analysis.factual_claims_count == 0 {
            100.0
        } else {
            (validated_citations.len() as f64 / requirement_analysis.factual_claims_count as f64 * 100.0).min(100.0)
        };
        
        // Create coverage report
        let coverage_report = CitationCoverageReport {
            coverage_percentage,
            factual_claims_detected: requirement_analysis.factual_claims_count,
            citations: validated_citations.clone(),
            uncited_claims: Vec::new(), // Simplified for now
            coverage_gaps: Vec::new(),  // Simplified for now
            quality_score: quality_metrics.overall_quality_score,
        };
        
        Ok(ComprehensiveCitationResult {
            coverage_percentage,
            quality_score: quality_metrics.overall_quality_score,
            final_citations: validated_citations,
            fact_cache_utilized: false, // Would be true with real FACT integration
            validation_passed: quality_metrics.passed_quality_threshold,
            quality_metrics,
            coverage_report,
        })
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