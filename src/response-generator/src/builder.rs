//! Response builder for constructing high-quality responses with context assembly

use crate::{
    error::{Result, ResponseError},
    GenerationRequest, ContextChunk, IntermediateResponse,
    citation::Source,
};
use serde_json;
use std::collections::{HashMap, HashSet};
use tokio::time::{Duration, Instant};
use tracing::{debug, instrument, warn};
use uuid::Uuid;

/// Response builder for assembling responses from context
#[derive(Debug, Clone)]
pub struct ResponseBuilder {
    /// Request being processed
    request: GenerationRequest,
    
    /// Optimized query (may differ from original)
    optimized_query: Option<String>,
    
    /// Current response content
    content: String,
    
    /// Confidence factors collected during building
    confidence_factors: Vec<f64>,
    
    /// Source references used in response
    source_references: Vec<Uuid>,
    
    /// Context chunks ranked by relevance
    ranked_context: Vec<RankedContext>,
    
    /// Warnings accumulated during building
    warnings: Vec<String>,
    
    /// Metadata for the response (including FACT analysis)
    metadata: HashMap<String, serde_json::Value>,
    
    /// Content sections for structured building
    sections: Vec<ContentSection>,
}

/// Context chunk with relevance ranking
#[derive(Debug, Clone)]
struct RankedContext {
    chunk: ContextChunk,
    relevance_score: f64,
    quality_score: f64,
    usage_priority: f64,
}

/// Content section for structured response building
#[derive(Debug, Clone)]
struct ContentSection {
    /// Section title/type
    section_type: SectionType,
    
    /// Content for this section
    content: String,
    
    /// Supporting sources
    sources: Vec<Uuid>,
    
    /// Confidence in this section
    confidence: f64,
}

/// Types of content sections
#[derive(Debug, Clone, PartialEq)]
enum SectionType {
    /// Direct answer to the query
    DirectAnswer,
    
    /// Supporting evidence
    SupportingEvidence,
    
    /// Context background
    Background,
    
    /// Related information
    RelatedInfo,
    
    /// Caveats or limitations
    Caveats,
}

/// Response building strategy
#[derive(Debug, Clone)]
pub enum BuildingStrategy {
    /// Balanced approach using multiple sources
    Balanced,
    
    /// Focus on highest confidence sources
    HighConfidence,
    
    /// Comprehensive coverage of all relevant sources
    Comprehensive,
    
    /// Concise response from most relevant sources
    Concise,
}

/// Content quality metrics
#[derive(Debug, Clone)]
struct QualityMetrics {
    relevance: f64,
    coherence: f64,
    completeness: f64,
    accuracy_indicators: f64,
    source_diversity: f64,
}

impl ResponseBuilder {
    /// Create a new response builder
    pub fn new(request: GenerationRequest) -> Self {
        Self {
            request,
            optimized_query: None,
            content: String::new(),
            confidence_factors: Vec::new(),
            source_references: Vec::new(),
            ranked_context: Vec::new(),
            warnings: Vec::new(),
            metadata: HashMap::new(),
            sections: Vec::new(),
        }
    }

    /// Set building strategy
    pub fn with_strategy(mut self, strategy: BuildingStrategy) -> Self {
        self.metadata.insert("building_strategy".to_string(), 
                           serde_json::json!(format!("{:?}", strategy)));
        self
    }
    
    /// Set optimized query from FACT preprocessing
    pub fn set_optimized_query(&mut self, optimized_query: String) {
        self.optimized_query = Some(optimized_query);
    }
    
    /// Get optimized query or original query
    pub fn get_query(&self) -> &str {
        self.optimized_query.as_ref().unwrap_or(&self.request.query)
    }
    
    /// Set metadata value (used by FACT preprocessing)
    pub fn set_metadata(&mut self, key: &str, value: serde_json::Value) {
        self.metadata.insert(key.to_string(), value);
    }
    
    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Rank and prepare context for response building
    #[instrument(skip(self))]
    pub async fn prepare_context(&mut self) -> Result<()> {
        debug!("Preparing context from {} chunks", self.request.context.len());
        
        if self.request.context.is_empty() {
            self.warnings.push("No context provided for response generation".to_string());
            return Ok(());
        }

        // Calculate quality scores for each context chunk
        for chunk in &self.request.context {
            let quality_score = self.calculate_quality_score(chunk).await?;
            let usage_priority = self.calculate_usage_priority(chunk, quality_score).await?;
            
            self.ranked_context.push(RankedContext {
                chunk: chunk.clone(),
                relevance_score: chunk.relevance_score,
                quality_score,
                usage_priority,
            });
        }

        // Sort by usage priority (highest first)
        self.ranked_context.sort_by(|a, b| b.usage_priority.partial_cmp(&a.usage_priority).unwrap_or(std::cmp::Ordering::Equal));
        
        // Filter out low-quality context
        let quality_threshold = 0.3;
        let initial_count = self.ranked_context.len();
        self.ranked_context.retain(|ctx| ctx.quality_score >= quality_threshold);
        
        if self.ranked_context.len() < initial_count {
            self.warnings.push(format!(
                "Filtered out {} low-quality context chunks",
                initial_count - self.ranked_context.len()
            ));
        }

        debug!("Prepared {} high-quality context chunks", self.ranked_context.len());
        Ok(())
    }

    /// Build response content from prepared context
    #[instrument(skip(self))]
    pub async fn build_content(&mut self) -> Result<()> {
        debug!("Building response content");
        
        if self.ranked_context.is_empty() {
            return Err(ResponseError::generation_failed(
                "No usable context available for response generation"
            ));
        }

        // Analyze query to determine response structure
        let query_analysis = self.analyze_query().await?;
        
        // Build sections based on query analysis
        self.build_direct_answer(&query_analysis).await?;
        self.build_supporting_evidence(&query_analysis).await?;
        
        if query_analysis.needs_background {
            self.build_background_section(&query_analysis).await?;
        }
        
        if query_analysis.needs_caveats {
            self.build_caveats_section(&query_analysis).await?;
        }

        // Assemble final content from sections
        self.assemble_final_content().await?;
        
        // Validate content quality
        let quality = self.assess_content_quality().await?;
        if quality.overall_score() < 0.7 {
            self.warnings.push(format!(
                "Response quality score {:.2} below target 0.7",
                quality.overall_score()
            ));
        }

        debug!("Built response content: {} characters", self.content.len());
        Ok(())
    }

    /// Add source attribution and citations
    #[instrument(skip(self))]
    pub async fn add_citations(&mut self) -> Result<()> {
        debug!("Adding citations to response");
        
        // Extract source references from sections
        let mut source_ids = HashSet::new();
        for section in &self.sections {
            source_ids.extend(&section.sources);
        }
        
        self.source_references = source_ids.into_iter().collect();
        
        // Add inline citations based on content
        self.add_inline_citations().await?;
        
        debug!("Added {} source references", self.source_references.len());
        Ok(())
    }

    /// Optimize response for target constraints
    #[instrument(skip(self))]
    pub async fn optimize(&mut self) -> Result<()> {
        debug!("Optimizing response");
        
        // Check length constraints
        if let Some(max_length) = self.request.max_length {
            if self.content.len() > max_length {
                self.trim_content(max_length).await?;
                self.warnings.push(format!(
                    "Response trimmed to {} characters to meet length constraint",
                    max_length
                ));
            }
        }

        // Optimize for coherence
        self.improve_coherence().await?;
        
        // Remove redundancy
        self.remove_redundancy().await?;
        
        debug!("Response optimization complete");
        Ok(())
    }

    /// Build the final response
    #[instrument(skip(self))]
    pub async fn build(mut self) -> Result<IntermediateResponse> {
        let start_time = Instant::now();
        
        // Execute building pipeline
        self.prepare_context().await?;
        self.build_content().await?;
        self.add_citations().await?;
        self.optimize().await?;
        
        // Calculate overall confidence
        let _overall_confidence = if self.confidence_factors.is_empty() {
            0.5 // Default neutral confidence
        } else {
            self.confidence_factors.iter().sum::<f64>() / self.confidence_factors.len() as f64
        };

        let build_duration = start_time.elapsed();
        if build_duration > Duration::from_millis(50) {
            warn!("Response building took {}ms", build_duration.as_millis());
        }

        Ok(IntermediateResponse {
            content: self.content,
            confidence_factors: self.confidence_factors,
            source_references: self.source_references,
            warnings: self.warnings,
        })
    }

    /// Calculate quality score for a context chunk
    async fn calculate_quality_score(&self, chunk: &ContextChunk) -> Result<f64> {
        let mut quality_factors = Vec::new();
        
        // Content length factor (not too short, not too long)
        let length_factor = self.calculate_length_factor(chunk.content.len());
        quality_factors.push(length_factor);
        
        // Source credibility (from metadata)
        let credibility_factor = chunk.metadata.get("credibility_score")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.5);
        quality_factors.push(credibility_factor);
        
        // Content complexity/informativeness
        let info_factor = self.calculate_informativeness(&chunk.content);
        quality_factors.push(info_factor);
        
        // Coherence with query
        let coherence_factor = self.calculate_query_coherence(&chunk.content).await?;
        quality_factors.push(coherence_factor);
        
        Ok(quality_factors.iter().sum::<f64>() / quality_factors.len() as f64)
    }

    /// Calculate usage priority combining relevance and quality
    async fn calculate_usage_priority(&self, chunk: &ContextChunk, quality_score: f64) -> Result<f64> {
        // Weight relevance more heavily than quality
        let relevance_weight = 0.7;
        let quality_weight = 0.3;
        
        Ok(chunk.relevance_score * relevance_weight + quality_score * quality_weight)
    }

    /// Analyze query to determine response requirements
    async fn analyze_query(&self) -> Result<QueryAnalysis> {
        let query = &self.request.query.to_lowercase();
        
        let mut analysis = QueryAnalysis {
            query_type: self.determine_query_type(query),
            needs_background: false,
            needs_caveats: false,
            expected_sections: Vec::new(),
            complexity_level: self.assess_query_complexity(query),
        };

        // Determine if background context is needed
        analysis.needs_background = query.contains("what is") || 
                                    query.contains("explain") ||
                                    query.contains("describe");

        // Determine if caveats are needed
        analysis.needs_caveats = query.contains("always") ||
                                  query.contains("never") ||
                                  query.contains("all") ||
                                  analysis.complexity_level > 0.7;

        Ok(analysis)
    }

    /// Build direct answer section
    async fn build_direct_answer(&mut self, _analysis: &QueryAnalysis) -> Result<()> {
        let mut content = String::new();
        let mut sources = Vec::new();
        let mut confidence_sum = 0.0;
        let mut source_count = 0;

        // Use top-ranked context for direct answer
        let top_contexts = self.ranked_context.iter().take(3);
        
        for ranked_ctx in top_contexts {
            let chunk = &ranked_ctx.chunk;
            
            // Extract most relevant sentences
            let relevant_sentences = self.extract_relevant_sentences(&chunk.content, &self.request.query).await?;
            
            for sentence in relevant_sentences {
                if !content.is_empty() {
                    content.push_str(" ");
                }
                content.push_str(&sentence);
                sources.push(chunk.source.id);
                confidence_sum += ranked_ctx.quality_score;
                source_count += 1;
            }
        }

        if !content.is_empty() {
            let section_confidence = if source_count > 0 {
                confidence_sum / source_count as f64
            } else {
                0.0
            };

            self.sections.push(ContentSection {
                section_type: SectionType::DirectAnswer,
                content,
                sources,
                confidence: section_confidence,
            });

            self.confidence_factors.push(section_confidence);
        }

        Ok(())
    }

    /// Build supporting evidence section
    async fn build_supporting_evidence(&mut self, _analysis: &QueryAnalysis) -> Result<()> {
        let mut content = String::new();
        let mut sources = Vec::new();
        let mut confidence_sum = 0.0;
        let mut source_count = 0;

        // Use middle-ranked context for supporting evidence
        let supporting_contexts = self.ranked_context.iter().skip(3).take(5);
        
        for ranked_ctx in supporting_contexts {
            let chunk = &ranked_ctx.chunk;
            
            // Extract supporting information
            let supporting_info = self.extract_supporting_information(&chunk.content).await?;
            
            if !supporting_info.is_empty() {
                if !content.is_empty() {
                    content.push_str(" ");
                }
                content.push_str(&supporting_info);
                sources.push(chunk.source.id);
                confidence_sum += ranked_ctx.quality_score;
                source_count += 1;
            }
        }

        if !content.is_empty() {
            let section_confidence = if source_count > 0 {
                confidence_sum / source_count as f64
            } else {
                0.0
            };

            self.sections.push(ContentSection {
                section_type: SectionType::SupportingEvidence,
                content,
                sources,
                confidence: section_confidence,
            });

            self.confidence_factors.push(section_confidence);
        }

        Ok(())
    }

    /// Build background section
    async fn build_background_section(&mut self, _analysis: &QueryAnalysis) -> Result<()> {
        // Implementation for background section building
        // This would extract contextual information that helps understand the main answer
        Ok(())
    }

    /// Build caveats section
    async fn build_caveats_section(&mut self, _analysis: &QueryAnalysis) -> Result<()> {
        // Implementation for caveats section building
        // This would identify limitations, exceptions, or important considerations
        Ok(())
    }

    /// Assemble final content from sections
    async fn assemble_final_content(&mut self) -> Result<()> {
        let mut final_content = String::new();
        
        // Order sections appropriately
        let section_order = [
            SectionType::DirectAnswer,
            SectionType::SupportingEvidence,
            SectionType::Background,
            SectionType::RelatedInfo,
            SectionType::Caveats,
        ];

        for section_type in &section_order {
            if let Some(section) = self.sections.iter().find(|s| &s.section_type == section_type) {
                if !final_content.is_empty() {
                    final_content.push_str("\n\n");
                }
                final_content.push_str(&section.content);
            }
        }

        self.content = final_content;
        Ok(())
    }

    /// Add inline citations to content
    async fn add_inline_citations(&mut self) -> Result<()> {
        // This would add citation markers like [1], [2] etc. to the content
        // and maintain mapping to source references
        Ok(())
    }

    /// Trim content to fit length constraints
    async fn trim_content(&mut self, max_length: usize) -> Result<()> {
        if self.content.len() <= max_length {
            return Ok(());
        }

        // Find the best place to cut while preserving meaning
        let sentences: Vec<&str> = self.content.split('.').collect();
        let mut trimmed = String::new();
        
        for sentence in sentences {
            let potential_length = trimmed.len() + sentence.len() + 1; // +1 for period
            if potential_length <= max_length {
                if !trimmed.is_empty() {
                    trimmed.push('.');
                }
                trimmed.push_str(sentence);
            } else {
                break;
            }
        }

        if !trimmed.ends_with('.') && trimmed.len() < max_length {
            trimmed.push('.');
        }

        self.content = trimmed;
        Ok(())
    }

    /// Improve content coherence
    async fn improve_coherence(&mut self) -> Result<()> {
        // This would analyze and improve the flow between sentences/paragraphs
        // For now, basic implementation ensuring proper sentence structure
        
        // Ensure sentences end with proper punctuation
        if !self.content.is_empty() && !self.content.ends_with('.') && !self.content.ends_with('!') && !self.content.ends_with('?') {
            self.content.push('.');
        }

        Ok(())
    }

    /// Remove redundant content
    async fn remove_redundancy(&mut self) -> Result<()> {
        // This would identify and remove duplicate or highly similar content
        // Basic implementation removes exact duplicate sentences
        
        let sentences: Vec<&str> = self.content.split('.').collect();
        let mut unique_sentences = Vec::new();
        let mut seen = HashSet::new();

        for sentence in sentences {
            let trimmed = sentence.trim();
            if !trimmed.is_empty() && seen.insert(trimmed.to_string()) {
                unique_sentences.push(trimmed);
            }
        }

        self.content = unique_sentences.join(". ");
        if !self.content.is_empty() && !self.content.ends_with('.') {
            self.content.push('.');
        }

        Ok(())
    }

    /// Assess overall content quality
    async fn assess_content_quality(&self) -> Result<QualityMetrics> {
        Ok(QualityMetrics {
            relevance: self.assess_relevance(),
            coherence: self.assess_coherence(),
            completeness: self.assess_completeness(),
            accuracy_indicators: self.assess_accuracy_indicators(),
            source_diversity: self.assess_source_diversity(),
        })
    }

    // Helper methods for quality assessment
    fn assess_relevance(&self) -> f64 {
        // Analyze how well the content addresses the query
        // Basic implementation based on keyword overlap
        let query_words: HashSet<String> = self.request.query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let content_words: HashSet<String> = self.content
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let intersection_count = query_words.intersection(&content_words).count();
        intersection_count as f64 / query_words.len().max(1) as f64
    }

    fn assess_coherence(&self) -> f64 {
        // Basic coherence assessment based on sentence structure
        let sentences: Vec<&str> = self.content.split('.').collect();
        if sentences.len() < 2 {
            return 0.8; // Single sentence is considered coherent
        }

        // Count transitions between sentences (very basic)
        let transition_words = ["however", "therefore", "additionally", "furthermore", "moreover"];
        let transition_count = transition_words.iter()
            .map(|word| self.content.to_lowercase().matches(word).count())
            .sum::<usize>();

        (transition_count as f64 / sentences.len() as f64).min(1.0)
    }

    fn assess_completeness(&self) -> f64 {
        // Basic completeness based on content length and section coverage
        let length_factor = (self.content.len() as f64 / 500.0).min(1.0); // Target ~500 chars
        let section_factor = self.sections.len() as f64 / 3.0; // Target 3 sections
        
        (length_factor + section_factor) / 2.0
    }

    fn assess_accuracy_indicators(&self) -> f64 {
        // Basic accuracy indicators based on source quality
        if self.confidence_factors.is_empty() {
            return 0.5;
        }
        
        self.confidence_factors.iter().sum::<f64>() / self.confidence_factors.len() as f64
    }

    fn assess_source_diversity(&self) -> f64 {
        // Assess diversity of sources used
        let unique_sources = self.source_references.len();
        let total_context = self.request.context.len().max(1);
        
        (unique_sources as f64 / total_context as f64).min(1.0)
    }

    // Additional helper methods
    fn calculate_length_factor(&self, length: usize) -> f64 {
        // Optimal length is around 200-2000 characters
        if length < 50 {
            return length as f64 / 50.0;
        } else if length <= 2000 {
            return 1.0;
        } else {
            return 2000.0 / length as f64;
        }
    }

    fn calculate_informativeness(&self, content: &str) -> f64 {
        // Basic informativeness based on unique words and sentence structure
        let words: HashSet<String> = content
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        
        let unique_word_ratio = words.len() as f64 / content.split_whitespace().count().max(1) as f64;
        unique_word_ratio.min(1.0)
    }

    async fn calculate_query_coherence(&self, _content: &str) -> Result<f64> {
        // Calculate how well content relates to the query
        Ok(self.assess_relevance()) // Simplified implementation
    }

    fn determine_query_type(&self, query: &str) -> QueryType {
        if query.contains("what") || query.contains("define") {
            QueryType::Definitional
        } else if query.contains("how") {
            QueryType::Procedural
        } else if query.contains("why") {
            QueryType::Causal
        } else if query.contains("when") || query.contains("where") {
            QueryType::Factual
        } else if query.contains("compare") || query.contains("vs") {
            QueryType::Comparative
        } else {
            QueryType::General
        }
    }

    fn assess_query_complexity(&self, query: &str) -> f64 {
        // Basic complexity assessment
        let word_count = query.split_whitespace().count();
        let question_marks = query.chars().filter(|&c| c == '?').count();
        let complex_words = ["analyze", "evaluate", "compare", "contrast", "explain", "describe"];
        let complex_word_count = complex_words.iter()
            .map(|word| query.to_lowercase().matches(word).count())
            .sum::<usize>();

        let complexity = (word_count as f64 / 20.0) + 
                        (question_marks as f64 * 0.2) + 
                        (complex_word_count as f64 * 0.3);
        
        complexity.min(1.0)
    }

    async fn extract_relevant_sentences(&self, content: &str, query: &str) -> Result<Vec<String>> {
        // Extract sentences most relevant to the query
        let sentences: Vec<String> = content
            .split('.')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let query_words: HashSet<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let mut scored_sentences: Vec<(String, f64)> = sentences
            .into_iter()
            .map(|sentence| {
                let sentence_words: HashSet<String> = sentence
                    .to_lowercase()
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();

                let overlap = query_words.intersection(&sentence_words).count();
                let relevance_score = overlap as f64 / query_words.len().max(1) as f64;
                
                (sentence, relevance_score)
            })
            .collect();

        // Sort by relevance score
        scored_sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top 3 most relevant sentences
        Ok(scored_sentences
            .into_iter()
            .take(3)
            .map(|(sentence, _)| sentence)
            .collect())
    }

    async fn extract_supporting_information(&self, content: &str) -> Result<String> {
        // Extract supporting information that doesn't directly answer the query
        // but provides valuable context
        let sentences: Vec<&str> = content.split('.').collect();
        
        // Take middle portion of content as supporting info
        let start_idx = sentences.len() / 3;
        let end_idx = (sentences.len() * 2) / 3;
        
        let supporting_sentences: Vec<&str> = sentences
            .iter()
            .skip(start_idx)
            .take(end_idx - start_idx)
            .cloned()
            .collect();

        Ok(supporting_sentences.join(". ").trim().to_string())
    }
}

impl QualityMetrics {
    pub fn overall_score(&self) -> f64 {
        self.relevance * 0.3 + 
         self.coherence * 0.2 + 
         self.completeness * 0.2 + 
         self.accuracy_indicators * 0.2 + 
         self.source_diversity * 0.1
    }
}

/// Query analysis results
#[derive(Debug, Clone)]
struct QueryAnalysis {
    query_type: QueryType,
    needs_background: bool,
    needs_caveats: bool,
    expected_sections: Vec<SectionType>,
    complexity_level: f64,
}

/// Types of queries
#[derive(Debug, Clone, PartialEq)]
enum QueryType {
    Definitional,
    Procedural,
    Causal,
    Factual,
    Comparative,
    General,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_response_builder_creation() {
        let request = crate::GenerationRequest::builder()
            .query("Test query")
            .build()
            .unwrap();

        let builder = ResponseBuilder::new(request);
        assert!(builder.content.is_empty());
        assert!(builder.confidence_factors.is_empty());
    }

    #[tokio::test]
    async fn test_quality_score_calculation() {
        let request = crate::GenerationRequest::builder()
            .query("Test query")
            .build()
            .unwrap();

        let builder = ResponseBuilder::new(request);
        
        let chunk = ContextChunk {
            content: "This is a test content with reasonable length and information.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Test Source".to_string(),
                url: None,
                document_type: "text".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.8,
            position: Some(0),
            metadata: HashMap::new(),
        };

        let quality_score = builder.calculate_quality_score(&chunk).await.unwrap();
        assert!(quality_score > 0.0 && quality_score <= 1.0);
    }

    #[test]
    fn test_query_type_determination() {
        let request = crate::GenerationRequest::builder()
            .query("What is Rust programming language?")
            .build()
            .unwrap();

        let builder = ResponseBuilder::new(request);
        let query_type = builder.determine_query_type("what is rust programming language?");
        assert_eq!(query_type, QueryType::Definitional);
    }
}