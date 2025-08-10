//! Key term extraction component for identifying important query terms
//!
//! This module provides comprehensive key term extraction using TF-IDF,
//! N-gram analysis, and domain-specific term identification.

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{info, instrument};

use crate::config::TermExtractorConfig;
use crate::error::{ProcessorError, Result};
use crate::query::Query;
use crate::types::*;

/// Key term extractor for identifying important query terms
pub struct KeyTermExtractor {
    config: Arc<TermExtractorConfig>,
    tfidf_calculator: Option<TfIdfCalculator>,
    ngram_extractor: NgramExtractor,
    stop_words: HashSet<String>,
    domain_terms: HashMap<String, f64>,
}

impl KeyTermExtractor {
    /// Create a new key term extractor
    #[instrument(skip(config))]
    pub async fn new(config: Arc<TermExtractorConfig>) -> Result<Self> {
        info!("Initializing Key Term Extractor");
        
        let tfidf_calculator = if config.enable_tfidf {
            Some(TfIdfCalculator::new().await?)
        } else {
            None
        };
        
        let ngram_extractor = NgramExtractor::new(config.clone()).await?;
        
        let stop_words: HashSet<String> = config.stop_words.iter().cloned().collect();
        let domain_terms = Self::initialize_domain_terms();
        
        Ok(Self {
            config,
            tfidf_calculator,
            ngram_extractor,
            stop_words,
            domain_terms,
        })
    }

    /// Extract key terms from query
    #[instrument(skip(self, query, analysis))]
    pub async fn extract(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
    ) -> Result<Vec<KeyTerm>> {
        info!("Extracting key terms from query: {}", query.id());
        
        let mut all_terms = Vec::new();
        
        // Extract unigrams
        if self.config.ngram_sizes.contains(&1) {
            let unigrams = self.extract_unigrams(query, analysis).await?;
            all_terms.extend(unigrams);
        }
        
        // Extract n-grams
        if self.config.enable_ngrams {
            for &n in &self.config.ngram_sizes {
                if n > 1 {
                    let ngrams = self.ngram_extractor.extract_ngrams(query, analysis, n).await?;
                    all_terms.extend(ngrams);
                }
            }
        }
        
        // Calculate TF-IDF scores if enabled
        if let Some(ref calculator) = self.tfidf_calculator {
            for term in &mut all_terms {
                term.tfidf_score = Some(calculator.calculate_tfidf(&term.term, query.text()).await?);
            }
        }
        
        // Filter and score terms
        let filtered_terms = self.filter_and_score_terms(all_terms).await?;
        
        // Limit to max terms
        let limited_terms = self.limit_terms(filtered_terms);
        
        info!("Extracted {} key terms", limited_terms.len());
        Ok(limited_terms)
    }

    /// Extract unigrams (single words)
    async fn extract_unigrams(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
    ) -> Result<Vec<KeyTerm>> {
        let mut terms = Vec::new();
        let text = query.text().to_lowercase();
        let tokens: Vec<String> = text.split_whitespace()
            .map(|s| s.trim_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        
        // Count term frequencies
        let mut term_counts = HashMap::new();
        for token in &tokens {
            if !self.stop_words.contains(token) && token.len() > 2 {
                *term_counts.entry(token.clone()).or_insert(0) += 1;
            }
        }
        
        // Filter by minimum frequency
        for (term, frequency) in term_counts {
            if frequency >= self.config.min_frequency {
                let importance = self.calculate_term_importance(&term, frequency, analysis);
                let category = self.classify_term_category(&term, analysis);
                
                terms.push(KeyTerm {
                    term: term.clone(),
                    importance,
                    frequency,
                    category,
                    ngram_size: 1,
                    tfidf_score: None, // Will be calculated later if enabled
                    contexts: self.extract_term_contexts(&term, query.text()),
                });
            }
        }
        
        Ok(terms)
    }

    /// Calculate term importance score
    fn calculate_term_importance(
        &self,
        term: &str,
        frequency: usize,
        analysis: &SemanticAnalysis,
    ) -> f64 {
        let mut importance = 0.0;
        
        // Base importance from frequency
        importance += (frequency as f64).log2() * 0.3;
        
        // Domain-specific terms get higher importance
        if let Some(&domain_score) = self.domain_terms.get(term) {
            importance += domain_score * 0.4;
        }
        
        // Terms found in named entities get higher importance
        let in_named_entity = analysis.syntactic_features.named_entities
            .iter()
            .any(|entity| entity.text.to_lowercase().contains(term));
        
        if in_named_entity {
            importance += 0.3;
        }
        
        // Question words get lower importance (they're structural, not content)
        if analysis.syntactic_features.question_words.contains(&term.to_string()) {
            importance -= 0.2;
        }
        
        // Terms in noun phrases get slightly higher importance
        let in_noun_phrase = analysis.syntactic_features.noun_phrases
            .iter()
            .any(|phrase| phrase.text.to_lowercase().contains(term));
        
        if in_noun_phrase {
            importance += 0.1;
        }
        
        importance.clamp(0.0, 1.0)
    }

    /// Classify term category based on context and content
    fn classify_term_category(&self, term: &str, analysis: &SemanticAnalysis) -> TermCategory {
        // Check if it's a domain-specific term
        if self.domain_terms.contains_key(term) {
            return TermCategory::Domain;
        }
        
        // Check if it appears in named entities
        for entity in &analysis.syntactic_features.named_entities {
            if entity.text.to_lowercase().contains(term) {
                return match entity.entity_type.as_str() {
                    "STANDARD" | "REQUIREMENT" => TermCategory::Compliance,
                    "TECHNICAL_TERM" => TermCategory::Technical,
                    _ => TermCategory::Concept,
                };
            }
        }
        
        // Check if it's an action word (verb-like)
        let action_indicators = ["implement", "configure", "validate", "assess", "monitor", "review"];
        if action_indicators.iter().any(|&action| term.contains(action)) {
            return TermCategory::Action;
        }
        
        // Default classification
        TermCategory::General
    }

    /// Extract contexts where term appears
    fn extract_term_contexts(&self, term: &str, text: &str) -> Vec<String> {
        let mut contexts = Vec::new();
        let text_lower = text.to_lowercase();
        let term_lower = term.to_lowercase();
        
        let mut start = 0;
        while let Some(pos) = text_lower[start..].find(&term_lower) {
            let absolute_pos = start + pos;
            let context_start = absolute_pos.saturating_sub(20);
            let context_end = (absolute_pos + term.len() + 20).min(text.len());
            
            let context = text[context_start..context_end].to_string();
            contexts.push(context);
            
            start = absolute_pos + term.len();
            
            // Limit contexts to avoid excessive data
            if contexts.len() >= 3 {
                break;
            }
        }
        
        contexts
    }

    /// Filter terms by frequency and score them
    async fn filter_and_score_terms(&self, mut terms: Vec<KeyTerm>) -> Result<Vec<KeyTerm>> {
        // Filter by minimum frequency
        terms.retain(|term| term.frequency >= self.config.min_frequency);
        
        // Boost terms with TF-IDF scores if available
        for term in &mut terms {
            if let Some(tfidf) = term.tfidf_score {
                term.importance = (term.importance + tfidf * 0.3).clamp(0.0, 1.0);
            }
        }
        
        // Sort by importance
        terms.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(terms)
    }

    /// Limit terms to maximum count
    fn limit_terms(&self, mut terms: Vec<KeyTerm>) -> Vec<KeyTerm> {
        terms.truncate(self.config.max_terms);
        terms
    }

    /// Initialize domain-specific terms with importance scores
    fn initialize_domain_terms() -> HashMap<String, f64> {
        let mut terms = HashMap::new();
        
        // High importance compliance terms
        terms.insert("encryption".to_string(), 1.0);
        terms.insert("compliance".to_string(), 1.0);
        terms.insert("audit".to_string(), 0.95);
        terms.insert("security".to_string(), 0.95);
        terms.insert("vulnerability".to_string(), 0.90);
        terms.insert("authentication".to_string(), 0.90);
        terms.insert("authorization".to_string(), 0.85);
        terms.insert("firewall".to_string(), 0.80);
        terms.insert("monitoring".to_string(), 0.80);
        terms.insert("assessment".to_string(), 0.85);
        
        // Medium importance terms
        terms.insert("requirement".to_string(), 0.75);
        terms.insert("control".to_string(), 0.75);
        terms.insert("policy".to_string(), 0.70);
        terms.insert("procedure".to_string(), 0.70);
        terms.insert("documentation".to_string(), 0.65);
        terms.insert("implementation".to_string(), 0.70);
        terms.insert("validation".to_string(), 0.75);
        terms.insert("remediation".to_string(), 0.70);
        
        // PCI DSS specific terms
        terms.insert("cardholder".to_string(), 0.95);
        terms.insert("payment".to_string(), 0.80);
        terms.insert("merchant".to_string(), 0.75);
        terms.insert("acquirer".to_string(), 0.70);
        terms.insert("processor".to_string(), 0.70);
        
        terms
    }
}

/// N-gram extractor for multi-word phrases
pub struct NgramExtractor {
    config: Arc<TermExtractorConfig>,
}

impl NgramExtractor {
    pub async fn new(config: Arc<TermExtractorConfig>) -> Result<Self> {
        Ok(Self { config })
    }

    pub async fn extract_ngrams(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
        n: usize,
    ) -> Result<Vec<KeyTerm>> {
        let mut ngrams = Vec::new();
        let text = query.text().to_lowercase();
        let tokens: Vec<String> = text.split_whitespace()
            .map(|s| s.trim_matches(|c: char| c.is_ascii_punctuation()))
            .filter(|s| !s.is_empty() && s.len() > 2)
            .map(|s| s.to_string())
            .collect();
        
        if tokens.len() < n {
            return Ok(ngrams);
        }
        
        // Extract n-grams
        let mut ngram_counts = HashMap::new();
        for window in tokens.windows(n) {
            // Skip n-grams that are mostly stop words
            let stop_word_count = window.iter()
                .filter(|token| self.config.stop_words.contains(*token))
                .count();
            
            if stop_word_count >= n - 1 {
                continue;
            }
            
            let ngram = window.join(" ");
            *ngram_counts.entry(ngram).or_insert(0) += 1;
        }
        
        // Convert to KeyTerm objects
        for (ngram, frequency) in ngram_counts {
            if frequency >= self.config.min_frequency {
                let importance = self.calculate_ngram_importance(&ngram, frequency, analysis, n);
                let category = self.classify_ngram_category(&ngram, analysis);
                
                ngrams.push(KeyTerm {
                    term: ngram.clone(),
                    importance,
                    frequency,
                    category,
                    ngram_size: n,
                    tfidf_score: None,
                    contexts: vec![ngram.clone()], // For n-grams, the term itself is the context
                });
            }
        }
        
        Ok(ngrams)
    }

    fn calculate_ngram_importance(
        &self,
        ngram: &str,
        frequency: usize,
        analysis: &SemanticAnalysis,
        n: usize,
    ) -> f64 {
        let mut importance = 0.0;
        
        // Base importance from frequency
        importance += (frequency as f64).log2() * 0.2;
        
        // Longer n-grams get higher base importance
        importance += (n as f64) * 0.1;
        
        // Check if n-gram appears in named entities
        let in_named_entity = analysis.syntactic_features.named_entities
            .iter()
            .any(|entity| entity.text.to_lowercase().contains(ngram));
        
        if in_named_entity {
            importance += 0.4;
        }
        
        // Check if n-gram appears in noun phrases
        let in_noun_phrase = analysis.syntactic_features.noun_phrases
            .iter()
            .any(|phrase| phrase.text.to_lowercase().contains(ngram));
        
        if in_noun_phrase {
            importance += 0.3;
        }
        
        // Boost for compliance-related phrases
        let compliance_indicators = ["pci dss", "data security", "access control", "vulnerability scan"];
        if compliance_indicators.iter().any(|&indicator| ngram.contains(indicator)) {
            importance += 0.5;
        }
        
        importance.clamp(0.0, 1.0)
    }

    fn classify_ngram_category(&self, ngram: &str, _analysis: &SemanticAnalysis) -> TermCategory {
        // Classify based on content patterns
        if ngram.contains("encrypt") || ngram.contains("security") || ngram.contains("access control") {
            TermCategory::Technical
        } else if ngram.contains("compliance") || ngram.contains("audit") || ngram.contains("requirement") {
            TermCategory::Compliance
        } else if ngram.contains("implement") || ngram.contains("configure") || ngram.contains("validate") {
            TermCategory::Action
        } else {
            TermCategory::Concept
        }
    }
}

/// TF-IDF calculator for term importance scoring
pub struct TfIdfCalculator {
    // In a real implementation, this would maintain document corpus statistics
    // For now, it's a placeholder
}

impl TfIdfCalculator {
    pub async fn new() -> Result<Self> {
        Ok(Self {})
    }

    pub async fn calculate_tfidf(&self, term: &str, document: &str) -> Result<f64> {
        // Simplified TF-IDF calculation
        // In a real system, this would use a proper corpus and IDF calculations
        
        // Calculate term frequency
        let term_count = document.to_lowercase()
            .split_whitespace()
            .filter(|word| word.trim_matches(|c: char| c.is_ascii_punctuation()) == term)
            .count();
        
        let total_words = document.split_whitespace().count();
        let tf = if total_words > 0 {
            term_count as f64 / total_words as f64
        } else {
            0.0
        };
        
        // Simplified IDF (in real implementation, would use document collection)
        let idf = 1.0 + (1.0 / (1.0 + term_count as f64)).ln();
        
        Ok(tf * idf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TermExtractorConfig;

    fn create_test_analysis() -> SemanticAnalysis {
        SemanticAnalysis {
            syntactic_features: SyntacticFeatures {
                pos_tags: vec![],
                named_entities: vec![
                    NamedEntity::new(
                        "PCI DSS".to_string(),
                        "STANDARD".to_string(),
                        10,
                        17,
                        0.95,
                    ),
                ],
                noun_phrases: vec![
                    Phrase {
                        text: "encryption requirements".to_string(),
                        phrase_type: "NP".to_string(),
                        start: 20,
                        end: 40,
                        head: Some("requirements".to_string()),
                        modifiers: vec!["encryption".to_string()],
                    },
                ],
                verb_phrases: vec![],
                question_words: vec!["what".to_string()],
            },
            semantic_features: SemanticFeatures {
                semantic_roles: vec![],
                coreferences: vec![],
                sentiment: None,
                similarity_vectors: vec![],
            },
            dependencies: vec![],
            topics: vec![],
            confidence: 0.8,
        }
    }

    #[tokio::test]
    async fn test_key_term_extractor_creation() {
        let config = Arc::new(TermExtractorConfig::default());
        let extractor = KeyTermExtractor::new(config).await;
        assert!(extractor.is_ok());
    }

    #[tokio::test]
    async fn test_unigram_extraction() {
        let config = Arc::new(TermExtractorConfig::default());
        let extractor = KeyTermExtractor::new(config).await.unwrap();
        
        let query = Query::new("What are the encryption requirements in PCI DSS 4.0?");
        let analysis = create_test_analysis();
        
        let result = extractor.extract(&query, &analysis).await;
        assert!(result.is_ok());
        
        let terms = result.unwrap();
        assert!(!terms.is_empty());
        
        // Should find important terms like "encryption", "requirements", "PCI", "DSS"
        let term_texts: Vec<String> = terms.iter().map(|t| t.term.clone()).collect();
        assert!(term_texts.iter().any(|t| t.contains("encryption") || t.contains("requirements")));
    }

    #[tokio::test]
    async fn test_ngram_extraction() {
        let mut config = TermExtractorConfig::default();
        config.ngram_sizes = vec![1, 2, 3];
        config.enable_ngrams = true;
        let config = Arc::new(config);
        
        let extractor = KeyTermExtractor::new(config).await.unwrap();
        
        let query = Query::new("What are PCI DSS compliance requirements for data encryption?");
        let analysis = create_test_analysis();
        
        let result = extractor.extract(&query, &analysis).await;
        assert!(result.is_ok());
        
        let terms = result.unwrap();
        
        // Should have both unigrams and n-grams
        let has_unigrams = terms.iter().any(|t| t.ngram_size == 1);
        let has_bigrams = terms.iter().any(|t| t.ngram_size == 2);
        
        assert!(has_unigrams);
        assert!(has_bigrams);
        
        // Should find "PCI DSS" as a bigram
        assert!(terms.iter().any(|t| t.term == "pci dss"));
    }

    #[tokio::test]
    async fn test_term_importance_calculation() {
        let config = Arc::new(TermExtractorConfig::default());
        let extractor = KeyTermExtractor::new(config).await.unwrap();
        
        let analysis = create_test_analysis();
        
        // Domain term should have high importance
        let encryption_importance = extractor.calculate_term_importance("encryption", 3, &analysis);
        assert!(encryption_importance > 0.7);
        
        // Common word should have lower importance
        let the_importance = extractor.calculate_term_importance("the", 5, &analysis);
        assert!(the_importance < 0.5);
    }

    #[tokio::test]
    async fn test_term_categorization() {
        let config = Arc::new(TermExtractorConfig::default());
        let extractor = KeyTermExtractor::new(config).await.unwrap();
        
        let analysis = create_test_analysis();
        
        let encryption_category = extractor.classify_term_category("encryption", &analysis);
        assert_eq!(encryption_category, TermCategory::Domain);
        
        let implement_category = extractor.classify_term_category("implement", &analysis);
        assert_eq!(implement_category, TermCategory::Action);
    }

    #[tokio::test]
    async fn test_tfidf_calculation() {
        let calculator = TfIdfCalculator::new().await.unwrap();
        
        let document = "encryption is important for security encryption protects data";
        let tfidf = calculator.calculate_tfidf("encryption", document).await.unwrap();
        
        assert!(tfidf > 0.0);
        
        // More frequent terms should have higher TF component
        let security_tfidf = calculator.calculate_tfidf("security", document).await.unwrap();
        assert!(tfidf > security_tfidf); // "encryption" appears twice, "security" once
    }
}