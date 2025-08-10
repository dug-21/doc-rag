//! Entity extraction component for identifying and classifying entities
//!
//! This module provides comprehensive entity extraction capabilities including
//! named entity recognition, compliance-specific entities, and technical terms.

use async_trait::async_trait;
use chrono::Utc;
use regex::Regex;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, instrument, warn};

use crate::config::EntityExtractorConfig;
use crate::error::{ProcessorError, Result};
use crate::query::Query;
use crate::types::*;

/// Entity extractor for identifying entities in queries
pub struct EntityExtractor {
    config: Arc<EntityExtractorConfig>,
    ner_extractor: NamedEntityExtractor,
    compliance_extractor: ComplianceEntityExtractor,
    technical_extractor: TechnicalTermExtractor,
    patterns: HashMap<EntityCategory, Vec<EntityPattern>>,
}

impl EntityExtractor {
    /// Create a new entity extractor
    #[instrument(skip(config))]
    pub async fn new(config: Arc<EntityExtractorConfig>) -> Result<Self> {
        info!("Initializing Entity Extractor");
        
        let ner_extractor = NamedEntityExtractor::new(config.clone()).await?;
        let compliance_extractor = ComplianceEntityExtractor::new(config.clone()).await?;
        let technical_extractor = TechnicalTermExtractor::new(config.clone()).await?;
        let patterns = Self::initialize_patterns();
        
        Ok(Self {
            config,
            ner_extractor,
            compliance_extractor,
            technical_extractor,
            patterns,
        })
    }

    /// Extract entities from query
    #[instrument(skip(self, query, analysis))]
    pub async fn extract(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
    ) -> Result<Vec<ExtractedEntity>> {
        info!("Extracting entities from query: {}", query.id());
        
        let mut all_entities = Vec::new();
        
        // Extract named entities using NLP
        if self.config.enable_ner {
            let ner_entities = self.ner_extractor.extract(query, analysis).await?;
            all_entities.extend(ner_entities);
        }
        
        // Extract compliance-specific entities
        if self.config.enable_compliance_entities {
            let compliance_entities = self.compliance_extractor.extract(query, analysis).await?;
            all_entities.extend(compliance_entities);
        }
        
        // Extract technical terms
        if self.config.enable_technical_terms {
            let technical_entities = self.technical_extractor.extract(query, analysis).await?;
            all_entities.extend(technical_entities);
        }
        
        // Apply pattern-based extraction
        let pattern_entities = self.extract_pattern_entities(query, analysis).await?;
        all_entities.extend(pattern_entities);
        
        // Deduplicate and filter by confidence
        let filtered_entities = self.deduplicate_and_filter(all_entities).await?;
        
        // Limit to max entities
        let limited_entities = self.limit_entities(filtered_entities);
        
        info!("Extracted {} entities", limited_entities.len());
        Ok(limited_entities)
    }

    /// Extract entities using predefined patterns
    async fn extract_pattern_entities(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
    ) -> Result<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();
        let text = query.text();
        
        for (category, patterns) in &self.patterns {
            for pattern in patterns {
                let matches = pattern.find_matches(text)?;
                
                for entity_match in matches {
                    let entity = ExtractedEntity {
                        entity: NamedEntity::new(
                            entity_match.text.clone(),
                            category.to_string(),
                            entity_match.start,
                            entity_match.end,
                            entity_match.confidence,
                        ),
                        category: category.clone(),
                        metadata: EntityMetadata {
                            extraction_method: "pattern_based".to_string(),
                            extracted_at: Utc::now(),
                            context: self.extract_context(text, entity_match.start, entity_match.end),
                            normalization: Some(pattern.normalize(&entity_match.text)),
                            properties: HashMap::new(),
                        },
                        relationships: Vec::new(),
                    };
                    
                    entities.push(entity);
                }
            }
        }
        
        Ok(entities)
    }

    /// Extract context around entity
    fn extract_context(&self, text: &str, start: usize, end: usize) -> String {
        let context_window = 50; // characters
        let context_start = start.saturating_sub(context_window);
        let context_end = (end + context_window).min(text.len());
        
        text.chars()
            .skip(context_start)
            .take(context_end - context_start)
            .collect()
    }

    /// Deduplicate overlapping entities and filter by confidence
    async fn deduplicate_and_filter(
        &self,
        entities: Vec<ExtractedEntity>,
    ) -> Result<Vec<ExtractedEntity>> {
        let mut filtered = Vec::new();
        let mut sorted_entities = entities;
        
        // Sort by confidence (descending) and then by start position
        sorted_entities.sort_by(|a, b| {
            b.entity.confidence.partial_cmp(&a.entity.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.entity.start.cmp(&b.entity.start))
        });
        
        for entity in sorted_entities {
            // Filter by confidence threshold
            if entity.entity.confidence < self.config.confidence_threshold {
                continue;
            }
            
            // Check for overlap with already accepted entities
            let overlaps = filtered.iter().any(|existing: &ExtractedEntity| {
                self.entities_overlap(&entity, existing)
            });
            
            if !overlaps {
                filtered.push(entity);
            }
        }
        
        Ok(filtered)
    }

    /// Check if two entities overlap
    fn entities_overlap(&self, a: &ExtractedEntity, b: &ExtractedEntity) -> bool {
        let a_range = a.entity.start..a.entity.end;
        let b_range = b.entity.start..b.entity.end;
        
        // Check for any overlap
        a_range.start < b_range.end && b_range.start < a_range.end
    }

    /// Limit entities to maximum count
    fn limit_entities(&self, mut entities: Vec<ExtractedEntity>) -> Vec<ExtractedEntity> {
        if entities.len() <= self.config.max_entities {
            return entities;
        }
        
        // Sort by confidence and take top entities
        entities.sort_by(|a, b| {
            b.entity.confidence.partial_cmp(&a.entity.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        entities.truncate(self.config.max_entities);
        entities
    }

    /// Initialize entity extraction patterns
    fn initialize_patterns() -> HashMap<EntityCategory, Vec<EntityPattern>> {
        let mut patterns = HashMap::new();
        
        // Standard patterns (PCI DSS, HIPAA, etc.)
        patterns.insert(EntityCategory::Standard, vec![
            EntityPattern::new(
                r"(?i)PCI\s+DSS(?:\s+(\d+(?:\.\d+)*?))?",
                "PCI DSS standard detection",
                0.95,
            ),
            EntityPattern::new(
                r"(?i)HIPAA",
                "HIPAA standard detection", 
                0.95,
            ),
            EntityPattern::new(
                r"(?i)SOX",
                "Sarbanes-Oxley detection",
                0.90,
            ),
            EntityPattern::new(
                r"(?i)GDPR",
                "GDPR regulation detection",
                0.95,
            ),
            EntityPattern::new(
                r"(?i)ISO\s+27001",
                "ISO 27001 standard detection",
                0.95,
            ),
        ]);
        
        // Version patterns
        patterns.insert(EntityCategory::Version, vec![
            EntityPattern::new(
                r"\b\d+\.\d+(?:\.\d+)*\b",
                "Version number detection",
                0.80,
            ),
            EntityPattern::new(
                r"(?i)version\s+(\d+(?:\.\d+)*)",
                "Explicit version reference",
                0.90,
            ),
        ]);
        
        // Requirement patterns
        patterns.insert(EntityCategory::Requirement, vec![
            EntityPattern::new(
                r"(?i)requirement\s+(\d+(?:\.\d+)*)",
                "Requirement reference",
                0.90,
            ),
            EntityPattern::new(
                r"(?i)req\.?\s*(\d+(?:\.\d+)*)",
                "Abbreviated requirement reference",
                0.85,
            ),
            EntityPattern::new(
                r"(?i)section\s+(\d+(?:\.\d+)*)",
                "Section reference",
                0.85,
            ),
        ]);
        
        // Control patterns
        patterns.insert(EntityCategory::Control, vec![
            EntityPattern::new(
                r"(?i)control\s+(\w+(?:\.\w+)*)",
                "Security control reference",
                0.85,
            ),
            EntityPattern::new(
                r"(?i)compensating\s+control",
                "Compensating control reference",
                0.90,
            ),
        ]);
        
        // Technical term patterns
        patterns.insert(EntityCategory::TechnicalTerm, vec![
            EntityPattern::new(
                r"(?i)\b(?:encryption|decryption|cryptography)\b",
                "Cryptography terms",
                0.80,
            ),
            EntityPattern::new(
                r"(?i)\b(?:authentication|authorization)\b",
                "Auth terms",
                0.80,
            ),
            EntityPattern::new(
                r"(?i)\b(?:vulnerability|exploit|attack)\b",
                "Security threat terms",
                0.80,
            ),
            EntityPattern::new(
                r"(?i)\b(?:firewall|intrusion|detection|prevention)\b",
                "Security system terms",
                0.80,
            ),
        ]);
        
        // Date patterns
        patterns.insert(EntityCategory::Date, vec![
            EntityPattern::new(
                r"\b\d{4}-\d{2}-\d{2}\b",
                "ISO date format",
                0.95,
            ),
            EntityPattern::new(
                r"\b\d{1,2}/\d{1,2}/\d{4}\b",
                "US date format",
                0.90,
            ),
            EntityPattern::new(
                r"(?i)\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
                "Natural date format",
                0.85,
            ),
        ]);
        
        patterns
    }
}

/// Named entity extractor using NLP
pub struct NamedEntityExtractor {
    _config: Arc<EntityExtractorConfig>,
}

impl NamedEntityExtractor {
    pub async fn new(config: Arc<EntityExtractorConfig>) -> Result<Self> {
        Ok(Self {
            _config: config,
        })
    }

    pub async fn extract(
        &self,
        _query: &Query,
        analysis: &SemanticAnalysis,
    ) -> Result<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();
        
        // Convert named entities from analysis to extracted entities
        for named_entity in &analysis.syntactic_features.named_entities {
            let category = self.classify_entity_category(&named_entity.entity_type);
            
            let entity = ExtractedEntity {
                entity: named_entity.clone(),
                category,
                metadata: EntityMetadata {
                    extraction_method: "ner".to_string(),
                    extracted_at: Utc::now(),
                    context: "".to_string(), // Would be filled from original text
                    normalization: None,
                    properties: HashMap::new(),
                },
                relationships: Vec::new(),
            };
            
            entities.push(entity);
        }
        
        Ok(entities)
    }

    fn classify_entity_category(&self, entity_type: &str) -> EntityCategory {
        match entity_type.to_uppercase().as_str() {
            "STANDARD" => EntityCategory::Standard,
            "REQUIREMENT" => EntityCategory::Requirement,
            "CONTROL" => EntityCategory::Control,
            "VERSION" => EntityCategory::Version,
            "DATE" => EntityCategory::Date,
            "PERSON" => EntityCategory::Person,
            "ORG" | "ORGANIZATION" => EntityCategory::Organization,
            "LOC" | "LOCATION" => EntityCategory::Location,
            _ => EntityCategory::Unknown,
        }
    }
}

/// Compliance-specific entity extractor
pub struct ComplianceEntityExtractor {
    _config: Arc<EntityExtractorConfig>,
    compliance_terms: HashMap<String, EntityCategory>,
}

impl ComplianceEntityExtractor {
    pub async fn new(config: Arc<EntityExtractorConfig>) -> Result<Self> {
        let compliance_terms = Self::initialize_compliance_terms();
        
        Ok(Self {
            _config: config,
            compliance_terms,
        })
    }

    pub async fn extract(
        &self,
        query: &Query,
        _analysis: &SemanticAnalysis,
    ) -> Result<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();
        let text = query.text().to_lowercase();
        
        for (term, category) in &self.compliance_terms {
            if let Some(start) = text.find(term) {
                let end = start + term.len();
                
                let entity = ExtractedEntity {
                    entity: NamedEntity::new(
                        term.clone(),
                        category.to_string(),
                        start,
                        end,
                        0.85, // Standard confidence for compliance terms
                    ),
                    category: category.clone(),
                    metadata: EntityMetadata {
                        extraction_method: "compliance_specific".to_string(),
                        extracted_at: Utc::now(),
                        context: "".to_string(),
                        normalization: None,
                        properties: HashMap::new(),
                    },
                    relationships: Vec::new(),
                };
                
                entities.push(entity);
            }
        }
        
        Ok(entities)
    }

    fn initialize_compliance_terms() -> HashMap<String, EntityCategory> {
        let mut terms = HashMap::new();
        
        // Compliance standards
        terms.insert("pci dss".to_string(), EntityCategory::Standard);
        terms.insert("hipaa".to_string(), EntityCategory::Standard);
        terms.insert("sox".to_string(), EntityCategory::Standard);
        terms.insert("gdpr".to_string(), EntityCategory::Standard);
        
        // Security controls
        terms.insert("access control".to_string(), EntityCategory::Control);
        terms.insert("encryption".to_string(), EntityCategory::Control);
        terms.insert("key management".to_string(), EntityCategory::Control);
        terms.insert("vulnerability management".to_string(), EntityCategory::Control);
        
        // Compliance processes
        terms.insert("audit".to_string(), EntityCategory::Process);
        terms.insert("assessment".to_string(), EntityCategory::Process);
        terms.insert("remediation".to_string(), EntityCategory::Process);
        terms.insert("monitoring".to_string(), EntityCategory::Process);
        
        terms
    }
}

/// Technical term extractor
pub struct TechnicalTermExtractor {
    _config: Arc<EntityExtractorConfig>,
    technical_terms: HashMap<String, f64>, // term -> importance score
}

impl TechnicalTermExtractor {
    pub async fn new(config: Arc<EntityExtractorConfig>) -> Result<Self> {
        let technical_terms = Self::initialize_technical_terms();
        
        Ok(Self {
            _config: config,
            technical_terms,
        })
    }

    pub async fn extract(
        &self,
        query: &Query,
        _analysis: &SemanticAnalysis,
    ) -> Result<Vec<ExtractedEntity>> {
        let mut entities = Vec::new();
        let text = query.text().to_lowercase();
        
        for (term, importance) in &self.technical_terms {
            if let Some(start) = text.find(term) {
                let end = start + term.len();
                let confidence = importance * 0.8; // Scale importance to confidence
                
                let entity = ExtractedEntity {
                    entity: NamedEntity::new(
                        term.clone(),
                        "TECHNICAL_TERM".to_string(),
                        start,
                        end,
                        confidence,
                    ),
                    category: EntityCategory::TechnicalTerm,
                    metadata: EntityMetadata {
                        extraction_method: "technical_terms".to_string(),
                        extracted_at: Utc::now(),
                        context: "".to_string(),
                        normalization: None,
                        properties: {
                            let mut props = HashMap::new();
                            props.insert(
                                "importance".to_string(),
                                serde_json::Value::Number(serde_json::Number::from_f64(*importance).unwrap_or_else(|| serde_json::Number::from(0))),
                            );
                            props
                        },
                    },
                    relationships: Vec::new(),
                };
                
                entities.push(entity);
            }
        }
        
        Ok(entities)
    }

    fn initialize_technical_terms() -> HashMap<String, f64> {
        let mut terms = HashMap::new();
        
        // High importance terms
        terms.insert("encryption".to_string(), 1.0);
        terms.insert("authentication".to_string(), 1.0);
        terms.insert("authorization".to_string(), 0.95);
        terms.insert("vulnerability".to_string(), 0.95);
        terms.insert("penetration testing".to_string(), 0.90);
        
        // Medium importance terms
        terms.insert("firewall".to_string(), 0.80);
        terms.insert("intrusion detection".to_string(), 0.85);
        terms.insert("malware".to_string(), 0.75);
        terms.insert("phishing".to_string(), 0.70);
        terms.insert("ssl".to_string(), 0.80);
        terms.insert("tls".to_string(), 0.80);
        
        // Lower importance but relevant terms
        terms.insert("password".to_string(), 0.60);
        terms.insert("backup".to_string(), 0.55);
        terms.insert("logging".to_string(), 0.65);
        terms.insert("monitoring".to_string(), 0.70);
        
        terms
    }
}

/// Entity extraction pattern
#[derive(Debug, Clone)]
pub struct EntityPattern {
    regex: Regex,
    description: String,
    confidence: f64,
}

impl EntityPattern {
    pub fn new(pattern: &str, description: &str, confidence: f64) -> Self {
        let regex = Regex::new(pattern).expect("Invalid regex pattern");
        
        Self {
            regex,
            description: description.to_string(),
            confidence,
        }
    }

    pub fn find_matches(&self, text: &str) -> Result<Vec<EntityMatch>> {
        let mut matches = Vec::new();
        
        for cap in self.regex.captures_iter(text) {
            if let Some(full_match) = cap.get(0) {
                matches.push(EntityMatch {
                    text: full_match.as_str().to_string(),
                    start: full_match.start(),
                    end: full_match.end(),
                    confidence: self.confidence,
                    groups: cap.iter()
                        .skip(1) // Skip full match
                        .map(|m| m.map(|m| m.as_str().to_string()))
                        .collect(),
                });
            }
        }
        
        Ok(matches)
    }

    pub fn normalize(&self, text: &str) -> String {
        // Simple normalization - could be more sophisticated
        text.trim().to_lowercase()
    }
}

/// Entity match result
#[derive(Debug, Clone)]
pub struct EntityMatch {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f64,
    pub groups: Vec<Option<String>>,
}

impl EntityCategory {
    fn to_string(&self) -> String {
        match self {
            EntityCategory::Standard => "STANDARD".to_string(),
            EntityCategory::Requirement => "REQUIREMENT".to_string(),
            EntityCategory::Control => "CONTROL".to_string(),
            EntityCategory::TechnicalTerm => "TECHNICAL_TERM".to_string(),
            EntityCategory::Organization => "ORGANIZATION".to_string(),
            EntityCategory::Date => "DATE".to_string(),
            EntityCategory::Version => "VERSION".to_string(),
            EntityCategory::Person => "PERSON".to_string(),
            EntityCategory::Location => "LOCATION".to_string(),
            EntityCategory::Process => "PROCESS".to_string(),
            EntityCategory::System => "SYSTEM".to_string(),
            EntityCategory::Risk => "RISK".to_string(),
            EntityCategory::Objective => "OBJECTIVE".to_string(),
            EntityCategory::AuditElement => "AUDIT_ELEMENT".to_string(),
            EntityCategory::Unknown => "UNKNOWN".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EntityExtractorConfig;

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
                noun_phrases: vec![],
                verb_phrases: vec![],
                question_words: vec![],
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
    async fn test_entity_extractor_creation() {
        let config = Arc::new(EntityExtractorConfig::default());
        let extractor = EntityExtractor::new(config).await;
        assert!(extractor.is_ok());
    }

    #[tokio::test]
    async fn test_entity_extraction() {
        let config = Arc::new(EntityExtractorConfig::default());
        let extractor = EntityExtractor::new(config).await.unwrap();
        
        let query = Query::new("What are the encryption requirements in PCI DSS 4.0?");
        let analysis = create_test_analysis();
        
        let result = extractor.extract(&query, &analysis).await;
        assert!(result.is_ok());
        
        let entities = result.unwrap();
        assert!(!entities.is_empty());
        
        // Should find PCI DSS standard
        let has_pci_dss = entities.iter().any(|e| {
            e.entity.text.to_lowercase().contains("pci dss")
        });
        assert!(has_pci_dss);
    }

    #[tokio::test]
    async fn test_pattern_matching() {
        let pattern = EntityPattern::new(r"(?i)PCI\s+DSS", "PCI DSS detection", 0.95);
        let matches = pattern.find_matches("What is PCI DSS 4.0?").unwrap();
        
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "PCI DSS");
        assert_eq!(matches[0].confidence, 0.95);
    }

    #[tokio::test]
    async fn test_entity_deduplication() {
        let config = Arc::new(EntityExtractorConfig::default());
        let extractor = EntityExtractor::new(config).await.unwrap();
        
        // Create overlapping entities
        let entities = vec![
            ExtractedEntity {
                entity: NamedEntity::new("PCI DSS".to_string(), "STANDARD".to_string(), 0, 7, 0.95),
                category: EntityCategory::Standard,
                metadata: EntityMetadata {
                    extraction_method: "test".to_string(),
                    extracted_at: Utc::now(),
                    context: "".to_string(),
                    normalization: None,
                    properties: HashMap::new(),
                },
                relationships: vec![],
            },
            ExtractedEntity {
                entity: NamedEntity::new("PCI".to_string(), "STANDARD".to_string(), 0, 3, 0.80),
                category: EntityCategory::Standard,
                metadata: EntityMetadata {
                    extraction_method: "test".to_string(),
                    extracted_at: Utc::now(),
                    context: "".to_string(),
                    normalization: None,
                    properties: HashMap::new(),
                },
                relationships: vec![],
            },
        ];
        
        let filtered = extractor.deduplicate_and_filter(entities).await.unwrap();
        assert_eq!(filtered.len(), 1); // Should keep only the higher confidence entity
        assert_eq!(filtered[0].entity.text, "PCI DSS");
    }

    #[tokio::test]
    async fn test_compliance_entity_extraction() {
        let config = Arc::new(EntityExtractorConfig::default());
        let extractor = ComplianceEntityExtractor::new(config).await.unwrap();
        
        let query = Query::new("What are the HIPAA requirements for encryption?");
        let analysis = create_test_analysis();
        
        let entities = extractor.extract(&query, &analysis).await.unwrap();
        
        let has_hipaa = entities.iter().any(|e| {
            e.entity.text.to_lowercase().contains("hipaa")
        });
        assert!(has_hipaa);
    }
}