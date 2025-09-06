//! Search strategy selection and optimization component
//!
//! This module provides intelligent search strategy selection based on query 
//! characteristics, historical performance, and adaptive learning mechanisms.
//! It optimizes search performance by selecting the most appropriate strategy
//! for each query type.

// use async_trait::async_trait; // Unused
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{info, instrument}; // removed unused warn
// use uuid::Uuid; // Unused

use crate::config::ProcessorConfig;
use crate::error::Result; // removed unused ProcessorError
use crate::query::Query;
use crate::types::*;

/// Strategy selector for determining optimal search strategies
pub struct StrategySelector {
    config: Arc<ProcessorConfig>,
    strategy_engine: StrategyEngine,
    performance_tracker: PerformanceTracker,
    adaptive_learner: Option<AdaptiveLearner>,
    strategy_cache: StrategyCache,
}

impl StrategySelector {
    /// Create a new strategy selector
    #[instrument(skip(config))]
    pub async fn new(config: Arc<ProcessorConfig>) -> Result<Self> {
        info!("Initializing Strategy Selector");
        
        let strategy_engine = StrategyEngine::new(&config.strategy_selector)?;
        let performance_tracker = PerformanceTracker::new(&config.strategy_selector)?;
        
        let adaptive_learner = if config.strategy_selector.enable_adaptive {
            Some(AdaptiveLearner::new(&config.strategy_selector).await?)
        } else {
            None
        };

        let strategy_cache = StrategyCache::new(config.performance.cache.size)?;
        
        Ok(Self {
            config,
            strategy_engine,
            performance_tracker,
            adaptive_learner,
            strategy_cache,
        })
    }

    /// Select optimal search strategy for query
    #[instrument(skip(self, query, analysis, intent, entities))]
    pub async fn select(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
        intent: &QueryIntent,
        entities: &[ExtractedEntity],
    ) -> Result<StrategySelection> {
        info!("Selecting search strategy for query: {}", query.id());
        
        // Check cache first
        let cache_key = self.generate_cache_key(query, analysis, intent);
        if let Some(cached_selection) = self.strategy_cache.get(&cache_key) {
            info!("Found cached strategy selection for query");
            return Ok(cached_selection);
        }

        // Analyze query characteristics
        let characteristics = self.analyze_query_characteristics(query, analysis, intent, entities).await?;
        
        // Get strategy recommendations
        let recommendations = self.strategy_engine.recommend_strategies(&characteristics).await?;
        
        // Apply adaptive learning if enabled
        let final_recommendations = if let Some(ref learner) = self.adaptive_learner {
            learner.refine_recommendations(recommendations, &characteristics).await?
        } else {
            recommendations
        };

        // Select primary strategy
        let selection = self.select_primary_strategy(final_recommendations, &characteristics).await?;
        
        // Cache the result
        self.strategy_cache.put(cache_key, selection.clone());

        info!("Selected strategy: {:?} with confidence {:.3}", 
              selection.strategy, selection.confidence);
        
        Ok(selection)
    }

    /// Generate cache key for strategy selection
    fn generate_cache_key(&self, query: &Query, analysis: &SemanticAnalysis, intent: &QueryIntent) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.text().hash(&mut hasher);
        analysis.confidence.to_bits().hash(&mut hasher);
        std::mem::discriminant(intent).hash(&mut hasher);
        
        format!("strategy_{:x}", hasher.finish())
    }

    /// Analyze query characteristics for strategy selection
    async fn analyze_query_characteristics(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
        intent: &QueryIntent,
        entities: &[ExtractedEntity],
    ) -> Result<QueryCharacteristics> {
        let text = query.text();
        let complexity_score = self.calculate_complexity_score(query, analysis, entities);
        
        let characteristics = QueryCharacteristics {
            query_length: text.len(),
            word_count: text.split_whitespace().count(),
            entity_count: entities.len(),
            entity_density: entities.len() as f64 / text.split_whitespace().count().max(1) as f64,
            intent: intent.clone(),
            intent_confidence: intent.confidence_weight(),
            complexity_score,
            has_technical_terms: self.has_technical_terms(entities),
            has_compliance_terms: self.has_compliance_terms(entities),
            has_comparison_indicators: self.has_comparison_indicators(text),
            semantic_complexity: self.calculate_semantic_complexity(analysis),
            syntactic_complexity: self.calculate_syntactic_complexity(analysis),
            requires_high_accuracy: intent == &QueryIntent::Compliance || 
                                   query.metadata().required_accuracy.unwrap_or(0.0) > 0.9,
            performance_requirements: PerformanceRequirements {
                max_latency: query.metadata().expected_response_time
                    .unwrap_or(Duration::from_secs(2)),
                min_accuracy: query.metadata().required_accuracy.unwrap_or(0.95),
                min_recall: 0.9,
                min_precision: 0.95,
            },
        };

        Ok(characteristics)
    }

    /// Calculate overall query complexity score
    fn calculate_complexity_score(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
        entities: &[ExtractedEntity],
    ) -> f64 {
        let mut score = 0.0;
        
        // Length complexity
        let length_factor = (query.text().len() as f64 / 100.0).min(1.0);
        score += length_factor * 0.2;
        
        // Entity complexity
        let entity_factor = (entities.len() as f64 / 10.0).min(1.0);
        score += entity_factor * 0.3;
        
        // Semantic complexity
        score += analysis.confidence * 0.3;
        
        // Syntactic complexity
        let syntactic_factor = analysis.syntactic_features.pos_tags.len() as f64 / 20.0;
        score += syntactic_factor.min(1.0) * 0.2;
        
        score.min(1.0)
    }

    /// Check for technical terms in entities
    fn has_technical_terms(&self, entities: &[ExtractedEntity]) -> bool {
        entities.iter().any(|e| {
            matches!(e.category, EntityCategory::TechnicalTerm | EntityCategory::System)
        })
    }

    /// Check for compliance terms in entities
    fn has_compliance_terms(&self, entities: &[ExtractedEntity]) -> bool {
        entities.iter().any(|e| {
            matches!(e.category, 
                EntityCategory::Standard | 
                EntityCategory::Requirement | 
                EntityCategory::Control |
                EntityCategory::AuditElement
            )
        })
    }

    /// Check for comparison indicators in text
    fn has_comparison_indicators(&self, text: &str) -> bool {
        let text_lower = text.to_lowercase();
        let comparison_words = ["compare", "versus", "vs", "difference", "between", "than"];
        comparison_words.iter().any(|word| text_lower.contains(word))
    }

    /// Calculate semantic complexity
    fn calculate_semantic_complexity(&self, analysis: &SemanticAnalysis) -> f64 {
        let semantic = &analysis.semantic_features;
        let mut complexity = 0.0;
        
        complexity += semantic.semantic_roles.len() as f64 * 0.1;
        complexity += semantic.coreferences.len() as f64 * 0.1;
        
        if semantic.sentiment.is_some() {
            complexity += 0.1;
        }
        
        complexity.min(1.0)
    }

    /// Calculate syntactic complexity
    fn calculate_syntactic_complexity(&self, analysis: &SemanticAnalysis) -> f64 {
        let syntactic = &analysis.syntactic_features;
        let mut complexity = 0.0;
        
        complexity += syntactic.named_entities.len() as f64 * 0.05;
        complexity += syntactic.noun_phrases.len() as f64 * 0.03;
        complexity += syntactic.verb_phrases.len() as f64 * 0.03;
        complexity += analysis.dependencies.len() as f64 * 0.02;
        
        complexity.min(1.0)
    }

    /// Select primary strategy from recommendations
    async fn select_primary_strategy(
        &self,
        recommendations: Vec<StrategyRecommendation>,
        characteristics: &QueryCharacteristics,
    ) -> Result<StrategySelection> {
        if recommendations.is_empty() {
            return Ok(StrategySelection {
                strategy: self.get_default_strategy(characteristics),
                confidence: 0.5,
                reasoning: "No specific recommendations, using default strategy".to_string(),
                expected_metrics: self.estimate_performance_metrics(&SearchStrategy::Vector {
                    model: "default".to_string(),
                    similarity_threshold: 0.7,
                    max_results: 10,
                }, characteristics),
                fallbacks: vec![],
                predictions: crate::types::StrategyPredictions {
                    latency: 0.5,
                    accuracy: 0.5,
                    resource_usage: 0.3,
                },
            });
        }

        let best_recommendation = &recommendations[0];
        
        // Generate reasoning
        let reasoning = self.generate_reasoning(best_recommendation, characteristics);
        
        // Estimate performance metrics
        let expected_metrics = self.estimate_performance_metrics(&best_recommendation.strategy, characteristics);
        
        // Get fallback strategies
        let fallbacks = recommendations.iter()
            .skip(1)
            .take(3)
            .map(|r| r.strategy.clone())
            .collect();

        Ok(StrategySelection {
            strategy: best_recommendation.strategy.clone(),
            confidence: best_recommendation.confidence,
            reasoning,
            expected_metrics,
            fallbacks,
            predictions: crate::types::StrategyPredictions {
                latency: 0.4,
                accuracy: best_recommendation.confidence,
                resource_usage: 0.3,
            },
        })
    }

    /// Get default strategy based on characteristics
    fn get_default_strategy(&self, characteristics: &QueryCharacteristics) -> SearchStrategy {
        match &self.config.strategy_selector.default_strategy {
            crate::config::DefaultStrategy::Vector => SearchStrategy::Vector {
                model: "default".to_string(),
                similarity_threshold: 0.7,
                max_results: 10,
            },
            crate::config::DefaultStrategy::Keyword => SearchStrategy::Keyword {
                algorithm: KeywordAlgorithm::BM25,
                boost_factors: HashMap::new(),
                fuzzy_matching: false,
            },
            crate::config::DefaultStrategy::Hybrid => SearchStrategy::Hybrid {
                strategies: vec![
                    SearchStrategy::Vector {
                        model: "default".to_string(),
                        similarity_threshold: 0.7,
                        max_results: 10,
                    },
                    SearchStrategy::Keyword {
                        algorithm: KeywordAlgorithm::BM25,
                        boost_factors: HashMap::new(),
                        fuzzy_matching: false,
                    },
                ],
                weights: {
                    let mut weights = HashMap::new();
                    weights.insert("vector".to_string(), 0.6);
                    weights.insert("keyword".to_string(), 0.4);
                    weights
                },
                combination_method: CombinationMethod::WeightedSum,
            },
            crate::config::DefaultStrategy::Semantic => SearchStrategy::Semantic {
                graph_traversal: GraphTraversal::BreadthFirst,
                relationship_types: vec!["related_to".to_string()],
                depth_limit: 3,
            },
        }
    }

    /// Generate reasoning for strategy selection
    fn generate_reasoning(&self, recommendation: &StrategyRecommendation, characteristics: &QueryCharacteristics) -> String {
        let mut reasons = Vec::new();
        
        match &characteristics.intent {
            QueryIntent::Factual => reasons.push("Factual query benefits from precise matching"),
            QueryIntent::Comparison => reasons.push("Comparison query requires comprehensive search"),
            QueryIntent::Compliance => reasons.push("Compliance query demands high accuracy"),
            QueryIntent::Procedural => reasons.push("Procedural query needs structured information"),
            _ => {}
        }

        if characteristics.has_technical_terms {
            reasons.push("Technical terms present, using semantic search");
        }

        if characteristics.complexity_score > 0.7 {
            reasons.push("High complexity query, employing advanced strategy");
        }

        if characteristics.requires_high_accuracy {
            reasons.push("High accuracy required, selecting reliable strategy");
        }

        if reasons.is_empty() {
            format!("Selected {} based on query analysis", strategy_name(&recommendation.strategy))
        } else {
            format!("{}: {}", strategy_name(&recommendation.strategy), reasons.join(", "))
        }
    }

    /// Estimate performance metrics for a strategy
    fn estimate_performance_metrics(&self, strategy: &SearchStrategy, characteristics: &QueryCharacteristics) -> PerformanceMetrics {
        let base_accuracy = match strategy {
            SearchStrategy::Vector { .. } => 0.85,
            SearchStrategy::Keyword { .. } => 0.75,
            SearchStrategy::Hybrid { .. } => 0.90,
            SearchStrategy::Semantic { .. } => 0.88,
            SearchStrategy::Adaptive { .. } => 0.92,
            SearchStrategy::HybridSearch => 0.90,
            SearchStrategy::SemanticSearch => 0.88,
            SearchStrategy::KeywordSearch => 0.75,
            SearchStrategy::VectorSimilarity => 0.85,
            SearchStrategy::ExactMatch => 0.95,
            SearchStrategy::NeuralSearch => 0.91,
        };

        let base_latency = match strategy {
            SearchStrategy::Vector { .. } => Duration::from_millis(150),
            SearchStrategy::Keyword { .. } => Duration::from_millis(50),
            SearchStrategy::Hybrid { .. } => Duration::from_millis(200),
            SearchStrategy::Semantic { .. } => Duration::from_millis(300),
            SearchStrategy::Adaptive { .. } => Duration::from_millis(250),
            SearchStrategy::HybridSearch => Duration::from_millis(200),
            SearchStrategy::SemanticSearch => Duration::from_millis(300),
            SearchStrategy::KeywordSearch => Duration::from_millis(50),
            SearchStrategy::VectorSimilarity => Duration::from_millis(150),
            SearchStrategy::ExactMatch => Duration::from_millis(30),
            SearchStrategy::NeuralSearch => Duration::from_millis(180),
        };

        // Adjust based on characteristics
        let accuracy_adjustment = if characteristics.has_compliance_terms { 0.05 } else { 0.0 };
        let latency_adjustment = characteristics.complexity_score * 50.0; // ms

        PerformanceMetrics {
            expected_accuracy: f64::min(base_accuracy + accuracy_adjustment, 1.0),
            expected_response_time: base_latency + Duration::from_millis(latency_adjustment as u64),
            expected_recall: base_accuracy * 0.95,
            expected_precision: base_accuracy * 1.05,
            resource_usage: ResourceUsage {
                cpu_usage: characteristics.complexity_score * 0.3,
                memory_usage: (characteristics.word_count * 1024) as u64,
                network_io: 0,
                disk_io: 0,
            },
        }
    }

    /// Record strategy performance for learning
    pub async fn record_performance(
        &mut self,
        selection: &StrategySelection,
        actual_performance: ActualPerformance,
    ) -> Result<()> {
        self.performance_tracker.record_performance(selection, actual_performance.clone()).await?;
        
        if let Some(ref mut learner) = self.adaptive_learner {
            learner.update_from_performance(selection, &actual_performance).await?;
        }

        Ok(())
    }
}

/// Strategy engine for generating recommendations
#[derive(Debug)]
pub struct StrategyEngine {
    strategy_rules: Vec<StrategyRule>,
    strategy_templates: HashMap<String, SearchStrategy>,
}

impl StrategyEngine {
    pub fn new(config: &crate::config::StrategyConfig) -> Result<Self> {
        let strategy_rules = Self::load_strategy_rules(config)?;
        let strategy_templates = Self::load_strategy_templates()?;
        
        Ok(Self {
            strategy_rules,
            strategy_templates,
        })
    }

    pub async fn recommend_strategies(
        &self,
        characteristics: &QueryCharacteristics,
    ) -> Result<Vec<StrategyRecommendation>> {
        let mut recommendations = Vec::new();

        // Apply strategy rules
        for rule in &self.strategy_rules {
            if let Some(recommendation) = rule.evaluate(characteristics)? {
                recommendations.push(recommendation);
            }
        }

        // Sort by confidence
        recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        // Remove duplicates and limit results
        recommendations.dedup_by(|a, b| std::mem::discriminant(&a.strategy) == std::mem::discriminant(&b.strategy));
        recommendations.truncate(5);

        Ok(recommendations)
    }

    fn load_strategy_rules(config: &crate::config::StrategyConfig) -> Result<Vec<StrategyRule>> {
        let mut rules = Vec::new();

        // Intent-based rules
        rules.push(StrategyRule {
            name: "factual_vector_search".to_string(),
            conditions: vec![
                RuleCondition::IntentEquals(QueryIntent::Factual),
                RuleCondition::ComplexityBelow(0.5),
            ],
            strategy_template: "vector_search".to_string(),
            confidence: 0.8,
            parameters: HashMap::new(),
        });

        rules.push(StrategyRule {
            name: "comparison_hybrid_search".to_string(),
            conditions: vec![
                RuleCondition::IntentEquals(QueryIntent::Comparison),
            ],
            strategy_template: "hybrid_search".to_string(),
            confidence: 0.9,
            parameters: HashMap::new(),
        });

        rules.push(StrategyRule {
            name: "compliance_high_accuracy".to_string(),
            conditions: vec![
                RuleCondition::IntentEquals(QueryIntent::Compliance),
                RuleCondition::RequiresHighAccuracy,
            ],
            strategy_template: "semantic_search".to_string(),
            confidence: 0.95,
            parameters: HashMap::new(),
        });

        rules.push(StrategyRule {
            name: "procedural_structured_search".to_string(),
            conditions: vec![
                RuleCondition::IntentEquals(QueryIntent::Procedural),
            ],
            strategy_template: "adaptive_search".to_string(),
            confidence: 0.85,
            parameters: HashMap::new(),
        });

        // Complexity-based rules
        rules.push(StrategyRule {
            name: "simple_keyword_search".to_string(),
            conditions: vec![
                RuleCondition::ComplexityBelow(0.3),
                RuleCondition::EntityCountBelow(3),
            ],
            strategy_template: "keyword_search".to_string(),
            confidence: 0.7,
            parameters: HashMap::new(),
        });

        rules.push(StrategyRule {
            name: "complex_adaptive_search".to_string(),
            conditions: vec![
                RuleCondition::ComplexityAbove(0.8),
            ],
            strategy_template: "adaptive_search".to_string(),
            confidence: 0.9,
            parameters: HashMap::new(),
        });

        // Technical content rules
        rules.push(StrategyRule {
            name: "technical_semantic_search".to_string(),
            conditions: vec![
                RuleCondition::HasTechnicalTerms,
            ],
            strategy_template: "semantic_search".to_string(),
            confidence: 0.8,
            parameters: HashMap::new(),
        });

        Ok(rules)
    }

    fn load_strategy_templates() -> Result<HashMap<String, SearchStrategy>> {
        let mut templates = HashMap::new();

        templates.insert("vector_search".to_string(), SearchStrategy::Vector {
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            similarity_threshold: 0.7,
            max_results: 10,
        });

        templates.insert("keyword_search".to_string(), SearchStrategy::Keyword {
            algorithm: KeywordAlgorithm::BM25,
            boost_factors: {
                let mut factors = HashMap::new();
                factors.insert("title".to_string(), 2.0);
                factors.insert("headings".to_string(), 1.5);
                factors
            },
            fuzzy_matching: true,
        });

        templates.insert("hybrid_search".to_string(), SearchStrategy::Hybrid {
            strategies: vec![
                SearchStrategy::Vector {
                    model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                    similarity_threshold: 0.65,
                    max_results: 15,
                },
                SearchStrategy::Keyword {
                    algorithm: KeywordAlgorithm::BM25,
                    boost_factors: HashMap::new(),
                    fuzzy_matching: true,
                },
            ],
            weights: {
                let mut weights = HashMap::new();
                weights.insert("vector".to_string(), 0.7);
                weights.insert("keyword".to_string(), 0.3);
                weights
            },
            combination_method: CombinationMethod::RankFusion,
        });

        templates.insert("semantic_search".to_string(), SearchStrategy::Semantic {
            graph_traversal: GraphTraversal::BreadthFirst,
            relationship_types: vec![
                "related_to".to_string(),
                "part_of".to_string(),
                "implements".to_string(),
            ],
            depth_limit: 2,
        });

        templates.insert("adaptive_search".to_string(), SearchStrategy::Adaptive {
            base_strategy: Box::new(SearchStrategy::Hybrid {
                strategies: vec![
                    SearchStrategy::Vector {
                        model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                        similarity_threshold: 0.7,
                        max_results: 10,
                    },
                    SearchStrategy::Keyword {
                        algorithm: KeywordAlgorithm::BM25,
                        boost_factors: HashMap::new(),
                        fuzzy_matching: false,
                    },
                ],
                weights: {
                    let mut weights = HashMap::new();
                    weights.insert("vector".to_string(), 0.6);
                    weights.insert("keyword".to_string(), 0.4);
                    weights
                },
                combination_method: CombinationMethod::WeightedSum,
            }),
            adaptation_rules: vec![
                AdaptationRule {
                    name: "boost_accuracy".to_string(),
                    condition: AdaptationCondition::HistoricalPerformance(0.85),
                    action: AdaptationAction::ModifyParameters({
                        let mut params = HashMap::new();
                        params.insert("similarity_threshold".to_string(), 0.8);
                        params
                    }),
                    priority: 1,
                },
            ],
            learning_enabled: true,
        });

        Ok(templates)
    }
}

/// Strategy rule for rule-based recommendations
#[derive(Debug, Clone)]
pub struct StrategyRule {
    pub name: String,
    pub conditions: Vec<RuleCondition>,
    pub strategy_template: String,
    pub confidence: f64,
    pub parameters: HashMap<String, f64>,
}

impl StrategyRule {
    pub fn evaluate(&self, characteristics: &QueryCharacteristics) -> Result<Option<StrategyRecommendation>> {
        let mut matching_conditions = 0;
        
        for condition in &self.conditions {
            if condition.matches(characteristics) {
                matching_conditions += 1;
            }
        }

        if matching_conditions == self.conditions.len() {
            // All conditions match
            Ok(Some(StrategyRecommendation {
                strategy: self.create_strategy_from_template(characteristics)?,
                confidence: self.confidence,
                reasoning: format!("Rule '{}' matched all conditions", self.name),
                performance_estimate: None,
            }))
        } else if matching_conditions > 0 {
            // Partial match
            let partial_confidence = self.confidence * (matching_conditions as f64 / self.conditions.len() as f64);
            if partial_confidence > 0.3 {
                Ok(Some(StrategyRecommendation {
                    strategy: self.create_strategy_from_template(characteristics)?,
                    confidence: partial_confidence,
                    reasoning: format!("Rule '{}' partially matched ({}/{})", 
                                     self.name, matching_conditions, self.conditions.len()),
                    performance_estimate: None,
                }))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn create_strategy_from_template(&self, _characteristics: &QueryCharacteristics) -> Result<SearchStrategy> {
        // This would create a strategy from the template with parameters
        // For now, return a basic strategy
        match self.strategy_template.as_str() {
            "vector_search" => Ok(SearchStrategy::Vector {
                model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                similarity_threshold: self.parameters.get("similarity_threshold").copied().unwrap_or(0.7),
                max_results: self.parameters.get("max_results").copied().unwrap_or(10.0) as usize,
            }),
            "keyword_search" => Ok(SearchStrategy::Keyword {
                algorithm: KeywordAlgorithm::BM25,
                boost_factors: HashMap::new(),
                fuzzy_matching: self.parameters.get("fuzzy_matching").copied().unwrap_or(0.0) > 0.5,
            }),
            "hybrid_search" => Ok(SearchStrategy::Hybrid {
                strategies: vec![
                    SearchStrategy::Vector {
                        model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                        similarity_threshold: 0.7,
                        max_results: 10,
                    },
                    SearchStrategy::Keyword {
                        algorithm: KeywordAlgorithm::BM25,
                        boost_factors: HashMap::new(),
                        fuzzy_matching: true,
                    },
                ],
                weights: {
                    let mut weights = HashMap::new();
                    weights.insert("vector".to_string(), 0.6);
                    weights.insert("keyword".to_string(), 0.4);
                    weights
                },
                combination_method: CombinationMethod::RankFusion,
            }),
            "semantic_search" => Ok(SearchStrategy::Semantic {
                graph_traversal: GraphTraversal::BreadthFirst,
                relationship_types: vec!["related_to".to_string()],
                depth_limit: 2,
            }),
            _ => Ok(SearchStrategy::Vector {
                model: "default".to_string(),
                similarity_threshold: 0.7,
                max_results: 10,
            }),
        }
    }
}

/// Rule condition for strategy selection
#[derive(Debug, Clone)]
pub enum RuleCondition {
    IntentEquals(QueryIntent),
    ComplexityAbove(f64),
    ComplexityBelow(f64),
    EntityCountAbove(usize),
    EntityCountBelow(usize),
    HasTechnicalTerms,
    HasComplianceTerms,
    RequiresHighAccuracy,
    MaxLatencyBelow(Duration),
}

impl RuleCondition {
    pub fn matches(&self, characteristics: &QueryCharacteristics) -> bool {
        match self {
            RuleCondition::IntentEquals(intent) => &characteristics.intent == intent,
            RuleCondition::ComplexityAbove(threshold) => characteristics.complexity_score > *threshold,
            RuleCondition::ComplexityBelow(threshold) => characteristics.complexity_score < *threshold,
            RuleCondition::EntityCountAbove(threshold) => characteristics.entity_count > *threshold,
            RuleCondition::EntityCountBelow(threshold) => characteristics.entity_count < *threshold,
            RuleCondition::HasTechnicalTerms => characteristics.has_technical_terms,
            RuleCondition::HasComplianceTerms => characteristics.has_compliance_terms,
            RuleCondition::RequiresHighAccuracy => characteristics.requires_high_accuracy,
            RuleCondition::MaxLatencyBelow(threshold) => characteristics.performance_requirements.max_latency < *threshold,
        }
    }
}

/// Strategy recommendation from engine
#[derive(Debug, Clone)]
pub struct StrategyRecommendation {
    pub strategy: SearchStrategy,
    pub confidence: f64,
    pub reasoning: String,
    pub performance_estimate: Option<PerformanceMetrics>,
}

/// Query characteristics for strategy selection
#[derive(Debug, Clone)]
pub struct QueryCharacteristics {
    pub query_length: usize,
    pub word_count: usize,
    pub entity_count: usize,
    pub entity_density: f64,
    pub intent: QueryIntent,
    pub intent_confidence: f64,
    pub complexity_score: f64,
    pub has_technical_terms: bool,
    pub has_compliance_terms: bool,
    pub has_comparison_indicators: bool,
    pub semantic_complexity: f64,
    pub syntactic_complexity: f64,
    pub requires_high_accuracy: bool,
    pub performance_requirements: PerformanceRequirements,
}

/// Performance requirements for queries
#[derive(Debug, Clone)]
pub struct PerformanceRequirements {
    pub max_latency: Duration,
    pub min_accuracy: f64,
    pub min_recall: f64,
    pub min_precision: f64,
}

/// Performance tracker for strategy effectiveness
#[derive(Debug)]
pub struct PerformanceTracker {
    performance_history: HashMap<String, Vec<PerformanceRecord>>,
    strategy_metrics: HashMap<String, StrategyMetrics>,
}

impl PerformanceTracker {
    pub fn new(_config: &crate::config::StrategyConfig) -> Result<Self> {
        Ok(Self {
            performance_history: HashMap::new(),
            strategy_metrics: HashMap::new(),
        })
    }

    pub async fn record_performance(
        &mut self,
        selection: &StrategySelection,
        actual: ActualPerformance,
    ) -> Result<()> {
        let strategy_key = strategy_name(&selection.strategy);
        
        let record = PerformanceRecord {
            timestamp: chrono::Utc::now(),
            expected_metrics: selection.expected_metrics.clone(),
            actual_performance: actual.clone(),
            accuracy_delta: actual.accuracy - selection.expected_metrics.expected_accuracy,
            latency_delta: actual.latency.as_secs_f64() - selection.expected_metrics.expected_response_time.as_secs_f64(),
        };

        // Add to history
        self.performance_history
            .entry(strategy_key.clone())
            .or_insert_with(Vec::new)
            .push(record.clone());

        // Update aggregated metrics
        self.update_strategy_metrics(&strategy_key, &record);

        Ok(())
    }

    fn update_strategy_metrics(&mut self, strategy_key: &str, record: &PerformanceRecord) {
        let metrics = self.strategy_metrics
            .entry(strategy_key.to_string())
            .or_insert_with(StrategyMetrics::default);

        metrics.total_queries += 1;
        metrics.total_accuracy += record.actual_performance.accuracy;
        metrics.total_latency += record.actual_performance.latency;
        
        if record.actual_performance.accuracy >= 0.9 {
            metrics.high_accuracy_count += 1;
        }

        // Calculate running averages
        metrics.average_accuracy = metrics.total_accuracy / metrics.total_queries as f64;
        metrics.average_latency = Duration::from_secs_f64(
            metrics.total_latency.as_secs_f64() / metrics.total_queries as f64
        );
        
        metrics.accuracy_variance += record.accuracy_delta * record.accuracy_delta;
        metrics.latency_variance += record.latency_delta * record.latency_delta;
    }

    pub fn get_strategy_performance(&self, strategy: &SearchStrategy) -> Option<&StrategyMetrics> {
        let key = strategy_name(strategy);
        self.strategy_metrics.get(&key)
    }
}

/// Performance record for a strategy execution
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub expected_metrics: PerformanceMetrics,
    pub actual_performance: ActualPerformance,
    pub accuracy_delta: f64,
    pub latency_delta: f64,
}

/// Actual performance metrics after execution
#[derive(Debug, Clone)]
pub struct ActualPerformance {
    pub accuracy: f64,
    pub recall: f64,
    pub precision: f64,
    pub latency: Duration,
    pub resource_usage: ResourceUsage,
    pub result_count: usize,
    pub user_satisfaction: Option<f64>,
}

/// Strategy performance metrics
#[derive(Debug, Clone, Default)]
pub struct StrategyMetrics {
    pub total_queries: usize,
    pub high_accuracy_count: usize,
    pub total_accuracy: f64,
    pub average_accuracy: f64,
    pub total_latency: Duration,
    pub average_latency: Duration,
    pub accuracy_variance: f64,
    pub latency_variance: f64,
    pub last_updated: Option<chrono::DateTime<chrono::Utc>>,
}

/// Adaptive learner for strategy optimization
#[derive(Debug)]
pub struct AdaptiveLearner {
    learning_rate: f64,
    strategy_weights: HashMap<String, f64>,
    performance_threshold: f64,
}

impl AdaptiveLearner {
    pub async fn new(config: &crate::config::StrategyConfig) -> Result<Self> {
        Ok(Self {
            learning_rate: 0.01,
            strategy_weights: HashMap::new(),
            performance_threshold: config.thresholds.accuracy_threshold,
        })
    }

    pub async fn refine_recommendations(
        &self,
        mut recommendations: Vec<StrategyRecommendation>,
        _characteristics: &QueryCharacteristics,
    ) -> Result<Vec<StrategyRecommendation>> {
        // Adjust confidence based on learned weights
        for recommendation in &mut recommendations {
            let strategy_key = strategy_name(&recommendation.strategy);
            if let Some(&weight) = self.strategy_weights.get(&strategy_key) {
                recommendation.confidence *= weight;
            }
        }

        // Re-sort by adjusted confidence
        recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        Ok(recommendations)
    }

    pub async fn update_from_performance(
        &mut self,
        selection: &StrategySelection,
        actual: &ActualPerformance,
    ) -> Result<()> {
        let strategy_key = strategy_name(&selection.strategy);
        let current_weight = self.strategy_weights.get(&strategy_key).copied().unwrap_or(1.0);
        
        // Calculate performance score
        let performance_score = (actual.accuracy + actual.precision + actual.recall) / 3.0;
        
        // Update weight based on performance
        let weight_adjustment = if performance_score >= self.performance_threshold {
            self.learning_rate * (performance_score - self.performance_threshold)
        } else {
            -self.learning_rate * (self.performance_threshold - performance_score)
        };

        let new_weight = (current_weight + weight_adjustment).clamp(0.1, 2.0);
        self.strategy_weights.insert(strategy_key, new_weight);

        Ok(())
    }
}

/// Strategy cache for performance optimization
#[derive(Debug)]
pub struct StrategyCache {
    cache: Mutex<HashMap<String, StrategySelection>>,
    max_size: usize,
}

impl StrategyCache {
    pub fn new(max_size: usize) -> Result<Self> {
        Ok(Self {
            cache: Mutex::new(HashMap::new()),
            max_size,
        })
    }

    pub fn get(&self, key: &str) -> Option<StrategySelection> {
        self.cache.lock().unwrap().get(key).cloned()
    }

    pub fn put(&self, key: String, value: StrategySelection) {
        let mut cache = self.cache.lock().unwrap();
        if cache.len() >= self.max_size {
            // Simple eviction: remove first entry
            if let Some(first_key) = cache.keys().next().cloned() {
                cache.remove(&first_key);
            }
        }
        cache.insert(key, value);
    }
}

/// Helper function to get strategy name for metrics
fn strategy_name(strategy: &SearchStrategy) -> String {
    match strategy {
        SearchStrategy::Vector { model, .. } => format!("vector_{}", model),
        SearchStrategy::Keyword { algorithm, .. } => format!("keyword_{:?}", algorithm),
        SearchStrategy::Hybrid { .. } => "hybrid".to_string(),
        SearchStrategy::Semantic { .. } => "semantic".to_string(),
        SearchStrategy::Adaptive { .. } => "adaptive".to_string(),
        SearchStrategy::HybridSearch => "hybrid_search".to_string(),
        SearchStrategy::SemanticSearch => "semantic_search".to_string(),
        SearchStrategy::KeywordSearch => "keyword_search".to_string(),
        SearchStrategy::VectorSimilarity => "vector_similarity".to_string(),
        SearchStrategy::ExactMatch => "exact_match".to_string(),
        SearchStrategy::NeuralSearch => "neural_search".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProcessorConfig;

    #[tokio::test]
    async fn test_strategy_selector_creation() {
        let config = Arc::new(ProcessorConfig::default());
        let selector = StrategySelector::new(config).await;
        assert!(selector.is_ok());
    }

    #[tokio::test]
    async fn test_strategy_selection() {
        let config = Arc::new(ProcessorConfig::default());
        let selector = StrategySelector::new(config).await.unwrap();
        
        let query = crate::query::Query::new("What are the PCI DSS encryption requirements?");
        let analysis = create_test_analysis();
        let intent = QueryIntent::Compliance;
        let entities = vec![];
        
        let selection = selector.select(&query, &analysis, &intent, &entities).await.unwrap();
        assert!(selection.confidence > 0.0);
        assert!(!selection.reasoning.is_empty());
    }

    #[tokio::test]
    async fn test_comparison_query_strategy() -> Result<(), QueryError> {
        let config = Arc::new(ProcessorConfig::default());
        let selector = StrategySelector::new(config).await.unwrap();
        
        let query = crate::query::Query::new("Compare PCI DSS 3.2.1 versus 4.0")?;
        let analysis = create_test_analysis();
        let intent = QueryIntent::Comparison;
        let entities = vec![];
        
        let selection = selector.select(&query, &analysis, &intent, &entities).await.unwrap();
        
        // Comparison queries should prefer hybrid or semantic search
        match selection.strategy {
            SearchStrategy::Hybrid { .. } | SearchStrategy::Semantic { .. } => {},
            _ => {
                return Err(QueryError::InvalidStrategy(
                    format!("Unexpected strategy for comparison query: {:?}", selection.strategy)
                ));
            }
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_performance_tracking() {
        let config = crate::config::StrategyConfig::default();
        let mut tracker = PerformanceTracker::new(&config).unwrap();
        
        let selection = StrategySelection {
            strategy: SearchStrategy::Vector {
                model: "test".to_string(),
                similarity_threshold: 0.7,
                max_results: 10,
            },
            confidence: 0.8,
            reasoning: "Test".to_string(),
            expected_metrics: PerformanceMetrics {
                expected_accuracy: 0.9,
                expected_response_time: Duration::from_millis(100),
                expected_recall: 0.85,
                expected_precision: 0.9,
                resource_usage: ResourceUsage::default(),
            },
            fallbacks: vec![],
            predictions: crate::types::StrategyPredictions {
                latency: 0.1,
                accuracy: 0.9,
                resource_usage: 0.3,
            },
        };
        
        let actual = ActualPerformance {
            accuracy: 0.85,
            recall: 0.8,
            precision: 0.88,
            latency: Duration::from_millis(120),
            resource_usage: ResourceUsage::default(),
            result_count: 5,
            user_satisfaction: Some(0.9),
        };
        
        let result = tracker.record_performance(&selection, actual).await;
        assert!(result.is_ok());
        
        let metrics = tracker.get_strategy_performance(&selection.strategy);
        assert!(metrics.is_some());
        assert_eq!(metrics.unwrap().total_queries, 1);
    }

    fn create_test_analysis() -> SemanticAnalysis {
        use std::time::Duration;
        
        SemanticAnalysis {
            syntactic_features: SyntacticFeatures {
                pos_tags: vec![],
                named_entities: vec![
                    NamedEntity::new(
                        "PCI DSS".to_string(),
                        "STANDARD".to_string(),
                        0,
                        7,
                        0.95,
                    ),
                ],
                noun_phrases: vec![],
                verb_phrases: vec![],
                question_words: vec!["What".to_string()],
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
            timestamp: chrono::Utc::now(),
            processing_time: Duration::from_millis(100),
        }
    }
}