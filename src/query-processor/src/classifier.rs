//! Intent classification component for determining query intentions and patterns
//!
//! This module provides comprehensive intent classification capabilities including
//! rule-based classification, neural network integration, ensemble methods,
//! and multi-label classification support.

// use async_trait::async_trait; // Unused
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{info, instrument, warn};
// use uuid::Uuid; // Unused

use crate::config::ProcessorConfig;
use crate::error::{ProcessorError, Result};
use crate::query::Query;
use crate::types::*;

#[cfg(feature = "neural")]
use ruv_fann::NeuralNet;

/// Intent classifier for determining query intentions
#[derive(Debug)]
pub struct IntentClassifier {
    config: Arc<ProcessorConfig>,
    rule_classifier: RuleBasedClassifier,
    neural_classifier: Option<NeuralClassifier>,
    ensemble_classifier: Option<EnsembleClassifier>,
    feature_extractor: FeatureExtractor,
    classification_cache: ClassificationCache,
}

impl IntentClassifier {
    /// Create a new intent classifier
    #[instrument(skip(config))]
    pub async fn new(config: Arc<ProcessorConfig>) -> Result<Self> {
        info!("Initializing Intent Classifier");
        
        let rule_classifier = RuleBasedClassifier::new(&config.intent_classifier)?;
        
        let neural_classifier = if config.enable_neural {
            Some(NeuralClassifier::new(&config.intent_classifier).await?)
        } else {
            None
        };

        let ensemble_classifier = if matches!(config.intent_classifier.model_type, crate::config::ClassificationModel::Ensemble) {
            Some(EnsembleClassifier::new(&config.intent_classifier, &rule_classifier, neural_classifier.as_ref()).await?)
        } else {
            None
        };

        let feature_extractor = FeatureExtractor::new(&config.intent_classifier)?;
        let classification_cache = ClassificationCache::new(config.performance.cache.size)?;
        
        Ok(Self {
            config,
            rule_classifier,
            neural_classifier,
            ensemble_classifier,
            feature_extractor,
            classification_cache,
        })
    }

    /// Classify query intent
    #[instrument(skip(self, query, analysis))]
    pub async fn classify(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
    ) -> Result<IntentClassification> {
        info!("Classifying intent for query: {}", query.id());
        
        // Check cache first
        let cache_key = self.generate_cache_key(query, analysis);
        if let Some(cached_classification) = self.classification_cache.get(&cache_key) {
            info!("Found cached classification for query");
            return Ok(cached_classification);
        }

        // Extract features for classification
        let features = self.feature_extractor.extract_features(query, analysis).await?;
        
        // Perform classification based on configured model type
        let classification = match &self.config.intent_classifier.model_type {
            crate::config::ClassificationModel::RuleBased => {
                self.rule_classifier.classify(query, analysis, &features).await?
            },
            crate::config::ClassificationModel::Neural => {
                if let Some(ref neural) = self.neural_classifier {
                    neural.classify(query, analysis, &features).await?
                } else {
                    return Err(ProcessorError::IntentClassificationFailed {
                        reason: "Neural classifier not initialized".to_string(),
                    });
                }
            },
            crate::config::ClassificationModel::Ensemble => {
                if let Some(ref ensemble) = self.ensemble_classifier {
                    ensemble.classify(query, analysis, &features).await?
                } else {
                    return Err(ProcessorError::IntentClassificationFailed {
                        reason: "Ensemble classifier not initialized".to_string(),
                    });
                }
            },
            crate::config::ClassificationModel::External => {
                self.external_classify(query, analysis, &features).await?
            },
        };

        // Validate classification confidence
        if classification.confidence < self.config.intent_classifier.confidence_threshold {
            warn!(
                "Classification confidence {} below threshold {}",
                classification.confidence, 
                self.config.intent_classifier.confidence_threshold
            );
        }

        // Cache the result
        self.classification_cache.put(cache_key, classification.clone());

        info!("Classified intent as {:?} with confidence {:.3}", 
              classification.primary_intent, classification.confidence);
        
        Ok(classification)
    }

    /// Generate cache key for classification
    fn generate_cache_key(&self, query: &Query, analysis: &SemanticAnalysis) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.text().hash(&mut hasher);
        analysis.confidence.to_bits().hash(&mut hasher);
        
        format!("classification_{:x}", hasher.finish())
    }

    /// External classification using API
    async fn external_classify(
        &self,
        _query: &Query,
        _analysis: &SemanticAnalysis,
        _features: &ClassificationFeatures,
    ) -> Result<IntentClassification> {
        // Placeholder for external API integration
        Err(ProcessorError::IntentClassificationFailed {
            reason: "External classification not implemented".to_string(),
        })
    }
}

/// Rule-based intent classifier
#[derive(Debug)]
pub struct RuleBasedClassifier {
    rules: Vec<ClassificationRule>,
    intent_patterns: HashMap<QueryIntent, Vec<IntentPattern>>,
}

impl RuleBasedClassifier {
    pub fn new(config: &crate::config::IntentClassifierConfig) -> Result<Self> {
        let rules = Self::load_classification_rules()?;
        let intent_patterns = Self::load_intent_patterns()?;
        
        Ok(Self {
            rules,
            intent_patterns,
        })
    }

    pub async fn classify(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
        features: &ClassificationFeatures,
    ) -> Result<IntentClassification> {
        let mut intent_scores: HashMap<QueryIntent, f64> = HashMap::new();
        let text = query.text().to_lowercase();

        // Apply pattern-based classification
        for (intent, patterns) in &self.intent_patterns {
            let mut max_score = 0.0;
            
            for pattern in patterns {
                let score = pattern.calculate_score(&text, analysis, features)?;
                max_score = f64::max(max_score, score);
            }
            
            if max_score > 0.0 {
                intent_scores.insert(intent.clone(), max_score);
            }
        }

        // Apply rule-based classification
        for rule in &self.rules {
            if let Some(score) = rule.evaluate(&text, analysis, features)? {
                let current_score = intent_scores.get(&rule.intent).copied().unwrap_or(0.0);
                intent_scores.insert(rule.intent.clone(), current_score.max(score));
            }
        }

        // Store probabilities before moving intent_scores
        let probabilities = intent_scores.clone();
        
        // Determine primary and secondary intents
        let mut sorted_intents: Vec<_> = intent_scores.into_iter().collect();
        sorted_intents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if sorted_intents.is_empty() {
            return Ok(IntentClassification {
                primary_intent: QueryIntent::Unknown,
                secondary_intents: vec![],
                confidence: 0.3,
                probabilities: HashMap::new(),
                method: ClassificationMethod::RuleBased,
                features: features.feature_names.clone(),
            });
        }

        let primary_intent = sorted_intents[0].0.clone();
        let primary_confidence = sorted_intents[0].1;
        
        let secondary_intents = sorted_intents
            .into_iter()
            .skip(1)
            .take(2)
            .map(|(intent, _)| intent)
            .collect();

        Ok(IntentClassification {
            primary_intent,
            secondary_intents,
            confidence: primary_confidence,
            probabilities,
            method: ClassificationMethod::RuleBased,
            features: features.feature_names.clone(),
        })
    }

    fn load_classification_rules() -> Result<Vec<ClassificationRule>> {
        let mut rules = Vec::new();

        // Question word rules
        rules.push(ClassificationRule {
            name: "what_factual".to_string(),
            intent: QueryIntent::Factual,
            conditions: vec![
                RuleCondition::ContainsPattern(r"\bwhat\s+(is|are|was|were)\b".to_string()),
            ],
            confidence: 0.8,
        });

        rules.push(ClassificationRule {
            name: "definition_request".to_string(),
            intent: QueryIntent::Definition,
            conditions: vec![
                RuleCondition::ContainsPattern(r"\b(define|definition|meaning|means)\b".to_string()),
            ],
            confidence: 0.9,
        });

        rules.push(ClassificationRule {
            name: "how_procedural".to_string(),
            intent: QueryIntent::Procedural,
            conditions: vec![
                RuleCondition::ContainsPattern(r"\bhow\s+(do|to|can|should)\b".to_string()),
            ],
            confidence: 0.85,
        });

        rules.push(ClassificationRule {
            name: "compare_comparison".to_string(),
            intent: QueryIntent::Comparison,
            conditions: vec![
                RuleCondition::ContainsPattern(r"\b(compare|comparison|difference|versus|vs|between)\b".to_string()),
            ],
            confidence: 0.9,
        });

        rules.push(ClassificationRule {
            name: "summarize_summary".to_string(),
            intent: QueryIntent::Summary,
            conditions: vec![
                RuleCondition::ContainsPattern(r"\b(summary|summarize|overview|outline)\b".to_string()),
            ],
            confidence: 0.85,
        });

        rules.push(ClassificationRule {
            name: "compliance_check".to_string(),
            intent: QueryIntent::Compliance,
            conditions: vec![
                RuleCondition::ContainsPattern(r"\b(compliance|compliant|requirements?|standards?|regulations?)\b".to_string()),
                RuleCondition::ContainsEntity("STANDARD".to_string()),
            ],
            confidence: 0.9,
        });

        rules.push(ClassificationRule {
            name: "why_causal".to_string(),
            intent: QueryIntent::Causal,
            conditions: vec![
                RuleCondition::ContainsPattern(r"\b(why|because|reason|cause|due to)\b".to_string()),
            ],
            confidence: 0.8,
        });

        rules.push(ClassificationRule {
            name: "when_temporal".to_string(),
            intent: QueryIntent::Temporal,
            conditions: vec![
                RuleCondition::ContainsPattern(r"\b(when|timeline|schedule|date|time)\b".to_string()),
            ],
            confidence: 0.8,
        });

        Ok(rules)
    }

    fn load_intent_patterns() -> Result<HashMap<QueryIntent, Vec<IntentPattern>>> {
        let mut patterns = HashMap::new();

        // Factual patterns
        patterns.insert(QueryIntent::Factual, vec![
            IntentPattern {
                name: "direct_question".to_string(),
                keywords: vec!["what", "which", "where", "who"].iter().map(|s| s.to_string()).collect(),
                structures: vec!["question_word + verb + object".to_string()],
                confidence: 0.8,
                boost_factors: HashMap::new(),
            },
        ]);

        // Procedural patterns
        patterns.insert(QueryIntent::Procedural, vec![
            IntentPattern {
                name: "how_to".to_string(),
                keywords: vec!["how", "steps", "process", "procedure"].iter().map(|s| s.to_string()).collect(),
                structures: vec!["how + infinitive".to_string()],
                confidence: 0.85,
                boost_factors: {
                    let mut map = HashMap::new();
                    map.insert("implementation".to_string(), 1.2);
                    map.insert("setup".to_string(), 1.1);
                    map
                },
            },
        ]);

        // Analytical patterns
        patterns.insert(QueryIntent::Analytical, vec![
            IntentPattern {
                name: "analysis_request".to_string(),
                keywords: vec!["analyze", "analysis", "evaluate", "assess"].iter().map(|s| s.to_string()).collect(),
                structures: vec!["analyze + object".to_string()],
                confidence: 0.8,
                boost_factors: HashMap::new(),
            },
        ]);

        Ok(patterns)
    }
}

/// Classification rule for rule-based classifier
#[derive(Debug, Clone)]
pub struct ClassificationRule {
    pub name: String,
    pub intent: QueryIntent,
    pub conditions: Vec<RuleCondition>,
    pub confidence: f64,
}

impl ClassificationRule {
    pub fn evaluate(
        &self,
        text: &str,
        analysis: &SemanticAnalysis,
        _features: &ClassificationFeatures,
    ) -> Result<Option<f64>> {
        let mut matching_conditions = 0;
        
        for condition in &self.conditions {
            if condition.matches(text, analysis)? {
                matching_conditions += 1;
            }
        }

        if matching_conditions == self.conditions.len() {
            Ok(Some(self.confidence))
        } else if matching_conditions > 0 {
            // Partial match - reduced confidence
            let partial_confidence = self.confidence * (matching_conditions as f64 / self.conditions.len() as f64);
            Ok(Some(partial_confidence))
        } else {
            Ok(None)
        }
    }
}

/// Rule condition for classification
#[derive(Debug, Clone)]
pub enum RuleCondition {
    ContainsPattern(String),
    ContainsEntity(String),
    HasSentiment(String),
    QuestionWordCount(usize),
    TextLength(usize, usize), // min, max
}

impl RuleCondition {
    pub fn matches(&self, text: &str, analysis: &SemanticAnalysis) -> Result<bool> {
        match self {
            RuleCondition::ContainsPattern(pattern) => {
                let regex = regex::Regex::new(pattern).map_err(|e| {
                    ProcessorError::IntentClassificationFailed {
                        reason: format!("Invalid regex pattern: {}", e),
                    }
                })?;
                Ok(regex.is_match(text))
            },
            RuleCondition::ContainsEntity(entity_type) => {
                Ok(analysis.syntactic_features.named_entities
                    .iter()
                    .any(|entity| entity.entity_type == *entity_type))
            },
            RuleCondition::HasSentiment(sentiment_label) => {
                Ok(analysis.semantic_features.sentiment
                    .as_ref()
                    .map(|s| s.label == *sentiment_label)
                    .unwrap_or(false))
            },
            RuleCondition::QuestionWordCount(min_count) => {
                Ok(analysis.syntactic_features.question_words.len() >= *min_count)
            },
            RuleCondition::TextLength(min_len, max_len) => {
                let len = text.len();
                Ok(len >= *min_len && len <= *max_len)
            },
        }
    }
}

/// Intent pattern for pattern-based classification
#[derive(Debug, Clone)]
pub struct IntentPattern {
    pub name: String,
    pub keywords: Vec<String>,
    pub structures: Vec<String>,
    pub confidence: f64,
    pub boost_factors: HashMap<String, f64>,
}

impl IntentPattern {
    pub fn calculate_score(
        &self,
        text: &str,
        _analysis: &SemanticAnalysis,
        _features: &ClassificationFeatures,
    ) -> Result<f64> {
        let mut score = 0.0;
        let text_lower = text.to_lowercase();

        // Check for keywords
        let mut keyword_matches = 0;
        for keyword in &self.keywords {
            if text_lower.contains(&keyword.to_lowercase()) {
                keyword_matches += 1;
                
                // Apply boost factors if available
                if let Some(boost) = self.boost_factors.get(keyword) {
                    score += self.confidence * boost;
                } else {
                    score += self.confidence;
                }
            }
        }

        if keyword_matches > 0 {
            // Normalize by number of keywords
            score = score / self.keywords.len() as f64;
            
            // Bonus for multiple keyword matches
            if keyword_matches > 1 {
                score *= 1.0 + (0.1 * (keyword_matches - 1) as f64);
            }
        }

        Ok(score.min(1.0))
    }
}

/// Neural network classifier using ruv-FANN integration
#[derive(Debug)]
pub struct NeuralClassifier {
    model_config: NeuralModelConfig,
    feature_dimension: usize,
    #[cfg(feature = "neural")]
    neural_net: Option<NeuralNet>,
    #[cfg(not(feature = "neural"))]
    _phantom: std::marker::PhantomData<()>,
}

impl NeuralClassifier {
    /// Get pattern recognition results using ruv-FANN for semantic analysis
    #[cfg(feature = "neural")]
    pub async fn recognize_patterns(&self, features: &ClassificationFeatures) -> Result<Vec<PatternMatch>> {
        info!("Running pattern recognition with ruv-FANN");
        
        let input_vector = self.features_to_vector(features)?;
        let output = self.run_inference(&input_vector).await?;
        
        // Convert neural output to pattern matches
        let mut patterns = Vec::new();
        
        // Analyze output patterns for semantic features
        for (i, &score) in output.iter().enumerate() {
            if score > 0.1 { // Threshold for pattern detection
                let pattern = match i {
                    0 => PatternMatch {
                        pattern_type: "factual_query".to_string(),
                        confidence: score as f64,
                        semantic_weight: 0.9,
                    },
                    1 => PatternMatch {
                        pattern_type: "comparison_query".to_string(),
                        confidence: score as f64,
                        semantic_weight: 0.8,
                    },
                    2 => PatternMatch {
                        pattern_type: "analytical_query".to_string(),
                        confidence: score as f64,
                        semantic_weight: 0.85,
                    },
                    3 => PatternMatch {
                        pattern_type: "summary_query".to_string(),
                        confidence: score as f64,
                        semantic_weight: 0.7,
                    },
                    4 => PatternMatch {
                        pattern_type: "procedural_query".to_string(),
                        confidence: score as f64,
                        semantic_weight: 0.8,
                    },
                    5 => PatternMatch {
                        pattern_type: "definition_query".to_string(),
                        confidence: score as f64,
                        semantic_weight: 0.9,
                    },
                    6 => PatternMatch {
                        pattern_type: "causal_query".to_string(),
                        confidence: score as f64,
                        semantic_weight: 0.75,
                    },
                    7 => PatternMatch {
                        pattern_type: "temporal_query".to_string(),
                        confidence: score as f64,
                        semantic_weight: 0.7,
                    },
                    8 => PatternMatch {
                        pattern_type: "compliance_query".to_string(),
                        confidence: score as f64,
                        semantic_weight: 0.95,
                    },
                    _ => continue,
                };
                patterns.push(pattern);
            }
        }
        
        // Sort by confidence
        patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(patterns)
    }
    
    /// Train the neural network with provided training data
    #[cfg(feature = "neural")]
    pub async fn train(&mut self, training_data: &[(Vec<f32>, Vec<f32>)]) -> Result<()> {
        if let Some(ref neural_net) = self.neural_net {
            info!("Training ruv-FANN neural network with {} samples", training_data.len());
            
            // Prepare training data in ruv-FANN format
            let inputs: Vec<Vec<f32>> = training_data.iter().map(|(input, _)| input.clone()).collect();
            let outputs: Vec<Vec<f32>> = training_data.iter().map(|(_, output)| output.clone()).collect();
            
            // Configure training parameters
            neural_net.set_learning_rate(self.model_config.learning_rate)
                .map_err(|e| ProcessorError::IntentClassificationFailed {
                    reason: format!("Failed to set learning rate: {:?}", e),
                })?;
            
            // Set training algorithm
            let training_algo = match self.model_config.training_algorithm.as_str() {
                "incremental" => ruv_fann::TrainingAlgorithm::Incremental,
                "batch" => ruv_fann::TrainingAlgorithm::Batch,
                "rprop" => ruv_fann::TrainingAlgorithm::Rprop,
                "quickprop" => ruv_fann::TrainingAlgorithm::Quickprop,
                _ => ruv_fann::TrainingAlgorithm::Rprop, // Default
            };
            
            neural_net.set_training_algorithm(training_algo)
                .map_err(|e| ProcessorError::IntentClassificationFailed {
                    reason: format!("Failed to set training algorithm: {:?}", e),
                })?;
            
            // Train the network
            neural_net.train_on_data(
                &inputs,
                &outputs,
                self.model_config.max_epochs,
                0, // Reports every epoch (0 = no reports during training)
                self.model_config.desired_error
            ).map_err(|e| ProcessorError::IntentClassificationFailed {
                reason: format!("Training failed: {:?}", e),
            })?;
            
            info!("Neural network training completed successfully");
            Ok(())
        } else {
            Err(ProcessorError::IntentClassificationFailed {
                reason: "Neural network not initialized".to_string(),
            })
        }
    }
    
    /// Save the trained neural network to file
    #[cfg(feature = "neural")]
    pub async fn save_model(&self, path: &str) -> Result<()> {
        if let Some(ref neural_net) = self.neural_net {
            info!("Saving ruv-FANN neural network to: {}", path);
            neural_net.save(path)
                .map_err(|e| ProcessorError::IntentClassificationFailed {
                    reason: format!("Failed to save neural network to {}: {:?}", path, e),
                })?;
            info!("Neural network saved successfully");
            Ok(())
        } else {
            Err(ProcessorError::IntentClassificationFailed {
                reason: "Neural network not initialized".to_string(),
            })
        }
    }
    
    #[cfg(feature = "neural")]
    fn initialize_neural_network(config: &NeuralModelConfig) -> Result<Option<NeuralNet>> {
        info!("Initializing ruv-FANN neural network");
        
        match &config.model_path {
            Some(path) => {
                // Load pre-trained model
                info!("Loading neural network from: {}", path);
                let neural_net = NeuralNet::new_from_file(path)
                    .map_err(|e| ProcessorError::IntentClassificationFailed {
                        reason: format!("Failed to load neural network from {}: {:?}", path, e),
                    })?;
                Ok(Some(neural_net))
            },
            None => {
                // Create new neural network with specified architecture
                info!("Creating new neural network with architecture: input={}, hidden={:?}, output={}", 
                      config.input_size, config.hidden_layers, config.output_size);
                
                let mut layers = vec![config.input_size];
                layers.extend(&config.hidden_layers);
                layers.push(config.output_size);
                
                let neural_net = NeuralNet::new(&layers)
                    .map_err(|e| ProcessorError::IntentClassificationFailed {
                        reason: format!("Failed to create neural network: {:?}", e),
                    })?;
                
                // Configure activation functions
                neural_net.set_activation_function_hidden(
                    Self::parse_activation_function(&config.activation_function)
                ).map_err(|e| ProcessorError::IntentClassificationFailed {
                    reason: format!("Failed to set activation function: {:?}", e),
                })?;
                
                neural_net.set_activation_function_output(
                    ruv_fann::ActivationFunc::Linear
                ).map_err(|e| ProcessorError::IntentClassificationFailed {
                    reason: format!("Failed to set output activation function: {:?}", e),
                })?;
                
                Ok(Some(neural_net))
            }
        }
    }
    
    #[cfg(feature = "neural")]
    fn parse_activation_function(activation: &str) -> ruv_fann::ActivationFunc {
        match activation.to_lowercase().as_str() {
            "sigmoid" => ruv_fann::ActivationFunc::Sigmoid,
            "sigmoid_symmetric" => ruv_fann::ActivationFunc::SigmoidSymmetric,
            "linear" => ruv_fann::ActivationFunc::Linear,
            "threshold" => ruv_fann::ActivationFunc::Threshold,
            "threshold_symmetric" => ruv_fann::ActivationFunc::ThresholdSymmetric,
            "gaussian" => ruv_fann::ActivationFunc::Gaussian,
            "gaussian_symmetric" => ruv_fann::ActivationFunc::GaussianSymmetric,
            "elliot" => ruv_fann::ActivationFunc::Elliot,
            "elliot_symmetric" => ruv_fann::ActivationFunc::ElliotSymmetric,
            "sin_symmetric" => ruv_fann::ActivationFunc::SinSymmetric,
            "cos_symmetric" => ruv_fann::ActivationFunc::CosSymmetric,
            "sin" => ruv_fann::ActivationFunc::Sin,
            "cos" => ruv_fann::ActivationFunc::Cos,
            _ => ruv_fann::ActivationFunc::Sigmoid, // Default fallback
        }
    }
    pub async fn new(config: &crate::config::IntentClassifierConfig) -> Result<Self> {
        info!("Initializing Neural Classifier with ruv-FANN");
        
        let model_config = NeuralModelConfig::default();
        let feature_dimension = 128; // Standard feature vector size
        
        #[cfg(feature = "neural")]
        let neural_net = Self::initialize_neural_network(&model_config)?;
        
        Ok(Self {
            model_config,
            feature_dimension,
            #[cfg(feature = "neural")]
            neural_net,
            #[cfg(not(feature = "neural"))]
            _phantom: std::marker::PhantomData,
        })
    }

    pub async fn classify(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
        features: &ClassificationFeatures,
    ) -> Result<IntentClassification> {
        info!("Running neural classification with ruv-FANN for query: {}", query.id());
        
        // Convert features to neural network input
        let input_vector = self.features_to_vector(features)?;
        
        // Validate input vector dimension
        if input_vector.len() != self.feature_dimension {
            return Err(ProcessorError::IntentClassificationFailed {
                reason: format!(
                    "Input vector dimension mismatch: expected {}, got {}", 
                    self.feature_dimension, input_vector.len()
                ),
            });
        }
        
        // Run neural network inference using ruv-FANN
        let output = self.run_inference(&input_vector).await?;
        
        // Convert output to intent classification
        self.output_to_classification(output, features)
    }

    fn features_to_vector(&self, features: &ClassificationFeatures) -> Result<Vec<f32>> {
        let mut vector = vec![0.0; self.feature_dimension];
        
        // Simple feature encoding - in practice this would be more sophisticated
        vector[0] = features.question_word_count as f32;
        vector[1] = features.text_length as f32 / 1000.0; // Normalized
        vector[2] = features.entity_count as f32;
        vector[3] = features.sentiment_score;
        vector[4] = features.complexity_score;
        
        // One-hot encode some categorical features
        if features.has_question_mark { vector[10] = 1.0; }
        if features.has_comparison_words { vector[11] = 1.0; }
        if features.has_compliance_terms { vector[12] = 1.0; }
        
        Ok(vector)
    }

    async fn run_inference(&self, input: &[f32]) -> Result<Vec<f32>> {
        #[cfg(feature = "neural")]
        {
            if let Some(ref neural_net) = self.neural_net {
                // Use ruv-FANN for actual neural network inference
                let output = neural_net.run(input)
                    .map_err(|e| ProcessorError::IntentClassificationFailed {
                        reason: format!("ruv-FANN inference failed: {:?}", e),
                    })?;
                Ok(output)
            } else {
                return Err(ProcessorError::IntentClassificationFailed {
                    reason: "Neural network not properly initialized".to_string(),
                });
            }
        }
        
        #[cfg(not(feature = "neural"))]
        {
            // Fallback when neural feature is not enabled
            Err(ProcessorError::IntentClassificationFailed {
                reason: "Neural classification requires 'neural' feature to be enabled".to_string(),
            })
        }
    }

    fn output_to_classification(
        &self,
        output: Vec<f32>,
        features: &ClassificationFeatures,
    ) -> Result<IntentClassification> {
        let intents = vec![
            QueryIntent::Factual,
            QueryIntent::Comparison,
            QueryIntent::Analytical,
            QueryIntent::Summary,
            QueryIntent::Procedural,
            QueryIntent::Definition,
            QueryIntent::Causal,
            QueryIntent::Temporal,
            QueryIntent::Compliance,
            QueryIntent::Unknown,
        ];

        // Find the intent with highest probability
        let mut max_prob = 0.0;
        let mut max_index = 0;
        
        for (i, &prob) in output.iter().enumerate() {
            if prob > max_prob {
                max_prob = prob;
                max_index = i;
            }
        }

        let primary_intent = intents.get(max_index).cloned().unwrap_or(QueryIntent::Unknown);
        
        // Create probability distribution
        let mut probabilities = HashMap::new();
        for (intent, &prob) in intents.iter().zip(output.iter()) {
            if prob > 0.01 { // Only include significant probabilities
                probabilities.insert(intent.clone(), prob as f64);
            }
        }

        // Find secondary intents
        let mut intent_probs: Vec<_> = intents.iter()
            .zip(output.iter())
            .enumerate()
            .filter(|(i, _)| *i != max_index)
            .map(|(_, (intent, &prob))| (intent.clone(), prob))
            .collect();
        
        intent_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let secondary_intents = intent_probs
            .into_iter()
            .take(2)
            .filter(|(_, prob)| *prob > 0.1)
            .map(|(intent, _)| intent)
            .collect();

        Ok(IntentClassification {
            primary_intent,
            secondary_intents,
            confidence: max_prob as f64,
            probabilities,
            method: ClassificationMethod::Neural,
            features: features.feature_names.clone(),
        })
    }
}

/// Neural model configuration for ruv-FANN integration
#[derive(Debug, Clone)]
pub struct NeuralModelConfig {
    pub model_path: Option<String>,
    pub input_size: usize,
    pub hidden_layers: Vec<usize>,
    pub output_size: usize,
    pub activation_function: String,
    pub learning_rate: f32,
    pub training_algorithm: String,
    pub max_epochs: u32,
    pub desired_error: f32,
}

impl Default for NeuralModelConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            input_size: 128,
            hidden_layers: vec![64, 32],
            output_size: 10, // Number of intent classes
            activation_function: "sigmoid".to_string(),
            learning_rate: 0.01,
            training_algorithm: "rprop".to_string(),
            max_epochs: 1000,
            desired_error: 0.001,
        }
    }
}

/// Ensemble classifier combining multiple approaches
#[derive(Debug)]
pub struct EnsembleClassifier {
    rule_weight: f64,
    neural_weight: f64,
    combination_method: EnsembleCombination,
}

impl EnsembleClassifier {
    pub async fn new(
        _config: &crate::config::IntentClassifierConfig,
        _rule_classifier: &RuleBasedClassifier,
        _neural_classifier: Option<&NeuralClassifier>,
    ) -> Result<Self> {
        Ok(Self {
            rule_weight: 0.6,
            neural_weight: 0.4,
            combination_method: EnsembleCombination::WeightedAverage,
        })
    }

    pub async fn classify(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
        features: &ClassificationFeatures,
    ) -> Result<IntentClassification> {
        // This would combine multiple classifier outputs
        // For now, return a placeholder
        Ok(IntentClassification {
            primary_intent: QueryIntent::Unknown,
            secondary_intents: vec![],
            confidence: 0.5,
            probabilities: HashMap::new(),
            method: ClassificationMethod::Ensemble,
            features: features.feature_names.clone(),
        })
    }
}

/// Ensemble combination methods
#[derive(Debug, Clone)]
pub enum EnsembleCombination {
    WeightedAverage,
    Voting,
    Stacking,
}

/// Feature extractor for intent classification
#[derive(Debug)]
pub struct FeatureExtractor {
    stop_words: std::collections::HashSet<String>,
}

impl FeatureExtractor {
    pub fn new(config: &crate::config::IntentClassifierConfig) -> Result<Self> {
        // Load stop words
        let stop_words = vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"
        ].into_iter().map(|s| s.to_string()).collect();
        
        Ok(Self {
            stop_words,
        })
    }

    pub async fn extract_features(
        &self,
        query: &Query,
        analysis: &SemanticAnalysis,
    ) -> Result<ClassificationFeatures> {
        let text = query.text();
        let text_lower = text.to_lowercase();

        let features = ClassificationFeatures {
            text_length: text.len(),
            word_count: text.split_whitespace().count(),
            sentence_count: text.split('.').count(),
            question_word_count: analysis.syntactic_features.question_words.len(),
            entity_count: analysis.syntactic_features.named_entities.len(),
            has_question_mark: text.contains('?'),
            has_comparison_words: self.has_comparison_words(&text_lower),
            has_compliance_terms: self.has_compliance_terms(&text_lower),
            sentiment_score: analysis.semantic_features.sentiment
                .as_ref()
                .map(|s| s.score as f32)
                .unwrap_or(0.0),
            complexity_score: analysis.confidence as f32,
            pos_tag_distribution: self.calculate_pos_distribution(&analysis.syntactic_features.pos_tags),
            dependency_features: self.extract_dependency_features(&analysis.dependencies),
            semantic_features: self.extract_semantic_features(&analysis.semantic_features),
            feature_names: vec![
                "text_length".to_string(),
                "word_count".to_string(),
                "question_words".to_string(),
                "entities".to_string(),
                "sentiment".to_string(),
                "complexity".to_string(),
            ],
        };

        Ok(features)
    }

    fn has_comparison_words(&self, text: &str) -> bool {
        let comparison_words = ["compare", "comparison", "versus", "vs", "difference", "between", "than"];
        comparison_words.iter().any(|word| text.contains(word))
    }

    fn has_compliance_terms(&self, text: &str) -> bool {
        let compliance_terms = ["compliance", "requirement", "standard", "regulation", "audit", "control"];
        compliance_terms.iter().any(|word| text.contains(word))
    }

    fn calculate_pos_distribution(&self, pos_tags: &[PosTag]) -> HashMap<String, f32> {
        let mut distribution = HashMap::new();
        let total = pos_tags.len() as f32;
        
        if total == 0.0 {
            return distribution;
        }

        for tag in pos_tags {
            *distribution.entry(tag.tag.clone()).or_insert(0.0) += 1.0 / total;
        }

        distribution
    }

    fn extract_dependency_features(&self, dependencies: &[Dependency]) -> DependencyFeatures {
        DependencyFeatures {
            dependency_count: dependencies.len(),
            has_subject_verb: dependencies.iter().any(|d| d.relation == "nsubj"),
            has_direct_object: dependencies.iter().any(|d| d.relation == "dobj"),
            has_question_structure: dependencies.iter().any(|d| d.relation.contains("wh")),
        }
    }

    fn extract_semantic_features(&self, semantic: &SemanticFeatures) -> ExtractedSemanticFeatures {
        ExtractedSemanticFeatures {
            semantic_role_count: semantic.semantic_roles.len(),
            coreference_chain_count: semantic.coreferences.len(),
            has_sentiment: semantic.sentiment.is_some(),
            embedding_dimension: semantic.similarity_vectors.len(),
        }
    }
}

/// Classification features for intent classification
#[derive(Debug, Clone)]
pub struct ClassificationFeatures {
    pub text_length: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub question_word_count: usize,
    pub entity_count: usize,
    pub has_question_mark: bool,
    pub has_comparison_words: bool,
    pub has_compliance_terms: bool,
    pub sentiment_score: f32,
    pub complexity_score: f32,
    pub pos_tag_distribution: HashMap<String, f32>,
    pub dependency_features: DependencyFeatures,
    pub semantic_features: ExtractedSemanticFeatures,
    pub feature_names: Vec<String>,
}

/// Dependency-based features
#[derive(Debug, Clone)]
pub struct DependencyFeatures {
    pub dependency_count: usize,
    pub has_subject_verb: bool,
    pub has_direct_object: bool,
    pub has_question_structure: bool,
}

/// Semantic features for classification
#[derive(Debug, Clone)]
pub struct ExtractedSemanticFeatures {
    pub semantic_role_count: usize,
    pub coreference_chain_count: usize,
    pub has_sentiment: bool,
    pub embedding_dimension: usize,
}

/// Classification cache for performance
#[derive(Debug)]
pub struct ClassificationCache {
    cache: Mutex<std::collections::HashMap<String, IntentClassification>>,
    max_size: usize,
}

impl ClassificationCache {
    pub fn new(max_size: usize) -> Result<Self> {
        Ok(Self {
            cache: Mutex::new(HashMap::new()),
            max_size,
        })
    }

    pub fn get(&self, key: &str) -> Option<IntentClassification> {
        self.cache.lock().unwrap().get(key).cloned()
    }

    pub fn put(&self, key: String, value: IntentClassification) {
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

/// Pattern match result from neural pattern recognition
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_type: String,
    pub confidence: f64,
    pub semantic_weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProcessorConfig;

    #[tokio::test]
    async fn test_intent_classifier_creation() {
        let config = Arc::new(ProcessorConfig::default());
        let classifier = IntentClassifier::new(config).await;
        assert!(classifier.is_ok());
    }

    #[tokio::test]
    async fn test_rule_based_classification() {
        let config = Arc::new(ProcessorConfig::default());
        let classifier = IntentClassifier::new(config).await.unwrap();
        
        let query = crate::query::Query::new("What is PCI DSS?");
        let analysis = create_test_analysis();
        
        let classification = classifier.classify(&query, &analysis).await.unwrap();
        assert!(matches!(classification.primary_intent, QueryIntent::Factual | QueryIntent::Definition));
        assert!(classification.confidence > 0.0);
    }

    #[tokio::test]
    async fn test_procedural_intent() {
        let config = Arc::new(ProcessorConfig::default());
        let classifier = IntentClassifier::new(config).await.unwrap();
        
        let query = crate::query::Query::new("How do I implement PCI DSS requirements?");
        let analysis = create_test_analysis();
        
        let classification = classifier.classify(&query, &analysis).await.unwrap();
        assert_eq!(classification.primary_intent, QueryIntent::Procedural);
    }

    #[tokio::test]
    async fn test_comparison_intent() {
        let config = Arc::new(ProcessorConfig::default());
        let classifier = IntentClassifier::new(config).await.unwrap();
        
        let query = crate::query::Query::new("Compare PCI DSS 3.2.1 versus 4.0");
        let analysis = create_test_analysis();
        
        let classification = classifier.classify(&query, &analysis).await.unwrap();
        assert_eq!(classification.primary_intent, QueryIntent::Comparison);
    }

    #[tokio::test]
    async fn test_feature_extraction() {
        let config = crate::config::IntentClassifierConfig::default();
        let extractor = FeatureExtractor::new(&config).unwrap();
        
        let query = crate::query::Query::new("What are the PCI DSS encryption requirements?");
        let analysis = create_test_analysis();
        
        let features = extractor.extract_features(&query, &analysis).await.unwrap();
        assert!(features.text_length > 0);
        assert!(features.word_count > 0);
        assert!(!features.feature_names.is_empty());
    }

    fn create_test_analysis() -> SemanticAnalysis {
        use std::time::Duration;
        
        SemanticAnalysis {
            syntactic_features: SyntacticFeatures {
                pos_tags: vec![
                    PosTag {
                        token: "What".to_string(),
                        tag: "WP".to_string(),
                        start: 0,
                        end: 4,
                        confidence: 0.9,
                    },
                ],
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
                sentiment: Some(Sentiment::default()),
                similarity_vectors: vec![0.1, 0.2, 0.3],
            },
            dependencies: vec![],
            topics: vec![],
            confidence: 0.8,
            timestamp: chrono::Utc::now(),
            processing_time: Duration::from_millis(100),
        }
    }
}