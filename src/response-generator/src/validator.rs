//! Multi-stage validation pipeline for response accuracy and quality assurance

use crate::error::{Result, ResponseError};
use crate::{GenerationRequest, IntermediateResponse};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{Duration, Instant};
use tracing::{debug, instrument, warn, error};

/// Multi-stage validator for response quality assurance
#[derive(Debug)]
pub struct Validator {
    /// Validation configuration
    config: ValidationConfig,
    
    /// Ordered list of validation layers
    layers: Vec<Box<dyn ValidationLayer>>,
    
    /// Performance metrics
    metrics: ValidationMetrics,
}

/// Configuration for validation system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Minimum overall confidence threshold
    pub min_confidence_threshold: f64,
    
    /// Maximum validation time allowed
    pub max_validation_time: Duration,
    
    /// Enable strict validation mode
    pub strict_mode: bool,
    
    /// Parallel validation execution
    pub parallel_validation: bool,
    
    /// Layer-specific configurations
    pub layer_configs: HashMap<String, serde_json::Value>,
}

/// Validation result from a specific layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Layer that performed validation
    pub layer_name: String,
    
    /// Whether validation passed
    pub passed: bool,
    
    /// Confidence score from this layer
    pub confidence: f64,
    
    /// Detailed findings
    pub findings: Vec<ValidationFinding>,
    
    /// Processing time for this layer
    pub processing_time: Duration,
    
    /// Segment-specific results
    pub segment_start: usize,
    
    /// Segment end position
    pub segment_end: usize,
}

/// Individual validation finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFinding {
    /// Severity of the finding
    pub severity: ValidationSeverity,
    
    /// Type of validation issue
    pub finding_type: ValidationFindingType,
    
    /// Description of the finding
    pub message: String,
    
    /// Position in text where issue was found
    pub position: Option<usize>,
    
    /// Suggested resolution
    pub suggestion: Option<String>,
    
    /// Confidence in this finding
    pub confidence: f64,
}

/// Severity levels for validation findings
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum ValidationSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Types of validation findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationFindingType {
    /// Factual accuracy issues
    FactualAccuracy,
    
    /// Citation and source issues
    CitationIssue,
    
    /// Coherence and flow problems
    CoherenceIssue,
    
    /// Completeness problems
    CompletenessIssue,
    
    /// Bias detection
    BiasDetection,
    
    /// Hallucination detection
    HallucinationDetection,
    
    /// Consistency issues
    ConsistencyIssue,
    
    /// Language and grammar issues
    LanguageIssue,
}

/// Validation layer trait
#[async_trait]
pub trait ValidationLayer: Send + Sync + std::fmt::Debug {
    /// Name of the validation layer
    fn name(&self) -> &str;
    
    /// Validate response and return results
    async fn validate(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
        config: &ValidationConfig,
    ) -> Result<ValidationResult>;
    
    /// Get layer priority (higher number = higher priority)
    fn priority(&self) -> u8 {
        50
    }
    
    /// Whether this layer can run in parallel
    fn supports_parallel(&self) -> bool {
        true
    }
}

/// Performance metrics for validation
#[derive(Debug, Clone, Default)]
pub struct ValidationMetrics {
    /// Total validation calls
    pub total_validations: u64,
    
    /// Average validation time
    pub avg_validation_time: Duration,
    
    /// Validation pass rate
    pub pass_rate: f64,
    
    /// Layer performance metrics
    pub layer_metrics: HashMap<String, LayerMetrics>,
}

/// Performance metrics per layer
#[derive(Debug, Clone, Default)]
pub struct LayerMetrics {
    /// Total executions
    pub executions: u64,
    
    /// Average execution time
    pub avg_time: Duration,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Average confidence score
    pub avg_confidence: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.7,
            max_validation_time: Duration::from_millis(50),
            strict_mode: false,
            parallel_validation: true,
            layer_configs: HashMap::new(),
        }
    }
}

impl Validator {
    /// Create a new validator with configuration
    pub fn new(config: ValidationConfig) -> Self {
        let mut validator = Self {
            config,
            layers: Vec::new(),
            metrics: ValidationMetrics::default(),
        };
        
        // Initialize default validation layers
        validator.add_default_layers();
        validator
    }

    /// Add a validation layer
    pub fn add_layer(&mut self, layer: Box<dyn ValidationLayer>) {
        self.layers.push(layer);
        self.sort_layers_by_priority();
    }

    /// Validate response through all layers
    #[instrument(skip(self, response, request))]
    pub async fn validate(
        &mut self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
    ) -> Result<Vec<ValidationResult>> {
        let start_time = Instant::now();
        debug!("Starting validation with {} layers", self.layers.len());

        let results = if self.config.parallel_validation {
            self.validate_parallel(response, request).await?
        } else {
            self.validate_sequential(response, request).await?
        };

        let validation_time = start_time.elapsed();
        
        // Check validation time constraint
        if validation_time > self.config.max_validation_time {
            warn!("Validation took {}ms, exceeding target of {}ms", 
                  validation_time.as_millis(), 
                  self.config.max_validation_time.as_millis());
        }

        // Update metrics
        self.update_metrics(&results, validation_time).await?;

        // Check overall confidence threshold
        let overall_confidence = self.calculate_overall_confidence(&results);
        if overall_confidence < self.config.min_confidence_threshold {
            if self.config.strict_mode {
                return Err(ResponseError::ValidationFailed {
                    details: format!("Overall confidence {} below threshold {}", 
                                   overall_confidence, 
                                   self.config.min_confidence_threshold)
                });
            } else {
                warn!("Validation confidence {} below threshold {}", 
                      overall_confidence, 
                      self.config.min_confidence_threshold);
            }
        }

        debug!("Validation completed in {}ms with confidence {:.2}", 
               validation_time.as_millis(), overall_confidence);
        
        Ok(results)
    }

    /// Get validation pass count
    pub fn get_pass_count(&self) -> usize {
        self.layers.len()
    }

    /// Get current validation metrics
    pub fn get_metrics(&self) -> &ValidationMetrics {
        &self.metrics
    }

    /// Add default validation layers
    fn add_default_layers(&mut self) {
        self.layers.push(Box::new(FactualAccuracyLayer::new()));
        self.layers.push(Box::new(CitationValidationLayer::new()));
        self.layers.push(Box::new(CoherenceValidationLayer::new()));
        self.layers.push(Box::new(CompletenessValidationLayer::new()));
        self.layers.push(Box::new(BiasDetectionLayer::new()));
        self.layers.push(Box::new(HallucinationDetectionLayer::new()));
        self.layers.push(Box::new(ConsistencyValidationLayer::new()));
        
        self.sort_layers_by_priority();
    }

    /// Sort layers by priority
    fn sort_layers_by_priority(&mut self) {
        self.layers.sort_by(|a, b| b.priority().cmp(&a.priority()));
    }

    /// Validate using parallel execution
    async fn validate_parallel(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
    ) -> Result<Vec<ValidationResult>> {
        use futures::future::join_all;

        let futures: Vec<_> = self.layers
            .iter()
            .filter(|layer| layer.supports_parallel())
            .map(|layer| async {
                layer.validate(response, request, &self.config).await
            })
            .collect();

        let results = join_all(futures).await;
        let mut validation_results = Vec::new();

        for result in results {
            match result {
                Ok(validation_result) => validation_results.push(validation_result),
                Err(e) => {
                    error!("Validation layer failed: {}", e);
                    if self.config.strict_mode {
                        return Err(e);
                    }
                }
            }
        }

        // Run non-parallel layers sequentially
        for layer in &self.layers {
            if !layer.supports_parallel() {
                match layer.validate(response, request, &self.config).await {
                    Ok(result) => validation_results.push(result),
                    Err(e) => {
                        error!("Sequential validation layer failed: {}", e);
                        if self.config.strict_mode {
                            return Err(e);
                        }
                    }
                }
            }
        }

        Ok(validation_results)
    }

    /// Validate using sequential execution
    async fn validate_sequential(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
    ) -> Result<Vec<ValidationResult>> {
        let mut results = Vec::new();

        for layer in &self.layers {
            match layer.validate(response, request, &self.config).await {
                Ok(result) => {
                    results.push(result);
                }
                Err(e) => {
                    error!("Validation layer {} failed: {}", layer.name(), e);
                    if self.config.strict_mode {
                        return Err(e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Calculate overall confidence from all validation results
    fn calculate_overall_confidence(&self, results: &[ValidationResult]) -> f64 {
        if results.is_empty() {
            return 0.0;
        }

        // Weighted average based on layer priority and pass status
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for (result, layer) in results.iter().zip(&self.layers) {
            let weight = layer.priority() as f64 / 100.0;
            let confidence = if result.passed {
                result.confidence
            } else {
                result.confidence * 0.5 // Penalty for failed validation
            };
            
            weighted_sum += confidence * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Update performance metrics
    async fn update_metrics(
        &mut self,
        results: &[ValidationResult],
        total_time: Duration,
    ) -> Result<()> {
        self.metrics.total_validations += 1;
        
        // Update average validation time
        let current_avg_ms = self.metrics.avg_validation_time.as_millis() as f64;
        let new_time_ms = total_time.as_millis() as f64;
        let count = self.metrics.total_validations as f64;
        
        let new_avg_ms = (current_avg_ms * (count - 1.0) + new_time_ms) / count;
        self.metrics.avg_validation_time = Duration::from_millis(new_avg_ms as u64);

        // Update pass rate
        let passed = results.iter().filter(|r| r.passed).count() as f64;
        let total = results.len() as f64;
        let current_pass_rate = if total > 0.0 { passed / total } else { 0.0 };
        
        self.metrics.pass_rate = (self.metrics.pass_rate * (count - 1.0) + current_pass_rate) / count;

        // Update layer-specific metrics
        for result in results {
            let layer_metrics = self.metrics.layer_metrics
                .entry(result.layer_name.clone())
                .or_insert_with(LayerMetrics::default);

            layer_metrics.executions += 1;
            
            // Update average time
            let current_avg = layer_metrics.avg_time.as_millis() as f64;
            let new_time = result.processing_time.as_millis() as f64;
            let exec_count = layer_metrics.executions as f64;
            
            let new_avg = (current_avg * (exec_count - 1.0) + new_time) / exec_count;
            layer_metrics.avg_time = Duration::from_millis(new_avg as u64);

            // Update success rate
            let success = if result.passed { 1.0 } else { 0.0 };
            layer_metrics.success_rate = (layer_metrics.success_rate * (exec_count - 1.0) + success) / exec_count;

            // Update average confidence
            layer_metrics.avg_confidence = (layer_metrics.avg_confidence * (exec_count - 1.0) + result.confidence) / exec_count;
        }

        Ok(())
    }
}

// Validation layer implementations

/// Factual accuracy validation layer
#[derive(Debug)]
struct FactualAccuracyLayer {
    name: String,
}

impl FactualAccuracyLayer {
    fn new() -> Self {
        Self {
            name: "factual_accuracy".to_string(),
        }
    }
}

#[async_trait]
impl ValidationLayer for FactualAccuracyLayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u8 {
        90 // High priority
    }

    async fn validate(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
        _config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut findings = Vec::new();

        // Check for factual accuracy indicators
        let content = &response.content.to_lowercase();
        
        // Look for potentially inaccurate claims
        if content.contains("always") || content.contains("never") || content.contains("100%") {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::FactualAccuracy,
                message: "Absolute statements detected - may indicate overgeneralization".to_string(),
                position: None,
                suggestion: Some("Consider qualifying absolute statements".to_string()),
                confidence: 0.7,
            });
        }

        // Check for numerical claims without sources
        let has_numbers = content.chars().any(|c| c.is_numeric());
        let has_citations = !response.source_references.is_empty();
        
        if has_numbers && !has_citations {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::FactualAccuracy,
                message: "Numerical claims present without source citations".to_string(),
                position: None,
                suggestion: Some("Add citations for numerical claims".to_string()),
                confidence: 0.8,
            });
        }

        // Calculate overall confidence
        let base_confidence = response.confidence_factors
            .iter()
            .sum::<f64>() / response.confidence_factors.len().max(1) as f64;
        
        let penalty = findings.iter()
            .map(|f| match f.severity {
                ValidationSeverity::Critical => 0.3,
                ValidationSeverity::Error => 0.2,
                ValidationSeverity::Warning => 0.1,
                ValidationSeverity::Info => 0.0,
            })
            .sum::<f64>();

        let confidence = (base_confidence - penalty).max(0.0).min(1.0);
        let passed = confidence >= 0.6 && findings.iter().all(|f| f.severity != ValidationSeverity::Critical);

        Ok(ValidationResult {
            layer_name: self.name.clone(),
            passed,
            confidence,
            findings,
            processing_time: start_time.elapsed(),
            segment_start: 0,
            segment_end: response.content.len(),
        })
    }
}

/// Citation validation layer
#[derive(Debug)]
struct CitationValidationLayer {
    name: String,
}

impl CitationValidationLayer {
    fn new() -> Self {
        Self {
            name: "citation_validation".to_string(),
        }
    }
}

#[async_trait]
impl ValidationLayer for CitationValidationLayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u8 {
        80
    }

    async fn validate(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
        _config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut findings = Vec::new();

        // Check if sources are provided when needed
        let content = &response.content.to_lowercase();
        let needs_citations = content.contains("study") || 
                             content.contains("research") ||
                             content.contains("according to") ||
                             content.contains("data shows");

        if needs_citations && response.source_references.is_empty() {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Error,
                finding_type: ValidationFindingType::CitationIssue,
                message: "Content suggests research/studies but no sources are referenced".to_string(),
                position: None,
                suggestion: Some("Add appropriate source citations".to_string()),
                confidence: 0.9,
            });
        }

        // Check source diversity
        let unique_sources = response.source_references.len();
        if unique_sources > 0 && unique_sources < 2 {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::CitationIssue,
                message: "Limited source diversity - consider adding more sources".to_string(),
                position: None,
                suggestion: Some("Include additional diverse sources".to_string()),
                confidence: 0.6,
            });
        }

        let confidence = if findings.iter().any(|f| f.severity == ValidationSeverity::Error) {
            0.4
        } else if !findings.is_empty() {
            0.7
        } else {
            0.9
        };

        let passed = confidence >= 0.6;

        Ok(ValidationResult {
            layer_name: self.name.clone(),
            passed,
            confidence,
            findings,
            processing_time: start_time.elapsed(),
            segment_start: 0,
            segment_end: response.content.len(),
        })
    }
}

/// Coherence validation layer
#[derive(Debug)]
struct CoherenceValidationLayer {
    name: String,
}

impl CoherenceValidationLayer {
    fn new() -> Self {
        Self {
            name: "coherence_validation".to_string(),
        }
    }
}

#[async_trait]
impl ValidationLayer for CoherenceValidationLayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u8 {
        70
    }

    async fn validate(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
        _config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut findings = Vec::new();

        // Basic coherence checks
        let sentences: Vec<&str> = response.content.split('.').collect();
        
        if sentences.len() < 2 {
            let confidence = 0.8; // Single sentence is generally coherent
            return Ok(ValidationResult {
                layer_name: self.name.clone(),
                passed: true,
                confidence,
                findings,
                processing_time: start_time.elapsed(),
                segment_start: 0,
                segment_end: response.content.len(),
            });
        }

        // Check for transition words and coherence indicators
        let transition_words = ["however", "therefore", "additionally", "furthermore", "moreover", "consequently"];
        let transition_count = transition_words.iter()
            .map(|word| response.content.to_lowercase().matches(word).count())
            .sum::<usize>();

        let transition_ratio = transition_count as f64 / sentences.len() as f64;
        
        if transition_ratio < 0.1 {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::CoherenceIssue,
                message: "Limited use of transition words may affect flow".to_string(),
                position: None,
                suggestion: Some("Consider adding transition words to improve flow".to_string()),
                confidence: 0.6,
            });
        }

        // Check for repeated phrases (potential redundancy)
        let words: Vec<&str> = response.content.split_whitespace().collect();
        let mut word_counts = HashMap::new();
        for word in &words {
            *word_counts.entry(word.to_lowercase()).or_insert(0) += 1;
        }

        let excessive_repetition = word_counts.values().any(|&count| count > words.len() / 10);
        if excessive_repetition {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::CoherenceIssue,
                message: "Excessive word repetition detected".to_string(),
                position: None,
                suggestion: Some("Vary word choice to improve readability".to_string()),
                confidence: 0.8,
            });
        }

        let confidence = if findings.is_empty() {
            0.85 + (transition_ratio * 0.15).min(0.15)
        } else {
            0.7
        };

        Ok(ValidationResult {
            layer_name: self.name.clone(),
            passed: confidence >= 0.6,
            confidence,
            findings,
            processing_time: start_time.elapsed(),
            segment_start: 0,
            segment_end: response.content.len(),
        })
    }
}

/// Completeness validation layer
#[derive(Debug)]
struct CompletenessValidationLayer {
    name: String,
}

impl CompletenessValidationLayer {
    fn new() -> Self {
        Self {
            name: "completeness_validation".to_string(),
        }
    }
}

#[async_trait]
impl ValidationLayer for CompletenessValidationLayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u8 {
        60
    }

    async fn validate(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
        _config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut findings = Vec::new();

        // Check if response addresses the query
        let query_words: Vec<String> = request.query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let response_words: Vec<String> = response.content
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let query_coverage = query_words.iter()
            .filter(|word| response_words.contains(word))
            .count() as f64 / query_words.len() as f64;

        if query_coverage < 0.5 {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::CompletenessIssue,
                message: format!("Low query coverage ({:.1}%)", query_coverage * 100.0),
                position: None,
                suggestion: Some("Ensure response directly addresses the query".to_string()),
                confidence: 0.8,
            });
        }

        // Check response length appropriateness
        let word_count = response_words.len();
        if word_count < 10 {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::CompletenessIssue,
                message: "Response appears too brief".to_string(),
                position: None,
                suggestion: Some("Consider providing more comprehensive information".to_string()),
                confidence: 0.7,
            });
        }

        let confidence = query_coverage * 0.6 + 
                        ((word_count as f64).min(100.0) / 100.0) * 0.4;

        Ok(ValidationResult {
            layer_name: self.name.clone(),
            passed: confidence >= 0.6,
            confidence,
            findings,
            processing_time: start_time.elapsed(),
            segment_start: 0,
            segment_end: response.content.len(),
        })
    }
}

/// Bias detection layer
#[derive(Debug)]
struct BiasDetectionLayer {
    name: String,
}

impl BiasDetectionLayer {
    fn new() -> Self {
        Self {
            name: "bias_detection".to_string(),
        }
    }
}

#[async_trait]
impl ValidationLayer for BiasDetectionLayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u8 {
        50
    }

    async fn validate(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
        _config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut findings = Vec::new();

        let content = &response.content.to_lowercase();

        // Check for potential bias indicators
        let bias_indicators = [
            "obviously", "clearly", "everyone knows", "it's common sense",
            "all experts agree", "without question", "undoubtedly"
        ];

        for indicator in &bias_indicators {
            if content.contains(indicator) {
                findings.push(ValidationFinding {
                    severity: ValidationSeverity::Info,
                    finding_type: ValidationFindingType::BiasDetection,
                    message: format!("Potential bias indicator: '{}'", indicator),
                    position: None,
                    suggestion: Some("Consider more neutral language".to_string()),
                    confidence: 0.6,
                });
            }
        }

        let confidence = if findings.is_empty() { 0.9 } else { 0.7 };

        Ok(ValidationResult {
            layer_name: self.name.clone(),
            passed: true, // Bias detection is informational
            confidence,
            findings,
            processing_time: start_time.elapsed(),
            segment_start: 0,
            segment_end: response.content.len(),
        })
    }
}

/// Hallucination detection layer
#[derive(Debug)]
struct HallucinationDetectionLayer {
    name: String,
}

impl HallucinationDetectionLayer {
    fn new() -> Self {
        Self {
            name: "hallucination_detection".to_string(),
        }
    }
}

#[async_trait]
impl ValidationLayer for HallucinationDetectionLayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u8 {
        85
    }

    async fn validate(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
        _config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut findings = Vec::new();

        // Check if response content can be supported by context
        let has_context = !request.context.is_empty();
        let has_sources = !response.source_references.is_empty();

        if !has_context && !has_sources {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::HallucinationDetection,
                message: "Response generated without context or source references".to_string(),
                position: None,
                suggestion: Some("Ensure response is grounded in provided context".to_string()),
                confidence: 0.8,
            });
        }

        // Check for specific claims without support
        let content = &response.content.to_lowercase();
        let specific_claims = content.contains("on") && 
                             (content.contains("january") || content.contains("2023") || content.contains("2024"));

        if specific_claims && response.source_references.is_empty() {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::HallucinationDetection,
                message: "Specific date/time claims without source verification".to_string(),
                position: None,
                suggestion: Some("Verify specific claims against sources".to_string()),
                confidence: 0.9,
            });
        }

        let confidence = if findings.iter().any(|f| f.severity == ValidationSeverity::Error) {
            0.4
        } else if !findings.is_empty() {
            0.7
        } else {
            0.9
        };

        Ok(ValidationResult {
            layer_name: self.name.clone(),
            passed: confidence >= 0.6,
            confidence,
            findings,
            processing_time: start_time.elapsed(),
            segment_start: 0,
            segment_end: response.content.len(),
        })
    }
}

/// Consistency validation layer
#[derive(Debug)]
struct ConsistencyValidationLayer {
    name: String,
}

impl ConsistencyValidationLayer {
    fn new() -> Self {
        Self {
            name: "consistency_validation".to_string(),
        }
    }
}

#[async_trait]
impl ValidationLayer for ConsistencyValidationLayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn priority(&self) -> u8 {
        40
    }

    async fn validate(
        &self,
        response: &IntermediateResponse,
        request: &GenerationRequest,
        _config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let start_time = Instant::now();
        let mut findings = Vec::new();

        // Check for internal contradictions (basic implementation)
        let content = &response.content.to_lowercase();
        
        // Look for contradictory statements
        if content.contains("always") && content.contains("never") {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Warning,
                finding_type: ValidationFindingType::ConsistencyIssue,
                message: "Potentially contradictory absolute statements".to_string(),
                position: None,
                suggestion: Some("Review for internal consistency".to_string()),
                confidence: 0.6,
            });
        }

        if content.contains("increases") && content.contains("decreases") {
            findings.push(ValidationFinding {
                severity: ValidationSeverity::Info,
                finding_type: ValidationFindingType::ConsistencyIssue,
                message: "Opposing directional statements - verify context".to_string(),
                position: None,
                suggestion: Some("Ensure opposing statements are properly contextualized".to_string()),
                confidence: 0.5,
            });
        }

        let confidence = if findings.is_empty() { 0.8 } else { 0.7 };

        Ok(ValidationResult {
            layer_name: self.name.clone(),
            passed: confidence >= 0.6,
            confidence,
            findings,
            processing_time: start_time.elapsed(),
            segment_start: 0,
            segment_end: response.content.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GenerationRequest, ContextChunk, Source};

    #[tokio::test]
    async fn test_validator_creation() {
        let config = ValidationConfig::default();
        let validator = Validator::new(config);
        assert!(validator.layers.len() > 0);
    }

    #[tokio::test]
    async fn test_factual_accuracy_validation() {
        let layer = FactualAccuracyLayer::new();
        let response = IntermediateResponse {
            content: "This always happens in all cases".to_string(),
            confidence_factors: vec![0.8],
            source_references: vec![],
            warnings: vec![],
        };
        
        let request = GenerationRequest::builder()
            .query("Test query")
            .build()
            .unwrap();

        let config = ValidationConfig::default();
        let result = layer.validate(&response, &request, &config).await.unwrap();
        
        assert_eq!(result.layer_name, "factual_accuracy");
        assert!(!result.findings.is_empty()); // Should find absolute statements
    }

    #[tokio::test]
    async fn test_validation_confidence_calculation() {
        let config = ValidationConfig::default();
        let mut validator = Validator::new(config);
        
        let response = IntermediateResponse {
            content: "Test response".to_string(),
            confidence_factors: vec![0.8, 0.9],
            source_references: vec![],
            warnings: vec![],
        };
        
        let request = GenerationRequest::builder()
            .query("Test query")
            .build()
            .unwrap();

        let results = validator.validate(&response, &request).await.unwrap();
        assert!(!results.is_empty());
        
        let overall_confidence = validator.calculate_overall_confidence(&results);
        assert!(overall_confidence >= 0.0 && overall_confidence <= 1.0);
    }
}