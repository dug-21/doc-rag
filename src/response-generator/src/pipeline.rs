//! Processing pipeline for orchestrating response generation stages

use crate::{
    builder::ResponseBuilder,
    error::{Result, ResponseError},
    query_preprocessing::FACTQueryPreprocessingStage,
    GenerationRequest, ResponseChunk,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::time::{Duration, Instant};
use tracing::{debug, instrument, warn};

/// Processing pipeline for response generation
#[derive(Debug)]
pub struct Pipeline {
    /// Ordered list of processing stages
    pub stages: Vec<Box<dyn PipelineStage>>,
    
    /// Pipeline configuration
    config: PipelineConfig,
    
    /// Performance metrics
    metrics: PipelineMetrics,
}

/// Configuration for the processing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Maximum total processing time
    pub max_processing_time: Duration,
    
    /// Enable parallel stage execution where possible
    pub enable_parallelism: bool,
    
    /// Retry configuration for failed stages
    pub retry_config: RetryConfig,
    
    /// Performance monitoring
    pub enable_metrics: bool,
    
    /// Stage-specific configurations
    pub stage_configs: HashMap<String, serde_json::Value>,
}

/// Retry configuration for pipeline stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries per stage
    pub max_retries: u32,
    
    /// Base delay between retries
    pub base_delay: Duration,
    
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    
    /// Maximum delay between retries
    pub max_delay: Duration,
}

/// Processing context shared between pipeline stages
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Original request
    pub request: GenerationRequest,
    
    /// Processing start time
    pub start_time: Instant,
    
    /// Context variables for inter-stage communication
    pub variables: HashMap<String, serde_json::Value>,
    
    /// Accumulated warnings
    pub warnings: Vec<String>,
    
    /// Performance metrics
    pub stage_timings: HashMap<String, Duration>,
}

/// Pipeline processing stage trait
#[async_trait]
pub trait PipelineStage: Send + Sync + std::fmt::Debug {
    /// Stage name for identification
    fn name(&self) -> &str;
    
    /// Stage processing order (lower numbers execute first)
    fn order(&self) -> u32;
    
    /// Process the response builder
    async fn process(
        &self,
        builder: ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<ResponseBuilder>;
    
    /// Process with streaming support
    async fn process_streaming(
        &self,
        _builder: &mut ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<Option<ResponseChunk>> {
        // Default implementation: no streaming support
        Ok(None)
    }
    
    /// Whether this stage supports parallel execution
    fn supports_parallel(&self) -> bool {
        true
    }
    
    /// Stage dependencies (must complete before this stage)
    fn dependencies(&self) -> Vec<String> {
        Vec::new()
    }
    
    /// Validate stage preconditions
    async fn validate_preconditions(&self, context: &ProcessingContext) -> Result<()> {
        Ok(())
    }
    
    /// Cleanup after stage completion
    async fn cleanup(&self, context: &ProcessingContext) -> Result<()> {
        Ok(())
    }
}

/// Pipeline performance metrics
#[derive(Debug, Clone, Default)]
pub struct PipelineMetrics {
    /// Total pipeline executions
    pub total_executions: u64,
    
    /// Average total processing time
    pub avg_processing_time: Duration,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Stage-specific metrics
    pub stage_metrics: HashMap<String, StageMetrics>,
}

/// Performance metrics per stage
#[derive(Debug, Clone, Default)]
pub struct StageMetrics {
    /// Total executions
    pub executions: u64,
    
    /// Average processing time
    pub avg_time: Duration,
    
    /// Success rate
    pub success_rate: f64,
    
    /// Retry statistics
    pub retry_count: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            max_processing_time: Duration::from_millis(100),
            enable_parallelism: true,
            retry_config: RetryConfig::default(),
            enable_metrics: true,
            stage_configs: HashMap::new(),
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(10),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_millis(1000),
        }
    }
}

impl Pipeline {
    /// Create a new pipeline with default stages
    pub async fn new(stage_names: &[String]) -> Self {
        let mut pipeline = Self {
            stages: Vec::new(),
            config: PipelineConfig::default(),
            metrics: PipelineMetrics::default(),
        };
        
        // Add stages based on configuration
        for stage_name in stage_names {
            if let Some(stage) = pipeline.create_stage(stage_name).await {
                pipeline.add_stage(stage);
            }
        }
        
        // If no stages specified, add default stages
        if pipeline.stages.is_empty() {
            pipeline.add_default_stages().await;
        }
        
        pipeline
    }

    /// Create pipeline with custom configuration
    pub async fn with_config(config: PipelineConfig, stage_names: &[String]) -> Self {
        let mut pipeline = Self::new(stage_names).await;
        pipeline.config = config;
        pipeline
    }

    /// Add a processing stage
    pub fn add_stage(&mut self, stage: Box<dyn PipelineStage>) {
        self.stages.push(stage);
        self.sort_stages();
    }

    /// Process request through the pipeline
    #[instrument(skip(self, builder, context))]
    pub async fn process(
        &mut self,
        builder: ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<ResponseBuilder> {
        let start_time = Instant::now();
        debug!("Starting pipeline processing with {} stages", self.stages.len());

        // Validate pipeline preconditions
        self.validate_pipeline(context).await?;

        let mut current_builder = builder;
        let mut processing_context = context.clone();

        // Execute stages in order
        for stage in &self.stages {
            let stage_start = Instant::now();
            
            // Validate stage preconditions
            stage.validate_preconditions(&processing_context).await?;
            
            // Process with retry logic
            current_builder = self.process_stage_with_retry(
                stage.as_ref(),
                current_builder,
                &processing_context,
            ).await?;
            
            let stage_duration = stage_start.elapsed();
            processing_context.stage_timings.insert(stage.name().to_string(), stage_duration);
            
            // Stage cleanup
            stage.cleanup(&processing_context).await?;
            
            debug!("Stage '{}' completed in {}ms", stage.name(), stage_duration.as_millis());
        }

        let total_duration = start_time.elapsed();
        
        // Check processing time constraint
        if total_duration > self.config.max_processing_time {
            warn!("Pipeline processing took {}ms, exceeding target of {}ms",
                  total_duration.as_millis(),
                  self.config.max_processing_time.as_millis());
        }

        // Update metrics
        if self.config.enable_metrics {
            self.update_metrics(&processing_context, total_duration, true).await?;
        }

        debug!("Pipeline processing completed in {}ms", total_duration.as_millis());
        Ok(current_builder)
    }

    /// Get pipeline performance metrics
    pub fn get_metrics(&self) -> &PipelineMetrics {
        &self.metrics
    }

    /// Add default processing stages
    async fn add_default_stages(&mut self) {
        // Add FACT query preprocessing as the first stage
        if let Ok(fact_stage) = FACTQueryPreprocessingStage::new().await {
            self.stages.push(Box::new(fact_stage));
        }
        
        self.stages.push(Box::new(ContextPreprocessingStage::new()));
        self.stages.push(Box::new(ContentGenerationStage::new()));
        self.stages.push(Box::new(QualityEnhancementStage::new()));
        self.stages.push(Box::new(CitationProcessingStage::new()));
        self.stages.push(Box::new(FinalOptimizationStage::new()));
        
        self.sort_stages();
    }

    /// Create a stage by name
    async fn create_stage(&self, stage_name: &str) -> Option<Box<dyn PipelineStage>> {
        match stage_name {
            "fact_query_preprocessing" => {
                // Handle async stage creation
                if let Ok(stage) = FACTQueryPreprocessingStage::new().await {
                    Some(Box::new(stage))
                } else {
                    None
                }
            },
            "context_preprocessing" => Some(Box::new(ContextPreprocessingStage::new())),
            "content_generation" => Some(Box::new(ContentGenerationStage::new())),
            "quality_enhancement" => Some(Box::new(QualityEnhancementStage::new())),
            "citation_processing" => Some(Box::new(CitationProcessingStage::new())),
            "final_optimization" => Some(Box::new(FinalOptimizationStage::new())),
            _ => None,
        }
    }

    /// Sort stages by processing order
    fn sort_stages(&mut self) {
        self.stages.sort_by(|a, b| a.order().cmp(&b.order()));
    }

    /// Validate pipeline configuration and dependencies
    async fn validate_pipeline(&self, _context: &ProcessingContext) -> Result<()> {
        // Check for dependency cycles
        let mut dependencies: HashMap<String, Vec<String>> = HashMap::new();
        
        for stage in &self.stages {
            dependencies.insert(stage.name().to_string(), stage.dependencies());
        }
        
        // Simple cycle detection (for production, use a more sophisticated algorithm)
        for (stage_name, deps) in &dependencies {
            for dep in deps {
                if let Some(dep_deps) = dependencies.get(dep) {
                    if dep_deps.contains(stage_name) {
                        return Err(ResponseError::pipeline(
                            "dependency_validation",
                            format!("Circular dependency detected between {} and {}", stage_name, dep).as_str()
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Process a single stage with retry logic
    async fn process_stage_with_retry(
        &self,
        stage: &dyn PipelineStage,
        builder: ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<ResponseBuilder> {
        let mut attempts = 0;
        let mut last_error = None;
        let mut current_builder = builder;

        while attempts <= self.config.retry_config.max_retries {
            match stage.process(current_builder.clone(), context).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    attempts += 1;
                    
                    if attempts <= self.config.retry_config.max_retries {
                        let delay = self.calculate_retry_delay(attempts);
                        warn!("Stage '{}' failed (attempt {}), retrying in {}ms", 
                              stage.name(), attempts, delay.as_millis());
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| ResponseError::pipeline(
            stage.name(),
            "Maximum retries exceeded"
        )))
    }

    /// Calculate retry delay with exponential backoff
    fn calculate_retry_delay(&self, attempt: u32) -> Duration {
        let base_delay = self.config.retry_config.base_delay.as_millis() as f64;
        let multiplier = self.config.retry_config.backoff_multiplier;
        let max_delay = self.config.retry_config.max_delay;
        
        let delay_ms = base_delay * multiplier.powi(attempt as i32 - 1);
        let delay = Duration::from_millis(delay_ms as u64);
        
        std::cmp::min(delay, max_delay)
    }

    /// Update pipeline metrics
    async fn update_metrics(
        &mut self,
        context: &ProcessingContext,
        total_duration: Duration,
        success: bool,
    ) -> Result<()> {
        self.metrics.total_executions += 1;
        
        // Update average processing time
        let current_avg_ms = self.metrics.avg_processing_time.as_millis() as f64;
        let new_time_ms = total_duration.as_millis() as f64;
        let count = self.metrics.total_executions as f64;
        
        let new_avg_ms = (current_avg_ms * (count - 1.0) + new_time_ms) / count;
        self.metrics.avg_processing_time = Duration::from_millis(new_avg_ms as u64);

        // Update success rate
        let success_value = if success { 1.0 } else { 0.0 };
        self.metrics.success_rate = (self.metrics.success_rate * (count - 1.0) + success_value) / count;

        // Update stage-specific metrics
        for (stage_name, stage_duration) in &context.stage_timings {
            let stage_metrics = self.metrics.stage_metrics
                .entry(stage_name.clone())
                .or_insert_with(StageMetrics::default);

            stage_metrics.executions += 1;
            
            // Update average time
            let current_avg = stage_metrics.avg_time.as_millis() as f64;
            let new_time = stage_duration.as_millis() as f64;
            let exec_count = stage_metrics.executions as f64;
            
            let new_avg = (current_avg * (exec_count - 1.0) + new_time) / exec_count;
            stage_metrics.avg_time = Duration::from_millis(new_avg as u64);

            // Update success rate
            stage_metrics.success_rate = (stage_metrics.success_rate * (exec_count - 1.0) + success_value) / exec_count;
        }

        Ok(())
    }
}

impl ProcessingContext {
    /// Create a new processing context
    pub fn new(request: &GenerationRequest) -> Self {
        Self {
            request: request.clone(),
            start_time: Instant::now(),
            variables: HashMap::new(),
            warnings: Vec::new(),
            stage_timings: HashMap::new(),
        }
    }

    /// Set a context variable
    pub fn set_variable<T: serde::Serialize>(&mut self, key: String, value: T) -> Result<()> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| ResponseError::internal(format!("Failed to serialize context variable: {}", e)))?;
        self.variables.insert(key, json_value);
        Ok(())
    }

    /// Get a context variable
    pub fn get_variable<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        if let Some(value) = self.variables.get(key) {
            let result = serde_json::from_value(value.clone())
                .map_err(|e| ResponseError::internal(format!("Failed to deserialize context variable: {}", e)))?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Add a warning
    pub fn add_warning<S: Into<String>>(&mut self, warning: S) {
        self.warnings.push(warning.into());
    }

    /// Get elapsed processing time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

// Default pipeline stage implementations

/// Context preprocessing stage
#[derive(Debug)]
struct ContextPreprocessingStage {
    name: String,
}

impl ContextPreprocessingStage {
    fn new() -> Self {
        Self {
            name: "context_preprocessing".to_string(),
        }
    }
}

#[async_trait]
impl PipelineStage for ContextPreprocessingStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        10
    }

    async fn process(
        &self,
        mut builder: ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<ResponseBuilder> {
        debug!("Processing context preprocessing stage");
        
        // Prepare and rank context
        builder.prepare_context().await?;
        
        Ok(builder)
    }
}

/// Content generation stage
#[derive(Debug)]
struct ContentGenerationStage {
    name: String,
}

impl ContentGenerationStage {
    fn new() -> Self {
        Self {
            name: "content_generation".to_string(),
        }
    }
}

#[async_trait]
impl PipelineStage for ContentGenerationStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        20
    }

    fn dependencies(&self) -> Vec<String> {
        vec!["context_preprocessing".to_string()]
    }

    async fn process(
        &self,
        mut builder: ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<ResponseBuilder> {
        debug!("Processing content generation stage");
        
        // Build the main response content
        builder.build_content().await?;
        
        Ok(builder)
    }

    async fn process_streaming(
        &self,
        _builder: &mut ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<Option<ResponseChunk>> {
        // Generate a chunk of content for streaming
        // This is a simplified implementation
        Ok(Some(ResponseChunk {
            content: "Generated content chunk".to_string(),
            chunk_type: crate::ResponseChunkType::Partial,
            position: 0,
            is_final: false,
            confidence: Some(0.8),
            metadata: Some(crate::GenerationMetrics {
                total_duration: std::time::Duration::from_millis(0),
                validation_duration: std::time::Duration::from_millis(0),
                formatting_duration: std::time::Duration::from_millis(0),
                citation_duration: std::time::Duration::from_millis(0),
                validation_passes: 0,
                sources_used: 0,
                response_length: 0,
            }),
        }))
    }
}

/// Quality enhancement stage
#[derive(Debug)]
struct QualityEnhancementStage {
    name: String,
}

impl QualityEnhancementStage {
    fn new() -> Self {
        Self {
            name: "quality_enhancement".to_string(),
        }
    }
}

#[async_trait]
impl PipelineStage for QualityEnhancementStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        30
    }

    fn dependencies(&self) -> Vec<String> {
        vec!["content_generation".to_string()]
    }

    async fn process(
        &self,
        mut builder: ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<ResponseBuilder> {
        debug!("Processing quality enhancement stage");
        
        // Optimize and enhance response quality
        builder.optimize().await?;
        
        Ok(builder)
    }
}

/// Citation processing stage
#[derive(Debug)]
struct CitationProcessingStage {
    name: String,
}

impl CitationProcessingStage {
    fn new() -> Self {
        Self {
            name: "citation_processing".to_string(),
        }
    }
}

#[async_trait]
impl PipelineStage for CitationProcessingStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        40
    }

    fn dependencies(&self) -> Vec<String> {
        vec!["content_generation".to_string()]
    }

    async fn process(
        &self,
        mut builder: ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<ResponseBuilder> {
        debug!("Processing citation stage");
        
        // Add citations and source attribution
        builder.add_citations().await?;
        
        Ok(builder)
    }
}

/// Final optimization stage
#[derive(Debug)]
struct FinalOptimizationStage {
    name: String,
}

impl FinalOptimizationStage {
    fn new() -> Self {
        Self {
            name: "final_optimization".to_string(),
        }
    }
}

#[async_trait]
impl PipelineStage for FinalOptimizationStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        50
    }

    fn dependencies(&self) -> Vec<String> {
        vec!["quality_enhancement".to_string(), "citation_processing".to_string()]
    }

    async fn process(
        &self,
        builder: ResponseBuilder,
        context: &ProcessingContext,
    ) -> Result<ResponseBuilder> {
        debug!("Processing final optimization stage");
        
        // Final checks and optimizations
        if context.elapsed() > Duration::from_millis(80) {
            debug!("Processing time approaching limit, applying final optimizations");
        }
        
        Ok(builder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GenerationRequest;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let stage_names = vec!["content_generation".to_string()];
        let pipeline = Pipeline::new(&stage_names).await;
        assert!(pipeline.stages.len() > 0);
    }

    #[tokio::test]
    async fn test_processing_context() {
        let request = GenerationRequest::builder()
            .query("Test query")
            .build()
            .unwrap();

        let mut context = ProcessingContext::new(&request);
        
        // Test variable setting and getting
        context.set_variable("test_key".to_string(), "test_value").unwrap();
        let value: Option<String> = context.get_variable("test_key").unwrap();
        assert_eq!(value, Some("test_value".to_string()));

        // Test warnings
        context.add_warning("Test warning");
        assert_eq!(context.warnings.len(), 1);
    }

    #[tokio::test]
    async fn test_stage_ordering() {
        let mut pipeline = Pipeline::new(&[]).await;
        pipeline.add_default_stages().await;
        
        // Verify stages are sorted by order
        for i in 1..pipeline.stages.len() {
            assert!(pipeline.stages[i-1].order() <= pipeline.stages[i].order());
        }
    }

    #[tokio::test]
    async fn test_retry_delay_calculation() {
        let pipeline = Pipeline::new(&[]).await;
        
        let delay1 = pipeline.calculate_retry_delay(1);
        let delay2 = pipeline.calculate_retry_delay(2);
        let delay3 = pipeline.calculate_retry_delay(3);
        
        assert!(delay1 <= delay2);
        assert!(delay2 <= delay3);
        assert!(delay3 <= pipeline.config.retry_config.max_delay);
    }

    #[test]
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default();
        assert_eq!(config.max_processing_time, Duration::from_millis(100));
        assert!(config.enable_parallelism);
        assert!(config.enable_metrics);
        assert_eq!(config.retry_config.max_retries, 3);
    }
}