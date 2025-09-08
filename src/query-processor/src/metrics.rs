//! Performance metrics and observability for the Query Processor
//!
//! This module provides comprehensive metrics collection, performance tracking,
//! and observability features for monitoring query processing performance.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
// use std::sync::atomic::{AtomicU64, Ordering}; // Unused
use std::time::{Duration, Instant};
use tracing::{info, instrument};

use crate::query::{ProcessedQuery, ProcessingSummary};
use crate::types::*;

/// Classification metadata for intent classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetadata {
    /// Classifier used
    pub classifier: String,
    /// Classification timestamp
    pub classified_at: DateTime<Utc>,
    /// Features used for classification
    pub features: Vec<String>,
    /// Alternative classifications
    pub alternatives: Vec<String>,
}

/// Performance prediction for strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePrediction {
    /// Predicted accuracy
    pub accuracy: f64,
    /// Predicted latency
    pub latency: Duration,
    /// Predicted recall
    pub recall: f64,
    /// Predicted precision
    pub precision: f64,
    /// Overall confidence
    pub confidence: f64,
}

/// Selection metadata for strategy selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionMetadata {
    /// Selection algorithm used
    pub algorithm: String,
    /// Selection timestamp
    pub selected_at: DateTime<Utc>,
    /// Selection factors
    pub factors: Vec<String>,
    /// Query characteristics
    pub characteristics: HashMap<String, f64>,
}

/// Comprehensive metrics for query processor performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorMetrics {
    /// Total number of queries processed
    pub total_processed: u64,
    /// Number of successful processes
    pub successful_processes: u64,
    /// Number of failed processes
    pub failed_processes: u64,
    /// Processing latency statistics
    pub latency_stats: LatencyStatistics,
    /// Throughput measurements
    pub throughput_stats: ThroughputStatistics,
    /// Accuracy metrics
    pub accuracy_stats: AccuracyStatistics,
    /// Resource usage statistics
    pub resource_stats: ResourceStatistics,
    /// Stage-specific performance metrics
    pub stage_metrics: HashMap<String, StageMetrics>,
    /// Intent distribution metrics
    pub intent_distribution: HashMap<QueryIntent, IntentMetrics>,
    /// Strategy effectiveness metrics (keyed by strategy name)
    pub strategy_effectiveness: HashMap<String, StrategyMetrics>,
    /// Error distribution
    pub error_distribution: HashMap<String, u64>,
    /// Metrics collection start time
    pub start_time: DateTime<Utc>,
    /// Last update time
    pub last_updated: DateTime<Utc>,
    /// Recent processing summaries (circular buffer)
    recent_summaries: VecDeque<ProcessingSummary>,
}

impl ProcessorMetrics {
    /// Create new processor metrics
    pub fn new() -> Self {
        let now = Utc::now();
        
        Self {
            total_processed: 0,
            successful_processes: 0,
            failed_processes: 0,
            latency_stats: LatencyStatistics::new(),
            throughput_stats: ThroughputStatistics::new(),
            accuracy_stats: AccuracyStatistics::new(),
            resource_stats: ResourceStatistics::new(),
            stage_metrics: HashMap::new(),
            intent_distribution: HashMap::new(),
            strategy_effectiveness: HashMap::new(),
            error_distribution: HashMap::new(),
            start_time: now,
            last_updated: now,
            recent_summaries: VecDeque::with_capacity(1000),
        }
    }

    /// Record a successful query processing
    #[instrument(skip(self, processed_query))]
    pub fn record_processing(&mut self, duration: Duration, processed_query: &ProcessedQuery) {
        info!("Recording processing metrics for query: {}", processed_query.id());
        
        self.total_processed += 1;
        self.successful_processes += 1;
        self.last_updated = Utc::now();
        
        // Record latency
        self.latency_stats.record_latency(duration);
        
        // Record throughput
        self.throughput_stats.record_processing(Instant::now());
        
        // Record accuracy metrics
        let overall_confidence = processed_query.overall_confidence();
        self.accuracy_stats.record_confidence(overall_confidence);
        
        // Record resource usage
        self.resource_stats.record_usage(&processed_query.processing_metadata.statistics.resource_usage);
        
        // Record stage metrics
        for (stage, stage_duration) in &processed_query.processing_metadata.stage_durations {
            let stage_metrics = self.stage_metrics.entry(stage.clone()).or_insert_with(StageMetrics::new);
            stage_metrics.record_duration(*stage_duration);
        }
        
        // Record intent distribution
        let intent_metrics = self.intent_distribution
            .entry(processed_query.intent.primary_intent.clone())
            .or_insert_with(IntentMetrics::new);
        intent_metrics.record_processing(processed_query.intent.confidence);
        
        // Record strategy effectiveness
        let strategy_metrics = self.strategy_effectiveness
            .entry(format!("{:?}", processed_query.strategy.strategy))
            .or_insert_with(StrategyMetrics::new);
        strategy_metrics.record_usage(processed_query.strategy.confidence);
        
        // Store recent summary (circular buffer)
        let summary = processed_query.summary();
        if self.recent_summaries.len() >= 1000 {
            self.recent_summaries.pop_front();
        }
        self.recent_summaries.push_back(summary);
    }

    /// Record a failed processing
    pub fn record_failure(&mut self, error_category: String, duration: Option<Duration>) {
        self.total_processed += 1;
        self.failed_processes += 1;
        self.last_updated = Utc::now();
        
        // Record error
        *self.error_distribution.entry(error_category).or_insert(0) += 1;
        
        // Record latency if available
        if let Some(d) = duration {
            self.latency_stats.record_latency(d);
        }
        
        // Record throughput attempt
        self.throughput_stats.record_processing(Instant::now());
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_processed == 0 {
            1.0
        } else {
            self.successful_processes as f64 / self.total_processed as f64
        }
    }

    /// Get failure rate
    pub fn failure_rate(&self) -> f64 {
        1.0 - self.success_rate()
    }

    /// Get average latency
    pub fn average_latency(&self) -> Duration {
        self.latency_stats.average()
    }

    /// Get current throughput (queries per second)
    pub fn current_throughput(&self) -> f64 {
        self.throughput_stats.current_qps()
    }

    /// Get average confidence
    pub fn average_confidence(&self) -> f64 {
        self.accuracy_stats.average_confidence()
    }

    /// Get recent processing summaries
    pub fn recent_summaries(&self) -> &VecDeque<ProcessingSummary> {
        &self.recent_summaries
    }

    /// Get performance summary
    pub fn performance_summary(&self) -> PerformanceSummary {
        PerformanceSummary {
            total_processed: self.total_processed,
            success_rate: self.success_rate(),
            average_latency: self.average_latency(),
            current_throughput: self.current_throughput(),
            average_confidence: self.average_confidence(),
            top_intent: self.most_common_intent(),
            top_strategy: self.most_effective_strategy(),
            uptime: Utc::now().signed_duration_since(self.start_time).to_std().unwrap_or_default(),
            last_updated: self.last_updated,
        }
    }

    /// Get most common intent
    fn most_common_intent(&self) -> Option<QueryIntent> {
        self.intent_distribution
            .iter()
            .max_by_key(|(_, metrics)| metrics.count)
            .map(|(intent, _)| intent.clone())
    }

    /// Get most effective strategy
    fn most_effective_strategy(&self) -> Option<String> {
        self.strategy_effectiveness
            .iter()
            .max_by(|(_, a), (_, b)| {
                a.average_confidence().partial_cmp(&b.average_confidence())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(strategy, _)| strategy.clone())
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Export metrics to Prometheus format
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();
        
        // Basic counters
        output.push_str(&format!("# HELP query_processor_total_processed_total Total number of queries processed\n"));
        output.push_str(&format!("# TYPE query_processor_total_processed_total counter\n"));
        output.push_str(&format!("query_processor_total_processed_total {}\n", self.total_processed));
        
        output.push_str(&format!("# HELP query_processor_successful_processes_total Successful processes\n"));
        output.push_str(&format!("# TYPE query_processor_successful_processes_total counter\n"));
        output.push_str(&format!("query_processor_successful_processes_total {}\n", self.successful_processes));
        
        output.push_str(&format!("# HELP query_processor_failed_processes_total Failed processes\n"));
        output.push_str(&format!("# TYPE query_processor_failed_processes_total counter\n"));
        output.push_str(&format!("query_processor_failed_processes_total {}\n", self.failed_processes));
        
        // Latency metrics
        output.push_str(&format!("# HELP query_processor_latency_seconds Processing latency in seconds\n"));
        output.push_str(&format!("# TYPE query_processor_latency_seconds histogram\n"));
        output.push_str(&format!("query_processor_latency_seconds_average {}\n", 
                                self.latency_stats.average().as_secs_f64()));
        output.push_str(&format!("query_processor_latency_seconds_p50 {}\n", 
                                self.latency_stats.percentile(50).as_secs_f64()));
        output.push_str(&format!("query_processor_latency_seconds_p95 {}\n", 
                                self.latency_stats.percentile(95).as_secs_f64()));
        output.push_str(&format!("query_processor_latency_seconds_p99 {}\n", 
                                self.latency_stats.percentile(99).as_secs_f64()));
        
        // Throughput
        output.push_str(&format!("# HELP query_processor_throughput_qps Current throughput in queries per second\n"));
        output.push_str(&format!("# TYPE query_processor_throughput_qps gauge\n"));
        output.push_str(&format!("query_processor_throughput_qps {}\n", self.current_throughput()));
        
        // Accuracy
        output.push_str(&format!("# HELP query_processor_accuracy_ratio Average processing accuracy\n"));
        output.push_str(&format!("# TYPE query_processor_accuracy_ratio gauge\n"));
        output.push_str(&format!("query_processor_accuracy_ratio {}\n", self.average_confidence()));
        
        // Intent distribution
        for (intent, metrics) in &self.intent_distribution {
            output.push_str(&format!(
                "query_processor_intent_distribution{{intent=\"{:?}\"}} {}\n",
                intent, metrics.count
            ));
        }
        
        // Strategy effectiveness
        for (strategy, metrics) in &self.strategy_effectiveness {
            output.push_str(&format!(
                "query_processor_strategy_effectiveness{{strategy=\"{:?}\"}} {}\n",
                strategy, metrics.average_confidence()
            ));
        }
        
        output
    }
}

/// Latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStatistics {
    count: u64,
    total_duration: Duration,
    min_duration: Option<Duration>,
    max_duration: Option<Duration>,
    durations: VecDeque<Duration>, // For percentile calculation
}

impl LatencyStatistics {
    pub fn new() -> Self {
        Self {
            count: 0,
            total_duration: Duration::ZERO,
            min_duration: None,
            max_duration: None,
            durations: VecDeque::with_capacity(1000),
        }
    }

    pub fn record_latency(&mut self, duration: Duration) {
        self.count += 1;
        self.total_duration += duration;
        
        self.min_duration = Some(self.min_duration.map_or(duration, |min| min.min(duration)));
        self.max_duration = Some(self.max_duration.map_or(duration, |max| max.max(duration)));
        
        // Store for percentile calculation (circular buffer)
        if self.durations.len() >= 1000 {
            self.durations.pop_front();
        }
        self.durations.push_back(duration);
    }

    pub fn average(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.count as u32
        }
    }

    pub fn percentile(&self, p: u8) -> Duration {
        if self.durations.is_empty() {
            return Duration::ZERO;
        }
        
        let mut sorted: Vec<Duration> = self.durations.iter().cloned().collect();
        sorted.sort();
        
        let index = ((p as f64 / 100.0) * (sorted.len() as f64 - 1.0)) as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

/// Throughput statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStatistics {
    #[serde(skip)]
    processing_times: VecDeque<Instant>,
    window_duration: Duration,
}

impl ThroughputStatistics {
    pub fn new() -> Self {
        Self {
            processing_times: VecDeque::with_capacity(1000),
            window_duration: Duration::from_secs(60), // 1-minute window
        }
    }

    pub fn record_processing(&mut self, timestamp: Instant) {
        // Clean old entries outside window
        let cutoff = timestamp - self.window_duration;
        while let Some(&front) = self.processing_times.front() {
            if front < cutoff {
                self.processing_times.pop_front();
            } else {
                break;
            }
        }
        
        self.processing_times.push_back(timestamp);
    }

    pub fn current_qps(&self) -> f64 {
        let count = self.processing_times.len() as f64;
        let window_secs = self.window_duration.as_secs_f64();
        
        if window_secs == 0.0 {
            0.0
        } else {
            count / window_secs
        }
    }
}

/// Accuracy statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyStatistics {
    confidence_sum: f64,
    confidence_count: u64,
    min_confidence: Option<f64>,
    max_confidence: Option<f64>,
}

impl AccuracyStatistics {
    pub fn new() -> Self {
        Self {
            confidence_sum: 0.0,
            confidence_count: 0,
            min_confidence: None,
            max_confidence: None,
        }
    }

    pub fn record_confidence(&mut self, confidence: f64) {
        self.confidence_sum += confidence;
        self.confidence_count += 1;
        
        self.min_confidence = Some(self.min_confidence.map_or(confidence, |min| min.min(confidence)));
        self.max_confidence = Some(self.max_confidence.map_or(confidence, |max| max.max(confidence)));
    }

    pub fn average_confidence(&self) -> f64 {
        if self.confidence_count == 0 {
            0.0
        } else {
            self.confidence_sum / self.confidence_count as f64
        }
    }
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStatistics {
    peak_memory_usage: u64,
    total_cpu_time: Duration,
    total_api_calls: u64,
    cache_hits: u64,
    cache_misses: u64,
}

impl ResourceStatistics {
    pub fn new() -> Self {
        Self {
            peak_memory_usage: 0,
            total_cpu_time: Duration::ZERO,
            total_api_calls: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    pub fn record_usage(&mut self, usage: &crate::query::ResourceUsage) {
        self.peak_memory_usage = self.peak_memory_usage.max(usage.peak_memory);
        self.total_cpu_time += usage.cpu_time;
        self.total_api_calls += usage.api_calls as u64;
        self.cache_hits += usage.cache_hits as u64;
        self.cache_misses += usage.cache_misses as u64;
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let total_requests = self.cache_hits + self.cache_misses;
        if total_requests == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total_requests as f64
        }
    }
}

/// Stage-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageMetrics {
    count: u64,
    total_duration: Duration,
    average_duration: Duration,
}

impl StageMetrics {
    pub fn new() -> Self {
        Self {
            count: 0,
            total_duration: Duration::ZERO,
            average_duration: Duration::ZERO,
        }
    }

    pub fn record_duration(&mut self, duration: Duration) {
        self.count += 1;
        self.total_duration += duration;
        self.average_duration = self.total_duration / self.count as u32;
    }
}

/// Intent-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentMetrics {
    count: u64,
    total_confidence: f64,
    average_confidence: f64,
}

impl IntentMetrics {
    pub fn new() -> Self {
        Self {
            count: 0,
            total_confidence: 0.0,
            average_confidence: 0.0,
        }
    }

    pub fn record_processing(&mut self, confidence: f64) {
        self.count += 1;
        self.total_confidence += confidence;
        self.average_confidence = self.total_confidence / self.count as f64;
    }
}

/// Strategy effectiveness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    usage_count: u64,
    total_confidence: f64,
    success_count: u64,
}

impl StrategyMetrics {
    pub fn new() -> Self {
        Self {
            usage_count: 0,
            total_confidence: 0.0,
            success_count: 0,
        }
    }

    pub fn record_usage(&mut self, confidence: f64) {
        self.usage_count += 1;
        self.total_confidence += confidence;
        if confidence > 0.8 {
            self.success_count += 1;
        }
    }

    pub fn average_confidence(&self) -> f64 {
        if self.usage_count == 0 {
            0.0
        } else {
            self.total_confidence / self.usage_count as f64
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.usage_count == 0 {
            0.0
        } else {
            self.success_count as f64 / self.usage_count as f64
        }
    }
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub total_processed: u64,
    pub success_rate: f64,
    pub average_latency: Duration,
    pub current_throughput: f64,
    pub average_confidence: f64,
    pub top_intent: Option<QueryIntent>,
    pub top_strategy: Option<String>,
    pub uptime: Duration,
    pub last_updated: DateTime<Utc>,
}

impl Default for ProcessorMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{Query, ProcessedQuery};
    use crate::types::*;
    use std::collections::HashMap;
    use chrono::Utc;

    fn create_test_processed_query() -> ProcessedQuery {
        let query = Query::new("Test query").unwrap();
        let analysis = SemanticAnalysis {
            processing_time: std::time::Duration::from_millis(100),
            timestamp: Utc::now(),
            syntactic_features: SyntacticFeatures {
                pos_tags: vec![],
                named_entities: vec![],
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
        };
        let entities = vec![];
        let key_terms = vec![];
        let intent = IntentClassification {
            primary_intent: QueryIntent::Factual,
            confidence: 0.9,
            secondary_intents: vec![],
            probabilities: HashMap::new(),
            method: ClassificationMethod::RuleBased,
            features: vec!["test".to_string()],
        };
        let strategy = StrategySelection {
            strategy: SearchStrategy::VectorSimilarity,
            confidence: 0.85,
            fallbacks: vec![],
            reasoning: "Test reasoning".to_string(),
            expected_metrics: PerformanceMetrics {
                expected_accuracy: 0.9,
                expected_response_time: Duration::from_millis(100),
                expected_recall: 0.85,
                expected_precision: 0.90,
                resource_usage: ResourceUsage::default(),
            },
            predictions: StrategyPredictions {
                latency: 0.1,
                accuracy: 0.9,
                resource_usage: 0.5,
            },
        };
        
        let mut processed = ProcessedQuery::new(query, analysis, entities, key_terms, intent, strategy);
        processed.add_stage_duration("analysis".to_string(), Duration::from_millis(50));
        processed.add_stage_duration("extraction".to_string(), Duration::from_millis(30));
        
        processed
    }

    #[test]
    fn test_metrics_creation() {
        let metrics = ProcessorMetrics::new();
        assert_eq!(metrics.total_processed, 0);
        assert_eq!(metrics.success_rate(), 1.0);
        assert_eq!(metrics.average_latency(), Duration::ZERO);
    }

    #[test]
    fn test_record_processing() {
        let mut metrics = ProcessorMetrics::new();
        let query = create_test_processed_query();
        let duration = Duration::from_millis(200);
        
        metrics.record_processing(duration, &query);
        
        assert_eq!(metrics.total_processed, 1);
        assert_eq!(metrics.successful_processes, 1);
        assert_eq!(metrics.success_rate(), 1.0);
        assert_eq!(metrics.average_latency(), duration);
        
        // Check intent distribution
        assert!(metrics.intent_distribution.contains_key(&QueryIntent::Factual));
        
        // Check strategy effectiveness
        assert!(metrics.strategy_effectiveness.contains_key("VectorSimilarity"));
        
        // Check stage metrics
        assert!(metrics.stage_metrics.contains_key("analysis"));
        assert!(metrics.stage_metrics.contains_key("extraction"));
    }

    #[test]
    fn test_record_failure() {
        let mut metrics = ProcessorMetrics::new();
        
        metrics.record_failure("validation".to_string(), Some(Duration::from_millis(100)));
        
        assert_eq!(metrics.total_processed, 1);
        assert_eq!(metrics.failed_processes, 1);
        assert_eq!(metrics.success_rate(), 0.0);
        assert_eq!(metrics.failure_rate(), 1.0);
        assert_eq!(metrics.error_distribution.get("validation"), Some(&1));
    }

    #[test]
    fn test_latency_statistics() {
        let mut stats = LatencyStatistics::new();
        
        stats.record_latency(Duration::from_millis(100));
        stats.record_latency(Duration::from_millis(200));
        stats.record_latency(Duration::from_millis(150));
        
        assert_eq!(stats.count, 3);
        assert_eq!(stats.average(), Duration::from_millis(150));
        
        // Test percentiles
        let p50 = stats.percentile(50);
        assert!(p50 >= Duration::from_millis(100) && p50 <= Duration::from_millis(200));
    }

    #[test]
    fn test_throughput_statistics() {
        let mut stats = ThroughputStatistics::new();
        let now = Instant::now();
        
        // Record some processing times
        stats.record_processing(now);
        stats.record_processing(now + Duration::from_millis(100));
        stats.record_processing(now + Duration::from_millis(200));
        
        let qps = stats.current_qps();
        assert!(qps > 0.0);
    }

    #[test]
    fn test_accuracy_statistics() {
        let mut stats = AccuracyStatistics::new();
        
        stats.record_confidence(0.9);
        stats.record_confidence(0.8);
        stats.record_confidence(0.7);
        
        assert_eq!(stats.confidence_count, 3);
        assert!((stats.average_confidence() - 0.8).abs() < 0.01);
        assert_eq!(stats.min_confidence, Some(0.7));
        assert_eq!(stats.max_confidence, Some(0.9));
    }

    #[test]
    fn test_prometheus_export() {
        let mut metrics = ProcessorMetrics::new();
        let query = create_test_processed_query();
        
        metrics.record_processing(Duration::from_millis(200), &query);
        
        let prometheus_output = metrics.to_prometheus();
        
        // Check that output contains expected metrics
        assert!(prometheus_output.contains("query_processor_total_processed_total"));
        assert!(prometheus_output.contains("query_processor_latency_seconds"));
        assert!(prometheus_output.contains("query_processor_throughput_qps"));
        assert!(prometheus_output.contains("query_processor_accuracy_ratio"));
    }

    #[test]
    fn test_performance_summary() {
        let mut metrics = ProcessorMetrics::new();
        let query = create_test_processed_query();
        
        metrics.record_processing(Duration::from_millis(200), &query);
        
        let summary = metrics.performance_summary();
        
        assert_eq!(summary.total_processed, 1);
        assert_eq!(summary.success_rate, 1.0);
        assert_eq!(summary.average_latency, Duration::from_millis(200));
        assert_eq!(summary.top_intent, Some(QueryIntent::Factual));
        assert_eq!(summary.top_strategy, Some("VectorSimilarity".to_string()));
    }
}