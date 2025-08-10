//! Batch processing utilities for efficient embedding generation
//!
//! This module provides utilities for processing texts in optimal batch sizes,
//! memory management, and progress tracking.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};
use crate::EmbedderError;

/// Batch processor for managing text batching and memory efficiency
pub struct BatchProcessor {
    batch_size: usize,
    max_memory_mb: Option<usize>,
    progress_tracker: Arc<ProgressTracker>,
}

/// Progress tracking for batch processing
pub struct ProgressTracker {
    total_items: AtomicUsize,
    processed_items: AtomicUsize,
    current_batch: AtomicUsize,
    total_batches: AtomicUsize,
}

/// Statistics for a batch processing operation
#[derive(Debug, Clone)]
pub struct BatchStats {
    pub total_items: usize,
    pub total_batches: usize,
    pub processed_items: usize,
    pub current_batch: usize,
    pub avg_items_per_batch: f64,
    pub completion_percentage: f64,
}

/// Configuration for adaptive batching
#[derive(Debug, Clone)]
pub struct AdaptiveBatchConfig {
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub target_memory_mb: usize,
    pub avg_text_length_chars: usize,
    pub embedding_dimension: usize,
}

impl BatchProcessor {
    /// Create a new batch processor
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            max_memory_mb: None,
            progress_tracker: Arc::new(ProgressTracker::new()),
        }
    }
    
    /// Create a batch processor with memory constraints
    pub fn with_memory_limit(batch_size: usize, max_memory_mb: usize) -> Self {
        Self {
            batch_size,
            max_memory_mb: Some(max_memory_mb),
            progress_tracker: Arc::new(ProgressTracker::new()),
        }
    }
    
    /// Create batches from a slice of texts
    pub fn create_batches(&self, texts: &[String]) -> Vec<Vec<String>> {
        if texts.is_empty() {
            return Vec::new();
        }
        
        let effective_batch_size = self.calculate_effective_batch_size(texts);
        let mut batches = Vec::new();
        
        for chunk in texts.chunks(effective_batch_size) {
            batches.push(chunk.to_vec());
        }
        
        self.progress_tracker.initialize(texts.len(), batches.len());
        
        info!(
            "Created {} batches for {} texts (avg batch size: {:.1})",
            batches.len(),
            texts.len(),
            texts.len() as f64 / batches.len() as f64
        );
        
        batches
    }
    
    /// Calculate the effective batch size based on memory constraints and text characteristics
    fn calculate_effective_batch_size(&self, texts: &[String]) -> usize {
        if let Some(max_memory_mb) = self.max_memory_mb {
            let avg_text_length = texts.iter()
                .map(|t| t.len())
                .sum::<usize>() / texts.len().max(1);
            
            // Rough estimation: each character uses ~4 bytes in memory during processing
            // Plus embedding storage (~1536 bytes for typical models)
            let estimated_memory_per_text = (avg_text_length * 4) + 1536;
            let max_memory_bytes = max_memory_mb * 1_048_576;
            
            let memory_constrained_batch_size = max_memory_bytes / estimated_memory_per_text;
            let effective_batch_size = memory_constrained_batch_size.max(1).min(self.batch_size);
            
            if effective_batch_size < self.batch_size {
                warn!(
                    "Reducing batch size from {} to {} due to memory constraints",
                    self.batch_size, effective_batch_size
                );
            }
            
            effective_batch_size
        } else {
            self.batch_size
        }
    }
    
    /// Create adaptive batches based on text characteristics
    pub fn create_adaptive_batches(
        &self,
        texts: &[String],
        config: &AdaptiveBatchConfig,
    ) -> Vec<Vec<String>> {
        if texts.is_empty() {
            return Vec::new();
        }
        
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_batch_memory = 0;
        
        let target_memory_bytes = config.target_memory_mb * 1_048_576;
        let embedding_size_bytes = config.embedding_dimension * 4; // f32
        
        for text in texts {
            let text_memory = estimate_text_memory(text, embedding_size_bytes);
            
            // Check if adding this text would exceed memory target or batch size limits
            if (!current_batch.is_empty() && 
                (current_batch_memory + text_memory > target_memory_bytes ||
                 current_batch.len() >= config.max_batch_size)) ||
               current_batch.len() >= config.max_batch_size {
                
                batches.push(current_batch);
                current_batch = Vec::new();
                current_batch_memory = 0;
            }
            
            current_batch.push(text.clone());
            current_batch_memory += text_memory;
            
            // If a single text is too large, create a batch just for it
            if current_batch_memory > target_memory_bytes && current_batch.len() == 1 {
                batches.push(current_batch);
                current_batch = Vec::new();
                current_batch_memory = 0;
            }
        }
        
        // Add remaining texts
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }
        
        // Ensure minimum batch size where possible
        batches = merge_small_batches(batches, config.min_batch_size);
        
        self.progress_tracker.initialize(texts.len(), batches.len());
        
        info!(
            "Created {} adaptive batches for {} texts (target memory: {}MB)",
            batches.len(),
            texts.len(),
            config.target_memory_mb
        );
        
        batches
    }
    
    /// Mark a batch as completed (for progress tracking)
    pub fn mark_batch_completed(&self, batch_size: usize) {
        self.progress_tracker.mark_batch_completed(batch_size);
    }
    
    /// Get current progress statistics
    pub fn get_progress(&self) -> BatchStats {
        self.progress_tracker.get_stats()
    }
    
    /// Reset progress tracking
    pub fn reset_progress(&self) {
        self.progress_tracker.reset();
    }
    
    /// Calculate optimal batch size for given constraints
    pub fn calculate_optimal_batch_size(
        avg_text_length: usize,
        embedding_dimension: usize,
        available_memory_mb: usize,
        target_throughput_per_sec: usize,
    ) -> usize {
        // Memory-based calculation
        let text_memory_bytes = avg_text_length * 4; // Rough estimate
        let embedding_memory_bytes = embedding_dimension * 4; // f32
        let total_memory_per_item = text_memory_bytes + embedding_memory_bytes;
        
        let available_memory_bytes = available_memory_mb * 1_048_576;
        let memory_based_batch_size = available_memory_bytes / (total_memory_per_item * 4); // 25% utilization
        
        // Throughput-based calculation (rough heuristic)
        let throughput_based_batch_size = (target_throughput_per_sec / 10).max(1);
        
        // Take the minimum to respect memory constraints
        let optimal_size = memory_based_batch_size.min(throughput_based_batch_size);
        
        // Ensure reasonable bounds
        optimal_size.clamp(1, 128)
    }
}

impl ProgressTracker {
    fn new() -> Self {
        Self {
            total_items: AtomicUsize::new(0),
            processed_items: AtomicUsize::new(0),
            current_batch: AtomicUsize::new(0),
            total_batches: AtomicUsize::new(0),
        }
    }
    
    fn initialize(&self, total_items: usize, total_batches: usize) {
        self.total_items.store(total_items, Ordering::Relaxed);
        self.total_batches.store(total_batches, Ordering::Relaxed);
        self.processed_items.store(0, Ordering::Relaxed);
        self.current_batch.store(0, Ordering::Relaxed);
    }
    
    fn mark_batch_completed(&self, batch_size: usize) {
        self.processed_items.fetch_add(batch_size, Ordering::Relaxed);
        self.current_batch.fetch_add(1, Ordering::Relaxed);
        
        let progress = self.get_stats();
        debug!(
            "Completed batch {}/{} ({:.1}% complete)",
            progress.current_batch,
            progress.total_batches,
            progress.completion_percentage
        );
    }
    
    fn get_stats(&self) -> BatchStats {
        let total_items = self.total_items.load(Ordering::Relaxed);
        let processed_items = self.processed_items.load(Ordering::Relaxed);
        let current_batch = self.current_batch.load(Ordering::Relaxed);
        let total_batches = self.total_batches.load(Ordering::Relaxed);
        
        let avg_items_per_batch = if total_batches > 0 {
            total_items as f64 / total_batches as f64
        } else {
            0.0
        };
        
        let completion_percentage = if total_items > 0 {
            (processed_items as f64 / total_items as f64) * 100.0
        } else {
            0.0
        };
        
        BatchStats {
            total_items,
            total_batches,
            processed_items,
            current_batch,
            avg_items_per_batch,
            completion_percentage,
        }
    }
    
    fn reset(&self) {
        self.total_items.store(0, Ordering::Relaxed);
        self.processed_items.store(0, Ordering::Relaxed);
        self.current_batch.store(0, Ordering::Relaxed);
        self.total_batches.store(0, Ordering::Relaxed);
    }
}

/// Estimate memory usage for a text during processing
fn estimate_text_memory(text: &str, embedding_size_bytes: usize) -> usize {
    // Text storage (UTF-8) + tokenization overhead + embedding storage
    let text_bytes = text.len();
    let tokenization_overhead = text_bytes * 2; // Rough estimate
    let total_memory = text_bytes + tokenization_overhead + embedding_size_bytes;
    
    // Add 50% overhead for intermediate computations
    (total_memory as f64 * 1.5) as usize
}

/// Merge small batches to meet minimum size requirements
fn merge_small_batches(mut batches: Vec<Vec<String>>, min_batch_size: usize) -> Vec<Vec<String>> {
    if batches.is_empty() || min_batch_size <= 1 {
        return batches;
    }
    
    let mut merged_batches = Vec::new();
    let mut current_merge_batch = Vec::new();
    
    for batch in batches.drain(..) {
        if batch.len() >= min_batch_size {
            // This batch is large enough
            if !current_merge_batch.is_empty() {
                merged_batches.push(current_merge_batch);
                current_merge_batch = Vec::new();
            }
            merged_batches.push(batch);
        } else {
            // This batch is too small, add to merge batch
            current_merge_batch.extend(batch);
            
            // If merge batch is now large enough, finalize it
            if current_merge_batch.len() >= min_batch_size {
                merged_batches.push(current_merge_batch);
                current_merge_batch = Vec::new();
            }
        }
    }
    
    // Handle remaining items in merge batch
    if !current_merge_batch.is_empty() {
        if let Some(last_batch) = merged_batches.last_mut() {
            // Merge with the last batch if it exists
            last_batch.extend(current_merge_batch);
        } else {
            // No other batches, keep as is
            merged_batches.push(current_merge_batch);
        }
    }
    
    merged_batches
}

/// Batch processing utilities for working with different data types
pub struct BatchUtils;

impl BatchUtils {
    /// Split a large vector into chunks of specified size
    pub fn chunk_vector<T: Clone>(items: &[T], chunk_size: usize) -> Vec<Vec<T>> {
        if items.is_empty() || chunk_size == 0 {
            return Vec::new();
        }
        
        items.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
    
    /// Split items into a specific number of roughly equal batches
    pub fn split_into_n_batches<T: Clone>(items: &[T], n: usize) -> Vec<Vec<T>> {
        if items.is_empty() || n == 0 {
            return Vec::new();
        }
        
        let items_per_batch = (items.len() + n - 1) / n; // Ceiling division
        Self::chunk_vector(items, items_per_batch)
    }
    
    /// Balance batches to have roughly equal sizes
    pub fn balance_batches<T: Clone>(mut batches: Vec<Vec<T>>) -> Vec<Vec<T>> {
        if batches.len() <= 1 {
            return batches;
        }
        
        // Collect all items
        let all_items: Vec<T> = batches.drain(..)
            .flat_map(|batch| batch.into_iter())
            .collect();
        
        if all_items.is_empty() {
            return Vec::new();
        }
        
        let num_batches = batches.len().max(1);
        let items_per_batch = (all_items.len() + num_batches - 1) / num_batches;
        
        Self::chunk_vector(&all_items, items_per_batch)
    }
    
    /// Calculate memory usage for a batch of texts
    pub fn estimate_batch_memory(texts: &[String], embedding_dimension: usize) -> usize {
        let embedding_size_bytes = embedding_dimension * std::mem::size_of::<f32>();
        texts.iter()
            .map(|text| estimate_text_memory(text, embedding_size_bytes))
            .sum()
    }
}

impl Default for AdaptiveBatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 4,
            max_batch_size: 64,
            target_memory_mb: 512,
            avg_text_length_chars: 1000,
            embedding_dimension: 384,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new(32);
        assert_eq!(processor.batch_size, 32);
        assert!(processor.max_memory_mb.is_none());
    }
    
    #[test]
    fn test_create_batches() {
        let processor = BatchProcessor::new(3);
        let texts = vec![
            "text1".to_string(),
            "text2".to_string(), 
            "text3".to_string(),
            "text4".to_string(),
            "text5".to_string(),
        ];
        
        let batches = processor.create_batches(&texts);
        
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 2);
    }
    
    #[test]
    fn test_empty_input() {
        let processor = BatchProcessor::new(32);
        let texts: Vec<String> = vec![];
        let batches = processor.create_batches(&texts);
        assert!(batches.is_empty());
    }
    
    #[test]
    fn test_progress_tracking() {
        let processor = BatchProcessor::new(2);
        let texts = vec!["text1".to_string(), "text2".to_string(), "text3".to_string()];
        
        let batches = processor.create_batches(&texts);
        assert_eq!(batches.len(), 2);
        
        let progress = processor.get_progress();
        assert_eq!(progress.total_items, 3);
        assert_eq!(progress.total_batches, 2);
        assert_eq!(progress.processed_items, 0);
        
        processor.mark_batch_completed(2);
        let progress = processor.get_progress();
        assert_eq!(progress.processed_items, 2);
        assert_eq!(progress.current_batch, 1);
    }
    
    #[test]
    fn test_adaptive_batching() {
        let processor = BatchProcessor::new(32);
        let texts = vec![
            "short".to_string(),
            "medium length text here".to_string(),
            "this is a much longer text that should consume more memory during processing".to_string(),
            "short".to_string(),
        ];
        
        let config = AdaptiveBatchConfig {
            min_batch_size: 1,
            max_batch_size: 2,
            target_memory_mb: 1, // Very small to force splitting
            avg_text_length_chars: 50,
            embedding_dimension: 384,
        };
        
        let batches = processor.create_adaptive_batches(&texts, &config);
        
        // Should create more batches due to memory constraints
        assert!(batches.len() >= 2);
        
        // Each batch should respect max size
        for batch in &batches {
            assert!(batch.len() <= config.max_batch_size);
        }
    }
    
    #[test]
    fn test_optimal_batch_size_calculation() {
        let optimal_size = BatchProcessor::calculate_optimal_batch_size(
            1000, // avg_text_length
            384,  // embedding_dimension  
            1024, // available_memory_mb
            100,  // target_throughput_per_sec
        );
        
        assert!(optimal_size >= 1);
        assert!(optimal_size <= 128);
    }
    
    #[test]
    fn test_batch_utils_chunking() {
        let items = vec![1, 2, 3, 4, 5, 6, 7];
        let chunks = BatchUtils::chunk_vector(&items, 3);
        
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![1, 2, 3]);
        assert_eq!(chunks[1], vec![4, 5, 6]);
        assert_eq!(chunks[2], vec![7]);
    }
    
    #[test]
    fn test_batch_utils_split_into_n() {
        let items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let batches = BatchUtils::split_into_n_batches(&items, 3);
        
        assert_eq!(batches.len(), 3);
        // Each batch should have roughly equal size
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 3);
    }
    
    #[test]
    fn test_memory_estimation() {
        let texts = vec![
            "Hello world".to_string(),
            "This is a longer piece of text for testing".to_string(),
        ];
        
        let memory = BatchUtils::estimate_batch_memory(&texts, 384);
        assert!(memory > 0);
    }
    
    #[test]
    fn test_merge_small_batches() {
        let small_batches = vec![
            vec!["a".to_string()],
            vec!["b".to_string()],
            vec!["c".to_string(), "d".to_string(), "e".to_string()],
            vec!["f".to_string()],
        ];
        
        let merged = merge_small_batches(small_batches, 3);
        
        // Should merge small batches while keeping large ones separate
        assert!(merged.len() < 4);
        
        // All items should be preserved
        let total_items: usize = merged.iter().map(|b| b.len()).sum();
        assert_eq!(total_items, 6);
    }
}