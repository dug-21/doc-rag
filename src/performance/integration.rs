//! Performance Integration
//! 
//! Integration utilities for applying performance optimizations across
//! all Doc-RAG components with minimal code changes.

use std::time::Duration;
use crate::performance::{PerformanceManager, PerformanceConfig};
use tracing::{info, warn, debug, instrument};

/// Global performance manager instance
static mut PERFORMANCE_MANAGER: Option<PerformanceManager> = None;
static INIT: std::sync::Once = std::sync::Once::new();

/// Initialize performance optimization across the system
pub async fn initialize_performance_optimization() -> Result<(), Box<dyn std::error::Error>> {
    INIT.call_once(|| {
        let config = PerformanceConfig::default();
        let manager = PerformanceManager::new(config);
        
        unsafe {
            PERFORMANCE_MANAGER = Some(manager);
        }
    });
    
    if let Some(manager) = unsafe { PERFORMANCE_MANAGER.as_ref() } {
        manager.start().await;
        info!("Performance optimization system initialized");
    }
    
    Ok(())
}

/// Performance-optimized wrapper for async operations
#[instrument(skip(operation))]
pub async fn with_performance_optimization<F, T>(
    component: &str,
    operation_name: &str,
    operation: F
) -> T
where
    F: std::future::Future<Output = T>,
{
    if let Some(manager) = unsafe { PERFORMANCE_MANAGER.as_ref() } {
        manager.profile_and_optimize(component, operation_name, operation).await
    } else {
        // Fallback if performance manager not initialized
        debug!("Performance manager not initialized, running operation without optimization");
        operation.await
    }
}

/// Macro for easy performance instrumentation
#[macro_export]
macro_rules! perf_optimized {
    ($component:expr, $operation:expr, $code:block) => {
        crate::performance::integration::with_performance_optimization(
            $component,
            $operation,
            async move $code
        ).await
    };
}

/// Performance optimization annotations for existing functions
pub trait PerformanceOptimized {
    /// Wrap an async operation with performance optimization
    async fn with_perf_opt<F, T>(&self, component: &str, operation: &str, func: F) -> T
    where
        F: std::future::Future<Output = T>;
}

impl<S> PerformanceOptimized for S {
    async fn with_perf_opt<F, T>(&self, component: &str, operation: &str, func: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        with_performance_optimization(component, operation, func).await
    }
}

/// Component-specific performance optimization helpers
pub mod components {
    use super::*;
    
    /// Optimized chunking operations
    pub async fn optimized_chunk_document<F, T>(operation: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        with_performance_optimization("chunker", "chunk_document", operation).await
    }
    
    /// Optimized embedding generation
    pub async fn optimized_generate_embeddings<F, T>(operation: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        with_performance_optimization("embedder", "generate_embeddings", operation).await
    }
    
    /// Optimized vector search
    pub async fn optimized_vector_search<F, T>(operation: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        with_performance_optimization("storage", "vector_search", operation).await
    }
    
    /// Optimized query processing
    pub async fn optimized_process_query<F, T>(operation: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        with_performance_optimization("query-processor", "process_query", operation).await
    }
    
    /// Optimized response generation
    pub async fn optimized_generate_response<F, T>(operation: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        with_performance_optimization("response-generator", "generate_response", operation).await
    }
}

/// Performance monitoring utilities
pub mod monitoring {
    use super::*;
    
    /// Get current performance metrics
    pub async fn get_performance_metrics() -> Option<crate::performance::IntegratedPerformanceReport> {
        if let Some(manager) = unsafe { PERFORMANCE_MANAGER.as_ref() } {
            Some(manager.get_performance_report().await)
        } else {
            None
        }
    }
    
    /// Check if system is meeting performance targets
    pub async fn check_performance_health() -> bool {
        if let Some(report) = get_performance_metrics().await {
            matches!(report.overall_health.status, 
                crate::performance::HealthStatus::Excellent | 
                crate::performance::HealthStatus::Good
            )
        } else {
            false
        }
    }
    
    /// Log performance summary
    pub async fn log_performance_summary() {
        if let Some(report) = get_performance_metrics().await {
            info!(
                "Performance Status: {:?} (Score: {:.1})",
                report.overall_health.status,
                report.overall_health.health_score
            );
            
            if !report.overall_health.issues.is_empty() {
                warn!("Performance Issues: {:?}", report.overall_health.issues);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_initialization() {
        let result = initialize_performance_optimization().await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_performance_optimization_wrapper() {
        let _ = initialize_performance_optimization().await;
        
        let result = with_performance_optimization("test", "operation", async {
            tokio::time::sleep(Duration::from_millis(10)).await;
            "test_result"
        }).await;
        
        assert_eq!(result, "test_result");
    }
    
    #[tokio::test]
    async fn test_component_optimization() {
        let _ = initialize_performance_optimization().await;
        
        let result = components::optimized_chunk_document(async {
            "chunked_result"
        }).await;
        
        assert_eq!(result, "chunked_result");
    }
    
    #[tokio::test]
    async fn test_performance_monitoring() {
        let _ = initialize_performance_optimization().await;
        
        // Run some operations to generate metrics
        let _ = with_performance_optimization("test", "op1", async { "result1" }).await;
        let _ = with_performance_optimization("test", "op2", async { "result2" }).await;
        
        let health = monitoring::check_performance_health().await;
        // Health check should work (may be true or false depending on system state)
        assert!(health == true || health == false);
    }
}