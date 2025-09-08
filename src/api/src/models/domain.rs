use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
// Note: StorageUsage is imported via the re-exports in mod.rs

// ============================================================================
// Processing History and Statistics Domain Types
// ============================================================================



/// Processing status for entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStatus {
    Success,
    Failed(String),
    Partial,
    Pending,
    Processing,
}


/// Document information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInfo {
    pub id: Uuid,
    pub name: String,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
    pub chunk_count: usize,
    pub content_type: Option<String>,
}



/// Processing task information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTask {
    pub id: Uuid,
    pub document_id: Uuid,
    pub status: DomainTaskStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub progress_percent: Option<u8>,
    pub error_message: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

/// Processing statistics for the domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    pub total_processed: u64,
    pub success_count: u64,
    pub failed_count: u64,
    pub success_rate: f64,
    pub average_processing_time_ms: f64,
    pub last_updated: DateTime<Utc>,
}

/// Content type statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTypeStatistics {
    pub total_files: u64,
    pub by_type: HashMap<String, ContentTypeStat>,
    pub last_updated: DateTime<Utc>,
}

/// Individual content type statistic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTypeStat {
    pub count: u64,
    pub total_size_bytes: u64,
    pub percentage: f64,
}

/// Domain task status enumeration (to avoid conflicts with API TaskStatus)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainTaskStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
    Cancelled,
}


// ============================================================================
// Default Implementations
// ============================================================================


// ============================================================================
// Helper Methods
// ============================================================================

impl DomainTaskStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            DomainTaskStatus::Completed | DomainTaskStatus::Failed(_) | DomainTaskStatus::Cancelled
        )
    }

    pub fn is_processing(&self) -> bool {
        matches!(self, DomainTaskStatus::Pending | DomainTaskStatus::Processing)
    }
}

impl ProcessingStatus {
    pub fn is_successful(&self) -> bool {
        matches!(self, ProcessingStatus::Success)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, ProcessingStatus::Failed(_))
    }
}

impl Default for ProcessingStatistics {
    fn default() -> Self {
        Self {
            total_processed: 0,
            success_count: 0,
            failed_count: 0,
            success_rate: 0.0,
            average_processing_time_ms: 0.0,
            last_updated: Utc::now(),
        }
    }
}

impl Default for ContentTypeStatistics {
    fn default() -> Self {
        Self {
            total_files: 0,
            by_type: HashMap::new(),
            last_updated: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_status_methods() {
        assert!(DomainTaskStatus::Completed.is_terminal());
        assert!(DomainTaskStatus::Failed("error".to_string()).is_terminal());
        assert!(DomainTaskStatus::Cancelled.is_terminal());
        assert!(!DomainTaskStatus::Pending.is_terminal());
        assert!(!DomainTaskStatus::Processing.is_terminal());

        assert!(DomainTaskStatus::Pending.is_processing());
        assert!(DomainTaskStatus::Processing.is_processing());
        assert!(!DomainTaskStatus::Completed.is_processing());
    }

    #[test]
    fn test_processing_status_methods() {
        assert!(ProcessingStatus::Success.is_successful());
        assert!(!ProcessingStatus::Failed("error".to_string()).is_successful());
        
        assert!(ProcessingStatus::Failed("error".to_string()).is_failed());
        assert!(!ProcessingStatus::Success.is_failed());
    }

    #[test]
    fn test_default_implementations() {
        let storage = crate::models::StorageUsage {
            total_documents: 0,
            total_chunks: 0,
            total_size_bytes: 0,
            storage_by_type: Vec::new(),
            last_updated: Utc::now(),
        };
        assert_eq!(storage.total_size_bytes, 0);
        assert_eq!(storage.total_documents, 0);

        let stats = ProcessingStatistics {
            total_processed: 0,
            success_count: 0,
            failed_count: 0,
            success_rate: 0.0,
            average_processing_time_ms: 0.0,
            last_updated: Utc::now(),
        };
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.success_rate, 0.0);

        let content_stats = ContentTypeStatistics {
            total_files: 0,
            by_type: HashMap::new(),
            last_updated: Utc::now(),
        };
        assert_eq!(content_stats.total_files, 0);
        assert!(content_stats.by_type.is_empty());
    }
}