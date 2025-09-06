use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

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
        let storage = StorageUsage::default();
        assert_eq!(storage.total_bytes, 0);
        assert_eq!(storage.document_count, 0);

        let stats = ProcessingStatistics::default();
        assert_eq!(stats.total_processed, 0);
        assert_eq!(stats.success_rate, 0.0);

        let content_stats = ContentTypeStatistics::default();
        assert_eq!(content_stats.total_files, 0);
        assert!(content_stats.by_type.is_empty());
    }
}