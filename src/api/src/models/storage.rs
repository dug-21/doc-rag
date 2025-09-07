use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::api::{ProcessingStage, TaskStatus};

// ============================================================================
// Storage and Processing Models for StorageServiceClient
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingHistory {
    pub entries: Vec<ProcessingEntry>,
    pub total_processed: u64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingEntry {
    pub id: Uuid,
    pub document_id: Uuid,
    pub task_id: Uuid,
    pub stage: ProcessingStage,
    pub status: TaskStatus,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub processing_time_ms: Option<u64>,
    pub error_message: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageUsage {
    pub total_documents: u64,
    pub total_chunks: u64,
    pub total_size_bytes: u64,
    pub storage_by_type: Vec<ContentTypeUsage>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTypeUsage {
    pub content_type: String,
    pub document_count: u64,
    pub total_size_bytes: u64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentDocument {
    pub document_id: Uuid,
    pub title: Option<String>,
    pub content_type: Option<String>,
    pub size_bytes: u64,
    pub chunk_count: u32,
    pub created_at: DateTime<Utc>,
    pub last_accessed: Option<DateTime<Utc>>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    pub total_processing_tasks: u64,
    pub successful_tasks: u64,
    pub failed_tasks: u64,
    pub average_processing_time_ms: f64,
    pub processing_by_stage: Vec<StageStatistics>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageStatistics {
    pub stage: ProcessingStage,
    pub total_tasks: u64,
    pub successful_tasks: u64,
    pub failed_tasks: u64,
    pub average_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTypeStatistics {
    pub statistics: Vec<ContentTypeStats>,
    pub total_types: u32,
    pub most_common_type: Option<String>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTypeStats {
    pub content_type: String,
    pub document_count: u64,
    pub chunk_count: u64,
    pub total_size_bytes: u64,
    pub average_processing_time_ms: f64,
    pub success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatusResponse {
    pub task_id: Uuid,
    pub status: TaskStatus,
    pub message: Option<String>,
    pub progress_percent: Option<u8>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDetails {
    pub task_id: Uuid,
    pub document_id: Option<Uuid>,
    pub content: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub chunking_strategy: Option<crate::models::ChunkingStrategy>,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}