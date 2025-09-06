use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

// ============================================================================
// Request/Response Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct IngestRequest {
    #[validate(length(min = 1, max = 1000000, message = "Content must be between 1 and 1,000,000 characters"))]
    pub content: String,
    pub content_type: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub chunking_strategy: Option<ChunkingStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResponse {
    pub task_id: Uuid,
    pub document_id: Uuid,
    pub status: TaskStatus,
    pub message: String,
    pub chunks_created: Option<usize>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct BatchIngestRequest {
    #[validate(length(min = 1, max = 100, message = "Batch must contain 1-100 documents"))]
    pub documents: Vec<IngestRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchIngestResponse {
    pub batch_id: Uuid,
    pub total_documents: usize,
    pub successful_ingestions: usize,
    pub failed_ingestions: usize,
    pub results: Vec<IngestResponse>,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct QueryRequest {
    #[serde(default = "Uuid::new_v4")]
    pub query_id: Uuid,
    #[validate(length(min = 1, max = 10000, message = "Query must be between 1 and 10,000 characters"))]
    pub query: String,
    pub user_id: Option<Uuid>,
    pub context: Option<serde_json::Value>,
    pub preferences: Option<QueryPreferences>,
    #[validate(range(min = 1, max = 100, message = "Max results must be between 1 and 100"))]
    pub max_results: Option<usize>,
    pub include_sources: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub query_id: Uuid,
    pub answer: String,
    pub sources: Vec<Source>,
    pub confidence_score: f64,
    pub processing_time_ms: u64,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingQueryResponse {
    pub query_id: Uuid,
    pub chunk: String,
    pub chunk_index: usize,
    pub is_final: bool,
    pub sources: Option<Vec<Source>>,
    pub metadata: Option<serde_json::Value>,
}

// ============================================================================
// Authentication Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct LoginRequest {
    #[validate(email(message = "Invalid email format"))]
    pub email: String,
    #[validate(length(min = 8, message = "Password must be at least 8 characters"))]
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthResponse {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_in: u64,
    pub user: UserInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    pub id: Uuid,
    pub email: String,
    pub name: String,
    pub role: UserRole,
    pub created_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
}

// ============================================================================
// Health and System Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: HealthStatus,
    pub version: String,
    pub timestamp: DateTime<Utc>,
    pub uptime_seconds: u64,
    pub checks: Vec<ComponentHealth>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: HealthStatus,
    pub details: Option<String>,
    pub last_check: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub version: String,
    pub build_info: BuildInfo,
    pub runtime_info: RuntimeInfo,
    pub component_versions: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    pub version: String,
    pub commit_hash: String,
    pub build_date: String,
    pub rust_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub uptime_seconds: u64,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
    pub active_connections: usize,
    pub processed_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatusResponse {
    pub components: std::collections::HashMap<String, ComponentStatus>,
    pub overall_status: HealthStatus,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentStatus {
    pub status: HealthStatus,
    pub url: String,
    pub last_check: DateTime<Utc>,
    pub response_time_ms: Option<u64>,
    pub error_message: Option<String>,
    pub version: Option<String>,
}

// ============================================================================
// Query History and Analytics Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryHistoryRequest {
    pub limit: usize,
    pub offset: usize,
    pub user_id: Option<Uuid>,
    pub start_date: Option<DateTime<Utc>>,
    pub end_date: Option<DateTime<Utc>>,
    pub status_filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryHistoryResponse {
    pub queries: Vec<QueryHistoryItem>,
    pub total_count: usize,
    pub page_info: PageInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryHistoryItem {
    pub query_id: Uuid,
    pub query: String,
    pub user_id: Option<Uuid>,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub processing_time_ms: Option<u64>,
    pub confidence_score: Option<f64>,
    pub result_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_response_time_ms: f64,
    pub average_confidence_score: f64,
    pub queries_per_hour: Vec<HourlyMetric>,
    pub popular_topics: Vec<TopicMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HourlyMetric {
    pub hour: DateTime<Utc>,
    pub query_count: u64,
    pub average_response_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicMetric {
    pub topic: String,
    pub query_count: u64,
    pub average_confidence: f64,
}

// ============================================================================
// File and Document Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileUploadResponse {
    pub file_id: Uuid,
    pub filename: String,
    pub content_type: String,
    pub size_bytes: u64,
    pub upload_url: Option<String>,
    pub processing_status: TaskStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStatus {
    pub task_id: Uuid,
    pub document_id: Option<Uuid>,
    pub status: TaskStatus,
    pub progress_percent: u8,
    pub current_stage: ProcessingStage,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub error_message: Option<String>,
    pub chunks_processed: Option<usize>,
    pub total_chunks: Option<usize>,
}

// ============================================================================
// Supporting Types and Enums
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    #[serde(rename = "healthy")]
    Healthy,
    #[serde(rename = "degraded")]
    Degraded,
    #[serde(rename = "unhealthy")]
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    #[serde(rename = "pending")]
    Pending,
    #[serde(rename = "processing")]
    Processing,
    #[serde(rename = "completed")]
    Completed,
    #[serde(rename = "failed")]
    Failed,
    #[serde(rename = "cancelled")]
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStage {
    #[serde(rename = "validation")]
    Validation,
    #[serde(rename = "chunking")]
    Chunking,
    #[serde(rename = "embedding")]
    Embedding,
    #[serde(rename = "storage")]
    Storage,
    #[serde(rename = "indexing")]
    Indexing,
    #[serde(rename = "completed")]
    Completed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserRole {
    #[serde(rename = "user")]
    User,
    #[serde(rename = "admin")]
    Admin,
    #[serde(rename = "system")]
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingStrategy {
    pub strategy_type: ChunkingType,
    pub max_chunk_size: Option<usize>,
    pub overlap_size: Option<usize>,
    pub preserve_structure: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingType {
    #[serde(rename = "fixed")]
    Fixed,
    #[serde(rename = "semantic")]
    Semantic,
    #[serde(rename = "adaptive")]
    Adaptive,
    #[serde(rename = "neural")]
    Neural,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPreferences {
    pub response_format: Option<ResponseFormat>,
    pub include_citations: Option<bool>,
    pub max_context_length: Option<usize>,
    pub temperature: Option<f64>,
    pub streaming: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "markdown")]
    Markdown,
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "structured")]
    Structured,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub document_id: Uuid,
    pub chunk_id: Option<Uuid>,
    pub title: Option<String>,
    pub content_preview: String,
    pub relevance_score: f64,
    pub metadata: Option<serde_json::Value>,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageInfo {
    pub current_page: usize,
    pub per_page: usize,
    pub total_pages: usize,
    pub total_items: usize,
    pub has_next: bool,
    pub has_previous: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestTask {
    pub task_id: Uuid,
    pub document_id: Uuid,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

// ============================================================================
// Storage Service Models
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingHistory {
    pub total_documents: u64,
    pub successful_processing: u64,
    pub failed_processing: u64,
    pub processing_time_stats: ProcessingTimeStats,
    pub recent_activity: Vec<ProcessingActivity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingTimeStats {
    pub average_ms: f64,
    pub min_ms: u64,
    pub max_ms: u64,
    pub median_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingActivity {
    pub timestamp: DateTime<Utc>,
    pub document_id: Uuid,
    pub status: TaskStatus,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageUsage {
    pub total_documents: u64,
    pub total_chunks: u64,
    pub storage_size_bytes: u64,
    pub average_document_size_bytes: u64,
    pub average_chunks_per_document: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInfo {
    pub document_id: Uuid,
    pub title: Option<String>,
    pub content_type: Option<String>,
    pub created_at: DateTime<Utc>,
    pub chunk_count: usize,
    pub size_bytes: u64,
    pub status: TaskStatus,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    pub total_processed: u64,
    pub successful_processing: u64,
    pub failed_processing: u64,
    pub average_processing_time_ms: f64,
    pub documents_per_hour: f64,
    pub error_rate_percent: f64,
    pub last_24h_stats: DailyStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyStats {
    pub processed: u64,
    pub successful: u64,
    pub failed: u64,
    pub average_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTypeStatistics {
    pub content_types: Vec<ContentTypeStat>,
    pub total_documents: u64,
    pub most_common_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentTypeStat {
    pub content_type: String,
    pub document_count: u64,
    pub percentage: f64,
    pub average_size_bytes: u64,
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

// ============================================================================
// Default Implementations
// ============================================================================

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self {
            strategy_type: ChunkingType::Adaptive,
            max_chunk_size: Some(1024),
            overlap_size: Some(128),
            preserve_structure: Some(true),
        }
    }
}

impl Default for QueryPreferences {
    fn default() -> Self {
        Self {
            response_format: Some(ResponseFormat::Text),
            include_citations: Some(true),
            max_context_length: Some(4000),
            temperature: Some(0.7),
            streaming: Some(false),
        }
    }
}

impl Default for PageInfo {
    fn default() -> Self {
        Self {
            current_page: 1,
            per_page: 50,
            total_pages: 0,
            total_items: 0,
            has_next: false,
            has_previous: false,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

impl HealthStatus {
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    pub fn is_available(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }
}

impl TaskStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled)
    }

    pub fn is_processing(&self) -> bool {
        matches!(self, TaskStatus::Pending | TaskStatus::Processing)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_methods() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Degraded.is_healthy());
        assert!(!HealthStatus::Unhealthy.is_healthy());

        assert!(HealthStatus::Healthy.is_available());
        assert!(HealthStatus::Degraded.is_available());
        assert!(!HealthStatus::Unhealthy.is_available());
    }

    #[test]
    fn test_task_status_methods() {
        assert!(TaskStatus::Completed.is_terminal());
        assert!(TaskStatus::Failed.is_terminal());
        assert!(!TaskStatus::Pending.is_terminal());

        assert!(TaskStatus::Pending.is_processing());
        assert!(TaskStatus::Processing.is_processing());
        assert!(!TaskStatus::Completed.is_processing());
    }

    #[test]
    fn test_default_implementations() {
        let chunking = ChunkingStrategy::default();
        assert_eq!(chunking.max_chunk_size, Some(1024));
        assert_eq!(chunking.overlap_size, Some(128));

        let preferences = QueryPreferences::default();
        assert_eq!(preferences.temperature, Some(0.7));
        assert_eq!(preferences.include_citations, Some(true));

        let page_info = PageInfo::default();
        assert_eq!(page_info.current_page, 1);
        assert_eq!(page_info.per_page, 50);
    }

    #[test]
    fn test_uuid_generation() {
        let request = QueryRequest {
            query_id: Uuid::new_v4(),
            query: "test query".to_string(),
            user_id: None,
            context: None,
            preferences: None,
            max_results: Some(10),
            include_sources: Some(true),
        };

        assert_ne!(request.query_id, Uuid::nil());
    }
}