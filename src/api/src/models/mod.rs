// API request/response models
pub mod api;

// Storage-specific models for client operations
pub mod storage;

// Domain types for business logic  
pub mod domain;

// Re-export from api, but not the conflicting types
pub use api::{
    IngestRequest, IngestResponse, BatchIngestRequest, BatchIngestResponse,
    QueryRequest, QueryResponse, StreamingQueryResponse, LoginRequest, AuthResponse, UserInfo,
    HealthResponse, ComponentHealth, SystemInfo, BuildInfo, RuntimeInfo, ComponentStatusResponse, ComponentStatus,
    QueryHistoryRequest, QueryHistoryResponse, QueryHistoryItem, QueryMetrics, HourlyMetric, TopicMetric,
    FileUploadResponse, DocumentStatus, HealthStatus, TaskStatus, ProcessingStage, UserRole,
    ChunkingStrategy, ChunkingType, QueryPreferences, ResponseFormat, Source, PageInfo, IngestTask
};

// Re-export storage-specific types - avoid conflicts by specifying individual types
pub use storage::{
    ProcessingHistory, ProcessingEntry, StorageUsage, ContentTypeUsage, RecentDocument, 
    TaskDetails, TaskStatusResponse, ContentTypeStats
};

// Use storage ProcessingStatistics specifically
pub use storage::ProcessingStatistics as StorageProcessingStatistics;
pub use storage::StageStatistics;

// Re-export domain-specific types - avoid conflicts 
pub use domain::{
    ProcessingStatus, DocumentInfo, ProcessingTask, DomainTaskStatus, 
    ContentTypeStatistics, ContentTypeStat
};

// Use domain ProcessingStatistics specifically
pub use domain::ProcessingStatistics as DomainProcessingStatistics;