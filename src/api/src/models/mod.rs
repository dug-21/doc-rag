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

// Re-export storage-specific types
pub use storage::*;

// Re-export domain-specific types
pub use domain::*;