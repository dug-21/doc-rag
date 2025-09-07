//! Unit tests for MongoDB Vector Storage components

use std::collections::HashMap;
use uuid::Uuid;
use chrono::Utc;

use storage::{
    ChunkDocument, ChunkMetadata, CustomFieldValue, ChunkReference, ReferenceType,
    MetadataDocument, DocumentMetadata, ProcessingStats, QualityMetrics, SecurityLevel,
    VectorSimilarity, SearchQuery, SearchType, SearchFilters, SortOptions, SortField, SortDirection,
    StorageConfig, VectorSearchConfig, TextSearchConfig, PerformanceConfig,
    StorageError, ErrorContext, RecoveryStrategy, WithContext,
    StorageMetrics, OperationMetrics, PerformanceMetrics, ErrorMetrics,
    BulkInsertRequest, BulkInsertResponse, BulkInsertError,
};

#[test]
fn test_chunk_document_creation() {
    let document_id = Uuid::new_v4();
    let chunk_id = Uuid::new_v4();
    
    let metadata = ChunkMetadata::new(
        document_id,
        "Test Document".to_string(),
        0,
        5,
        "/path/to/test.txt".to_string(),
    );
    
    let chunk = ChunkDocument::new(
        chunk_id,
        "This is test content for the chunk.".to_string(),
        metadata,
    );
    
    assert_eq!(chunk.chunk_id, chunk_id);
    assert_eq!(chunk.content, "This is test content for the chunk.");
    assert_eq!(chunk.version, 1);
    assert!(chunk.embedding.is_none());
    assert!(chunk.references.is_empty());
    assert!(chunk.validate().is_ok());
}

#[test]
fn test_chunk_with_embedding() {
    let document_id = Uuid::new_v4();
    let chunk_id = Uuid::new_v4();
    let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    
    let metadata = ChunkMetadata::new(
        document_id,
        "Test Document".to_string(),
        0,
        1,
        "/path/to/test.txt".to_string(),
    );
    
    let chunk = ChunkDocument::new(
        chunk_id,
        "Test content".to_string(),
        metadata,
    ).with_embedding(embedding.clone());
    
    assert_eq!(chunk.embedding, Some(embedding));
    assert_eq!(chunk.version, 2); // Incremented when embedding added
    assert!(chunk.validate().is_ok());
}

#[test]
fn test_chunk_references() {
    let document_id = Uuid::new_v4();
    let chunk_id = Uuid::new_v4();
    let reference_id = Uuid::new_v4();
    
    let metadata = ChunkMetadata::new(
        document_id,
        "Test Document".to_string(),
        0,
        1,
        "/path/to/test.txt".to_string(),
    );
    
    let mut chunk = ChunkDocument::new(
        chunk_id,
        "Test content".to_string(),
        metadata,
    );
    
    let reference = ChunkReference {
        chunk_id: reference_id,
        reference_type: ReferenceType::Sequential,
        confidence: 0.95,
        context: "Previous chunk in sequence".to_string(),
    };
    
    let initial_version = chunk.version;
    chunk.add_reference(reference.clone());
    
    assert_eq!(chunk.references.len(), 1);
    assert_eq!(chunk.references[0].chunk_id, reference_id);
    assert_eq!(chunk.version, initial_version + 1); // Version incremented
}

#[test]
fn test_chunk_metadata() {
    let document_id = Uuid::new_v4();
    
    let mut metadata = ChunkMetadata::new(
        document_id,
        "Test Document".to_string(),
        2,
        10,
        "/documents/test.pdf".to_string(),
    );
    
    // Test basic fields
    assert_eq!(metadata.document_id, document_id);
    assert_eq!(metadata.chunk_index, 2);
    assert_eq!(metadata.total_chunks, 10);
    assert_eq!(metadata.language, "en");
    
    // Test tag operations
    metadata.add_tag("important".to_string());
    metadata.add_tag("reviewed".to_string());
    metadata.add_tag("important".to_string()); // Duplicate should not be added
    
    assert_eq!(metadata.tags.len(), 2);
    assert!(metadata.tags.contains(&"important".to_string()));
    assert!(metadata.tags.contains(&"reviewed".to_string()));
    
    // Test custom fields
    metadata.set_custom_field(
        "priority".to_string(),
        CustomFieldValue::String("high".to_string()),
    );
    metadata.set_custom_field(
        "score".to_string(),
        CustomFieldValue::Number(85.5),
    );
    metadata.set_custom_field(
        "approved".to_string(),
        CustomFieldValue::Boolean(true),
    );
    
    assert_eq!(metadata.custom_fields.len(), 3);
    
    match metadata.get_custom_field("priority") {
        Some(CustomFieldValue::String(s)) => assert_eq!(s, "high"),
        _ => panic!("Expected string value for priority"),
    }
    
    match metadata.get_custom_field("score") {
        Some(CustomFieldValue::Number(n)) => assert_eq!(*n, 85.5),
        _ => panic!("Expected number value for score"),
    }
    
    match metadata.get_custom_field("approved") {
        Some(CustomFieldValue::Boolean(b)) => assert!(b),
        _ => panic!("Expected boolean value for approved"),
    }
}

#[test]
fn test_chunk_validation() {
    let document_id = Uuid::new_v4();
    
    // Valid chunk
    let valid_metadata = ChunkMetadata::new(
        document_id,
        "Test Document".to_string(),
        0,
        1,
        "/test.txt".to_string(),
    );
    
    let valid_chunk = ChunkDocument::new(
        Uuid::new_v4(),
        "Valid content".to_string(),
        valid_metadata,
    );
    
    assert!(valid_chunk.validate().is_ok());
    
    // Invalid chunk - empty content
    let mut invalid_chunk = valid_chunk.clone();
    invalid_chunk.content = String::new();
    
    assert!(invalid_chunk.validate().is_err());
    
    // Invalid chunk - chunk index >= total chunks
    let mut invalid_metadata = ChunkMetadata::new(
        document_id,
        "Test Document".to_string(),
        5,
        5, // chunk_index should be < total_chunks
        "/test.txt".to_string(),
    );
    invalid_metadata.chunk_size = 100;
    
    let invalid_chunk2 = ChunkDocument::new(
        Uuid::new_v4(),
        "Content".to_string(),
        invalid_metadata,
    );
    
    assert!(invalid_chunk2.validate().is_err());
    
    // Invalid chunk - empty embedding if present
    let mut invalid_chunk3 = valid_chunk.clone();
    invalid_chunk3.embedding = Some(vec![]);
    
    assert!(invalid_chunk3.validate().is_err());
}

#[test]
fn test_metadata_document() {
    let document_id = Uuid::new_v4();
    
    let doc_metadata = DocumentMetadata {
        title: "Test Document".to_string(),
        author: Some("Test Author".to_string()),
        document_created_at: Some(Utc::now()),
        file_size: 1024,
        format: "PDF".to_string(),
        summary: Some("This is a test document".to_string()),
        keywords: vec!["test".to_string(), "document".to_string()],
        classification: Some("Internal".to_string()),
        security_level: SecurityLevel::Internal,
    };
    
    let metadata_doc = MetadataDocument::new(document_id, doc_metadata);
    
    assert_eq!(metadata_doc.document_id, document_id);
    assert_eq!(metadata_doc.metadata.title, "Test Document");
    assert_eq!(metadata_doc.metadata.file_size, 1024);
    assert!(metadata_doc.metadata.author.is_some());
}

#[test]
fn test_vector_similarity_calculations() {
    // Test cosine similarity
    let vec1 = vec![1.0, 2.0, 3.0];
    let vec2 = vec![4.0, 5.0, 6.0];
    
    let cosine_sim = VectorSimilarity::cosine_similarity(&vec1, &vec2).unwrap();
    assert!(cosine_sim > 0.0 && cosine_sim <= 1.0);
    
    // Test identical vectors
    let identical_sim = VectorSimilarity::cosine_similarity(&vec1, &vec1).unwrap();
    assert!((identical_sim - 1.0).abs() < 1e-10); // Should be exactly 1.0
    
    // Test orthogonal vectors
    let ortho1 = vec![1.0, 0.0];
    let ortho2 = vec![0.0, 1.0];
    let ortho_sim = VectorSimilarity::cosine_similarity(&ortho1, &ortho2).unwrap();
    assert!(ortho_sim.abs() < 1e-10); // Should be 0.0
    
    // Test euclidean distance
    let euclidean_dist = VectorSimilarity::euclidean_distance(&vec1, &vec2).unwrap();
    assert!(euclidean_dist > 0.0);
    
    // Test distance to similarity conversion
    let similarity = VectorSimilarity::distance_to_similarity(euclidean_dist, 10.0);
    assert!(similarity >= 0.0 && similarity <= 1.0);
    
    // Test error cases
    let vec3 = vec![1.0, 2.0]; // Different dimension
    assert!(VectorSimilarity::cosine_similarity(&vec1, &vec3).is_err());
    assert!(VectorSimilarity::euclidean_distance(&vec1, &vec3).is_err());
}

#[test]
fn test_search_query_builder() {
    let embedding = vec![0.1, 0.2, 0.3];
    let document_ids = vec![Uuid::new_v4(), Uuid::new_v4()];
    
    let query = SearchQuery {
        query_embedding: Some(embedding.clone()),
        text_query: Some("test query".to_string()),
        search_type: SearchType::Hybrid,
        limit: 20,
        offset: 10,
        min_score: Some(0.5),
        filters: SearchFilters {
            document_ids: Some(document_ids.clone()),
            tags: Some(vec!["important".to_string()]),
            language: Some("en".to_string()),
            ..Default::default()
        },
        sort: SortOptions {
            field: SortField::CreatedAt,
            direction: SortDirection::Descending,
            secondary: Some(Box::new(SortOptions {
                field: SortField::Relevance,
                direction: SortDirection::Descending,
                secondary: None,
            })),
        },
    };
    
    assert_eq!(query.query_embedding, Some(embedding));
    assert_eq!(query.text_query, Some("test query".to_string()));
    assert_eq!(query.limit, 20);
    assert_eq!(query.offset, 10);
    assert_eq!(query.min_score, Some(0.5));
    
    assert_eq!(query.filters.document_ids, Some(document_ids));
    assert_eq!(query.filters.tags, Some(vec!["important".to_string()]));
    assert_eq!(query.filters.language, Some("en".to_string()));
    
    assert!(matches!(query.sort.field, SortField::CreatedAt));
    assert!(matches!(query.sort.direction, SortDirection::Descending));
    assert!(query.sort.secondary.is_some());
}

#[test]
fn test_default_search_query() {
    let query = SearchQuery::default();
    
    assert!(query.query_embedding.is_none());
    assert!(query.text_query.is_none());
    assert!(matches!(query.search_type, SearchType::Hybrid));
    assert_eq!(query.limit, 10);
    assert_eq!(query.offset, 0);
    assert!(query.min_score.is_none());
    assert!(matches!(query.sort.field, SortField::Relevance));
    assert!(matches!(query.sort.direction, SortDirection::Descending));
}

#[test]
fn test_storage_config() {
    let config = StorageConfig::default();
    
    // Test validation of default config
    assert!(config.validate().is_ok());
    
    // Test invalid configurations
    let mut invalid_config = config.clone();
    invalid_config.connection_string = String::new();
    assert!(invalid_config.validate().is_err());
    
    let mut invalid_config2 = config.clone();
    invalid_config2.max_pool_size = 0;
    assert!(invalid_config2.validate().is_err());
    
    let mut invalid_config3 = config.clone();
    invalid_config3.min_pool_size = 20;
    invalid_config3.max_pool_size = 10;
    assert!(invalid_config3.validate().is_err());
    
    // Test testing config
    let test_config = StorageConfig::for_testing();
    assert!(test_config.validate().is_ok());
    assert!(test_config.database_name.starts_with("test_rag_storage_"));
    assert_eq!(test_config.connection_timeout_secs, 5);
}

#[test]
fn test_config_durations() {
    let config = StorageConfig::default();
    
    assert_eq!(config.connection_timeout().as_secs(), 10);
    assert_eq!(config.operation_timeout().as_secs(), 30);
    assert_eq!(config.health_check_interval().as_secs(), 60);
}

#[test]
fn test_storage_errors() {
    let chunk_id = Uuid::new_v4();
    let document_id = Uuid::new_v4();
    
    // Test error categorization
    let connection_error = StorageError::ConnectionError("Failed to connect".to_string());
    assert!(connection_error.is_retryable());
    assert!(connection_error.is_server_error());
    assert!(!connection_error.is_client_error());
    assert_eq!(connection_error.category(), "connection");
    assert_eq!(connection_error.status_code(), 503);
    
    let not_found_error = StorageError::ChunkNotFound(chunk_id);
    assert!(!not_found_error.is_retryable());
    assert!(!not_found_error.is_server_error());
    assert!(not_found_error.is_client_error());
    assert_eq!(not_found_error.category(), "not_found");
    assert_eq!(not_found_error.status_code(), 404);
    
    let validation_error = StorageError::ValidationError("Invalid data".to_string());
    assert!(!validation_error.is_retryable());
    assert!(validation_error.is_client_error());
    assert_eq!(validation_error.category(), "validation");
    assert_eq!(validation_error.status_code(), 400);
}

#[test]
fn test_error_context() {
    let chunk_id = Uuid::new_v4();
    let error = StorageError::ChunkNotFound(chunk_id);
    
    let context = ErrorContext::new("get_chunk")
        .with_context("chunk_id", chunk_id.to_string())
        .with_request_id("req-123");
    
    let contextual_error = error.with_detailed_context(context).unwrap_err();
    
    let error_string = contextual_error.to_string();
    assert!(error_string.contains("get_chunk"));
    assert!(error_string.contains("req-123"));
    assert!(error_string.contains(&chunk_id.to_string()));
}

#[test]
fn test_recovery_strategy() {
    let connection_error = StorageError::ConnectionError("Failed to connect".to_string());
    let strategy = RecoveryStrategy::for_error(&connection_error);
    
    match strategy {
        RecoveryStrategy::Retry { max_attempts, base_delay_ms } => {
            assert_eq!(max_attempts, 3);
            assert_eq!(base_delay_ms, 1000);
            
            // Test delay calculation
            let delay1 = strategy.calculate_delay(1).unwrap();
            let delay2 = strategy.calculate_delay(2).unwrap();
            let delay3 = strategy.calculate_delay(3).unwrap();
            let delay4 = strategy.calculate_delay(4);
            
            assert_eq!(delay1.as_millis(), 1000);
            assert_eq!(delay2.as_millis(), 2000);
            assert_eq!(delay3.as_millis(), 4000);
            assert!(delay4.is_none()); // Exceeded max attempts
        }
        _ => panic!("Expected retry strategy for connection error"),
    }
    
    let not_found_error = StorageError::ChunkNotFound(Uuid::new_v4());
    let fail_fast_strategy = RecoveryStrategy::for_error(&not_found_error);
    
    assert!(matches!(fail_fast_strategy, RecoveryStrategy::FailFast));
}

#[test]
fn test_metrics() {
    let metrics = StorageMetrics::new();
    
    // Test operation recording
    metrics.record_operation_duration("insert", std::time::Duration::from_millis(100));
    metrics.record_operation_duration("insert", std::time::Duration::from_millis(200));
    metrics.record_operation_error("insert", "validation", "Invalid data");
    
    let insert_metrics = metrics.get_operation_metrics("insert").unwrap();
    assert_eq!(insert_metrics.count, 3);
    assert_eq!(insert_metrics.success_count, 2);
    assert_eq!(insert_metrics.error_count, 1);
    assert_eq!(insert_metrics.avg_duration_ms(), 100.0); // (100 + 200) / 2 operations with duration
    assert_eq!(insert_metrics.success_rate(), 2.0 / 3.0);
    assert_eq!(insert_metrics.error_rate(), 1.0 / 3.0);
    
    // Test performance metrics
    metrics.record_documents_processed(100);
    metrics.record_bytes_processed(1024);
    
    let perf = metrics.get_performance_metrics();
    assert_eq!(perf.documents_processed, 100);
    assert_eq!(perf.bytes_processed, 1024);
    
    // Test cache metrics
    metrics.record_cache_hit();
    metrics.record_cache_hit();
    metrics.record_cache_miss();
    
    let updated_perf = metrics.get_performance_metrics();
    assert!(updated_perf.cache_hit_ratio > 0.0);
    assert!(updated_perf.cache_miss_ratio > 0.0);
    
    // Test metrics snapshot
    let snapshot = metrics.snapshot();
    assert!(!snapshot.operations.is_empty());
    assert_eq!(snapshot.performance.documents_processed, 100);
}

#[test]
fn test_bulk_operations() {
    let document_id = Uuid::new_v4();
    let chunks: Vec<ChunkDocument> = (0..5)
        .map(|i| {
            let metadata = ChunkMetadata::new(
                document_id,
                format!("Doc {}", i),
                i,
                5,
                format!("/test/{}.txt", i),
            );
            ChunkDocument::new(
                Uuid::new_v4(),
                format!("Content {}", i),
                metadata,
            )
        })
        .collect();
    
    let request = BulkInsertRequest {
        chunks: chunks.clone(),
        upsert: false,
        batch_size: Some(2),
    };
    
    assert_eq!(request.chunks.len(), 5);
    assert_eq!(request.batch_size, Some(2));
    assert!(!request.upsert);
    
    // Test bulk response
    let response = BulkInsertResponse {
        inserted_count: 5,
        modified_count: 0,
        failed_count: 0,
        errors: vec![],
        processing_time_ms: 100,
    };
    
    assert_eq!(response.inserted_count, 5);
    assert_eq!(response.failed_count, 0);
    assert!(response.errors.is_empty());
}

#[test]
fn test_operation_metrics_percentiles() {
    let mut metrics = OperationMetrics::default();
    
    // Add durations from 1 to 100
    for i in 1..=100 {
        metrics.recent_durations.push(i);
    }
    
    assert_eq!(metrics.p95_duration_ms(), 95.0);
    assert_eq!(metrics.p99_duration_ms(), 99.0);
    
    // Test with fewer data points
    let mut small_metrics = OperationMetrics::default();
    small_metrics.recent_durations = vec![10, 20, 30];
    
    assert_eq!(small_metrics.p95_duration_ms(), 30.0);
    
    // Test empty metrics
    let empty_metrics = OperationMetrics::default();
    assert_eq!(empty_metrics.p95_duration_ms(), 0.0);
}

#[test]
fn test_reference_types() {
    let chunk_id = Uuid::new_v4();
    
    let sequential_ref = ChunkReference {
        chunk_id,
        reference_type: ReferenceType::Sequential,
        confidence: 1.0,
        context: "Next chunk".to_string(),
    };
    
    assert!(matches!(sequential_ref.reference_type, ReferenceType::Sequential));
    
    let custom_ref = ChunkReference {
        chunk_id,
        reference_type: ReferenceType::Custom("related_topic".to_string()),
        confidence: 0.8,
        context: "Related by topic".to_string(),
    };
    
    assert!(matches!(custom_ref.reference_type, ReferenceType::Custom(_)));
}

#[test]
fn test_security_levels() {
    assert!(matches!(SecurityLevel::default(), SecurityLevel::Internal));
    
    let levels = vec![
        SecurityLevel::Public,
        SecurityLevel::Internal,
        SecurityLevel::Confidential,
        SecurityLevel::Restricted,
        SecurityLevel::TopSecret,
    ];
    
    assert_eq!(levels.len(), 5);
}

#[test]
fn test_quality_metrics() {
    let quality = QualityMetrics {
        avg_boundary_confidence: 0.85,
        coherence_score: 0.92,
        information_density: 0.78,
        overlap_quality: 0.88,
        overall_quality: 86.0,
    };
    
    assert!(quality.avg_boundary_confidence > 0.0);
    assert!(quality.overall_quality > 80.0);
}

#[test]
fn test_processing_stats() {
    let mut stats = ProcessingStats::default();
    stats.processing_time_ms = 1500;
    stats.chunk_count = 25;
    stats.avg_chunk_size = 512;
    stats.embedding_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
    stats.embedding_dimension = 384;
    
    assert_eq!(stats.chunk_count, 25);
    assert_eq!(stats.embedding_dimension, 384);
    assert!(stats.processing_time_ms > 0);
}