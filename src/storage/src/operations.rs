//! Database operations for MongoDB Vector Storage
//! 
//! This module provides comprehensive CRUD operations with transaction support,
//! bulk operations, and proper error handling for the vector storage system.

use std::time::{Duration, Instant};
use std::collections::HashMap;

use mongodb::{
    bson::{doc, Document, to_document, from_document},
    options::{
        FindOptions, UpdateOptions, DeleteOptions, InsertManyOptions,
        ReadPreference, WriteConcern, ReadConcern, TransactionOptions,
    },
    results::{InsertManyResult, UpdateResult, DeleteResult},
    ClientSession, Collection,
};
use futures::stream::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::{Result, Context, anyhow};
use async_trait::async_trait;
use tracing::{info, warn, error, debug, instrument};

use crate::{VectorStorage, ChunkDocument, MetadataDocument, BulkInsertRequest, BulkInsertResponse, BulkInsertError};
use crate::error::StorageError;

/// Trait defining database operations
#[async_trait]
pub trait DatabaseOperations {
    /// Insert a single chunk
    async fn insert_chunk(&self, chunk: ChunkDocument) -> Result<Uuid>;
    
    /// Insert multiple chunks
    async fn insert_chunks(&self, chunks: Vec<ChunkDocument>) -> Result<BulkInsertResponse>;
    
    /// Update an existing chunk
    async fn update_chunk(&self, chunk_id: Uuid, chunk: ChunkDocument) -> Result<bool>;
    
    /// Delete a chunk by ID
    async fn delete_chunk(&self, chunk_id: Uuid) -> Result<bool>;
    
    /// Delete all chunks for a document
    async fn delete_document_chunks(&self, document_id: Uuid) -> Result<u64>;
    
    /// Get a chunk by ID
    async fn get_chunk(&self, chunk_id: Uuid) -> Result<Option<ChunkDocument>>;
    
    /// Get all chunks for a document
    async fn get_document_chunks(&self, document_id: Uuid) -> Result<Vec<ChunkDocument>>;
    
    /// Count chunks for a document
    async fn count_document_chunks(&self, document_id: Uuid) -> Result<u64>;
    
    /// Check if a chunk exists
    async fn chunk_exists(&self, chunk_id: Uuid) -> Result<bool>;
}

/// Transaction operations trait
#[async_trait]
pub trait TransactionOperations {
    /// Execute operations within a transaction
    async fn with_transaction<F, R>(&self, operations: F) -> Result<R>
    where
        F: for<'a> FnOnce(&'a mut ClientSession) -> futures::future::BoxFuture<'a, Result<R>> + Send,
        R: Send;
    
    /// Bulk insert with transaction support
    async fn transactional_bulk_insert(&self, request: BulkInsertRequest) -> Result<BulkInsertResponse>;
}

/// Batch processing operations
pub struct BatchOperations;

impl BatchOperations {
    /// Default batch size for bulk operations
    pub const DEFAULT_BATCH_SIZE: usize = 100;
    
    /// Maximum batch size to prevent memory issues
    pub const MAX_BATCH_SIZE: usize = 1000;
    
    /// Split large batches into smaller chunks
    pub fn split_into_batches<T>(items: Vec<T>, batch_size: Option<usize>) -> Vec<Vec<T>> {
        let batch_size = batch_size
            .unwrap_or(Self::DEFAULT_BATCH_SIZE)
            .min(Self::MAX_BATCH_SIZE);
        
        items.chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

#[async_trait]
impl DatabaseOperations for VectorStorage {
    #[instrument(skip(self, chunk))]
    async fn insert_chunk(&self, chunk: ChunkDocument) -> Result<Uuid> {
        let start_time = Instant::now();
        
        // Validate chunk before insertion
        chunk.validate()
            .map_err(|e| StorageError::ValidationError(e))?;
        
        let chunk_id = chunk.chunk_id;
        
        // Convert to document and insert
        let result = self.chunk_collection
            .insert_one(&chunk, None)
            .await
            .context("Failed to insert chunk")?;
        
        // Update metrics
        let duration = start_time.elapsed();
        self.metrics.record_operation_duration("insert_chunk", duration);
        self.metrics.increment_operation_count("insert_chunk");
        
        info!("Inserted chunk {} in {:?}", chunk_id, duration);
        Ok(chunk_id)
    }
    
    #[instrument(skip(self, chunks))]
    async fn insert_chunks(&self, chunks: Vec<ChunkDocument>) -> Result<BulkInsertResponse> {
        let start_time = Instant::now();
        let total_chunks = chunks.len();
        
        if chunks.is_empty() {
            return Ok(BulkInsertResponse {
                inserted_count: 0,
                modified_count: 0,
                failed_count: 0,
                errors: Vec::new(),
                processing_time_ms: 0,
            });
        }
        
        // Validate all chunks
        let mut validation_errors = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            if let Err(e) = chunk.validate() {
                validation_errors.push(BulkInsertError {
                    document_index: index,
                    error_message: e,
                    error_code: None,
                });
            }
        }
        
        if !validation_errors.is_empty() {
            return Ok(BulkInsertResponse {
                inserted_count: 0,
                modified_count: 0,
                failed_count: total_chunks as u64,
                errors: validation_errors,
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            });
        }
        
        // Process in batches
        let batches = BatchOperations::split_into_batches(chunks, None);
        let mut total_inserted = 0u64;
        let mut total_errors = Vec::new();
        
        for (batch_index, batch) in batches.into_iter().enumerate() {
            let batch_start = batch_index * BatchOperations::DEFAULT_BATCH_SIZE;
            
            // Insert batch with proper error handling
            let options = InsertManyOptions::builder()
                .ordered(false) // Continue on errors
                .build();
            
            match self.chunk_collection.insert_many(&batch, options).await {
                Ok(result) => {
                    total_inserted += result.inserted_ids.len() as u64;
                    info!("Inserted batch {}: {} documents", batch_index, result.inserted_ids.len());
                }
                Err(e) => {
                    // Handle bulk write errors
                    let failed_count = batch.len();
                    for i in 0..failed_count {
                        total_errors.push(BulkInsertError {
                            document_index: batch_start + i,
                            error_message: e.to_string(),
                            error_code: None,
                        });
                    }
                    warn!("Failed to insert batch {}: {}", batch_index, e);
                }
            }
        }
        
        let duration = start_time.elapsed();
        let failed_count = total_chunks as u64 - total_inserted;
        
        // Update metrics
        self.metrics.record_operation_duration("insert_chunks", duration);
        self.metrics.increment_operation_count("insert_chunks");
        
        info!(
            "Bulk insert completed: {} inserted, {} failed in {:?}",
            total_inserted, failed_count, duration
        );
        
        Ok(BulkInsertResponse {
            inserted_count: total_inserted,
            modified_count: 0,
            failed_count,
            errors: total_errors,
            processing_time_ms: duration.as_millis() as u64,
        })
    }
    
    #[instrument(skip(self, chunk))]
    async fn update_chunk(&self, chunk_id: Uuid, mut chunk: ChunkDocument) -> Result<bool> {
        let start_time = Instant::now();
        
        // Validate chunk
        chunk.validate()
            .map_err(|e| StorageError::ValidationError(e))?;
        
        // Update timestamp and version
        chunk.updated_at = Utc::now();
        chunk.version += 1;
        
        let filter = doc! { "chunk_id": chunk_id.to_string() };
        let update = doc! { 
            "$set": to_document(&chunk)
                .context("Failed to serialize chunk for update")?
        };
        
        let options = UpdateOptions::builder()
            .upsert(false)
            .build();
        
        let result = self.chunk_collection
            .update_one(filter, update, options)
            .await
            .context("Failed to update chunk")?;
        
        let duration = start_time.elapsed();
        let updated = result.modified_count > 0;
        
        // Update metrics
        self.metrics.record_operation_duration("update_chunk", duration);
        self.metrics.increment_operation_count("update_chunk");
        
        info!("Updated chunk {}: {} in {:?}", chunk_id, updated, duration);
        Ok(updated)
    }
    
    #[instrument(skip(self))]
    async fn delete_chunk(&self, chunk_id: Uuid) -> Result<bool> {
        let start_time = Instant::now();
        
        let filter = doc! { "chunk_id": chunk_id.to_string() };
        let result = self.chunk_collection
            .delete_one(filter, None)
            .await
            .context("Failed to delete chunk")?;
        
        let duration = start_time.elapsed();
        let deleted = result.deleted_count > 0;
        
        // Update metrics
        self.metrics.record_operation_duration("delete_chunk", duration);
        self.metrics.increment_operation_count("delete_chunk");
        
        info!("Deleted chunk {}: {} in {:?}", chunk_id, deleted, duration);
        Ok(deleted)
    }
    
    #[instrument(skip(self))]
    async fn delete_document_chunks(&self, document_id: Uuid) -> Result<u64> {
        let start_time = Instant::now();
        
        let filter = doc! { "metadata.document_id": document_id.to_string() };
        let result = self.chunk_collection
            .delete_many(filter, None)
            .await
            .context("Failed to delete document chunks")?;
        
        let duration = start_time.elapsed();
        let deleted_count = result.deleted_count;
        
        // Update metrics
        self.metrics.record_operation_duration("delete_document_chunks", duration);
        self.metrics.increment_operation_count("delete_document_chunks");
        
        info!("Deleted {} chunks for document {} in {:?}", deleted_count, document_id, duration);
        Ok(deleted_count)
    }
    
    #[instrument(skip(self))]
    async fn get_chunk(&self, chunk_id: Uuid) -> Result<Option<ChunkDocument>> {
        let start_time = Instant::now();
        
        let filter = doc! { "chunk_id": chunk_id.to_string() };
        let result = self.chunk_collection
            .find_one(filter, None)
            .await
            .context("Failed to get chunk")?;
        
        let duration = start_time.elapsed();
        
        // Update metrics
        self.metrics.record_operation_duration("get_chunk", duration);
        self.metrics.increment_operation_count("get_chunk");
        
        debug!("Retrieved chunk {}: {} in {:?}", chunk_id, result.is_some(), duration);
        Ok(result)
    }
    
    #[instrument(skip(self))]
    async fn get_document_chunks(&self, document_id: Uuid) -> Result<Vec<ChunkDocument>> {
        let start_time = Instant::now();
        
        let filter = doc! { "metadata.document_id": document_id.to_string() };
        let options = FindOptions::builder()
            .sort(doc! { "metadata.chunk_index": 1 })
            .build();
        
        let cursor = self.chunk_collection
            .find(filter, options)
            .await
            .context("Failed to query document chunks")?;
        
        let chunks: Result<Vec<ChunkDocument>, _> = cursor.try_collect().await;
        let chunks = chunks.context("Failed to collect document chunks")?;
        
        let duration = start_time.elapsed();
        let count = chunks.len();
        
        // Update metrics
        self.metrics.record_operation_duration("get_document_chunks", duration);
        self.metrics.increment_operation_count("get_document_chunks");
        
        info!("Retrieved {} chunks for document {} in {:?}", count, document_id, duration);
        Ok(chunks)
    }
    
    #[instrument(skip(self))]
    async fn count_document_chunks(&self, document_id: Uuid) -> Result<u64> {
        let start_time = Instant::now();
        
        let filter = doc! { "metadata.document_id": document_id.to_string() };
        let count = self.chunk_collection
            .count_documents(filter, None)
            .await
            .context("Failed to count document chunks")?;
        
        let duration = start_time.elapsed();
        
        // Update metrics
        self.metrics.record_operation_duration("count_document_chunks", duration);
        self.metrics.increment_operation_count("count_document_chunks");
        
        debug!("Counted {} chunks for document {} in {:?}", count, document_id, duration);
        Ok(count)
    }
    
    #[instrument(skip(self))]
    async fn chunk_exists(&self, chunk_id: Uuid) -> Result<bool> {
        let start_time = Instant::now();
        
        let filter = doc! { "chunk_id": chunk_id.to_string() };
        let options = FindOptions::builder()
            .projection(doc! { "_id": 1 })
            .limit(1)
            .build();
        
        let exists = self.chunk_collection
            .find_one(filter, options)
            .await
            .context("Failed to check chunk existence")?
            .is_some();
        
        let duration = start_time.elapsed();
        
        // Update metrics
        self.metrics.record_operation_duration("chunk_exists", duration);
        self.metrics.increment_operation_count("chunk_exists");
        
        debug!("Checked chunk {} exists: {} in {:?}", chunk_id, exists, duration);
        Ok(exists)
    }
}

#[async_trait]
impl TransactionOperations for VectorStorage {
    #[instrument(skip(self, operations))]
    async fn with_transaction<F, R>(&self, operations: F) -> Result<R>
    where
        F: for<'a> FnOnce(&'a mut ClientSession) -> futures::future::BoxFuture<'a, Result<R>> + Send,
        R: Send,
    {
        let mut session = self.client
            .start_session(None)
            .await
            .context("Failed to start MongoDB session")?;
        
        let transaction_options = TransactionOptions::builder()
            .read_concern(ReadConcern::majority())
            .write_concern(WriteConcern::majority())
            .read_preference(ReadPreference::primary())
            .max_commit_time(Some(Duration::from_secs(30)))
            .build();
        
        session.start_transaction(transaction_options)
            .await
            .context("Failed to start transaction")?;
        
        let result = match operations(&mut session).await {
            Ok(result) => {
                session.commit_transaction()
                    .await
                    .context("Failed to commit transaction")?;
                Ok(result)
            }
            Err(e) => {
                if let Err(abort_err) = session.abort_transaction().await {
                    error!("Failed to abort transaction: {}", abort_err);
                }
                Err(e)
            }
        };
        
        result
    }
    
    #[instrument(skip(self, request))]
    async fn transactional_bulk_insert(&self, request: BulkInsertRequest) -> Result<BulkInsertResponse> {
        let start_time = Instant::now();
        
        self.with_transaction(|session| {
            let chunks = request.chunks.clone();
            let chunk_collection = self.chunk_collection.clone();
            
            Box::pin(async move {
                // Validate all chunks first
                for chunk in &chunks {
                    chunk.validate()
                        .map_err(|e| StorageError::ValidationError(e))?;
                }
                
                // Process in batches within transaction
                let batches = BatchOperations::split_into_batches(chunks, request.batch_size);
                let mut total_inserted = 0u64;
                let mut errors = Vec::new();
                
                for (batch_index, batch) in batches.into_iter().enumerate() {
                    let options = InsertManyOptions::builder()
                        .ordered(false)
                        .build();
                    
                    match chunk_collection.insert_many_with_session(&batch, options, session).await {
                        Ok(result) => {
                            total_inserted += result.inserted_ids.len() as u64;
                        }
                        Err(e) => {
                            let batch_start = batch_index * BatchOperations::DEFAULT_BATCH_SIZE;
                            for i in 0..batch.len() {
                                errors.push(BulkInsertError {
                                    document_index: batch_start + i,
                                    error_message: e.to_string(),
                                    error_code: None,
                                });
                            }
                        }
                    }
                }
                
                Ok(BulkInsertResponse {
                    inserted_count: total_inserted,
                    modified_count: 0,
                    failed_count: errors.len() as u64,
                    errors,
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                })
            })
        }).await
    }
}

/// Utility functions for database operations
pub struct OperationUtils;

impl OperationUtils {
    /// Build a compound filter for chunk queries
    pub fn build_chunk_filter(
        document_id: Option<Uuid>,
        tags: Option<&[String]>,
        date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    ) -> Document {
        let mut filter = doc! {};
        
        if let Some(doc_id) = document_id {
            filter.insert("metadata.document_id", doc_id.to_string());
        }
        
        if let Some(tags) = tags {
            if !tags.is_empty() {
                filter.insert("metadata.tags", doc! { "$in": tags });
            }
        }
        
        if let Some((start, end)) = date_range {
            filter.insert("created_at", doc! {
                "$gte": start,
                "$lte": end
            });
        }
        
        filter
    }
    
    /// Build update document for partial chunk updates
    pub fn build_update_document(updates: HashMap<String, mongodb::bson::Bson>) -> Document {
        let mut update_doc = doc! {
            "$set": {
                "updated_at": Utc::now()
            }
        };
        
        if let Ok(set_doc) = update_doc.get_document_mut("$set") {
            for (key, value) in updates {
                set_doc.insert(key, value);
            }
        }
        
        update_doc
    }
    
    /// Extract error information from MongoDB bulk write errors
    pub fn extract_bulk_write_errors(error: &mongodb::error::Error) -> Vec<BulkInsertError> {
        // This would extract detailed error information from MongoDB bulk write errors
        // For now, return a generic error
        vec![BulkInsertError {
            document_index: 0,
            error_message: error.to_string(),
            error_code: None,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChunkMetadata, StorageConfig};
    use tokio_test;
    
    #[tokio::test]
    async fn test_batch_operations() {
        let items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let batches = BatchOperations::split_into_batches(items, Some(3));
        
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0], vec![1, 2, 3]);
        assert_eq!(batches[1], vec![4, 5, 6]);
        assert_eq!(batches[2], vec![7, 8, 9]);
        assert_eq!(batches[3], vec![10]);
    }
    
    #[test]
    fn test_filter_building() {
        let document_id = Uuid::new_v4();
        let tags = vec!["tag1".to_string(), "tag2".to_string()];
        let start_date = Utc::now() - chrono::Duration::days(7);
        let end_date = Utc::now();
        
        let filter = OperationUtils::build_chunk_filter(
            Some(document_id),
            Some(&tags),
            Some((start_date, end_date)),
        );
        
        assert!(filter.contains_key("metadata.document_id"));
        assert!(filter.contains_key("metadata.tags"));
        assert!(filter.contains_key("created_at"));
    }
    
    #[tokio::test]
    async fn test_database_operations() {
        // This test would require a running MongoDB instance
        // In practice, we'd use testcontainers for integration tests
        
        let config = StorageConfig::default();
        if let Ok(storage) = VectorStorage::new(config).await {
            let metadata = ChunkMetadata::new(
                Uuid::new_v4(),
                "Test Document".to_string(),
                0,
                1,
                "/test/path".to_string(),
            );
            
            let chunk = ChunkDocument::new(
                Uuid::new_v4(),
                "Test content".to_string(),
                metadata,
            );
            
            let chunk_id = chunk.chunk_id;
            
            // Test insert
            if let Ok(inserted_id) = storage.insert_chunk(chunk).await {
                assert_eq!(inserted_id, chunk_id);
                
                // Test retrieval
                if let Ok(Some(retrieved)) = storage.get_chunk(chunk_id).await {
                    assert_eq!(retrieved.content, "Test content");
                    
                    // Test deletion
                    let deleted = storage.delete_chunk(chunk_id).await.unwrap();
                    assert!(deleted);
                }
            }
        }
    }
}