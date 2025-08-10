//! Vector and hybrid search functionality for MongoDB Vector Storage
//! 
//! This module provides comprehensive search capabilities including:
//! - Vector similarity search using KNN algorithms
//! - Hybrid search combining vector and text search
//! - Advanced filtering and pagination
//! - Query optimization and caching

use std::time::{Duration, Instant};
use std::collections::HashMap;

use mongodb::{
    bson::{doc, Document, Bson},
    options::{AggregateOptions, FindOptions},
    Cursor,
};
use futures::stream::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::{Result, Context, anyhow};
use async_trait::async_trait;
use tracing::{info, warn, error, debug, instrument};
use ndarray::{Array1, Array2};

use crate::{VectorStorage, ChunkDocument, ChunkMetadata};
use crate::error::StorageError;
use crate::operations::DatabaseOperations;

/// Search query parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Query embedding for vector search
    pub query_embedding: Option<Vec<f64>>,
    
    /// Text query for keyword search
    pub text_query: Option<String>,
    
    /// Search type
    pub search_type: SearchType,
    
    /// Number of results to return
    pub limit: usize,
    
    /// Offset for pagination
    pub offset: usize,
    
    /// Minimum similarity score threshold
    pub min_score: Option<f32>,
    
    /// Filters to apply
    pub filters: SearchFilters,
    
    /// Result sorting preferences
    pub sort: SortOptions,
}

/// Types of search operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchType {
    /// Vector similarity search only
    Vector,
    
    /// Text search only
    Text,
    
    /// Hybrid search combining vector and text
    Hybrid,
    
    /// Semantic search with query expansion
    Semantic,
}

/// Search filters for refining results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilters {
    /// Filter by document IDs
    pub document_ids: Option<Vec<Uuid>>,
    
    /// Filter by tags
    pub tags: Option<Vec<String>>,
    
    /// Filter by date range
    pub date_range: Option<DateRange>,
    
    /// Filter by language
    pub language: Option<String>,
    
    /// Filter by content length
    pub content_length_range: Option<(usize, usize)>,
    
    /// Custom metadata filters
    pub custom_filters: HashMap<String, FilterValue>,
}

/// Date range filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Filter value types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Array(Vec<String>),
    Range { min: f64, max: f64 },
}

/// Sorting options for search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortOptions {
    /// Primary sort field
    pub field: SortField,
    
    /// Sort direction
    pub direction: SortDirection,
    
    /// Secondary sort criteria
    pub secondary: Option<Box<SortOptions>>,
}

/// Available sort fields
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortField {
    Relevance,
    CreatedAt,
    UpdatedAt,
    ChunkIndex,
    ContentLength,
}

/// Sort directions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortDirection {
    Ascending,
    Descending,
}

/// Search result containing chunk and relevance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matching chunk
    pub chunk: ChunkDocument,
    
    /// Relevance score (0.0 to 1.0)
    pub score: f32,
    
    /// Vector similarity score if applicable
    pub vector_score: Option<f32>,
    
    /// Text relevance score if applicable
    pub text_score: Option<f32>,
    
    /// Explanation of why this result was selected
    pub explanation: Option<String>,
    
    /// Highlighted text snippets
    pub highlights: Vec<TextHighlight>,
}

/// Text highlighting information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextHighlight {
    /// Field that was highlighted
    pub field: String,
    
    /// Highlighted text with markers
    pub highlighted_text: String,
    
    /// Character positions of highlights
    pub positions: Vec<(usize, usize)>,
}

/// Search response containing results and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Search results
    pub results: Vec<SearchResult>,
    
    /// Total number of matches (without pagination)
    pub total_count: u64,
    
    /// Search execution time in milliseconds
    pub search_time_ms: u64,
    
    /// Query that was executed
    pub query: SearchQuery,
    
    /// Search performance metrics
    pub metrics: SearchMetrics,
}

/// Performance metrics for search operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchMetrics {
    /// Vector search time
    pub vector_search_ms: Option<u64>,
    
    /// Text search time
    pub text_search_ms: Option<u64>,
    
    /// Filtering time
    pub filter_time_ms: Option<u64>,
    
    /// Sorting time
    pub sort_time_ms: Option<u64>,
    
    /// Number of documents scanned
    pub documents_scanned: u64,
    
    /// Number of documents that passed filters
    pub documents_filtered: u64,
}

/// Trait defining search operations
#[async_trait]
pub trait SearchOperations {
    /// Perform vector similarity search
    async fn vector_search(
        &self,
        query_embedding: &[f64],
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>>;
    
    /// Perform text search
    async fn text_search(
        &self,
        query: &str,
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>>;
    
    /// Perform hybrid search
    async fn hybrid_search(&self, query: SearchQuery) -> Result<SearchResponse>;
    
    /// Find similar chunks to a given chunk
    async fn find_similar(&self, chunk_id: Uuid, limit: usize) -> Result<Vec<SearchResult>>;
    
    /// Get recommendations based on user interaction
    async fn get_recommendations(
        &self,
        viewed_chunks: &[Uuid],
        limit: usize,
    ) -> Result<Vec<SearchResult>>;
}

/// Vector similarity calculations
pub struct VectorSimilarity;

impl VectorSimilarity {
    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(anyhow!("Vector dimensions must match: {} vs {}", a.len(), b.len()));
        }
        
        let a_array = Array1::from_vec(a.to_vec());
        let b_array = Array1::from_vec(b.to_vec());
        
        let dot_product = a_array.dot(&b_array);
        let norm_a = a_array.dot(&a_array).sqrt();
        let norm_b = b_array.dot(&b_array).sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }
        
        Ok((dot_product / (norm_a * norm_b)) as f32)
    }
    
    /// Calculate Euclidean distance between vectors
    pub fn euclidean_distance(a: &[f64], b: &[f64]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(anyhow!("Vector dimensions must match: {} vs {}", a.len(), b.len()));
        }
        
        let sum_sq: f64 = a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();
        
        Ok(sum_sq.sqrt() as f32)
    }
    
    /// Convert distance to similarity score (0-1)
    pub fn distance_to_similarity(distance: f32, max_distance: f32) -> f32 {
        if max_distance <= 0.0 {
            return 0.0;
        }
        (max_distance - distance.min(max_distance)) / max_distance
    }
}

#[async_trait]
impl SearchOperations for VectorStorage {
    #[instrument(skip(self, query_embedding))]
    async fn vector_search(
        &self,
        query_embedding: &[f64],
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        let start_time = Instant::now();
        
        if query_embedding.is_empty() {
            return Err(StorageError::InvalidQuery("Query embedding cannot be empty".to_string()).into());
        }
        
        // Build aggregation pipeline for vector search
        let mut pipeline = vec![];
        
        // Add filter stage if provided
        if let Some(filters) = &filters {
            if let Some(filter_doc) = self.build_filter_document(filters)? {
                pipeline.push(doc! { "$match": filter_doc });
            }
        }
        
        // Add vector search stage (using MongoDB Vector Search)
        // Note: This requires MongoDB Atlas Vector Search or MongoDB 7.0+ with vector search
        pipeline.push(doc! {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": (limit * 10) as i32, // Oversample for better results
                "limit": limit as i32,
                "index": &self.vector_index_name
            }
        });
        
        // Add metadata score
        pipeline.push(doc! {
            "$addFields": {
                "score": { "$meta": "vectorSearchScore" }
            }
        });
        
        // Execute aggregation
        let options = AggregateOptions::builder()
            .allow_disk_use(Some(true))
            .build();
        
        let mut cursor = self.chunk_collection
            .aggregate(pipeline, options)
            .await
            .context("Failed to execute vector search")?;
        
        let mut results = Vec::new();
        
        while let Some(doc) = cursor.next().await {
            let doc = doc.context("Failed to read search result")?;
            
            // Extract score
            let score = doc.get_f64("score").unwrap_or(0.0) as f32;
            
            // Convert document to ChunkDocument
            if let Ok(chunk) = mongodb::bson::from_document::<ChunkDocument>(doc) {
                results.push(SearchResult {
                    chunk,
                    score,
                    vector_score: Some(score),
                    text_score: None,
                    explanation: Some("Vector similarity match".to_string()),
                    highlights: Vec::new(),
                });
            }
        }
        
        let duration = start_time.elapsed();
        
        // Update metrics
        self.metrics.record_operation_duration("vector_search", duration);
        self.metrics.increment_operation_count("vector_search");
        
        info!("Vector search completed: {} results in {:?}", results.len(), duration);
        Ok(results)
    }
    
    #[instrument(skip(self, query))]
    async fn text_search(
        &self,
        query: &str,
        limit: usize,
        filters: Option<SearchFilters>,
    ) -> Result<Vec<SearchResult>> {
        let start_time = Instant::now();
        
        if query.trim().is_empty() {
            return Err(StorageError::InvalidQuery("Query cannot be empty".to_string()).into());
        }
        
        // Build aggregation pipeline for text search
        let mut pipeline = vec![];
        
        // Text search stage
        pipeline.push(doc! {
            "$match": {
                "$text": {
                    "$search": query,
                    "$caseSensitive": false,
                    "$diacriticSensitive": false
                }
            }
        });
        
        // Add filter stage if provided
        if let Some(filters) = &filters {
            if let Some(filter_doc) = self.build_filter_document(filters)? {
                pipeline.push(doc! { "$match": filter_doc });
            }
        }
        
        // Add text score
        pipeline.push(doc! {
            "$addFields": {
                "textScore": { "$meta": "textScore" }
            }
        });
        
        // Sort by text score
        pipeline.push(doc! {
            "$sort": { "textScore": { "$meta": "textScore" } }
        });
        
        // Limit results
        pipeline.push(doc! { "$limit": limit as i32 });
        
        // Execute aggregation
        let options = AggregateOptions::builder()
            .allow_disk_use(Some(true))
            .build();
        
        let mut cursor = self.chunk_collection
            .aggregate(pipeline, options)
            .await
            .context("Failed to execute text search")?;
        
        let mut results = Vec::new();
        
        while let Some(doc) = cursor.next().await {
            let doc = doc.context("Failed to read search result")?;
            
            // Extract text score
            let text_score = doc.get_f64("textScore").unwrap_or(0.0) as f32;
            
            // Convert document to ChunkDocument
            if let Ok(chunk) = mongodb::bson::from_document::<ChunkDocument>(doc) {
                // Generate highlights
                let highlights = self.generate_text_highlights(&chunk.content, query);
                
                results.push(SearchResult {
                    chunk,
                    score: text_score,
                    vector_score: None,
                    text_score: Some(text_score),
                    explanation: Some("Text search match".to_string()),
                    highlights,
                });
            }
        }
        
        let duration = start_time.elapsed();
        
        // Update metrics
        self.metrics.record_operation_duration("text_search", duration);
        self.metrics.increment_operation_count("text_search");
        
        info!("Text search completed: {} results in {:?}", results.len(), duration);
        Ok(results)
    }
    
    #[instrument(skip(self, query))]
    async fn hybrid_search(&self, query: SearchQuery) -> Result<SearchResponse> {
        let start_time = Instant::now();
        let mut metrics = SearchMetrics::default();
        
        let mut all_results = Vec::new();
        
        // Perform vector search if embedding provided
        if let Some(ref embedding) = query.query_embedding {
            let vector_start = Instant::now();
            
            let vector_results = self.vector_search(
                embedding,
                query.limit * 2, // Get more results for fusion
                Some(query.filters.clone()),
            ).await?;
            
            metrics.vector_search_ms = Some(vector_start.elapsed().as_millis() as u64);
            all_results.extend(vector_results);
        }
        
        // Perform text search if text query provided
        if let Some(ref text) = query.text_query {
            let text_start = Instant::now();
            
            let text_results = self.text_search(
                text,
                query.limit * 2, // Get more results for fusion
                Some(query.filters.clone()),
            ).await?;
            
            metrics.text_search_ms = Some(text_start.elapsed().as_millis() as u64);
            all_results.extend(text_results);
        }
        
        // If hybrid search, combine and re-rank results
        if query.search_type == SearchType::Hybrid {
            all_results = self.combine_and_rerank_results(all_results, &query)?;
        }
        
        // Apply post-processing filters
        let filter_start = Instant::now();
        all_results = self.apply_post_filters(all_results, &query)?;
        metrics.filter_time_ms = Some(filter_start.elapsed().as_millis() as u64);
        
        // Sort results
        let sort_start = Instant::now();
        self.sort_results(&mut all_results, &query.sort)?;
        metrics.sort_time_ms = Some(sort_start.elapsed().as_millis() as u64);
        
        // Apply pagination
        let total_count = all_results.len() as u64;
        let end_index = (query.offset + query.limit).min(all_results.len());
        let paginated_results = if query.offset < all_results.len() {
            all_results[query.offset..end_index].to_vec()
        } else {
            Vec::new()
        };
        
        metrics.documents_scanned = total_count;
        metrics.documents_filtered = paginated_results.len() as u64;
        
        let search_time_ms = start_time.elapsed().as_millis() as u64;
        
        // Update metrics
        self.metrics.record_operation_duration("hybrid_search", start_time.elapsed());
        self.metrics.increment_operation_count("hybrid_search");
        
        info!(
            "Hybrid search completed: {} results (total: {}) in {:?}",
            paginated_results.len(),
            total_count,
            start_time.elapsed()
        );
        
        Ok(SearchResponse {
            results: paginated_results,
            total_count,
            search_time_ms,
            query,
            metrics,
        })
    }
    
    #[instrument(skip(self))]
    async fn find_similar(&self, chunk_id: Uuid, limit: usize) -> Result<Vec<SearchResult>> {
        let start_time = Instant::now();
        
        // Get the source chunk with its embedding
        let source_chunk = self.get_chunk(chunk_id).await?
            .ok_or_else(|| StorageError::ChunkNotFound(chunk_id))?;
        
        let embedding = source_chunk.embedding
            .ok_or_else(|| StorageError::EmbeddingNotFound(chunk_id))?;
        
        // Perform vector search excluding the source chunk
        let filters = SearchFilters::default();
        // Add filter to exclude the source chunk (would need custom implementation)
        
        let results = self.vector_search(&embedding, limit + 1, Some(filters)).await?;
        
        // Remove the source chunk from results and limit
        let filtered_results: Vec<SearchResult> = results
            .into_iter()
            .filter(|result| result.chunk.chunk_id != chunk_id)
            .take(limit)
            .collect();
        
        let duration = start_time.elapsed();
        
        // Update metrics
        self.metrics.record_operation_duration("find_similar", duration);
        self.metrics.increment_operation_count("find_similar");
        
        info!("Found {} similar chunks to {} in {:?}", filtered_results.len(), chunk_id, duration);
        Ok(filtered_results)
    }
    
    #[instrument(skip(self, viewed_chunks))]
    async fn get_recommendations(
        &self,
        viewed_chunks: &[Uuid],
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let start_time = Instant::now();
        
        if viewed_chunks.is_empty() {
            return Ok(Vec::new());
        }
        
        // Get embeddings for viewed chunks
        let mut viewed_embeddings = Vec::new();
        for &chunk_id in viewed_chunks {
            if let Some(chunk) = self.get_chunk(chunk_id).await? {
                if let Some(embedding) = chunk.embedding {
                    viewed_embeddings.push(embedding);
                }
            }
        }
        
        if viewed_embeddings.is_empty() {
            return Ok(Vec::new());
        }
        
        // Calculate centroid of viewed embeddings
        let centroid = self.calculate_embedding_centroid(&viewed_embeddings)?;
        
        // Find similar chunks to the centroid
        let results = self.vector_search(&centroid, limit * 2, None).await?;
        
        // Filter out already viewed chunks
        let recommendations: Vec<SearchResult> = results
            .into_iter()
            .filter(|result| !viewed_chunks.contains(&result.chunk.chunk_id))
            .take(limit)
            .collect();
        
        let duration = start_time.elapsed();
        
        // Update metrics
        self.metrics.record_operation_duration("get_recommendations", duration);
        self.metrics.increment_operation_count("get_recommendations");
        
        info!("Generated {} recommendations in {:?}", recommendations.len(), duration);
        Ok(recommendations)
    }
}

impl VectorStorage {
    /// Build MongoDB filter document from search filters
    fn build_filter_document(&self, filters: &SearchFilters) -> Result<Option<Document>> {
        let mut filter_doc = doc! {};
        
        if let Some(ref doc_ids) = filters.document_ids {
            let doc_id_strings: Vec<String> = doc_ids.iter().map(|id| id.to_string()).collect();
            filter_doc.insert("metadata.document_id", doc! { "$in": doc_id_strings });
        }
        
        if let Some(ref tags) = filters.tags {
            filter_doc.insert("metadata.tags", doc! { "$in": tags });
        }
        
        if let Some(ref date_range) = filters.date_range {
            filter_doc.insert("created_at", doc! {
                "$gte": date_range.start,
                "$lte": date_range.end
            });
        }
        
        if let Some(ref language) = filters.language {
            filter_doc.insert("metadata.language", language);
        }
        
        if let Some((min_len, max_len)) = filters.content_length_range {
            filter_doc.insert("metadata.chunk_size", doc! {
                "$gte": min_len as i32,
                "$lte": max_len as i32
            });
        }
        
        // Handle custom filters
        for (key, value) in &filters.custom_filters {
            let field_key = format!("metadata.custom_fields.{}", key);
            match value {
                FilterValue::String(s) => filter_doc.insert(field_key, s),
                FilterValue::Number(n) => filter_doc.insert(field_key, *n),
                FilterValue::Boolean(b) => filter_doc.insert(field_key, *b),
                FilterValue::Array(arr) => filter_doc.insert(field_key, doc! { "$in": arr }),
                FilterValue::Range { min, max } => {
                    filter_doc.insert(field_key, doc! { "$gte": *min, "$lte": *max })
                }
            };
        }
        
        if filter_doc.is_empty() {
            Ok(None)
        } else {
            Ok(Some(filter_doc))
        }
    }
    
    /// Combine and re-rank results from vector and text search
    fn combine_and_rerank_results(
        &self,
        results: Vec<SearchResult>,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>> {
        // Remove duplicates by chunk ID, keeping the best score
        let mut deduped = HashMap::new();
        
        for result in results {
            let chunk_id = result.chunk.chunk_id;
            
            match deduped.get(&chunk_id) {
                Some(existing_result) => {
                    // Combine scores for hybrid ranking
                    let combined_score = self.combine_scores(&result, existing_result, query)?;
                    if combined_score > existing_result.score {
                        let mut new_result = result;
                        new_result.score = combined_score;
                        new_result.explanation = Some("Hybrid search match".to_string());
                        deduped.insert(chunk_id, new_result);
                    }
                }
                None => {
                    deduped.insert(chunk_id, result);
                }
            }
        }
        
        let mut combined_results: Vec<SearchResult> = deduped.into_values().collect();
        combined_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(combined_results)
    }
    
    /// Combine vector and text scores for hybrid ranking
    fn combine_scores(
        &self,
        result1: &SearchResult,
        result2: &SearchResult,
        _query: &SearchQuery,
    ) -> Result<f32> {
        // Weighted combination of vector and text scores
        let vector_weight = 0.6;
        let text_weight = 0.4;
        
        let vector_score = result1.vector_score
            .or(result2.vector_score)
            .unwrap_or(0.0);
        
        let text_score = result1.text_score
            .or(result2.text_score)
            .unwrap_or(0.0);
        
        let combined = (vector_score * vector_weight) + (text_score * text_weight);
        Ok(combined)
    }
    
    /// Apply post-processing filters to results
    fn apply_post_filters(
        &self,
        mut results: Vec<SearchResult>,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>> {
        // Apply minimum score threshold
        if let Some(min_score) = query.min_score {
            results.retain(|result| result.score >= min_score);
        }
        
        Ok(results)
    }
    
    /// Sort results according to sort options
    fn sort_results(&self, results: &mut [SearchResult], sort: &SortOptions) -> Result<()> {
        results.sort_by(|a, b| {
            let ordering = match sort.field {
                SortField::Relevance => b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal),
                SortField::CreatedAt => match sort.direction {
                    SortDirection::Ascending => a.chunk.created_at.cmp(&b.chunk.created_at),
                    SortDirection::Descending => b.chunk.created_at.cmp(&a.chunk.created_at),
                },
                SortField::UpdatedAt => match sort.direction {
                    SortDirection::Ascending => a.chunk.updated_at.cmp(&b.chunk.updated_at),
                    SortDirection::Descending => b.chunk.updated_at.cmp(&a.chunk.updated_at),
                },
                SortField::ChunkIndex => match sort.direction {
                    SortDirection::Ascending => a.chunk.metadata.chunk_index.cmp(&b.chunk.metadata.chunk_index),
                    SortDirection::Descending => b.chunk.metadata.chunk_index.cmp(&a.chunk.metadata.chunk_index),
                },
                SortField::ContentLength => match sort.direction {
                    SortDirection::Ascending => a.chunk.content.len().cmp(&b.chunk.content.len()),
                    SortDirection::Descending => b.chunk.content.len().cmp(&a.chunk.content.len()),
                },
            };
            
            // Apply secondary sort if primary sort results in equality
            if ordering == std::cmp::Ordering::Equal {
                if let Some(ref secondary) = sort.secondary {
                    return self.sort_results_single(a, b, secondary).unwrap_or(std::cmp::Ordering::Equal);
                }
            }
            
            ordering
        });
        
        Ok(())
    }
    
    /// Apply single sort criterion
    fn sort_results_single(
        &self,
        a: &SearchResult,
        b: &SearchResult,
        sort: &SortOptions,
    ) -> Result<std::cmp::Ordering> {
        let ordering = match sort.field {
            SortField::Relevance => b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal),
            SortField::CreatedAt => match sort.direction {
                SortDirection::Ascending => a.chunk.created_at.cmp(&b.chunk.created_at),
                SortDirection::Descending => b.chunk.created_at.cmp(&a.chunk.created_at),
            },
            SortField::UpdatedAt => match sort.direction {
                SortDirection::Ascending => a.chunk.updated_at.cmp(&b.chunk.updated_at),
                SortDirection::Descending => b.chunk.updated_at.cmp(&a.chunk.updated_at),
            },
            SortField::ChunkIndex => match sort.direction {
                SortDirection::Ascending => a.chunk.metadata.chunk_index.cmp(&b.chunk.metadata.chunk_index),
                SortDirection::Descending => b.chunk.metadata.chunk_index.cmp(&a.chunk.metadata.chunk_index),
            },
            SortField::ContentLength => match sort.direction {
                SortDirection::Ascending => a.chunk.content.len().cmp(&b.chunk.content.len()),
                SortDirection::Descending => b.chunk.content.len().cmp(&a.chunk.content.len()),
            },
        };
        
        Ok(ordering)
    }
    
    /// Generate text highlights for search results
    fn generate_text_highlights(&self, content: &str, query: &str) -> Vec<TextHighlight> {
        let mut highlights = Vec::new();
        
        // Simple highlighting implementation - in practice, use a proper text highlighter
        let query_terms: Vec<&str> = query.split_whitespace().collect();
        let content_lower = content.to_lowercase();
        
        for term in query_terms {
            let term_lower = term.to_lowercase();
            let mut start = 0;
            let mut positions = Vec::new();
            
            while let Some(pos) = content_lower[start..].find(&term_lower) {
                let actual_pos = start + pos;
                positions.push((actual_pos, actual_pos + term.len()));
                start = actual_pos + 1;
            }
            
            if !positions.is_empty() {
                // Create highlighted text (simplified)
                let highlighted = content.replace(term, &format!("<mark>{}</mark>", term));
                
                highlights.push(TextHighlight {
                    field: "content".to_string(),
                    highlighted_text: highlighted,
                    positions,
                });
            }
        }
        
        highlights
    }
    
    /// Calculate centroid of multiple embeddings
    fn calculate_embedding_centroid(&self, embeddings: &[Vec<f64>]) -> Result<Vec<f64>> {
        if embeddings.is_empty() {
            return Err(anyhow!("Cannot calculate centroid of empty embeddings"));
        }
        
        let dimension = embeddings[0].len();
        if dimension == 0 {
            return Err(anyhow!("Cannot calculate centroid of zero-dimensional embeddings"));
        }
        
        // Verify all embeddings have the same dimension
        for embedding in embeddings {
            if embedding.len() != dimension {
                return Err(anyhow!("All embeddings must have the same dimension"));
            }
        }
        
        // Calculate centroid
        let mut centroid = vec![0.0; dimension];
        for embedding in embeddings {
            for (i, &value) in embedding.iter().enumerate() {
                centroid[i] += value;
            }
        }
        
        // Average the values
        let count = embeddings.len() as f64;
        for value in &mut centroid {
            *value /= count;
        }
        
        Ok(centroid)
    }
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            query_embedding: None,
            text_query: None,
            search_type: SearchType::Hybrid,
            limit: 10,
            offset: 0,
            min_score: None,
            filters: SearchFilters::default(),
            sort: SortOptions {
                field: SortField::Relevance,
                direction: SortDirection::Descending,
                secondary: None,
            },
        }
    }
}

impl Default for SortOptions {
    fn default() -> Self {
        Self {
            field: SortField::Relevance,
            direction: SortDirection::Descending,
            secondary: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_similarity_calculations() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        
        let cosine_sim = VectorSimilarity::cosine_similarity(&vec1, &vec2).unwrap();
        assert!(cosine_sim > 0.0 && cosine_sim <= 1.0);
        
        let euclidean_dist = VectorSimilarity::euclidean_distance(&vec1, &vec2).unwrap();
        assert!(euclidean_dist > 0.0);
        
        let similarity = VectorSimilarity::distance_to_similarity(euclidean_dist, 10.0);
        assert!(similarity >= 0.0 && similarity <= 1.0);
    }
    
    #[test]
    fn test_search_query_builder() {
        let query = SearchQuery {
            text_query: Some("test query".to_string()),
            limit: 20,
            filters: SearchFilters {
                tags: Some(vec!["tag1".to_string(), "tag2".to_string()]),
                ..Default::default()
            },
            ..Default::default()
        };
        
        assert_eq!(query.limit, 20);
        assert!(query.text_query.is_some());
        assert!(query.filters.tags.is_some());
    }
    
    #[test]
    fn test_centroid_calculation() {
        let storage = create_test_storage(); // Would need mock implementation
        
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        if let Ok(centroid) = storage.calculate_embedding_centroid(&embeddings) {
            assert_eq!(centroid, vec![4.0, 5.0, 6.0]);
        }
    }
    
    // Helper function for tests (would need proper implementation)
    fn create_test_storage() -> VectorStorage {
        // This would create a mock VectorStorage for testing
        // In practice, you'd use a test configuration or mock
        unimplemented!("Test storage creation not implemented")
    }
}