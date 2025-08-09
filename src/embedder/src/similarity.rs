//! Similarity calculations for embeddings
//!
//! This module provides efficient implementations of various similarity metrics,
//! with SIMD optimizations where available.

use anyhow::Result;
use crate::EmbedderError;

/// Calculate cosine similarity between two embeddings
///
/// Returns a value between -1.0 and 1.0, where 1.0 indicates identical vectors,
/// 0.0 indicates orthogonal vectors, and -1.0 indicates opposite vectors.
///
/// # Arguments
/// * `emb1` - First embedding vector
/// * `emb2` - Second embedding vector
///
/// # Returns
/// * `Ok(f32)` - The cosine similarity
/// * `Err(EmbedderError)` - If the vectors have different dimensions or are invalid
pub fn cosine_similarity(emb1: &[f32], emb2: &[f32]) -> Result<f32> {
    if emb1.len() != emb2.len() {
        return Err(EmbedderError::DimensionMismatch {
            expected: emb1.len(),
            actual: emb2.len(),
        }.into());
    }
    
    if emb1.is_empty() {
        return Err(EmbedderError::InvalidEmbedding {
            message: "Empty embedding vectors".to_string(),
        }.into());
    }
    
    let dot_product = dot_product_simd(emb1, emb2);
    let norm1 = l2_norm_simd(emb1);
    let norm2 = l2_norm_simd(emb2);
    
    if norm1 == 0.0 || norm2 == 0.0 {
        return Err(EmbedderError::InvalidEmbedding {
            message: "Zero-magnitude embedding vector".to_string(),
        }.into());
    }
    
    let similarity = dot_product / (norm1 * norm2);
    
    // Clamp to valid range to handle floating point precision issues
    Ok(similarity.clamp(-1.0, 1.0))
}

/// Calculate cosine similarities between one query embedding and a batch of embeddings
///
/// This is more efficient than calling `cosine_similarity` in a loop.
///
/// # Arguments
/// * `query_embedding` - The query embedding vector
/// * `embeddings` - Vector of embedding vectors to compare against
///
/// # Returns
/// * `Ok(Vec<f32>)` - Vector of cosine similarities
/// * `Err(EmbedderError)` - If dimensions don't match or embeddings are invalid
pub fn batch_cosine_similarity(query_embedding: &[f32], embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    
    // Pre-calculate query norm
    let query_norm = l2_norm_simd(query_embedding);
    if query_norm == 0.0 {
        return Err(EmbedderError::InvalidEmbedding {
            message: "Zero-magnitude query embedding".to_string(),
        }.into());
    }
    
    let mut similarities = Vec::with_capacity(embeddings.len());
    
    for embedding in embeddings {
        if embedding.len() != query_embedding.len() {
            return Err(EmbedderError::DimensionMismatch {
                expected: query_embedding.len(),
                actual: embedding.len(),
            }.into());
        }
        
        let dot_product = dot_product_simd(query_embedding, embedding);
        let embedding_norm = l2_norm_simd(embedding);
        
        if embedding_norm == 0.0 {
            return Err(EmbedderError::InvalidEmbedding {
                message: "Zero-magnitude embedding in batch".to_string(),
            }.into());
        }
        
        let similarity = dot_product / (query_norm * embedding_norm);
        similarities.push(similarity.clamp(-1.0, 1.0));
    }
    
    Ok(similarities)
}

/// Calculate Euclidean distance between two embeddings
///
/// # Arguments
/// * `emb1` - First embedding vector
/// * `emb2` - Second embedding vector
///
/// # Returns
/// * `Ok(f32)` - The Euclidean distance
/// * `Err(EmbedderError)` - If the vectors have different dimensions
pub fn euclidean_distance(emb1: &[f32], emb2: &[f32]) -> Result<f32> {
    if emb1.len() != emb2.len() {
        return Err(EmbedderError::DimensionMismatch {
            expected: emb1.len(),
            actual: emb2.len(),
        }.into());
    }
    
    let squared_distance = emb1.iter()
        .zip(emb2.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>();
    
    Ok(squared_distance.sqrt())
}

/// Calculate Manhattan (L1) distance between two embeddings
///
/// # Arguments
/// * `emb1` - First embedding vector
/// * `emb2` - Second embedding vector
///
/// # Returns
/// * `Ok(f32)` - The Manhattan distance
/// * `Err(EmbedderError)` - If the vectors have different dimensions
pub fn manhattan_distance(emb1: &[f32], emb2: &[f32]) -> Result<f32> {
    if emb1.len() != emb2.len() {
        return Err(EmbedderError::DimensionMismatch {
            expected: emb1.len(),
            actual: emb2.len(),
        }.into());
    }
    
    let distance = emb1.iter()
        .zip(emb2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    
    Ok(distance)
}

/// Normalize an embedding to unit length (L2 normalization)
///
/// # Arguments
/// * `embedding` - The embedding vector to normalize
///
/// # Returns
/// * `Ok(Vec<f32>)` - The normalized embedding
/// * `Err(EmbedderError)` - If the embedding has zero magnitude
pub fn normalize_l2(embedding: &[f32]) -> Result<Vec<f32>> {
    let norm = l2_norm_simd(embedding);
    
    if norm == 0.0 {
        return Err(EmbedderError::InvalidEmbedding {
            message: "Cannot normalize zero-magnitude vector".to_string(),
        }.into());
    }
    
    Ok(embedding.iter().map(|&x| x / norm).collect())
}

/// Calculate L2 norm (Euclidean norm) of an embedding vector
///
/// # Arguments
/// * `embedding` - The embedding vector
///
/// # Returns
/// * `f32` - The L2 norm
pub fn l2_norm(embedding: &[f32]) -> f32 {
    l2_norm_simd(embedding)
}

/// Calculate dot product of two vectors with SIMD optimization where available
#[inline]
fn dot_product_simd(vec1: &[f32], vec2: &[f32]) -> f32 {
    // Use simple implementation - in production, you'd use SIMD intrinsics
    // or a library like `simdeez` for better performance
    vec1.iter()
        .zip(vec2.iter())
        .map(|(a, b)| a * b)
        .sum()
}

/// Calculate L2 norm with SIMD optimization where available
#[inline]
fn l2_norm_simd(vec: &[f32]) -> f32 {
    // Use simple implementation - in production, you'd use SIMD intrinsics
    let sum_of_squares: f32 = vec.iter()
        .map(|&x| x * x)
        .sum();
    
    sum_of_squares.sqrt()
}

/// Find the top K most similar embeddings to a query embedding
///
/// # Arguments
/// * `query_embedding` - The query embedding vector
/// * `embeddings` - Vector of (embedding, index) pairs to search
/// * `k` - Number of top results to return
///
/// # Returns
/// * `Ok(Vec<(usize, f32)>)` - Vector of (index, similarity) pairs sorted by similarity (descending)
/// * `Err(EmbedderError)` - If dimensions don't match or embeddings are invalid
pub fn find_top_k_similar(
    query_embedding: &[f32],
    embeddings: &[(Vec<f32>, usize)],
    k: usize,
) -> Result<Vec<(usize, f32)>> {
    if embeddings.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    
    let query_norm = l2_norm_simd(query_embedding);
    if query_norm == 0.0 {
        return Err(EmbedderError::InvalidEmbedding {
            message: "Zero-magnitude query embedding".to_string(),
        }.into());
    }
    
    let mut similarities = Vec::with_capacity(embeddings.len());
    
    for (embedding, index) in embeddings {
        if embedding.len() != query_embedding.len() {
            return Err(EmbedderError::DimensionMismatch {
                expected: query_embedding.len(),
                actual: embedding.len(),
            }.into());
        }
        
        let dot_product = dot_product_simd(query_embedding, embedding);
        let embedding_norm = l2_norm_simd(embedding);
        
        if embedding_norm == 0.0 {
            continue; // Skip zero-magnitude embeddings
        }
        
        let similarity = dot_product / (query_norm * embedding_norm);
        similarities.push((*index, similarity.clamp(-1.0, 1.0)));
    }
    
    // Sort by similarity (descending) and take top k
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    similarities.truncate(k);
    
    Ok(similarities)
}

/// Calculate pairwise cosine similarities for a batch of embeddings
///
/// Returns a symmetric similarity matrix where element (i,j) is the similarity
/// between embeddings[i] and embeddings[j].
///
/// # Arguments
/// * `embeddings` - Vector of embedding vectors
///
/// # Returns
/// * `Ok(Vec<Vec<f32>>)` - Similarity matrix
/// * `Err(EmbedderError)` - If embeddings have different dimensions or are invalid
pub fn pairwise_cosine_similarities(embeddings: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }
    
    let n = embeddings.len();
    let mut similarity_matrix = vec![vec![0.0; n]; n];
    
    // Calculate norms once
    let norms: Result<Vec<f32>, _> = embeddings.iter()
        .map(|emb| {
            let norm = l2_norm_simd(emb);
            if norm == 0.0 {
                Err(EmbedderError::InvalidEmbedding {
                    message: "Zero-magnitude embedding in batch".to_string(),
                })
            } else {
                Ok(norm)
            }
        })
        .collect();
    
    let norms = norms?;
    
    // Calculate similarities
    for i in 0..n {
        similarity_matrix[i][i] = 1.0; // Self-similarity is always 1.0
        
        for j in (i + 1)..n {
            if embeddings[i].len() != embeddings[j].len() {
                return Err(EmbedderError::DimensionMismatch {
                    expected: embeddings[i].len(),
                    actual: embeddings[j].len(),
                }.into());
            }
            
            let dot_product = dot_product_simd(&embeddings[i], &embeddings[j]);
            let similarity = (dot_product / (norms[i] * norms[j])).clamp(-1.0, 1.0);
            
            similarity_matrix[i][j] = similarity;
            similarity_matrix[j][i] = similarity; // Symmetric
        }
    }
    
    Ok(similarity_matrix)
}

/// Clustering embeddings using k-means algorithm (simple implementation)
///
/// # Arguments
/// * `embeddings` - Vector of embedding vectors to cluster
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum number of iterations
///
/// # Returns
/// * `Ok((Vec<usize>, Vec<Vec<f32>>))` - (cluster assignments, centroids)
/// * `Err(EmbedderError)` - If clustering fails
pub fn simple_kmeans_clustering(
    embeddings: &[Vec<f32>],
    k: usize,
    max_iterations: usize,
) -> Result<(Vec<usize>, Vec<Vec<f32>>)> {
    if embeddings.is_empty() || k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    
    if k > embeddings.len() {
        return Err(EmbedderError::ConfigError {
            message: format!("k ({}) cannot be greater than number of embeddings ({})", k, embeddings.len()),
        }.into());
    }
    
    let dimension = embeddings[0].len();
    let mut centroids = Vec::with_capacity(k);
    let mut assignments = vec![0; embeddings.len()];
    
    // Initialize centroids randomly (using first k embeddings for simplicity)
    for i in 0..k {
        centroids.push(embeddings[i % embeddings.len()].clone());
    }
    
    for _ in 0..max_iterations {
        let mut new_assignments = vec![0; embeddings.len()];
        let mut changed = false;
        
        // Assign points to nearest centroid
        for (i, embedding) in embeddings.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_distance = f32::INFINITY;
            
            for (j, centroid) in centroids.iter().enumerate() {
                let distance = euclidean_distance(embedding, centroid)?;
                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = j;
                }
            }
            
            new_assignments[i] = best_cluster;
            if new_assignments[i] != assignments[i] {
                changed = true;
            }
        }
        
        assignments = new_assignments;
        
        if !changed {
            break; // Converged
        }
        
        // Update centroids
        let mut new_centroids = vec![vec![0.0; dimension]; k];
        let mut cluster_counts = vec![0; k];
        
        for (i, embedding) in embeddings.iter().enumerate() {
            let cluster = assignments[i];
            cluster_counts[cluster] += 1;
            
            for (j, &value) in embedding.iter().enumerate() {
                new_centroids[cluster][j] += value;
            }
        }
        
        // Average to get centroids
        for (i, centroid) in new_centroids.iter_mut().enumerate() {
            if cluster_counts[i] > 0 {
                for value in centroid.iter_mut() {
                    *value /= cluster_counts[i] as f32;
                }
            }
        }
        
        centroids = new_centroids;
    }
    
    Ok((assignments, centroids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_cosine_similarity() {
        let emb1 = vec![1.0, 0.0, 0.0];
        let emb2 = vec![0.0, 1.0, 0.0];
        let emb3 = vec![1.0, 0.0, 0.0];
        
        let sim1 = cosine_similarity(&emb1, &emb2).unwrap();
        let sim2 = cosine_similarity(&emb1, &emb3).unwrap();
        
        assert_relative_eq!(sim1, 0.0, epsilon = 1e-6);
        assert_relative_eq!(sim2, 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_batch_cosine_similarity() {
        let query = vec![1.0, 0.0, 0.0];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![-1.0, 0.0, 0.0],
        ];
        
        let similarities = batch_cosine_similarity(&query, &embeddings).unwrap();
        
        assert_relative_eq!(similarities[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(similarities[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(similarities[2], -1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_euclidean_distance() {
        let emb1 = vec![0.0, 0.0];
        let emb2 = vec![3.0, 4.0];
        
        let distance = euclidean_distance(&emb1, &emb2).unwrap();
        assert_relative_eq!(distance, 5.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_manhattan_distance() {
        let emb1 = vec![0.0, 0.0];
        let emb2 = vec![3.0, 4.0];
        
        let distance = manhattan_distance(&emb1, &emb2).unwrap();
        assert_relative_eq!(distance, 7.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_normalize_l2() {
        let embedding = vec![3.0, 4.0, 0.0];
        let normalized = normalize_l2(&embedding).unwrap();
        
        assert_relative_eq!(normalized[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(normalized[1], 0.8, epsilon = 1e-6);
        assert_relative_eq!(normalized[2], 0.0, epsilon = 1e-6);
        
        // Check that it's unit length
        let norm = l2_norm(&normalized);
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_find_top_k_similar() {
        let query = vec![1.0, 0.0, 0.0];
        let embeddings = vec![
            (vec![1.0, 0.0, 0.0], 0), // similarity = 1.0
            (vec![0.0, 1.0, 0.0], 1), // similarity = 0.0
            (vec![-1.0, 0.0, 0.0], 2), // similarity = -1.0
            (vec![0.5, 0.5, 0.0], 3), // similarity ≈ 0.707
        ];
        
        let top_k = find_top_k_similar(&query, &embeddings, 2).unwrap();
        
        assert_eq!(top_k.len(), 2);
        assert_eq!(top_k[0].0, 0); // Best match
        assert_eq!(top_k[1].0, 3); // Second best match
        assert!(top_k[0].1 > top_k[1].1); // Similarities in descending order
    }
    
    #[test]
    fn test_dimension_mismatch_error() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![1.0, 0.0, 0.0];
        
        let result = cosine_similarity(&emb1, &emb2);
        assert!(result.is_err());
        
        match result.unwrap_err().downcast::<EmbedderError>().unwrap() {
            EmbedderError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }
    
    #[test]
    fn test_zero_magnitude_error() {
        let emb1 = vec![0.0, 0.0, 0.0];
        let emb2 = vec![1.0, 0.0, 0.0];
        
        let result = cosine_similarity(&emb1, &emb2);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_pairwise_similarities() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ];
        
        let similarities = pairwise_cosine_similarities(&embeddings).unwrap();
        
        assert_eq!(similarities.len(), 3);
        assert_eq!(similarities[0].len(), 3);
        
        // Check diagonal (self-similarity)
        for i in 0..3 {
            assert_relative_eq!(similarities[i][i], 1.0, epsilon = 1e-6);
        }
        
        // Check symmetry
        assert_relative_eq!(similarities[0][1], similarities[1][0], epsilon = 1e-6);
        
        // Check specific values
        assert_relative_eq!(similarities[0][1], 0.0, epsilon = 1e-6); // orthogonal
        assert_relative_eq!(similarities[0][2], 1.0, epsilon = 1e-6); // identical
    }
    
    #[test]
    fn test_simple_kmeans() {
        let embeddings = vec![
            vec![1.0, 1.0],
            vec![1.1, 1.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];
        
        let (assignments, centroids) = simple_kmeans_clustering(&embeddings, 2, 10).unwrap();
        
        assert_eq!(assignments.len(), 4);
        assert_eq!(centroids.len(), 2);
        
        // Check that similar points are in the same cluster
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[2], assignments[3]);
        assert_ne!(assignments[0], assignments[2]);
    }
}