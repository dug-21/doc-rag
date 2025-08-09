use crate::{chunk::ChunkReference, chunk::ReferenceType};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ReferenceTracker {
    config: ReferenceConfig,
}

#[derive(Debug, Clone)]
pub struct ReferenceConfig {
    pub min_confidence: f32,
    pub max_references_per_chunk: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedReference {
    pub base: ChunkReference,
    pub metadata: ReferenceMetadata,
    pub quality: ReferenceQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceMetadata {
    pub source_position: (usize, usize),
    pub surrounding_context: String,
    pub resolution_status: ResolutionStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceQuality {
    pub overall_score: f32,
    pub detection_confidence: f32,
    pub resolution_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionStatus {
    Resolved,
    Unresolved,
    Ambiguous,
    Invalid,
}

#[derive(Debug, Clone)]
pub struct ReferenceGraph {
    nodes: HashMap<String, crate::Chunk>,
    edges: HashMap<String, Vec<(String, ChunkReference)>>,
}

#[derive(Debug, Clone)]
pub struct ReferenceValidationResult {
    pub is_valid: bool,
    pub confidence: f64,
    pub issues: Vec<String>,
}

impl Default for ReferenceConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            max_references_per_chunk: 20,
        }
    }
}

impl ReferenceTracker {
    pub fn new() -> Self {
        Self {
            config: ReferenceConfig::default(),
        }
    }

    pub fn extract_references(&self, content: &str) -> Vec<ChunkReference> {
        let mut references = Vec::new();
        
        let section_regex = Regex::new(r"(?i)section\s+(\d+)").unwrap();
        for captures in section_regex.captures_iter(content) {
            if let Some(section_id) = captures.get(1) {
                references.push(ChunkReference {
                    reference_type: ReferenceType::SectionReference,
                    target_id: format!("section-{}", section_id.as_str()),
                    context: Some(captures.get(0).unwrap().as_str().to_string()),
                    confidence: 0.7,
                });
            }
        }
        
        let citation_regex = Regex::new(r"\[(\d+)\]").unwrap();
        for captures in citation_regex.captures_iter(content) {
            if let Some(num) = captures.get(1) {
                references.push(ChunkReference {
                    reference_type: ReferenceType::Citation,
                    target_id: format!("citation-{}", num.as_str()),
                    context: Some(captures.get(0).unwrap().as_str().to_string()),
                    confidence: 0.8,
                });
            }
        }

        references.retain(|r| r.confidence >= self.config.min_confidence as f64);
        references.truncate(self.config.max_references_per_chunk);
        references
    }

    pub fn extract_extended_references(&self, content: &str, _chunk_id: Uuid) -> Vec<ExtendedReference> {
        let base_references = self.extract_references(content);
        base_references.into_iter().map(|base_ref| {
            ExtendedReference {
                metadata: ReferenceMetadata {
                    source_position: (0, content.len()),
                    surrounding_context: content[..std::cmp::min(100, content.len())].to_string(),
                    resolution_status: ResolutionStatus::Unresolved,
                },
                quality: ReferenceQuality {
                    overall_score: base_ref.confidence as f32,
                    detection_confidence: base_ref.confidence as f32,
                    resolution_confidence: 0.5,
                },
                base: base_ref,
            }
        }).collect()
    }

    pub fn validate_references(&self, references: &[ChunkReference]) -> Vec<ReferenceValidationResult> {
        references.iter().map(|r| self.validate_single_reference(r)).collect()
    }

    pub fn validate_single_reference(&self, reference: &ChunkReference) -> ReferenceValidationResult {
        let mut issues = Vec::new();
        
        if reference.target_id.is_empty() {
            issues.push("Empty target".to_string());
        }
        
        if reference.confidence < 0.0 || reference.confidence > 1.0 {
            issues.push("Invalid confidence".to_string());
        }

        ReferenceValidationResult {
            is_valid: issues.is_empty(),
            confidence: reference.confidence,
            issues,
        }
    }

    pub fn build_reference_graph(&self, chunks: &[crate::Chunk]) -> ReferenceGraph {
        let mut graph = ReferenceGraph::new();
        for chunk in chunks {
            graph.add_node(chunk.id.to_string(), chunk.clone());
        }
        for chunk in chunks {
            for reference in &chunk.references {
                graph.add_edge(
                    chunk.id.to_string(),
                    reference.target_id.clone(),
                    reference.clone(),
                );
            }
        }
        graph
    }
}

impl ReferenceGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, id: String, chunk: crate::Chunk) {
        self.nodes.insert(id, chunk);
    }

    pub fn add_edge(&mut self, from: String, to: String, reference: ChunkReference) {
        self.edges.entry(from).or_insert_with(Vec::new).push((to, reference));
    }

    pub fn get_outgoing_edges(&self, node_id: &str) -> Option<&Vec<(String, ChunkReference)>> {
        self.edges.get(node_id)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|edges| edges.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_extraction() {
        let tracker = ReferenceTracker::new();
        let content = "Please see section 2 for more details. Also check [1] for references.";
        let references = tracker.extract_references(content);
        
        assert!(!references.is_empty());
    }

    #[test]
    fn test_reference_validation() {
        let tracker = ReferenceTracker::new();
        let reference = ChunkReference {
            reference_type: ReferenceType::CrossReference,
            target_id: "section-1".to_string(),
            context: Some("test".to_string()),
            confidence: 0.8,
        };
        
        let validation = tracker.validate_single_reference(&reference);
        assert!(validation.is_valid);
    }

    #[test]
    fn test_empty_target_validation() {
        let tracker = ReferenceTracker::new();
        let reference = ChunkReference {
            reference_type: ReferenceType::CrossReference,
            target_id: String::new(),
            context: Some("test".to_string()),
            confidence: 0.8,
        };
        
        let validation = tracker.validate_single_reference(&reference);
        assert!(!validation.is_valid);
    }
}