use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Processed document for graph storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDocument {
    pub id: Uuid,
    pub title: String,
    pub version: String,
    pub doc_type: DocumentType,
    pub hierarchy: DocumentHierarchy,
    pub requirements: Vec<Requirement>,
    pub cross_references: Vec<CrossReference>,
    pub metadata: DocumentMetadata,
    pub created_at: DateTime<Utc>,
}

/// Document type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DocumentType {
    PciDss,
    Iso27001,
    Soc2,
    Nist,
    Unknown,
}

impl std::fmt::Display for DocumentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocumentType::PciDss => write!(f, "PCI-DSS"),
            DocumentType::Iso27001 => write!(f, "ISO-27001"),
            DocumentType::Soc2 => write!(f, "SOC2"),
            DocumentType::Nist => write!(f, "NIST"),
            DocumentType::Unknown => write!(f, "Unknown"),
        }
    }
}

impl DocumentType {
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s.to_uppercase().as_str() {
            "PCI-DSS" | "PCIDSS" => Ok(DocumentType::PciDss),
            "ISO-27001" | "ISO27001" => Ok(DocumentType::Iso27001),
            "SOC2" => Ok(DocumentType::Soc2),
            "NIST" => Ok(DocumentType::Nist),
            "UNKNOWN" => Ok(DocumentType::Unknown),
            _ => Err(format!("Unknown document type: {}", s)),
        }
    }
}

/// Document hierarchy structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentHierarchy {
    pub sections: Vec<Section>,
    pub total_sections: usize,
    pub max_depth: usize,
}

impl DocumentHierarchy {
    pub fn get_section(&self, section_id: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.id == section_id)
    }
}

/// Document section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    pub id: String,
    pub number: String,
    pub title: String,
    pub text: String,
    pub page_range: (u32, u32),
    pub subsections: Vec<Section>,
    pub parent_id: Option<String>,
    pub section_type: SectionType,
}

/// Section classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SectionType {
    Requirements,
    Definitions,
    Procedures,
    Appendices,
    Examples,
    References,
}

impl std::fmt::Display for SectionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SectionType::Requirements => write!(f, "Requirements"),
            SectionType::Definitions => write!(f, "Definitions"),
            SectionType::Procedures => write!(f, "Procedures"),
            SectionType::Appendices => write!(f, "Appendices"),
            SectionType::Examples => write!(f, "Examples"),
            SectionType::References => write!(f, "References"),
        }
    }
}

/// Individual requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Requirement {
    pub id: String,
    pub text: String,
    pub section: String,
    pub requirement_type: RequirementType,
    pub domain: String,
    pub priority: Priority,
    pub cross_references: Vec<String>,
    pub created_at: DateTime<Utc>,
}

/// Requirement type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RequirementType {
    Must,       // Mandatory (MUST/SHALL)
    Should,     // Recommended (SHOULD)
    May,        // Optional (MAY/CAN)
    Guideline,  // Best practice guidance
}

impl std::fmt::Display for RequirementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RequirementType::Must => write!(f, "MUST"),
            RequirementType::Should => write!(f, "SHOULD"),
            RequirementType::May => write!(f, "MAY"),
            RequirementType::Guideline => write!(f, "GUIDELINE"),
        }
    }
}

impl RequirementType {
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s.to_uppercase().as_str() {
            "MUST" | "SHALL" => Ok(RequirementType::Must),
            "SHOULD" => Ok(RequirementType::Should),
            "MAY" | "CAN" => Ok(RequirementType::May),
            "GUIDELINE" => Ok(RequirementType::Guideline),
            _ => Err(format!("Unknown requirement type: {}", s)),
        }
    }
}

/// Requirement priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Priority {
    Critical,   // Security-critical requirements
    High,       // Important compliance requirements
    Medium,     // Standard requirements
    Low,        // Nice-to-have requirements
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Priority::Critical => write!(f, "CRITICAL"),
            Priority::High => write!(f, "HIGH"),
            Priority::Medium => write!(f, "MEDIUM"),
            Priority::Low => write!(f, "LOW"),
        }
    }
}

impl Priority {
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s.to_uppercase().as_str() {
            "CRITICAL" => Ok(Priority::Critical),
            "HIGH" => Ok(Priority::High),
            "MEDIUM" => Ok(Priority::Medium),
            "LOW" => Ok(Priority::Low),
            _ => Err(format!("Unknown priority: {}", s)),
        }
    }
}

/// Cross-reference between requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossReference {
    pub from_requirement: String,
    pub to_requirement: String,
    pub reference_type: ReferenceType,
    pub context: String,
    pub confidence: f64,
}

/// Type of cross-reference
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReferenceType {
    Direct,     // Direct citation (e.g., "see section 3.2.1")
    Implied,    // Implied relationship
    Example,    // Example or illustration
    Exception,  // Exception to the rule
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: String,
    pub version: String,
    pub publication_date: Option<DateTime<Utc>>,
    pub author: Option<String>,
    pub page_count: u32,
    pub word_count: u32,
}

impl DocumentMetadata {
    /// Get a metadata field value by key
    pub fn get(&self, key: &str) -> Option<String> {
        match key {
            "title" => Some(self.title.clone()),
            "version" => Some(self.version.clone()),
            "author" => self.author.clone(),
            _ => None,
        }
    }
}

/// Graph node representing a requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequirementNode {
    pub id: String,
    pub neo4j_id: i64,
    pub text: String,
    pub requirement_type: RequirementType,
    pub section: String,
    pub domain: String,
    pub priority: Priority,
}

/// Graph node representing a section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionNode {
    pub id: String,
    pub neo4j_id: i64,
    pub section_number: String,
    pub title: String,
    pub section_type: SectionType,
}

/// Graph relationship between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEdge {
    pub id: i64,
    pub from_node: String,
    pub to_node: String,
    pub relationship_type: RelationshipType,
    pub properties: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

/// Types of relationships in the graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationshipType {
    References,   // Direct citation
    DependsOn,    // Logical dependency
    Exception,    // Override relationship
    Implements,   // Implementation relationship
    Contains,     // Hierarchical containment
}

impl std::fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelationshipType::References => write!(f, "REFERENCES"),
            RelationshipType::DependsOn => write!(f, "DEPENDS_ON"),
            RelationshipType::Exception => write!(f, "EXCEPTION"),
            RelationshipType::Implements => write!(f, "IMPLEMENTS"),
            RelationshipType::Contains => write!(f, "CONTAINS"),
        }
    }
}

impl RelationshipType {
    pub fn from_string(s: &str) -> Result<Self, String> {
        match s.to_uppercase().as_str() {
            "REFERENCES" => Ok(RelationshipType::References),
            "DEPENDS_ON" => Ok(RelationshipType::DependsOn),
            "EXCEPTION" => Ok(RelationshipType::Exception),
            "IMPLEMENTS" => Ok(RelationshipType::Implements),
            "CONTAINS" => Ok(RelationshipType::Contains),
            _ => Err(format!("Unknown relationship type: {}", s)),
        }
    }
}

/// Complete document graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentGraph {
    pub document_id: String,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<RelationshipEdge>,
    pub metadata: DocumentMetadata,
    pub created_at: DateTime<Utc>,
}

impl DocumentGraph {
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_relationships(&self) -> usize {
        self.edges.len()
    }

    pub async fn validate_completeness(&self) -> Result<bool, String> {
        // Extract node IDs from the nodes vector
        let all_node_ids: std::collections::HashSet<String> = 
            self.nodes.iter().map(|node| {
                match node {
                    GraphNode::Document(d) => d.id.clone(),
                    GraphNode::Section(s) => s.id.clone(),
                    GraphNode::Requirement(r) => r.id.clone(),
                }
            }).collect();

        // Validate relationships point to existing nodes
        for rel in &self.edges {
            if !all_node_ids.contains(&rel.from_node) {
                return Err(format!("Relationship references non-existent from_node: {}", rel.from_node));
            }
            if !all_node_ids.contains(&rel.to_node) {
                return Err(format!("Relationship references non-existent to_node: {}", rel.to_node));
            }
        }

        Ok(true)
    }
}

/// Result of graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalResult {
    pub start_requirement_id: String,
    pub related_requirements: Vec<RelatedRequirement>,
    pub traversal_paths: Vec<TraversalPath>,
    pub execution_time_ms: u64,
    pub total_paths: usize,
}

/// Individual path in graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalPath {
    pub start_node: String,
    pub end_node: String,
    pub path_length: usize,
    pub relationship_chain: Vec<RelationshipType>,
    pub total_weight: f64,
}

/// Related requirement from traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedRequirement {
    pub id: String,
    pub relationship_type: RelationshipType,
    pub distance: usize,
    pub confidence: f64,
}

/// Document node in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentNode {
    pub id: String,
    pub neo4j_id: i64,
    pub title: String,
    pub document_type: String,
    pub created_at: DateTime<Utc>,
    pub metadata: DocumentMetadata,
}

/// Graph node types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphNode {
    Document(DocumentNode),
    Section(SectionNode),
    Requirement(RequirementNode),
}