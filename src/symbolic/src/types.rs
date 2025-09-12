// src/symbolic/src/types.rs
// Common types for symbolic reasoning system

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Requirement classification based on modal verbs
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RequirementType {
    Must,       // MUST/SHALL - mandatory
    Should,     // SHOULD - recommended
    May,        // MAY - optional
    MustNot,    // MUST NOT/SHALL NOT - prohibited
    Guideline,  // Best practice guidance
}

impl RequirementType {
    pub fn from_string(s: &str) -> Result<Self, crate::error::SymbolicError> {
        match s.to_lowercase().as_str() {
            "must" | "shall" => Ok(RequirementType::Must),
            "should" => Ok(RequirementType::Should),
            "may" | "can" => Ok(RequirementType::May),
            "must not" | "shall not" => Ok(RequirementType::MustNot),
            "guideline" => Ok(RequirementType::Guideline),
            _ => Err(crate::error::SymbolicError::ParseError(
                format!("Unknown requirement type: {}", s)
            )),
        }
    }
}

impl ToString for RequirementType {
    fn to_string(&self) -> String {
        match self {
            RequirementType::Must => "must".to_string(),
            RequirementType::Should => "should".to_string(),
            RequirementType::May => "may".to_string(),
            RequirementType::MustNot => "must not".to_string(),
            RequirementType::Guideline => "guideline".to_string(),
        }
    }
}

/// Priority level for requirements
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    Critical,   // Security-critical requirements
    High,       // Important compliance requirements
    Medium,     // Standard requirements
    Low,        // Nice-to-have requirements
}

impl Priority {
    pub fn from_string(s: &str) -> Result<Self, crate::error::SymbolicError> {
        match s.to_lowercase().as_str() {
            "critical" => Ok(Priority::Critical),
            "high" => Ok(Priority::High),
            "medium" => Ok(Priority::Medium),
            "low" => Ok(Priority::Low),
            _ => Err(crate::error::SymbolicError::ParseError(
                format!("Unknown priority: {}", s)
            )),
        }
    }
}

impl ToString for Priority {
    fn to_string(&self) -> String {
        match self {
            Priority::Critical => "critical".to_string(),
            Priority::High => "high".to_string(),
            Priority::Medium => "medium".to_string(),
            Priority::Low => "low".to_string(),
        }
    }
}

/// Entity extracted from requirement text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub entity_type: String,
    pub confidence: f64,
    pub normalized_form: String,
}

/// Action verb extracted from requirement text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub verb: String,
    pub predicate: String,
    pub confidence: f64,
}

/// Condition or constraint in requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub condition_type: String,
    pub value: String,
    pub operator: Option<String>,
    pub confidence: f64,
}

/// Cross-reference to other requirements or documents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossReference {
    pub reference: String,
    pub reference_type: String,
    pub context: String,
}

/// Query type classification for routing
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    Existence,    // "Does X exist?"
    Relationship, // "What relates to X?"
    Inference,    // "What are the implications of X?"
    Compliance,   // "Is X compliant with Y?"
}

/// Proof step in logical inference chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStep {
    pub step_number: usize,
    pub rule: String,
    pub premises: Vec<String>,
    pub conclusion: String,
    pub source_section: String,
    pub conditions: Vec<String>,
    pub confidence: f64,
}

/// Citation for proof step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub source_document: String,
    pub section_reference: Option<String>,
    pub page_number: Option<u32>,
    pub quoted_text: String,
    pub context: String,
}

/// Performance metrics for symbolic reasoning
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_queries: u64,
    pub total_rules_added: u64,
    pub average_query_time_ms: f64,
    pub cache_hit_count: u64,
    pub cache_miss_count: u64,
    pub parse_time_ms: f64,
    pub inference_time_ms: f64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn cache_hit_rate(&self) -> f64 {
        if self.cache_hit_count + self.cache_miss_count == 0 {
            0.0
        } else {
            self.cache_hit_count as f64 / (self.cache_hit_count + self.cache_miss_count) as f64
        }
    }
}

/// Validation result for proof chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofValidation {
    pub is_complete: bool,
    pub confidence_score: f64,
    pub has_gaps: bool,
    pub has_circular_dependencies: bool,
    pub missing_premises: Vec<String>,
    pub validation_errors: Vec<String>,
}

impl Default for ProofValidation {
    fn default() -> Self {
        Self {
            is_complete: true,
            confidence_score: 1.0,
            has_gaps: false,
            has_circular_dependencies: false,
            missing_premises: Vec::new(),
            validation_errors: Vec::new(),
        }
    }
}

/// Temporal constraint in requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    pub constraint_type: String,
    pub value: String,
    pub unit: String,
}

/// Quantifier in logical expressions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quantifier {
    pub quantifier_type: String, // forall, exists, etc.
    pub variable: String,
    pub scope: String,
}

/// Conditional logic structure (if-then-else)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalStructure {
    pub condition_type: String,
    pub if_condition: String,
    pub then_clause: String,
    pub else_clause: Option<String>,
}

impl Default for ConditionalStructure {
    fn default() -> Self {
        Self {
            condition_type: "none".to_string(),
            if_condition: "".to_string(),
            then_clause: "".to_string(),
            else_clause: None,
        }
    }
}

/// Exception clause in requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Exception {
    pub condition: String,
    pub exception_type: String,
    pub scope: String,
}

/// Alternative interpretation for ambiguous requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeInterpretation {
    pub subject: String,
    pub predicate: String,
    pub confidence: f64,
    pub rationale: String,
}

/// Parsed requirement components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedRequirement {
    pub original_text: String,
    pub requirement_type: RequirementType,
    pub entities: Vec<Entity>,
    pub actions: Vec<Action>,
    pub conditions: Vec<Condition>,
    pub cross_references: Vec<CrossReference>,
    pub confidence: f64,
}

