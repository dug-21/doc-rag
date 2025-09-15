// src/symbolic/src/lib.rs
// Symbolic reasoning module - Datalog and Prolog engines for neurosymbolic RAG

// Core modules
pub mod datalog;
pub mod prolog;
pub mod error;
pub mod types;

// High-level engines
pub mod datalog_engine;
pub mod neural_classifier;
pub mod neurosymbolic;
pub mod neurosymbolic_processor;

// Core logic processing modules
pub mod logic_parser;
pub mod rule_parser;
pub mod inference;
pub mod proof_chains;

// Re-export main types for easy usage (avoiding conflicts)
pub use datalog_engine::{DatalogEngine, DatalogRule, DatalogError, DatalogFact};
pub use neurosymbolic_processor::{NeurosymbolicProcessor, NeurosymbolicMetrics};
pub use neural_classifier::{NeuralClassifier, NeuralClassifierSystem, ClassificationResult};
pub use neurosymbolic::{NeurosymbolicQuery, NeurosymbolicResult, ProcessorError};
pub use error::{SymbolicError, ClassificationError};
pub use inference::{InferenceEngine};
pub use proof_chains::{ProofChainBuilder};
pub use logic_parser::{LogicParser};
pub use rule_parser::{RuleParser};
pub use prolog::engine::{PrologEngine, PrologQuery, ProofResult};

// Re-export types with module prefix to avoid conflicts
pub use types::{RequirementType as TypesRequirementType, Priority as TypesPriority, QueryResult as TypesQueryResult, ProofStep as TypesProofStep, RequirementRule as TypesRequirementRule, SymbolicFact, ProofChain, SymbolicRule, ReasoningType};

// Legacy compatibility - common types (renamed to avoid conflicts)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LegacyRequirementType {
    Security,
    Compliance,
    Functional,
    Performance,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LegacyPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests;

// Include standalone validation tests
#[cfg(test)]
mod standalone_validation;