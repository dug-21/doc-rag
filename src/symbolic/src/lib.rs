// src/symbolic/src/lib.rs
// Symbolic reasoning module - Datalog and Prolog engines for neurosymbolic RAG

pub mod datalog;
pub mod prolog; 
pub mod logic_parser;
pub mod types;
pub mod error;

// Re-export main types for easy usage
pub use datalog::{DatalogEngine, DatalogRule, QueryResult};
pub use prolog::{PrologEngine, PrologQuery, ProofResult};
pub use logic_parser::{LogicParser, ParsedLogic};
pub use types::{RequirementType, Priority, Entity, Action, Condition, CrossReference};
pub use error::{SymbolicError, Result};

#[cfg(test)]
mod tests;