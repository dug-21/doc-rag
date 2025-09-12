// src/symbolic/src/prolog/mod.rs
// Prolog engine module

pub mod engine;
pub mod inference;
pub mod knowledge_base;

pub use engine::{PrologEngine, PrologQuery, ProofResult};
pub use inference::InferenceEngine;
pub use knowledge_base::KnowledgeBase;