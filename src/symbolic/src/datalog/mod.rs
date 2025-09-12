// src/symbolic/src/datalog/mod.rs
// Datalog engine module

pub mod engine;
pub mod rule_compiler;
pub mod query_processor;

pub use engine::{DatalogEngine, DatalogRule, QueryResult, QueryResultItem};
pub use rule_compiler::RuleCompiler;
pub use query_processor::QueryProcessor;