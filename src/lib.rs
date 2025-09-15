//! # Neurosymbolic RAG System
//!
//! A high-performance document retrieval and augmented generation system
//! that combines neural networks with symbolic reasoning for enhanced
//! accuracy and explainability.
//!
//! ## Architecture
//!
//! This system follows neurosymbolic principles by:
//! - Using neural networks for classification and pattern recognition (not generation)
//! - Leveraging symbolic reasoning for logical inference and validation
//! - Maintaining clear separation between neural and symbolic components
//! - Ensuring symbolic components control the overall system behavior
//!
//! ## Constraints
//!
//! - CONSTRAINT-003: Neural networks are used for classification only, not generation
//! - CONSTRAINT-004: Symbolic reasoning controls system behavior
//! - CONSTRAINT-005: Clear separation between neural and symbolic components

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::module_name_repetitions)]

// Re-export the main integration module as the primary interface
pub use integration::*;

// Import workspace crates - these are external dependencies in this workspace setup
extern crate integration;
extern crate graph;
extern crate fact;
extern crate symbolic;

/// Common types and traits used throughout the system
pub mod prelude {
    //! Prelude module containing commonly used types and traits
    //!
    //! This module provides convenient access to the most frequently used
    //! types without requiring explicit imports from individual modules.
    //!
    //! ## Neurosymbolic Architecture Compliance
    //!
    //! This prelude carefully follows neurosymbolic constraints:
    //! - Only exposes classification-related neural components (CONSTRAINT-003)
    //! - Prioritizes symbolic reasoning types and traits
    //! - Maintains clear separation between neural and symbolic exports

    // Core integration types
    pub use integration::{
        QueryRequest, QueryResponse, Citation, ResponseFormat,
        SystemHealth, HealthStatus, ComponentHealth,
        IntegrationConfig, IntegrationError,
        FullSystemIntegration as SystemIntegration,
    };

    // Graph database types (symbolic reasoning)
    pub use graph::{
        GraphDatabase, GraphConfig, GraphError, GraphResult,
        GraphPerformanceMetrics, RequirementFilter,
    };

    // Graph models and types
    pub use graph::models::{
        ProcessedDocument, DocumentGraph, Requirement, RequirementNode,
        RelationshipType, RelationshipEdge, TraversalResult,
    };

    // Symbolic reasoning types (primary interface)
    pub use symbolic::{
        // Engines
        DatalogEngine, DatalogRule, DatalogError, DatalogFact,
        NeurosymbolicProcessor, NeurosymbolicMetrics,
        InferenceEngine, ProofChainBuilder, LogicParser, RuleParser,
        PrologEngine, PrologQuery, ProofResult,

        // Core types
        NeurosymbolicQuery, NeurosymbolicResult, ProcessorError,
        SymbolicError, ClassificationError,

        // Data types (with type prefixes to avoid conflicts)
        TypesRequirementType as RequirementType,
        TypesPriority as Priority,
        TypesQueryResult as QueryResult,
        TypesProofStep as ProofStep,
        TypesRequirementRule as RequirementRule,
        SymbolicFact, ProofChain, SymbolicRule, ReasoningType,
    };

    // Neural classification ONLY (CONSTRAINT-003: No generation)
    pub use symbolic::{
        NeuralClassifier, NeuralClassifierSystem, ClassificationResult,
    };

    // FACT caching system
    pub use fact::{
        FACTCache as Cache, FACTConfig as CacheConfig, FACTClient,
        CacheStats, CachedEntry,
    };

    // Note: External dependencies like async_trait, anyhow, serde, uuid, chrono
    // should be imported directly by users as needed. This keeps the prelude
    // focused on our domain-specific types and avoids dependency issues.

    // Common collections and concurrency types
    pub use std::collections::HashMap;
    pub use std::sync::Arc;
    pub use tokio::sync::{RwLock, Mutex};

    // Re-export Result type with default error
    pub type Result<T> = std::result::Result<T, IntegrationError>;
}

// Note: Individual modules are imported through workspace dependencies
// and not directly included here to maintain the workspace structure