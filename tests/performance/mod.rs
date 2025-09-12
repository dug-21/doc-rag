//! Performance testing and validation modules
//! 
//! This module contains comprehensive performance testing capabilities including:
//! - Phase 2 performance harness (tests/performance/phase2_performance_harness.rs)
//! - Performance validator for claims verification
//! - CONSTRAINT-006 compliance validation
//! - Load testing and benchmarking utilities

pub mod phase2_performance_harness;
pub mod performance_validator;
pub mod validate_performance_claims;

pub use phase2_performance_harness::*;
pub use performance_validator::*;