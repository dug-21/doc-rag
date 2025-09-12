//! Template Engine Test Module
//!
//! Comprehensive London TDD test suite for template response generation with:
//! - <1s response time validation (CONSTRAINT-006)
//! - CONSTRAINT-004 deterministic generation enforcement
//! - Variable substitution from proof chains
//! - Citation formatting and audit trail validation

pub mod template_selection_tests;
pub mod variable_substitution_tests;
pub mod citation_formatting_tests;
pub mod constraint_004_tests;
pub mod performance_tests;
pub mod audit_trail_tests;
pub mod template_validation_tests;
pub mod integration_tests;

use crate::fixtures::*;