//! Mock implementations for FACT integration testing
//! 
//! This module provides comprehensive mock implementations and test utilities
//! for testing FACT integration components.

pub mod mock_fact_client;

// Re-export the main mock client and utilities
pub use mock_fact_client::{
    MockFACTClient, 
    FACTClientTrait,
    FACTTestFixtures,
    RealisticFACTSimulator,
    SimulatorConfig,
    MockFACTClientFactory,
    FACTError,
    QueryResult,
    Citation,
    ConsensusVote,
    FACTMetrics,
    HealthStatus,
};