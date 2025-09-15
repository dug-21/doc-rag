//! Proof chain generation and validation
//! CONSTRAINT-001: Logic Programming Foundation - <100ms query response time

use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tracing::{info, debug, warn, instrument};
use uuid::Uuid;

use crate::error::{SymbolicError, Result};
use crate::types::{ProofStep, ProofChain, SymbolicFact};

/// Simple proof validation result
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProofValidation {
    pub is_complete: bool,
    pub confidence_score: f64,
    pub validation_errors: Vec<String>,
}

/// Proof chain builder for symbolic reasoning
#[derive(Debug)]
pub struct ProofChainBuilder {
    current_chain: Vec<ProofStep>,
    fact_database: HashMap<String, SymbolicFact>,
    chain_id: Uuid,
}

impl ProofChainBuilder {
    pub fn new() -> Self {
        Self {
            current_chain: Vec::new(),
            fact_database: HashMap::new(),
            chain_id: Uuid::new_v4(),
        }
    }

    /// Build proof chain for a query with performance constraint
    #[instrument(skip(self))]
    pub async fn build_proof_chain(
        &mut self,
        query: &str,
        conclusion: SymbolicFact,
        steps: Vec<ProofStep>,
    ) -> Result<ProofChain> {
        let start = Instant::now();

        debug!("Building proof chain for query: {}", query);

        // Validate proof steps
        let validated_steps = self.validate_proof_steps(&steps).await?;

        // Calculate overall confidence
        let confidence = self.calculate_confidence(&validated_steps);

        let elapsed = start.elapsed();
        let inference_time_ms = elapsed.as_millis() as u64;

        // CONSTRAINT-001: Must be <100ms
        if inference_time_ms >= 100 {
            warn!("Proof chain building took {}ms, exceeds 100ms constraint", inference_time_ms);
        }

        let proof_chain = ProofChain {
            query: query.to_string(),
            conclusion,
            steps: validated_steps,
            confidence,
            inference_time_ms,
            created_at: chrono::Utc::now(),
        };

        info!("Proof chain built with {} steps in {}ms",
              proof_chain.steps.len(), inference_time_ms);

        Ok(proof_chain)
    }

    /// Validate individual proof steps
    async fn validate_proof_steps(&self, steps: &[ProofStep]) -> Result<Vec<ProofStep>> {
        let mut validated_steps = Vec::new();

        for step in steps {
            if !step.rule.is_empty() && !step.conclusion.is_empty() {
                validated_steps.push(step.clone());
            } else {
                warn!("Invalid proof step detected: {:?}", step);
            }
        }

        Ok(validated_steps)
    }

    /// Calculate confidence based on proof steps
    fn calculate_confidence(&self, steps: &[ProofStep]) -> f64 {
        if steps.is_empty() {
            return 0.0;
        }

        let total_confidence: f64 = steps.iter().map(|s| s.confidence).sum();
        total_confidence / steps.len() as f64
    }

    /// Add fact to the database
    pub fn add_fact(&mut self, predicate: String, fact: SymbolicFact) {
        self.fact_database.insert(predicate, fact);
    }
}

impl Default for ProofChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}
