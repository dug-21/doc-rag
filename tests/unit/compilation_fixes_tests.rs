//! London TDD Tests for Compilation Fixes
//! 
//! Test-first implementation for missing methods and types to resolve
//! all 21+ compilation errors using TDD methodology.

use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;
use chrono::{DateTime, Utc};

// Import the modules we're testing
use query_processor::{
    Query, ProcessedQuery, QueryMetadata, QueryPriority, 
    SemanticAnalysis, ExtractedEntity, KeyTerm, IntentClassification,
    StrategySelection, ConsensusResult, ProcessingMetadata, ProcessingStatistics,
    ResourceUsage, ValidationResult, ValidationStatus, PerformanceMetrics,
    SyntacticFeatures, SemanticFeatures, Dependency, Topic
};

use response_generator::{ResponseError, Result};

/// Test suite for the missing add_metadata method on ProcessedQuery
#[cfg(test)]
mod add_metadata_tests {
    use super::*;

    #[test]
    fn test_add_metadata_method_exists() {
        // This test will fail until we implement add_metadata
        let query = create_test_query();
        let processed_query = create_test_processed_query(query);
        
        // Test that add_metadata method exists and works
        let key = "test_key".to_string();
        let value = "test_value".to_string();
        
        // This should compile once we add the method
        let result = processed_query.add_metadata(key.clone(), value.clone());
        assert!(result.is_ok(), "add_metadata should succeed with valid inputs");
        
        // Verify the metadata was added
        // Note: We'll need access to the metadata to verify this
        // This drives the interface design
    }

    #[test] 
    fn test_add_metadata_with_empty_key() {
        let query = create_test_query();
        let mut processed_query = create_test_processed_query(query);
        
        let result = processed_query.add_metadata("".to_string(), "value".to_string());
        assert!(result.is_err(), "add_metadata should fail with empty key");
    }

    #[test]
    fn test_add_metadata_overwrites_existing() {
        let query = create_test_query();
        let mut processed_query = create_test_processed_query(query);
        
        let key = "test_key".to_string();
        
        // Add initial value
        processed_query.add_metadata(key.clone(), "initial_value".to_string()).unwrap();
        
        // Overwrite with new value
        processed_query.add_metadata(key.clone(), "new_value".to_string()).unwrap();
        
        // Verify the new value is present (need getter method)
        // This drives the need for a get_metadata method as well
    }

    #[test]
    fn test_add_metadata_with_null_values() {
        let query = create_test_query();
        let mut processed_query = create_test_processed_query(query);
        
        // Test various edge cases
        let result = processed_query.add_metadata("null_test".to_string(), "".to_string());
        assert!(result.is_ok(), "empty string value should be allowed");
    }

    #[test]
    fn test_add_metadata_returns_previous_value() {
        let query = create_test_query();
        let mut processed_query = create_test_processed_query(query);
        
        let key = "return_test".to_string();
        let initial_value = "initial".to_string();
        let new_value = "new".to_string();
        
        // First addition should return None (no previous value)
        let result1 = processed_query.add_metadata(key.clone(), initial_value.clone()).unwrap();
        assert_eq!(result1, None, "First add_metadata should return None");
        
        // Second addition should return the previous value
        let result2 = processed_query.add_metadata(key.clone(), new_value.clone()).unwrap();
        assert_eq!(result2, Some(initial_value), "Second add_metadata should return previous value");
    }
}

/// Test suite for missing methods in EnhancedQueryClassifier  
#[cfg(test)]
mod classifier_tests {
    use super::*;
    use query_processor::{EnhancedQueryClassifier, EnhancedClassifierConfig};

    #[tokio::test]
    async fn test_classify_method_exists() {
        // This test will fail until we implement classify method
        let config = EnhancedClassifierConfig::default();
        let classifier = EnhancedQueryClassifier::new(config).await.unwrap();
        
        let query = create_test_query();
        let analysis = create_test_semantic_analysis();
        
        // Test that classify method exists and works
        let result = classifier.classify(&query, &analysis).await;
        assert!(result.is_ok(), "classify method should exist and work");
        
        let classification_result = result.unwrap();
        // Verify result structure - this drives the interface design
        assert!(!classification_result.intent.to_string().is_empty(), "Should return valid intent");
    }

    #[tokio::test]
    async fn test_classify_with_invalid_query() {
        let config = EnhancedClassifierConfig::default();
        let classifier = EnhancedQueryClassifier::new(config).await.unwrap();
        
        // Test with empty query text
        let empty_query = Query::new("").unwrap(); // This might fail, which is good to test
        let analysis = create_test_semantic_analysis();
        
        let result = classifier.classify(&empty_query, &analysis).await;
        // Should handle gracefully or return appropriate error
        // This drives error handling requirements
    }

    #[tokio::test]
    async fn test_classify_performance_constraint() {
        let config = EnhancedClassifierConfig::default();
        let classifier = EnhancedQueryClassifier::new(config).await.unwrap();
        
        let query = create_test_query();
        let analysis = create_test_semantic_analysis();
        
        let start = std::time::Instant::now();
        let _result = classifier.classify(&query, &analysis).await.unwrap();
        let elapsed = start.elapsed();
        
        // CONSTRAINT: Classification should be fast
        assert!(elapsed < Duration::from_millis(100), "Classification should be under 100ms");
    }
}

/// Test suite for missing ResponseError variants
#[cfg(test)]
mod response_error_tests {
    use super::*;

    #[test]
    fn test_constraint_violation_variant_exists() {
        // Test that ConstraintViolation variant exists
        let constraint_error = ResponseError::ConstraintViolation {
            constraint: "CONSTRAINT-004".to_string(),
            violation: "Free text generation attempted".to_string(),
            context: Some("Template engine bypassed".to_string()),
        };
        
        assert!(constraint_error.to_string().contains("ConstraintViolation"));
        assert!(constraint_error.to_string().contains("CONSTRAINT-004"));
    }

    #[test]
    fn test_constraint_violation_serialization() {
        let constraint_error = ResponseError::ConstraintViolation {
            constraint: "CONSTRAINT-003".to_string(),
            violation: "Neural inference time exceeded 10ms".to_string(),
            context: Some("99.5ms inference time measured".to_string()),
        };
        
        // Should be serializable for logging and debugging
        let debug_str = format!("{:?}", constraint_error);
        assert!(debug_str.contains("ConstraintViolation"));
    }

    #[test]
    fn test_constraint_violation_without_context() {
        let constraint_error = ResponseError::ConstraintViolation {
            constraint: "CONSTRAINT-001".to_string(),
            violation: "Accuracy below 99%".to_string(),
            context: None,
        };
        
        // Should handle optional context gracefully
        let error_msg = constraint_error.to_string();
        assert!(error_msg.contains("CONSTRAINT-001"));
        assert!(error_msg.contains("Accuracy below 99%"));
    }
}

/// Test suite for missing CitationDeduplicationStrategy
#[cfg(test)]
mod citation_deduplication_tests {
    use super::*;
    use response_generator::{CitationDeduplicationStrategy, Citation, Source};

    #[test]
    fn test_citation_deduplication_strategy_enum_exists() {
        // Test that the enum variants exist
        let strategy1 = CitationDeduplicationStrategy::Exact;
        let strategy2 = CitationDeduplicationStrategy::Fuzzy;
        let strategy3 = CitationDeduplicationStrategy::Semantic;
        let strategy4 = CitationDeduplicationStrategy::Hybrid;
        
        // Should be different variants
        assert_ne!(format!("{:?}", strategy1), format!("{:?}", strategy2));
        assert_ne!(format!("{:?}", strategy2), format!("{:?}", strategy3));
        assert_ne!(format!("{:?}", strategy3), format!("{:?}", strategy4));
    }

    #[test]
    fn test_citation_deduplication_strategy_default() {
        let default_strategy = CitationDeduplicationStrategy::default();
        // Default should be a reasonable choice - probably Hybrid for best results
        assert_eq!(format!("{:?}", default_strategy), "Hybrid");
    }

    #[test]
    fn test_citation_deduplication_strategy_application() {
        let citations = vec![
            create_test_citation("source1", "page1"),
            create_test_citation("source1", "page1"), // Exact duplicate
            create_test_citation("source1", "page2"),  // Same source, different page
        ];
        
        let strategy = CitationDeduplicationStrategy::Exact;
        
        // Should be able to apply the strategy to deduplicate
        let deduplicated = strategy.deduplicate(citations);
        assert_eq!(deduplicated.len(), 2, "Exact strategy should remove exact duplicate");
    }
}

// Helper functions to create test data
fn create_test_query() -> Query {
    Query::new("What are the encryption requirements for stored payment card data?").unwrap()
}

fn create_test_processed_query(query: Query) -> ProcessedQuery {
    let analysis = create_test_semantic_analysis();
    let entities = vec![];
    let key_terms = vec![];
    let intent = create_test_intent_classification();
    let strategy = create_test_strategy_selection();
    
    ProcessedQuery::new(query, analysis, entities, key_terms, intent, strategy)
}

fn create_test_semantic_analysis() -> SemanticAnalysis {
    SemanticAnalysis::new(
        SyntacticFeatures::default(),
        SemanticFeatures::default(),
        vec![],
        vec![],
        0.8,
        Duration::from_millis(10),
    )
}

fn create_test_intent_classification() -> IntentClassification {
    IntentClassification {
        intent: query_processor::QueryIntent::Factual,
        confidence: 0.85,
        alternatives: vec![],
        reasoning: "Test reasoning".to_string(),
        timestamp: Utc::now(),
    }
}

fn create_test_strategy_selection() -> StrategySelection {
    StrategySelection {
        strategy: query_processor::SearchStrategy::Hybrid,
        confidence: 0.90,
        reasoning: "Test strategy reasoning".to_string(),
        alternatives: vec![],
        parameters: HashMap::new(),
        timestamp: Utc::now(),
    }
}

fn create_test_citation(source_name: &str, page: &str) -> Citation {
    Citation {
        id: Uuid::new_v4(),
        source: Source {
            id: Uuid::new_v4(),
            title: source_name.to_string(),
            url: Some(format!("https://example.com/{}", source_name)),
            author: Some("Test Author".to_string()),
            publication_date: Some(Utc::now()),
            source_type: "document".to_string(),
            metadata: HashMap::new(),
        },
        page_number: Some(page.to_string()),
        section: None,
        quote: None,
        confidence_score: 0.95,
        relevance_score: 0.88,
        context: "Test context".to_string(),
        timestamp: Utc::now(),
    }
}