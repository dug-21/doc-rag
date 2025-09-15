//! FACT cache performance verification test
//!
//! This test verifies that the FACT cache system meets the <50ms SLA requirement
//! for cached response retrieval as mandated by Phase 2 requirements.

use response_generator::fact_cache_optimized::{OptimizedFACTCache, OptimizedCacheConfig};
use response_generator::citation::{Citation, Source, CitationType, TextRange};
use serde_json::json;
use std::time::Instant;

#[tokio::test]
async fn test_fact_cache_performance_sla() {
    // Create FACT system with reasonable cache size
    let config = OptimizedCacheConfig::default();
    let fact_cache = OptimizedFACTCache::new(config);

    // Create test citation using neurosymbolic architecture types
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("page".to_string(), "42".to_string());
    metadata.insert("section".to_string(), "Section 4.2".to_string());
    metadata.insert("version".to_string(), "v4.0".to_string());
    metadata.insert("authors".to_string(), "PCI Security Standards Council".to_string());

    let source = Source {
        id: uuid::Uuid::new_v4(),
        title: "PCI DSS Compliance Guide".to_string(),
        url: Some("file://test_document.pdf".to_string()),
        document_type: "compliance_standard".to_string(),
        metadata,
    };

    let citation = Citation {
        id: uuid::Uuid::new_v4(),
        source,
        text_range: TextRange { start: 100, end: 200, length: 100 },
        confidence: 0.95,
        citation_type: CitationType::SupportingEvidence,
        relevance_score: 0.95,
        supporting_text: Some("Section 4.2".to_string()),
    };

    // Store test response using neurosymbolic FACT cache
    let test_query = "What are the encryption requirements for stored payment card data?";
    let test_response = json!({
        "answer": "PCI DSS requires strong cryptographic protection for stored cardholder data...",
        "citations": [citation],
        "confidence": 0.95,
        "response_type": "compliance_requirement"
    });

    // Store in FACT cache
    fact_cache.put(
        test_query.to_string(),
        test_response.clone(),
        Some("PCI DSS requires strong cryptographic protection...")
    ).await.expect("Should store response successfully");

    // Verify cache retrieval meets <50ms SLA
    let start = Instant::now();
    let result = fact_cache.get(test_query).await;
    let elapsed = start.elapsed();

    // Verify successful retrieval
    assert!(result.is_some(), "FACT cache should retrieve stored response");

    // Verify <50ms SLA compliance
    assert!(elapsed.as_millis() < 50,
        "FACT cache retrieval took {}ms, exceeding 50ms SLA",
        elapsed.as_millis()
    );

    let cached_response = result.unwrap();
    assert_eq!(cached_response["answer"], "PCI DSS requires strong cryptographic protection for stored cardholder data...");
    assert!(cached_response["citations"].is_array());

    println!("✅ FACT cache performance: {}ms (target: <50ms)", elapsed.as_millis());
}

#[tokio::test]
async fn test_fact_cache_hit_rate_tracking() {
    let config = OptimizedCacheConfig::default();
    let fact_cache = OptimizedFACTCache::new(config);

    // Initially hit rate should be 0
    let initial_metrics = fact_cache.get_performance_metrics();
    assert_eq!(initial_metrics.hit_rate, 0.0);

    // Store a response
    let test_response = json!({
        "answer": "test response",
        "confidence": 0.9
    });

    fact_cache.put(
        "test query".to_string(),
        test_response,
        Some("test response")
    ).await.expect("Should store response");

    // First access should increase hit rate
    let _result = fact_cache.get("test query").await;

    // Hit rate should be positive after cache hit
    let final_metrics = fact_cache.get_performance_metrics();
    assert!(final_metrics.hit_rate > 0.0);

    println!("✅ FACT cache hit rate: {:.2}", final_metrics.hit_rate);
}

#[tokio::test]
async fn test_fact_cache_invalidation() {
    let config = OptimizedCacheConfig::default();
    let fact_cache = OptimizedFACTCache::new(config);

    // Store response
    let test_response = json!({
        "answer": "temp response",
        "confidence": 0.8
    });

    fact_cache.put(
        "temp query".to_string(),
        test_response,
        Some("temp response")
    ).await.expect("Should store response");

    // Verify it's cached
    let cached_result = fact_cache.get("temp query").await;
    assert!(cached_result.is_some(), "Response should be cached");

    // Clear cache
    fact_cache.clear().await;

    // Verify cache miss after clearing
    let cleared_result = fact_cache.get("temp query").await;
    assert!(cleared_result.is_none(), "Response should be cleared from cache");

    println!("✅ FACT cache invalidation works correctly");
}