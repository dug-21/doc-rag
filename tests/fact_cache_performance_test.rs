//! FACT cache performance verification test
//! 
//! This test verifies that the FACT cache system meets the <50ms SLA requirement
//! for cached response retrieval as mandated by Phase 2 requirements.

use fact::{FactSystem, Citation};
use std::time::Instant;

#[tokio::test]
async fn test_fact_cache_performance_sla() {
    // Create FACT system with reasonable cache size
    let fact_system = FactSystem::new(1000);
    
    // Create test citation
    let citation = Citation {
        source: "test_document.pdf".to_string(),
        page: Some(42),
        section: Some("Section 4.2".to_string()),
        relevance_score: 0.95,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };
    
    // Store test response
    let test_query = "What are the encryption requirements for stored payment card data?";
    let test_response = "PCI DSS requires strong cryptographic protection for stored cardholder data...";
    
    fact_system.store_response(
        test_query.to_string(),
        test_response.to_string(),
        vec![citation]
    );
    
    // Verify cache retrieval meets <50ms SLA
    let start = Instant::now();
    let result = fact_system.process_query(test_query);
    let elapsed = start.elapsed();
    
    // Verify successful retrieval
    assert!(result.is_ok(), "FACT cache should retrieve stored response");
    
    // Verify <50ms SLA compliance
    assert!(elapsed.as_millis() < 50, 
        "FACT cache retrieval took {}ms, exceeding 50ms SLA", 
        elapsed.as_millis()
    );
    
    let cached_response = result.unwrap();
    assert_eq!(cached_response.content, test_response);
    assert_eq!(cached_response.citations.len(), 1);
    assert_eq!(cached_response.citations[0].source, "test_document.pdf");
    
    println!("✅ FACT cache performance: {}ms (target: <50ms)", elapsed.as_millis());
}

#[tokio::test]
async fn test_fact_cache_hit_rate_tracking() {
    let fact_system = FactSystem::new(100);
    
    // Initially hit rate should be 0
    assert_eq!(fact_system.cache.hit_rate(), 0.0);
    
    // Store a response
    fact_system.store_response(
        "test query".to_string(),
        "test response".to_string(),
        vec![]
    );
    
    // First access should increase hit rate
    let _result = fact_system.process_query("test query");
    
    // Hit rate should be positive after cache hit
    assert!(fact_system.cache.hit_rate() > 0.0);
    
    println!("✅ FACT cache hit rate: {:.2}", fact_system.cache.hit_rate());
}

#[tokio::test]
async fn test_fact_cache_invalidation() {
    let fact_system = FactSystem::new(100);
    
    // Store response
    fact_system.store_response(
        "temp query".to_string(),
        "temp response".to_string(),
        vec![]
    );
    
    // Verify it's cached
    assert!(fact_system.process_query("temp query").is_ok());
    
    // Clear cache
    fact_system.cache.clear();
    fact_system.tracker.clear();
    
    // Verify cache miss after clearing
    assert!(fact_system.process_query("temp query").is_err());
    
    println!("✅ FACT cache invalidation works correctly");
}