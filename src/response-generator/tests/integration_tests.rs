//! Integration tests for the response generator system

use response_generator::{
    Config, ResponseGenerator, GenerationRequest, ContextChunk, Source, OutputFormat,
    error::Result,
};
use std::collections::HashMap;
use tokio_test;
use tokio_stream::StreamExt;
use uuid::Uuid;

/// Test basic response generation flow
#[tokio::test]
async fn test_basic_response_generation() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    let request = GenerationRequest::builder()
        .query("What is Rust programming language?")
        .build()?;
    
    let response = generator.generate(request).await?;
    
    assert!(!response.content.is_empty());
    assert!(response.confidence_score > 0.0);
    assert!(response.metrics.total_duration.as_millis() > 0);
    
    Ok(())
}

/// Test response generation with context
#[tokio::test]
async fn test_response_with_context() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    let context_chunk = ContextChunk {
        content: "Rust is a systems programming language that focuses on safety, speed, and concurrency.".to_string(),
        source: Source {
            id: Uuid::new_v4(),
            title: "Rust Programming Guide".to_string(),
            url: Some("https://doc.rust-lang.org/".to_string()),
            document_type: "documentation".to_string(),
            metadata: HashMap::new(),
        },
        relevance_score: 0.9,
        position: Some(0),
        metadata: HashMap::new(),
    };
    
    let request = GenerationRequest::builder()
        .query("What is Rust?")
        .add_context(context_chunk)
        .build()?;
    
    let response = generator.generate(request).await?;
    
    assert!(!response.content.is_empty());
    assert!(response.confidence_score > 0.5);
    assert!(!response.citations.is_empty());
    
    Ok(())
}

/// Test different output formats
#[tokio::test]
async fn test_output_formats() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    let formats = vec![
        OutputFormat::Json,
        OutputFormat::Markdown,
        OutputFormat::Text,
        OutputFormat::Html,
        OutputFormat::Xml,
        OutputFormat::Yaml,
    ];
    
    for format in formats {
        let request = GenerationRequest::builder()
            .query("Test query")
            .format(format.clone())
            .build()?;
        
        let response = generator.generate(request).await?;
        assert!(!response.content.is_empty());
        assert_eq!(response.format, format);
    }
    
    Ok(())
}

/// Test confidence threshold filtering
#[tokio::test]
async fn test_confidence_threshold() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    // Test with high confidence requirement
    let request = GenerationRequest::builder()
        .query("What is artificial intelligence?")
        .min_confidence(0.95) // Very high threshold
        .build()?;
    
    // This should either succeed with high confidence or fail
    let result = generator.generate(request).await;
    
    match result {
        Ok(response) => {
            assert!(response.confidence_score >= 0.95);
        }
        Err(_) => {
            // Expected if confidence is insufficient
        }
    }
    
    Ok(())
}

/// Test streaming response generation
#[tokio::test]
async fn test_streaming_response() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    let request = GenerationRequest::builder()
        .query("Explain machine learning")
        .build()?;
    
    let mut generator = generator;
    let stream = generator.generate_stream(request).await?;
    
    // Collect all chunks
    let chunks: Vec<_> = tokio_stream::StreamExt::collect(stream).await;
    
    assert!(!chunks.is_empty());
    
    // Verify all chunks are Ok
    for chunk_result in chunks {
        assert!(chunk_result.is_ok());
    }
    
    Ok(())
}

/// Test performance targets
#[tokio::test]
async fn test_performance_targets() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    let start_time = std::time::Instant::now();
    
    let request = GenerationRequest::builder()
        .query("What is cloud computing?")
        .build()?;
    
    let response = generator.generate(request).await?;
    let elapsed = start_time.elapsed();
    
    // Verify response generation is under 200ms (allowing some margin)
    assert!(elapsed.as_millis() < 200, "Response took {}ms, expected <200ms", elapsed.as_millis());
    
    // Verify internal metrics also show good performance
    assert!(response.metrics.total_duration.as_millis() < 150);
    
    Ok(())
}

/// Test validation layers
#[tokio::test]
async fn test_validation_layers() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    let request = GenerationRequest::builder()
        .query("This is a test query with specific requirements")
        .build()?;
    
    let response = generator.generate(request).await?;
    
    // Verify validation results are present
    assert!(!response.validation_results.is_empty());
    
    // Check that validation layers ran
    let layer_names: Vec<String> = response.validation_results
        .iter()
        .map(|vr| vr.layer_name.clone())
        .collect();
    
    assert!(layer_names.contains(&"factual_accuracy".to_string()));
    assert!(layer_names.contains(&"coherence_validation".to_string()));
    
    Ok(())
}

/// Test citation tracking
#[tokio::test]
async fn test_citation_tracking() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    // Create multiple context chunks with different sources
    let context_chunks = vec![
        ContextChunk {
            content: "First source of information about the topic.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Source 1".to_string(),
                url: Some("https://source1.com".to_string()),
                document_type: "article".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.8,
            position: Some(0),
            metadata: HashMap::new(),
        },
        ContextChunk {
            content: "Second source with different perspective on the topic.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Source 2".to_string(),
                url: Some("https://source2.com".to_string()),
                document_type: "research".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.7,
            position: Some(0),
            metadata: HashMap::new(),
        },
    ];
    
    let request = GenerationRequest::builder()
        .query("What does research say about this topic?")
        .context(context_chunks)
        .build()?;
    
    let response = generator.generate(request).await?;
    
    // Verify citations are generated
    assert!(!response.citations.is_empty());
    
    // Verify citation quality
    for citation in &response.citations {
        assert!(citation.confidence > 0.0);
        assert!(citation.relevance_score > 0.0);
        assert!(!citation.source.title.is_empty());
    }
    
    Ok(())
}

/// Test error handling and recovery
#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    // Test with empty query
    let request = GenerationRequest::builder()
        .query("")
        .build()?;
    
    let result = generator.generate(request).await;
    
    // Should handle gracefully (either succeed with warning or fail appropriately)
    match result {
        Ok(response) => {
            // If it succeeds, should have warnings
            assert!(!response.warnings.is_empty());
        }
        Err(_) => {
            // Expected error is also acceptable
        }
    }
    
    Ok(())
}

/// Test concurrent request handling
#[tokio::test]
async fn test_concurrent_requests() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    // Create multiple concurrent requests
    let requests = (0..10).map(|i| {
        GenerationRequest::builder()
            .query(format!("Query number {}", i))
            .build()
            .unwrap()
    }).collect::<Vec<_>>();
    
    // Execute all requests concurrently
    let futures: Vec<_> = requests.into_iter().map(|request| {
        generator.generate(request)
    }).collect();
    
    let results = futures::future::join_all(futures).await;
    
    // Verify all requests completed
    assert_eq!(results.len(), 10);
    
    // Verify at least some succeeded
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    assert!(success_count >= 8, "Expected at least 8 successful requests, got {}", success_count);
    
    Ok(())
}

/// Test configuration loading and validation
#[tokio::test]
async fn test_configuration() -> Result<()> {
    // Test default configuration
    let config = Config::default();
    assert!(config.validate().is_ok());
    
    // Test custom configuration
    let custom_config = Config::builder()
        .max_response_length(2048)
        .default_confidence_threshold(0.8)
        .build();
    
    assert!(custom_config.validate().is_ok());
    assert_eq!(custom_config.max_response_length, 2048);
    
    Ok(())
}

/// Test memory usage and cleanup
#[tokio::test]
async fn test_memory_management() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    // Generate many responses to test memory management
    for i in 0..50 {
        let request = GenerationRequest::builder()
            .query(format!("Memory test query {}", i))
            .build()?;
        
        let _response = generator.generate(request).await?;
        
        // Memory usage should remain reasonable
        // This is a basic test - in production, you'd use more sophisticated memory monitoring
    }
    
    Ok(())
}

/// Integration test with realistic data
#[tokio::test]
async fn test_realistic_scenario() -> Result<()> {
    let config = Config::default();
    let generator = ResponseGenerator::new(config).await;
    
    // Create realistic context chunks
    let context_chunks = vec![
        ContextChunk {
            content: "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can work and react like humans. Some of the activities computers with artificial intelligence are designed for include speech recognition, learning, planning, and problem-solving.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Introduction to Artificial Intelligence".to_string(),
                url: Some("https://www.ibm.com/topics/artificial-intelligence".to_string()),
                document_type: "article".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("publish_date".to_string(), "2023-01-15T10:00:00Z".to_string());
                    meta.insert("author".to_string(), "IBM Research".to_string());
                    meta
                },
            },
            relevance_score: 0.95,
            position: Some(0),
            metadata: HashMap::new(),
        },
        ContextChunk {
            content: "Machine learning is a subset of AI that focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. Machine learning is an important component of the growing field of data science.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "What is Machine Learning?".to_string(),
                url: Some("https://www.ibm.com/topics/machine-learning".to_string()),
                document_type: "article".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("publish_date".to_string(), "2023-02-20T14:30:00Z".to_string());
                    meta.insert("credibility_score".to_string(), "0.9".to_string());
                    meta
                },
            },
            relevance_score: 0.88,
            position: Some(1),
            metadata: HashMap::new(),
        },
    ];
    
    let request = GenerationRequest::builder()
        .query("What is the relationship between artificial intelligence and machine learning?")
        .context(context_chunks)
        .format(OutputFormat::Markdown)
        .min_confidence(0.7)
        .max_length(500)
        .build()?;
    
    let response = generator.generate(request).await?;
    
    // Comprehensive verification
    assert!(!response.content.is_empty());
    assert!(response.content.len() <= 500); // Respects length limit
    assert!(response.confidence_score >= 0.7); // Meets confidence threshold
    assert_eq!(response.format, OutputFormat::Markdown);
    assert!(!response.citations.is_empty()); // Has citations
    assert!(!response.validation_results.is_empty()); // Ran validation
    assert!(response.metrics.total_duration.as_millis() < 100); // Performance target
    
    // Verify citations point to correct sources
    let source_titles: Vec<String> = response.citations
        .iter()
        .map(|c| c.source.title.clone())
        .collect();
    
    assert!(source_titles.contains(&"Introduction to Artificial Intelligence".to_string()) || 
           source_titles.contains(&"What is Machine Learning?".to_string()));
    
    Ok(())
}