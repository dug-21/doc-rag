//! Advanced features demonstration for the response generator

use response_generator::{
    Config, ResponseGenerator, GenerationRequest, ContextChunk, Source, OutputFormat,
    citation::{CitationStyle, CitationConfig},
    formatter::FormatterConfig,
    validator::ValidationConfig,
};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging with detailed output
    tracing_subscriber::fmt()
        .with_env_filter("response_generator=debug")
        .init();

    println!("Response Generator - Advanced Features Demo");
    println!("==========================================\n");

    // Example 1: Custom Configuration
    println!("Example 1: Custom Configuration");
    println!("-------------------------------");
    
    let custom_config = Config::builder()
        .max_response_length(1000)
        .default_confidence_threshold(0.85)
        .build();

    let custom_generator = ResponseGenerator::new(custom_config).await;
    
    let request = GenerationRequest::builder()
        .query("Explain quantum computing comprehensively")
        .format(OutputFormat::Markdown)
        .min_confidence(0.8)
        .build()?;

    let response = custom_generator.generate(request).await?;
    
    println!("Custom configured response:");
    println!("Length: {} chars (limit: 1000)", response.content.len());
    println!("Confidence: {:.2} (threshold: 0.85)", response.confidence_score);
    println!("Generation time: {}ms\n", response.metrics.total_duration.as_millis());

    // Example 2: Multiple Context Sources with Rich Metadata
    println!("Example 2: Rich Context with Multiple Sources");
    println!("---------------------------------------------");
    
    let context_chunks = vec![
        ContextChunk {
            content: "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Introduction to Machine Learning".to_string(),
                url: Some("https://www.ibm.com/topics/machine-learning".to_string()),
                document_type: "article".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("publish_date".to_string(), "2023-01-15T10:00:00Z".to_string());
                    meta.insert("author".to_string(), "IBM Research Team".to_string());
                    meta.insert("credibility_score".to_string(), "0.9".to_string());
                    meta.insert("peer_reviewed".to_string(), "false".to_string());
                    meta
                },
            },
            relevance_score: 0.95,
            position: Some(0),
            metadata: HashMap::new(),
        },
        ContextChunk {
            content: "Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Deep Learning Fundamentals".to_string(),
                url: Some("https://www.tensorflow.org/learn".to_string()),
                document_type: "documentation".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("publish_date".to_string(), "2023-03-20T14:30:00Z".to_string());
                    meta.insert("author".to_string(), "TensorFlow Team".to_string());
                    meta.insert("credibility_score".to_string(), "0.95".to_string());
                    meta.insert("citation_count".to_string(), "1500".to_string());
                    meta
                },
            },
            relevance_score: 0.88,
            position: Some(1),
            metadata: HashMap::new(),
        },
        ContextChunk {
            content: "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Natural Language Processing Overview".to_string(),
                url: Some("https://research.stanford.edu/nlp".to_string()),
                document_type: "academic".to_string(),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("publish_date".to_string(), "2023-02-10T09:15:00Z".to_string());
                    meta.insert("author".to_string(), "Stanford AI Lab".to_string());
                    meta.insert("credibility_score".to_string(), "0.98".to_string());
                    meta.insert("peer_reviewed".to_string(), "true".to_string());
                    meta
                },
            },
            relevance_score: 0.75,
            position: Some(2),
            metadata: HashMap::new(),
        },
    ];

    let generator = ResponseGenerator::new(Config::default()).await;
    let request = GenerationRequest::builder()
        .query("What are the main branches of artificial intelligence and how do they relate?")
        .context(context_chunks)
        .format(OutputFormat::Markdown)
        .max_length(800)
        .min_confidence(0.7)
        .metadata("topic".to_string(), "AI_overview".to_string())
        .build()?;

    let response = generator.generate(request).await?;
    
    println!("Multi-source response:");
    println!("{}", response.content);
    println!("\nCitation Analysis:");
    for (i, citation) in response.citations.iter().enumerate() {
        println!("  {}. {} (Confidence: {:.2}, Type: {:?})", 
                 i + 1, 
                 citation.source.title, 
                 citation.confidence,
                 citation.citation_type);
        if let Some(url) = &citation.source.url {
            println!("     URL: {}", url);
        }
    }
    println!();

    // Example 3: Performance Analysis
    println!("Example 3: Performance Analysis");
    println!("-------------------------------");
    
    let perf_request = GenerationRequest::builder()
        .query("Explain the principles of distributed computing")
        .format(OutputFormat::Json)
        .build()?;

    let start_time = std::time::Instant::now();
    let perf_response = generator.generate(perf_request).await?;
    let total_time = start_time.elapsed();

    println!("Performance Metrics:");
    println!("  Total time: {}ms", total_time.as_millis());
    println!("  Internal total: {}ms", perf_response.metrics.total_duration.as_millis());
    println!("  Validation time: {}ms", perf_response.metrics.validation_duration.as_millis());
    println!("  Citation processing: {}ms", perf_response.metrics.citation_duration.as_millis());
    println!("  Formatting time: {}ms", perf_response.metrics.formatting_duration.as_millis());
    println!("  Validation passes: {}", perf_response.metrics.validation_passes);
    println!("  Sources used: {}", perf_response.metrics.sources_used);
    println!("  Response length: {} chars", perf_response.metrics.response_length);
    
    // Performance target check
    if total_time.as_millis() <= 100 {
        println!("  ✅ Performance target met (<100ms)");
    } else {
        println!("  ⚠️  Performance target missed (>100ms)");
    }
    println!();

    // Example 4: Validation Layer Analysis
    println!("Example 4: Validation Layer Analysis");
    println!("------------------------------------");
    
    let validation_request = GenerationRequest::builder()
        .query("All experts always agree that AI will never cause any problems")
        .format(OutputFormat::Text)
        .build()?;

    let validation_response = generator.generate(validation_request).await?;
    
    println!("Query with potential issues: '{}'", "All experts always agree that AI will never cause any problems");
    println!("Validation Results:");
    for result in &validation_response.validation_results {
        let status = if result.passed { "✅ PASS" } else { "❌ FAIL" };
        println!("  {} {}: {:.2} confidence in {}ms", 
                 status,
                 result.layer_name, 
                 result.confidence,
                 result.processing_time.as_millis());
        
        for finding in &result.findings {
            println!("    - {:?}: {}", finding.severity, finding.message);
            if let Some(suggestion) = &finding.suggestion {
                println!("      Suggestion: {}", suggestion);
            }
        }
    }
    println!();

    // Example 5: Different Output Formats Comparison
    println!("Example 5: Output Format Comparison");
    println!("-----------------------------------");
    
    let formats_to_test = vec![
        ("JSON", OutputFormat::Json),
        ("Markdown", OutputFormat::Markdown),
        ("HTML", OutputFormat::Html),
        ("XML", OutputFormat::Xml),
        ("YAML", OutputFormat::Yaml),
    ];

    for (format_name, format) in formats_to_test {
        let format_request = GenerationRequest::builder()
            .query("What is containerization in software development?")
            .format(format)
            .max_length(300)
            .build()?;

        let format_response = generator.generate(format_request).await?;
        
        println!("{} Format ({} chars):", format_name, format_response.content.len());
        let preview = if format_response.content.len() > 100 {
            format!("{}...", &format_response.content[..100])
        } else {
            format_response.content.clone()
        };
        println!("  Preview: {}", preview.replace('\n', "\\n"));
        println!("  Content-Type: {}", 
                 match format_response.format {
                     OutputFormat::Json => "application/json",
                     OutputFormat::Markdown => "text/markdown", 
                     OutputFormat::Html => "text/html",
                     OutputFormat::Xml => "application/xml",
                     OutputFormat::Yaml => "application/x-yaml",
                     _ => "text/plain",
                 });
        println!();
    }

    // Example 6: Error Handling and Recovery
    println!("Example 6: Error Handling Demonstration");
    println!("--------------------------------------");
    
    // Test with very high confidence threshold
    let high_confidence_request = GenerationRequest::builder()
        .query("What is the exact number of atoms in the universe?")
        .min_confidence(0.99) // Very high threshold
        .build()?;

    match generator.generate(high_confidence_request).await {
        Ok(response) => {
            println!("High confidence request succeeded:");
            println!("  Confidence: {:.3}", response.confidence_score);
            println!("  Warnings: {:?}", response.warnings);
        }
        Err(e) => {
            println!("High confidence request failed (as expected): {}", e);
            println!("  Error type: {:?}", e);
            println!("  Recovery strategy: {:?}", e.recovery_strategy());
        }
    }

    // Test with empty query
    let empty_request = GenerationRequest::builder()
        .query("")
        .build()?;

    match generator.generate(empty_request).await {
        Ok(response) => {
            println!("Empty query handled gracefully:");
            println!("  Response length: {} chars", response.content.len());
            println!("  Warnings: {:?}", response.warnings);
        }
        Err(e) => {
            println!("Empty query failed: {}", e);
        }
    }

    println!("\nAdvanced features demonstration completed!");
    println!("✅ All examples executed successfully");
    
    Ok(())
}