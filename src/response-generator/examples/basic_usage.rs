//! Basic usage example for the response generator

use response_generator::{
    Config, ResponseGenerator, GenerationRequest, ContextChunk, Source, OutputFormat,
};
use std::collections::HashMap;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Response Generator - Basic Usage Example");
    println!("========================================\n");

    // Create a response generator with default configuration
    let config = Config::default();
    let generator = ResponseGenerator::new(config);

    // Example 1: Simple query without context
    println!("Example 1: Simple Query");
    println!("-----------------------");
    
    let request = GenerationRequest::builder()
        .query("What is artificial intelligence?")
        .format(OutputFormat::Markdown)
        .build()?;

    let response = generator.generate(request).await?;
    
    println!("Query: What is artificial intelligence?");
    println!("Response:");
    println!("{}", response.content);
    println!("Confidence: {:.2}", response.confidence_score);
    println!("Generation time: {}ms\n", response.metrics.total_duration.as_millis());

    // Example 2: Query with context
    println!("Example 2: Query with Context");
    println!("-----------------------------");
    
    let context_chunk = ContextChunk {
        content: "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.".to_string(),
        source: Source {
            id: Uuid::new_v4(),
            title: "AI Overview - Encyclopedia".to_string(),
            url: Some("https://example.com/ai-overview".to_string()),
            document_type: "encyclopedia".to_string(),
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("publish_date".to_string(), "2023-01-15".to_string());
                meta.insert("author".to_string(), "AI Research Team".to_string());
                meta
            },
        },
        relevance_score: 0.9,
        position: Some(0),
        metadata: HashMap::new(),
    };

    let request = GenerationRequest::builder()
        .query("What are the key characteristics of AI?")
        .add_context(context_chunk)
        .format(OutputFormat::Markdown)
        .min_confidence(0.7)
        .build()?;

    let response = generator.generate(request).await?;
    
    println!("Query: What are the key characteristics of AI?");
    println!("Response:");
    println!("{}", response.content);
    println!("Confidence: {:.2}", response.confidence_score);
    println!("Citations: {}", response.citations.len());
    println!("Generation time: {}ms\n", response.metrics.total_duration.as_millis());

    // Example 3: Multiple output formats
    println!("Example 3: Multiple Output Formats");
    println!("----------------------------------");
    
    let formats = vec![
        ("JSON", OutputFormat::Json),
        ("Plain Text", OutputFormat::Text),
        ("HTML", OutputFormat::Html),
    ];

    for (format_name, format) in formats {
        let request = GenerationRequest::builder()
            .query("Explain machine learning briefly")
            .format(format)
            .max_length(200)
            .build()?;

        let response = generator.generate(request).await?;
        
        println!("Format: {}", format_name);
        println!("Response (truncated):");
        let truncated = if response.content.len() > 150 {
            format!("{}...", &response.content[..150])
        } else {
            response.content.clone()
        };
        println!("{}\n", truncated);
    }

    // Example 4: Streaming response
    println!("Example 4: Streaming Response");
    println!("-----------------------------");
    
    let request = GenerationRequest::builder()
        .query("Explain the benefits of cloud computing in detail")
        .format(OutputFormat::Markdown)
        .build()?;

    println!("Query: Explain the benefits of cloud computing in detail");
    println!("Streaming response chunks:");
    
    let mut stream = generator.generate_stream(request).await?;
    let mut chunk_count = 0;
    
    use tokio_stream::StreamExt;
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                chunk_count += 1;
                println!("Chunk {}: {} characters", chunk_count, chunk.content.len());
                if let Some(confidence) = chunk.confidence {
                    println!("  Confidence: {:.2}", confidence);
                }
            }
            Err(e) => {
                println!("Error in chunk: {}", e);
                break;
            }
        }
        
        if chunk_count >= 3 {
            println!("  (truncating for example)");
            break;
        }
    }

    // Example 5: Custom configuration
    println!("\nExample 5: Custom Configuration");
    println!("-------------------------------");
    
    let custom_config = Config::builder()
        .max_response_length(300)
        .default_confidence_threshold(0.8)
        .build();

    let custom_generator = ResponseGenerator::new(custom_config);
    
    let request = GenerationRequest::builder()
        .query("What is the future of artificial intelligence?")
        .format(OutputFormat::Markdown)
        .build()?;

    let response = custom_generator.generate(request).await?;
    
    println!("Query: What is the future of artificial intelligence?");
    println!("Response (max 300 chars):");
    println!("{}", response.content);
    println!("Length: {} characters", response.content.len());
    println!("Confidence: {:.2}", response.confidence_score);
    
    // Show validation results
    if !response.validation_results.is_empty() {
        println!("\nValidation Results:");
        for validation in &response.validation_results {
            println!("  {}: {} (confidence: {:.2})", 
                    validation.layer_name, 
                    if validation.passed { "PASSED" } else { "FAILED" }, 
                    validation.confidence);
        }
    }

    println!("\nExample completed successfully!");
    Ok(())
}