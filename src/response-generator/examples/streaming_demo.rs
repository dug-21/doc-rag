//! Streaming response demonstration

use response_generator::{
    ResponseGenerator, GenerationRequest, ContextChunk, Source, OutputFormat,
};
use std::collections::HashMap;
use tokio_stream::StreamExt;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("Response Generator - Streaming Demo");
    println!("==================================\n");

    let mut generator = ResponseGenerator::default();

    // Example 1: Basic Streaming
    println!("Example 1: Basic Streaming Response");
    println!("----------------------------------");
    
    let request = GenerationRequest::builder()
        .query("Explain the evolution of programming languages from assembly to modern high-level languages")
        .format(OutputFormat::Markdown)
        .build()?;

    println!("Query: {}", request.query);
    println!("Streaming response chunks:\n");

    let mut stream = generator.generate_stream(request).await?;
    let mut chunk_count = 0;
    let mut total_content = String::new();
    let start_time = std::time::Instant::now();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                chunk_count += 1;
                total_content.push_str(&chunk.content);
                
                println!("Chunk {}: {} chars at position {}", 
                        chunk_count, 
                        chunk.content.len(),
                        chunk.position);
                
                if let Some(confidence) = chunk.confidence {
                    println!("  Confidence: {:.2}", confidence);
                }
                
                println!("  Content: {}", 
                        if chunk.content.len() > 80 { 
                            format!("{}...", &chunk.content[..80])
                        } else { 
                            chunk.content.clone() 
                        });
                
                if chunk.is_final {
                    println!("  [FINAL CHUNK]");
                    break;
                }
                println!();
                
                // Simulate processing time
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            Err(e) => {
                println!("❌ Error in chunk {}: {}", chunk_count + 1, e);
                break;
            }
        }
    }
    
    let streaming_duration = start_time.elapsed();
    println!("\nStreaming Summary:");
    println!("  Total chunks: {}", chunk_count);
    println!("  Total content length: {} chars", total_content.len());
    println!("  Streaming duration: {}ms", streaming_duration.as_millis());
    println!("  Average chunk size: {:.1} chars", 
             total_content.len() as f64 / chunk_count as f64);

    // Example 2: Streaming with Context
    println!("\n\nExample 2: Streaming with Rich Context");
    println!("--------------------------------------");
    
    let context_chunks = vec![
        ContextChunk {
            content: "Microservices architecture is a method of developing software systems that tries to focus on building single-function modules with well-defined interfaces and operations.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Microservices Architecture Guide".to_string(),
                url: Some("https://microservices.io/".to_string()),
                document_type: "guide".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.9,
            position: Some(0),
            metadata: HashMap::new(),
        },
        ContextChunk {
            content: "Container orchestration platforms like Kubernetes provide automated deployment, scaling, and management of containerized applications.".to_string(),
            source: Source {
                id: Uuid::new_v4(),
                title: "Kubernetes Documentation".to_string(),
                url: Some("https://kubernetes.io/docs/".to_string()),
                document_type: "documentation".to_string(),
                metadata: HashMap::new(),
            },
            relevance_score: 0.85,
            position: Some(1),
            metadata: HashMap::new(),
        },
    ];

    let context_request = GenerationRequest::builder()
        .query("How do microservices and containerization work together in modern software architecture?")
        .context(context_chunks)
        .format(OutputFormat::Markdown)
        .build()?;

    println!("Query: {}", context_request.query);
    println!("Context sources: {} chunks provided", context_request.context.len());
    println!("Streaming response:\n");

    let mut context_stream = generator.generate_stream(context_request).await?;
    let mut context_chunk_count = 0;
    let context_start_time = std::time::Instant::now();
    let mut accumulated_content = String::new();

    while let Some(chunk_result) = context_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                context_chunk_count += 1;
                accumulated_content.push_str(&chunk.content);
                
                print!("Chunk {}: ", context_chunk_count);
                if let Some(confidence) = chunk.confidence {
                    print!("[Conf: {:.2}] ", confidence);
                }
                print!("{} chars - ", chunk.content.len());
                
                // Show first 60 characters
                let preview = if chunk.content.len() > 60 {
                    format!("{}...", &chunk.content[..60])
                } else {
                    chunk.content.clone()
                };
                println!("{}", preview.replace('\n', " "));
                
                if chunk.is_final {
                    println!("  ✅ Final chunk received");
                    break;
                }
                
                // Limit to first 5 chunks for demo
                if context_chunk_count >= 5 {
                    println!("  (Demo limited to 5 chunks)");
                    break;
                }
                
                // Simulate real-time processing
                tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
            }
            Err(e) => {
                println!("❌ Streaming error: {}", e);
                break;
            }
        }
    }
    
    let context_streaming_duration = context_start_time.elapsed();
    println!("\nContext Streaming Summary:");
    println!("  Chunks processed: {}", context_chunk_count);
    println!("  Content accumulated: {} chars", accumulated_content.len());
    println!("  Duration: {}ms", context_streaming_duration.as_millis());

    // Example 3: Streaming Performance Comparison
    println!("\n\nExample 3: Streaming vs Non-Streaming Performance");
    println!("-------------------------------------------------");
    
    let perf_request = GenerationRequest::builder()
        .query("Explain the principles of distributed databases and their advantages over traditional databases")
        .format(OutputFormat::Text)
        .build()?;

    // Non-streaming generation
    let non_stream_start = std::time::Instant::now();
    let regular_response = generator.generate(perf_request.clone()).await?;
    let non_stream_duration = non_stream_start.elapsed();
    
    println!("Non-streaming generation:");
    println!("  Duration: {}ms", non_stream_duration.as_millis());
    println!("  Response length: {} chars", regular_response.content.len());
    println!("  Time to first byte: {}ms (same as total)", non_stream_duration.as_millis());

    // Streaming generation
    let stream_start = std::time::Instant::now();
    let mut perf_stream = generator.generate_stream(perf_request).await?;
    let mut first_byte_time = None;
    let mut stream_chunk_count = 0;
    let mut stream_total_length = 0;

    while let Some(chunk_result) = perf_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                if first_byte_time.is_none() {
                    first_byte_time = Some(stream_start.elapsed());
                }
                
                stream_chunk_count += 1;
                stream_total_length += chunk.content.len();
                
                if chunk.is_final || stream_chunk_count >= 3 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    
    let stream_duration = stream_start.elapsed();
    let ttfb = first_byte_time.unwrap_or(stream_duration);
    
    println!("\nStreaming generation:");
    println!("  Total duration: {}ms", stream_duration.as_millis());
    println!("  Time to first byte: {}ms", ttfb.as_millis());
    println!("  Chunks received: {}", stream_chunk_count);
    println!("  Content length: {} chars", stream_total_length);
    
    // Calculate improvement
    let ttfb_improvement = ((non_stream_duration.as_millis() as f64 - ttfb.as_millis() as f64) 
                           / non_stream_duration.as_millis() as f64) * 100.0;
    
    println!("\nPerformance Comparison:");
    println!("  Time-to-first-byte improvement: {:.1}%", ttfb_improvement);
    println!("  Streaming allows progressive content display");
    println!("  Better user experience for long responses");

    // Example 4: Error Handling in Streaming
    println!("\n\nExample 4: Streaming Error Handling");
    println!("-----------------------------------");
    
    let error_request = GenerationRequest::builder()
        .query("Generate an extremely long response about every possible topic")
        .max_length(50) // Very small limit to potentially trigger errors
        .min_confidence(0.95) // High confidence requirement
        .build()?;

    println!("Testing error handling with restrictive constraints...");
    
    let mut error_stream = generator.generate_stream(error_request).await?;
    let mut error_count = 0;
    let mut success_count = 0;

    while let Some(chunk_result) = error_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                success_count += 1;
                println!("✅ Chunk {}: {} chars", success_count, chunk.content.len());
                
                if chunk.is_final || success_count >= 3 {
                    break;
                }
            }
            Err(e) => {
                error_count += 1;
                println!("❌ Error in chunk {}: {}", error_count, e);
                
                if error_count >= 2 {
                    println!("  Stopping after multiple errors");
                    break;
                }
            }
        }
    }
    
    println!("\nError Handling Summary:");
    println!("  Successful chunks: {}", success_count);
    println!("  Errors encountered: {}", error_count);
    println!("  Graceful degradation: {}", if success_count > 0 { "✅ Yes" } else { "❌ No" });

    println!("\n🚀 Streaming demonstration completed!");
    println!("Key benefits demonstrated:");
    println!("  • Progressive content delivery");
    println!("  • Reduced time-to-first-byte");
    println!("  • Better user experience for long responses");
    println!("  • Graceful error handling");
    println!("  • Real-time confidence feedback");
    
    Ok(())
}