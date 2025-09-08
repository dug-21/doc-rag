//! Neural boundary detection demo using ruv-fann

use chunker::{DocumentChunker, BoundaryDetector};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Neural Boundary Detection Demo using ruv-fann");
    println!("================================================");
    
    // Test boundary detector directly
    println!("\n1. Testing BoundaryDetector...");
    let mut detector = BoundaryDetector::new()?;
    
    let sample_text = "This is the first paragraph with some content. It contains multiple sentences for analysis.\n\nThis is the second paragraph with different content. The neural network should detect this boundary.\n\nFinally, this is the third paragraph which concludes our test.";
    
    let boundaries = detector.detect_boundaries(sample_text)?;
    println!("Detector found {} boundaries at positions: {:?}", 
        boundaries.len(), 
        boundaries.iter().map(|b| b.position).collect::<Vec<_>>()
    );
    
    // Test complete document chunking
    println!("\n2. Testing DocumentChunker...");
    let mut chunker = DocumentChunker::new(150, 20)?;
    
    let document = "Document chunking is the process of breaking down large documents into smaller, manageable pieces. This is essential for efficient document processing and retrieval.

Neural networks can enhance this process by learning to identify semantic boundaries. Unlike simple rule-based approaches, neural networks can understand context and meaning.

The ruv-fann library provides a powerful Rust implementation of neural networks. It supports various activation functions and training algorithms, making it ideal for boundary detection tasks.

In this example, we demonstrate how to use neural networks for intelligent document chunking. The system learns to identify natural break points based on content analysis and semantic understanding.";

    let chunks = chunker.chunk_document(document)?;
    
    println!("Created {} chunks using boundary detection:", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!("\nChunk {}: ({}-{}, {} chars, {} words)", 
                i + 1, 
                chunk.metadata.start_offset, 
                chunk.metadata.end_offset,
                chunk.content_length(),
                chunk.word_count());
        
        let preview = if chunk.content.len() > 100 {
            format!("{}...", &chunk.content[..100])
        } else {
            chunk.content.clone()
        };
        println!("Content: \"{}\"", preview.replace('\n', " "));
    }
    
    println!("\n✅ Demo completed successfully!");
    
    Ok(())
}