//! Standalone test for neural classifier functionality
//! Tests ruv-fann integration without DAA dependencies

use symbolic::neural_classifier::{NeuralClassifierSystem, ClassificationResult};

#[tokio::test]
async fn test_neural_classifier_initialization() {
    let mut classifier = NeuralClassifierSystem::new();
    let result = classifier.initialize().await;
    assert!(result.is_ok(), "Neural classifier should initialize successfully");
}

#[tokio::test]
async fn test_neural_classification_performance() {
    let mut classifier = NeuralClassifierSystem::new();
    classifier.initialize().await.unwrap();
    
    let start = std::time::Instant::now();
    let result = classifier.classify_query("What are the encryption requirements?").await;
    let elapsed = start.elapsed();
    
    assert!(result.is_ok(), "Query classification should succeed");
    let classification = result.unwrap();
    
    // CONSTRAINT-003: Must be <10ms
    assert!(elapsed.as_millis() < 10, "Classification took {:?}, exceeds 10ms constraint", elapsed);
    assert!(classification.confidence > 0.0, "Classification should have confidence > 0");
    assert_eq!(classification.features.len(), 50, "Should have 50 features for query classification");
    
    println!("Neural classification completed in {:?} with confidence {:.2}", 
             elapsed, classification.confidence);
}

#[tokio::test]
async fn test_document_classification() {
    let mut classifier = NeuralClassifierSystem::new();
    classifier.initialize().await.unwrap();
    
    let doc_text = "PCI DSS Payment Card Industry Data Security Standard";
    let result = classifier.classify_document(doc_text).await;
    
    assert!(result.is_ok(), "Document classification should succeed");
    let classification = result.unwrap();
    
    assert!(classification.inference_time_ms < 10, "Inference should be <10ms");
    assert!(classification.confidence > 0.0, "Should have confidence > 0");
    assert_eq!(classification.features.len(), 100, "Should have 100 features for document classification");
    
    println!("Document classified as '{}' with confidence {:.2}", 
             classification.classification, classification.confidence);
}

#[tokio::test]
async fn test_section_classification() {
    let mut classifier = NeuralClassifierSystem::new();
    classifier.initialize().await.unwrap();
    
    let section_text = "Requirements for encryption of cardholder data";
    let result = classifier.classify_section(section_text).await;
    
    assert!(result.is_ok(), "Section classification should succeed");
    let classification = result.unwrap();
    
    assert!(classification.inference_time_ms < 10, "Inference should be <10ms");
    assert_eq!(classification.features.len(), 80, "Should have 80 features for section classification");
    
    println!("Section classified as '{}' with confidence {:.2}", 
             classification.classification, classification.confidence);
}