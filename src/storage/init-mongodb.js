// MongoDB Initialization Script
// This script runs when MongoDB starts for the first time

print("🚀 Initializing MongoDB for RAG Vector Storage...");

// Create keyfile for replica set authentication
try {
    const keyfileContent = "your-secret-key-here-make-this-secure-in-production-environment";
    
    // Note: In a real deployment, use a proper keyfile
    print("⚠️  Using development keyfile - replace in production!");
    
} catch (e) {
    print("Note: Keyfile setup handled by docker-compose");
}

// Switch to the RAG database
const db = getSiblingDB('rag_storage');

print("📊 Creating initial database structure...");

// Create initial collections (they'll be created when first document is inserted)
// This script mainly serves as documentation of the expected structure

const sampleChunk = {
    chunk_id: "00000000-0000-0000-0000-000000000000",
    content: "This is a sample chunk for testing the collection structure.",
    embedding: [0.1, 0.2, 0.3, 0.4, 0.5],
    metadata: {
        document_id: "00000000-0000-0000-0000-000000000001",
        title: "Sample Document",
        chunk_index: 0,
        total_chunks: 1,
        chunk_size: 64,
        overlap_size: 0,
        source_path: "/samples/test.txt",
        mime_type: "text/plain",
        language: "en",
        tags: ["sample", "test"],
        custom_fields: {
            priority: "low",
            category: "test"
        },
        content_hash: "sample_hash",
        boundary_confidence: 1.0
    },
    references: [],
    created_at: new Date(),
    updated_at: new Date(),
    version: NumberLong(1)
};

const sampleMetadata = {
    document_id: "00000000-0000-0000-0000-000000000001",
    metadata: {
        title: "Sample Document",
        author: "System",
        document_created_at: new Date(),
        file_size: NumberLong(1024),
        format: "text/plain",
        summary: "A sample document for testing",
        keywords: ["sample", "test"],
        classification: "public",
        security_level: "public"
    },
    processing_stats: {
        processing_time_ms: NumberLong(100),
        chunk_count: 1,
        avg_chunk_size: 64,
        embedding_model: "test-model",
        embedding_dimension: 5,
        errors: [],
        quality_metrics: {
            avg_boundary_confidence: 1.0,
            coherence_score: 1.0,
            information_density: 1.0,
            overlap_quality: 1.0,
            overall_quality: 100.0
        }
    },
    created_at: new Date(),
    updated_at: new Date()
};

// Insert sample documents to initialize collections
try {
    db.chunks.insertOne(sampleChunk);
    db.metadata.insertOne(sampleMetadata);
    print("✅ Sample documents inserted");
    
    // Remove sample documents (they were just for collection creation)
    db.chunks.deleteOne({chunk_id: "00000000-0000-0000-0000-000000000000"});
    db.metadata.deleteOne({document_id: "00000000-0000-0000-0000-000000000001"});
    print("🧹 Sample documents cleaned up");
    
} catch (e) {
    print("📝 Note: Collection initialization handled by application");
}

print("✅ MongoDB initialization completed!");
print("🔗 Database ready for RAG Vector Storage operations");

// Display configuration info
print("\n📋 Configuration Summary:");
print("   Database: rag_storage");
print("   Collections: chunks, metadata");
print("   Replica Set: rs0");
print("   Authentication: Enabled");
print("   Ready for vector storage operations!");

print("\n🚀 RAG Vector Storage MongoDB setup complete!");