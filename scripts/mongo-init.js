// MongoDB initialization script for doc-rag
print('Initializing doc-rag MongoDB database...');

// Switch to the doc_rag database
db = db.getSiblingDB('doc_rag');

// Create collections with validation
db.createCollection("documents", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["id", "content", "metadata"],
      properties: {
        id: { bsonType: "string" },
        content: { bsonType: "string" },
        metadata: { bsonType: "object" },
        chunks: { bsonType: "array" },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" }
      }
    }
  }
});

db.createCollection("embeddings", {
  validator: {
    $jsonSchema: {
      bsonType: "object", 
      required: ["document_id", "chunk_id", "vector"],
      properties: {
        document_id: { bsonType: "string" },
        chunk_id: { bsonType: "string" },
        vector: { bsonType: "array" },
        metadata: { bsonType: "object" }
      }
    }
  }
});

// Create indexes for performance
db.documents.createIndex({ "id": 1 }, { unique: true });
db.documents.createIndex({ "metadata.type": 1 });
db.documents.createIndex({ "created_at": 1 });

db.embeddings.createIndex({ "document_id": 1 });
db.embeddings.createIndex({ "chunk_id": 1 });

print('doc-rag MongoDB initialization complete');