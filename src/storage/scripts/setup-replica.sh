#!/bin/bash
# MongoDB Replica Set Setup Script

set -e

echo "Setting up MongoDB replica set..."

# Wait for MongoDB to be fully ready
sleep 15

# Connect to MongoDB and initialize replica set
mongosh --host mongodb:27017 -u admin -p ${MONGODB_PASSWORD:-password123} --authenticationDatabase admin --eval "
try {
    var config = {
        '_id': 'rs0',
        'version': 1,
        'members': [
            {
                '_id': 0,
                'host': 'mongodb:27017',
                'priority': 1
            }
        ]
    };
    
    var result = rs.initiate(config);
    print('Replica set initiate result:', JSON.stringify(result));
    
    if (result.ok === 1) {
        print('Replica set initialized successfully');
    } else {
        print('Failed to initialize replica set:', result);
    }
} catch (e) {
    if (e.message.includes('already initialized')) {
        print('Replica set already initialized');
    } else {
        print('Error initializing replica set:', e);
        throw e;
    }
}
"

# Wait for replica set to be ready
echo "Waiting for replica set to be ready..."
sleep 10

# Create application user and database
mongosh --host mongodb:27017 -u admin -p ${MONGODB_PASSWORD:-password123} --authenticationDatabase admin --eval "
try {
    // Switch to rag_storage database
    db = db.getSiblingDB('rag_storage');
    
    // Create application user
    db.createUser({
        user: 'rag_user',
        pwd: '${RAG_USER_PASSWORD:-rag123}',
        roles: [
            { role: 'readWrite', db: 'rag_storage' },
            { role: 'dbAdmin', db: 'rag_storage' }
        ]
    });
    
    print('Application user created successfully');
    
    // Create collections with schema validation
    db.createCollection('chunks', {
        validator: {
            \$jsonSchema: {
                bsonType: 'object',
                required: ['chunk_id', 'content', 'metadata', 'created_at', 'updated_at', 'version'],
                properties: {
                    chunk_id: { bsonType: 'string' },
                    content: { bsonType: 'string' },
                    embedding: { 
                        bsonType: 'array',
                        items: { bsonType: 'double' }
                    },
                    metadata: {
                        bsonType: 'object',
                        required: ['document_id', 'title', 'chunk_index', 'total_chunks'],
                        properties: {
                            document_id: { bsonType: 'string' },
                            title: { bsonType: 'string' },
                            chunk_index: { bsonType: 'int' },
                            total_chunks: { bsonType: 'int' }
                        }
                    },
                    created_at: { bsonType: 'date' },
                    updated_at: { bsonType: 'date' },
                    version: { bsonType: 'long' }
                }
            }
        }
    });
    
    db.createCollection('metadata', {
        validator: {
            \$jsonSchema: {
                bsonType: 'object',
                required: ['document_id', 'metadata', 'processing_stats', 'created_at', 'updated_at'],
                properties: {
                    document_id: { bsonType: 'string' },
                    created_at: { bsonType: 'date' },
                    updated_at: { bsonType: 'date' }
                }
            }
        }
    });
    
    print('Collections created with schema validation');
    
    // Create vector search index (Atlas only - commenting out for local development)
    /*
    db.chunks.createSearchIndex(
        'vector_search_idx',
        {
            definition: {
                mappings: {
                    dynamic: true,
                    fields: {
                        embedding: {
                            type: 'knnVector',
                            dimensions: 384,
                            similarity: 'cosine'
                        },
                        'metadata.document_id': { type: 'token' },
                        'metadata.tags': { type: 'token' },
                        content: { type: 'string' }
                    }
                }
            }
        }
    );
    */
    
    // Create regular indexes for efficient querying
    db.chunks.createIndex({ 'chunk_id': 1 }, { unique: true });
    db.chunks.createIndex({ 'metadata.document_id': 1 });
    db.chunks.createIndex({ 'metadata.chunk_index': 1 });
    db.chunks.createIndex({ 'metadata.tags': 1 });
    db.chunks.createIndex({ 'created_at': -1 });
    db.chunks.createIndex({ 'updated_at': -1 });
    
    // Text search index
    db.chunks.createIndex(
        { 
            'content': 'text', 
            'metadata.title': 'text',
            'metadata.tags': 'text' 
        },
        {
            name: 'text_search_idx',
            weights: {
                'content': 10,
                'metadata.title': 5,
                'metadata.tags': 3
            }
        }
    );
    
    // Metadata collection indexes
    db.metadata.createIndex({ 'document_id': 1 }, { unique: true });
    db.metadata.createIndex({ 'created_at': -1 });
    
    print('Indexes created successfully');
    
} catch (e) {
    if (e.message.includes('already exists')) {
        print('User or collection already exists');
    } else {
        print('Error setting up database:', e);
        throw e;
    }
}
"

echo "MongoDB replica set setup completed successfully!"