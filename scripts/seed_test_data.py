#!/usr/bin/env python3
"""
Test Data Seeder
Seeds test databases with realistic data for testing
"""

import os
import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Database connections
try:
    import pymongo
    from pymongo import MongoClient
    HAS_MONGO = True
except ImportError:
    HAS_MONGO = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestDataSeeder:
    def __init__(self):
        self.mongo_url = os.environ.get('MONGODB_URL', 'mongodb://test-mongodb:27017/doc_rag_test')
        self.redis_url = os.environ.get('REDIS_URL', 'redis://test-redis:6379')
        
        # Initialize connections
        self.mongo_client = None
        self.redis_client = None
        self.db = None
        
    def connect_databases(self):
        """Establish database connections"""
        # MongoDB connection
        if HAS_MONGO:
            try:
                self.mongo_client = MongoClient(self.mongo_url, serverSelectionTimeoutMS=5000)
                self.db = self.mongo_client.get_default_database()
                # Test connection
                self.mongo_client.server_info()
                logger.info("✅ Connected to MongoDB")
            except Exception as e:
                logger.error(f"❌ Failed to connect to MongoDB: {e}")
                self.mongo_client = None
        else:
            logger.warning("⚠️  pymongo not available, skipping MongoDB seeding")
        
        # Redis connection
        if HAS_REDIS:
            try:
                self.redis_client = redis.from_url(self.redis_url, socket_timeout=5)
                self.redis_client.ping()
                logger.info("✅ Connected to Redis")
            except Exception as e:
                logger.error(f"❌ Failed to connect to Redis: {e}")
                self.redis_client = None
        else:
            logger.warning("⚠️  redis-py not available, skipping Redis seeding")
    
    def generate_test_documents(self) -> List[Dict[str, Any]]:
        """Generate realistic test documents"""
        documents = []
        
        # Sample document types and content
        doc_types = [
            {
                'name': 'Technical Specification',
                'content': 'This document outlines the technical requirements for the Doc-RAG system...',
                'size': random.randint(50000, 200000)
            },
            {
                'name': 'User Manual',
                'content': 'Welcome to Doc-RAG! This manual will guide you through the system features...',
                'size': random.randint(30000, 100000)
            },
            {
                'name': 'Research Paper',
                'content': 'Abstract: We present a novel approach to document retrieval using neural networks...',
                'size': random.randint(80000, 300000)
            },
            {
                'name': 'API Documentation',
                'content': 'The Doc-RAG API provides endpoints for document upload, processing, and querying...',
                'size': random.randint(25000, 75000)
            },
            {
                'name': 'Financial Report',
                'content': 'Executive Summary: This quarterly report shows significant improvements...',
                'size': random.randint(40000, 150000)
            }
        ]
        
        for i in range(20):  # Generate 20 test documents
            doc_type = random.choice(doc_types)
            doc_id = f"test_doc_{i+1:03d}_{random.randint(1000, 9999)}"
            
            document = {
                '_id': doc_id,
                'name': f"{doc_type['name']} {i+1}",
                'content': doc_type['content'],
                'size': doc_type['size'],
                'type': doc_type['name'].lower().replace(' ', '_'),
                'upload_time': datetime.utcnow() - timedelta(days=random.randint(1, 30)),
                'processed': True,
                'chunks_count': random.randint(5, 50),
                'embedding_model': 'test-embedder-v1.0',
                'metadata': {
                    'pages': random.randint(1, 100),
                    'language': 'en',
                    'format': 'pdf',
                    'version': f"1.{random.randint(0, 9)}"
                },
                'tags': random.sample(['important', 'draft', 'public', 'internal', 'archived'], 
                                    random.randint(1, 3))
            }
            
            documents.append(document)
        
        return documents
    
    def generate_test_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate test document chunks"""
        chunks = []
        
        for doc in documents:
            doc_id = doc['_id']
            chunks_count = doc['chunks_count']
            
            for chunk_idx in range(chunks_count):
                chunk = {
                    '_id': f"{doc_id}_chunk_{chunk_idx:03d}",
                    'document_id': doc_id,
                    'chunk_index': chunk_idx,
                    'content': f"This is chunk {chunk_idx + 1} of document {doc['name']}. " +
                              f"It contains relevant information about {doc['type']} topics. " * 
                              random.randint(2, 8),
                    'embedding': [random.uniform(-1, 1) for _ in range(768)],  # Mock embedding vector
                    'page_number': random.randint(1, doc['metadata']['pages']),
                    'start_char': chunk_idx * 500,
                    'end_char': (chunk_idx + 1) * 500 - 1,
                    'confidence': random.uniform(0.7, 1.0),
                    'created_at': doc['upload_time'] + timedelta(minutes=random.randint(1, 60))
                }
                
                chunks.append(chunk)
        
        return chunks
    
    def generate_test_queries(self) -> List[Dict[str, Any]]:
        """Generate test queries and responses"""
        queries = []
        
        sample_questions = [
            "What is the main objective of this system?",
            "How do I upload a document?",
            "What are the performance benchmarks?",
            "Explain the neural chunking approach",
            "What are the system requirements?",
            "How does the caching mechanism work?",
            "What is the accuracy of the embeddings?",
            "Describe the API endpoints available",
            "How is security implemented?",
            "What are the deployment options?"
        ]
        
        for i, question in enumerate(sample_questions):
            query_id = f"test_query_{i+1:03d}"
            
            query = {
                '_id': query_id,
                'question': question,
                'document_ids': random.sample([d['_id'] for d in self.documents], 
                                            random.randint(1, 3)),
                'answer': f"Based on the analysis of the documents, {question.lower()} " +
                         f"The system demonstrates high accuracy with confidence score {random.uniform(0.8, 0.99):.3f}.",
                'confidence': random.uniform(0.8, 0.99),
                'processing_time_ms': random.randint(50, 500),
                'timestamp': datetime.utcnow() - timedelta(hours=random.randint(1, 72)),
                'citations': [
                    {
                        'source': random.choice([d['name'] for d in self.documents]),
                        'page': random.randint(1, 20),
                        'relevance': random.uniform(0.7, 0.95),
                        'text': f"Relevant excerpt supporting the answer to: {question}"
                    }
                ],
                'metadata': {
                    'chunks_analyzed': random.randint(3, 15),
                    'cache_hit': random.choice([True, False]),
                    'model_version': 'test-v1.0'
                }
            }
            
            queries.append(query)
        
        return queries
    
    def seed_mongodb(self):
        """Seed MongoDB with test data"""
        if not self.mongo_client:
            logger.warning("⚠️  MongoDB not connected, skipping seeding")
            return
        
        logger.info("🌱 Seeding MongoDB with test data...")
        
        try:
            # Clear existing test data
            collections = ['documents', 'chunks', 'queries', 'cache']
            for collection_name in collections:
                collection = self.db[collection_name]
                result = collection.delete_many({'_id': {'$regex': '^test_'}})
                logger.info(f"   Cleared {result.deleted_count} existing test records from {collection_name}")
            
            # Generate test data
            self.documents = self.generate_test_documents()
            chunks = self.generate_test_chunks(self.documents)
            queries = self.generate_test_queries()
            
            # Insert documents
            if self.documents:
                result = self.db.documents.insert_many(self.documents)
                logger.info(f"   Inserted {len(result.inserted_ids)} test documents")
            
            # Insert chunks
            if chunks:
                # Insert in batches to avoid memory issues
                batch_size = 100
                inserted_count = 0
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    result = self.db.chunks.insert_many(batch)
                    inserted_count += len(result.inserted_ids)
                logger.info(f"   Inserted {inserted_count} test chunks")
            
            # Insert queries
            if queries:
                result = self.db.queries.insert_many(queries)
                logger.info(f"   Inserted {len(result.inserted_ids)} test queries")
            
            # Create indexes for better performance
            self.db.documents.create_index([('type', 1), ('processed', 1)])
            self.db.chunks.create_index([('document_id', 1), ('chunk_index', 1)])
            self.db.queries.create_index([('timestamp', -1)])
            logger.info("   Created database indexes")
            
            logger.info("✅ MongoDB seeding completed successfully")
            
        except Exception as e:
            logger.error(f"❌ MongoDB seeding failed: {e}")
    
    def seed_redis(self):
        """Seed Redis with test cache data"""
        if not self.redis_client:
            logger.warning("⚠️  Redis not connected, skipping seeding")
            return
        
        logger.info("🌱 Seeding Redis with test cache data...")
        
        try:
            # Clear existing test data
            pattern = "test:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"   Cleared {deleted} existing test keys from Redis")
            
            # Generate test cache entries
            cache_entries = []
            
            # Document metadata cache
            for doc in getattr(self, 'documents', []):
                key = f"test:doc:meta:{doc['_id']}"
                value = json.dumps({
                    'name': doc['name'],
                    'type': doc['type'],
                    'size': doc['size'],
                    'chunks_count': doc['chunks_count']
                })
                cache_entries.append((key, value))
            
            # Query results cache
            for i in range(10):
                key = f"test:query:result:{i}"
                value = json.dumps({
                    'answer': f'Cached answer for test query {i}',
                    'confidence': random.uniform(0.8, 0.99),
                    'cached_at': datetime.utcnow().isoformat()
                })
                cache_entries.append((key, value))
            
            # Performance metrics cache
            metrics_keys = ['response_time', 'cache_hit_rate', 'throughput', 'accuracy']
            for metric in metrics_keys:
                key = f"test:metrics:{metric}"
                value = json.dumps({
                    'value': random.uniform(0.5, 1.0) if metric == 'accuracy' else random.randint(10, 1000),
                    'timestamp': datetime.utcnow().isoformat(),
                    'unit': 'ms' if metric == 'response_time' else 'percent' if 'rate' in metric else 'requests/sec' if metric == 'throughput' else 'score'
                })
                cache_entries.append((key, value))
            
            # Insert cache entries
            pipe = self.redis_client.pipeline()
            for key, value in cache_entries:
                pipe.set(key, value, ex=3600)  # 1 hour expiry
            
            result = pipe.execute()
            success_count = sum(1 for r in result if r)
            logger.info(f"   Cached {success_count} test entries in Redis")
            
            # Set some additional test counters
            counters = {
                'test:stats:total_queries': random.randint(100, 1000),
                'test:stats:cache_hits': random.randint(50, 500),
                'test:stats:documents_processed': len(getattr(self, 'documents', [])),
            }
            
            for key, value in counters.items():
                self.redis_client.set(key, value)
            
            logger.info("✅ Redis seeding completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Redis seeding failed: {e}")
    
    def verify_seeded_data(self):
        """Verify that seeded data is accessible"""
        logger.info("🔍 Verifying seeded data...")
        
        # Verify MongoDB
        if self.mongo_client:
            try:
                doc_count = self.db.documents.count_documents({'_id': {'$regex': '^test_'}})
                chunk_count = self.db.chunks.count_documents({'_id': {'$regex': '^test_'}})
                query_count = self.db.queries.count_documents({'_id': {'$regex': '^test_'}})
                
                logger.info(f"   MongoDB: {doc_count} documents, {chunk_count} chunks, {query_count} queries")
            except Exception as e:
                logger.error(f"   MongoDB verification failed: {e}")
        
        # Verify Redis
        if self.redis_client:
            try:
                test_keys = len(self.redis_client.keys("test:*"))
                logger.info(f"   Redis: {test_keys} test keys")
            except Exception as e:
                logger.error(f"   Redis verification failed: {e}")
        
        logger.info("✅ Data verification completed")
    
    def run(self):
        """Main execution method"""
        logger.info("🚀 Starting test data seeding...")
        
        # Connect to databases
        self.connect_databases()
        
        # Seed databases
        self.seed_mongodb()
        self.seed_redis()
        
        # Verify seeded data
        self.verify_seeded_data()
        
        # Close connections
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("✅ Test data seeding completed successfully!")

def main():
    seeder = TestDataSeeder()
    seeder.run()

if __name__ == '__main__':
    main()