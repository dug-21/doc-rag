-- Database initialization script for Doc-RAG system
-- This script sets up the basic schema and indexes

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create custom types
CREATE TYPE document_status AS ENUM ('pending', 'processing', 'indexed', 'error', 'archived');
CREATE TYPE chunk_type AS ENUM ('text', 'code', 'table', 'image', 'metadata');
CREATE TYPE processing_status AS ENUM ('queued', 'processing', 'completed', 'failed', 'retrying');

-- Documents table - stores document metadata
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(512) NOT NULL,
    original_name VARCHAR(512) NOT NULL,
    mime_type VARCHAR(128) NOT NULL,
    file_size BIGINT NOT NULL,
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    storage_path VARCHAR(1024) NOT NULL,
    
    -- Content metadata
    title VARCHAR(512),
    author VARCHAR(256),
    language VARCHAR(10) DEFAULT 'en',
    page_count INTEGER,
    word_count INTEGER,
    
    -- Processing metadata
    status document_status DEFAULT 'pending',
    processing_started_at TIMESTAMP WITH TIME ZONE,
    processing_completed_at TIMESTAMP WITH TIME ZONE,
    processing_error TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    indexed_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata JSON
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT valid_file_size CHECK (file_size > 0),
    CONSTRAINT valid_page_count CHECK (page_count IS NULL OR page_count > 0),
    CONSTRAINT valid_word_count CHECK (word_count IS NULL OR word_count >= 0)
);

-- Chunks table - stores document chunks
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    
    -- Chunk content
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    chunk_type chunk_type DEFAULT 'text',
    
    -- Position and structure
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    paragraph_index INTEGER,
    sentence_index INTEGER,
    
    -- Size metrics
    char_count INTEGER NOT NULL,
    word_count INTEGER NOT NULL,
    token_count INTEGER,
    
    -- Embeddings metadata
    embedding_model VARCHAR(128),
    embedding_dimensions INTEGER,
    embedding_created_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Additional metadata
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT valid_chunk_index CHECK (chunk_index >= 0),
    CONSTRAINT valid_char_count CHECK (char_count > 0),
    CONSTRAINT valid_word_count CHECK (word_count > 0),
    CONSTRAINT valid_token_count CHECK (token_count IS NULL OR token_count > 0),
    CONSTRAINT valid_page_number CHECK (page_number IS NULL OR page_number > 0),
    CONSTRAINT valid_embedding_dims CHECK (embedding_dimensions IS NULL OR embedding_dimensions > 0),
    
    UNIQUE(document_id, chunk_index)
);

-- Processing jobs table - tracks async processing
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(64) NOT NULL, -- 'chunk', 'embed', 'index', etc.
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    
    -- Job status
    status processing_status DEFAULT 'queued',
    priority INTEGER DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Timing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Error handling
    error_message TEXT,
    error_details JSONB,
    
    -- Job configuration
    parameters JSONB DEFAULT '{}',
    
    CONSTRAINT valid_priority CHECK (priority >= 0),
    CONSTRAINT valid_retry_count CHECK (retry_count >= 0),
    CONSTRAINT valid_max_retries CHECK (max_retries >= 0)
);

-- Search queries table - stores search history and analytics
CREATE TABLE search_queries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64) NOT NULL,
    
    -- Query metadata
    query_type VARCHAR(32) DEFAULT 'semantic', -- 'semantic', 'keyword', 'hybrid'
    language VARCHAR(10) DEFAULT 'en',
    
    -- Search parameters
    top_k INTEGER DEFAULT 10,
    similarity_threshold REAL DEFAULT 0.7,
    filters JSONB DEFAULT '{}',
    
    -- Results metadata
    total_results INTEGER,
    processing_time_ms INTEGER,
    
    -- Analytics
    user_session VARCHAR(128),
    user_id UUID,
    ip_address INET,
    user_agent TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_top_k CHECK (top_k > 0 AND top_k <= 1000),
    CONSTRAINT valid_similarity_threshold CHECK (similarity_threshold >= 0 AND similarity_threshold <= 1),
    CONSTRAINT valid_total_results CHECK (total_results IS NULL OR total_results >= 0),
    CONSTRAINT valid_processing_time CHECK (processing_time_ms IS NULL OR processing_time_ms >= 0)
);

-- System configuration table
CREATE TABLE system_config (
    key VARCHAR(128) PRIMARY KEY,
    value JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for optimal performance
-- Documents indexes
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX idx_documents_file_hash ON documents(file_hash);
CREATE INDEX idx_documents_mime_type ON documents(mime_type);
CREATE INDEX idx_documents_metadata ON documents USING gin(metadata);

-- Chunks indexes
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_type ON chunks(chunk_type);
CREATE INDEX idx_chunks_hash ON chunks(content_hash);
CREATE INDEX idx_chunks_created_at ON chunks(created_at DESC);
CREATE INDEX idx_chunks_metadata ON chunks USING gin(metadata);
CREATE INDEX idx_chunks_content_trgm ON chunks USING gin(content gin_trgm_ops);

-- Processing jobs indexes
CREATE INDEX idx_processing_jobs_status ON processing_jobs(status);
CREATE INDEX idx_processing_jobs_type ON processing_jobs(job_type);
CREATE INDEX idx_processing_jobs_created_at ON processing_jobs(created_at DESC);
CREATE INDEX idx_processing_jobs_priority ON processing_jobs(priority DESC, created_at ASC);
CREATE INDEX idx_processing_jobs_document_id ON processing_jobs(document_id);

-- Search queries indexes
CREATE INDEX idx_search_queries_created_at ON search_queries(created_at DESC);
CREATE INDEX idx_search_queries_hash ON search_queries(query_hash);
CREATE INDEX idx_search_queries_session ON search_queries(user_session);
CREATE INDEX idx_search_queries_text_trgm ON search_queries USING gin(query_text gin_trgm_ops);

-- Composite indexes for common query patterns
CREATE INDEX idx_documents_status_created ON documents(status, created_at DESC);
CREATE INDEX idx_chunks_doc_index ON chunks(document_id, chunk_index);
CREATE INDEX idx_processing_jobs_status_priority ON processing_jobs(status, priority DESC, created_at ASC);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chunks_updated_at 
    BEFORE UPDATE ON chunks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_config_updated_at 
    BEFORE UPDATE ON system_config 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial system configuration
INSERT INTO system_config (key, value, description) VALUES
('chunking_strategy', '"adaptive"', 'Default chunking strategy'),
('max_chunk_size', '1024', 'Maximum chunk size in characters'),
('overlap_size', '128', 'Overlap size between chunks'),
('embedding_model', '"sentence-transformers/all-MiniLM-L6-v2"', 'Default embedding model'),
('similarity_threshold', '0.7', 'Default similarity threshold for searches'),
('max_search_results', '100', 'Maximum number of search results'),
('indexing_batch_size', '50', 'Batch size for indexing operations'),
('cache_ttl_seconds', '3600', 'Default cache TTL in seconds');

-- Views for common queries
CREATE VIEW document_stats AS
SELECT 
    status,
    COUNT(*) as count,
    AVG(file_size) as avg_file_size,
    SUM(file_size) as total_file_size,
    AVG(word_count) as avg_word_count,
    SUM(word_count) as total_word_count
FROM documents 
GROUP BY status;

CREATE VIEW recent_documents AS
SELECT 
    id,
    filename,
    original_name,
    mime_type,
    file_size,
    status,
    created_at,
    processing_completed_at
FROM documents 
ORDER BY created_at DESC 
LIMIT 100;

CREATE VIEW processing_queue AS
SELECT 
    id,
    job_type,
    document_id,
    status,
    priority,
    retry_count,
    created_at,
    started_at
FROM processing_jobs 
WHERE status IN ('queued', 'processing') 
ORDER BY priority DESC, created_at ASC;

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO docrag;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO docrag;