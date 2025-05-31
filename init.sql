-- init.sql - Database initialization with pgvector
-- This file will run automatically when the database container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create main baggage policies table
CREATE TABLE IF NOT EXISTS baggage_policies (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(32) UNIQUE NOT NULL,
    quality_score REAL DEFAULT 0.0,
    extraction_type VARCHAR(50) DEFAULT 'unknown',
    url TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    embedding vector(384)  -- Add embedding column immediately
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_source ON baggage_policies(source);
CREATE INDEX IF NOT EXISTS idx_content_hash ON baggage_policies(content_hash);
CREATE INDEX IF NOT EXISTS idx_quality_score ON baggage_policies(quality_score);
CREATE INDEX IF NOT EXISTS idx_created_at ON baggage_policies(created_at);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_content_fts 
ON baggage_policies 
USING GIN (to_tsvector('english', content));

-- Vector similarity index (only create if embedding column has data)
-- This will be created later when embeddings are added

-- Embedding statistics view
CREATE OR REPLACE VIEW embedding_stats AS
SELECT 
    COUNT(*) as total_policies,
    COUNT(embedding) as embedded_policies,
    COUNT(*) - COUNT(embedding) as missing_embeddings,
    CASE 
        WHEN COUNT(*) > 0 THEN ROUND((COUNT(embedding)::float / COUNT(*) * 100)::numeric, 2)
        ELSE 0
    END as coverage_percent
FROM baggage_policies;

-- Function to safely create vector index when we have embeddings
CREATE OR REPLACE FUNCTION create_vector_index_if_needed() 
RETURNS void AS $$
BEGIN
    -- Only create index if we have embeddings and index doesn't exist
    IF EXISTS (SELECT 1 FROM baggage_policies WHERE embedding IS NOT NULL LIMIT 1) THEN
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_embedding_cosine') THEN
            CREATE INDEX idx_embedding_cosine 
            ON baggage_policies 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
            
            RAISE NOTICE 'Vector index created successfully';
        END IF;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialized successfully with pgvector extension';
    RAISE NOTICE 'Tables created: baggage_policies';
    RAISE NOTICE 'Indexes created: source, content_hash, quality_score, created_at, content_fts';
    RAISE NOTICE 'Views created: embedding_stats';
    RAISE NOTICE 'Ready for data and embeddings';
END $$;