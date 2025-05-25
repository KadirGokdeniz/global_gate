-- init.sql - Database initialization
CREATE EXTENSION IF NOT EXISTS vector;

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
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance için indexler
CREATE INDEX IF NOT EXISTS idx_source ON baggage_policies(source);
CREATE INDEX IF NOT EXISTS idx_content_hash ON baggage_policies(content_hash);
CREATE INDEX IF NOT EXISTS idx_quality_score ON baggage_policies(quality_score);
CREATE INDEX IF NOT EXISTS idx_created_at ON baggage_policies(created_at);

-- Full-text search için
CREATE INDEX IF NOT EXISTS idx_content_fts 
ON baggage_policies 
USING GIN (to_tsvector('english', content));

-- Test verisi ekle
INSERT INTO baggage_policies (source, content, content_hash, quality_score, extraction_type) 
VALUES (
    'test_source',
    'Welcome to Turkish Airlines Baggage Policy API - Test Data',
    'test123',
    1.0,
    'manual'
) ON CONFLICT (content_hash) DO NOTHING;