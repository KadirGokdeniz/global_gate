-- Multi-Airline Database Schema for Turkish Airlines + Pegasus
-- Bu dosya config/init.sql olarak kaydedilmeli

-- pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Ana baggage policies tablosu - MULTI-AIRLINE SUPPORT
DROP TABLE IF EXISTS baggage_policies CASCADE;

CREATE TABLE baggage_policies (
    id SERIAL PRIMARY KEY,
    
    -- YENİ: Airline identifier
    airline VARCHAR(50) NOT NULL DEFAULT 'turkish_airlines',
    
    -- Existing fields
    source VARCHAR(50) NOT NULL,  -- checked_baggage, sports_equipment, etc.
    content TEXT NOT NULL,
    content_hash VARCHAR(32) NOT NULL,
    quality_score REAL DEFAULT 0.0,
    extraction_type VARCHAR(50) DEFAULT 'unknown',
    url TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Vector embedding
    embedding vector(384),  -- 384 dimension for multilingual model
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- UNIQUE constraint: airline + source + content_hash
    CONSTRAINT unique_airline_content UNIQUE (airline, source, content_hash)
);

-- Performance indexes
CREATE INDEX idx_airline ON baggage_policies(airline);
CREATE INDEX idx_source ON baggage_policies(source);
CREATE INDEX idx_airline_source ON baggage_policies(airline, source);
CREATE INDEX idx_quality_score ON baggage_policies(quality_score);
CREATE INDEX idx_created_at ON baggage_policies(created_at);
CREATE INDEX idx_updated_at ON baggage_policies(updated_at);

-- Full-text search index
CREATE INDEX idx_content_fts 
ON baggage_policies 
USING GIN (to_tsvector('english', content));

-- Vector similarity search index
CREATE INDEX idx_embedding_cosine 
ON baggage_policies 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Updated at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_baggage_policies_updated_at 
    BEFORE UPDATE ON baggage_policies 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Multi-airline stats view
CREATE OR REPLACE VIEW multi_airline_stats AS
SELECT 
    airline,
    source,
    COUNT(*) as total_policies,
    COUNT(embedding) as embedded_policies,
    COUNT(*) - COUNT(embedding) as missing_embeddings,
    ROUND((COUNT(embedding)::float / COUNT(*) * 100)::numeric, 2) as embedding_coverage_percent,
    AVG(quality_score) as avg_quality_score,
    MIN(created_at) as oldest_policy,
    MAX(created_at) as newest_policy
FROM baggage_policies 
GROUP BY airline, source
ORDER BY airline, source;

-- Overall stats view
CREATE OR REPLACE VIEW overall_stats AS
SELECT 
    COUNT(*) as total_policies,
    COUNT(DISTINCT airline) as total_airlines,
    COUNT(DISTINCT source) as total_sources,
    COUNT(embedding) as total_embedded,
    ROUND((COUNT(embedding)::float / COUNT(*) * 100)::numeric, 2) as overall_embedding_coverage,
    AVG(quality_score) as overall_avg_quality
FROM baggage_policies;

-- Data quality check function
CREATE OR REPLACE FUNCTION check_data_quality() 
RETURNS TABLE(
    airline text, 
    source text,
    total_policies bigint,
    quality_issues bigint,
    avg_content_length numeric
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.airline::text,
        p.source::text,
        COUNT(*) as total_policies,
        COUNT(*) FILTER (
            WHERE p.content IS NULL 
               OR LENGTH(p.content) < 10 
               OR p.quality_score < 0.1
        ) as quality_issues,
        ROUND(AVG(LENGTH(p.content))::numeric, 2) as avg_content_length
    FROM baggage_policies p
    GROUP BY p.airline, p.source
    ORDER BY p.airline, p.source;
END;
$$ LANGUAGE plpgsql;

-- Sample data insertion (for testing)
INSERT INTO baggage_policies (airline, source, content, content_hash, quality_score, extraction_type) 
VALUES 
    ('turkish_airlines', 'test_source', 'Test Turkish Airlines policy content', md5('test_thy'), 0.8, 'manual'),
    ('pegasus', 'test_source', 'Test Pegasus Airlines policy content', md5('test_pegasus'), 0.7, 'manual')
ON CONFLICT (airline, source, content_hash) DO NOTHING;

-- Grant permissions (if needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Database ready confirmation
DO $$
BEGIN
    RAISE NOTICE '✅ Multi-airline database schema created successfully';
    RAISE NOTICE '📊 Tables: baggage_policies';
    RAISE NOTICE '📈 Views: multi_airline_stats, overall_stats';
    RAISE NOTICE '🔍 Functions: check_data_quality()';
    RAISE NOTICE '🚀 Ready for Turkish Airlines + Pegasus data';
END $$;