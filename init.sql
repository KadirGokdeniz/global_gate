-- Simplified Airlines Policy Database Schema 
-- Core functionality only - no complex KPI tracking

-- pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main policy table - MULTI-AIRLINE SUPPORT
DROP TABLE IF EXISTS policy CASCADE;

CREATE TABLE policy (
    id SERIAL PRIMARY KEY,
    airline VARCHAR(50) NOT NULL,
    
    source VARCHAR(50) NOT NULL,  -- checked_baggage, sports_equipment, etc.
    content TEXT NOT NULL,
    content_hash VARCHAR(32) NOT NULL,
    quality_score REAL DEFAULT 0.0,
    extraction_type VARCHAR(50) DEFAULT 'unknown',
    url TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Vector embedding
    embedding vector(768),  -- 384 dimension for multilingual model
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- UNIQUE constraint: airline + source + content_hash
    CONSTRAINT unique_airline_content UNIQUE (airline, source, content_hash)
);

-- Performance indexes
CREATE INDEX idx_airline ON policy(airline);
CREATE INDEX idx_source ON policy(source);
CREATE INDEX idx_airline_source ON policy(airline, source);
CREATE INDEX idx_quality_score ON policy(quality_score);
CREATE INDEX idx_created_at ON policy(created_at);
CREATE INDEX idx_updated_at ON policy(updated_at);

-- Full-text search index
CREATE INDEX idx_content_fts 
ON policy 
USING GIN (to_tsvector('english', content));

-- Vector similarity search index
CREATE INDEX idx_embedding_cosine 
ON policy 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- ==============================================
-- OPTIONAL: Simple session logging (if needed)
-- ==============================================

-- Simple session tracking (much lighter than complex KPI tracking)
CREATE TABLE IF NOT EXISTS query_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    question TEXT NOT NULL,
    ai_provider VARCHAR(50), -- 'openai', 'claude'
    model_used VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Simple index
CREATE INDEX idx_query_sessions_created_at ON query_sessions(created_at DESC);

-- ==============================================
-- CORE VIEWS AND FUNCTIONS
-- ==============================================

-- Updated at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_policy_updated_at 
    BEFORE UPDATE ON policy 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Simple stats view per airline
CREATE OR REPLACE VIEW airline_stats AS
SELECT 
    airline,
    source,
    COUNT(*) as total_policies,
    COUNT(embedding) as embedded_policies,
    COUNT(*) - COUNT(embedding) as missing_embeddings,
    ROUND((COUNT(embedding)::NUMERIC / COUNT(*) * 100), 2) as embedding_coverage_percent,
    AVG(quality_score) as avg_quality_score,
    MIN(created_at) as oldest_policy,
    MAX(created_at) as newest_policy
FROM policy 
GROUP BY airline, source
ORDER BY airline, source;

-- Overall stats view 
CREATE OR REPLACE VIEW overall_stats AS
SELECT 
    COUNT(*) as total_policies,
    COUNT(DISTINCT airline) as total_airlines,
    COUNT(DISTINCT source) as total_sources,
    COUNT(embedding) as total_embedded,
    ROUND((COUNT(embedding)::NUMERIC / COUNT(*) * 100), 2) as overall_embedding_coverage,
    AVG(quality_score) as overall_avg_quality
FROM policy;

-- Simple data quality check function
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
        ROUND(AVG(LENGTH(p.content))::NUMERIC, 2) as avg_content_length
    FROM policy p
    GROUP BY p.airline, p.source
    ORDER BY p.airline, p.source;
END;
$$ LANGUAGE plpgsql;

-- Sample data insertion (for testing)
INSERT INTO policy (airline, source, content, content_hash, quality_score, extraction_type) 
VALUES 
    ('turkish_airlines', 'checked_baggage', 'Turkish Airlines checked baggage policy: Passengers are allowed one piece of checked baggage up to 23kg for economy class.', md5('thy_checked_baggage'), 0.8, 'manual'),
    ('turkish_airlines', 'carry_on', 'Turkish Airlines carry-on baggage policy: Economy passengers may bring one carry-on bag up to 8kg and dimensions of 55x40x23cm.', md5('thy_carry_on'), 0.9, 'manual'),
    ('pegasus', 'checked_baggage', 'Pegasus Airlines checked baggage policy: Basic fare includes no checked baggage. Additional baggage can be purchased.', md5('pegasus_checked_baggage'), 0.7, 'manual'),
    ('pegasus', 'carry_on', 'Pegasus Airlines carry-on policy: All passengers may bring one small personal item. Larger carry-on bags require additional purchase.', md5('pegasus_carry_on'), 0.8, 'manual')
ON CONFLICT (airline, source, content_hash) DO NOTHING;

-- Grant permissions (if needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Database ready confirmation
DO $$
BEGIN
    RAISE NOTICE 'Airlines database schema created successfully';
    RAISE NOTICE 'Core Tables: policy, query_sessions (simple)';
    RAISE NOTICE 'Views: airline_stats, overall_stats';  
    RAISE NOTICE 'Functions: check_data_quality()';
    RAISE NOTICE 'Ready for Turkish Airlines + Pegasus data';
    RAISE NOTICE 'Simplified schema - focus on core functionality';
END $$;