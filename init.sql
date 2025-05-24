-- init.sql - Database initialization with Vector Embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Updated table with embedding column
CREATE TABLE IF NOT EXISTS baggage_policies (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(32) UNIQUE NOT NULL,
    quality_score REAL DEFAULT 0.0,
    extraction_type VARCHAR(50) DEFAULT 'unknown',
    url TEXT,
    metadata JSONB DEFAULT '{}',
    embedding vector(384),  -- NEW: OpenAI Ada-002 embeddings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

-- NEW: Vector similarity search index
CREATE INDEX IF NOT EXISTS idx_baggage_policies_embedding 
ON baggage_policies 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Function to update updated_at automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for automatic updated_at
CREATE TRIGGER update_baggage_policies_updated_at 
    BEFORE UPDATE ON baggage_policies 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample test data
INSERT INTO baggage_policies (source, content, content_hash, quality_score, extraction_type) 
VALUES (
    'test_source',
    'Welcome to Turkish Airlines Baggage Policy API - Test Data. This service provides comprehensive information about carry-on luggage, checked baggage weight limits, and restricted items for international and domestic flights.',
    'test123updated', 
    1.0,
    'manual'
) ON CONFLICT (content_hash) DO NOTHING;

-- Insert more realistic test data
INSERT INTO baggage_policies (source, content, content_hash, quality_score, extraction_type, metadata) 
VALUES 
(
    'turkish_airlines',
    'Passengers are allowed to carry one piece of carry-on baggage weighing up to 8 kg (17.6 lbs) and measuring maximum 55x40x23 cm (21.5x15.5x9 inches). Personal items such as laptop bags, purses, and camera bags are allowed in addition to carry-on baggage.',
    MD5('turkish_airlines_carry_on_policy'),
    0.95,
    'manual',
    '{"policy_type": "carry_on", "airline": "turkish_airlines", "updated": "2024-01-15"}'
),
(
    'turkish_airlines', 
    'Electronic devices including laptops, tablets, smartphones, and cameras are permitted in carry-on baggage. Lithium batteries must be carried in carry-on luggage only. Power banks are limited to 100Wh capacity.',
    MD5('turkish_airlines_electronics_policy'),
    0.92,
    'manual',
    '{"policy_type": "electronics", "airline": "turkish_airlines", "updated": "2024-01-15"}'
),
(
    'general_aviation',
    'Liquids, gels, and aerosols in carry-on baggage must be in containers of 100ml or less and placed in a clear, resealable plastic bag. Each passenger is limited to one bag measuring 20x20cm.',
    MD5('general_liquids_policy'),
    0.88,
    'manual',
    '{"policy_type": "liquids", "source": "general_aviation", "updated": "2024-01-15"}'
) ON CONFLICT (content_hash) DO NOTHING;