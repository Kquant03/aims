-- scripts/init_db.sql
-- PostgreSQL initialization script for AIMS

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Create main schema
CREATE SCHEMA IF NOT EXISTS aims;

-- Set search path
SET search_path TO aims, public;

-- Create custom types
CREATE TYPE emotion_state AS (
    pleasure FLOAT,
    arousal FLOAT,
    dominance FLOAT
);

CREATE TYPE personality_traits AS (
    openness FLOAT,
    conscientiousness FLOAT,
    extraversion FLOAT,
    agreeableness FLOAT,
    neuroticism FLOAT
);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for all tables with updated_at
CREATE TRIGGER update_memories_updated_at BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();