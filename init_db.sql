-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create episodic memories table
CREATE TABLE IF NOT EXISTS episodic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL,
    context JSONB,
    embedding vector(1536),
    salience_score JSONB,
    metadata JSONB,
    importance FLOAT DEFAULT 0.5,
    attention_focus TEXT,
    emotional_state JSONB,
    INDEXES:
    CREATE INDEX idx_session_id ON episodic_memories(session_id);
    CREATE INDEX idx_user_id ON episodic_memories(user_id);
    CREATE INDEX idx_timestamp ON episodic_memories(timestamp);
);

-- Create vector similarity index
CREATE INDEX episodic_memory_embedding_idx 
ON episodic_memories 
USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);