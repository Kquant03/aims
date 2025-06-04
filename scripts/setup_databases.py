#!/usr/bin/env python3
# scripts/setup_databases.py - Database initialization script
import asyncio
import asyncpg
import os
import sys
from pathlib import Path
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging

logger = setup_logging('INFO', 'logs/db_setup.log')

async def setup_postgresql():
    """Set up PostgreSQL database"""
    logger.info("Setting up PostgreSQL...")
    
    # Connection parameters
    host = os.environ.get('POSTGRES_HOST', 'localhost')
    port = int(os.environ.get('POSTGRES_PORT', '5433'))
    database = os.environ.get('POSTGRES_DB', 'aims_memory')
    user = os.environ.get('POSTGRES_USER', 'aims')
    password = os.environ.get('POSTGRES_PASSWORD', 'aims_secure_password')
    
    try:
        # Connect to PostgreSQL
        conn = await asyncpg.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        
        # Create pgvector extension
        await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        logger.info("Created pgvector extension")
        
        # Create tables
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(768),
                importance FLOAT DEFAULT 0.5,
                emotional_context JSONB,
                associations TEXT[],
                decay_rate FLOAT DEFAULT 0.1,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                metadata JSONB
            )
        ''')
        logger.info("Created memories table")
        
        # Create indexes
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
        ''')
        
        # Create HNSW index for vector similarity
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_memories_embedding 
            ON memories USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        ''')
        logger.info("Created indexes")
        
        # Create conversation history table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS conversation_history (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_refs TEXT[],
                created_at TIMESTAMP DEFAULT NOW(),
                metadata JSONB
            )
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_conversation_session 
            ON conversation_history(session_id);
            CREATE INDEX IF NOT EXISTS idx_conversation_user 
            ON conversation_history(user_id);
        ''')
        logger.info("Created conversation history table")
        
        # Create user states table
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS user_states (
                user_id TEXT PRIMARY KEY,
                consciousness_state JSONB,
                personality_state JSONB,
                emotional_state JSONB,
                last_interaction TIMESTAMP,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        ''')
        logger.info("Created user states table")
        
        await conn.close()
        logger.info("PostgreSQL setup complete")
        
    except Exception as e:
        logger.error(f"PostgreSQL setup failed: {e}")
        raise

def setup_qdrant():
    """Set up Qdrant vector database"""
    logger.info("Setting up Qdrant...")
    
    host = os.environ.get('QDRANT_HOST', 'localhost')
    port = int(os.environ.get('QDRANT_PORT', '6333'))
    
    try:
        client = QdrantClient(host=host, port=port)
        
        # Create memories collection
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "aims_memories" not in collection_names:
            client.create_collection(
                collection_name="aims_memories",
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )
            logger.info("Created aims_memories collection")
        else:
            logger.info("aims_memories collection already exists")
        
        # Create embeddings collection for faster search
        if "aims_embeddings" not in collection_names:
            client.create_collection(
                collection_name="aims_embeddings",
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                ),
                optimizers_config={
                    "indexing_threshold": 10000,
                    "memmap_threshold": 50000
                }
            )
            logger.info("Created aims_embeddings collection")
        
        logger.info("Qdrant setup complete")
        
    except Exception as e:
        logger.error(f"Qdrant setup failed: {e}")
        raise

async def verify_redis():
    """Verify Redis is running"""
    logger.info("Verifying Redis...")
    
    import redis
    
    host = os.environ.get('REDIS_HOST', 'localhost')
    port = int(os.environ.get('REDIS_PORT', '6379'))
    
    try:
        r = redis.Redis(host=host, port=port)
        r.ping()
        logger.info("Redis is running and accessible")
        
        # Set some default configurations
        r.config_set('maxmemory', '4gb')
        r.config_set('maxmemory-policy', 'allkeys-lru')
        logger.info("Redis configuration updated")
        
    except Exception as e:
        logger.error(f"Redis verification failed: {e}")
        raise

async def main():
    """Main setup function"""
    logger.info("Starting AIMS database setup...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Setup PostgreSQL
    await setup_postgresql()
    
    # Setup Qdrant
    setup_qdrant()
    
    # Verify Redis
    await verify_redis()
    
    logger.info("All databases setup complete!")
    
    # Print connection info
    print("\nDatabase setup complete! Connection details:")
    print(f"PostgreSQL: {os.environ.get('POSTGRES_HOST', 'localhost')}:{os.environ.get('POSTGRES_PORT', '5433')}")
    print(f"Redis: {os.environ.get('REDIS_HOST', 'localhost')}:{os.environ.get('REDIS_PORT', '6379')}")
    print(f"Qdrant: {os.environ.get('QDRANT_HOST', 'localhost')}:{os.environ.get('QDRANT_PORT', '6333')}")

if __name__ == "__main__":
    asyncio.run(main())