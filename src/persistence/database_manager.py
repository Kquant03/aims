# Enhanced database_manager.py with actual implementations
import asyncio
import asyncpg
import redis.asyncio as redis
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import uuid
import json
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Production-ready database connection management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.qdrant_client: Optional[QdrantClient] = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize all database connections"""
        try:
            # PostgreSQL connection pool
            self.pg_pool = await asyncpg.create_pool(
                host=self.config['pg_host'],
                port=self.config['pg_port'],
                database=self.config['pg_database'],
                user=self.config['pg_user'],
                password=self.config['pg_password'],
                min_size=10,
                max_size=50,
                command_timeout=60
            )
            
            # Create tables if needed
            await self._init_postgres_schema()
            
            # Redis connection
            self.redis_client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                decode_responses=True,
                max_connections=100
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            
            # Qdrant connection
            self.qdrant_client = QdrantClient(
                host=self.config['qdrant_host'],
                port=self.config['qdrant_port']
            )
            
            # Create collection if needed
            await self._init_qdrant_collection()
            
            self.initialized = True
            logger.info("All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
            raise
    
    async def _init_postgres_schema(self):
        """Initialize PostgreSQL schema"""
        async with self.pg_pool.acquire() as conn:
            # Create pgvector extension
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            
            # Create memories table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(768),
                    importance FLOAT DEFAULT 0.5,
                    emotional_context JSONB,
                    associations TEXT[],
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB
                )
            ''')
            
            # Create indexes
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
                CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at);
                CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories 
                USING ivfflat (embedding vector_cosine_ops);
            ''')
            
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
            
            # Create index for conversation history
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversation_session 
                ON conversation_history(session_id);
            ''')
    
    async def _init_qdrant_collection(self):
        """Initialize Qdrant collection"""
        collection_name = "aims_memories"
        
        # Check if collection exists
        collections = await self.qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if collection_name not in collection_names:
            # Create collection
            await self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=768,  # Embedding dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
    
    async def store_memory(self, 
                          content: str, 
                          embedding: List[float], 
                          metadata: Dict[str, Any]) -> str:
        """Store a memory with embedding across all databases"""
        memory_id = str(uuid.uuid4())
        
        # Store in PostgreSQL
        async with self.pg_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO memories 
                (id, user_id, content, embedding, importance, emotional_context, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''', 
                memory_id,
                metadata.get('user_id', 'anonymous'),
                content,
                embedding,
                metadata.get('importance', 0.5),
                json.dumps(metadata.get('emotional_context', {})),
                json.dumps(metadata)
            )
        
        # Store in Qdrant
        await self.qdrant_client.upsert(
            collection_name="aims_memories",
            points=[
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "content": content,
                        "user_id": metadata.get('user_id', 'anonymous'),
                        **metadata
                    }
                )
            ]
        )
        
        # Cache in Redis with TTL
        cache_key = f"memory:{memory_id}"
        await self.redis_client.setex(
            cache_key,
            3600,  # 1 hour TTL
            json.dumps({
                "content": content,
                "metadata": metadata
            })
        )
        
        return memory_id
    
    async def search_memories(self, 
                            query_embedding: List[float], 
                            limit: int = 10,
                            user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search memories by vector similarity"""
        # Search in Qdrant
        search_params = {
            "collection_name": "aims_memories",
            "query_vector": query_embedding,
            "limit": limit
        }
        
        if user_id:
            search_params["query_filter"] = {
                "must": [
                    {"key": "user_id", "match": {"value": user_id}}
                ]
            }
        
        results = await self.qdrant_client.search(**search_params)
        
        # Format results
        memories = []
        for result in results:
            memories.append({
                "id": result.id,
                "content": result.payload.get("content"),
                "score": result.score,
                "metadata": result.payload
            })
        
        return memories
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        # Check Redis cache first
        cache_key = f"memory:{memory_id}"
        cached = await self.redis_client.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        # Fetch from PostgreSQL
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM memories WHERE id = $1
            ''', memory_id)
            
            if row:
                memory = {
                    "id": str(row['id']),
                    "content": row['content'],
                    "importance": row['importance'],
                    "emotional_context": row['emotional_context'],
                    "metadata": row['metadata'],
                    "created_at": row['created_at'].isoformat()
                }
                
                # Cache it
                await self.redis_client.setex(
                    cache_key,
                    3600,
                    json.dumps(memory)
                )
                
                return memory
        
        return None
    
    async def save_conversation_turn(self,
                                   session_id: str,
                                   user_id: str,
                                   role: str,
                                   content: str,
                                   memory_refs: Optional[List[str]] = None):
        """Save a conversation turn"""
        async with self.pg_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO conversation_history 
                (session_id, user_id, role, content, memory_refs)
                VALUES ($1, $2, $3, $4, $5)
            ''',
                session_id,
                user_id,
                role,
                content,
                memory_refs or []
            )
    
    async def get_conversation_history(self,
                                     session_id: str,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT * FROM conversation_history
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            ''', session_id, limit)
            
            history = []
            for row in reversed(rows):  # Reverse to get chronological order
                history.append({
                    "role": row['role'],
                    "content": row['content'],
                    "memory_refs": row['memory_refs'],
                    "created_at": row['created_at'].isoformat()
                })
            
            return history
    
    async def cache_set(self, key: str, value: Any, ttl: int = 3600):
        """Set a value in cache"""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        
        await self.redis_client.setex(key, ttl, value)
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        value = await self.redis_client.get(key)
        
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        
        return None
    
    async def close(self):
        """Close all database connections"""
        if self.pg_pool:
            await self.pg_pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("All database connections closed")