# three_tier_memory.py - Core Three-Tier Memory Architecture (FIXED)
import asyncio
import redis.asyncio as redis
import asyncpg
from sqlalchemy import Column, String, DateTime, Text, Integer, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import chromadb
from typing import Optional, Dict, List, Any, TypeVar, Generic
from abc import ABC, abstractmethod
import json
import time
from datetime import datetime, timedelta
import numpy as np
import logging
import uuid

logger = logging.getLogger(__name__)

Base = declarative_base()

# PostgreSQL Episodic Memory Schema
class EpisodicMemory(Base):
    __tablename__ = 'episodic_memories'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String, index=True, nullable=False)
    user_id = Column(String, index=True)
    timestamp = Column(DateTime, index=True, default=datetime.utcnow)
    content = Column(Text, nullable=False)
    context = Column(JSON)
    embedding = Column(Vector(1536))  # OpenAI ada-002 dimensions
    salience_score = Column(JSON)  # Multi-dimensional scores
    memory_memory_metadata = Column(JSON)
    importance = Column(Float, default=0.5)
    attention_focus = Column(Text)  # What AIMS was focusing on
    emotional_state = Column(JSON)  # PAD values at time of memory

class ThreeTierMemorySystem:
    """Complete implementation of the three-tier memory architecture"""
    
    def __init__(self, config: dict):
        self.config = config
        self.initialized = False
        
    async def initialize(self):
        """Initialize all memory stores"""
        try:
            # Working Memory (Redis)
            self.redis_client = await redis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'), 
                decode_responses=True,
                max_connections=50
            )
            
            # Test Redis connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Episodic Memory (PostgreSQL with pgvector)
            self.pg_engine = create_async_engine(
                self.config.get('postgres_url', 'postgresql+asyncpg://aims:aims_password@localhost:5432/aims_memory'),
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True
            )
            
            # Create session maker
            self.async_session = async_sessionmaker(
                self.pg_engine, 
                class_=AsyncSession, 
                expire_on_commit=False
            )
            
            # Create tables if they don't exist
            async with self.pg_engine.begin() as conn:
                await conn.run_sync(Base.memory_metadata.create_all)
                
                # Create indices for vector search
                await conn.execute("""
                    CREATE EXTENSION IF NOT EXISTS vector;
                    
                    CREATE INDEX IF NOT EXISTS episodic_memory_embedding_idx 
                    ON episodic_memories 
                    USING ivfflat (embedding vector_cosine_ops) 
                    WITH (lists = 100);
                """)
            
            logger.info("PostgreSQL with pgvector initialized")
            
            # Semantic Memory (ChromaDB for development)
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.get('chroma_persist_dir', './data/chroma')
            )
            self.semantic_collection = self.chroma_client.get_or_create_collection(
                name="semantic_memory",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("ChromaDB semantic memory initialized")
            
            # Initialize memory subsystems
            self.working_memory = WorkingMemory(self.redis_client, ttl_seconds=300)
            self.episodic_store = EpisodicMemoryStore(self.pg_engine, self.async_session)
            self.semantic_store = SemanticMemoryStore(self.semantic_collection)
            
            self.initialized = True
            logger.info("Three-tier memory system fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            raise
    
    async def close(self):
        """Close all connections"""
        if hasattr(self, 'redis_client'):
            await self.redis_client.close()
        if hasattr(self, 'pg_engine'):
            await self.pg_engine.dispose()

class WorkingMemory:
    """Redis-based working memory with 5-minute TTL"""
    
    def __init__(self, redis_client, ttl_seconds: int = 300):
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.memory_prefix = "wm"  # working_memory prefix
        
    async def store_interaction(self, session_id: str, interaction_data: dict):
        """Store interaction with automatic expiration"""
        key = f"{self.memory_prefix}:{session_id}:interactions"
        
        # Add timestamp to interaction
        interaction_data['timestamp'] = time.time()
        interaction_data['id'] = str(uuid.uuid4())
        
        # Store in Redis sorted set by timestamp
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.zadd(
                key, 
                {json.dumps(interaction_data): interaction_data['timestamp']}
            )
            pipe.expire(key, self.ttl)
            await pipe.execute()
        
        # Also store current context separately
        context_key = f"{self.memory_prefix}:{session_id}:context"
        await self.redis.setex(
            context_key, 
            self.ttl, 
            json.dumps(interaction_data.get('context', {}))
        )
        
        # Update attention focus
        if 'attention_focus' in interaction_data:
            await self.update_attention_focus(session_id, interaction_data['attention_focus'])
    
    async def get_recent_interactions(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Retrieve recent interactions within TTL window"""
        key = f"{self.memory_prefix}:{session_id}:interactions"
        
        # Get most recent interactions using ZREVRANGE
        raw_interactions = await self.redis.zrevrange(key, 0, limit - 1)
        
        interactions = []
        for raw in raw_interactions:
            try:
                interactions.append(json.loads(raw))
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode interaction: {raw}")
                continue
                
        return interactions
    
    async def update_attention_focus(self, session_id: str, focus_data: Any):
        """Update current attention focus with sliding window"""
        key = f"{self.memory_prefix}:{session_id}:attention"
        
        # Store with automatic expiration
        await self.redis.setex(
            key, 
            self.ttl, 
            json.dumps(focus_data) if not isinstance(focus_data, str) else focus_data
        )
        
        # Extend TTL on active sessions
        await self.redis.expire(f"{self.memory_prefix}:{session_id}:interactions", self.ttl)
    
    async def get_working_memory_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of current working memory state"""
        interactions = await self.get_recent_interactions(session_id, limit=7)
        
        # Get current context
        context_key = f"{self.memory_prefix}:{session_id}:context"
        context = await self.redis.get(context_key)
        
        # Get attention focus
        attention_key = f"{self.memory_prefix}:{session_id}:attention"
        attention = await self.redis.get(attention_key)
        
        return {
            'interaction_count': len(interactions),
            'recent_interactions': interactions,
            'current_context': json.loads(context) if context else {},
            'attention_focus': json.loads(attention) if attention else None,
            'oldest_timestamp': interactions[-1]['timestamp'] if interactions else None
        }

class EpisodicMemoryStore:
    """PostgreSQL-based episodic memory with vector similarity search"""
    
    def __init__(self, engine, session_maker):
        self.engine = engine
        self.Session = session_maker
        
    async def store_episode(self, episode_data: dict) -> str:
        """Store episodic memory with full context"""
        async with self.Session() as session:
            episode = EpisodicMemory(
                session_id=episode_data['session_id'],
                user_id=episode_data.get('user_id'),
                content=episode_data['content'],
                embedding=episode_data.get('embedding'),
                context=episode_data.get('context', {}),
                salience_score=episode_data.get('salience_score', {}),
                memory_metadata=episode_data.get('metadata', {}),
                importance=episode_data.get('importance', 0.5),
                attention_focus=episode_data.get('attention_focus'),
                emotional_state=episode_data.get('emotional_state', {})
            )
            
            session.add(episode)
            await session.commit()
            await session.refresh(episode)
            
            logger.debug(f"Stored episodic memory {episode.id}: {episode.content[:50]}...")
            return str(episode.id)
    
    async def search_similar_episodes(
        self, 
        query_embedding: Optional[List[float]] = None, 
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        time_window_hours: Optional[int] = None,
        limit: int = 10,
        importance_threshold: float = 0.0
    ) -> List[EpisodicMemory]:
        """Vector similarity search with temporal and importance filtering"""
        async with self.Session() as session:
            from sqlalchemy import select, and_, text
            
            # Base query with vector similarity
            query = select(EpisodicMemory)
            
            # Apply filters
            filters = []
            if session_id:
                filters.append(EpisodicMemory.session_id == session_id)
            if user_id:
                filters.append(EpisodicMemory.user_id == user_id)
            if time_window_hours:
                cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
                filters.append(EpisodicMemory.timestamp >= cutoff_time)
            if importance_threshold > 0:
                filters.append(EpisodicMemory.importance >= importance_threshold)
            
            if filters:
                query = query.where(and_(*filters))
            
            # Order by cosine similarity if embedding provided
            if query_embedding:
                # Convert to string format for PostgreSQL
                embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
                query = query.order_by(
                    text(f"embedding <=> '{embedding_str}'::vector")
                )
            else:
                # Order by timestamp if no embedding
                query = query.order_by(EpisodicMemory.timestamp.desc())
            
            query = query.limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_temporal_sequence(
        self, 
        session_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[EpisodicMemory]:
        """Retrieve episodic memories in temporal order"""
        async with self.Session() as session:
            from sqlalchemy import select, and_
            
            query = select(EpisodicMemory).where(
                and_(
                    EpisodicMemory.session_id == session_id,
                    EpisodicMemory.timestamp >= start_time,
                    EpisodicMemory.timestamp <= end_time
                )
            ).order_by(EpisodicMemory.timestamp)
            
            result = await session.execute(query)
            return result.scalars().all()
    
    async def get_user_memories(
        self, 
        user_id: str, 
        limit: int = 100
    ) -> List[EpisodicMemory]:
        """Get all memories for a specific user"""
        async with self.Session() as session:
            from sqlalchemy import select
            
            query = select(EpisodicMemory).where(
                EpisodicMemory.user_id == user_id
            ).order_by(
                EpisodicMemory.timestamp.desc()
            ).limit(limit)
            
            result = await session.execute(query)
            return result.scalars().all()

class SemanticMemoryStore:
    """ChromaDB implementation for semantic memory storage"""
    
    def __init__(self, collection):
        self.collection = collection
        
    async def store_semantic_knowledge(self, knowledge_data: dict) -> str:
        """Store semantic memory with metadata"""
        doc_id = f"{knowledge_data.get('category', 'general')}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
        
        # Prepare document
        document = knowledge_data['content']
        embedding = knowledge_data.get('embedding')
        metadata = {
            "category": knowledge_data.get('category', 'general'),
            "source": knowledge_data.get('source', 'episodic_consolidation'),
            "confidence": knowledge_data.get('confidence', 1.0),
            "timestamp": time.time(),
            "importance": knowledge_data.get('importance', 0.5),
            "related_concepts": json.dumps(knowledge_data.get('related_concepts', [])),
            "user_id": knowledge_data.get('user_id'),
            "consolidated_from": json.dumps(knowledge_data.get('consolidated_from', []))
        }
        
        # Store in ChromaDB
        if embedding:
            self.collection.add(
                documents=[document],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[doc_id]
            )
        else:
            self.collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
        
        logger.debug(f"Stored semantic memory {doc_id}: {document[:50]}...")
        return doc_id
    
    async def semantic_search(
        self, 
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        category_filter: Optional[str] = None,
        user_filter: Optional[str] = None,
        confidence_threshold: float = 0.0,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Semantic similarity search with filtering"""
        # Build where clause
        where_clause = {}
        if confidence_threshold > 0:
            where_clause["confidence"] = {"$gte": confidence_threshold}
        if category_filter:
            where_clause["category"] = category_filter
        if user_filter:
            where_clause["user_id"] = user_filter
        
        # Perform search
        if query_embedding:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                where=where_clause if where_clause else None,
                n_results=top_k
            )
        elif query_text:
            results = self.collection.query(
                query_texts=[query_text],
                where=where_clause if where_clause else None,
                n_results=top_k
            )
        else:
            # Get all matching documents
            results = self.collection.get(
                where=where_clause if where_clause else None,
                limit=top_k
            )
            # Convert to query format
            results = {
                'documents': [results.get('documents', [])],
                'metadatas': [results.get('metadatas', [])],
                'distances': [[0.0] * len(results.get('documents', []))]
            }
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                formatted_results.append({
                    'content': doc,
                    'score': 1.0 - results['distances'][0][i] if results['distances'] else 1.0,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })
        
        return formatted_results
    
    async def get_concepts_graph(self, user_id: Optional[str] = None) -> Dict[str, List[str]]:
        """Build a graph of related concepts from semantic memory"""
        where_clause = {"user_id": user_id} if user_id else None
        
        # Get all semantic memories
        results = self.collection.get(
            where=where_clause,
            limit=1000  # Adjust as needed
        )
        
        # Build concept graph
        concept_graph = {}
        
        for metadata in results.get('metadatas', []):
            category = metadata.get('category', 'general')
            related = json.loads(metadata.get('related_concepts', '[]'))
            
            if category not in concept_graph:
                concept_graph[category] = []
            
            concept_graph[category].extend(related)
        
        # Deduplicate
        for category in concept_graph:
            concept_graph[category] = list(set(concept_graph[category]))
        
        return concept_graph