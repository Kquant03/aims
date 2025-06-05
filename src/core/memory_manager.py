# memory_manager.py - Fixed version with enhanced conversation storage
import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import logging
from collections import deque
import hnswlib
from typing import Tuple
import msgpack

# Add missing torch import
try:
    import torch
except ImportError:
    torch = None
    
logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Represents a single memory item"""
    id: str
    content: str
    timestamp: datetime
    importance: float
    emotional_context: Dict[str, float]
    associations: List[str]
    decay_rate: float = 0.1
    
    def calculate_salience(self, current_time: datetime) -> float:
        """Calculate current salience based on importance and time decay"""
        time_delta = (current_time - self.timestamp).total_seconds() / 86400  # Days
        decay_factor = np.exp(-self.decay_rate * time_delta)
        
        # Emotional boost - stronger emotions make memories more salient
        emotional_intensity = np.mean(list(self.emotional_context.values()))
        emotional_boost = 1 + (emotional_intensity - 0.5) * 0.5
        
        return self.importance * decay_factor * emotional_boost

class PersistentMemoryManager:
    """Manages persistent memory storage and retrieval"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memories: Dict[str, MemoryItem] = {}
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.consolidation_threshold = config.get('consolidation_threshold', 0.3)
        
        # Initialize storage connections (simplified for now)
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage connections"""
        logger.info("Initialized memory storage (in-memory mode)")
    
    async def store_memory(self, content: str, context: Dict[str, Any]) -> str:
        """Store a new memory"""
        # Generate memory ID
        memory_id = hashlib.sha256(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Create memory item
        memory = MemoryItem(
            id=memory_id,
            content=content,
            timestamp=datetime.now(),
            importance=context.get('importance', 0.5),
            emotional_context=context.get('emotional_state', {
                'pleasure': 0.5,
                'arousal': 0.5,
                'dominance': 0.5
            }),
            associations=context.get('associations', [])
        )
        
        # Store in memory
        self.memories[memory_id] = memory
        
        logger.debug(f"Stored memory {memory_id}: {content[:50]}...")
        return memory_id
    
    async def retrieve_memories(self, query: str, k: int = 5) -> List[MemoryItem]:
        """Retrieve relevant memories based on query"""
        # Simple implementation - in production, use vector similarity
        current_time = datetime.now()
        
        # Calculate salience for all memories
        memory_scores = []
        for memory in self.memories.values():
            # Simple keyword matching (replace with embeddings in production)
            query_words = set(query.lower().split())
            memory_words = set(memory.content.lower().split())
            overlap = len(query_words.intersection(memory_words))
            
            relevance = overlap / max(len(query_words), 1)
            salience = memory.calculate_salience(current_time)
            
            score = relevance * 0.7 + salience * 0.3
            memory_scores.append((score, memory))
        
        # Sort by score and return top k
        memory_scores.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in memory_scores[:k]]
    
    async def save_conversation_turn(self, session_id: str, role: str, 
                                   content: str, **metadata):
        """Save a conversation turn with flexible metadata"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        turn = {
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content
        }
        
        # Add any additional metadata
        turn.update(metadata)
        
        self.conversation_history[session_id].append(turn)
        
        logger.debug(f"Saved {role} turn for session {session_id}")
    
    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        if session_id not in self.conversation_history:
            return []
        
        history = self.conversation_history[session_id]
        return history[-limit:] if len(history) > limit else history
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            'total_memories': len(self.memories),
            'total_sessions': len(self.conversation_history),
            'oldest_memory': min(
                (m.timestamp for m in self.memories.values()),
                default=None
            ),
            'average_importance': np.mean(
                [m.importance for m in self.memories.values()]
            ) if self.memories else 0
        }
    
    async def consolidate_memories(self):
        """Consolidate memories based on importance and recency"""
        current_time = datetime.now()
        
        # Remove memories below salience threshold
        memories_to_remove = []
        for memory_id, memory in self.memories.items():
            if memory.calculate_salience(current_time) < self.consolidation_threshold:
                memories_to_remove.append(memory_id)
        
        for memory_id in memories_to_remove:
            del self.memories[memory_id]
        
        logger.info(f"Consolidated memories: removed {len(memories_to_remove)} low-salience memories")

class AdvancedMemoryManager(PersistentMemoryManager):
    """Enhanced memory manager with HNSW indexing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_dim = config.get('embedding_dim', 768)
        self.use_gpu = torch.cuda.is_available() if torch else False
        
        # Initialize HNSW index
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.hnsw_index.init_index(
            max_elements=config.get('max_memories', 1000000),
            ef_construction=200,
            M=32
        )
        self.hnsw_index.set_ef(100)  # Query time parameter
        
        # Memory consolidation parameters
        self.consolidation_threshold = config.get('consolidation_threshold', 0.3)
        self.importance_decay_rate = config.get('importance_decay_rate', 0.01)
        
        # Keep track of memory embeddings
        self.memory_embeddings = {}
        self.embedding_to_id = {}
        
    async def store_memory_with_embedding(self, content: str, embedding: np.ndarray, 
                                        context: Dict[str, Any]) -> str:
        """Store memory with pre-computed embedding"""
        # Store the memory using parent method
        memory_id = await self.store_memory(content, context)
        
        # Add to HNSW index
        self.hnsw_index.add_items(
            embedding.reshape(1, -1),
            [len(self.memory_embeddings)]
        )
        
        # Track embeddings
        self.memory_embeddings[memory_id] = embedding
        self.embedding_to_id[len(self.memory_embeddings) - 1] = memory_id
        
        return memory_id
    
    async def retrieve_memories_by_embedding(self, query_embedding: np.ndarray, 
                                           k: int = 5) -> List[MemoryItem]:
        """Retrieve memories using embedding similarity"""
        if len(self.memory_embeddings) == 0:
            return []
        
        # Search in HNSW index
        labels, distances = self.hnsw_index.knn_query(
            query_embedding.reshape(1, -1), 
            k=min(k, len(self.memory_embeddings))
        )
        
        # Get memory items
        results = []
        for idx in labels[0]:
            if idx in self.embedding_to_id:
                memory_id = self.embedding_to_id[idx]
                if memory_id in self.memories:
                    results.append(self.memories[memory_id])
        
        return results