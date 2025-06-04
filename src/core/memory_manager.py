# memory_manager.py - Persistent Memory Management System
import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import logging
from collections import deque
import faiss
import hnswlib
from typing import Tuple
import msgpack

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
        # In a real implementation, this would connect to Redis, PostgreSQL, and Qdrant
        # For now, we'll use in-memory storage
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
                                   content: str, memory_refs: Optional[List[str]] = None):
        """Save a conversation turn"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        turn = {
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content,
            'memory_refs': memory_refs or []
        }
        
        self.conversation_history[session_id].append(turn)
    
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
    """Enhanced memory manager with HNSW indexing and GPU acceleration"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embedding_dim = config.get('embedding_dim', 768)
        self.use_gpu = torch.cuda.is_available() and config.get('use_gpu', True)
        
        # Initialize HNSW index
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.hnsw_index.init_index(
            max_elements=config.get('max_memories', 1000000),
            ef_construction=200,
            M=32
        )
        self.hnsw_index.set_ef(100)  # Query time parameter
        
        # Initialize FAISS GPU index if available
        if self.use_gpu:
            self._init_faiss_gpu()
        
        # Memory consolidation parameters
        self.consolidation_threshold = config.get('consolidation_threshold', 0.3)
        self.importance_decay_rate = config.get('importance_decay_rate', 0.01)
        
    def _init_faiss_gpu(self):
        """Initialize FAISS GPU index"""
        try:
            import faiss
            
            # Create GPU resources
            self.gpu_res = faiss.StandardGpuResources()
            
            # Create GPU index
            self.gpu_index = faiss.GpuIndexFlatL2(
                self.gpu_res,
                self.embedding_dim
            )
            
            logger.info("FAISS GPU index initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize FAISS GPU: {e}")
            self.use_gpu = False
    
    async def store_memory_with_embedding(self, 
                                        content: str, 
                                        embedding: np.ndarray,
                                        context: Dict[str, Any]) -> str:
        """Store memory with pre-computed embedding"""
        memory_id = await self.store_memory(content, context)
        
        # Add to HNSW index
        self.hnsw_index.add_items(
            embedding.reshape(1, -1),
            [hash(memory_id) % 2**32]  # Convert to uint32
        )
        
        # Add to GPU index if available
        if self.use_gpu and hasattr(self, 'gpu_index'):
            self.gpu_index.add(embedding.reshape(1, -1))
        
        return memory_id
    
    async def retrieve_memories_by_embedding(self, 
                                           query_embedding: np.ndarray,
                                           k: int = 10) -> List[Tuple[MemoryItem, float]]:
        """Retrieve memories using embedding similarity"""
        # Search in HNSW index
        labels, distances = self.hnsw_index.knn_query(
            query_embedding.reshape(1, -1),
            k=k
        )
        
        # Retrieve actual memories
        memories_with_scores = []
        for label, distance in zip(labels[0], distances[0]):
            # Convert distance to similarity score
            similarity = 1 - distance
            
            # Find memory by hash (simplified - in production use proper mapping)
            for memory_id, memory in self.memories.items():
                if hash(memory_id) % 2**32 == label:
                    memories_with_scores.append((memory, similarity))
                    break
        
        return memories_with_scores
    
    async def consolidate_memories_advanced(self):
        """Advanced memory consolidation with importance weighting"""
        current_time = datetime.now()
        
        # Group memories by semantic similarity
        memory_clusters = await self._cluster_memories()
        
        # Consolidate each cluster
        for cluster_id, cluster_memories in memory_clusters.items():
            # Calculate cluster importance
            cluster_importance = np.mean([
                m.calculate_salience(current_time) for m in cluster_memories
            ])
            
            if cluster_importance < self.consolidation_threshold:
                # Create summary memory
                summary_content = self._summarize_cluster(cluster_memories)
                
                # Store summary with high importance
                await self.store_memory(
                    content=summary_content,
                    context={
                        'type': 'consolidated',
                        'source_memories': [m.id for m in cluster_memories],
                        'importance': min(1.0, cluster_importance * 1.5)
                    }
                )
                
                # Remove original memories
                for memory in cluster_memories:
                    del self.memories[memory.id]
        
        logger.info(f"Consolidated {len(memory_clusters)} memory clusters")
    
    async def _cluster_memories(self) -> Dict[int, List[MemoryItem]]:
        """Cluster memories by semantic similarity"""
        # Simplified clustering - in production use proper clustering algorithms
        clusters = {}
        cluster_id = 0
        
        processed = set()
        
        for memory_id, memory in self.memories.items():
            if memory_id in processed:
                continue
            
            # Create new cluster
            cluster = [memory]
            processed.add(memory_id)
            
            # Find similar memories
            for other_id, other_memory in self.memories.items():
                if other_id in processed:
                    continue
                
                # Simple similarity check
                if self._calculate_similarity(memory, other_memory) > 0.8:
                    cluster.append(other_memory)
                    processed.add(other_id)
            
            if len(cluster) > 1:
                clusters[cluster_id] = cluster
                cluster_id += 1
        
        return clusters
    
    def _calculate_similarity(self, mem1: MemoryItem, mem2: MemoryItem) -> float:
        """Calculate similarity between two memories"""
        # Simplified - in production use proper embeddings
        words1 = set(mem1.content.lower().split())
        words2 = set(mem2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _summarize_cluster(self, memories: List[MemoryItem]) -> str:
        """Create summary of memory cluster"""
        # Simplified summarization
        contents = [m.content for m in memories]
        
        # Find common themes
        all_words = []
        for content in contents:
            all_words.extend(content.lower().split())
        
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        summary = f"Consolidated memory of {len(memories)} related experiences involving: "
        summary += ", ".join([word for word, _ in top_words[:5]])
        
        return summary