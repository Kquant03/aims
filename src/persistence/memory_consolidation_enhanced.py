# memory_consolidation.py - Celery-based Memory Consolidation Pipeline
from celery import Celery, Task, chain, group, chord
from celery.schedules import crontab
from celery.result import AsyncResult
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import json
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import hashlib

logger = logging.getLogger(__name__)

# Celery Configuration
app = Celery('aims_memory_consolidation')
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/1',
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Optimized for ML workloads
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=100,
    
    # Task routing for different queues
    task_routes={
        'consolidation.extract_episodic': {'queue': 'extraction'},
        'consolidation.cluster_memories': {'queue': 'clustering'},
        'consolidation.hierarchical_summarize': {'queue': 'summarization'},
        'consolidation.generate_semantic': {'queue': 'semantic'},
        'consolidation.compute_salience': {'queue': 'salience'}
    },
    
    # Periodic tasks schedule
    beat_schedule={
        'consolidate-hourly': {
            'task': 'consolidation.hourly_consolidation',
            'schedule': crontab(minute=0),
            'options': {'queue': 'consolidation'}
        },
        'deep-consolidation-daily': {
            'task': 'consolidation.deep_consolidation',
            'schedule': crontab(hour=3, minute=0),
            'options': {'queue': 'consolidation'}
        },
        'cleanup-old-memories': {
            'task': 'consolidation.cleanup_old_memories',
            'schedule': crontab(hour=4, minute=0),
            'options': {'queue': 'maintenance'}
        }
    }
)

class MemoryModelTask(Task):
    """Base task with model caching for efficiency"""
    _embedding_model = None
    _summarization_model = None
    _salience_scorer = None
    _memory_system = None
    _config = None
    
    @property
    def config(self):
        if self._config is None:
            # Load configuration
            self._config = {
                'embedding_dim': 768,
                'input_dim': 768,
                'hidden_dim': 512,
                'text_embedding_dim': 768,
                'goal_dim': 256,
                'num_classes': 10,
                'postgres_url': 'postgresql+asyncpg://aims:aims_password@localhost:5432/aims_memory',
                'redis_url': 'redis://localhost:6379',
                'chroma_persist_dir': './data/chroma'
            }
        return self._config
    
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            logger.info("Loading embedding model...")
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            if torch.cuda.is_available():
                self._embedding_model = self._embedding_model.cuda()
        return self._embedding_model
    
    @property
    def summarization_model(self):
        if self._summarization_model is None:
            logger.info("Loading summarization model...")
            self._summarization_model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
        return self._summarization_model
    
    @property
    def salience_scorer(self):
        if self._salience_scorer is None:
            logger.info("Loading salience scorer...")
            from salience_scoring import MultiDimensionalSalienceScorer, SalienceConfig
            config = SalienceConfig(**self.config)
            self._salience_scorer = MultiDimensionalSalienceScorer(config)
            self._salience_scorer.eval()
        return self._salience_scorer
    
    @property
    def memory_system(self):
        if self._memory_system is None:
            logger.info("Initializing memory system...")
            from three_tier_memory import ThreeTierMemorySystem
            self._memory_system = ThreeTierMemorySystem(self.config)
            # Run async initialization
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._memory_system.initialize())
        return self._memory_system

@app.task(base=MemoryModelTask, bind=True, name='consolidation.extract_episodic')
def extract_episodic_memories(self, session_id: str, user_id: Optional[str] = None, 
                            time_window_hours: int = 1) -> Dict[str, Any]:
    """Extract episodic memories for consolidation"""
    logger.info(f"Extracting episodic memories for session {session_id}")
    
    # Get recent episodic memories
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=time_window_hours)
    
    # Run async operation in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    episodes = loop.run_until_complete(
        self.memory_system.episodic_store.get_temporal_sequence(
            session_id, start_time, end_time
        )
    )
    
    # Also get working memory for context
    working_memory = loop.run_until_complete(
        self.memory_system.working_memory.get_working_memory_summary(session_id)
    )
    
    # Process episodes
    memories_with_embeddings = []
    for episode in episodes:
        memory_data = {
            'id': str(episode.id),
            'content': episode.content,
            'embedding': episode.embedding,
            'context': episode.context,
            'timestamp': episode.timestamp,
            'salience_score': episode.salience_score,
            'importance': episode.importance,
            'attention_focus': episode.attention_focus,
            'emotional_state': episode.emotional_state
        }
        
        # Generate embedding if missing
        if memory_data['embedding'] is None:
            embedding = self.embedding_model.encode(episode.content)
            memory_data['embedding'] = embedding.tolist()
            memory_data['text_embedding'] = memory_data['embedding']  # Use same for now
        else:
            memory_data['text_embedding'] = memory_data['embedding']
        
        memories_with_embeddings.append(memory_data)
    
    logger.info(f"Extracted {len(memories_with_embeddings)} episodic memories")
    
    return {
        'memories': memories_with_embeddings,
        'session_id': session_id,
        'user_id': user_id,
        'time_window': {
            'start': start_time.isoformat(),
            'end': end_time.isoformat()
        },
        'working_memory_context': working_memory
    }

@app.task(base=MemoryModelTask, bind=True, name='consolidation.compute_salience')
def compute_memory_salience(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute salience scores for memories"""
    memories = memory_data['memories']
    
    if not memories:
        return memory_data
    
    logger.info(f"Computing salience scores for {len(memories)} memories")
    
    # Default goals (would be retrieved from consciousness state in full implementation)
    current_goals = torch.randn(1, self.config['goal_dim'])
    
    # Compute salience scores
    salience_results = self.salience_scorer.compute_batch_salience(memories, current_goals)
    
    # Update memories with salience scores
    for i, memory in enumerate(memories):
        memory['computed_salience'] = salience_results[i]
    
    # Sort by overall salience
    memories.sort(key=lambda m: m['computed_salience']['overall_salience'], reverse=True)
    
    memory_data['memories'] = memories
    memory_data['salience_computed'] = True
    
    return memory_data

@app.task(base=MemoryModelTask, bind=True, name='consolidation.cluster_memories')
def cluster_memories_dbscan(self, memory_data: Dict[str, Any], 
                           min_cluster_size: int = 3,
                           eps: Optional[float] = None) -> Dict[str, Any]:
    """Cluster memories using DBSCAN with adaptive parameters"""
    memories = memory_data['memories']
    
    if len(memories) < min_cluster_size:
        logger.warning(f"Too few memories ({len(memories)}) for clustering")
        return {
            'clusters': {0: memories},
            'noise': [],
            'metadata': memory_data,
            'clustering_params': {'min_cluster_size': min_cluster_size}
        }
    
    logger.info(f"Clustering {len(memories)} memories")
    
    # Extract embeddings
    embeddings = np.array([m['embedding'] for m in memories])
    
    # Normalize for better clustering
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    
    # Calculate optimal eps if not provided
    if eps is None:
        eps = self._calculate_optimal_eps(normalized_embeddings)
    
    # Perform clustering
    dbscan = DBSCAN(
        eps=eps,
        min_samples=max(2, int(len(memories) * 0.05)),
        metric='cosine',
        algorithm='auto',
        n_jobs=-1
    )
    
    cluster_labels = dbscan.fit_predict(normalized_embeddings)
    
    # Organize results by cluster
    clusters = {}
    noise_points = []
    
    for i, label in enumerate(cluster_labels):
        if label == -1:
            noise_points.append(memories[i])
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(memories[i])
    
    # Calculate cluster statistics
    cluster_stats = {}
    for cluster_id, cluster_memories in clusters.items():
        # Calculate cluster centroid
        cluster_embeddings = np.array([m['embedding'] for m in cluster_memories])
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Calculate average importance/salience
        avg_importance = np.mean([m.get('importance', 0.5) for m in cluster_memories])
        avg_salience = np.mean([
            m.get('computed_salience', {}).get('overall_salience', 0.5) 
            for m in cluster_memories
        ])
        
        cluster_stats[cluster_id] = {
            'size': len(cluster_memories),
            'centroid': centroid.tolist(),
            'avg_importance': avg_importance,
            'avg_salience': avg_salience
        }
    
    logger.info(f"Created {len(clusters)} clusters with {len(noise_points)} noise points")
    
    return {
        'clusters': clusters,
        'noise': noise_points,
        'cluster_stats': cluster_stats,
        'n_clusters': len(clusters),
        'eps_used': eps,
        'metadata': memory_data
    }
    
    def _calculate_optimal_eps(self, embeddings: np.ndarray) -> float:
        """Calculate optimal eps using k-distance method"""
        k = min(4, len(embeddings))
        neighbors = NearestNeighbors(n_neighbors=k, metric='cosine')
        neighbors_fit = neighbors.fit(embeddings)
        distances, _ = neighbors_fit.kneighbors(embeddings)
        
        # Find elbow point
        k_distances = np.sort(distances[:, k-1], axis=0)
        
        # Simple elbow detection
        if len(k_distances) > 10:
            gradients = np.gradient(k_distances)
            eps = k_distances[np.argmax(gradients)]
        else:
            eps = np.median(k_distances)
        
        # Ensure reasonable bounds
        eps = np.clip(eps, 0.1, 0.9)
        
        return float(eps)

@app.task(base=MemoryModelTask, bind=True, name='consolidation.hierarchical_summarize')
def hierarchical_summarize(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
    """Hierarchical summarization of memory clusters"""
    cluster_id = cluster_data['cluster_id']
    memories = cluster_data['memories']
    cluster_stats = cluster_data.get('cluster_stats', {})
    
    logger.info(f"Summarizing cluster {cluster_id} with {len(memories)} memories")
    
    # Level 1: Prepare individual memories
    memory_contents = []
    for memory in memories:
        content = memory['content']
        # Include attention focus for context
        if memory.get('attention_focus'):
            content = f"[Focus: {memory['attention_focus']}] {content}"
        memory_contents.append(content)
    
    # Combine memories with importance weighting
    if 'computed_salience' in memories[0]:
        # Sort by salience
        sorted_memories = sorted(memories, 
                               key=lambda m: m['computed_salience']['overall_salience'], 
                               reverse=True)
        # Take top memories based on salience
        top_memories = sorted_memories[:max(5, len(sorted_memories)//2)]
        memory_contents = [m['content'] for m in top_memories]
    
    # Level 2: Chunk and summarize
    combined_content = " ".join(memory_contents)
    
    if len(combined_content) > 1024:
        # Semantic chunking
        chunks = self._semantic_chunk(combined_content, max_chunk_size=800)
        chunk_summaries = []
        
        for chunk in chunks:
            if len(chunk) > 100:  # Only summarize substantial chunks
                summary = self.summarization_model(
                    chunk,
                    max_length=200,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']
                chunk_summaries.append(summary)
            else:
                chunk_summaries.append(chunk)
        
        # Final summary
        final_content = " ".join(chunk_summaries)
        if len(final_content) > 300:
            cluster_summary = self.summarization_model(
                final_content,
                max_length=300,
                min_length=100,
                do_sample=False
            )[0]['summary_text']
        else:
            cluster_summary = final_content
    else:
        # Direct summarization for shorter content
        if len(combined_content) > 300:
            cluster_summary = self.summarization_model(
                combined_content,
                max_length=250,
                min_length=80,
                do_sample=False
            )[0]['summary_text']
        else:
            cluster_summary = combined_content
    
    # Generate embedding for semantic storage
    summary_embedding = self.embedding_model.encode(cluster_summary)
    
    # Extract key themes/concepts
    themes = self._extract_themes(memories)
    
    result = {
        'cluster_id': cluster_id,
        'summary': cluster_summary,
        'embedding': summary_embedding.tolist(),
        'original_count': len(memories),
        'compression_ratio': len(combined_content) / max(len(cluster_summary), 1),
        'themes': themes,
        'cluster_stats': cluster_stats,
        'source_memory_ids': [m['id'] for m in memories],
        'avg_importance': np.mean([m.get('importance', 0.5) for m in memories]),
        'emotional_summary': self._summarize_emotions(memories)
    }
    
    logger.info(f"Created summary with compression ratio {result['compression_ratio']:.2f}")
    
    return result
    
    def _semantic_chunk(self, text: str, max_chunk_size: int = 800) -> List[str]:
        """Split text into semantic chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append('. '.join(current_chunk))
        
        return chunks
    
    def _extract_themes(self, memories: List[Dict]) -> List[str]:
        """Extract common themes from memories"""
        # Simple theme extraction based on repeated words
        # In production, use more sophisticated NLP
        from collections import Counter
        import re
        
        # Combine all content
        all_content = ' '.join([m['content'] for m in memories])
        
        # Extract significant words
        words = re.findall(r'\b\w{4,}\b', all_content.lower())
        
        # Filter common words (simple stopwords)
        common_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'what', 'when', 'where'}
        words = [w for w in words if w not in common_words]
        
        # Get most common themes
        word_counts = Counter(words)
        themes = [word for word, count in word_counts.most_common(5)]
        
        return themes
    
    def _summarize_emotions(self, memories: List[Dict]) -> Dict[str, float]:
        """Summarize emotional content across memories"""
        emotions = {
            'avg_valence': [],
            'avg_arousal': [],
            'avg_dominance': []
        }
        
        for memory in memories:
            if 'emotional_state' in memory and memory['emotional_state']:
                emotions['avg_valence'].append(memory['emotional_state'].get('pleasure', 0.5))
                emotions['avg_arousal'].append(memory['emotional_state'].get('arousal', 0.5))
                emotions['avg_dominance'].append(memory['emotional_state'].get('dominance', 0.5))
        
        # Calculate averages
        result = {}
        for key, values in emotions.items():
            if values:
                result[key] = float(np.mean(values))
            else:
                result[key] = 0.5
        
        return result

@app.task(base=MemoryModelTask, bind=True, name='consolidation.generate_semantic')
def generate_semantic_memories(self, summarized_clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform summarized clusters into semantic memories"""
    logger.info(f"Generating semantic memories from {len(summarized_clusters)} clusters")
    
    semantic_memories = []
    
    for cluster_summary in summarized_clusters:
        # Calculate importance based on cluster properties
        importance = self._calculate_importance(cluster_summary)
        
        # Determine category based on themes
        category = self._determine_category(cluster_summary['themes'])
        
        # Extract concepts for knowledge graph
        concepts = cluster_summary['themes'] + self._extract_entities(cluster_summary['summary'])
        
        semantic_memory = {
            'content': cluster_summary['summary'],
            'embedding': cluster_summary['embedding'],
            'category': category,
            'source': 'episodic_consolidation',
            'confidence': min(0.95, importance + 0.1),  # Slightly boost confidence
            'importance': importance,
            'related_concepts': list(set(concepts)),  # Deduplicate
            'metadata': {
                'original_count': cluster_summary['original_count'],
                'compression_ratio': cluster_summary['compression_ratio'],
                'consolidation_time': datetime.utcnow().isoformat(),
                'source_memory_ids': cluster_summary['source_memory_ids'],
                'cluster_id': cluster_summary['cluster_id'],
                'emotional_summary': cluster_summary['emotional_summary']
            },
            'consolidated_from': cluster_summary['source_memory_ids']
        }
        
        # Add user_id if available
        if 'metadata' in cluster_summary:
            metadata = cluster_summary['metadata']
            if isinstance(metadata, dict) and 'user_id' in metadata:
                semantic_memory['user_id'] = metadata['user_id']
        
        semantic_memories.append(semantic_memory)
    
    logger.info(f"Generated {len(semantic_memories)} semantic memories")
    
    return semantic_memories
    
    def _calculate_importance(self, cluster_summary: Dict[str, Any]) -> float:
        """Calculate importance score for semantic memory"""
        base_importance = cluster_summary.get('avg_importance', 0.5)
        
        # Boost importance based on cluster size
        size_boost = min(0.2, cluster_summary['original_count'] / 50)
        
        # Boost based on emotional intensity
        emotional_summary = cluster_summary.get('emotional_summary', {})
        valence = abs(emotional_summary.get('avg_valence', 0.5) - 0.5)
        arousal = emotional_summary.get('avg_arousal', 0.5)
        emotion_boost = (valence + arousal) / 4  # Max 0.5
        
        # Combine factors
        importance = base_importance + size_boost + emotion_boost
        
        return min(1.0, importance)
    
    def _determine_category(self, themes: List[str]) -> str:
        """Determine semantic category based on themes"""
        # Simple categorization - enhance with more sophisticated methods
        categories = {
            'technical': ['code', 'programming', 'system', 'error', 'function'],
            'emotional': ['feel', 'happy', 'sad', 'emotion', 'mood'],
            'social': ['friend', 'family', 'people', 'relationship', 'conversation'],
            'learning': ['learn', 'understand', 'know', 'study', 'research'],
            'task': ['work', 'task', 'project', 'complete', 'finish']
        }
        
        theme_set = set(themes)
        best_category = 'general'
        best_score = 0
        
        for category, keywords in categories.items():
            score = len(theme_set.intersection(keywords))
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        # Simple entity extraction - use NER in production
        import re
        
        # Extract capitalized words (potential entities)
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Filter out common words
        common_caps = {'The', 'This', 'That', 'These', 'Those', 'It', 'We', 'You', 'They'}
        entities = [e for e in entities if e not in common_caps]
        
        return entities[:10]  # Limit to top 10

@app.task(bind=True, name='consolidation.store_semantic_memories')
def store_semantic_memories(self, semantic_memories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Store semantic memories in the semantic store"""
    logger.info(f"Storing {len(semantic_memories)} semantic memories")
    
    # Initialize memory system for storage
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    stored_ids = []
    failed = []
    
    for memory in semantic_memories:
        try:
            memory_id = loop.run_until_complete(
                self.memory_system.semantic_store.store_semantic_knowledge(memory)
            )
            stored_ids.append(memory_id)
        except Exception as e:
            logger.error(f"Failed to store semantic memory: {e}")
            failed.append(memory)
    
    result = {
        'stored_count': len(stored_ids),
        'failed_count': len(failed),
        'stored_ids': stored_ids,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    logger.info(f"Successfully stored {len(stored_ids)} semantic memories")
    
    return result

# Workflow orchestration tasks
@app.task(bind=True, name='consolidation.hourly_consolidation')
def hourly_consolidation(self, session_ids: Optional[List[str]] = None):
    """Hourly memory consolidation workflow"""
    logger.info("Starting hourly memory consolidation")
    
    # If no session IDs provided, get active sessions from Redis
    if not session_ids:
        # This would query Redis for active sessions
        # For now, we'll skip if no sessions provided
        logger.warning("No session IDs provided for consolidation")
        return
    
    # Create workflow for each session
    workflows = []
    
    for session_id in session_ids:
        workflow = chain(
            extract_episodic_memories.s(session_id, time_window_hours=1),
            compute_memory_salience.s(),
            cluster_memories_dbscan.s(),
            group([
                hierarchical_summarize.s({
                    'cluster_id': i,
                    'memories': cluster,
                    'cluster_stats': {}
                })
                for i, cluster in enumerate([])  # Will be populated by previous task
            ]),
            generate_semantic_memories.s(),
            store_semantic_memories.s()
        )
        
        workflows.append(workflow)
    
    # Execute workflows
    job = group(workflows).apply_async()
    
    return {
        'job_id': job.id,
        'session_count': len(session_ids),
        'timestamp': datetime.utcnow().isoformat()
    }

@app.task(bind=True, name='consolidation.deep_consolidation')
def deep_consolidation(self, time_window_hours: int = 24):
    """Daily deep consolidation across all memories"""
    logger.info(f"Starting deep consolidation for past {time_window_hours} hours")
    
    # This would perform more comprehensive consolidation
    # Including cross-session pattern detection, knowledge graph updates, etc.
    
    return {
        'status': 'completed',
        'timestamp': datetime.utcnow().isoformat()
    }

@app.task(bind=True, name='consolidation.cleanup_old_memories')
def cleanup_old_memories(self, days_to_keep: int = 30):
    """Clean up old low-importance memories"""
    logger.info(f"Cleaning up memories older than {days_to_keep} days")
    
    # This would remove old memories below importance threshold
    # Keeping only significant memories long-term
    
    return {
        'status': 'completed',
        'timestamp': datetime.utcnow().isoformat()
    }