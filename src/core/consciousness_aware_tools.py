# src/core/memory_consolidation_enhanced.py
"""
Phase 7: Enhanced Memory Consolidation Pipeline
GPU-accelerated clustering and hierarchical summarization
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

from celery import Celery, Task, group, chain, chord
from celery.result import AsyncResult
from sklearn.cluster import DBSCAN
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import torch
import logging
from dataclasses import dataclass
from collections import defaultdict
import json

# Try to import RAPIDS cuML for GPU acceleration
try:
    import cupy as cp
    import cuml
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    GPU_AVAILABLE = True
    logger.info("RAPIDS cuML available - using GPU acceleration")
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("RAPIDS cuML not available - falling back to CPU")

logger = logging.getLogger(__name__)

# Enhanced Celery configuration for ML workloads
app = Celery('aims_consolidation_enhanced')
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/1',
    task_serializer='json',
    accept_content=['json', 'pickle'],  # Allow pickle for numpy arrays
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Optimized for ML workloads
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=50,  # Restart after 50 tasks to free memory
    
    # Queue configuration with priorities
    task_routes={
        'consolidation.gpu.*': {'queue': 'gpu', 'priority': 10},
        'consolidation.cluster.*': {'queue': 'gpu', 'priority': 8},
        'consolidation.summarize.*': {'queue': 'cpu', 'priority': 6},
        'consolidation.semantic.*': {'queue': 'cpu', 'priority': 4},
        'consolidation.background.*': {'queue': 'background', 'priority': 1}
    },
    
    # Task time limits
    task_time_limit=3600,  # 1 hour hard limit
    task_soft_time_limit=3000,  # 50 min soft limit
    
    # Result expiration
    result_expires=86400,  # 24 hours
    
    # Beat schedule for periodic consolidation
    beat_schedule={
        'rapid-consolidation': {
            'task': 'consolidation.rapid_consolidation',
            'schedule': 300.0,  # Every 5 minutes
            'options': {'queue': 'gpu', 'priority': 9}
        },
        'hourly-consolidation': {
            'task': 'consolidation.hourly_consolidation',
            'schedule': 3600.0,  # Every hour
            'options': {'queue': 'gpu', 'priority': 7}
        },
        'daily-deep-consolidation': {
            'task': 'consolidation.deep_consolidation',
            'schedule': 86400.0,  # Daily at midnight
            'options': {'queue': 'background', 'priority': 3}
        },
        'memory-pruning': {
            'task': 'consolidation.prune_old_memories',
            'schedule': 86400.0,  # Daily
            'options': {'queue': 'background', 'priority': 1}
        }
    }
)

@dataclass
class ConsolidationMetrics:
    """Metrics for consolidation performance"""
    total_memories: int
    clusters_found: int
    semantic_memories_created: int
    compression_ratio: float
    processing_time: float
    gpu_memory_used: float = 0.0
    
class GPUMemoryPool:
    """Manages GPU memory allocation for consolidation tasks"""
    
    def __init__(self, max_memory_mb: int = 20000):  # 20GB for RTX 3090
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.allocated = 0
        self.lock = asyncio.Lock()
        
    async def allocate(self, size_bytes: int) -> bool:
        """Try to allocate GPU memory"""
        async with self.lock:
            if self.allocated + size_bytes <= self.max_memory:
                self.allocated += size_bytes
                return True
            return False
            
    async def release(self, size_bytes: int):
        """Release GPU memory"""
        async with self.lock:
            self.allocated = max(0, self.allocated - size_bytes)

# Global GPU memory pool
gpu_pool = GPUMemoryPool()

class EnhancedConsolidationTask(Task):
    """Base task with GPU support and enhanced caching"""
    
    _embedding_cache = {}
    _model_cache = {}
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if GPU_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
        
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
            if hasattr(cp, 'get_default_memory_pool'):
                cp.get_default_memory_pool().free_all_blocks()

@app.task(base=EnhancedConsolidationTask, bind=True, name='consolidation.gpu.cluster_memories')
def gpu_cluster_memories(self, memory_batch: Dict[str, Any]) -> Dict[str, Any]:
    """GPU-accelerated memory clustering using HDBSCAN"""
    memories = memory_batch['memories']
    
    if len(memories) < 5:
        return {
            'clusters': {0: memories},
            'metrics': {'n_clusters': 1, 'gpu_used': False}
        }
    
    logger.info(f"GPU clustering {len(memories)} memories")
    
    # Extract embeddings
    embeddings = np.array([m['embedding'] for m in memories])
    
    if GPU_AVAILABLE:
        try:
            # Transfer to GPU
            embeddings_gpu = cp.asarray(embeddings)
            
            # Use HDBSCAN for better clustering
            clusterer = cuHDBSCAN(
                min_cluster_size=max(2, len(memories) // 20),
                min_samples=2,
                metric='euclidean',
                cluster_selection_epsilon=0.3,
                cluster_selection_method='eom',  # Excess of Mass
                prediction_data=True
            )
            
            # Perform clustering
            cluster_labels = clusterer.fit_predict(embeddings_gpu)
            
            # Transfer results back to CPU
            cluster_labels = cp.asnumpy(cluster_labels)
            
            # Get cluster probabilities
            probabilities = cp.asnumpy(clusterer.probabilities_)
            
            gpu_memory_used = self.get_gpu_memory_usage()
            self.clear_gpu_cache()
            
        except Exception as e:
            logger.warning(f"GPU clustering failed, falling back to CPU: {e}")
            return cpu_fallback_clustering(embeddings, memories)
    else:
        return cpu_fallback_clustering(embeddings, memories)
    
    # Organize memories by cluster
    clusters = defaultdict(list)
    noise_points = []
    
    for i, (label, prob) in enumerate(zip(cluster_labels, probabilities)):
        memory = memories[i].copy()
        memory['cluster_probability'] = float(prob)
        
        if label == -1:
            noise_points.append(memory)
        else:
            clusters[int(label)].append(memory)
    
    # Calculate cluster quality metrics
    cluster_metrics = {}
    for cluster_id, cluster_memories in clusters.items():
        cluster_embeddings = np.array([m['embedding'] for m in cluster_memories])
        
        # Intra-cluster distance (cohesion)
        centroid = np.mean(cluster_embeddings, axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=2)
        
        cluster_metrics[cluster_id] = {
            'size': len(cluster_memories),
            'cohesion': float(np.mean(distances)),
            'avg_probability': float(np.mean([m['cluster_probability'] for m in cluster_memories]))
        }
    
    return {
        'clusters': dict(clusters),
        'noise': noise_points,
        'metrics': {
            'n_clusters': len(clusters),
            'n_noise': len(noise_points),
            'gpu_used': True,
            'gpu_memory_mb': gpu_memory_used,
            'cluster_quality': cluster_metrics
        }
    }

def cpu_fallback_clustering(embeddings: np.ndarray, memories: List[Dict]) -> Dict[str, Any]:
    """CPU fallback using standard HDBSCAN"""
    from sklearn.cluster import HDBSCAN
    
    clusterer = HDBSCAN(
        min_cluster_size=max(2, len(memories) // 20),
        min_samples=2,
        metric='euclidean',
        cluster_selection_epsilon=0.3
    )
    
    cluster_labels = clusterer.fit_predict(embeddings)
    
    clusters = defaultdict(list)
    noise_points = []
    
    for i, label in enumerate(cluster_labels):
        if label == -1:
            noise_points.append(memories[i])
        else:
            clusters[int(label)].append(memories[i])
    
    return {
        'clusters': dict(clusters),
        'noise': noise_points,
        'metrics': {
            'n_clusters': len(clusters),
            'n_noise': len(noise_points),
            'gpu_used': False
        }
    }

@app.task(base=EnhancedConsolidationTask, bind=True, name='consolidation.summarize.hierarchical')
def hierarchical_summarize_cluster(self, cluster_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced hierarchical summarization with importance weighting"""
    cluster_id = cluster_data['cluster_id']
    memories = cluster_data['memories']
    cluster_metrics = cluster_data.get('metrics', {})
    
    logger.info(f"Hierarchical summarization of cluster {cluster_id} ({len(memories)} memories)")
    
    # Sort memories by importance and recency
    sorted_memories = sorted(
        memories,
        key=lambda m: (
            m.get('importance', 0.5) * 0.6 + 
            m.get('cluster_probability', 0.5) * 0.4
        ),
        reverse=True
    )
    
    # Implement multi-level summarization
    summaries = []
    
    # Level 1: Group by semantic similarity within cluster
    subgroups = create_semantic_subgroups(sorted_memories)
    
    # Level 2: Summarize each subgroup
    for subgroup in subgroups:
        if len(subgroup) > 1:
            subgroup_summary = summarize_memory_group(subgroup)
            summaries.append(subgroup_summary)
        else:
            summaries.append(subgroup[0]['content'])
    
    # Level 3: Create meta-summary
    if len(summaries) > 3:
        meta_summary = create_meta_summary(summaries)
    else:
        meta_summary = ' '.join(summaries)
    
    # Extract key insights
    insights = extract_key_insights(sorted_memories)
    
    # Calculate information retention
    original_length = sum(len(m['content']) for m in memories)
    summary_length = len(meta_summary)
    retention_ratio = summary_length / max(original_length, 1)
    
    return {
        'cluster_id': cluster_id,
        'summary': meta_summary,
        'insights': insights,
        'subgroup_summaries': summaries[:5],  # Top 5 subgroup summaries
        'metrics': {
            'original_memories': len(memories),
            'subgroups_found': len(subgroups),
            'compression_ratio': 1 - retention_ratio,
            'avg_importance': np.mean([m.get('importance', 0.5) for m in memories]),
            'cluster_metrics': cluster_metrics
        },
        'source_memory_ids': [m['id'] for m in sorted_memories[:10]]  # Top 10 most important
    }

def create_semantic_subgroups(memories: List[Dict], max_subgroups: int = 5) -> List[List[Dict]]:
    """Create semantic subgroups within a cluster"""
    if len(memories) <= max_subgroups:
        return [[m] for m in memories]
    
    # Use mini-batch k-means for efficiency
    from sklearn.cluster import MiniBatchKMeans
    
    embeddings = np.array([m['embedding'] for m in memories])
    n_clusters = min(max_subgroups, len(memories) // 2)
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    subgroups = defaultdict(list)
    for i, label in enumerate(labels):
        subgroups[label].append(memories[i])
    
    return list(subgroups.values())

def summarize_memory_group(memories: List[Dict]) -> str:
    """Summarize a group of related memories"""
    # Extract key information
    contents = [m['content'] for m in memories]
    importance_weights = [m.get('importance', 0.5) for m in memories]
    
    # Weighted combination (simplified - in production use transformer model)
    weighted_contents = []
    for content, weight in zip(contents, importance_weights):
        # Take more from important memories
        char_limit = int(100 * weight)
        weighted_contents.append(content[:char_limit])
    
    combined = ' '.join(weighted_contents)
    
    # Simple extractive summary (in production, use BART or T5)
    sentences = combined.split('. ')
    if len(sentences) > 3:
        # Take first, middle, and last sentence
        summary = f"{sentences[0]}. {sentences[len(sentences)//2]}. {sentences[-1]}"
    else:
        summary = combined
    
    return summary.strip()

def create_meta_summary(summaries: List[str]) -> str:
    """Create a meta-summary from multiple summaries"""
    combined = ' '.join(summaries)
    
    # Extract most common themes
    words = combined.lower().split()
    word_freq = defaultdict(int)
    
    for word in words:
        if len(word) > 4:  # Focus on meaningful words
            word_freq[word] += 1
    
    # Get top themes
    top_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    theme_words = [word for word, _ in top_themes]
    
    # Create summary focusing on themes
    theme_sentence = f"Key themes include: {', '.join(theme_words)}."
    
    # Add first and last summary sentences
    first_sentence = summaries[0].split('.')[0] + '.'
    last_sentence = summaries[-1].split('.')[-1].strip() + '.'
    
    return f"{first_sentence} {theme_sentence} {last_sentence}"

def extract_key_insights(memories: List[Dict]) -> List[str]:
    """Extract key insights from memories"""
    insights = []
    
    # Emotional patterns
    emotions = [m.get('emotional_state', {}) for m in memories if 'emotional_state' in m]
    if emotions:
        avg_pleasure = np.mean([e.get('pleasure', 0.5) for e in emotions])
        if avg_pleasure > 0.7:
            insights.append("Predominantly positive emotional context")
        elif avg_pleasure < 0.3:
            insights.append("Challenging emotional period identified")
    
    # Importance patterns
    importances = [m.get('importance', 0.5) for m in memories]
    if max(importances) > 0.8:
        insights.append("Contains highly significant memories")
    
    # Temporal patterns
    if 'timestamp' in memories[0]:
        timestamps = [m['timestamp'] for m in memories]
        # Check for temporal clustering
        insights.append(f"Memories span {len(set(timestamps))} distinct time periods")
    
    return insights

@app.task(bind=True, name='consolidation.semantic.generate')
def generate_semantic_memories(self, summarized_clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate semantic memories from summarized clusters"""
    semantic_memories = []
    
    for cluster in summarized_clusters:
        # Skip low-quality clusters
        if cluster['metrics']['avg_importance'] < 0.3:
            continue
        
        semantic_memory = {
            'id': f"semantic_{int(datetime.now().timestamp())}_{cluster['cluster_id']}",
            'content': cluster['summary'],
            'type': 'semantic',
            'category': determine_memory_category(cluster),
            'insights': cluster['insights'],
            'confidence': calculate_confidence(cluster),
            'importance': cluster['metrics']['avg_importance'],
            'source': 'consolidation',
            'metadata': {
                'cluster_id': cluster['cluster_id'],
                'original_count': cluster['metrics']['original_memories'],
                'compression_ratio': cluster['metrics']['compression_ratio'],
                'subgroup_summaries': cluster['subgroup_summaries'],
                'consolidation_time': datetime.now().isoformat()
            },
            'related_concepts': extract_concepts(cluster),
            'consolidated_from': cluster['source_memory_ids']
        }
        
        semantic_memories.append(semantic_memory)
    
    return {
        'semantic_memories': semantic_memories,
        'total_generated': len(semantic_memories),
        'timestamp': datetime.now().isoformat()
    }

def determine_memory_category(cluster: Dict[str, Any]) -> str:
    """Determine category based on cluster content"""
    summary = cluster['summary'].lower()
    insights = ' '.join(cluster['insights']).lower()
    combined = f"{summary} {insights}"
    
    categories = {
        'technical': ['code', 'system', 'error', 'function', 'algorithm'],
        'emotional': ['feel', 'emotion', 'happy', 'sad', 'stress'],
        'learning': ['learn', 'understand', 'discover', 'realize', 'insight'],
        'social': ['conversation', 'discuss', 'share', 'connect', 'relationship'],
        'planning': ['goal', 'plan', 'future', 'intend', 'strategy']
    }
    
    scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in combined)
        scores[category] = score
    
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    return 'general'

def calculate_confidence(cluster: Dict[str, Any]) -> float:
    """Calculate confidence score for semantic memory"""
    base_confidence = 0.7
    
    # Adjust based on cluster quality
    if 'cluster_metrics' in cluster['metrics']:
        metrics = cluster['metrics']['cluster_metrics']
        avg_probability = np.mean([m.get('avg_probability', 0.5) for m in metrics.values()])
        base_confidence = base_confidence * 0.5 + avg_probability * 0.5
    
    # Adjust based on compression ratio
    compression = cluster['metrics']['compression_ratio']
    if compression > 0.9:  # Very high compression
        base_confidence *= 0.9
    elif compression < 0.5:  # Low compression
        base_confidence *= 1.1
    
    return min(0.95, base_confidence)

def extract_concepts(cluster: Dict[str, Any]) -> List[str]:
    """Extract key concepts from cluster"""
    # Combine summary and insights
    text = f"{cluster['summary']} {' '.join(cluster['insights'])}"
    
    # Simple concept extraction (in production use NER)
    words = text.split()
    concepts = []
    
    # Look for capitalized words (potential entities)
    for word in words:
        if word[0].isupper() and len(word) > 3:
            concepts.append(word)
    
    # Look for repeated meaningful words
    word_counts = defaultdict(int)
    for word in words:
        if len(word) > 5:
            word_counts[word.lower()] += 1
    
    # Add frequently mentioned words
    frequent_words = [w for w, c in word_counts.items() if c >= 2]
    concepts.extend(frequent_words)
    
    return list(set(concepts))[:10]  # Limit to 10 concepts

# Workflow orchestration
@app.task(bind=True, name='consolidation.rapid_consolidation')
def rapid_consolidation(self, session_ids: List[str]) -> Dict[str, Any]:
    """Rapid consolidation for recent memories (5-minute intervals)"""
    logger.info(f"Starting rapid consolidation for {len(session_ids)} sessions")
    
    consolidation_workflow = chain(
        extract_recent_memories.s(session_ids, time_window_minutes=5),
        gpu_cluster_memories.s(),
        process_clusters.s(),
        generate_semantic_memories.s()
    )
    
    result = consolidation_workflow.apply_async()
    
    return {
        'task_id': result.id,
        'session_count': len(session_ids),
        'type': 'rapid',
        'timestamp': datetime.now().isoformat()
    }

@app.task(bind=True, name='consolidation.extract_recent_memories')
def extract_recent_memories(self, session_ids: List[str], 
                           time_window_minutes: int = 60) -> Dict[str, Any]:
    """Extract recent memories for consolidation"""
    # This would interface with your memory manager
    # Placeholder implementation
    memories = []
    
    for session_id in session_ids:
        # In production, query your memory store
        session_memories = [
            {
                'id': f"mem_{session_id}_{i}",
                'content': f"Memory content {i}",
                'embedding': np.random.randn(768).tolist(),  # Placeholder
                'importance': np.random.random(),
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            }
            for i in range(np.random.randint(5, 20))
        ]
        memories.extend(session_memories)
    
    return {
        'memories': memories,
        'total_count': len(memories),
        'time_window': time_window_minutes
    }

@app.task(bind=True, name='consolidation.process_clusters')
def process_clusters(self, cluster_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process clusters through hierarchical summarization"""
    clusters = cluster_result['clusters']
    
    # Create parallel summarization tasks
    summarization_tasks = []
    
    for cluster_id, memories in clusters.items():
        cluster_data = {
            'cluster_id': cluster_id,
            'memories': memories,
            'metrics': cluster_result['metrics'].get('cluster_quality', {}).get(cluster_id, {})
        }
        
        summarization_tasks.append(
            hierarchical_summarize_cluster.s(cluster_data)
        )
    
    # Execute in parallel
    job = group(summarization_tasks).apply_async()
    results = job.get()
    
    return results

# Monitoring and metrics
@app.task(bind=True, name='consolidation.get_metrics')
def get_consolidation_metrics(self) -> Dict[str, Any]:
    """Get current consolidation metrics"""
    return {
        'gpu_available': GPU_AVAILABLE,
        'gpu_memory_usage_mb': self.get_gpu_memory_usage() if GPU_AVAILABLE else 0,
        'active_tasks': app.control.inspect().active(),
        'scheduled_tasks': app.control.inspect().scheduled(),
        'timestamp': datetime.now().isoformat()
    }