# consciousness_aware_tools.py - Fixed version without RAPIDS dependency
"""
Memory Consolidation with CPU fallback
"""
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
from sklearn.cluster import HDBSCAN, DBSCAN
import asyncio

logger = logging.getLogger(__name__)

# Try to import GPU libraries (optional)
GPU_AVAILABLE = False
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        logger.info("GPU available for acceleration")
except ImportError:
    logger.info("PyTorch not available, using CPU only")

@dataclass
class ConsolidationMetrics:
    """Metrics for consolidation performance"""
    total_memories: int
    clusters_found: int
    semantic_memories_created: int
    compression_ratio: float
    processing_time: float
    gpu_memory_used: float = 0.0

def cpu_cluster_memories(memories: List[Dict], min_cluster_size: int = 3) -> Dict[str, Any]:
    """CPU-based memory clustering using HDBSCAN"""
    if len(memories) < min_cluster_size:
        return {
            'clusters': {0: memories},
            'noise': [],
            'metrics': {'n_clusters': 1, 'gpu_used': False}
        }
    
    logger.info(f"Clustering {len(memories)} memories using CPU")
    
    # Extract embeddings
    embeddings = np.array([m['embedding'] for m in memories])
    
    # Use HDBSCAN for clustering
    clusterer = HDBSCAN(
        min_cluster_size=max(2, len(memories) // 20),
        min_samples=2,
        metric='euclidean',
        cluster_selection_epsilon=0.3
    )
    
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Organize by cluster
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

def summarize_memory_group(memories: List[Dict]) -> str:
    """Summarize a group of related memories"""
    contents = [m['content'] for m in memories]
    importance_weights = [m.get('importance', 0.5) for m in memories]
    
    # Weighted combination
    weighted_contents = []
    for content, weight in zip(contents, importance_weights):
        char_limit = int(100 * weight)
        weighted_contents.append(content[:char_limit])
    
    combined = ' '.join(weighted_contents)
    
    # Simple extractive summary
    sentences = combined.split('. ')
    if len(sentences) > 3:
        summary = f"{sentences[0]}. {sentences[len(sentences)//2]}. {sentences[-1]}"
    else:
        summary = combined
    
    return summary.strip()

def extract_themes(memories: List[Dict]) -> List[str]:
    """Extract common themes from memories"""
    from collections import Counter
    import re
    
    all_content = ' '.join([m['content'] for m in memories])
    words = re.findall(r'\b\w{4,}\b', all_content.lower())
    
    # Filter common words
    common_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'what', 'when', 'where'}
    words = [w for w in words if w not in common_words]
    
    word_counts = Counter(words)
    themes = [word for word, count in word_counts.most_common(5)]
    
    return themes

class MemoryConsolidationPipeline:
    """Memory consolidation without GPU dependencies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def consolidate_memories(self, memories: List[Dict], 
                                 session_id: str) -> Dict[str, Any]:
        """Consolidate memories into semantic knowledge"""
        start_time = datetime.now()
        
        # Cluster memories
        cluster_result = cpu_cluster_memories(memories)
        
        # Summarize each cluster
        semantic_memories = []
        
        for cluster_id, cluster_memories in cluster_result['clusters'].items():
            if len(cluster_memories) > 1:
                summary = summarize_memory_group(cluster_memories)
                themes = extract_themes(cluster_memories)
                
                semantic_memory = {
                    'cluster_id': cluster_id,
                    'summary': summary,
                    'themes': themes,
                    'source_count': len(cluster_memories),
                    'importance': np.mean([m.get('importance', 0.5) for m in cluster_memories]),
                    'timestamp': datetime.now().isoformat()
                }
                
                semantic_memories.append(semantic_memory)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'semantic_memories': semantic_memories,
            'metrics': ConsolidationMetrics(
                total_memories=len(memories),
                clusters_found=len(cluster_result['clusters']),
                semantic_memories_created=len(semantic_memories),
                compression_ratio=len(semantic_memories) / max(len(memories), 1),
                processing_time=processing_time
            ).__dict__
        }
