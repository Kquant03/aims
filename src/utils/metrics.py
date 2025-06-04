"""
metrics.py - Metrics collection and monitoring for AIMS
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and tracks metrics for consciousness system"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.start_time = datetime.now()
    
    def record(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': datetime.now(),
            'tags': tags or {}
        })
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if metric_name not in self.metrics:
            return {}
        
        values = [m['value'] for m in self.metrics[metric_name]]
        if not values:
            return {}
        
        # Simple statistics without numpy
        count = len(values)
        mean = sum(values) / count
        sorted_values = sorted(values)
        
        return {
            'count': count,
            'mean': mean,
            'min': min(values),
            'max': max(values),
            'p50': sorted_values[count // 2],
            'p95': sorted_values[int(count * 0.95)] if count > 20 else max(values),
            'p99': sorted_values[int(count * 0.99)] if count > 100 else max(values)
        }
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics"""
        return {name: self.get_stats(name) for name in self.metrics}

class ConsciousnessMetrics:
    """Specific metrics for consciousness system"""
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.coherence_history = deque(maxlen=100)
    
    def record_coherence(self, coherence: float):
        """Record consciousness coherence score"""
        self.collector.record('consciousness.coherence', coherence)
        self.coherence_history.append({
            'value': coherence,
            'timestamp': datetime.now()
        })
    
    def record_memory_operation(self, operation: str, duration: float, success: bool):
        """Record memory operation metrics"""
        self.collector.record(
            f'memory.{operation}.duration',
            duration,
            tags={'success': str(success)}
        )
    
    def record_emotional_state(self, pleasure: float, arousal: float, dominance: float):
        """Record emotional state metrics"""
        self.collector.record('emotion.pleasure', pleasure)
        self.collector.record('emotion.arousal', arousal)
        self.collector.record('emotion.dominance', dominance)
    
    def record_api_call(self, api: str, duration: float, tokens: int, success: bool):
        """Record API call metrics"""
        self.collector.record(
            f'api.{api}.duration',
            duration,
            tags={'success': str(success)}
        )
        self.collector.record(
            f'api.{api}.tokens',
            tokens,
            tags={'success': str(success)}
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        coherence_stats = self.collector.get_stats('consciousness.coherence')
        
        # Determine health based on coherence
        if not coherence_stats:
            health = 'unknown'
        elif coherence_stats['mean'] > 0.8:
            health = 'healthy'
        elif coherence_stats['mean'] > 0.6:
            health = 'degraded'
        else:
            health = 'unhealthy'
        
        return {
            'status': health,
            'uptime_seconds': (datetime.now() - self.collector.start_time).total_seconds(),
            'coherence': coherence_stats,
            'metrics_summary': self.collector.get_all_metrics()
        }

# Global metrics instance
consciousness_metrics = ConsciousnessMetrics()
