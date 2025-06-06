# src/core/consciousness_state.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
import json

@dataclass
class ConsciousnessState:
    """Core state representation for the consciousness system"""
    
    # Core consciousness metrics
    global_coherence: float = 0.5
    attention_focus: str = "general awareness"
    
    # Memory and interaction tracking
    working_memory: deque = field(default_factory=lambda: deque(maxlen=10))
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    
    # Goals and intentions
    active_goals: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            'global_coherence': self.global_coherence,
            'attention_focus': self.attention_focus,
            'working_memory': list(self.working_memory),
            'interaction_count': self.interaction_count,
            'last_interaction': self.last_interaction.isoformat() if self.last_interaction else None,
            'active_goals': self.active_goals,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsciousnessState':
        """Create state from dictionary"""
        state = cls()
        
        state.global_coherence = data.get('global_coherence', 0.5)
        state.attention_focus = data.get('attention_focus', 'general awareness')
        
        # Restore working memory
        working_memory_list = data.get('working_memory', [])
        state.working_memory = deque(working_memory_list, maxlen=10)
        
        state.interaction_count = data.get('interaction_count', 0)
        
        # Parse datetime fields
        if data.get('last_interaction'):
            state.last_interaction = datetime.fromisoformat(data['last_interaction'])
        
        state.active_goals = data.get('active_goals', [])
        
        if data.get('created_at'):
            state.created_at = datetime.fromisoformat(data['created_at'])
        if data.get('last_updated'):
            state.last_updated = datetime.fromisoformat(data['last_updated'])
        
        return state
    
    def update_interaction(self, timestamp: Optional[datetime] = None):
        """Update interaction tracking"""
        self.interaction_count += 1
        self.last_interaction = timestamp or datetime.now()
        self.last_updated = datetime.now()


class ConsciousnessCore:
    """Core consciousness management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.state = ConsciousnessState()
        
        # Memory buffer for working memory
        self.memory_buffer = deque(maxlen=self.config.get('working_memory_size', 10))
        
        # Configuration
        self.cycle_frequency = self.config.get('cycle_frequency', 1.0)  # Hz
        self._running = False
        
    def process_input(self, content: str, context: Dict[str, Any] = None):
        """Process new input and update consciousness state"""
        # Add to working memory
        self.memory_buffer.append(content[:200])  # Truncate for memory
        
        # Update state
        self.state.update_interaction()
        
        # Update attention based on context
        if context and 'attention_result' in context:
            self.state.attention_focus = context['attention_result'].get(
                'primary_focus', 
                self.state.attention_focus
            )
    
    async def _update_attention(self):
        """Update attention mechanisms"""
        # Placeholder for attention update logic
        pass
    
    async def _consolidate_memory(self):
        """Consolidate working memory"""
        # Placeholder for memory consolidation
        pass
    
    async def _update_emotional_state(self):
        """Update emotional state"""
        # Placeholder for emotional update
        pass
    
    def _calculate_coherence(self):
        """Calculate global coherence score"""
        # Simple coherence calculation based on working memory consistency
        if len(self.memory_buffer) < 2:
            return
        
        # Placeholder - in reality this would be more sophisticated
        # For now, slight random walk
        import random
        delta = random.uniform(-0.02, 0.02)
        self.state.global_coherence = max(0.1, min(1.0, self.state.global_coherence + delta))
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current consciousness state"""
        return {
            'coherence': self.state.global_coherence,
            'attention_focus': self.state.attention_focus,
            'working_memory_size': len(self.memory_buffer),
            'interaction_count': self.state.interaction_count,
            'active_goals': self.state.active_goals,
            'last_interaction': self.state.last_interaction.isoformat() if self.state.last_interaction else None
        }
    
    def shutdown(self):
        """Shutdown consciousness loops"""
        self._running = False