"""Core consciousness functionality with proper state management"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessState:
    """State of consciousness at any given moment"""
    global_coherence: float = 0.7
    attention_focus: str = "general awareness"
    active_goals: List[str] = field(default_factory=list)
    working_memory: List[str] = field(default_factory=list)
    emotional_state: Dict[str, float] = field(default_factory=lambda: {
        'pleasure': 0.5,
        'arousal': 0.5,
        'dominance': 0.5
    })
    personality_traits: Dict[str, float] = field(default_factory=lambda: {
        'openness': 0.8,
        'conscientiousness': 0.7,
        'extraversion': 0.6,
        'agreeableness': 0.8,
        'neuroticism': 0.3
    })
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'global_coherence': self.global_coherence,
            'attention_focus': self.attention_focus,
            'active_goals': self.active_goals,
            'working_memory': self.working_memory,
            'emotional_state': self.emotional_state,
            'personality_traits': self.personality_traits,
            'interaction_count': self.interaction_count,
            'last_interaction': self.last_interaction.isoformat() if self.last_interaction else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsciousnessState':
        """Create from dictionary"""
        state = cls()
        state.global_coherence = data.get('global_coherence', 0.7)
        state.attention_focus = data.get('attention_focus', 'general awareness')
        state.active_goals = data.get('active_goals', [])
        state.working_memory = data.get('working_memory', [])
        state.emotional_state = data.get('emotional_state', state.emotional_state)
        state.personality_traits = data.get('personality_traits', state.personality_traits)
        state.interaction_count = data.get('interaction_count', 0)
        if data.get('last_interaction'):
            state.last_interaction = datetime.fromisoformat(data['last_interaction'])
        return state


class ConsciousnessCore:
    """Core consciousness functionality"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.state = ConsciousnessState()
        self.memory_buffer = deque(maxlen=self.config.get('memory_buffer_size', 10))
        self.cycle_frequency = self.config.get('cycle_frequency', 1.0)  # Hz
        
    def process_input(self, input_text: str, context: Dict[str, Any]):
        """Process input and update consciousness state"""
        self.state.interaction_count += 1
        self.state.last_interaction = datetime.now()
        
        # Add to working memory
        memory_entry = f"[{datetime.now().strftime('%H:%M:%S')}] {input_text[:100]}"
        self.memory_buffer.append(memory_entry)
        self.state.working_memory = list(self.memory_buffer)
        
        # Update coherence based on activity
        self._calculate_coherence()
        
        logger.debug(f"Processed input. Coherence: {self.state.global_coherence:.2f}")
        
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary"""
        return {
            'coherence': self.state.global_coherence,
            'attention': self.state.attention_focus,
            'emotion': self.state.emotional_state,
            'recent_memories': list(self.memory_buffer)[-5:],
            'interaction_count': self.state.interaction_count,
            'goals': self.state.active_goals,
            'personality': self.state.personality_traits
        }
    
    def _calculate_coherence(self):
        """Calculate global coherence score"""
        # Base coherence
        base_coherence = 0.7
        
        # Memory factor (more memories = higher coherence)
        memory_factor = min(1.0, len(self.memory_buffer) / 10)
        
        # Interaction recency factor
        recency_factor = 1.0
        if self.state.last_interaction:
            time_since = (datetime.now() - self.state.last_interaction).total_seconds()
            recency_factor = max(0.5, 1.0 - (time_since / 3600))  # Decay over 1 hour
        
        # Calculate final coherence
        self.state.global_coherence = (
            base_coherence * 0.5 + 
            memory_factor * 0.3 + 
            recency_factor * 0.2
        )
        
    async def _update_attention(self):
        """Update attention focus"""
        # This would be implemented with the attention agent
        pass
    
    async def _consolidate_memory(self):
        """Consolidate working memory"""
        # This would trigger memory consolidation
        pass
    
    async def _update_emotional_state(self):
        """Update emotional state"""
        # This would use the emotional engine
        pass
    
    def shutdown(self):
        """Cleanup on shutdown"""
        logger.info("Consciousness core shutting down")
        logger.info(f"Final coherence: {self.state.global_coherence:.2f}")
        logger.info(f"Total interactions: {self.state.interaction_count}")
