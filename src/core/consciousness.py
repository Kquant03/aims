# consciousness.py - Core AIMS Consciousness System
import asyncio
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessState:
    """Represents the current state of consciousness"""
    timestamp: datetime
    attention_focus: str
    emotional_state: Dict[str, float]  # PAD model: pleasure, arousal, dominance
    personality_traits: Dict[str, float]  # OCEAN: openness, conscientiousness, extraversion, agreeableness, neuroticism
    working_memory: List[str]
    global_coherence: float
    active_goals: List[str]
    last_interaction: Optional[datetime] = None
    interaction_count: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize state for persistence"""
        state_dict = asdict(self)
        state_dict['timestamp'] = self.timestamp.isoformat()
        if self.last_interaction:
            state_dict['last_interaction'] = self.last_interaction.isoformat()
        return state_dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConsciousnessState':
        """Deserialize from persistent storage"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('last_interaction'):
            data['last_interaction'] = datetime.fromisoformat(data['last_interaction'])
        return cls(**data)

class AttentionMechanism(nn.Module):
    """Simplified attention mechanism for consciousness focus"""
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.size(-1))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        attended = torch.matmul(attention_weights, V)
        return self.output_proj(attended)

class ConsciousnessCore:
    """Main consciousness system coordinating all components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = self._initialize_state()
        self.attention = AttentionMechanism()
        self.memory_buffer = deque(maxlen=config.get('working_memory_size', 7))
        self.cycle_frequency = config.get('cycle_frequency', 2.0)  # 2Hz default
        self._running = False
        self._coherence_threshold = config.get('coherence_threshold', 0.7)
        
        # Initialize with GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention.to(self.device)
        logger.info(f"Consciousness core initialized on {self.device}")
        
    def _initialize_state(self) -> ConsciousnessState:
        """Create initial consciousness state"""
        return ConsciousnessState(
            timestamp=datetime.now(),
            attention_focus="initialization",
            emotional_state={
                "pleasure": 0.6,
                "arousal": 0.5,
                "dominance": 0.5
            },
            personality_traits={
                "openness": 0.8,
                "conscientiousness": 0.7,
                "extraversion": 0.6,
                "agreeableness": 0.8,
                "neuroticism": 0.3
            },
            working_memory=[],
            global_coherence=1.0,
            active_goals=["maintain_coherence", "be_helpful", "learn_continuously"]
        )
    
    async def consciousness_loop(self):
        """Main consciousness processing loop"""
        self._running = True
        cycle_duration = 1.0 / self.cycle_frequency
        
        while self._running:
            cycle_start = asyncio.get_event_loop().time()
            
            try:
                # Update consciousness state
                await self._update_attention()
                await self._consolidate_memory()
                await self._update_emotional_state()
                self._calculate_coherence()
                
                # Persist state periodically
                if self.state.interaction_count % 10 == 0:
                    await self._persist_state()
                
            except Exception as e:
                logger.error(f"Error in consciousness loop: {e}")
                self.state.global_coherence *= 0.9  # Reduce coherence on errors
            
            # Maintain cycle timing
            elapsed = asyncio.get_event_loop().time() - cycle_start
            if elapsed < cycle_duration:
                await asyncio.sleep(cycle_duration - elapsed)
    
    async def _update_attention(self):
        """Update attention focus based on current inputs"""
        if self.memory_buffer:
            # Simple attention mechanism - focus on most recent important item
            recent_memories = list(self.memory_buffer)
            
            # Convert to embeddings (simplified - in production use real embeddings)
            embeddings = torch.randn(len(recent_memories), 768).to(self.device)
            
            # Apply attention
            with torch.no_grad():
                attended = self.attention(embeddings.unsqueeze(0))
            
            # Update focus based on attention weights
            attention_scores = attended.squeeze().mean(dim=1)
            max_attention_idx = attention_scores.argmax().item()
            
            if max_attention_idx < len(recent_memories):
                self.state.attention_focus = recent_memories[max_attention_idx]
    
    async def _consolidate_memory(self):
        """Consolidate working memory items"""
        if len(self.memory_buffer) > 5:
            # Keep only most salient memories
            self.state.working_memory = list(self.memory_buffer)[-5:]
    
    async def _update_emotional_state(self):
        """Update emotional state with gradual transitions"""
        # Simple emotional drift toward baseline
        baseline = {"pleasure": 0.6, "arousal": 0.5, "dominance": 0.5}
        drift_rate = 0.05
        
        for dimension, baseline_value in baseline.items():
            current = self.state.emotional_state[dimension]
            self.state.emotional_state[dimension] = (
                current + drift_rate * (baseline_value - current)
            )
    
    def _calculate_coherence(self):
        """Calculate global coherence score"""
        # Simplified coherence based on memory consistency and emotional stability
        memory_coherence = min(1.0, len(self.state.working_memory) / 5.0)
        
        emotional_stability = 1.0 - np.std(list(self.state.emotional_state.values()))
        
        self.state.global_coherence = 0.7 * memory_coherence + 0.3 * emotional_stability
    
    async def _persist_state(self):
        """Persist current state to storage"""
        # This will be handled by the persistence manager
        logger.debug(f"Persisting state at {self.state.timestamp}")
    
    def process_input(self, input_text: str, context: Optional[Dict] = None):
        """Process new input and update consciousness accordingly"""
        self.state.interaction_count += 1
        self.state.last_interaction = datetime.now()
        
        # Add to working memory
        self.memory_buffer.append(input_text[:100])  # Truncate for memory
        
        # Update emotional state based on input sentiment
        # (Simplified - in production use real sentiment analysis)
        if context and 'sentiment' in context:
            sentiment = context['sentiment']
            self.state.emotional_state['pleasure'] = (
                0.7 * self.state.emotional_state['pleasure'] + 
                0.3 * sentiment.get('pleasure', 0.6)
            )
            self.state.emotional_state['arousal'] = (
                0.7 * self.state.emotional_state['arousal'] + 
                0.3 * sentiment.get('arousal', 0.5)
            )
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current consciousness state for API responses"""
        return {
            "coherence": self.state.global_coherence,
            "attention": self.state.attention_focus,
            "emotion": self.state.emotional_state,
            "personality": self.state.personality_traits,
            "recent_memories": list(self.memory_buffer)[-3:],
            "interaction_count": self.state.interaction_count,
            "goals": self.state.active_goals
        }
    
    def shutdown(self):
        """Graceful shutdown of consciousness loop"""
        self._running = False
        logger.info("Consciousness core shutting down")

# Example usage
if __name__ == "__main__":
    config = {
        "cycle_frequency": 2.0,
        "working_memory_size": 7,
        "coherence_threshold": 0.7
    }
    
    consciousness = ConsciousnessCore(config)
    
    # Start consciousness loop
    async def main():
        task = asyncio.create_task(consciousness.consciousness_loop())
        
        # Simulate some interactions
        consciousness.process_input("Hello, how are you today?")
        await asyncio.sleep(1)
        
        consciousness.process_input("I'd like to discuss consciousness")
        await asyncio.sleep(1)
        
        print(consciousness.get_state_summary())
        
        consciousness.shutdown()
        await task
    
    asyncio.run(main())