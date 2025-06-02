# test_core.py - Core System Tests
import pytest
import asyncio
import torch
from datetime import datetime
import json
import tempfile
from pathlib import Path

from src.core.consciousness import ConsciousnessCore, ConsciousnessState
from src.core.memory_manager import PersistentMemoryManager, MemoryItem
from src.core.personality import PersonalityEngine, PersonalityProfile
from src.core.emotional_engine import EmotionalEngine, EmotionalState


class TestConsciousnessCore:
    """Test consciousness core functionality"""
    
    @pytest.fixture
    def consciousness(self):
        config = {
            'cycle_frequency': 10.0,  # Fast for testing
            'working_memory_size': 5,
            'coherence_threshold': 0.7
        }
        return ConsciousnessCore(config)
    
    def test_initialization(self, consciousness):
        """Test consciousness initialization"""
        assert consciousness.state is not None
        assert consciousness.state.global_coherence == 1.0
        assert len(consciousness.state.working_memory) == 0
        assert consciousness.cycle_frequency == 10.0
    
    def test_process_input(self, consciousness):
        """Test input processing"""
        consciousness.process_input("Hello, world!", {'test': True})
        
        assert consciousness.state.interaction_count == 1
        assert len(consciousness.memory_buffer) == 1
        assert consciousness.state.last_interaction is not None
    
    @pytest.mark.asyncio
    async def test_consciousness_loop(self, consciousness):
        """Test consciousness loop execution"""
        # Run for a short time
        task = asyncio.create_task(consciousness.consciousness_loop())
        await asyncio.sleep(0.3)  # 3 cycles at 10Hz
        
        consciousness.shutdown()
        await task
        
        # Should have processed some cycles
        assert consciousness.state.timestamp != consciousness._initialize_state().timestamp
    
    def test_coherence_calculation(self, consciousness):
        """Test coherence calculation"""
        # Add some memories
        for i in range(3):
            consciousness.process_input(f"Memory {i}")
        
        consciousness._calculate_coherence()
        
        assert 0 <= consciousness.state.global_coherence <= 1.0
    
    def test_state_serialization(self, consciousness):
        """Test state serialization and deserialization"""
        # Process some inputs
        consciousness.process_input("Test input")
        
        # Serialize
        state_dict = consciousness.state.to_dict()
        
        # Deserialize
        restored_state = ConsciousnessState.from_dict(state_dict)
        
        assert restored_state.interaction_count == consciousness.state.interaction_count
        assert restored_state.attention_focus == consciousness.state.attention_focus


class TestMemoryManager:
    """Test memory management functionality"""
    
    @pytest.fixture
    async def memory_manager(self):
        config = {
            'redis_host': 'localhost',
            'redis_port': 6379,
            'pg_host': 'localhost',
            'pg_port': 5432,
            'pg_database': 'test_aims_memory',
            'pg_user': 'test_user',
            'pg_password': 'test_password',
            'qdrant_host': 'localhost',
            'qdrant_port': 6333
        }
        
        # Note: This requires test databases to be running
        # In practice, you might use test containers or mocks
        return PersistentMemoryManager(config)
    
    @pytest.mark.asyncio
    async def test_store_memory(self, memory_manager):
        """Test memory storage"""
        memory_id = await memory_manager.store_memory(
            content="This is a test memory",
            context={
                'user_id': 'test_user',
                'emotional_state': {
                    'pleasure': 0.7,
                    'arousal': 0.5,
                    'dominance': 0.6
                },
                'importance': 0.8
            }
        )
        
        assert memory_id is not None
        assert len(memory_id) == 16  # SHA256 truncated
    
    @pytest.mark.asyncio
    async def test_retrieve_memories(self, memory_manager):
        """Test memory retrieval"""
        # Store some memories
        memory_ids = []
        for i in range(5):
            memory_id = await memory_manager.store_memory(
                content=f"Test memory {i}",
                context={'user_id': 'test_user', 'importance': 0.5 + i * 0.1}
            )
            memory_ids.append(memory_id)
        
        # Retrieve memories
        memories = await memory_manager.retrieve_memories("test memory", k=3)
        
        assert len(memories) <= 3
        assert all(isinstance(m, MemoryItem) for m in memories)
    
    def test_memory_salience_calculation(self):
        """Test memory salience calculation"""
        memory = MemoryItem(
            id="test_id",
            content="Test content",
            timestamp=datetime.now(),
            importance=0.8,
            emotional_context={'pleasure': 0.9, 'arousal': 0.7, 'dominance': 0.5},
            associations=[],
            decay_rate=0.1
        )
        
        current_time = datetime.now()
        salience = memory.calculate_salience(current_time)
        
        assert 0 <= salience <= 1.5  # Can be slightly above 1 due to emotional boost


class TestPersonalityEngine:
    """Test personality system"""
    
    @pytest.fixture
    def personality_engine(self):
        return PersonalityEngine()
    
    def test_trait_initialization(self, personality_engine):
        """Test personality trait initialization"""
        traits = personality_engine.profile.get_traits()
        
        assert 'openness' in traits
        assert 'conscientiousness' in traits
        assert 'extraversion' in traits
        assert 'agreeableness' in traits
        assert 'neuroticism' in traits
        
        # All traits should be within bounds
        for trait, value in traits.items():
            bounds = personality_engine.profile.trait_bounds[trait]
            assert bounds[0] <= value <= bounds[1]
    
    def test_trait_update(self, personality_engine):
        """Test personality trait updates"""
        initial_openness = personality_engine.profile.openness
        
        # Process interaction that should increase openness
        personality_engine.process_interaction({
            'topic_complexity': 0.9,
            'user_sentiment': 0.8
        })
        
        # Openness should have increased slightly
        assert personality_engine.profile.openness >= initial_openness
    
    def test_behavioral_modifiers(self, personality_engine):
        """Test behavioral modifier calculation"""
        modifiers = personality_engine.get_behavioral_modifiers()
        
        assert 'response_length' in modifiers
        assert 'emotional_expression' in modifiers
        assert 'formality' in modifiers
        assert 'creativity' in modifiers
        
        # All modifiers should be within [-1, 1]
        for modifier, value in modifiers.items():
            assert -1.0 <= value <= 1.0


class TestEmotionalEngine:
    """Test emotional system"""
    
    @pytest.fixture
    def emotional_engine(self):
        return EmotionalEngine()
    
    def test_emotional_state_initialization(self, emotional_engine):
        """Test emotional state initialization"""
        state = emotional_engine.current_state
        
        assert 0 <= state.pleasure <= 1
        assert 0 <= state.arousal <= 1
        assert 0 <= state.dominance <= 1
    
    def test_emotional_update(self, emotional_engine):
        """Test emotional state updates"""
        initial_state = EmotionalState(
            pleasure=emotional_engine.current_state.pleasure,
            arousal=emotional_engine.current_state.arousal,
            dominance=emotional_engine.current_state.dominance
        )
        
        # Apply positive stimulus
        emotional_engine.update_emotional_state({
            'sentiment': 0.9,
            'urgency': 0.7,
            'success': True
        })
        
        # State should have changed
        assert emotional_engine.current_state.distance_to(initial_state) > 0
    
    def test_emotion_labeling(self, emotional_engine):
        """Test emotion label identification"""
        # Set a specific emotional state
        emotional_engine.current_state = EmotionalState(
            pleasure=0.8,
            arousal=0.7,
            dominance=0.7
        )
        
        label, confidence = emotional_engine.get_closest_emotion_label()
        
        assert label in emotional_engine.emotion_categories
        assert 0 <= confidence <= 1
        
        # Should be close to 'joy' given the values
        assert label in ['joy', 'excitement']
    
    def test_baseline_pull(self, emotional_engine):
        """Test baseline emotional pull"""
        # Set extreme emotional state
        emotional_engine.current_state = EmotionalState(
            pleasure=1.0,
            arousal=1.0,
            dominance=1.0
        )
        
        # Apply neutral stimulus multiple times
        for _ in range(10):
            emotional_engine.update_emotional_state({'sentiment': 0.5})
        
        # Should have moved toward baseline
        assert emotional_engine.current_state.pleasure < 1.0
        assert emotional_engine.current_state.arousal < 1.0


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_consciousness_memory_integration(self):
        """Test integration between consciousness and memory"""
        # This would require a full system setup
        # Placeholder for integration test
        pass
    
    @pytest.mark.asyncio
    async def test_full_interaction_flow(self):
        """Test a complete interaction flow"""
        # This would test the entire pipeline from input to response
        # Placeholder for integration test
        pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])