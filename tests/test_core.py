# tests/test_core.py - Comprehensive core functionality tests
import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.consciousness import ConsciousnessCore, ConsciousnessState
from src.core.emotional_engine import EmotionalEngine, EmotionalState
from src.core.personality import PersonalityEngine, PersonalityProfile
from src.core.memory_manager import AdvancedMemoryManager, MemoryItem

# Async fixtures
@pytest.fixture
async def consciousness_core():
    """Async fixture for consciousness core"""
    config = {
        'cycle_frequency': 10.0,  # Fast for testing
        'working_memory_size': 5,
        'coherence_threshold': 0.7
    }
    core = ConsciousnessCore(config)
    yield core
    core.shutdown()

@pytest.fixture
async def emotional_engine():
    """Async fixture for emotional engine"""
    return EmotionalEngine()

@pytest.fixture
async def personality_engine():
    """Async fixture for personality engine"""
    return PersonalityEngine()

@pytest.fixture
async def memory_manager():
    """Async fixture for memory manager"""
    config = {
        'embedding_dim': 768,
        'max_memories': 1000,
        'use_gpu': False  # CPU for testing
    }
    return AdvancedMemoryManager(config)

class TestConsciousnessCore:
    """Test consciousness core functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, consciousness_core):
        """Test consciousness core initialization"""
        assert consciousness_core.state is not None
        assert consciousness_core.state.global_coherence == 1.0
        assert len(consciousness_core.memory_buffer) == 0
        assert consciousness_core.cycle_frequency == 10.0
    
    @pytest.mark.asyncio
    async def test_consciousness_loop(self, consciousness_core):
        """Test consciousness processing loop"""
        # Run loop for short duration
        loop_task = asyncio.create_task(consciousness_core.consciousness_loop())
        
        # Let it run for 0.5 seconds
        await asyncio.sleep(0.5)
        
        # Should have completed several cycles
        assert consciousness_core._running
        
        # Shutdown
        consciousness_core.shutdown()
        await loop_task
        
        assert not consciousness_core._running
    
    @pytest.mark.asyncio
    async def test_process_input(self, consciousness_core):
        """Test input processing"""
        initial_count = consciousness_core.state.interaction_count
        
        consciousness_core.process_input("Test input", {
            'sentiment': {'pleasure': 0.8, 'arousal': 0.5}
        })
        
        assert consciousness_core.state.interaction_count == initial_count + 1
        assert len(consciousness_core.memory_buffer) == 1
        assert consciousness_core.state.emotional_state['pleasure'] > 0.6
    
    @pytest.mark.asyncio
    async def test_coherence_calculation(self, consciousness_core):
        """Test coherence score calculation"""
        # Add memories to buffer
        for i in range(5):
            consciousness_core.memory_buffer.append(f"Memory {i}")
        
        consciousness_core._calculate_coherence()
        
        # Should have high coherence with consistent memories
        assert consciousness_core.state.global_coherence > 0.5
        assert consciousness_core.state.global_coherence <= 1.0
    
    @pytest.mark.asyncio
    async def test_attention_update(self, consciousness_core):
        """Test attention focus updates"""
        # Mock GPU tensor operations
        with patch('torch.randn') as mock_randn:
            mock_tensor = Mock()
            mock_tensor.unsqueeze.return_value = mock_tensor
            mock_tensor.squeeze.return_value = mock_tensor
            mock_tensor.mean.return_value = mock_tensor
            mock_tensor.argmax.return_value.item.return_value = 0
            mock_randn.return_value = mock_tensor
            
            consciousness_core.memory_buffer.append("Focus on this")
            await consciousness_core._update_attention()
            
            assert consciousness_core.state.attention_focus == "Focus on this"

class TestEmotionalEngine:
    """Test emotional engine functionality"""
    
    @pytest.mark.asyncio
    async def test_emotional_state_update(self, emotional_engine):
        """Test emotional state transitions"""
        initial_state = EmotionalState(
            pleasure=emotional_engine.current_state.pleasure,
            arousal=emotional_engine.current_state.arousal,
            dominance=emotional_engine.current_state.dominance
        )
        
        # Apply positive stimulus
        emotional_engine.update_emotional_state({
            'sentiment': 0.9,
            'urgency': 0.2,
            'success': True
        })
        
        # Should increase pleasure and dominance
        assert emotional_engine.current_state.pleasure > initial_state.pleasure
        assert emotional_engine.current_state.dominance > initial_state.dominance
    
    @pytest.mark.asyncio
    async def test_emotion_labeling(self, emotional_engine):
        """Test emotion label identification"""
        # Set to joy state
        emotional_engine.current_state = EmotionalState(
            pleasure=0.8,
            arousal=0.7,
            dominance=0.7
        )
        
        label, confidence = emotional_engine.get_closest_emotion_label()
        assert label == 'joy'
        assert confidence > 0.7
    
    @pytest.mark.asyncio
    async def test_emotional_intensity(self, emotional_engine):
        """Test emotional intensity calculation"""
        # Neutral state
        emotional_engine.current_state = EmotionalState(0.5, 0.5, 0.5)
        assert emotional_engine.get_emotional_intensity() < 0.1
        
        # Extreme state
        emotional_engine.current_state = EmotionalState(1.0, 1.0, 1.0)
        intensity = emotional_engine.get_emotional_intensity()
        assert intensity > 0.8

class TestPersonalityEngine:
    """Test personality engine functionality"""
    
    @pytest.mark.asyncio
    async def test_trait_updates(self, personality_engine):
        """Test personality trait evolution"""
        initial_openness = personality_engine.profile.openness
        
        # Process creative interaction
        personality_engine.process_interaction({
            'topic_complexity': 0.9,
            'novelty': 0.8,
            'creativity_required': 0.9
        })
        
        # Openness should increase slightly
        assert personality_engine.profile.openness > initial_openness
    
    @pytest.mark.asyncio
    async def test_trait_bounds(self, personality_engine):
        """Test personality trait boundaries"""
        # Try to set extreme values
        personality_engine.profile.update_trait('neuroticism', 10.0)
        
        # Should be clamped to bounds
        assert personality_engine.profile.neuroticism <= 0.7
        
        personality_engine.profile.update_trait('openness', -10.0)
        assert personality_engine.profile.openness >= 0.4
    
    @pytest.mark.asyncio
    async def test_behavioral_modifiers(self, personality_engine):
        """Test behavioral modifier calculation"""
        modifiers = personality_engine.get_behavioral_modifiers()
        
        # All modifiers should be in reasonable range
        for key, value in modifiers.items():
            assert 0.0 <= value <= 1.0

class TestMemoryManager:
    """Test memory management functionality"""
    
    @pytest.mark.asyncio
    async def test_memory_storage(self, memory_manager):
        """Test memory storage and retrieval"""
        # Store a memory
        memory_id = await memory_manager.store_memory(
            "Important test memory",
            {'importance': 0.9, 'user_id': 'test_user'}
        )
        
        assert memory_id in memory_manager.memories
        assert memory_manager.memories[memory_id].importance == 0.9
    
    @pytest.mark.asyncio
    async def test_memory_retrieval(self, memory_manager):
        """Test memory retrieval by query"""
        # Store multiple memories
        await memory_manager.store_memory("Python programming test", {'importance': 0.8})
        await memory_manager.store_memory("JavaScript testing", {'importance': 0.6})
        await memory_manager.store_memory("Unrelated memory", {'importance': 0.5})
        
        # Retrieve by query
        results = await memory_manager.retrieve_memories("programming test", k=2)
        
        assert len(results) == 2
        assert "Python" in results[0].content
    
    @pytest.mark.asyncio
    async def test_memory_consolidation(self, memory_manager):
        """Test memory consolidation process"""
        # Create old memories with low salience
        for i in range(10):
            memory = MemoryItem(
                id=f"old_{i}",
                content=f"Old memory {i}",
                timestamp=datetime(2020, 1, 1),  # Very old
                importance=0.1,
                emotional_context={},
                associations=[]
            )
            memory_manager.memories[memory.id] = memory
        
        initial_count = len(memory_manager.memories)
        await memory_manager.consolidate_memories()
        
        # Should have fewer memories after consolidation
        assert len(memory_manager.memories) < initial_count

class TestIntegration:
    """Integration tests across components"""
    
    @pytest.mark.asyncio
    async def test_consciousness_emotional_integration(self, consciousness_core, emotional_engine):
        """Test integration between consciousness and emotions"""
        # Process input that affects emotions
        consciousness_core.process_input("This makes me very happy!", {
            'sentiment': {'pleasure': 0.9, 'arousal': 0.7}
        })
        
        # Update emotional engine based on consciousness state
        emotional_engine.update_emotional_state({
            'sentiment': consciousness_core.state.emotional_state['pleasure']
        })
        
        # Both should reflect positive state
        assert consciousness_core.state.emotional_state['pleasure'] > 0.6
        assert emotional_engine.current_state.pleasure > 0.6
    
    @pytest.mark.asyncio
    async def test_memory_consciousness_integration(self, consciousness_core, memory_manager):
        """Test memory and consciousness integration"""
        # Add memories to consciousness
        test_memories = ["Memory 1", "Memory 2", "Memory 3"]
        for mem in test_memories:
            consciousness_core.process_input(mem)
            await memory_manager.store_memory(mem, {'source': 'consciousness'})
        
        # Consciousness should track memories
        assert len(consciousness_core.memory_buffer) >= 3
        
        # Memory manager should have stored them
        assert len(memory_manager.memories) >= 3

# Performance tests
class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_consciousness_cycle_performance(self, consciousness_core):
        """Test consciousness cycle performance"""
        import time
        
        # Measure cycle time
        start = time.time()
        await consciousness_core._update_attention()
        await consciousness_core._consolidate_memory()
        await consciousness_core._update_emotional_state()
        consciousness_core._calculate_coherence()
        
        cycle_time = time.time() - start
        
        # Should complete in reasonable time
        assert cycle_time < 0.1  # 100ms max
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_scaling(self, memory_manager):
        """Test memory system scaling"""
        import time
        
        # Store many memories
        start = time.time()
        for i in range(1000):
            await memory_manager.store_memory(
                f"Memory {i} with unique content",
                {'index': i}
            )
        
        storage_time = time.time() - start
        
        # Should handle 1000 memories efficiently
        assert storage_time < 10.0  # 10 seconds max
        
        # Test retrieval performance
        start = time.time()
        results = await memory_manager.retrieve_memories("unique content", k=10)
        retrieval_time = time.time() - start
        
        assert retrieval_time < 0.1  # 100ms max
        assert len(results) == 10

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])