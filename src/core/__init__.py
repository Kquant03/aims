from .consciousness_core import ConsciousnessCore, ConsciousnessState
from .memory_manager import ThreeTierMemorySystem  # This is the actual class name
from .personality import PersonalityEngine
from .emotional_engine import EmotionalEngine
from .living_consciousness import LivingConsciousnessArtifact

# Create an alias for compatibility
PersistentMemoryManager = ThreeTierMemorySystem

__all__ = [
    'ConsciousnessCore',
    'ConsciousnessState', 
    'ThreeTierMemorySystem',
    'PersistentMemoryManager',  # Alias
    'PersonalityEngine',
    'EmotionalEngine',
    'LivingConsciousnessArtifact'
]
