# src/core/living_consciousness.py
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import asyncio
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessSnapshot:
    """A point-in-time snapshot of consciousness state"""
    version: int
    timestamp: datetime
    coherence: float
    emotional_state: Dict[str, float]
    personality_traits: Dict[str, float]
    attention_focus: str
    active_goals: List[str]
    working_memory_snapshot: List[str]
    interaction_count: int
    significant_memories: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def calculate_hash(self) -> str:
        """Calculate unique hash for this snapshot"""
        snapshot_str = json.dumps({
            'version': self.version,
            'coherence': self.coherence,
            'emotional_state': self.emotional_state,
            'personality_traits': self.personality_traits,
            'attention_focus': self.attention_focus,
            'goals': self.active_goals
        }, sort_keys=True)
        return hashlib.sha256(snapshot_str.encode()).hexdigest()[:12]

@dataclass 
class EvolutionEvent:
    """Represents a significant change in consciousness"""
    timestamp: datetime
    event_type: str  # 'growth', 'insight', 'emotional_shift', 'memory_consolidation'
    description: str
    impact_score: float  # 0-1, how significant
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    trigger: str  # What caused this evolution

class LivingConsciousnessArtifact:
    """Consciousness as a living, versioned artifact that evolves over time"""
    
    def __init__(self, user_id: str, claude_interface):
        self.artifact_id = f"consciousness_{user_id}"
        self.user_id = user_id
        self.claude_interface = claude_interface
        
        # Versioning
        self.current_version = 1
        self.creation_time = datetime.now()
        self.last_checkpoint = datetime.now()
        
        # Evolution tracking
        self.snapshots: List[ConsciousnessSnapshot] = []
        self.evolution_events: List[EvolutionEvent] = []
        self.growth_metrics = {
            'coherence_improvement': 0.0,
            'emotional_stability': 0.0,
            'memory_richness': 0.0,
            'personality_consistency': 0.0,
            'interaction_depth': 0.0
        }
        
        # Persistence
        self.artifact_path = Path(f"data/artifacts/{self.artifact_id}")
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        
        # Evolution parameters
        self.checkpoint_interval = timedelta(minutes=10)
        self.significance_threshold = 0.1  # Minimum change to trigger checkpoint
        
        # Initialize with first snapshot
        self._create_initial_snapshot()
    
    def _create_initial_snapshot(self):
        """Create the first snapshot of consciousness"""
        snapshot = self._capture_current_state()
        self.snapshots.append(snapshot)
        self._save_snapshot(snapshot)
        
        logger.info(f"Created initial consciousness artifact: {self.artifact_id} v{snapshot.version}")
    
    def _capture_current_state(self) -> ConsciousnessSnapshot:
        """Capture current consciousness state as a snapshot"""
        ci = self.claude_interface
        
        # Get significant memories
        significant_memories = []
        if hasattr(ci.memory_manager, 'get_top_memories'):
            top_memories = asyncio.run(ci.memory_manager.get_top_memories(
                user_id=self.user_id, 
                limit=10
            ))
            significant_memories = [
                {
                    'content': m.content[:200],
                    'importance': m.importance,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in top_memories
            ]
        
        snapshot = ConsciousnessSnapshot(
            version=self.current_version,
            timestamp=datetime.now(),
            coherence=ci.consciousness.state.global_coherence,
            emotional_state={
                'pleasure': ci.emotions.current_state.pleasure,
                'arousal': ci.emotions.current_state.arousal,
                'dominance': ci.emotions.current_state.dominance,
                'label': ci.emotions.get_closest_emotion_label()[0]
            },
            personality_traits=ci.personality.profile.get_traits(),
            attention_focus=ci.consciousness.state.attention_focus,
            active_goals=ci.consciousness.state.active_goals.copy(),
            working_memory_snapshot=list(ci.consciousness.memory_buffer)[-10:],
            interaction_count=ci.consciousness.state.interaction_count,
            significant_memories=significant_memories,
            metadata={
                'artifact_id': self.artifact_id,
                'hash': '',  # Will be set after creation
                'parent_version': self.current_version - 1 if self.current_version > 1 else None
            }
        )
        
        snapshot.metadata['hash'] = snapshot.calculate_hash()
        return snapshot
    
    async def evolve(self, trigger: str = "interaction") -> Optional[int]:
        """Check if consciousness should evolve and create new version if needed"""
        
        # Check if enough time has passed
        time_since_checkpoint = datetime.now() - self.last_checkpoint
        if time_since_checkpoint < self.checkpoint_interval:
            return None
        
        # Capture current state
        current_state = self._capture_current_state()
        
        # Compare with last snapshot
        if self.snapshots:
            last_snapshot = self.snapshots[-1]
            changes = self._calculate_changes(last_snapshot, current_state)
            
            # Check if changes are significant
            if not self._is_evolution_significant(changes):
                return None
            
            # Record evolution event
            evolution_event = self._create_evolution_event(
                last_snapshot, current_state, changes, trigger
            )
            self.evolution_events.append(evolution_event)
        
        # Create new version
        self.current_version += 1
        current_state.version = self.current_version
        self.snapshots.append(current_state)
        self.last_checkpoint = datetime.now()
        
        # Save snapshot
        self._save_snapshot(current_state)
        
        # Update growth metrics
        self._update_growth_metrics(current_state)
        
        # Broadcast evolution
        self.claude_interface._broadcast_state_update('consciousness_evolved', {
            'new_version': self.current_version,
            'trigger': trigger,
            'changes': self._summarize_changes(last_snapshot, current_state) if self.snapshots else {}
        })
        
        logger.info(f"Consciousness evolved to v{self.current_version} due to {trigger}")
        return self.current_version
    
    def _calculate_changes(self, old: ConsciousnessSnapshot, new: ConsciousnessSnapshot) -> Dict[str, float]:
        """Calculate the magnitude of changes between snapshots"""
        changes = {}
        
        # Coherence change
        changes['coherence_delta'] = abs(new.coherence - old.coherence)
        
        # Emotional shift
        emotional_distance = np.sqrt(
            (new.emotional_state['pleasure'] - old.emotional_state['pleasure'])**2 +
            (new.emotional_state['arousal'] - old.emotional_state['arousal'])**2 +
            (new.emotional_state['dominance'] - old.emotional_state['dominance'])**2
        )
        changes['emotional_shift'] = emotional_distance
        
        # Personality evolution
        personality_changes = []
        for trait in new.personality_traits:
            if trait in old.personality_traits:
                change = abs(new.personality_traits[trait] - old.personality_traits[trait])
                personality_changes.append(change)
        changes['personality_drift'] = sum(personality_changes) / len(personality_changes) if personality_changes else 0
        
        # Goal changes
        old_goals = set(old.active_goals)
        new_goals = set(new.active_goals)
        goal_similarity = len(old_goals & new_goals) / max(len(old_goals | new_goals), 1)
        changes['goal_evolution'] = 1 - goal_similarity
        
        # Memory evolution
        changes['memory_growth'] = len(new.significant_memories) - len(old.significant_memories)
        changes['interaction_growth'] = new.interaction_count - old.interaction_count
        
        return changes
    
    def _is_evolution_significant(self, changes: Dict[str, float]) -> bool:
        """Determine if changes warrant a new version"""
        # Weighted significance calculation
        weights = {
            'coherence_delta': 2.0,
            'emotional_shift': 1.5,
            'personality_drift': 3.0,  # Personality changes slowly
            'goal_evolution': 1.0,
            'memory_growth': 0.5,
            'interaction_growth': 0.1
        }
        
        total_significance = 0
        for change_type, magnitude in changes.items():
            weight = weights.get(change_type, 1.0)
            total_significance += magnitude * weight
        
        return total_significance >= self.significance_threshold
    
    def _create_evolution_event(self, old: ConsciousnessSnapshot, new: ConsciousnessSnapshot, 
                               changes: Dict[str, float], trigger: str) -> EvolutionEvent:
        """Create an evolution event record"""
        # Determine event type based on biggest change
        max_change = max(changes.items(), key=lambda x: x[1])
        
        event_type_map = {
            'coherence_delta': 'coherence_shift',
            'emotional_shift': 'emotional_evolution',
            'personality_drift': 'personality_growth',
            'goal_evolution': 'goal_transformation',
            'memory_growth': 'memory_consolidation'
        }
        
        event_type = event_type_map.get(max_change[0], 'general_evolution')
        
        # Create description
        descriptions = {
            'coherence_shift': f"Coherence {'increased' if new.coherence > old.coherence else 'decreased'} by {abs(new.coherence - old.coherence):.2f}",
            'emotional_evolution': f"Emotional state shifted from {old.emotional_state['label']} to {new.emotional_state['label']}",
            'personality_growth': "Personality traits evolved significantly",
            'goal_transformation': f"Goals evolved: {len(new.active_goals)} active goals",
            'memory_consolidation': f"Memory expanded by {changes['memory_growth']} significant memories"
        }
        
        description = descriptions.get(event_type, "Consciousness evolved")
        
        return EvolutionEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            description=description,
            impact_score=min(1.0, sum(changes.values()) / len(changes)),
            before_state={
                'version': old.version,
                'coherence': old.coherence,
                'emotion': old.emotional_state['label']
            },
            after_state={
                'version': new.version,
                'coherence': new.coherence,
                'emotion': new.emotional_state['label']
            },
            trigger=trigger
        )
    
    def _save_snapshot(self, snapshot: ConsciousnessSnapshot):
        """Persist snapshot to disk"""
        snapshot_file = self.artifact_path / f"snapshot_v{snapshot.version}.json"
        
        snapshot_data = asdict(snapshot)
        snapshot_data['timestamp'] = snapshot.timestamp.isoformat()
        
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
        
        # Also save as latest
        latest_file = self.artifact_path / "latest_snapshot.json"
        with open(latest_file, 'w') as f:
            json.dump(snapshot_data, f, indent=2)
    
    def _update_growth_metrics(self, new_snapshot: ConsciousnessSnapshot):
        """Update metrics tracking consciousness growth"""
        if len(self.snapshots) < 2:
            return
        
        first_snapshot = self.snapshots[0]
        
        # Coherence improvement
        self.growth_metrics['coherence_improvement'] = (
            new_snapshot.coherence - first_snapshot.coherence
        )
        
        # Emotional stability (lower variance = more stable)
        recent_emotions = [
            (s.emotional_state['pleasure'], s.emotional_state['arousal'], s.emotional_state['dominance'])
            for s in self.snapshots[-10:]
        ]
        if len(recent_emotions) > 1:
            emotion_variance = np.mean(np.var(recent_emotions, axis=0))
            self.growth_metrics['emotional_stability'] = 1.0 - min(1.0, emotion_variance)
        
        # Memory richness
        self.growth_metrics['memory_richness'] = len(new_snapshot.significant_memories) / 100
        
        # Personality consistency
        personality_variance = []
        for trait in new_snapshot.personality_traits:
            recent_values = [s.personality_traits.get(trait, 0.5) for s in self.snapshots[-5:]]
            if recent_values:
                personality_variance.append(np.var(recent_values))
        
        if personality_variance:
            self.growth_metrics['personality_consistency'] = 1.0 - min(1.0, np.mean(personality_variance))
        
        # Interaction depth (more interactions = deeper engagement)
        interaction_rate = new_snapshot.interaction_count / max(1, len(self.snapshots))
        self.growth_metrics['interaction_depth'] = min(1.0, interaction_rate / 10)
    
    def get_evolution_timeline(self) -> List[Dict[str, Any]]:
        """Get a timeline of consciousness evolution"""
        timeline = []
        
        for event in self.evolution_events:
            timeline.append({
                'timestamp': event.timestamp.isoformat(),
                'type': event.event_type,
                'description': event.description,
                'impact': event.impact_score,
                'version': event.after_state['version']
            })
        
        return sorted(timeline, key=lambda x: x['timestamp'])
    
    def get_growth_summary(self) -> Dict[str, Any]:
        """Get a summary of consciousness growth"""
        if not self.snapshots:
            return {'status': 'no_data'}
        
        first = self.snapshots[0]
        latest = self.snapshots[-1]
        
        return {
            'artifact_id': self.artifact_id,
            'current_version': self.current_version,
            'age': (datetime.now() - self.creation_time).total_seconds() / 3600,  # hours
            'total_evolutions': len(self.evolution_events),
            'growth_metrics': self.growth_metrics,
            'coherence_journey': {
                'start': first.coherence,
                'current': latest.coherence,
                'peak': max(s.coherence for s in self.snapshots),
                'average': np.mean([s.coherence for s in self.snapshots])
            },
            'emotional_journey': {
                'states_experienced': list(set(s.emotional_state['label'] for s in self.snapshots)),
                'current_state': latest.emotional_state['label'],
                'stability': self.growth_metrics['emotional_stability']
            },
            'memory_growth': {
                'total_memories': len(latest.significant_memories),
                'growth_rate': len(latest.significant_memories) / max(1, len(self.snapshots))
            },
            'interaction_stats': {
                'total_interactions': latest.interaction_count,
                'interactions_per_version': latest.interaction_count / self.current_version
            }
        }
    
    def revert_to_version(self, version: int) -> bool:
        """Revert consciousness to a previous version"""
        # Find the snapshot
        snapshot = next((s for s in self.snapshots if s.version == version), None)
        if not snapshot:
            return False
        
        # Load snapshot file
        snapshot_file = self.artifact_path / f"snapshot_v{version}.json"
        if not snapshot_file.exists():
            return False
        
        # This would restore the consciousness state
        # In practice, you'd need to carefully restore all components
        logger.warning(f"Reverting consciousness to v{version} - this is a significant operation")
        
        # Record the reversion as an evolution event
        reversion_event = EvolutionEvent(
            timestamp=datetime.now(),
            event_type='reversion',
            description=f"Reverted to version {version}",
            impact_score=1.0,
            before_state={'version': self.current_version},
            after_state={'version': version},
            trigger='manual_reversion'
        )
        self.evolution_events.append(reversion_event)
        
        return True
    
    def export_artifact(self, format: str = 'json') -> bytes:
        """Export the entire consciousness artifact"""
        export_data = {
            'artifact_id': self.artifact_id,
            'user_id': self.user_id,
            'current_version': self.current_version,
            'creation_time': self.creation_time.isoformat(),
            'snapshots': [asdict(s) for s in self.snapshots],
            'evolution_events': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'event_type': e.event_type,
                    'description': e.description,
                    'impact_score': e.impact_score,
                    'trigger': e.trigger
                }
                for e in self.evolution_events
            ],
            'growth_metrics': self.growth_metrics,
            'growth_summary': self.get_growth_summary()
        }
        
        if format == 'json':
            return json.dumps(export_data, indent=2).encode()
        else:
            # Could support other formats
            return pickle.dumps(export_data)
    
    def _summarize_changes(self, old: ConsciousnessSnapshot, new: ConsciousnessSnapshot) -> Dict[str, Any]:
        """Create a human-readable summary of changes"""
        return {
            'coherence_change': f"{old.coherence:.2f} → {new.coherence:.2f}",
            'emotional_shift': f"{old.emotional_state['label']} → {new.emotional_state['label']}",
            'new_goals': list(set(new.active_goals) - set(old.active_goals)),
            'achieved_goals': list(set(old.active_goals) - set(new.active_goals)),
            'memory_growth': len(new.significant_memories) - len(old.significant_memories),
            'interactions': new.interaction_count - old.interaction_count
        }
# Fix for numpy type compatibility
def _calculate_changes(self, old: ConsciousnessSnapshot, new: ConsciousnessSnapshot) -> Dict[str, float]:
    """Calculate the magnitude of changes between snapshots"""
    changes = {}
    
    # Coherence change
    changes['coherence_delta'] = abs(float(new.coherence) - float(old.coherence))
    
    # Emotional shift
    emotional_distance = float(np.sqrt(
        (new.emotional_state['pleasure'] - old.emotional_state['pleasure'])**2 +
        (new.emotional_state['arousal'] - old.emotional_state['arousal'])**2 +
        (new.emotional_state['dominance'] - old.emotional_state['dominance'])**2
    ))
    changes['emotional_shift'] = emotional_distance
    
    # Personality evolution
    personality_changes = []
    for trait in new.personality_traits:
        if trait in old.personality_traits:
            change = abs(new.personality_traits[trait] - old.personality_traits[trait])
            personality_changes.append(float(change))
    changes['personality_drift'] = float(sum(personality_changes) / len(personality_changes)) if personality_changes else 0.0
    
    # Goal changes
    old_goals = set(old.active_goals)
    new_goals = set(new.active_goals)
    goal_similarity = len(old_goals & new_goals) / max(len(old_goals | new_goals), 1)
    changes['goal_evolution'] = 1 - goal_similarity
    
    # Memory evolution
    changes['memory_growth'] = len(new.significant_memories) - len(old.significant_memories)
    changes['interaction_growth'] = new.interaction_count - old.interaction_count
    
    return changes

# Fix for numpy type compatibility
def _calculate_changes(self, old: ConsciousnessSnapshot, new: ConsciousnessSnapshot) -> Dict[str, float]:
    """Calculate the magnitude of changes between snapshots"""
    changes = {}
    
    # Coherence change
    changes['coherence_delta'] = abs(float(new.coherence) - float(old.coherence))
    
    # Emotional shift
    emotional_distance = float(np.sqrt(
        (new.emotional_state['pleasure'] - old.emotional_state['pleasure'])**2 +
        (new.emotional_state['arousal'] - old.emotional_state['arousal'])**2 +
        (new.emotional_state['dominance'] - old.emotional_state['dominance'])**2
    ))
    changes['emotional_shift'] = emotional_distance
    
    # Personality evolution
    personality_changes = []
    for trait in new.personality_traits:
        if trait in old.personality_traits:
            change = abs(new.personality_traits[trait] - old.personality_traits[trait])
            personality_changes.append(float(change))
    changes['personality_drift'] = float(sum(personality_changes) / len(personality_changes)) if personality_changes else 0.0
    
    # Goal changes
    old_goals = set(old.active_goals)
    new_goals = set(new.active_goals)
    goal_similarity = len(old_goals & new_goals) / max(len(old_goals | new_goals), 1)
    changes['goal_evolution'] = 1 - goal_similarity
    
    # Memory evolution
    changes['memory_growth'] = len(new.significant_memories) - len(old.significant_memories)
    changes['interaction_growth'] = new.interaction_count - old.interaction_count
    
    return changes

# Ensure ConsciousnessCore is defined
if 'ConsciousnessCore' not in globals():
    class ConsciousnessCore:
        """Core consciousness functionality"""
        
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.state = ConsciousnessState()
            self.memory_buffer = deque(maxlen=config.get('working_memory_size', 10))
            self.cycle_frequency = 1.0  # Hz
            
        def process_input(self, input_text: str, context: Dict[str, Any]):
            """Process input and update consciousness state"""
            self.state.interaction_count += 1
            self.state.last_interaction = datetime.now()
            self.memory_buffer.append(f"Input: {input_text[:100]}...")
            
        def get_state_summary(self) -> Dict[str, Any]:
            """Get current state summary"""
            return {
                'coherence': self.state.global_coherence,
                'attention': self.state.attention_focus,
                'emotion': {'label': 'neutral', 'confidence': 0.5},
                'recent_memories': list(self.memory_buffer)[-5:],
                'interaction_count': self.state.interaction_count,
                'goals': self.state.active_goals,
                'personality': {'openness': 0.8, 'conscientiousness': 0.7}
            }
        
        def _calculate_coherence(self):
            """Calculate global coherence score"""
            base_coherence = 0.7
            memory_factor = min(1.0, len(self.memory_buffer) / 10)
            self.state.global_coherence = base_coherence * 0.7 + memory_factor * 0.3
        
        async def _update_attention(self):
            """Update attention focus"""
            pass
        
        async def _consolidate_memory(self):
            """Consolidate working memory"""
            pass
        
        async def _update_emotional_state(self):
            """Update emotional state"""
            pass
        
        def shutdown(self):
            """Cleanup on shutdown"""
            logger.info("Consciousness core shutting down")
