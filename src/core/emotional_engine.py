# emotional_engine.py - PAD Emotional Model with Smooth Transitions
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmotionalState:
    """PAD (Pleasure-Arousal-Dominance) emotional state"""
    pleasure: float = 0.6  # Valence: negative to positive
    arousal: float = 0.5   # Activation: calm to excited
    dominance: float = 0.5 # Control: submissive to dominant
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector"""
        return np.array([self.pleasure, self.arousal, self.dominance])
    
    def from_vector(self, vector: np.ndarray):
        """Update from numpy vector"""
        self.pleasure = float(vector[0])
        self.arousal = float(vector[1])
        self.dominance = float(vector[2])
    
    def distance_to(self, other: 'EmotionalState') -> float:
        """Calculate Euclidean distance to another emotional state"""
        return np.linalg.norm(self.to_vector() - other.to_vector())

class EmotionalEngine:
    """Manages emotional states and transitions using PAD model"""
    
    def __init__(self, baseline_state: Optional[EmotionalState] = None):
        self.current_state = baseline_state or EmotionalState()
        self.baseline_state = EmotionalState(
            pleasure=self.current_state.pleasure,
            arousal=self.current_state.arousal,
            dominance=self.current_state.dominance
        )
        
        # Emotion history for smooth transitions
        self.state_history = deque(maxlen=20)
        self.state_history.append(self.current_state)
        
        # Transition parameters
        self.transition_speed = 0.1  # How quickly emotions change
        self.baseline_pull = 0.05    # How strongly we return to baseline
        self.inertia = 0.7          # Resistance to change
        
        # Define emotion categories in PAD space
        self.emotion_categories = self._define_emotion_categories()
    
    def _define_emotion_categories(self) -> Dict[str, EmotionalState]:
        """Define common emotions in PAD space"""
        return {
            # High pleasure, high arousal, high dominance
            'joy': EmotionalState(0.8, 0.7, 0.7),
            'excitement': EmotionalState(0.7, 0.9, 0.6),
            
            # High pleasure, low arousal
            'contentment': EmotionalState(0.7, 0.3, 0.6),
            'serenity': EmotionalState(0.8, 0.2, 0.5),
            
            # Low pleasure, high arousal
            'anger': EmotionalState(0.2, 0.8, 0.8),
            'fear': EmotionalState(0.2, 0.8, 0.2),
            'anxiety': EmotionalState(0.3, 0.7, 0.3),
            
            # Low pleasure, low arousal
            'sadness': EmotionalState(0.2, 0.3, 0.3),
            'boredom': EmotionalState(0.4, 0.2, 0.5),
            
            # Neutral states
            'neutral': EmotionalState(0.5, 0.5, 0.5),
            'curiosity': EmotionalState(0.6, 0.6, 0.6),
            
            # Complex emotions
            'surprise': EmotionalState(0.5, 0.8, 0.4),
            'disgust': EmotionalState(0.2, 0.6, 0.6),
            'contempt': EmotionalState(0.3, 0.4, 0.7),
            'pride': EmotionalState(0.7, 0.5, 0.8),
            'shame': EmotionalState(0.2, 0.4, 0.2),
            'guilt': EmotionalState(0.3, 0.5, 0.3)
        }
    
    def update_emotional_state(self, stimulus: Dict[str, Any]):
        """Update emotional state based on stimulus"""
        # Calculate target emotional state based on stimulus
        target_state = self._calculate_target_state(stimulus)
        
        # Smooth transition from current to target
        new_state = self._smooth_transition(self.current_state, target_state)
        
        # Apply baseline pull
        new_state = self._apply_baseline_pull(new_state)
        
        # Ensure bounds [0, 1]
        new_state.pleasure = max(0, min(1, new_state.pleasure))
        new_state.arousal = max(0, min(1, new_state.arousal))
        new_state.dominance = max(0, min(1, new_state.dominance))
        
        # Update current state
        self.current_state = new_state
        self.state_history.append(new_state)
        
        logger.debug(f"Emotional state updated: P={new_state.pleasure:.2f}, "
                    f"A={new_state.arousal:.2f}, D={new_state.dominance:.2f}")
    
    def _calculate_target_state(self, stimulus: Dict[str, Any]) -> EmotionalState:
        """Calculate target emotional state from stimulus"""
        target = EmotionalState(
            pleasure=self.current_state.pleasure,
            arousal=self.current_state.arousal,
            dominance=self.current_state.dominance
        )
        
        # Sentiment affects pleasure
        if 'sentiment' in stimulus:
            sentiment = stimulus['sentiment']
            target.pleasure += (sentiment - 0.5) * 0.4
        
        # Urgency affects arousal
        if 'urgency' in stimulus:
            target.arousal += stimulus['urgency'] * 0.3
        
        # Success/failure affects dominance
        if 'success' in stimulus:
            if stimulus['success']:
                target.dominance += 0.2
                target.pleasure += 0.1
            else:
                target.dominance -= 0.1
                target.pleasure -= 0.1
        
        # Social interaction effects
        if stimulus.get('social_interaction', False):
            target.arousal += 0.1
            if stimulus.get('positive_interaction', True):
                target.pleasure += 0.1
        
        return target
    
    def _smooth_transition(self, current: EmotionalState, 
                          target: EmotionalState) -> EmotionalState:
        """Smooth transition between emotional states"""
        current_vec = current.to_vector()
        target_vec = target.to_vector()
        
        # Calculate transition with inertia
        direction = target_vec - current_vec
        step = direction * self.transition_speed * (1 - self.inertia)
        
        # Add some noise for naturalness
        noise = np.random.normal(0, 0.01, 3)
        
        new_vec = current_vec + step + noise
        
        new_state = EmotionalState()
        new_state.from_vector(new_vec)
        
        return new_state
    
    def _apply_baseline_pull(self, state: EmotionalState) -> EmotionalState:
        """Apply gradual pull toward baseline emotional state"""
        current_vec = state.to_vector()
        baseline_vec = self.baseline_state.to_vector()
        
        pull_direction = baseline_vec - current_vec
        pull_strength = self.baseline_pull * (1 - self.get_emotional_intensity())
        
        new_vec = current_vec + pull_direction * pull_strength
        
        new_state = EmotionalState()
        new_state.from_vector(new_vec)
        
        return new_state
    
    def get_emotional_intensity(self) -> float:
        """Get current emotional intensity (distance from neutral)"""
        neutral = EmotionalState(0.5, 0.5, 0.5)
        return min(1.0, self.current_state.distance_to(neutral) / np.sqrt(0.75))
    
    def get_closest_emotion_label(self) -> Tuple[str, float]:
        """Get closest named emotion and confidence"""
        min_distance = float('inf')
        closest_emotion = 'neutral'
        
        for emotion_name, emotion_state in self.emotion_categories.items():
            distance = self.current_state.distance_to(emotion_state)
            if distance < min_distance:
                min_distance = distance
                closest_emotion = emotion_name
        
        # Convert distance to confidence (closer = higher confidence)
        confidence = max(0, 1 - min_distance / np.sqrt(3))
        
        return closest_emotion, confidence
    
    def get_emotional_color(self) -> Tuple[int, int, int]:
        """Get RGB color representation of current emotion"""
        # Map PAD to RGB for visualization
        r = int(255 * self.current_state.pleasure)
        g = int(255 * (1 - self.current_state.arousal))
        b = int(255 * self.current_state.dominance)
        
        return (r, g, b)
    
    def get_state_trajectory(self, n_steps: int = 10) -> List[EmotionalState]:
        """Get recent emotional trajectory"""
        if len(self.state_history) < n_steps:
            return list(self.state_history)
        return list(self.state_history)[-n_steps:]
    
    def modulate_response(self, base_response: str) -> str:
        """Modulate a response based on current emotional state"""
        # This is a placeholder - in practice, you'd use more sophisticated
        # natural language processing to adjust tone, word choice, etc.
        
        emotion_label, confidence = self.get_closest_emotion_label()
        intensity = self.get_emotional_intensity()
        
        # Add emotional context to response
        if confidence > 0.7 and intensity > 0.5:
            emotional_prefix = {
                'joy': "I'm delighted to say that ",
                'excitement': "How exciting! ",
                'contentment': "I'm pleased to share that ",
                'sadness': "I must sadly inform you that ",
                'fear': "I'm concerned that ",
                'anger': "I must express that ",
                'curiosity': "Interestingly, ",
            }.get(emotion_label, "")
            
            return emotional_prefix + base_response
        
        return base_response