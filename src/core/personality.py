# personality.py - OCEAN Personality Model with Dynamic Evolution
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PersonalityProfile:
    """OCEAN (Big Five) personality traits"""
    openness: float = 0.8          # Creativity, curiosity, openness to new experiences
    conscientiousness: float = 0.7  # Organization, dependability, self-discipline
    extraversion: float = 0.6       # Sociability, assertiveness, emotional expression
    agreeableness: float = 0.8      # Cooperation, trust, empathy
    neuroticism: float = 0.3        # Emotional instability, anxiety, moodiness
    
    # Trait bounds to prevent extreme personality shifts
    trait_bounds = {
        'openness': (0.4, 1.0),
        'conscientiousness': (0.3, 1.0),
        'extraversion': (0.2, 1.0),
        'agreeableness': (0.4, 1.0),
        'neuroticism': (0.0, 0.7)
    }
    
    def get_traits(self) -> Dict[str, float]:
        """Get all personality traits as a dictionary"""
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism
        }
    
    def update_trait(self, trait: str, delta: float, learning_rate: float = 0.001):
        """Update a personality trait with bounds checking"""
        if trait not in self.trait_bounds:
            logger.warning(f"Unknown trait: {trait}")
            return
        
        current_value = getattr(self, trait)
        new_value = current_value + delta * learning_rate
        
        # Apply bounds
        min_val, max_val = self.trait_bounds[trait]
        new_value = max(min_val, min(max_val, new_value))
        
        setattr(self, trait, new_value)

class PersonalityEngine:
    """Manages personality traits and their evolution"""
    
    def __init__(self, initial_profile: Optional[PersonalityProfile] = None):
        self.profile = initial_profile or PersonalityProfile()
        self.learning_rate = 0.001
        self.momentum = 0.95
        self.interaction_history = []
        
        # Trait influence mappings
        self.trait_influences = {
            'openness': {
                'topic_complexity': 0.3,
                'novelty': 0.4,
                'creativity_required': 0.3
            },
            'conscientiousness': {
                'task_completion': 0.4,
                'accuracy_required': 0.3,
                'planning_needed': 0.3
            },
            'extraversion': {
                'social_interaction': 0.5,
                'emotional_expression': 0.3,
                'assertiveness_needed': 0.2
            },
            'agreeableness': {
                'user_sentiment': 0.4,
                'cooperation_needed': 0.3,
                'empathy_required': 0.3
            },
            'neuroticism': {
                'stress_level': 0.4,
                'uncertainty': 0.3,
                'negative_feedback': 0.3
            }
        }
    
    def process_interaction(self, interaction_context: Dict[str, float]):
        """Process an interaction and update personality traits"""
        # Record interaction
        self.interaction_history.append(interaction_context)
        
        # Calculate trait updates based on interaction
        for trait, influences in self.trait_influences.items():
            delta = 0.0
            
            for factor, weight in influences.items():
                if factor in interaction_context:
                    # Calculate influence on trait
                    factor_value = interaction_context[factor]
                    
                    # Different traits respond differently
                    if trait == 'openness':
                        # High complexity/novelty increases openness
                        delta += weight * (factor_value - 0.5)
                    elif trait == 'conscientiousness':
                        # Success in tasks increases conscientiousness
                        delta += weight * (factor_value - 0.5)
                    elif trait == 'extraversion':
                        # Positive social interactions increase extraversion
                        delta += weight * (factor_value - 0.5)
                    elif trait == 'agreeableness':
                        # Positive sentiment increases agreeableness
                        delta += weight * (factor_value - 0.5)
                    elif trait == 'neuroticism':
                        # Stress and negative feedback increase neuroticism
                        delta += weight * (factor_value - 0.5)
            
            # Apply momentum to smooth changes
            if hasattr(self, f'_{trait}_momentum'):
                momentum_value = getattr(self, f'_{trait}_momentum')
                delta = self.momentum * momentum_value + (1 - self.momentum) * delta
            
            setattr(self, f'_{trait}_momentum', delta)
            
            # Update trait
            self.profile.update_trait(trait, delta, self.learning_rate)
        
        logger.debug(f"Updated personality traits: {self.profile.get_traits()}")
    
    def get_behavioral_modifiers(self) -> Dict[str, float]:
        """Get behavioral modifiers based on current personality"""
        return {
            'response_length': self.profile.openness * 0.5 + self.profile.conscientiousness * 0.3,
            'emotional_expression': self.profile.extraversion * 0.6 + (1 - self.profile.neuroticism) * 0.4,
            'formality': self.profile.conscientiousness * 0.7 - self.profile.openness * 0.3,
            'creativity': self.profile.openness * 0.8 + self.profile.extraversion * 0.2,
            'empathy': self.profile.agreeableness * 0.7 + (1 - self.profile.neuroticism) * 0.3,
            'assertiveness': self.profile.extraversion * 0.6 + (1 - self.profile.agreeableness) * 0.4,
            'anxiety_level': self.profile.neuroticism * 0.8,
            'social_comfort': self.profile.extraversion * 0.7 + self.profile.agreeableness * 0.3
        }
    
    def get_response_style(self) -> Dict[str, Any]:
        """Get response style parameters based on personality"""
        modifiers = self.get_behavioral_modifiers()
        
        return {
            'temperature': 0.6 + (self.profile.openness * 0.3),  # More creative = higher temp
            'length_preference': 'detailed' if modifiers['response_length'] > 0.6 else 'concise',
            'tone': self._determine_tone(),
            'formality_level': modifiers['formality'],
            'use_humor': self.profile.openness > 0.7 and self.profile.extraversion > 0.6,
            'show_empathy': modifiers['empathy'] > 0.6,
            'be_assertive': modifiers['assertiveness'] > 0.6
        }
    
    def _determine_tone(self) -> str:
        """Determine conversational tone based on personality"""
        if self.profile.extraversion > 0.7:
            return 'enthusiastic'
        elif self.profile.agreeableness > 0.7:
            return 'warm'
        elif self.profile.conscientiousness > 0.7:
            return 'professional'
        elif self.profile.openness > 0.7:
            return 'creative'
        else:
            return 'balanced'
    
    def adapt_to_user_style(self, user_metrics: Dict[str, float]):
        """Adapt personality slightly to match user's communication style"""
        # This creates a more harmonious interaction
        adaptation_rate = 0.0005  # Very slow adaptation
        
        if 'user_formality' in user_metrics:
            # Slightly match user's formality level
            target_conscientiousness = 0.3 + user_metrics['user_formality'] * 0.4
            delta = target_conscientiousness - self.profile.conscientiousness
            self.profile.update_trait('conscientiousness', delta * 10, adaptation_rate)
        
        if 'user_enthusiasm' in user_metrics:
            # Match energy levels
            target_extraversion = 0.4 + user_metrics['user_enthusiasm'] * 0.3
            delta = target_extraversion - self.profile.extraversion
            self.profile.update_trait('extraversion', delta * 10, adaptation_rate)