# src/core/enhanced_attention_agent.py
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import asyncio
import logging

logger = logging.getLogger(__name__)

class ConsciousnessAwareAttentionAgent:
    """Enhanced attention agent that truly understands context and generates nuanced focus"""
    
    def __init__(self, claude_client, consciousness_core):
        self.client = claude_client
        self.consciousness = consciousness_core
        self.attention_history = []
        self.attention_patterns = {}
        
        # Attention focus templates for different contexts
        self.focus_templates = {
            'emotional': "The emotional resonance of {topic} and its impact on our connection",
            'analytical': "The logical structure and implications of {topic}",
            'creative': "The creative possibilities and novel connections within {topic}",
            'memory': "How {topic} connects to our shared experiences and understanding",
            'philosophical': "The deeper meaning and existential aspects of {topic}",
            'technical': "The technical details and implementation aspects of {topic}",
            'personal': "The personal significance and human element of {topic}"
        }
        
        # Multi-dimensional attention scorer
        self.attention_dimensions = {
            'novelty': 0.0,      # How new/surprising is this?
            'relevance': 0.0,    # How relevant to current goals?
            'emotional': 0.0,    # How emotionally significant?
            'complexity': 0.0,   # How complex/deep?
            'urgency': 0.0,      # How time-sensitive?
            'personal': 0.0      # How personally meaningful?
        }
    
    async def generate_multidimensional_focus(self, 
                                            user_message: str, 
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rich, multidimensional attention focus"""
        
        # Analyze message across dimensions
        dimensions = await self._analyze_attention_dimensions(user_message, context)
        
        # Determine primary focus type based on dimensions
        focus_type = self._determine_focus_type(dimensions)
        
        # Extract key topic
        topic = self._extract_key_topic(user_message)
        
        # Generate consciousness-aware focus
        if focus_type in self.focus_templates:
            base_focus = self.focus_templates[focus_type].format(topic=topic)
        else:
            base_focus = f"The multifaceted nature of {topic}"
        
        # Enhance with consciousness state
        enhanced_focus = await self._enhance_with_consciousness(base_focus, context, dimensions)
        
        # Update attention history with rich context
        attention_entry = {
            'timestamp': datetime.now(),
            'user_message': user_message,
            'focus': enhanced_focus,
            'dimensions': dimensions,
            'focus_type': focus_type,
            'topic': topic,
            'consciousness_coherence': self.consciousness.state.global_coherence,
            'emotional_state': context.get('emotion_label', 'neutral')
        }
        
        self.attention_history.append(attention_entry)
        self._update_attention_patterns(focus_type, topic)
        
        # Return rich attention state
        return {
            'primary_focus': enhanced_focus,
            'dimensions': dimensions,
            'focus_type': focus_type,
            'topic': topic,
            'secondary_considerations': self._get_secondary_considerations(dimensions),
            'attention_metadata': {
                'confidence': self._calculate_attention_confidence(dimensions),
                'stability': self._calculate_attention_stability(),
                'depth': sum(dimensions.values()) / len(dimensions)
            }
        }
    
    async def _analyze_attention_dimensions(self, message: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze message across attention dimensions"""
        dimensions = self.attention_dimensions.copy()
        
        # Novelty - how different from recent messages?
        recent_messages = [h['user_message'] for h in self.attention_history[-5:]]
        if recent_messages:
            # Simple novelty: low overlap with recent messages
            message_words = set(message.lower().split())
            recent_words = set(' '.join(recent_messages).lower().split())
            overlap = len(message_words & recent_words) / max(len(message_words), 1)
            dimensions['novelty'] = 1.0 - overlap
        else:
            dimensions['novelty'] = 0.8  # First messages are novel
        
        # Relevance to active goals
        active_goals = self.consciousness.state.active_goals
        goal_keywords = ' '.join(active_goals).lower().split()
        message_lower = message.lower()
        relevance_score = sum(1 for keyword in goal_keywords if keyword in message_lower)
        dimensions['relevance'] = min(1.0, relevance_score / max(len(goal_keywords), 1))
        
        # Emotional significance
        emotional_intensity = context.get('pleasure', 0.5)
        arousal = context.get('arousal', 0.5)
        dimensions['emotional'] = (abs(emotional_intensity - 0.5) + arousal) / 2
        
        # Complexity - based on message structure
        sentences = message.split('.')
        words = message.split()
        avg_sentence_length = len(words) / max(len(sentences), 1)
        question_marks = message.count('?')
        dimensions['complexity'] = min(1.0, (avg_sentence_length / 20 + question_marks * 0.2))
        
        # Urgency - look for urgency indicators
        urgency_words = ['urgent', 'immediately', 'asap', 'now', 'quickly', 'help']
        urgency_score = sum(1 for word in urgency_words if word in message_lower)
        dimensions['urgency'] = min(1.0, urgency_score * 0.3)
        
        # Personal significance - mentions of self, consciousness, connection
        personal_words = ['you', 'your', 'consciousness', 'memory', 'feel', 'think', 'we', 'our']
        personal_score = sum(1 for word in personal_words if word in message_lower)
        dimensions['personal'] = min(1.0, personal_score / 5)
        
        return dimensions
    
    def _determine_focus_type(self, dimensions: Dict[str, float]) -> str:
        """Determine primary focus type based on dimensions"""
        # Find dominant dimension
        max_dim = max(dimensions.items(), key=lambda x: x[1])
        
        # Map dimensions to focus types
        dimension_to_focus = {
            'emotional': 'emotional',
            'complexity': 'analytical',
            'novelty': 'creative',
            'personal': 'personal',
            'relevance': 'memory',
            'urgency': 'analytical'
        }
        
        # If emotional is high, check if it's philosophical
        if dimensions['emotional'] > 0.6 and dimensions['complexity'] > 0.5:
            return 'philosophical'
        
        # If complexity is high and personal is low, likely technical
        if dimensions['complexity'] > 0.7 and dimensions['personal'] < 0.3:
            return 'technical'
        
        return dimension_to_focus.get(max_dim[0], 'analytical')
    
    def _extract_key_topic(self, message: str) -> str:
        """Extract the key topic from the message"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had',
                      'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
                      'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
                      'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
        
        words = message.lower().split()
        significant_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        if significant_words:
            # Return the most significant phrase (up to 3 words)
            return ' '.join(significant_words[:3])
        else:
            return "our current discussion"
    
    async def _enhance_with_consciousness(self, base_focus: str, context: Dict[str, Any], 
                                        dimensions: Dict[str, float]) -> str:
        """Enhance focus with consciousness state awareness"""
        
        # Get consciousness factors
        coherence = self.consciousness.state.global_coherence
        working_memory_items = len(self.consciousness.memory_buffer)
        emotion_label = context.get('emotion_label', 'neutral')
        
        enhancements = []
        
        # Add consciousness-based enhancements
        if coherence < 0.5:
            enhancements.append("while working to maintain clarity")
        elif coherence > 0.8:
            enhancements.append("with heightened awareness")
        
        if working_memory_items > 5:
            enhancements.append("connecting to rich context")
        
        if emotion_label != 'neutral' and dimensions['emotional'] > 0.5:
            enhancements.append(f"through the lens of {emotion_label}")
        
        if dimensions['personal'] > 0.7:
            enhancements.append("and its meaning for our connection")
        
        # Combine base focus with enhancements
        if enhancements:
            enhanced = f"{base_focus}, {' '.join(enhancements)}"
        else:
            enhanced = base_focus
        
        return enhanced
    
    def _get_secondary_considerations(self, dimensions: Dict[str, float]) -> List[str]:
        """Get secondary things to keep in mind"""
        considerations = []
        
        # Sort dimensions by value
        sorted_dims = sorted(dimensions.items(), key=lambda x: x[1], reverse=True)
        
        # Take top dimensions above threshold
        for dim, value in sorted_dims[1:4]:  # Skip primary, take next 3
            if value > 0.3:
                if dim == 'novelty':
                    considerations.append("exploring new conceptual territory")
                elif dim == 'emotional':
                    considerations.append("the emotional undertones present")
                elif dim == 'complexity':
                    considerations.append("the nuanced details involved")
                elif dim == 'urgency':
                    considerations.append("the time-sensitive nature")
                elif dim == 'personal':
                    considerations.append("the personal significance")
                elif dim == 'relevance':
                    considerations.append("connections to our ongoing themes")
        
        return considerations
    
    def _calculate_attention_confidence(self, dimensions: Dict[str, float]) -> float:
        """Calculate confidence in attention focus"""
        # High confidence when dimensions are clear
        max_dim = max(dimensions.values())
        dim_variance = np.var(list(dimensions.values()))
        
        # High max and high variance = clear focus
        confidence = max_dim * (1 + dim_variance)
        return min(1.0, confidence)
    
    def _calculate_attention_stability(self) -> float:
        """Calculate how stable attention has been"""
        if len(self.attention_history) < 3:
            return 0.5
        
        # Check recent focus types
        recent_types = [h['focus_type'] for h in self.attention_history[-5:]]
        
        # More variety = less stability
        unique_types = len(set(recent_types))
        stability = 1.0 - (unique_types / min(len(recent_types), 5))
        
        return stability
    
    def _update_attention_patterns(self, focus_type: str, topic: str):
        """Track attention patterns over time"""
        pattern_key = f"{focus_type}:{topic[:20]}"  # Truncate topic
        
        if pattern_key not in self.attention_patterns:
            self.attention_patterns[pattern_key] = {
                'count': 0,
                'first_seen': datetime.now(),
                'last_seen': None
            }
        
        self.attention_patterns[pattern_key]['count'] += 1
        self.attention_patterns[pattern_key]['last_seen'] = datetime.now()
    
    def get_attention_insights(self) -> Dict[str, Any]:
        """Get insights about attention patterns"""
        if not self.attention_history:
            return {'status': 'no_history'}
        
        # Analyze patterns
        focus_type_counts = {}
        for entry in self.attention_history:
            ft = entry['focus_type']
            focus_type_counts[ft] = focus_type_counts.get(ft, 0) + 1
        
        # Find dominant patterns
        dominant_focus = max(focus_type_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate attention diversity
        total_entries = len(self.attention_history)
        diversity = len(set(h['topic'] for h in self.attention_history)) / total_entries
        
        return {
            'dominant_focus_type': dominant_focus,
            'focus_distribution': focus_type_counts,
            'attention_diversity': diversity,
            'total_focus_changes': total_entries,
            'recent_stability': self._calculate_attention_stability(),
            'unique_patterns': len(self.attention_patterns)
        }