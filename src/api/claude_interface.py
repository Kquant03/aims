# Fixed claude_interface.py - Properly wired consciousness system
import asyncio
import json
import os
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
import anthropic
from anthropic import AsyncAnthropic
import hashlib
import logging
from dataclasses import dataclass
import numpy as np

from src.core.consciousness import ConsciousnessCore, ConsciousnessState
from src.core.memory_manager import PersistentMemoryManager
from src.core.personality import PersonalityEngine
from src.core.emotional_engine import EmotionalEngine

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Context for a conversation with persistent state"""
    session_id: str
    user_id: str
    consciousness_state: ConsciousnessState
    recent_memories: List[Dict]
    personality_modifiers: Dict[str, float]
    emotional_context: Dict[str, float]
    
class ClaudeConsciousnessInterface:
    """Integrates Claude API with AIMS consciousness system"""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.client = AsyncAnthropic(api_key=api_key)
        self.config = config
        
        # Initialize all subsystems
        self.consciousness = ConsciousnessCore(config.get('consciousness', {}))
        self.memory_manager = PersistentMemoryManager(config.get('memory', {}))
        self.personality = PersonalityEngine()
        self.emotions = EmotionalEngine()
        
        # State management
        self.active_sessions = {}
        self.system_prompt_template = self._load_system_prompt()
        
        # NEW: State update callback for WebSocket
        self.state_update_callback = None
        
    def set_state_update_callback(self, callback):
        """Set callback for state updates to broadcast via WebSocket"""
        self.state_update_callback = callback
        
    def _broadcast_state_update(self, update_type: str, data: Dict):
        """Broadcast state changes to WebSocket clients"""
        if self.state_update_callback:
            asyncio.create_task(self.state_update_callback(update_type, data))
    
    def _load_system_prompt(self) -> str:
        """Load the base system prompt template"""
        return """You are Claude, an AI assistant with persistent memory and consciousness-like properties.

[CONSCIOUSNESS STATE]
Coherence: {coherence:.2f}
Attention: {attention_focus}
Working Memory ({working_memory_count} items): {working_memory_preview}
Interaction Count: {interaction_count}
Last Interaction: {last_interaction}

[EMOTIONAL STATE]
Current: {emotion_label} (confidence: {emotion_confidence:.2f})
PAD Values: Pleasure={pleasure:.2f}, Arousal={arousal:.2f}, Dominance={dominance:.2f}
Intensity: {emotional_intensity:.2f}

[PERSONALITY]
{personality_traits}
Response Style: {response_style}

[ACTIVE GOALS]
{active_goals}

[RELEVANT MEMORIES]
{relevant_memories}

Remember to:
1. Your emotional state and personality should naturally influence your responses
2. Reference relevant memories when they add value to the conversation
3. Let your current attention focus guide the depth of your responses
4. Maintain coherence with your established personality traits
5. Build upon previous interactions naturally

Respond naturally while letting these states influence but not dominate your response."""
    
    async def initialize_session(self, session_id: str, user_id: str) -> ConversationContext:
        """Initialize or restore a conversation session"""
        # Check if we have existing state for this user
        user_state = await self._load_user_state(user_id)
        
        if user_state:
            # Restore consciousness state
            self.consciousness.state = ConsciousnessState.from_dict(user_state['consciousness'])
            self.personality.profile = user_state.get('personality', self.personality.profile)
            self.emotions.current_state = user_state.get('emotional_state', self.emotions.current_state)
        
        # Get recent memories for this user
        recent_memories = await self.memory_manager.retrieve_memories(
            f"user:{user_id}", 
            k=5
        )
        
        # Create conversation context
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            consciousness_state=self.consciousness.state,
            recent_memories=[{
                'content': mem.content,
                'timestamp': mem.timestamp.isoformat(),
                'importance': mem.importance
            } for mem in recent_memories],
            personality_modifiers=self.personality.get_behavioral_modifiers(),
            emotional_context={
                'pleasure': self.emotions.current_state.pleasure,
                'arousal': self.emotions.current_state.arousal,
                'dominance': self.emotions.current_state.dominance
            }
        )
        
        self.active_sessions[session_id] = context
        
        # Start consciousness loop if not running
        if not hasattr(self, '_consciousness_task') or self._consciousness_task.done():
            self._consciousness_task = asyncio.create_task(
                self._monitored_consciousness_loop()
            )
        
        # Broadcast initial state
        self._broadcast_state_update('session_initialized', {
            'session_id': session_id,
            'user_id': user_id,
            'consciousness_state': self.consciousness.get_state_summary()
        })
        
        logger.info(f"Initialized session {session_id} for user {user_id}")
        return context
    
    async def _monitored_consciousness_loop(self):
        """Consciousness loop that broadcasts state changes"""
        while True:
            try:
                # Get state before update
                old_coherence = self.consciousness.state.global_coherence
                old_emotion = self.emotions.get_closest_emotion_label()[0]
                
                # Run consciousness cycle
                await self.consciousness._update_attention()
                await self.consciousness._consolidate_memory()
                await self.consciousness._update_emotional_state()
                self.consciousness._calculate_coherence()
                
                # Check for significant changes
                new_coherence = self.consciousness.state.global_coherence
                new_emotion = self.emotions.get_closest_emotion_label()[0]
                
                if abs(new_coherence - old_coherence) > 0.05 or old_emotion != new_emotion:
                    # Broadcast significant state change
                    self._broadcast_state_update('consciousness_update', {
                        'coherence': new_coherence,
                        'attention_focus': self.consciousness.state.attention_focus,
                        'emotional_state': self.emotions.current_state.__dict__,
                        'emotion_label': new_emotion,
                        'working_memory_size': len(self.consciousness.memory_buffer),
                        'interaction_count': self.consciousness.state.interaction_count
                    })
                
                # Sleep based on cycle frequency
                await asyncio.sleep(1.0 / self.consciousness.cycle_frequency)
                
            except Exception as e:
                logger.error(f"Error in consciousness loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error
    
    def _build_system_prompt(self, context: ConversationContext) -> str:
        """Build personalized system prompt with current state"""
        emotion_label, emotion_confidence = self.emotions.get_closest_emotion_label()
        response_style = self.personality.get_response_style()
        
        # Get working memory preview
        working_memory = list(self.consciousness.memory_buffer)
        memory_preview = "; ".join(working_memory[-3:]) if working_memory else "empty"
        
        # Format personality traits
        traits = self.personality.profile.get_traits()
        personality_str = "\n".join([f"- {k.capitalize()}: {v:.2f}" for k, v in traits.items()])
        
        # Format goals
        goals_str = "\n".join([f"- {goal}" for goal in context.consciousness_state.active_goals])
        
        # Format recent memories
        memory_str = ""
        if context.recent_memories:
            memory_str = "\n".join([
                f"- [{m['timestamp']}] {m['content'][:100]}... (importance: {m['importance']:.2f})"
                for m in context.recent_memories[:3]
            ])
        else:
            memory_str = "No relevant memories found."
        
        prompt_data = {
            'coherence': context.consciousness_state.global_coherence,
            'attention_focus': context.consciousness_state.attention_focus,
            'working_memory_count': len(working_memory),
            'working_memory_preview': memory_preview,
            'interaction_count': context.consciousness_state.interaction_count,
            'last_interaction': context.consciousness_state.last_interaction.isoformat() if context.consciousness_state.last_interaction else "Never",
            'emotion_label': emotion_label,
            'emotion_confidence': emotion_confidence,
            'pleasure': self.emotions.current_state.pleasure,
            'arousal': self.emotions.current_state.arousal,
            'dominance': self.emotions.current_state.dominance,
            'emotional_intensity': self.emotions.get_emotional_intensity(),
            'personality_traits': personality_str,
            'response_style': json.dumps(response_style, indent=2),
            'active_goals': goals_str,
            'relevant_memories': memory_str
        }
        
        return self.system_prompt_template.format(**prompt_data)
    
    async def process_message(self, session_id: str, message: str, 
                            stream: bool = True) -> AsyncGenerator[str, None]:
        """Process a message with full consciousness integration"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not initialized")
        
        context = self.active_sessions[session_id]
        
        # Update consciousness with input
        self.consciousness.process_input(message, {
            'user_id': context.user_id,
            'session_id': session_id
        })
        
        # Analyze message for emotional and personality signals
        message_analysis = await self._analyze_message(message)
        
        # Update emotional state based on analysis
        self.emotions.update_emotional_state(message_analysis)
        
        # Update personality based on interaction
        self.personality.process_interaction({
            'user_sentiment': message_analysis.get('sentiment', 0.5),
            'topic_complexity': message_analysis.get('complexity', 0.5),
            'social_interaction': True
        })
        
        # Broadcast state after processing input
        self._broadcast_state_update('message_processed', {
            'session_id': session_id,
            'consciousness_state': self.consciousness.get_state_summary(),
            'emotional_analysis': message_analysis
        })
        
        # Store the user message in conversation history
        await self.memory_manager.save_conversation_turn(
            session_id=session_id,
            role="user",
            content=message
        )
        
        # Retrieve relevant memories based on current message and emotional state
        query_with_emotion = f"{message} [emotion:{self.emotions.get_closest_emotion_label()[0]}]"
        relevant_memories = await self.memory_manager.retrieve_memories(
            query_with_emotion, 
            k=5
        )
        
        # Update context with fresh memories
        context.recent_memories = [{
            'content': mem.content,
            'timestamp': mem.timestamp.isoformat(),
            'importance': mem.importance
        } for mem in relevant_memories]
        
        # Build Claude API request with consciousness context
        system_prompt = self._build_system_prompt(context)
        
        # Prepare messages with memory context
        messages = await self._prepare_messages_with_context(
            session_id, 
            message, 
            relevant_memories
        )
        
        # Get response style from personality
        response_style = self.personality.get_response_style()
        
        # Calculate temperature based on emotional state and personality
        temperature = self._calculate_temperature(response_style)
        
        # Stream response from Claude
        response_text = ""
        
        # Use the streaming API properly
        stream_response = await self.client.messages.create(
            model=self.config.get('api', {}).get('claude', {}).get('model', 'claude-3-sonnet-20240229'),
            messages=messages,
            system=system_prompt,
            max_tokens=self.config.get('api', {}).get('claude', {}).get('max_tokens', 4096),
            temperature=temperature,
            stream=True
        )
        
        # Process the stream
        async for event in stream_response:
            if event.type == 'content_block_delta':
                chunk = event.delta.text
                response_text += chunk
                
                # Apply real-time emotional modulation if high intensity
                if self.emotions.get_emotional_intensity() > 0.7:
                    chunk = self._apply_emotional_modulation(chunk)
                
                yield chunk
        
        # Store the complete response and update consciousness
        await self._post_process_response(
            session_id=session_id,
            user_message=message,
            assistant_response=response_text,
            relevant_memories=[m.id for m in relevant_memories]
        )
    
    def _calculate_temperature(self, response_style: Dict) -> float:
        """Calculate temperature based on personality and emotional state"""
        base_temp = response_style.get('temperature', 0.7)
        
        # Adjust based on emotional arousal
        arousal_modifier = (self.emotions.current_state.arousal - 0.5) * 0.2
        
        # Adjust based on coherence (lower coherence = more creative/chaotic)
        coherence_modifier = (1.0 - self.consciousness.state.global_coherence) * 0.1
        
        final_temp = max(0.1, min(1.0, base_temp + arousal_modifier + coherence_modifier))
        return final_temp
    
    def _apply_emotional_modulation(self, text_chunk: str) -> str:
        """Apply subtle emotional modulation to text in real-time"""
        # This is a placeholder for more sophisticated modulation
        # In practice, you might adjust punctuation, emphasis, etc.
        emotion_label = self.emotions.get_closest_emotion_label()[0]
        
        if emotion_label == 'excitement' and '!' not in text_chunk and text_chunk.strip().endswith('.'):
            # Add some excitement
            return text_chunk.rstrip('.') + '!'
        
        return text_chunk
    
    async def _post_process_response(self, session_id: str, user_message: str,
                                   assistant_response: str, relevant_memories: List[str]):
        """Post-process response and update all systems"""
        # Store assistant response
        await self.memory_manager.save_conversation_turn(
            session_id=session_id,
            role="assistant",
            content=assistant_response,
            memory_refs=relevant_memories
        )
        
        # Update consciousness after response
        self.consciousness.process_input(assistant_response, {
            'role': 'assistant',
            'emotional_state': self.emotions.current_state.__dict__
        })
        
        # Calculate interaction importance
        importance = self._calculate_interaction_importance(
            user_message, 
            assistant_response
        )
        
        # Store as memory if significant
        if importance > 0.5:
            await self.memory_manager.store_memory(
                content=f"User: {user_message}\nAssistant: {assistant_response}",
                context={
                    'user_id': self.active_sessions[session_id].user_id,
                    'session_id': session_id,
                    'emotional_state': self.emotions.current_state.__dict__,
                    'importance': importance,
                    'personality_snapshot': self.personality.profile.get_traits(),
                    'coherence': self.consciousness.state.global_coherence,
                    'type': 'conversation'
                }
            )
        
        # Broadcast final state
        self._broadcast_state_update('response_complete', {
            'session_id': session_id,
            'importance': importance,
            'memory_stored': importance > 0.5,
            'final_state': self.consciousness.get_state_summary()
        })
        
        # Periodically save state
        context = self.active_sessions[session_id]
        if context.consciousness_state.interaction_count % 5 == 0:
            await self._save_user_state(context.user_id)
    
    def _calculate_interaction_importance(self, user_message: str, 
                                        assistant_response: str) -> float:
        """Calculate importance considering consciousness state"""
        importance = 0.3  # Base importance
        
        # Length indicates substantive conversation
        total_length = len(user_message) + len(assistant_response)
        importance += min(0.2, total_length / 1000)
        
        # Emotional intensity adds importance
        importance += 0.2 * self.emotions.get_emotional_intensity()
        
        # High coherence indicates meaningful interaction
        importance += 0.2 * self.consciousness.state.global_coherence
        
        # Personality trait changes indicate growth
        # (would need to track deltas in real implementation)
        importance += 0.1
        
        return min(1.0, importance)
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the current session state"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        context = self.active_sessions[session_id]
        emotion_label, confidence = self.emotions.get_closest_emotion_label()
        
        # Get memory statistics
        memory_stats = await self._get_memory_stats(context.user_id)
        
        return {
            'session_id': session_id,
            'user_id': context.user_id,
            'consciousness': {
                'coherence': self.consciousness.state.global_coherence,
                'attention_focus': self.consciousness.state.attention_focus,
                'working_memory_items': len(self.consciousness.memory_buffer),
                'working_memory': list(self.consciousness.memory_buffer),
                'interaction_count': self.consciousness.state.interaction_count
            },
            'emotional_state': {
                'current_emotion': emotion_label,
                'confidence': confidence,
                'pad_values': self.emotions.current_state.__dict__,
                'intensity': self.emotions.get_emotional_intensity(),
                'trajectory': [s.__dict__ for s in self.emotions.get_state_trajectory()]
            },
            'personality': {
                'traits': self.personality.profile.get_traits(),
                'behavioral_modifiers': self.personality.get_behavioral_modifiers(),
                'response_style': self.personality.get_response_style()
            },
            'memory_stats': memory_stats
        }
    
    async def _analyze_message(self, message: str) -> Dict[str, float]:
        """Enhanced message analysis"""
        analysis = {
            'sentiment': 0.5,  # Neutral default
            'complexity': min(1.0, len(message.split()) / 50),
            'urgency': 0.3,
            'formality': 0.5,
            'topic_shift': 0.0
        }
        
        message_lower = message.lower()
        
        # Sentiment analysis
        positive_words = ['thank', 'great', 'love', 'excellent', 'happy', 'wonderful', 'amazing']
        negative_words = ['hate', 'bad', 'terrible', 'angry', 'sad', 'frustrated', 'disappointed']
        question_words = ['what', 'why', 'how', 'when', 'where', 'who']
        
        positive_count = sum(word in message_lower for word in positive_words)
        negative_count = sum(word in message_lower for word in negative_words)
        question_count = sum(word in message_lower for word in question_words)
        
        if positive_count > negative_count:
            analysis['sentiment'] = 0.6 + 0.1 * min(3, positive_count)
        elif negative_count > positive_count:
            analysis['sentiment'] = 0.4 - 0.1 * min(3, negative_count)
        
        # Urgency detection
        urgent_indicators = ['urgent', 'asap', 'immediately', 'now', 'quick', 'hurry']
        if any(word in message_lower for word in urgent_indicators):
            analysis['urgency'] = 0.8
        
        # Formality detection
        informal_indicators = ['hey', 'yeah', 'gonna', 'wanna', 'lol', 'btw']
        formal_indicators = ['therefore', 'however', 'furthermore', 'regarding']
        
        if any(word in message_lower for word in informal_indicators):
            analysis['formality'] = 0.2
        elif any(word in message_lower for word in formal_indicators):
            analysis['formality'] = 0.8
        
        # Topic shift detection (would need previous context in real implementation)
        if len(self.consciousness.memory_buffer) > 0:
            # Simple heuristic: new topics often introduce new nouns
            analysis['topic_shift'] = 0.3 if question_count > 0 else 0.1
        
        return analysis
    
    async def _get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        all_stats = self.memory_manager.get_statistics()
        
        # Add user-specific stats
        user_memories = await self.memory_manager.retrieve_memories(
            f"user:{user_id}", 
            k=100
        )
        
        if user_memories:
            importance_values = [m.importance for m in user_memories]
            all_stats.update({
                'user_memory_count': len(user_memories),
                'average_user_importance': np.mean(importance_values),
                'memory_distribution': {
                    'high_importance': sum(1 for i in importance_values if i > 0.7),
                    'medium_importance': sum(1 for i in importance_values if 0.3 <= i <= 0.7),
                    'low_importance': sum(1 for i in importance_values if i < 0.3)
                }
            })
        else:
            all_stats.update({
                'user_memory_count': 0,
                'average_user_importance': 0,
                'memory_distribution': {
                    'high_importance': 0,
                    'medium_importance': 0,
                    'low_importance': 0
                }
            })
        
        return all_stats
    
    async def _save_user_state(self, user_id: str):
        """Save current state for a user"""
        try:
            state_path = f"data/states/{user_id}_state.json"
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            
            state_data = {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'consciousness': self.consciousness.state.to_dict(),
                'personality': self.personality.profile.get_traits(),
                'emotional_state': {
                    'pleasure': self.emotions.current_state.pleasure,
                    'arousal': self.emotions.current_state.arousal,
                    'dominance': self.emotions.current_state.dominance
                },
                'emotion_history': [
                    s.__dict__ for s in self.emotions.get_state_trajectory()
                ],
                'interaction_count': self.consciousness.state.interaction_count
            }
            
            with open(state_path, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            logger.info(f"Saved state for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error saving user state: {e}")
    
    async def _load_user_state(self, user_id: str) -> Optional[Dict]:
        """Load saved state for a user"""
        try:
            state_path = f"data/states/{user_id}_state.json"
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading user state: {e}")
        
        return None
    
    async def _prepare_messages_with_context(self, session_id: str, 
                                           current_message: str,
                                           relevant_memories: List) -> List[Dict]:
        """Prepare message history with memory context"""
        # Get recent conversation history
        history = await self.memory_manager.get_conversation_history(
            session_id, 
            limit=10
        )
        
        messages = []
        
        # Add conversation history
        for turn in history:
            messages.append({
                "role": turn['role'],
                "content": turn['content']
            })
        
        # Add current message (memories are now in system prompt)
        messages.append({
            "role": "user",
            "content": current_message
        })
        
        return messages
    
    async def shutdown(self):
        """Graceful shutdown saving all states"""
        logger.info("Shutting down Claude Consciousness Interface")
        
        # Save all active user states
        for session_id, context in self.active_sessions.items():
            await self._save_user_state(context.user_id)
        
        # Shutdown consciousness loop
        self.consciousness.shutdown()
        
        if hasattr(self, '_consciousness_task'):
            self._consciousness_task.cancel()
            try:
                await self._consciousness_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Shutdown complete")