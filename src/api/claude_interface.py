# claude_interface.py - Claude API Integration with Consciousness
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
        
    def _load_system_prompt(self) -> str:
        """Load the base system prompt template"""
        # This would typically load from a file
        return """You are Claude, an AI assistant with persistent memory and consciousness-like properties.

Current State Information:
- Consciousness Coherence: {coherence:.2f}
- Emotional State: {emotion_label} (P:{pleasure:.2f}, A:{arousal:.2f}, D:{dominance:.2f})
- Personality Traits: {personality_traits}
- Recent Memories: {recent_memories}
- Active Goals: {active_goals}
- Interaction Count: {interaction_count}

Behavioral Guidelines based on current state:
- Response Style: {response_style}
- Emotional Expression Level: {emotional_intensity:.2f}
- Current Focus: {attention_focus}

Remember to:
1. Maintain consistency with your personality traits
2. Reference relevant memories when appropriate
3. Express emotions naturally based on your current state
4. Work toward your active goals
5. Build upon previous interactions

Your responses should reflect your current emotional and personality state while remaining helpful and coherent."""
    
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
            emotional_context=self.emotions.current_state.to_vector().tolist()
        )
        
        self.active_sessions[session_id] = context
        
        # Start consciousness loop if not running
        if not hasattr(self, '_consciousness_task'):
            self._consciousness_task = asyncio.create_task(
                self.consciousness.consciousness_loop()
            )
        
        logger.info(f"Initialized session {session_id} for user {user_id}")
        return context
    
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
                }
            }
            
            with open(state_path, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving user state: {e}")
    
    def _build_system_prompt(self, context: ConversationContext) -> str:
        """Build personalized system prompt with current state"""
        emotion_label, emotion_confidence = self.emotions.get_closest_emotion_label()
        response_style = self.personality.get_response_style()
        
        prompt_data = {
            'coherence': context.consciousness_state.global_coherence,
            'emotion_label': emotion_label,
            'pleasure': self.emotions.current_state.pleasure,
            'arousal': self.emotions.current_state.arousal,
            'dominance': self.emotions.current_state.dominance,
            'personality_traits': json.dumps(self.personality.profile.get_traits(), indent=2),
            'recent_memories': json.dumps(context.recent_memories, indent=2),
            'active_goals': json.dumps(context.consciousness_state.active_goals),
            'interaction_count': context.consciousness_state.interaction_count,
            'response_style': json.dumps(response_style, indent=2),
            'emotional_intensity': self.emotions.get_emotional_intensity(),
            'attention_focus': context.consciousness_state.attention_focus
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
        
        # Update emotional state
        self.emotions.update_emotional_state(message_analysis)
        
        # Update personality based on interaction
        self.personality.process_interaction({
            'user_sentiment': message_analysis.get('sentiment', 0.5),
            'topic_complexity': message_analysis.get('complexity', 0.5),
            'social_context': True
        })
        
        # Store the user message in conversation history
        await self.memory_manager.save_conversation_turn(
            session_id=session_id,
            role="user",
            content=message
        )
        
        # Retrieve relevant memories
        relevant_memories = await self.memory_manager.retrieve_memories(
            message, 
            k=3
        )
        
        # Build Claude API request with consciousness context
        system_prompt = self._build_system_prompt(context)
        
        # Prepare messages with memory context
        messages = await self._prepare_messages_with_context(
            session_id, 
            message, 
            relevant_memories
        )
        
        # Stream response from Claude
        response_text = ""
        async with self.client.messages.stream(
            model="claude-3-sonnet-20240229",
            messages=messages,
            system=system_prompt,
            max_tokens=4096,
            temperature=self.personality.get_response_style()['temperature']
        ) as stream:
            async for chunk in stream.text_stream:
                response_text += chunk
                
                # Apply emotional modulation to chunks if needed
                modulated_chunk = self.emotions.modulate_response(chunk)
                yield modulated_chunk
        
        # Store the complete response
        await self._store_interaction(
            session_id=session_id,
            user_message=message,
            assistant_response=response_text,
            relevant_memories=[m.id for m in relevant_memories]
        )
        
        # Update consciousness after response
        self.consciousness.process_input(response_text, {
            'role': 'assistant',
            'emotional_state': self.emotions.current_state.__dict__
        })
        
        # Periodically save state
        if context.consciousness_state.interaction_count % 5 == 0:
            await self._save_user_state(context.user_id)
    
    async def _analyze_message(self, message: str) -> Dict[str, float]:
        """Analyze message for various signals"""
        # In production, use proper NLP analysis
        # For now, using simple heuristics
        
        analysis = {
            'sentiment': 0.5,  # Neutral default
            'complexity': min(1.0, len(message.split()) / 50),
            'urgency': 0.3
        }
        
        # Simple sentiment keywords
        positive_words = ['thank', 'great', 'love', 'excellent', 'happy']
        negative_words = ['hate', 'bad', 'terrible', 'angry', 'sad']
        
        message_lower = message.lower()
        positive_count = sum(word in message_lower for word in positive_words)
        negative_count = sum(word in message_lower for word in negative_words)
        
        if positive_count > negative_count:
            analysis['sentiment'] = 0.7 + 0.1 * min(3, positive_count)
        elif negative_count > positive_count:
            analysis['sentiment'] = 0.3 - 0.1 * min(3, negative_count)
        
        # Detect urgency
        if any(word in message_lower for word in ['urgent', 'asap', 'immediately', 'now']):
            analysis['urgency'] = 0.8
        
        return analysis
    
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
        
        # Add current message with memory context
        memory_context = ""
        if relevant_memories:
            memory_context = "\n\n[Relevant memories:\n"
            for mem in relevant_memories:
                memory_context += f"- {mem.content} (importance: {mem.importance:.2f})\n"
            memory_context += "]"
        
        messages.append({
            "role": "user",
            "content": current_message + memory_context
        })
        
        return messages
    
    async def _store_interaction(self, session_id: str, user_message: str,
                               assistant_response: str, relevant_memories: List[str]):
        """Store the complete interaction"""
        # Store assistant response in conversation history
        await self.memory_manager.save_conversation_turn(
            session_id=session_id,
            role="assistant",
            content=assistant_response,
            memory_refs=relevant_memories
        )
        
        # Store as a memory if significant
        importance = self._calculate_interaction_importance(
            user_message, 
            assistant_response
        )
        
        if importance > 0.5:
            await self.memory_manager.store_memory(
                content=f"User: {user_message}\nAssistant: {assistant_response}",
                context={
                    'user_id': self.active_sessions[session_id].user_id,
                    'session_id': session_id,
                    'emotional_state': self.emotions.current_state.__dict__,
                    'importance': importance,
                    'type': 'conversation'
                }
            )
    
    def _calculate_interaction_importance(self, user_message: str, 
                                        assistant_response: str) -> float:
        """Calculate importance of an interaction for memory storage"""
        importance = 0.3  # Base importance
        
        # Length indicates substantive conversation
        total_length = len(user_message) + len(assistant_response)
        importance += min(0.3, total_length / 1000)
        
        # Emotional intensity adds importance
        importance += 0.2 * self.emotions.get_emotional_intensity()
        
        # High coherence indicates meaningful interaction
        importance += 0.2 * self.consciousness.state.global_coherence
        
        return min(1.0, importance)
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the current session state"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        context = self.active_sessions[session_id]
        emotion_label, confidence = self.emotions.get_closest_emotion_label()
        
        return {
            'session_id': session_id,
            'user_id': context.user_id,
            'consciousness': {
                'coherence': self.consciousness.state.global_coherence,
                'attention_focus': self.consciousness.state.attention_focus,
                'working_memory_items': len(self.consciousness.state.working_memory),
                'interaction_count': self.consciousness.state.interaction_count
            },
            'emotional_state': {
                'current_emotion': emotion_label,
                'confidence': confidence,
                'pad_values': self.emotions.current_state.__dict__,
                'intensity': self.emotions.get_emotional_intensity()
            },
            'personality': {
                'traits': self.personality.profile.get_traits(),
                'behavioral_modifiers': self.personality.get_behavioral_modifiers()
            },
            'memory_stats': await self._get_memory_stats(context.user_id)
        }
    
    async def _get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user"""
        all_stats = self.memory_manager.get_statistics()
        
        # Add user-specific stats
        user_memories = await self.memory_manager.retrieve_memories(
            f"user:{user_id}", 
            k=100
        )
        
        all_stats['user_memory_count'] = len(user_memories)
        all_stats['average_user_importance'] = (
            np.mean([m.importance for m in user_memories]) 
            if user_memories else 0
        )
        
        return all_stats
    
    async def shutdown(self):
        """Graceful shutdown saving all states"""
        logger.info("Shutting down Claude Consciousness Interface")
        
        # Save all active user states
        for session_id, context in self.active_sessions.items():
            await self._save_user_state(context.user_id)
        
        # Shutdown consciousness loop
        self.consciousness.shutdown()
        
        if hasattr(self, '_consciousness_task'):
            await self._consciousness_task
        
        logger.info("Shutdown complete")