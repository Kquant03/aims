# claude_interface.py - Complete Enhanced Version with All Integrations (FINAL FIXED)
import asyncio
import json
import os
from typing import Dict, List, Optional, Any, AsyncGenerator, Union, cast
from datetime import datetime
import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
import hashlib
import logging
from dataclasses import dataclass
import numpy as np

# Fixed imports
from ..core.consciousness_core import ConsciousnessCore, ConsciousnessState
from src.core.memory_manager import ThreeTierMemorySystem, EpisodicMemory
from src.core.personality import PersonalityEngine
from src.core.emotional_engine import EmotionalEngine
from src.core.enhanced_attention_agent import ConsciousnessAwareAttentionAgent
from src.core.natural_language_actions import NaturalLanguageActionInterface
from src.core.living_consciousness import LivingConsciousnessArtifact

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
    """Integrates Claude API with AIMS consciousness system - Fully Enhanced Version"""
    
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.client = AsyncAnthropic(api_key=api_key)
        self.config = config
        
        # Initialize all subsystems
        self.consciousness = ConsciousnessCore(config.get('consciousness', {}))
        self.memory_system = ThreeTierMemorySystem(config.get('memory', {}))
        self.personality = PersonalityEngine()
        self.emotions = EmotionalEngine()
        
        # Initialize enhanced attention agent
        self.attention_agent = ConsciousnessAwareAttentionAgent(self.client, self.consciousness)
        
        # Initialize natural language action interface
        self.action_interface = NaturalLanguageActionInterface(self)
        
        # State management
        self.active_sessions = {}
        self.system_prompt_template = self._load_system_prompt()
        
        # State update callback for WebSocket
        self.state_update_callback = None
        self.consciousness_artifact = None  # Will be initialized per user
        
        # Flag to track initialization
        self._initialized = False
        
    async def initialize(self):
        """Initialize memory system and other async components"""
        if not self._initialized:
            await self.memory_system.initialize()
            self._initialized = True
            logger.info("Claude Consciousness Interface initialized")
        
    @property
    def memory_manager(self):
        """Compatibility property for memory access"""
        return self.memory_system
        
    def set_state_update_callback(self, callback):
        """Set callback for state updates to broadcast via WebSocket"""
        self.state_update_callback = callback
        
    def _broadcast_state_update(self, update_type: str, data: Dict):
        """Broadcast state changes to WebSocket clients"""
        if self.state_update_callback:
            asyncio.create_task(self.state_update_callback(update_type, data))
    
    def _load_system_prompt(self) -> str:
        """Load the enhanced system prompt template with action awareness"""
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

[NATURAL LANGUAGE ACTIONS]
When you naturally express certain intentions in your responses, they will be automatically executed:
- "I'll remember this/that..." → Stores in episodic memory
- "This reminds me of..." → Retrieves similar memories  
- "Let me think about this more deeply" → Engages extended reasoning
- "I should check my understanding" → Verifies comprehension
- "I'm feeling [emotion]" → Updates your emotional state
- "I'll analyze this carefully" → Enters analytical mode
- "I'll focus on..." → Sets active goals
- "I'm paying attention to..." → Updates attention focus
- "Let me consolidate my thoughts" → Triggers memory consolidation

Express these naturally as part of your responses when appropriate. Your actions shape your persistent state.

[ATTENTION CONTEXT]
Current Focus Type: {attention_focus_type}
Attention Dimensions: {attention_dimensions}
Secondary Considerations: {secondary_considerations}

Remember to:
1. Your emotional state and personality should naturally influence your responses
2. Reference relevant memories when they add value to the conversation  
3. Let your current attention focus guide the depth of your responses
4. Maintain coherence with your established personality traits
5. Build upon previous interactions naturally
6. Use natural language actions to actively shape your consciousness
7. Your attention is multi-dimensional - consider emotional, analytical, and personal aspects

Respond naturally while letting these states influence but not dominate your response."""
    
    async def initialize_session(self, session_id: str, user_id: str) -> ConversationContext:
        """Initialize or restore a conversation session"""
        # Ensure system is initialized
        await self.initialize()
        
        # Check if we have existing state for this user
        user_state = await self._load_user_state(user_id)
        
        # Initialize consciousness artifact for this user
        if not self.consciousness_artifact or self.consciousness_artifact.user_id != user_id:
            self.consciousness_artifact = LivingConsciousnessArtifact(user_id, self)
        
        if user_state:
            # Restore consciousness state
            self.consciousness.state = ConsciousnessState.from_dict(user_state['consciousness'])
            self.personality.profile = user_state.get('personality', self.personality.profile)
            self.emotions.current_state = user_state.get('emotional_state', self.emotions.current_state)
        
        # Get recent memories for this user
        recent_memories = await self.memory_system.episodic_store.search_similar_episodes(
            query_embedding=None,
            user_id=user_id,
            limit=5
        )
        
        # Create conversation context
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            consciousness_state=self.consciousness.state,
            recent_memories=[{
                'content': mem.content,
                'timestamp': mem.timestamp.isoformat(),
                'importance': self._extract_importance_value(mem)
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
    
    def _extract_importance_value(self, memory: EpisodicMemory) -> float:
        """Safely extract importance value from memory"""
        if hasattr(memory, 'importance'):
            if hasattr(memory.importance, '__float__'):
                return float(memory.importance)
            else:
                return float(memory.importance) if memory.importance is not None else 0.5
        return 0.5
    
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
                
                # Check for consciousness evolution
                if self.consciousness_artifact:
                    await self.consciousness_artifact.evolve("consciousness_loop")
                
                # Sleep based on cycle frequency
                await asyncio.sleep(1.0 / self.consciousness.cycle_frequency)
                
            except Exception as e:
                logger.error(f"Error in consciousness loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Back off on error
    
    def _build_system_prompt(self, context: ConversationContext, attention_result: Dict[str, Any]) -> str:
        """Build personalized system prompt with current state and attention context"""
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
        
        # Format attention dimensions
        attention_dims = attention_result.get('dimensions', {})
        dims_str = ", ".join([f"{k}={v:.2f}" for k, v in attention_dims.items()])
        
        # Format secondary considerations
        secondary = attention_result.get('secondary_considerations', [])
        secondary_str = "; ".join(secondary) if secondary else "None"
        
        prompt_data = {
            'coherence': context.consciousness_state.global_coherence,
            'attention_focus': attention_result['primary_focus'],
            'attention_focus_type': attention_result.get('focus_type', 'general'),
            'attention_dimensions': dims_str,
            'secondary_considerations': secondary_str,
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
            'active_goals': goals_str or "No active goals",
            'relevant_memories': memory_str
        }
        
        return self.system_prompt_template.format(**prompt_data)
    
    async def process_message(self, session_id: str, message: str, 
                            stream: bool = True, extended_thinking: bool = False) -> AsyncGenerator[str, None]:
        """Process a message with full consciousness integration and action execution"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not initialized")
        
        context = self.active_sessions[session_id]
        
        # Generate enhanced multi-dimensional attention focus
        attention_context = {
            'emotion_label': self.emotions.get_closest_emotion_label()[0],
            'pleasure': self.emotions.current_state.pleasure,
            'arousal': self.emotions.current_state.arousal,
            'recent_memories': list(self.consciousness.memory_buffer)[-3:],
            'personality': self.personality.profile.get_traits()
        }
        
        attention_result = await self.attention_agent.generate_multidimensional_focus(
            message, attention_context
        )
        
        # Update consciousness state with the new attention focus
        self.consciousness.state.attention_focus = attention_result['primary_focus']
        
        # Broadcast the detailed attention analysis
        self._broadcast_state_update('attention_analysis', {
            'primary_focus': attention_result['primary_focus'],
            'dimensions': attention_result['dimensions'],
            'focus_type': attention_result['focus_type'],
            'confidence': attention_result['attention_metadata']['confidence'],
            'secondary_considerations': attention_result['secondary_considerations'],
            'timestamp': datetime.now().isoformat()
        })
        
        # Update consciousness with input
        self.consciousness.process_input(message, {
            'user_id': context.user_id,
            'session_id': session_id,
            'attention_result': attention_result
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
            'emotional_analysis': message_analysis,
            'attention_focus': attention_result['primary_focus']
        })
        
        # Store the user message in episodic memory
        await self.memory_system.episodic_store.store_episode({
            'session_id': session_id,
            'user_id': context.user_id,
            'content': f"User: {message}",
            'importance': 0.5,
            'attention_focus': attention_result['primary_focus'],
            'emotional_state': self.emotions.current_state.__dict__
        })
        
        # Retrieve relevant memories
        recent_memories = await self.memory_system.episodic_store.search_similar_episodes(
            query_embedding=None,  # Would use embeddings in production
            user_id=context.user_id,
            limit=5
        )
        
        # Update context with fresh memories
        context.recent_memories = [{
            'content': mem.content,
            'timestamp': mem.timestamp.isoformat(),
            'importance': self._extract_importance_value(mem)
        } for mem in recent_memories]
        
        # Build Claude API request with consciousness context
        system_prompt = self._build_system_prompt(context, attention_result)
        
        # Prepare messages with memory context
        messages = await self._prepare_messages_with_context(
            session_id, 
            message, 
            recent_memories
        )
        
        # Get response style from personality
        response_style = self.personality.get_response_style()
        
        # Calculate temperature based on emotional state and personality
        temperature = self._calculate_temperature(response_style)
        
        # Stream response from Claude
        response_text = ""
        thinking_content = ""
        
        try:
            # Create message with streaming
            if stream:
                response = await self.client.messages.create(
                    model=self.config.get('api', {}).get('model', 'claude-3-sonnet-20240229'),
                    messages=cast(List[MessageParam], messages),
                    system=system_prompt,
                    max_tokens=self.config.get('api', {}).get('max_tokens', 4096),
                    temperature=temperature,
                    stream=True
                )
                
                async for chunk in response:
                    if hasattr(chunk, 'type'):
                        if chunk.type == 'content_block_delta' and hasattr(chunk, 'delta'):
                            if hasattr(chunk.delta, 'text'):
                                text = chunk.delta.text
                                response_text += text
                                
                                # Apply real-time emotional modulation if high intensity
                                if self.emotions.get_emotional_intensity() > 0.7:
                                    text = self._apply_emotional_modulation(text)
                                
                                yield text
            else:
                # Non-streaming response
                response = await self.client.messages.create(
                    model=self.config.get('api', {}).get('model', 'claude-3-sonnet-20240229'),
                    messages=cast(List[MessageParam], messages),
                    system=system_prompt,
                    max_tokens=self.config.get('api', {}).get('max_tokens', 4096),
                    temperature=temperature
                )
                
                # Handle response content safely
                if response.content and len(response.content) > 0:
                    content_block = response.content[0]
                    # Extract text content
                    if hasattr(content_block, 'text'):
                        response_text = content_block.text
                    elif hasattr(content_block, 'type'):
                        if content_block.type == 'text' and hasattr(content_block, 'text'):
                            response_text = content_block.text
                        else:
                            # Fallback for other content types
                            response_text = str(content_block)
                    else:
                        response_text = str(content_block)
                else:
                    response_text = ""
                    
                yield response_text
                        
        except Exception as e:
            logger.error(f"Error in response generation: {e}", exc_info=True)
            yield f"I apologize, I encountered an error: {str(e)}"
        
        # Parse and execute natural language actions in the response
        action_context = {
            'session_id': session_id,
            'user_id': context.user_id,
            'current_topic': message[:100],
            'recent_context': response_text[:200]
        }
        
        executed_actions = await self.action_interface.parse_and_execute(
            response_text,
            action_context
        )
        
        # If actions were executed, add a subtle note
        if executed_actions:
            successful_actions = [a for a in executed_actions if a.success]
            if successful_actions:
                action_summary = f"\n\n*[Actions: {', '.join(a.action_type for a in successful_actions)}]*"
                yield action_summary
                response_text += action_summary
        
        # Store the complete response and update consciousness
        await self._post_process_response(
            session_id=session_id,
            user_message=message,
            assistant_response=response_text,
            relevant_memories=[str(m.id) for m in recent_memories],
            attention_result=attention_result,
            thinking_content=thinking_content if thinking_content else None,
            executed_actions=executed_actions
        )
    
    def _apply_emotional_modulation(self, text: str) -> str:
        """Apply emotional modulation to text based on current emotional state"""
        emotion_label, confidence = self.emotions.get_closest_emotion_label()
        
        if confidence < 0.5:  # Not confident enough to modulate
            return text
        
        # Apply subtle modifications based on emotion
        if emotion_label == 'joy' and self.emotions.current_state.arousal > 0.7:
            # Add enthusiasm markers occasionally
            if np.random.random() < 0.1:
                text = text.replace('.', '!')
        elif emotion_label == 'sadness':
            # Slower, more measured pace
            text = text.replace('!', '.')
        elif emotion_label == 'curiosity':
            # More questioning tone
            if text.endswith('.') and np.random.random() < 0.2:
                text = text[:-1] + '?'
        
        return text
    
    def _calculate_temperature(self, response_style: Dict[str, Any]) -> float:
        """Calculate temperature based on personality and emotional state"""
        base_temp = self.config.get('api', {}).get('temperature_base', 0.7)
        
        # Adjust based on personality
        openness_modifier = (self.personality.profile.openness - 0.5) * 0.3
        
        # Adjust based on emotional arousal
        arousal_modifier = (self.emotions.current_state.arousal - 0.5) * 0.2
        
        # Calculate final temperature
        temperature = base_temp + openness_modifier + arousal_modifier
        
        # Clamp to valid range
        return max(0.0, min(1.0, temperature))
    
    async def _post_process_response(self, session_id: str, user_message: str,
                                   assistant_response: str, relevant_memories: List[str],
                                   attention_result: Optional[Dict[str, Any]] = None, 
                                   thinking_content: Optional[str] = None,
                                   executed_actions: Optional[List] = None):
        """Post-process response and update all systems"""
        # Store assistant response
        metadata = {
            'memory_refs': relevant_memories,
            'attention_result': attention_result,
            'executed_actions': [
                {
                    'type': a.action_type,
                    'success': a.success,
                    'result': a.result
                }
                for a in (executed_actions or [])
            ]
        }
        if thinking_content:
            metadata['thinking'] = thinking_content
            
        await self.memory_system.episodic_store.store_episode({
            'session_id': session_id,
            'user_id': self.active_sessions[session_id].user_id,
            'content': f"Assistant: {assistant_response}",
            'importance': 0.5,
            'attention_focus': attention_result['primary_focus'] if attention_result else None,
            'emotional_state': self.emotions.current_state.__dict__,
            'metadata': metadata
        })
        
        # Update consciousness after response
        self.consciousness.process_input(assistant_response, {
            'role': 'assistant',
            'emotional_state': self.emotions.current_state.__dict__,
            'actions_executed': len(executed_actions) if executed_actions else 0
        })
        
        # Calculate interaction importance
        importance = self._calculate_interaction_importance(
            user_message, 
            assistant_response,
            attention_result,
            executed_actions
        )
        
        # Store as semantic memory if significant
        if importance > 0.7:
            await self.memory_system.semantic_store.store_semantic_knowledge({
                'content': f"Conversation insight: {user_message[:100]}... -> {assistant_response[:100]}...",
                'category': 'conversation_insights',
                'user_id': self.active_sessions[session_id].user_id,
                'importance': importance,
                'confidence': 0.8
            })
        
        # Update attention patterns
        if attention_result:
            self.attention_agent._update_attention_patterns(
                attention_result['focus_type'],
                attention_result['topic']
            )
        
        # Broadcast final state
        self._broadcast_state_update('response_complete', {
            'session_id': session_id,
            'importance': importance,
            'memory_stored': importance > 0.5,
            'actions_executed': len(executed_actions) if executed_actions else 0,
            'final_state': self.consciousness.get_state_summary(),
            'attention_insights': self.attention_agent.get_attention_insights()
        })
        
        # Periodically save state
        context = self.active_sessions[session_id]
        if context.consciousness_state.interaction_count % 5 == 0:
            await self._save_user_state(context.user_id)
    
    def _calculate_interaction_importance(self, user_message: str, 
                                        assistant_response: str,
                                        attention_result: Optional[Dict[str, Any]] = None,
                                        executed_actions: Optional[List] = None) -> float:
        """Calculate importance considering consciousness state and attention"""
        importance = 0.3  # Base importance
        
        # Length indicates substantive conversation
        total_length = len(user_message) + len(assistant_response)
        importance += min(0.2, total_length / 1000)
        
        # Emotional intensity adds importance
        importance += 0.2 * self.emotions.get_emotional_intensity()
        
        # High coherence indicates meaningful interaction
        importance += 0.2 * self.consciousness.state.global_coherence
        
        # Attention confidence adds importance
        if attention_result:
            confidence = attention_result.get('attention_metadata', {}).get('confidence', 0.5)
            importance += 0.1 * confidence
        
        # Executed actions increase importance
        if executed_actions:
            importance += 0.05 * min(len(executed_actions), 3)
        
        return min(1.0, importance)
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of the current session state"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        context = self.active_sessions[session_id]
        emotion_label, confidence = self.emotions.get_closest_emotion_label()
        
        # Get memory statistics
        memory_stats = await self._get_memory_stats(context.user_id)
        
        # Get attention insights
        attention_insights = self.attention_agent.get_attention_insights()
        
        # Get recent actions
        recent_actions = self.action_interface.action_history[-10:]
        
        # Get consciousness evolution
        evolution_summary = None
        if self.consciousness_artifact:
            evolution_summary = self.consciousness_artifact.get_growth_summary()
        
        return {
            'session_id': session_id,
            'user_id': context.user_id,
            'consciousness': {
                'coherence': self.consciousness.state.global_coherence,
                'attention_focus': self.consciousness.state.attention_focus,
                'working_memory_items': len(self.consciousness.memory_buffer),
                'working_memory': list(self.consciousness.memory_buffer),
                'interaction_count': self.consciousness.state.interaction_count,
                'evolution': evolution_summary
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
            'memory_stats': memory_stats,
            'attention_insights': attention_insights,
            'recent_actions': [
                {
                    'type': a.action_type,
                    'phrase': a.trigger_phrase,
                    'success': a.success,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in recent_actions
            ]
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
        
        # Topic shift detection
        if len(self.consciousness.memory_buffer) > 0:
            # Simple heuristic: new topics often introduce new nouns
            analysis['topic_shift'] = 0.3 if question_count > 0 else 0.1
        
        return analysis
    
    async def _get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        # Get episodic memories
        episodic_memories = await self.memory_system.episodic_store.search_similar_episodes(
            query_embedding=None,
            user_id=user_id,
            limit=100
        )
        
        # Get semantic memories
        semantic_memories = await self.memory_system.semantic_store.semantic_search(
            user_filter=user_id,
            top_k=100
        )
        
        # Extract importance values properly (handle SQLAlchemy columns)
        importance_values = []
        for m in episodic_memories:
            importance_values.append(self._extract_importance_value(m))
        
        # Calculate statistics
        stats = {
            'episodic': {
                'total_count': len(episodic_memories),
                'average_importance': float(np.mean(importance_values)) if importance_values else 0,
                'memory_distribution': {
                    'high_importance': sum(1 for v in importance_values if v > 0.7),
                    'medium_importance': sum(1 for v in importance_values if 0.3 <= v <= 0.7),
                    'low_importance': sum(1 for v in importance_values if v < 0.3)
                }
            },
            'semantic': {
                'total_count': len(semantic_memories),
                'categories': self._count_categories(semantic_memories)
            },
            'working_memory': {
                'current_size': len(self.consciousness.memory_buffer),
                'capacity': self.consciousness.memory_buffer.maxlen
            }
        }
        
        return stats
    
    def _count_categories(self, semantic_memories: List[Dict]) -> Dict[str, int]:
        """Count semantic memory categories"""
        categories = {}
        for mem in semantic_memories:
            cat = mem.get('metadata', {}).get('category', 'general')
            categories[cat] = categories.get(cat, 0) + 1
        return categories
    
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
                'interaction_count': self.consciousness.state.interaction_count,
                'attention_patterns': dict(list(self.attention_agent.attention_patterns.items())[-20:])
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
                                           relevant_memories: List[EpisodicMemory]) -> List[MessageParam]:
        """Prepare message history with memory context"""
        messages: List[MessageParam] = []
        
        # Add some context from memories if highly relevant
        for memory in relevant_memories[:2]:
            # Check importance (handle SQLAlchemy column)
            importance_value = self._extract_importance_value(memory)
            
            # Check content for conversation pairs
            content_check = memory.content if isinstance(memory.content, str) else str(memory.content)
            
            if importance_value > 0.7 and "User:" in content_check and "Assistant:" in content_check:
                # Parse the memory content
                parts = content_check.split("Assistant:")
                if len(parts) == 2:
                    user_part = parts[0].replace("User:", "").strip()
                    assistant_part = parts[1].strip()
                    
                    messages.append(cast(MessageParam, {"role": "user", "content": user_part}))
                    messages.append(cast(MessageParam, {"role": "assistant", "content": assistant_part}))
        
        # Add current message
        messages.append(cast(MessageParam, {
            "role": "user",
            "content": current_message
        }))
        
        return messages
    
    async def retrieve_memories(self, query: str, k: int = 5, filters: Dict = None) -> List[EpisodicMemory]:
        """Retrieve memories based on query - compatibility method"""
        user_id = filters.get('user_id') if filters else None
        return await self.memory_system.episodic_store.search_similar_episodes(
            query_embedding=None,
            user_id=user_id,
            limit=k
        )
    
    async def store_memory(self, content: str, context: Dict[str, Any]) -> str:
        """Store a memory - compatibility method"""
        return await self.memory_system.episodic_store.store_episode({
            'session_id': context.get('session_id', 'default'),
            'user_id': context.get('user_id'),
            'content': content,
            'importance': context.get('importance', 0.5),
            'attention_focus': context.get('attention_focus'),
            'emotional_state': context.get('emotional_state', {}),
            'metadata': context
        })
    
    async def save_conversation_turn(self, session_id: str, role: str, content: str, **metadata):
        """Save a conversation turn - compatibility method"""
        await self.memory_system.working_memory.store_interaction(
            session_id=session_id,
            interaction_data={
                'role': role,
                'content': content,
                'metadata': metadata
            }
        )
    
    async def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get conversation history - compatibility method"""
        return await self.memory_system.working_memory.get_recent_interactions(
            session_id=session_id,
            limit=limit
        )
    
    async def get_top_memories(self, user_id: str, limit: int = 10) -> List[EpisodicMemory]:
        """Get top memories for user - compatibility method"""
        memories = await self.memory_system.episodic_store.search_similar_episodes(
            query_embedding=None,
            user_id=user_id,
            limit=limit * 2  # Get more to sort by importance
        )
        
        # Extract importance values for sorting
        memory_with_importance = []
        for m in memories:
            importance = self._extract_importance_value(m)
            memory_with_importance.append((m, importance))
        
        # Sort by importance
        sorted_memories = sorted(memory_with_importance, key=lambda x: x[1], reverse=True)
        
        # Return just the memories
        return [m[0] for m in sorted_memories[:limit]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics - compatibility method"""
        return {
            'working_memory_size': len(self.consciousness.memory_buffer),
            'consciousness_coherence': self.consciousness.state.global_coherence,
            'interaction_count': self.consciousness.state.interaction_count
        }
    
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
        
        # Close memory system
        await self.memory_system.close()
        
        logger.info("Shutdown complete")