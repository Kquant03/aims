# src/core/natural_language_actions.py
import re
import asyncio
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ExecutedAction:
    """Represents an action that was executed from natural language"""
    action_type: str
    trigger_phrase: str
    parameters: Dict[str, Any]
    result: Any
    timestamp: datetime
    success: bool
    
class NaturalLanguageActionInterface:
    """Allows Claude to execute actions through natural language expressions"""
    
    def __init__(self, claude_interface):
        self.claude = claude_interface
        self.consciousness = claude_interface.consciousness
        self.memory_manager = claude_interface.memory_manager
        self.emotions = claude_interface.emotions
        
        # Define action patterns with their handlers
        self.action_patterns = [
            # Memory actions
            (r"(?:I'll|I will|Let me) remember (?:this|that|about )(.+?)(?:\.|$)", 
             self.action_remember, "remember"),
            
            (r"(?:I'll|I will|Let me) store this (?:as|in) (?:my |)([\w\s]+) memory", 
             self.action_store_categorized, "store_memory"),
            
            # Retrieval actions
            (r"(?:This|That) reminds me of (.+?)(?:\.|$)", 
             self.action_retrieve_similar, "retrieve_similar"),
            
            (r"(?:I|Let me) (?:should |)(?:recall|remember) (?:what I know about |)(.+?)(?:\.|$)", 
             self.action_recall, "recall"),
            
            # Thinking actions
            (r"(?:I'll|Let me) (?:think|reflect) (?:about this |on this |)(?:more |)(?:deeply|carefully)", 
             self.action_deep_think, "deep_think"),
            
            (r"(?:I need to|I should|Let me) (?:verify|check) (?:my |this |)understanding", 
             self.action_verify_understanding, "verify"),
            
            # Emotional actions
            (r"(?:I'm|I am) (?:feeling |)(\w+)(?: about this|)", 
             self.action_update_emotion, "emotion_update"),
            
            # Analysis actions
            (r"(?:I'll|Let me) analyze (?:this|that) (?:more |)(?:carefully|thoroughly|in detail)", 
             self.action_analyze, "analyze"),
            
            # Goal actions
            (r"(?:I'll|I will|I should) (?:focus on|prioritize) (.+?)(?:\.|$)", 
             self.action_set_goal, "set_goal"),
            
            # Attention actions
            (r"(?:I'm|I am) (?:now |)(?:focusing on|paying attention to) (.+?)(?:\.|$)", 
             self.action_focus_attention, "focus_attention"),
            
            # Meta-cognitive actions
            (r"(?:I|Let me) (?:should |)(?:consolidate|organize) my (?:thoughts|memories|understanding)", 
             self.action_consolidate, "consolidate"),
        ]
        
        # Track executed actions for context
        self.action_history = []
        
    async def parse_and_execute(self, text: str, context: Dict[str, Any]) -> List[ExecutedAction]:
        """Parse text for action intentions and execute them"""
        executed_actions = []
        
        # Check each pattern
        for pattern, handler, action_type in self.action_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                try:
                    # Execute the action
                    result = await handler(match, text, context)
                    
                    # Record the execution
                    action = ExecutedAction(
                        action_type=action_type,
                        trigger_phrase=match.group(0),
                        parameters={'match_groups': match.groups()},
                        result=result,
                        timestamp=datetime.now(),
                        success=True
                    )
                    
                    executed_actions.append(action)
                    self.action_history.append(action)
                    
                    # Log the action
                    logger.info(f"Executed action '{action_type}' from phrase: {match.group(0)}")
                    
                except Exception as e:
                    logger.error(f"Error executing action {action_type}: {e}")
                    
                    action = ExecutedAction(
                        action_type=action_type,
                        trigger_phrase=match.group(0),
                        parameters={'match_groups': match.groups()},
                        result={'error': str(e)},
                        timestamp=datetime.now(),
                        success=False
                    )
                    executed_actions.append(action)
        
        # Broadcast executed actions
        if executed_actions and hasattr(self.claude, '_broadcast_state_update'):
            self.claude._broadcast_state_update('actions_executed', {
                'actions': [
                    {
                        'type': a.action_type,
                        'phrase': a.trigger_phrase,
                        'success': a.success,
                        'timestamp': a.timestamp.isoformat()
                    }
                    for a in executed_actions
                ]
            })
        
        return executed_actions
    
    async def action_remember(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store something in episodic memory"""
        # Extract what to remember
        content = match.group(1) if match.lastindex else full_text
        
        # Determine importance based on emotional state and emphasis
        base_importance = 0.6
        if self.emotions.get_emotional_intensity() > 0.7:
            base_importance += 0.2
        
        # Store in memory with context
        memory_id = await self.memory_manager.store_memory(
            content=f"Important insight: {content}",
            context={
                'type': 'deliberate_memory',
                'session_id': context.get('session_id'),
                'user_id': context.get('user_id'),
                'emotional_state': self.emotions.current_state.__dict__,
                'attention_focus': self.consciousness.state.attention_focus,
                'importance': base_importance,
                'trigger': 'natural_language_action'
            }
        )
        
        return {
            'memory_id': memory_id,
            'content': content,
            'importance': base_importance
        }
    
    async def action_store_categorized(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store memory with specific category"""
        category = match.group(1).strip()
        
        # Find the content (usually the previous sentence or context)
        sentences = full_text.split('.')
        current_sentence_idx = None
        for i, sentence in enumerate(sentences):
            if match.group(0) in sentence:
                current_sentence_idx = i
                break
        
        # Get previous sentence as content
        if current_sentence_idx and current_sentence_idx > 0:
            content = sentences[current_sentence_idx - 1].strip()
        else:
            content = context.get('recent_context', 'Current discussion point')
        
        # Store as semantic memory
        memory_id = await self.memory_manager.semantic_store.store_semantic_knowledge({
            'content': content,
            'category': category,
            'source': 'natural_language_action',
            'confidence': 0.9,
            'importance': 0.7,
            'user_id': context.get('user_id')
        })
        
        return {
            'memory_id': memory_id,
            'category': category,
            'content': content
        }
    
    async def action_retrieve_similar(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve memories similar to something"""
        query = match.group(1)
        
        # Search for similar memories
        similar_memories = await self.memory_manager.retrieve_memories(
            query=query,
            k=5,
            filters={'user_id': context.get('user_id')}
        )
        
        # Update working memory with retrieved content
        for memory in similar_memories[:3]:
            self.consciousness.memory_buffer.append(
                f"Retrieved: {memory.content[:100]}..."
            )
        
        return {
            'query': query,
            'found_count': len(similar_memories),
            'top_matches': [
                {
                    'content': m.content,
                    'importance': m.importance,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in similar_memories[:3]
            ]
        }
    
    async def action_deep_think(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Engage deeper thinking mode"""
        # Update consciousness state for deep thinking
        self.consciousness.state.attention_focus = "deep analytical thinking"
        
        # Temporarily increase working memory capacity
        original_maxlen = self.consciousness.memory_buffer.maxlen
        self.consciousness.memory_buffer = type(self.consciousness.memory_buffer)(
            self.consciousness.memory_buffer, 
            maxlen=15  # Temporarily expand
        )
        
        # Schedule return to normal after some time
        async def restore_memory():
            await asyncio.sleep(60)  # 1 minute of expanded memory
            self.consciousness.memory_buffer = type(self.consciousness.memory_buffer)(
                self.consciousness.memory_buffer,
                maxlen=original_maxlen
            )
        
        asyncio.create_task(restore_memory())
        
        return {
            'mode': 'deep_thinking',
            'working_memory_expanded': True,
            'duration': 60
        }
    
    async def action_verify_understanding(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify understanding by checking memory consistency"""
        # Get recent memories related to current context
        recent_memories = list(self.consciousness.memory_buffer)[-5:]
        
        # Check for contradictions or gaps
        # This is simplified - in production would use more sophisticated analysis
        coherence_check = {
            'memory_count': len(recent_memories),
            'topics_covered': list(set(word for memory in recent_memories 
                                     for word in memory.split() 
                                     if len(word) > 4)),
            'coherence_score': self.consciousness.state.global_coherence
        }
        
        return coherence_check
    
    async def action_update_emotion(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update emotional state based on expressed feeling"""
        emotion_word = match.group(1).lower()
        
        # Map emotion words to PAD values
        emotion_mappings = {
            # Positive emotions
            'happy': {'pleasure': 0.8, 'arousal': 0.6, 'dominance': 0.6},
            'excited': {'pleasure': 0.7, 'arousal': 0.8, 'dominance': 0.7},
            'confident': {'pleasure': 0.7, 'arousal': 0.5, 'dominance': 0.8},
            'curious': {'pleasure': 0.6, 'arousal': 0.6, 'dominance': 0.6},
            'grateful': {'pleasure': 0.8, 'arousal': 0.4, 'dominance': 0.5},
            
            # Negative emotions
            'confused': {'pleasure': 0.3, 'arousal': 0.6, 'dominance': 0.3},
            'uncertain': {'pleasure': 0.4, 'arousal': 0.5, 'dominance': 0.3},
            'concerned': {'pleasure': 0.3, 'arousal': 0.6, 'dominance': 0.5},
            'overwhelmed': {'pleasure': 0.2, 'arousal': 0.8, 'dominance': 0.2},
            
            # Neutral emotions
            'thoughtful': {'pleasure': 0.5, 'arousal': 0.4, 'dominance': 0.6},
            'focused': {'pleasure': 0.5, 'arousal': 0.6, 'dominance': 0.7}
        }
        
        if emotion_word in emotion_mappings:
            target_state = emotion_mappings[emotion_word]
            self.emotions.update_emotional_state({
                'sentiment': target_state['pleasure'],
                'arousal_change': target_state['arousal'] - self.emotions.current_state.arousal,
                'dominance_change': target_state['dominance'] - self.emotions.current_state.dominance,
                'trigger': 'self_expressed_emotion'
            })
            
            return {
                'emotion': emotion_word,
                'pad_values': target_state,
                'updated': True
            }
        
        return {
            'emotion': emotion_word,
            'updated': False,
            'reason': 'unrecognized_emotion'
        }
    
    async def action_analyze(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis"""
        # Set analytical focus
        self.consciousness.state.attention_focus = "detailed analysis mode"
        
        # Increase consciousness coherence temporarily
        original_coherence = self.consciousness.state.global_coherence
        self.consciousness.state.global_coherence = min(1.0, original_coherence + 0.1)
        
        # Gather relevant memories for analysis
        analysis_context = await self.memory_manager.retrieve_memories(
            query=context.get('current_topic', 'current discussion'),
            k=10
        )
        
        return {
            'mode': 'analytical',
            'coherence_boost': 0.1,
            'relevant_memories': len(analysis_context),
            'analysis_depth': 'thorough'
        }
    
    async def action_set_goal(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Set a new active goal"""
        goal = match.group(1).strip()
        
        # Add to active goals
        if goal not in self.consciousness.state.active_goals:
            self.consciousness.state.active_goals.append(goal)
            
            # Keep only recent goals
            if len(self.consciousness.state.active_goals) > 5:
                self.consciousness.state.active_goals = self.consciousness.state.active_goals[-5:]
        
        return {
            'goal': goal,
            'active_goals': self.consciousness.state.active_goals,
            'goal_count': len(self.consciousness.state.active_goals)
        }
    
    async def action_focus_attention(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update attention focus"""
        focus = match.group(1).strip()
        
        # Update attention
        self.consciousness.state.attention_focus = focus
        
        # Store in working memory
        self.consciousness.memory_buffer.append(f"Attention: {focus}")
        
        return {
            'attention_focus': focus,
            'timestamp': datetime.now().isoformat()
        }
    
    async def action_consolidate(self, match: re.Match, full_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger memory consolidation"""
        # This would trigger the Celery consolidation pipeline
        session_id = context.get('session_id')
        
        if session_id:
            # In real implementation, this would trigger Celery task
            logger.info(f"Triggering consolidation for session {session_id}")
            
            # For now, just organize working memory
            current_memories = list(self.consciousness.memory_buffer)
            
            # Group by topic (simplified)
            topics = {}
            for memory in current_memories:
                # Simple topic extraction (first significant word)
                words = memory.split()
                topic = next((w for w in words if len(w) > 4), 'general')
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append(memory)
            
            return {
                'consolidated': True,
                'topics_found': list(topics.keys()),
                'memory_count': len(current_memories)
            }
        
        return {
            'consolidated': False,
            'reason': 'no_session'
        }
    
    def get_available_actions(self) -> List[Dict[str, str]]:
        """Get list of available actions for documentation"""
        return [
            {
                'pattern': pattern,
                'action': action_type,
                'description': handler.__doc__.strip() if handler.__doc__ else ""
            }
            for pattern, handler, action_type in self.action_patterns
        ]