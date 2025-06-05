# attention_agent.py - Generates contextual attention focus insights
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AttentionAgent:
    """Generates contextual attention insights as Claude's initial reaction to inputs"""
    
    def __init__(self, claude_client):
        self.client = claude_client
        self.last_focus = ""
        self.focus_history = []
        
    async def generate_attention_focus(self, 
                                     user_message: str, 
                                     context: Dict[str, Any]) -> str:
        """Generate Claude's knee-jerk reaction/attention focus for the input"""
        
        # Build a lightweight prompt for quick attention generation
        prompt = f"""Based on this user message and context, provide a 1-2 sentence "attention focus" - your immediate, instinctive reaction or what seems most important to address. Be concise and direct.

User message: "{user_message}"

Current emotional state: {context.get('emotion_label', 'neutral')} (pleasure: {context.get('pleasure', 0.5):.1f})
Recent topics: {', '.join(context.get('recent_memories', [])[:3])}
Personality dominance: {context.get('personality', {}).get('openness', 0.8):.1f}

Your attention focus (1-2 sentences max):"""

        try:
            # Use a faster model for quick reactions
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",  # Faster model for quick responses
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            # Extract the focus text
            focus_text = response.content[0].text.strip()
            
            # Store in history
            self.focus_history.append({
                'timestamp': datetime.now(),
                'user_message': user_message[:100],
                'focus': focus_text
            })
            
            # Keep only recent history
            if len(self.focus_history) > 20:
                self.focus_history = self.focus_history[-20:]
            
            self.last_focus = focus_text
            return focus_text
            
        except Exception as e:
            logger.error(f"Error generating attention focus: {e}")
            # Fallback to simple analysis
            if "?" in user_message:
                return "The user is asking a question that needs addressing."
            elif any(word in user_message.lower() for word in ['help', 'problem', 'issue', 'error']):
                return "The user seems to need assistance with something."
            else:
                return "Processing the user's input and formulating a response."
    
    def get_focus_history(self) -> list:
        """Get recent attention focus history"""
        return self.focus_history[-10:]  # Last 10 focuses