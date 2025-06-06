# message_fix.py - Fix for Anthropic API message format
from typing import List, Dict, Any

def format_messages_for_claude(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Format messages for Claude API"""
    formatted = []
    for msg in messages:
        formatted.append({
            "role": str(msg.get("role", "user")),
            "content": str(msg.get("content", ""))
        })
    return formatted
