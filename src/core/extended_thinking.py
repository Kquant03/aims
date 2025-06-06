# src/core/extended_thinking.py
"""
Phase 5: Extended Thinking Integration
Real-time streaming of AI reasoning with progressive disclosure
"""
import asyncio
import json
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from collections import deque
import re

logger = logging.getLogger(__name__)

class ThoughtType(Enum):
    """Types of thoughts for color-coding and categorization"""
    ANALYTICAL = "analytical"      # Blue #2563EB - Logic and analysis
    EMOTIONAL = "emotional"        # Purple #7C3AED - Intuition and empathy
    CREATIVE = "creative"          # Orange #EA580C - Innovation and ideation
    UNCERTAIN = "uncertain"        # Gray #6B7280 - Exploration and questioning
    HYPOTHESIS = "hypothesis"      # Green #059669 - Theory formation
    VALIDATION = "validation"      # Indigo #4F46E5 - Checking/verifying
    DECISION = "decision"          # Red #DC2626 - Choice points
    META = "meta"                  # Pink #DB2777 - Thinking about thinking

@dataclass
class ThoughtFragment:
    """Represents a single thought in the reasoning chain"""
    id: str
    type: ThoughtType
    content: str
    confidence: float
    timestamp: datetime
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
        if self.metadata is None:
            self.metadata = {}

class ExtendedThinkingEngine:
    """
    Captures and streams AI reasoning in real-time with structured thought chains
    """
    
    def __init__(self, websocket_server=None, flush_interval_ms: int = 100):
        self.websocket_server = websocket_server
        self.flush_interval = flush_interval_ms / 1000.0  # Convert to seconds
        
        # Thought storage
        self.thought_buffer = deque(maxlen=1000)
        self.thought_tree: Dict[str, ThoughtFragment] = {}
        self.current_thought_id: Optional[str] = None
        
        # Streaming state
        self.is_streaming = False
        self.stream_queue: asyncio.Queue = asyncio.Queue()
        self.flush_task: Optional[asyncio.Task] = None
        
        # Pattern matchers for thought classification
        self.thought_patterns = {
            ThoughtType.ANALYTICAL: [
                r"(?:analyzing|considering|examining|evaluating)",
                r"(?:therefore|thus|because|since|given that)",
                r"(?:data suggests|evidence shows|analysis reveals)"
            ],
            ThoughtType.EMOTIONAL: [
                r"(?:feel|feeling|sense|emotion)",
                r"(?:empathy|compassion|concern|care)",
                r"(?:intuition|gut feeling|instinct)"
            ],
            ThoughtType.CREATIVE: [
                r"(?:imagine|envision|what if|suppose)",
                r"(?:creative|innovative|novel|unique)",
                r"(?:idea|concept|possibility)"
            ],
            ThoughtType.UNCERTAIN: [
                r"(?:uncertain|unsure|unclear|ambiguous)",
                r"(?:perhaps|maybe|possibly|might)",
                r"(?:question|wonder|curious)"
            ],
            ThoughtType.HYPOTHESIS: [
                r"(?:hypothesis|theory|propose|suggest)",
                r"(?:if.*then|assuming|suppose that)",
                r"(?:prediction|expect|anticipate)"
            ],
            ThoughtType.VALIDATION: [
                r"(?:verify|check|confirm|validate)",
                r"(?:correct|accurate|true|false)",
                r"(?:testing|examining|reviewing)"
            ],
            ThoughtType.DECISION: [
                r"(?:decide|choose|select|opt for)",
                r"(?:best option|preferred|optimal)",
                r"(?:conclusion|judgment|determination)"
            ],
            ThoughtType.META: [
                r"(?:thinking about|reflecting on|considering my)",
                r"(?:reasoning process|thought process|approach)",
                r"(?:meta-cognitive|self-aware|introspective)"
            ]
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {
            thought_type: [re.compile(pattern, re.IGNORECASE) 
                          for pattern in patterns]
            for thought_type, patterns in self.thought_patterns.items()
        }
    
    def classify_thought(self, content: str) -> ThoughtType:
        """Classify a thought fragment based on content patterns"""
        # Check each thought type's patterns
        for thought_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(content):
                    return thought_type
        
        # Default to analytical if no pattern matches
        return ThoughtType.ANALYTICAL
    
    def calculate_confidence(self, content: str, thought_type: ThoughtType) -> float:
        """Calculate confidence level for a thought"""
        confidence = 0.7  # Base confidence
        
        # Adjust based on uncertainty markers
        uncertainty_markers = ['maybe', 'perhaps', 'possibly', 'might', 'could']
        certainty_markers = ['definitely', 'certainly', 'clearly', 'obviously']
        
        content_lower = content.lower()
        
        for marker in uncertainty_markers:
            if marker in content_lower:
                confidence -= 0.1
        
        for marker in certainty_markers:
            if marker in content_lower:
                confidence += 0.1
        
        # Uncertain thoughts have inherently lower confidence
        if thought_type == ThoughtType.UNCERTAIN:
            confidence *= 0.7
        elif thought_type == ThoughtType.VALIDATION:
            confidence *= 1.1
        
        return max(0.1, min(1.0, confidence))
    
    async def start_thinking_stream(self, session_id: str, extended_mode: bool = True) -> str:
        """Start a new extended thinking stream"""
        self.is_streaming = True
        stream_id = f"stream_{session_id}_{int(datetime.now().timestamp())}"
        
        # Start flush task if not running
        if not self.flush_task or self.flush_task.done():
            self.flush_task = asyncio.create_task(self._flush_loop())
        
        # Broadcast stream start
        if self.websocket_server:
            await self.websocket_server.broadcast_update('thinking_started', {
                'stream_id': stream_id,
                'extended_mode': extended_mode,
                'timestamp': datetime.now().isoformat()
            })
        
        return stream_id
    
    async def capture_thought(self, content: str, parent_id: Optional[str] = None,
                            metadata: Optional[Dict[str, Any]] = None) -> ThoughtFragment:
        """Capture a thought fragment and add to stream"""
        # Generate thought ID
        thought_id = f"thought_{int(datetime.now().timestamp() * 1000)}"
        
        # Classify the thought
        thought_type = self.classify_thought(content)
        
        # Calculate confidence
        confidence = self.calculate_confidence(content, thought_type)
        
        # Create thought fragment
        thought = ThoughtFragment(
            id=thought_id,
            type=thought_type,
            content=content,
            confidence=confidence,
            timestamp=datetime.now(),
            parent_id=parent_id or self.current_thought_id,
            metadata=metadata or {}
        )
        
        # Update parent's children if applicable
        if thought.parent_id and thought.parent_id in self.thought_tree:
            self.thought_tree[thought.parent_id].children_ids.append(thought_id)
        
        # Store thought
        self.thought_tree[thought_id] = thought
        self.thought_buffer.append(thought)
        
        # Update current thought for chaining
        self.current_thought_id = thought_id
        
        # Queue for streaming
        await self.stream_queue.put(thought)
        
        return thought
    
    async def _flush_loop(self):
        """Flush thoughts to WebSocket clients at regular intervals"""
        batch = []
        
        while self.is_streaming:
            try:
                # Collect thoughts for flush interval
                deadline = asyncio.create_task(asyncio.sleep(self.flush_interval))
                
                while True:
                    try:
                        # Get thought with timeout
                        thought = await asyncio.wait_for(
                            self.stream_queue.get(),
                            timeout=self.flush_interval
                        )
                        batch.append(thought)
                        
                        # Flush if batch is getting large
                        if len(batch) >= 10:
                            break
                            
                    except asyncio.TimeoutError:
                        break
                
                # Flush batch if not empty
                if batch and self.websocket_server:
                    await self._flush_thoughts(batch)
                    batch = []
                
                # Wait for remainder of interval
                await deadline
                
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _flush_thoughts(self, thoughts: List[ThoughtFragment]):
        """Send batch of thoughts to WebSocket clients"""
        thought_data = []
        
        for thought in thoughts:
            data = {
                'id': thought.id,
                'type': thought.type.value,
                'content': thought.content,
                'confidence': thought.confidence,
                'timestamp': thought.timestamp.isoformat(),
                'parent_id': thought.parent_id,
                'metadata': thought.metadata,
                'color': self._get_thought_color(thought.type)
            }
            thought_data.append(data)
        
        await self.websocket_server.broadcast_update('thinking_update', {
            'thoughts': thought_data,
            'batch_size': len(thought_data),
            'timestamp': datetime.now().isoformat()
        })
    
    def _get_thought_color(self, thought_type: ThoughtType) -> str:
        """Get color code for thought type (WCAG AA compliant)"""
        color_map = {
            ThoughtType.ANALYTICAL: "#2563EB",   # Blue
            ThoughtType.EMOTIONAL: "#7C3AED",    # Purple
            ThoughtType.CREATIVE: "#EA580C",     # Orange
            ThoughtType.UNCERTAIN: "#6B7280",    # Gray
            ThoughtType.HYPOTHESIS: "#059669",   # Green
            ThoughtType.VALIDATION: "#4F46E5",   # Indigo
            ThoughtType.DECISION: "#DC2626",     # Red
            ThoughtType.META: "#DB2777"          # Pink
        }
        return color_map.get(thought_type, "#6B7280")
    
    async def stop_thinking_stream(self, summary: Optional[str] = None):
        """Stop the thinking stream and send summary"""
        self.is_streaming = False
        
        # Cancel flush task
        if self.flush_task and not self.flush_task.done():
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        # Generate thought summary
        thought_summary = self._generate_thought_summary()
        
        # Broadcast stream end
        if self.websocket_server:
            await self.websocket_server.broadcast_update('thinking_ended', {
                'summary': summary or "Thinking process completed",
                'thought_stats': thought_summary,
                'total_thoughts': len(self.thought_tree),
                'timestamp': datetime.now().isoformat()
            })
        
        # Reset state
        self.current_thought_id = None
    
    def _generate_thought_summary(self) -> Dict[str, Any]:
        """Generate summary statistics of thought process"""
        if not self.thought_tree:
            return {}
        
        # Count thoughts by type
        type_counts = {}
        confidence_by_type = {}
        
        for thought in self.thought_tree.values():
            thought_type = thought.type.value
            
            if thought_type not in type_counts:
                type_counts[thought_type] = 0
                confidence_by_type[thought_type] = []
            
            type_counts[thought_type] += 1
            confidence_by_type[thought_type].append(thought.confidence)
        
        # Calculate average confidence by type
        avg_confidence = {
            t: sum(confs) / len(confs) if confs else 0
            for t, confs in confidence_by_type.items()
        }
        
        # Find deepest thought chain
        max_depth = 0
        for thought in self.thought_tree.values():
            depth = self._calculate_thought_depth(thought.id)
            max_depth = max(max_depth, depth)
        
        return {
            'type_distribution': type_counts,
            'average_confidence': avg_confidence,
            'max_chain_depth': max_depth,
            'total_thoughts': len(self.thought_tree)
        }
    
    def _calculate_thought_depth(self, thought_id: str, depth: int = 0) -> int:
        """Calculate depth of a thought in the tree"""
        thought = self.thought_tree.get(thought_id)
        if not thought or not thought.parent_id:
            return depth
        
        return self._calculate_thought_depth(thought.parent_id, depth + 1)
    
    def get_thought_chain(self, thought_id: str) -> List[ThoughtFragment]:
        """Get complete chain of thoughts leading to a specific thought"""
        chain = []
        current_id = thought_id
        
        while current_id:
            thought = self.thought_tree.get(current_id)
            if not thought:
                break
            
            chain.append(thought)
            current_id = thought.parent_id
        
        return list(reversed(chain))
    
    def get_thought_tree_visualization(self) -> Dict[str, Any]:
        """Get thought tree in format suitable for D3.js visualization"""
        nodes = []
        links = []
        
        for thought in self.thought_tree.values():
            # Add node
            nodes.append({
                'id': thought.id,
                'type': thought.type.value,
                'content': thought.content[:100] + '...' if len(thought.content) > 100 else thought.content,
                'confidence': thought.confidence,
                'color': self._get_thought_color(thought.type)
            })
            
            # Add links to children
            for child_id in thought.children_ids:
                links.append({
                    'source': thought.id,
                    'target': child_id,
                    'value': 1
                })
        
        return {
            'nodes': nodes,
            'links': links
        }