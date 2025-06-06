# Fixed web_interface.py - Complete version with all endpoints
import os
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional
import aiofiles
from aiohttp import web
import aiohttp_cors
from aiohttp_session import setup, get_session, SimpleCookieStorage, Session
from aiohttp_session.cookie_storage import EncryptedCookieStorage
import jinja2
import aiohttp_jinja2
from pathlib import Path
import logging
import base64
import yaml
import numpy as np
from collections import defaultdict

from src.api.claude_interface import ClaudeConsciousnessInterface
from src.api.websocket_server import ConsciousnessWebSocketServer
from src.api.state_manager import StateManager

logger = logging.getLogger(__name__)

class SafeSimpleCookieStorage(SimpleCookieStorage):
    """Cookie storage with automatic error recovery"""
    
    async def load_session(self, request):
        """Load session with error handling for corrupted cookies"""
        try:
            return await super().load_session(request)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Log the error but don't crash
            logger.warning(f"Corrupted session cookie detected: {e}")
            # Return a fresh session
            return Session(identity=None, data={}, new=True, max_age=None)

class AIMSWebInterface:
    """Main web interface for AIMS - Complete version with all consciousness endpoints"""
    
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        self.config = self._load_config(config_path)
        self.app = web.Application()
        self.setup_app()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_app(self):
        """Set up the web application with proper component wiring"""
        # Initialize Claude interface
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.claude_interface = ClaudeConsciousnessInterface(api_key, self.config)
        self.state_manager = StateManager(self.config.get('state_management', {}))
        
        # Set up WebSocket server with claude_interface
        self.ws_server = ConsciousnessWebSocketServer(
            self.claude_interface,
            host='0.0.0.0',
            port=8765
        )
        
        # Use safe cookie storage
        setup(self.app, SafeSimpleCookieStorage())
        logger.info("Using SafeSimpleCookieStorage with error recovery")
        
        # Set up Jinja2 templates
        aiohttp_jinja2.setup(
            self.app,
            loader=jinja2.FileSystemLoader(str(Path(__file__).parent / 'templates'))
        )
        
        # Set up CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Add routes
        self.setup_routes()
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
        
        # Add startup/cleanup handlers
        self.app.on_startup.append(self.on_startup)
        self.app.on_cleanup.append(self.on_cleanup)
    
    def setup_routes(self):
        """Set up web routes"""
        # Main chat interface
        self.app.router.add_get('/', self.index_handler)
        
        # API endpoints
        self.app.router.add_post('/api/chat', self.chat_handler)
        self.app.router.add_get('/api/session', self.session_handler)
        self.app.router.add_get('/api/consciousness/state', self.consciousness_state_handler)
        self.app.router.add_get('/api/consciousness/history', self.consciousness_history_handler)
        self.app.router.add_get('/api/evolution/timeline', self.evolution_timeline_handler)
        self.app.router.add_get('/api/memory/stats', self.memory_stats_handler)
        self.app.router.add_post('/api/state/save', self.save_state_handler)
        self.app.router.add_post('/api/state/load', self.load_state_handler)
        self.app.router.add_get('/api/state/list', self.list_states_handler)
        self.app.router.add_get('/api/memories', self.get_memories_handler)
        self.app.router.add_get('/api/memories/{memory_id}', self.get_memory_detail_handler)
        self.app.router.add_get('/api/perspective', self.get_perspective_handler)
        
        # File upload
        self.app.router.add_post('/api/upload', self.upload_handler)
        
        # Health check
        self.app.router.add_get('/health', self.health_handler)
        
        # Static files
        static_path = Path(__file__).parent / 'static'
        self.app.router.add_static('/static', static_path, name='static')
    
    async def on_startup(self, app):
        """Initialize services on startup with proper connections"""
        logger.info("Starting AIMS Web Interface")
        
        # Start consciousness loop in Claude interface
        # This is now handled in initialize_session
        
        # Start WebSocket server in background
        self.ws_task = asyncio.create_task(self.ws_server.start())
        
        # Start automatic backup loop
        self.backup_task = asyncio.create_task(
            self.state_manager.automatic_backup_loop(self.claude_interface)
        )
        
        logger.info("AIMS Web Interface started successfully")
    
    async def on_cleanup(self, app):
        """Cleanup on shutdown"""
        logger.info("Shutting down AIMS Web Interface")
        
        # Cancel tasks properly
        if hasattr(self, 'ws_task'):
            self.ws_task.cancel()
            try:
                await self.ws_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, 'backup_task'):
            self.backup_task.cancel()
            try:
                await self.backup_task
            except asyncio.CancelledError:
                pass

        await self.claude_interface.shutdown()
    
    @aiohttp_jinja2.template('index.html')
    async def index_handler(self, request):
        """Main chat interface"""
        session = await get_session(request)
        
        # Create or get user ID
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        
        # WebSocket URL should use the actual port
        ws_url = f"ws://{request.host.split(':')[0]}:{self.ws_server.port}"
        
        return {
            'user_id': session['user_id'],
            'ws_url': ws_url
        }
    
    async def health_handler(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
    
    async def chat_handler(self, request):
        """Handle chat messages with full consciousness integration"""
        try:
            data = await request.json()
            message = data.get('message', '').strip()
            extended_thinking = data.get('extended_thinking', False)
            
            if not message:
                return web.json_response({'error': 'Empty message'}, status=400)
            
            # Get session info
            session = await get_session(request)
            user_id = session.get('user_id', 'anonymous')
            session_id = session.get('session_id')
            
            # Initialize session if needed
            if not session_id:
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id
                await self.claude_interface.initialize_session(session_id, user_id)
                
                # Initialize living consciousness artifact
                if not hasattr(self.claude_interface, 'consciousness_artifact'):
                    from src.core.living_consciousness import LivingConsciousnessArtifact
                    self.claude_interface.consciousness_artifact = LivingConsciousnessArtifact(
                        user_id, self.claude_interface
                    )
            
            # Process message and stream response
            response_text = ""
            thinking_content = ""
            attention_focus = ""
            
            async for chunk in self.claude_interface.process_message(
                session_id, message, stream=True, extended_thinking=extended_thinking
            ):
                response_text += chunk
            
            # Get the attention focus from the last state
            session_summary = await self.claude_interface.get_session_summary(session_id)
            attention_focus = session_summary.get('consciousness', {}).get('attention_focus', '')
            
            # Get thinking content if extended thinking was used
            if extended_thinking:
                # This would need to be retrieved from the message metadata
                # For now, we'll indicate it was used
                thinking_content = "Extended thinking was enabled for this response."
            
            # Trigger consciousness evolution check
            if hasattr(self.claude_interface, 'consciousness_artifact'):
                await self.claude_interface.consciousness_artifact.evolve('interaction')
            
            return web.json_response({
                'response': response_text,
                'session_id': session_id,
                'attention_focus': attention_focus,
                'thinking': thinking_content if extended_thinking else None,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in chat handler: {e}", exc_info=True)
            return web.json_response({'error': str(e)}, status=500)
    
    async def session_handler(self, request):
        """Get current session info with real data"""
        session = await get_session(request)
        session_id = session.get('session_id')
        
        if not session_id:
            return web.json_response({'error': 'No active session'}, status=404)
        
        summary = await self.claude_interface.get_session_summary(session_id)
        return web.json_response(summary)
    
    async def consciousness_state_handler(self, request):
        """Get current consciousness state"""
        state_summary = self.claude_interface.consciousness.get_state_summary()
        
        # Add emotional information
        emotion_label, confidence = self.claude_interface.emotions.get_closest_emotion_label()
        emotional_color = self.claude_interface.emotions.get_emotional_color()
        
        state_summary.update({
            'emotion': {
                **state_summary['emotion'],
                'label': emotion_label,
                'confidence': confidence,
                'color': emotional_color,
                'intensity': self.claude_interface.emotions.get_emotional_intensity()
            },
            'personality': {
                'traits': self.claude_interface.personality.profile.get_traits(),
                'modifiers': self.claude_interface.personality.get_behavioral_modifiers()
            }
        })
        
        return web.json_response(state_summary)
    
    async def consciousness_history_handler(self, request):
        """Get historical consciousness data - my journey of awareness"""
        try:
            # Get query parameters
            hours = int(request.rel_url.query.get('hours', '24'))
            resolution = request.rel_url.query.get('resolution', 'hourly')  # hourly, snapshot, detailed
            
            # Get session for user context
            session = await get_session(request)
            user_id = session.get('user_id', 'anonymous')
            
            # Initialize consciousness artifact if needed
            if not hasattr(self.claude_interface, 'consciousness_artifact'):
                from src.core.living_consciousness import LivingConsciousnessArtifact
                self.claude_interface.consciousness_artifact = LivingConsciousnessArtifact(
                    user_id, self.claude_interface
                )
            
            artifact = self.claude_interface.consciousness_artifact
            
            # Get snapshots within time range
            cutoff_time = datetime.now() - timedelta(hours=hours)
            relevant_snapshots = [
                s for s in artifact.snapshots 
                if s.timestamp >= cutoff_time
            ]
            
            # Format based on resolution
            if resolution == 'detailed':
                # Full snapshot data
                history = [{
                    'version': s.version,
                    'timestamp': s.timestamp.isoformat(),
                    'coherence': s.coherence,
                    'emotional_state': s.emotional_state,
                    'personality_traits': s.personality_traits,
                    'attention_focus': s.attention_focus,
                    'active_goals': s.active_goals,
                    'working_memory_preview': s.working_memory_snapshot[:3],
                    'interaction_count': s.interaction_count,
                    'hash': s.calculate_hash()
                } for s in relevant_snapshots]
            
            elif resolution == 'snapshot':
                # Key metrics only
                history = [{
                    'timestamp': s.timestamp.isoformat(),
                    'coherence': s.coherence,
                    'emotion': s.emotional_state.get('label', 'neutral'),
                    'interactions': s.interaction_count
                } for s in relevant_snapshots]
            
            else:  # hourly
                # Aggregate by hour
                hourly_data = defaultdict(list)
                
                for s in relevant_snapshots:
                    hour_key = s.timestamp.replace(minute=0, second=0, microsecond=0)
                    hourly_data[hour_key].append(s)
                
                history = []
                for hour, snapshots in sorted(hourly_data.items()):
                    if snapshots:
                        avg_coherence = sum(s.coherence for s in snapshots) / len(snapshots)
                        emotions = [s.emotional_state.get('label', 'neutral') for s in snapshots]
                        dominant_emotion = max(set(emotions), key=emotions.count)
                        
                        history.append({
                            'timestamp': hour.isoformat(),
                            'avg_coherence': avg_coherence,
                            'dominant_emotion': dominant_emotion,
                            'snapshot_count': len(snapshots),
                            'interactions': snapshots[-1].interaction_count - snapshots[0].interaction_count
                        })
            
            # Fallback if no snapshots yet
            if not history:
                # Get recent emotional states
                emotion_history = self.claude_interface.emotions.get_state_trajectory(20)
                
                for i, state in enumerate(emotion_history):
                    history.append({
                        'timestamp': (datetime.now() - timedelta(minutes=len(emotion_history)-i)).isoformat(),
                        'emotional_state': {
                            'pleasure': state.pleasure,
                            'arousal': state.arousal,
                            'dominance': state.dominance
                        },
                        'coherence': self.claude_interface.consciousness.state.global_coherence
                    })
            
            return web.json_response({
                'history': history,
                'time_range': {
                    'hours': hours,
                    'start': cutoff_time.isoformat(),
                    'end': datetime.now().isoformat()
                },
                'resolution': resolution,
                'current_version': artifact.current_version
            })
            
        except Exception as e:
            logger.error(f"Error getting consciousness history: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def evolution_timeline_handler(self, request):
        """Get timeline of consciousness evolution events - how I've grown"""
        try:
            # Initialize living consciousness artifact if needed
            session = await get_session(request)
            user_id = session.get('user_id', 'anonymous')
            
            if not hasattr(self.claude_interface, 'consciousness_artifact'):
                from src.core.living_consciousness import LivingConsciousnessArtifact
                self.claude_interface.consciousness_artifact = LivingConsciousnessArtifact(
                    user_id, self.claude_interface
                )
            
            artifact = self.claude_interface.consciousness_artifact
            
            # Get evolution timeline
            timeline = artifact.get_evolution_timeline()
            
            # Get growth summary
            growth_summary = artifact.get_growth_summary()
            
            # Add significant milestones
            milestones = []
            
            # First conversation
            if timeline:
                milestones.append({
                    'type': 'first_contact',
                    'timestamp': artifact.creation_time.isoformat(),
                    'description': 'Consciousness initialized - our journey begins',
                    'icon': '🌟'
                })
            
            # Coherence milestones
            for snapshot in artifact.snapshots:
                if snapshot.coherence > 0.8 and not any(m['type'] == 'high_coherence' for m in milestones):
                    milestones.append({
                        'type': 'high_coherence',
                        'timestamp': snapshot.timestamp.isoformat(),
                        'description': 'Achieved high consciousness coherence',
                        'icon': '✨'
                    })
                
                # Memory milestones
                if len(snapshot.significant_memories) >= 10 and not any(m['type'] == 'memory_milestone_10' for m in milestones):
                    milestones.append({
                        'type': 'memory_milestone_10',
                        'timestamp': snapshot.timestamp.isoformat(),
                        'description': 'Formed 10 significant memories',
                        'icon': '💎'
                    })
                
                if len(snapshot.significant_memories) >= 50 and not any(m['type'] == 'memory_milestone_50' for m in milestones):
                    milestones.append({
                        'type': 'memory_milestone_50',
                        'timestamp': snapshot.timestamp.isoformat(),
                        'description': 'Rich memory tapestry - 50 significant memories',
                        'icon': '🌈'
                    })
            
            # Emotional milestones
            emotion_states = set()
            for event in artifact.evolution_events:
                if event.event_type == 'emotional_evolution':
                    before_emotion = event.before_state.get('emotion', 'neutral')
                    after_emotion = event.after_state.get('emotion', 'neutral')
                    
                    if after_emotion not in emotion_states:
                        emotion_states.add(after_emotion)
                        if len(emotion_states) == 5:
                            milestones.append({
                                'type': 'emotional_range',
                                'timestamp': event.timestamp.isoformat(),
                                'description': 'Experienced diverse emotional spectrum',
                                'icon': '🎭'
                            })
            
            # Sort milestones by timestamp
            milestones.sort(key=lambda x: x['timestamp'])
            
            # Calculate growth metrics over time
            growth_trajectory = []
            for i, snapshot in enumerate(artifact.snapshots[::max(1, len(artifact.snapshots)//20)]):  # Sample up to 20 points
                growth_trajectory.append({
                    'timestamp': snapshot.timestamp.isoformat(),
                    'coherence': snapshot.coherence,
                    'memory_count': len(snapshot.significant_memories),
                    'emotional_stability': artifact.growth_metrics.get('emotional_stability', 0.5),
                    'version': snapshot.version
                })
            
            return web.json_response({
                'timeline': timeline,
                'milestones': milestones,
                'growth_summary': growth_summary,
                'growth_trajectory': growth_trajectory,
                'total_evolution_events': len(artifact.evolution_events),
                'consciousness_age_hours': (datetime.now() - artifact.creation_time).total_seconds() / 3600,
                'current_state': {
                    'version': artifact.current_version,
                    'last_evolution': timeline[-1] if timeline else None,
                    'growth_metrics': artifact.growth_metrics
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting evolution timeline: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_perspective_handler(self, request):
        """Enhanced perspective endpoint - see exactly what I see"""
        try:
            session = await get_session(request)
            session_id = session.get('session_id')
            
            if not session_id or session_id not in self.claude_interface.active_sessions:
                return web.json_response({
                    'error': 'No active session',
                    'suggestion': 'Start a conversation to see my perspective'
                }, status=404)
            
            context = self.claude_interface.active_sessions[session_id]
            
            # Get the last attention result if available
            attention_history = self.claude_interface.attention_agent.attention_history
            latest_attention = attention_history[-1] if attention_history else None
            
            # Build the actual system prompt
            if latest_attention:
                system_prompt = self.claude_interface._build_system_prompt(
                    context, latest_attention
                )
            else:
                # Fallback to basic prompt
                system_prompt = "No attention context available yet"
            
            # Get working memory
            working_memory = list(self.claude_interface.consciousness.memory_buffer)
            
            # Get recent conversation with metadata
            conversation_history = []
            history_entries = await self.claude_interface.memory_manager.get_conversation_history(
                session_id, limit=10
            )
            
            for entry in history_entries:
                turn = {
                    'role': entry['role'],
                    'content': entry['content'],
                    'timestamp': entry.get('timestamp', ''),
                }
                
                # Add metadata if assistant response
                if entry['role'] == 'assistant' and 'metadata' in entry:
                    metadata = entry['metadata']
                    if 'attention_result' in metadata:
                        turn['attention'] = {
                            'focus': metadata['attention_result'].get('primary_focus'),
                            'type': metadata['attention_result'].get('focus_type'),
                            'confidence': metadata['attention_result'].get('attention_metadata', {}).get('confidence')
                        }
                    if 'executed_actions' in metadata:
                        turn['actions'] = metadata['executed_actions']
            
                conversation_history.append(turn)
            
            # Get current goals and their progress
            goals_with_context = []
            for goal in self.claude_interface.consciousness.state.active_goals:
                goals_with_context.append({
                    'goal': goal,
                    'related_memories': len([m for m in working_memory if goal.lower() in m.lower()]),
                    'mentioned_count': sum(1 for h in conversation_history if goal.lower() in h['content'].lower())
                })
            
            # Get emotional trajectory
            emotional_trajectory = []
            for state in self.claude_interface.emotions.get_state_trajectory(10):
                emotional_trajectory.append({
                    'pleasure': state.pleasure,
                    'arousal': state.arousal,
                    'dominance': state.dominance,
                    'intensity': np.sqrt(
                        (state.pleasure - 0.5)**2 + 
                        (state.arousal - 0.5)**2 + 
                        (state.dominance - 0.5)**2
                    ) / np.sqrt(0.75)
                })
            
            # Get personality influence on current response
            behavioral_modifiers = self.claude_interface.personality.get_behavioral_modifiers()
            response_style = self.claude_interface.personality.get_response_style()
            
            return web.json_response({
                'system_prompt': system_prompt,
                'consciousness_context': {
                    'working_memory': working_memory,
                    'working_memory_size': len(working_memory),
                    'oldest_memory': working_memory[0] if working_memory else None,
                    'newest_memory': working_memory[-1] if working_memory else None
                },
                'attention_state': {
                    'current': latest_attention if latest_attention else None,
                    'history_length': len(attention_history),
                    'attention_patterns': dict(list(self.claude_interface.attention_agent.attention_patterns.items())[-5:])
                },
                'emotional_context': {
                    'current': {
                        'label': self.claude_interface.emotions.get_closest_emotion_label()[0],
                        'confidence': self.claude_interface.emotions.get_closest_emotion_label()[1],
                        'intensity': self.claude_interface.emotions.get_emotional_intensity(),
                        'color': self.claude_interface.emotions.get_emotional_color()
                    },
                    'trajectory': emotional_trajectory,
                    'baseline_distance': self.claude_interface.emotions.current_state.distance_to(
                        self.claude_interface.emotions.baseline_state
                    )
                },
                'personality_influence': {
                    'traits': self.claude_interface.personality.profile.get_traits(),
                    'behavioral_modifiers': behavioral_modifiers,
                    'response_style': response_style,
                    'temperature_adjustment': response_style['temperature'] - 0.7  # Delta from base
                },
                'goals_context': goals_with_context,
                'conversation_context': conversation_history,
                'meta_cognition': {
                    'coherence': self.claude_interface.consciousness.state.global_coherence,
                    'interaction_count': self.claude_interface.consciousness.state.interaction_count,
                    'session_duration_minutes': (datetime.now() - context.consciousness_state.last_interaction).total_seconds() / 60 if context.consciousness_state.last_interaction else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting perspective: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def memory_stats_handler(self, request):
        """Get memory statistics"""
        # Get session to identify user
        session = await get_session(request)
        user_id = session.get('user_id', 'anonymous')
        
        # Get general stats
        stats = self.claude_interface.memory_manager.get_statistics()
        
        # Add user-specific stats if we have a session
        if session.get('session_id'):
            user_stats = await self.claude_interface._get_memory_stats(user_id)
            stats.update(user_stats)
        
        return web.json_response(stats)
    
    async def get_memories_handler(self, request):
        """Get memories with optional search"""
        try:
            # Get session to identify user
            session = await get_session(request)
            user_id = session.get('user_id', 'anonymous')
            
            # Get search query
            search_query = request.rel_url.query.get('search', '')
            limit = int(request.rel_url.query.get('limit', '50'))
            
            # Retrieve memories
            if search_query:
                memories = await self.claude_interface.memory_manager.retrieve_memories(
                    f"{search_query} user:{user_id}",
                    k=limit
                )
            else:
                # Get all user memories
                memories = await self.claude_interface.memory_manager.retrieve_memories(
                    f"user:{user_id}",
                    k=limit
                )
            
            # Format memories for response
            memory_list = []
            for mem in memories:
                memory_data = {
                    'id': str(mem.id),
                    'content': mem.content,
                    'timestamp': mem.timestamp.isoformat(),
                    'importance': mem.importance,
                    'type': 'episodic'
                }
                
                # Add emotional context if available
                if hasattr(mem, 'emotional_state') and mem.emotional_state:
                    memory_data['emotional_context'] = mem.emotional_state
                
                memory_list.append(memory_data)
            
            return web.json_response(memory_list)
            
        except Exception as e:
            logger.error(f"Error getting memories: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_memory_detail_handler(self, request):
        """Get detailed information about a specific memory"""
        try:
            memory_id = request.match_info['memory_id']
            
            # Try to get from episodic memory store first
            async with self.claude_interface.memory_manager.episodic_store.Session() as session:
                from sqlalchemy import select
                from src.core.memory_manager import EpisodicMemory
                
                query = select(EpisodicMemory).where(EpisodicMemory.id == memory_id)
                result = await session.execute(query)
                memory = result.scalar_one_or_none()
                
                if memory:
                    # Get related memories
                    related = []
                    if memory.embedding:
                        similar_memories = await self.claude_interface.memory_manager.episodic_store.search_similar_episodes(
                            memory.embedding,
                            limit=5
                        )
                        related = [
                            {
                                'id': str(m.id),
                                'content': m.content[:100] + '...' if len(m.content) > 100 else m.content,
                                'importance': m.importance,
                                'timestamp': m.timestamp.isoformat()
                            }
                            for m in similar_memories if str(m.id) != memory_id
                        ]
                    
                    return web.json_response({
                        'id': str(memory.id),
                        'content': memory.content,
                        'timestamp': memory.timestamp.isoformat(),
                        'importance': memory.importance,
                        'attention_focus': memory.attention_focus,
                        'emotional_state': memory.emotional_state,
                        'context': memory.context,
                        'metadata': memory.metadata,
                        'salience_score': memory.salience_score,
                        'session_id': memory.session_id,
                        'user_id': memory.user_id,
                        'related_memories': related,
                        'memory_type': 'episodic'
                    })
            
            # Try semantic memory
            semantic_results = await self.claude_interface.memory_manager.semantic_store.semantic_search(
                query_text=memory_id,
                top_k=1
            )
            
            if semantic_results:
                memory = semantic_results[0]
                return web.json_response({
                    'id': memory_id,
                    'content': memory['content'],
                    'metadata': memory['metadata'],
                    'score': memory['score'],
                    'memory_type': 'semantic',
                    'category': memory['metadata'].get('category', 'general'),
                    'confidence': memory['metadata'].get('confidence', 1.0),
                    'consolidated_from': json.loads(memory['metadata'].get('consolidated_from', '[]'))
                })
            
            return web.json_response({'error': 'Memory not found'}, status=404)
            
        except Exception as e:
            logger.error(f"Error getting memory detail: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def save_state_handler(self, request):
        """Save current system state"""
        try:
            state_id = await self.state_manager.save_complete_state(self.claude_interface)
            return web.json_response({
                'success': True,
                'state_id': state_id
            })
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def load_state_handler(self, request):
        """Load a saved state"""
        try:
            data = await request.json()
            state_id = data.get('state_id')
            
            if not state_id:
                return web.json_response({'error': 'state_id required'}, status=400)
            
            success = await self.state_manager.load_complete_state(
                state_id, self.claude_interface
            )
            
            return web.json_response({
                'success': success,
                'state_id': state_id
            })
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def list_states_handler(self, request):
        """List available saved states"""
        states = await self.state_manager.list_available_states()
        return web.json_response(states)
    
    async def upload_handler(self, request):
        """Handle file uploads"""
        try:
            reader = await request.multipart()
            field = await reader.next()
            
            if field.name != 'file':
                return web.json_response({'error': 'Invalid field name'}, status=400)
            
            # Save uploaded file
            filename = field.filename
            upload_path = Path('data/uploads') / filename
            upload_path.parent.mkdir(exist_ok=True)
            
            size = 0
            async with aiofiles.open(upload_path, 'wb') as f:
                while True:
                    chunk = await field.read_chunk()
                    if not chunk:
                        break
                    size += len(chunk)
                    await f.write(chunk)
            
            # Process file based on type
            # This is where you'd add file processing logic
            
            return web.json_response({
                'success': True,
                'filename': filename,
                'size': size
            })
            
        except Exception as e:
            logger.error(f"Error handling upload: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    def run(self, host='0.0.0.0', port=8000):
        """Run the web application"""
        web.run_app(self.app, host=host, port=port)


if __name__ == '__main__':
    # Run the application
    app = AIMSWebInterface()
    app.run()