# websocket_server.py - WebSocket Server for Real-time Consciousness Updates
import asyncio
import json
import logging
from typing import Set, Dict, Any
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol
import aiohttp
from aiohttp import web
import aiohttp_cors

logger = logging.getLogger(__name__)

class ConsciousnessWebSocketServer:
    """WebSocket server for real-time consciousness state updates"""
    
    def __init__(self, claude_interface, host='localhost', port=8765):
        self.claude_interface = claude_interface
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.update_interval = 0.5  # Send updates every 500ms
        
    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        logger.info(f"Client {websocket.remote_address} connected")
        
        # Send initial state
        await self.send_current_state(websocket)
    
    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Remove a WebSocket client"""
        self.clients.remove(websocket)
        logger.info(f"Client {websocket.remote_address} disconnected")
    
    async def send_current_state(self, websocket: WebSocketServerProtocol):
        """Send current consciousness state to a client"""
        try:
            state_data = {
                'type': 'consciousness_update',
                'timestamp': datetime.now().isoformat(),
                'data': {
                    'coherence': self.claude_interface.consciousness.state.global_coherence,
                    'attention_focus': self.claude_interface.consciousness.state.attention_focus,
                    'emotional_state': self.claude_interface.emotions.current_state.__dict__,
                    'emotion_label': self.claude_interface.emotions.get_closest_emotion_label()[0],
                    'personality_traits': self.claude_interface.personality.profile.get_traits(),
                    'working_memory': list(self.claude_interface.consciousness.memory_buffer),
                    'interaction_count': self.claude_interface.consciousness.state.interaction_count
                }
            }
            
            await websocket.send(json.dumps(state_data))
        except Exception as e:
            logger.error(f"Error sending state update: {e}")
    
    async def broadcast_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast an update to all connected clients"""
        if not self.clients:
            return
        
        message = json.dumps({
            'type': update_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        })
        
        # Send to all clients concurrently
        await asyncio.gather(
            *[client.send(message) for client in self.clients],
            return_exceptions=True
        )
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a WebSocket client connection"""
        await self.register_client(websocket)
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming client messages"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'get_state':
                await self.send_current_state(websocket)
            
            elif message_type == 'get_memory_stats':
                stats = self.claude_interface.memory_manager.get_statistics()
                await websocket.send(json.dumps({
                    'type': 'memory_stats',
                    'data': stats
                }))
            
            elif message_type == 'get_emotion_trajectory':
                trajectory = self.claude_interface.emotions.get_state_trajectory()
                await websocket.send(json.dumps({
                    'type': 'emotion_trajectory',
                    'data': [{'pleasure': s.pleasure, 
                             'arousal': s.arousal, 
                             'dominance': s.dominance} for s in trajectory]
                }))
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def periodic_update_loop(self):
        """Send periodic updates to all clients"""
        while True:
            await self.broadcast_update('consciousness_update', {
                'coherence': self.claude_interface.consciousness.state.global_coherence,
                'attention_focus': self.claude_interface.consciousness.state.attention_focus,
                'emotional_state': self.claude_interface.emotions.current_state.__dict__,
                'emotion_label': self.claude_interface.emotions.get_closest_emotion_label()[0],
                'working_memory_size': len(self.claude_interface.consciousness.memory_buffer)
            })
            
            await asyncio.sleep(self.update_interval)
    
    async def start(self):
        """Start the WebSocket server"""
        # Start the WebSocket server
        server = await websockets.serve(
            self.handle_client, 
            self.host, 
            self.port
        )
        
        # Start periodic update loop
        update_task = asyncio.create_task(self.periodic_update_loop())
        
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
        
        # Keep server running
        await asyncio.Future()