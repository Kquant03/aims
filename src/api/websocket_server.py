# Fixed websocket_server.py - Real state broadcasting
import asyncio
import socket
import json
import logging
import time
from typing import Set, Dict, List, Any, Optional, Union
from datetime import datetime
from collections import deque
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

logger = logging.getLogger(__name__)

def find_free_port(start_port: int, max_attempts: int = 10) -> int:
    """Find a free port starting from start_port"""
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")

class MessageQueue:
    """Message queue for handling offline clients"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.queues: Dict[str, deque] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
    def add_message(self, client_id: str, message: Dict[str, Any]):
        """Add a message to a client's queue"""
        if client_id not in self.queues:
            self.queues[client_id] = deque(maxlen=self.max_size)
        
        message['queued_at'] = time.time()
        self.queues[client_id].append(message)
    
    def get_messages(self, client_id: str) -> List[Dict[str, Any]]:
        """Get and clear messages for a client"""
        if client_id not in self.queues:
            return []
        
        current_time = time.time()
        messages = []
        
        # Filter out expired messages
        while self.queues[client_id]:
            msg = self.queues[client_id].popleft()
            if current_time - msg['queued_at'] < self.ttl_seconds:
                messages.append(msg)
        
        # Clear the queue
        if client_id in self.queues:
            del self.queues[client_id]
        
        return messages

class ClientConnection:
    """Represents a client connection with metadata"""
    
    def __init__(self, websocket: WebSocketServerProtocol, client_id: str):
        self.websocket = websocket
        self.client_id = client_id
        self.connected_at = time.time()
        self.last_ping = time.time()
        self.last_pong = time.time()
        self.reconnect_count = 0
        self.user_agent = websocket.request_headers.get('User-Agent', 'Unknown')
        
    def update_ping(self):
        """Update last ping time"""
        self.last_ping = time.time()
    
    def update_pong(self):
        """Update last pong time"""
        self.last_pong = time.time()
    
    def is_alive(self, timeout: float = 60.0) -> bool:
        """Check if connection is still alive"""
        return (time.time() - self.last_pong) < timeout

class ConsciousnessWebSocketServer:
    """WebSocket server that broadcasts real consciousness state"""
    
    def __init__(self, claude_interface, host='localhost', port=8765):
        self.claude_interface = claude_interface
        self.host = host
        self.port = port
        self.connections: Dict[str, ClientConnection] = {}
        self.message_queue = MessageQueue()
        self.update_interval = 0.5  # Send updates every 500ms
        self.ping_interval = 30.0  # Ping every 30 seconds
        self.connection_timeout = 60.0  # Consider connection dead after 60s
        
        # Rate limiting
        self.rate_limits = {}
        self.max_messages_per_minute = 60
        
        # Set callback in claude_interface
        self.claude_interface.set_state_update_callback(self.handle_state_update)
        
    async def handle_state_update(self, update_type: str, data: Dict[str, Any]):
        """Handle state updates from consciousness system"""
        message = {
            'type': update_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Broadcast to all connected clients
        await self.broadcast_update(update_type, data)
        
    async def register_client(self, websocket: WebSocketServerProtocol, client_id: Optional[str] = None):
        """Register a new WebSocket client with reconnection support"""
        # Generate client ID if not provided
        if not client_id:
            client_id = f"client_{int(time.time() * 1000)}_{websocket.remote_address[0]}"
        
        # Check if this is a reconnection
        if client_id in self.connections:
            old_connection = self.connections[client_id]
            old_connection.reconnect_count += 1
            logger.info(f"Client {client_id} reconnected (count: {old_connection.reconnect_count})")
            
            # Close old connection if still open
            try:
                await old_connection.websocket.close()
            except:
                pass
        
        # Create new connection
        connection = ClientConnection(websocket, client_id)
        self.connections[client_id] = connection
        
        logger.info(f"Client {client_id} connected from {websocket.remote_address}")
        
        # Send initial state
        await self.send_current_state(connection)
        
        # Send any queued messages
        queued_messages = self.message_queue.get_messages(client_id)
        for msg in queued_messages:
            try:
                await websocket.send(json.dumps(msg))
            except:
                pass
        
        return client_id
    
    async def unregister_client(self, client_id: str):
        """Remove a WebSocket client"""
        if client_id in self.connections:
            connection = self.connections[client_id]
            logger.info(f"Client {client_id} disconnected after {time.time() - connection.connected_at:.1f}s")
            del self.connections[client_id]
    
    async def send_current_state(self, connection: ClientConnection):
        """Send current consciousness state to a client"""
        try:
            # Get real state from consciousness system
            consciousness_state = self.claude_interface.consciousness.get_state_summary()
            emotion_label, emotion_confidence = self.claude_interface.emotions.get_closest_emotion_label()
            
            state_data = {
                'type': 'consciousness_update',
                'timestamp': datetime.now().isoformat(),
                'client_id': connection.client_id,
                'data': {
                    'coherence': consciousness_state['coherence'],
                    'attention_focus': consciousness_state['attention'],
                    'emotional_state': consciousness_state['emotion'],
                    'emotion_label': emotion_label,
                    'emotion_confidence': emotion_confidence,
                    'personality_traits': consciousness_state['personality'],
                    'working_memory': consciousness_state['recent_memories'],
                    'working_memory_size': len(consciousness_state['recent_memories']),
                    'interaction_count': consciousness_state['interaction_count'],
                    'goals': consciousness_state['goals']
                }
            }
            
            await connection.websocket.send(json.dumps(state_data))
        except ConnectionClosed:
            logger.debug(f"Connection closed while sending state to {connection.client_id}")
        except Exception as e:
            logger.error(f"Error sending state update: {e}")
    
    async def broadcast_update(self, update_type: str, data: Dict[str, Any], exclude_client: Optional[str] = None):
        """Broadcast an update to all connected clients"""
        message = {
            'type': update_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # Send to connected clients
        disconnected_clients = []
        for client_id, connection in self.connections.items():
            if client_id == exclude_client:
                continue
            
            try:
                await connection.websocket.send(json.dumps(message))
            except (ConnectionClosed, ConnectionClosedError):
                disconnected_clients.append(client_id)
                # Queue message for disconnected client
                self.message_queue.add_message(client_id, message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.unregister_client(client_id)
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a WebSocket client connection"""
        client_id = None
        
        try:
            # Check for client ID in query parameters (for reconnection)
            query_params = {}
            if '?' in path:
                query_string = path.split('?')[1]
                for param in query_string.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        query_params[key] = value
            
            client_id = query_params.get('client_id')
            
            # Register client
            client_id = await self.register_client(websocket, client_id)
            
            # Send client ID for future reconnections
            await websocket.send(json.dumps({
                'type': 'connection_established',
                'client_id': client_id
            }))
            
            # Handle messages
            async for message in websocket:
                if not await self.check_rate_limit(client_id):
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Rate limit exceeded'
                    }))
                    continue
                
                # Handle both string and bytes messages
                if isinstance(message, bytes):
                    message_str = message.decode('utf-8')
                else:
                    message_str = str(message)

                await self.handle_client_message(client_id, message_str)
        
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"Client {client_id} connection closed normally")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            if client_id:
                await self.unregister_client(client_id)
    
    async def handle_client_message(self, client_id: str, message: str):
        """Handle incoming client messages"""
        connection = self.connections.get(client_id)
        if not connection:
            return
        
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'ping':
                # Handle ping/pong for connection health
                connection.update_ping()
                await connection.websocket.send(json.dumps({
                    'type': 'pong',
                    'timestamp': time.time()
                }))
                connection.update_pong()
            
            elif message_type == 'get_state':
                await self.send_current_state(connection)
            
            elif message_type == 'get_memory_stats':
                stats = self.claude_interface.memory_manager.get_statistics()
                await connection.websocket.send(json.dumps({
                    'type': 'memory_stats',
                    'data': stats
                }))
            
            elif message_type == 'get_emotion_trajectory':
                trajectory = self.claude_interface.emotions.get_state_trajectory()
                await connection.websocket.send(json.dumps({
                    'type': 'emotion_trajectory',
                    'data': [{'pleasure': s.pleasure, 
                             'arousal': s.arousal, 
                             'dominance': s.dominance} for s in trajectory]
                }))
            
            elif message_type == 'get_personality':
                traits = self.claude_interface.personality.profile.get_traits()
                modifiers = self.claude_interface.personality.get_behavioral_modifiers()
                await connection.websocket.send(json.dumps({
                    'type': 'personality_update',
                    'data': {
                        'traits': traits,
                        'modifiers': modifiers
                    }
                }))
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received from {client_id}: {message}")
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = time.time()
        
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = deque()
        
        # Remove old entries
        while self.rate_limits[client_id] and self.rate_limits[client_id][0] < current_time - 60:
            self.rate_limits[client_id].popleft()
        
        # Check limit
        if len(self.rate_limits[client_id]) >= self.max_messages_per_minute:
            return False
        
        # Add current request
        self.rate_limits[client_id].append(current_time)
        return True
    
    async def periodic_update_loop(self):
        """Send periodic updates to all clients"""
        while True:
            try:
                # Only send if we have clients
                if self.connections:
                    # Get current state from consciousness system
                    consciousness_state = self.claude_interface.consciousness.get_state_summary()
                    emotion_label, emotion_confidence = self.claude_interface.emotions.get_closest_emotion_label()
                    
                    await self.broadcast_update('consciousness_update', {
                        'coherence': consciousness_state['coherence'],
                        'attention_focus': consciousness_state['attention'],
                        'emotional_state': consciousness_state['emotion'],
                        'emotion_label': emotion_label,
                        'emotion_confidence': emotion_confidence,
                        'working_memory_size': len(consciousness_state['recent_memories']),
                        'interaction_count': consciousness_state['interaction_count']
                    })
                
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in periodic update loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    async def connection_health_monitor(self):
        """Monitor connection health and clean up dead connections"""
        while True:
            try:
                current_time = time.time()
                dead_connections = []
                
                for client_id, connection in self.connections.items():
                    # Send ping to check connection
                    try:
                        await connection.websocket.ping()
                        connection.update_ping()
                    except:
                        dead_connections.append(client_id)
                        continue
                    
                    # Check if connection is alive
                    if not connection.is_alive(self.connection_timeout):
                        dead_connections.append(client_id)
                
                # Clean up dead connections
                for client_id in dead_connections:
                    logger.info(f"Removing dead connection: {client_id}")
                    await self.unregister_client(client_id)
                
                await asyncio.sleep(self.ping_interval)
            except Exception as e:
                logger.error(f"Error in connection health monitor: {e}")
                await asyncio.sleep(30)
    
    async def start(self):
        """Start the WebSocket server with all background tasks"""
        # Find available port if default is in use
        port = self.port
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Try to start server
                server = await websockets.serve(
                    self.handle_client, 
                    self.host, 
                    port,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.connection_timeout
                )
                break
            except OSError as e:
                if "address already in use" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Port {port} is already in use, finding alternative...")
                    port = find_free_port(port + 1)
                    self.port = port
                    logger.info(f"Using alternative port: {port}")
                else:
                    raise
        
        # Start background tasks
        update_task = asyncio.create_task(self.periodic_update_loop())
        health_task = asyncio.create_task(self.connection_health_monitor())
        
        logger.info(f"WebSocket server started on ws://{self.host}:{port}")
        
        # Keep server running
        await asyncio.Future()