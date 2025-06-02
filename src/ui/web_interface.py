# web_interface.py - Main Web Application Interface
import os
import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional
import aiofiles
from aiohttp import web
import aiohttp_cors
from aiohttp_session import setup, get_session
from aiohttp_session.cookie_storage import EncryptedCookieStorage
import jinja2
import aiohttp_jinja2
from pathlib import Path
import logging

from src.api.claude_interface import ClaudeConsciousnessInterface
from src.api.websocket_server import ConsciousnessWebSocketServer
from src.api.state_manager import StateManager

logger = logging.getLogger(__name__)

class AIMSWebInterface:
    """Main web interface for AIMS"""
    
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        self.config = self._load_config(config_path)
        self.app = web.Application()
        self.setup_app()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_app(self):
        """Set up the web application"""
        # Initialize Claude interface
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.claude_interface = ClaudeConsciousnessInterface(api_key, self.config)
        self.state_manager = StateManager(self.config.get('state_management', {}))
        
        # Set up WebSocket server
        self.ws_server = ConsciousnessWebSocketServer(
            self.claude_interface,
            host='0.0.0.0',
            port=8765
        )
        
        # Set up session middleware
        secret_key = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')
        setup(self.app, EncryptedCookieStorage(secret_key))
        
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
        self.app.router.add_get('/api/memory/stats', self.memory_stats_handler)
        self.app.router.add_post('/api/state/save', self.save_state_handler)
        self.app.router.add_post('/api/state/load', self.load_state_handler)
        self.app.router.add_get('/api/state/list', self.list_states_handler)
        
        # File upload
        self.app.router.add_post('/api/upload', self.upload_handler)
        
        # Static files
        static_path = Path(__file__).parent / 'static'
        self.app.router.add_static('/static', static_path, name='static')
    
    async def on_startup(self, app):
        """Initialize services on startup"""
        logger.info("Starting AIMS Web Interface")
        
        # Start WebSocket server in background
        asyncio.create_task(self.ws_server.start())
        
        # Start automatic backup loop
        asyncio.create_task(
            self.state_manager.automatic_backup_loop(self.claude_interface)
        )
        
        logger.info("AIMS Web Interface started successfully")
    
    async def on_cleanup(self, app):
        """Cleanup on shutdown"""
        logger.info("Shutting down AIMS Web Interface")
        await self.claude_interface.shutdown()
    
    @aiohttp_jinja2.template('index.html')
    async def index_handler(self, request):
        """Main chat interface"""
        session = await get_session(request)
        
        # Create or get user ID
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        
        return {
            'user_id': session['user_id'],
            'ws_url': f"ws://{request.host}/ws"
        }
    
    async def chat_handler(self, request):
        """Handle chat messages"""
        try:
            data = await request.json()
            message = data.get('message', '').strip()
            
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
            
            # Process message and stream response
            response_text = ""
            async for chunk in self.claude_interface.process_message(
                session_id, message, stream=True
            ):
                response_text += chunk
            
            return web.json_response({
                'response': response_text,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in chat handler: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def session_handler(self, request):
        """Get current session info"""
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
        state_summary['emotion'] = {
            'label': emotion_label,
            'confidence': confidence,
            'color': self.claude_interface.emotions.get_emotional_color()
        }
        
        return web.json_response(state_summary)
    
    async def memory_stats_handler(self, request):
        """Get memory statistics"""
        stats = self.claude_interface.memory_manager.get_statistics()
        return web.json_response(stats)
    
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


# Create the HTML template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIMS - Claude Consciousness Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            height: 100vh;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            height: 100vh;
        }
        
        /* Consciousness Panel */
        .consciousness-panel {
            width: 300px;
            background: #1a1a1a;
            border-right: 1px solid #333;
            padding: 20px;
            overflow-y: auto;
        }
        
        .metric {
            margin-bottom: 20px;
        }
        
        .metric-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 300;
        }
        
        .progress-bar {
            width: 100%;
            height: 4px;
            background: #333;
            border-radius: 2px;
            overflow: hidden;
            margin-top: 5px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00ff88, #0088ff);
            transition: width 0.3s ease;
        }
        
        /* Chat Area */
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0f0f0f;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message-user {
            text-align: right;
        }
        
        .message-content {
            display: inline-block;
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 16px;
            background: #1a1a1a;
            border: 1px solid #333;
        }
        
        .message-user .message-content {
            background: #0066cc;
            border-color: #0066cc;
        }
        
        /* Input Area */
        .input-area {
            padding: 20px;
            border-top: 1px solid #333;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
        }
        
        .message-input {
            flex: 1;
            background: #1a1a1a;
            border: 1px solid #333;
            color: #e0e0e0;
            padding: 12px 16px;
            border-radius: 24px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .message-input:focus {
            border-color: #0066cc;
        }
        
        .send-button {
            background: #0066cc;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 24px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }
        
        .send-button:hover {
            background: #0052a3;
        }
        
        .send-button:disabled {
            background: #333;
            cursor: not-allowed;
        }
        
        /* Emotional State Indicator */
        .emotion-indicator {
            width: 100%;
            height: 60px;
            background: #111;
            border-radius: 8px;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        
        .emotion-gradient {
            position: absolute;
            inset: 0;
            opacity: 0.8;
            transition: background 1s ease;
        }
        
        .emotion-label {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 14px;
            font-weight: 500;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Consciousness Monitoring Panel -->
        <div class="consciousness-panel">
            <h2 style="margin-bottom: 20px; font-weight: 300;">Consciousness State</h2>
            
            <div class="emotion-indicator">
                <div class="emotion-gradient" id="emotionGradient"></div>
                <div class="emotion-label" id="emotionLabel">Neutral</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Coherence</div>
                <div class="metric-value" id="coherenceValue">1.00</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="coherenceBar" style="width: 100%"></div>
                </div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Attention Focus</div>
                <div class="metric-value" id="attentionFocus" style="font-size: 14px;">Initialization</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Working Memory</div>
                <div class="metric-value" id="workingMemory">0</div>
                <div class="metric-label">items</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Interactions</div>
                <div class="metric-value" id="interactionCount">0</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Total Memories</div>
                <div class="metric-value" id="totalMemories">0</div>
            </div>
        </div>
        
        <!-- Chat Interface -->
        <div class="chat-area">
            <div class="messages" id="messages"></div>
            
            <div class="input-area">
                <div class="input-container">
                    <input 
                        type="text" 
                        class="message-input" 
                        id="messageInput" 
                        placeholder="Type your message..."
                        autocomplete="off"
                    >
                    <button class="send-button" id="sendButton">Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection
        let ws = null;
        const wsUrl = "{{ ws_url }}";
        
        // Connect to WebSocket
        function connectWebSocket() {
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                console.log('Connected to consciousness stream');
                ws.send(JSON.stringify({ type: 'get_state' }));
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateConsciousnessDisplay(data);
            };
            
            ws.onclose = () => {
                console.log('Disconnected from consciousness stream');
                setTimeout(connectWebSocket, 3000); // Reconnect after 3 seconds
            };
        }
        
        // Update consciousness display
        function updateConsciousnessDisplay(data) {
            if (data.type === 'consciousness_update') {
                const state = data.data;
                
                // Update coherence
                document.getElementById('coherenceValue').textContent = state.coherence.toFixed(2);
                document.getElementById('coherenceBar').style.width = `${state.coherence * 100}%`;
                
                // Update emotion
                document.getElementById('emotionLabel').textContent = state.emotion_label;
                const emotionColor = `rgb(${Math.floor(state.emotional_state.pleasure * 255)}, 
                                         ${Math.floor((1 - state.emotional_state.arousal) * 255)}, 
                                         ${Math.floor(state.emotional_state.dominance * 255)})`;
                document.getElementById('emotionGradient').style.background = 
                    `radial-gradient(circle, ${emotionColor}, transparent)`;
                
                // Update other metrics
                document.getElementById('attentionFocus').textContent = state.attention_focus;
                document.getElementById('workingMemory').textContent = state.working_memory_size || 0;
                document.getElementById('interactionCount').textContent = state.interaction_count;
            } else if (data.type === 'memory_stats') {
                document.getElementById('totalMemories').textContent = data.data.total_memories;
            }
        }
        
        // Chat functionality
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const messagesContainer = document.getElementById('messages');
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Disable input
            messageInput.disabled = true;
            sendButton.disabled = true;
            
            // Add user message to chat
            addMessage(message, 'user');
            
            // Clear input
            messageInput.value = '';
            
            try {
                // Send to API
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message }),
                });
                
                const data = await response.json();
                
                if (data.response) {
                    addMessage(data.response, 'assistant');
                }
                
                // Request updated stats
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'get_state' }));
                    ws.send(JSON.stringify({ type: 'get_memory_stats' }));
                }
                
            } catch (error) {
                console.error('Error sending message:', error);
                addMessage('Sorry, there was an error processing your message.', 'assistant');
            } finally {
                // Re-enable input
                messageInput.disabled = false;
                sendButton.disabled = false;
                messageInput.focus();
            }
        }
        
        function addMessage(content, sender) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message message-${sender}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            
            messageDiv.appendChild(contentDiv);
            messagesContainer.appendChild(messageDiv);
            
            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Initialize
        connectWebSocket();
        messageInput.focus();
    </script>
</body>
</html>'''

# Save the template
def save_template():
    template_dir = Path(__file__).parent / 'templates'
    template_dir.mkdir(exist_ok=True)
    
    with open(template_dir / 'index.html', 'w') as f:
        f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    # Save the template
    save_template()
    
    # Run the application
    app = AIMSWebInterface()
    app.run()