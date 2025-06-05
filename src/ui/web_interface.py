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


if __name__ == '__main__':
    # Save the template
    
    # Run the application
    app = AIMSWebInterface()
    app.run()