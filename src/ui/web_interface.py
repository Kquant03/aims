"""Web Interface for AIMS with WebSocket support"""
import os
import aiohttp
from aiohttp import web
import aiohttp_cors
import logging
from ..api.websocket_server import ConsciousnessWebSocketServer

logger = logging.getLogger(__name__)

class AIMSWebInterface:
    """Web interface with WebSocket server"""
    
    def __init__(self, claude_interface=None):
        self.app = web.Application()
        self.claude_interface = claude_interface
        
        # Initialize WebSocket server
        if claude_interface:
            self.ws_server = ConsciousnessWebSocketServer(
                claude_interface, 
                host='localhost', 
                port=8765
            )
        else:
            self.ws_server = None
            
        self.setup_routes()
        self.setup_cors()
    
    def setup_routes(self):
        """Setup web routes"""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/api/status', self.status)
        self.app.router.add_get('/api/session', self.get_session)
        self.app.router.add_static('/', path='src/ui/static', name='static')
    
    def setup_cors(self):
        """Setup CORS"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def index(self, request):
        """Serve index page"""
        html_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
        if os.path.exists(html_path):
            with open(html_path, 'r') as f:
                return web.Response(text=f.read(), content_type='text/html')
        else:
            # Fallback HTML
            return web.Response(text='''
            <!DOCTYPE html>
            <html>
            <head>
                <title>AIMS - Autonomous Intelligent Memory System</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <link rel="stylesheet" href="/css/main.css">
            </head>
            <body>
                <div id="root"></div>
                <script src="/js/bundle.js"></script>
            </body>
            </html>
            ''', content_type='text/html')
    
    async def status(self, request):
        """Get system status"""
        status = {
            'online': True,
            'websocket_url': 'ws://localhost:8765',
            'version': '1.0.0'
        }
        
        if self.claude_interface:
            status['consciousness'] = {
                'coherence': self.claude_interface.consciousness.state.global_coherence,
                'interaction_count': self.claude_interface.consciousness.state.interaction_count
            }
        
        return web.json_response(status)
    
    async def get_session(self, request):
        """Get or create session"""
        session_id = request.cookies.get('session_id', f'session_{os.urandom(8).hex()}')
        response = web.json_response({'session_id': session_id})
        response.set_cookie('session_id', session_id, max_age=86400)  # 24 hours
        return response
    
    async def start(self, host='localhost', port=8000):
        """Start web server"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        logger.info(f"Web interface started on http://{host}:{port}")
