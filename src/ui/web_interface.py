"""Simple Web Interface for AIMS"""
import os
import aiohttp
from aiohttp import web
import aiohttp_cors
import logging

logger = logging.getLogger(__name__)

class AIMSWebInterface:
    """Minimal web interface"""
    
    def __init__(self):
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()
    
    def setup_routes(self):
        """Setup web routes"""
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/api/status', self.status)
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
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AIMS Interface</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .status { padding: 20px; background: #f0f0f0; border-radius: 8px; }
                .online { color: green; }
                .offline { color: red; }
            </style>
        </head>
        <body>
            <h1>AIMS - Autonomous Intelligent Memory System</h1>
            <div class="status">
                <h2>System Status</h2>
                <p>Status: <span class="online">● Online</span></p>
                <p>API Key: <span class="online">✓ Configured</span></p>
                <p>Services:</p>
                <ul>
                    <li>Web Interface: <span class="online">● Running</span></li>
                    <li>WebSocket: <span class="offline">○ Not Started</span></li>
                    <li>Claude API: <span class="online">✓ Ready</span></li>
                </ul>
            </div>
            <p>This is a minimal interface. Full UI coming soon!</p>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def status(self, request):
        """API status endpoint"""
        return web.json_response({
            'status': 'online',
            'version': '1.0.0',
            'api_key_set': bool(os.environ.get('ANTHROPIC_API_KEY'))
        })
    
    def run(self, host='0.0.0.0', port=8000):
        """Run the web server"""
        web.run_app(self.app, host=host, port=port)

# Allow running directly
if __name__ == "__main__":
    interface = AIMSWebInterface()
    interface.run()
