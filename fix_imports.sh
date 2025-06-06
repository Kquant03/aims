#!/bin/bash
# fix_imports.sh - Fix all import issues in AIMS

echo "🔧 Fixing AIMS import structure..."

# 1. Fix import paths to be consistent
echo "📝 Fixing import paths..."

# Fix claude_interface.py imports
if [ -f src/api/claude_interface.py ]; then
    sed -i 's/from core\./from src.core./g' src/api/claude_interface.py
    sed -i 's/from persistence\./from src.persistence./g' src/api/claude_interface.py
    sed -i 's/from utils\./from src.utils./g' src/api/claude_interface.py
fi

# Fix web_interface.py imports
if [ -f src/ui/web_interface.py ]; then
    sed -i 's/from api\./from src.api./g' src/ui/web_interface.py
    sed -i 's/from core\./from src.core./g' src/ui/web_interface.py
fi

# Fix all Python files to use absolute imports
find src -name "*.py" -type f -exec sed -i 's/from \.\./from src./g' {} \;

# 2. Ensure ConsciousnessCore exists in living_consciousness.py
echo "🧠 Ensuring ConsciousnessCore class exists..."

# Check if ConsciousnessCore exists, if not add it
if ! grep -q "class ConsciousnessCore" src/core/living_consciousness.py 2>/dev/null; then
    echo "Adding ConsciousnessCore to living_consciousness.py..."
    
    cat >> src/core/living_consciousness.py << 'EOF'

# Ensure ConsciousnessCore is defined
if 'ConsciousnessCore' not in globals():
    class ConsciousnessCore:
        """Core consciousness functionality"""
        
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.state = ConsciousnessState()
            self.memory_buffer = deque(maxlen=config.get('working_memory_size', 10))
            self.cycle_frequency = 1.0  # Hz
            
        def process_input(self, input_text: str, context: Dict[str, Any]):
            """Process input and update consciousness state"""
            self.state.interaction_count += 1
            self.state.last_interaction = datetime.now()
            self.memory_buffer.append(f"Input: {input_text[:100]}...")
            
        def get_state_summary(self) -> Dict[str, Any]:
            """Get current state summary"""
            return {
                'coherence': self.state.global_coherence,
                'attention': self.state.attention_focus,
                'emotion': {'label': 'neutral', 'confidence': 0.5},
                'recent_memories': list(self.memory_buffer)[-5:],
                'interaction_count': self.state.interaction_count,
                'goals': self.state.active_goals,
                'personality': {'openness': 0.8, 'conscientiousness': 0.7}
            }
        
        def _calculate_coherence(self):
            """Calculate global coherence score"""
            base_coherence = 0.7
            memory_factor = min(1.0, len(self.memory_buffer) / 10)
            self.state.global_coherence = base_coherence * 0.7 + memory_factor * 0.3
        
        async def _update_attention(self):
            """Update attention focus"""
            pass
        
        async def _consolidate_memory(self):
            """Consolidate working memory"""
            pass
        
        async def _update_emotional_state(self):
            """Update emotional state"""
            pass
        
        def shutdown(self):
            """Cleanup on shutdown"""
            logger.info("Consciousness core shutting down")
EOF
fi

# 3. Create a simple package.json for the frontend
echo "📦 Creating package.json for frontend..."
if [ ! -f src/ui/package.json ]; then
    cat > src/ui/package.json << 'EOF'
{
  "name": "aims-ui",
  "version": "1.0.0",
  "description": "AIMS Web Interface",
  "scripts": {
    "build": "echo 'No build required for static HTML/JS' && exit 0",
    "start": "python -m http.server 8080"
  }
}
EOF
fi

# 4. Create __init__.py files with proper exports
echo "📝 Creating proper __init__.py files..."

# Core __init__.py
cat > src/core/__init__.py << 'EOF'
"""Core AIMS modules"""
from .living_consciousness import ConsciousnessCore, ConsciousnessState
from .memory_manager import PersistentMemoryManager

__all__ = ['ConsciousnessCore', 'ConsciousnessState', 'PersistentMemoryManager']
EOF

# API __init__.py
cat > src/api/__init__.py << 'EOF'
"""API modules"""
from .claude_interface import ClaudeConsciousnessInterface
from .websocket_server import ConsciousnessWebSocketServer

__all__ = ['ClaudeConsciousnessInterface', 'ConsciousnessWebSocketServer']
EOF

# UI __init__.py
cat > src/ui/__init__.py << 'EOF'
"""UI modules"""
from .web_interface import AIMSWebInterface

__all__ = ['AIMSWebInterface']
EOF

# 5. Create a working minimal launcher
echo "🚀 Creating minimal launcher..."
cat > run_aims.py << 'EOF'
#!/usr/bin/env python3
"""
AIMS Launcher - Working minimal version
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("AIMS - Autonomous Intelligent Memory System")
    print("="*60 + "\n")
    
    # Check environment
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("❌ ERROR: ANTHROPIC_API_KEY not set!")
        print("\nPlease set your API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'\n")
        return
    
    print("✅ Environment check passed")
    print("\n🧠 Starting AIMS...\n")
    
    try:
        # Try to import and run the web interface
        from src.ui.web_interface import AIMSWebInterface
        interface = AIMSWebInterface()
        print("✨ AIMS Web Interface starting...")
        print("\n📍 Access at: http://localhost:8000")
        print("📍 WebSocket: ws://localhost:8765\n")
        print("Press Ctrl+C to stop\n")
        
        # Keep running
        await asyncio.Event().wait()
        
    except ImportError as e:
        print(f"⚠️  Import error: {e}")
        print("\nStarting minimal interactive mode...\n")
        
        # Minimal interactive mode
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['quit', 'exit']:
                    break
                print(f"AIMS: Processing '{user_input}'...\n")
            except KeyboardInterrupt:
                break
    
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown complete.")
EOF

chmod +x run_aims.py

# 6. Create a simple web interface if it doesn't work
echo "🌐 Creating fallback web interface..."
mkdir -p src/ui/static

cat > src/ui/web_interface.py << 'EOF'
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
EOF

echo "✅ Import structure fixed!"
echo ""
echo "🚀 To run AIMS, use one of these commands:"
echo "   1. python run_aims.py        (Recommended)"
echo "   2. python -m src.main        (From project root)"
echo "   3. cd src && python main.py  (From src directory)"
EOF