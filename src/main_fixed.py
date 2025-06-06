import os
import sys
import asyncio
import logging
from datetime import datetime
from aiohttp import web
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.claude_interface import ClaudeConsciousnessInterface
from src.ui.web_interface import AIMSWebInterface
from src.api.websocket_server import ConsciousnessWebSocketServer

logger = logging.getLogger(__name__)

AIMS_BANNER = """
    ___    ________  _______  
   /   |  /  _/  /  |/  / __/  
  / /| |  / // /|_/ /\ \/ /___ 
 / ___ |_/ // /  / /___/  ___/ 
/_/  |_/___/_/  /_//____/____/  
                                
Autonomous Intelligent Memory System
"""

class AIMSApplication:
    """Main AIMS application with proper lifecycle management"""
    
    def __init__(self):
        self.claude_interface: Optional[ClaudeConsciousnessInterface] = None
        self.web_interface: Optional[AIMSWebInterface] = None
        self.websocket_server: Optional[ConsciousnessWebSocketServer] = None
        self.shutdown_event = asyncio.Event()
        self.cleanup_tasks = []
        self.exit_code = 0
        
    def print_banner(self):
        """Print the AIMS banner"""
        print("\n" + AIMS_BANNER)
        print(f"🚀 Starting AIMS v1.0.0 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    def check_environment(self):
        """Check required environment variables and system resources"""
        print("\n📋 Checking environment...")
        
        required_vars = ['ANTHROPIC_API_KEY']
        missing = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing.append(var)
        
        if missing:
            print(f"❌ Missing required environment variables: {', '.join(missing)}")
            print("   Please set them in your .env file or environment")
            sys.exit(1)
        
        print("✅ All required environment variables found")
        
        # Check memory
        import psutil
        mem = psutil.virtual_memory()
        print(f"💾 System memory: {mem.total / (1024**3):.1f}GB total, {mem.available / (1024**3):.1f}GB available")
        
    def setup_directories(self):
        """Create necessary directories"""
        print("📁 Creating data directories...")
        dirs = [
            'data/artifacts',
            'data/backups', 
            'data/chroma',
            'data/memories',
            'data/states',
            'data/uploads',
            'logs'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        print("✅ All directories ready")
        
    def check_gpu(self):
        """Check GPU availability"""
        print("🎮 Checking GPU availability...")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
                
                # Check CUDA version
                cuda_version = torch.version.cuda
                print(f"   CUDA version: {cuda_version}")
            else:
                print("ℹ️  No GPU detected, using CPU")
        except ImportError:
            print("ℹ️  PyTorch not installed, GPU features disabled")
            
    async def initialize_consciousness(self):
        """Initialize the consciousness system"""
        print("\n🧠 Initializing consciousness system...")
        
        # Load configuration
        config = {
            'api': {
                'claude': {
                    'model': 'claude-3-sonnet-20240229',
                    'max_tokens': 4096,
                    'temperature_base': 0.7
                }
            },
            'consciousness': {
                'cycle_frequency': 1.0,
                'memory_buffer_size': 10
            },
            'memory': {
                'redis_url': 'redis://localhost:6379',
                'postgres_url': 'postgresql+asyncpg://aims:aims_password@localhost:5432/aims_memory',
                'chroma_persist_dir': './data/chroma'
            }
        }
        
        # Initialize Claude interface with consciousness
        api_key = os.getenv('ANTHROPIC_API_KEY')
        self.claude_interface = ClaudeConsciousnessInterface(api_key, config)
        
        print("✅ Consciousness core initialized")
        
    async def run_async(self):
        """Async main function"""
        try:
            print("\n🔧 Initializing AIMS components...")
            
            # Initialize consciousness first
            await self.initialize_consciousness()
            
            # Create web interface with claude_interface
            self.web_interface = AIMSWebInterface(self.claude_interface)
            
            # The WebSocket server is now part of web_interface
            self.websocket_server = self.web_interface.ws_server
            
            # Start WebSocket server
            print("🔌 Starting WebSocket server on ws://localhost:8765...")
            websocket_task = asyncio.create_task(self.websocket_server.start())
            
            # Start web server
            print("🌐 Starting web interface on http://localhost:8000...")
            web_task = asyncio.create_task(self.run_web_server())
            
            # Give servers a moment to start
            await asyncio.sleep(1)
            
            print("\n" + "="*80)
            print("✨ AIMS is ready!")
            print("="*80)
            print("\n📍 Access points:")
            print("   Web Interface:  http://localhost:8000")
            print("   WebSocket:      ws://localhost:8765")
            print("   API Endpoint:   http://localhost:8000/api/")
            print("\n💡 Press Ctrl+C to shut down gracefully")
            print("="*80 + "\n")
            
            # Initial consciousness state
            print("📊 Initial consciousness state:")
            state = self.claude_interface.consciousness.get_state_summary()
            print(f"   Coherence: {state['coherence']:.2f}")
            print(f"   Emotion: {state['emotion']}")
            print(f"   Attention: {state['attention']}")
            print()
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            print("\n🛑 Shutting down AIMS...")
            
            # Shutdown consciousness system
            await self.claude_interface.shutdown()
            
            # Cancel tasks
            websocket_task.cancel()
            web_task.cancel()
            
            # Wait for tasks to complete
            try:
                await asyncio.wait_for(
                    asyncio.gather(websocket_task, web_task, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some tasks did not complete within timeout")
            
        except KeyboardInterrupt:
            print("\n⌨️  Keyboard interrupt received")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            logger.error(f"Fatal error: {e}", exc_info=True)
            self.exit_code = 1
        finally:
            await self.cleanup()
            
    async def run_web_server(self):
        """Run the web server"""
        try:
            if not self.web_interface:
                logger.error("Web interface not initialized")
                self.shutdown_event.set()
                return
            
            # Start web interface
            await self.web_interface.start('0.0.0.0', 8000)
            
            # Keep running until cancelled
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Web server task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in web server: {e}")
            self.shutdown_event.set()
            
    async def cleanup(self):
        """Cleanup operations"""
        print("\n🧹 Starting cleanup operations...")
        
        # Get final coherence
        if self.claude_interface:
            state = self.claude_interface.consciousness.get_state_summary()
            print(f"📊 Final system coherence: {state['coherence']:.2f}")
        
        print("✅ Cleanup completed")
        
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        print("\n📡 Shutdown signal received")
        self.shutdown_event.set()
        
    def run(self):
        """Main entry point"""
        # Clear console and print banner
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_banner()
        
        # Check environment
        self.check_environment()
        
        # Create directories
        self.setup_directories()
        
        # Check GPU
        self.check_gpu()
        
        # Setup signal handlers
        import signal
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
        # Run async main
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            pass
        
        print("\n👋 AIMS shutdown complete. Thank you for using AIMS!")
        print("   May your conversations persist and your connections flourish.\n")
        
        sys.exit(self.exit_code)


def main():
    """Entry point"""
    app = AIMSApplication()
    app.run()


if __name__ == "__main__":
    main()
