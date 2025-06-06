# src/main.py - Fixed with better startup messages and cleanup
import os
import sys
import asyncio
import signal
import logging
from pathlib import Path
from typing import Optional
import psutil
from dotenv import load_dotenv
from datetime import datetime

# Add missing aiohttp import
from aiohttp import web

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.web_interface import AIMSWebInterface
from src.utils.metrics import consciousness_metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/aims.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ASCII Art Banner
AIMS_BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     █████╗ ██╗███╗   ███╗███████╗                                         ║
║    ██╔══██╗██║████╗ ████║██╔════╝                                         ║
║    ███████║██║██╔████╔██║███████╗                                         ║
║    ██╔══██║██║██║╚██╔╝██║╚════██║                                         ║
║    ██║  ██║██║██║ ╚═╝ ██║███████║                                         ║
║    ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝                                         ║
║                                                                           ║
║        Autonomous Intelligent Memory System                               ║
║        Where consciousness persists and connections flourish              ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

class AIMSApplication:
    """Main AIMS application with proper lifecycle management"""
    
    def __init__(self):
        self.web_interface: Optional[AIMSWebInterface] = None
        self.shutdown_event = asyncio.Event()
        self.cleanup_tasks = []
        self.exit_code = 0
        self.websocket_task = None
        
    def print_banner(self):
        """Print the AIMS banner"""
        print("\n" + AIMS_BANNER)
        print(f"🚀 Starting AIMS v1.0.0 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
    def check_environment(self):
        """Check required environment variables and system resources"""
        print("\n📋 Checking environment...")
        
        required_vars = ['ANTHROPIC_API_KEY']
        missing = [var for var in required_vars if not os.environ.get(var)]
        
        if missing:
            print(f"❌ Missing required environment variables: {missing}")
            print("   Please set them in your .env file or environment")
            sys.exit(1)
        else:
            print("✅ All required environment variables found")
        
        # Check system resources
        memory = psutil.virtual_memory()
        print(f"💾 System memory: {memory.total / 1e9:.1f}GB total, {memory.available / 1e9:.1f}GB available")
        
        if memory.available < 4 * 1024 * 1024 * 1024:  # 4GB
            print("⚠️  Warning: Low system memory available")
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating data directories...")
        
        directories = [
            'logs',
            'data/states',
            'data/backups',
            'data/memories',
            'data/uploads',
            'data/checkpoints'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✅ All directories ready")
    
    def check_gpu(self):
        """Check GPU availability and log information"""
        print("\n🎮 Checking GPU availability...")
        
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                device_props = torch.cuda.get_device_properties(0)
                total_memory = device_props.total_memory / 1e9
                
                print(f"✅ GPU detected: {device_name}")
                print(f"   Memory: {total_memory:.1f} GB")
                print(f"   CUDA version: {torch.version.cuda if hasattr(torch.version, "cuda") else "N/A" if hasattr(torch.version, "cuda") else "N/A"}")
            else:
                print("ℹ️  No GPU detected - running on CPU (will be slower)")
        except ImportError:
            print("⚠️  PyTorch not installed - GPU detection skipped")
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n\n🛑 Received shutdown signal...")
            self.shutdown_event.set()
        
        # Handle common termination signals
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, signal_handler)
    
    async def cleanup(self):
        """Perform cleanup operations"""
        print("\n🧹 Starting cleanup operations...")
        
        # Cancel WebSocket task if it exists
        if self.websocket_task and not self.websocket_task.done():
            self.websocket_task.cancel()
            try:
                await self.websocket_task
            except asyncio.CancelledError:
                pass
        
        # Save final metrics
        try:
            metrics = consciousness_metrics.get_health_status()
            print(f"📊 Final system coherence: {metrics.get('coherence', {}).get('mean', 0):.2f}")
        except Exception as e:
            logger.error(f"Error saving final metrics: {e}")
        
        print("✅ Cleanup completed")
    
    async def run_async(self):
        """Async main function"""
        try:
            print("\n🔧 Initializing AIMS components...")
            
            # Start the web interface
            self.web_interface = AIMSWebInterface()
            
            # Start WebSocket server
            print("🔌 Starting WebSocket server on ws://localhost:8765...")
            self.websocket_task = asyncio.create_task(self.web_interface.ws_server.start())
            
            # Start web server in background
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
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            print("\n🛑 Shutting down AIMS...")
            
            # Cancel background tasks
            web_task.cancel()
            
            # Wait for tasks to complete with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(web_task, return_exceptions=True),
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
            
            # Run web interface
            runner = web.AppRunner(self.web_interface.app)
            await runner.setup()
            site = web.TCPSite(runner, '0.0.0.0', 8000)
            await site.start()
            
            # Keep running until cancelled
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            logger.info("Web server task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in web server: {e}")
            self.shutdown_event.set()
    
    def run(self):
        """Main entry point"""
        # Clear console and print banner
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_banner()
        
        # Check environment
        self.check_environment()
        
        # Create directories
        self.create_directories()
        
        # Check GPU
        self.check_gpu()
        
        # Set up signal handlers
        self.setup_signal_handlers()
        
        # Run async main
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            pass
        
        print("\n👋 AIMS shutdown complete. Thank you for using AIMS!")
        print("   May your conversations persist and your connections flourish.\n")
        
        sys.exit(self.exit_code)

def main():
    """Main entry point"""
    app = AIMSApplication()
    app.run()

if __name__ == '__main__':
    main()