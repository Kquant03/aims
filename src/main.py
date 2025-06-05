# src/main.py - Fixed with proper imports
import os
import sys
import asyncio
import signal
import logging
from pathlib import Path
from typing import Optional
import psutil
from dotenv import load_dotenv

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

class AIMSApplication:
    """Main AIMS application with proper lifecycle management"""
    
    def __init__(self):
        self.web_interface: Optional[AIMSWebInterface] = None
        self.shutdown_event = asyncio.Event()
        self.cleanup_tasks = []
        self.exit_code = 0
        
    def check_environment(self):
        """Check required environment variables and system resources"""
        required_vars = ['ANTHROPIC_API_KEY']
        missing = [var for var in required_vars if not os.environ.get(var)]
        
        if missing:
            logger.error(f"Missing required environment variables: {missing}")
            logger.error("Please set them in your .env file or environment")
            sys.exit(1)
        
        # Check system resources
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024 * 1024 * 1024:  # 4GB
            logger.warning(f"Low system memory: {memory.available / 1e9:.1f}GB available")
    
    def create_directories(self):
        """Create necessary directories"""
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
            logger.debug(f"Created directory: {dir_path}")
    
    def check_gpu(self):
        """Check GPU availability and log information"""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                device_props = torch.cuda.get_device_properties(0)
                total_memory = device_props.total_memory / 1e9
                
                logger.info(f"GPU available: {device_name}")
                logger.info(f"GPU memory: {total_memory:.1f} GB")
            else:
                logger.warning("No GPU detected, running on CPU (will be slower)")
        except ImportError:
            logger.warning("PyTorch not installed, GPU detection skipped")
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.shutdown_event.set()
        
        # Handle common termination signals
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, signal_handler)
    
    async def cleanup(self):
        """Perform cleanup operations"""
        logger.info("Starting cleanup operations...")
        
        # Save final metrics
        try:
            metrics = consciousness_metrics.get_health_status()
            logger.info(f"Final system metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error saving final metrics: {e}")
        
        logger.info("Cleanup completed")
    
    async def run_async(self):
        """Async main function"""
        try:
            # Start the web interface
            self.web_interface = AIMSWebInterface()
            
            # Start web server in background
            web_task = asyncio.create_task(self.run_web_server())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            logger.info("Shutdown initiated...")
            
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
            logger.info("Received keyboard interrupt")
        except Exception as e:
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
            
            logger.info("Web interface started on http://0.0.0.0:8000")
            
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
        logger.info("="*60)
        logger.info("Starting AIMS - Autonomous Intelligent Memory System")
        logger.info("="*60)
        
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
        
        logger.info("AIMS shutdown complete")
        sys.exit(self.exit_code)

def main():
    """Main entry point"""
    app = AIMSApplication()
    app.run()

if __name__ == '__main__':
    main()
