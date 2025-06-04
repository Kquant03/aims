# src/main.py - Enhanced Main entry point for AIMS with graceful shutdown
import os
import sys
import asyncio
import signal
import logging
from pathlib import Path
from typing import Optional
import psutil
from dotenv import load_dotenv

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
        
        # Optional but recommended
        optional_vars = ['OPENAI_API_KEY', 'SESSION_SECRET']
        missing_optional = [var for var in optional_vars if not os.environ.get(var)]
        
        if missing_optional:
            logger.warning(f"Missing optional environment variables: {missing_optional}")
        
        # Check system resources
        memory = psutil.virtual_memory()
        if memory.available < 4 * 1024 * 1024 * 1024:  # 4GB
            logger.warning(f"Low system memory: {memory.available / 1e9:.1f}GB available")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 10 * 1024 * 1024 * 1024:  # 10GB
            logger.warning(f"Low disk space: {disk.free / 1e9:.1f}GB free")
    
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
                
                # Check if it's RTX 3090
                if "3090" in device_name:
                    logger.info("RTX 3090 detected - optimal configuration will be used")
                else:
                    logger.info(f"Non-RTX 3090 GPU detected - using adaptive settings")
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
        
        # Windows-specific signal handling
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, signal_handler)
    
    async def cleanup(self):
        """Perform cleanup operations"""
        logger.info("Starting cleanup operations...")
        
        # Save final metrics
        try:
            metrics = consciousness_metrics.get_health_status()
            logger.info(f"Final system metrics: {metrics}")
        except Exception as e:
            logger.error(f"Error saving final metrics: {e}")
        
        # Run all registered cleanup tasks
        for task_name, cleanup_coro in self.cleanup_tasks:
            try:
                logger.info(f"Running cleanup task: {task_name}")
                await cleanup_coro()
            except Exception as e:
                logger.error(f"Error in cleanup task {task_name}: {e}")
        
        logger.info("Cleanup completed")
    
    def register_cleanup_task(self, name: str, coro):
        """Register a cleanup coroutine to run on shutdown"""
        self.cleanup_tasks.append((name, coro))
    
    async def run_async(self):
        """Async main function"""
        try:
            # Start the web interface
            self.web_interface = AIMSWebInterface()
            
            # Register cleanup tasks
            self.register_cleanup_task(
                "consciousness_state",
                self.save_consciousness_state
            )
            self.register_cleanup_task(
                "web_interface",
                self.shutdown_web_interface
            )
            
            # Start web server in background
            web_task = asyncio.create_task(
                self.run_web_server()
            )
            
            # Start monitoring task
            monitor_task = asyncio.create_task(
                self.monitor_system_health()
            )
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            logger.info("Shutdown initiated...")
            
            # Cancel background tasks
            web_task.cancel()
            monitor_task.cancel()
            
            # Wait for tasks to complete with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(web_task, monitor_task, return_exceptions=True),
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
    
    async def monitor_system_health(self):
        """Monitor system health and log warnings"""
        try:
            while not self.shutdown_event.is_set():
                # Check memory usage
                memory = psutil.virtual_memory()
                if memory.percent > 90:
                    logger.warning(f"High memory usage: {memory.percent}%")
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                
                # Check consciousness coherence
                if hasattr(self.web_interface, 'claude_interface'):
                    coherence = self.web_interface.claude_interface.consciousness.state.global_coherence
                    if coherence < 0.5:
                        logger.warning(f"Low consciousness coherence: {coherence}")
                
                # Sleep for monitoring interval
                await asyncio.sleep(30)
                
        except asyncio.CancelledError:
            logger.info("System monitor task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in system monitor: {e}")
    
    async def save_consciousness_state(self):
        """Save consciousness state before shutdown"""
        if self.web_interface and hasattr(self.web_interface, 'claude_interface'):
            try:
                logger.info("Saving consciousness state...")
                await self.web_interface.state_manager.save_complete_state(
                    self.web_interface.claude_interface
                )
                logger.info("Consciousness state saved successfully")
            except Exception as e:
                logger.error(f"Error saving consciousness state: {e}")
    
    async def shutdown_web_interface(self):
        """Shutdown web interface gracefully"""
        if self.web_interface:
            try:
                await self.web_interface.claude_interface.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down web interface: {e}")
    
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