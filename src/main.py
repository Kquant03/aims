# src/main.py - Main entry point for AIMS
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.web_interface import AIMSWebInterface
# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ui.web_interface import AIMSWebInterface

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

def check_environment():
    """Check required environment variables"""
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

def main():
    """Main entry point"""
    logger.info("Starting AIMS - Autonomous Intelligent Memory System")
    
    # Check environment
    check_environment()
    
    # Create necessary directories
    for dir_path in ['logs', 'data/states', 'data/backups', 'data/memories', 'data/uploads']:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.warning("No GPU detected, running on CPU (will be slower)")
    except ImportError:
        logger.warning("PyTorch not installed, GPU detection skipped")
    
    # Start the web interface
    try:
        app = AIMSWebInterface()
        app.run(host='0.0.0.0', port=8000)
    except KeyboardInterrupt:
        logger.info("Shutting down AIMS...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()