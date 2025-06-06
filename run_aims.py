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
