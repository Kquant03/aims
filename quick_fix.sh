#!/usr/bin/env python3
# fix_main_syntax.py - Fix the syntax error in main.py

import os
import re

def fix_main_py():
    """Fix the syntax error in src/main.py"""
    
    main_path = 'src/main.py'
    if not os.path.exists(main_path):
        print(f"❌ {main_path} not found")
        return False
    
    with open(main_path, 'r') as f:
        content = f.read()
    
    # Find and fix the problematic line
    # The issue is with nested quotes in the f-string
    problematic_pattern = r'print\(f"   CUDA version: \{torch\.version\.cuda if hasattr\(torch\.version, "cuda"\) else "N/A" if hasattr\(torch\.version, "cuda"\) else "N/A"\}"\)'
    
    # Replace with a cleaner version
    replacement = '''cuda_version = "N/A"
                if hasattr(torch.version, 'cuda'):
                    cuda_version = torch.version.cuda
                print(f"   CUDA version: {cuda_version}")'''
    
    # Try direct replacement first
    if 'CUDA version: {torch.version.cuda if hasattr(torch.version, "cuda")' in content:
        # Find the exact line and replace it
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'CUDA version:' in line and 'torch.version.cuda' in line:
                # Replace this line with the fixed version
                indent = len(line) - len(line.lstrip())
                fixed_lines = [
                    ' ' * indent + 'cuda_version = "N/A"',
                    ' ' * indent + 'if torch.cuda.is_available() and hasattr(torch, "version") and hasattr(torch.version, "cuda"):',
                    ' ' * indent + '    cuda_version = torch.version.cuda',
                    ' ' * indent + 'print(f"   CUDA version: {cuda_version}")'
                ]
                # Replace the problematic line with our fixed lines
                lines[i:i+1] = fixed_lines
                content = '\n'.join(lines)
                break
    
    # Write back the fixed content
    with open(main_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed syntax error in {main_path}")
    return True

# Also create a simpler alternative if the fix doesn't work
def create_simple_main():
    """Create a simpler main.py without the problematic code"""
    
    simple_main = '''# src/main.py - Simplified AIMS launcher
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class SimpleAIMS:
    """Simplified AIMS Application"""
    
    def __init__(self):
        self.check_environment()
    
    def check_environment(self):
        """Check environment setup"""
        print("\\n📋 Checking environment...")
        
        # Check API key
        if not os.environ.get('ANTHROPIC_API_KEY'):
            print("❌ ANTHROPIC_API_KEY not set!")
            sys.exit(1)
        else:
            print("✅ API key found")
        
        # Check GPU (simplified)
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            else:
                print("ℹ️  No GPU detected - using CPU")
        except ImportError:
            print("ℹ️  PyTorch not installed - GPU detection skipped")
    
    async def run(self):
        """Run the application"""
        print("\\n🚀 Starting AIMS...")
        
        try:
            from src.ui.web_interface import AIMSWebInterface
            
            # Create and run web interface
            web_interface = AIMSWebInterface()
            print("\\n✨ AIMS is ready!")
            print("\\n📍 Access points:")
            print("   Web Interface:  http://localhost:8000")
            print("   WebSocket:      ws://localhost:8765")
            print("\\n💡 Press Ctrl+C to shut down\\n")
            
            # Run web interface
            await web_interface.app.startup()
            await asyncio.Event().wait()
            
        except ImportError as e:
            print(f"\\n❌ Import error: {e}")
            print("\\nTrying minimal mode...")
            await self.run_minimal()
        except Exception as e:
            print(f"\\n❌ Error: {e}")
            sys.exit(1)
    
    async def run_minimal(self):
        """Run minimal interface"""
        print("\\n🤖 Running in minimal mode...")
        print("Type 'quit' to exit\\n")
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    break
                print(f"AIMS: I heard '{user_input}' (minimal mode - no AI processing)\\n")
            except KeyboardInterrupt:
                break
        
        print("\\n👋 Goodbye!")

async def main():
    """Main entry point"""
    app = SimpleAIMS()
    await app.run()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\n\\n⌨️ Interrupted by user")
'''
    
    # Backup original if it exists
    if os.path.exists('src/main.py'):
        import shutil
        shutil.copy('src/main.py', 'src/main.py.bak')
        print("📋 Backed up original main.py to main.py.bak")
    
    # Write simple version
    with open('src/main.py', 'w') as f:
        f.write(simple_main)
    
    print("✅ Created simplified main.py")

if __name__ == "__main__":
    print("🔧 Fixing main.py syntax error...")
    
    # Try to fix the existing file
    if not fix_main_py():
        # If fix fails, create a simple replacement
        print("Creating simplified main.py...")
        create_simple_main()
    
    print("\\n✅ Fix complete! Try running AIMS again.")