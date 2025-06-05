#!/usr/bin/env python3
"""
Fix AIMS import errors in web_interface.py
"""
import os
import sys

def fix_imports():
    """Fix the import errors in web_interface.py"""
    
    web_interface_path = "src/ui/web_interface.py"
    
    if not os.path.exists(web_interface_path):
        print("❌ Error: web_interface.py not found. Run from AIMS root directory.")
        sys.exit(1)
    
    # Read the current file
    with open(web_interface_path, 'r') as f:
        content = f.read()
    
    # Create backup
    import shutil
    backup_path = web_interface_path + ".import_fix_backup"
    shutil.copy2(web_interface_path, backup_path)
    print(f"📄 Backup saved to: {backup_path}")
    
    # Fix the imports section
    # The correct imports should be:
    correct_imports = """# web_interface.py - Main Web Application Interface
import os
import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional
import aiofiles
from aiohttp import web
import aiohttp_cors
from aiohttp_session import setup, get_session, SimpleCookieStorage, Session
from aiohttp_session.cookie_storage import EncryptedCookieStorage
import jinja2
import aiohttp_jinja2
from pathlib import Path
import logging

from src.api.claude_interface import ClaudeConsciousnessInterface
from src.api.websocket_server import ConsciousnessWebSocketServer
from src.api.state_manager import StateManager

logger = logging.getLogger(__name__)"""
    
    # Find where imports end (at the logger line)
    logger_pos = content.find("logger = logging.getLogger(__name__)")
    if logger_pos == -1:
        print("❌ Error: Could not find logger initialization")
        sys.exit(1)
    
    # Find the start of the file
    import_start = content.find("# web_interface.py")
    if import_start == -1:
        import_start = 0
    
    # Extract everything after logger initialization
    after_imports = content[logger_pos + len("logger = logging.getLogger(__name__)"):]
    
    # Rebuild the file
    new_content = correct_imports + after_imports
    
    # Write the fixed file
    with open(web_interface_path, 'w') as f:
        f.write(new_content)
    
    print("✅ Fixed import errors in web_interface.py!")
    print("🚀 AIMS should now start without import errors")

if __name__ == "__main__":
    print("🔧 Fixing AIMS import errors...")
    fix_imports()