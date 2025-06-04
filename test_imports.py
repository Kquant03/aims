#!/usr/bin/env python3
"""
Test basic imports to identify what's missing
"""
import sys

print("Testing AIMS imports...")
print("=" * 50)

# Test core imports
modules_to_test = [
    ("logging", "Python logging"),
    ("asyncio", "Async support"),
    ("torch", "PyTorch"),
    ("numpy", "NumPy"),
    ("anthropic", "Anthropic API"),
    ("aiohttp", "Web framework"),
    ("redis", "Redis client"),
    ("websockets", "WebSocket support"),
    ("jinja2", "Template engine"),
    ("aiohttp_session", "Session support"),
    ("aiohttp_jinja2", "Jinja2 integration"),
    ("yaml", "YAML support"),
    ("psutil", "System utilities"),
    ("dotenv", "Environment variables"),
    ("cryptography", "Cryptography")
]

failed = []
succeeded = []

for module, description in modules_to_test:
    try:
        if module == "dotenv":
            __import__("dotenv", fromlist=["load_dotenv"])
        elif module == "yaml":
            __import__("yaml")
        else:
            __import__(module)
        succeeded.append(module)
        print(f"✅ {module:<20} - {description}")
    except ImportError as e:
        failed.append((module, str(e)))
        print(f"❌ {module:<20} - {description} (MISSING)")

print("\n" + "=" * 50)
print(f"Summary: {len(succeeded)}/{len(modules_to_test)} modules available")

if failed:
    print("\nMissing modules:")
    for module, error in failed:
        print(f"  - {module}")
    
    print("\nTo install missing modules:")
    print(f"  {sys.executable} -m pip install " + " ".join([m[0] for m in failed]))
else:
    print("\n✨ All core modules are available!")
    print("\nNow testing AIMS imports...")
    
    # Test AIMS specific imports
    try:
        print("\nTesting consciousness.py import...")
        # First fix the logger issue
        import subprocess
        subprocess.run([
            "sed", "-i", 
            "s/logger\\.warning(\"Flash Attention not available, using standard attention\")/print(\"WARNING: Flash Attention not available, using standard attention\")/",
            "src/core/consciousness.py"
        ])
        
        from src.core.consciousness import ConsciousnessCore
        print("✅ ConsciousnessCore imported successfully")
    except Exception as e:
        print(f"❌ Error importing ConsciousnessCore: {e}")
    
    try:
        from src.ui.web_interface import AIMSWebInterface
        print("✅ AIMSWebInterface imported successfully")
    except Exception as e:
        print(f"❌ Error importing AIMSWebInterface: {e}")

print("\nDone!")