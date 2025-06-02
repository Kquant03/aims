#!/usr/bin/env python3
# check_venv.py - Verify AIMS virtual environment is properly configured

import sys
import os
import subprocess
from pathlib import Path

# Colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def check_python_version():
    """Check if we're running Python 3.10"""
    version = sys.version_info
    print(f"\n{BLUE}Python Version Check:{NC}")
    print(f"Current Python: {sys.version}")
    
    if version.major == 3 and version.minor == 10:
        print(f"{GREEN}✅ Python 3.10 detected - Perfect!{NC}")
        return True
    else:
        print(f"{YELLOW}⚠️  Python {version.major}.{version.minor} detected - Python 3.10 is recommended{NC}")
        return False

def check_virtual_env():
    """Check if we're in a virtual environment"""
    print(f"\n{BLUE}Virtual Environment Check:{NC}")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"{GREEN}✅ Running in virtual environment{NC}")
        print(f"   Location: {sys.prefix}")
        
        # Check if it's our project's venv
        if 'aims' in sys.prefix.lower() and 'venv' in sys.prefix:
            print(f"{GREEN}✅ This appears to be the AIMS virtual environment{NC}")
        return True
    else:
        print(f"{RED}❌ Not running in virtual environment{NC}")
        print("   Activate with: source venv/bin/activate")
        return False

def check_core_packages():
    """Check if core packages are installed"""
    print(f"\n{BLUE}Core Package Check:{NC}")
    
    core_packages = {
        'anthropic': 'Anthropic API (Claude)',
        'torch': 'PyTorch (Deep Learning)',
        'aiohttp': 'Async Web Framework',
        'websockets': 'WebSocket Support',
        'redis': 'Redis Client',
        'psycopg2': 'PostgreSQL Client',
        'mem0ai': 'Memory Management',
        'transformers': 'Hugging Face Transformers'
    }
    
    all_installed = True
    installed_count = 0
    
    for package, description in core_packages.items():
        try:
            __import__(package)
            print(f"{GREEN}✅ {package:<15} - {description}{NC}")
            installed_count += 1
        except ImportError:
            print(f"{RED}❌ {package:<15} - {description} (NOT INSTALLED){NC}")
            all_installed = False
    
    print(f"\n{installed_count}/{len(core_packages)} core packages installed")
    return all_installed

def check_torch_cuda():
    """Check PyTorch CUDA support"""
    print(f"\n{BLUE}PyTorch CUDA Check:{NC}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"{GREEN}✅ CUDA is available{NC}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print(f"{YELLOW}⚠️  CUDA not available - will use CPU{NC}")
            return False
    except ImportError:
        print(f"{RED}❌ PyTorch not installed{NC}")
        return False

def check_package_versions():
    """Check specific package versions"""
    print(f"\n{BLUE}Package Version Check:{NC}")
    
    critical_versions = {
        'anthropic': '0.25.0',
        'numpy': '1.24.0',
        'pandas': '2.0.0',
        'aiohttp': '3.9.0'
    }
    
    for package, min_version in critical_versions.items():
        try:
            module = __import__(package)
            if hasattr(module, '__version__'):
                version = module.__version__
                print(f"{package:<15} {version:<10} (minimum: {min_version})")
        except ImportError:
            print(f"{package:<15} NOT INSTALLED")

def count_total_packages():
    """Count total installed packages"""
    print(f"\n{BLUE}Package Statistics:{NC}")
    
    try:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            # Skip header lines
            package_count = len(lines) - 2
            print(f"Total packages installed: {package_count}")
            
            # Check size
            result = subprocess.run(['pip', 'show', '-f', 'torch'], capture_output=True, text=True)
            if 'Location:' in result.stdout:
                location = [line for line in result.stdout.split('\n') if 'Location:' in line][0]
                print(f"Package location: {location.split(': ')[1]}")
    except:
        pass

def check_environment_variables():
    """Check if required environment variables are set"""
    print(f"\n{BLUE}Environment Variables:{NC}")
    
    env_vars = {
        'ANTHROPIC_API_KEY': 'Required for Claude API',
        'OPENAI_API_KEY': 'Optional for embeddings',
        'VIRTUAL_ENV': 'Virtual environment path'
    }
    
    for var, description in env_vars.items():
        value = os.environ.get(var)
        if value:
            if var.endswith('_KEY'):
                # Don't show full API keys
                display_value = value[:8] + '...' + value[-4:] if len(value) > 12 else 'SET'
            else:
                display_value = value
            print(f"{GREEN}✅ {var}: {display_value}{NC}")
        else:
            status = "⚠️ " if 'Optional' in description else "❌"
            print(f"{YELLOW if 'Optional' in description else RED}{status} {var}: NOT SET - {description}{NC}")

def main():
    print(f"{BLUE}{'='*60}{NC}")
    print(f"{BLUE}AIMS Virtual Environment Verification{NC}")
    print(f"{BLUE}{'='*60}{NC}")
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_env),
        ("Core Packages", check_core_packages),
        ("PyTorch CUDA", check_torch_cuda),
        ("Package Versions", check_package_versions),
        ("Package Count", count_total_packages),
        ("Environment Variables", check_environment_variables)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"{RED}Error in {name}: {e}{NC}")
            results.append((name, False))
    
    # Summary
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}Summary:{NC}")
    print(f"{BLUE}{'='*60}{NC}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{GREEN}✅ PASS{NC}" if result else f"{RED}❌ FAIL{NC}"
        print(f"{name:<25} {status}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print(f"\n{GREEN}🎉 Your AIMS virtual environment is properly configured!{NC}")
        print(f"\nYou can now run: {BLUE}python -m src.main{NC}")
    else:
        print(f"\n{YELLOW}⚠️  Some issues were found. Run the following to fix:{NC}")
        print(f"   pip install -r requirements.txt")
        print(f"\nFor CUDA support, install PyTorch with:")
        print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main()