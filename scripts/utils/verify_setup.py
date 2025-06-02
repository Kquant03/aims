#!/usr/bin/env python3
"""
verify_setup.py - Comprehensive AIMS setup verification
"""

import sys
import os
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_header(title):
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}{title}{NC}")
    print(f"{BLUE}{'='*60}{NC}")

def check_mark(success):
    return f"{GREEN}✅{NC}" if success else f"{RED}❌{NC}"

def main():
    print_header("AIMS Setup Verification")
    
    all_good = True
    
    # 1. Check Python version
    print(f"\n{BLUE}1. Python Version Check:{NC}")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    is_correct_version = sys.version_info.major == 3 and sys.version_info.minor == 10
    print(f"{check_mark(is_correct_version)} Python {python_version}")
    if not is_correct_version:
        print(f"{YELLOW}   Warning: Python 3.10 is recommended{NC}")
        all_good = False
    
    # 2. Check virtual environment
    print(f"\n{BLUE}2. Virtual Environment Check:{NC}")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"{check_mark(in_venv)} Running in virtual environment")
    if in_venv:
        print(f"   Location: {sys.prefix}")
    else:
        print(f"{RED}   Not in virtual environment! Run: source venv/bin/activate{NC}")
        all_good = False
    
    # 3. Check critical packages
    print(f"\n{BLUE}3. Package Installation Check:{NC}")
    packages = {
        'anthropic': 'Claude API',
        'torch': 'PyTorch',
        'aiohttp': 'Web Framework',
        'redis': 'Redis Client',
        'mem0ai': 'Memory System',
        'transformers': 'Transformers'
    }
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"{check_mark(True)} {package:<15} - {description}")
        except ImportError:
            print(f"{check_mark(False)} {package:<15} - {description} (NOT INSTALLED)")
            all_good = False
    
    # 4. Check GPU/CUDA
    print(f"\n{BLUE}4. GPU/CUDA Check:{NC}")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        print(f"{check_mark(has_cuda)} CUDA available: {has_cuda}")
        if has_cuda:
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        print(f"{check_mark(False)} Error checking CUDA: {e}")
    
    # 5. Check environment variables
    print(f"\n{BLUE}5. Environment Variables:{NC}")
    env_vars = {
        'ANTHROPIC_API_KEY': ('Required', True),
        'OPENAI_API_KEY': ('Optional', False),
        'SESSION_SECRET': ('Recommended', False)
    }
    
    for var, (desc, required) in env_vars.items():
        value = os.environ.get(var)
        has_value = bool(value and value != f'your-{var.lower().replace("_", "-")}-here')
        
        if has_value:
            display_value = value[:8] + '...' if 'KEY' in var else 'SET'
            print(f"{check_mark(True)} {var}: {display_value}")
        else:
            symbol = check_mark(False) if required else f"{YELLOW}⚠️{NC} "
            print(f"{symbol} {var}: NOT SET ({desc})")
            if required:
                all_good = False
    
    # 6. Check directories
    print(f"\n{BLUE}6. Directory Structure:{NC}")
    required_dirs = ['data/states', 'data/backups', 'data/memories', 'logs', 'src/ui/templates']
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        print(f"{check_mark(exists)} {dir_path}")
        if not exists:
            all_good = False
    
    # 7. Check Docker
    print(f"\n{BLUE}7. Docker Services:{NC}")
    try:
        import subprocess
        
        # Check if Docker is running
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        docker_running = result.returncode == 0
        print(f"{check_mark(docker_running)} Docker is accessible")
        
        if docker_running:
            # Check docker-compose services
            result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                # Check for running services
                services = {'postgres': False, 'redis': False, 'qdrant': False}
                for line in lines:
                    for service in services:
                        if f'aims_{service}' in line and 'Up' in line:
                            services[service] = True
                
                for service, running in services.items():
                    print(f"{check_mark(running)} {service:<10} {'Running' if running else 'Not running'}")
                    if not running:
                        all_good = False
        else:
            print(f"{RED}   Cannot access Docker. Run: sudo usermod -aG docker $USER{NC}")
            print(f"{RED}   Then log out and back in or run: newgrp docker{NC}")
            all_good = False
            
    except FileNotFoundError:
        print(f"{check_mark(False)} Docker not installed")
        all_good = False
    except Exception as e:
        print(f"{check_mark(False)} Error checking Docker: {e}")
    
    # 8. Check .env file
    print(f"\n{BLUE}8. Configuration File:{NC}")
    env_exists = Path('.env').exists()
    print(f"{check_mark(env_exists)} .env file exists")
    if not env_exists:
        print(f"{YELLOW}   Run: cp .env.example .env{NC}")
        all_good = False
    
    # Summary
    print_header("Summary")
    if all_good:
        print(f"\n{GREEN}🎉 Everything looks good! Your AIMS setup is complete.{NC}")
        print(f"\nTo start AIMS:")
        print(f"1. Make sure you're in the virtual environment:")
        print(f"   {BLUE}source venv/bin/activate{NC}")
        print(f"\n2. Run AIMS:")
        print(f"   {BLUE}python -m src.main{NC}")
        print(f"\n3. Open your browser to:")
        print(f"   {BLUE}http://localhost:8000{NC}")
    else:
        print(f"\n{YELLOW}⚠️  Some issues were found. Please fix them before running AIMS.{NC}")
        print(f"\nKey issues to fix:")
        if not in_venv:
            print(f"- Activate virtual environment: {BLUE}source venv/bin/activate{NC}")
        if not env_exists:
            print(f"- Create .env file: {BLUE}cp .env.example .env{NC}")
        if not os.environ.get('ANTHROPIC_API_KEY'):
            print(f"- Add your Anthropic API key to .env file")
        print(f"\nFor Docker permission issues:")
        print(f"- Run: {BLUE}sudo usermod -aG docker $USER{NC}")
        print(f"- Then: {BLUE}newgrp docker{NC}")

if __name__ == "__main__":
    main()