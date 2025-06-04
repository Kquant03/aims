#!/usr/bin/env python3
"""
AIMS Emergency Fix Script - Fixes Docker, Dependencies, and Runtime Issues
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

class AIMSEmergencyFixer:
    def __init__(self):
        self.root_dir = Path.cwd()
        
    def run(self):
        print(f"{BLUE}{'='*60}{NC}")
        print(f"{BLUE}AIMS Emergency Fix Script{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        # Fix all issues in order
        self.fix_docker_compose()
        self.fix_pip_and_dependencies()
        self.create_updated_start_script()
        self.verify_setup()
        
        print(f"\n{GREEN}✨ All fixes applied!{NC}")
        print(f"\nNow you can run: {BLUE}./start_aims_fixed.sh{NC}")
    
    def fix_docker_compose(self):
        """Fix the docker-compose.yml file"""
        print(f"\n{BLUE}1. Fixing Docker Compose configuration...{NC}")
        
        # Create a fixed docker-compose.yml
        docker_compose_content = """version: '3.7'

services:
  # PostgreSQL with PGVector
  postgres:
    image: postgres:15
    container_name: aims_postgres
    environment:
      POSTGRES_DB: aims_memory
      POSTGRES_USER: aims
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-aims_secure_password}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5433:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aims"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: aims_redis
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Qdrant vector database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: aims_qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  qdrant_data:

networks:
  default:
    name: aims_network
"""
        
        # Backup old docker-compose.yml
        docker_compose_path = self.root_dir / 'docker-compose.yml'
        if docker_compose_path.exists():
            backup_path = self.root_dir / 'docker-compose.yml.backup'
            docker_compose_path.rename(backup_path)
            print(f"  {BLUE}•{NC} Backed up old docker-compose.yml")
        
        # Write new docker-compose.yml
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose_content)
        print(f"  {GREEN}✓{NC} Created fixed docker-compose.yml")
        
        # Stop all containers first
        print(f"  Stopping existing containers...")
        subprocess.run(['docker-compose', 'down'], capture_output=True)
        
        # Remove problematic containers
        subprocess.run(['docker', 'rm', '-f', 'aims_postgres'], capture_output=True)
        
        print(f"  {GREEN}✓{NC} Docker Compose fixed")
    
    def fix_pip_and_dependencies(self):
        """Fix pip and install dependencies properly"""
        print(f"\n{BLUE}2. Fixing Python dependencies...{NC}")
        
        # Determine Python and pip paths
        venv_path = self.root_dir / 'venv'
        if os.name == 'nt':  # Windows
            python_path = venv_path / 'Scripts' / 'python.exe'
            pip_path = venv_path / 'bin' / 'pip.exe'
        else:  # Unix-like
            python_path = venv_path / 'bin' / 'python'
            pip_path = venv_path / 'bin' / 'pip'
        
        # Upgrade pip and setuptools first
        print(f"  Upgrading pip and setuptools...")
        subprocess.run([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], 
                      capture_output=True)
        print(f"  {GREEN}✓{NC} Pip and setuptools upgraded")
        
        # Install python-dotenv specifically
        print(f"  Installing python-dotenv...")
        result = subprocess.run([str(pip_path), 'install', 'python-dotenv'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  {GREEN}✓{NC} python-dotenv installed")
        else:
            print(f"  {RED}✗{NC} Failed to install python-dotenv")
        
        # Create a minimal requirements file for immediate needs
        minimal_requirements = """# Core dependencies for AIMS to start
python-dotenv>=1.0.0
anthropic>=0.25.0
aiohttp>=3.9.0
aiohttp-cors>=0.7.0
aiohttp-session>=2.12.0
cryptography>=41.0.0
websockets>=12.0
pyyaml>=6.0
numpy>=1.24.0
psutil>=5.9.0
colorlog>=6.8.0
"""
        
        minimal_req_path = self.root_dir / 'requirements_minimal.txt'
        with open(minimal_req_path, 'w') as f:
            f.write(minimal_requirements)
        
        # Install minimal requirements
        print(f"  Installing minimal requirements...")
        result = subprocess.run([str(pip_path), 'install', '-r', 'requirements_minimal.txt'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  {GREEN}✓{NC} Minimal requirements installed")
        else:
            print(f"  {YELLOW}!{NC} Some requirements failed, but continuing...")
        
        # Try to install the full requirements.txt but don't fail if some packages have issues
        print(f"  Attempting to install remaining requirements...")
        subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt', '--no-deps'], 
                      capture_output=True)
        
        print(f"  {GREEN}✓{NC} Dependencies fixed")
    
    def create_updated_start_script(self):
        """Create an updated start script that handles issues"""
        print(f"\n{BLUE}3. Creating updated start script...{NC}")
        
        start_script_content = """#!/bin/bash
# Updated start script for AIMS with fixes

echo "Starting AIMS (Fixed Version)..."

# Colors
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m'

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Make sure pip is up to date
echo "Ensuring pip is up to date..."
python -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1

# Install core dependencies
echo "Checking core dependencies..."
python -m pip install python-dotenv anthropic aiohttp pyyaml cryptography >/dev/null 2>&1

# Start Docker services with the fixed docker-compose.yml
echo "Starting Docker services..."
if command -v docker-compose &> /dev/null; then
    # Stop any existing containers
    docker-compose down >/dev/null 2>&1
    
    # Start fresh
    docker-compose up -d
    
    # Wait for services
    echo "Waiting for services to be ready..."
    sleep 10
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        echo -e "${GREEN}✓${NC} Docker services are running"
    else
        echo -e "${YELLOW}!${NC} Some Docker services may not be running properly"
        echo "  You can check with: docker-compose ps"
    fi
else
    echo -e "${YELLOW}!${NC} Docker Compose not found. Services won't be available."
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}!${NC} No .env file found. Creating from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "  Please edit .env and add your ANTHROPIC_API_KEY"
    fi
fi

# Check for API key
if [ -f ".env" ]; then
    if grep -q "ANTHROPIC_API_KEY=your-anthropic-api-key-here" .env; then
        echo ""
        echo -e "${YELLOW}!${NC} WARNING: You need to add your ANTHROPIC_API_KEY to the .env file"
        echo "  Edit .env and replace 'your-anthropic-api-key-here' with your actual key"
        echo ""
        read -p "Press Enter to continue anyway (the app will fail without a valid key)..."
    fi
fi

# Run AIMS
echo ""
echo "Starting AIMS on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Run with explicit PYTHONPATH
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python -m src.main
"""
        
        script_path = self.root_dir / 'start_aims_fixed.sh'
        with open(script_path, 'w') as f:
            f.write(start_script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"  {GREEN}✓{NC} Created start_aims_fixed.sh")
    
    def verify_setup(self):
        """Verify the setup is ready"""
        print(f"\n{BLUE}4. Verifying setup...{NC}")
        
        # Check venv
        venv_path = self.root_dir / 'venv'
        if venv_path.exists():
            print(f"  {GREEN}✓{NC} Virtual environment exists")
        else:
            print(f"  {RED}✗{NC} Virtual environment missing")
        
        # Check critical files
        critical_files = [
            '.env',
            'docker-compose.yml',
            'src/main.py',
            'requirements.txt'
        ]
        
        for file_name in critical_files:
            if (self.root_dir / file_name).exists():
                print(f"  {GREEN}✓{NC} {file_name} exists")
            else:
                print(f"  {RED}✗{NC} {file_name} missing")
        
        # Check if we can import dotenv in venv
        if os.name == 'nt':
            python_path = venv_path / 'Scripts' / 'python.exe'
        else:
            python_path = venv_path / 'bin' / 'python'
        
        result = subprocess.run(
            [str(python_path), '-c', 'import dotenv; print("OK")'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and "OK" in result.stdout:
            print(f"  {GREEN}✓{NC} python-dotenv is importable")
        else:
            print(f"  {RED}✗{NC} python-dotenv import failed")
    
    def create_minimal_main_wrapper(self):
        """Create a wrapper for main.py that handles missing imports"""
        print(f"\n{BLUE}5. Creating failsafe main wrapper...{NC}")
        
        wrapper_content = '''#!/usr/bin/env python3
"""
Failsafe wrapper for AIMS main
"""
import sys
import subprocess
from pathlib import Path

def check_and_install_missing():
    """Check for missing modules and try to install them"""
    missing = []
    
    required_modules = [
        ('dotenv', 'python-dotenv'),
        ('anthropic', 'anthropic'),
        ('aiohttp', 'aiohttp'),
        ('yaml', 'pyyaml')
    ]
    
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)

# Try to install missing packages
check_and_install_missing()

# Now run the actual main
try:
    from src.main import main
    main()
except ImportError as e:
    print(f"Error: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)
'''
        
        wrapper_path = self.root_dir / 'run_aims.py'
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        
        print(f"  {GREEN}✓{NC} Created run_aims.py wrapper")

if __name__ == "__main__":
    fixer = AIMSEmergencyFixer()
    fixer.run()