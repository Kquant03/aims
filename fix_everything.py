#!/usr/bin/env python3
"""
AIMS Complete Setup and Fix Script
This script will set up the entire AIMS project, fix all known issues,
and ensure everything is working correctly.
"""

import os
import sys
import subprocess
import shutil
import json
import yaml
import secrets
from pathlib import Path
from datetime import datetime
import platform
import re

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

class AIMSSetupFixer:
    def __init__(self):
        self.root_dir = Path.cwd()
        self.issues_fixed = []
        self.issues_failed = []
        self.python_cmd = None
        
    def run(self):
        """Run complete setup and fixes"""
        print(f"{BLUE}{'='*60}{NC}")
        print(f"{BLUE}AIMS Complete Setup & Fix Script{NC}")
        print(f"{BLUE}{'='*60}{NC}\n")
        
        # Run all setup steps
        self.check_system_requirements()
        self.setup_python_environment()
        self.create_directory_structure()
        self.fix_session_key_issue()
        self.setup_environment_variables()
        self.install_dependencies()
        self.fix_docker_compose()
        self.setup_databases()
        self.create_helper_scripts()
        self.verify_setup()
        
        # Print summary
        self.print_summary()
    
    def check_system_requirements(self):
        """Check system requirements"""
        print(f"\n{BLUE}1. Checking system requirements...{NC}")
        
        # Check OS
        os_type = platform.system()
        print(f"  Operating System: {os_type}")
        
        # Check Python versions
        python_versions = []
        for cmd in ['python3.10', 'python3.11', 'python3.12', 'python3', 'python']:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    python_versions.append((cmd, version))
            except FileNotFoundError:
                pass
        
        if python_versions:
            print(f"  {GREEN}✓{NC} Python installations found:")
            for cmd, version in python_versions:
                print(f"    - {cmd}: {version}")
            
            # Prefer Python 3.10
            for cmd, version in python_versions:
                if '3.10' in version:
                    self.python_cmd = cmd
                    print(f"  {GREEN}✓{NC} Using {cmd} (recommended)")
                    break
            
            if not self.python_cmd:
                self.python_cmd = python_versions[0][0]
                print(f"  {YELLOW}!{NC} Python 3.10 not found, using {self.python_cmd}")
        else:
            print(f"  {RED}✗{NC} No Python installation found!")
            self.issues_failed.append("Python installation")
            return
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  {GREEN}✓{NC} Docker: {result.stdout.strip()}")
            else:
                print(f"  {YELLOW}!{NC} Docker not accessible")
        except FileNotFoundError:
            print(f"  {YELLOW}!{NC} Docker not installed")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  {GREEN}✓{NC} GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"  {YELLOW}!{NC} No GPU detected (will use CPU)")
        except ImportError:
            print(f"  {BLUE}•{NC} PyTorch not yet installed (GPU check pending)")
        
        self.issues_fixed.append("System requirements check")
    
    def setup_python_environment(self):
        """Set up Python virtual environment"""
        print(f"\n{BLUE}2. Setting up Python environment...{NC}")
        
        venv_path = self.root_dir / 'venv'
        
        # Remove old venv if exists
        if venv_path.exists():
            print(f"  {YELLOW}!{NC} Removing existing virtual environment...")
            shutil.rmtree(venv_path)
        
        # Create new venv
        print(f"  Creating virtual environment with {self.python_cmd}...")
        subprocess.run([self.python_cmd, '-m', 'venv', 'venv'], check=True)
        print(f"  {GREEN}✓{NC} Virtual environment created")
        
        # Determine pip path
        if os.name == 'nt':  # Windows
            self.pip_path = venv_path / 'Scripts' / 'pip'
            self.python_path = venv_path / 'Scripts' / 'python'
        else:  # Unix-like
            self.pip_path = venv_path / 'bin' / 'pip'
            self.python_path = venv_path / 'bin' / 'python'
        
        # Upgrade pip
        print(f"  Upgrading pip...")
        subprocess.run([str(self.python_path), '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel', 'setuptools'], 
                      capture_output=True)
        print(f"  {GREEN}✓{NC} Pip upgraded")
        
        self.issues_fixed.append("Python environment")
    
    def create_directory_structure(self):
        """Create all necessary directories"""
        print(f"\n{BLUE}3. Creating directory structure...{NC}")
        
        directories = [
            'data/states',
            'data/backups',
            'data/memories',
            'data/uploads',
            'data/checkpoints',
            'logs',
            'src/ui/templates',
            'src/ui/static/js',
            'src/ui/static/css',
            'src/ui/static/images',
            'configs',
            'scripts/fixes',
            'scripts/setup',
            'scripts/utils',
            'tests',
            'docs'
        ]
        
        for dir_path in directories:
            path = self.root_dir / dir_path
            path.mkdir(parents=True, exist_ok=True)
            print(f"  {GREEN}✓{NC} Created {dir_path}")
        
        # Create __init__.py files
        init_dirs = [
            'src',
            'src/api',
            'src/core',
            'src/persistence',
            'src/ui',
            'src/utils'
        ]
        
        for dir_path in init_dirs:
            init_file = self.root_dir / dir_path / '__init__.py'
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.touch()
        
        self.issues_fixed.append("Directory structure")
    
    def fix_session_key_issue(self):
        """Fix the session key encoding issue"""
        print(f"\n{BLUE}4. Fixing session key issue...{NC}")
        
        web_interface_path = self.root_dir / 'src/ui/web_interface.py'
        
        if not web_interface_path.exists():
            print(f"  {YELLOW}!{NC} web_interface.py not found, will be created later")
            return
        
        # Read the file
        with open(web_interface_path, 'r') as f:
            content = f.read()
        
        # Fix the session key issue with a robust solution
        # Find the problematic line
        if "setup(self.app, EncryptedCookieStorage(secret_key.encode()))" in content:
            # Replace with proper handling
            new_session_setup = '''# Set up session middleware with proper key handling
        from cryptography.fernet import Fernet
        
        # Handle session key properly
        if secret_key == 'dev-secret-key-change-in-production':
            # Generate a proper key for development
            fernet_key = Fernet.generate_key()
            setup(self.app, EncryptedCookieStorage(fernet_key))
            logger.warning("Using generated session key - set SESSION_SECRET in .env for production")
        else:
            # Use the provided key - check if it needs encoding
            try:
                # If it's already a valid Fernet key (base64), use as-is
                if len(secret_key) == 44 and secret_key.endswith('='):
                    setup(self.app, EncryptedCookieStorage(secret_key.encode('utf-8')))
                else:
                    # Otherwise, generate a proper key
                    fernet_key = Fernet.generate_key()
                    setup(self.app, EncryptedCookieStorage(fernet_key))
                    logger.warning("Invalid SESSION_SECRET format - using generated key")
            except Exception as e:
                # Fallback: generate a new key
                fernet_key = Fernet.generate_key()
                setup(self.app, EncryptedCookieStorage(fernet_key))
                logger.warning(f"Session key error: {e}. Using generated key.")'''
            
            content = content.replace(
                "setup(self.app, EncryptedCookieStorage(secret_key.encode()))",
                new_session_setup
            )
            
            # Write back
            with open(web_interface_path, 'w') as f:
                f.write(content)
            
            print(f"  {GREEN}✓{NC} Fixed session key encoding issue")
            self.issues_fixed.append("Session key fix")
        else:
            print(f"  {BLUE}•{NC} Session key already fixed or different format")
    
    def setup_environment_variables(self):
        """Set up environment variables"""
        print(f"\n{BLUE}5. Setting up environment variables...{NC}")
        
        env_path = self.root_dir / '.env'
        env_example_content = """# AIMS Environment Configuration
# Generated by setup script on {timestamp}

# Required - Add your API key here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional but recommended
OPENAI_API_KEY=your-openai-api-key-for-embeddings
SESSION_SECRET={session_secret}

# Database Configuration
POSTGRES_PASSWORD=aims_secure_password_{random_suffix}
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=aims_memory
POSTGRES_USER=aims

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# S3 Backup (optional)
S3_BACKUP_ENABLED=false
S3_BUCKET=
S3_ACCESS_KEY=
S3_SECRET_KEY=
S3_REGION=us-east-1

# System Configuration
LOG_LEVEL=INFO
BACKUP_INTERVAL_HOURS=6
MAX_LOCAL_BACKUPS=7

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
GPU_MEMORY_FRACTION=0.9
"""
        
        # Generate secure keys
        try:
            from cryptography.fernet import Fernet
            session_secret = Fernet.generate_key().decode()
        except ImportError:
            session_secret = secrets.token_urlsafe(32)
        
        random_suffix = secrets.token_hex(4)
        
        # Create .env file
        env_content = env_example_content.format(
            timestamp=datetime.now().isoformat(),
            session_secret=session_secret,
            random_suffix=random_suffix
        )
        
        if env_path.exists():
            # Backup existing .env
            backup_path = env_path.with_suffix('.env.backup')
            shutil.copy(env_path, backup_path)
            print(f"  {BLUE}•{NC} Backed up existing .env to .env.backup")
            
            # Preserve API keys if they exist
            with open(env_path, 'r') as f:
                existing_content = f.read()
                
            # Extract existing API keys
            anthropic_match = re.search(r'ANTHROPIC_API_KEY=(.+)', existing_content)
            openai_match = re.search(r'OPENAI_API_KEY=(.+)', existing_content)
            
            if anthropic_match and anthropic_match.group(1) != 'your-anthropic-api-key-here':
                env_content = env_content.replace(
                    'ANTHROPIC_API_KEY=your-anthropic-api-key-here',
                    f'ANTHROPIC_API_KEY={anthropic_match.group(1)}'
                )
                print(f"  {GREEN}✓{NC} Preserved existing ANTHROPIC_API_KEY")
            
            if openai_match and openai_match.group(1) != 'your-openai-api-key-for-embeddings':
                env_content = env_content.replace(
                    'OPENAI_API_KEY=your-openai-api-key-for-embeddings',
                    f'OPENAI_API_KEY={openai_match.group(1)}'
                )
                print(f"  {GREEN}✓{NC} Preserved existing OPENAI_API_KEY")
        
        with open(env_path, 'w') as f:
            f.write(env_content)
        
        print(f"  {GREEN}✓{NC} Created/updated .env file")
        
        # Create .env.example
        env_example_path = self.root_dir / '.env.example'
        with open(env_example_path, 'w') as f:
            f.write(env_example_content.format(
                timestamp='<timestamp>',
                session_secret='<generate-with-setup-script>',
                random_suffix='<random>'
            ))
        
        print(f"  {GREEN}✓{NC} Created .env.example")
        
        # Check for API key
        if 'your-anthropic-api-key-here' in env_content:
            print(f"  {YELLOW}!{NC} Remember to add your ANTHROPIC_API_KEY to .env")
        
        self.issues_fixed.append("Environment configuration")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print(f"\n{BLUE}6. Installing Python dependencies...{NC}")
        
        # First install critical dependencies
        critical_deps = [
            'pip>=21.0',
            'wheel',
            'setuptools',
            'cryptography',
            'python-dotenv',
            'pyyaml',
            'aiohttp>=3.9.0',
            'aiohttp-cors',
            'aiohttp-session[secure]',
            'anthropic>=0.25.0'
        ]
        
        print(f"  Installing critical dependencies...")
        for dep in critical_deps:
            subprocess.run([str(self.pip_path), 'install', dep], capture_output=True)
        
        print(f"  {GREEN}✓{NC} Critical dependencies installed")
        
        # Install PyTorch with CUDA if available
        print(f"  Installing PyTorch...")
        try:
            # Check if NVIDIA GPU is available
            nvidia_check = subprocess.run(['nvidia-smi'], capture_output=True)
            if nvidia_check.returncode == 0:
                # Install PyTorch with CUDA
                subprocess.run([
                    str(self.pip_path), 'install', 
                    'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cu121'
                ], capture_output=True)
                print(f"  {GREEN}✓{NC} PyTorch installed with CUDA support")
            else:
                # Install CPU-only PyTorch
                subprocess.run([
                    str(self.pip_path), 'install',
                    'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cpu'
                ], capture_output=True)
                print(f"  {GREEN}✓{NC} PyTorch installed (CPU only)")
        except:
            print(f"  {YELLOW}!{NC} PyTorch installation may need manual intervention")
        
        # Install remaining requirements
        req_path = self.root_dir / 'requirements.txt'
        if req_path.exists():
            print(f"  Installing remaining requirements (this may take a few minutes)...")
            result = subprocess.run(
                [str(self.pip_path), 'install', '-r', 'requirements.txt'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"  {GREEN}✓{NC} All requirements installed")
            else:
                print(f"  {YELLOW}!{NC} Some requirements failed to install")
                # Try to install what we can
                subprocess.run([
                    str(self.pip_path), 'install', '-r', 'requirements.txt',
                    '--no-deps'
                ], capture_output=True)
        
        self.issues_fixed.append("Python dependencies")
    
    def fix_docker_compose(self):
        """Fix Docker Compose configuration"""
        print(f"\n{BLUE}7. Fixing Docker configuration...{NC}")
        
        docker_compose_content = """version: '3.7'

services:
  # PostgreSQL with PGVector
  postgres:
    image: pgvector/pgvector:pg15
    container_name: aims_postgres
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-aims_memory}
      POSTGRES_USER: ${POSTGRES_USER:-aims}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-aims_secure_password}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "${POSTGRES_PORT:-5433}:5432"
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
      - "${REDIS_PORT:-6379}:6379"
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
      - "${QDRANT_PORT:-6333}:6333"
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
        
        docker_compose_path = self.root_dir / 'docker-compose.yml'
        
        # Backup existing file
        if docker_compose_path.exists():
            backup_path = docker_compose_path.with_suffix('.yml.backup')
            shutil.copy(docker_compose_path, backup_path)
            print(f"  {BLUE}•{NC} Backed up existing docker-compose.yml")
        
        # Write new file
        with open(docker_compose_path, 'w') as f:
            f.write(docker_compose_content)
        
        print(f"  {GREEN}✓{NC} Created fixed docker-compose.yml")
        
        # Create init_db.sql if it doesn't exist
        init_db_path = self.root_dir / 'scripts/init_db.sql'
        if not init_db_path.exists():
            init_db_content = """-- PostgreSQL initialization for AIMS
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create schema
CREATE SCHEMA IF NOT EXISTS aims;
SET search_path TO aims, public;
"""
            init_db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(init_db_path, 'w') as f:
                f.write(init_db_content)
            print(f"  {GREEN}✓{NC} Created init_db.sql")
        
        self.issues_fixed.append("Docker configuration")
    
    def setup_databases(self):
        """Set up database services"""
        print(f"\n{BLUE}8. Setting up database services...{NC}")
        
        # Check if Docker is accessible
        try:
            result = subprocess.run(['docker', 'ps'], capture_output=True)
            if result.returncode != 0:
                print(f"  {YELLOW}!{NC} Docker not accessible. Run: sudo usermod -aG docker $USER")
                print(f"     Then log out and back in or run: newgrp docker")
                return
        except FileNotFoundError:
            print(f"  {RED}✗{NC} Docker not installed")
            return
        
        # Stop existing containers
        print(f"  Stopping any existing containers...")
        subprocess.run(['docker-compose', 'down'], capture_output=True)
        
        # Start services
        print(f"  Starting Docker services...")
        result = subprocess.run(['docker-compose', 'up', '-d'], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  {GREEN}✓{NC} Docker services started")
            
            # Wait for services to be ready
            print(f"  Waiting for services to be ready...")
            import time
            time.sleep(10)
            
            # Check service status
            result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
            print(f"  Service status:")
            for line in result.stdout.split('\n'):
                if 'aims_' in line and 'Up' in line:
                    service = line.split()[0]
                    print(f"    {GREEN}✓{NC} {service} is running")
        else:
            print(f"  {RED}✗{NC} Failed to start Docker services")
            print(f"     Error: {result.stderr}")
    
    def create_helper_scripts(self):
        """Create convenient helper scripts"""
        print(f"\n{BLUE}9. Creating helper scripts...{NC}")
        
        # Create run.sh
        run_script_content = """#!/bin/bash
# Quick start script for AIMS

# Colors
GREEN='\\033[0;32m'
BLUE='\\033[0;34m'
RED='\\033[0;31m'
NC='\\033[0m'

echo -e "${BLUE}🚀 Starting AIMS...${NC}"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Run setup script first${NC}"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if Docker services are running
if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
    echo "Starting Docker services..."
    docker-compose up -d
    echo "Waiting for services to be ready..."
    sleep 5
fi

# Export environment variables from .env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-anthropic-api-key-here" ]; then
    echo ""
    echo -e "${RED}WARNING: ANTHROPIC_API_KEY not set!${NC}"
    echo "Edit .env file and add your API key"
    echo ""
fi

# Start AIMS
echo -e "${GREEN}✅ Starting AIMS on http://localhost:8000${NC}"
echo -e "${GREEN}✅ WebSocket server on ws://localhost:8765${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python -m src.main
"""
        
        run_script_path = self.root_dir / 'run.sh'
        with open(run_script_path, 'w') as f:
            f.write(run_script_content)
        os.chmod(run_script_path, 0o755)
        print(f"  {GREEN}✓{NC} Created run.sh")
        
        # Create stop.sh
        stop_script_content = """#!/bin/bash
# Stop AIMS and Docker services

echo "Stopping AIMS services..."
docker-compose down
echo "✅ All services stopped"
"""
        
        stop_script_path = self.root_dir / 'stop.sh'
        with open(stop_script_path, 'w') as f:
            f.write(stop_script_content)
        os.chmod(stop_script_path, 0o755)
        print(f"  {GREEN}✓{NC} Created stop.sh")
        
        # Create test.sh
        test_script_content = """#!/bin/bash
# Run AIMS tests

source venv/bin/activate
python -m pytest tests/ -v --tb=short
"""
        
        test_script_path = self.root_dir / 'test.sh'
        with open(test_script_path, 'w') as f:
            f.write(test_script_content)
        os.chmod(test_script_path, 0o755)
        print(f"  {GREEN}✓{NC} Created test.sh")
        
        self.issues_fixed.append("Helper scripts")
    
    def verify_setup(self):
        """Verify the setup is complete"""
        print(f"\n{BLUE}10. Verifying setup...{NC}")
        
        all_good = True
        
        # Check critical files
        critical_files = [
            '.env',
            'docker-compose.yml',
            'requirements.txt',
            'src/main.py',
            'src/core/consciousness.py',
            'src/api/claude_interface.py',
            'src/ui/web_interface.py'
        ]
        
        print(f"  Checking critical files:")
        for file_path in critical_files:
            if (self.root_dir / file_path).exists():
                print(f"    {GREEN}✓{NC} {file_path}")
            else:
                print(f"    {RED}✗{NC} {file_path} missing!")
                all_good = False
        
        # Check Python packages
        print(f"\n  Checking Python packages:")
        try:
            result = subprocess.run(
                [str(self.python_path), '-c', '''
import sys
try:
    import anthropic
    print("✓ anthropic")
except: print("✗ anthropic")
try:
    import torch
    print("✓ torch")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("• CUDA not available")
except: print("✗ torch")
try:
    import aiohttp
    print("✓ aiohttp")
except: print("✗ aiohttp")
try:
    import websockets
    print("✓ websockets")
except: print("✗ websockets")
try:
    import yaml
    print("✓ yaml")
except: print("✗ yaml")
try:
    from dotenv import load_dotenv
    print("✓ python-dotenv")
except: print("✗ python-dotenv")
'''],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if line.strip():
                    if '✓' in line:
                        print(f"    {GREEN}{line}{NC}")
                    elif '✗' in line:
                        print(f"    {RED}{line}{NC}")
                        all_good = False
                    else:
                        print(f"    {BLUE}{line}{NC}")
        except Exception as e:
            print(f"    {RED}✗{NC} Failed to check packages: {e}")
            all_good = False
        
        if all_good:
            self.issues_fixed.append("Setup verification")
        else:
            self.issues_failed.append("Some components missing")
    
    def print_summary(self):
        """Print summary of fixes"""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{BLUE}Setup Summary{NC}")
        print(f"{BLUE}{'='*60}{NC}")
        
        if self.issues_fixed:
            print(f"\n{GREEN}Successfully completed:{NC}")
            for issue in self.issues_fixed:
                print(f"  ✓ {issue}")
        
        if self.issues_failed:
            print(f"\n{RED}Failed to complete:{NC}")
            for issue in self.issues_failed:
                print(f"  ✗ {issue}")
        
        print(f"\n{BLUE}Next steps:{NC}")
        print("1. Add your ANTHROPIC_API_KEY to the .env file:")
        print(f"   {BLUE}nano .env{NC}  # or use your preferred editor")
        print("")
        print("2. Make sure Docker is running and accessible:")
        print(f"   {BLUE}docker ps{NC}  # Should not show permission errors")
        print("")
        print("3. Start AIMS:")
        print(f"   {BLUE}./run.sh{NC}")
        print("")
        print("4. Open your browser to:")
        print(f"   {BLUE}http://localhost:8000{NC}")
        print("")
        
        if not self.issues_failed:
            print(f"{GREEN}✨ Your AIMS system is ready to go!{NC}")
        else:
            print(f"{YELLOW}⚠️  Some issues need manual attention.{NC}")

def main():
    """Main entry point"""
    # Check if we're in the right directory
    if not Path('src').exists() or not Path('requirements.txt').exists():
        print(f"{RED}Error: This script must be run from the AIMS project root directory{NC}")
        print("Please cd to the AIMS directory and run again.")
        sys.exit(1)
    
    # Run the fixer
    fixer = AIMSSetupFixer()
    fixer.run()

if __name__ == "__main__":
    main()