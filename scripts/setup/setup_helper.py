import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
import getpass

class AIMSSetupHelper:
    """Interactive setup assistant for AIMS"""
    
    def __init__(self):
        self.root_dir = Path.cwd()
        self.config = {}
        
    def run(self):
        """Run the setup process"""
        print("🚀 AIMS Setup Assistant")
        print("=" * 50)
        
        # Check Python version
        if not self.check_python_version():
            return
        
        # Check GPU
        self.check_gpu()
        
        # Setup environment
        self.setup_environment()
        
        # Check Docker
        self.check_docker()
        
        # Configure API keys
        self.configure_api_keys()
        
        # Create directories
        self.create_directories()
        
        # Install dependencies
        self.install_dependencies()
        
        # Initialize databases
        self.initialize_databases()
        
        print("\n✅ Setup complete!")
        print("\nTo start AIMS, run:")
        print("  python -m src.main")
        print("\nThen open http://localhost:8000 in your browser")
    
    def check_python_version(self):
        """Check if Python version is 3.10+"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            print("❌ Python 3.10+ is required")
            print(f"   Current version: {version.major}.{version.minor}")
            return False
        print(f"✅ Python {version.major}.{version.minor} detected")
        return True
    
    def check_gpu(self):
        """Check for GPU availability"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
                
                if "3090" in gpu_name:
                    print("   Perfect! RTX 3090 is the recommended GPU")
                elif gpu_memory >= 20:
                    print("   Good! Your GPU has sufficient memory")
                else:
                    print("   ⚠️  Limited GPU memory - may need to adjust settings")
            else:
                print("⚠️  No GPU detected - will run on CPU (slower)")
        except ImportError:
            print("⚠️  PyTorch not installed yet - GPU detection will happen after installation")
    
    def setup_environment(self):
        """Set up environment variables"""
        print("\n📝 Setting up environment variables...")
        
        env_file = self.root_dir / ".env"
        env_example = self.root_dir / ".env.example"
        
        if not env_file.exists() and env_example.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
        elif env_file.exists():
            print("✅ .env file already exists")
        else:
            print("❌ .env.example not found - creating basic .env")
            self.create_basic_env()
    
    def configure_api_keys(self):
        """Configure API keys"""
        print("\n🔑 Configuring API keys...")
        
        env_file = self.root_dir / ".env"
        
        # Read existing env
        env_vars = {}
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value
        
        # Check Anthropic API key
        if not env_vars.get('ANTHROPIC_API_KEY') or env_vars['ANTHROPIC_API_KEY'] == 'your-anthropic-api-key-here':
            api_key = input("\nEnter your Anthropic API key (required): ").strip()
            if api_key:
                env_vars['ANTHROPIC_API_KEY'] = api_key
                print("✅ Anthropic API key configured")
            else:
                print("⚠️  No API key provided - you'll need to add it to .env manually")
        else:
            print("✅ Anthropic API key already configured")
        
        # Optional OpenAI key
        if not env_vars.get('OPENAI_API_KEY') or env_vars['OPENAI_API_KEY'] == 'your-openai-api-key-for-embeddings':
            response = input("\nDo you have an OpenAI API key for embeddings? (optional) [y/N]: ")
            if response.lower() == 'y':
                api_key = input("Enter your OpenAI API key: ").strip()
                if api_key:
                    env_vars['OPENAI_API_KEY'] = api_key
                    print("✅ OpenAI API key configured")
        
        # Write back to .env
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
    
    def check_docker(self):
        """Check if Docker is installed and running"""
        print("\n🐳 Checking Docker...")
        
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker is installed")
                
                # Check if Docker is running
                result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Docker is running")
                    
                    # Start services
                    response = input("\nStart Docker services now? [Y/n]: ")
                    if response.lower() != 'n':
                        print("Starting Docker services...")
                        subprocess.run(['docker-compose', 'up', '-d'])
                        print("✅ Docker services started")
                else:
                    print("❌ Docker is not running - please start Docker")
            else:
                print("❌ Docker not found - please install Docker")
        except FileNotFoundError:
            print("❌ Docker not found - please install Docker")
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")
        
        directories = [
            'data/states',
            'data/backups',
            'data/memories',
            'data/uploads',
            'logs',
            'src/ui/templates',
            'src/ui/static'
        ]
        
        for dir_path in directories:
            path = self.root_dir / dir_path
            path.mkdir(parents=True, exist_ok=True)
        
        print("✅ Directories created")
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("\n📦 Installing Python dependencies...")
        
        response = input("Install Python dependencies now? [Y/n]: ")
        if response.lower() != 'n':
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
                print("✅ Dependencies installed")
            except subprocess.CalledProcessError:
                print("❌ Error installing dependencies - please run manually:")
                print("   pip install -r requirements.txt")
    
    def initialize_databases(self):
        """Initialize database schemas"""
        print("\n🗄️  Initializing databases...")
        
        # This would normally create database schemas
        # For now, just inform the user
        print("ℹ️  Databases will be initialized on first run")
        print("   Make sure PostgreSQL, Redis, and Qdrant are running")
    
    def create_basic_env(self):
        """Create a basic .env file"""
        basic_env = """# Required
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional
OPENAI_API_KEY=your-openai-api-key-for-embeddings
SESSION_SECRET=change-this-to-a-random-string

# Database Configuration
POSTGRES_PASSWORD=aims_secure_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=aims_memory
POSTGRES_USER=aims

# Redis Configuration  
REDIS_HOST=localhost
REDIS_PORT=6379

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# System Configuration
LOG_LEVEL=INFO
BACKUP_INTERVAL_HOURS=6
MAX_LOCAL_BACKUPS=7
"""
        
        with open('.env', 'w') as f:
            f.write(basic_env)


if __name__ == "__main__":
    helper = AIMSSetupHelper()
    helper.run()