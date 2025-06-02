#!/bin/bash
# setup_ubuntu.sh - Ubuntu 24.04 setup script for AIMS

set -e  # Exit on error

echo "🚀 AIMS Ubuntu Setup Script"
echo "=========================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo ""
echo "🐍 Checking Python version..."
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
PYTHON_CMD="python3"

# Ubuntu 24.04 comes with Python 3.12, but we recommend 3.10 for compatibility
if command_exists python3.10; then
    PYTHON_CMD="python3.10"
    echo -e "${GREEN}✅ Python 3.10 found (recommended)${NC}"
elif [[ "$PYTHON_VERSION" == "3.11" ]] || [[ "$PYTHON_VERSION" == "3.12" ]]; then
    echo -e "${YELLOW}⚠️  Python $PYTHON_VERSION detected. Python 3.10 is recommended for best compatibility.${NC}"
    read -p "Install Python 3.10? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        sudo apt install -y python3.10 python3.10-venv python3.10-dev
        PYTHON_CMD="python3.10"
    fi
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Update system packages
echo ""
echo "📦 Updating system packages..."
sudo apt update

# Install system dependencies
echo ""
echo "🔧 Installing system dependencies..."
PACKAGES=(
    "python3.10"
    "python3.10-venv"
    "python3.10-dev"
    "python3-pip"
    "build-essential"
    "postgresql-client"
    "redis-tools"
    "curl"
    "git"
    "libpq-dev"
    "nvidia-cuda-toolkit"  # For GPU support
)

for package in "${PACKAGES[@]}"; do
    if dpkg -l | grep -q "^ii  $package"; then
        echo -e "${GREEN}✅ $package already installed${NC}"
    else
        echo "Installing $package..."
        sudo apt install -y "$package"
    fi
done

# Check Docker installation
echo ""
echo "🐳 Checking Docker installation..."
if command_exists docker; then
    echo -e "${GREEN}✅ Docker is installed${NC}"
else
    echo -e "${YELLOW}⚠️  Docker not found${NC}"
    read -p "Install Docker? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        # Install Docker
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        echo -e "${GREEN}✅ Docker installed${NC}"
        echo -e "${YELLOW}⚠️  You may need to log out and back in for Docker permissions${NC}"
    fi
fi

# Check Docker Compose installation  
echo ""
echo "🐳 Checking Docker Compose installation..."
if command_exists docker-compose; then
    echo -e "${GREEN}✅ Docker Compose is installed${NC}"
else
    echo "Installing Docker Compose..."
    sudo apt install -y docker-compose
fi

# Check NVIDIA drivers (optional)
echo ""
echo "🎮 Checking NVIDIA GPU..."
if command_exists nvidia-smi; then
    echo -e "${GREEN}✅ NVIDIA drivers installed${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠️  NVIDIA drivers not found - GPU acceleration will not be available${NC}"
fi

# Create project structure
echo ""
echo "📁 Creating project structure..."
mkdir -p data/{states,backups,memories,uploads}
mkdir -p logs
mkdir -p src/ui/{templates,static}

# Set up Python virtual environment
echo ""
echo "🐍 Setting up Python virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "Delete and recreate with Python 3.10? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "Removed old virtual environment"
    fi
fi

if [ ! -d "venv" ]; then
    # Create venv specifically with Python 3.10
    if command_exists python3.10; then
        echo "Creating virtual environment with Python 3.10..."
        python3.10 -m venv venv
        echo -e "${GREEN}✅ Virtual environment created with Python 3.10${NC}"
    else
        echo -e "${RED}❌ Python 3.10 not found. Please install it first.${NC}"
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Verify we're using Python 3.10 in the venv
VENV_PYTHON_VERSION=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$VENV_PYTHON_VERSION" == "3.10" ]]; then
    echo -e "${GREEN}✅ Virtual environment is using Python 3.10${NC}"
else
    echo -e "${RED}❌ Virtual environment is using Python $VENV_PYTHON_VERSION instead of 3.10${NC}"
    echo "Please recreate the virtual environment with Python 3.10"
    exit 1
fi

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install Python requirements
echo ""
echo "📦 Installing Python requirements..."
if [ -f "requirements.txt" ]; then
    # Make sure we're in the virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo "Installing all project dependencies in virtual environment..."
        
        # Install build dependencies for compiled packages
        pip install --upgrade pip wheel setuptools
        
        # Install PyTorch with CUDA support first (for RTX 3090)
        echo "Installing PyTorch with CUDA 12.1 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        
        # Install all other requirements
        pip install -r requirements.txt
        
        # Verify critical packages
        echo ""
        echo "Verifying installation..."
        python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
except ImportError:
    print('PyTorch not installed')
    
try:
    import anthropic
    print('Anthropic SDK: Installed')
except ImportError:
    print('Anthropic SDK: Not installed')
"
        
        echo -e "${GREEN}✅ All Python dependencies installed in virtual environment${NC}"
    else
        echo -e "${RED}❌ Not in virtual environment! Aborting.${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
fi

# Create .env file if it doesn't exist
echo ""
echo "🔧 Setting up environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env file from template${NC}"
        echo -e "${YELLOW}⚠️  Please edit .env and add your API keys${NC}"
    else
        echo -e "${RED}❌ .env.example not found${NC}"
    fi
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Set permissions
echo ""
echo "🔒 Setting file permissions..."
chmod +x scripts/*.sh 2>/dev/null || true
chmod 755 data logs

# Start Docker services
echo ""
read -p "Start Docker services now? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    echo "Starting Docker services..."
    docker-compose up -d
    echo -e "${GREEN}✅ Docker services started${NC}"
    
    # Wait for services to be ready
    echo "Waiting for services to be ready..."
    sleep 10
    
    # Check service status
    docker-compose ps
fi

# Final summary
echo ""
echo "✨ Setup complete!"
echo ""
echo "📦 Virtual Environment Summary:"
echo "  Location: $(pwd)/venv"
echo "  Python version: $(python --version)"
echo "  Packages installed: $(pip list | wc -l) packages"
echo ""
echo "🚀 To use AIMS:"
echo "1. Always activate the virtual environment first:"
echo "   source venv/bin/activate"
echo ""
echo "2. Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "3. Run AIMS:"
echo "   python -m src.main"
echo ""
echo "4. Open http://localhost:8000 in your browser"
echo ""
echo "💡 Tips:"
echo "- The virtual environment contains ALL project dependencies"
echo "- No system-wide Python packages are needed"
echo "- To deactivate the venv, type: deactivate"
echo "- To see installed packages: pip list"
echo ""
echo "For quick activation, you can also use:"
echo "  source activate_aims.sh"