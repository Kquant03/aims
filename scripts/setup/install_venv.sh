#!/bin/bash
# install_venv.sh - Complete virtual environment setup for AIMS with Python 3.10

set -e  # Exit on any error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}AIMS Complete Virtual Environment Setup${NC}"
echo "========================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Ensure Python 3.10 is installed
echo -e "\n${BLUE}Step 1: Checking Python 3.10${NC}"
if command_exists python3.10; then
    echo -e "${GREEN}✅ Python 3.10 is installed${NC}"
    PYTHON_CMD="python3.10"
else
    echo -e "${YELLOW}⚠️  Python 3.10 not found${NC}"
    
    # Detect OS and install Python 3.10
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        if [[ "$ID" == "ubuntu" ]]; then
            echo "Installing Python 3.10 on Ubuntu..."
            
            # Add deadsnakes PPA for Python 3.10
            echo "Adding deadsnakes PPA for Python 3.10..."
            sudo apt update
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
            
            # Install Python 3.10
            echo "Installing Python 3.10 packages..."
            sudo apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
            
            # Verify installation
            if command_exists python3.10; then
                echo -e "${GREEN}✅ Python 3.10 successfully installed${NC}"
                PYTHON_CMD="python3.10"
            else
                echo -e "${RED}❌ Failed to install Python 3.10${NC}"
                exit 1
            fi
        else
            echo -e "${RED}Please install Python 3.10 manually for your OS${NC}"
            exit 1
        fi
    fi
fi

# Step 2: Create virtual environment
echo -e "\n${BLUE}Step 2: Creating virtual environment${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
    read -p "Delete and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "Removed old virtual environment"
    else
        echo "Using existing virtual environment"
    fi
fi

if [ ! -d "venv" ]; then
    echo "Creating new virtual environment with Python 3.10..."
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Step 3: Activate virtual environment
echo -e "\n${BLUE}Step 3: Activating virtual environment${NC}"
source venv/bin/activate

# Verify we're in the venv and using Python 3.10
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
    echo "   Location: $VIRTUAL_ENV"
    
    VENV_PYTHON_VERSION=$(python --version | cut -d' ' -f2)
    echo "   Python version: $VENV_PYTHON_VERSION"
else
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi

# Step 4: Upgrade pip and install build tools
echo -e "\n${BLUE}Step 4: Upgrading pip and build tools${NC}"
python -m pip install --upgrade pip wheel setuptools
echo -e "${GREEN}✅ Pip and build tools upgraded${NC}"

# Step 5: Install PyTorch with CUDA support
echo -e "\n${BLUE}Step 5: Installing PyTorch${NC}"
if command_exists nvidia-smi; then
    echo "NVIDIA GPU detected, installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo -e "${GREEN}✅ PyTorch installed${NC}"

# Step 6: Install all requirements
echo -e "\n${BLUE}Step 6: Installing all project requirements${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✅ All requirements installed${NC}"
else
    echo -e "${RED}❌ requirements.txt not found${NC}"
    exit 1
fi

# Step 7: Verify installation
echo -e "\n${BLUE}Step 7: Verifying installation${NC}"
python -c "
import sys
print(f'Python: {sys.version}')
print(f'Executable: {sys.executable}')

# Check critical imports
try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('❌ PyTorch not installed')

try:
    import anthropic
    print('✅ Anthropic SDK installed')
except ImportError:
    print('❌ Anthropic SDK not installed')

try:
    import aiohttp
    print('✅ aiohttp installed')
except ImportError:
    print('❌ aiohttp not installed')

# Count total packages
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
if result.returncode == 0:
    package_count = len(result.stdout.strip().split('\\n')) - 2
    print(f'\\nTotal packages installed: {package_count}')
"

# Step 8: Create activation script
echo -e "\n${BLUE}Step 8: Creating convenience scripts${NC}"
cat > activate_aims_env.sh << 'EOF'
#!/bin/bash
# Quick activation script for AIMS virtual environment

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "AIMS virtual environment activated"
    echo "Python: $(python --version)"
    echo "Location: $VIRTUAL_ENV"
else
    echo "Virtual environment not found!"
    echo "Run ./install_venv.sh first"
fi
EOF

chmod +x activate_aims_env.sh
echo -e "${GREEN}✅ Created activate_aims_env.sh${NC}"

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Virtual environment setup complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "The virtual environment contains ALL project dependencies:"
echo "- Python 3.10"
echo "- PyTorch (with CUDA support if GPU available)"
echo "- All packages from requirements.txt"
echo ""
echo "To use AIMS:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run AIMS: python -m src.main"
echo ""
echo "Or use the convenience script:"
echo "   source activate_aims_env.sh"
echo ""
echo -e "${YELLOW}Note: This virtual environment is completely self-contained.${NC}"
echo -e "${YELLOW}No system Python packages are needed or used.${NC}"