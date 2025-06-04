#!/bin/bash
# fix_venv.sh - Fix the virtual environment and reinstall all packages

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔧 Fixing AIMS Virtual Environment${NC}"
echo "===================================="

# Step 1: Remove broken venv
if [ -d "venv" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf venv
fi

# Step 2: Create new venv with Python 3.10
echo -e "${BLUE}Creating fresh virtual environment...${NC}"

# Find Python 3.10
PYTHON_CMD=""
for cmd in python3.10 python3; do
    if command -v $cmd >/dev/null 2>&1; then
        VERSION=$($cmd --version 2>&1 | awk '{print $2}')
        if [[ $VERSION == 3.10.* ]]; then
            PYTHON_CMD=$cmd
            echo -e "${GREEN}✅ Found Python $VERSION${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}❌ Python 3.10 not found!${NC}"
    echo "Please install Python 3.10:"
    echo "  sudo apt install python3.10 python3.10-venv"
    exit 1
fi

# Create venv
$PYTHON_CMD -m venv venv
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to create virtual environment${NC}"
    echo "Make sure python3.10-venv is installed:"
    echo "  sudo apt install python3.10-venv"
    exit 1
fi

echo -e "${GREEN}✅ Virtual environment created${NC}"

# Step 3: Upgrade pip and install wheel
echo -e "${BLUE}Upgrading pip and installing build tools...${NC}"
./venv/bin/python -m pip install --upgrade pip wheel setuptools

# Step 4: Install critical packages first
echo -e "${BLUE}Installing critical packages...${NC}"
./venv/bin/pip install \
    psutil \
    cryptography \
    python-dotenv \
    pyyaml \
    aiohttp \
    anthropic

# Step 5: Install PyTorch
echo -e "${BLUE}Installing PyTorch...${NC}"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU detected, installing CUDA-enabled PyTorch..."
    ./venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "No GPU detected, installing CPU-only PyTorch..."
    ./venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Step 6: Install remaining requirements
echo -e "${BLUE}Installing all requirements...${NC}"
./venv/bin/pip install -r requirements.txt

# Step 7: Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
./venv/bin/python -c "
import sys
print(f'Python: {sys.version}')

packages = {
    'psutil': 'System utilities',
    'anthropic': 'Claude API',
    'torch': 'PyTorch',
    'aiohttp': 'Async HTTP',
    'redis': 'Redis client',
    'asyncpg': 'PostgreSQL client',
    'transformers': 'Hugging Face',
    'websockets': 'WebSocket support'
}

all_good = True
for package, desc in packages.items():
    try:
        __import__(package)
        print(f'✅ {package:<15} - {desc}')
    except ImportError:
        print(f'❌ {package:<15} - {desc} (MISSING)')
        all_good = False

if not all_good:
    print('\\n⚠️  Some packages failed to install')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✨ Virtual environment fixed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Make sure Docker is running:"
    echo "   docker-compose up -d"
    echo ""
    echo "2. Use the fixed run script:"
    echo "   ./run_fixed.sh"
    echo ""
    echo "Or manually:"
    echo "   ./venv/bin/python -m src.main"
else
    echo ""
    echo -e "${RED}❌ Some packages failed to install${NC}"
    echo "Check the error messages above"
fi