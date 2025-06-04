#!/bin/bash
# AIMS Debug and Fix Script

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}AIMS Debug and Fix${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 1. Check virtual environment
echo -e "${BLUE}1. Checking virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "  ${GREEN}✓${NC} venv directory exists"
    
    # Check Python in venv
    if [ -f "venv/bin/python" ]; then
        echo -e "  ${GREEN}✓${NC} venv Python found"
        VENV_PYTHON=$(venv/bin/python --version 2>&1)
        echo "     Version: $VENV_PYTHON"
    else
        echo -e "  ${RED}✗${NC} venv Python not found!"
    fi
    
    # Check pip in venv
    if [ -f "venv/bin/pip" ]; then
        echo -e "  ${GREEN}✓${NC} venv pip found"
    else
        echo -e "  ${RED}✗${NC} venv pip not found!"
    fi
else
    echo -e "  ${RED}✗${NC} No virtual environment found!"
    exit 1
fi

# 2. List installed packages in venv
echo -e "\n${BLUE}2. Checking installed packages in venv...${NC}"
echo "  Installed packages:"
venv/bin/pip list | head -20
echo "  ..."
TOTAL_PACKAGES=$(venv/bin/pip list | wc -l)
echo "  Total packages: $((TOTAL_PACKAGES - 2))"

# 3. Check specific packages
echo -e "\n${BLUE}3. Checking critical packages...${NC}"
PACKAGES=("psutil" "websockets" "anthropic" "torch" "aiohttp" "aiohttp-session" "cryptography" "python-dotenv" "pyyaml")

for pkg in "${PACKAGES[@]}"; do
    if venv/bin/pip show "$pkg" >/dev/null 2>&1; then
        VERSION=$(venv/bin/pip show "$pkg" | grep Version | cut -d' ' -f2)
        echo -e "  ${GREEN}✓${NC} $pkg==$VERSION"
    else
        echo -e "  ${RED}✗${NC} $pkg NOT INSTALLED"
    fi
done

# 4. Fix missing packages
echo -e "\n${BLUE}4. Installing missing packages...${NC}"

# First, ensure websockets installs correctly
echo "  Installing websockets..."
venv/bin/pip install --upgrade websockets

# Install psutil
echo "  Installing psutil..."
venv/bin/pip install psutil

# Install any other missing critical packages
echo "  Installing other missing packages..."
venv/bin/pip install aiofiles aioboto3 mem0ai redis psycopg2-binary pgvector qdrant-client neo4j websockets pyyaml python-dotenv anthropic openai aiohttp aiohttp-cors "aiohttp-session[secure]" cryptography

# 5. Create a fixed run.sh script
echo -e "\n${BLUE}5. Creating fixed run.sh script...${NC}"
cat > run_fixed.sh << 'EOF'
#!/bin/bash
# Fixed run script for AIMS

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting AIMS...${NC}"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Run setup script first${NC}"
    exit 1
fi

# Use absolute path for Python
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Activate venv (for environment variables)
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
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-anthropic-api-key-here" ]; then
    echo ""
    echo -e "${YELLOW}WARNING: ANTHROPIC_API_KEY not set!${NC}"
    echo "Edit .env file and add your API key"
    echo ""
fi

# Use the venv Python explicitly
echo -e "${GREEN}✅ Starting AIMS on http://localhost:8000${NC}"
echo -e "${GREEN}✅ WebSocket server on ws://localhost:8765${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run with full path to venv Python
"${SCRIPT_DIR}/venv/bin/python" -m src.main
EOF

chmod +x run_fixed.sh
echo -e "  ${GREEN}✓${NC} Created run_fixed.sh"

# 6. Test imports
echo -e "\n${BLUE}6. Testing imports...${NC}"
venv/bin/python -c "
import sys
print(f'Python executable: {sys.executable}')
print(f'Python version: {sys.version}')
print(f'Python path:')
for p in sys.path[:5]:
    print(f'  {p}')

print('\nTesting imports:')
try:
    import psutil
    print('✓ psutil')
except ImportError as e:
    print(f'✗ psutil: {e}')

try:
    import websockets
    print('✓ websockets')
except ImportError as e:
    print(f'✗ websockets: {e}')

try:
    import anthropic
    print('✓ anthropic')
except ImportError as e:
    print(f'✗ anthropic: {e}')

try:
    import aiohttp
    print('✓ aiohttp')
except ImportError as e:
    print(f'✗ aiohttp: {e}')

try:
    from dotenv import load_dotenv
    print('✓ python-dotenv')
except ImportError as e:
    print(f'✗ python-dotenv: {e}')

try:
    import torch
    print(f'✓ torch (CUDA: {torch.cuda.is_available()})')
except ImportError as e:
    print(f'✗ torch: {e}')
"

# 7. Fix the original run.sh
echo -e "\n${BLUE}7. Fixing original run.sh...${NC}"
cp run.sh run.sh.backup
cat > run.sh << 'EOF'
#!/bin/bash
# Quick start script for AIMS

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting AIMS...${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Run setup script first${NC}"
    exit 1
fi

# Check if Docker services are running
if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
    echo "Starting Docker services..."
    docker-compose up -d
    echo "Waiting for services to be ready..."
    sleep 5
fi

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-anthropic-api-key-here" ]; then
    echo ""
    echo -e "${YELLOW}WARNING: ANTHROPIC_API_KEY not set!${NC}"
    echo "Edit .env file and add your API key"
    echo ""
fi

# Start AIMS using venv Python directly
echo -e "${GREEN}✅ Starting AIMS on http://localhost:8000${NC}"
echo -e "${GREEN}✅ WebSocket server on ws://localhost:8765${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Use absolute path to venv Python
exec "${SCRIPT_DIR}/venv/bin/python" -m src.main
EOF

chmod +x run.sh
echo -e "  ${GREEN}✓${NC} Fixed run.sh"

echo -e "\n${BLUE}================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Fixes applied:"
echo "1. Installed missing packages (psutil, websockets, etc.)"
echo "2. Created run_fixed.sh with proper venv handling"
echo "3. Fixed original run.sh to use venv Python directly"
echo ""
echo -e "${GREEN}Now try running AIMS with:${NC}"
echo "  ./run.sh"
echo ""
echo "If that still fails, try:"
echo "  ./run_fixed.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python -m src.main"