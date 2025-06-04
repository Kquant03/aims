#!/bin/bash
# fix_and_run.sh - Fix all issues and run AIMS

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🔧 Fixing AIMS and installing dependencies${NC}"
echo "=========================================="

# Step 1: Fix the consciousness.py logger issue
echo -e "\n${BLUE}Step 1: Fixing consciousness.py...${NC}"
sed -i 's/logger\.warning("Flash Attention not available, using standard attention")/print("WARNING: Flash Attention not available, using standard attention")/' src/core/consciousness.py
echo -e "${GREEN}✅ Fixed logger issue${NC}"

# Step 2: Install critical missing packages
echo -e "\n${BLUE}Step 2: Installing missing packages...${NC}"
echo "This may take a minute..."

# Install in batches to avoid conflicts
echo "Installing web framework packages..."
./venv/bin/pip install -q aiohttp-session aiohttp-jinja2 aiohttp-cors

echo "Installing database packages..."
./venv/bin/pip install -q redis asyncpg psycopg2-binary

echo "Installing core packages..."
./venv/bin/pip install -q websockets msgpack python-dateutil prometheus-client

# Try to install optional packages (don't fail if they don't work)
echo "Installing optional packages..."
./venv/bin/pip install -q qdrant-client 2>/dev/null || true
./venv/bin/pip install -q transformers 2>/dev/null || true

echo -e "${GREEN}✅ Packages installed${NC}"

# Step 3: Check environment
echo -e "\n${BLUE}Step 3: Checking environment...${NC}"

# Check .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}⚠️  Created .env from template - add your API keys!${NC}"
    fi
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

# Load environment
set -a
source .env 2>/dev/null || true
set +a

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-anthropic-api-key-here" ]; then
    echo -e "${YELLOW}⚠️  ANTHROPIC_API_KEY not set - some features won't work${NC}"
else
    echo -e "${GREEN}✅ ANTHROPIC_API_KEY is set${NC}"
fi

# Check Docker
if docker ps >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker is running${NC}"
    
    # Start services if needed
    if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
        echo "Starting Docker services..."
        docker-compose up -d >/dev/null 2>&1
        sleep 3
    fi
else
    echo -e "${YELLOW}⚠️  Docker not accessible${NC}"
fi

# Step 4: Run AIMS
echo -e "\n${BLUE}Step 4: Starting AIMS...${NC}"
echo "========================================"
echo -e "${GREEN}Web interface: http://localhost:8000${NC}"
echo -e "${GREEN}WebSocket: ws://localhost:8765${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run with proper error handling
exec ./venv/bin/python -m src.main 2>&1