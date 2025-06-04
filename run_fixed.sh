#!/bin/bash
# run_aims.sh - Quick script to run AIMS with minimal dependencies

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting AIMS (Quick Start)${NC}"
echo "=============================="

# Install only the critical missing packages
echo -e "${YELLOW}Installing critical missing packages...${NC}"
./venv/bin/pip install -q redis websockets aiohttp-session aiohttp-jinja2

# Check .env file
if [ ! -f ".env" ]; then
    echo -e "${RED}No .env file found!${NC}"
    echo "Creating .env from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        echo -e "${RED}No .env.example found either!${NC}"
        exit 1
    fi
fi

# Export environment variables
set -a
source .env
set +a

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "your-anthropic-api-key-here" ]; then
    echo -e "${YELLOW}WARNING: ANTHROPIC_API_KEY not set!${NC}"
    echo "Edit .env file and add your API key"
    echo ""
fi

# Check Docker
if docker ps >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker is accessible${NC}"
    
    # Start services if needed
    if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
        echo "Starting Docker services..."
        docker-compose up -d
        sleep 5
    fi
else
    echo -e "${YELLOW}⚠️  Docker not accessible - some features may not work${NC}"
fi

# Run AIMS
echo -e "${GREEN}Starting AIMS...${NC}"
echo "Web interface: http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Use the venv Python directly
exec ./venv/bin/python -m src.main