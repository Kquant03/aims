#!/bin/bash
# run.sh - Quick start script for AIMS

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 Starting AIMS...${NC}"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Virtual environment not found. Run ./setup.sh first${NC}"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo -e "${RED}Warning: Not using Python 3.10 (using $PYTHON_VERSION)${NC}"
fi

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

# Start AIMS
echo -e "${GREEN}✅ Starting AIMS on http://localhost:8000${NC}"
echo -e "${GREEN}✅ WebSocket server on ws://localhost:8765${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python -m src.main
