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
