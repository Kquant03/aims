#!/bin/bash
# setup.sh - Comprehensive AIMS setup script for Ubuntu

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 AIMS - Autonomous Intelligent Memory System${NC}"
echo -e "${BLUE}   Comprehensive Setup Script${NC}"
echo "=============================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check Python version
echo -e "\n${BLUE}Checking Python version...${NC}"
if command_exists python3.10; then
    echo -e "${GREEN}✅ Python 3.10 found${NC}"
    PYTHON_CMD="python3.10"
else
    echo -e "${YELLOW}⚠️  Python 3.10 not found${NC}"
    echo "Would you like to install Python 3.10? [Y/n]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]] || [[ -z "$response" ]]; then
        scripts/setup/fix_python310_ubuntu24.sh
        PYTHON_CMD="python3.10"
    else
        PYTHON_CMD="python3"
    fi
fi

# 2. Set up virtual environment
echo -e "\n${BLUE}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate venv
source venv/bin/activate

# 3. Install Python dependencies
echo -e "\n${BLUE}Installing Python dependencies...${NC}"
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install aioboto3  # Additional dependency
echo -e "${GREEN}✅ Dependencies installed${NC}"

# 4. Set up environment variables
echo -e "\n${BLUE}Setting up environment variables...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Created .env file - please add your API keys${NC}"
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

# Generate proper SESSION_SECRET if needed
if grep -q "dev-secret-key-change-in-production" .env; then
    echo "Generating secure session key..."
    NEW_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
    sed -i "s/SESSION_SECRET=.*/SESSION_SECRET=$NEW_KEY/" .env
    echo -e "${GREEN}✅ Generated secure session key${NC}"
fi

# 5. Check Docker
echo -e "\n${BLUE}Checking Docker...${NC}"
if ! docker ps >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Docker not accessible${NC}"
    echo "Run: sudo usermod -aG docker $USER && newgrp docker"
else
    echo -e "${GREEN}✅ Docker is accessible${NC}"
    
    # Start services
    echo "Start Docker services? [Y/n]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]] || [[ -z "$response" ]]; then
        docker-compose up -d
        echo -e "${GREEN}✅ Docker services started${NC}"
    fi
fi

# 6. Create necessary directories
echo -e "\n${BLUE}Creating directories...${NC}"
mkdir -p data/{states,backups,memories,uploads}
mkdir -p logs
mkdir -p src/ui/{templates,static}
echo -e "${GREEN}✅ Directories created${NC}"

# 7. Run verification
echo -e "\n${BLUE}Running setup verification...${NC}"
if [ -f "scripts/utils/verify_setup.py" ]; then
    python scripts/utils/verify_setup.py
else
    echo "Verification script will be available after cleanup"
fi

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo ""
echo "To start AIMS:"
echo "1. source venv/bin/activate"
echo "2. python -m src.main"
echo ""
echo "Or simply run: ./run.sh"
echo ""
echo "Then open: http://localhost:8000"
