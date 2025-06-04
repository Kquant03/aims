#!/bin/bash
# install_missing.sh - Install missing packages for AIMS

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Installing missing packages for AIMS${NC}"
echo "===================================="

# First, let's fix the faiss issue by installing faiss-cpu instead
echo -e "${BLUE}Installing faiss-cpu (instead of faiss-gpu)...${NC}"
./venv/bin/pip install faiss-cpu

# Install critical missing packages one by one
echo -e "${BLUE}Installing critical missing packages...${NC}"

# Redis
echo "Installing redis..."
./venv/bin/pip install redis

# PostgreSQL async driver
echo "Installing asyncpg..."
./venv/bin/pip install asyncpg

# WebSockets
echo "Installing websockets..."
./venv/bin/pip install websockets

# Transformers
echo "Installing transformers..."
./venv/bin/pip install transformers

# Other database packages
echo "Installing database packages..."
./venv/bin/pip install psycopg2-binary pgvector qdrant-client

# Memory packages
echo "Installing memory packages..."
./venv/bin/pip install mem0ai

# Additional web packages
echo "Installing web packages..."
./venv/bin/pip install aiohttp-cors aiohttp-session aiohttp-jinja2 aiofiles

# FastAPI (optional but useful)
echo "Installing FastAPI packages..."
./venv/bin/pip install fastapi uvicorn

# Data processing
echo "Installing data processing packages..."
./venv/bin/pip install pandas numpy scipy scikit-learn

# Additional packages
echo "Installing additional packages..."
./venv/bin/pip install \
    click \
    colorlog \
    pydantic-settings \
    msgpack \
    lz4 \
    prometheus-client \
    python-dateutil \
    pytz \
    tenacity \
    cachetools \
    diskcache \
    python-multipart \
    validators

# Language model packages (without conflicting versions)
echo "Installing language model packages..."
./venv/bin/pip install langchain langchain-community sentence-transformers

# Testing packages
echo "Installing testing packages..."
./venv/bin/pip install pytest pytest-asyncio pytest-cov pytest-mock

# Verify installation
echo -e "\n${BLUE}Verifying installation...${NC}"
./venv/bin/python -c "
import sys
print(f'Python: {sys.version}')
print()

packages = {
    'psutil': 'System utilities',
    'anthropic': 'Claude API',
    'torch': 'PyTorch',
    'aiohttp': 'Async HTTP',
    'redis': 'Redis client',
    'asyncpg': 'PostgreSQL client',
    'transformers': 'Hugging Face',
    'websockets': 'WebSocket support',
    'pandas': 'Data processing',
    'fastapi': 'FastAPI',
    'langchain': 'LangChain',
    'mem0ai': 'Memory management'
}

all_good = True
for package, desc in packages.items():
    try:
        __import__(package)
        print(f'✅ {package:<15} - {desc}')
    except ImportError:
        print(f'❌ {package:<15} - {desc} (MISSING)')
        all_good = False

print()
if all_good:
    print('✨ All packages successfully installed!')
else:
    print('⚠️  Some packages are still missing')
"

echo -e "\n${GREEN}Installation complete!${NC}"
echo "Now you can run AIMS with:"
echo "  ./venv/bin/python -m src.main"