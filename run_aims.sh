#!/bin/bash
# AIMS Quick Start Script

set -e

echo "🚀 Starting AIMS..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Creating from template..."
    python fix_startup_issues.py
    echo ""
    echo "⚠️  Please edit .env and add your ANTHROPIC_API_KEY, then run this script again."
    exit 1
fi

# Check if API key is set
if grep -q "ANTHROPIC_API_KEY=your_anthropic_api_key_here" .env; then
    echo "❌ ANTHROPIC_API_KEY not set in .env file"
    echo "   Please edit .env and add your API key"
    exit 1
fi

# Check if Docker is needed and running
if command -v docker &> /dev/null; then
    if ! docker info &> /dev/null; then
        echo "⚠️  Docker is installed but not running."
        echo "   Start Docker or run AIMS without database backends."
    else
        # Check if database containers are running
        if [ -f "docker-compose.yml" ]; then
            echo "🐳 Checking Docker services..."
            running_containers=$(docker-compose ps --services --filter "status=running" 2>/dev/null || echo "")
            
            if [ -z "$running_containers" ]; then
                echo "📦 Starting database services..."
                docker-compose up -d postgres redis qdrant
                echo "⏳ Waiting for services to be ready..."
                sleep 5
            else
                echo "✅ Docker services are running"
            fi
        fi
    fi
fi

# Clear screen for clean start
clear

# Run AIMS
echo "Starting AIMS..."
python src/main.py