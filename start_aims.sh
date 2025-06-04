#!/bin/bash
# Quick start script for AIMS

echo "Starting AIMS..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if needed
pip install -q -r requirements.txt

# Start Docker services if not running
if ! docker-compose ps 2>/dev/null | grep -q "Up"; then
    echo "Starting Docker services..."
    docker-compose up -d
    sleep 5
fi

# Run AIMS
echo "Starting AIMS on http://localhost:8000"
python -m src.main
