#!/bin/bash

echo "🚀 Starting AIMS System..."

# Start Docker services
echo "📦 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services..."
sleep 10

# Check if services are running
docker-compose ps

# Activate virtual environment
source venv/bin/activate

# Build frontend
echo "🎨 Building frontend..."
cd src/ui && npm run build && cd ../..

# Start AIMS
echo "🧠 Starting AIMS consciousness..."
python -m src.main

# Cleanup on exit
trap "docker-compose down" EXIT