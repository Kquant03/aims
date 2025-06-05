#!/bin/bash
# start_aims.sh - Simple AIMS startup script

echo "🚀 Starting AIMS..."

# Kill any existing processes
pkill -f "python.*src/main.py" 2>/dev/null
pkill -f "python.*src.main" 2>/dev/null

# Free ports
fuser -k 8000/tcp 2>/dev/null
fuser -k 8765/tcp 2>/dev/null

# Wait a moment
sleep 1

# Check environment
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ Error: ANTHROPIC_API_KEY not set"
    echo "Please run: export ANTHROPIC_API_KEY='your-key-here'"
    exit 1
fi

# Activate venv if needed
if [ -z "$VIRTUAL_ENV" ] && [ -d "venv" ]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Start AIMS
echo "✨ Starting AIMS web interface..."
echo "📝 TIP: Use an incognito/private browser window to avoid cookie issues"
echo "=================================================================================="
python -m src.main
