#!/bin/bash
# AIMS Quick Start Script

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║     █████╗ ██╗███╗   ███╗███████╗                                       ║"
echo "║    ██╔══██╗██║████╗ ████║██╔════╝                                       ║"
echo "║    ███████║██║██╔████╔██║███████╗                                       ║"
echo "║    ██╔══██║██║██║╚██╔╝██║╚════██║                                       ║"
echo "║    ██║  ██║██║██║ ╚═╝ ██║███████║                                       ║"
echo "║    ╚═╝  ╚═╝╚═╝╚═╝     ╚═╝╚══════╝                                       ║"
echo "║                                                                           ║"
echo "║        Autonomous Intelligent Memory System                               ║"
echo "║        Quick Start Setup                                                  ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    
    # Generate session secret
    echo "🔐 Generating secure session secret..."
    SESSION_SECRET=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode('utf-8'))")
    
    # Update .env with the generated secret
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/SESSION_SECRET=.*/SESSION_SECRET=$SESSION_SECRET/" .env
    else
        # Linux
        sed -i "s/SESSION_SECRET=.*/SESSION_SECRET=$SESSION_SECRET/" .env
    fi
    
    echo "✅ Session secret generated and saved to .env"
else
    echo "✅ .env file already exists"
fi

# Check for Anthropic API key
if ! grep -q "ANTHROPIC_API_KEY=sk-" .env 2>/dev/null; then
    echo ""
    echo "⚠️  WARNING: ANTHROPIC_API_KEY not set in .env file!"
    echo "Please edit .env and add your Anthropic API key"
    echo ""
    read -p "Do you want to enter it now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your Anthropic API key: " API_KEY
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$API_KEY/" .env
        else
            sed -i "s/ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$API_KEY/" .env
        fi
        echo "✅ API key saved to .env"
    fi
fi

# Create required directories
echo "📁 Creating data directories..."
mkdir -p logs data/states data/backups data/memories data/uploads
echo "✅ Directories created"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo "✅ Virtual environment created and dependencies installed"
else
    echo "✅ Virtual environment already exists"
    source venv/bin/activate
fi

# Export environment variables
echo "🔧 Loading environment variables..."
export $(grep -v '^#' .env | xargs)

echo ""
echo "🚀 Starting AIMS..."
echo "=================================================================================="
echo ""

# Run AIMS
python src/main.py