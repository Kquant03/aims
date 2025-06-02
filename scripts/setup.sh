# scripts/setup.sh
#!/bin/bash
# Setup script for AIMS

echo "Setting up AIMS - Autonomous Intelligent Memory System"

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{states,backups,memories,uploads}
mkdir -p logs
mkdir -p src/ui/{templates,static}

# Check for .env file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your API keys"
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Initialize databases (if running locally)
if command -v psql &> /dev/null; then
    echo "PostgreSQL found. Setting up database..."
    # This would include database setup commands
fi

echo "Setup complete!"
echo "To start AIMS, run: python -m src.main"