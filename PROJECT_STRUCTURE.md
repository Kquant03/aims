# AIMS Project Structure

## Quick Start
```bash
./setup.sh    # First time setup
./run.sh      # Start AIMS
```

## Directory Structure
```
aims/
├── src/              # Source code
│   ├── api/         # API interfaces (Claude, WebSocket, etc.)
│   ├── core/        # Core systems (consciousness, memory, etc.)
│   ├── ui/          # Web interface
│   ├── persistence/ # State management
│   └── utils/       # Utilities
├── configs/          # Configuration files
├── data/            # Runtime data
│   ├── states/      # Saved states
│   ├── backups/     # Backups
│   ├── memories/    # Memory storage
│   └── uploads/     # User uploads
├── docker/          # Docker configurations
├── logs/            # Application logs
├── scripts/         # Setup and utility scripts
│   ├── setup/       # Setup scripts
│   ├── fixes/       # Fix scripts
│   └── utils/       # Utility scripts
├── tests/           # Test suite
├── docs/            # Documentation
├── venv/            # Python virtual environment
├── .env             # Environment variables (create from .env.example)
├── requirements.txt # Python dependencies
├── docker-compose.yml
├── setup.sh         # Main setup script
└── run.sh           # Quick start script
```

## Configuration
Edit `.env` file for:
- ANTHROPIC_API_KEY (required)
- OPENAI_API_KEY (optional)
- Database settings
- Other options

## Development
1. Always activate venv: `source venv/bin/activate`
2. Format code: `black src/`
3. Lint: `flake8 src/`
4. Test: `pytest tests/ -v`

## Troubleshooting
- If Docker permission issues: `sudo usermod -aG docker $USER && newgrp docker`
- If port 8000 is in use: Change port in `src/main.py`
- If memory issues: Reduce batch sizes in config
- Check logs in `logs/aims.log`
