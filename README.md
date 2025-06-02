# AIMS - Autonomous Intelligent Memory System

A sophisticated consciousness-aware AI system that maintains persistent memory, personality, and emotional continuity across conversations. Built for RTX 3090 (24GB VRAM) with fallback CPU support.

## 🌟 Features

- **Persistent Memory**: Remembers conversations and learns from interactions using Mem0 + PostgreSQL + Redis
- **Consciousness Simulation**: Implements simplified attention and global workspace theories
- **Emotional Intelligence**: PAD (Pleasure-Arousal-Dominance) emotional model with smooth transitions
- **Dynamic Personality**: OCEAN personality traits that evolve based on interactions
- **Real-time Monitoring**: WebSocket-based consciousness state visualization
- **GPU Optimized**: Leverages RTX 3090 for accelerated processing
- **Backup & Recovery**: Comprehensive state management with automatic backups

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- NVIDIA GPU (RTX 3090 recommended) with CUDA 12.1+
- 32GB+ System RAM
- Anthropic API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/aims.git
cd aims
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

3. **Start Docker services**
```bash
docker-compose up -d
```

4. **Install Python dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

5. **Initialize the system**
```bash
python scripts/setup.sh
```

6. **Run AIMS**
```bash
python -m src.main
```

7. **Access the interface**
Open http://localhost:8000 in your browser

## 📁 Project Structure

```
aims/
├── src/
│   ├── core/              # Core consciousness systems
│   │   ├── consciousness.py
│   │   ├── memory_manager.py
│   │   ├── personality.py
│   │   └── emotional_engine.py
│   ├── api/               # API integrations
│   │   ├── claude_interface.py
│   │   └── websocket_server.py
│   ├── persistence/       # State management
│   │   ├── backup_manager.py
│   │   └── state_serializer.py
│   └── ui/               # Web interface
├── configs/              # Configuration files
├── data/                # Persistent data storage
├── tests/               # Test suite
└── docker/              # Docker configurations
```

## 🧠 How It Works

### Consciousness Loop
The system runs a continuous consciousness loop at 2Hz (configurable), processing:
1. **Attention Updates**: Focuses on salient information
2. **Memory Consolidation**: Stores important interactions
3. **Emotional Processing**: Updates emotional state
4. **Coherence Calculation**: Maintains internal consistency

### Memory Architecture
- **Short-term**: Working memory buffer (7±2 items)
- **Long-term**: Semantic memories with importance weighting
- **Episodic**: Conversation history with emotional context
- **Consolidation**: Gradual decay with importance-based retention

### Personality System
- **OCEAN Model**: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
- **Dynamic Evolution**: Traits evolve slowly based on interactions
- **Bounded Changes**: Core personality remains stable

### Emotional Model
- **PAD Space**: 3D emotional representation
- **Smooth Transitions**: Natural emotional state changes
- **Baseline Pull**: Gradual return to neutral state
- **Context Integration**: Emotions influence responses

## 🔧 Configuration

### Core Settings (configs/default_config.yaml)
```yaml
consciousness:
  cycle_frequency: 2.0      # Hz
  working_memory_size: 7
  coherence_threshold: 0.7

personality:
  learning_rate: 0.001
  momentum: 0.95

emotions:
  transition_speed: 0.1
  baseline_pull: 0.05
```

### GPU Optimization
The system automatically detects and optimizes for available GPU:
- **RTX 3090**: Full features with 2-5Hz consciousness cycles
- **Other GPUs**: Adjusted batch sizes and reduced features
- **CPU Only**: Basic functionality at reduced speed

## 📊 Monitoring

### Web Dashboard
- Real-time consciousness metrics
- Emotional state visualization
- Memory statistics
- Personality trait display

### WebSocket API
Connect to `ws://localhost:8765` for real-time updates:
```javascript
const ws = new WebSocket('ws://localhost:8765');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Consciousness update:', data);
};
```

## 🔐 Privacy & Security

- All data stored locally by default
- Optional S3 backup encryption
- Session-based user isolation
- No data sharing with external services (except Claude API)

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Adding New Features
1. Create feature branch
2. Add tests in `tests/`
3. Implement in appropriate module
4. Update documentation
5. Submit pull request

### Code Style
```bash
black src/
flake8 src/
```

## 🐛 Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
nvidia-smi  # Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Memory Errors**
- Reduce `cycle_frequency` in config
- Lower batch sizes
- Enable gradient checkpointing

**Database Connection Issues**
```bash
docker-compose ps  # Check service status
docker-compose logs postgres  # View logs
```

## 📈 Performance Tuning

### For RTX 3090
- Optimal batch size: 32-64
- Memory allocation: 20-22GB VRAM
- Consciousness cycles: 2-5Hz

### For Limited Hardware
- Reduce working memory size
- Increase consolidation threshold
- Disable real-time visualization
- Use CPU-only mode

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Anthropic for Claude API
- Open source communities for supporting libraries
- Consciousness researchers for theoretical foundations

## ⚠️ Disclaimer

AIMS simulates consciousness-like properties but does not claim actual consciousness. The system is designed for research and educational purposes. Always use AI responsibly and ethically.

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: aims-support@example.com

---

Built with ❤️ for advancing human-AI interaction