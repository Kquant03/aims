# Contributing to AIMS

Thank you for your interest in contributing to the Autonomous Intelligent Memory System (AIMS)! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Code Style](#code-style)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/aims.git
   cd aims
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/originalowner/aims.git
   ```
4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

## Development Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- Write clean, documented code
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed

### 3. Commit Your Changes

Follow conventional commit format:
```bash
git commit -m "feat: add consciousness state visualization"
git commit -m "fix: resolve memory leak in emotional engine"
git commit -m "docs: update API documentation"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Build process or auxiliary tool changes

### 4. Keep Your Fork Updated

```bash
git fetch upstream
git checkout main
git merge upstream/main
git checkout your-branch
git rebase main
```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Descriptive variable names
consciousness_state = calculate_coherence(memory_buffer)

# Good: Type hints
def process_memory(content: str, importance: float = 0.5) -> MemoryItem:
    """Process and store a memory item.
    
    Args:
        content: The memory content to store
        importance: Importance score (0-1)
        
    Returns:
        MemoryItem: The created memory object
    """
    pass

# Good: Async/await patterns
async def update_consciousness_state(self) -> None:
    """Update consciousness state asynchronously."""
    async with self.state_lock:
        self.state = await self.calculate_new_state()
```

### Documentation

- Use Google-style docstrings
- Document all public APIs
- Include usage examples for complex functions
- Keep comments concise and meaningful

### Code Organization

```
src/
├── core/           # Core consciousness components
├── api/            # API interfaces
├── persistence/    # Data persistence layer
├── ui/             # User interface
└── utils/          # Utility functions
```

## Testing Guidelines

### Writing Tests

```python
import pytest
from unittest.mock import Mock, AsyncMock

class TestConsciousnessCore:
    """Test consciousness core functionality."""
    
    @pytest.fixture
    async def consciousness_core(self):
        """Create consciousness core fixture."""
        config = {'cycle_frequency': 10.0}
        core = ConsciousnessCore(config)
        yield core
        core.shutdown()
    
    @pytest.mark.asyncio
    async def test_coherence_calculation(self, consciousness_core):
        """Test coherence score calculation."""
        # Arrange
        consciousness_core.memory_buffer.extend(['mem1', 'mem2'])
        
        # Act
        consciousness_core._calculate_coherence()
        
        # Assert
        assert 0 <= consciousness_core.state.global_coherence <= 1
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run tests matching pattern
pytest -k "consciousness" -v
```

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test performance metrics
- **End-to-End Tests**: Test complete workflows

## Pull Request Process

### Before Submitting

1. **Test your changes**:
   ```bash
   pytest
   black src/ tests/  # Format code
   flake8 src/ tests/  # Lint code
   mypy src/  # Type checking
   ```

2. **Update documentation**:
   - Update README.md if needed
   - Add/update docstrings
   - Update API documentation

3. **Ensure compatibility**:
   - Test on Python 3.10+
   - Verify GPU functionality (if applicable)
   - Check all database integrations

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process

1. Submit PR against `main` branch
2. Ensure CI/CD checks pass
3. Address reviewer feedback
4. Maintainers will merge when approved

## Reporting Issues

### Bug Reports

Include:
- Python version
- GPU model and CUDA version
- Steps to reproduce
- Expected vs actual behavior
- Error logs

### Feature Requests

Include:
- Use case description
- Proposed implementation (if any)
- Impact on existing functionality

## Development Setup Tips

### GPU Development

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Database Development

```bash
# Start only databases
docker-compose up -d postgres redis qdrant

# Connect to PostgreSQL
psql -h localhost -p 5433 -U aims -d aims_memory

# Monitor Redis
redis-cli monitor
```

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# GPU memory debugging
import torch
torch.cuda.empty_cache()
print(torch.cuda.memory_summary())
```

## Questions?

- Check existing issues and PRs
- Ask in discussions
- Contact maintainers

Thank you for contributing to AIMS! 🚀
```

## 7. Session Encoding Verification

The session encoding in `src/ui/web_interface.py` is already correctly implemented using `EncryptedCookieStorage`. No changes needed, but I'll add a verification test:

```python
# Add to tests/test_session.py
import pytest
from aiohttp_session import get_session
from src.ui.web_interface import AIMSWebInterface

class TestSessionManagement:
    """Test session encoding and management"""
    
    @pytest.mark.asyncio
    async def test_session_encoding(self):
        """Verify session encoding works correctly"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            interface = AIMSWebInterface()
            
            # Create test request
            async with interface.app.test_client() as client:
                # First request should create session
                resp = await client.get('/')
                assert resp.status == 200
                
                # Check session was created
                cookies = resp.cookies
                assert any('aiohttp_session' in cookie for cookie in cookies)
                
                # Second request should maintain session
                resp2 = await client.get('/', cookies=cookies)
                assert resp2.status == 200
    
    @pytest.mark.asyncio
    async def test_session_persistence(self):
        """Test session data persistence"""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'}):
            interface = AIMSWebInterface()
            
            async with interface.app.test_client() as client:
                # Store data in session
                async with client.post('/api/chat', json={'message': 'test'}) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    session_id = data.get('session_id')
                
                # Verify session persists
                async with client.get('/api/session') as resp:
                    assert resp.status == 200
                    session_data = await resp.json()
                    assert session_data.get('session_id') == session_id