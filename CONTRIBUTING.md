# CONTRIBUTING.md - Contributing Guidelines

# Contributing to AIMS

Thank you for your interest in contributing to AIMS! This document provides guidelines for contributing to the project.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/aims.git
   cd aims
   ```
3. **Set up the development environment**:
   ```bash
   python setup_helper.py
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 💻 Development Process

### 1. Before You Start
- Check existing issues and pull requests
- Create an issue to discuss major changes
- Ensure your idea aligns with the project's goals

### 2. Writing Code
- Follow the existing code style
- Write clear, self-documenting code
- Add comments for complex logic
- Keep functions small and focused

### 3. Code Style
We use:
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** where appropriate

Run before committing:
```bash
black src/
flake8 src/ --max-line-length=100
```

### 4. Testing
- Write tests for new features
- Ensure all tests pass:
  ```bash
  pytest tests/ -v
  ```
- Aim for high test coverage
- Test edge cases

### 5. Documentation
- Update README.md if needed
- Add docstrings to new functions/classes
- Update configuration examples
- Document breaking changes

## 📝 Pull Request Process

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** with:
   - Clear title and description
   - Reference to related issues
   - Summary of changes
   - Screenshots (if UI changes)

4. **PR Requirements**:
   - All tests must pass
   - Code must be formatted
   - Documentation updated
   - No merge conflicts

5. **Review Process**:
   - Be patient and respectful
   - Address reviewer feedback
   - Make requested changes
   - Re-request review when ready

## 🏗️ Architecture Guidelines

### Core Principles
1. **Modularity**: Keep components loosely coupled
2. **Efficiency**: Optimize for RTX 3090 but support CPU
3. **Persistence**: Ensure state can be saved/restored
4. **Monitoring**: Add metrics for new features

### Adding New Features

#### Consciousness Components
- Extend `ConsciousnessCore` for new cognitive features
- Maintain the 2-5Hz update cycle
- Document theoretical basis

#### Memory Systems
- Use `PersistentMemoryManager` for storage
- Consider memory efficiency
- Implement consolidation strategies

#### Emotional/Personality
- Maintain bounded trait evolution
- Ensure smooth transitions
- Preserve core personality

## 🐛 Reporting Issues

### Bug Reports Should Include:
- System information (OS, GPU, Python version)
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/logs
- Configuration used

### Feature Requests Should Include:
- Clear use case
- Expected behavior
- Implementation ideas (optional)
- Theoretical basis (for consciousness features)

## 🎯 Areas for Contribution

### High Priority
- Performance optimizations
- Memory efficiency improvements
- Additional consciousness theories
- Better visualization tools

### Good First Issues
- Documentation improvements
- Test coverage expansion
- UI/UX enhancements
- Configuration examples

### Research Areas
- Alternative emotional models
- Advanced memory consolidation
- Multi-modal consciousness
- Distributed consciousness

## 📊 Performance Contributions

When optimizing performance:
1. Profile before and after
2. Document benchmarks
3. Test on multiple hardware configs
4. Consider CPU fallbacks

## 🔒 Security Contributions

For security-related contributions:
- Report vulnerabilities privately first
- Follow responsible disclosure
- Add security tests
- Document security implications

## 📚 Resources

### Recommended Reading
- Global Workspace Theory papers
- Attention Schema Theory research
- PAD emotional model literature
- OCEAN personality framework

### Development Tools
- PyTorch documentation
- Anthropic API reference
- Docker best practices
- WebSocket protocols

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Thanked in project updates

## 💬 Communication

- **GitHub Issues**: Bug reports and features
- **GitHub Discussions**: General questions
- **Pull Requests**: Code contributions

## ⚖️ License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to AIMS! Together we're advancing human-AI interaction. 🚀