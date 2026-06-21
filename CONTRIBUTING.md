# Contributing to FlashVLM

Thank you for your interest in contributing to FlashVLM! This document provides guidelines for contributing.

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/FlashVLM.git
cd FlashVLM
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[all]"
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

## Code Standards

- Follow PEP 8 and use `ruff` for linting
- Add type hints to all function signatures
- Write docstrings for all public functions and classes
- Keep line length under 100 characters

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear, atomic commits
3. Add or update tests as needed
4. Ensure all tests pass: `pytest tests/`
5. Update documentation if applicable
6. Submit a pull request with a clear description

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- Provide system information (OS, Python version, GPU)

## Code of Conduct

Be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.
