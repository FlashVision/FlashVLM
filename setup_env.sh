#!/bin/bash
set -e

echo "=== FlashVLM Environment Setup ==="

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}

if ! command -v $PYTHON &> /dev/null; then
    echo "Error: $PYTHON not found. Please install Python 3.9+."
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found Python $PY_VERSION"

if [[ $(echo "$PY_VERSION < 3.9" | bc -l 2>/dev/null || python3 -c "print(1 if $PY_VERSION < 3.9 else 0)") == "1" ]]; then
    echo "Error: Python 3.9+ required, found $PY_VERSION"
    exit 1
fi

echo "Creating virtual environment in $VENV_DIR..."
$PYTHON -m venv $VENV_DIR
source $VENV_DIR/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing FlashVLM with all dependencies..."
pip install -e ".[all]"

echo "Installing pre-commit hooks..."
pre-commit install

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with: source $VENV_DIR/bin/activate"
echo "Run tests with: pytest tests/"
echo "Start the CLI with: flashvlm --help"
