#!/bin/bash

# Quick start script for Pradel-JAX

echo "==========================================="
echo "Setting up Pradel-JAX environment..."
echo "==========================================="

# Create virtual environment
python3 -m venv pradel_env
source pradel_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Run statistical inference demo with dipper dataset
echo "==========================================="
echo "Running statistical inference demo..."
echo "==========================================="

python examples/statistical_inference_demo.py

echo "==========================================="
echo "Running integration test to verify setup..."
echo "==========================================="

python -m pytest tests/integration/test_optimization_minimal.py -v

echo "==========================================="
echo "Setup complete! ✅"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source pradel_env/bin/activate"
echo "2. Try the examples in examples/"
echo "3. Check docs/ for user guides and tutorials"
echo "4. Run tests with: python -m pytest tests/"
echo "==========================================="