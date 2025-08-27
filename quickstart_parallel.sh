#!/bin/bash

# Quick start script for Pradel-JAX with parallel/large-scale demo

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

# Run large-scale scalability demonstration
echo "==========================================="
echo "Running large-scale scalability demo..."
echo "==========================================="

python examples/benchmarks/large_scale_scalability_demonstration.py

# Run GPU acceleration benchmark if available
echo "==========================================="
echo "Testing GPU acceleration (if available)..."
echo "==========================================="

python examples/benchmarks/gpu_acceleration_benchmark.py || echo "GPU not available - CPU mode only"

# Run comprehensive integration tests
echo "==========================================="
echo "Running comprehensive integration tests..."
echo "==========================================="

python -m pytest tests/integration/ -v --tb=short

echo "==========================================="
echo "Setup complete! ✅"
echo "==========================================="
echo ""
echo "Performance features demonstrated:"
echo "1. Large-scale dataset processing"
echo "2. GPU acceleration (if available)"
echo "3. Parallel optimization strategies"
echo "4. Memory-efficient processing"
echo ""
echo "Next steps:"
echo "1. Activate environment: source pradel_env/bin/activate"
echo "2. Check examples/nebraska/ for real-world analysis"
echo "3. Run benchmarks with: python -m pytest tests/benchmarks/"
echo "==========================================="