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

# Add src/ to PYTHONPATH so imports work
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Run test analysis on sample data
echo "==========================================="
echo "Running test analysis on subset (5,000 individuals)..."
echo "==========================================="

python examples/run_analysis.py \
  --data data/wf.dat.csv \
  --output results/test_results.csv \
  --models data/models.csv \
  --subset \
  --cores 4

echo "==========================================="
echo "Setup complete! Check results/test_results.csv for output."
echo "==========================================="