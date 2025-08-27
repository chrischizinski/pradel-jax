#!/bin/bash

# Quick start script for Pradel-JAX (Unified Parallel Version)

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

# Run full analysis on 5,000-row dataset using the unified parallel script
echo "==========================================="
echo "Running analysis on full dataset ..."
echo "==========================================="

python examples/run_analysis.py \
  --data data/wf.dat.csv \
  --output results/test_results.csv \
  --models models/models.csv \
  --mode parallel \
  --cores 4 \
  --gpu

echo "==========================================="
echo "Setup complete! Check results/test_results.csv for output."
echo "==========================================="