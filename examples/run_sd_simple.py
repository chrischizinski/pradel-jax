#!/usr/bin/env python3
"""
Simple South Dakota analysis script using reliable L-BFGS-B optimization only.
"""

import os
import sys
import subprocess

def main():
    repo_root = "/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax"
    script_path = os.path.join(repo_root, "examples", "nebraska", "nebraska_sample_analysis.py")

    cmd = [
        sys.executable, script_path,
        "--dataset", "south_dakota",
        "--sample-size", "0",  # Full dataset
        "--max-models", "64",
        "--strategy", "scipy_lbfgs",  # Use reliable L-BFGS-B only
        "--parallel",
        "--n-workers", "4",  # Reduce workers
        "--batch-size", "4",   # Reduce batch size
        "--prefer-tv-only",
        "--warm-start", "intercept"
        # Remove problematic options that were causing failures
    ]

    env = os.environ.copy()
    env['PYTHONPATH'] = repo_root

    print(f"🚀 Starting South Dakota analysis with L-BFGS-B strategy...")
    print(f"Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True, env=env, cwd=repo_root)

if __name__ == "__main__":
    main()