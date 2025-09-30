#!/usr/bin/env python3
"""
Convenience script to run the full Nebraska (NE) model fitting with
recommended robust options and time‑varying covariates.

Usage:
  python examples/run_ne_full.py
"""

import os
import sys
import subprocess


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = os.path.join(repo_root, "examples", "nebraska", "nebraska_sample_analysis.py")

    env = os.environ.copy()
    # Ensure repo is on PYTHONPATH
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        script,
        "--dataset",
        "nebraska",
        "--sample-size",
        "0",
        "--max-models",
        "64",
        "--strategy",
        "hybrid",
        "--parallel",
        "--n-workers",
        "8",
        "--batch-size",
        "8",
        "--prefer-tv-only",
        "--warm-start",
        "intercept",
        "--penalty",
        "ridge",
        "--lambda-penalty",
        "1e-4",
        "--boundary-prior",
        "jeffreys",
        "--boundary-weight",
        "1e-4",
        "--robust-se",
        "--robust-se-on",
        "top",
        "--robust-se-top-k",
        "5",
        "--bootstrap",
        "--bootstrap-samples",
        "200",
        "--firth-refine",
        "--firth-steps",
        "2",
    ]

    print("Running NE full analysis with recommended options...\n")
    subprocess.run(cmd, check=True, env=env, cwd=repo_root)


if __name__ == "__main__":
    main()

