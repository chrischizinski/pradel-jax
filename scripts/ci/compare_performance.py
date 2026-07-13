#!/usr/bin/env python3
"""Compare PR vs main branch performance reports (performance-tests.yml
benchmark-comparison job, PRs only). Run from tests/benchmarks/:

    python ../../scripts/ci/compare_performance.py
"""

import json
import sys

try:
    with open("pr_performance.json") as f:
        pr_data = json.load(f)
    with open("main_performance.json") as f:
        main_data = json.load(f)
except FileNotFoundError as e:
    print(f"Could not load performance data: {e}")
    sys.exit(1)

pr_time = pr_data["performance_summary"]["avg_time"]
main_time = main_data["performance_summary"]["avg_time"]
time_ratio = pr_time / main_time if main_time > 0 else 1.0

pr_success = pr_data["performance_summary"]["success_rate"]
main_success = main_data["performance_summary"]["success_rate"]

print("=== Performance Comparison ===")
print(f"PR average time: {pr_time:.3f}s")
print(f"Main average time: {main_time:.3f}s")
print(f"Performance ratio: {time_ratio:.2f}x")
print(f"PR success rate: {pr_success:.1%}")
print(f"Main success rate: {main_success:.1%}")

# Shared CI runners are noisy; only flag large, clear regressions.
if time_ratio > 2.0:
    print("PR is significantly slower than main")
    sys.exit(1)
elif time_ratio < 0.5:
    print("PR shows a large performance improvement")
else:
    print("Performance is comparable to main")

if pr_success < main_success * 0.9:
    print("PR has a lower success rate than main")
    sys.exit(1)
