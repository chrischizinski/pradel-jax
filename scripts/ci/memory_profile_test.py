#!/usr/bin/env python3
"""Memory profiling smoke test for CI (performance-tests.yml memory-profiling job).

Run from repo root (needs memory_profiler installed):

    python -m memory_profiler scripts/ci/memory_profile_test.py
"""

import time

import pradel_jax as pj

try:
    from memory_profiler import profile
except ImportError:  # profiler decorator is a no-op if the package is absent

    def profile(func):
        return func


@profile
def test_memory_usage():
    data = pj.load_data("data/dipper_dataset.csv")
    model = pj.PradelModel()
    formula = pj.create_simple_spec(phi="~1", p="~1", f="~1")

    for strategy in ["scipy_lbfgs", "hybrid"]:
        print(f"Testing {strategy}...")
        pj.fit_model(model, formula, data, strategy=strategy)
        time.sleep(0.1)


if __name__ == "__main__":
    test_memory_usage()
