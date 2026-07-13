#!/usr/bin/env python3
"""Quick performance smoke test for CI (ci.yml quick-performance job).

Fits the intercept-only Pradel model on the dipper dataset and checks it
converges in reasonable time. Run from the repo root:

    python scripts/ci/quick_performance_smoke_test.py
"""

import time

import pradel_jax as pj

print("Running quick performance smoke test...")

data = pj.load_data("data/dipper_dataset.csv")
model = pj.PradelModel()
formula = pj.create_simple_spec(phi="~1", p="~1", f="~1")

start_time = time.time()
result = pj.fit_model(model, formula, data, strategy="scipy_lbfgs")
elapsed = time.time() - start_time

print(f"Basic optimization completed in {elapsed:.2f}s")
print(f"Success: {result.success}")
print(f"AIC: {result.aic:.2f}")

assert result.success, "Basic optimization should succeed"
assert elapsed < 30.0, f"Basic optimization took too long: {elapsed:.2f}s"
# Dipper intercept-only AIC is ~1435 (verified locally); 1000 was never
# reachable for this dataset/model and made this check fail unconditionally.
assert result.aic < 2000, f"AIC seems unreasonably high: {result.aic}"

print("Quick performance smoke test passed")
