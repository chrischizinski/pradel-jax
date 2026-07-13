# Example Datasets & Using the `pradel_jax` Python API

This directory only ever contains real published example data (the dipper
dataset) or fully synthetic data. **Never place real hunter/customer data
here** — see the repo-root `CLAUDE.md` and `.gitignore` for the policy and
why (person_id/customer_id/FUZZY identifiers are banned from this repo).

## What's in this directory

| File | What it is |
|---|---|
| `dipper_dataset.csv` | The classic Lebreton et al. (1992) European Dipper dataset — 294 birds, 7 capture occasions, `ch` + `sex` columns. Standard textbook/RMark validation dataset. |
| `dipper_minimal.csv`, `dipper_processed.csv` | Pre-processed variants of the dipper dataset used by some tests/examples. |
| `test_datasets/` | Fixtures used by the test suite (dipper CSV + cached RMark comparison results). |
| `synthetic_capture_recapture_data.csv` | Fully synthetic, generated dataset with the **same column schema** as the real Nebraska/South Dakota HIP hunter-tier exports (`customer_id`, `tier_2016..tier_2024`, `gender`, `age_2016..age_2024`, `th`, `ch`). Every value comes from a known simulated Pradel process — see `generate_synthetic_hunter_data.py`. Safe to use, share, and commit. |
| `generate_synthetic_hunter_data.py` | The generator for `synthetic_capture_recapture_data.csv`. Re-run it any time; it's seeded (`SEED = 42`) so output is reproducible. |

## What `pradel_jax` does

`pradel_jax` fits Pradel (1996) capture-recapture models — the temporal-symmetry
parameterization that jointly estimates, per capture occasion:

- **φ (phi)** — apparent survival probability
- **p** — detection probability
- **f** — per-capita recruitment rate (MARK's "Pradrec" parameterization)

from which the population growth rate **λ = φ + f** and seniority
**γ = φ/λ** follow directly. Models are specified with R-style formulas
(`~1 + gender + age`) and fit via JAX-based optimization (L-BFGS-B, SLSQP,
Adam, or multi-start, auto-selected), with standard errors computed from
the exact `jax.hessian` of the log-likelihood (not finite differences).

## Quick start: fitting the dipper dataset

```python
import pradel_jax as pj

data_context = pj.load_data("data/dipper_dataset.csv")
formula_spec = pj.create_formula_spec(phi="~1 + sex", p="~1 + sex", f="~1")
result = pj.fit_model(model=pj.PradelModel(), formula=formula_spec, data=data_context)

print(result.success, result.log_likelihood, result.aic)
# True -714.387 1438.77  (matches published/RMark phi~0.56, p~0.90)

for name, est, se in zip(result.parameter_names, result.parameters, result.parameter_se):
    print(f"{name:20s} {est: .4f}  (SE {se:.4f})")
```

## Quick start: fitting the synthetic hunter dataset

`synthetic_capture_recapture_data.csv` has per-year tier codes (0 = not captured,
1/2 = tier level) rather than a single `ch` column, so it needs a small
amount of prep before `pradel_jax.load_data()` — this mirrors exactly what
`scripts/analysis/run_mr_nebraska_sd.R` does to the real NE/SD exports:

```python
import pandas as pd
import numpy as np
import pradel_jax as pj

df = pd.read_csv("data/synthetic_capture_recapture_data.csv")
tier_cols = [c for c in df.columns if c.startswith("tier_")]
tier = df[tier_cols].to_numpy()

prepped = pd.DataFrame({
    "individual_id": range(len(df)),
    "ch": ["".join("1" if v > 0 else "0" for v in row) for row in tier],
    "gender": df["gender"].map({"M": 1, "F": 0}),  # UNKNOWN -> NaN, dropped below
    "first_age": df["age_2016"],
    "tier2_dummy": (tier == 2).any(axis=1).astype(int),
}).dropna(subset=["gender", "first_age"])
prepped["gender"] = prepped["gender"].astype(int)
prepped["first_age"] = (prepped["first_age"] - prepped["first_age"].mean()) / prepped["first_age"].std()

prepped.to_csv("/tmp/synthetic_prepped.csv", index=False)
data_context = pj.load_data("/tmp/synthetic_prepped.csv")

formula_spec = pj.create_formula_spec(
    phi="~1 + gender + first_age + tier2_dummy",
    p="~1",
    f="~1 + gender",
)
result = pj.fit_model(model=pj.PradelModel(), formula=formula_spec, data=data_context)
```

Or just run the full R pipeline against it directly — edit
`scripts/analysis/run_mr_nebraska_sd.R`'s `DATA_FILES` to point at
`data/synthetic_capture_recapture_data.csv` (it already does all of the above prep,
plus fits and ranks all 12 candidate models by AIC). This is the fastest
way to test that pipeline end-to-end without any real data.

## Regenerating the synthetic dataset

```bash
python data/generate_synthetic_hunter_data.py
```

Edit `TRUE_PARAMS` at the top of the script to change sample size,
survival/detection/recruitment rates, or covariate effects — everything
is simulated from those values, so there's no risk of ever leaking real
data through this file.
