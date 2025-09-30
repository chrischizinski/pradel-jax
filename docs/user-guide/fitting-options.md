# Fitting Options & Robust Inference

This guide summarizes the recommended CLI options for robust, scalable fitting on large datasets, with time‑varying covariates and boundary‑aware inference.

## Time‑Varying Covariates
- Age: Columns like `age_2016..age_2024` are assembled automatically into a 2D `age` matrix and expanded to per‑occasion columns (`age_t0..`).
- Tier (categorical): Columns like `tier_2016..tier_2024` are assembled into a 2D `tier` matrix and expanded to per‑occasion dummies (reference level dropped).
- Prefer TV‑only: `--prefer-tv-only` will prefer `age`/`tier` over single‑year proxies (`age_2017`, etc.).

## Stability & Boundaries
- Warm‑start: `--warm-start intercept` quickly fits an intercept‑only model and seeds the target model.
- Ridge: `--penalty ridge --lambda-penalty 1e-4` adds mild shrinkage to stabilize weakly identified terms.
- Boundary priors (φ,p):
  - Jeffreys (recommended default): `--boundary-prior jeffreys --boundary-weight 1e-4`
  - Barrier (stronger) if needed: `--boundary-prior barrier --boundary-weight 1e-4`
- Firth refinement (post‑MLE): `--firth-refine --firth-steps 2` for logistic components on the best model.

## Robust Uncertainty
- Robust SEs (SVD-based): `--robust-se --robust-se-on top --robust-se-top-k 5`
- Bootstrap CIs: `--bootstrap --bootstrap-samples 200`
- Diagnostics in exports: `fisher_condition`, `near_bound_any/count/params` printed in `*_model_selection_*.csv`.

## Parallel & Aggregation
- Parallel: `--parallel --n-workers 8 --batch-size 8` (falls back to sequential in restricted environments).
- Aggregation (intercept-only): `--aggregate by_history` collapses identical capture histories and uses `weights`.

## Example Commands

NE (64 models, TV age+tier, Jeffreys prior):
```
python examples/nebraska/nebraska_sample_analysis.py \
  --dataset nebraska --sample-size 0 --max-models 64 \
  --strategy hybrid --parallel --n-workers 8 --batch-size 8 \
  --prefer-tv-only \
  --warm-start intercept --penalty ridge --lambda-penalty 1e-4 \
  --boundary-prior jeffreys --boundary-weight 1e-4 \
  --robust-se --robust-se-on top --robust-se-top-k 5 \
  --bootstrap --bootstrap-samples 200 \
  --firth-refine --firth-steps 2
```

SD (64 models, TV‑only, Jeffreys prior):
```
python examples/nebraska/nebraska_sample_analysis.py \
  --dataset south_dakota --sample-size 0 --max-models 64 \
  --strategy hybrid --parallel --n-workers 8 --batch-size 8 \
  --prefer-tv-only \
  --warm-start intercept --penalty ridge --lambda-penalty 1e-4 \
  --boundary-prior jeffreys --boundary-weight 1e-4 \
  --robust-se --robust-se-on top --robust-se-top-k 5 \
  --bootstrap --bootstrap-samples 200 \
  --firth-refine --firth-steps 2
```

## Tuning Advice
- Start with Jeffreys 1e‑4; increase to 5e‑4 only if boundary flags persist.
- Use Barrier only when Jeffreys doesn’t sufficiently reduce boundary pile‑ups.
- Always prefer bootstrap/profile CIs for boundary‑affected parameters.

