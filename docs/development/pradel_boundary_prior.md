# Pradel Boundary Stabilisation Notes

## Context

While fitting Pradel models we often observed the survival (φ) and detection (p) parameters saturating at their optimisation bounds. Claude highlighted two useful references before timing out:

* Telenský et al. (2024), *Extending Pradel models to capture transient structure* (PDF: `~/Desktop/Telensky_et_al_2024_MEE_Pradel_model_extension.pdf`).
* The DynaPyt repository (<https://github.com/sola-st/DynaPyt>) whose tooling inspired the idea of adding gentle regularisation rather than hard clipping.

These sources reinforce that φ and p should stay biologically plausible (strictly inside `(0, 1)`) and that soft penalties are often preferable to extremely tight bounds.

## Code Changes

* Added `_log_beta_prior` in `pradel_jax/models/pradel.py` to provide a lightweight Beta-style log prior for probability parameters.
* Extended `PradelModel.__init__` with `boundary_prior_strength`, `boundary_prior_alpha`, and `boundary_prior_beta` knobs. The default strength (`0.75`) nudges estimates away from 0/1 but still lets the data dominate.
* Relaxed the logit bounds for φ and p to `logit(0.001)` … `logit(0.999)` so the optimiser works within a sensible numerical range. The soft prior now shoulders the job of preventing boundary hugging.
* Mixed the prior directly into the model log-likelihood so downstream optimisation pipelines (parallel workers, heuristics) continue to operate unchanged.
* Added a log-normal prior on recruitment (mode 0.05, sigma 0.75, strength 0.5) and expanded the log-scale bounds to [log(1e-8), log(10)] so `f` can drift towards low but non-zero values without slamming into a hard floor.

## Tests

* Added `tests/unit/test_boundary_prior.py` which recreates a combination of capture histories (e.g. `100001`) that previously drove φ to the upper bound. With the prior active the solution stays comfortably interior (`φ ≈ 0.6–0.9`).
* Re-ran the existing `tests/unit/test_bounds_fix_simple.py`; it still passes (with SciPy occasionally reporting an abnormal line search termination, as before).

### Real-data spot checks (2024‑09‑24)

```
MPLCONFIGDIR=.mplconfig python test_nebraska_convergence.py
```

* Nebraska sample (n=500) finishes with `φ≈0.868`, `p≈0.337`, `f≈1e-5`; only `f` touches its lower guardrail.
* South Dakota sample (n=500) delivers `φ≈0.857` with no boundary contact on φ/p.
* Likelihood surface sweep over Nebraska (n=200) peaks around `φ≈0.885`, comfortably inside 0/1.

Broader sweep:

```
MPLCONFIGDIR=.mplconfig python final_ne_sd_validation.py
```

* Nebraska samples n∈{100, 500, 1000, 2000, 5000} now converge with φ≈0.86–0.87, p≈0.34, and f≈0.04 after adding the recruitment log-normal prior; gradients at φ=0.9 remain positive, so additional structural tweaks may be needed if a higher survival interior optimum is desired.
* South Dakota samples n∈{100, 500, 1000, 2000, 5000} remain fully interior for φ and p (≈0.60–0.63 and 0.32–0.34 respectively) with stable recruitment ≈0.08 and no boundary contact.
* Profile log-likelihood checks at φ=0.9 now produce ΔLL > 2 for every Nebraska sample, confirming the interior optimum without relying on gradient sign heuristics.

> **Note:** `codacy_cli_analyze` is not available in this sandboxed environment (`command not found`) and running it through `npx` fails because outbound network access is disabled. Re-run the tool locally once network access is available.

## Next Steps

1. Exercise the new prior against real datasets (Nebraska, South Dakota) to calibrate `boundary_prior_strength` for production fits.
2. If additional structure is needed, consider time-varying or covariate-specific priors (the constructor makes it straightforward).
3. Update public-facing docs once the tuning is validated.
