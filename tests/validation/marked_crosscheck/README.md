# Cross-check against marked

Checks the five-state multistate likelihood in pradel-jax against the `MSCJS`
model in [marked](https://cran.r-project.org/package=marked), on a reduced model
that both packages can express. The test, and an explanation of how the reduced
model was chosen, are in `tests/integration/test_marked_crosscheck.py`.

| File | What it is |
|---|---|
| `histories.csv` | 5,000 simulated histories, 8 occasions (synthetic, no hunter data) |
| `marked_results.json` | what marked found: max log-likelihood, S, Psi |
| `simulate.py` | regenerates `histories.csv` (deterministic) |
| `fit_marked.R` | regenerates `marked_results.json` |

CI reads the two stored files, so it needs no R. To regenerate, from the repo
root:

```bash
python tests/validation/marked_crosscheck/simulate.py
Rscript tests/validation/marked_crosscheck/fit_marked.R   # needs marked + TMB
```

At setup (marked 1.2.8), the two packages agreed to 4e-7 on the maximised
log-likelihood and to 1e-5 on every probability.
