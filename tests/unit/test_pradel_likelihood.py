"""
Regression tests for the Pradel (1996) likelihood implementation.

These tests pin down *why* the implementation is correct, not merely that it
runs:

1. The forward marginal of the Pradel temporal-symmetry likelihood is exactly a
   Cormack-Jolly-Seber (CJS) likelihood, so the survival/detection MLEs for a
   constant model on the European dipper data must reproduce the long-published
   RMark/MARK CJS estimates (phi ~ 0.56, p ~ 0.90). This is the single strongest
   check that the survival/detection bookkeeping is right.

2. The chi / xi tail probabilities are iterates of an affine recursion; the
   closed form used in the model must equal the literal recursion. If these ever
   diverge, the likelihood is silently wrong for histories with gaps before the
   first or after the last capture.

3. Recruitment f must be identified (lambda finite and biologically plausible),
   not railed to a parameter bound -- the failure mode of the previous
   implementation.

4. log_likelihood() must return the *unpenalized* Pradel log-likelihood by
   default, otherwise AIC/BIC and likelihood-ratio tests are invalid.
"""

import warnings

import numpy as np
import pytest

import pradel_jax as pj
from pradel_jax.models.pradel import (
    PradelModel,
    _affine_iterate,
    calculate_seniority_gamma,
    inv_logit,
    exp_link,
)

DIPPER_PATH = "data/dipper_dataset.csv"

# Long-published constant-model CJS estimates for the European dipper
# (Lebreton et al. 1992; reproduced by MARK/RMark): phi ~ 0.560, p ~ 0.903.
RMARK_PHI = 0.560
RMARK_P = 0.903


def _fit_constant_dipper():
    data = pj.load_data(DIPPER_PATH)
    formula = pj.create_formula_spec(phi="~1", p="~1", f="~1")
    model = PradelModel()
    result = pj.fit_model(model=model, formula=formula, data=data)
    design = model.build_design_matrices(formula, data)
    split = model._split_parameters(result.parameters, design)
    phi = float(inv_logit(split["phi"][0]))
    p = float(inv_logit(split["p"][0]))
    f = float(exp_link(split["f"][0]))
    return result, phi, p, f


def test_affine_iterate_matches_recursion():
    """chi/xi closed form must equal the literal affine recursion it replaces."""
    for rate, p in [(0.8, 0.5), (0.56, 0.9), (0.3, 0.2), (0.99, 0.1)]:
        for n in range(0, 6):
            expected = 1.0
            for _ in range(n):
                expected = (1.0 - rate) + rate * (1.0 - p) * expected
            got = float(_affine_iterate(1.0, rate, p, float(n)))
            assert got == pytest.approx(expected, rel=1e-6, abs=1e-9), (
                f"affine_iterate(rate={rate}, p={p}, n={n}) = {got}, "
                f"recursion = {expected}"
            )


def test_seniority_gamma_uses_lambda_phi_plus_f():
    """gamma must be phi/lambda with lambda = phi + f (MARK Pradel-f model)."""
    phi, f = 0.56, 0.6
    gamma = float(calculate_seniority_gamma(phi, f))
    assert gamma == pytest.approx(phi / (phi + f), rel=1e-6)


def test_dipper_survival_detection_match_rmark():
    """Constant Pradel phi/p on dipper must match published CJS estimates.

    The forward marginal of the Pradel likelihood is a CJS likelihood, so these
    estimates are a hard reference, not an approximation to eyeball.
    """
    _, phi, p, _ = _fit_constant_dipper()
    assert phi == pytest.approx(RMARK_PHI, abs=0.02), f"phi={phi}, expected ~{RMARK_PHI}"
    assert p == pytest.approx(RMARK_P, abs=0.02), f"p={p}, expected ~{RMARK_P}"


def test_dipper_recruitment_is_identified():
    """f/lambda must be finite and biologically plausible, not railed to a bound."""
    _, phi, _, f = _fit_constant_dipper()
    lam = phi + f
    # Bounds allow f up to 10 (lambda up to ~10.5); a railed fit lands there.
    assert f < 5.0, f"recruitment f={f} looks railed to the upper bound"
    assert 0.8 < lam < 1.6, f"lambda={lam} outside plausible range for dipper"


def test_log_likelihood_unpenalized_by_default():
    """Default log_likelihood is the true Pradel LL (no prior) so AIC is valid."""
    data = pj.load_data(DIPPER_PATH)
    formula = pj.create_formula_spec(phi="~1", p="~1", f="~1")
    design = PradelModel().build_design_matrices(formula, data)
    theta = PradelModel().get_initial_parameters(data, design)

    default_model = PradelModel()
    no_prior_model = PradelModel(
        boundary_prior_strength=0.0, recruitment_prior_strength=0.0
    )
    ll_default = float(default_model.log_likelihood(theta, data, design))
    ll_no_prior = float(no_prior_model.log_likelihood(theta, data, design))
    assert ll_default == pytest.approx(ll_no_prior, rel=1e-9), (
        "log_likelihood() must not include a prior penalty by default"
    )


def test_never_captured_history_contributes_zero():
    """The conditional Pradel likelihood ignores all-zero capture histories."""
    from pradel_jax.models.pradel import _pradel_individual_likelihood

    zero_history = np.zeros(7, dtype=float)
    ll = float(_pradel_individual_likelihood(zero_history, 0.56, 0.9, 0.6))
    assert ll == pytest.approx(0.0, abs=1e-9)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    raise SystemExit(pytest.main([__file__, "-v"]))
