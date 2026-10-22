#!/usr/bin/env python3
"""Regression test ensuring the soft boundary prior keeps estimates interior."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import pradel_jax as pj
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import FormulaSpec, ParameterType
from pradel_jax.models.pradel import PradelModel, inv_logit


def _build_problematic_dataset() -> pj.DataContext:
    """Create a small capture history dataset that used to hit the phi upper bound."""

    # Patterns chosen from previous boundary investigations (e.g. 10001).
    patterns = [
        "100001",
        "100001",
        "100010",
        "100001",
        "111100",
        "011110",
    ]

    df = pd.DataFrame(
        {
            "individual_id": np.arange(len(patterns)),
            "ch": patterns,
        }
    )

    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    try:
        df.to_csv(handle.name, index=False)
        handle.close()
        return pj.load_data(handle.name)
    finally:
        os.unlink(handle.name)


def _fit(data_context, spec, **prior_kwargs):
    """Fit the constant-parameter model and return (result, bounds, model)."""
    model = PradelModel(**prior_kwargs)
    design_mats = model.build_design_matrices(spec, data_context)
    bounds = model.get_parameter_bounds(data_context, design_mats)
    initial = model.get_initial_parameters(data_context, design_mats)

    def objective(theta):
        return -float(model.log_likelihood(theta, data_context, design_mats))

    result = minimize(
        objective,
        np.asarray(initial),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200},
    )
    return result, bounds


def _spec() -> FormulaSpec:
    parser = FormulaParser()
    return FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1"),
    )


@pytest.mark.unit
def test_without_priors_survival_pins_to_the_upper_bound():
    """Establishes the pathology the priors exist to fix.

    This dataset is small enough that the unpenalised MLE sits on the phi
    upper bound.  If this ever stops happening the two tests below are no
    longer testing anything, because their starting pathology is gone.
    """
    data_context = _build_problematic_dataset()
    result, _ = _fit(data_context, _spec())

    assert result.success, f"optimisation failed: {result.message}"
    assert float(inv_logit(result.x[0])) > 0.99


@pytest.mark.unit
def test_boundary_prior_keeps_survival_interior():
    """The Beta prior governs phi and p, and only those.

    Recruitment is deliberately not asserted here: the Beta prior is applied
    to phi and p alone, so it has no mechanism to hold f off zero.  That is
    test_recruitment_prior_keeps_recruitment_interior's job.
    """
    data_context = _build_problematic_dataset()
    # Stronger penalty weight makes the effect visible despite the small sample size.
    result, bounds = _fit(data_context, _spec(), boundary_prior_strength=1.25)

    assert result.success, f"optimisation failed: {result.message}"

    phi_hat = float(inv_logit(result.x[0]))
    # Stay comfortably away from the upper bound (0.999) and not degenerate.
    assert phi_hat < 0.97
    assert phi_hat > 0.4

    # Ensure we did not finish exactly on the numerical bound.
    upper_logit = bounds[0][1]
    assert abs(result.x[0] - upper_logit) > 1e-3


@pytest.mark.unit
def test_recruitment_prior_keeps_recruitment_interior():
    """Only the log-normal prior can hold f away from zero.

    Under float32 this assertion used to pass with the boundary prior alone,
    because the optimiser could not descend to the true optimum and stopped
    short of f = 0.  In double precision it reaches f ~ 1e-07 unless the
    recruitment prior is actually switched on, so the assertion now tests the
    prior rather than the arithmetic.
    """
    data_context = _build_problematic_dataset()
    result, _ = _fit(
        data_context,
        _spec(),
        boundary_prior_strength=1.25,
        recruitment_prior_strength=1.0,
    )

    assert result.success, f"optimisation failed: {result.message}"
    assert float(np.exp(result.x[2])) > 5e-4
