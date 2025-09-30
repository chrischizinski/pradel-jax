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


@pytest.mark.unit
def test_boundary_prior_keeps_survival_interior():
    """Optimisation should avoid sticking to the survival upper bound."""

    data_context = _build_problematic_dataset()

    parser = FormulaParser()
    spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1"),
    )

    # Stronger penalty weight makes the effect visible despite the small sample size.
    model = PradelModel(boundary_prior_strength=1.25)
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

    assert result.success, f"optimisation failed: {result.message}"

    phi_hat = float(inv_logit(result.x[0]))
    # Stay comfortably away from the upper bound (0.999) and not degenerate.
    assert phi_hat < 0.97
    assert phi_hat > 0.4

    f_hat = float(np.exp(result.x[2]))
    assert f_hat > 5e-4

    # Ensure we did not finish exactly on the numerical bound.
    upper_logit = bounds[0][1]
    assert abs(result.x[0] - upper_logit) > 1e-3
