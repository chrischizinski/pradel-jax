"""Regression tests for the exact frequency-weighted likelihood backend."""

import numpy as np
import pandas as pd

import pradel_jax as pj
from pradel_jax.data.grouped import group_data_context
from pradel_jax.models.pradel import PradelModel


def test_grouped_context_preserves_complete_covariate_trajectories(tmp_path):
    frame = pd.DataFrame(
        {
            "individual_id": range(5),
            "ch": ["1011", "1011", "1011", "1011", "1011"],
            "sex": [0, 0, 0, 0, 1],
            "tier_2016": [1, 1, 1, 1, 1],
            "tier_2017": [1, 1, 1, 2, 1],
        }
    )
    path = tmp_path / "histories.csv"
    frame.to_csv(path, index=False)
    grouped = group_data_context(pj.load_data(path))

    assert grouped.n_individuals == 3
    assert sorted(np.asarray(grouped.frequency).tolist()) == [1.0, 1.0, 3.0]


def test_grouped_likelihood_equals_individual_likelihood(tmp_path):
    frame = pd.DataFrame(
        {
            "individual_id": range(8),
            "ch": ["1011", "1011", "1011", "1101", "1101", "0011", "0011", "0011"],
            "sex": [0, 0, 0, 1, 1, 0, 0, 0],
        }
    )
    path = tmp_path / "histories.csv"
    frame.to_csv(path, index=False)
    data = pj.load_data(path)
    formula = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")
    model = PradelModel()
    design = model.build_design_matrices(formula, data)
    theta = model.get_initial_parameters(data, design)

    grouped = group_data_context(data)
    grouped_design = model.build_design_matrices(formula, grouped)
    individual_ll = float(model.log_likelihood(theta, data, design))
    grouped_ll = float(model.log_likelihood(theta, grouped, grouped_design))

    np.testing.assert_allclose(grouped_ll, individual_ll, rtol=1e-6, atol=1e-6)


def test_grouped_likelihood_matches_individual_with_priors_active(tmp_path):
    """Priors are per-individual, so grouping must not dilute them.

    With unweighted priors the grouped backend applied the penalty once per
    unique record instead of once per individual, so the two backends
    disagreed for any model with a prior switched on.  Defaults are zero, so
    only a non-zero strength exercises this path.
    """
    frame = pd.DataFrame(
        {
            "individual_id": range(8),
            "ch": ["1011", "1011", "1011", "1101", "1101", "0011", "0011", "0011"],
            "sex": [0, 0, 0, 1, 1, 0, 0, 0],
        }
    )
    path = tmp_path / "histories.csv"
    frame.to_csv(path, index=False)
    data = pj.load_data(path)
    formula = pj.create_formula_spec(phi="~1 + sex", p="~1", f="~1")
    model = PradelModel(
        boundary_prior_strength=1.0,
        recruitment_prior_strength=1.0,
    )
    design = model.build_design_matrices(formula, data)
    theta = model.get_initial_parameters(data, design)

    grouped = group_data_context(data)
    grouped_design = model.build_design_matrices(formula, grouped)
    individual_ll = float(model.log_likelihood(theta, data, design))
    grouped_ll = float(model.log_likelihood(theta, grouped, grouped_design))

    np.testing.assert_allclose(grouped_ll, individual_ll, rtol=1e-9, atol=1e-9)


def test_likelihood_runs_in_double_precision():
    """float32 ulp at the NE/SD likelihood scale is 0.0625, larger than the
    nested-model log-likelihood differences the AIC comparison depends on."""
    import jax

    assert jax.config.jax_enable_x64 is True
