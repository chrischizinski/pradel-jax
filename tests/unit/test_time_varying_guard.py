"""The design matrix must refuse to flatten time-varying covariates.

A 2D covariate expands to one column per occasion, but a model whose
parameters are a single value per individual then collapses those columns into
one linear predictor.  Before the guard this converged and reported an AIC, so
an "annual tier" model would have ranked against the others while actually
fitting one time-constant survival driven additively by every year at once.
These tests exist so that failure stays loud until parameters are genuinely
indexed by occasion.
"""

import pandas as pd
import pytest

import pradel_jax as pj
from pradel_jax.core.exceptions import ModelSpecificationError
from pradel_jax.formulas.design_matrix import build_design_matrix
from pradel_jax.models.pradel import PradelModel


def _annual_tier_context(tmp_path) -> pj.DataContext:
    """Four occasions with genuinely annual tier status."""
    frame = pd.DataFrame(
        {
            "individual_id": range(6),
            "ch": ["1011", "1101", "0111", "1011", "1101", "0110"],
            "tier_2016": [1, 1, 2, 1, 2, 1],
            "tier_2017": [1, 2, 2, 1, 2, 1],
            "tier_2018": [2, 2, 1, 1, 2, 2],
            "tier_2019": [2, 1, 1, 2, 2, 2],
        }
    )
    path = tmp_path / "annual_tier.csv"
    frame.to_csv(path, index=False)
    return pj.load_data(path)


def test_adapter_still_assembles_the_time_varying_matrix(tmp_path):
    """Guard rejects at design-matrix time, not by dropping the data.

    If the adapter ever stopped building the (n, T) matrix, the guard below
    would pass for the wrong reason.
    """
    data = _annual_tier_context(tmp_path)
    assert data.covariates["tier"].shape == (6, 4)


def test_time_varying_covariate_is_rejected_not_flattened(tmp_path):
    data = _annual_tier_context(tmp_path)
    model = PradelModel()
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")

    with pytest.raises(ModelSpecificationError) as excinfo:
        model.build_design_matrices(spec, data)

    message = str(excinfo.value)
    # The message has to name the covariate and the collapse, or it will not
    # tell the next person what actually went wrong.
    assert "tier" in message
    assert "4 occasions" in message


def test_time_constant_covariate_is_unaffected(tmp_path):
    """The guard must not catch ordinary single-year columns."""
    data = _annual_tier_context(tmp_path)
    model = PradelModel()
    spec = pj.create_formula_spec(phi="~1 + tier_2016", p="~1", f="~1")

    design = model.build_design_matrices(spec, data)

    assert design["phi"].matrix.shape == (6, 2)
    assert design["phi"].column_names == ["(Intercept)", "tier_2016"]


def test_opt_in_restores_per_occasion_expansion(tmp_path):
    """Occasion-indexed builders keep the expansion via allow_time_varying.

    This is the seam the occasion-specific parameter work will use; without it
    the guard would have to be deleted rather than switched on.
    """
    data = _annual_tier_context(tmp_path)
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")

    design = build_design_matrix(spec.phi, data, allow_time_varying=True)

    # One dummy per occasion for the non-reference tier level, plus intercept.
    assert design.matrix.shape[1] > 2
    assert any(name.endswith("_t3") for name in design.column_names)
