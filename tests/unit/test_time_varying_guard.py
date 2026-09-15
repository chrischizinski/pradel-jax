"""A time-varying covariate is either indexed by occasion or refused.

A 2D covariate has one value per occasion, but a model whose parameters take a
single value per individual has nowhere to put them.  The old behaviour was to
expand the covariate into one design column per occasion and let the linear
predictor add them all together, which converged and reported an AIC -- so an
"annual tier" model ranked against the others while actually fitting one
time-constant survival driven additively by every year at once.

Two things now stand between that failure and a user.  The design-matrix builder
refuses by default, so any caller that has not thought about occasions gets an
error rather than a plausible number.  And `PradelModel` opts in, because its
parameters really are indexed by occasion -- see
tests/unit/test_occasion_specific_parameters.py for what that indexing has to
satisfy.  These tests pin both halves: the refusal must stay the default, and
the opt-in must produce a period axis rather than the old flattening.
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


def test_builders_that_have_not_opted_in_still_refuse(tmp_path):
    """The refusal is the default, and it has to name what went wrong.

    `allow_time_varying` defaults to False precisely because the failure it
    prevents is silent.  Any future model that forgets to index by occasion
    should hit this rather than fit a collapsed covariate.
    """
    data = _annual_tier_context(tmp_path)
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")

    with pytest.raises(ModelSpecificationError) as excinfo:
        build_design_matrix(spec.phi, data)

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


def test_pradel_opts_in_and_gets_a_period_axis_not_more_columns(tmp_path):
    """The opt-in is what the guard was holding the door open for.

    The distinction that matters is where the occasions live.  They belong on
    their own axis of the design matrix, so the model keeps one tier
    coefficient applied at whatever tier the hunter held that year.  The old
    expansion put them in the *columns*, which silently turned `~ tier` into a
    tier-by-year model collapsed back down to one number.
    """
    data = _annual_tier_context(tmp_path)
    model = PradelModel()
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")

    design = model.build_design_matrices(spec, data)
    phi = design["phi"]

    # 6 individuals, 3 intervals between 4 occasions, intercept + one dummy for
    # the single non-reference tier level present in this data.
    assert phi.matrix.shape == (6, 3, 2)
    assert phi.parameter_count == 2
    assert not any("_t" in name for name in phi.column_names), (
        "per-occasion columns are the old flattening; occasions belong on the "
        "period axis"
    )
