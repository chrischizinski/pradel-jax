"""Annual tier status the model can use, and why the recorded one cannot be.

In the HIP data a hunter's tier is recorded only in years they registered, and
the capture history is *built from* that same field: `tier_<year> > 0` is
identical to "captured in <year>".  Putting the recorded annual tier on the
right-hand side of a survival model therefore puts the response there too --
the model would be explaining detection with detection.  It converges and
reports an AIC, which is what makes it dangerous rather than merely wrong.

`tier_state_<year>` is the tier regime the hunter is *under*: the last tier they
were observed at, carried forward across years they did not register, and their
first observed tier carried backward across the years before they ever did.
The regulation attaches to the hunter, not to the act of registering.

The backward fill is the part that is easy to get wrong, and getting it wrong is
worse than leaving the covariate out.  Pradel reads each history in both
directions: survival phi drives the forward term, and the *same* phi drives the
reverse term through the seniority rate gamma = phi / (phi + f).  Intervals
before first capture enter only through that reverse term.  Leaving them at a
distinct "not yet registered" level hands the model a phi that acts on gamma
alone, decoupled from the phi acting on survival, which breaks the
temporal-symmetry constraint that identifies f at all.  Fitted that way on the
real data the model gained roughly 220,000 AIC and returned phi near 0.003 --
a structural loophole wearing the costume of a better model.

With both fills the state is always 1 or 2, so the fitted coefficient is exactly
the Tier II versus Tier I contrast, and the level set no longer encodes whether
the hunter had entered.
"""

import numpy as np
import pandas as pd
import pytest

import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel

YEARS = [2016, 2017, 2018, 2019]
HISTORIES = ["1011", "1101", "0111", "1011", "0011", "0110"]
# Recorded tier: nonzero exactly where the hunter was captured.
RECORDED = [
    [1, 0, 1, 2],
    [1, 2, 0, 1],
    [0, 2, 2, 2],
    [1, 0, 1, 1],
    [0, 0, 1, 2],
    [0, 1, 1, 0],
]
# Last observed tier carried forward, first observed tier carried backward.
FILLED = [
    [1, 1, 1, 2],
    [1, 2, 2, 1],
    [2, 2, 2, 2],
    [1, 1, 1, 1],
    [1, 1, 1, 2],
    [1, 1, 1, 1],
]


def _context(tmp_path) -> pj.DataContext:
    frame = {"individual_id": range(len(HISTORIES)), "ch": HISTORIES}
    for j, year in enumerate(YEARS):
        frame[f"tier_{year}"] = [row[j] for row in RECORDED]
        frame[f"tier_state_{year}"] = [row[j] for row in FILLED]
    path = tmp_path / "tier_state.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return pj.load_data(path)


def test_recorded_tier_is_the_capture_indicator(tmp_path):
    """The premise. If this ever stops holding, tier_state stops being needed.

    The R pipeline asserts this identity when it builds the capture histories;
    asserting it here too is what makes the next test a statement about
    endogeneity rather than an arbitrary comparison of two arrays.
    """
    data = _context(tmp_path)
    recorded = np.asarray(data.covariates["tier"])
    captured = np.asarray(data.capture_matrix)

    assert np.array_equal((recorded > 0).astype(float), captured)


def test_the_filled_state_is_not_the_capture_indicator(tmp_path):
    """The point of the whole exercise."""
    data = _context(tmp_path)
    state = np.asarray(data.covariates["tier_state"])
    captured = np.asarray(data.capture_matrix)

    assert not np.array_equal((state > 0).astype(float), captured)


def test_no_inactive_level_survives(tmp_path):
    """A zero anywhere would put "had not registered yet" back in the model.

    That level is what decouples phi from gamma and destroys the identification
    of f, so its absence is the invariant worth pinning -- not the fill
    procedure that happens to produce it.
    """
    data = _context(tmp_path)
    state = np.asarray(data.covariates["tier_state"])

    assert set(np.unique(state)) == {1.0, 2.0}


def test_the_filled_state_still_varies_over_time(tmp_path):
    """It has to remain a time-varying covariate to be worth the machinery.

    Both fills applied too aggressively -- or a bug collapsing the matrix to one
    value per hunter -- would leave a covariate that is merely `tier2_dummy`
    again, with all the endogeneity that carries and none of the annual detail.
    """
    data = _context(tmp_path)
    state = np.asarray(data.covariates["tier_state"])

    varies = (state != state[:, :1]).any(axis=1)
    assert varies.sum() >= 3, "too few hunters change tier for this to be annual"


def test_the_two_series_do_not_collide(tmp_path):
    """`tier_state_2016` must not be swept into the `tier` series.

    Both are assembled by prefix, and "tier_state_2016" does start with
    "tier_". Only the requirement that everything after the prefix is a year
    keeps them apart.
    """
    data = _context(tmp_path)

    assert np.asarray(data.covariates["tier"]).shape == (6, 4)
    assert np.asarray(data.covariates["tier_state"]).shape == (6, 4)
    assert not np.array_equal(
        np.asarray(data.covariates["tier"]), np.asarray(data.covariates["tier_state"])
    )


def test_the_coefficient_is_the_tier_two_versus_tier_one_contrast(tmp_path):
    """One dummy, and it means the thing the study is asking about."""
    data = _context(tmp_path)
    model = PradelModel()
    spec = pj.create_formula_spec(phi="~1 + tier_state", p="~1", f="~1")

    design = model.build_design_matrices(spec, data)
    phi = design["phi"]

    assert phi.matrix.shape == (6, len(YEARS) - 1, 2)
    assert phi.column_names == ["(Intercept)", "tier_state_2"]


def test_a_carried_forward_model_fits(tmp_path):
    """End to end: the covariate the analysis will actually use has to fit.

    Six hunters cannot identify much, so this checks that the machinery runs and
    names its coefficients, not that the estimates mean anything.  Standard
    errors are deliberately not asserted here: on a dataset this small the
    existing SE path returns None for every model, time-constant ones included,
    so it would not be testing anything about tier_state.  The Hessian itself is
    covered directly in test_occasion_specific_parameters.py.
    """
    data = _context(tmp_path)
    spec = pj.create_formula_spec(phi="~1 + tier_state", p="~1", f="~1")

    result = pj.fit_model(model=PradelModel(), formula=spec, data=data)

    assert result.success
    assert result.parameter_names[:2] == ["phi_(Intercept)", "phi_tier_state_2"]
    assert np.all(np.isfinite(np.asarray(result.parameters)))
