"""The real-data entry point for the five-state model.

test_multistate_recovery.py established that the model can recover what it
generated. These tests check the pieces the real data adds on top: the calendar
(the regime indicators), grouping of identical records, the initial
distribution at the first occasion, and the two extensions the first real fit
showed were needed -- year-specific entry and an age slope on cessation. Each
extension's tests reproduce the symptom it was added to cure.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pradel_jax.models import multistate_fit as F
from pradel_jax.models.multistate import (
    N_STATES,
    OBS_NONE,
    OBS_TIER1,
    OBS_TIER2,
    STATE_TIER1,
    STATE_TIER2,
    conditional_log_likelihood,
)

YEARS = np.arange(2016, 2026)
N_INTERVALS = len(YEARS) - 1
FIRST_TIER2_YEAR = 2021
TRUTH = {
    "entry": -1.6,
    "out_tier1": -1.2,
    "out_tier1_post": -1.1,
    "out_tier2": -0.8,
    "ret": -1.3,
    "cease": -2.2,
    "to_tier2": -1.8,
    "switch_up": -2.5,
    "switch_down": -3.0,
}
TRUE_CONTRAST = TRUTH["out_tier2"] - TRUTH["out_tier1_post"]
INITIAL_TRUTH = {"new": 0.30, "tier1": 0.55, "out": 0.15}
# Steady recruitment: a NEW hunter enters in each remaining interval with equal
# chance, so (deaths aside) the same number start every year. A constant entry
# rate cannot produce that.
STEADY_ENTRY = {
    f"entry_{t}": float(
        np.log(1 / (N_INTERVALS - t)) - np.log1p(-1 / (N_INTERVALS - t))
    )
    for t in range(N_INTERVALS - 1)
}
# Cessation rising with age: logit slope per year of age, centred at 45.
CEASE_PER_YEAR = 0.04
AGE_REFERENCE = 45.0


def _mortality(ages, male):
    return np.clip(
        np.where(male[:, None], 0.0005, 0.0003) * np.exp(0.08 * ages), 0.0, 0.6
    )


def _simulate(n, rng, params=TRUTH, cease_per_year=0.0, initial_truth=INITIAL_TRUTH):
    """Histories on the real calendar, keeping only hunters seen at least once."""
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    ages = rng.integers(12, 80, size=n)[:, None] + np.arange(N_INTERVALS)[None, :]
    mortality = _mortality(ages, rng.random(n) < 0.9)
    shifted = dict(params)
    shifted["cease_age"] = cease_per_year
    matrices = np.asarray(
        F.transition_matrices(
            {k: jnp.asarray(v) for k, v in shifted.items()},
            jnp.array(mortality),
            avail,
            era,
            age_z=jnp.array(ages - AGE_REFERENCE, dtype=float),
        )
    )
    initial = np.zeros(N_STATES)
    initial[[0, 1, 3]] = [initial_truth[k] for k in ("new", "tier1", "out")]

    def code(states):
        return np.where(
            states == STATE_TIER1,
            OBS_TIER1,
            np.where(states == STATE_TIER2, OBS_TIER2, OBS_NONE),
        )

    states = (rng.random(n)[:, None] > np.cumsum(initial)).sum(axis=1)
    history = np.zeros((n, len(YEARS)), dtype=np.int32)
    history[:, 0] = code(states)
    for t in range(1, len(YEARS)):
        rows = matrices[np.arange(n), t - 1, states]
        states = (rng.random(n)[:, None] > np.cumsum(rows, axis=1)).sum(axis=1)
        history[:, t] = code(states)
    seen = history.max(axis=1) > 0
    return history[seen], mortality[seen], ages[seen], initial


def test_tier2_can_be_recorded_in_its_first_year():
    """The off-by-one that would make real data impossible.

    Tier II existed from 2021, so a 2021 Tier II record is real and must have
    positive probability. That needs Tier II reachable on the interval that
    *ends* in 2021. The like-for-like comparison, by contrast, starts on the
    interval that *begins* in 2021 -- only then could a hunter at its start have
    been in either tier.
    """
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    interval_into_2021 = list(YEARS).index(2021) - 1
    assert avail[interval_into_2021] == 1 and era[interval_into_2021] == 0
    assert avail[interval_into_2021 - 1] == 0
    assert era[interval_into_2021 + 1] == 1

    history = np.zeros(len(YEARS), dtype=np.int32)
    history[list(YEARS).index(2021)] = OBS_TIER2
    params = {k: jnp.asarray(v) for k, v in TRUTH.items()}
    matrices = F.transition_matrices(
        params, jnp.full((1, N_INTERVALS), 0.01), avail, era
    )[0]
    initial = jnp.zeros(N_STATES).at[0].set(1.0)
    assert np.isfinite(float(conditional_log_likelihood(history, matrices, initial)))


def test_grouping_is_exact():
    """Collapsing identical records must not change the likelihood at all."""
    rng = np.random.default_rng(1)
    history, mortality, _, initial = _simulate(3000, rng)
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    params = {k: jnp.asarray(v) for k, v in TRUTH.items()}

    def total(obs, mort, weights):
        matrices = F.transition_matrices(params, jnp.array(mort), avail, era)
        pi = jnp.broadcast_to(jnp.array(initial), (len(obs), N_STATES))
        return float(
            jnp.sum(
                jnp.array(weights)
                * jax.vmap(conditional_log_likelihood)(jnp.array(obs), matrices, pi)
            )
        )

    grouped = F.group_records(history, mortality)
    assert len(grouped[-1]) < len(history), "nothing collapsed; test is vacuous"
    assert total(*grouped) == pytest.approx(
        total(history, mortality, np.ones(len(history))), rel=1e-12
    )


# --------------------------------------------------------------------------
# Constant truth: recovery and the initial distribution
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fits():
    rng = np.random.default_rng(3)
    history, mortality, _, initial = _simulate(80_000, rng)
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    return {
        "fixed": F.fit(
            history, mortality, avail, era, initial=initial, entry="constant"
        ),
        "estimated": F.fit(history, mortality, avail, era, entry="constant"),
        "annual": F.fit(history, mortality, avail, era, entry="annual"),
    }


def test_the_fit_recovers_the_truth_on_the_real_calendar(fits):
    fit = fits["fixed"]
    assert fit["converged"], fit["message"]
    errors = {name: fit["estimates"][name] - value for name, value in TRUTH.items()}
    bad = {k: round(v, 3) for k, v in errors.items() if abs(v) > 0.2}
    assert not bad, f"not recovered (logit-scale error): {bad}"
    assert abs(fit["tier_contrast"] - TRUE_CONTRAST) < 3 * fit["tier_contrast_se"]


def test_the_initial_distribution_is_estimable_and_does_not_move_the_contrast(fits):
    """The 2016 weights can be estimated, and the study's contrast ignores them.

    This was expected to fail: the study cannot tell a genuine 2016 entrant
    from a hunter who lapsed in 2015. Under the model's own assumptions it can,
    because entrants and returners follow different transitions afterwards
    (entry versus ret, and only returners can cease while absent), though the
    OUT weight is the less well determined of the two. On real data that
    identification leans on the constant rates also describing the years
    before 2016, which is why it is reported rather than relied on -- and why
    this test also pins that the tier contrast does not depend on it.
    """
    estimated = fits["estimated"]
    assert estimated["converged"], estimated["message"]
    for state, truth in INITIAL_TRUTH.items():
        assert (
            abs(estimated["initial"][state] - truth) < 0.06
        ), f"initial {state}: {estimated['initial'][state]:.3f} vs {truth}"
    assert estimated["tier_contrast"] == pytest.approx(
        fits["fixed"]["tier_contrast"], abs=0.01
    )


def test_annual_entry_contains_the_constant_model(fits):
    """Year-specific entry loses nothing when recruitment really is constant.

    Conditional on being seen, the constant model's spread of first entries
    over the years is one particular distribution; annual entry can match any.
    So its maximised likelihood can only be higher, and the tier contrast --
    which does not involve entry -- should not move.
    """
    annual, constant = fits["annual"], fits["estimated"]
    assert annual["converged"], annual["message"]
    assert annual["log_likelihood"] >= constant["log_likelihood"] - 0.01
    assert annual["tier_contrast"] == pytest.approx(constant["tier_contrast"], abs=0.01)


# --------------------------------------------------------------------------
# (A) Steady recruitment: why entry is annual
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def steady():
    params = {k: v for k, v in TRUTH.items() if k != "entry"} | STEADY_ENTRY
    rng = np.random.default_rng(8)
    initial_truth = {"new": 0.45, "tier1": 0.42, "out": 0.13}
    history, mortality, _, _ = _simulate(
        80_000, rng, params=params, initial_truth=initial_truth
    )
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    return {
        "constant": F.fit(history, mortality, avail, era, entry="constant"),
        "annual": F.fit(history, mortality, avail, era, entry="annual"),
        "initial_truth": initial_truth,
    }


def test_steady_recruitment_drives_constant_entry_toward_the_boundary(steady):
    """Reproduces the Nebraska symptom, so the cause is known, not guessed.

    A constant entry rate is a pool draining geometrically. Given recruits that
    arrive at a steady rate, the best it can do is an enormous pool with a
    near-zero rate, which mimics steady arrivals closely -- the likelihood is
    barely worse (about 8 units here) -- but mislabels who was who in 2016: the
    Tier I weight collapses from 0.42 to a few percent. Nebraska's fit went the
    rest of the way to the boundary (NEW -> 1, entry -> 0). Here deaths while
    waiting make recruitment decline slightly, which stops it short. The tier
    contrast does not involve entry and is unaffected.
    """
    constant, annual = steady["constant"], steady["annual"]
    assert constant["initial"]["new"] > 0.85, constant["initial"]
    assert constant["initial"]["tier1"] < 0.15, constant["initial"]
    assert constant["probabilities"]["entry"] < 0.02
    assert constant["tier_contrast"] == pytest.approx(annual["tier_contrast"], abs=0.01)


def test_annual_entry_recovers_steady_recruitment(steady):
    annual = steady["annual"]
    assert annual["converged"], annual["message"]
    for state, truth in steady["initial_truth"].items():
        assert abs(annual["initial"][state] - truth) < 0.05, annual["initial"]
    for name, truth in STEADY_ENTRY.items():
        assert (
            abs(annual["estimates"][name] - truth) < 0.25
        ), f"{name}: {annual['estimates'][name]:.3f} vs {truth:.3f}"
    assert abs(annual["tier_contrast"] - TRUE_CONTRAST) < 3 * annual["tier_contrast_se"]
    assert annual["log_likelihood"] > steady["constant"]["log_likelihood"]


# --------------------------------------------------------------------------
# (B) Cessation that rises with age: why cease has an age slope
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def aging():
    rng = np.random.default_rng(12)
    history, mortality, ages, _ = _simulate(80_000, rng, cease_per_year=CEASE_PER_YEAR)
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    out = {}
    for label, factor in (("table", 1.0), ("double", 2.0)):
        scaled = np.clip(mortality * factor, 0.0, 0.95)
        out[("no_age", label)] = F.fit(history, scaled, avail, era)
        out[("age", label)] = F.fit(history, scaled, avail, era, ages=ages)
    return out


def test_unmodelled_age_in_cessation_makes_more_mortality_fit_better(aging):
    """Reproduces the symptom from the first real fit.

    Death and quitting both lead to GONE. If quitting rises with age but the
    model's cessation is constant, the fixed age-varying mortality schedule is
    the only age-dependent route out, so inflating it improves the fit -- even
    though the true schedule is the one given. On the real data the likelihood
    rose from "no mortality" to "life table" to "double"; this shows an age
    effect on quitting is enough to produce that.
    """
    assert (
        aging[("no_age", "double")]["log_likelihood"]
        > aging[("no_age", "table")]["log_likelihood"]
    )


def test_an_age_slope_on_cessation_removes_the_symptom_and_is_recovered(aging):
    """With the slope in the model, the true mortality schedule fits best."""
    table, double = aging[("age", "table")], aging[("age", "double")]
    assert table["converged"], table["message"]
    assert table["log_likelihood"] > double["log_likelihood"]

    per_year = table["estimates"]["cease_age"] / table["age_scale"]
    assert per_year == pytest.approx(CEASE_PER_YEAR, abs=0.006)
    assert abs(table["tier_contrast"] - TRUE_CONTRAST) < 3 * table["tier_contrast_se"]


def test_records_with_no_sighting_are_rejected():
    history = np.zeros((2, len(YEARS)), dtype=np.int32)
    history[0, 3] = OBS_TIER1
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    with pytest.raises(ValueError, match="at least one record"):
        F.fit(history, np.zeros((2, N_INTERVALS)), avail, era)


def test_one_boundary_parameter_does_not_erase_every_standard_error():
    """A parameter at 0 or 1 gets no SE; the others keep theirs.

    On the South Dakota data one entry rate went to 1, the whole Hessian became
    singular, and the tier contrast lost its SE even though nothing about it
    was at a boundary.
    """
    interior = np.array([[4.0, 1.0], [1.0, 2.0]])
    hessian = np.zeros((3, 3))
    hessian[np.ix_([0, 2], [0, 2])] = interior  # parameter 1 is flat
    covariance = F.boundary_aware_covariance(hessian, [False, True, False])
    assert np.isnan(covariance[1]).all() and np.isnan(covariance[:, 1]).all()
    np.testing.assert_allclose(
        covariance[np.ix_([0, 2], [0, 2])], np.linalg.inv(interior)
    )


# --------------------------------------------------------------------------
# Age bands: why the initial weights need age
# --------------------------------------------------------------------------

BANDS = ["young", "middle", "old", "unknown"]
REFERENCE_BAND = 1
# Calibrated to the NE full-model estimates (life table, primary sample),
# collapsed to three bands: the young are mostly yet to start and almost never
# lapsed hunters; older people first seen later lean the other way.
BAND_TRUTH = {
    "initial_new_age": np.array([1.5, -0.8, 0.0]),
    "initial_out_age": np.array([-3.0, 0.15, 0.0]),
    "entry_age": np.array([-0.5, 0.15, 0.0]),
    "out_age": np.array([0.5, -0.2, 0.0]),
    "ret_age": np.array([-0.8, 0.3, 0.0]),
}


def _bands(ages):
    return np.where(ages < 25, 0, np.where(ages < 55, 1, 2))


def _simulate_banded(n, rng):
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    ages = rng.integers(12, 80, size=n)[:, None] + np.arange(N_INTERVALS)[None, :]
    bands = _bands(ages)
    mortality = _mortality(ages, rng.random(n) < 0.9)
    one_hot = jnp.asarray(np.delete(np.eye(len(BANDS))[bands], REFERENCE_BAND, -1))
    params = {k: jnp.asarray(v) for k, v in TRUTH.items() if k != "entry"}
    params |= {k: jnp.asarray(v) for k, v in STEADY_ENTRY.items()}
    params |= {k: jnp.asarray(v) for k, v in BAND_TRUTH.items()}
    matrices = np.asarray(
        F.transition_matrices(params, jnp.asarray(mortality), avail, era, band=one_hot)
    )
    base = jnp.asarray([-0.2, -1.0])  # initial_new, initial_out
    first = one_hot[:, 0, :]
    shifts = jnp.stack(
        [first @ params["initial_new_age"], first @ params["initial_out_age"]], -1
    )
    initial = np.asarray(jax.vmap(lambda s: F.initial_distribution(base + s))(shifts))

    def code(states):
        return np.where(
            states == STATE_TIER1,
            OBS_TIER1,
            np.where(states == STATE_TIER2, OBS_TIER2, OBS_NONE),
        )

    states = (rng.random(n)[:, None] > np.cumsum(initial, axis=1)).sum(axis=1)
    history = np.zeros((n, len(YEARS)), dtype=np.int32)
    history[:, 0] = code(states)
    for t in range(1, len(YEARS)):
        rows = matrices[np.arange(n), t - 1, states]
        states = (rng.random(n)[:, None] > np.cumsum(rows, axis=1)).sum(axis=1)
        history[:, t] = code(states)
    seen = history.max(axis=1) > 0
    return history[seen], mortality[seen], bands[seen]


@pytest.fixture(scope="module")
def banded():
    history, mortality, bands = _simulate_banded(60_000, np.random.default_rng(21))
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    common = dict(age_bands=bands, band_labels=BANDS, reference_band=REFERENCE_BAND)
    out = {}
    for label, effects in (("with_initial", ("entry", "initial", "out", "ret")),):
        for k in (1.0, 2.0):
            out[(label, k)] = F.fit(
                history,
                np.clip(mortality * k, 0.0, 0.95),
                avail,
                era,
                age_effects=effects,
                **common,
            )
    return out


def test_age_band_effects_are_recovered_and_the_true_mortality_fits_best(banded):
    """With age on entry, the initial weights, sitting out and returning, the
    band effects come back, the true mortality schedule fits better than a
    doubled one, and the tier contrast is unaffected.

    What this does NOT show: that leaving age off the initial weights is what
    made the real NE/SD fits prefer an inflated life table. Simulated with
    these same (calibrated) effects, the age-blind model still prefers the
    true mortality, so that mechanism is not established -- only the
    real-data result that adding these effects removed the preference.
    """
    right, inflated = banded[("with_initial", 1.0)], banded[("with_initial", 2.0)]
    assert right["converged"], right["message"]
    assert right["log_likelihood"] > inflated["log_likelihood"]

    for vector, truth in BAND_TRUTH.items():
        for i, label in enumerate(["young", "old"]):
            name = f"{vector[:-4]}_age_{label}"
            assert (
                abs(right["estimates"][name] - truth[i]) < 0.3
            ), f"{name}: {right['estimates'][name]:+.3f} vs {truth[i]:+.3f}"
    assert abs(right["tier_contrast"] - TRUE_CONTRAST) < 3 * right["tier_contrast_se"]


def test_every_start_reaches_the_same_maximum(banded):
    """The staged and cold starts should agree; if not, the fit is multimodal
    and the reported maximum is only the best one found."""
    lls = banded[("with_initial", 1.0)]["start_log_likelihoods"]
    assert len(lls) == 2
    assert max(lls) - min(lls) < 1.0, lls


def test_an_empty_age_band_is_dropped_not_left_to_erase_the_standard_errors(banded):
    """The simulated data has no hunter of unknown age, as NE's primary sample
    has none. That band's coefficients carry no information; kept, they made
    the Hessian singular and the tier contrast's SE NaN on the NE data. They
    must be dropped, and everything else keep its standard error."""
    fit = banded[("with_initial", 1.0)]
    assert all(name.endswith("_age_unknown") for name in fit["dropped_bands"])
    assert len(fit["dropped_bands"]) == 5  # entry, initial_new, initial_out, out, ret
    assert not any(name.endswith("_age_unknown") for name in fit["estimates"])
    assert np.isfinite(fit["tier_contrast_se"]) and fit["tier_contrast_se"] > 0
    assert fit["hessian_min_eigenvalue"] > 0


def test_a_parameter_made_irrelevant_by_another_is_found_and_excluded():
    """An interior parameter with no curvature, not caught by the boundary
    check, must be reported as not estimable instead of erasing every SE.
    Non-finite Hessian rows are handled the same way."""
    interior = np.array([[4.0, 1.0], [1.0, 2.0]])
    hessian = np.zeros((4, 4))
    hessian[np.ix_([0, 2], [0, 2])] = interior  # parameter 1: flat, interior
    hessian[3, :] = hessian[:, 3] = np.nan  # parameter 3: non-finite
    covariance, excluded, smallest = F.estimable_covariance(
        hessian, [False, False, False, False]
    )
    assert sorted(excluded) == [1, 3]  # the flat one and the non-finite one
    assert np.isnan(covariance[3]).all() and np.isnan(covariance[1]).all()
    np.testing.assert_allclose(
        covariance[np.ix_([0, 2], [0, 2])], np.linalg.inv(interior)
    )
    assert smallest > 0
