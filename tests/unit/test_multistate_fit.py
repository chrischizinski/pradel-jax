"""The real-data entry point for the five-state model.

test_multistate_recovery.py established that the model can recover what it
generated. These tests check the pieces the real data adds on top: the calendar
(the regime indicators), grouping of identical records, and the initial
distribution at the first occasion, which the recovery studies held fixed.
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
FIRST_TIER2_YEAR = 2021
TRUTH = np.array([-1.6, -1.2, -1.1, -0.8, -1.3, -2.2, -1.8, -2.5, -3.0])
INITIAL_TRUTH = {"new": 0.30, "tier1": 0.55, "out": 0.15}


def _mortality(ages, male):
    return np.clip(
        np.where(male[:, None], 0.0005, 0.0003) * np.exp(0.08 * ages), 0.0, 0.6
    )


def _simulate(n, rng):
    """Histories on the real calendar, keeping only hunters seen at least once."""
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    ages = rng.integers(12, 80, size=n)[:, None] + np.arange(len(YEARS) - 1)[None, :]
    mortality = _mortality(ages, rng.random(n) < 0.9)
    matrices = np.asarray(
        F.transition_matrices(jnp.array(TRUTH), jnp.array(mortality), avail, era)
    )
    initial = np.zeros(N_STATES)
    initial[[0, 1, 3]] = [INITIAL_TRUTH[k] for k in ("new", "tier1", "out")]

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
    return history[seen], mortality[seen], initial


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
    matrices = F.transition_matrices(
        jnp.array(TRUTH), jnp.full((1, len(YEARS) - 1), 0.01), avail, era
    )[0]
    initial = jnp.zeros(N_STATES).at[0].set(1.0)
    assert np.isfinite(float(conditional_log_likelihood(history, matrices, initial)))


def test_grouping_is_exact():
    """Collapsing identical records must not change the likelihood at all."""
    rng = np.random.default_rng(1)
    history, mortality, initial = _simulate(3000, rng)
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    initial = jnp.broadcast_to(jnp.array(initial), (len(history), N_STATES))

    def total(obs, mort, weights):
        matrices = F.transition_matrices(jnp.array(TRUTH), jnp.array(mort), avail, era)
        pi = initial[: len(obs)]
        return float(
            jnp.sum(
                jnp.array(weights)
                * jax.vmap(conditional_log_likelihood)(jnp.array(obs), matrices, pi)
            )
        )

    grouped = F.group_records(history, mortality)
    assert len(grouped[2]) < len(history), "nothing collapsed; test is vacuous"
    assert total(*grouped) == pytest.approx(
        total(history, mortality, np.ones(len(history))), rel=1e-12
    )


@pytest.fixture(scope="module")
def fits():
    rng = np.random.default_rng(3)
    history, mortality, initial = _simulate(80_000, rng)
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    return {
        "fixed": F.fit(history, mortality, avail, era, initial=initial),
        "estimated": F.fit(history, mortality, avail, era),
    }


def test_the_fit_recovers_the_truth_on_the_real_calendar(fits):
    fit = fits["fixed"]
    assert fit["converged"], fit["message"]
    errors = {
        name: fit["estimates"][name] - TRUTH[i] for i, name in enumerate(F.PARAMETERS)
    }
    bad = {k: round(v, 3) for k, v in errors.items() if abs(v) > 0.2}
    assert not bad, f"not recovered (logit-scale error): {bad}"
    true_contrast = (
        TRUTH[F.PARAMETERS.index("out_tier2")]
        - TRUTH[F.PARAMETERS.index("out_tier1_post")]
    )
    assert abs(fit["tier_contrast"] - true_contrast) < 3 * fit["tier_contrast_se"]


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


def test_records_with_no_sighting_are_rejected():
    history = np.zeros((2, len(YEARS)), dtype=np.int32)
    history[0, 3] = OBS_TIER1
    avail, era = F.regime_indicators(YEARS, FIRST_TIER2_YEAR)
    with pytest.raises(ValueError, match="at least one record"):
        F.fit(history, np.zeros((2, len(YEARS) - 1)), avail, era)
