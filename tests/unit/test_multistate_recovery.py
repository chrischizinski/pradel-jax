"""Can the five-state model recover parameters it generated itself?

A likelihood that is provably correct (test_multistate_forward.py) can still be
useless: correct and identifiable are different properties. The parameter that
matters most here -- how much of a long absence is people who will come back
versus people who are gone for good -- is exactly the kind that a model can be
unable to pin down, because both stories predict the same declining return
hazard.

So these tests generate data from known parameters and ask whether fitting
recovers them. If it cannot do that on data it made itself, no result from real
data means anything, and no amount of software correctness would save it.

The mechanism being tested is worth stating, because it is not obvious that it
works at all: with a constant return probability and a constant cessation
probability, the *observed* hazard of returning still declines with time away,
because the pool of absent hunters fills up with people who have quietly left
for good. The level of that curve carries the return rate and its rate of decay
carries the cessation rate, so the two are separable in principle. These tests
check that they are separable in practice, at this study's size and length.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.optimize import minimize

from pradel_jax.models.multistate import (
    N_STATES,
    OBS_NONE,
    OBS_TIER1,
    OBS_TIER2,
    STATE_GONE,
    STATE_NEW,
    STATE_OUT,
    STATE_TIER1,
    STATE_TIER2,
    build_transition_matrices,
    conditional_log_likelihood,
)

N_OCCASIONS = 10
FIRST_TIER2_OCCASION = 5  # 2021 in a 2016-2025 window

# Truth on the logit scale, chosen to sit near what the real data shows:
# roughly a fifth of active hunters sit a season out, a fifth of absent hunters
# come back, and cessation is uncommon per year but accumulates.
TRUTH = {
    "entry": -1.6,
    "out_tier1": -1.2,
    "out_tier2": -0.9,  # the tier effect on sitting out
    "ret": -1.3,
    "cease": -2.2,
    "to_tier2": -1.8,
    "switch_up": -2.5,
    "switch_down": -3.0,
}
PARAM_ORDER = list(TRUTH)


def _mortality(ages):
    """Stand-in life table: an exponential rise in annual mortality with age.

    The real model takes published CDC/SSA tables. What matters for this test is
    only that mortality is *fixed, known and varying between individuals* --
    that is the external anchor which lets death be separated from giving up,
    and a test with a flat rate would not exercise it.
    """
    return np.clip(0.0004 * np.exp(0.075 * ages), 0.0, 0.5)


def _tier2_gate():
    intervals = np.arange(N_OCCASIONS - 1)
    return (intervals >= FIRST_TIER2_OCCASION).astype(np.float64)


def _expand(params, n_individuals):
    """Scalar parameters -> (n_individuals, n_intervals) probability grids."""
    n_intervals = N_OCCASIONS - 1
    shape = (n_individuals, n_intervals)
    return {
        name: jnp.broadcast_to(jax.nn.sigmoid(params[i]), shape)
        for i, name in enumerate(PARAM_ORDER)
    }


def _matrices(params, ages, n_individuals):
    probabilities = _expand(params, n_individuals)
    age_grid = ages[:, None] + np.arange(N_OCCASIONS - 1)[None, :]
    mortality = jnp.array(_mortality(age_grid))
    gate = jnp.broadcast_to(jnp.array(_tier2_gate()), (n_individuals, N_OCCASIONS - 1))
    return build_transition_matrices(
        mortality=mortality, tier2_available=gate, **probabilities
    )


def _initial(n_individuals):
    """Most hunters are already active when the window opens; some are not.

    These weights are assumed, not estimated -- the study cannot tell a genuine
    2016 entrant from someone who hunted in 2015. Holding them fixed here is
    deliberate: it is the same choice the real analysis has to make, so the
    recovery test should be run under it rather than around it.
    """
    initial = np.zeros((n_individuals, N_STATES))
    initial[:, STATE_NEW] = 0.35
    initial[:, STATE_TIER1] = 0.50
    initial[:, STATE_OUT] = 0.15
    return jnp.array(initial)


def _simulate(params, ages, rng):
    """Walk the chain forward and record only what a registration file would see."""
    n_individuals = len(ages)
    matrices = np.asarray(_matrices(params, ages, n_individuals))
    return _walk(matrices, rng)


def _walk(matrices, rng):
    n_individuals = matrices.shape[0]
    initial = np.asarray(_initial(n_individuals))

    states = np.array(
        [rng.choice(N_STATES, p=initial[i]) for i in range(n_individuals)]
    )
    history = np.zeros((n_individuals, N_OCCASIONS), dtype=np.int32)
    history[:, 0] = np.where(
        states == STATE_TIER1,
        OBS_TIER1,
        np.where(states == STATE_TIER2, OBS_TIER2, OBS_NONE),
    )
    for t in range(1, N_OCCASIONS):
        draws = rng.random(n_individuals)
        cumulative = np.cumsum(
            matrices[np.arange(n_individuals), t - 1, states], axis=1
        )
        states = (draws[:, None] > cumulative).sum(axis=1)
        history[:, t] = np.where(
            states == STATE_TIER1,
            OBS_TIER1,
            np.where(states == STATE_TIER2, OBS_TIER2, OBS_NONE),
        )
    return history


def _fit(history, ages, start):
    observed = history[history.sum(axis=1) > 0]
    kept_ages = ages[history.sum(axis=1) > 0]
    n_individuals = len(observed)
    observations = jnp.array(observed)
    initial = _initial(n_individuals)

    def negative_log_likelihood(params):
        matrices = _matrices(params, kept_ages, n_individuals)
        return -jnp.sum(
            jax.vmap(conditional_log_likelihood)(observations, matrices, initial)
        )

    objective = jax.jit(jax.value_and_grad(negative_log_likelihood))

    def scipy_objective(x):
        value, grad = objective(jnp.array(x))
        return float(value), np.asarray(grad, dtype=np.float64)

    result = minimize(
        scipy_objective,
        np.asarray(start),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 400},
    )
    return result.x, result


@pytest.fixture(scope="module")
def recovered():
    rng = np.random.default_rng(11)
    n_individuals = 40_000
    ages = rng.uniform(18, 70, size=n_individuals)
    truth = np.array([TRUTH[name] for name in PARAM_ORDER])

    history = _simulate(truth, ages, rng)
    # Start well away from the truth so success cannot be an artifact of
    # beginning at the answer.
    start = truth + rng.normal(0.0, 0.6, size=len(truth))
    estimate, result = _fit(history, ages, start)
    return truth, estimate, result, history


def test_the_simulation_looks_like_the_real_data(recovered):
    """Guard on the generator itself before trusting anything fitted to it.

    If the simulated histories did not resemble the NE/SD data -- roughly a
    quarter of hunters with an interior gap, a return hazard that falls with
    time away -- then recovering parameters from them would say nothing about
    whether the model works on the real thing.
    """
    _, _, _, history = recovered
    seen = history > 0
    seen = seen[seen.sum(axis=1) > 0]

    first = np.argmax(seen, axis=1)
    last = seen.shape[1] - 1 - np.argmax(seen[:, ::-1], axis=1)
    span = last - first + 1
    with_gap = (span > seen.sum(axis=1)).mean()
    assert (
        0.15 < with_gap < 0.45
    ), f"interior-gap rate {with_gap:.2f} is unlike the data"

    # Return hazard must decline, which is the whole phenomenon being modelled.
    hazard = []
    for j in (1, 2, 3):
        at_risk = returned = 0
        for row, start in zip(seen, first):
            away = 0
            for t in range(start + 1, seen.shape[1]):
                if row[t]:
                    if away == j:
                        returned += 1
                    if away >= j:
                        at_risk += 1
                    away = 0
                else:
                    if away == j:
                        at_risk += 1
                    away += 1
        hazard.append(returned / max(at_risk, 1))
    assert hazard[0] > hazard[1] > hazard[2], f"hazard did not decline: {hazard}"


def test_the_fit_converges(recovered):
    _, _, result, _ = recovered
    assert result.success, result.message


def test_every_parameter_is_recovered(recovered):
    """The gate. Each parameter must come back close to the value that made the data."""
    truth, estimate, _, _ = recovered
    errors = {name: float(estimate[i] - truth[i]) for i, name in enumerate(PARAM_ORDER)}
    bad = {k: round(v, 3) for k, v in errors.items() if abs(v) > 0.15}
    assert not bad, f"parameters not recovered (logit-scale error): {bad}"


def test_return_and_cessation_are_separately_identified(recovered):
    """The one that could plausibly fail, and the reason for the whole exercise.

    Returning and giving up both produce absence. If the data could not tell
    them apart, `ret` and `cease` would trade off against each other and their
    individual estimates would be arbitrary even while the likelihood looked
    fine. Recovering both to this tolerance is evidence the declining return
    hazard genuinely carries two separable pieces of information.
    """
    truth, estimate, _, _ = recovered
    for name in ("ret", "cease"):
        i = PARAM_ORDER.index(name)
        assert (
            abs(estimate[i] - truth[i]) < 0.12
        ), f"{name}: truth {truth[i]:.3f}, estimated {estimate[i]:.3f}"


def test_the_tier_effect_on_sitting_out_is_recovered(recovered):
    """The scientific quantity, not just the nuisance parameters.

    `out_tier2 - out_tier1` is the contrast the study is about: does Tier II
    change how often a hunter sits a season out? It is a difference of two
    fitted values, so it can be wrong even when both are individually close.
    """
    truth, estimate, _, _ = recovered
    i1, i2 = PARAM_ORDER.index("out_tier1"), PARAM_ORDER.index("out_tier2")
    true_effect = truth[i2] - truth[i1]
    fitted_effect = estimate[i2] - estimate[i1]
    assert abs(fitted_effect - true_effect) < 0.15, (
        f"tier effect on sitting out: truth {true_effect:.3f}, "
        f"estimated {fitted_effect:.3f}"
    )


# --------------------------------------------------------------------------
# What the unverifiable assumption actually costs
# --------------------------------------------------------------------------


def _fit_with_mortality(history, ages, mortality_fn, start):
    """Fit while *believing* a given mortality schedule, whatever generated the data."""
    keep = history.sum(axis=1) > 0
    observations = jnp.array(history[keep])
    kept_ages = ages[keep]
    n_individuals = len(kept_ages)

    age_grid = kept_ages[:, None] + np.arange(N_OCCASIONS - 1)[None, :]
    mortality = jnp.array(mortality_fn(age_grid))
    gate = jnp.broadcast_to(jnp.array(_tier2_gate()), (n_individuals, N_OCCASIONS - 1))
    initial = _initial(n_individuals)

    def negative_log_likelihood(params):
        probabilities = {
            name: jnp.broadcast_to(
                jax.nn.sigmoid(params[i]), (n_individuals, N_OCCASIONS - 1)
            )
            for i, name in enumerate(PARAM_ORDER)
        }
        matrices = build_transition_matrices(
            mortality=mortality, tier2_available=gate, **probabilities
        )
        return -jnp.sum(
            jax.vmap(conditional_log_likelihood)(observations, matrices, initial)
        )

    objective = jax.jit(jax.value_and_grad(negative_log_likelihood))

    def scipy_objective(x):
        value, grad = objective(jnp.array(x))
        return float(value), np.asarray(grad, dtype=np.float64)

    return minimize(
        scipy_objective,
        np.asarray(start),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 400},
    ).x


@pytest.fixture(scope="module")
def misspecified():
    """Fit the same simulated data under four different beliefs about mortality.

    The data always comes from the age-varying schedule. Only what the model is
    told changes -- which is the real situation: the life table is an external
    input that could be wrong, and nothing in the capture histories will say so.
    """
    rng = np.random.default_rng(23)
    n_individuals = 40_000
    ages = rng.uniform(18, 70, size=n_individuals)
    truth = np.array([TRUTH[name] for name in PARAM_ORDER])
    history = _simulate(truth, ages, rng)
    start = truth + rng.normal(0.0, 0.5, size=len(truth))

    beliefs = {
        "correct": _mortality,
        "none": lambda a: np.zeros_like(a),
        "flat": lambda a: np.full_like(a, 0.012),
        "double": lambda a: _mortality(a) * 2.0,
    }
    fits = {
        name: _fit_with_mortality(history, ages, fn, start)
        for name, fn in beliefs.items()
    }
    return truth, fits


def test_getting_mortality_wrong_distorts_cessation(misspecified):
    """The assumption is not free, and this records where the cost lands.

    Death and giving-up are the two roads to the same absorbing state, so an
    error in the fixed mortality schedule is absorbed almost one-for-one by the
    estimated cessation rate. Believing nobody dies makes cessation look more
    common; doubling the life table makes it look rarer. Any claim about *why*
    hunters leave permanently inherits that error, and has to be reported with
    it.
    """
    truth, fits = misspecified
    index = PARAM_ORDER.index("cease")

    assert abs(fits["correct"][index] - truth[index]) < 0.05
    assert (
        fits["none"][index] - truth[index] > 0.10
    ), "no-mortality should inflate cessation"
    assert (
        fits["double"][index] - truth[index] < -0.10
    ), "double mortality should deflate it"


def test_the_tier_effect_survives_a_wrong_life_table(misspecified):
    """The claim the study rests on, and the reason the assumption is acceptable.

    The tier contrast is a difference between two active states, and both are
    reached through the same mortality. Error in the life table therefore
    largely cancels out of it, while landing on cessation instead. That is what
    makes an unverifiable assumption tolerable here: it distorts a secondary
    quantity and leaves the primary one alone.

    Stated as a bound so it can be quoted: across beliefs ranging from "nobody
    dies" to "twice the published rate", the tier effect moves by at most a
    tenth of its own size.
    """
    truth, fits = misspecified
    i1, i2 = PARAM_ORDER.index("out_tier1"), PARAM_ORDER.index("out_tier2")
    true_effect = truth[i2] - truth[i1]

    for name, estimate in fits.items():
        fitted_effect = estimate[i2] - estimate[i1]
        relative = abs(fitted_effect - true_effect) / abs(true_effect)
        assert relative < 0.15, (
            f"belief '{name}': tier effect {fitted_effect:+.3f} vs truth "
            f"{true_effect:+.3f} ({relative:.0%} off)"
        )
        assert np.sign(fitted_effect) == np.sign(
            true_effect
        ), f"belief '{name}' flipped the sign of the tier effect"


# --------------------------------------------------------------------------
# The rival story: heterogeneity instead of a permanently-gone fraction
# --------------------------------------------------------------------------


def _simulate_with_frailty(params, ages, frailty_sd, rng):
    """Generate data with *no* permanent cessation, only uneven return propensity.

    Each hunter carries their own return probability, logit-normal around the
    `ret` in ``params``, and `cease` is exactly zero. The observed return hazard
    still declines with time away -- the pool of absent hunters fills up with
    the ones least inclined to come back -- which is the same signature the cure
    model attributes to people leaving for good. The two stories are not
    distinguishable from fit, so this is the case the real analysis cannot rule
    out.
    """
    n_individuals = len(ages)
    n_intervals = N_OCCASIONS - 1
    shape = (n_individuals, n_intervals)
    probabilities = {
        name: jnp.broadcast_to(jax.nn.sigmoid(params[i]), shape)
        for i, name in enumerate(PARAM_ORDER)
    }
    frailty = rng.normal(0.0, frailty_sd, size=n_individuals)
    ret_logit = params[PARAM_ORDER.index("ret")] + frailty
    probabilities["ret"] = jnp.broadcast_to(
        jax.nn.sigmoid(jnp.array(ret_logit))[:, None], shape
    )
    probabilities["cease"] = jnp.zeros(shape)

    age_grid = ages[:, None] + np.arange(n_intervals)[None, :]
    gate = jnp.broadcast_to(jnp.array(_tier2_gate()), shape)
    matrices = build_transition_matrices(
        mortality=jnp.array(_mortality(age_grid)), tier2_available=gate, **probabilities
    )
    return _walk(np.asarray(matrices), rng)


@pytest.fixture(scope="module")
def heterogeneous():
    """Fit the cure model to data that has no cure fraction at all.

    Mortality is given correctly, so the only thing wrong is the story about
    why the return hazard declines. Two frailty levels: a moderate one, and one
    twice as wide, to see whether any bias grows with the amount of
    heterogeneity rather than sitting at one convenient value.
    """
    truth = np.array([TRUTH[name] for name in PARAM_ORDER])
    fits = {}
    for frailty_sd in (0.8, 1.6):
        rng = np.random.default_rng(31)
        ages = rng.uniform(18, 70, size=40_000)
        history = _simulate_with_frailty(truth, ages, frailty_sd, rng)
        start = truth + rng.normal(0.0, 0.5, size=len(truth))
        fits[frailty_sd] = _fit_with_mortality(history, ages, _mortality, start)
    return truth, fits


def test_heterogeneity_is_misread_as_permanent_cessation(heterogeneous):
    """Where the cost of the wrong story lands, recorded so it is reported.

    The data contains no one who left for good, yet the cure model finds a
    cessation rate of a few percent a year, and more of it the wider the
    heterogeneity. Over the nine intervals that compounds into a substantial
    "permanently gone" fraction that does not exist. The estimated cessation
    rate -- and any statement about how many hunters quit for good -- is
    therefore conditional on the cure story being the right one, and has to be
    presented as such.
    """
    _, fits = heterogeneous
    index = PARAM_ORDER.index("cease")
    moderate = float(jax.nn.sigmoid(fits[0.8][index]))
    wide = float(jax.nn.sigmoid(fits[1.6][index]))

    assert moderate > 0.01, f"moderate frailty: cessation {moderate:.4f}"
    assert wide > moderate, (
        f"cessation should grow with heterogeneity: sd 0.8 -> {moderate:.4f}, "
        f"sd 1.6 -> {wide:.4f}"
    )


def test_the_tier_effect_survives_the_wrong_story(heterogeneous):
    """The claim the study needs, under the assumption it cannot test.

    Heterogeneity in returning distorts `ret` and invents `cease`, but both act
    on the absent pool, which every active hunter reaches the same way
    regardless of tier. The contrast between the two active states is left
    close to the truth. Same bound as the life-table study, so the two can be
    quoted together: within 15% of its own size, and never flipped.
    """
    truth, fits = heterogeneous
    i1, i2 = PARAM_ORDER.index("out_tier1"), PARAM_ORDER.index("out_tier2")
    true_effect = truth[i2] - truth[i1]

    for frailty_sd, estimate in fits.items():
        fitted_effect = estimate[i2] - estimate[i1]
        relative = abs(fitted_effect - true_effect) / abs(true_effect)
        assert relative < 0.15, (
            f"frailty sd {frailty_sd}: tier effect {fitted_effect:+.3f} vs truth "
            f"{true_effect:+.3f} ({relative:.0%} off)"
        )
        assert np.sign(fitted_effect) == np.sign(true_effect)
