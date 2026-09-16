"""The five-state forward recursion must equal brute-force path enumeration.

A hidden-Markov forward pass is easy to write and easy to get subtly wrong:
an off-by-one in which transition matrix applies to which interval, a mask
applied before propagation instead of after, a rescaling that drops a factor.
None of those announce themselves -- the likelihood stays finite, the optimiser
still converges, and the estimates are simply wrong.

So these tests do not check the recursion against a tidied-up version of itself.
They check it against the definition: enumerate every possible sequence of
hidden states, score each one, and add up those consistent with what was
observed. That is exponential and useless at scale, which is the entire reason
the forward algorithm exists -- but at five occasions it is 3,125 paths, exact,
and depends on none of the code under test.
"""

import itertools

import numpy as np
import jax.numpy as jnp
import pytest

from pradel_jax.models.multistate import (
    EMISSION,
    N_STATES,
    OBS_NONE,
    OBS_TIER1,
    OBS_TIER2,
    STATE_GONE,
    STATE_NEW,
    STATE_OUT,
    STATE_TIER1,
    STATE_TIER2,
    conditional_log_likelihood,
    forward_log_likelihood,
    log_prob_ever_observed,
)


def _random_transitions(rng, n_intervals, structured=True):
    """Row-stochastic transition matrices, optionally with the model's structure.

    `structured=False` deliberately allows transitions the real model forbids.
    The recursion must not depend on those zeros being there -- if it did, the
    enumeration check would be comparing two implementations that share an
    assumption, which is exactly the kind of agreement that proves nothing.
    """
    matrices = []
    for _ in range(n_intervals):
        matrix = rng.dirichlet(np.ones(N_STATES), size=N_STATES)
        if structured:
            # GONE is absorbing.
            matrix[STATE_GONE, :] = 0.0
            matrix[STATE_GONE, STATE_GONE] = 1.0
            # NEW is never re-entered, and nobody returns to it.
            matrix[:, STATE_NEW] = 0.0
            matrix[STATE_NEW, STATE_NEW] = rng.uniform(0.2, 0.7)
            # OUT cannot be occupied before a hunter has ever been active.
            matrix[STATE_NEW, STATE_OUT] = 0.0
            matrix[STATE_GONE, :] = 0.0
            matrix[STATE_GONE, STATE_GONE] = 1.0
            rows = matrix.sum(axis=1, keepdims=True)
            matrix = matrix / rows
        matrices.append(matrix)
    return jnp.array(np.stack(matrices), dtype=jnp.float64)


def _random_initial(rng, structured=True):
    """Distribution over states at the first occasion.

    Not everyone starts in NEW. Hunters were already hunting in 2016 -- the
    study window opened on an ongoing process, it did not create one -- so the
    first occasion has mass on the active states and on OUT as well. Putting
    everyone in NEW would make a record at the first occasion impossible, which
    is both wrong and would silently weaken these tests by excluding every
    history that starts with a sighting.

    This mixture is the left-truncation problem in concrete form: its weights
    are not estimable from the data alone, and the study cannot tell a genuine
    new entrant in 2016 from someone who hunted in 2015.
    """
    if structured:
        initial = np.zeros(N_STATES)
        weights = rng.dirichlet(np.ones(3))
        initial[STATE_NEW] = weights[0]
        initial[STATE_TIER1] = weights[1]
        initial[STATE_OUT] = weights[2]
        return jnp.array(initial, dtype=jnp.float64)
    return jnp.array(rng.dirichlet(np.ones(N_STATES)), dtype=jnp.float64)


def _enumerate_log_likelihood(observations, transitions, initial):
    """Sum over every hidden path. The definition, computed the slow way."""
    obs = np.asarray(observations)
    trans = np.asarray(transitions)
    init = np.asarray(initial)
    emission = np.asarray(EMISSION)

    total = 0.0
    for path in itertools.product(range(N_STATES), repeat=len(obs)):
        # A path contributes only if every state it visits could have produced
        # the observation recorded at that occasion.
        if any(emission[obs[t], path[t]] == 0.0 for t in range(len(obs))):
            continue
        probability = init[path[0]]
        for t in range(1, len(obs)):
            probability *= trans[t - 1, path[t - 1], path[t]]
        total += probability
    return np.log(total) if total > 0 else -np.inf


def _enumerate_never_seen(n_occasions, transitions, initial):
    return _enumerate_log_likelihood(
        np.full(n_occasions, OBS_NONE), transitions, initial
    )


HISTORIES = [
    [OBS_NONE, OBS_TIER1, OBS_NONE, OBS_TIER1, OBS_NONE],
    [OBS_TIER1, OBS_TIER1, OBS_TIER2, OBS_NONE, OBS_TIER2],
    [OBS_NONE, OBS_NONE, OBS_TIER2, OBS_NONE, OBS_NONE],
    [OBS_TIER1, OBS_NONE, OBS_NONE, OBS_NONE, OBS_TIER2],
    [OBS_NONE, OBS_NONE, OBS_NONE, OBS_NONE, OBS_TIER1],
    [OBS_TIER2, OBS_NONE, OBS_NONE, OBS_NONE, OBS_NONE],
]


@pytest.mark.parametrize("history", HISTORIES)
def test_forward_equals_enumeration_under_model_structure(history):
    """The case that matters: structured transitions, realistic histories."""
    rng = np.random.default_rng(len(history) * 7 + sum(history))
    observations = jnp.array(history, dtype=jnp.int32)
    transitions = _random_transitions(rng, len(history) - 1, structured=True)
    initial = _random_initial(rng, structured=True)

    fast = float(forward_log_likelihood(observations, transitions, initial))
    slow = _enumerate_log_likelihood(observations, transitions, initial)

    assert fast == pytest.approx(slow, rel=0, abs=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_forward_equals_enumeration_without_structural_zeros(seed):
    """Unrestricted transitions, so the recursion cannot be leaning on structure.

    If the forward pass silently assumed GONE was absorbing, or that NEW could
    not be re-entered, it would still agree with a structured enumeration. It
    would not agree with this one.
    """
    rng = np.random.default_rng(seed)
    history = rng.integers(0, 3, size=5)
    observations = jnp.array(history, dtype=jnp.int32)
    transitions = _random_transitions(rng, 4, structured=False)
    initial = _random_initial(rng, structured=False)

    fast = float(forward_log_likelihood(observations, transitions, initial))
    slow = _enumerate_log_likelihood(observations, transitions, initial)

    assert fast == pytest.approx(slow, rel=0, abs=1e-10)


def test_transition_matrices_are_applied_in_the_right_order():
    """Time-varying transitions must land on the interval they belong to.

    With the same matrix at every interval this test could not fail. The two
    intervals here are deliberately different, and reversing them changes the
    answer -- which is the off-by-one that a constant-rate test would miss.
    """
    observations = jnp.array([OBS_TIER1, OBS_NONE, OBS_TIER2], dtype=jnp.int32)
    initial = jnp.zeros(N_STATES).at[STATE_TIER1].set(1.0)

    early = np.zeros((N_STATES, N_STATES))
    early[STATE_TIER1, STATE_OUT] = 0.9
    early[STATE_TIER1, STATE_TIER1] = 0.1
    late = np.zeros((N_STATES, N_STATES))
    late[STATE_OUT, STATE_TIER2] = 0.5
    late[STATE_OUT, STATE_OUT] = 0.5
    late[STATE_TIER1, STATE_TIER2] = 0.25
    late[STATE_TIER1, STATE_TIER1] = 0.75
    for matrix in (early, late):
        empty = matrix.sum(axis=1) == 0
        matrix[empty, STATE_GONE] = 1.0

    forward_order = jnp.array(np.stack([early, late]), dtype=jnp.float64)
    reversed_order = jnp.array(np.stack([late, early]), dtype=jnp.float64)

    correct = float(forward_log_likelihood(observations, forward_order, initial))
    swapped = float(forward_log_likelihood(observations, reversed_order, initial))

    # Only one route exists: TIER1 -> OUT (0.9) -> TIER2 (0.5).
    assert correct == pytest.approx(np.log(0.9 * 0.5), rel=0, abs=1e-12)
    assert not np.isclose(correct, swapped)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_never_observed_probability_matches_enumeration(seed):
    """The conditioning term has to be right or every estimate is scaled wrong."""
    rng = np.random.default_rng(100 + seed)
    transitions = _random_transitions(rng, 4, structured=True)
    initial = _random_initial(rng, structured=True)

    log_seen = float(log_prob_ever_observed(transitions, initial))
    log_never = _enumerate_never_seen(5, transitions, initial)

    assert log_seen == pytest.approx(np.log(-np.expm1(log_never)), rel=0, abs=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_conditional_likelihood_sums_to_one_over_observable_histories(seed):
    """The real test of the conditioning: it must be a probability distribution.

    Summing the conditional likelihood over every history that could actually
    have been recorded -- that is, every one except all-zeros -- has to give
    exactly 1. If the conditioning term were wrong in any way, this total would
    drift off 1 while each individual value still looked perfectly plausible.
    """
    rng = np.random.default_rng(200 + seed)
    n_occasions = 4
    transitions = _random_transitions(rng, n_occasions - 1, structured=True)
    initial = _random_initial(rng, structured=True)

    total = 0.0
    for history in itertools.product(
        [OBS_NONE, OBS_TIER1, OBS_TIER2], repeat=n_occasions
    ):
        if all(o == OBS_NONE for o in history):
            continue  # never recorded, so never in the data
        observations = jnp.array(history, dtype=jnp.int32)
        total += float(
            np.exp(
                float(conditional_log_likelihood(observations, transitions, initial))
            )
        )

    assert total == pytest.approx(1.0, rel=0, abs=1e-9)
