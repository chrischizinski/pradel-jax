"""Does the five-state likelihood agree with an established package?

The brute-force tests (test_multistate_forward.py) check the recursion against
itself. This checks it against marked (Laake, Johnson & Conn 2013), which a
reader can trust without reading this code -- the answer to "why not MARK?".

marked cannot express the full model: it has no conditioning on "seen at least
once", no NEW state, no Tier II gate and no fixed per-individual mortality. So
both sides fit a reduced model they *can* both express, and it is chosen so
the two parameter spaces are identical, which means their maxima must be too:

    multistate CJS, conditioned on first capture
    strata  A = Tier I, B = Tier II (seen with certainty), C = sitting out (never seen)
    S       one survival for all strata          <- pradel-jax: 1 - cease, mortality 0
    Psi     six free transitions, constant       <- pradel-jax: the sequential
                                                    out / switch / return steps

pradel-jax builds transitions as a chain of conditional steps; marked uses a
multinomial logit per origin. Agreement therefore also checks that the
sequential construction covers the same space and lands on the same point.

What this does not cover, because marked cannot: the ever-seen conditioning,
NEW, the gate, and mortality. Those are covered by enumeration.

The marked side is stored in tests/validation/marked_crosscheck/ so CI needs no
R; that directory's README says how to regenerate it.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from pradel_jax.models.multistate import (
    N_STATES,
    build_transition_matrices,
    forward_log_likelihood,
)

HERE = Path(__file__).resolve().parents[1] / "validation" / "marked_crosscheck"
N_OCCASIONS = 8
NAMES = [
    "cease",
    "out_tier1",
    "switch_up",
    "out_tier2",
    "switch_down",
    "ret",
    "to_tier2",
]
TRUTH = np.array([-2.2, -1.2, -2.5, -0.9, -3.0, -1.3, -1.8])
CODES = {"0": 0, "A": 1, "B": 2}


def reduced_matrices(params, n_individuals):
    """The reduced model: no mortality, Tier II always available."""
    shape = (n_individuals, N_OCCASIONS - 1)
    probabilities = {
        name: jnp.broadcast_to(jax.nn.sigmoid(params[i]), shape)
        for i, name in enumerate(NAMES)
    }
    # NEW is unreachable once histories start at first capture; any value will do.
    probabilities["entry"] = jnp.full(shape, 0.5)
    return build_transition_matrices(
        mortality=jnp.zeros(shape), tier2_available=jnp.ones(shape), **probabilities
    )


def _first_capture_inputs(history):
    """Condition on first capture using the unmodified forward recursion.

    Before first capture the transitions are replaced by the identity and the
    first observation is repeated, so those occasions contribute probability 1,
    and the hunter starts with certainty in the state they were seen in.
    Observation codes for the two tiers equal their state indices.
    """
    n = len(history)
    first = np.argmax(history > 0, axis=1)
    observations = history.copy()
    for i in range(n):
        observations[i, : first[i]] = history[i, first[i]]
    initial = np.zeros((n, N_STATES))
    initial[np.arange(n), history[np.arange(n), first]] = 1.0
    before = np.arange(N_OCCASIONS - 1)[None, :] < first[:, None]
    return jnp.array(observations), jnp.array(initial), jnp.array(before)


def _read_histories():
    lines = (HERE / "histories.csv").read_text().split()[1:]
    return np.array([[CODES[c] for c in line] for line in lines], dtype=np.int32)


@pytest.fixture(scope="module")
def both():
    history = _read_histories()
    observations, initial, before = _first_capture_inputs(history)
    identity = jnp.eye(N_STATES)

    def negative_log_likelihood(params):
        matrices = reduced_matrices(params, len(history))
        matrices = jnp.where(before[:, :, None, None], identity, matrices)
        return -jnp.sum(
            jax.vmap(forward_log_likelihood)(observations, matrices, initial)
        )

    objective = jax.jit(jax.value_and_grad(negative_log_likelihood))

    def scipy_objective(x):
        value, grad = objective(jnp.array(x))
        return float(value), np.asarray(grad, dtype=np.float64)

    result = minimize(
        scipy_objective,
        np.zeros(len(NAMES)),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "gtol": 1e-9},
    )
    assert result.success, result.message
    p = dict(zip(NAMES, jax.nn.sigmoid(jnp.array(result.x)).tolist()))
    ours = {
        "loglik": -float(result.fun),
        "S": 1.0 - p["cease"],
        "psi": {
            "A": {
                "A": (1 - p["out_tier1"]) * (1 - p["switch_up"]),
                "B": (1 - p["out_tier1"]) * p["switch_up"],
                "C": p["out_tier1"],
            },
            "B": {
                "A": (1 - p["out_tier2"]) * p["switch_down"],
                "B": (1 - p["out_tier2"]) * (1 - p["switch_down"]),
                "C": p["out_tier2"],
            },
            "C": {
                "A": p["ret"] * (1 - p["to_tier2"]),
                "B": p["ret"] * p["to_tier2"],
                "C": 1 - p["ret"],
            },
        },
    }
    marked = json.loads((HERE / "marked_results.json").read_text())
    return ours, marked


def test_maximum_log_likelihood_matches_marked(both):
    """Same data, same parameter space: the maximised likelihoods must be equal.

    The tolerance is the optimisers' own noise, not slack for disagreement --
    the two agreed to 4e-7 when this was set up. A bug in the recursion, the
    unobservable state, or first-capture conditioning moves this by whole units.
    """
    ours, marked = both
    assert ours["loglik"] == pytest.approx(marked["loglik"], abs=1e-3)


def test_fitted_probabilities_match_marked(both):
    """Equal likelihoods at different points would mean a different model.

    Checks survival and all nine transitions, which is also what shows the
    sequential construction and marked's multinomial logit are the same space.
    """
    ours, marked = both
    assert ours["S"] == pytest.approx(marked["S"], abs=1e-4)
    for origin, row in marked["psi"].items():
        for destination, value in row.items():
            assert ours["psi"][origin][destination] == pytest.approx(
                value, abs=1e-4
            ), f"Psi {origin}->{destination}"
