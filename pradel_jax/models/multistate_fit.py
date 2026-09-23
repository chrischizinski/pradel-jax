"""Fit the five-state model to real registration histories.

:mod:`pradel_jax.models.multistate` provides the likelihood of one history given
transition matrices. This module supplies everything between that and a data
file: the parameterisation, the regime indicators, grouping of identical
records, the optimiser and standard errors.

The parameterisation is deliberately small -- nine constant probabilities -- and
is the one the recovery studies in ``tests/unit/test_multistate_recovery.py``
validated. Covariates belong in a later step; the first real-data fit should be
of the model whose behaviour has been measured.

Parameters (all on the logit scale):

    entry           NEW -> active: a hunter not yet in the frame first registers
    out_tier1       Tier I hunter sits the next season out, before Tier II existed
    out_tier1_post  the same, once Tier II was on offer
    out_tier2       Tier II hunter sits the next season out
    ret             a hunter sitting out comes back
    cease           a living hunter leaves for good
    to_tier2        a hunter entering or returning does so at Tier II
    switch_up       Tier I -> Tier II between seasons
    switch_down     Tier II -> Tier I between seasons

The contrast the study is about is ``out_tier2 - out_tier1_post``: Tier II
against Tier I in the same years. It is an association, not the effect of the
tier -- see ``test_self_selection_can_reverse_the_tier_effect``.
"""

from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from .multistate import (
    N_STATES,
    STATE_NEW,
    STATE_OUT,
    STATE_TIER1,
    build_transition_matrices,
    conditional_log_likelihood,
)

PARAMETERS = (
    "entry",
    "out_tier1",
    "out_tier1_post",
    "out_tier2",
    "ret",
    "cease",
    "to_tier2",
    "switch_up",
    "switch_down",
)
# Initial-state logits, relative to TIER1, when the initial distribution is
# estimated rather than fixed.
INITIAL_PARAMETERS = ("initial_new", "initial_out")


def regime_indicators(years: Sequence[int], first_tier2_year: int):
    """The two per-interval indicators the two-tier regulation needs.

    Interval ``t`` runs from occasion ``t`` to ``t + 1``. They answer different
    questions, and conflating them is an off-by-one error with real
    consequences:

    ``tier2_available[t]``: can a hunter *be* in Tier II at the end of the
        interval? True when occasion ``t + 1`` is on or after the first Tier II
        year. If this were keyed to the start instead, a Tier II record in the
        first year of the regime would be impossible under the model and its
        log-likelihood would be -inf.

    ``comparison_era[t]``: is this interval one where a hunter at its start
        could have been in either tier? True when occasion ``t`` is on or after
        the first Tier II year. Tier I sitting-out on these intervals is the
        like-for-like comparator for Tier II.
    """
    years = np.asarray(years)
    tier2_available = (years[1:] >= first_tier2_year).astype(np.float64)
    comparison_era = (years[:-1] >= first_tier2_year).astype(np.float64)
    return tier2_available, comparison_era


def transition_matrices(params, mortality, tier2_available, comparison_era):
    """Logit parameters -> (n, n_intervals, N_STATES, N_STATES) matrices."""
    shape = mortality.shape
    logits = {name: params[i] for i, name in enumerate(PARAMETERS)}
    era = jnp.broadcast_to(jnp.asarray(comparison_era) > 0, shape)
    logits["out_tier1"] = jnp.where(era, logits["out_tier1_post"], logits["out_tier1"])
    del logits["out_tier1_post"]
    probabilities = {
        name: jnp.broadcast_to(jax.nn.sigmoid(value), shape)
        for name, value in logits.items()
    }
    gate = jnp.broadcast_to(jnp.asarray(tier2_available), shape)
    return build_transition_matrices(
        mortality=mortality, tier2_available=gate, **probabilities
    )


def initial_distribution(initial_logits):
    """Softmax over NEW, TIER1 and OUT, with TIER1 as the reference.

    Nobody can start in Tier II (it did not exist at the first occasion) or in
    GONE (they would never be seen, and the data holds only hunters who were).
    """
    new, out = initial_logits[0], initial_logits[1]
    weights = jax.nn.softmax(jnp.stack([new, 0.0, out]))
    initial = jnp.zeros(N_STATES)
    return (
        initial.at[STATE_NEW]
        .set(weights[0])
        .at[STATE_TIER1]
        .set(weights[1])
        .at[STATE_OUT]
        .set(weights[2])
    )


def group_records(observations: np.ndarray, mortality: np.ndarray):
    """Collapse hunters whose history and mortality schedule are identical.

    Exact, not an approximation: such hunters contribute identical terms, so
    one term times a count gives the same likelihood. Mortality comes from a
    life table keyed on age and sex, so hunters of the same age and sex with
    the same history collapse.
    """
    stacked = np.concatenate([observations.astype(np.float64), mortality], axis=1)
    unique, inverse, counts = np.unique(
        stacked, axis=0, return_inverse=True, return_counts=True
    )
    n_occasions = observations.shape[1]
    return (
        unique[:, :n_occasions].astype(np.int32),
        unique[:, n_occasions:],
        counts.astype(np.float64),
    )


def fit(
    observations: np.ndarray,
    mortality: np.ndarray,
    tier2_available: np.ndarray,
    comparison_era: np.ndarray,
    initial: Optional[np.ndarray] = None,
    start: Optional[np.ndarray] = None,
) -> Dict:
    """Maximum-likelihood fit of the constant five-state model.

    Args:
        observations: (n, T) codes OBS_NONE / OBS_TIER1 / OBS_TIER2. Every row
            must contain at least one record.
        mortality: (n, T - 1) annual probability of death for each hunter over
            each interval, from a life table. Fixed, not estimated.
        tier2_available, comparison_era: (T - 1,) from :func:`regime_indicators`.
        initial: (N_STATES,) fixed initial distribution, or None to estimate
            the NEW and OUT weights (2 extra parameters).
        start: optional starting logits.

    Returns:
        dict with ``estimates`` and ``std_errors`` (logit scale, by name),
        ``probabilities``, ``tier_contrast`` and its ``tier_contrast_se``,
        ``log_likelihood``, ``n_parameters``, ``n_individuals``, ``n_groups``,
        ``converged`` and ``message``.
    """
    observations = np.asarray(observations)
    mortality = np.asarray(mortality, dtype=np.float64)
    if (observations.max(axis=1) == 0).any():
        raise ValueError("every history must contain at least one record")

    grouped_obs, grouped_mortality, counts = group_records(observations, mortality)
    obs_j = jnp.asarray(grouped_obs)
    mortality_j = jnp.asarray(grouped_mortality)
    counts_j = jnp.asarray(counts)
    n_groups = len(counts)

    estimate_initial = initial is None
    names = list(PARAMETERS) + (list(INITIAL_PARAMETERS) if estimate_initial else [])
    fixed_initial = None if estimate_initial else jnp.asarray(initial)

    def negative_log_likelihood(x):
        matrices = transition_matrices(
            x[: len(PARAMETERS)], mortality_j, tier2_available, comparison_era
        )
        pi = (
            initial_distribution(x[len(PARAMETERS) :])
            if estimate_initial
            else fixed_initial
        )
        pi = jnp.broadcast_to(pi, (n_groups, N_STATES))
        per_group = jax.vmap(conditional_log_likelihood)(obs_j, matrices, pi)
        return -jnp.sum(counts_j * per_group)

    objective = jax.jit(jax.value_and_grad(negative_log_likelihood))

    def scipy_objective(x):
        value, grad = objective(jnp.asarray(x))
        return float(value), np.asarray(grad, dtype=np.float64)

    x0 = np.full(len(names), -1.5) if start is None else np.asarray(start, float)
    result = minimize(
        scipy_objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 2000, "gtol": 1e-6},
    )

    hessian = np.asarray(jax.hessian(negative_log_likelihood)(jnp.asarray(result.x)))
    try:
        covariance = np.linalg.inv(hessian)
        std_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    except np.linalg.LinAlgError:
        covariance = np.full_like(hessian, np.nan)
        std_errors = np.full(len(names), np.nan)

    i1, i2 = names.index("out_tier1_post"), names.index("out_tier2")
    contrast_variance = (
        covariance[i1, i1] + covariance[i2, i2] - 2.0 * covariance[i1, i2]
    )

    out = {
        "estimates": dict(zip(names, result.x.tolist())),
        "std_errors": dict(zip(names, std_errors.tolist())),
        "probabilities": {
            name: float(jax.nn.sigmoid(result.x[i]))
            for i, name in enumerate(PARAMETERS)
        },
        "tier_contrast": float(result.x[i2] - result.x[i1]),
        "tier_contrast_se": float(np.sqrt(max(contrast_variance, 0.0))),
        "log_likelihood": -float(result.fun),
        "n_parameters": len(names),
        "n_individuals": int(counts.sum()),
        "n_groups": int(n_groups),
        "converged": bool(result.success),
        "message": str(result.message),
    }
    if estimate_initial:
        pi = np.asarray(initial_distribution(result.x[len(PARAMETERS) :]))
        out["initial"] = {
            "new": float(pi[STATE_NEW]),
            "tier1": float(pi[STATE_TIER1]),
            "out": float(pi[STATE_OUT]),
        }
    return out
