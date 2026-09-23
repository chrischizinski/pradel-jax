"""Fit the five-state model to real registration histories.

:mod:`pradel_jax.models.multistate` provides the likelihood of one history given
transition matrices. This module supplies everything between that and a data
file: the parameterisation, the regime indicators, grouping of identical
records, the optimiser and standard errors.

The parameterisation is the one the recovery studies in
``tests/unit/test_multistate_recovery.py`` validated, with two extensions that
real data turned out to need (each switched by an argument to :func:`fit`):

    entry           NEW -> active. ``entry="constant"``: one rate for every
                    year. ``entry="annual"``: one rate per interval, the last
                    fixed at 1 (see :func:`parameter_names`).
    out_tier1       Tier I hunter sits the next season out, before Tier II existed
    out_tier1_post  the same, once Tier II was on offer
    out_tier2       Tier II hunter sits the next season out
    ret             a hunter sitting out comes back
    cease           a living hunter leaves for good
    cease_age       with ``ages`` given: slope of ``cease`` on standardised age
    to_tier2        a hunter entering or returning does so at Tier II
    switch_up       Tier I -> Tier II between seasons
    switch_down     Tier II -> Tier I between seasons

All on the logit scale. Tier is chosen by the hunter each year, so the
switches are a real part of the process, not a nuisance.

The contrast the study is about is ``out_tier2 - out_tier1_post``: Tier II
against Tier I in the same years. It is an association, not the effect of the
tier -- see ``test_self_selection_can_reverse_the_tier_effect``.
"""

from typing import Dict, List, Optional, Sequence

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

BEHAVIOUR = (
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


def parameter_names(
    n_intervals: int,
    entry: str = "annual",
    cease_age: bool = False,
    estimate_initial: bool = True,
) -> List[str]:
    """The free parameters, in the order the optimiser sees them.

    With ``entry="annual"`` there is one entry rate per interval except the
    last, which is fixed at 1. That is a definition, not an assumption. Once the
    likelihood is conditioned on being seen, hunters who would never have
    entered during the study are invisible, so how many of them there are
    cannot be estimated. Fixing the last rate at 1 defines NEW as "will start
    hunting within the study window", which removes that unidentifiable mass.
    It is the same convention as the Schwarz-Arnason entry probabilities
    summing to one.

    A single constant rate cannot do this. It implies a finite pool that
    drains geometrically, so recruits must decline year on year; data with
    steady recruitment push it to the boundary of an infinite pool and a zero
    rate. That is what happened on the Nebraska data.
    """
    if entry == "constant":
        names = ["entry"]
    elif entry == "annual":
        names = [f"entry_{t}" for t in range(n_intervals - 1)]
    else:
        raise ValueError(f"entry must be 'constant' or 'annual', not {entry!r}")
    names += list(BEHAVIOUR)
    if cease_age:
        names.append("cease_age")
    if estimate_initial:
        names += list(INITIAL_PARAMETERS)
    return names


# A probability-type logit beyond this is within ~3e-7 of 0 or 1: at the
# boundary of the parameter space, where the likelihood is flat in it.
BOUNDARY_LOGIT = 15.0


def boundary_aware_covariance(hessian: np.ndarray, at_boundary: Sequence[bool]):
    """Inverse Hessian over the interior parameters; NaN for boundary ones.

    A parameter at the boundary has a (numerically) zero row in the Hessian, so
    inverting the whole matrix fails or returns garbage -- and one such
    parameter would wipe out every standard error, including the tier
    contrast's. The usual practice (MARK does the same) is to treat boundary
    estimates as fixed: invert over the rest and report no SE for them.
    """
    at_boundary = np.asarray(at_boundary, dtype=bool)
    covariance = np.full_like(hessian, np.nan)
    interior = np.where(~at_boundary)[0]
    try:
        covariance[np.ix_(interior, interior)] = np.linalg.inv(
            hessian[np.ix_(interior, interior)]
        )
    except np.linalg.LinAlgError:
        pass
    return covariance


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


def transition_matrices(
    params: Dict, mortality, tier2_available, comparison_era, age_z=None
):
    """Named logit parameters -> (n, n_intervals, N_STATES, N_STATES) matrices.

    ``params`` holds ``entry`` or ``entry_0 .. entry_{T-3}`` (see
    :func:`parameter_names`), every name in :data:`BEHAVIOUR`, and
    ``cease_age`` when ``age_z`` -- standardised age, shaped like
    ``mortality`` -- is given.
    """
    shape = mortality.shape
    n_intervals = shape[-1]

    if "entry" in params:
        entry = jax.nn.sigmoid(params["entry"])
    else:
        free = jnp.stack([params[f"entry_{t}"] for t in range(n_intervals - 1)])
        entry = jnp.concatenate([jax.nn.sigmoid(free), jnp.ones(1)])

    era = jnp.broadcast_to(jnp.asarray(comparison_era) > 0, shape)
    logits = {
        "out_tier1": jnp.where(era, params["out_tier1_post"], params["out_tier1"]),
        "cease": params["cease"]
        + (0.0 if age_z is None else params["cease_age"] * age_z),
    }
    for name in ("out_tier2", "ret", "to_tier2", "switch_up", "switch_down"):
        logits[name] = params[name]

    probabilities = {
        name: jnp.broadcast_to(jax.nn.sigmoid(value), shape)
        for name, value in logits.items()
    }
    probabilities["entry"] = jnp.broadcast_to(entry, shape)
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


def group_records(observations: np.ndarray, *per_interval: np.ndarray):
    """Collapse hunters whose history and every per-interval input are identical.

    Exact, not an approximation: such hunters contribute identical terms, so
    one term times a count gives the same likelihood. Mortality and age come
    from age and sex, so hunters of the same age and sex with the same history
    collapse.

    Returns the grouped observations, each grouped per-interval array in the
    order given, and the counts.
    """
    stacked = np.concatenate(
        [observations.astype(np.float64)] + [np.asarray(a) for a in per_interval],
        axis=1,
    )
    unique, counts = np.unique(stacked, axis=0, return_counts=True)
    n_occasions = observations.shape[1]
    n_intervals = n_occasions - 1
    pieces = [unique[:, :n_occasions].astype(np.int32)]
    for k in range(len(per_interval)):
        start = n_occasions + k * n_intervals
        pieces.append(unique[:, start : start + n_intervals])
    return (*pieces, counts.astype(np.float64))


def fit(
    observations: np.ndarray,
    mortality: np.ndarray,
    tier2_available: np.ndarray,
    comparison_era: np.ndarray,
    initial: Optional[np.ndarray] = None,
    entry: str = "annual",
    ages: Optional[np.ndarray] = None,
    start: Optional[np.ndarray] = None,
) -> Dict:
    """Maximum-likelihood fit of the five-state model.

    Args:
        observations: (n, T) codes OBS_NONE / OBS_TIER1 / OBS_TIER2. Every row
            must contain at least one record.
        mortality: (n, T - 1) annual probability of death for each hunter over
            each interval, from a life table. Fixed, not estimated.
        tier2_available, comparison_era: (T - 1,) from :func:`regime_indicators`.
        initial: (N_STATES,) fixed initial distribution, or None to estimate
            the NEW and OUT weights (2 extra parameters).
        entry: ``"annual"`` (default) or ``"constant"``; see
            :func:`parameter_names`.
        ages: optional (n, T - 1) age over each interval. Given, cessation gets
            a slope on age, standardised by the mean and SD of these values
            (returned as ``age_center`` and ``age_scale``).
        start: optional starting logits.

    Returns:
        dict with ``estimates`` and ``std_errors`` (logit scale, by name),
        ``probabilities`` (for the intercept-type parameters),
        ``tier_contrast`` and its ``tier_contrast_se``, ``log_likelihood``,
        ``n_parameters``, ``n_individuals``, ``n_groups``, ``converged`` and
        ``message``; ``initial`` when estimated; ``age_center`` and
        ``age_scale`` when ``ages`` is given.
    """
    observations = np.asarray(observations)
    mortality = np.asarray(mortality, dtype=np.float64)
    if (observations.max(axis=1) == 0).any():
        raise ValueError("every history must contain at least one record")
    n_intervals = observations.shape[1] - 1

    extras = [mortality]
    if ages is not None:
        ages = np.asarray(ages, dtype=np.float64)
        age_center, age_scale = float(ages.mean()), float(ages.std())
        extras.append((ages - age_center) / age_scale)
    grouped = group_records(observations, *extras)
    obs_j = jnp.asarray(grouped[0])
    mortality_j = jnp.asarray(grouped[1])
    age_z_j = jnp.asarray(grouped[2]) if ages is not None else None
    counts_j = jnp.asarray(grouped[-1])
    n_groups = len(grouped[-1])

    estimate_initial = initial is None
    names = parameter_names(
        n_intervals,
        entry=entry,
        cease_age=ages is not None,
        estimate_initial=estimate_initial,
    )
    fixed_initial = None if estimate_initial else jnp.asarray(initial)

    def negative_log_likelihood(x):
        params = dict(zip(names, x))
        matrices = transition_matrices(
            params, mortality_j, tier2_available, comparison_era, age_z_j
        )
        pi = (
            initial_distribution(jnp.stack([params[k] for k in INITIAL_PARAMETERS]))
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

    if start is None:
        x0 = np.full(len(names), -1.5)
        if "cease_age" in names:
            x0[names.index("cease_age")] = 0.0
    else:
        x0 = np.asarray(start, float)
    result = minimize(
        scipy_objective,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 3000, "gtol": 1e-6},
    )

    hessian = np.asarray(jax.hessian(negative_log_likelihood)(jnp.asarray(result.x)))
    boundary = [
        name
        for name, value in zip(names, result.x)
        if name != "cease_age" and abs(value) > BOUNDARY_LOGIT
    ]
    covariance = boundary_aware_covariance(
        hessian, [name in boundary for name in names]
    )
    std_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    i1, i2 = names.index("out_tier1_post"), names.index("out_tier2")
    contrast_variance = (
        covariance[i1, i1] + covariance[i2, i2] - 2.0 * covariance[i1, i2]
    )
    estimates = dict(zip(names, result.x.tolist()))

    out = {
        "estimates": estimates,
        "std_errors": dict(zip(names, std_errors.tolist())),
        "probabilities": {
            name: float(jax.nn.sigmoid(value))
            for name, value in estimates.items()
            if name not in INITIAL_PARAMETERS and name != "cease_age"
        },
        "tier_contrast": float(result.x[i2] - result.x[i1]),
        "tier_contrast_se": float(np.sqrt(max(contrast_variance, 0.0))),
        "log_likelihood": -float(result.fun),
        "n_parameters": len(names),
        "n_individuals": int(grouped[-1].sum()),
        "n_groups": int(n_groups),
        "converged": bool(result.success),
        "message": str(result.message),
        "at_boundary": boundary,
    }
    if estimate_initial:
        pi = np.asarray(
            initial_distribution(jnp.stack([estimates[k] for k in INITIAL_PARAMETERS]))
        )
        out["initial"] = {
            "new": float(pi[STATE_NEW]),
            "tier1": float(pi[STATE_TIER1]),
            "out": float(pi[STATE_OUT]),
        }
    if ages is not None:
        out["age_center"] = age_center
        out["age_scale"] = age_scale
    return out
