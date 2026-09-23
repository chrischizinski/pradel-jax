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
    <p>_age_<band>  with ``age_bands`` given: an effect of age band on process
                    <p> (entry, initial_new, initial_out, cease, out, ret),
                    relative to a reference band; see :func:`fit`
    cease_female    with ``female`` given: shift of ``cease`` for women
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
# Processes an age-band effect can act on. "initial" expands to both initial
# logits; "out" shifts sitting out in both tiers alike, so the tier contrast
# is still a single number.
AGE_PROCESSES = ("entry", "initial", "cease", "out", "ret")


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


# An eigenvalue of the Hessian below this fraction of the largest is treated
# as zero: a direction the data do not determine.
NULL_EIGENVALUE_RATIO = 1e-8
# A parameter loading more than this on such a direction is reported as not
# estimable.
NULL_LOADING = 0.3


def estimable_covariance(hessian: np.ndarray, at_boundary: Sequence[bool]):
    """Covariance over the parameters the data determine; NaN for the rest.

    Boundary estimates are excluded first (see
    :func:`boundary_aware_covariance`). That is not always enough: a
    parameter at a boundary can make another one irrelevant without that one
    being at a boundary itself. On the SD data every hunter of unknown age
    went to NEW at the first occasion, so how the rest of that band split
    between Tier I and OUT stopped mattering -- a flat, interior direction
    that made the Hessian singular and cost every standard error. So any
    direction of near-zero curvature left after the boundary exclusions is
    found from the eigen-decomposition, the parameters that load on it are
    excluded too, and the process repeats until the rest invert. Rows that
    are not finite are excluded the same way, one at a time. This is how MARK reports
    parameters it cannot estimate.

    Returns (covariance, names-free list of excluded indices, smallest
    eigenvalue of what was finally inverted).
    """
    n = hessian.shape[0]
    keep = ~np.asarray(at_boundary, dtype=bool)
    not_estimable = []
    covariance = np.full((n, n), np.nan)
    smallest = float("nan")
    while keep.any():
        index = np.where(keep)[0]
        block = hessian[np.ix_(index, index)]
        bad = (~np.isfinite(block)).sum(axis=1)
        if bad.any():
            # A non-finite row also puts non-finite entries in every other
            # row's column, so drop the worst one and look again.
            worst = index[int(np.argmax(bad))]
            not_estimable.append(int(worst))
            keep[worst] = False
            continue
        eigenvalues, vectors = np.linalg.eigh(block)
        tolerance = NULL_EIGENVALUE_RATIO * np.abs(eigenvalues).max()
        null = eigenvalues < tolerance
        if not null.any():
            covariance[np.ix_(index, index)] = np.linalg.inv(block)
            smallest = float(eigenvalues.min())
            break
        loading = np.abs(vectors[:, null]).max(axis=1)
        drop = index[loading > NULL_LOADING]
        if len(drop) == 0:  # spread thinly; drop the largest loading
            drop = index[[int(np.argmax(loading))]]
        not_estimable += drop.tolist()
        keep[drop] = False
    return covariance, not_estimable, smallest


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
    params: Dict,
    mortality,
    tier2_available,
    comparison_era,
    age_z=None,
    band=None,
    female=None,
):
    """Named logit parameters -> (n, n_intervals, N_STATES, N_STATES) matrices.

    ``params`` holds ``entry`` or ``entry_0 .. entry_{T-3}`` (see
    :func:`parameter_names`) and every name in :data:`BEHAVIOUR`. Optionally:
    ``cease_age`` with ``age_z`` (standardised age, shaped like
    ``mortality``); ``<process>_age`` vectors with ``band``, a one-hot
    (n, n_intervals, n_bands - 1) matrix with the reference band dropped;
    ``cease_female`` with ``female``, shaped like ``mortality``.
    """
    shape = mortality.shape
    n_intervals = shape[-1]

    def banded(process):
        if band is None or f"{process}_age" not in params:
            return 0.0
        return band @ params[f"{process}_age"]

    if "entry" in params:
        entry_logit = jnp.broadcast_to(params["entry"], shape) + banded("entry")
        entry = jax.nn.sigmoid(entry_logit)
    else:
        free = jnp.concatenate(
            [
                jnp.stack([params[f"entry_{t}"] for t in range(n_intervals - 1)]),
                jnp.zeros(1),
            ]
        )
        entry_logit = jnp.broadcast_to(free, shape) + banded("entry")
        last = jnp.arange(n_intervals) == n_intervals - 1
        entry = jnp.where(last, 1.0, jax.nn.sigmoid(entry_logit))

    era = jnp.broadcast_to(jnp.asarray(comparison_era) > 0, shape)
    out_shift = banded("out")
    logits = {
        "out_tier1": jnp.where(era, params["out_tier1_post"], params["out_tier1"])
        + out_shift,
        "out_tier2": params["out_tier2"] + out_shift,
        "ret": params["ret"] + banded("ret"),
        "cease": params["cease"]
        + (0.0 if age_z is None else params["cease_age"] * age_z)
        + banded("cease")
        + (0.0 if female is None else params["cease_female"] * female),
    }
    for name in ("to_tier2", "switch_up", "switch_down"):
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


# Optimiser settings. The default L-BFGS-B stopping rule (relative change in
# the objective) stops a likelihood of ~5e5 with gradients up to ~10; these
# drive the gradient to ~1e-3. Found on the SD data, where a loose stop also
# left one fit at a local maximum 1,580 log-likelihood units short.
OPTIMIZER_OPTIONS = {"maxiter": 20000, "gtol": 1e-5, "ftol": 1e-15, "maxls": 50}


def _minimize(objective, x0):
    def scipy_objective(x):
        value, grad = objective(jnp.asarray(x))
        return float(value), np.asarray(grad, dtype=np.float64)

    return minimize(
        scipy_objective, x0, jac=True, method="L-BFGS-B", options=OPTIMIZER_OPTIONS
    )


def fit(
    observations: np.ndarray,
    mortality: np.ndarray,
    tier2_available: np.ndarray,
    comparison_era: np.ndarray,
    initial: Optional[np.ndarray] = None,
    entry: str = "annual",
    ages: Optional[np.ndarray] = None,
    start: Optional[np.ndarray] = None,
    age_bands: Optional[np.ndarray] = None,
    band_labels: Optional[Sequence[str]] = None,
    reference_band: Optional[int] = None,
    age_effects: Sequence[str] = (),
    female: Optional[np.ndarray] = None,
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
        age_bands: optional (n, T - 1) integer age band over each interval,
            indexing ``band_labels``. A hunter of unknown age should get a band
            of its own rather than a guessed age: a guessed age puts them in a
            real band, where their pattern is taken for that band's.
        band_labels, reference_band: names of the bands, and which one the
            effects are relative to.
        age_effects: which of :data:`AGE_PROCESSES` get band effects.
            ``"initial"`` uses the band over the first interval: age decides
            how likely a hunter first seen later was already hunting in the
            first year (OUT) rather than yet to start (NEW). On the NE and SD
            data, simpler specifications fit better with the life table
            inflated (up to 3x); adding age on entry and on these weights
            removed that. Simulation did not reproduce the inflation from their
            absence, so why it happened is not established.
        female: optional (n,) indicator; given, ``cease`` gets a female shift.

    Returns:
        dict with ``estimates`` and ``std_errors`` (logit scale, by name),
        ``probabilities`` (for the intercept-type parameters),
        ``tier_contrast`` and its ``tier_contrast_se``, ``log_likelihood``,
        ``n_parameters``, ``n_individuals``, ``n_groups``, ``converged``,
        ``message``, ``at_boundary`` and ``start_log_likelihoods`` (every
        start tried; they should agree); ``initial`` when estimated and not
        age-dependent; ``age_center`` and ``age_scale`` when ``ages`` is given.
    """
    observations = np.asarray(observations)
    mortality = np.asarray(mortality, dtype=np.float64)
    if (observations.max(axis=1) == 0).any():
        raise ValueError("every history must contain at least one record")
    n_intervals = observations.shape[1] - 1
    age_effects = tuple(age_effects)
    unknown = set(age_effects) - set(AGE_PROCESSES)
    if unknown:
        raise ValueError(f"unknown age_effects {sorted(unknown)}")
    if age_effects and age_bands is None:
        raise ValueError("age_effects need age_bands")
    if "initial" in age_effects and initial is not None:
        raise ValueError("age effects on the initial weights need them estimated")

    extras = [mortality]
    if ages is not None:
        ages = np.asarray(ages, dtype=np.float64)
        age_center, age_scale = float(ages.mean()), float(ages.std())
        extras.append((ages - age_center) / age_scale)
    if age_effects:
        extras.append(np.asarray(age_bands, dtype=np.float64))
    if female is not None:
        extras.append(
            np.broadcast_to(
                np.asarray(female, dtype=np.float64)[:, None], mortality.shape
            )
        )
    grouped = group_records(observations, *extras)
    pieces = iter(grouped[1:-1])
    obs_j = jnp.asarray(grouped[0])
    mortality_j = jnp.asarray(next(pieces))
    age_z_j = jnp.asarray(next(pieces)) if ages is not None else None
    band_j = None
    if age_effects:
        n_bands = len(band_labels)
        one_hot = np.eye(n_bands)[next(pieces).astype(int)]
        band_j = jnp.asarray(np.delete(one_hot, reference_band, axis=-1))
    female_j = jnp.asarray(next(pieces)) if female is not None else None
    counts_j = jnp.asarray(grouped[-1])
    n_groups = len(grouped[-1])

    estimate_initial = initial is None
    scalar_names = parameter_names(
        n_intervals,
        entry=entry,
        cease_age=ages is not None,
        estimate_initial=estimate_initial,
    )
    if female is not None:
        scalar_names.append("cease_female")
    vector_names = []
    for process in age_effects:
        vector_names += (
            ["initial_new_age", "initial_out_age"]
            if process == "initial"
            else [f"{process}_age"]
        )
    band_names = (
        [label for i, label in enumerate(band_labels) if i != reference_band]
        if age_effects
        else []
    )
    # A band with no hunters in it has no information: its coefficient is
    # flat, the Hessian singular, and every standard error -- the tier
    # contrast's included -- would be lost. Keep, for each effect, only the
    # bands that occur where that effect acts: the first interval for the
    # initial weights, any interval otherwise. (NE's primary sample, for one,
    # has no hunter of unknown age.)
    kept_bands, dropped_bands = {}, []
    if age_effects:
        band_np = np.asarray(band_j)
        counts_np = np.asarray(counts_j)
        anywhere = np.einsum("g,gtb->b", counts_np, band_np) > 0
        at_first = np.einsum("g,gb->b", counts_np, band_np[:, 0, :]) > 0
        for vector in vector_names:
            present = at_first if vector.startswith("initial_") else anywhere
            kept_bands[vector] = np.where(present)[0]
            dropped_bands += [
                f"{vector[:-4]}_age_{band_names[b]}" for b in np.where(~present)[0]
            ]
    names = scalar_names + [
        f"{vector[:-4]}_age_{band_names[b]}"
        for vector in vector_names
        for b in kept_bands[vector]
    ]
    n_scalar, n_band = len(scalar_names), len(band_names)
    fixed_initial = None if estimate_initial else jnp.asarray(initial)

    def unpack(x):
        params = dict(zip(scalar_names, x[:n_scalar]))
        offset = n_scalar
        for vector in vector_names:
            kept = kept_bands[vector]
            params[vector] = (
                jnp.zeros(n_band).at[kept].set(x[offset : offset + len(kept)])
            )
            offset += len(kept)
        return params

    def initial_weights(params):
        if not estimate_initial:
            return jnp.broadcast_to(fixed_initial, (n_groups, N_STATES))
        base = jnp.stack([params[k] for k in INITIAL_PARAMETERS])
        if "initial" not in age_effects:
            return jnp.broadcast_to(initial_distribution(base), (n_groups, N_STATES))
        first = band_j[:, 0, :]
        shifts = jnp.stack(
            [first @ params["initial_new_age"], first @ params["initial_out_age"]],
            axis=-1,
        )
        return jax.vmap(lambda shift: initial_distribution(base + shift))(shifts)

    def negative_log_likelihood(x):
        params = unpack(x)
        matrices = transition_matrices(
            params,
            mortality_j,
            tier2_available,
            comparison_era,
            age_z=age_z_j,
            band=band_j,
            female=female_j,
        )
        per_group = jax.vmap(conditional_log_likelihood)(
            obs_j, matrices, initial_weights(params)
        )
        return -jnp.sum(counts_j * per_group)

    objective = jax.jit(jax.value_and_grad(negative_log_likelihood))

    starts = []
    if start is not None:
        starts.append(np.asarray(start, float))
    else:
        cold = np.full(len(names), -1.5)
        cold[n_scalar:] = 0.0
        for name in ("cease_age", "cease_female"):
            if name in scalar_names:
                cold[scalar_names.index(name)] = 0.0
        starts.append(cold)
        if age_effects or female is not None:
            # Staged start: the same model without the age and sex effects,
            # extended with zeros. Richer models here have local maxima, and
            # one start is not enough to trust.
            simpler = fit(
                observations,
                mortality,
                tier2_available,
                comparison_era,
                initial=initial,
                entry=entry,
                ages=ages,
            )
            staged = np.zeros(len(names))
            for i, name in enumerate(scalar_names):
                staged[i] = simpler["estimates"].get(name, 0.0)
            starts.append(staged)

    results = [_minimize(objective, x0) for x0 in starts]
    result = max(results, key=lambda r: -r.fun)

    hessian = np.asarray(jax.hessian(negative_log_likelihood)(jnp.asarray(result.x)))
    boundary = [
        name
        for name, value in zip(names, result.x)
        if name not in ("cease_age", "cease_female") and abs(value) > BOUNDARY_LOGIT
    ]
    at_boundary = [name in boundary for name in names]
    covariance, excluded, smallest = estimable_covariance(hessian, at_boundary)
    not_estimable = [names[i] for i in excluded]
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
            name: float(jax.nn.sigmoid(estimates[name]))
            for name in scalar_names
            if name not in INITIAL_PARAMETERS
            and name not in ("cease_age", "cease_female")
        },
        "tier_contrast": float(result.x[i2] - result.x[i1]),
        "tier_contrast_se": float(np.sqrt(max(contrast_variance, 0.0))),
        "log_likelihood": -float(result.fun),
        "start_log_likelihoods": [-float(r.fun) for r in results],
        "n_parameters": len(names),
        "n_individuals": int(grouped[-1].sum()),
        "n_groups": int(n_groups),
        "converged": bool(result.success),
        "message": str(result.message),
        "at_boundary": boundary,
        "dropped_bands": dropped_bands,
        "not_estimable": not_estimable,
        "hessian_min_eigenvalue": smallest,
    }
    if estimate_initial and "initial" not in age_effects:
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
