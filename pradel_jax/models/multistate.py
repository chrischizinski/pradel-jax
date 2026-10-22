"""Forward algorithm for the five-state hunter participation model.

The Pradel model treats tier as a covariate. It is a state, and the difference
is not cosmetic: every year a hunter does not register, their tier is not
missing data to be filled in, it is information about which state they occupied.
This module provides the likelihood for the state formulation.

States
------
Two are observed directly. Three emit nothing and are told apart only by where
they can sit in a history and what can follow them.

    0  NEW      not yet in this state's HIP frame; only before first record
    1  TIER1    active, Tier I     -- observed
    2  TIER2    active, Tier II    -- observed
    3  OUT      alive, not participating this year; may return
    4  GONE     permanently out: dead, or ceased for good. Absorbing.

The observation process is deterministic, which is what makes this tractable.
HIP registration is effectively complete, so a hunter who hunts is in the frame
and their tier is recorded exactly: detection is 1 for TIER1 and TIER2 and 0 for
the rest. There is no detection parameter, and no "was active but missed"
ambiguity to spend identification on -- all of it goes to the transitions.

That also means the emission step is a *mask*, not a probability. Seeing a Tier
II record rules out every state but TIER2. Seeing nothing rules out TIER1 and
TIER2 and leaves NEW, OUT and GONE alive as possibilities, which is exactly the
ambiguity the model exists to resolve.

Scope
-----
This module computes the likelihood of an observed history given transition
matrices. It does not build those matrices from covariates -- that is the design
matrix work, and keeping it out means this can be validated against brute-force
enumeration without any of that machinery in the way.
"""

from typing import Optional

import jax
import jax.numpy as jnp

# State indices. Kept as module constants rather than an enum so they can be
# used inside jitted code without tracing problems.
STATE_NEW = 0
STATE_TIER1 = 1
STATE_TIER2 = 2
STATE_OUT = 3
STATE_GONE = 4
N_STATES = 5

# Observation codes.
OBS_NONE = 0
OBS_TIER1 = 1
OBS_TIER2 = 2

# Which states are consistent with each observation. Rows are observations,
# columns are states. This is the deterministic emission described above: every
# entry is 0 or 1, never a probability.
EMISSION = jnp.array(
    [
        # no record: not yet arrived, sitting out, or gone for good
        [1.0, 0.0, 0.0, 1.0, 1.0],
        # seen at Tier I
        [0.0, 1.0, 0.0, 0.0, 0.0],
        # seen at Tier II
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ],
    dtype=jnp.float64,
)


def _rescale(vector):
    """Normalise a probability vector, returning (normalised, log of the mass).

    A history that the model says is impossible has zero mass, and the honest
    answer for it is a log-likelihood of -inf -- the parameters are falsified by
    that record. Flooring the mass at a tiny epsilon instead would return a
    large finite number, and an optimiser would happily walk around in a region
    where the model cannot produce the data at all.

    The guard that is needed is against nan, not against -inf: dividing a zero
    vector by zero mass gives nan, which then poisons every later occasion and
    every gradient. Returning a zero vector keeps the arithmetic clean, and the
    -inf carried in the accumulator is preserved all the way out.

    The log needs the same guard for its derivative. ``log(0)`` is the right
    value, but its derivative is inf, and inf times the zero sensitivity of the
    mass is nan. Zero mass is not only impossible histories: a hunter certain
    to be recorded (no mortality, last entry rate fixed at 1) has a zero
    probability of never being seen, and the conditioning term took the log of
    exactly that -- turning the gradient and Hessian nan in a no-mortality fit
    while its value stayed finite. The double ``where`` keeps the value and
    gives a zero derivative instead.
    """
    mass = jnp.sum(vector)
    positive = mass > 0
    safe = jnp.where(positive, vector / jnp.where(positive, mass, 1.0), 0.0)
    log_mass = jnp.where(positive, jnp.log(jnp.where(positive, mass, 1.0)), -jnp.inf)
    return safe, log_mass


@jax.jit
def forward_log_likelihood(
    observations: jnp.ndarray,
    transitions: jnp.ndarray,
    initial: jnp.ndarray,
) -> float:
    """Log-probability of one observed history under the state model.

    Runs the standard hidden-Markov forward recursion, rescaling at every
    occasion and accumulating the logs of the scale factors. Rescaling rather
    than working in log-space keeps the recursion a plain matrix-vector product,
    which matters because ten occasions of five states is small enough that the
    scan cost is dominated by overhead.

    Args:
        observations: (T,) integer codes, one per occasion -- OBS_NONE,
            OBS_TIER1 or OBS_TIER2.
        transitions: (T-1, N_STATES, N_STATES) row-stochastic matrices.
            ``transitions[t, i, j]`` is P(state j at occasion t+1 | state i at
            occasion t). Structural zeros -- GONE being absorbing, NEW being
            unreachable once left -- are the caller's to impose; nothing here
            assumes them, so the same recursion validates against an unrestricted
            enumeration.
        initial: (N_STATES,) distribution over states at the first occasion.

    Returns:
        log P(observations). Not conditioned on the hunter ever being seen; see
        :func:`log_prob_ever_observed` for the conditioning term.
    """
    alpha, log_likelihood = _rescale(initial * EMISSION[observations[0]])

    def step(carry, t):
        alpha_prev, running = carry
        # alpha_prev is a row vector of state probabilities; propagating it is a
        # vector-matrix product, then the mask knocks out states the next
        # observation rules out.
        propagated = alpha_prev @ transitions[t - 1]
        masked, log_mass = _rescale(propagated * EMISSION[observations[t]])
        return (masked, running + log_mass), None

    n_occasions = observations.shape[0]
    (_, log_likelihood), _ = jax.lax.scan(
        step, (alpha, log_likelihood), jnp.arange(1, n_occasions)
    )
    return log_likelihood


@jax.jit
def log_prob_ever_observed(transitions: jnp.ndarray, initial: jnp.ndarray) -> float:
    """Log-probability that a hunter is recorded at least once.

    Individuals who never appear are not in the data, so the likelihood has to
    be conditioned on having been seen -- the same conditioning the Pradel
    likelihood applies. This computes the complement: propagate the chain while
    forbidding TIER1 and TIER2 at every occasion, and what survives is the
    probability of never being recorded.
    """
    never_seen = EMISSION[OBS_NONE]

    alpha, log_never = _rescale(initial * never_seen)

    def step(carry, t):
        alpha_prev, running = carry
        masked, log_mass = _rescale((alpha_prev @ transitions[t - 1]) * never_seen)
        return (masked, running + log_mass), None

    n_occasions = transitions.shape[0] + 1
    (_, log_never), _ = jax.lax.scan(
        step, (alpha, log_never), jnp.arange(1, n_occasions)
    )
    # log(1 - P(never seen)), guarded so that a chain which is certain never to
    # be seen returns -inf rather than a nan.
    return jnp.log(-jnp.expm1(jnp.minimum(log_never, -1e-12)))


@jax.jit
def conditional_log_likelihood(
    observations: jnp.ndarray,
    transitions: jnp.ndarray,
    initial: jnp.ndarray,
) -> float:
    """Forward likelihood conditioned on the hunter having been seen at least once.

    This is the quantity to sum over a dataset: histories that are all zeros
    cannot appear in the data, so leaving them in the sample space would spread
    probability mass over records that could never have been collected.
    """
    return forward_log_likelihood(observations, transitions, initial) - (
        log_prob_ever_observed(transitions, initial)
    )


def _transition_rows(
    mortality: jnp.ndarray,
    entry: jnp.ndarray,
    out_tier1: jnp.ndarray,
    out_tier2: jnp.ndarray,
    ret: jnp.ndarray,
    cease: jnp.ndarray,
    to_tier2: jnp.ndarray,
    switch_up: jnp.ndarray,
    switch_down: jnp.ndarray,
) -> jnp.ndarray:
    """Assemble one transition matrix from the probabilities that define it.

    Every argument broadcasts to the same shape, so this builds a single matrix
    or a whole ``(n_individuals, n_intervals)`` grid of them with the same code.

    The rows are written as a sequence of conditional steps rather than a table
    of free probabilities, which is what keeps them summing to one by
    construction instead of by a normalisation that would quietly hide an error:

        die first (mortality, fixed externally from a life table)
        -> if alive, cease for good
        -> if still participating, sit this season out
        -> if still active, possibly switch tier

    ``mortality`` is the only one that is not a parameter. It is data, taken
    from published life tables for the hunter's age and sex, and it is what lets
    "permanently gone" be split into death and giving up rather than assumed.
    """
    zero = jnp.zeros_like(mortality)
    alive = 1.0 - mortality
    # Leaving for good, whatever the reason: death, or ceasing while alive.
    gone = mortality + alive * cease
    staying = alive * (1.0 - cease)

    # NEW: has not been recorded yet. Cannot sit a season out -- there is no
    # participation to interrupt -- so the only moves are entering or dying.
    row_new = jnp.stack(
        [
            alive * (1.0 - entry),
            alive * entry * (1.0 - to_tier2),
            alive * entry * to_tier2,
            zero,
            mortality,
        ],
        axis=-1,
    )

    active_tier1 = staying * (1.0 - out_tier1)
    row_tier1 = jnp.stack(
        [
            zero,
            active_tier1 * (1.0 - switch_up),
            active_tier1 * switch_up,
            staying * out_tier1,
            gone,
        ],
        axis=-1,
    )

    active_tier2 = staying * (1.0 - out_tier2)
    row_tier2 = jnp.stack(
        [
            zero,
            active_tier2 * switch_down,
            active_tier2 * (1.0 - switch_down),
            staying * out_tier2,
            gone,
        ],
        axis=-1,
    )

    # OUT: returning hunters re-enter at whichever tier is on offer.
    row_out = jnp.stack(
        [
            zero,
            staying * ret * (1.0 - to_tier2),
            staying * ret * to_tier2,
            staying * (1.0 - ret),
            gone,
        ],
        axis=-1,
    )

    row_gone = jnp.stack([zero, zero, zero, zero, jnp.ones_like(mortality)], axis=-1)

    return jnp.stack([row_new, row_tier1, row_tier2, row_out, row_gone], axis=-2)


def build_transition_matrices(
    mortality: jnp.ndarray,
    tier2_available: jnp.ndarray,
    entry: jnp.ndarray,
    out_tier1: jnp.ndarray,
    out_tier2: jnp.ndarray,
    ret: jnp.ndarray,
    cease: jnp.ndarray,
    to_tier2: jnp.ndarray,
    switch_up: jnp.ndarray,
    switch_down: jnp.ndarray,
) -> jnp.ndarray:
    """Transition matrices for every individual and interval.

    ``tier2_available`` is the before/after indicator for the regulation change:
    0 on intervals that end before Tier II existed, 1 afterwards. Everything
    that can put a hunter into Tier II is multiplied by it, so the model cannot
    place anyone in a state the regulation had not created yet -- the same
    anachronism that had to be fixed in the covariate version, here made
    structurally impossible rather than left to a data-preparation step.
    """
    gate = tier2_available
    return _transition_rows(
        mortality=mortality,
        entry=entry,
        out_tier1=out_tier1,
        out_tier2=out_tier2,
        ret=ret,
        cease=cease,
        to_tier2=to_tier2 * gate,
        switch_up=switch_up * gate,
        switch_down=switch_down * gate,
    )
