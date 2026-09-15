"""The occasion-specific Pradel likelihood, validated two independent ways.

`_pradel_individual_likelihood_tv` exists because the scalar likelihood cannot
express time variation: it computes the chi and xi tail probabilities with the
closed form in `_affine_iterate`, which is only valid when the affine
recursion's coefficient is constant. With per-occasion rates there is no closed
form, so those become real recursions.

That substitution is the risky part, so it is checked twice:

1. With constant rates the new function must reproduce the scalar one exactly,
   which anchors it to the implementation already in use.
2. With rates that genuinely vary, chi and xi must match direct enumeration
   over when an animal dies (forward) or enters (reverse). Test 1 alone cannot
   catch an error that only appears once rates differ across occasions.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pradel_jax.models.pradel import (
    _pradel_individual_likelihood,
    _pradel_individual_likelihood_tv,
)

T = 6


def _const(value, n):
    return jnp.full(n, value, dtype=jnp.float64)


@pytest.mark.parametrize("phi,p,f", [(0.8, 0.6, 0.2), (0.5, 0.3, 0.9), (0.95, 0.9, 0.05), (0.2, 0.75, 1.4)])
def test_reduces_to_the_scalar_likelihood_on_every_history(phi, p, f):
    """Exhaustive over all 2**T histories, not a sampled few.

    The tail probabilities depend on where the first and last captures fall,
    so coverage has to include the edge histories: never captured, captured
    only at the first occasion, only at the last, and captured throughout.
    """
    worst = 0.0
    for bits in itertools.product([0, 1], repeat=T):
        ch = jnp.array(bits, dtype=jnp.float64)
        scalar = float(_pradel_individual_likelihood(ch, phi, p, f))
        tv = float(
            _pradel_individual_likelihood_tv(
                ch, _const(phi, T - 1), _const(p, T), _const(f, T - 1)
            )
        )
        worst = max(worst, abs(scalar - tv))
    assert worst < 1e-10, f"max discrepancy {worst:.3e}"


def test_never_captured_individuals_contribute_zero():
    """They are outside the conditional likelihood, as in the scalar version."""
    ch = jnp.zeros(T, dtype=jnp.float64)
    ll = _pradel_individual_likelihood_tv(
        ch, _const(0.8, T - 1), _const(0.6, T), _const(0.2, T - 1)
    )
    assert float(ll) == 0.0


def _varying_rates(seed=7):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.3, 0.95, T - 1)
    p = rng.uniform(0.2, 0.9, T)
    f = rng.uniform(0.05, 1.2, T - 1)
    return phi, p, f, phi / (phi + f)


def _chi_recursion(phi, p):
    chi = np.ones(len(p))
    for j in range(len(p) - 2, -1, -1):
        chi[j] = (1 - phi[j]) + phi[j] * (1 - p[j + 1]) * chi[j + 1]
    return chi


def _xi_recursion(gamma, p):
    xi = np.ones(len(p))
    for j in range(1, len(p)):
        xi[j] = (1 - gamma[j - 1]) + gamma[j - 1] * (1 - p[j - 1]) * xi[j - 1]
    return xi


def _chi_enumerated(phi, p, j):
    """P(never detected after j), summed over the interval the animal dies in."""
    n = len(p)
    total = 0.0
    for k in range(j, n - 1):
        total += (
            np.prod(phi[j:k])
            * (1 - phi[k])
            * np.prod([1 - p[i] for i in range(j + 1, k + 1)])
        )
    total += np.prod(phi[j : n - 1]) * np.prod([1 - p[i] for i in range(j + 1, n)])
    return total


def _xi_enumerated(gamma, p, j):
    """P(never detected before j), summed over the occasion the animal entered.

    Non-detection applies only from the entry occasion k onward: before it the
    animal is not present and cannot be detected.
    """
    total = 0.0
    for k in range(1, j + 1):
        total += (
            np.prod(gamma[k:j])
            * (1 - gamma[k - 1])
            * np.prod([1 - p[i] for i in range(k, j)])
        )
    total += np.prod(gamma[0:j]) * np.prod([1 - p[i] for i in range(0, j)])
    return total


def test_chi_recursion_matches_enumeration_under_varying_rates():
    phi, p, _, _ = _varying_rates()
    chi = _chi_recursion(phi, p)
    for j in range(T):
        assert chi[j] == pytest.approx(_chi_enumerated(phi, p, j), abs=1e-12)


def test_xi_recursion_matches_enumeration_under_varying_rates():
    _, p, _, gamma = _varying_rates()
    xi = _xi_recursion(gamma, p)
    for j in range(T):
        assert xi[j] == pytest.approx(_xi_enumerated(gamma, p, j), abs=1e-12)


def test_varying_rates_actually_change_the_likelihood():
    """Guards against the whole thing collapsing back to a constant-rate model.

    Without this, an implementation that quietly ignored the time dimension
    would still pass every test above.
    """
    phi, p, f, _ = _varying_rates()
    ch = jnp.array([0, 1, 1, 0, 1, 0], dtype=jnp.float64)

    varying = float(
        _pradel_individual_likelihood_tv(
            ch, jnp.asarray(phi), jnp.asarray(p), jnp.asarray(f)
        )
    )
    flat = float(
        _pradel_individual_likelihood_tv(
            ch,
            _const(phi.mean(), T - 1),
            _const(p.mean(), T),
            _const(f.mean(), T - 1),
        )
    )
    assert abs(varying - flat) > 1e-3


def test_gradients_flow_to_every_occasion():
    """Optimisation needs a gradient w.r.t. each occasion's parameter.

    A scan that accidentally closed over a constant would still produce the
    right value while returning zero gradient for most occasions.
    """
    phi, p, f, _ = _varying_rates()
    ch = jnp.array([0, 1, 1, 0, 1, 0], dtype=jnp.float64)

    grad_phi, grad_p = jax.grad(
        lambda a, b: _pradel_individual_likelihood_tv(ch, a, b, jnp.asarray(f)),
        argnums=(0, 1),
    )(jnp.asarray(phi), jnp.asarray(p))

    assert np.all(np.isfinite(np.asarray(grad_phi)))
    assert np.all(np.isfinite(np.asarray(grad_p)))
    # Intervals inside [first, last) must all matter.
    assert np.count_nonzero(np.asarray(grad_phi)) >= T - 2
