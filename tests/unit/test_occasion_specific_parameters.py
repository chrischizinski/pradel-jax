"""Occasion-specific Pradel parameters: does an annual covariate reach the right interval?

The prior architecture had one parameter value per individual, so a covariate
recorded once per year could only enter the model by being collapsed into a
single time-constant effect.  That collapse is now replaced by a design matrix
carrying a period axis, and these tests exist to check the two things that can
go wrong with it:

1. the covariate reaches the likelihood at all (a time effect that is silently
   averaged away still fits and still reports an AIC);
2. it reaches the *right* period -- a value recorded in 2019 must move survival
   over 2019->2020 and nothing else.

Plan section 2.2 asks specifically for the second.  Locality is the property
that distinguishes a genuinely time-varying model from one that merely has more
coefficients, and it cannot be seen from a likelihood value or a fit summary,
only by perturbing one occasion and watching where the change lands.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import pradel_jax as pj
from pradel_jax.data.adapters import CovariateInfo, DataContext
from pradel_jax.models.pradel import PradelModel

N_OCCASIONS = 5
N_INDIVIDUALS = 8


def _capture_matrix() -> jnp.ndarray:
    """Eight histories, every one with at least one capture.

    Individuals never captured contribute zero to the conditional Pradel
    likelihood, so a matrix containing them would let a broken perturbation test
    pass by moving nothing.
    """
    histories = [
        "10110",
        "11010",
        "01110",
        "10011",
        "11101",
        "01011",
        "00111",
        "11111",
    ]
    return jnp.array([[float(c) for c in h] for h in histories], dtype=jnp.float64)


def _context(**covariates) -> DataContext:
    covariate_info = {
        name: CovariateInfo(name=name, dtype="float64")
        for name in covariates
        if not name.endswith(("_is_categorical", "_categories", "_is_time_varying"))
    }
    return DataContext(
        capture_matrix=_capture_matrix(),
        covariates=covariates,
        covariate_info=covariate_info,
        n_individuals=N_INDIVIDUALS,
        n_occasions=N_OCCASIONS,
    )


def _annual_tier(rng: np.random.Generator) -> np.ndarray:
    """Annual tier status, one column per occasion, values in {0, 1, 2}."""
    return rng.integers(0, 3, size=(N_INDIVIDUALS, N_OCCASIONS)).astype(np.float64)


def _tier_context(tier: np.ndarray) -> DataContext:
    return _context(
        tier=jnp.array(tier, dtype=jnp.float64),
        tier_is_categorical=True,
        tier_categories=["0", "1", "2"],
        tier_is_time_varying=True,
    )


def _year_context() -> DataContext:
    """`year` as a time factor: the occasion index, identical for everyone.

    Declared categorical so that it expands to one dummy per non-reference
    occasion, which is what "a separate value each year" means.  It carries no
    individual variation at all, so any effect it has must come from the period
    axis.
    """
    year = np.tile(np.arange(N_OCCASIONS, dtype=np.float64), (N_INDIVIDUALS, 1))
    return _context(
        year=jnp.array(year),
        year_is_categorical=True,
        year_categories=[str(t) for t in range(N_OCCASIONS)],
        year_is_time_varying=True,
    )


def _fitted_shapes(spec, context):
    model = PradelModel()
    matrices = model.build_design_matrices(spec, context)
    return {name: info.matrix.shape for name, info in matrices.items()}


# --------------------------------------------------------------------------
# Design matrices carry a period axis, and the coefficient count does not grow
# --------------------------------------------------------------------------


def test_time_constant_model_is_untouched():
    """A model with no annual covariate must produce exactly what it did before.

    This is the regression guard for every analysis already run: if the
    time-constant path had gained a period axis, every existing fit would have
    silently changed.
    """
    context = _context(
        gender=jnp.array([0.0, 1.0] * (N_INDIVIDUALS // 2)),
    )
    spec = pj.create_formula_spec(phi="~1 + gender", p="~1", f="~1")
    shapes = _fitted_shapes(spec, context)

    assert shapes["phi"] == (N_INDIVIDUALS, 2)
    assert shapes["p"] == (N_INDIVIDUALS, 1)
    assert shapes["f"] == (N_INDIVIDUALS, 1)


def test_phi_gets_an_interval_axis_and_p_an_occasion_axis():
    """phi acts between occasions, p at them, so their time axes differ by one.

    Getting this off by one would misalign every annual covariate by a year
    while still producing a matrix of plausible shape.
    """
    rng = np.random.default_rng(0)
    context = _tier_context(_annual_tier(rng))
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1 + tier", f="~1")
    shapes = _fitted_shapes(spec, context)

    # 3 tier levels -> intercept + 2 dummies = 3 coefficients, for both.
    assert shapes["phi"] == (N_INDIVIDUALS, N_OCCASIONS - 1, 3)
    assert shapes["p"] == (N_INDIVIDUALS, N_OCCASIONS, 3)
    assert shapes["f"] == (N_INDIVIDUALS, 1)


def test_annual_covariate_costs_one_coefficient_not_one_per_year():
    """`phi ~ tier` is one tier effect applied annually, not a tier-by-year model.

    If the period structure leaked into the parameter vector instead of the
    design matrix, this model would quietly become `tier * year` -- a different
    hypothesis with many more parameters and a different AIC.
    """
    rng = np.random.default_rng(1)
    context = _tier_context(_annual_tier(rng))
    model = PradelModel()
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")
    matrices = model.build_design_matrices(spec, context)

    assert matrices["phi"].parameter_count == 3
    assert matrices["phi"].column_names == ["(Intercept)", "tier_1", "tier_2"]

    initial = model.get_initial_parameters(context, matrices)
    bounds = model.get_parameter_bounds(context, matrices)
    # phi (3) + p (1) + f (1)
    assert initial.shape == (5,)
    assert len(bounds) == 5


# --------------------------------------------------------------------------
# Plan 2.2: a covariate change at one occasion moves only that interval
# --------------------------------------------------------------------------


def _phi_by_interval(context, spec, parameters):
    """Per-interval survival implied by a parameter vector."""
    from pradel_jax.models.pradel import _linear_predictor, inv_logit

    model = PradelModel()
    matrices = model.build_design_matrices(spec, context)
    split = model._split_parameters(parameters, matrices)
    return inv_logit(_linear_predictor(matrices["phi"].matrix, split["phi"]))


def test_changing_tier_at_one_occasion_moves_only_that_interval():
    """Plan 2.2, `phi ~ tier`: the perturbation must stay in its own year."""
    rng = np.random.default_rng(2)
    tier = _annual_tier(rng)
    # Force a change that is guaranteed to be a change.
    tier[3, 1] = 0.0
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")
    parameters = jnp.array([0.7, 0.3, -0.4, 0.1, -1.2])

    baseline = _phi_by_interval(_tier_context(tier), spec, parameters)

    perturbed_tier = tier.copy()
    perturbed_tier[3, 1] = 2.0
    perturbed = _phi_by_interval(_tier_context(perturbed_tier), spec, parameters)

    moved = ~np.isclose(np.asarray(baseline), np.asarray(perturbed), rtol=0, atol=0)
    assert moved[3, 1], "the perturbed individual-interval did not move at all"
    assert moved.sum() == 1, (
        f"the change leaked to {moved.sum() - 1} other individual-intervals; "
        "it must stay in the interval whose covariate changed"
    )


def test_tier_in_the_final_year_cannot_reach_survival():
    """Survival has one fewer period than detection, so the last year is p-only.

    There is no interval starting at the final occasion, so a tier value
    recorded there has nowhere to go in phi.  If it moved phi, the covariate
    would be shifted a year relative to the interval it is supposed to describe.
    """
    rng = np.random.default_rng(3)
    tier = _annual_tier(rng)
    tier[2, N_OCCASIONS - 1] = 0.0
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")
    parameters = jnp.array([0.7, 0.3, -0.4, 0.1, -1.2])

    baseline = _phi_by_interval(_tier_context(tier), spec, parameters)
    tier[2, N_OCCASIONS - 1] = 2.0
    perturbed = _phi_by_interval(_tier_context(tier), spec, parameters)

    assert np.allclose(np.asarray(baseline), np.asarray(perturbed))


def test_changing_tier_at_one_occasion_changes_the_likelihood():
    """The interval that moved has to move the objective too.

    A design matrix can be perfectly occasion-specific and still be ignored by
    the likelihood, which is the failure the old collapse produced.
    """
    rng = np.random.default_rng(4)
    tier = _annual_tier(rng)
    tier[3, 1] = 0.0
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")
    parameters = jnp.array([0.7, 0.3, -0.4, 0.1, -1.2])
    model = PradelModel()

    def ll(t):
        context = _tier_context(t)
        return float(
            model.log_likelihood(
                parameters, context, model.build_design_matrices(spec, context)
            )
        )

    before = ll(tier)
    tier[3, 1] = 2.0
    after = ll(tier)

    assert not np.isclose(before, after)


def test_plan_2_2_year_on_phi_and_on_p():
    """Plan 2.2, `phi ~ year, p ~ 1` and `phi ~ tier, p ~ year`.

    `year` varies only across occasions, never across individuals, so a fit that
    still collapses the period axis would produce an intercept-only phi and an
    identical likelihood for every value of the year coefficients.
    """
    model = PradelModel()
    year_context = _year_context()

    spec = pj.create_formula_spec(phi="~1 + year", p="~1", f="~1")
    matrices = model.build_design_matrices(spec, year_context)
    # 5 occasions -> phi spans 4 intervals; dummies for years 1..4 exist but
    # only the first 4 occasions are in range, so the year-4 dummy is all-zero.
    assert matrices["phi"].matrix.shape == (N_INDIVIDUALS, N_OCCASIONS - 1, N_OCCASIONS)

    base = jnp.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.2, -1.1])
    bumped = base.at[2].set(0.9)
    ll_base = float(model.log_likelihood(base, year_context, matrices))
    ll_bumped = float(model.log_likelihood(bumped, year_context, matrices))
    assert not np.isclose(ll_base, ll_bumped), (
        "a year-specific phi coefficient changed nothing, so the period axis "
        "is not reaching the likelihood"
    )

    rng = np.random.default_rng(5)
    tier = _annual_tier(rng)
    mixed_context = _context(
        tier=jnp.array(tier),
        tier_is_categorical=True,
        tier_categories=["0", "1", "2"],
        tier_is_time_varying=True,
        year=jnp.array(
            np.tile(np.arange(N_OCCASIONS, dtype=np.float64), (N_INDIVIDUALS, 1))
        ),
        year_is_categorical=True,
        year_categories=[str(t) for t in range(N_OCCASIONS)],
        year_is_time_varying=True,
    )
    mixed_spec = pj.create_formula_spec(phi="~1 + tier", p="~1 + year", f="~1")
    mixed = model.build_design_matrices(mixed_spec, mixed_context)
    assert mixed["phi"].matrix.shape == (N_INDIVIDUALS, N_OCCASIONS - 1, 3)
    assert mixed["p"].matrix.shape == (N_INDIVIDUALS, N_OCCASIONS, N_OCCASIONS)


# --------------------------------------------------------------------------
# The occasion-specific path must agree with the reference path
# --------------------------------------------------------------------------


def test_constant_covariate_reproduces_the_time_constant_likelihood():
    """A covariate that does not actually vary must give the old answer exactly.

    This is the bridge between the two backends: it pins the new code to the
    retained scalar likelihood rather than only to itself.
    """
    model = PradelModel()
    constant_tier = np.ones((N_INDIVIDUALS, N_OCCASIONS), dtype=np.float64)
    tv_context = _context(
        tier=jnp.array(constant_tier),
        tier_is_categorical=True,
        tier_categories=["0", "1"],
        tier_is_time_varying=True,
    )
    tv_spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")
    tv_matrices = model.build_design_matrices(tv_spec, tv_context)
    assert tv_matrices["phi"].matrix.ndim == 3

    flat_context = _context(
        tier_flat=jnp.ones(N_INDIVIDUALS, dtype=jnp.float64),
    )
    flat_spec = pj.create_formula_spec(phi="~1 + tier_flat", p="~1", f="~1")
    flat_matrices = model.build_design_matrices(flat_spec, flat_context)
    assert flat_matrices["phi"].matrix.ndim == 2

    parameters = jnp.array([0.6, -0.35, 0.15, -1.05])
    tv_ll = float(model.log_likelihood(parameters, tv_context, tv_matrices))
    flat_ll = float(model.log_likelihood(parameters, flat_context, flat_matrices))

    assert tv_ll == pytest.approx(flat_ll, rel=0, abs=1e-10)


def test_gradients_reach_every_coefficient():
    """Optimization and the Hessian standard errors both need this.

    The coefficient vector is unchanged in length by the period axis, so
    jax.grad and jax.hessian work on it unmodified -- but only if the period
    axis is differentiable, which lax.scan inside the likelihood makes easy to
    break.
    """
    rng = np.random.default_rng(6)
    context = _tier_context(_annual_tier(rng))
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")
    model = PradelModel()
    matrices = model.build_design_matrices(spec, context)
    parameters = jnp.array([0.7, 0.3, -0.4, 0.1, -1.2])

    grad = jax.grad(lambda pars: model.log_likelihood(pars, context, matrices))(
        parameters
    )
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.abs(np.asarray(grad)) > 0)

    hessian = jax.hessian(lambda pars: model.log_likelihood(pars, context, matrices))(
        parameters
    )
    assert hessian.shape == (5, 5)
    assert np.all(np.isfinite(np.asarray(hessian)))


def test_grouped_and_individual_backends_agree_under_time_varying_parameters():
    """Frequency weighting must survive the period axis.

    The grouped backend collapses identical rows and multiplies by a count. That
    multiplication happens per individual, while the parameters are now per
    individual per occasion, so the weight has to broadcast rather than line up
    positionally.
    """
    rng = np.random.default_rng(7)
    tier = _annual_tier(rng)
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")
    parameters = jnp.array([0.7, 0.3, -0.4, 0.1, -1.2])
    model = PradelModel()

    context = _tier_context(tier)
    matrices = model.build_design_matrices(spec, context)
    unweighted = float(model.log_likelihood(parameters, context, matrices))

    weighted_context = _tier_context(tier)
    weighted_context.frequency = jnp.ones(N_INDIVIDUALS, dtype=jnp.float64) * 3.0
    weighted_matrices = model.build_design_matrices(spec, weighted_context)
    weighted = float(
        model.log_likelihood(parameters, weighted_context, weighted_matrices)
    )

    assert weighted == pytest.approx(3.0 * unweighted, rel=1e-12)


def test_priors_are_weighted_per_individual_under_time_varying_parameters():
    """The boundary prior is a per-row penalty; the period axis must not break it.

    The grouped backend collapses duplicate capture histories and carries the
    count in `frequency`, so a penalty applied once per stored row would be
    weaker than the same penalty on the ungrouped data -- and the two backends
    would then rank models differently for the same fit.  With occasion-specific
    parameters the penalty array gains a time axis that the frequency vector
    does not have, which is exactly where that weighting is easy to lose.
    """
    rng = np.random.default_rng(8)
    tier = _annual_tier(rng)
    spec = pj.create_formula_spec(phi="~1 + tier", p="~1", f="~1")
    parameters = jnp.array([0.7, 0.3, -0.4, 0.1, -1.2])
    model = PradelModel(
        boundary_prior_strength=0.5,
        recruitment_prior_strength=0.5,
    )

    context = _tier_context(tier)
    matrices = model.build_design_matrices(spec, context)
    unweighted = float(model.log_likelihood(parameters, context, matrices))

    weighted_context = _tier_context(tier)
    weighted_context.frequency = jnp.full(N_INDIVIDUALS, 3.0, dtype=jnp.float64)
    weighted = float(
        model.log_likelihood(
            parameters,
            weighted_context,
            model.build_design_matrices(spec, weighted_context),
        )
    )

    assert weighted == pytest.approx(3.0 * unweighted, rel=1e-12)
