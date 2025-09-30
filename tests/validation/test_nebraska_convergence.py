#!/usr/bin/env python3
"""
Test Nebraska data with current Pradel implementation to see if it also hits boundaries.
This will help us determine if the issue is data-specific or implementation-specific.
"""

from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import warnings
import pytest

warnings.filterwarnings("ignore", ".*TPU.*")

from nebraska_data_loader import load_and_prepare_nebraska_data
from pradel_jax.models.pradel import PradelModel, logit, inv_logit, log_link, exp_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEBRASKA_DATA = PROJECT_ROOT / "data/20250904_ne_hip_tier_data.csv"
SOUTH_DAKOTA_DATA = PROJECT_ROOT / "data/20250903_sd_hip_tier_data.csv"

pytestmark = [pytest.mark.slow]

if not (NEBRASKA_DATA.exists() and SOUTH_DAKOTA_DATA.exists()):  # pragma: no cover
    pytestmark.append(
        pytest.mark.skip(reason="Nebraska/South Dakota datasets not available")
    )

def test_nebraska_boundary_behavior():
    """Test if Nebraska data also hits phi boundary."""
    print("NEBRASKA DATA BOUNDARY TEST")
    print("="*50)

    # Load Nebraska data
    data_context, df = load_and_prepare_nebraska_data(n_sample=200, random_state=42)
    if data_context is None:
        pytest.skip("Nebraska dataset unavailable for convergence regression")

    print(f"✅ Loaded {data_context.n_individuals} Nebraska individuals")

    # Create simple null model
    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )

    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    param_names = model.get_parameter_names(design_matrices)

    def objective(params):
        return -model.log_likelihood(jnp.array(params), data_context, design_matrices)

    def gradient(params):
        grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
        return -np.array(grad_fn(jnp.array(params)))

    # Fit model
    print("Fitting null model...")
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        options={'maxiter': 500}
    )

    if result.success:
        phi_est = inv_logit(result.x[0])
        p_est = inv_logit(result.x[1])
        f_est = np.exp(result.x[2])

        print(f"✅ Optimization successful:")
        print(f"   φ (survival): {phi_est:.3f}")
        print(f"   p (detection): {p_est:.3f}")
        print(f"   f (recruitment): {f_est:.6f}")
        print(f"   Log-likelihood: {-result.fun:.3f}")

        # Check boundary hits
        boundary_hits = []
        for i, (param_val, (lower, upper)) in enumerate(zip(result.x, bounds)):
            if abs(param_val - lower) < 1e-6:
                boundary_hits.append(f"{param_names[i]} (lower)")
            elif abs(param_val - upper) < 1e-6:
                boundary_hits.append(f"{param_names[i]} (upper)")

        if boundary_hits:
            print(f"⚠️  Boundary hits: {boundary_hits}")
        else:
            print(f"✅ No boundary hits")

        assert not any("phi" in hit and "upper" in hit for hit in boundary_hits), (
            "Nebraska fit still hits phi upper boundary"
        )
        return phi_est, boundary_hits

    else:
        print(f"❌ Optimization failed: {result.message}")
        return None, None

def test_likelihood_surface_nebraska():
    """Test likelihood surface for Nebraska data."""
    print(f"\n" + "="*50)
    print("NEBRASKA LIKELIHOOD SURFACE TEST")
    print("="*50)

    data_context, df = load_and_prepare_nebraska_data(n_sample=200, random_state=42)
    if data_context is None:
        pytest.skip("Nebraska dataset unavailable for likelihood surface regression")

    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )

    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)

    # Test likelihood across phi values
    phi_values_prob = np.linspace(0.5, 0.99, 15)
    phi_values_logit = logit(phi_values_prob)

    # Fix p and f based on Nebraska patterns
    p_logit = logit(0.25)  # Lower capture rate
    f_log = log_link(0.05)  # Lower recruitment

    print(f"Testing Nebraska likelihood surface:")
    print(f"Fixed p = 0.25, f = 0.05")
    print(f"Phi (prob) | Log-Likelihood")
    print(f"-" * 30)

    likelihoods = []
    for phi_prob, phi_logit in zip(phi_values_prob, phi_values_logit):
        params = jnp.array([phi_logit, p_logit, f_log])
        try:
            ll = model.log_likelihood(params, data_context, design_matrices)
            ll_value = float(ll)
            likelihoods.append(ll_value)
            print(f"{phi_prob:.3f}     | {ll_value:.3f}")
        except Exception as e:
            print(f"{phi_prob:.3f}     | ERROR: {e}")
            likelihoods.append(np.nan)

    # Find maximum
    valid_ll = [ll for ll in likelihoods if not np.isnan(ll)]
    if not valid_ll:
        pytest.fail("Failed to evaluate likelihood surface for Nebraska data")

    max_ll = np.max(valid_ll)
    max_idx = likelihoods.index(max_ll)
    best_phi = phi_values_prob[max_idx]

    print(f"\nNebraska best phi: {best_phi:.3f}")
    assert max_idx < len(phi_values_prob) - 2, "Nebraska likelihood surface peaks at boundary"
    return best_phi

def compare_datasets():
    """Compare how the same model performs on both datasets."""
    print(f"\n" + "="*50)
    print("NEBRASKA vs SOUTH DAKOTA COMPARISON")
    print("="*50)

    # Test Nebraska
    print("Testing Nebraska...")
    ne_phi, ne_boundaries = test_nebraska_boundary_behavior()

    print(f"\n" + "-"*30)

    # Test South Dakota
    print("Testing South Dakota...")
    from south_dakota_data_loader import load_and_prepare_south_dakota_data

    data_context, df = load_and_prepare_south_dakota_data(n_sample=500, random_state=42)
    if data_context is None:
        return

    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )

    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    initial_params = model.get_initial_parameters(data_context, design_matrices)

    def objective(params):
        return -model.log_likelihood(jnp.array(params), data_context, design_matrices)

    def gradient(params):
        grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
        return -np.array(grad_fn(jnp.array(params)))

    result = minimize(objective, initial_params, method='L-BFGS-B', jac=gradient, bounds=bounds)

    if result.success:
        sd_phi = inv_logit(result.x[0])
        print(f"South Dakota φ: {sd_phi:.3f}")

        sd_boundaries = []
        param_names = model.get_parameter_names(design_matrices)
        for i, (param_val, (lower, upper)) in enumerate(zip(result.x, bounds)):
            if abs(param_val - upper) < 1e-6:
                sd_boundaries.append(param_names[i])

        print(f"Boundary hits: {sd_boundaries}")

    # Summary
    print(f"\n" + "="*50)
    print("DATASET COMPARISON SUMMARY")
    print("="*50)

    if ne_phi is not None:
        print(f"Nebraska φ estimate: {ne_phi:.3f}")
        if ne_boundaries:
            print(f"Nebraska boundary hits: {ne_boundaries}")
        else:
            print(f"Nebraska: No boundary hits")

    print(f"South Dakota φ estimate: {sd_phi:.3f}")
    if sd_boundaries:
        print(f"South Dakota boundary hits: {sd_boundaries}")

    if ne_phi is not None and ne_phi < 0.99 and sd_phi >= 0.99:
        print(f"\n🎯 KEY FINDING: Nebraska works, South Dakota hits boundary!")
        print(f"This confirms there's a real data-specific issue to debug.")

def main():
    """Run Nebraska vs South Dakota comparison."""
    test_likelihood_surface_nebraska()
    compare_datasets()

if __name__ == "__main__":
    main()
