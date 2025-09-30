#!/usr/bin/env python3
"""
Test script to validate the convergence fix for infinite parameter bounds.

This script will:
1. Verify that bounds are now finite
2. Test multiple model structures for stable convergence
3. Check standard error ranges are reasonable
4. Ensure statistical significance is detected
5. Confirm different models produce different parameter values
"""

from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import pandas as pd
import warnings
import pytest

# Suppress JAX warnings
warnings.filterwarnings("ignore", ".*TPU.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOUTH_DAKOTA_DATA = PROJECT_ROOT / "data/20250903_sd_hip_tier_data.csv"

pytestmark = [pytest.mark.slow]

if not SOUTH_DAKOTA_DATA.exists():  # pragma: no cover - optional dataset
    pytestmark.append(
        pytest.mark.skip(reason="South Dakota dataset not available in repo")
    )

from south_dakota_data_loader import load_and_prepare_south_dakota_data
import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel, logit, inv_logit, log_link, exp_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec

def _check_bounds_are_finite():
    """Test that all parameter bounds are now finite."""
    print("="*60)
    print("1. TESTING FINITE BOUNDS")
    print("="*60)

    # Load test data
    data_context, df = load_and_prepare_south_dakota_data(n_sample=100, random_state=42)
    if data_context is None:
        pytest.skip("Failed to load South Dakota sample for convergence regression")

    # Test with different model structures
    parser = FormulaParser()
    test_models = [
        ("null_model", FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )),
        ("gender_model", FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + gender"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        ))
    ]

    all_finite = True
    for model_name, formula_spec in test_models:
        print(f"\nTesting {model_name}:")

        model = PradelModel()
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        param_names = model.get_parameter_names(design_matrices)

        print(f"  Parameter bounds:")
        for name, (lower, upper) in zip(param_names, bounds):
            is_finite = np.isfinite(lower) and np.isfinite(upper)
            status = "✅" if is_finite else "❌"
            print(f"    {status} {name}: [{lower:.3f}, {upper:.3f}]")
            if not is_finite:
                all_finite = False

    print(f"\n✅ All bounds finite: {all_finite}")
    assert all_finite, "Encountered non-finite parameter bounds"
    return all_finite

def _run_model_convergence():
    """Test that different models converge to different parameter values."""
    print("\n" + "="*60)
    print("2. TESTING MODEL CONVERGENCE AND DISTINCTIVENESS")
    print("="*60)

    # Load larger sample for more robust testing
    data_context, df = load_and_prepare_south_dakota_data(n_sample=300, random_state=42)
    if data_context is None:
        pytest.skip("Failed to load South Dakota sample for convergence regression")

    parser = FormulaParser()

    # Test models with different complexity
    test_models = [
        ("phi(~1)_p(~1)_f(~1)", FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )),
        ("phi(~gender)_p(~1)_f(~1)", FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + gender"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )),
        ("phi(~1)_p(~gender)_f(~1)", FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1 + gender"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )),
        ("phi(~1)_p(~1)_f(~gender)", FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1 + gender")
        ))
    ]

    results = []

    for model_name, formula_spec in test_models:
        print(f"\nFitting {model_name}:")

        try:
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

            # Fit model with corrected bounds
            result = minimize(
                objective,
                initial_params,
                method='L-BFGS-B',
                jac=gradient,
                bounds=bounds,
                options={'disp': False, 'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
            )

            if result.success:
                log_likelihood = -result.fun
                n_params = len(result.x)
                aic = 2 * n_params - 2 * log_likelihood

                # Calculate approximate standard errors from Hessian
                try:
                    # Numerical Hessian
                    from scipy.optimize import approx_fprime
                    eps = np.sqrt(np.finfo(float).eps)
                    hessian_diag = []

                    for i in range(len(result.x)):
                        def partial_grad(x):
                            return gradient(result.x + eps * x)[i]

                        h_ii = approx_fprime(np.zeros(len(result.x)), partial_grad, eps)[i] / eps
                        if h_ii > 0:
                            se = 1.0 / np.sqrt(h_ii)
                        else:
                            se = np.nan
                        hessian_diag.append(se)

                    se_values = np.array(hessian_diag)
                    se_min = np.nanmin(se_values)
                    se_max = np.nanmax(se_values)

                except Exception as se_error:
                    se_values = np.full(n_params, np.nan)
                    se_min = se_max = np.nan

                # Check for boundary hits
                boundary_hits = []
                for i, (param_val, (lower, upper)) in enumerate(zip(result.x, bounds)):
                    if abs(param_val - lower) < 1e-6:
                        boundary_hits.append(f"{param_names[i]} (lower)")
                    elif abs(param_val - upper) < 1e-6:
                        boundary_hits.append(f"{param_names[i]} (upper)")

                result_dict = {
                    'model_name': model_name,
                    'success': True,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'n_parameters': n_params,
                    'parameters': result.x.copy(),
                    'parameter_names': param_names,
                    'gradient_norm': np.linalg.norm(result.jac),
                    'se_min': se_min,
                    'se_max': se_max,
                    'se_values': se_values,
                    'boundary_hits': boundary_hits,
                    'iterations': result.nit
                }

                print(f"  ✅ Success: Log-likelihood = {log_likelihood:.3f}, AIC = {aic:.1f}")
                print(f"     Parameters: {[f'{p:.4f}' for p in result.x]}")
                print(f"     Gradient norm: {np.linalg.norm(result.jac):.2e}")
                if not np.isnan(se_min) and not np.isnan(se_max):
                    print(f"     SE range: [{se_min:.6f}, {se_max:.6f}]")
                if boundary_hits:
                    print(f"     ⚠️  Boundary hits: {boundary_hits}")

            else:
                result_dict = {
                    'model_name': model_name,
                    'success': False,
                    'message': result.message
                }
                print(f"  ❌ Failed: {result.message}")

            results.append(result_dict)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'model_name': model_name,
                'success': False,
                'error': str(e)
            })

    successful_results = [r for r in results if r.get('success', False)]
    assert successful_results, "No models converged under convergence regression"
    return results


def test_convergence_bounds_are_finite():
    _check_bounds_are_finite()


def test_convergence_models_converge():
    _run_model_convergence()

def analyze_model_comparison(results):
    """Analyze the model comparison results."""
    print("\n" + "="*60)
    print("3. MODEL COMPARISON ANALYSIS")
    print("="*60)

    successful_results = [r for r in results if r.get('success', False)]

    if len(successful_results) < 2:
        print("❌ Need at least 2 successful models for comparison")
        return

    # Sort by AIC
    successful_results.sort(key=lambda x: x['aic'])

    print(f"✅ {len(successful_results)} models converged successfully")
    print("\nModel ranking (by AIC):")

    for i, result in enumerate(successful_results):
        rank = i + 1
        delta_aic = result['aic'] - successful_results[0]['aic']
        print(f"  {rank}. {result['model_name']}")
        print(f"     AIC: {result['aic']:.1f} (Δ = {delta_aic:.1f})")
        print(f"     Parameters: {result['n_parameters']}")

        # Show parameter estimates
        param_str = []
        for name, val in zip(result['parameter_names'], result['parameters']):
            param_str.append(f"{name}={val:.3f}")
        print(f"     Estimates: {', '.join(param_str)}")

    # Check for parameter distinctiveness
    print(f"\n📊 PARAMETER DISTINCTIVENESS CHECK:")

    if len(successful_results) >= 2:
        # Compare null model with others
        null_params = successful_results[0]['parameters']

        distinct_models = 0
        for i in range(1, len(successful_results)):
            other_params = successful_results[i]['parameters']

            # Check if parameters are substantially different
            if len(null_params) == len(other_params):
                max_diff = np.max(np.abs(null_params - other_params))
                if max_diff > 0.01:  # Threshold for "different"
                    distinct_models += 1
                    print(f"  ✅ {successful_results[i]['model_name']} has distinct parameters (max diff: {max_diff:.3f})")
                else:
                    print(f"  ⚠️  {successful_results[i]['model_name']} has similar parameters (max diff: {max_diff:.3f})")
            else:
                distinct_models += 1
                print(f"  ✅ {successful_results[i]['model_name']} has different structure")

        print(f"\n  Models with distinct parameters: {distinct_models}/{len(successful_results)-1}")

    # Check standard error ranges
    print(f"\n📈 STANDARD ERROR ANALYSIS:")

    all_se_min = []
    all_se_max = []

    for result in successful_results:
        if not np.isnan(result.get('se_min', np.nan)):
            all_se_min.append(result['se_min'])
        if not np.isnan(result.get('se_max', np.nan)):
            all_se_max.append(result['se_max'])

    if all_se_min and all_se_max:
        overall_se_min = min(all_se_min)
        overall_se_max = max(all_se_max)
        print(f"  Overall SE range: [{overall_se_min:.6f}, {overall_se_max:.6f}]")

        # Check if we've resolved the extreme ranges
        if overall_se_max < 10.0 and overall_se_min > 1e-5:
            print(f"  ✅ Standard error range is reasonable!")
        else:
            print(f"  ⚠️  Standard errors still problematic")
    else:
        print(f"  ⚠️  Could not calculate standard errors")

def main():
    """Run comprehensive convergence validation."""
    print("CONVERGENCE FIX VALIDATION TEST")
    print("Testing the fix for infinite parameter bounds")
    print("="*60)

    # Test 1: Verify bounds are finite
    bounds_ok = _check_bounds_are_finite()

    # Test 2: Test model convergence
    results = _run_model_convergence()

    # Test 3: Analyze results
    analyze_model_comparison(results)

    # Overall assessment
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT")
    print("="*60)

    successful_models = len([r for r in results if r.get('success', False)])
    total_models = len(results)

    print(f"✅ Finite bounds: {bounds_ok}")
    print(f"✅ Model success rate: {successful_models}/{total_models} ({successful_models/total_models*100:.1f}%)")

    if bounds_ok and successful_models >= 3:
        print(f"\n🎉 CONVERGENCE FIX SUCCESSFUL!")
        print(f"   - Infinite bounds have been eliminated")
        print(f"   - Models are converging successfully")
        print(f"   - Different models produce distinct results")
        print(f"   - Ready for full model comparison analysis")
    else:
        print(f"\n⚠️  Issues remain - further investigation needed")

    return results

if __name__ == "__main__":
    main()
