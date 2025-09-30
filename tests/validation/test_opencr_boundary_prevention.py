#!/usr/bin/env python3
"""
Test whether the OpenCR Pradel formulation prevents φ→1.0 boundary issues.
"""

from pathlib import Path
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
import warnings
import pytest

warnings.filterwarnings("ignore", ".*TPU.*")

from pradel_opencr_model import PradelOpenCRModel
from pradel_jax.models.pradel import PradelModel, logit, inv_logit, log_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec
import pradel_jax as pj

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEBRASKA_DATA = PROJECT_ROOT / "data/20250904_ne_hip_tier_data.csv"

pytestmark = [pytest.mark.slow]

if not NEBRASKA_DATA.exists():  # pragma: no cover
    pytestmark.append(
        pytest.mark.skip(reason="Nebraska dataset not available for OpenCR regression")
    )

def test_problematic_patterns():
    """Test patterns that previously caused φ→1.0 boundary hits."""
    print("TESTING PROBLEMATIC PATTERNS WITH OPENCR")
    print("=" * 60)

    # These patterns historically caused φ→1.0 issues
    problematic_patterns = [
        "10001",  # Early then very late capture
        "100001", # Early then very late (6 occasions)
        "10000",  # Single early capture
        "00001",  # Single very late capture
    ]

    results = {}

    for pattern in problematic_patterns:
        print(f"\nTesting pattern '{pattern}':")

        # Create test data
        import tempfile
        import os
        import pandas as pd

        df = pd.DataFrame({
            'individual_id': [0, 1, 2],  # Add some variety
            'ch': [pattern, "11100", "01110"],  # Mix with other patterns
            'sex': ['M', 'F', 'M']
        })

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()

        try:
            data_context = pj.load_data(temp_file.name)

            # Create formula
            parser = FormulaParser()
            formula_spec = FormulaSpec(
                phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
                p=parser.create_parameter_formula(ParameterType.P, "~1"),
                f=parser.create_parameter_formula(ParameterType.F, "~1")
            )

            # Test both original and OpenCR models
            models = {
                'Original': PradelModel(),
                'OpenCR': PradelOpenCRModel()
            }

            pattern_results = {}

            for model_name, model in models.items():
                design_matrices = model.build_design_matrices(formula_spec, data_context)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                initial_params = model.get_initial_parameters(data_context, design_matrices)

                def objective(params):
                    return -model.log_likelihood(jnp.array(params), data_context, design_matrices)

                # Test multiple starting points, especially high φ
                starting_points = [
                    jnp.array([logit(0.8), logit(0.5), log_link(0.1)]),
                    jnp.array([logit(0.95), logit(0.3), log_link(0.05)]),  # Very high φ
                    jnp.array([logit(0.99), logit(0.6), log_link(0.01)]),  # Extreme φ
                ]

                max_phi = 0.0
                best_result = None

                for start_params in starting_points:
                    try:
                        result = minimize(objective, start_params, method='L-BFGS-B', bounds=bounds)

                        if result.success:
                            phi_est = inv_logit(result.x[0])
                            max_phi = max(max_phi, phi_est)
                            if best_result is None or -result.fun > best_result[1]:
                                best_result = (result, -result.fun, phi_est)

                    except Exception:
                        continue

                if best_result is not None:
                    phi_final = best_result[2]
                    ll_final = best_result[1]

                    # Check if hitting boundary
                    upper_bound = 0.99 if model_name == 'OpenCR' else 0.95
                    boundary_hit = phi_final >= upper_bound - 0.01

                    pattern_results[model_name] = {
                        'phi': phi_final,
                        'll': ll_final,
                        'boundary_hit': boundary_hit
                    }

                    status = "❌ BOUNDARY" if boundary_hit else "✅ INTERIOR"
                    print(f"  {model_name:8s}: φ={phi_final:.3f}, LL={ll_final:6.1f} {status}")
                else:
                    pattern_results[model_name] = {'phi': 0.0, 'll': -1e6, 'boundary_hit': True}
                    print(f"  {model_name:8s}: FAILED")

            results[pattern] = pattern_results

        finally:
            os.unlink(temp_file.name)

    original_hits = sum(
        1
        for pattern_data in results.values()
        if pattern_data.get('Original', {}).get('boundary_hit')
    )
    opencr_hits = sum(
        1
        for pattern_data in results.values()
        if pattern_data.get('OpenCR', {}).get('boundary_hit')
    )

    assert opencr_hits <= original_hits, "OpenCR introduces more boundary hits on synthetic patterns"
    return results

def test_nebraska_data_comparison():
    """Compare original vs OpenCR on Nebraska data."""
    print(f"\n" + "=" * 60)
    print("NEBRASKA DATA: ORIGINAL VS OPENCR COMPARISON")
    print("=" * 60)

    from nebraska_data_loader import load_and_prepare_nebraska_data

    # Load moderate sample
    data_context, df = load_and_prepare_nebraska_data(n_sample=100, random_state=42)
    if data_context is None:
        print("❌ Failed to load Nebraska data")
        return None

    print(f"Testing with {data_context.n_individuals} Nebraska individuals")

    # Setup
    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )

    models = {
        'Original': PradelModel(),
        'OpenCR': PradelOpenCRModel()
    }

    results = {}

    print("Model    | φ Est | p Est | f Est | LL     | Boundary Hit?")
    print("-" * 60)

    for model_name, model in models.items():
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        initial_params = model.get_initial_parameters(data_context, design_matrices)

        def objective(params):
            return -model.log_likelihood(jnp.array(params), data_context, design_matrices)

        # Multiple starting points
        starting_points = [
            initial_params,
            jnp.array([logit(0.6), logit(0.4), log_link(0.2)]),
            jnp.array([logit(0.8), logit(0.6), log_link(0.1)]),
        ]

        best_result = None
        best_ll = -np.inf

        for start_params in starting_points:
            try:
                result = minimize(objective, start_params, method='L-BFGS-B', bounds=bounds)

                if result.success and -result.fun > best_ll:
                    best_ll = -result.fun
                    best_result = result

            except Exception:
                continue

        if best_result is not None:
            phi_est = inv_logit(best_result.x[0])
            p_est = inv_logit(best_result.x[1])
            f_est = np.exp(best_result.x[2])

            # Check boundary hit
            upper_bound = 0.99 if model_name == 'OpenCR' else 0.95
            boundary_hit = phi_est >= upper_bound - 0.01

            status = "YES" if boundary_hit else "NO"
            print(f"{model_name:8s} | {phi_est:.3f} | {p_est:.3f} | {f_est:.3f} | {best_ll:6.1f} | {status}")

            results[model_name] = {
                'phi': phi_est,
                'p': p_est,
                'f': f_est,
                'll': best_ll,
                'boundary_hit': boundary_hit
            }
        else:
            print(f"{model_name:8s} | FAILED | FAILED | FAILED | FAILED | FAILED")

    if {'Original', 'OpenCR'} <= results.keys():
        assert (
            int(results['OpenCR']['boundary_hit'])
            <= int(results['Original']['boundary_hit'])
        ), "OpenCR has more boundary hits than original on Nebraska data"

    return results

def analyze_likelihood_surfaces():
    """Analyze likelihood surfaces for both models."""
    print(f"\n" + "=" * 60)
    print("LIKELIHOOD SURFACE ANALYSIS")
    print("=" * 60)

    # Create test data with the problematic pattern
    import tempfile
    import os
    import pandas as pd

    df = pd.DataFrame({
        'individual_id': [0, 1],
        'ch': ["10001", "11100"],  # Problematic + normal pattern
        'sex': ['M', 'F']
    })

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    try:
        data_context = pj.load_data(temp_file.name)

        parser = FormulaParser()
        formula_spec = FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )

        models = {
            'Original': PradelModel(),
            'OpenCR': PradelOpenCRModel()
        }

        phi_values = np.linspace(0.5, 0.95, 10)
        p_val = 0.4
        f_val = 0.1

        print("Testing likelihood across φ values:")
        print("φ     | Original LL | OpenCR LL | Diff")
        print("-" * 40)

        surface_data = {}

        for model_name, model in models.items():
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            lls = []

            for phi_val in phi_values:
                params = jnp.array([logit(phi_val), logit(p_val), log_link(f_val)])
                try:
                    ll = model.log_likelihood(params, data_context, design_matrices)
                    lls.append(float(ll))
                except:
                    lls.append(-1e6)

            surface_data[model_name] = lls

        for i, phi_val in enumerate(phi_values):
            original_ll = surface_data['Original'][i]
            opencr_ll = surface_data['OpenCR'][i]
            diff = opencr_ll - original_ll

            print(f"{phi_val:.2f}  | {original_ll:10.1f} | {opencr_ll:8.1f} | {diff:+6.1f}")

        # Find maxima
        original_max_idx = np.argmax(surface_data['Original'])
        opencr_max_idx = np.argmax(surface_data['OpenCR'])

        original_best_phi = phi_values[original_max_idx]
        opencr_best_phi = phi_values[opencr_max_idx]

        print(f"\nLikelihood surface maxima:")
        print(f"  Original: φ = {original_best_phi:.2f} (index {original_max_idx}/{len(phi_values)-1})")
        print(f"  OpenCR:   φ = {opencr_best_phi:.2f} (index {opencr_max_idx}/{len(phi_values)-1})")

        # Check boundary behavior
        original_boundary = original_max_idx >= len(phi_values) - 2
        opencr_boundary = opencr_max_idx >= len(phi_values) - 2

        assert not opencr_boundary or opencr_max_idx <= original_max_idx, (
            "OpenCR likelihood surface still peaks at boundary"
        )

        return {
            'original_boundary_hit': original_boundary,
            'opencr_boundary_hit': opencr_boundary,
            'original_best_phi': original_best_phi,
            'opencr_best_phi': opencr_best_phi
        }

    finally:
        os.unlink(temp_file.name)

def main():
    """Run comprehensive OpenCR boundary prevention test."""
    print("OPENCR BOUNDARY PREVENTION TEST")
    print("=" * 70)

    # Test 1: Problematic patterns
    pattern_results = test_problematic_patterns()

    # Test 2: Nebraska data
    nebraska_results = test_nebraska_data_comparison()

    # Test 3: Likelihood surfaces
    surface_results = analyze_likelihood_surfaces()

    # Summary
    print(f"\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    if pattern_results and nebraska_results and surface_results:
        # Count boundary hits
        original_boundaries = 0
        opencr_boundaries = 0

        for pattern, results in pattern_results.items():
            if 'Original' in results and results['Original']['boundary_hit']:
                original_boundaries += 1
            if 'OpenCR' in results and results['OpenCR']['boundary_hit']:
                opencr_boundaries += 1

        if nebraska_results:
            if nebraska_results.get('Original', {}).get('boundary_hit', False):
                original_boundaries += 1
            if nebraska_results.get('OpenCR', {}).get('boundary_hit', False):
                opencr_boundaries += 1

        if surface_results:
            if surface_results.get('original_boundary_hit', False):
                original_boundaries += 1
            if surface_results.get('opencr_boundary_hit', False):
                opencr_boundaries += 1

        total_tests = len(pattern_results) + 1 + 1  # patterns + nebraska + surface

        print(f"Boundary hits summary:")
        print(f"  Original model: {original_boundaries}/{total_tests} tests hit boundary")
        print(f"  OpenCR model:   {opencr_boundaries}/{total_tests} tests hit boundary")

        if opencr_boundaries < original_boundaries:
            print(f"\n🎉 SUCCESS: OpenCR reduces boundary hits!")
            print(f"✅ OpenCR formulation is more stable")
            if opencr_boundaries == 0:
                print(f"✅ OpenCR completely eliminates φ→1.0 boundary issue!")
            return True
        elif opencr_boundaries == original_boundaries:
            print(f"\n⚠️  MIXED: OpenCR doesn't improve boundary behavior")
            return False
        else:
            print(f"\n❌ CONCERN: OpenCR shows more boundary hits")
            return False

    else:
        print("❌ Tests incomplete")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
