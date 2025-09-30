#!/usr/bin/env python3
"""
Test the corrected Pradel implementation to verify it fixes the phi → 1.0 boundary issue.
"""

import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
import warnings
import pytest

warnings.filterwarnings("ignore", ".*TPU.*")

from pradel_jax.models.pradel import PradelModel, logit, inv_logit, log_link, exp_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec
from pradel_jax.data.adapters import DataContext
import pradel_jax as pj

pytestmark = [pytest.mark.slow]

def create_test_data():
    """Create synthetic test data."""
    # Simple capture histories
    capture_histories = [
        "10100",  # Captured at 0 and 2
        "01010",  # Captured at 1 and 3
        "00100",  # Captured only at 2
        "11100",  # Captured at 0,1,2
        "00010",  # Captured only at 3
    ]

    import tempfile
    import os
    import pandas as pd

    # Create DataFrame
    df = pd.DataFrame({
        'individual_id': range(len(capture_histories)),
        'ch': capture_histories,
        'sex': ['M', 'F', 'M', 'F', 'M']
    })

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    try:
        data_context = pj.load_data(temp_file.name)
        return data_context
    finally:
        os.unlink(temp_file.name)

def _check_likelihood_surface():
    """Test likelihood surface with corrected implementation."""
    print("TESTING CORRECTED PRADEL IMPLEMENTATION")
    print("="*50)

    # Create test data
    data_context = create_test_data()
    print(f"✅ Created test data: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")

    # Create simple null model
    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )

    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)

    # Test likelihood across phi values
    phi_values = np.linspace(0.5, 0.95, 10)
    p_val = 0.4
    f_val = 0.1

    print("\nTesting likelihood surface:")
    print("Phi    | Log-Likelihood")
    print("-" * 25)

    likelihoods = []
    for phi_val in phi_values:
        # Create parameter vector
        params = jnp.array([
            logit(phi_val),  # phi on logit scale
            logit(p_val),    # p on logit scale
            log_link(f_val)  # f on log scale
        ])

        try:
            ll = model.log_likelihood(params, data_context, design_matrices)
            ll_value = float(ll)
            likelihoods.append(ll_value)
            print(f"{phi_val:.2f}   | {ll_value:.3f}")
        except Exception as e:
            print(f"{phi_val:.2f}   | ERROR: {e}")
            likelihoods.append(np.nan)

    # Find maximum
    valid_ll = [ll for ll in likelihoods if not np.isnan(ll)]
    assert valid_ll, "Likelihood evaluation failed across grid"

    max_ll = np.max(valid_ll)
    max_idx = likelihoods.index(max_ll)
    best_phi = phi_values[max_idx]

    print(f"\nCorrected implementation results:")
    print(f"Best phi: {best_phi:.3f}")
    interior = 1 < max_idx < len(phi_values) - 2
    if not interior:
        pytest.fail("Corrected implementation still maximises at boundary")

    print("✅ Maximum found in interior!")
    return True

def _check_full_optimization():
    """Test full optimization with corrected implementation."""
    print(f"\n" + "="*50)
    print("TESTING FULL OPTIMIZATION")
    print("="*50)

    # Create test data
    data_context = create_test_data()

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

    # Fit model
    print("Running optimization...")
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500}
    )

    assert result.success, f"Optimization failed: {result.message}"

    phi_est = inv_logit(result.x[0])
    p_est = inv_logit(result.x[1])
    f_est = np.exp(result.x[2])

    print(f"✅ Optimization successful:")
    print(f"   φ (survival): {phi_est:.3f}")
    print(f"   p (detection): {p_est:.3f}")
    print(f"   f (recruitment): {f_est:.6f}")
    print(f"   Log-likelihood: {-result.fun:.3f}")

    boundary_hits = []
    for i, (param_val, (lower, upper)) in enumerate(zip(result.x, bounds)):
        if abs(param_val - lower) < 1e-6:
            boundary_hits.append(f"{param_names[i]} (lower)")
        elif abs(param_val - upper) < 1e-6:
            boundary_hits.append(f"{param_names[i]} (upper)")

    assert not boundary_hits, f"Parameters stuck on numerical bounds: {boundary_hits}"
    print(f"✅ No boundary hits - optimization successful!")
    return True

def main():
    """Run all tests."""
    print("TESTING CORRECTED PRADEL IMPLEMENTATION")
    print("="*60)

    # Test likelihood surface
    surface_success = _check_likelihood_surface()

    # Test full optimization
    opt_success = _check_full_optimization()

    print(f"\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    if surface_success and opt_success:
        print("✅ SUCCESS: Corrected implementation fixes phi → 1.0 boundary issue!")
        print("✅ Both likelihood surface and optimization show interior maximum")
    elif surface_success:
        print("✅ PARTIAL SUCCESS: Likelihood surface shows interior maximum")
        print("⚠️  But optimization still has issues")
    elif opt_success:
        print("✅ PARTIAL SUCCESS: Optimization finds interior maximum")
        print("⚠️  But likelihood surface analysis shows boundary hit")
    else:
        print("❌ FAILURE: Corrected implementation still has boundary issues")
        print("❌ Both tests show boundary hits")

    return surface_success and opt_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


def test_corrected_likelihood_surface_interior():
    assert _check_likelihood_surface()


def test_corrected_implementation_optimization():
    assert _check_full_optimization()
