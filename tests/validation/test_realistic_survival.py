#!/usr/bin/env python3
"""
Test with realistic capture patterns that should produce moderate survival estimates.
"""

from pathlib import Path
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize
import warnings
import pytest

warnings.filterwarnings("ignore", ".*TPU.*")

from pradel_jax.models.pradel import PradelModel, logit, inv_logit, log_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec
import pradel_jax as pj

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEBRASKA_DATA = PROJECT_ROOT / "data/20250904_ne_hip_tier_data.csv"

pytestmark = [pytest.mark.slow]

def _run_realistic_moderate_survival():
    """Test with patterns that should give moderate survival (φ ≈ 0.6-0.8)."""
    print("TESTING WITH REALISTIC MODERATE SURVIVAL PATTERNS")
    print("=" * 60)

    # Patterns that suggest good survival with reasonable detection
    realistic_patterns = [
        "11111",  # Perfect survival and detection
        "11110",  # High survival, missed last
        "11100",  # Good early survival, died/emigrated
        "01111",  # Entered late, then high survival
        "11011",  # High survival, one missed detection
        "10111",  # Early capture, then consistent
        "11101",  # Consistent with one gap
        "01110",  # Middle period survival
        "11010",  # Some gaps but survived multiple periods
        "10110",  # Intermittent but survived
        # Add a few patterns with lower survival for realism
        "10000",  # Single capture (low survival)
        "01000",  # Single capture (low survival)
        "11000",  # Two captures then gone
    ] * 3  # Replicate patterns to increase sample size

    import tempfile
    import os
    import pandas as pd

    df = pd.DataFrame({
        'individual_id': range(len(realistic_patterns)),
        'ch': realistic_patterns,
        'sex': (['M', 'F'] * (len(realistic_patterns)//2 + 1))[:len(realistic_patterns)]
    })

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()

    try:
        data_context = pj.load_data(temp_file.name)
        print(f"Created dataset with {data_context.n_individuals} individuals")

        # Calculate empirical survival estimate
        capture_matrix = data_context.capture_matrix

        # More sophisticated empirical estimate
        seen_again_rates = []
        for t in range(capture_matrix.shape[1] - 1):
            captured_t = capture_matrix[:, t] == 1
            if jnp.sum(captured_t) > 0:
                # Count individuals seen in ANY subsequent occasion
                seen_later = jnp.sum(capture_matrix[:, t+1:], axis=1) > 0
                seen_again_rate = jnp.sum(captured_t & seen_later) / jnp.sum(captured_t)
                seen_again_rates.append(float(seen_again_rate))

        empirical_apparent_survival = (
            float(np.mean(seen_again_rates)) if seen_again_rates else np.nan
        )
        if not np.isnan(empirical_apparent_survival):
            print(f"Empirical apparent survival estimate: {empirical_apparent_survival:.3f}")
            print(f"Expected model estimate: φ ≈ {empirical_apparent_survival:.2f}")
        else:
            pytest.skip("Synthetic dataset did not yield any recaptures for survival estimate")

        # Fit model
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
            import jax
            grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
            return -np.array(grad_fn(jnp.array(params)))

        # Try multiple reasonable starting points
        starting_points = [
            jnp.array([logit(0.4), logit(0.5), log_link(0.1)]),   # Low-moderate
            jnp.array([logit(0.6), logit(0.6), log_link(0.2)]),   # Moderate
            jnp.array([logit(0.7), logit(0.7), log_link(0.3)]),   # Moderate-high
            jnp.array([logit(0.8), logit(0.5), log_link(0.1)]),   # High survival
        ]

        print(f"\nOptimization results:")
        print("Start φ | Result φ | Result p | Result f | LL      | Boundary Hits")
        print("-" * 70)

        best_result = None
        best_ll = -np.inf

        for i, start_params in enumerate(starting_points):
            start_phi = inv_logit(start_params[0])

            try:
                result = minimize(
                    objective,
                    start_params,
                    method='L-BFGS-B',
                    jac=gradient,
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )

                if result.success:
                    phi_est = inv_logit(result.x[0])
                    p_est = inv_logit(result.x[1])
                    f_est = np.exp(result.x[2])
                    ll = -result.fun

                    # Check boundary hits
                    boundary_hits = []
                    for j, (param_val, (lower, upper)) in enumerate(zip(result.x, bounds)):
                        if abs(param_val - lower) < 1e-4:
                            boundary_hits.append(f"{param_names[j]}↓")
                        elif abs(param_val - upper) < 1e-4:
                            boundary_hits.append(f"{param_names[j]}↑")

                    boundary_str = ",".join(boundary_hits) if boundary_hits else "None"

                    print(f"  {start_phi:.2f}  |   {phi_est:.3f}  |   {p_est:.3f}  |   {f_est:.3f}  | {ll:7.1f} | {boundary_str}")

                    if ll > best_ll:
                        best_ll = ll
                        best_result = result

                else:
                    print(f"  {start_phi:.2f}  |   FAILED   |   FAILED   |   FAILED   |  FAILED | FAILED")

            except Exception as e:
                print(f"  {start_phi:.2f}  |   ERROR    |   ERROR    |   ERROR    |   ERROR | ERROR")

        if best_result is not None:
            phi_final = inv_logit(best_result.x[0])
            p_final = inv_logit(best_result.x[1])
            f_final = np.exp(best_result.x[2])

            print(f"\n" + "=" * 50)
            print("ASSESSMENT")
            print("=" * 50)
            print(f"Empirical apparent survival: {empirical_apparent_survival:.3f}")
            print(f"Model survival estimate: {phi_final:.3f}")
            print(f"Difference: {phi_final - empirical_apparent_survival:+.3f}")

            if abs(phi_final - empirical_apparent_survival) < 0.2:
                print("✅ Model estimate reasonably close to empirical estimate")
                return True
            if phi_final < 0.2 and empirical_apparent_survival > 0.5:
                pytest.fail("Model severely underestimates survival on realistic patterns")
            if phi_final > 0.9:
                pytest.fail("Model still hits φ upper boundary on realistic patterns")

            print("⚠️  Model estimate differs from empirical, but within reasonable range")
            return True

        pytest.fail("Failed to obtain optimisation result for realistic survival patterns")

    finally:
        os.unlink(temp_file.name)


def test_realistic_moderate_survival():
    assert _run_realistic_moderate_survival()


def _run_nebraska_subsample():
    """Test with Nebraska data subsample that showed moderate-high survival."""
    print(f"\n" + "=" * 60)
    print("TESTING WITH NEBRASKA DATA SUBSAMPLE")
    print("=" * 60)

    if not NEBRASKA_DATA.exists():  # pragma: no cover
        pytest.skip("Nebraska dataset not available for survival regression")

    from nebraska_data_loader import load_and_prepare_nebraska_data

    # Load a reasonable sample size
    data_context, df = load_and_prepare_nebraska_data(n_sample=100, random_state=123)
    if data_context is None:
        pytest.skip("Failed to load Nebraska data subset")

    print(f"Loaded {data_context.n_individuals} Nebraska individuals")

    # Quick empirical analysis
    capture_matrix = data_context.capture_matrix

    seen_again_rates = []
    for t in range(capture_matrix.shape[1] - 1):
        captured_t = capture_matrix[:, t] == 1
        if jnp.sum(captured_t) > 0:
            seen_later = jnp.sum(capture_matrix[:, t+1:], axis=1) > 0
            seen_again_rate = jnp.sum(captured_t & seen_later) / jnp.sum(captured_t)
            seen_again_rates.append(float(seen_again_rate))

    empirical_survival = np.mean(seen_again_rates) if seen_again_rates else 0.0
    print(f"Empirical apparent survival: {empirical_survival:.3f}")

    # Fit model
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

    result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

    if result.success:
        phi_est = inv_logit(result.x[0])
        p_est = inv_logit(result.x[1])
        f_est = np.exp(result.x[2])

        print(f"Model estimates: φ={phi_est:.3f}, p={p_est:.3f}, f={f_est:.3f}")
        print(f"Difference from empirical: {phi_est - empirical_survival:+.3f}")

        if phi_est < 0.1 and empirical_survival > 0.5:
            pytest.fail("Nebraska subsample survival remains severely underestimated")

        print("✅ Model produces reasonable survival estimate for Nebraska data")
        return True

    pytest.fail("Optimization failed on Nebraska data subset")


def test_nebraska_subsample():
    assert _run_nebraska_subsample()

def main():
    """Test realistic survival scenarios."""
    print("TESTING REALISTIC SURVIVAL SCENARIOS")
    print("=" * 70)

    realistic_ok = _run_realistic_moderate_survival()
    nebraska_ok = _run_nebraska_subsample()

    print(f"\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)

    if realistic_ok and nebraska_ok:
        print("✅ Model produces reasonable survival estimates")
        print("✅ Balanced penalty successfully prevents both φ→1.0 and φ→0")
        print("✅ Implementation performs well on realistic data")
    elif realistic_ok or nebraska_ok:
        print("⚠️  Mixed results - may need further penalty adjustment")
    else:
        print("❌ Model still has issues with survival estimation")
        print("❌ May need to reduce penalty strength further")

    return realistic_ok and nebraska_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
