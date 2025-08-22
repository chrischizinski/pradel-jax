#!/usr/bin/env python3
"""
Now that we know JAX gradients are correct, test why optimizers are still failing.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel, logit, log_link, inv_logit, exp_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec
import tempfile
import os

def create_test_data():
    """Create simple test data."""
    np.random.seed(42)
    encounter_data = np.array([
        [1, 0, 1],  # Individual 1: captured at 1 and 3
        [0, 1, 1],  # Individual 2: captured at 2 and 3  
        [1, 1, 0],  # Individual 3: captured at 1 and 2
    ])
    
    df_data = []
    for i, history in enumerate(encounter_data):
        ch = ''.join(map(str, history))
        df_data.append({'individual_id': i, 'ch': ch})
    
    df = pd.DataFrame(df_data)
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        data_context = pj.load_data(temp_file.name)
    finally:
        os.unlink(temp_file.name)
    
    return data_context

def test_simple_optimization():
    """Test simple optimization with correct gradients."""
    print("Testing simple optimization with correct JAX gradients")
    print("=" * 60)
    
    # Create test data and model
    data_context = create_test_data()
    
    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )
    
    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    print("Data and model setup:")
    print(f"  {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
    
    # Get initial parameters
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    print(f"  Initial parameters: {initial_params}")
    
    phi_init = inv_logit(initial_params[0])
    p_init = inv_logit(initial_params[1])
    f_init = exp_link(initial_params[2])
    print(f"  Initial natural scale: phi={phi_init:.3f}, p={p_init:.3f}, f={f_init:.3f}")
    
    # Define objective and gradient functions
    def objective(params):
        """Negative log-likelihood (for minimization)."""
        return -float(model.log_likelihood(jnp.array(params), data_context, design_matrices))
    
    def gradient(params):
        """Gradient of negative log-likelihood."""
        grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
        return -np.array(grad_fn(jnp.array(params)))
    
    # Test objective and gradient at initial point
    initial_obj = objective(initial_params)
    initial_grad = gradient(initial_params)
    
    print(f"\nAt initial point:")
    print(f"  Objective: {initial_obj:.6f}")
    print(f"  Gradient: {initial_grad}")
    print(f"  Gradient magnitude: {np.linalg.norm(initial_grad):.6f}")
    
    # Test at a different point
    test_params = jnp.array([logit(0.7), logit(0.6), log_link(0.1)])
    test_obj = objective(test_params)
    test_grad = gradient(test_params)
    
    print(f"\nAt test point [phi=0.7, p=0.6, f=0.1]:")
    print(f"  Parameters: {test_params}")
    print(f"  Objective: {test_obj:.6f}")
    print(f"  Gradient: {test_grad}")
    print(f"  Gradient magnitude: {np.linalg.norm(test_grad):.6f}")
    
    # Now try optimization
    print(f"\nTesting different optimizers:")
    print("-" * 40)
    
    # Test L-BFGS-B
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    result_lbfgs = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        options={'disp': True, 'maxiter': 100}
    )
    
    print(f"L-BFGS-B result:")
    print(f"  Success: {result_lbfgs.success}")
    print(f"  Iterations: {result_lbfgs.nit}")
    print(f"  Function evaluations: {result_lbfgs.nfev}")
    print(f"  Final parameters: {result_lbfgs.x}")
    print(f"  Final objective: {result_lbfgs.fun:.6f}")
    print(f"  Message: {result_lbfgs.message}")
    
    # Convert to natural scale
    phi_final = inv_logit(result_lbfgs.x[0])
    p_final = inv_logit(result_lbfgs.x[1])
    f_final = exp_link(result_lbfgs.x[2])
    print(f"  Final natural scale: phi={phi_final:.3f}, p={p_final:.3f}, f={f_final:.3f}")
    
    return result_lbfgs

def test_gradient_magnitude_scaling():
    """Test if gradient magnitudes are causing convergence issues."""
    print(f"\n" + "="*60)
    print("Testing gradient magnitude scaling")
    print("="*60)
    
    data_context = create_test_data()
    
    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )
    
    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, design_matrices)
    
    def gradient_magnitude_at_point(phi_nat, p_nat, f_nat):
        """Get gradient magnitude at a natural scale point."""
        params = jnp.array([logit(phi_nat), logit(p_nat), log_link(f_nat)])
        grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
        grad = grad_fn(params)
        return np.linalg.norm(grad)
    
    # Test at various points
    test_points = [
        (0.5, 0.3, 0.05),  # Low values
        (0.7, 0.6, 0.1),   # Medium values  
        (0.9, 0.8, 0.2),   # High values
    ]
    
    for phi, p, f in test_points:
        grad_mag = gradient_magnitude_at_point(phi, p, f)
        print(f"Point phi={phi:.1f}, p={p:.1f}, f={f:.2f}: gradient magnitude = {grad_mag:.6f}")

if __name__ == "__main__":
    test_simple_optimization()
    test_gradient_magnitude_scaling()