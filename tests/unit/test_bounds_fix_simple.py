#!/usr/bin/env python3
"""
Simple test to validate the parameter bounds fix using direct scipy optimization.
"""

import sys
import numpy as np
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
        [1, 1, 1],  # Individual 4: captured at all occasions
        [0, 0, 1],  # Individual 5: captured only at last occasion
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

def test_bounds_fix():
    """Test the parameter bounds fix with direct scipy optimization."""
    print("Testing Parameter Bounds Fix")
    print("=" * 40)
    
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
    
    print(f"Data: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
    
    # Get the NEW bounds from the fixed model
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    print("Fixed parameter bounds:")
    param_names = ['phi', 'p', 'f']
    for i, (lower, upper) in enumerate(bounds):
        param_name = param_names[i]
        if param_name in ['phi', 'p']:
            nat_lower = inv_logit(lower)
            nat_upper = inv_logit(upper)
            print(f"  {param_name}: [{lower:.3f}, {upper:.3f}] -> natural: [{nat_lower:.4f}, {nat_upper:.4f}]")
        else:  # f
            nat_lower = exp_link(lower)
            nat_upper = exp_link(upper)
            print(f"  {param_name}: [{lower:.3f}, {upper:.3f}] -> natural: [{nat_lower:.4f}, {nat_upper:.4f}]")
    
    # Get initial parameters
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    
    phi_init = inv_logit(initial_params[0])
    p_init = inv_logit(initial_params[1])
    f_init = exp_link(initial_params[2])
    print(f"\nInitial parameters (natural): phi={phi_init:.4f}, p={p_init:.4f}, f={f_init:.4f}")
    
    # Define objective and gradient
    def objective(params):
        return -float(model.log_likelihood(jnp.array(params), data_context, design_matrices))
    
    def gradient(params):
        import jax
        grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
        return -np.array(grad_fn(jnp.array(params)))
    
    # Test optimization with L-BFGS-B
    print(f"\nRunning L-BFGS-B optimization...")
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        jac=gradient,
        bounds=bounds,
        options={'disp': False, 'maxiter': 100}
    )
    
    print(f"\nResults:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Final objective: {result.fun:.6f}")
    print(f"  Final log-likelihood: {-result.fun:.6f}")
    
    if result.success:
        phi_final = inv_logit(result.x[0])
        p_final = inv_logit(result.x[1])
        f_final = exp_link(result.x[2])
        print(f"  Final parameters (natural): phi={phi_final:.4f}, p={p_final:.4f}, f={f_final:.4f}")
        
        # Check if we hit bounds
        hit_bounds = []
        for i, (param_val, (lower, upper)) in enumerate(zip(result.x, bounds)):
            param_name = param_names[i]
            if abs(param_val - lower) < 1e-3:
                hit_bounds.append(f"{param_name} (lower)")
            elif abs(param_val - upper) < 1e-3:
                hit_bounds.append(f"{param_name} (upper)")
        
        if hit_bounds:
            print(f"  ⚠️  Hit bounds for: {', '.join(hit_bounds)}")
        else:
            print(f"  ✅ Did not hit bounds - optimization converged internally")
            
        # Calculate AIC
        n_params = len(result.x)
        aic = 2 * n_params - 2 * (-result.fun)
        print(f"  AIC: {aic:.2f}")
        
        print(f"\n✅ OPTIMIZATION SUCCESSFUL - Parameter bounds fix worked!")
        
    else:
        print(f"\n❌ OPTIMIZATION FAILED")
        
    return result

def compare_with_old_bounds():
    """Compare with the old restrictive bounds to show the difference."""
    print(f"\n" + "="*40)
    print("Comparing with OLD restrictive bounds")
    print("="*40)
    
    data_context = create_test_data()
    
    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )
    
    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    
    # OLD RESTRICTIVE BOUNDS (what was causing the problem)
    old_bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 5.0)]
    
    def objective(params):
        return -float(model.log_likelihood(jnp.array(params), data_context, design_matrices))
    
    def gradient(params):
        import jax
        grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
        return -np.array(grad_fn(jnp.array(params)))
    
    print("Testing with OLD bounds [-10, 10] for phi,p and [-10, 5] for f...")
    
    result_old = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        jac=gradient,
        bounds=old_bounds,
        options={'disp': False, 'maxiter': 100}
    )
    
    print(f"OLD bounds result:")
    print(f"  Success: {result_old.success}")
    print(f"  Final objective: {result_old.fun:.6f}")
    print(f"  Final parameters: {result_old.x}")
    
    # Check if hit bounds
    hit_bounds = []
    for i, (param_val, (lower, upper)) in enumerate(zip(result_old.x, old_bounds)):
        param_name = ['phi', 'p', 'f'][i]
        if abs(param_val - lower) < 1e-3:
            hit_bounds.append(f"{param_name} (lower)")
        elif abs(param_val - upper) < 1e-3:
            hit_bounds.append(f"{param_name} (upper)")
    
    if hit_bounds:
        print(f"  ❌ Hit bounds for: {', '.join(hit_bounds)} - This was the problem!")
    else:
        print(f"  ✅ Did not hit bounds")

if __name__ == "__main__":
    test_bounds_fix()
    compare_with_old_bounds()