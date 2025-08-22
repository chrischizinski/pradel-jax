#!/usr/bin/env python3
"""
Fix the optimization by adjusting parameter bounds and testing with better settings.
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

def test_with_better_bounds():
    """Test optimization with more reasonable bounds."""
    print("Testing optimization with better bounds")
    print("=" * 50)
    
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
    
    # Get initial parameters
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    print(f"Initial parameters: {initial_params}")
    
    phi_init = inv_logit(initial_params[0])
    p_init = inv_logit(initial_params[1])
    f_init = exp_link(initial_params[2])
    print(f"Initial natural scale: phi={phi_init:.3f}, p={p_init:.3f}, f={f_init:.3f}")
    
    # Define objective and gradient functions
    def objective(params):
        return -float(model.log_likelihood(jnp.array(params), data_context, design_matrices))
    
    def gradient(params):
        grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
        return -np.array(grad_fn(jnp.array(params)))
    
    # BETTER BOUNDS: More reasonable parameter ranges
    # For logit(phi) and logit(p): allow probabilities from 0.01 to 0.99
    # For log(f): allow recruitment from 0.001 to 1.0
    better_bounds = [
        (logit(0.01), logit(0.99)),  # phi: 0.01 to 0.99
        (logit(0.01), logit(0.99)),  # p: 0.01 to 0.99  
        (log_link(0.001), log_link(1.0)),  # f: 0.001 to 1.0
    ]
    
    print(f"\nBounds (transformed scale):")
    for i, (lower, upper) in enumerate(better_bounds):
        param_name = ['phi', 'p', 'f'][i]
        if param_name in ['phi', 'p']:
            nat_lower = inv_logit(lower)
            nat_upper = inv_logit(upper)
        else:  # f
            nat_lower = exp_link(lower)
            nat_upper = exp_link(upper)
        print(f"  {param_name}: [{lower:.3f}, {upper:.3f}] -> natural: [{nat_lower:.3f}, {nat_upper:.3f}]")
    
    # Try L-BFGS-B with better bounds
    result = minimize(
        objective,
        initial_params,
        method='L-BFGS-B',
        jac=gradient,
        bounds=better_bounds,
        options={'disp': False, 'maxiter': 200, 'ftol': 1e-9}
    )
    
    print(f"\nOptimization result:")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Final objective: {result.fun:.6f}")
    print(f"  Final log-likelihood: {-result.fun:.6f}")
    print(f"  Message: {result.message}")
    
    print(f"\nFinal parameters (transformed): {result.x}")
    
    # Convert to natural scale
    phi_final = inv_logit(result.x[0])
    p_final = inv_logit(result.x[1])
    f_final = exp_link(result.x[2])
    print(f"Final parameters (natural): phi={phi_final:.4f}, p={p_final:.4f}, f={f_final:.4f}")
    
    # Calculate AIC
    n_params = len(result.x)
    aic = 2 * n_params - 2 * (-result.fun)
    print(f"AIC: {aic:.2f}")
    
    return result

def test_different_optimizers():
    """Test different optimization methods."""
    print(f"\n" + "="*50)
    print("Testing different optimization methods")
    print("="*50)
    
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
    
    def objective(params):
        return -float(model.log_likelihood(jnp.array(params), data_context, design_matrices))
    
    def gradient(params):
        grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
        return -np.array(grad_fn(jnp.array(params)))
    
    # Better bounds
    bounds = [
        (logit(0.01), logit(0.99)),
        (logit(0.01), logit(0.99)),  
        (log_link(0.001), log_link(1.0)),
    ]
    
    methods = ['L-BFGS-B', 'SLSQP']
    
    for method in methods:
        print(f"\nTesting {method}:")
        
        try:
            if method == 'SLSQP':
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    jac=gradient,
                    bounds=bounds,
                    options={'disp': False, 'maxiter': 200, 'ftol': 1e-9}
                )
            else:  # L-BFGS-B
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    jac=gradient,
                    bounds=bounds,
                    options={'disp': False, 'maxiter': 200, 'ftol': 1e-9}
                )
            
            phi_final = inv_logit(result.x[0])
            p_final = inv_logit(result.x[1])
            f_final = exp_link(result.x[2])
            
            print(f"  Success: {result.success}")
            print(f"  Iterations: {result.nit}")
            print(f"  Final LL: {-result.fun:.6f}")
            print(f"  Parameters: phi={phi_final:.4f}, p={p_final:.4f}, f={f_final:.4f}")
            
        except Exception as e:
            print(f"  Failed: {e}")

def compare_with_pradel_jax_api():
    """Compare with the high-level pradel-jax API."""
    print(f"\n" + "="*50)
    print("Comparing with pradel-jax high-level API")
    print("="*50)
    
    data_context = create_test_data()
    
    # Use the high-level API
    try:
        result = pj.fit_model(
            model=pj.PradelModel(),
            formula=pj.create_formula_spec(phi="~1", p="~1", f="~1"),
            data=data_context
        )
        
        print(f"High-level API result:")
        print(f"  Success: {result.success}")
        print(f"  Strategy used: {result.strategy_used}")
        print(f"  Log-likelihood: {result.log_likelihood:.6f}")
        print(f"  AIC: {result.aic:.2f}")
        
        # Extract parameters if available
        if hasattr(result, 'parameters') and result.parameters is not None:
            params = result.parameters
            print(f"  Parameters: {params}")
        
    except Exception as e:
        print(f"High-level API failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_better_bounds()
    test_different_optimizers()
    compare_with_pradel_jax_api()