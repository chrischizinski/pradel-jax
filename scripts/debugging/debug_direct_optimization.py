#!/usr/bin/env python3
"""
Test direct optimization bypassing the orchestrator to isolate issues.
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

def test_direct_scipy():
    """Test direct scipy optimization bypassing orchestrator."""
    print("Direct SciPy Optimization Test")
    print("=" * 40)
    
    # Generate simple test data (same as before)
    np.random.seed(42)
    encounter_data = np.zeros((100, 5), dtype=int)
    
    for i in range(100):
        alive = True
        for t in range(5):
            if alive:
                if np.random.binomial(1, 0.6):
                    encounter_data[i, t] = 1
                if t < 4:
                    alive = np.random.binomial(1, 0.75)
    
    print(f"Generated data: {encounter_data.shape}")
    
    # Create data context
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
    
    # Create model
    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )
    
    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    print("Model and matrices created successfully")
    
    # Define objective function with error handling
    def objective(params):
        try:
            ll = model.log_likelihood(params, data_context, design_matrices)
            if np.isnan(ll) or np.isinf(ll):
                return 1e10
            return -ll  # Minimize negative log-likelihood
        except Exception as e:
            print(f"Error in objective: {e}")
            return 1e10
    
    # Test with different starting points and optimizers
    methods = ['L-BFGS-B', 'SLSQP', 'TNC']
    
    for method in methods:
        print(f"\nTesting {method}:")
        print("-" * 20)
        
        # Get initial parameters
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        print(f"Initial params: {initial_params}")
        print(f"Bounds: {bounds}")
        
        # Test objective at initial point
        initial_obj = objective(initial_params)
        print(f"Initial objective: {initial_obj:.6f}")
        
        if initial_obj >= 1e10:
            print("❌ Invalid initial objective")
            continue
        
        # Try optimization
        try:
            result = minimize(
                objective,
                initial_params,
                method=method,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                # Convert to natural scale
                phi_est = inv_logit(result.x[0])
                p_est = inv_logit(result.x[1]) 
                f_est = exp_link(result.x[2])
                
                print(f"✅ SUCCESS with {method}")
                print(f"Final objective: {result.fun:.6f}")
                print(f"Iterations: {result.nit}")
                print(f"Estimates: phi={phi_est:.4f}, p={p_est:.4f}, f={f_est:.4f}")
                
                # Expected: phi≈0.75, p≈0.6, f≈0.15
                phi_error = abs(phi_est - 0.75) / 0.75 * 100
                p_error = abs(p_est - 0.6) / 0.6 * 100  
                f_error = abs(f_est - 0.15) / 0.15 * 100
                
                print(f"Errors: phi={phi_error:.1f}%, p={p_error:.1f}%, f={f_error:.1f}%")
                
            else:
                print(f"❌ FAILED with {method}")
                print(f"Message: {result.message}")
                print(f"Final objective: {result.fun:.6f}")
                
        except Exception as e:
            print(f"❌ ERROR with {method}: {e}")
    
    return True

def test_manual_gradient_check():
    """Test if gradients are working properly."""
    print(f"\n" + "=" * 40)
    print("Manual Gradient Check")
    print("=" * 40)
    
    # Use same data setup as above
    np.random.seed(42)
    encounter_data = np.zeros((50, 5), dtype=int)  # Smaller for faster testing
    
    for i in range(50):
        alive = True
        for t in range(5):
            if alive:
                if np.random.binomial(1, 0.6):
                    encounter_data[i, t] = 1
                if t < 4:
                    alive = np.random.binomial(1, 0.75)
    
    # Create model setup
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
    
    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )
    
    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    # Test likelihood at different parameter values
    test_points = [
        (logit(0.7), logit(0.5), log_link(0.1)),
        (logit(0.75), logit(0.6), log_link(0.15)),
        (logit(0.8), logit(0.7), log_link(0.2)),
    ]
    
    print("Testing likelihood at different parameter values:")
    for i, (phi_logit, p_logit, f_log) in enumerate(test_points):
        params = jnp.array([phi_logit, p_logit, f_log])
        
        try:
            ll = model.log_likelihood(params, data_context, design_matrices)
            phi_nat = inv_logit(phi_logit)
            p_nat = inv_logit(p_logit) 
            f_nat = exp_link(f_log)
            
            print(f"Point {i+1}: phi={phi_nat:.3f}, p={p_nat:.3f}, f={f_nat:.3f} → LL={ll:.4f}")
            
            if np.isnan(ll) or np.isinf(ll):
                print("  ❌ Invalid likelihood!")
            
        except Exception as e:
            print(f"Point {i+1}: ERROR - {e}")

if __name__ == "__main__":
    test_direct_scipy()
    test_manual_gradient_check()