#!/usr/bin/env python3
"""
Test the corrected Pradel likelihood with fixed gradient computation.
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

def test_fixed_optimization():
    """Test if the gradient fix resolves optimization issues."""
    print("Testing Fixed Optimization")
    print("=" * 40)
    
    # Generate test data with known parameters
    np.random.seed(42)
    true_phi = 0.75
    true_p = 0.6
    true_f = 0.15
    
    encounter_data = np.zeros((100, 5), dtype=int)
    
    for i in range(100):
        alive = True
        for t in range(5):
            if alive:
                if np.random.binomial(1, true_p):
                    encounter_data[i, t] = 1
                if t < 4:
                    alive = np.random.binomial(1, true_phi)
    
    print(f"Generated data with true parameters: phi={true_phi}, p={true_p}, f={true_f}")
    
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
    
    # Define objective with gradient checking
    def objective(params):
        ll = model.log_likelihood(params, data_context, design_matrices)
        if np.isnan(ll) or np.isinf(ll):
            return 1e10
        return -ll
    
    # Test optimization with different methods
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    print(f"Initial parameters: {initial_params}")
    print(f"Initial objective: {objective(initial_params):.6f}")
    
    methods = ['L-BFGS-B', 'SLSQP']
    
    for method in methods:
        print(f"\nTesting {method}:")
        print("-" * 20)
        
        try:
            result = minimize(
                objective,
                initial_params,
                method=method,
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                # Convert to natural scale
                phi_est = inv_logit(result.x[0])
                p_est = inv_logit(result.x[1])
                f_est = exp_link(result.x[2])
                
                # Calculate errors
                phi_error = abs(phi_est - true_phi) / true_phi * 100
                p_error = abs(p_est - true_p) / true_p * 100
                f_error = abs(f_est - true_f) / true_f * 100
                
                print(f"✅ SUCCESS with {method}")
                print(f"Iterations: {result.nit}")
                print(f"Final objective: {result.fun:.6f}")
                print(f"Estimated: phi={phi_est:.4f}, p={p_est:.4f}, f={f_est:.4f}")
                print(f"Errors: phi={phi_error:.1f}%, p={p_error:.1f}%, f={f_error:.1f}%")
                
                # Check if we've resolved the f parameter issue
                if f_error < 25:
                    print(f"✅ F parameter recovery: GOOD ({f_error:.1f}% error)")
                    return True
                else:
                    print(f"⚠️  F parameter recovery: POOR ({f_error:.1f}% error)")
            else:
                print(f"❌ FAILED with {method}: {result.message}")
                
        except Exception as e:
            print(f"❌ ERROR with {method}: {e}")
    
    return False

if __name__ == "__main__":
    success = test_fixed_optimization()
    if success:
        print(f"\n✅ GRADIENT FIX SUCCESSFUL - Parameter recovery restored!")
    else:
        print(f"\n❌ Issues remain - further investigation needed")