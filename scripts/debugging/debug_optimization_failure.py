#!/usr/bin/env python3
"""
Debug optimization failure in f parameter investigation.
"""

import sys
import numpy as np
import jax.numpy as jnp
import pandas as pd

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel, logit, log_link, inv_logit, exp_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec
from pradel_jax.optimization.orchestrator import optimize_model
import tempfile
import os

def simple_test():
    """Simple test to isolate the optimization failure."""
    print("Simple Optimization Failure Debug")
    print("=" * 40)
    
    # Use simple synthetic data
    np.random.seed(42)
    
    # Generate simple test data (known to work from earlier tests)
    encounter_data = np.zeros((100, 5), dtype=int)
    
    for i in range(100):
        alive = True
        for t in range(5):
            if alive:
                # Detection with prob 0.6
                if np.random.binomial(1, 0.6):
                    encounter_data[i, t] = 1
                
                # Survival with prob 0.75
                if t < 4:
                    alive = np.random.binomial(1, 0.75)
    
    print(f"Generated simple data: {encounter_data.shape}")
    print(f"Captured individuals: {np.sum(np.sum(encounter_data, axis=1) > 0)}")
    
    # Create data context
    df_data = []
    for i, history in enumerate(encounter_data):
        ch = ''.join(map(str, history))
        df_data.append({'individual_id': i, 'ch': ch})
    
    df = pd.DataFrame(df_data)
    
    # Save and load
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        data_context = pj.load_data(temp_file.name)
        print(f"Data context loaded: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return False
    finally:
        os.unlink(temp_file.name)
    
    # Create model
    try:
        parser = FormulaParser()
        formula_spec = FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )
        print("Formula spec created successfully")
        
        model = PradelModel()
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        print("Design matrices built successfully")
        
        # Test likelihood evaluation at reasonable parameters
        phi_logit = logit(0.75)
        p_logit = logit(0.6)
        f_log = log_link(0.15)
        test_params = jnp.array([phi_logit, p_logit, f_log])
        
        ll = model.log_likelihood(test_params, data_context, design_matrices)
        print(f"Likelihood at test parameters: {ll:.4f}")
        
        if np.isnan(ll) or np.isinf(ll):
            print("ERROR: Likelihood is NaN or Inf!")
            return False
        
    except Exception as e:
        print(f"ERROR in model setup: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test optimization with detailed error reporting
    try:
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                if np.isnan(ll) or np.isinf(ll):
                    return 1e10  # Large penalty for invalid likelihood
                return -ll
            except Exception as e:
                print(f"ERROR in objective function: {e}")
                return 1e10
        
        initial_parameters = model.get_initial_parameters(data_context, design_matrices)
        print(f"Initial parameters: {initial_parameters}")
        
        # Test objective at initial parameters
        initial_obj = objective(initial_parameters)
        print(f"Initial objective value: {initial_obj}")
        
        if initial_obj >= 1e10:
            print("ERROR: Invalid initial objective value!")
            return False
        
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        print(f"Parameter bounds: {bounds}")
        
        # Try optimization
        result = optimize_model(
            objective_function=objective,
            initial_parameters=initial_parameters,
            context=data_context,
            bounds=bounds
        )
        
        print(f"Optimization result:")
        print(f"  Success: {result.success}")
        print(f"  Strategy: {result.strategy_used}")
        
        if hasattr(result, 'message'):
            print(f"  Message: {result.message}")
        
        if hasattr(result, 'x'):
            print(f"  Final parameters: {result.x}")
            
            # Convert to natural scale
            phi_est = inv_logit(result.x[0])
            p_est = inv_logit(result.x[1])
            f_est = exp_link(result.x[2])
            
            print(f"  Natural scale: phi={phi_est:.4f}, p={p_est:.4f}, f={f_est:.4f}")
        
        return result.success
        
    except Exception as e:
        print(f"ERROR in optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")