#!/usr/bin/env python3
"""
Test the fixed optimization with the updated PradelModel bounds.
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
from pradel_jax.optimization import optimize_model
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

def test_fixed_optimization():
    """Test optimization with the fixed PradelModel bounds."""
    print("Testing optimization with fixed PradelModel bounds")
    print("=" * 55)
    
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
    
    # Check the new bounds
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    print("New parameter bounds:")
    param_names = ['phi', 'p', 'f']
    for i, (lower, upper) in enumerate(bounds):
        param_name = param_names[i]
        if param_name in ['phi', 'p']:
            nat_lower = inv_logit(lower)
            nat_upper = inv_logit(upper)
        else:  # f
            nat_lower = exp_link(lower)
            nat_upper = exp_link(upper)
        print(f"  {param_name}: [{lower:.3f}, {upper:.3f}] -> natural: [{nat_lower:.4f}, {nat_upper:.4f}]")
    
    # Test optimization using the optimization framework
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    
    def objective_function(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    print(f"\nInitial parameters: {initial_params}")
    phi_init = inv_logit(initial_params[0])
    p_init = inv_logit(initial_params[1])
    f_init = exp_link(initial_params[2])
    print(f"Initial natural scale: phi={phi_init:.4f}, p={p_init:.4f}, f={f_init:.4f}")
    
    # Test with the optimization framework
    result = optimize_model(
        objective_function=objective_function,
        initial_parameters=initial_params,
        context=data_context,
        bounds=bounds,
        preferred_strategy="scipy_lbfgs"
    )
    
    print(f"\nOptimization result:")
    print(f"  Success: {result.success}")
    print(f"  Strategy used: {result.strategy}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Function evaluations: {result.function_evaluations}")
    print(f"  Final objective: {result.objective:.6f}")
    print(f"  Final log-likelihood: {-result.objective:.6f}")
    
    if result.success and result.parameters is not None:
        phi_final = inv_logit(result.parameters[0])
        p_final = inv_logit(result.parameters[1])
        f_final = exp_link(result.parameters[2])
        print(f"  Final parameters (natural): phi={phi_final:.4f}, p={p_final:.4f}, f={f_final:.4f}")
        
        # Calculate AIC
        n_params = len(result.parameters)
        aic = 2 * n_params - 2 * (-result.objective)
        print(f"  AIC: {aic:.2f}")
    
    return result

def test_multiple_strategies():
    """Test multiple optimization strategies with the fixed bounds."""
    print(f"\n" + "="*55)
    print("Testing multiple optimization strategies")
    print("="*55)
    
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
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    def objective_function(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    # Test different strategies
    strategies = ["scipy_lbfgs", "scipy_slsqp", "multi_start"]
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        try:
            result = optimize_model(
                objective_function=objective_function,
                initial_parameters=initial_params,
                context=data_context,
                bounds=bounds,
                preferred_strategy=strategy
            )
            
            if result.success and result.parameters is not None:
                phi_final = inv_logit(result.parameters[0])
                p_final = inv_logit(result.parameters[1])
                f_final = exp_link(result.parameters[2])
                
                print(f"  ✅ Success: {result.success}")
                print(f"  Iterations: {result.iterations}")
                print(f"  Log-likelihood: {-result.objective:.6f}")
                print(f"  Parameters: phi={phi_final:.4f}, p={p_final:.4f}, f={f_final:.4f}")
            else:
                print(f"  ❌ Failed: {result.message if hasattr(result, 'message') else 'Unknown error'}")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")

if __name__ == "__main__":
    test_fixed_optimization()
    test_multiple_strategies()