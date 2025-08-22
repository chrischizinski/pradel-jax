#!/usr/bin/env python3
"""
Debug gradient computation issues in the Pradel likelihood.

Check if JAX is computing gradients correctly and if they're being used by optimizers.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel, logit, log_link, inv_logit, exp_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec
import tempfile
import os

def test_gradient_computation():
    """Test if JAX gradients are being computed correctly."""
    print("JAX Gradient Computation Test")
    print("=" * 40)
    
    # Generate test data
    np.random.seed(42)
    encounter_data = np.zeros((50, 5), dtype=int)
    
    for i in range(50):
        alive = True
        for t in range(5):
            if alive:
                if np.random.binomial(1, 0.6):
                    encounter_data[i, t] = 1
                if t < 4:
                    alive = np.random.binomial(1, 0.75)
    
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
    
    print(f"Created model with {data_context.n_individuals} individuals")
    
    # Create a simple likelihood function wrapper for JAX
    def likelihood_for_jax(params):
        """Simple likelihood wrapper for JAX gradient computation."""
        return model.log_likelihood(params, data_context, design_matrices)
    
    # Test at different parameter values
    test_params = [
        jnp.array([logit(0.6), logit(0.5), log_link(0.1)]),  # Lower values
        jnp.array([logit(0.75), logit(0.6), log_link(0.15)]), # Target values
        jnp.array([logit(0.9), logit(0.7), log_link(0.2)]),  # Higher values
    ]
    
    # Compile gradient function
    grad_fn = jax.grad(likelihood_for_jax)
    
    print("\nTesting likelihood and gradients at different points:")
    print("=" * 60)
    
    for i, params in enumerate(test_params):
        phi_nat = inv_logit(params[0])
        p_nat = inv_logit(params[1])
        f_nat = exp_link(params[2])
        
        print(f"\nPoint {i+1}: phi={phi_nat:.3f}, p={p_nat:.3f}, f={f_nat:.3f}")
        print(f"Parameters (transformed): {params}")
        
        try:
            # Compute likelihood
            ll = likelihood_for_jax(params)
            print(f"Log-likelihood: {ll:.6f}")
            
            # Compute gradients
            gradients = grad_fn(params)
            print(f"Gradients: {gradients}")
            print(f"Gradient magnitudes: {np.abs(gradients)}")
            print(f"Max gradient magnitude: {np.max(np.abs(gradients)):.6f}")
            
            # Check if gradients are effectively zero
            if np.max(np.abs(gradients)) < 1e-6:
                print("❌ GRADIENTS ARE ESSENTIALLY ZERO - This explains optimizer convergence issues!")
            else:
                print("✅ Gradients are non-zero")
                
        except Exception as e:
            print(f"❌ Error computing gradients: {e}")
    
    return True

def test_finite_differences():
    """Test finite difference gradients for comparison."""
    print(f"\n" + "=" * 40)
    print("Finite Difference Gradient Test")
    print("=" * 40)
    
    # Use same setup as above
    np.random.seed(42)
    encounter_data = np.zeros((20, 5), dtype=int)  # Even smaller for FD test
    
    for i in range(20):
        alive = True
        for t in range(5):
            if alive:
                if np.random.binomial(1, 0.6):
                    encounter_data[i, t] = 1
                if t < 4:
                    alive = np.random.binomial(1, 0.75)
    
    # Create model
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
    
    # Test at one parameter point
    params = jnp.array([logit(0.75), logit(0.6), log_link(0.15)])
    
    def objective(p):
        return float(model.log_likelihood(jnp.array(p), data_context, design_matrices))
    
    # Compute finite difference gradients
    h = 1e-6
    fd_gradients = np.zeros_like(params)
    
    base_ll = objective(params)
    print(f"Base likelihood: {base_ll:.6f}")
    
    for i in range(len(params)):
        params_plus = params.at[i].add(h)
        params_minus = params.at[i].add(-h)
        
        ll_plus = objective(params_plus)
        ll_minus = objective(params_minus)
        
        fd_gradients[i] = (ll_plus - ll_minus) / (2 * h)
        
        print(f"Parameter {i}: LL+h={ll_plus:.6f}, LL-h={ll_minus:.6f}, FD grad={fd_gradients[i]:.6f}")
    
    print(f"\nFinite difference gradients: {fd_gradients}")
    print(f"FD gradient magnitudes: {np.abs(fd_gradients)}")
    
    # Compare with JAX gradients
    try:
        jax_grad = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))(params)
        print(f"JAX gradients: {jax_grad}")
        print(f"Difference (JAX - FD): {jax_grad - fd_gradients}")
        
        # Check if they're close
        if np.allclose(jax_grad, fd_gradients, atol=1e-4):
            print("✅ JAX and finite difference gradients match!")
        else:
            print("❌ JAX and finite difference gradients differ!")
            
    except Exception as e:
        print(f"❌ Error computing JAX gradients: {e}")

if __name__ == "__main__":
    test_gradient_computation()
    test_finite_differences()