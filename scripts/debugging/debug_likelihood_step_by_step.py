#!/usr/bin/env python3
"""
Step-by-step debugging of the Pradel likelihood function.

This will help us understand exactly where the gradient computation is failing.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel, logit, log_link, inv_logit, exp_link, _pradel_individual_likelihood, _pradel_vectorized_likelihood
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec
import tempfile
import os

def create_simple_test_data():
    """Create very simple test data for debugging."""
    # Just 3 individuals, 3 occasions
    np.random.seed(123)
    encounter_data = np.array([
        [1, 0, 1],  # Individual 1: captured at 1 and 3
        [0, 1, 1],  # Individual 2: captured at 2 and 3  
        [1, 1, 0],  # Individual 3: captured at 1 and 2
    ])
    
    # Create DataFrame
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
    
    return data_context, encounter_data

def test_individual_likelihood():
    """Test individual likelihood computation step by step."""
    print("Testing individual likelihood computation")
    print("=" * 50)
    
    # Simple test case: individual captured at occasions 0 and 2
    capture_history = jnp.array([1, 0, 1])
    phi = 0.7  # Survival
    p = 0.6    # Detection
    f = 0.1    # Recruitment
    
    print(f"Capture history: {capture_history}")
    print(f"Parameters: phi={phi}, p={p}, f={f}")
    
    # Test the individual likelihood function directly
    ll = _pradel_individual_likelihood(capture_history, phi, p, f)
    print(f"Individual log-likelihood: {ll}")
    
    # Test gradient w.r.t. each parameter
    def ll_wrt_phi(phi_val):
        return _pradel_individual_likelihood(capture_history, phi_val, p, f)
    
    def ll_wrt_p(p_val):
        return _pradel_individual_likelihood(capture_history, phi, p_val, f)
    
    def ll_wrt_f(f_val):
        return _pradel_individual_likelihood(capture_history, phi, p, f_val)
    
    # Compute gradients
    grad_phi = jax.grad(ll_wrt_phi)(phi)
    grad_p = jax.grad(ll_wrt_p)(p)
    grad_f = jax.grad(ll_wrt_f)(f)
    
    print(f"Gradients: d/dphi={grad_phi:.6f}, d/dp={grad_p:.6f}, d/df={grad_f:.6f}")
    
    # Test finite differences
    h = 1e-6
    fd_grad_phi = (ll_wrt_phi(phi + h) - ll_wrt_phi(phi - h)) / (2 * h)
    fd_grad_p = (ll_wrt_p(p + h) - ll_wrt_p(p - h)) / (2 * h)
    fd_grad_f = (ll_wrt_f(f + h) - ll_wrt_f(f - h)) / (2 * h)
    
    print(f"FD gradients: d/dphi={fd_grad_phi:.6f}, d/dp={fd_grad_p:.6f}, d/df={fd_grad_f:.6f}")
    print(f"Differences: phi={grad_phi - fd_grad_phi:.6f}, p={grad_p - fd_grad_p:.6f}, f={grad_f - fd_grad_f:.6f}")
    
    return True

def test_vectorized_likelihood():
    """Test vectorized likelihood and its gradients."""
    print("\nTesting vectorized likelihood computation")
    print("=" * 50)
    
    # Create test data
    data_context, encounter_data = create_simple_test_data()
    capture_matrix = jnp.array(encounter_data)
    
    # Test parameters - constant across individuals for simple intercept model
    n_individuals = 3
    phi_values = jnp.array([0.7, 0.7, 0.7])  # Same for all individuals
    p_values = jnp.array([0.6, 0.6, 0.6])    # Same for all individuals
    f_values = jnp.array([0.1, 0.1, 0.1])    # Same for all individuals
    
    print(f"Capture matrix shape: {capture_matrix.shape}")
    print(f"Capture matrix:\n{capture_matrix}")
    print(f"Parameter arrays: phi={phi_values}, p={p_values}, f={f_values}")
    
    # Test vectorized likelihood
    total_ll = _pradel_vectorized_likelihood(phi_values, p_values, f_values, capture_matrix)
    print(f"Total log-likelihood: {total_ll}")
    
    # Compare with sum of individual likelihoods
    individual_lls = []
    for i in range(n_individuals):
        ll = _pradel_individual_likelihood(capture_matrix[i], phi_values[i], p_values[i], f_values[i])
        individual_lls.append(ll)
        print(f"Individual {i} log-likelihood: {ll}")
    
    sum_individual = sum(individual_lls)
    print(f"Sum of individual likelihoods: {sum_individual}")
    print(f"Difference: {total_ll - sum_individual:.10f}")
    
    # Test gradients of vectorized function
    def vec_ll_phi(phi_arr):
        return _pradel_vectorized_likelihood(phi_arr, p_values, f_values, capture_matrix)
    
    def vec_ll_p(p_arr):
        return _pradel_vectorized_likelihood(phi_values, p_arr, f_values, capture_matrix)
    
    def vec_ll_f(f_arr):
        return _pradel_vectorized_likelihood(phi_values, p_values, f_arr, capture_matrix)
    
    grad_phi_vec = jax.grad(vec_ll_phi)(phi_values)
    grad_p_vec = jax.grad(vec_ll_p)(p_values)
    grad_f_vec = jax.grad(vec_ll_f)(f_values)
    
    print(f"Vectorized gradients:")
    print(f"  d/dphi: {grad_phi_vec}")
    print(f"  d/dp: {grad_p_vec}")
    print(f"  d/df: {grad_f_vec}")
    
    return True

def test_full_model_likelihood():
    """Test the full model likelihood including parameter transformation."""
    print("\nTesting full model likelihood with parameter transformation")
    print("=" * 50)
    
    # Create test data and model
    data_context, encounter_data = create_simple_test_data()
    
    parser = FormulaParser()
    formula_spec = FormulaSpec(
        phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
        p=parser.create_parameter_formula(ParameterType.P, "~1"),
        f=parser.create_parameter_formula(ParameterType.F, "~1")
    )
    
    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    print("Design matrices:")
    for param, info in design_matrices.items():
        print(f"  {param}: shape={info.matrix.shape}, has_intercept={info.has_intercept}")
        print(f"    Matrix:\n{info.matrix}")
    
    # Test parameters on natural scale
    phi_logit = logit(0.7)
    p_logit = logit(0.6)
    f_log = log_link(0.1)
    
    parameters = jnp.array([phi_logit, p_logit, f_log])
    print(f"Parameters (transformed): {parameters}")
    print(f"Parameters (natural): phi={inv_logit(phi_logit):.3f}, p={inv_logit(p_logit):.3f}, f={exp_link(f_log):.3f}")
    
    # Test full model likelihood
    ll = model.log_likelihood(parameters, data_context, design_matrices)
    print(f"Full model log-likelihood: {ll}")
    
    # Test gradients
    grad_fn = jax.grad(model.log_likelihood, argnums=0)  # Gradient w.r.t. first argument (parameters)
    gradients = grad_fn(parameters, data_context, design_matrices)
    print(f"Full model gradients: {gradients}")
    
    # Test finite differences on full model
    def full_model_ll(params):
        return float(model.log_likelihood(params, data_context, design_matrices))
    
    h = 1e-6
    fd_gradients = np.zeros_like(parameters)
    base_ll = full_model_ll(parameters)
    
    for i in range(len(parameters)):
        params_plus = parameters.at[i].add(h)
        params_minus = parameters.at[i].add(-h)
        
        ll_plus = full_model_ll(params_plus)
        ll_minus = full_model_ll(params_minus)
        
        fd_gradients[i] = (ll_plus - ll_minus) / (2 * h)
        print(f"Parameter {i}: base={base_ll:.6f}, +h={ll_plus:.6f}, -h={ll_minus:.6f}")
    
    print(f"Finite difference gradients: {fd_gradients}")
    print(f"JAX gradients: {gradients}")
    print(f"Gradient differences: {gradients - fd_gradients}")
    
    # Check if gradients are close
    if np.allclose(gradients, fd_gradients, atol=1e-4):
        print("✅ JAX and finite difference gradients match!")
    else:
        print("❌ JAX and finite difference gradients differ!")
        # Identify which parameters have problematic gradients
        for i, (jax_grad, fd_grad) in enumerate(zip(gradients, fd_gradients)):
            diff = abs(jax_grad - fd_grad)
            if diff > 1e-4:
                print(f"  Parameter {i}: JAX={jax_grad:.6f}, FD={fd_grad:.6f}, diff={diff:.6f}")
    
    return True

if __name__ == "__main__":
    test_individual_likelihood()
    test_vectorized_likelihood()
    test_full_model_likelihood()