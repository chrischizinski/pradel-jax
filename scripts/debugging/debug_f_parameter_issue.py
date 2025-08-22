#!/usr/bin/env python3
"""
Focus specifically on the f parameter gradient issue.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

from pradel_jax.models.pradel import _pradel_individual_likelihood, calculate_seniority_gamma

def test_f_parameter_components():
    """Test how f parameter affects different components of the likelihood."""
    print("Testing f parameter components")
    print("=" * 40)
    
    # Simple test case
    capture_history = jnp.array([1, 0, 1])  # Captured at times 0 and 2
    phi = 0.7
    p = 0.6
    f = 0.1
    
    print(f"Capture history: {capture_history}")
    print(f"Parameters: phi={phi}, p={p}, f={f}")
    
    # Calculate derived parameters
    gamma = calculate_seniority_gamma(phi, f)
    lambda_pop = 1.0 + f
    
    print(f"Derived: gamma={gamma:.6f}, lambda={lambda_pop:.6f}")
    
    # Test gradient of gamma w.r.t. f
    def gamma_func(f_val):
        return calculate_seniority_gamma(phi, f_val)
    
    grad_gamma = jax.grad(gamma_func)(f)
    print(f"d(gamma)/d(f) = {grad_gamma:.6f}")
    
    # Finite difference check
    h = 1e-6
    fd_gamma = (gamma_func(f + h) - gamma_func(f - h)) / (2 * h)
    print(f"FD d(gamma)/d(f) = {fd_gamma:.6f}")
    print(f"Difference: {grad_gamma - fd_gamma:.6f}")
    
    # Now test how gamma is used in the likelihood
    n_occasions = len(capture_history)
    first_capture = 0  # First element is 1
    
    # This is how gamma is used in the entry probability
    entry_prob_formula = lambda f_val: jnp.where(
        first_capture > 0,
        calculate_seniority_gamma(phi, f_val) ** first_capture,
        1.0
    )
    
    entry_prob = entry_prob_formula(f)
    grad_entry_prob = jax.grad(entry_prob_formula)(f)
    print(f"Entry probability: {entry_prob:.6f}")
    print(f"d(entry_prob)/d(f) = {grad_entry_prob:.6f}")
    
    # For never-captured individuals, lambda is used differently
    def never_captured_component(f_val):
        lambda_val = 1.0 + f_val
        prob_never_enter = 1.0 / (lambda_val ** (n_occasions - 1))
        prob_enter_not_detected = (1.0 - prob_never_enter) * ((1.0 - p) ** n_occasions)
        total_prob = prob_never_enter + prob_enter_not_detected
        return jnp.log(jnp.maximum(total_prob, 1e-12))
    
    never_ll = never_captured_component(f)
    grad_never = jax.grad(never_captured_component)(f)
    print(f"Never captured log-likelihood: {never_ll:.6f}")
    print(f"d(never_ll)/d(f) = {grad_never:.6f}")
    
    return True

def test_conditional_branches():
    """Test if the conditional branches are causing gradient issues."""
    print("\nTesting conditional branches")
    print("=" * 40)
    
    # Test different capture patterns
    test_cases = [
        ([1, 0, 1], "Captured at 0 and 2"),
        ([0, 1, 1], "Captured at 1 and 2"),
        ([1, 1, 0], "Captured at 0 and 1"),
        ([0, 0, 0], "Never captured"),
    ]
    
    phi = 0.7
    p = 0.6
    f = 0.1
    
    for capture_history, description in test_cases:
        capture_history = jnp.array(capture_history)
        print(f"\nCase: {description}")
        print(f"History: {capture_history}")
        
        # Compute likelihood
        ll = _pradel_individual_likelihood(capture_history, phi, p, f)
        print(f"Log-likelihood: {ll:.6f}")
        
        # Compute gradient w.r.t. f
        def ll_f(f_val):
            return _pradel_individual_likelihood(capture_history, phi, p, f_val)
        
        grad_f = jax.grad(ll_f)(f)
        
        # Finite difference gradient
        h = 1e-6
        fd_f = (ll_f(f + h) - ll_f(f - h)) / (2 * h)
        
        print(f"JAX gradient d/df: {grad_f:.6f}")
        print(f"FD gradient d/df: {fd_f:.6f}")
        print(f"Difference: {grad_f - fd_f:.6f}")
        
        if abs(grad_f - fd_f) > 1e-4:
            print("❌ Gradient mismatch!")
        else:
            print("✅ Gradients match")

def test_problematic_operations():
    """Test specific JAX operations that might cause gradient issues."""
    print("\nTesting problematic operations")
    print("=" * 40)
    
    f = 0.1
    phi = 0.7
    
    # Test power operations with JAX
    def test_power(f_val):
        lambda_val = 1.0 + f_val
        # This is used in never_captured_likelihood
        return 1.0 / (lambda_val ** 2)  # n_occasions - 1 = 2
    
    result = test_power(f)
    grad_power = jax.grad(test_power)(f)
    
    # Finite difference
    h = 1e-6
    fd_power = (test_power(f + h) - test_power(f - h)) / (2 * h)
    
    print(f"Power operation: 1/lambda^2 = {result:.6f}")
    print(f"JAX gradient: {grad_power:.6f}")
    print(f"FD gradient: {fd_power:.6f}")
    print(f"Difference: {grad_power - fd_power:.6f}")
    
    # Test gamma power operation
    def test_gamma_power(f_val):
        gamma = calculate_seniority_gamma(phi, f_val)
        return gamma ** 0  # This should be 1 always, so gradient should be 0
    
    result_gamma = test_gamma_power(f)
    grad_gamma_power = jax.grad(test_gamma_power)(f)
    
    print(f"\nGamma^0 operation: {result_gamma:.6f}")
    print(f"JAX gradient (should be 0): {grad_gamma_power:.6f}")
    
    if abs(grad_gamma_power) > 1e-10:
        print("❌ Gradient should be zero but isn't!")
    else:
        print("✅ Gradient is correctly zero")

if __name__ == "__main__":
    test_f_parameter_components()
    test_conditional_branches()
    test_problematic_operations()