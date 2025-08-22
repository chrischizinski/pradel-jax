#!/usr/bin/env python3
"""
Test different finite difference step sizes to understand why FD is inaccurate.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

from pradel_jax.models.pradel import _pradel_individual_likelihood

def test_fd_step_sizes():
    """Test different finite difference step sizes."""
    print("Testing different finite difference step sizes")
    print("=" * 50)
    
    capture_history = jnp.array([0, 1, 1])
    phi = 0.7
    p = 0.6
    f = 0.1
    
    def ll_f(f_val):
        return float(_pradel_individual_likelihood(capture_history, phi, p, f_val))
    
    # Get JAX gradient (our "truth")
    jax_grad = jax.grad(lambda f_val: _pradel_individual_likelihood(capture_history, phi, p, f_val))(f)
    print(f"JAX gradient (truth): {jax_grad:.8f}")
    
    # Test different step sizes
    step_sizes = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    
    for h in step_sizes:
        # Central difference
        fd_grad = (ll_f(f + h) - ll_f(f - h)) / (2 * h)
        
        # Forward difference
        fd_grad_forward = (ll_f(f + h) - ll_f(f)) / h
        
        error_central = abs(fd_grad - jax_grad)
        error_forward = abs(fd_grad_forward - jax_grad)
        
        print(f"h={h:9.0e}: Central={fd_grad:9.6f} (err={error_central:.6f}), Forward={fd_grad_forward:9.6f} (err={error_forward:.6f})")

def examine_function_behavior():
    """Examine how the likelihood function behaves around f=0.1."""
    print("\nExamining function behavior around f=0.1")
    print("=" * 50)
    
    capture_history = jnp.array([0, 1, 1])
    phi = 0.7
    p = 0.6
    f_center = 0.1
    
    def ll_f(f_val):
        return float(_pradel_individual_likelihood(capture_history, phi, p, f_val))
    
    # Test function values around f=0.1
    f_values = np.linspace(f_center - 0.01, f_center + 0.01, 21)
    
    print("f_value     likelihood    f_diff      ll_diff")
    print("-" * 50)
    
    base_ll = ll_f(f_center)
    
    for f_val in f_values:
        ll_val = ll_f(f_val)
        f_diff = f_val - f_center
        ll_diff = ll_val - base_ll
        print(f"{f_val:7.4f}   {ll_val:9.6f}   {f_diff:8.4f}   {ll_diff:9.6f}")

def test_epsilon_effect():
    """Test if the epsilon value is affecting gradients."""
    print("\nTesting epsilon effect on gradients")
    print("=" * 40)
    
    capture_history = jnp.array([0, 1, 1])
    phi = 0.7
    p = 0.6
    f = 0.1
    
    # Create version with different epsilon values
    @jax.jit
    def ll_with_epsilon(f_val, eps):
        """Likelihood with configurable epsilon."""
        from pradel_jax.models.pradel import calculate_seniority_gamma
        
        gamma = calculate_seniority_gamma(phi, f_val)
        
        # First capture contribution
        entry_prob = gamma  # gamma^1 since first_capture = 1
        not_detected_prob = 1.0 - p
        detected_prob = p
        
        first_contrib = jnp.log(jnp.maximum(entry_prob * not_detected_prob * detected_prob, eps))
        
        # Other contributions
        survival_contrib = jnp.log(jnp.maximum(phi, eps))
        detection_contrib = jnp.log(jnp.maximum(p, eps))
        
        return first_contrib + survival_contrib + detection_contrib
    
    # Test different epsilon values
    epsilons = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6]
    
    for eps in epsilons:
        grad_eps = jax.grad(lambda f_val: ll_with_epsilon(f_val, eps))(f)
        
        # Finite difference with same epsilon
        h = 1e-6
        fd_eps = (ll_with_epsilon(f + h, eps) - ll_with_epsilon(f - h, eps)) / (2 * h)
        
        print(f"eps={eps:8.0e}: JAX={grad_eps:.6f}, FD={fd_eps:.6f}, diff={abs(grad_eps - fd_eps):.6f}")

if __name__ == "__main__":
    test_fd_step_sizes()
    examine_function_behavior()
    test_epsilon_effect()