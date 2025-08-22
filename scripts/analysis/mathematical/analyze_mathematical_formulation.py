#!/usr/bin/env python3
"""
Analyze the mathematical formulation of the Pradel likelihood to identify the gradient issue.

Focus on understanding the exact mathematical relationship between f and the likelihood,
and check if the formulation itself has issues.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

from pradel_jax.models.pradel import calculate_seniority_gamma

def analytical_gradient_captured_case():
    """
    Analyze the gradient for a captured individual analytically.
    
    For individual captured at times 1 and 2 (history [0,1,1]):
    - First capture at t=1
    - Last capture at t=2
    """
    print("Analytical gradient analysis for captured case [0,1,1]")
    print("=" * 60)
    
    phi = 0.7
    p = 0.6
    f = 0.1
    
    gamma = calculate_seniority_gamma(phi, f)  # φ/(1+f)
    
    print(f"Parameters: phi={phi}, p={p}, f={f}")
    print(f"Gamma = phi/(1+f) = {gamma:.6f}")
    
    # For this case:
    # first_capture = 1
    # last_capture = 2
    
    # Part 1: First capture contribution
    # entry_prob = gamma^1 = gamma  (since first_capture = 1)
    # not_detected_prob = (1-p)^1 = (1-p)  (not detected at t=0)
    # detected_prob = p  (detected at t=1)
    
    entry_prob = gamma  # gamma^1
    not_detected_prob = 1.0 - p  # (1-p)^1
    detected_prob = p
    
    first_capture_contrib = entry_prob * not_detected_prob * detected_prob
    
    print(f"\nFirst capture contribution:")
    print(f"  entry_prob = gamma^1 = {entry_prob:.6f}")
    print(f"  not_detected_prob = (1-p)^1 = {not_detected_prob:.6f}")
    print(f"  detected_prob = p = {detected_prob:.6f}")
    print(f"  Product = {first_capture_contrib:.6f}")
    print(f"  Log contribution = {np.log(first_capture_contrib):.6f}")
    
    # Part 2: Survival from t=1 to t=2
    # Since individual is captured at both t=1 and t=2, they survived
    survival_contrib = np.log(phi)
    print(f"\nSurvival contribution:")
    print(f"  log(phi) = {survival_contrib:.6f}")
    
    # Part 3: Detection at t=2 (last capture)
    # This is automatic since it's the last capture
    last_detection_contrib = np.log(p)
    print(f"\nLast detection contribution:")
    print(f"  log(p) = {last_detection_contrib:.6f}")
    
    # Part 4: Not seen after t=2
    # occasions_after_last = 3 - 1 - 2 = 0, so no contribution
    
    total_ll = np.log(first_capture_contrib) + survival_contrib + last_detection_contrib
    print(f"\nTotal log-likelihood = {total_ll:.6f}")
    
    # Now calculate analytical gradient w.r.t. f
    # d/df of log(entry_prob * not_detected_prob * detected_prob) + log(phi) + log(p)
    # = d/df [log(entry_prob)] + 0 + 0
    # = d/df [log(gamma)] 
    # = (1/gamma) * d/df[gamma]
    # = (1/gamma) * d/df[phi/(1+f)]
    # = (1/gamma) * phi * d/df[1/(1+f)]
    # = (1/gamma) * phi * (-1/(1+f)^2) * 1
    # = (1/gamma) * phi * (-1/(1+f)^2)
    
    dgamma_df = -phi / ((1 + f) ** 2)  # Derivative of phi/(1+f) w.r.t. f
    analytical_grad = (1/gamma) * dgamma_df
    
    print(f"\nAnalytical gradient calculation:")
    print(f"  d(gamma)/df = -phi/(1+f)^2 = {dgamma_df:.6f}")
    print(f"  d(log(gamma))/df = (1/gamma) * d(gamma)/df = {analytical_grad:.6f}")
    
    return analytical_grad

def compare_with_jax_implementation():
    """Compare analytical gradient with JAX implementation."""
    print(f"\n" + "="*60)
    print("Comparison with JAX implementation")
    print("="*60)
    
    from pradel_jax.models.pradel import _pradel_individual_likelihood
    
    capture_history = jnp.array([0, 1, 1])
    phi = 0.7
    p = 0.6
    f = 0.1
    
    # Get analytical gradient
    analytical_grad = analytical_gradient_captured_case()
    
    # Get JAX gradient
    def ll_f(f_val):
        return _pradel_individual_likelihood(capture_history, phi, p, f_val)
    
    jax_grad = jax.grad(ll_f)(f)
    
    # Get finite difference gradient  
    h = 1e-6
    fd_grad = (ll_f(f + h) - ll_f(f - h)) / (2 * h)
    
    print(f"\nGradient comparison:")
    print(f"  Analytical: {analytical_grad:.6f}")
    print(f"  JAX:        {jax_grad:.6f}")
    print(f"  FD:         {fd_grad:.6f}")
    
    print(f"\nDifferences:")
    print(f"  JAX - Analytical: {jax_grad - analytical_grad:.6f}")
    print(f"  FD - Analytical:  {fd_grad - analytical_grad:.6f}")
    print(f"  JAX - FD:         {jax_grad - fd_grad:.6f}")

def test_gamma_derivative():
    """Test the gamma derivative calculation in isolation."""
    print(f"\n" + "="*60)
    print("Testing gamma derivative in isolation")
    print("="*60)
    
    phi = 0.7
    f = 0.1
    
    def gamma_func(f_val):
        return calculate_seniority_gamma(phi, f_val)
    
    # JAX gradient
    jax_grad_gamma = jax.grad(gamma_func)(f)
    
    # Analytical gradient
    analytical_grad_gamma = -phi / ((1 + f) ** 2)
    
    # Finite difference
    h = 1e-6
    fd_grad_gamma = (gamma_func(f + h) - gamma_func(f - h)) / (2 * h)
    
    print(f"Gamma = phi/(1+f) = {gamma_func(f):.6f}")
    print(f"Analytical d(gamma)/df = -phi/(1+f)^2 = {analytical_grad_gamma:.6f}")
    print(f"JAX d(gamma)/df = {jax_grad_gamma:.6f}")
    print(f"FD d(gamma)/df = {fd_grad_gamma:.6f}")
    
    print(f"\nDifferences:")
    print(f"  JAX - Analytical: {jax_grad_gamma - analytical_grad_gamma:.6f}")
    print(f"  FD - Analytical:  {fd_grad_gamma - analytical_grad_gamma:.6f}")

def examine_scan_operation():
    """Examine if the jax.lax.scan operation is causing issues."""
    print(f"\n" + "="*60)
    print("Examining JAX scan operation")
    print("="*60)
    
    # Simplified version without scan
    capture_history = jnp.array([0, 1, 1])
    phi = 0.7
    p = 0.6
    f = 0.1
    
    gamma = calculate_seniority_gamma(phi, f)
    
    # Manual computation without scan
    def manual_likelihood(f_val):
        gamma_val = calculate_seniority_gamma(phi, f_val)
        
        # First capture at t=1
        entry_prob = gamma_val  # gamma^1
        not_detected_prob = 1.0 - p  # (1-p)^1
        detected_prob = p
        
        first_contrib = jnp.log(entry_prob * not_detected_prob * detected_prob)
        
        # Survival from t=1 to t=2
        survival_contrib = jnp.log(phi)
        
        # Detection at t=2
        last_detection_contrib = jnp.log(p)
        
        return first_contrib + survival_contrib + last_detection_contrib
    
    # Test gradients
    manual_grad = jax.grad(manual_likelihood)(f)
    
    # Compare with full implementation
    from pradel_jax.models.pradel import _pradel_individual_likelihood
    
    def full_ll(f_val):
        return _pradel_individual_likelihood(capture_history, phi, p, f_val)
    
    full_grad = jax.grad(full_ll)(f)
    
    print(f"Manual implementation gradient: {manual_grad:.6f}")
    print(f"Full implementation gradient:   {full_grad:.6f}")
    print(f"Difference: {full_grad - manual_grad:.6f}")
    
    # Test likelihood values
    manual_ll = manual_likelihood(f)
    full_ll_val = full_ll(f)
    
    print(f"\nLikelihood values:")
    print(f"Manual: {manual_ll:.6f}")
    print(f"Full:   {full_ll_val:.6f}")
    print(f"Difference: {full_ll_val - manual_ll:.6f}")

if __name__ == "__main__":
    analytical_gradient_captured_case()
    compare_with_jax_implementation()
    test_gamma_derivative()
    examine_scan_operation()