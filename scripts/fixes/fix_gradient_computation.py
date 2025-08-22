#!/usr/bin/env python3
"""
Fix the gradient computation issue in the Pradel likelihood.

The main issue is with integer powers in JAX - using gamma**first_capture where first_capture 
is an integer can cause gradient computation issues. We need to use jnp.power instead.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
from pradel_jax.models.pradel import calculate_seniority_gamma

@jax.jit
def _pradel_individual_likelihood_fixed(
    capture_history: jnp.ndarray,
    phi: float,
    p: float,
    f: float
) -> float:
    """
    FIXED version of Pradel likelihood with proper gradient computation.
    
    Key fixes:
    1. Use jnp.power instead of ** for integer powers
    2. Ensure all operations are JAX-differentiable
    3. Handle edge cases more carefully
    """
    n_occasions = len(capture_history)
    total_captures = jnp.sum(capture_history)
    
    # Small constant to prevent log(0)
    epsilon = 1e-12
    
    # Calculate derived parameters using Pradel (1996) relationships
    gamma = calculate_seniority_gamma(phi, f)  # γ = φ/(1+f)
    lambda_pop = 1.0 + f                       # λ = 1 + f
    
    def never_captured_likelihood():
        """Fixed never-captured likelihood with proper gradient computation."""
        # Probability of never entering the population during the study
        # FIX: Use jnp.power instead of ** for better gradient computation
        prob_never_enter = 1.0 / jnp.power(lambda_pop, n_occasions - 1)
        
        # Probability of entering but never being detected
        # FIX: Use jnp.power instead of ** for better gradient computation  
        prob_enter_not_detected = (1.0 - prob_never_enter) * jnp.power(1.0 - p, n_occasions)
        
        total_prob = prob_never_enter + prob_enter_not_detected
        return jnp.log(jnp.maximum(total_prob, epsilon))
    
    def captured_likelihood():
        """Fixed captured likelihood with proper gradient computation."""
        # Find first and last capture occasions
        indices = jnp.arange(n_occasions)
        
        # Get first capture occasion
        capture_indices = jnp.where(capture_history == 1, indices, n_occasions)
        first_capture = jnp.min(capture_indices)
        
        # Get last capture occasion
        last_capture_indices = jnp.where(capture_history == 1, indices, -1)
        last_capture = jnp.max(last_capture_indices)
        
        # Initialize log-likelihood
        log_likelihood = 0.0
        
        # Part 1: Probability of first capture at occasion 'first_capture'
        
        # FIX: Use jnp.power for integer powers to ensure proper gradients
        entry_prob = jnp.where(
            first_capture > 0,
            jnp.power(gamma, first_capture),  # FIXED: Use jnp.power instead of **
            1.0
        )
        
        # FIX: Use jnp.power for integer powers
        not_detected_prob = jnp.where(
            first_capture > 0,
            jnp.power(1.0 - p, first_capture),  # FIXED: Use jnp.power instead of **
            1.0
        )
        
        # Probability of detection at first_capture
        detected_prob = p
        
        # Add first capture contribution
        first_capture_contrib = entry_prob * not_detected_prob * detected_prob
        log_likelihood += jnp.log(jnp.maximum(first_capture_contrib, epsilon))
        
        # Part 2: Process occasions between first and last capture
        def process_intermediate_occasion(carry, t):
            running_ll = carry
            
            # Only process occasions after first capture and before/at last
            in_active_period = (t > first_capture) & (t <= last_capture)
            
            survival_contrib = jnp.where(
                in_active_period,
                jnp.log(jnp.maximum(phi, epsilon)),  # Survived to this occasion
                0.0
            )
            
            # Detection contribution (for occasions before last)
            before_last = t < last_capture
            in_detection_period = (t > first_capture) & before_last
            
            captured_at_t = capture_history[t] == 1
            detection_contrib = jnp.where(
                in_detection_period,
                jnp.where(
                    captured_at_t,
                    jnp.log(jnp.maximum(p, epsilon)),        # Detected
                    jnp.log(jnp.maximum(1.0 - p, epsilon))   # Not detected
                ),
                0.0
            )
            
            # For last capture, we know it was detected (no choice probability)
            at_last = t == last_capture
            last_detection_contrib = jnp.where(
                at_last,
                jnp.log(jnp.maximum(p, epsilon)),  # Must be detected at last
                0.0
            )
            
            new_ll = running_ll + survival_contrib + detection_contrib + last_detection_contrib
            return new_ll, new_ll
        
        # Scan over all occasions
        final_ll, _ = jax.lax.scan(
            process_intermediate_occasion,
            log_likelihood,
            jnp.arange(n_occasions)
        )
        
        # Part 3: Probability of not being seen after last capture
        occasions_after_last = n_occasions - 1 - last_capture
        
        # FIX: Use jnp.power for integer powers
        not_available_prob = jnp.where(
            occasions_after_last > 0,
            jnp.power(1.0 - phi * p, occasions_after_last),  # FIXED: Use jnp.power instead of **
            1.0  # No occasions after, so probability is 1
        )
        
        final_ll += jnp.where(
            occasions_after_last > 0,
            jnp.log(jnp.maximum(not_available_prob, epsilon)),
            0.0  # No contribution if no occasions after
        )
        
        return final_ll
    
    # Return appropriate likelihood based on capture status
    return jnp.where(
        total_captures > 0,
        captured_likelihood(),
        never_captured_likelihood()
    )

def test_fixed_likelihood():
    """Test the fixed likelihood function."""
    print("Testing fixed likelihood function")
    print("=" * 40)
    
    # Test the same cases as before
    test_cases = [
        ([1, 0, 1], "Captured at 0 and 2"),
        ([0, 1, 1], "Captured at 1 and 2"),
        ([1, 1, 0], "Captured at 0 and 1"),
        ([0, 0, 0], "Never captured"),
    ]
    
    phi = 0.7
    p = 0.6
    f = 0.1
    
    from pradel_jax.models.pradel import _pradel_individual_likelihood as original_ll
    
    for capture_history, description in test_cases:
        capture_history = jnp.array(capture_history)
        print(f"\nCase: {description}")
        print(f"History: {capture_history}")
        
        # Compare original vs fixed
        ll_original = original_ll(capture_history, phi, p, f)
        ll_fixed = _pradel_individual_likelihood_fixed(capture_history, phi, p, f)
        
        print(f"Original LL: {ll_original:.6f}")
        print(f"Fixed LL: {ll_fixed:.6f}")
        print(f"Difference: {ll_fixed - ll_original:.6f}")
        
        # Test gradients for fixed version
        def ll_f_fixed(f_val):
            return _pradel_individual_likelihood_fixed(capture_history, phi, p, f_val)
        
        grad_f_fixed = jax.grad(ll_f_fixed)(f)
        
        # Finite difference gradient for fixed version
        h = 1e-6
        fd_f_fixed = (ll_f_fixed(f + h) - ll_f_fixed(f - h)) / (2 * h)
        
        print(f"Fixed JAX gradient d/df: {grad_f_fixed:.6f}")
        print(f"Fixed FD gradient d/df: {fd_f_fixed:.6f}")
        print(f"Fixed difference: {grad_f_fixed - fd_f_fixed:.6f}")
        
        if abs(grad_f_fixed - fd_f_fixed) > 1e-4:
            print("❌ Still gradient mismatch!")
        else:
            print("✅ Fixed gradients match")

if __name__ == "__main__":
    test_fixed_likelihood()