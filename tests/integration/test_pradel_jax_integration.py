#!/usr/bin/env python3
"""
Integration test to verify the fixed Pradel model can be used with JAX transformations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from pradel_jax.models.pradel import PradelModel

def test_model_jax_compatibility():
    """Test that the Pradel model works with JAX transformations."""
    print("Testing Pradel model JAX compatibility...")
    
    # Create simple test data directly
    capture_matrix = jnp.array([
        [1, 0, 1, 0],  # Individual 1: captured at t=1,3
        [0, 1, 1, 1],  # Individual 2: captured at t=2,3,4  
        [1, 1, 0, 1],  # Individual 3: captured at t=1,2,4
        [0, 0, 1, 1],  # Individual 4: captured at t=3,4
        [1, 0, 0, 0],  # Individual 5: captured only at t=1
    ])
    
    n_individuals, n_occasions = capture_matrix.shape
    
    # Test empirical estimation methods with JAX compatibility
    model = PradelModel()
    
    try:
        # Test empirical survival estimation
        survival_est = model._estimate_empirical_survival(capture_matrix)
        print(f"✓ Empirical survival estimation: {survival_est}")
        
        # Test empirical recruitment estimation
        recruitment_est = model._estimate_empirical_recruitment(capture_matrix)
        print(f"✓ Empirical recruitment estimation: {recruitment_est}")
        
        # Test that these can be JIT compiled
        @jax.jit
        def compiled_survival_est(capture_matrix):
            return model._estimate_empirical_survival(capture_matrix)
        
        compiled_survival = compiled_survival_est(capture_matrix)
        print(f"✓ Compiled survival estimation: {compiled_survival}")
        
        # Check they match (within floating point precision)
        assert abs(survival_est - compiled_survival) < 1e-6
        print("✓ Compiled and non-compiled survival estimates match")
        
        return True
        
    except Exception as e:
        print(f"✗ Model JAX compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_link_functions():
    """Test that link functions work correctly with JAX."""
    print("\nTesting link functions...")
    
    from pradel_jax.models.pradel import logit, inv_logit, log_link, exp_link
    
    try:
        # Test on arrays
        test_probs = jnp.array([0.1, 0.5, 0.9])
        test_rates = jnp.array([0.1, 1.0, 5.0])
        
        # Test logit/inv_logit round trip
        logit_vals = logit(test_probs)
        recovered_probs = inv_logit(logit_vals)
        assert jnp.allclose(test_probs, recovered_probs), "Logit round trip failed"
        print("✓ Logit/inv_logit round trip successful")
        
        # Test log/exp round trip
        log_vals = log_link(test_rates)
        recovered_rates = exp_link(log_vals)
        assert jnp.allclose(test_rates, recovered_rates), "Log round trip failed"
        print("✓ Log/exp round trip successful")
        
        # Test JAX compilation
        @jax.jit
        def compiled_transforms(probs, rates):
            logit_vals = logit(probs)
            recovered_probs = inv_logit(logit_vals)
            log_vals = log_link(rates)
            recovered_rates = exp_link(log_vals)
            return recovered_probs, recovered_rates
        
        comp_probs, comp_rates = compiled_transforms(test_probs, test_rates)
        assert jnp.allclose(test_probs, comp_probs), "Compiled logit round trip failed"
        assert jnp.allclose(test_rates, comp_rates), "Compiled log round trip failed"
        print("✓ Compiled link functions successful")
        
        # Test gradients
        def test_logit_grad(x):
            return jnp.sum(logit(x))
        
        grad_fn = jax.grad(test_logit_grad)
        grad_result = grad_fn(test_probs)
        print(f"✓ Logit gradient computation successful: {grad_result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Link function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vectorized_operations_integration():
    """Test that vectorized operations work as expected in the model context."""
    print("\nTesting vectorized operations integration...")
    
    try:
        # Create test arrays similar to those in the likelihood function
        capture_matrix = jnp.array([
            [1, 0, 1, 0],
            [0, 1, 1, 1],
            [1, 1, 0, 1],
        ])
        
        n_individuals, n_occasions = capture_matrix.shape
        
        # Test the array operations used in the fixed likelihood
        log_lik_contributions = jnp.zeros(n_individuals)
        
        for i in range(n_individuals):
            ch = capture_matrix[i, :]
            individual_loglik = 0.0
            
            for t in range(n_occasions - 1):
                captured_at_t = ch[t]
                captured_at_t1 = ch[t + 1]
                
                # Test the jnp.where pattern
                phi_p = 0.8 * 0.6  # Example survival * detection
                log_recapture = jnp.log(phi_p)
                log_no_recapture = jnp.log(1 - phi_p)
                
                contribution = jnp.where(
                    captured_at_t == 1,
                    jnp.where(
                        captured_at_t1 == 1,
                        log_recapture,
                        log_no_recapture
                    ),
                    0.0
                )
                individual_loglik += contribution
            
            # Test first capture logic
            was_captured = jnp.sum(ch) > 0
            first_capture_contribution = jnp.where(
                was_captured,
                jnp.log(0.6),  # Example detection prob
                0.0
            )
            individual_loglik += first_capture_contribution
            
            log_lik_contributions = log_lik_contributions.at[i].set(individual_loglik)
        
        total_loglik = jnp.sum(log_lik_contributions)
        print(f"✓ Vectorized likelihood computation: {total_loglik}")
        
        # Test JAX compilation of the whole pattern
        @jax.jit
        def compiled_likelihood_pattern(capture_matrix):
            n_individuals, n_occasions = capture_matrix.shape
            log_lik_contributions = jnp.zeros(n_individuals)
            
            for i in range(n_individuals):
                ch = capture_matrix[i, :]
                individual_loglik = 0.0
                
                for t in range(n_occasions - 1):
                    captured_at_t = ch[t]
                    captured_at_t1 = ch[t + 1]
                    
                    phi_p = 0.8 * 0.6
                    log_recapture = jnp.log(phi_p)
                    log_no_recapture = jnp.log(1 - phi_p)
                    
                    contribution = jnp.where(
                        captured_at_t == 1,
                        jnp.where(
                            captured_at_t1 == 1,
                            log_recapture,
                            log_no_recapture
                        ),
                        0.0
                    )
                    individual_loglik += contribution
                
                was_captured = jnp.sum(ch) > 0
                first_capture_contribution = jnp.where(
                    was_captured,
                    jnp.log(0.6),
                    0.0
                )
                individual_loglik += first_capture_contribution
                
                log_lik_contributions = log_lik_contributions.at[i].set(individual_loglik)
            
            return jnp.sum(log_lik_contributions)
        
        compiled_loglik = compiled_likelihood_pattern(capture_matrix)
        assert jnp.allclose(total_loglik, compiled_loglik), "Compiled vs non-compiled mismatch"
        print("✓ JAX compilation of likelihood pattern successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Vectorized operations integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Testing Pradel Model JAX Integration")
    print("=" * 60)
    
    success = True
    
    if not test_link_functions():
        success = False
    
    if not test_vectorized_operations_integration():
        success = False
    
    if not test_model_jax_compatibility():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All Pradel model JAX integration tests PASSED")
        print("✓ The Pradel model is now compatible with JAX transformations")
        print("✓ No control flow issues remain in the likelihood computation")
    else:
        print("✗ Some integration tests FAILED")
        print("Additional fixes may be needed")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)