#!/usr/bin/env python3
"""
Test script to verify JAX control flow fixes in the Pradel model.

This script tests that the likelihood function can be properly traced and 
compiled by JAX without control flow errors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from pradel_jax.models.pradel import PradelModel
from pradel_jax.formulas.spec import FormulaSpec, ParameterFormula
from pradel_jax.formulas.design_matrix import DesignMatrixInfo
from pradel_jax.data.adapters import DataContext

def create_test_data():
    """Create simple test data for JAX tracing tests."""
    # Simple capture matrix: 5 individuals, 4 occasions
    capture_matrix = jnp.array([
        [1, 0, 1, 0],  # Individual 1: captured at t=1,3
        [0, 1, 1, 1],  # Individual 2: captured at t=2,3,4  
        [1, 1, 0, 1],  # Individual 3: captured at t=1,2,4
        [0, 0, 1, 1],  # Individual 4: captured at t=3,4
        [1, 0, 0, 0],  # Individual 5: captured only at t=1
    ])
    
    n_individuals, n_occasions = capture_matrix.shape
    
    # Create simple covariates (just intercept for this test)
    covariates = {
        "intercept": jnp.ones(n_individuals)
    }
    
    # Create covariate info for the intercept
    from pradel_jax.data.adapters import CovariateInfo
    covariate_info = {
        "intercept": CovariateInfo(
            name="intercept",
            dtype="float64",
            is_time_varying=False,
            is_categorical=False
        )
    }
    
    data_context = DataContext(
        capture_matrix=capture_matrix,
        covariates=covariates,
        covariate_info=covariate_info,
        n_individuals=n_individuals,
        n_occasions=n_occasions
    )
    
    return data_context

def create_test_formula():
    """Create simple formula specification."""
    # Simple intercept-only model for all parameters
    from pradel_jax.formulas.spec import ParameterType
    
    phi_formula = ParameterFormula(parameter=ParameterType.PHI, formula_string="1")  # Survival
    p_formula = ParameterFormula(parameter=ParameterType.P, formula_string="1")    # Detection  
    f_formula = ParameterFormula(parameter=ParameterType.F, formula_string="1")    # Recruitment
    
    return FormulaSpec(phi=phi_formula, p=p_formula, f=f_formula)

def create_test_design_matrices(data_context):
    """Create simple design matrices for testing."""
    n_individuals = data_context.n_individuals
    
    # Simple intercept-only design matrices
    design_matrix = jnp.ones((n_individuals, 1))
    
    design_matrices = {
        "phi": DesignMatrixInfo(
            matrix=design_matrix,
            column_names=["intercept"],
            parameter_count=1,
            has_intercept=True
        ),
        "p": DesignMatrixInfo(
            matrix=design_matrix,
            column_names=["intercept"], 
            parameter_count=1,
            has_intercept=True
        ),
        "f": DesignMatrixInfo(
            matrix=design_matrix,
            column_names=["intercept"],
            parameter_count=1,
            has_intercept=True
        )
    }
    
    return design_matrices

def test_likelihood_compilation():
    """Test that the likelihood function can be JAX compiled."""
    print("Testing JAX compilation of likelihood function...")
    
    # Create test data
    data_context = create_test_data()
    formula_spec = create_test_formula()
    design_matrices = create_test_design_matrices(data_context)
    
    # Create model
    model = PradelModel()
    
    # Get initial parameters
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    print(f"Initial parameters shape: {initial_params.shape}")
    print(f"Initial parameters: {initial_params}")
    
    # Test basic likelihood computation
    try:
        loglik = model.log_likelihood(initial_params, data_context, design_matrices)
        print(f"✓ Basic likelihood computation successful: {loglik}")
    except Exception as e:
        print(f"✗ Basic likelihood computation failed: {e}")
        return False
    
    # Test JAX compilation with jit
    try:
        @jax.jit
        def compiled_likelihood(params):
            return model.log_likelihood(params, data_context, design_matrices)
        
        # Test compilation
        loglik_compiled = compiled_likelihood(initial_params)
        print(f"✓ JAX compilation successful: {loglik_compiled}")
        
        # Verify results match
        if jnp.allclose(loglik, loglik_compiled):
            print("✓ Compiled and non-compiled results match")
        else:
            print(f"✗ Results differ: {loglik} vs {loglik_compiled}")
            return False
            
    except Exception as e:
        print(f"✗ JAX compilation failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False
    
    # Test gradient computation 
    try:
        @jax.jit
        def likelihood_and_grad(params):
            return jax.value_and_grad(lambda p: model.log_likelihood(p, data_context, design_matrices))(params)
        
        loglik_grad, grad = likelihood_and_grad(initial_params)
        print(f"✓ Gradient computation successful")
        print(f"  Likelihood: {loglik_grad}")
        print(f"  Gradient shape: {grad.shape}")
        print(f"  Gradient norm: {jnp.linalg.norm(grad)}")
        
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        return False
    
    return True

def test_vectorized_operations():
    """Test that vectorized operations work correctly."""
    print("\nTesting vectorized operations...")
    
    # Test jnp.where operations with different array shapes
    test_arrays = [
        jnp.array([1, 0, 1]),
        jnp.array([[1, 0], [0, 1]]),
        jnp.array([[[1, 0, 1], [0, 1, 0]]])
    ]
    
    for i, arr in enumerate(test_arrays):
        try:
            # Test basic where operation
            result = jnp.where(arr == 1, 0.9, 0.1)
            print(f"✓ jnp.where test {i+1} passed, shape: {result.shape}")
        except Exception as e:
            print(f"✗ jnp.where test {i+1} failed: {e}")
            return False
    
    return True

def main():
    """Run all JAX control flow tests."""
    print("=" * 60)
    print("Testing JAX Control Flow Fixes in Pradel Model")
    print("=" * 60)
    
    success = True
    
    # Test vectorized operations
    if not test_vectorized_operations():
        success = False
    
    # Test likelihood compilation
    if not test_likelihood_compilation():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All JAX control flow tests PASSED")
        print("The fixes successfully resolved JAX tracing issues")
    else:
        print("✗ Some JAX control flow tests FAILED")
        print("Additional debugging may be needed")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)