#!/usr/bin/env python3
"""
Quick test to verify the tolerance fix for large-scale optimization
"""

import sys
sys.path.append('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
from pradel_jax.optimization.strategy import StrategySelector, ModelCharacteristics, OptimizationConfig

# Create a mock context that represents our large-scale problem
class MockContext:
    def __init__(self, n_individuals, n_parameters, n_occasions):
        self.n_individuals = n_individuals
        self.n_parameters = n_parameters  
        self.n_occasions = n_occasions
        self.capture_matrix = None  # Will be handled in characteristics
        
    def get_condition_estimate(self):
        return None

# Test tolerance selection for different problem sizes
def test_tolerance_selection():
    print("=== TOLERANCE CONFIGURATION TEST ===")
    
    selector = StrategySelector()
    
    # Small dataset (should use default tolerance)
    small_context = MockContext(n_individuals=1000, n_parameters=10, n_occasions=5)
    small_chars = ModelCharacteristics(
        n_parameters=10,
        n_individuals=1000, 
        n_occasions=5,
        parameter_ratio=0.01,
        data_sparsity=0.7
    )
    
    small_config = selector._generate_config("scipy_lbfgs", small_chars)
    print(f"Small dataset (1k individuals): tolerance = {small_config.tolerance}")
    
    # Large dataset (should use relaxed tolerance)
    large_context = MockContext(n_individuals=50000, n_parameters=64, n_occasions=10)
    large_chars = ModelCharacteristics(
        n_parameters=64,
        n_individuals=50000,
        n_occasions=10, 
        parameter_ratio=0.0013,
        data_sparsity=0.8
    )
    
    large_config = selector._generate_config("scipy_lbfgs", large_chars)
    print(f"Large dataset (50k individuals): tolerance = {large_config.tolerance}")
    
    # Very large dataset (Nebraska scale)
    xl_context = MockContext(n_individuals=111697, n_parameters=64, n_occasions=10)
    xl_chars = ModelCharacteristics(
        n_parameters=64,
        n_individuals=111697,
        n_occasions=10,
        parameter_ratio=0.0006,
        data_sparsity=0.85
    )
    
    xl_config = selector._generate_config("scipy_lbfgs", xl_chars)
    print(f"Very large dataset (111k individuals): tolerance = {xl_config.tolerance}")
    
    # Verify that large datasets get more relaxed tolerances (larger tolerance values)
    assert small_config.tolerance <= large_config.tolerance, "Large datasets should have more relaxed (larger) tolerance"
    assert large_config.tolerance <= xl_config.tolerance, "Very large datasets should have most relaxed tolerance"
    
    print("\n✅ Tolerance scaling works correctly!")
    print(f"   Small: {small_config.tolerance:.0e}")
    print(f"   Large: {large_config.tolerance:.0e}")  
    print(f"   X-Large: {xl_config.tolerance:.0e}")
    
if __name__ == "__main__":
    test_tolerance_selection()