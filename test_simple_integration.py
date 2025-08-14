#!/usr/bin/env python3
"""
Simple test to verify optimization framework integration works.
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path
import logging

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from pradel_jax.optimization import optimize_model
from pradel_jax.formulas.spec import FormulaSpec, ParameterFormula, ParameterType
from pradel_jax.data.adapters import DataContext

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SimpleDataContext(DataContext):
    """Simple data context for testing."""
    
    def __init__(self, n_individuals=100, n_occasions=5, n_parameters=3):
        self.n_individuals = n_individuals
        self.n_occasions = n_occasions
        self.n_parameters = n_parameters  # Add required attribute
        
        # Create simple capture matrix
        np.random.seed(42)
        self.capture_matrix = jnp.array(
            np.random.binomial(1, 0.3, (n_individuals, n_occasions)),
            dtype=jnp.int32
        )
        
        # Simple covariates
        self.covariates = {
            'sex': jnp.array(np.random.choice([0, 1], n_individuals), dtype=jnp.int32)
        }
    
    def get_condition_estimate(self):
        """Mock condition number estimate."""
        return 1e6


def simple_objective_function(params):
    """Simple quadratic objective function for testing."""
    params = jnp.asarray(params)
    # Simple quadratic with minimum at [1, 1, 1]
    target = jnp.ones_like(params)
    return jnp.sum((params - target)**2)


def test_simple_optimization():
    """Test basic optimization without complex model."""
    logger.info("Testing simple optimization")
    
    context = SimpleDataContext()
    initial_params = jnp.array([0.0, 0.0, 0.0])
    
    response = optimize_model(
        objective_function=simple_objective_function,
        initial_parameters=initial_params,
        context=context
    )
    
    logger.info(f"Success: {response.success}")
    logger.info(f"Strategy: {response.strategy_used}")
    logger.info(f"Final params: {response.result.x}")
    logger.info(f"Final objective: {response.result.fun:.6f}")
    
    return response


def test_formula_creation():
    """Test formula creation without model integration."""
    logger.info("Testing formula creation")
    
    try:
        # Test simple constant formula
        phi_formula = ParameterFormula(
            parameter=ParameterType.PHI, 
            formula_string="1"
        )
        p_formula = ParameterFormula(
            parameter=ParameterType.P, 
            formula_string="1"
        )
        f_formula = ParameterFormula(
            parameter=ParameterType.F, 
            formula_string="1"
        )
        
        spec = FormulaSpec(phi=phi_formula, p=p_formula, f=f_formula)
        logger.info(f"Created formula spec: {spec}")
        return True
        
    except Exception as e:
        logger.error(f"Formula creation failed: {e}")
        return False


def main():
    """Run simple integration tests."""
    logger.info("Simple Integration Test")
    logger.info("=" * 40)
    
    # Test 1: Formula creation
    formula_success = test_formula_creation()
    
    # Test 2: Simple optimization
    try:
        response = test_simple_optimization()
        optimization_success = response.success
    except Exception as e:
        logger.error(f"Simple optimization failed: {e}")
        optimization_success = False
    
    # Summary
    logger.info("\n" + "=" * 40)
    logger.info("Test Results:")
    logger.info(f"Formula creation: {'✓' if formula_success else '✗'}")
    logger.info(f"Simple optimization: {'✓' if optimization_success else '✗'}")
    
    overall_success = formula_success and optimization_success
    logger.info(f"Overall: {'PASS' if overall_success else 'FAIL'}")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)