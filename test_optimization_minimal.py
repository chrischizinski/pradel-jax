#!/usr/bin/env python3
"""
Minimal test to validate optimization framework integration with Pradel models.
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path
import logging

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from pradel_jax.optimization import optimize_model
from pradel_jax.models import PradelModel
from pradel_jax.data.adapters import DataContext
from pradel_jax.formulas.parser import create_simple_spec

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MinimalDataContext(DataContext):
    """Minimal data context for testing."""
    
    def __init__(self, n_individuals=50, n_occasions=4):
        self.n_individuals = n_individuals
        self.n_occasions = n_occasions
        self.n_parameters = 3  # phi, p, f
        
        # Create simple capture matrix
        np.random.seed(42)
        self.capture_matrix = jnp.array(
            np.random.binomial(1, 0.4, (n_individuals, n_occasions)),
            dtype=jnp.int32
        )
        
        # Simple covariates (numeric values for sex)
        self.covariates = {
            'sex': jnp.array(np.random.choice([0, 1], n_individuals), dtype=jnp.float32)
        }
    
    def get_condition_estimate(self):
        """Mock condition number estimate."""
        return 1e5


def test_pradel_optimization():
    """Test Pradel model optimization integration."""
    logger.info("Testing Pradel model optimization integration")
    
    # Create data context
    data_context = MinimalDataContext(n_individuals=50, n_occasions=4)
    
    # Create simple formula spec
    formula_spec = create_simple_spec("~1", "~1", "~1", "Constant model")
    
    # Create model
    model = PradelModel()
    
    try:
        # Validate data
        model.validate_data(data_context)
        logger.info("✓ Data validation passed")
        
        # Build design matrices
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        logger.info(f"✓ Design matrices built: {len(design_matrices)} parameters")
        
        # Get initial parameters
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        logger.info(f"✓ Initial parameters: {len(initial_params)} values")
        
        # Get bounds
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        logger.info(f"✓ Parameter bounds: {len(bounds)} constraints")
        
        # Define objective function
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                return -float(ll)  # Minimize negative log-likelihood
            except Exception as e:
                logger.warning(f"Objective function error: {e}")
                return 1e10
        
        # Test objective function
        initial_obj = objective(initial_params)
        logger.info(f"✓ Initial objective value: {initial_obj:.6f}")
        
        # Run optimization
        response = optimize_model(
            objective_function=objective,
            initial_parameters=initial_params,
            context=data_context,
            bounds=bounds
        )
        
        if response.success:
            logger.info(f"✓ Optimization succeeded!")
            logger.info(f"  Strategy used: {response.strategy_used}")
            logger.info(f"  Final objective: {response.result.fun:.6f}")
            logger.info(f"  Optimization time: {response.total_time:.2f}s")
            logger.info(f"  Parameters: {response.result.x}")
            return True
        else:
            logger.error(f"✗ Optimization failed")
            if hasattr(response, 'result') and hasattr(response.result, 'message'):
                logger.error(f"  Error: {response.result.message}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run minimal integration test."""
    logger.info("Pradel-JAX Optimization Framework - Minimal Integration Test")
    logger.info("=" * 60)
    
    success = test_pradel_optimization()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 INTEGRATION TEST PASSED!")
        logger.info("The optimization framework is successfully integrated with Pradel models.")
    else:
        logger.info("❌ INTEGRATION TEST FAILED!")
        logger.info("There are issues with the optimization framework integration.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)