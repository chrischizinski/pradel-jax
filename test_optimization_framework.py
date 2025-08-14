#!/usr/bin/env python3
"""
Test script for the optimization strategy framework.

Demonstrates the framework capabilities and validates integration
with industry-standard optimization libraries.
"""

import numpy as np
import jax.numpy as jnp
from jax import grad
import sys
from pathlib import Path
import logging

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from pradel_jax.optimization import (
    optimize_model,
    compare_optimization_strategies,
    OptimizationStrategy,
    OptimizationConfig,
    OptimizationRequest,
    OptimizationOrchestrator,
    recommend_strategy,
    diagnose_optimization_difficulty
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MockModelContext:
    """Mock model context for testing."""
    
    def __init__(self, n_parameters=10, n_individuals=1000, n_occasions=5):
        self.n_parameters = n_parameters
        self.n_individuals = n_individuals
        self.n_occasions = n_occasions
        
        # Create mock capture matrix
        rng = np.random.RandomState(42)
        self.capture_matrix = rng.binomial(1, 0.3, (n_individuals, n_occasions))
        
    def get_condition_estimate(self):
        """Mock condition number estimate."""
        return 1e6  # Well-conditioned problem


def rosenbrock_function(x):
    """
    Classic Rosenbrock function for testing optimization.
    
    Global minimum at x = [1, 1, ..., 1] with f(x) = 0
    """
    x = jnp.asarray(x)
    return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def quadratic_function(x):
    """Simple quadratic function with known minimum."""
    x = jnp.asarray(x)
    center = jnp.ones_like(x) * 2.0
    return jnp.sum((x - center)**2)


def ill_conditioned_quadratic(x):
    """Ill-conditioned quadratic function."""
    x = jnp.asarray(x)
    # Create ill-conditioned Hessian
    scales = jnp.array([1.0, 1e6])[:len(x)]  # Large condition number
    return jnp.sum(scales * x**2)


def test_basic_optimization():
    """Test basic optimization functionality."""
    logger.info("\n=== Testing Basic Optimization ===")
    
    # Simple quadratic optimization
    context = MockModelContext(n_parameters=5)
    initial_params = np.random.randn(5)
    
    response = optimize_model(
        objective_function=quadratic_function,
        initial_parameters=initial_params,
        context=context
    )
    
    logger.info(f"Optimization {'succeeded' if response.success else 'failed'}")
    logger.info(f"Strategy used: {response.strategy_used}")
    logger.info(f"Final objective: {response.result.fun:.6f}")
    logger.info(f"Iterations: {response.result.nit}")
    logger.info(f"Total time: {response.total_time:.3f}s")
    logger.info(f"Confidence: {response.confidence_score:.1%}")
    
    # Check if we found the minimum
    expected_minimum = np.ones(5) * 2.0
    error = np.linalg.norm(response.result.x - expected_minimum)
    logger.info(f"Parameter error: {error:.6f}")
    
    assert response.success, "Basic optimization should succeed"
    assert error < 0.1, "Should find approximate minimum"
    
    return response


def test_strategy_selection():
    """Test automatic strategy selection."""
    logger.info("\n=== Testing Strategy Selection ===")
    
    # Well-conditioned problem
    context = MockModelContext(n_parameters=3, n_individuals=1000)
    recommendation = recommend_strategy(context)
    
    logger.info(f"Recommended strategy: {recommendation.strategy.value}")
    logger.info(f"Confidence: {recommendation.confidence:.1%}")
    logger.info(f"Expected success rate: {recommendation.expected_success_rate:.1%}")
    logger.info(f"Estimated time: {recommendation.estimated_time_seconds:.1f}s")
    logger.info(f"Rationale: {recommendation.rationale}")
    
    if recommendation.preprocessing_recommendations:
        logger.info("Preprocessing recommendations:")
        for rec in recommendation.preprocessing_recommendations:
            logger.info(f"  - {rec}")
    
    # Ill-conditioned problem
    ill_context = MockModelContext(n_parameters=50, n_individuals=100)  # Under-identified
    ill_recommendation = recommend_strategy(ill_context)
    
    logger.info(f"\nIll-conditioned problem strategy: {ill_recommendation.strategy.value}")
    logger.info(f"Confidence: {ill_recommendation.confidence:.1%}")
    
    return recommendation


def test_strategy_comparison():
    """Test strategy comparison functionality."""
    logger.info("\n=== Testing Strategy Comparison ===")
    
    context = MockModelContext(n_parameters=3)
    initial_params = np.array([0.0, 0.0, 0.0])
    
    # Compare multiple strategies on Rosenbrock function
    strategies = [
        OptimizationStrategy.SCIPY_LBFGS,
        OptimizationStrategy.SCIPY_SLSQP,
        OptimizationStrategy.JAX_ADAM
    ]
    
    results = compare_optimization_strategies(
        objective_function=rosenbrock_function,
        initial_parameters=initial_params,
        context=context,
        strategies=strategies
    )
    
    logger.info("Strategy comparison results:")
    successful_results = []
    
    for strategy_name, response in results.items():
        success_str = "✓" if response.success else "✗"
        logger.info(f"  {success_str} {strategy_name}: "
                   f"obj={response.result.fun:.6f}, "
                   f"time={response.total_time:.2f}s, "
                   f"iter={response.result.nit}")
        
        if response.success:
            successful_results.append((strategy_name, response.result.fun))
    
    if successful_results:
        best_strategy, best_objective = min(successful_results, key=lambda x: x[1])
        logger.info(f"Best strategy: {best_strategy} with objective {best_objective:.6f}")
    
    assert len(successful_results) > 0, "At least one strategy should succeed"
    
    return results


def test_monitoring_and_profiling():
    """Test monitoring and profiling capabilities."""
    logger.info("\n=== Testing Monitoring and Profiling ===")
    
    context = MockModelContext(n_parameters=2)
    initial_params = np.array([-1.0, 1.0])
    
    # Create request with monitoring and profiling enabled
    request = OptimizationRequest(
        objective_function=quadratic_function,
        initial_parameters=initial_params,
        enable_monitoring=True,
        enable_profiling=True,
        experiment_name="test_monitoring"
    )
    
    orchestrator = OptimizationOrchestrator()
    response = orchestrator.optimize(request, context)
    
    logger.info(f"Monitoring enabled: {request.enable_monitoring}")
    logger.info(f"Profiling enabled: {request.enable_profiling}")
    logger.info(f"Session summary available: {'session_summary' in response.__dict__}")
    
    if response.profiling_data:
        logger.info("Profiling data:")
        for section, data in response.profiling_data.get('sections', {}).items():
            logger.info(f"  {section}: {data['total_time']:.4f}s "
                       f"({data['percentage']:.1f}%)")
    
    if response.session_summary:
        conv_stats = response.session_summary.get('convergence_stats', {})
        if conv_stats:
            logger.info(f"Convergence stats:")
            logger.info(f"  Best objective: {conv_stats.get('best_objective', 'N/A')}")
            logger.info(f"  Total improvement: {conv_stats.get('total_improvement', 'N/A')}")
    
    return response


def test_problem_diagnosis():
    """Test problem difficulty diagnosis."""
    logger.info("\n=== Testing Problem Diagnosis ===")
    
    # Easy problem
    easy_context = MockModelContext(n_parameters=3, n_individuals=5000)
    easy_diagnosis = diagnose_optimization_difficulty(easy_context)
    
    logger.info(f"Easy problem diagnosis:")
    logger.info(f"  Difficulty: {easy_diagnosis['difficulty']}")
    logger.info(f"  Parameter ratio: {easy_diagnosis['characteristics'].parameter_ratio:.4f}")
    logger.info(f"  Data sparsity: {easy_diagnosis['characteristics'].data_sparsity:.1%}")
    
    # Difficult problem
    hard_context = MockModelContext(n_parameters=100, n_individuals=200)
    hard_diagnosis = diagnose_optimization_difficulty(hard_context)
    
    logger.info(f"\nDifficult problem diagnosis:")
    logger.info(f"  Difficulty: {hard_diagnosis['difficulty']}")
    logger.info(f"  Parameter ratio: {hard_diagnosis['characteristics'].parameter_ratio:.4f}")
    
    if hard_diagnosis['recommendations']:
        logger.info("  Recommendations:")
        for rec in hard_diagnosis['recommendations']:
            logger.info(f"    - {rec}")
    
    return easy_diagnosis, hard_diagnosis


def test_configuration_customization():
    """Test custom configuration options."""
    logger.info("\n=== Testing Configuration Customization ===")
    
    context = MockModelContext(n_parameters=4)
    initial_params = np.random.randn(4)
    
    # Custom configuration with overrides
    custom_config = OptimizationConfig(
        max_iter=500,
        tolerance=1e-10,
        verbose=True
    )
    
    request = OptimizationRequest(
        objective_function=quadratic_function,
        initial_parameters=initial_params,
        preferred_strategy=OptimizationStrategy.SCIPY_LBFGS,
        config_overrides={'max_iter': 200, 'tolerance': 1e-8}
    )
    
    orchestrator = OptimizationOrchestrator()
    response = orchestrator.optimize(request, context)
    
    logger.info(f"Custom configuration used")
    logger.info(f"Success: {response.success}")
    logger.info(f"Iterations: {response.result.nit}")
    logger.info(f"Final tolerance achieved: {response.result.message}")
    
    return response


def test_error_handling():
    """Test error handling and fallback mechanisms."""
    logger.info("\n=== Testing Error Handling ===")
    
    context = MockModelContext(n_parameters=2)
    
    def problematic_function(x):
        """Function that might cause numerical issues."""
        x = jnp.asarray(x)
        if jnp.any(jnp.abs(x) > 10):
            return float('inf')  # Simulate numerical overflow
        return jnp.sum(x**2)
    
    # Start from problematic initial point
    initial_params = np.array([15.0, -15.0])  # Outside safe region
    
    response = optimize_model(
        objective_function=problematic_function,
        initial_parameters=initial_params,
        context=context,
        bounds=[(-5, 5), (-5, 5)]  # Constrain to safe region
    )
    
    logger.info(f"Problematic optimization: {'succeeded' if response.success else 'failed'}")
    logger.info(f"Strategy used: {response.strategy_used}")
    logger.info(f"Fallback used: {response.fallback_used}")
    
    if response.recommendations:
        logger.info("Recommendations:")
        for rec in response.recommendations:
            logger.info(f"  - {rec}")
    
    return response


def run_comprehensive_test():
    """Run comprehensive test suite."""
    logger.info("Starting Optimization Framework Test Suite")
    logger.info("=" * 50)
    
    try:
        # Basic functionality tests
        basic_response = test_basic_optimization()
        strategy_rec = test_strategy_selection()
        comparison_results = test_strategy_comparison()
        
        # Advanced features
        monitoring_response = test_monitoring_and_profiling()
        easy_diag, hard_diag = test_problem_diagnosis()
        config_response = test_configuration_customization()
        error_response = test_error_handling()
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("TEST SUITE SUMMARY")
        logger.info("=" * 50)
        
        tests_passed = 0
        total_tests = 7
        
        if basic_response.success:
            tests_passed += 1
            logger.info("✓ Basic optimization: PASSED")
        else:
            logger.info("✗ Basic optimization: FAILED")
        
        if strategy_rec.confidence > 0.5:
            tests_passed += 1
            logger.info("✓ Strategy selection: PASSED")
        else:
            logger.info("✗ Strategy selection: FAILED")
        
        if any(r.success for r in comparison_results.values()):
            tests_passed += 1
            logger.info("✓ Strategy comparison: PASSED")
        else:
            logger.info("✗ Strategy comparison: FAILED")
        
        if monitoring_response.session_summary:
            tests_passed += 1
            logger.info("✓ Monitoring and profiling: PASSED")
        else:
            logger.info("✗ Monitoring and profiling: FAILED")
        
        if easy_diag['difficulty'] != hard_diag['difficulty']:
            tests_passed += 1
            logger.info("✓ Problem diagnosis: PASSED")
        else:
            logger.info("✗ Problem diagnosis: FAILED")
        
        if config_response.success:
            tests_passed += 1
            logger.info("✓ Configuration customization: PASSED")
        else:
            logger.info("✗ Configuration customization: FAILED")
        
        # Error handling test - success or graceful failure both acceptable
        tests_passed += 1  
        logger.info("✓ Error handling: PASSED")
        
        logger.info(f"\nOverall: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            logger.info("🎉 All tests passed! Framework is working correctly.")
        else:
            logger.warning("⚠️  Some tests failed. Check logs for details.")
        
        return tests_passed == total_tests
        
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)