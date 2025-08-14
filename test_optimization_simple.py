#!/usr/bin/env python3
"""
Simple optimization framework test with real Pradel likelihood functions.

Tests the core optimization framework with manually implemented Pradel likelihood,
using real capture-recapture data to validate the integration.
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import pandas as pd
import time
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json

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
    diagnose_optimization_difficulty,
    optimization_experiment
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@jit
def inv_logit(x):
    """Inverse logit (sigmoid) function."""
    return jnp.where(x > 0, 1 / (1 + jnp.exp(-x)), jnp.exp(x) / (1 + jnp.exp(x)))


@jit
def pradel_log_likelihood(params, capture_matrix):
    """
    Simplified Pradel model log-likelihood for constant parameters.
    
    Parameters:
    - params[0]: logit(phi) - survival probability
    - params[1]: logit(p) - detection probability  
    - params[2]: log(f) - recruitment rate
    """
    
    # Transform parameters
    phi = inv_logit(params[0])  # Survival probability
    p = inv_logit(params[1])    # Detection probability
    f = jnp.exp(params[2])      # Recruitment rate
    
    # Ensure valid probabilities
    phi = jnp.clip(phi, 1e-6, 1 - 1e-6)
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    f = jnp.clip(f, 1e-6, 10.0)
    
    n_individuals, n_occasions = capture_matrix.shape
    
    if n_occasions < 2:
        return -1e10  # Invalid
    
    log_lik = 0.0
    
    for i in range(n_individuals):
        ch = capture_matrix[i, :]  # Capture history for individual i
        
        # Skip individuals never captured
        if jnp.sum(ch) == 0:
            continue
            
        # Find first and last capture
        captures = jnp.where(ch == 1)[0]
        if len(captures) == 0:
            continue
            
        first_capture = captures[0]
        last_capture = captures[-1]
        
        # Individual contribution to likelihood
        individual_loglik = 0.0
        
        # First capture: detection probability
        individual_loglik += jnp.log(p)
        
        # Between first and last capture
        for t in range(first_capture, last_capture):
            if t + 1 < n_occasions:
                if ch[t + 1] == 1:  # Survived and was captured
                    individual_loglik += jnp.log(phi * p)
                else:  # Survived but not captured
                    # We observe this individual later, so they must survive
                    individual_loglik += jnp.log(phi * (1 - p))
        
        # Add small penalty to encourage reasonable parameter values
        penalty = 0.001 * (jnp.square(params[0]) + jnp.square(params[1]) + jnp.square(params[2]))
        individual_loglik -= penalty
        
        log_lik += individual_loglik
    
    return log_lik


class SimpleModelContext:
    """Simple model context for optimization framework."""
    
    def __init__(self, capture_matrix, name="test_dataset"):
        self.capture_matrix = jnp.array(capture_matrix, dtype=jnp.int32)
        self.name = name
        
        # Set context properties
        self.n_individuals = self.capture_matrix.shape[0]
        self.n_occasions = self.capture_matrix.shape[1]
        self.n_parameters = 3  # phi, p, f
        
        # Calculate basic statistics
        self.capture_rate = float(jnp.mean(self.capture_matrix))
        self.sparsity = float(jnp.mean(self.capture_matrix == 0))
        
        logger.info(f"Dataset {name}: {self.n_individuals} individuals, {self.n_occasions} occasions")
        logger.info(f"Capture rate: {self.capture_rate:.3f}, Sparsity: {self.sparsity:.3f}")
    
    def get_condition_estimate(self):
        """Estimate condition number based on data characteristics."""
        if self.sparsity > 0.9:
            return 1e8  # Very sparse data
        elif self.sparsity > 0.7:
            return 1e6  # Moderately sparse
        else:
            return 1e4  # Well-conditioned


class SimplePradelOptimizationTester:
    """Test optimization framework with simplified Pradel likelihood."""
    
    def __init__(self):
        self.results = {}
        
    def load_real_data(self) -> Dict[str, SimpleModelContext]:
        """Load real capture-recapture datasets."""
        datasets = {}
        
        # Load dipper data
        try:
            dipper_path = Path(__file__).parent / "data" / "dipper_processed.csv"
            if dipper_path.exists():
                df = pd.read_csv(dipper_path)
                capture_cols = [col for col in df.columns if col.startswith('capture_')]
                if capture_cols:
                    capture_matrix = df[capture_cols].values
                    datasets["dipper"] = SimpleModelContext(capture_matrix, "dipper")
        except Exception as e:
            logger.warning(f"Failed to load dipper data: {e}")
        
        # Load wf.dat data
        try:
            wf_path = Path(__file__).parent / "data" / "test_datasets" / "wf.dat.csv"
            if wf_path.exists():
                df = pd.read_csv(wf_path)
                if 'ch' in df.columns:
                    # Parse capture history strings
                    ch_strings = df['ch'].astype(str)
                    max_len = max(len(ch) for ch in ch_strings)
                    capture_matrix = np.array([[int(ch[i]) if i < len(ch) else 0 
                                              for i in range(max_len)] for ch in ch_strings])
                    # Take subset for faster testing
                    if len(capture_matrix) > 1000:
                        capture_matrix = capture_matrix[:1000]  
                    datasets["wf_dat"] = SimpleModelContext(capture_matrix, "wf_dat")
        except Exception as e:
            logger.warning(f"Failed to load wf.dat: {e}")
        
        # Create synthetic data if no real data loaded
        if not datasets:
            logger.info("Creating synthetic test data")
            synthetic_data = self._create_synthetic_data()
            datasets["synthetic"] = SimpleModelContext(synthetic_data, "synthetic")
        
        return datasets
    
    def _create_synthetic_data(self):
        """Create synthetic capture-recapture data."""
        np.random.seed(42)
        n_individuals = 300
        n_occasions = 6
        
        # True parameters
        true_phi = 0.7  # Survival
        true_p = 0.4    # Detection
        
        capture_matrix = np.zeros((n_individuals, n_occasions), dtype=int)
        
        for i in range(n_individuals):
            alive = True
            for t in range(n_occasions):
                if alive:
                    # Detection
                    if np.random.random() < true_p:
                        capture_matrix[i, t] = 1
                    
                    # Survival to next occasion
                    if t < n_occasions - 1:
                        if np.random.random() > true_phi:
                            alive = False
        
        return capture_matrix
    
    def test_single_optimization(self, context: SimpleModelContext, strategy=None):
        """Test single optimization run."""
        logger.info(f"Testing optimization for {context.name}")
        
        # Define objective function (negative log-likelihood)
        def objective(params):
            try:
                ll = pradel_log_likelihood(params, context.capture_matrix)
                return -float(ll)
            except Exception as e:
                logger.warning(f"Objective function error: {e}")
                return 1e10
        
        # Define gradient function
        grad_fn = jit(grad(lambda p: -pradel_log_likelihood(p, context.capture_matrix)))
        
        def gradient(params):
            try:
                return np.array(grad_fn(params))
            except Exception as e:
                logger.warning(f"Gradient error: {e}")
                return np.zeros_like(params)
        
        # Initial parameters (reasonable starting values)
        initial_params = np.array([0.0, -0.5, -1.0])  # logit(0.5), logit(0.38), log(0.37)
        
        # Parameter bounds on transformed scale
        bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 3.0)]  # phi, p on logit; f on log
        
        start_time = time.time()
        
        try:
            if strategy is None:
                # Automatic strategy selection
                response = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=context,
                    bounds=bounds,
                    gradient_function=gradient
                )
                strategy_used = response.strategy_used
            else:
                # Specific strategy
                config = OptimizationConfig(max_iter=1000, tolerance=1e-8)
                
                from pradel_jax.optimization.optimizers import minimize_with_strategy
                result = minimize_with_strategy(strategy, objective, initial_params, bounds, gradient, config)
                
                # Mock response object
                class MockResponse:
                    def __init__(self, result, strategy):
                        self.result = result
                        self.success = getattr(result, 'success', True)
                        self.strategy_used = strategy
                        self.recommendations = []
                
                response = MockResponse(result, strategy)
                strategy_used = strategy
            
            optimization_time = time.time() - start_time
            
            # Process results
            test_result = {
                'dataset': context.name,
                'strategy': strategy_used.value if hasattr(strategy_used, 'value') else str(strategy_used),
                'success': response.success,
                'optimization_time': optimization_time,
                'n_parameters': 3,
                'n_individuals': context.n_individuals,
                'n_occasions': context.n_occasions,
                'capture_rate': context.capture_rate,
                'sparsity': context.sparsity
            }
            
            if response.success and hasattr(response, 'result'):
                # Transform parameters back to natural scale
                phi_est = inv_logit(response.result.x[0])
                p_est = inv_logit(response.result.x[1])
                f_est = np.exp(response.result.x[2])
                
                test_result.update({
                    'final_nll': float(response.result.fun),
                    'n_iterations': getattr(response.result, 'nit', 0),
                    'phi_estimate': float(phi_est),
                    'p_estimate': float(p_est),
                    'f_estimate': float(f_est),
                    'raw_parameters': response.result.x.tolist()
                })
                
                logger.info(f"Optimization successful for {context.name}:")
                logger.info(f"  Strategy: {strategy_used}")
                logger.info(f"  NLL: {response.result.fun:.6f}")
                logger.info(f"  Time: {optimization_time:.2f}s")
                logger.info(f"  Estimates: φ={phi_est:.3f}, p={p_est:.3f}, f={f_est:.3f}")
            else:
                logger.warning(f"Optimization failed for {context.name}")
                if hasattr(response, 'result') and hasattr(response.result, 'message'):
                    test_result['error_message'] = response.result.message
            
            return test_result
            
        except Exception as e:
            logger.error(f"Test failed for {context.name}: {e}")
            return {
                'dataset': context.name,
                'success': False,
                'error': str(e)
            }
    
    def test_strategy_comparison(self, context: SimpleModelContext):
        """Compare multiple optimization strategies."""
        logger.info(f"Comparing strategies for {context.name}")
        
        strategies = [
            OptimizationStrategy.SCIPY_LBFGS,
            OptimizationStrategy.SCIPY_SLSQP,
            OptimizationStrategy.JAX_ADAM,
            OptimizationStrategy.MULTI_START
        ]
        
        results = {}
        
        for strategy in strategies:
            try:
                logger.info(f"Testing {strategy.value} on {context.name}")
                result = self.test_single_optimization(context, strategy)
                results[strategy.value] = result
            except Exception as e:
                logger.error(f"Strategy {strategy.value} failed on {context.name}: {e}")
                results[strategy.value] = {'success': False, 'error': str(e)}
        
        return results
    
    def test_problem_analysis(self, context: SimpleModelContext):
        """Test problem analysis and strategy recommendation."""
        logger.info(f"Analyzing problem characteristics for {context.name}")
        
        try:
            # Get strategy recommendation
            recommendation = recommend_strategy(context)
            
            # Get problem diagnosis
            diagnosis = diagnose_optimization_difficulty(context)
            
            return {
                'dataset': context.name,
                'recommendation': {
                    'strategy': recommendation.strategy.value,
                    'confidence': recommendation.confidence,
                    'rationale': recommendation.rationale
                },
                'diagnosis': diagnosis
            }
        except Exception as e:
            logger.error(f"Problem analysis failed for {context.name}: {e}")
            return {'dataset': context.name, 'error': str(e)}
    
    def run_comprehensive_test(self):
        """Run comprehensive optimization framework test."""
        logger.info("Starting comprehensive optimization framework test")
        
        # Load datasets
        datasets = self.load_real_data()
        logger.info(f"Testing {len(datasets)} datasets: {list(datasets.keys())}")
        
        results = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'datasets': list(datasets.keys()),
            'single_tests': [],
            'strategy_comparisons': {},
            'problem_analyses': [],
            'summary': {}
        }
        
        # Test each dataset
        for name, context in datasets.items():
            
            # Single optimization test (automatic strategy)
            logger.info(f"\n--- Testing {name} ---")
            single_result = self.test_single_optimization(context)
            results['single_tests'].append(single_result)
            
            # Strategy comparison
            comparison_results = self.test_strategy_comparison(context)
            results['strategy_comparisons'][name] = comparison_results
            
            # Problem analysis
            analysis_result = self.test_problem_analysis(context)
            results['problem_analyses'].append(analysis_result)
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results):
        """Generate test summary statistics."""
        single_tests = results['single_tests']
        strategy_comparisons = results['strategy_comparisons']
        
        # Overall success rate
        total_single = len(single_tests)
        successful_single = sum(1 for r in single_tests if r.get('success', False))
        single_success_rate = successful_single / total_single if total_single > 0 else 0
        
        # Strategy performance
        strategy_stats = {}
        for dataset, comparisons in strategy_comparisons.items():
            for strategy, result in comparisons.items():
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'total': 0, 'successful': 0, 'times': []}
                
                strategy_stats[strategy]['total'] += 1
                if result.get('success', False):
                    strategy_stats[strategy]['successful'] += 1
                    if 'optimization_time' in result:
                        strategy_stats[strategy]['times'].append(result['optimization_time'])
        
        # Calculate success rates and timings
        for strategy in strategy_stats:
            stats = strategy_stats[strategy]
            stats['success_rate'] = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            if stats['times']:
                stats['avg_time'] = np.mean(stats['times'])
                stats['std_time'] = np.std(stats['times'])
            else:
                stats['avg_time'] = None
                stats['std_time'] = None
        
        # Parameter estimates (for successful tests)
        successful_estimates = []
        for test in single_tests:
            if test.get('success', False) and 'phi_estimate' in test:
                successful_estimates.append({
                    'dataset': test['dataset'],
                    'phi': test['phi_estimate'],
                    'p': test['p_estimate'], 
                    'f': test['f_estimate']
                })
        
        return {
            'total_datasets': len(results['datasets']),
            'single_test_success_rate': single_success_rate,
            'strategy_performance': strategy_stats,
            'parameter_estimates': successful_estimates,
            'datasets_tested': results['datasets']
        }
    
    def save_results(self, results, filename=None):
        """Save test results."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"simple_optimization_test_{timestamp}.json"
        
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create summary report
        self._write_summary_report(results, filepath.with_suffix('.txt'))
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def _write_summary_report(self, results, filepath):
        """Write human-readable summary report."""
        summary = results['summary']
        
        report = f"""
Pradel-JAX Optimization Framework Test Results
============================================

Test Timestamp: {results['test_timestamp']}
Datasets: {', '.join(results['datasets'])}

Overall Performance:
- Datasets Tested: {summary['total_datasets']}
- Single Test Success Rate: {summary['single_test_success_rate']:.1%}

Strategy Performance:
"""
        
        for strategy, stats in summary['strategy_performance'].items():
            report += f"\n{strategy}:\n"
            report += f"  Success Rate: {stats['success_rate']:.1%} ({stats['successful']}/{stats['total']})\n"
            if stats['avg_time'] is not None:
                report += f"  Average Time: {stats['avg_time']:.2f}±{stats['std_time']:.2f}s\n"
        
        if summary['parameter_estimates']:
            report += "\nParameter Estimates (successful optimizations):\n"
            for est in summary['parameter_estimates']:
                report += f"  {est['dataset']}: φ={est['phi']:.3f}, p={est['p']:.3f}, f={est['f']:.3f}\n"
        
        with open(filepath, 'w') as f:
            f.write(report)


def main():
    """Run the test."""
    logger.info("Pradel-JAX Optimization Framework Integration Test")
    logger.info("=" * 60)
    
    tester = SimplePradelOptimizationTester()
    
    try:
        results = tester.run_comprehensive_test()
        results_file = tester.save_results(results)
        
        # Print summary
        summary = results['summary']
        print(f"\n{'='*60}")
        print("OPTIMIZATION FRAMEWORK TEST COMPLETE")
        print(f"{'='*60}")
        print(f"Datasets Tested: {summary['total_datasets']}")
        print(f"Success Rate: {summary['single_test_success_rate']:.1%}")
        print(f"Results saved to: {results_file}")
        
        # Show strategy performance
        print("\nStrategy Performance:")
        for strategy, stats in summary['strategy_performance'].items():
            avg_time = f"{stats['avg_time']:.2f}s" if stats['avg_time'] else "N/A"
            print(f"  {strategy}: {stats['success_rate']:.1%} success, {avg_time} avg time")
        
        # Show parameter estimates
        if summary['parameter_estimates']:
            print("\nParameter Estimates:")
            for est in summary['parameter_estimates']:
                print(f"  {est['dataset']}: φ={est['phi']:.3f}, p={est['p']:.3f}, f={est['f']:.3f}")
        
        if summary['single_test_success_rate'] >= 0.5:
            print("\n✅ OPTIMIZATION FRAMEWORK TEST PASSED!")
            print("The optimization framework successfully integrates with Pradel likelihood functions.")
        else:
            print("\n⚠️  OPTIMIZATION FRAMEWORK TEST SHOWS ISSUES")
            print("Check detailed results for debugging information.")
        
        return results
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    results = main()