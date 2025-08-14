#!/usr/bin/env python3
"""
Optimization Framework Demo

Demonstrates how to use the pradel-jax optimization framework for
capture-recapture model fitting with automatic strategy selection,
monitoring, and experiment tracking.
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pradel_jax.optimization import (
    optimize_model,
    compare_optimization_strategies,
    OptimizationStrategy,
    OptimizationRequest,
    OptimizationOrchestrator,
    optimization_experiment,
    recommend_strategy
)


class ExampleModelContext:
    """Example model context for capture-recapture data."""
    
    def __init__(self, n_individuals=2000, n_occasions=6, n_parameters=8):
        self.n_individuals = n_individuals
        self.n_occasions = n_occasions 
        self.n_parameters = n_parameters
        
        # Simulate realistic capture-recapture data
        key = random.PRNGKey(42)
        
        # True parameters (survival=0.8, detection=0.4, recruitment=0.1)
        self.true_phi = 0.8  # survival
        self.true_p = 0.4    # detection
        self.true_f = 0.1    # recruitment
        
        # Generate capture histories
        self.capture_matrix = self._simulate_capture_histories(key)
        
        print(f"Generated {n_individuals} individuals over {n_occasions} occasions")
        print(f"Data sparsity: {jnp.mean(self.capture_matrix == 0):.1%}")
        
    def _simulate_capture_histories(self, key):
        """Simulate realistic capture histories."""
        key1, key2, key3 = random.split(key, 3)
        
        # Simple simulation - each individual has probability of capture
        capture_probs = random.uniform(key1, (self.n_individuals, self.n_occasions), 
                                     minval=0.2, maxval=0.6)
        
        captures = random.bernoulli(key2, capture_probs)
        
        # Ensure some individuals are never captured (realistic)
        never_captured = random.bernoulli(key3, 0.1, (self.n_individuals,))
        captures = captures * (1 - never_captured[:, None])
        
        return captures.astype(jnp.float32)
    
    def get_condition_estimate(self):
        """Estimate condition number of problem."""
        return 1e5  # Moderately conditioned


def pradel_log_likelihood(params, context):
    """
    Simplified Pradel model log-likelihood for demonstration.
    
    In practice, this would be the full likelihood from the Pradel model.
    """
    # Split parameters (intercepts only for simplicity)
    phi_logit = params[0]  # Survival logit
    p_logit = params[1]    # Detection logit  
    f_log = params[2]      # Recruitment log
    
    # Transform to natural scale
    phi = jax.nn.sigmoid(phi_logit)
    p = jax.nn.sigmoid(p_logit)
    f = jnp.exp(f_log)
    
    # Simple likelihood calculation (simplified for demo)
    capture_matrix = context.capture_matrix
    n_individuals, n_occasions = capture_matrix.shape
    
    log_lik = 0.0
    
    # For each individual
    for i in range(n_individuals):
        ch = capture_matrix[i, :]
        
        # First capture
        first_cap = jnp.argmax(ch)
        if jnp.any(ch):
            log_lik += jnp.log(p)  # Detection probability
        
        # Subsequent occasions
        for t in range(n_occasions - 1):
            if ch[t] == 1:  # Captured at t
                if ch[t + 1] == 1:  # Recaptured at t+1
                    log_lik += jnp.log(phi * p)
                else:  # Not recaptured
                    log_lik += jnp.log(1 - phi * p)
    
    # Add recruitment component (simplified)
    log_lik += jnp.sum(jnp.log(1 + f)) * 0.1  # Recruitment term
    
    return -log_lik  # Return negative log-likelihood for minimization


def demo_basic_optimization():
    """Demonstrate basic optimization with automatic strategy selection."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Optimization with Automatic Strategy Selection")
    print("="*60)
    
    # Create model context
    context = ExampleModelContext(n_individuals=1500, n_occasions=5, n_parameters=3)
    
    # Initial parameter guess (on transformed scale)
    initial_params = np.array([0.0, -0.5, -2.0])  # phi_logit, p_logit, f_log
    
    # Parameter bounds (logit/log scale)
    bounds = [(-5, 5), (-5, 5), (-5, 2)]
    
    print("Running optimization with automatic strategy selection...")
    
    # Run optimization
    response = optimize_model(
        objective_function=lambda params: pradel_log_likelihood(params, context),
        initial_parameters=initial_params,
        context=context,
        bounds=bounds,
        enable_monitoring=True
    )
    
    # Print results
    print(f"\nOptimization Results:")
    print(f"  Success: {response.success}")
    print(f"  Strategy used: {response.strategy_used}")
    print(f"  Final log-likelihood: {-response.result.fun:.4f}")
    print(f"  Iterations: {response.result.nit}")
    print(f"  Time: {response.total_time:.2f} seconds")
    print(f"  Confidence: {response.confidence_score:.1%}")
    
    # Transform parameters back to natural scale
    if response.success:
        phi_est = jax.nn.sigmoid(response.result.x[0])
        p_est = jax.nn.sigmoid(response.result.x[1])
        f_est = np.exp(response.result.x[2])
        
        print(f"\nParameter Estimates:")
        print(f"  Survival (φ): {phi_est:.3f} (true: {context.true_phi:.3f})")
        print(f"  Detection (p): {p_est:.3f} (true: {context.true_p:.3f})")
        print(f"  Recruitment (f): {f_est:.3f} (true: {context.true_f:.3f})")
    
    return response


def demo_strategy_comparison():
    """Demonstrate strategy comparison and benchmarking."""
    print("\n" + "="*60)
    print("DEMO 2: Strategy Comparison and Benchmarking")
    print("="*60)
    
    # Create smaller problem for faster comparison
    context = ExampleModelContext(n_individuals=800, n_occasions=4, n_parameters=3)
    initial_params = np.array([0.2, -0.3, -1.5])
    bounds = [(-3, 3), (-3, 3), (-3, 1)]
    
    # Compare different strategies
    strategies_to_compare = [
        OptimizationStrategy.SCIPY_LBFGS,
        OptimizationStrategy.SCIPY_SLSQP,
        OptimizationStrategy.JAX_ADAM
    ]
    
    print(f"Comparing {len(strategies_to_compare)} optimization strategies...")
    
    results = compare_optimization_strategies(
        objective_function=lambda params: pradel_log_likelihood(params, context),
        initial_parameters=initial_params,
        context=context,
        bounds=bounds,
        strategies=strategies_to_compare
    )
    
    # Analyze results
    print(f"\nStrategy Comparison Results:")
    print("-" * 80)
    print(f"{'Strategy':<15} {'Success':<8} {'Objective':<12} {'Time':<8} {'Iterations':<10}")
    print("-" * 80)
    
    successful_strategies = []
    
    for strategy_name, response in results.items():
        success_symbol = "✓" if response.success else "✗"
        obj_str = f"{response.result.fun:.6f}" if response.success else "Failed"
        time_str = f"{response.total_time:.2f}s"
        iter_str = str(response.result.nit) if response.success else "-"
        
        print(f"{strategy_name:<15} {success_symbol:<8} {obj_str:<12} {time_str:<8} {iter_str:<10}")
        
        if response.success:
            successful_strategies.append((strategy_name, response.result.fun, response.total_time))
    
    if successful_strategies:
        # Find best strategy by objective value
        best_strategy = min(successful_strategies, key=lambda x: x[1])
        fastest_strategy = min(successful_strategies, key=lambda x: x[2])
        
        print(f"\nSummary:")
        print(f"  Best objective: {best_strategy[0]} ({best_strategy[1]:.6f})")
        print(f"  Fastest: {fastest_strategy[0]} ({fastest_strategy[2]:.2f}s)")
        print(f"  Success rate: {len(successful_strategies)}/{len(results)} "
              f"({len(successful_strategies)/len(results):.1%})")
    
    return results


def demo_experiment_tracking():
    """Demonstrate experiment tracking and monitoring."""
    print("\n" + "="*60)
    print("DEMO 3: Experiment Tracking and Monitoring")
    print("="*60)
    
    context = ExampleModelContext(n_individuals=1000, n_occasions=4, n_parameters=3)
    
    # Run multiple optimizations with different settings
    experiment_name = "pradel_parameter_sensitivity"
    
    with optimization_experiment(experiment_name, "Testing parameter estimation sensitivity"):
        print(f"Running experiment: {experiment_name}")
        
        # Try different starting points
        starting_points = [
            np.array([0.0, 0.0, -1.0]),    # Reasonable start
            np.array([2.0, -2.0, 0.0]),    # Extreme start
            np.array([-1.0, 1.0, -3.0])   # Another extreme start
        ]
        
        results = []
        
        for i, start_point in enumerate(starting_points):
            print(f"\n  Run {i+1}: Starting from {start_point}")
            
            request = OptimizationRequest(
                objective_function=lambda params: pradel_log_likelihood(params, context),
                initial_parameters=start_point,
                bounds=[(-4, 4), (-4, 4), (-4, 2)],
                enable_monitoring=True,
                enable_profiling=True,
                experiment_name=experiment_name
            )
            
            orchestrator = OptimizationOrchestrator()
            response = orchestrator.optimize(request, context)
            
            print(f"    Success: {response.success}")
            print(f"    Strategy: {response.strategy_used}")
            print(f"    Final objective: {response.result.fun:.6f}")
            print(f"    Time: {response.total_time:.2f}s")
            
            results.append(response)
        
        # Analyze experiment results
        successful_runs = [r for r in results if r.success]
        
        if successful_runs:
            objectives = [r.result.fun for r in successful_runs]
            times = [r.total_time for r in successful_runs]
            
            print(f"\nExperiment Summary:")
            print(f"  Successful runs: {len(successful_runs)}/{len(results)}")
            print(f"  Best objective: {min(objectives):.6f}")
            print(f"  Worst objective: {max(objectives):.6f}")
            print(f"  Avg time: {np.mean(times):.2f}s")
            print(f"  Std time: {np.std(times):.2f}s")
    
    return results


def demo_strategy_recommendation():
    """Demonstrate automatic strategy recommendation."""
    print("\n" + "="*60)
    print("DEMO 4: Automatic Strategy Recommendation")
    print("="*60)
    
    # Test different problem types
    problem_types = [
        ("Small well-conditioned", ExampleModelContext(500, 4, 3)),
        ("Large well-conditioned", ExampleModelContext(5000, 6, 5)), 
        ("Under-identified", ExampleModelContext(200, 5, 15))
    ]
    
    for problem_name, context in problem_types:
        print(f"\n{problem_name} problem:")
        print(f"  {context.n_individuals} individuals, {context.n_occasions} occasions, {context.n_parameters} parameters")
        print(f"  Parameter ratio: {context.n_parameters/context.n_individuals:.3f}")
        
        # Get recommendation
        recommendation = recommend_strategy(context)
        
        print(f"  Recommended strategy: {recommendation.strategy.value}")
        print(f"  Confidence: {recommendation.confidence:.1%}")
        print(f"  Expected success rate: {recommendation.expected_success_rate:.1%}")
        print(f"  Estimated time: {recommendation.estimated_time_seconds:.1f}s")
        print(f"  Rationale: {recommendation.rationale}")
        
        if recommendation.preprocessing_recommendations:
            print(f"  Recommendations:")
            for rec in recommendation.preprocessing_recommendations:
                print(f"    - {rec}")


def main():
    """Run all demonstrations."""
    print("Pradel-JAX Optimization Framework Demo")
    print("=" * 60)
    print("This demo shows how to use the optimization framework for")
    print("capture-recapture model fitting with automatic strategy selection.")
    
    try:
        # Run demonstrations
        basic_result = demo_basic_optimization()
        comparison_results = demo_strategy_comparison()
        experiment_results = demo_experiment_tracking()
        demo_strategy_recommendation()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE")
        print("="*60)
        print("All demonstrations completed successfully!")
        print("\nKey takeaways:")
        print("1. The framework automatically selects optimal strategies")
        print("2. Multiple strategies can be compared easily")  
        print("3. Comprehensive monitoring and experiment tracking")
        print("4. Robust error handling and fallback mechanisms")
        print("5. Industry-standard integration with SciPy, JAX, etc.")
        
        return True
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)