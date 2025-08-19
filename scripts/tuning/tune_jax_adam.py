#!/usr/bin/env python3
"""
JAX Adam parameter tuning for optimal convergence.
Tests different learning rates, tolerances, and max iterations.
"""

import time
import numpy as np
import pradel_jax as pj
from pradel_jax.models import PradelModel
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy, OptimizationConfig

def test_jax_adam_config(learning_rate, tolerance, max_iter, data_context, formula_spec, model):
    """Test JAX Adam with specific configuration."""
    
    try:
        start_time = time.perf_counter()
        
        # Build optimization setup
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Add required attributes
        if not hasattr(data_context, 'n_parameters'):
            data_context.n_parameters = len(initial_params)
        if not hasattr(data_context, 'get_condition_estimate'):
            data_context.get_condition_estimate = lambda: 1e5
        
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                return -ll  # Return JAX array directly
            except Exception:
                return 1e10
        
        # Create custom config
        config = OptimizationConfig(
            max_iter=max_iter,
            tolerance=tolerance,
            learning_rate=learning_rate,
            verbose=False
        )
        
        # Use the lower-level optimizer directly to control config
        from pradel_jax.optimization.optimizers import JAXAdamOptimizer
        
        optimizer = JAXAdamOptimizer(config)
        result = optimizer.minimize(
            objective=objective,
            x0=initial_params,
            bounds=bounds
        )
        
        elapsed = time.perf_counter() - start_time
        
        if result.success:
            final_nll = result.fun  # Direct access to OptimizationResult
            n_params = len(initial_params)
            aic = 2 * final_nll + 2 * n_params
        else:
            aic = float('inf')
        
        return {
            'learning_rate': learning_rate,
            'tolerance': tolerance,
            'max_iter': max_iter,
            'success': result.success,
            'time': elapsed,
            'aic': aic,
            'iterations': result.nit if result.success else max_iter,
            'message': result.message
        }
        
    except Exception as e:
        return {
            'learning_rate': learning_rate,
            'tolerance': tolerance,
            'max_iter': max_iter,
            'success': False,
            'time': 0,
            'aic': float('inf'),
            'iterations': 0,
            'message': str(e)
        }

def main():
    print("=== JAX Adam Parameter Tuning ===")
    
    # Load data
    data_path = 'data/dipper_dataset.csv'
    data_context = pj.load_data(data_path)
    formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
    model = pj.PradelModel()
    
    # Test configurations
    test_configs = [
        # Learning rate variations
        (0.001, 1e-8, 2000),   # Lower learning rate, more iterations
        (0.005, 1e-8, 2000),   # 
        (0.01, 1e-8, 2000),    # Current default
        (0.02, 1e-8, 2000),    # Higher learning rate
        (0.05, 1e-8, 2000),    # Much higher learning rate
        
        # Tolerance variations with adjusted learning rate
        (0.005, 1e-6, 2000),   # Relaxed tolerance
        (0.005, 1e-7, 2000),   # Medium tolerance
        (0.005, 1e-9, 2000),   # Tighter tolerance
        
        # Max iterations variations
        (0.005, 1e-6, 5000),   # More iterations
        (0.005, 1e-6, 10000),  # Many more iterations
        
        # Conservative configs
        (0.001, 1e-6, 5000),   # Very conservative
        (0.002, 1e-6, 3000),   # Conservative
    ]
    
    results = []
    
    print(f"Testing {len(test_configs)} configurations...")
    print("LR      | Tol    | MaxIter | Success | Time    | AIC      | Iter | Message")
    print("-" * 80)
    
    for lr, tol, max_iter in test_configs:
        result = test_jax_adam_config(lr, tol, max_iter, data_context, formula_spec, model)
        results.append(result)
        
        print(f"{lr:7.3f} | {tol:6.0e} | {max_iter:7d} | "
              f"{'✓' if result['success'] else '✗':7} | "
              f"{result['time']:7.3f} | "
              f"{result['aic']:8.2f} | "
              f"{result['iterations']:4d} | "
              f"{result['message'][:20]}")
    
    # Find best configuration
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        print(f"\n=== Analysis ===")
        print(f"Success rate: {len(successful_results)}/{len(results)} = {len(successful_results)/len(results):.1%}")
        
        # Best by AIC
        best_aic = min(successful_results, key=lambda x: x['aic'])
        print(f"\nBest AIC configuration:")
        print(f"  Learning rate: {best_aic['learning_rate']}")
        print(f"  Tolerance: {best_aic['tolerance']:.0e}")
        print(f"  Max iterations: {best_aic['max_iter']}")
        print(f"  AIC: {best_aic['aic']:.2f}")
        print(f"  Time: {best_aic['time']:.3f}s")
        print(f"  Iterations: {best_aic['iterations']}")
        
        # Fastest successful
        fastest = min(successful_results, key=lambda x: x['time'])
        print(f"\nFastest successful configuration:")
        print(f"  Learning rate: {fastest['learning_rate']}")
        print(f"  Tolerance: {fastest['tolerance']:.0e}")
        print(f"  Max iterations: {fastest['max_iter']}")
        print(f"  Time: {fastest['time']:.3f}s")
        print(f"  AIC: {fastest['aic']:.2f}")
        
        # Recommend optimal config
        print(f"\n=== Recommendation ===")
        if best_aic == fastest:
            print("Best AIC and fastest configurations are the same!")
            recommended = best_aic
        else:
            # Choose based on balance of performance and speed
            if best_aic['time'] <= fastest['time'] * 2:  # If best AIC is not much slower
                recommended = best_aic
                print("Recommending best AIC configuration (good balance)")
            else:
                recommended = fastest
                print("Recommending fastest configuration (much faster)")
        
        print(f"\nOptimal JAX Adam configuration:")
        print(f"  learning_rate: {recommended['learning_rate']}")
        print(f"  tolerance: {recommended['tolerance']:.0e}")
        print(f"  max_iter: {recommended['max_iter']}")
        
    else:
        print("\n=== FAILURE ANALYSIS ===")
        print("No configurations converged successfully!")
        print("Need to investigate further...")
        
        # Show closest attempts
        print("\nClosest attempts (by iteration count):")
        sorted_results = sorted(results, key=lambda x: x['iterations'], reverse=True)
        for result in sorted_results[:3]:
            print(f"  LR={result['learning_rate']}, Tol={result['tolerance']:.0e}, "
                  f"Iter={result['iterations']}, Msg={result['message']}")

if __name__ == "__main__":
    main()