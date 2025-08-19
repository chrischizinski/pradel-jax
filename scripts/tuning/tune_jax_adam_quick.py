#!/usr/bin/env python3
"""
Quick JAX Adam parameter tuning test.
"""

import time
import numpy as np
import pradel_jax as pj
from pradel_jax.models import PradelModel
from pradel_jax.optimization.strategy import OptimizationConfig
from pradel_jax.optimization.optimizers import JAXAdamOptimizer

def test_jax_adam_config(learning_rate, tolerance, max_iter, data_context, formula_spec, model):
    """Test JAX Adam with specific configuration."""
    
    try:
        start_time = time.perf_counter()
        
        # Build optimization setup
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
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
        
        optimizer = JAXAdamOptimizer(config)
        result = optimizer.minimize(
            objective=objective,
            x0=initial_params,
            bounds=bounds
        )
        
        elapsed = time.perf_counter() - start_time
        
        if result.success:
            final_nll = result.fun
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
            'message': str(e)[:50]
        }

def main():
    print("=== Quick JAX Adam Parameter Tuning ===")
    
    # Load data
    data_path = 'data/dipper_dataset.csv'
    data_context = pj.load_data(data_path)
    formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
    model = pj.PradelModel()
    
    # Test a few key configurations
    test_configs = [
        # Current default with more iterations
        (0.01, 1e-8, 3000),    
        
        # Lower learning rate, relaxed tolerance 
        (0.005, 1e-6, 3000),   
        
        # Much lower learning rate, relaxed tolerance
        (0.001, 1e-6, 5000),   
        
        # Very conservative
        (0.001, 1e-5, 5000),   
    ]
    
    results = []
    
    print(f"Testing {len(test_configs)} configurations...")
    print("LR      | Tol    | MaxIter | Success | Time    | AIC      | Iter | Message")
    print("-" * 80)
    
    for lr, tol, max_iter in test_configs:
        print(f"Testing LR={lr}, Tol={tol:.0e}, MaxIter={max_iter}...")
        result = test_jax_adam_config(lr, tol, max_iter, data_context, formula_spec, model)
        results.append(result)
        
        print(f"{lr:7.3f} | {tol:6.0e} | {max_iter:7d} | "
              f"{'✓' if result['success'] else '✗':7} | "
              f"{result['time']:7.3f} | "
              f"{result['aic']:8.2f} | "
              f"{result['iterations']:4d} | "
              f"{result['message'][:30]}")
    
    # Find best configuration
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        print(f"\n=== Success! ===")
        print(f"Success rate: {len(successful_results)}/{len(results)} = {len(successful_results)/len(results):.1%}")
        
        # Best by AIC
        best_aic = min(successful_results, key=lambda x: x['aic'])
        print(f"\nRecommended JAX Adam configuration:")
        print(f"  learning_rate: {best_aic['learning_rate']}")
        print(f"  tolerance: {best_aic['tolerance']:.0e}")
        print(f"  max_iter: {best_aic['max_iter']}")
        print(f"  AIC: {best_aic['aic']:.2f}")
        print(f"  Time: {best_aic['time']:.3f}s")
        print(f"  Iterations: {best_aic['iterations']}")
        
    else:
        print("\n=== Still No Success ===")
        print("All configurations failed to converge!")
        
        # Show details of failures
        print("\nFailure details:")
        for result in results:
            print(f"  LR={result['learning_rate']}, Iter={result['iterations']}")
            print(f"    Message: {result['message']}")

if __name__ == "__main__":
    main()