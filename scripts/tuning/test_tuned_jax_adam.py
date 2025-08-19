#!/usr/bin/env python3
"""
Test the tuned JAX Adam configuration.
"""

import time
import numpy as np
import pradel_jax as pj
from pradel_jax.models import PradelModel
from pradel_jax.optimization.strategy import OptimizationConfig
from pradel_jax.optimization.optimizers import JAXAdamOptimizer

def test_tuned_jax_adam():
    print("=== Testing Tuned JAX Adam Configuration ===")
    
    # Load data
    data_path = 'data/dipper_dataset.csv'
    data_context = pj.load_data(data_path)
    formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
    model = pj.PradelModel()
    
    # Build optimization setup
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    def objective(params):
        try:
            ll = model.log_likelihood(params, data_context, design_matrices)
            return -ll
        except Exception:
            return 1e10
    
    # Test configurations
    configs_to_test = [
        # Newly tuned config
        {"lr": 0.0001, "tol": 1e-3, "max_iter": 5000, "name": "Tuned"},
        
        # A few variations
        {"lr": 0.0002, "tol": 1e-3, "max_iter": 3000, "name": "Faster LR"},
        {"lr": 0.0001, "tol": 1e-4, "max_iter": 3000, "name": "Tighter Tol"},
        {"lr": 0.00005, "tol": 1e-3, "max_iter": 5000, "name": "Conservative"},
    ]
    
    print("Config      | Success | Time    | AIC      | Iter | Final Grad Norm")
    print("-" * 70)
    
    results = []
    
    for config_info in configs_to_test:
        start_time = time.perf_counter()
        
        try:
            config = OptimizationConfig(
                max_iter=config_info["max_iter"],
                tolerance=config_info["tol"],
                learning_rate=config_info["lr"],
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
                
                # Get final gradient norm
                from jax import grad, jit
                gradient = jit(grad(objective))
                final_grad = gradient(result.x)
                final_grad_norm = float(np.linalg.norm(np.array(final_grad)))
            else:
                aic = float('inf')
                final_grad_norm = float('inf')
            
            results.append({
                'name': config_info["name"],
                'success': result.success,
                'time': elapsed,
                'aic': aic,
                'iterations': result.nit,
                'grad_norm': final_grad_norm
            })
            
            print(f"{config_info['name']:11} | "
                  f"{'✓' if result.success else '✗':7} | "
                  f"{elapsed:7.3f} | "
                  f"{aic:8.2f} | "
                  f"{result.nit:4d} | "
                  f"{final_grad_norm:12.2e}")
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"{config_info['name']:11} | "
                  f"{'✗':7} | "
                  f"{elapsed:7.3f} | "
                  f"{'inf':8} | "
                  f"{'0':4} | "
                  f"Error: {str(e)[:20]}")
    
    # Summary
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        print(f"\n=== Success! ===")
        print(f"Success rate: {len(successful_results)}/{len(results)} = {len(successful_results)/len(results):.1%}")
        
        best_result = min(successful_results, key=lambda x: x['aic'])
        print(f"\nBest result ({best_result['name']}):")
        print(f"  AIC: {best_result['aic']:.2f}")
        print(f"  Time: {best_result['time']:.3f}s") 
        print(f"  Iterations: {best_result['iterations']}")
        print(f"  Final gradient norm: {best_result['grad_norm']:.2e}")
        
        print(f"\nJAX Adam is now properly tuned! ✓")
        
    else:
        print(f"\n=== Still Need More Tuning ===")
        print("No configurations converged successfully.")

if __name__ == "__main__":
    test_tuned_jax_adam()