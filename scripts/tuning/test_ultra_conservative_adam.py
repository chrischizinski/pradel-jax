#!/usr/bin/env python3
"""
Test ultra-conservative JAX Adam configuration.
"""

import time
import numpy as np
import pradel_jax as pj
from pradel_jax.models import PradelModel
from pradel_jax.optimization.strategy import OptimizationConfig
from pradel_jax.optimization.optimizers import JAXAdamOptimizer

def test_ultra_conservative_adam():
    print("=== Testing Ultra-Conservative JAX Adam ===")
    
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
    
    # Test ultra-conservative configurations
    # Gradient norm was ~370, so learning rate should be much smaller
    configs_to_test = [
        {"lr": 0.00001, "tol": 1e-2, "max_iter": 3000, "name": "Ultra-Conservative"},
        {"lr": 0.000005, "tol": 1e-2, "max_iter": 5000, "name": "Extremely Small LR"},
        {"lr": 0.00002, "tol": 1e-2, "max_iter": 2000, "name": "Small LR Fast"},
    ]
    
    print("Config           | Success | Time    | AIC      | Iter | Grad Norm | Loss Progress")
    print("-" * 90)
    
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
                
                # Get loss progress
                if hasattr(result, 'convergence_history') and result.convergence_history:
                    initial_loss = result.convergence_history[0]
                    final_loss = result.convergence_history[-1]
                    loss_progress = f"{initial_loss:.1f}→{final_loss:.1f}"
                else:
                    loss_progress = f"N/A→{final_nll:.1f}"
            else:
                aic = float('inf')
                final_grad_norm = float('inf')
                
                # Try to get progress even from failed run
                if hasattr(result, 'convergence_history') and result.convergence_history:
                    initial_loss = result.convergence_history[0]
                    final_loss = result.convergence_history[-1]
                    loss_progress = f"{initial_loss:.1f}→{final_loss:.1f}"
                else:
                    loss_progress = "No progress"
            
            print(f"{config_info['name']:16} | "
                  f"{'✓' if result.success else '✗':7} | "
                  f"{elapsed:7.3f} | "
                  f"{aic:8.2f} | "
                  f"{result.nit:4d} | "
                  f"{final_grad_norm:9.2e} | "
                  f"{loss_progress}")
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            print(f"{config_info['name']:16} | "
                  f"{'✗':7} | "
                  f"{elapsed:7.3f} | "
                  f"{'inf':8} | "
                  f"{'0':4} | "
                  f"{'Error':9} | "
                  f"{str(e)[:30]}")

    # Try comparing with reference L-BFGS solution
    print(f"\n=== Reference L-BFGS Solution ===")
    try:
        import scipy.optimize
        from jax import grad, jit
        
        def objective_np(params):
            return float(objective(params))
        
        gradient = jit(grad(objective))
        def gradient_np(params):
            return np.array(gradient(params))
        
        start_time = time.perf_counter()
        scipy_result = scipy.optimize.minimize(
            fun=objective_np,
            x0=initial_params,
            method='L-BFGS-B',
            jac=gradient_np,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        elapsed = time.perf_counter() - start_time
        
        if scipy_result.success:
            aic = 2 * scipy_result.fun + 2 * len(initial_params)
            grad_norm = np.linalg.norm(scipy_result.jac)
        else:
            aic = float('inf')
            grad_norm = float('inf')
        
        print(f"L-BFGS           | "
              f"{'✓' if scipy_result.success else '✗':7} | "
              f"{elapsed:7.3f} | "
              f"{aic:8.2f} | "
              f"{scipy_result.nit:4d} | "
              f"{grad_norm:9.2e} | "
              f"879→{scipy_result.fun:.1f}")
              
        if scipy_result.success:
            print(f"\nL-BFGS target: AIC={aic:.2f}, GradNorm={grad_norm:.2e}")
            print(f"JAX Adam should achieve similar performance with proper tuning.")
        
    except Exception as e:
        print(f"L-BFGS reference failed: {e}")

if __name__ == "__main__":
    test_ultra_conservative_adam()