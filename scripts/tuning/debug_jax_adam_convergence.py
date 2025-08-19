#!/usr/bin/env python3
"""
Debug JAX Adam convergence behavior to understand why it's not converging.
"""

import time
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import pradel_jax as pj
from pradel_jax.models import PradelModel
from pradel_jax.optimization.strategy import OptimizationConfig

def debug_adam_convergence():
    print("=== Debugging JAX Adam Convergence ===")
    
    # Load data
    data_path = 'data/dipper_dataset.csv'
    data_context = pj.load_data(data_path)
    formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
    model = pj.PradelModel()
    
    # Setup optimization
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    
    print(f"Initial parameters: {initial_params}")
    print(f"Parameter count: {len(initial_params)}")
    
    # Define objective and gradient
    def objective(params):
        try:
            ll = model.log_likelihood(params, data_context, design_matrices)
            return -ll  # Negative log-likelihood
        except Exception as e:
            print(f"Objective evaluation failed: {e}")
            return 1e10
    
    gradient = jit(grad(objective))
    
    # Test initial objective and gradient
    initial_loss = objective(initial_params)
    initial_grad = gradient(initial_params)
    initial_grad_norm = jnp.linalg.norm(initial_grad)
    
    print(f"\nInitial state:")
    print(f"  Loss: {initial_loss}")
    print(f"  Gradient norm: {initial_grad_norm}")
    print(f"  Gradient: {initial_grad}")
    
    # Test a few steps of Adam manually
    print(f"\n=== Manual Adam Steps ===")
    
    # Adam parameters
    learning_rate = 0.005
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    
    params = jnp.array(initial_params)
    m = jnp.zeros_like(params)  # First moment
    v = jnp.zeros_like(params)  # Second moment
    
    print(f"{'Step':>4} | {'Loss':>12} | {'Grad Norm':>12} | {'Max Grad':>12} | {'Max Param':>12}")
    print("-" * 70)
    
    for t in range(1, 21):  # Test 20 steps
        loss = objective(params)
        grads = gradient(params)
        grad_norm = jnp.linalg.norm(grads)
        
        print(f"{t:4d} | {float(loss):12.6f} | {float(grad_norm):12.6e} | {float(jnp.max(jnp.abs(grads))):12.6e} | {float(jnp.max(jnp.abs(params))):12.6f}")
        
        # Check for issues
        if jnp.isnan(loss) or jnp.isinf(loss):
            print(f"  -> Loss became invalid at step {t}")
            break
        if jnp.any(jnp.isnan(grads)) or jnp.any(jnp.isinf(grads)):
            print(f"  -> Gradients became invalid at step {t}")
            break
            
        # Adam update
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * grads**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Parameter update
        update = learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        params = params - update
        
        # Check convergence manually
        if grad_norm < 1e-6:
            print(f"  -> Converged at step {t} with grad norm {float(grad_norm):.2e}")
            break
        if grad_norm < 1e-5 and t >= 10:
            print(f"  -> Near convergence at step {t} with grad norm {float(grad_norm):.2e}")
    
    # Compare with working L-BFGS
    print(f"\n=== L-BFGS Comparison ===")
    try:
        import scipy.optimize
        
        def objective_np(params):
            return float(objective(params))
        
        def gradient_np(params):
            return np.array(gradient(params))
        
        scipy_result = scipy.optimize.minimize(
            fun=objective_np,
            x0=initial_params,
            method='L-BFGS-B',
            jac=gradient_np,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        
        print(f"L-BFGS result:")
        print(f"  Success: {scipy_result.success}")
        print(f"  Iterations: {scipy_result.nit}")
        print(f"  Final loss: {scipy_result.fun:.6f}")
        print(f"  Final grad norm: {np.linalg.norm(scipy_result.jac):.6e}")
        
        # Calculate what Adam gradient norm would be at this solution
        adam_grad_at_solution = gradient(scipy_result.x)
        adam_grad_norm_at_solution = float(jnp.linalg.norm(adam_grad_at_solution))
        print(f"  JAX gradient norm at L-BFGS solution: {adam_grad_norm_at_solution:.6e}")
        
    except Exception as e:
        print(f"L-BFGS comparison failed: {e}")
    
    print(f"\n=== Analysis ===")
    print(f"JAX Adam may need:")
    if initial_grad_norm > 1e-3:
        print(f"  - Much smaller learning rate (gradient norm is high: {float(initial_grad_norm):.2e})")
    if float(initial_loss) > 1000:
        print(f"  - Better initialization (initial loss is high: {float(initial_loss):.2f})")
    
    print(f"  - Looser tolerance (e.g., 1e-4 or 1e-3 instead of 1e-6)")
    print(f"  - More iterations (current tests use 3000-5000)")
    print(f"  - Learning rate schedule (decay over time)")

if __name__ == "__main__":
    debug_adam_convergence()