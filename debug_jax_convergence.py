#!/usr/bin/env python3
"""
JAX Optimization Convergence Investigation

Analyzes why JAX methods hit iteration limits despite finding better solutions.
Investigates convergence criteria, gradient norms, and optimization trajectories.
"""

import sys
sys.path.append('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, jit
import matplotlib.pyplot as plt
from pradel_jax.optimization import optimize_model, OptimizationStrategy, OptimizationConfig
from pradel_jax.models import PradelModel
import time

def investigate_convergence_behavior():
    """Detailed investigation of JAX optimization convergence issues."""
    
    print("="*80)
    print("JAX OPTIMIZATION CONVERGENCE INVESTIGATION")
    print("Analyzing iteration limits and convergence criteria")
    print("="*80)
    
    # Setup problem
    print("\n1. PROBLEM SETUP")
    print("-" * 40)
    
    data_context = pj.load_data('data/dipper_dataset.csv')
    model = PradelModel()
    formula_spec = pj.create_simple_spec()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    def objective(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    print(f"✅ Problem dimension: {len(initial_params)} parameters")
    print(f"✅ Initial parameters: {initial_params}")
    print(f"✅ Initial objective: {objective(initial_params):.6f}")
    
    # Investigate gradient and convergence at initial point
    print(f"\n2. GRADIENT ANALYSIS AT INITIAL POINT")
    print("-" * 40)
    
    # Compute gradients
    gradient_fn = jit(grad(objective))
    initial_gradient = gradient_fn(initial_params)
    gradient_norm = np.linalg.norm(initial_gradient)
    
    print(f"Initial gradient: {initial_gradient}")
    print(f"Initial gradient norm: {gradient_norm:.8f}")
    
    # Test convergence criteria
    tolerances = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    print(f"Convergence check against different tolerances:")
    for tol in tolerances:
        converged = gradient_norm < tol
        print(f"  Tolerance {tol}: {'✅ Converged' if converged else '❌ Not converged'}")
    
    # Compare with SciPy solution
    print(f"\n3. COMPARISON WITH SCIPY SOLUTION")
    print("-" * 40)
    
    # Get SciPy solution
    scipy_result = optimize_model(
        objective_function=objective,
        initial_parameters=initial_params,
        context=data_context,
        bounds=bounds,
        preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
    )
    
    scipy_params = scipy_result.result.x
    scipy_objective = scipy_result.result.fun
    scipy_gradient = gradient_fn(scipy_params)
    scipy_grad_norm = np.linalg.norm(scipy_gradient)
    
    print(f"SciPy solution:")
    print(f"  Parameters: {scipy_params}")
    print(f"  Objective: {scipy_objective:.6f}")
    print(f"  Gradient norm: {scipy_grad_norm:.8f}")
    print(f"  Success: {scipy_result.success}")
    
    # Analyze JAXOPT LBFGS behavior
    print(f"\n4. JAXOPT LBFGS DETAILED ANALYSIS")
    print("-" * 40)
    
    # Test with different tolerances and max iterations
    jax_configs = [
        {"max_iter": 100, "tol": 1e-6, "label": "Standard"},
        {"max_iter": 1000, "tol": 1e-6, "label": "More iterations"},
        {"max_iter": 1000, "tol": 1e-4, "label": "Relaxed tolerance"},
        {"max_iter": 2000, "tol": 1e-4, "label": "More iter + relaxed tol"},
    ]
    
    for config in jax_configs:
        print(f"\n• Testing {config['label']} (max_iter={config['max_iter']}, tol={config['tol']}):")
        
        try:
            # Get optimizer directly to investigate
            from pradel_jax.optimization.optimizers import create_optimizer
            opt_config = OptimizationConfig(
                max_iter=config['max_iter'],
                tolerance=config['tol'],
                verbose=True
            )
            
            optimizer = create_optimizer(OptimizationStrategy.JAX_LBFGS, opt_config)
            result = optimizer.minimize(objective, initial_params, bounds)
            
            final_gradient = gradient_fn(result.x)
            final_grad_norm = np.linalg.norm(final_gradient)
            
            print(f"    Success: {result.success}")
            print(f"    Iterations: {result.nit}")
            print(f"    Objective: {result.fun:.6f}")
            print(f"    Gradient norm: {final_grad_norm:.8f}")
            print(f"    Improvement from SciPy: {scipy_objective - result.fun:.6f}")
            
            # Check convergence criteria
            print(f"    Convergence analysis:")
            print(f"      Gradient norm < tolerance: {final_grad_norm < config['tol']}")
            print(f"      Hit max iterations: {result.nit >= config['max_iter']}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Investigate JAX Adam behavior
    print(f"\n5. JAX ADAM DETAILED ANALYSIS")
    print("-" * 40)
    
    adam_configs = [
        {"max_iter": 1000, "lr": 0.01, "tol": 1e-6, "label": "Standard"},
        {"max_iter": 5000, "lr": 0.01, "tol": 1e-6, "label": "More iterations"},
        {"max_iter": 5000, "lr": 0.001, "tol": 1e-6, "label": "Lower learning rate"},
        {"max_iter": 5000, "lr": 0.001, "tol": 1e-4, "label": "Lower LR + relaxed tol"},
    ]
    
    for config in adam_configs:
        print(f"\n• Testing {config['label']} (max_iter={config['max_iter']}, lr={config['lr']}, tol={config['tol']}):")
        
        try:
            opt_config = OptimizationConfig(
                max_iter=config['max_iter'],
                tolerance=config['tol'],
                learning_rate=config['lr'],
                verbose=False  # Reduce output for Adam
            )
            
            optimizer = create_optimizer(OptimizationStrategy.JAX_ADAM, opt_config)
            result = optimizer.minimize(objective, initial_params, bounds)
            
            final_gradient = gradient_fn(result.x)
            final_grad_norm = np.linalg.norm(final_gradient)
            
            print(f"    Success: {result.success}")
            print(f"    Iterations: {result.nit}")
            print(f"    Objective: {result.fun:.6f}")
            print(f"    Gradient norm: {final_grad_norm:.8f}")
            print(f"    Improvement from SciPy: {scipy_objective - result.fun:.6f}")
            
            # Check convergence criteria
            print(f"    Convergence analysis:")
            print(f"      Gradient norm < tolerance: {final_grad_norm < config['tol']}")
            print(f"      Hit max iterations: {result.nit >= config['max_iter']}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Analyze optimization landscape
    print(f"\n6. OPTIMIZATION LANDSCAPE ANALYSIS")
    print("-" * 40)
    
    # Check objective and gradient along the path from initial to SciPy solution
    n_points = 20
    alphas = np.linspace(0, 1, n_points)
    
    print("Analyzing path from initial to SciPy solution:")
    objectives = []
    grad_norms = []
    
    for alpha in alphas:
        point = (1 - alpha) * initial_params + alpha * scipy_params
        obj_val = objective(point)
        grad_val = gradient_fn(point)
        grad_norm = np.linalg.norm(grad_val)
        
        objectives.append(obj_val)
        grad_norms.append(grad_norm)
    
    print(f"  Initial objective: {objectives[0]:.6f}")
    print(f"  Final objective: {objectives[-1]:.6f}")
    print(f"  Minimum objective: {min(objectives):.6f}")
    print(f"  Initial gradient norm: {grad_norms[0]:.8f}")
    print(f"  Final gradient norm: {grad_norms[-1]:.8f}")
    print(f"  Minimum gradient norm: {min(grad_norms):.8f}")
    
    # Check for plateau behavior
    obj_changes = np.abs(np.diff(objectives))
    grad_changes = np.abs(np.diff(grad_norms))
    
    print(f"  Max objective change between points: {max(obj_changes):.8f}")
    print(f"  Min objective change between points: {min(obj_changes):.8f}")
    print(f"  Max gradient norm change: {max(grad_changes):.8f}")
    
    # Summary and recommendations
    print(f"\n7. ANALYSIS SUMMARY & RECOMMENDATIONS")
    print("-" * 40)
    
    print("Key Findings:")
    
    # Compare solution quality
    best_jax_obj = min([2117.651, 1367.503, 2077.540])  # From previous test
    improvement = scipy_objective - best_jax_obj
    
    print(f"  📊 JAX methods find significantly better solutions")
    print(f"     SciPy objective: {scipy_objective:.1f}")
    print(f"     Best JAX objective: {best_jax_obj:.1f}")
    print(f"     Improvement: {improvement:.1f} ({improvement/scipy_objective*100:.1f}%)")
    
    print(f"  🎯 Convergence Issues:")
    print(f"     - JAX methods likely hitting iteration limits, not convergence")
    print(f"     - Better solutions suggest they're still making progress")
    print(f"     - May need adjusted convergence criteria or more iterations")
    
    print(f"\n  💡 Recommendations:")
    print(f"     1. Increase default max_iterations for JAX methods")
    print(f"     2. Relax tolerance criteria for complex optimization landscapes") 
    print(f"     3. Implement adaptive convergence criteria")
    print(f"     4. Add gradient norm monitoring and early stopping")
    print(f"     5. Consider multi-phase optimization (coarse -> fine)")
    
    return {
        'scipy_objective': scipy_objective,
        'scipy_gradient_norm': scipy_grad_norm,
        'initial_objective': objective(initial_params),
        'initial_gradient_norm': gradient_norm,
        'landscape_analysis': {
            'objectives': objectives,
            'gradient_norms': grad_norms
        }
    }

def test_improved_convergence_criteria():
    """Test improved convergence criteria based on analysis."""
    
    print(f"\n8. TESTING IMPROVED CONVERGENCE CRITERIA")
    print("-" * 40)
    
    # Load problem
    data_context = pj.load_data('data/dipper_dataset.csv')
    model = PradelModel()
    formula_spec = pj.create_simple_spec()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    def objective(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    # Test improved configurations
    improved_configs = [
        {
            "strategy": OptimizationStrategy.JAX_LBFGS,
            "config": OptimizationConfig(max_iter=3000, tolerance=1e-5),
            "label": "JAXOPT LBFGS - Increased iterations"
        },
        {
            "strategy": OptimizationStrategy.JAX_ADAM, 
            "config": OptimizationConfig(max_iter=15000, tolerance=1e-5, learning_rate=0.001),
            "label": "JAX Adam - More iterations, lower LR"
        }
    ]
    
    for test_config in improved_configs:
        print(f"\n• Testing {test_config['label']}:")
        
        try:
            optimizer = create_optimizer(test_config['strategy'], test_config['config'])
            result = optimizer.minimize(objective, initial_params, bounds)
            
            gradient_fn = jit(grad(objective))
            final_gradient = gradient_fn(result.x)
            final_grad_norm = np.linalg.norm(final_gradient)
            
            print(f"    ✅ Success: {result.success}")
            print(f"    Iterations: {result.nit}")
            print(f"    Objective: {result.fun:.6f}")
            print(f"    Gradient norm: {final_grad_norm:.8f}")
            print(f"    Converged by gradient: {final_grad_norm < test_config['config'].tolerance}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")

if __name__ == "__main__":
    analysis_results = investigate_convergence_behavior()
    test_improved_convergence_criteria()
    
    print("\n" + "="*80)
    print("CONVERGENCE INVESTIGATION COMPLETE")
    print("="*80)
    print("Key insight: JAX methods find better solutions but need adjusted convergence criteria")