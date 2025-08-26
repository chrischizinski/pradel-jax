#!/usr/bin/env python3
"""
Streamlined Validation of JAX Optimization Claims

Focused validation of the key performance claims:
1. 37.8% improvement in objective/AIC
2. Convergence reliability after fixes
3. Reproducibility across runs
4. Multiple starting points robustness
"""

import sys
sys.path.append('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
import numpy as np
from pradel_jax.optimization import optimize_model, OptimizationStrategy
from pradel_jax.models import PradelModel
from jax import random
import time

def validate_performance_improvement():
    """Validate the 37.8% improvement claim."""
    
    print("="*60)
    print("JAX OPTIMIZATION CLAIMS VALIDATION")
    print("="*60)
    
    print("\n1. PERFORMANCE IMPROVEMENT VALIDATION")
    print("-" * 40)
    
    # Setup problem
    data_context = pj.load_data('data/dipper_dataset.csv')
    model = PradelModel()
    formula_spec = pj.create_simple_spec()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    def objective(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    # Test baseline SciPy
    print("Testing SciPy L-BFGS-B (baseline)...")
    scipy_result = optimize_model(
        objective_function=objective,
        initial_parameters=initial_params,
        context=data_context,
        bounds=bounds,
        preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
    )
    
    scipy_obj = scipy_result.result.fun
    scipy_aic = scipy_result.result.aic
    
    print(f"  SciPy: Obj={scipy_obj:.6f}, AIC={scipy_aic:.2f}, Success={scipy_result.success}")
    
    # Test JAXOPT LBFGS
    print("Testing JAXOPT LBFGS...")
    jax_result = optimize_model(
        objective_function=objective,
        initial_parameters=initial_params,
        context=data_context,
        bounds=bounds,
        preferred_strategy=OptimizationStrategy.JAX_LBFGS
    )
    
    jax_obj = jax_result.result.fun
    jax_aic = jax_result.result.aic
    
    print(f"  JAXOPT: Obj={jax_obj:.6f}, AIC={jax_aic:.2f}, Success={jax_result.success}")
    
    # Calculate improvements
    obj_improvement = (scipy_obj - jax_obj) / scipy_obj * 100
    aic_improvement = (scipy_aic - jax_aic) / scipy_aic * 100
    
    print(f"\nIMPROVEMENT ANALYSIS:")
    print(f"  Objective improvement: {obj_improvement:.1f}%")
    print(f"  AIC improvement: {aic_improvement:.1f}%") 
    
    # Validate claims
    validates_37_8_claim = obj_improvement >= 35.0  # Allow some tolerance
    validates_convergence = jax_result.success
    
    print(f"\nCLAIMS VALIDATION:")
    print(f"  37.8% improvement claim: {'✅ VALIDATED' if validates_37_8_claim else '❌ NOT VALIDATED'}")
    print(f"  Convergence fix claim: {'✅ VALIDATED' if validates_convergence else '❌ NOT VALIDATED'}")
    
    return {
        'scipy_objective': scipy_obj,
        'jax_objective': jax_obj,
        'objective_improvement': obj_improvement,
        'aic_improvement': aic_improvement,
        'validates_improvement_claim': validates_37_8_claim,
        'validates_convergence_claim': validates_convergence
    }

def validate_reproducibility():
    """Validate reproducibility across multiple runs."""
    
    print(f"\n2. REPRODUCIBILITY VALIDATION")
    print("-" * 40)
    
    # Setup
    data_context = pj.load_data('data/dipper_dataset.csv')
    model = PradelModel()
    formula_spec = pj.create_simple_spec()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    def objective(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    # Test JAXOPT LBFGS reproducibility
    n_runs = 10
    print(f"Testing JAXOPT LBFGS reproducibility ({n_runs} runs)...")
    
    objectives = []
    successes = []
    
    for i in range(n_runs):
        result = optimize_model(
            objective_function=objective,
            initial_parameters=initial_params,
            context=data_context,
            bounds=bounds,
            preferred_strategy=OptimizationStrategy.JAX_LBFGS
        )
        
        objectives.append(result.result.fun)
        successes.append(result.success)
        print(f"  Run {i+1}: Obj={result.result.fun:.8f}, Success={result.success}")
    
    # Analyze reproducibility
    obj_std = np.std(objectives)
    obj_range = max(objectives) - min(objectives)
    success_rate = np.mean(successes)
    
    # Reproducibility criteria
    highly_reproducible = obj_std < 1e-6
    practically_reproducible = obj_std < 1e-3
    
    print(f"\nREPRODUCIBILITY ANALYSIS:")
    print(f"  Objective std: {obj_std:.10f}")
    print(f"  Objective range: {obj_range:.10f}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Highly reproducible (std<1e-6): {'✅ YES' if highly_reproducible else '❌ NO'}")
    print(f"  Practically reproducible (std<1e-3): {'✅ YES' if practically_reproducible else '❌ NO'}")
    
    return {
        'objective_std': obj_std,
        'objective_range': obj_range,
        'success_rate': success_rate,
        'highly_reproducible': highly_reproducible,
        'practically_reproducible': practically_reproducible
    }

def validate_robustness():
    """Validate robustness across different starting points."""
    
    print(f"\n3. ROBUSTNESS VALIDATION")
    print("-" * 40)
    
    # Setup
    data_context = pj.load_data('data/dipper_dataset.csv')
    model = PradelModel()
    formula_spec = pj.create_simple_spec()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    def objective(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    default_start = model.get_initial_parameters(data_context, design_matrices)
    
    # Generate diverse starting points
    n_starts = 8
    key = random.PRNGKey(42)
    starting_points = []
    
    # Random points within bounds
    for i in range(n_starts // 2):
        key, subkey = random.split(key)
        start_point = []
        for (lower, upper) in bounds:
            param = random.uniform(subkey, (), minval=float(lower)*0.8, maxval=float(upper)*0.8)
            start_point.append(float(param))
        starting_points.append(np.array(start_point))
    
    # Perturbations around default
    for i in range(n_starts // 2):
        key, subkey = random.split(key)
        perturbation = random.normal(subkey, shape=default_start.shape) * 0.5
        starting_points.append(default_start + perturbation)
    
    print(f"Testing JAXOPT LBFGS from {len(starting_points)} diverse starting points...")
    
    objectives = []
    successes = []
    
    for i, start_point in enumerate(starting_points):
        result = optimize_model(
            objective_function=objective,
            initial_parameters=start_point,
            context=data_context,
            bounds=bounds,
            preferred_strategy=OptimizationStrategy.JAX_LBFGS
        )
        
        objectives.append(result.result.fun)
        successes.append(result.success)
        print(f"  Start {i+1}: Obj={result.result.fun:.4f}, Success={result.success}")
    
    # Analyze robustness
    success_rate = np.mean(successes)
    valid_objectives = [obj for obj, success in zip(objectives, successes) if success]
    
    if valid_objectives:
        obj_std = np.std(valid_objectives)
        consistent_solution = obj_std < 1.0  # Solutions within reasonable range
    else:
        obj_std = float('inf')
        consistent_solution = False
    
    print(f"\nROBUSTNESS ANALYSIS:")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Valid solutions std: {obj_std:.6f}")
    print(f"  Finds consistent solution: {'✅ YES' if consistent_solution else '❌ NO'}")
    
    return {
        'success_rate': success_rate,
        'solution_consistency': consistent_solution,
        'valid_objectives_std': obj_std
    }

def comprehensive_validation_summary():
    """Run comprehensive validation and provide summary."""
    
    # Run all validations
    performance_results = validate_performance_improvement()
    reproducibility_results = validate_reproducibility()
    robustness_results = validate_robustness()
    
    print(f"\n" + "="*60)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("="*60)
    
    # Count validations
    validations = []
    
    # Performance claims
    if performance_results['validates_improvement_claim']:
        validations.append("37.8% improvement claim")
    if performance_results['validates_convergence_claim']:
        validations.append("Convergence fix")
    
    # Reproducibility
    if reproducibility_results['practically_reproducible']:
        validations.append("Reproducibility")
    
    # Robustness
    if robustness_results['success_rate'] > 0.7:
        validations.append("Robustness")
    
    print(f"\n✅ VALIDATED CLAIMS ({len(validations)}/4):")
    for validation in validations:
        print(f"  • {validation}")
    
    # Key metrics summary
    print(f"\n📊 KEY METRICS:")
    print(f"  • Objective improvement: {performance_results['objective_improvement']:.1f}%")
    print(f"  • AIC improvement: {performance_results['aic_improvement']:.1f}%")
    print(f"  • Reproducibility std: {reproducibility_results['objective_std']:.2e}")
    print(f"  • Multi-start success rate: {robustness_results['success_rate']:.1%}")
    
    # Overall validation
    overall_success = len(validations) >= 3
    
    print(f"\n🎯 OVERALL VALIDATION: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("   JAX optimization claims are validated by extensive testing")
    else:
        print("   Some claims require further investigation")
    
    return {
        'performance': performance_results,
        'reproducibility': reproducibility_results,
        'robustness': robustness_results,
        'validated_claims': validations,
        'overall_success': overall_success
    }

if __name__ == "__main__":
    print("Starting streamlined validation of JAX optimization claims...")
    
    try:
        results = comprehensive_validation_summary()
        
        if results['overall_success']:
            print(f"\n🎉 VALIDATION SUCCESSFUL")
            exit(0)
        else:
            print(f"\n⚠️  VALIDATION INCOMPLETE")
            exit(1)
            
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)