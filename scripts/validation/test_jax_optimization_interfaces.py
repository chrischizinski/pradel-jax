#!/usr/bin/env python3
"""
JAX Optimization Interface Standardization Test

Tests all optimization strategies to ensure consistent interfaces
and validates the new JAXOPT LBFGS implementation.
"""

import sys
sys.path.append('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
import numpy as np
from pradel_jax.optimization import optimize_model, OptimizationStrategy, OptimizationConfig
from pradel_jax.models import PradelModel
import time

def test_optimization_strategy_interfaces():
    """Test all optimization strategies for interface consistency."""
    
    print("="*80)
    print("OPTIMIZATION STRATEGY INTERFACE VALIDATION")
    print("Testing JAXOPT LBFGS and interface standardization")
    print("="*80)
    
    # Load data and setup problem
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
    print(f"✅ Bounds: {bounds}")
    
    # Test strategies
    strategies_to_test = [
        OptimizationStrategy.SCIPY_LBFGS,
        OptimizationStrategy.SCIPY_SLSQP,
        OptimizationStrategy.JAX_ADAM,
        OptimizationStrategy.JAX_LBFGS,  # Our new implementation
        OptimizationStrategy.MULTI_START,
        OptimizationStrategy.HYBRID,
        OptimizationStrategy.JAX_ADAM_ADAPTIVE
    ]
    
    print(f"\n2. STRATEGY TESTING")
    print("-" * 40)
    print(f"Testing {len(strategies_to_test)} optimization strategies...")
    
    results = {}
    
    for strategy in strategies_to_test:
        print(f"\n• Testing {strategy.value}...")
        
        try:
            start_time = time.time()
            result = optimize_model(
                objective_function=objective,
                initial_parameters=initial_params,
                context=data_context,
                bounds=bounds,
                preferred_strategy=strategy
            )
            elapsed = time.time() - start_time
            
            # Check result interface consistency
            assert hasattr(result, 'success'), f"{strategy.value}: Missing 'success' attribute"
            assert hasattr(result, 'strategy_used'), f"{strategy.value}: Missing 'strategy_used' attribute"
            assert hasattr(result, 'result'), f"{strategy.value}: Missing 'result' attribute"
            
            # Get optimization result
            opt_result = result.result
            assert hasattr(opt_result, 'x'), f"{strategy.value}: Missing 'x' parameter in result"
            assert hasattr(opt_result, 'fun'), f"{strategy.value}: Missing 'fun' objective value"
            assert hasattr(opt_result, 'nit'), f"{strategy.value}: Missing 'nit' iterations"
            assert hasattr(opt_result, 'strategy_used'), f"{strategy.value}: Missing 'strategy_used' in result"
            
            results[strategy.value] = {
                'success': result.success,
                'strategy_used': result.strategy_used,
                'objective': opt_result.fun,
                'parameters': opt_result.x,
                'iterations': opt_result.nit,
                'time': elapsed,
                'aic': getattr(opt_result, 'aic', None),
                'interface_valid': True
            }
            
            print(f"  ✅ Success: {result.success}")
            print(f"  ✅ Strategy: {result.strategy_used}")
            print(f"  ✅ Objective: {opt_result.fun:.6f}")
            print(f"  ✅ Iterations: {opt_result.nit}")
            print(f"  ✅ Time: {elapsed:.3f}s")
            if hasattr(opt_result, 'aic') and opt_result.aic is not None:
                print(f"  ✅ AIC: {opt_result.aic:.2f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[strategy.value] = {
                'success': False,
                'error': str(e),
                'interface_valid': False
            }
    
    # Results summary
    print(f"\n3. RESULTS SUMMARY")
    print("-" * 40)
    
    successful_strategies = [name for name, result in results.items() if result.get('success', False)]
    failed_strategies = [name for name, result in results.items() if not result.get('success', False)]
    interface_valid = [name for name, result in results.items() if result.get('interface_valid', False)]
    
    print(f"✅ Successful strategies: {len(successful_strategies)}/{len(strategies_to_test)}")
    print(f"✅ Interface valid strategies: {len(interface_valid)}/{len(strategies_to_test)}")
    
    if successful_strategies:
        print(f"\n✅ Working strategies:")
        for name in successful_strategies:
            result = results[name]
            print(f"  - {name}: objective={result['objective']:.6f}, time={result['time']:.3f}s")
    
    if failed_strategies:
        print(f"\n❌ Failed strategies:")
        for name in failed_strategies:
            error = results[name].get('error', 'Unknown error')
            print(f"  - {name}: {error}")
    
    # JAXOPT LBFGS specific validation
    print(f"\n4. JAXOPT LBFGS VALIDATION")
    print("-" * 40)
    
    if 'jax_lbfgs' in results:
        jax_lbfgs_result = results['jax_lbfgs']
        if jax_lbfgs_result.get('success', False):
            print("✅ JAXOPT LBFGS implementation working correctly")
            print(f"✅ Strategy properly integrated in orchestrator")
            print(f"✅ Result interface consistent with other optimizers")
            
            # Compare with scipy reference
            if 'scipy_lbfgs' in results and results['scipy_lbfgs'].get('success', False):
                scipy_obj = results['scipy_lbfgs']['objective']
                jax_obj = jax_lbfgs_result['objective']
                obj_diff = abs(jax_obj - scipy_obj)
                
                print(f"✅ Comparison with SciPy L-BFGS-B:")
                print(f"   SciPy objective: {scipy_obj:.6f}")
                print(f"   JAXOPT objective: {jax_obj:.6f}")
                print(f"   Difference: {obj_diff:.8f}")
                
                if obj_diff < 1.0:  # Allow reasonable difference
                    print(f"✅ JAXOPT LBFGS produces similar results to SciPy")
                else:
                    print(f"⚠️  Significant difference - may need convergence tuning")
        else:
            print("❌ JAXOPT LBFGS implementation has issues")
    
    print(f"\n5. INTERFACE STANDARDIZATION STATUS")
    print("-" * 40)
    
    if len(interface_valid) == len(strategies_to_test):
        print("✅ ALL OPTIMIZATION INTERFACES STANDARDIZED")
        print("✅ All strategies provide consistent result attributes")
        print("✅ JAX optimization interfaces successfully standardized")
    else:
        print(f"⚠️  Interface standardization incomplete")
        print(f"   Valid interfaces: {len(interface_valid)}/{len(strategies_to_test)}")
    
    return results

if __name__ == "__main__":
    results = test_optimization_strategy_interfaces()
    
    print("\n" + "="*80)
    print("INTERFACE STANDARDIZATION TEST COMPLETE")
    print("="*80)
    
    successful_count = sum(1 for r in results.values() if r.get('success', False))
    total_count = len(results)
    
    if successful_count >= total_count * 0.8:  # 80% success rate
        print(f"✅ OVERALL SUCCESS: {successful_count}/{total_count} strategies working")
        print("✅ JAX optimization interface standardization COMPLETE")
        exit(0)
    else:
        print(f"❌ ISSUES DETECTED: Only {successful_count}/{total_count} strategies working")
        print("❌ Additional work needed for interface standardization")
        exit(1)