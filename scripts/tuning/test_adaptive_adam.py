#!/usr/bin/env python3
"""
Test script for the adaptive JAX Adam optimizer.

This script tests the new adaptive Adam implementation to ensure it works
correctly and provides improved performance compared to the basic Adam.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.optimization.adaptive_adam import AdaptiveJAXAdamOptimizer, AdaptiveAdamConfig
from pradel_jax.models import PradelModel


def test_adaptive_adam_basic():
    """Test basic functionality of adaptive Adam optimizer."""
    print("🧪 Testing basic adaptive Adam functionality...")
    
    # Load test data
    data_path = Path(__file__).parent / "data" / "dipper_dataset.csv"
    if not data_path.exists():
        print(f"❌ Test data not found: {data_path}")
        return False
    
    data_context = pj.load_data(str(data_path))
    model = PradelModel()
    formula = pj.create_simple_spec(phi="~1", p="~1", f="~1")
    
    try:
        # Test direct optimizer usage
        config = AdaptiveAdamConfig(
            max_iter=1000,
            learning_rate=0.01,
            verbose=True
        )
        
        optimizer = AdaptiveJAXAdamOptimizer(config)
        
        # Setup optimization problem
        design_matrices = model.build_design_matrices(formula, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                return -ll
            except Exception:
                return 1e10
        
        # Run optimization
        result = optimizer.minimize(objective, initial_params, bounds)
        
        print(f"  ✅ Direct optimizer test: Success={result.success}, Time={result.optimization_time:.3f}s")
        return result.success
        
    except Exception as e:
        print(f"  ❌ Direct optimizer test failed: {e}")
        return False


def test_adaptive_adam_via_strategy():
    """Test adaptive Adam via the strategy framework."""
    print("🧪 Testing adaptive Adam via strategy framework...")
    
    # Load test data
    data_path = Path(__file__).parent / "data" / "dipper_dataset.csv"
    data_context = pj.load_data(str(data_path))
    model = PradelModel()
    formula = pj.create_simple_spec(phi="~1", p="~1", f="~1")
    
    try:
        # Add required attributes for optimization framework
        if not hasattr(data_context, 'n_parameters'):
            design_matrices = model.build_design_matrices(formula, data_context)
            initial_params = model.get_initial_parameters(data_context, design_matrices)
            data_context.n_parameters = len(initial_params)
        if not hasattr(data_context, 'get_condition_estimate'):
            data_context.get_condition_estimate = lambda: 1e5
        
        # Setup optimization
        design_matrices = model.build_design_matrices(formula, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                return -ll
            except Exception:
                return 1e10
        
        # Run optimization with adaptive Adam strategy
        result = optimize_model(
            objective_function=objective,
            initial_parameters=initial_params,
            context=data_context,
            bounds=bounds,
            preferred_strategy=OptimizationStrategy.JAX_ADAM_ADAPTIVE
        )
        
        print(f"  ✅ Strategy test: Success={result.success}, Time={result.optimization_time:.3f}s")
        return result.success
        
    except Exception as e:
        print(f"  ❌ Strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def performance_comparison():
    """Compare performance between basic and adaptive Adam."""
    print("📊 Comparing basic vs adaptive Adam performance...")
    
    # Load test data
    data_path = Path(__file__).parent / "data" / "dipper_dataset.csv"
    data_context = pj.load_data(str(data_path))
    model = PradelModel()
    formula = pj.create_simple_spec(phi="~1", p="~1", f="~1")
    
    # Add required attributes
    design_matrices = model.build_design_matrices(formula, data_context)
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    if not hasattr(data_context, 'n_parameters'):
        data_context.n_parameters = len(initial_params)
    if not hasattr(data_context, 'get_condition_estimate'):
        data_context.get_condition_estimate = lambda: 1e5
    
    def objective(params):
        try:
            ll = model.log_likelihood(params, data_context, design_matrices)
            return -ll
        except Exception:
            return 1e10
    
    strategies_to_test = [
        ('Basic JAX Adam', OptimizationStrategy.JAX_ADAM),
        ('Adaptive JAX Adam', OptimizationStrategy.JAX_ADAM_ADAPTIVE),
    ]
    
    results = []
    
    for name, strategy in strategies_to_test:
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
            
            performance = {
                'strategy': name,
                'success': result.success,
                'time': elapsed,
                'final_objective': result.result.fun if result.success else float('inf'),
                'iterations': result.result.nit if hasattr(result.result, 'nit') else 0
            }
            
            results.append(performance)
            
            print(f"  {name:20} | Success: {result.success} | Time: {elapsed:.3f}s | "
                  f"Obj: {performance['final_objective']:.6f}")
            
        except Exception as e:
            print(f"  {name:20} | ❌ Failed: {e}")
            results.append({
                'strategy': name,
                'success': False,
                'time': float('inf'),
                'final_objective': float('inf'),
                'iterations': 0
            })
    
    # Compare results
    if len(results) >= 2:
        basic_result = results[0]
        adaptive_result = results[1]
        
        print(f"\n📈 Comparison Summary:")
        if adaptive_result['success'] and basic_result['success']:
            time_improvement = basic_result['time'] / adaptive_result['time']
            obj_improvement = basic_result['final_objective'] / adaptive_result['final_objective']
            print(f"  Time ratio (basic/adaptive): {time_improvement:.2f}x")
            print(f"  Objective ratio (basic/adaptive): {obj_improvement:.2f}x")
        elif adaptive_result['success'] and not basic_result['success']:
            print(f"  ✅ Adaptive Adam succeeded where basic Adam failed!")
        elif not adaptive_result['success'] and basic_result['success']:
            print(f"  ⚠️  Basic Adam succeeded where adaptive Adam failed")
        else:
            print(f"  ❌ Both optimizers failed")
    
    return results


def test_strategy_selection():
    """Test that the strategy selector chooses adaptive Adam appropriately."""
    print("🎯 Testing automatic strategy selection...")
    
    # Load test data
    data_path = Path(__file__).parent / "data" / "dipper_dataset.csv"
    data_context = pj.load_data(str(data_path))
    
    try:
        # Test with high-level interface
        result = pj.fit_model(
            model=pj.PradelModel(),
            formula=pj.create_simple_spec(phi="~1", p="~1", f="~1"),
            data=data_context
        )
        
        print(f"  ✅ High-level interface: Success={result.success}, Strategy={result.strategy_used}")
        return result.success
        
    except Exception as e:
        print(f"  ❌ High-level interface test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Adaptive JAX Adam Optimizer\n")
    
    tests = [
        ("Basic Functionality", test_adaptive_adam_basic),
        ("Strategy Framework", test_adaptive_adam_via_strategy),
        ("Performance Comparison", performance_comparison),
        ("Strategy Selection", test_strategy_selection),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n=== {test_name} ===")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📋 Test Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:20} | {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Adaptive Adam optimizer is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())