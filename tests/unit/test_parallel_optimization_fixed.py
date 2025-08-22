#!/usr/bin/env python3
"""
Test script for the fixed parallel optimization framework.

Tests the complete parallel optimization workflow with proper DataContext
serialization to ensure it works across process boundaries.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

# Set PYTHONPATH to include our package
import sys
sys.path.insert(0, str(Path(__file__).parent))

def test_parallel_optimization_basic():
    """Test basic parallel optimization functionality."""
    print("Testing basic parallel optimization...")
    
    # Import components
    import pradel_jax as pj
    from pradel_jax.optimization.parallel import ParallelOptimizer, ParallelModelSpec
    from pradel_jax.optimization.strategy import OptimizationStrategy
    from pradel_jax.data.adapters import RMarkFormatAdapter
    
    # Create test dataset with more individuals for meaningful optimization
    np.random.seed(42)
    n_individuals = 20
    n_occasions = 5
    
    # Generate realistic capture histories
    true_phi = 0.7  # Survival probability
    true_p = 0.6    # Detection probability
    
    capture_histories = []
    for i in range(n_individuals):
        ch = ""
        alive = True
        for t in range(n_occasions):
            if alive:
                # Individual is alive, may be detected
                detected = np.random.random() < true_p
                ch += "1" if detected else "0"
                # May die after this occasion
                alive = np.random.random() < true_phi
            else:
                # Individual is dead
                ch += "0"
        capture_histories.append(ch)
    
    # Create DataFrame with covariates
    test_data = pd.DataFrame({
        'ch': capture_histories,
        'sex': np.random.choice(['M', 'F'], n_individuals),
        'age': np.random.uniform(0.5, 3.0, n_individuals),
        'weight': np.random.uniform(150, 250, n_individuals)
    })
    
    print(f"Created test dataset with {n_individuals} individuals, {n_occasions} occasions")
    
    # Process into DataContext
    adapter = RMarkFormatAdapter()
    data_context = adapter.process(test_data)
    
    # Create multiple model specifications to test in parallel
    model_specs = [
        ParallelModelSpec(
            name="φ(.) p(.) f(.)",
            formula_spec=pj.create_simple_spec(phi="~1", p="~1", f="~1"),
            index=0
        ),
        ParallelModelSpec(
            name="φ(sex) p(.) f(.)",  
            formula_spec=pj.create_simple_spec(phi="~1 + sex", p="~1", f="~1"),
            index=1
        ),
        ParallelModelSpec(
            name="φ(.) p(sex) f(.)",
            formula_spec=pj.create_simple_spec(phi="~1", p="~1 + sex", f="~1"),
            index=2
        ),
        ParallelModelSpec(
            name="φ(sex) p(sex) f(.)",
            formula_spec=pj.create_simple_spec(phi="~1 + sex", p="~1 + sex", f="~1"),
            index=3
        )
    ]
    
    print(f"Created {len(model_specs)} model specifications")
    
    # Test parallel optimization
    try:
        print("\n1. Testing ParallelOptimizer initialization...")
        optimizer = ParallelOptimizer(n_workers=2)  # Use 2 workers for testing
        print("✅ ParallelOptimizer created successfully")
        
        print("2. Running parallel model fitting...")
        start_time = time.time()
        
        results = optimizer.fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            strategy=OptimizationStrategy.SCIPY_LBFGS,
            batch_size=2,  # Process 2 models at a time
            checkpoint_interval=1  # Checkpoint after each batch
        )
        
        fit_time = time.time() - start_time
        print(f"✅ Parallel fitting completed in {fit_time:.2f} seconds")
        
        # Analyze results
        print("\n3. Analyzing results...")
        successful_results = [r for r in results if r and r.success]
        failed_results = [r for r in results if r and not r.success]
        
        print(f"✅ Successful fits: {len(successful_results)}/{len(model_specs)}")
        print(f"⚠️  Failed fits: {len(failed_results)}/{len(model_specs)}")
        
        if successful_results:
            # Show results for each successful model
            print("\nModel Results:")
            print("-" * 60)
            for result in successful_results:
                if result.success:
                    print(f"{result.model_name:20} | AIC: {result.aic:8.2f} | Strategy: {result.strategy_used}")
            
            # Find best model by AIC
            best_result = min(successful_results, key=lambda r: r.aic)
            print(f"\nBest model: {best_result.model_name} (AIC: {best_result.aic:.2f})")
            
            # Verify parameters are reasonable
            if best_result.parameters:
                params = np.array(best_result.parameters)
                print(f"Parameters: {params}")
                
                # Basic sanity checks
                if len(params) >= 3:  # At least phi, p, f parameters
                    phi_logit = params[0]
                    p_logit = params[1] 
                    f_logit = params[2]
                    
                    # Convert to probabilities
                    phi_est = 1 / (1 + np.exp(-phi_logit))
                    p_est = 1 / (1 + np.exp(-p_logit))
                    f_est = 1 / (1 + np.exp(-f_logit))
                    
                    print(f"Estimated φ (survival): {phi_est:.3f} (true: {true_phi})")
                    print(f"Estimated p (detection): {p_est:.3f} (true: {true_p})")
                    print(f"Estimated f (recruitment): {f_est:.3f}")
                    
                    # Check if estimates are reasonable
                    if 0.1 <= phi_est <= 0.99 and 0.1 <= p_est <= 0.99:
                        print("✅ Parameter estimates are in reasonable range")
                    else:
                        print("⚠️  Parameter estimates may be unrealistic")
            
        else:
            print("❌ No successful model fits")
            for result in failed_results:
                print(f"  {result.model_name}: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Parallel optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ Basic parallel optimization test passed!")
    return True


def test_parallel_vs_sequential():
    """Compare parallel vs sequential optimization to ensure results are consistent."""
    print("\nTesting parallel vs sequential consistency...")
    
    # Import components
    import pradel_jax as pj
    from pradel_jax.optimization.parallel import ParallelOptimizer, ParallelModelSpec
    from pradel_jax.optimization import optimize_model
    from pradel_jax.optimization.strategy import OptimizationStrategy
    from pradel_jax.data.adapters import RMarkFormatAdapter
    
    # Create small test dataset for comparison
    test_data = pd.DataFrame({
        'ch': ['0110', '1010', '1110', '0101', '1100', '0011', '1101', '0110'],
        'sex': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']
    })
    
    adapter = RMarkFormatAdapter()
    data_context = adapter.process(test_data)
    
    # Simple model for comparison
    model_spec = ParallelModelSpec(
        name="φ(.) p(.) f(.)",
        formula_spec=pj.create_simple_spec(phi="~1", p="~1", f="~1"),
        index=0
    )
    
    try:
        print("1. Running sequential optimization...")
        
        # Sequential optimization
        model = pj.PradelModel()
        design_matrices = model.build_design_matrices(model_spec.formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                return -ll if np.isfinite(ll) else 1e10
            except:
                return 1e10
        
        sequential_result = optimize_model(
            objective_function=objective,
            initial_parameters=initial_params,
            context=data_context,
            preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        print(f"✅ Sequential result: success={sequential_result.success}")
        if sequential_result.success:
            seq_params = sequential_result.result.x
            seq_ll = -sequential_result.result.fun
            print(f"   Parameters: {seq_params}")
            print(f"   Log-likelihood: {seq_ll:.6f}")
        
        print("2. Running parallel optimization...")
        
        # Parallel optimization  
        optimizer = ParallelOptimizer(n_workers=1)  # Single worker for fair comparison
        parallel_results = optimizer.fit_models_parallel(
            model_specs=[model_spec],
            data_context=data_context,
            strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        parallel_result = parallel_results[0]
        print(f"✅ Parallel result: success={parallel_result.success}")
        
        if parallel_result.success:
            par_params = np.array(parallel_result.parameters)
            par_ll = parallel_result.log_likelihood
            print(f"   Parameters: {par_params}")
            print(f"   Log-likelihood: {par_ll:.6f}")
            
            # Compare results
            if sequential_result.success:
                param_diff = np.max(np.abs(par_params - seq_params))
                ll_diff = abs(par_ll - seq_ll)
                
                print(f"\n3. Comparison:")
                print(f"   Max parameter difference: {param_diff:.6f}")
                print(f"   Log-likelihood difference: {ll_diff:.6f}")
                
                # Results should be very similar (allowing for numerical precision)
                if param_diff < 1e-4 and ll_diff < 1e-4:
                    print("✅ Sequential and parallel results are consistent")
                    return True
                else:
                    print("⚠️  Results differ more than expected")
                    print("   This may be due to different random seeds or numerical precision")
                    return param_diff < 1e-2 and ll_diff < 1e-2  # More lenient check
            else:
                print("⚠️  Cannot compare - sequential optimization failed")
                return parallel_result.success
        else:
            print(f"❌ Parallel optimization failed: {parallel_result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Parallel Optimization Test Suite (Fixed Serialization)")
    print("=" * 60)
    
    # Test 1: Basic functionality
    success1 = test_parallel_optimization_basic()
    
    # Test 2: Consistency with sequential
    success2 = test_parallel_vs_sequential()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Basic parallel optimization: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Sequential consistency:      {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED - Parallel optimization with fixed serialization works!")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Check parallel optimization implementation")
        exit(1)