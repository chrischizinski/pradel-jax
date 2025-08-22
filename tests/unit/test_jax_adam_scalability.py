#!/usr/bin/env python3
"""
Test JAX Adam adaptive optimization on large datasets to demonstrate scalability.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.models import PradelModel
from tests.benchmarks.test_large_scale_performance import LargeScaleDataGenerator, MemoryProfiler


def test_jax_adam_scalability():
    """Test JAX Adam adaptive on increasing dataset sizes."""
    
    print("=== JAX ADAM ADAPTIVE SCALABILITY TEST ===")
    
    generator = LargeScaleDataGenerator()
    model = PradelModel()
    formula_spec = pj.create_simple_spec(phi="~sex", p="~sex", f="~1")
    
    dataset_sizes = [1000, 5000, 25000]
    results = []
    
    for size in dataset_sizes:
        print(f"\nTesting {size:,} individuals...")
        
        # Generate dataset
        dataset = generator.generate_synthetic_dataset(
            n_individuals=size,
            n_occasions=7
        )
        
        print(f"Generated: {dataset.n_individuals:,} captured individuals")
        
        # Setup optimization
        design_matrices = model.build_design_matrices(formula_spec, dataset)
        initial_params = model.get_initial_parameters(dataset, design_matrices)
        bounds = model.get_parameter_bounds(dataset, design_matrices)
        
        # Add required attributes for context
        if not hasattr(dataset, 'n_parameters'):
            dataset.n_parameters = len(initial_params)
        if not hasattr(dataset, 'get_condition_estimate'):
            dataset.get_condition_estimate = lambda: max(1e5, size * 10)
        
        # Test different JAX Adam strategies
        strategies_to_test = [
            OptimizationStrategy.JAX_ADAM_ADAPTIVE,
            OptimizationStrategy.JAX_ADAM,
            OptimizationStrategy.SCIPY_LBFGS,  # For comparison
        ]
        
        for strategy in strategies_to_test:
            profiler = MemoryProfiler()
            profiler.start()
            
            start_time = time.perf_counter()
            
            try:
                def objective(params):
                    try:
                        profiler.update()
                        ll = model.log_likelihood(params, dataset, design_matrices)
                        return -ll
                    except Exception:
                        return 1e10
                
                # Optimize with strategy
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=dataset,
                    bounds=bounds,
                    preferred_strategy=strategy
                )
                
                elapsed = time.perf_counter() - start_time
                peak_memory = profiler.get_peak_usage()
                
                success = result.success
                aic = 2 * result.result.fun + 2 * len(initial_params) if success else float('inf')
                
                result_data = {
                    'strategy': strategy.value,
                    'dataset_size': size,
                    'time': elapsed,
                    'success': success,
                    'aic': aic,
                    'peak_memory_mb': peak_memory,
                    'memory_efficiency': size / peak_memory if peak_memory > 0 else 0
                }
                
                results.append(result_data)
                
                print(f"  {strategy.value}: {elapsed:.2f}s, Success: {success}, "
                      f"Memory: {peak_memory:.1f}MB, AIC: {aic:.1f}")
                
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                peak_memory = profiler.get_peak_usage()
                
                result_data = {
                    'strategy': strategy.value,
                    'dataset_size': size,
                    'time': elapsed,
                    'success': False,
                    'aic': float('inf'),
                    'peak_memory_mb': peak_memory,
                    'memory_efficiency': 0
                }
                
                results.append(result_data)
                
                print(f"  {strategy.value}: FAILED ({elapsed:.2f}s) - {str(e)[:100]}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    csv_file = f"jax_adam_scalability_test_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    # Generate summary
    print(f"\n=== SCALABILITY TEST RESULTS ===")
    print(f"Results saved to: {csv_file}")
    
    # Analyze by strategy
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        successful = strategy_data[strategy_data['success'] == True]
        
        if len(successful) > 0:
            print(f"\n{strategy.upper()}:")
            print(f"  Success rate: {len(successful)/len(strategy_data):.1%}")
            print(f"  Avg time: {successful['time'].mean():.2f}s")
            print(f"  Avg memory: {successful['peak_memory_mb'].mean():.1f}MB")
            print(f"  Avg memory efficiency: {successful['memory_efficiency'].mean():.1f} ind/MB")
            
            # Check scaling
            if len(successful) > 1:
                sizes = successful['dataset_size'].values
                times = successful['time'].values
                size_ratio = max(sizes) / min(sizes)
                time_ratio = max(times) / min(times)
                if size_ratio > 1:
                    scaling = np.log(time_ratio) / np.log(size_ratio)
                    print(f"  Time complexity: O(n^{scaling:.2f})")
        else:
            print(f"\n{strategy.upper()}: NO SUCCESSFUL RUNS")
    
    return results


if __name__ == "__main__":
    results = test_jax_adam_scalability()