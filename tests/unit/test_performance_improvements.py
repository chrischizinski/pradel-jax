#!/usr/bin/env python3
"""
Test script to validate performance improvements.
Compares original vs optimized likelihood computation.
"""

import numpy as np
import time
import pandas as pd
import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel
from pradel_jax.models.pradel_optimized import OptimizedPradelModel

def create_test_data(n_individuals: int, n_occasions: int = 9, seed: int = 42):
    """Create synthetic test data."""
    np.random.seed(seed)
    
    # Create synthetic capture histories
    capture_prob = 0.3
    capture_matrix = np.random.binomial(1, capture_prob, (n_individuals, n_occasions))
    
    # Create DataFrame in expected format
    data = []
    for i in range(n_individuals):
        record = {
            'individual': f'ind_{i}',
            'gender': np.random.choice([0, 1, 2]),
            'age_1': np.random.normal(2, 1),
            'tier_1': np.random.choice([0, 1, 2])
        }
        
        # Add occasion data
        for j in range(n_occasions):
            record[f'occasion_{j+1}'] = capture_matrix[i, j]
        
        data.append(record)
    
    return pd.DataFrame(data)

def benchmark_model_performance(model, data_context, sample_sizes):
    """Benchmark model performance across different sample sizes."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model.__class__.__name__}")
    print(f"{'='*60}")
    
    results = []
    
    for n in sample_sizes:
        print(f"\n📊 Testing with {n:,} individuals...")
        
        # Create subset of data
        subset_data = create_test_data(n)
        temp_file = f"temp_test_{n}.csv"
        subset_data.to_csv(temp_file, index=False)
        
        try:
            # Load data
            data_ctx = pj.load_data(temp_file)
            
            # Create simple model specification
            formula_spec = pj.create_simple_spec(phi="~1 + age_1", p="~1", f="~1 + gender")
            
            # Build design matrices
            design_matrices = model.build_design_matrices(formula_spec, data_ctx)
            initial_params = model.get_initial_parameters(data_ctx, design_matrices)
            
            # Time likelihood computation (multiple runs for accuracy)
            n_runs = 10
            times = []
            
            for run in range(n_runs):
                start_time = time.time()
                ll = model.log_likelihood(initial_params, data_ctx, design_matrices)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            print(f"   ⏱️  Average time: {avg_time*1000:.1f} ± {std_time*1000:.1f} ms")
            print(f"   📈 Time per individual: {avg_time/n*1000:.3f} ms")
            print(f"   🔢 Log-likelihood: {ll:.2f}")
            
            results.append({
                'model': model.__class__.__name__,
                'n_individuals': n,
                'avg_time_ms': avg_time * 1000,
                'time_per_individual_ms': avg_time / n * 1000,
                'log_likelihood': ll
            })
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            
        finally:
            # Cleanup
            import os
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    return results

def test_scaling_performance():
    """Test performance scaling with different sample sizes."""
    sample_sizes = [100, 500, 1000, 2000]
    
    print("🧪 Performance Testing: Original vs Optimized Pradel Model")
    print("=" * 80)
    
    # Test original model
    original_model = PradelModel()
    original_results = benchmark_model_performance(original_model, None, sample_sizes)
    
    # Test optimized model
    optimized_model = OptimizedPradelModel()
    optimized_results = benchmark_model_performance(optimized_model, None, sample_sizes)
    
    # Compare results
    print(f"\n{'='*80}")
    print("📊 PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    print(f"{'Size':<8} {'Original (ms)':<15} {'Optimized (ms)':<16} {'Speedup':<10} {'Scaling':<10}")
    print("-" * 80)
    
    speedups = []
    for orig, opt in zip(original_results, optimized_results):
        if orig['n_individuals'] == opt['n_individuals']:
            speedup = orig['avg_time_ms'] / opt['avg_time_ms']
            speedups.append(speedup)
            
            # Calculate scaling factor
            if len(speedups) > 1:
                scaling = opt['time_per_individual_ms'] / optimized_results[0]['time_per_individual_ms']
            else:
                scaling = 1.0
            
            print(f"{orig['n_individuals']:<8} {orig['avg_time_ms']:<15.1f} {opt['avg_time_ms']:<16.1f} "
                  f"{speedup:<10.1f}x {scaling:<10.2f}x")
    
    print("-" * 80)
    print(f"Average speedup: {np.mean(speedups):.1f}x")
    
    # Estimate full dataset performance
    if len(optimized_results) >= 2:
        # Use last two points to estimate scaling
        r1, r2 = optimized_results[-2], optimized_results[-1]
        scaling_factor = (r2['time_per_individual_ms'] / r1['time_per_individual_ms']) / (r2['n_individuals'] / r1['n_individuals'])
        
        print(f"\n🔮 PERFORMANCE PROJECTIONS")
        print(f"{'='*50}")
        
        # Project performance for larger datasets
        base_time = optimized_results[-1]['time_per_individual_ms']
        base_size = optimized_results[-1]['n_individuals']
        
        projections = [5000, 25000, 111000]
        
        for proj_size in projections:
            # Assume slightly worse than linear scaling
            scaling = (proj_size / base_size) ** scaling_factor
            proj_time_per_ind = base_time * scaling
            total_time_sec = proj_time_per_ind * proj_size / 1000
            
            print(f"{proj_size:,} individuals:")
            print(f"  Single model: {total_time_sec:.1f} seconds ({total_time_sec/60:.1f} minutes)")
            print(f"  64 models: {total_time_sec * 64 / 3600:.1f} hours")
            print(f"  64 models (8 parallel): {total_time_sec * 64 / 3600 / 8:.1f} hours")
            print()

def main():
    """Run performance tests."""
    try:
        test_scaling_performance()
        
        print("\n✅ Performance testing completed!")
        print("\n💡 Key Improvements:")
        print("   1. Vectorized likelihood computation (eliminates Python loops)")
        print("   2. JAX JIT compilation for maximum speed")
        print("   3. Parallel processing across CPU cores")
        print("   4. Memory-efficient operations")
        
    except Exception as e:
        print(f"❌ Performance test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()