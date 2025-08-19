#!/usr/bin/env python3
"""
Focus on JAX Adam scalability analysis with 50k+ individual datasets.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, '.')

from tests.benchmarks.test_large_scale_performance import LargeScaleBenchmarker

def main():
    """Run JAX Adam scalability analysis."""
    
    print("="*80)
    print("JAX ADAM SCALABILITY ANALYSIS")
    print("Demonstrating JAX Adam's advantages on large datasets")
    print("="*80)
    
    benchmarker = LargeScaleBenchmarker()
    
    # Focus on JAX Adam scalability 
    strategies = ['scipy_lbfgs', 'jax_adam']  # Compare against scipy baseline
    dataset_sizes = [5000, 25000, 50000, 100000]  # Scale up to 100k
    
    print(f"\nConfiguration:")
    print(f"  Strategies: {strategies}")
    print(f"  Dataset sizes: {[f'{s:,}' for s in dataset_sizes]}")
    print(f"  Focus: JAX Adam scalability advantages")
    
    # Run focused benchmarks
    start_time = time.time()
    
    results = {}
    
    # Test JAX Adam scalability first
    print(f"\n{'='*60}")
    print(f"TESTING JAX ADAM SCALABILITY")
    print(f"{'='*60}")
    
    jax_results = benchmarker.benchmark_scalability(
        strategy='jax_adam',
        dataset_sizes=dataset_sizes,
        n_runs=2
    )
    results['jax_adam'] = jax_results
    
    # Test scipy L-BFGS for comparison (smaller sizes due to expected slowness)
    print(f"\n{'='*60}")
    print(f"TESTING SCIPY L-BFGS FOR COMPARISON")
    print(f"{'='*60}")
    
    scipy_sizes = [5000, 25000]  # Limit scipy to smaller sizes
    scipy_results = benchmarker.benchmark_scalability(
        strategy='scipy_lbfgs',
        dataset_sizes=scipy_sizes,
        n_runs=2
    )
    results['scipy_lbfgs'] = scipy_results
    
    # Save results
    output_dir = Path("tests/benchmarks")
    json_file, csv_file, report_file = benchmarker.save_results(results, output_dir)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"JAX ADAM SCALABILITY ANALYSIS COMPLETED in {elapsed/60:.1f} minutes")
    print(f"{'='*80}")
    
    # Analyze scalability
    print(f"\nJAX Adam Scalability Results:")
    print(f"{'Dataset Size':<12} {'Time (s)':<10} {'Memory (MB)':<12} {'Efficiency':<12}")
    print(f"{'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    
    for result in jax_results:
        print(f"{result.dataset_size:>8,} {result.avg_time:>8.1f}s {result.peak_memory_mb:>10.1f}MB "
              f"{result.memory_efficiency:>8.1f} i/MB")
    
    # Calculate scaling characteristics
    if len(jax_results) > 1:
        largest = jax_results[-1]
        smallest = jax_results[0]
        
        size_ratio = largest.dataset_size / smallest.dataset_size
        time_ratio = largest.avg_time / smallest.avg_time
        memory_ratio = largest.peak_memory_mb / smallest.peak_memory_mb
        
        time_scaling = np.log(time_ratio) / np.log(size_ratio)
        memory_scaling = np.log(memory_ratio) / np.log(size_ratio)
        
        print(f"\nJAX Adam Scaling Analysis:")
        print(f"  Dataset size increase: {size_ratio:.1f}x ({smallest.dataset_size:,} → {largest.dataset_size:,})")
        print(f"  Time increase: {time_ratio:.1f}x ({smallest.avg_time:.1f}s → {largest.avg_time:.1f}s)")
        print(f"  Memory increase: {memory_ratio:.1f}x ({smallest.peak_memory_mb:.1f}MB → {largest.peak_memory_mb:.1f}MB)")
        print(f"  Time complexity: O(n^{time_scaling:.2f})")
        print(f"  Memory complexity: O(n^{memory_scaling:.2f})")
    
    # Compare with scipy where available
    if scipy_results:
        print(f"\nPerformance Comparison (on {scipy_sizes[-1]:,} individuals):")
        
        scipy_large = scipy_results[-1] 
        jax_large = next((r for r in jax_results if r.dataset_size == scipy_sizes[-1]), None)
        
        if jax_large:
            speedup = scipy_large.avg_time / jax_large.avg_time
            memory_diff = scipy_large.peak_memory_mb - jax_large.peak_memory_mb
            
            print(f"  JAX Adam: {jax_large.avg_time:.1f}s, {jax_large.peak_memory_mb:.1f}MB")
            print(f"  Scipy L-BFGS: {scipy_large.avg_time:.1f}s, {scipy_large.peak_memory_mb:.1f}MB")
            print(f"  JAX Adam is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
            print(f"  JAX Adam uses {abs(memory_diff):.1f}MB {'less' if memory_diff > 0 else 'more'} memory")
    
    print(f"\nDetailed analysis: {report_file}")
    
    return results


if __name__ == "__main__":
    import numpy as np
    results = main()