#!/usr/bin/env python3
"""
Run comprehensive large-scale benchmarks for JAX Adam scalability analysis.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, '.')

from tests.benchmarks.test_large_scale_performance import LargeScaleBenchmarker

def main():
    """Run comprehensive large-scale benchmark suite."""
    
    print("="*80)
    print("LARGE-SCALE PRADEL-JAX BENCHMARK SUITE")
    print("Testing JAX Adam scalability on 50k+ individual datasets")
    print("="*80)
    
    benchmarker = LargeScaleBenchmarker()
    
    # Define test configurations
    strategies = ['scipy_lbfgs', 'jax_adam', 'multi_start']
    dataset_sizes = [1000, 5000, 25000, 50000]  # Scale up to 50k
    
    print(f"\nConfiguration:")
    print(f"  Strategies: {strategies}")
    print(f"  Dataset sizes: {dataset_sizes}")
    print(f"  Runs per test: 2")
    
    # Run comprehensive benchmarks
    start_time = time.time()
    
    try:
        results = benchmarker.compare_strategies_large_scale(
            strategies=strategies,
            dataset_sizes=dataset_sizes,
            n_runs=2
        )
        
        # Save results
        output_dir = Path("tests/benchmarks")
        json_file, csv_file, report_file = benchmarker.save_results(results, output_dir)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK SUITE COMPLETED in {elapsed/60:.1f} minutes")
        print(f"{'='*80}")
        
        print(f"\nResults saved to:")
        print(f"  📊 Summary: {csv_file}")  
        print(f"  📋 Report: {report_file}")
        print(f"  📁 Raw data: {json_file}")
        
        # Print quick summary
        print(f"\nQuick Performance Summary:")
        print(f"{'Strategy':<15} {'50k Dataset':<12} {'Memory Eff':<12} {'Success':<8}")
        print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*8}")
        
        for strategy, strategy_results in results.items():
            # Find 50k result
            large_result = None
            for result in strategy_results:
                if result.dataset_size >= 50000:
                    large_result = result
                    break
            
            if large_result:
                print(f"{strategy:<15} {large_result.avg_time:>8.1f}s     "
                      f"{large_result.memory_efficiency:>8.1f} i/MB {large_result.success_rate:>6.1%}")
            else:
                print(f"{strategy:<15} {'N/A':<12} {'N/A':<12} {'N/A':<8}")
        
        print(f"\nFor detailed analysis, see: {report_file}")
        
    except KeyboardInterrupt:
        print(f"\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()