#!/usr/bin/env python3

import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.models import PradelModel

def benchmark_strategy(strategy, objective_fn, initial_params, bounds, context, n_runs=3):
    """Benchmark a single optimization strategy."""
    results = []
    
    for i in range(n_runs):
        start_time = time.perf_counter()
        try:
            result = optimize_model(
                objective_function=objective_fn,
                initial_parameters=initial_params,
                context=context,
                bounds=bounds,
                preferred_strategy=strategy
            )
            end_time = time.perf_counter()
            
            results.append({
                'run': i + 1,
                'success': result.success,
                'time': end_time - start_time,
                'fun_value': result.fun,
                'iterations': getattr(result, 'nit', None),
                'function_evaluations': getattr(result, 'nfev', None)
            })
        except Exception as e:
            end_time = time.perf_counter()
            results.append({
                'run': i + 1,
                'success': False,
                'time': end_time - start_time,
                'fun_value': None,
                'iterations': None,
                'function_evaluations': None,
                'error': str(e)
            })
    
    # Calculate summary statistics
    successful_runs = [r for r in results if r['success']]
    
    if successful_runs:
        times = [r['time'] for r in successful_runs]
        fun_values = [r['fun_value'] for r in successful_runs]
        
        summary = {
            'strategy': strategy.value,
            'success_rate': len(successful_runs) / n_runs,
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'best_fun_value': min(fun_values),
            'avg_fun_value': sum(fun_values) / len(fun_values),
            'total_runs': n_runs,
            'successful_runs': len(successful_runs)
        }
    else:
        summary = {
            'strategy': strategy.value,
            'success_rate': 0.0,
            'avg_time': None,
            'min_time': None,
            'max_time': None,
            'best_fun_value': None,
            'avg_fun_value': None,
            'total_runs': n_runs,
            'successful_runs': 0
        }
    
    return summary, results

def main():
    """Run production benchmark suite."""
    print("🚀 Pradel-JAX Production Benchmark Suite")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Timestamp: {timestamp}")
    
    # Load data
    print("📊 Loading data...")
    data_context = pj.load_data("data/dipper_dataset.csv")
    print(f"Data: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
    
    # Create simple model specification
    formula_spec = pj.create_simple_spec(
        phi="~1",  # Constant survival
        p="~1",    # Constant detection
        f="~1"     # Constant recruitment
    )
    
    # Setup model
    model = PradelModel()
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    initial_params = model.get_initial_parameters(data_context, design_matrices)
    bounds = model.get_parameter_bounds(data_context, design_matrices)
    
    def objective(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    # Test strategies (reduced set for speed)
    strategies = [
        OptimizationStrategy.SCIPY_LBFGS,
        OptimizationStrategy.SCIPY_SLSQP, 
        OptimizationStrategy.MULTI_START
    ]
    
    print(f"🧪 Testing {len(strategies)} optimization strategies...")
    print(f"Strategies: {', '.join([s.value for s in strategies])}")
    print()
    
    all_results = []
    
    for strategy in strategies:
        print(f"Testing {strategy.value}...")
        start = time.perf_counter()
        
        summary, detailed = benchmark_strategy(
            strategy=strategy,
            objective_fn=objective,
            initial_params=initial_params,
            bounds=bounds,
            context=data_context,
            n_runs=3
        )
        
        elapsed = time.perf_counter() - start
        
        print(f"  ✅ Success rate: {summary['success_rate']:.1%}")
        if summary['success_rate'] > 0:
            print(f"  ⏱️  Average time: {summary['avg_time']:.3f}s")
            print(f"  🎯 Best objective: {summary['best_fun_value']:.2f}")
        print(f"  🕐 Total time: {elapsed:.1f}s")
        print()
        
        summary['strategy_total_time'] = elapsed
        summary['timestamp'] = timestamp
        all_results.append(summary)
    
    # Generate performance summary
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Sort by success rate then by time
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(['success_rate', 'avg_time'], ascending=[False, True])
    
    print("Strategy Performance (sorted by success rate, then speed):")
    for _, row in results_df.iterrows():
        if row['success_rate'] > 0:
            print(f"  {row['strategy']:12} | Success: {row['success_rate']:5.1%} | Time: {row['avg_time']:6.3f}s | Best: {row['best_fun_value']:8.2f}")
        else:
            print(f"  {row['strategy']:12} | Success: {row['success_rate']:5.1%} | FAILED")
    
    # Save results
    results_dir = Path("tests/benchmarks")
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"production_benchmark_{timestamp}.json"
    summary_file = results_dir / f"production_summary_{timestamp}.csv"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    results_df.to_csv(summary_file, index=False)
    
    print(f"\n📁 Results saved:")
    print(f"  JSON: {results_file}")
    print(f"  CSV:  {summary_file}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    successful_strategies = results_df[results_df['success_rate'] > 0]
    if len(successful_strategies) > 0:
        best_strategy = successful_strategies.iloc[0]
        print(f"🏆 Best overall strategy: {best_strategy['strategy']}")
        print(f"   Success rate: {best_strategy['success_rate']:.1%}")
        if best_strategy['avg_time'] is not None:
            print(f"   Average time: {best_strategy['avg_time']:.3f}s")
        
        if len(successful_strategies) > 1:
            fastest = successful_strategies.loc[successful_strategies['avg_time'].idxmin()]
            if fastest['strategy'] != best_strategy['strategy']:
                print(f"⚡ Fastest strategy: {fastest['strategy']} ({fastest['avg_time']:.3f}s)")
    else:
        print("❌ No strategies succeeded consistently")
    
    print(f"\n✅ Benchmark complete!")
    total_time = sum(row['strategy_total_time'] for row in all_results)
    print(f"Total benchmark time: {total_time:.1f}s")

if __name__ == "__main__":
    main()