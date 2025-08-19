#!/usr/bin/env python3
"""
Local performance monitoring script for Pradel-JAX optimization framework.

This script provides convenient local performance monitoring and regression detection
for developers working on optimization improvements.

Usage:
    python scripts/monitor_performance.py --help
    python scripts/monitor_performance.py --quick          # Quick performance check
    python scripts/monitor_performance.py --full           # Comprehensive monitoring
    python scripts/monitor_performance.py --baseline       # Create new baselines
    python scripts/monitor_performance.py --compare        # Compare with baselines
"""

import argparse
import sys
import time
import json
import psutil
import tracemalloc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pradel_jax as pj
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.models import PradelModel


class PerformanceMonitor:
    """Local performance monitoring for development workflow."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
        
    def log(self, message: str):
        """Log message if verbose mode enabled."""
        if self.verbose:
            print(message)
    
    def measure_strategy_performance(self, 
                                   strategy: str,
                                   data_context,
                                   formula_spec,
                                   model: PradelModel,
                                   n_runs: int = 3) -> Dict[str, Any]:
        """Measure performance for a single strategy."""
        
        self.log(f"📊 Testing {strategy} strategy...")
        
        times = []
        successes = []
        memories = []
        iterations = []
        
        for run in range(n_runs):
            # Start monitoring
            tracemalloc.start()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.perf_counter()
            
            try:
                # Setup optimization
                design_matrices = model.build_design_matrices(formula_spec, data_context)
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                # Add required attributes
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
                
                # Run optimization
                from pradel_jax.optimization import optimize_model
                strategy_enum = OptimizationStrategy(strategy)
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=data_context,
                    bounds=bounds,
                    preferred_strategy=strategy_enum
                )
                
                # Record metrics
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
                successes.append(result.success)
                
                # Memory usage
                end_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024
                memory_used = max(end_memory - start_memory, peak_memory)
                memories.append(memory_used)
                
                # Iterations
                iterations.append(result.result.nit if hasattr(result.result, 'nit') else 0)
                
            except Exception as e:
                self.log(f"  ⚠️  Run {run+1} failed: {e}")
                times.append(float('inf'))
                successes.append(False)
                memories.append(0)
                iterations.append(0)
                
            finally:
                tracemalloc.stop()
        
        # Calculate summary statistics
        valid_times = [t for t in times if t != float('inf')]
        
        result_summary = {
            'strategy': strategy,
            'n_runs': n_runs,
            'avg_time': sum(valid_times) / len(valid_times) if valid_times else float('inf'),
            'min_time': min(valid_times) if valid_times else float('inf'),
            'max_time': max(valid_times) if valid_times else float('inf'),
            'success_rate': sum(successes) / len(successes),
            'avg_memory_mb': sum(memories) / len(memories) if memories else 0,
            'avg_iterations': sum(iterations) / len(iterations) if iterations else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log results
        if result_summary['success_rate'] > 0:
            self.log(f"  ✅ Success rate: {result_summary['success_rate']:.1%}")
            self.log(f"  ⏱️  Time: {result_summary['avg_time']:.3f}s ({result_summary['min_time']:.3f}-{result_summary['max_time']:.3f}s)")
            self.log(f"  💾 Memory: {result_summary['avg_memory_mb']:.1f}MB")
            self.log(f"  🔄 Iterations: {result_summary['avg_iterations']:.0f}")
        else:
            self.log(f"  ❌ All runs failed")
        
        return result_summary
    
    def quick_check(self) -> Dict[str, Any]:
        """Quick performance check with essential strategies."""
        self.log("🚀 Running quick performance check...")
        
        # Load test data
        data_path = Path(__file__).parent.parent / "data" / "dipper_dataset.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Test data not found: {data_path}")
        
        data_context = pj.load_data(str(data_path))
        model = PradelModel()
        formula = pj.create_simple_spec(phi="~1", p="~1", f="~1")
        
        # Test core strategies
        strategies = ['scipy_lbfgs', 'hybrid']
        results = []
        
        for strategy in strategies:
            try:
                result = self.measure_strategy_performance(
                    strategy=strategy,
                    data_context=data_context,
                    formula_spec=formula,
                    model=model,
                    n_runs=2  # Quick check
                )
                results.append(result)
            except Exception as e:
                self.log(f"❌ Strategy {strategy} failed: {e}")
                results.append({
                    'strategy': strategy,
                    'error': str(e),
                    'success_rate': 0.0
                })
        
        summary = {
            'test_type': 'quick_check',
            'timestamp': datetime.now().isoformat(),
            'strategies_tested': len(strategies),
            'results': results,
            'overall_success_rate': sum(r.get('success_rate', 0) for r in results) / len(results)
        }
        
        self.log(f"\n📈 Quick Check Summary:")
        self.log(f"  Strategies tested: {summary['strategies_tested']}")
        self.log(f"  Overall success rate: {summary['overall_success_rate']:.1%}")
        
        return summary
    
    def comprehensive_monitor(self) -> Dict[str, Any]:
        """Comprehensive performance monitoring."""
        self.log("🔬 Running comprehensive performance monitoring...")
        
        # Load test data
        data_path = Path(__file__).parent.parent / "data" / "dipper_dataset.csv"
        data_context = pj.load_data(str(data_path))
        model = PradelModel()
        
        # Test configurations
        test_configs = [
            ('scipy_lbfgs', 'simple'),
            ('scipy_slsqp', 'simple'),
            ('hybrid', 'simple'),
            ('jax_adam', 'simple'),
            ('multi_start', 'simple'),
            ('scipy_lbfgs', 'moderate'),
            ('hybrid', 'moderate'),
        ]
        
        formulas = {
            'simple': pj.create_simple_spec(phi="~1", p="~1", f="~1"),
            'moderate': pj.create_simple_spec(phi="~sex", p="~sex", f="~1")
        }
        
        results = []
        
        for strategy, formula_complexity in test_configs:
            self.log(f"\n🧪 Testing {strategy} with {formula_complexity} formula...")
            
            try:
                result = self.measure_strategy_performance(
                    strategy=strategy,
                    data_context=data_context,
                    formula_spec=formulas[formula_complexity],
                    model=model,
                    n_runs=3
                )
                result['formula_complexity'] = formula_complexity
                results.append(result)
                
            except Exception as e:
                self.log(f"❌ Test failed: {e}")
                results.append({
                    'strategy': strategy,
                    'formula_complexity': formula_complexity,
                    'error': str(e),
                    'success_rate': 0.0
                })
        
        # Generate comprehensive summary
        successful_results = [r for r in results if r.get('success_rate', 0) > 0]
        
        summary = {
            'test_type': 'comprehensive',
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(test_configs),
            'successful_tests': len(successful_results),
            'results': results,
            'performance_summary': {
                'avg_time': sum(r['avg_time'] for r in successful_results) / len(successful_results) if successful_results else 0,
                'fastest_strategy': min(successful_results, key=lambda x: x['avg_time'])['strategy'] if successful_results else None,
                'most_reliable': max(successful_results, key=lambda x: x['success_rate'])['strategy'] if successful_results else None,
                'avg_memory': sum(r['avg_memory_mb'] for r in successful_results) / len(successful_results) if successful_results else 0
            }
        }
        
        # Print summary
        self.log(f"\n📊 Comprehensive Monitoring Summary:")
        self.log(f"  Total tests: {summary['total_tests']}")
        self.log(f"  Successful: {summary['successful_tests']}")
        self.log(f"  Average time: {summary['performance_summary']['avg_time']:.3f}s")
        self.log(f"  Fastest strategy: {summary['performance_summary']['fastest_strategy']}")
        self.log(f"  Most reliable: {summary['performance_summary']['most_reliable']}")
        
        return summary
    
    def compare_with_baseline(self, baseline_file: str = None) -> Dict[str, Any]:
        """Compare current performance with baseline."""
        self.log("📊 Comparing performance with baseline...")
        
        if baseline_file is None:
            baseline_file = str(Path(__file__).parent.parent / "tests" / "benchmarks" / "performance_baselines.json")
        
        baseline_path = Path(baseline_file)
        if not baseline_path.exists():
            self.log(f"❌ Baseline file not found: {baseline_path}")
            self.log("💡 Run with --baseline to create initial baselines")
            return {'error': 'No baseline file found'}
        
        # Load baselines
        with open(baseline_path, 'r') as f:
            baselines = json.load(f)
        
        # Run current performance tests
        current_results = self.quick_check()
        
        # Compare with baselines
        comparisons = []
        for result in current_results['results']:
            strategy = result['strategy']
            baseline_key = f"{strategy}_simple"  # Assuming simple formula for quick check
            
            if baseline_key in baselines:
                baseline = baselines[baseline_key]
                
                time_ratio = result['avg_time'] / baseline['avg_time_seconds']
                success_change = result['success_rate'] - baseline['success_rate']
                
                comparison = {
                    'strategy': strategy,
                    'current_time': result['avg_time'],
                    'baseline_time': baseline['avg_time_seconds'],
                    'time_ratio': time_ratio,
                    'current_success_rate': result['success_rate'],
                    'baseline_success_rate': baseline['success_rate'],
                    'success_change': success_change,
                    'regression_detected': time_ratio > 1.5 or success_change < -0.1
                }
                
                comparisons.append(comparison)
                
                # Log comparison
                if comparison['regression_detected']:
                    self.log(f"  ⚠️  {strategy}: Potential regression detected")
                    if time_ratio > 1.5:
                        self.log(f"    Time: {time_ratio:.2f}x slower")
                    if success_change < -0.1:
                        self.log(f"    Success rate: {success_change:.1%} lower")
                else:
                    self.log(f"  ✅ {strategy}: Performance within normal range")
        
        comparison_summary = {
            'test_type': 'baseline_comparison',
            'timestamp': datetime.now().isoformat(),
            'baseline_file': baseline_file,
            'comparisons': comparisons,
            'regressions_detected': sum(1 for c in comparisons if c['regression_detected']),
            'total_comparisons': len(comparisons)
        }
        
        self.log(f"\n📈 Baseline Comparison Summary:")
        self.log(f"  Comparisons made: {comparison_summary['total_comparisons']}")
        self.log(f"  Regressions detected: {comparison_summary['regressions_detected']}")
        
        return comparison_summary


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Pradel-JAX Performance Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/monitor_performance.py --quick
  python scripts/monitor_performance.py --full --output results.json
  python scripts/monitor_performance.py --compare
  python scripts/monitor_performance.py --baseline
        """
    )
    
    # Main action arguments
    parser.add_argument('--quick', action='store_true',
                       help='Run quick performance check')
    parser.add_argument('--full', action='store_true',
                       help='Run comprehensive performance monitoring')
    parser.add_argument('--compare', action='store_true',
                       help='Compare current performance with baseline')
    parser.add_argument('--baseline', action='store_true',
                       help='Create new performance baselines')
    
    # Configuration arguments
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Verbose output (default: True)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode (minimal output)')
    
    args = parser.parse_args()
    
    # Handle quiet mode
    verbose = args.verbose and not args.quiet
    
    # Create monitor
    monitor = PerformanceMonitor(verbose=verbose)
    
    # Determine action
    if args.quick:
        results = monitor.quick_check()
    elif args.full:
        results = monitor.comprehensive_monitor()
    elif args.compare:
        results = monitor.compare_with_baseline()
    elif args.baseline:
        # Create baselines using regression tester
        sys.path.append(str(Path(__file__).parent.parent / "tests" / "benchmarks"))
        from test_performance_regression import PerformanceRegressionTester
        
        print("🎯 Creating performance baselines...")
        tester = PerformanceRegressionTester()
        
        # Load test data
        data_path = Path(__file__).parent.parent / "data" / "dipper_dataset.csv"
        data_context = pj.load_data(str(data_path))
        
        formulas = {
            'simple': pj.create_simple_spec(phi="~1", p="~1", f="~1"),
            'moderate': pj.create_simple_spec(phi="~sex", p="~sex", f="~1")
        }
        
        model = PradelModel()
        
        # Core baseline strategies
        baseline_configs = [
            ('scipy_lbfgs', 'simple'),
            ('scipy_lbfgs', 'moderate'),
            ('scipy_slsqp', 'simple'),
            ('hybrid', 'simple'),
        ]
        
        for strategy, formula_complexity in baseline_configs:
            result = tester.measure_performance(
                strategy=strategy,
                data_context=data_context,
                formula_spec=formulas[formula_complexity],
                model=model
            )
            tester.update_baseline(result)
            print(f"✅ Baseline created: {strategy}/{formula_complexity}")
        
        results = {'baseline_creation': 'completed', 'strategies': len(baseline_configs)}
    else:
        # Default to quick check
        print("No action specified, running quick check...")
        results = monitor.quick_check()
    
    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if verbose:
            print(f"\n💾 Results saved to: {output_path}")
    
    # Exit with appropriate code
    if 'regressions_detected' in results and results['regressions_detected'] > 0:
        if verbose:
            print("\n⚠️  Performance regressions detected!")
        sys.exit(1)
    elif 'error' in results:
        if verbose:
            print(f"\n❌ Error: {results['error']}")
        sys.exit(1)
    else:
        if verbose:
            print("\n✅ Performance monitoring completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()