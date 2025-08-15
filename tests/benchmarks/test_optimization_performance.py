#!/usr/bin/env python3
"""
Comprehensive performance benchmarking suite for Pradel-JAX optimization framework.
Tests optimization strategy performance across different scenarios and datasets.
"""

import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.models import PradelModel


class PerformanceBenchmarker:
    """Performance benchmarking framework for optimization strategies."""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        
    def benchmark_strategy(self, 
                          strategy: str,
                          data_context,
                          formula_spec,
                          model: PradelModel,
                          n_runs: int = 3) -> Dict[str, Any]:
        """Benchmark a specific optimization strategy."""
        times = []
        successes = []
        aics = []
        
        for run in range(n_runs):
            start_time = time.perf_counter()
            
            try:
                # Build design matrices and get optimization setup
                design_matrices = model.build_design_matrices(formula_spec, data_context)
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                # Add required attributes to data context for optimization framework
                if not hasattr(data_context, 'n_parameters'):
                    data_context.n_parameters = len(initial_params)
                if not hasattr(data_context, 'get_condition_estimate'):
                    data_context.get_condition_estimate = lambda: 1e5
                
                def objective(params):
                    try:
                        ll = model.log_likelihood(params, data_context, design_matrices)
                        # For JAX compatibility, don't use float() - return JAX array directly
                        return -ll
                    except Exception:
                        return 1e10
                
                strategy_enum = OptimizationStrategy(strategy)
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=data_context,
                    bounds=bounds,
                    preferred_strategy=strategy_enum
                )
                
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
                successes.append(result.success)
                
                if result.success:
                    # Calculate AIC from final objective value
                    final_nll = result.result.fun  # scipy optimization result
                    n_params = len(initial_params)
                    aic = 2 * final_nll + 2 * n_params
                    aics.append(aic)
                else:
                    aics.append(float('inf'))
                    
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
                successes.append(False)
                aics.append(float('inf'))
        
        return {
            'strategy': strategy,
            'n_runs': n_runs,
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'success_rate': np.mean(successes),
            'avg_aic': np.mean([aic for aic in aics if aic != float('inf')]) if any(aic != float('inf') for aic in aics) else float('inf'),
            'best_aic': np.min(aics) if any(aic != float('inf') for aic in aics) else float('inf')
        }


@pytest.fixture
def dipper_data():
    """Load dipper dataset for benchmarking."""
    data_path = Path(__file__).parent.parent.parent / "data" / "dipper_dataset.csv"
    return pj.load_data(str(data_path))


@pytest.fixture
def formula_specs():
    """Create test formula specifications of varying complexity."""
    return {
        'simple': pj.create_simple_spec(phi="~1", p="~1", f="~1"),
        'sex_effects': pj.create_simple_spec(phi="~sex", p="~sex", f="~1"),
        'complex': pj.create_simple_spec(phi="~sex", p="~sex", f="~sex")
    }


@pytest.fixture
def benchmarker():
    """Create benchmarker instance."""
    return PerformanceBenchmarker()


class TestOptimizationPerformance:
    """Test suite for optimization performance benchmarking."""
    
    def test_strategy_comparison_simple_model(self, dipper_data, formula_specs, benchmarker):
        """Benchmark all optimization strategies on simple model."""
        model = pj.PradelModel()
        formula = formula_specs['simple']
        
        strategies = ['scipy_lbfgs', 'scipy_slsqp', 'jax_adam', 'multi_start']
        results = []
        
        for strategy in strategies:
            result = benchmarker.benchmark_strategy(
                strategy=strategy,
                data_context=dipper_data,
                formula_spec=formula,
                model=model,
                n_runs=3
            )
            results.append(result)
            
        # Verify all strategies complete
        assert len(results) == len(strategies)
        
        # At least one strategy should succeed
        success_rates = [r['success_rate'] for r in results]
        assert max(success_rates) > 0, "No optimization strategy succeeded"
        
        # Print performance summary
        print("\n=== Simple Model Strategy Performance ===")
        for result in results:
            print(f"{result['strategy']:12} | "
                  f"Time: {result['avg_time']:.3f}±{result['std_time']:.3f}s | "
                  f"Success: {result['success_rate']:.1%} | "
                  f"AIC: {result['best_aic']:.2f}")
    
    def test_strategy_comparison_complex_model(self, dipper_data, formula_specs, benchmarker):
        """Benchmark strategies on complex model with covariates."""
        model = pj.PradelModel()
        formula = formula_specs['sex_effects']
        
        strategies = ['scipy_lbfgs', 'scipy_slsqp', 'multi_start']
        results = []
        
        for strategy in strategies:
            result = benchmarker.benchmark_strategy(
                strategy=strategy,
                data_context=dipper_data,
                formula_spec=formula,
                model=model,
                n_runs=2  # Fewer runs for complex models
            )
            results.append(result)
            
        # Print performance summary
        print("\n=== Complex Model Strategy Performance ===")
        for result in results:
            print(f"{result['strategy']:12} | "
                  f"Time: {result['avg_time']:.3f}±{result['std_time']:.3f}s | "
                  f"Success: {result['success_rate']:.1%} | "
                  f"AIC: {result['best_aic']:.2f}")
    
    def test_scaling_performance(self, dipper_data, formula_specs, benchmarker):
        """Test performance scaling with dataset size."""
        model = pj.PradelModel()
        formula = formula_specs['simple']
        
        # Test different data sizes  
        sizes = [100, len(dipper_data.capture_histories)]  # Up to full dataset size
        strategy = 'scipy_lbfgs'  # Use fastest strategy
        
        results = []
        for size in sizes:
            # Create subset if needed
            if size < len(dipper_data.capture_histories):
                subset_data = type(dipper_data)(
                    capture_histories=dipper_data.capture_histories[:size],
                    covariates={k: v[:size] for k, v in dipper_data.covariates.items()},
                    n_occasions=dipper_data.n_occasions,
                    n_individuals=size
                )
            else:
                subset_data = dipper_data
                
            result = benchmarker.benchmark_strategy(
                strategy=strategy,
                data_context=subset_data,
                formula_spec=formula,
                model=model,
                n_runs=2
            )
            result['data_size'] = size
            results.append(result)
        
        # Print scaling results
        print("\n=== Performance Scaling ===")
        for result in results:
            print(f"Size {result['data_size']:3d} | "
                  f"Time: {result['avg_time']:.3f}s | "
                  f"Success: {result['success_rate']:.1%}")
    
    def test_convergence_reliability(self, dipper_data, formula_specs, benchmarker):
        """Test convergence reliability across multiple runs."""
        model = pj.PradelModel()
        formula = formula_specs['simple']
        
        # Test with more runs to assess reliability
        result = benchmarker.benchmark_strategy(
            strategy='scipy_lbfgs',
            data_context=dipper_data,
            formula_spec=formula,
            model=model,
            n_runs=10
        )
        
        print("\n=== Convergence Reliability (10 runs) ===")
        print(f"Success rate: {result['success_rate']:.1%}")
        print(f"Time variation: {result['avg_time']:.3f}±{result['std_time']:.3f}s")
        print(f"CV: {result['std_time']/result['avg_time']*100:.1f}%")
        
        # High success rate expected for simple model
        assert result['success_rate'] >= 0.8, f"Low success rate: {result['success_rate']:.1%}"


@pytest.mark.benchmark
def test_comprehensive_benchmark_suite(dipper_data, formula_specs):
    """Run comprehensive benchmark suite and save results."""
    benchmarker = PerformanceBenchmarker()
    model = pj.PradelModel()
    
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test configurations
    test_configs = [
        ('simple', 'scipy_lbfgs'),
        ('simple', 'scipy_slsqp'),
        ('simple', 'jax_adam'),
        ('simple', 'multi_start'),
        ('sex_effects', 'scipy_lbfgs'),
        ('sex_effects', 'scipy_slsqp'),
        ('sex_effects', 'multi_start'),
    ]
    
    print(f"\n=== Comprehensive Benchmark Suite ===")
    print(f"Timestamp: {timestamp}")
    
    for formula_name, strategy in test_configs:
        print(f"\nTesting {formula_name} model with {strategy} strategy...")
        
        result = benchmarker.benchmark_strategy(
            strategy=strategy,
            data_context=dipper_data,
            formula_spec=formula_specs[formula_name],
            model=model,
            n_runs=3
        )
        
        result.update({
            'formula_complexity': formula_name,
            'timestamp': timestamp,
            'dataset': 'dipper',
            'dataset_size': len(dipper_data.capture_histories)
        })
        
        all_results.append(result)
        
        print(f"  Result: {result['avg_time']:.3f}s, "
              f"Success: {result['success_rate']:.1%}, "
              f"AIC: {result['best_aic']:.2f}")
    
    # Save results
    results_dir = Path(__file__).parent.parent.parent / "tests" / "benchmarks"
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"benchmark_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary report
    df = pd.DataFrame(all_results)
    summary_file = results_dir / f"benchmark_summary_{timestamp}.csv"
    df.to_csv(summary_file, index=False)
    
    print(f"\nResults saved to:")
    print(f"  {results_file}")
    print(f"  {summary_file}")
    
    # Basic validation
    assert len(all_results) == len(test_configs)
    assert all(r['avg_time'] > 0 for r in all_results)


if __name__ == "__main__":
    # Run benchmarks directly
    pytest.main([__file__, "-v", "-s", "--tb=short"])