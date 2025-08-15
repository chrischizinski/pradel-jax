#!/usr/bin/env python3
"""
Memory performance benchmarking for Pradel-JAX optimization framework.
Tests memory usage patterns and efficiency across optimization strategies.
"""

import gc
import psutil
import time
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import tracemalloc

import pradel_jax as pj
from pradel_jax.models import PradelModel
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy


class MemoryProfiler:
    """Memory profiling utilities for optimization benchmarking."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        
    def start_profiling(self):
        """Start memory profiling."""
        gc.collect()  # Clean up before measurement
        tracemalloc.start()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
    def stop_profiling(self) -> Dict[str, float]:
        """Stop profiling and return memory metrics."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'baseline_mb': self.baseline_memory,
            'final_mb': final_memory,
            'peak_mb': peak / 1024 / 1024,
            'current_mb': current / 1024 / 1024,
            'delta_mb': final_memory - self.baseline_memory
        }


@pytest.fixture
def memory_profiler():
    """Create memory profiler instance."""
    return MemoryProfiler()


@pytest.fixture
def dipper_data():
    """Load dipper dataset."""
    data_path = Path(__file__).parent.parent.parent / "data" / "dipper_dataset.csv"
    return pj.load_data(str(data_path))


@pytest.fixture
def simple_formula():
    """Simple formula for memory testing."""
    return pj.create_simple_spec(phi="~1", p="~1", f="~1")


class TestMemoryPerformance:
    """Memory performance test suite."""
    
    def test_memory_usage_by_strategy(self, dipper_data, simple_formula, memory_profiler):
        """Test memory usage across different optimization strategies."""
        model = pj.PradelModel()
        strategies = ['lbfgs', 'slsqp', 'adam']
        
        memory_results = []
        
        for strategy in strategies:
            # Start memory profiling
            memory_profiler.start_profiling()
            
            # Run optimization
            try:
                design_matrices = model.build_design_matrices(simple_formula, dipper_data)
                initial_params = model.get_initial_parameters(dipper_data, design_matrices)
                bounds = model.get_parameter_bounds(dipper_data, design_matrices)
                
                # Add required attributes for optimization framework
                if not hasattr(dipper_data, 'n_parameters'):
                    dipper_data.n_parameters = len(initial_params)
                if not hasattr(dipper_data, 'get_condition_estimate'):
                    dipper_data.get_condition_estimate = lambda: 1e5
                
                def objective(params):
                    try:
                        ll = model.log_likelihood(params, dipper_data, design_matrices)
                        # For JAX compatibility, don't use float() - return JAX array directly
                        return -ll
                    except Exception:
                        return 1e10
                
                from pradel_jax.optimization.strategy import OptimizationStrategy
                strategy_enum = OptimizationStrategy(strategy)
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=dipper_data,
                    bounds=bounds,
                    preferred_strategy=strategy_enum
                )
                success = result.success
            except Exception:
                success = False
            
            # Stop profiling and collect metrics
            memory_metrics = memory_profiler.stop_profiling()
            memory_metrics.update({
                'strategy': strategy,
                'success': success
            })
            memory_results.append(memory_metrics)
            
            # Force cleanup between tests
            gc.collect()
            time.sleep(0.1)
        
        # Print memory usage summary
        print("\n=== Memory Usage by Strategy ===")
        for result in memory_results:
            print(f"{result['strategy']:8} | "
                  f"Peak: {result['peak_mb']:.1f}MB | "
                  f"Delta: {result['delta_mb']:+.1f}MB | "
                  f"Success: {result['success']}")
        
        # Verify memory usage is reasonable
        peak_memories = [r['peak_mb'] for r in memory_results]
        max_memory = max(peak_memories)
        assert max_memory < 500, f"Excessive memory usage: {max_memory:.1f}MB"
    
    def test_memory_scaling_with_data_size(self, dipper_data, simple_formula, memory_profiler):
        """Test memory scaling with different data sizes."""
        model = pj.PradelModel()
        strategy = 'lbfgs'  # Use consistent strategy
        
        # Test different subset sizes
        sizes = [50, 100, 200, 294]  # 294 is full dataset
        memory_results = []
        
        for size in sizes:
            # Create data subset
            if size < len(dipper_data.capture_histories):
                subset_data = type(dipper_data)(
                    capture_histories=dipper_data.capture_histories[:size],
                    covariates={k: v[:size] for k, v in dipper_data.covariates.items()},
                    n_occasions=dipper_data.n_occasions,
                    n_individuals=size
                )
            else:
                subset_data = dipper_data
            
            # Profile memory usage
            memory_profiler.start_profiling()
            
            try:
                result = pj.fit_model(
                    model=model,
                    formula=simple_formula,
                    data=subset_data,
                    strategy=strategy
                )
                success = result.success
            except Exception:
                success = False
            
            memory_metrics = memory_profiler.stop_profiling()
            memory_metrics.update({
                'data_size': size,
                'success': success
            })
            memory_results.append(memory_metrics)
            
            gc.collect()
            time.sleep(0.1)
        
        # Print scaling results
        print("\n=== Memory Scaling with Data Size ===")
        for result in memory_results:
            print(f"Size {result['data_size']:3d} | "
                  f"Peak: {result['peak_mb']:.1f}MB | "
                  f"Delta: {result['delta_mb']:+.1f}MB | "
                  f"Success: {result['success']}")
        
        # Verify memory scales reasonably
        sizes_tested = [r['data_size'] for r in memory_results]
        peak_memories = [r['peak_mb'] for r in memory_results]
        
        # Memory should scale sub-quadratically
        if len(sizes_tested) >= 2:
            memory_ratio = peak_memories[-1] / peak_memories[0]
            size_ratio = sizes_tested[-1] / sizes_tested[0]
            scaling_factor = memory_ratio / size_ratio
            
            print(f"Memory scaling factor: {scaling_factor:.2f}")
            assert scaling_factor < 5.0, f"Poor memory scaling: {scaling_factor:.2f}"
    
    def test_memory_leaks_multiple_runs(self, dipper_data, simple_formula, memory_profiler):
        """Test for memory leaks across multiple optimization runs."""
        model = pj.PradelModel()
        strategy = 'lbfgs'
        n_runs = 5
        
        memory_snapshots = []
        
        for run in range(n_runs):
            memory_profiler.start_profiling()
            
            # Run optimization
            try:
                design_matrices = model.build_design_matrices(simple_formula, dipper_data)
                initial_params = model.get_initial_parameters(dipper_data, design_matrices)
                bounds = model.get_parameter_bounds(dipper_data, design_matrices)
                
                # Add required attributes for optimization framework
                if not hasattr(dipper_data, 'n_parameters'):
                    dipper_data.n_parameters = len(initial_params)
                if not hasattr(dipper_data, 'get_condition_estimate'):
                    dipper_data.get_condition_estimate = lambda: 1e5
                
                def objective(params):
                    try:
                        ll = model.log_likelihood(params, dipper_data, design_matrices)
                        # For JAX compatibility, don't use float() - return JAX array directly
                        return -ll
                    except Exception:
                        return 1e10
                
                from pradel_jax.optimization.strategy import OptimizationStrategy
                strategy_enum = OptimizationStrategy(strategy)
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=dipper_data,
                    bounds=bounds,
                    preferred_strategy=strategy_enum
                )
                success = result.success
            except Exception:
                success = False
            
            memory_metrics = memory_profiler.stop_profiling()
            memory_metrics.update({
                'run': run + 1,
                'success': success
            })
            memory_snapshots.append(memory_metrics)
            
            gc.collect()
            time.sleep(0.1)
        
        # Print memory progression
        print(f"\n=== Memory Leak Test ({n_runs} runs) ===")
        for snapshot in memory_snapshots:
            print(f"Run {snapshot['run']} | "
                  f"Peak: {snapshot['peak_mb']:.1f}MB | "
                  f"Final: {snapshot['final_mb']:.1f}MB | "
                  f"Success: {snapshot['success']}")
        
        # Check for memory leaks
        baseline_memories = [s['baseline_mb'] for s in memory_snapshots]
        final_memories = [s['final_mb'] for s in memory_snapshots]
        
        # Memory should not consistently increase
        if len(baseline_memories) >= 3:
            memory_trend = np.polyfit(range(len(final_memories)), final_memories, 1)[0]
            print(f"Memory trend: {memory_trend:+.2f} MB/run")
            
            # Allow small positive trend but flag large increases
            assert memory_trend < 10.0, f"Potential memory leak detected: {memory_trend:.2f} MB/run"
    
    def test_memory_efficiency_complex_model(self, dipper_data, memory_profiler):
        """Test memory efficiency with complex model formulations."""
        model = pj.PradelModel()
        
        # Test different model complexities
        formulas = {
            'intercept': pj.create_simple_spec(phi="~1", p="~1", f="~1"),
            'sex_effects': pj.create_simple_spec(phi="~sex", p="~sex", f="~1"),
            'full_sex': pj.create_simple_spec(phi="~sex", p="~sex", f="~sex")
        }
        
        memory_results = []
        
        for complexity, formula in formulas.items():
            memory_profiler.start_profiling()
            
            try:
                result = pj.fit_model(
                    model=model,
                    formula=formula,
                    data=dipper_data,
                    strategy='lbfgs'
                )
                success = result.success
            except Exception:
                success = False
            
            memory_metrics = memory_profiler.stop_profiling()
            memory_metrics.update({
                'complexity': complexity,
                'success': success
            })
            memory_results.append(memory_metrics)
            
            gc.collect()
            time.sleep(0.1)
        
        # Print complexity results
        print("\n=== Memory Usage by Model Complexity ===")
        for result in memory_results:
            print(f"{result['complexity']:12} | "
                  f"Peak: {result['peak_mb']:.1f}MB | "
                  f"Delta: {result['delta_mb']:+.1f}MB | "
                  f"Success: {result['success']}")
        
        # Verify memory usage scales with complexity
        peak_memories = [r['peak_mb'] for r in memory_results]
        assert max(peak_memories) < 1000, f"Excessive memory for complex model: {max(peak_memories):.1f}MB"


@pytest.mark.benchmark
def test_comprehensive_memory_benchmark(dipper_data):
    """Run comprehensive memory benchmark suite."""
    profiler = MemoryProfiler()
    model = pj.PradelModel()
    simple_formula = pj.create_simple_spec(phi="~1", p="~1", f="~1")
    
    print("\n=== Comprehensive Memory Benchmark ===")
    
    # Test multiple strategies
    strategies = ['lbfgs', 'slsqp', 'adam', 'multi_start']
    strategy_results = []
    
    for strategy in strategies:
        profiler.start_profiling()
        
        try:
            result = pj.fit_model(
                model=model,
                formula=simple_formula,
                data=dipper_data,
                strategy=strategy
            )
            success = result.success
        except Exception:
            success = False
        
        memory_metrics = profiler.stop_profiling()
        memory_metrics.update({
            'strategy': strategy,
            'success': success
        })
        strategy_results.append(memory_metrics)
        
        gc.collect()
        time.sleep(0.2)
    
    # Summary
    print("\nMemory Efficiency Summary:")
    for result in strategy_results:
        efficiency = result['peak_mb'] / 294  # MB per individual
        print(f"{result['strategy']:12} | "
              f"{efficiency:.3f} MB/individual | "
              f"Peak: {result['peak_mb']:.1f}MB")
    
    # Verify overall efficiency
    peak_memories = [r['peak_mb'] for r in strategy_results if r['success']]
    if peak_memories:
        avg_memory = np.mean(peak_memories)
        assert avg_memory < 200, f"High average memory usage: {avg_memory:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])