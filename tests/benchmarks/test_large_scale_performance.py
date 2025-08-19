#!/usr/bin/env python3
"""
Large-scale performance benchmarking suite for Pradel-JAX optimization framework.
Tests JAX Adam's scalability advantages on 50k+ individual datasets.
"""

import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import psutil
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.models import PradelModel
from pradel_jax.data.adapters import DataContext, CovariateInfo


@dataclass
class BenchmarkResult:
    """Structured benchmark result."""
    strategy: str
    dataset_size: int
    n_occasions: int
    avg_time: float
    std_time: float
    peak_memory_mb: float
    success_rate: float
    avg_aic: float
    convergence_iterations: Optional[int] = None
    memory_efficiency: Optional[float] = None


class LargeScaleDataGenerator:
    """Generate synthetic capture-recapture datasets for large-scale testing."""
    
    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)
        
    def generate_synthetic_dataset(self, 
                                 n_individuals: int,
                                 n_occasions: int = 7,
                                 detection_prob: float = 0.6,
                                 survival_prob: float = 0.75,
                                 recruitment_prob: float = 0.2) -> DataContext:
        """Generate large synthetic capture-recapture dataset."""
        
        # Generate capture histories using realistic survival/detection model
        all_capture_histories = []
        all_covariates = {
            'sex': [],
            'age': []
        }
        
        for i in range(n_individuals):
            # Generate individual covariates
            sex = self.rng.choice(['M', 'F'])
            age = self.rng.choice(['juvenile', 'adult'], p=[0.3, 0.7])
            
            # Adjust probabilities based on covariates
            sex_effect = 1.1 if sex == 'M' else 1.0
            age_effect = 0.9 if age == 'juvenile' else 1.0
            
            individual_survival = min(0.95, survival_prob * sex_effect * age_effect)
            individual_detection = min(0.9, detection_prob * sex_effect)
            
            # Simulate capture history
            history = []
            alive = True
            first_capture = False
            
            for occasion in range(n_occasions):
                if not alive:
                    history.append(0)
                    continue
                    
                # Detection process
                if self.rng.random() < individual_detection:
                    history.append(1)
                    first_capture = True
                else:
                    history.append(0)
                
                # Survival process (don't apply on last occasion)
                if occasion < n_occasions - 1:
                    if first_capture and self.rng.random() > individual_survival:
                        alive = False
            
            # Only include individuals that were captured at least once
            if sum(history) > 0:
                all_capture_histories.append(history)
                all_covariates['sex'].append(sex)
                all_covariates['age'].append(age)
        
        # Convert to arrays
        import jax.numpy as jnp
        
        capture_matrix = jnp.array(all_capture_histories)
        n_captured = len(all_capture_histories)
        
        # Create covariate arrays
        covariates = {}
        covariate_info = {}
        
        # Sex covariate (categorical)
        sex_values = jnp.array([1 if sex == 'M' else 0 for sex in all_covariates['sex']])
        covariates['sex'] = sex_values
        covariate_info['sex'] = CovariateInfo(
            name='sex',
            dtype='categorical',
            is_categorical=True,
            levels=['F', 'M']
        )
        
        # Age covariate (categorical) 
        age_values = jnp.array([1 if age == 'adult' else 0 for age in all_covariates['age']])
        covariates['age'] = age_values
        covariate_info['age'] = CovariateInfo(
            name='age', 
            dtype='categorical',
            is_categorical=True,
            levels=['juvenile', 'adult']
        )
        
        # Create DataContext
        return DataContext(
            capture_matrix=capture_matrix,
            covariates=covariates,
            covariate_info=covariate_info,
            n_individuals=n_captured,
            n_occasions=n_occasions,
            occasion_names=[f'occasion_{i+1}' for i in range(n_occasions)]
        )


class MemoryProfiler:
    """Profile memory usage during optimization."""
    
    def __init__(self):
        self.peak_memory = 0
        self.initial_memory = 0
        
    def start(self):
        """Start memory profiling."""
        gc.collect()  # Force garbage collection
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        
    def update(self):
        """Update peak memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def get_peak_usage(self) -> float:
        """Get peak memory usage above baseline."""
        return self.peak_memory - self.initial_memory


class LargeScaleBenchmarker:
    """Large-scale performance benchmarking framework."""
    
    def __init__(self):
        self.data_generator = LargeScaleDataGenerator()
        self.results = []
        
    def benchmark_scalability(self,
                            strategy: str,
                            dataset_sizes: List[int],
                            n_occasions: int = 7,
                            n_runs: int = 3) -> List[BenchmarkResult]:
        """Benchmark strategy across different dataset sizes."""
        
        results = []
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi="~sex", p="~sex", f="~1")
        
        for size in dataset_sizes:
            print(f"\nBenchmarking {strategy} on {size:,} individuals...")
            
            # Generate dataset
            print(f"  Generating synthetic dataset...")
            dataset = self.data_generator.generate_synthetic_dataset(
                n_individuals=size,
                n_occasions=n_occasions
            )
            
            print(f"  Generated dataset: {dataset.n_individuals:,} captured individuals")
            
            # Run benchmark
            run_results = []
            memory_usage = []
            
            for run in range(n_runs):
                profiler = MemoryProfiler()
                profiler.start()
                
                start_time = time.perf_counter()
                
                try:
                    # Setup optimization
                    design_matrices = model.build_design_matrices(formula_spec, dataset)
                    initial_params = model.get_initial_parameters(dataset, design_matrices)
                    bounds = model.get_parameter_bounds(dataset, design_matrices)
                    
                    # Add required attributes
                    if not hasattr(dataset, 'n_parameters'):
                        dataset.n_parameters = len(initial_params)
                    if not hasattr(dataset, 'get_condition_estimate'):
                        dataset.get_condition_estimate = lambda: max(1e5, size * 10)
                    
                    def objective(params):
                        try:
                            profiler.update()
                            ll = model.log_likelihood(params, dataset, design_matrices)
                            return -ll
                        except Exception:
                            return 1e10
                    
                    # Optimize
                    strategy_enum = OptimizationStrategy(strategy)
                    result = optimize_model(
                        objective_function=objective,
                        initial_parameters=initial_params,
                        context=dataset,
                        bounds=bounds,
                        preferred_strategy=strategy_enum
                    )
                    
                    elapsed = time.perf_counter() - start_time
                    peak_memory = profiler.get_peak_usage()
                    
                    run_results.append({
                        'time': elapsed,
                        'success': result.success,
                        'aic': 2 * result.result.fun + 2 * len(initial_params) if result.success else float('inf'),
                        'iterations': getattr(result.result, 'nit', None) if result.success else None
                    })
                    memory_usage.append(peak_memory)
                    
                    print(f"    Run {run+1}: {elapsed:.2f}s, Success: {result.success}, Memory: {peak_memory:.1f}MB")
                    
                except Exception as e:
                    elapsed = time.perf_counter() - start_time
                    peak_memory = profiler.get_peak_usage()
                    
                    run_results.append({
                        'time': elapsed,
                        'success': False,
                        'aic': float('inf'),
                        'iterations': None
                    })
                    memory_usage.append(peak_memory)
                    
                    print(f"    Run {run+1}: FAILED ({elapsed:.2f}s) - {str(e)[:100]}")
                
                # Force cleanup
                gc.collect()
            
            # Aggregate results
            times = [r['time'] for r in run_results]
            successes = [r['success'] for r in run_results]
            aics = [r['aic'] for r in run_results if r['aic'] != float('inf')]
            iterations = [r['iterations'] for r in run_results if r['iterations'] is not None]
            
            benchmark_result = BenchmarkResult(
                strategy=strategy,
                dataset_size=size,
                n_occasions=n_occasions,
                avg_time=np.mean(times),
                std_time=np.std(times),
                peak_memory_mb=np.mean(memory_usage),
                success_rate=np.mean(successes),
                avg_aic=np.mean(aics) if aics else float('inf'),
                convergence_iterations=np.mean(iterations) if iterations else None,
                memory_efficiency=size / np.mean(memory_usage) if memory_usage else None  # individuals per MB
            )
            
            results.append(benchmark_result)
            
            print(f"  Summary: {benchmark_result.avg_time:.2f}±{benchmark_result.std_time:.2f}s, "
                  f"{benchmark_result.success_rate:.1%} success, {benchmark_result.peak_memory_mb:.1f}MB peak")
        
        return results
    
    def compare_strategies_large_scale(self,
                                     strategies: List[str],
                                     dataset_sizes: List[int],
                                     n_runs: int = 3) -> Dict[str, List[BenchmarkResult]]:
        """Compare multiple strategies on large-scale datasets."""
        
        all_results = {}
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"BENCHMARKING STRATEGY: {strategy.upper()}")
            print(f"{'='*60}")
            
            try:
                results = self.benchmark_scalability(
                    strategy=strategy,
                    dataset_sizes=dataset_sizes,
                    n_runs=n_runs
                )
                all_results[strategy] = results
                
            except Exception as e:
                print(f"FAILED to benchmark {strategy}: {e}")
                all_results[strategy] = []
        
        return all_results
    
    def save_results(self, results: Dict[str, List[BenchmarkResult]], output_dir: Path):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert to serializable format
        serializable_results = {}
        all_rows = []
        
        for strategy, strategy_results in results.items():
            serializable_results[strategy] = []
            
            for result in strategy_results:
                result_dict = {
                    'strategy': result.strategy,
                    'dataset_size': result.dataset_size,
                    'n_occasions': result.n_occasions,
                    'avg_time': result.avg_time,
                    'std_time': result.std_time,
                    'peak_memory_mb': result.peak_memory_mb,
                    'success_rate': result.success_rate,
                    'avg_aic': result.avg_aic,
                    'convergence_iterations': result.convergence_iterations,
                    'memory_efficiency': result.memory_efficiency,
                    'timestamp': timestamp
                }
                
                serializable_results[strategy].append(result_dict)
                all_rows.append(result_dict)
        
        # Save JSON results
        json_file = output_dir / f"large_scale_benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save CSV summary
        df = pd.DataFrame(all_rows)
        csv_file = output_dir / f"large_scale_benchmark_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Generate report
        report_file = output_dir / f"large_scale_benchmark_report_{timestamp}.md"
        self._generate_scalability_report(results, report_file, timestamp)
        
        return json_file, csv_file, report_file
    
    def _generate_scalability_report(self, results: Dict[str, List[BenchmarkResult]], 
                                   report_file: Path, timestamp: str):
        """Generate comprehensive scalability report."""
        
        with open(report_file, 'w') as f:
            f.write(f"# Large-Scale Pradel-JAX Benchmark Report\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Test Suite:** Large-scale dataset scalability analysis\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report evaluates the scalability of JAX Adam optimization compared to ")
            f.write("traditional scipy optimizers on large capture-recapture datasets (50k+ individuals).\n\n")
            
            # Performance comparison table
            f.write("## Strategy Performance Comparison\n\n")
            f.write("| Strategy | Dataset Size | Avg Time (s) | Memory (MB) | Success Rate | Memory Efficiency |\n")
            f.write("|----------|-------------|--------------|-------------|--------------|------------------|\n")
            
            for strategy, strategy_results in results.items():
                for result in strategy_results:
                    f.write(f"| {result.strategy} | {result.dataset_size:,} | "
                           f"{result.avg_time:.2f}±{result.std_time:.2f} | "
                           f"{result.peak_memory_mb:.1f} | {result.success_rate:.1%} | "
                           f"{result.memory_efficiency:.1f} ind/MB |\n")
            
            f.write("\n## Scalability Analysis\n\n")
            
            # Analyze scalability trends
            for strategy, strategy_results in results.items():
                f.write(f"### {strategy.replace('_', '-').title()}\n\n")
                
                if len(strategy_results) > 1:
                    # Calculate scaling factors
                    sizes = [r.dataset_size for r in strategy_results]
                    times = [r.avg_time for r in strategy_results]
                    memories = [r.peak_memory_mb for r in strategy_results]
                    
                    # Time scaling
                    size_ratio = max(sizes) / min(sizes)
                    time_ratio = max(times) / min(times)
                    time_scaling = np.log(time_ratio) / np.log(size_ratio)
                    
                    # Memory scaling
                    memory_ratio = max(memories) / min(memories)
                    memory_scaling = np.log(memory_ratio) / np.log(size_ratio)
                    
                    f.write(f"- **Time Complexity:** O(n^{time_scaling:.2f})\n")
                    f.write(f"- **Memory Complexity:** O(n^{memory_scaling:.2f})\n")
                    f.write(f"- **Largest Dataset:** {max(sizes):,} individuals\n")
                    f.write(f"- **Best Performance:** {min(times):.2f}s on {sizes[times.index(min(times))]:,} individuals\n")
                
                f.write("\n")
            
            f.write("## Key Findings\n\n")
            
            # Identify best performing strategy
            best_strategy = None
            best_time = float('inf')
            
            for strategy, strategy_results in results.items():
                avg_time_across_sizes = np.mean([r.avg_time for r in strategy_results if r.success_rate > 0])
                if avg_time_across_sizes < best_time:
                    best_time = avg_time_across_sizes
                    best_strategy = strategy
            
            f.write(f"- **Fastest Strategy:** {best_strategy} (avg {best_time:.2f}s across dataset sizes)\n")
            
            # Memory efficiency
            best_memory_strategy = None
            best_memory_eff = 0
            
            for strategy, strategy_results in results.items():
                avg_memory_eff = np.mean([r.memory_efficiency for r in strategy_results if r.memory_efficiency])
                if avg_memory_eff > best_memory_eff:
                    best_memory_eff = avg_memory_eff
                    best_memory_strategy = strategy
            
            f.write(f"- **Most Memory Efficient:** {best_memory_strategy} ({best_memory_eff:.1f} ind/MB)\n")
            
            # Reliability
            most_reliable = max(results.keys(), 
                              key=lambda s: np.mean([r.success_rate for r in results[s]]))
            avg_success = np.mean([r.success_rate for r in results[most_reliable]])
            f.write(f"- **Most Reliable:** {most_reliable} ({avg_success:.1%} success rate)\n")
            
            f.write(f"\n## Recommendations\n\n")
            f.write(f"Based on this large-scale analysis:\n\n")
            f.write(f"1. For datasets >50k individuals, use **{best_strategy}** for optimal speed\n")
            f.write(f"2. For memory-constrained environments, use **{best_memory_strategy}**\n")
            f.write(f"3. For production reliability, use **{most_reliable}**\n")
            f.write(f"4. JAX Adam shows {'excellent' if 'jax_adam' == best_strategy else 'good'} scalability characteristics\n")


@pytest.fixture
def large_scale_benchmarker():
    """Create large-scale benchmarker instance."""
    return LargeScaleBenchmarker()


@pytest.fixture
def dataset_sizes():
    """Define dataset sizes for scalability testing."""
    return [1000, 5000, 25000, 50000, 100000]


@pytest.fixture
def strategies():
    """Define strategies for comparison."""
    return ['scipy_lbfgs', 'jax_adam', 'multi_start']


class TestLargeScalePerformance:
    """Test suite for large-scale performance benchmarking."""
    
    @pytest.mark.slow
    def test_jax_adam_scalability(self, large_scale_benchmarker, dataset_sizes):
        """Test JAX Adam scalability on increasing dataset sizes."""
        
        results = large_scale_benchmarker.benchmark_scalability(
            strategy='jax_adam',
            dataset_sizes=dataset_sizes[:3],  # Start with smaller sizes
            n_runs=2
        )
        
        # Verify all tests complete
        assert len(results) == 3
        
        # Check for reasonable scaling
        times = [r.avg_time for r in results if r.success_rate > 0]
        if len(times) > 1:
            # Time should increase with dataset size but not exponentially
            time_growth = times[-1] / times[0] if times[0] > 0 else float('inf')
            size_growth = dataset_sizes[2] / dataset_sizes[0]
            
            # Expect sub-quadratic scaling
            assert time_growth < size_growth ** 1.5, f"Poor scaling: {time_growth:.2f}x time for {size_growth:.2f}x data"
    
    @pytest.mark.slow
    def test_strategy_comparison_50k(self, large_scale_benchmarker, strategies):
        """Compare strategies on 50k individual dataset."""
        
        large_dataset_size = [50000]
        
        results = large_scale_benchmarker.compare_strategies_large_scale(
            strategies=strategies,
            dataset_sizes=large_dataset_size,
            n_runs=2
        )
        
        # Verify all strategies were tested
        assert len(results) == len(strategies)
        
        # At least one strategy should succeed
        success_rates = []
        for strategy_results in results.values():
            if strategy_results:
                success_rates.extend([r.success_rate for r in strategy_results])
        
        assert max(success_rates) > 0, "No strategy succeeded on large dataset"
    
    @pytest.mark.comprehensive
    def test_comprehensive_large_scale_benchmark(self, large_scale_benchmarker, strategies, dataset_sizes):
        """Run comprehensive large-scale benchmark suite."""
        
        # Use subset of sizes for comprehensive test
        test_sizes = [5000, 25000, 50000]
        
        results = large_scale_benchmarker.compare_strategies_large_scale(
            strategies=strategies,
            dataset_sizes=test_sizes,
            n_runs=2
        )
        
        # Save results
        output_dir = Path(__file__).parent
        json_file, csv_file, report_file = large_scale_benchmarker.save_results(results, output_dir)
        
        print(f"\nLarge-scale benchmark results saved:")
        print(f"  JSON: {json_file}")
        print(f"  CSV: {csv_file}")
        print(f"  Report: {report_file}")
        
        # Basic validation
        assert json_file.exists()
        assert csv_file.exists() 
        assert report_file.exists()
        
        # Verify comprehensive results
        total_tests = len(strategies) * len(test_sizes)
        actual_results = sum(len(strategy_results) for strategy_results in results.values())
        
        # Allow for some failures but expect most tests to complete
        assert actual_results >= total_tests * 0.7, f"Too many test failures: {actual_results}/{total_tests}"


if __name__ == "__main__":
    # Run large-scale benchmarks directly
    benchmarker = LargeScaleBenchmarker()
    
    # Quick scalability test
    print("Running JAX Adam scalability test...")
    results = benchmarker.benchmark_scalability(
        strategy='jax_adam',
        dataset_sizes=[1000, 5000, 10000],
        n_runs=2
    )
    
    for result in results:
        print(f"Size {result.dataset_size:,}: {result.avg_time:.2f}s, "
              f"{result.success_rate:.1%} success, {result.peak_memory_mb:.1f}MB")