#!/usr/bin/env python3
"""
Performance Benchmarking for New Optimization Strategies

This module provides comprehensive performance testing for the hybrid and adaptive Adam
optimization strategies, comparing them against baseline optimizers across multiple
scenarios and datasets.

Key Performance Metrics:
1. Convergence speed and reliability
2. Solution quality (AIC, log-likelihood)
3. Robustness to different problem characteristics
4. Scalability with dataset size
5. Memory efficiency

Author: Claude Code
Date: August 2025
"""

import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import logging
import psutil
import tracemalloc

import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.models import PradelModel


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for optimization strategies."""
    strategy: str
    dataset_name: str
    formula_complexity: str
    
    # Execution metrics
    execution_time: float
    memory_peak_mb: float
    convergence_iterations: int
    
    # Solution quality
    final_aic: float
    final_log_likelihood: float
    convergence_success: bool
    
    # Robustness metrics
    parameter_stability: float  # Std dev across runs
    convergence_reliability: float  # Success rate across runs
    
    # Efficiency metrics
    time_per_iteration: float
    memory_efficiency: float  # MB per parameter
    
    # Relative performance
    relative_speed: float  # Compared to baseline
    relative_quality: float  # AIC improvement over baseline
    
    # Problem characteristics
    n_parameters: int
    n_individuals: int
    data_sparsity: float
    
    # Additional notes
    notes: List[str]


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report for optimization strategies."""
    benchmark_id: str
    timestamp: str
    
    # Test configuration
    strategies_tested: List[str]
    datasets_tested: List[str]
    formula_complexities: List[str]
    
    # Performance results
    performance_metrics: List[PerformanceMetrics]
    
    # Aggregate analysis
    performance_summary: Dict[str, Any]
    scalability_analysis: Dict[str, Any]
    robustness_analysis: Dict[str, Any]
    
    # Conclusions
    best_performing_strategy: str
    most_reliable_strategy: str
    most_efficient_strategy: str
    recommendations: List[str]


class NewOptimizerBenchmarker:
    """Specialized benchmarker for new optimization strategies."""
    
    def __init__(self):
        self.baseline_strategy = OptimizationStrategy.SCIPY_LBFGS
        self.new_strategies = [
            OptimizationStrategy.HYBRID,
            OptimizationStrategy.JAX_ADAM_ADAPTIVE
        ]
        self.results_history = []
    
    def benchmark_strategy_comprehensive(
        self,
        strategy: OptimizationStrategy,
        data_context,
        formula_spec,
        n_runs: int = 5,
        measure_memory: bool = True
    ) -> PerformanceMetrics:
        """
        Comprehensive benchmarking of a single optimization strategy.
        
        Measures execution time, memory usage, convergence reliability,
        and solution quality across multiple runs.
        """
        logger.info(f"Benchmarking {strategy.value} with {n_runs} runs")
        
        # Initialize metrics
        execution_times = []
        memory_peaks = []
        aics = []
        log_likelihoods = []
        successes = []
        iterations = []
        parameters_history = []
        
        model = PradelModel()
        
        for run_idx in range(n_runs):
            logger.debug(f"Run {run_idx + 1}/{n_runs}")
            
            # Memory tracking setup
            if measure_memory:
                tracemalloc.start()
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.perf_counter()
            
            try:
                # Build optimization setup
                design_matrices = model.build_design_matrices(formula_spec, data_context)
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                # Ensure data context has required attributes
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
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=data_context,
                    bounds=bounds,
                    preferred_strategy=strategy
                )
                
                elapsed_time = time.perf_counter() - start_time
                execution_times.append(elapsed_time)
                
                # Memory measurement
                if measure_memory:
                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    memory_peak = memory_after - memory_before
                    memory_peaks.append(max(0, memory_peak))
                    tracemalloc.stop()
                
                # Extract results
                if result.success:
                    final_params = result.result.x
                    final_nll = result.result.fun
                    n_params = len(initial_params)
                    aic = 2 * final_nll + 2 * n_params
                    
                    aics.append(aic)
                    log_likelihoods.append(-final_nll)
                    successes.append(True)
                    iterations.append(getattr(result.result, 'nit', 0))
                    parameters_history.append(final_params)
                    
                else:
                    successes.append(False)
                    aics.append(float('inf'))
                    log_likelihoods.append(float('-inf'))
                    iterations.append(0)
                    
            except Exception as e:
                logger.warning(f"Run {run_idx + 1} failed: {e}")
                execution_times.append(time.perf_counter() - start_time)
                successes.append(False)
                aics.append(float('inf'))
                log_likelihoods.append(float('-inf'))
                iterations.append(0)
                
                if measure_memory:
                    memory_peaks.append(0)
                    tracemalloc.stop()
        
        # Calculate aggregate metrics
        successful_runs = [i for i, s in enumerate(successes) if s]
        
        if not successful_runs:
            return PerformanceMetrics(
                strategy=strategy.value,
                dataset_name="unknown",
                formula_complexity="unknown",
                execution_time=np.mean(execution_times),
                memory_peak_mb=np.mean(memory_peaks) if memory_peaks else 0,
                convergence_iterations=0,
                final_aic=float('inf'),
                final_log_likelihood=float('-inf'),
                convergence_success=False,
                parameter_stability=float('inf'),
                convergence_reliability=0.0,
                time_per_iteration=float('inf'),
                memory_efficiency=float('inf'),
                relative_speed=float('inf'),
                relative_quality=float('inf'),
                n_parameters=data_context.n_parameters,
                n_individuals=data_context.n_individuals,
                data_sparsity=np.mean(data_context.capture_matrix == 0),
                notes=[f"All {n_runs} runs failed"]
            )
        
        # Successful run statistics
        successful_times = [execution_times[i] for i in successful_runs]
        successful_aics = [aics[i] for i in successful_runs]
        successful_lls = [log_likelihoods[i] for i in successful_runs]
        successful_iters = [iterations[i] for i in successful_runs]
        successful_params = [parameters_history[i] for i in successful_runs]
        
        # Parameter stability (coefficient of variation)
        parameter_stability = 0.0
        if len(successful_params) > 1:
            param_matrix = np.array(successful_params)
            param_stds = np.std(param_matrix, axis=0)
            param_means = np.abs(np.mean(param_matrix, axis=0))
            param_cvs = param_stds / (param_means + 1e-8)
            parameter_stability = np.mean(param_cvs)
        
        # Calculate metrics
        avg_time = np.mean(successful_times)
        avg_memory = np.mean(memory_peaks) if memory_peaks else 0
        avg_iterations = np.mean(successful_iters)
        best_aic = min(successful_aics)
        best_ll = max(successful_lls)
        convergence_reliability = len(successful_runs) / n_runs
        
        time_per_iteration = avg_time / max(avg_iterations, 1)
        memory_efficiency = avg_memory / max(data_context.n_parameters, 1)
        
        return PerformanceMetrics(
            strategy=strategy.value,
            dataset_name=getattr(data_context, 'name', 'unknown'),
            formula_complexity=self._assess_formula_complexity(formula_spec),
            execution_time=avg_time,
            memory_peak_mb=avg_memory,
            convergence_iterations=int(avg_iterations),
            final_aic=best_aic,
            final_log_likelihood=best_ll,
            convergence_success=convergence_reliability >= 0.8,
            parameter_stability=parameter_stability,
            convergence_reliability=convergence_reliability,
            time_per_iteration=time_per_iteration,
            memory_efficiency=memory_efficiency,
            relative_speed=1.0,  # Will be calculated relative to baseline
            relative_quality=1.0,  # Will be calculated relative to baseline
            n_parameters=data_context.n_parameters,
            n_individuals=data_context.n_individuals,
            data_sparsity=np.mean(data_context.capture_matrix == 0),
            notes=[]
        )
    
    def benchmark_scalability(
        self,
        strategy: OptimizationStrategy,
        base_data_context,
        formula_spec,
        size_factors: List[float] = [0.25, 0.5, 1.0, 2.0]
    ) -> Dict[str, Any]:
        """
        Test scalability of optimization strategy with varying dataset sizes.
        
        Creates subsets and supersets (with synthetic data) to test performance scaling.
        """
        logger.info(f"Testing scalability of {strategy.value}")
        
        scalability_results = []
        base_size = base_data_context.n_individuals
        
        for factor in size_factors:
            target_size = int(base_size * factor)
            logger.info(f"Testing with {target_size} individuals (factor: {factor})")
            
            # Create dataset with target size
            if factor <= 1.0:
                # Subset existing data
                subset_size = target_size
                test_context = type(base_data_context)(
                    capture_histories=base_data_context.capture_histories[:subset_size],
                    covariates={k: v[:subset_size] for k, v in base_data_context.covariates.items()},
                    n_occasions=base_data_context.n_occasions,
                    n_individuals=subset_size
                )
            else:
                # Extend with synthetic data (simplified approach)
                extra_size = target_size - base_size
                
                # Generate synthetic capture histories
                synthetic_histories = []
                for _ in range(extra_size):
                    # Simple synthetic generation based on observed patterns
                    p_capture = 0.6  # Average capture probability
                    history = [1 if np.random.random() < p_capture else 0 
                             for _ in range(base_data_context.n_occasions)]
                    # Ensure at least one capture
                    if sum(history) == 0:
                        history[np.random.randint(0, len(history))] = 1
                    synthetic_histories.append(history)
                
                # Combine original and synthetic data
                all_histories = list(base_data_context.capture_histories) + synthetic_histories
                
                # Extend covariates
                extended_covariates = {}
                for name, values in base_data_context.covariates.items():
                    # Repeat existing covariate patterns
                    extended_values = list(values)
                    for _ in range(extra_size):
                        extended_values.append(values[np.random.randint(0, len(values))])
                    extended_covariates[name] = np.array(extended_values)
                
                test_context = type(base_data_context)(
                    capture_histories=all_histories,
                    covariates=extended_covariates,
                    n_occasions=base_data_context.n_occasions,
                    n_individuals=target_size
                )
            
            # Benchmark at this size
            metrics = self.benchmark_strategy_comprehensive(
                strategy=strategy,
                data_context=test_context,
                formula_spec=formula_spec,
                n_runs=3,  # Fewer runs for scalability testing
                measure_memory=True
            )
            
            scalability_results.append({
                'size_factor': factor,
                'n_individuals': target_size,
                'execution_time': metrics.execution_time,
                'memory_peak_mb': metrics.memory_peak_mb,
                'convergence_reliability': metrics.convergence_reliability,
                'final_aic': metrics.final_aic,
                'time_per_individual': metrics.execution_time / target_size
            })
        
        # Analyze scaling behavior
        times = [r['execution_time'] for r in scalability_results]
        sizes = [r['n_individuals'] for r in scalability_results]
        
        # Fit scaling curve (simple linear regression on log scale)
        if len(times) > 2:
            log_times = np.log(times)
            log_sizes = np.log(sizes)
            scaling_coefficient = np.corrcoef(log_sizes, log_times)[0, 1]
            
            # Estimate scaling exponent
            slope, intercept = np.polyfit(log_sizes, log_times, 1)
            scaling_exponent = slope
        else:
            scaling_coefficient = 0.0
            scaling_exponent = 1.0
        
        return {
            'strategy': strategy.value,
            'results': scalability_results,
            'scaling_coefficient': scaling_coefficient,
            'scaling_exponent': scaling_exponent,
            'scaling_quality': self._assess_scaling_quality(scaling_exponent)
        }
    
    def benchmark_robustness(
        self,
        strategy: OptimizationStrategy,
        data_context,
        formula_specs: List[Any],
        perturbation_levels: List[float] = [0.0, 0.1, 0.2]
    ) -> Dict[str, Any]:
        """
        Test robustness of optimization strategy to different problem characteristics.
        
        Tests performance across different model complexities and data perturbations.
        """
        logger.info(f"Testing robustness of {strategy.value}")
        
        robustness_results = []
        
        for formula_spec in formula_specs:
            for perturbation in perturbation_levels:
                # Create perturbed data if needed
                if perturbation > 0:
                    perturbed_context = self._perturb_data_context(data_context, perturbation)
                else:
                    perturbed_context = data_context
                
                # Benchmark on this configuration
                metrics = self.benchmark_strategy_comprehensive(
                    strategy=strategy,
                    data_context=perturbed_context,
                    formula_spec=formula_spec,
                    n_runs=3
                )
                
                robustness_results.append({
                    'formula_complexity': self._assess_formula_complexity(formula_spec),
                    'perturbation_level': perturbation,
                    'convergence_reliability': metrics.convergence_reliability,
                    'execution_time': metrics.execution_time,
                    'parameter_stability': metrics.parameter_stability,
                    'final_aic': metrics.final_aic
                })
        
        # Analyze robustness
        reliability_scores = [r['convergence_reliability'] for r in robustness_results]
        stability_scores = [r['parameter_stability'] for r in robustness_results 
                          if r['parameter_stability'] != float('inf')]
        
        avg_reliability = np.mean(reliability_scores)
        avg_stability = np.mean(stability_scores) if stability_scores else float('inf')
        robustness_score = avg_reliability * (1.0 / (1.0 + avg_stability))
        
        return {
            'strategy': strategy.value,
            'results': robustness_results,
            'avg_reliability': avg_reliability,
            'avg_stability': avg_stability,
            'robustness_score': robustness_score,
            'robustness_level': self._assess_robustness_level(robustness_score)
        }
    
    def _assess_formula_complexity(self, formula_spec) -> str:
        """Assess complexity level of formula specification."""
        # Count non-intercept terms
        complexity_score = 0
        for param_formula in [formula_spec.phi, formula_spec.p, formula_spec.f]:
            # Extract formula string from ParameterFormula object
            formula_str = param_formula.formula_string if hasattr(param_formula, 'formula_string') else str(param_formula)
            if formula_str != "1":
                complexity_score += 1
                # Simple heuristic for interaction terms
                if "*" in formula_str or ":" in formula_str:
                    complexity_score += 2
        
        if complexity_score == 0:
            return "simple"
        elif complexity_score <= 2:
            return "moderate"
        elif complexity_score <= 4:
            return "complex"
        else:
            return "very_complex"
    
    def _assess_scaling_quality(self, scaling_exponent: float) -> str:
        """Assess scaling quality based on exponent."""
        if scaling_exponent < 1.2:
            return "excellent"
        elif scaling_exponent < 1.5:
            return "good"
        elif scaling_exponent < 2.0:
            return "acceptable"
        else:
            return "poor"
    
    def _assess_robustness_level(self, robustness_score: float) -> str:
        """Assess robustness level based on composite score."""
        if robustness_score > 0.8:
            return "excellent"
        elif robustness_score > 0.6:
            return "good"
        elif robustness_score > 0.4:
            return "acceptable"
        else:
            return "poor"
    
    def _perturb_data_context(self, data_context, perturbation_level: float):
        """Create perturbed version of data context for robustness testing."""
        # Simple perturbation: add noise to capture probabilities
        perturbed_matrix = data_context.capture_matrix.copy()
        
        # Add random noise to capture decisions
        n_perturbations = int(perturbation_level * perturbed_matrix.size)
        flat_indices = np.random.choice(perturbed_matrix.size, n_perturbations, replace=False)
        
        for flat_idx in flat_indices:
            i, j = np.unravel_index(flat_idx, perturbed_matrix.shape)
            perturbed_matrix[i, j] = 1 - perturbed_matrix[i, j]  # Flip capture decision
        
        # Create new data context with perturbed matrix
        perturbed_context = type(data_context)(
            capture_histories=[
                [int(perturbed_matrix[i, j]) for j in range(data_context.n_occasions)]
                for i in range(data_context.n_individuals)
            ],
            covariates=data_context.covariates.copy(),
            n_occasions=data_context.n_occasions,
            n_individuals=data_context.n_individuals
        )
        
        return perturbed_context


@pytest.fixture
def dipper_data():
    """Load dipper dataset for benchmarking."""
    data_path = Path(__file__).parent.parent.parent / "data" / "dipper_dataset.csv"
    data = pj.load_data(str(data_path))
    data.name = "dipper"  # Add name for reporting
    return data


@pytest.fixture
def formula_specs():
    """Create test formula specifications of varying complexity."""
    return {
        'simple': pj.create_simple_spec(phi="~1", p="~1", f="~1"),
        'moderate': pj.create_simple_spec(phi="~sex", p="~1", f="~1"),
        'complex': pj.create_simple_spec(phi="~sex", p="~sex", f="~1"),
        'very_complex': pj.create_simple_spec(phi="~sex", p="~sex", f="~sex")
    }


@pytest.fixture
def benchmarker():
    """Create benchmarker instance."""
    return NewOptimizerBenchmarker()


class TestNewOptimizerPerformance:
    """Performance test suite for new optimization strategies."""
    
    def test_hybrid_optimizer_performance(self, dipper_data, formula_specs, benchmarker):
        """Comprehensive performance test for HYBRID optimizer."""
        logger.info("=== Testing HYBRID Optimizer Performance ===")
        
        # Test on simple model
        metrics = benchmarker.benchmark_strategy_comprehensive(
            strategy=OptimizationStrategy.HYBRID,
            data_context=dipper_data,
            formula_spec=formula_specs['simple'],
            n_runs=5
        )
        
        # Performance assertions
        assert metrics.convergence_reliability >= 0.8, f"Low reliability: {metrics.convergence_reliability:.1%}"
        assert metrics.execution_time < 30.0, f"Slow execution: {metrics.execution_time:.1f}s"
        assert metrics.final_aic != float('inf'), "Failed to converge"
        
        print(f"\n🚀 HYBRID Performance Results:")
        print(f"   Execution Time: {metrics.execution_time:.2f}s")
        print(f"   Memory Peak: {metrics.memory_peak_mb:.1f} MB")
        print(f"   Convergence Reliability: {metrics.convergence_reliability:.1%}")
        print(f"   Parameter Stability: {metrics.parameter_stability:.4f}")
        print(f"   Final AIC: {metrics.final_aic:.2f}")
        print(f"   Time per Iteration: {metrics.time_per_iteration:.4f}s")
    
    def test_adaptive_adam_performance(self, dipper_data, formula_specs, benchmarker):
        """Comprehensive performance test for JAX_ADAM_ADAPTIVE optimizer."""
        logger.info("=== Testing ADAPTIVE ADAM Optimizer Performance ===")
        
        # Test on simple model
        metrics = benchmarker.benchmark_strategy_comprehensive(
            strategy=OptimizationStrategy.JAX_ADAM_ADAPTIVE,
            data_context=dipper_data,
            formula_spec=formula_specs['simple'],
            n_runs=5
        )
        
        # Performance assertions
        assert metrics.convergence_reliability >= 0.7, f"Low reliability: {metrics.convergence_reliability:.1%}"
        assert metrics.execution_time < 25.0, f"Slow execution: {metrics.execution_time:.1f}s"
        
        print(f"\n⚡ ADAPTIVE ADAM Performance Results:")
        print(f"   Execution Time: {metrics.execution_time:.2f}s")
        print(f"   Memory Peak: {metrics.memory_peak_mb:.1f} MB")
        print(f"   Convergence Reliability: {metrics.convergence_reliability:.1%}")
        print(f"   Parameter Stability: {metrics.parameter_stability:.4f}")
        print(f"   Final AIC: {metrics.final_aic:.2f}")
        print(f"   Iterations: {metrics.convergence_iterations}")
    
    def test_relative_performance_comparison(self, dipper_data, formula_specs, benchmarker):
        """Compare new optimizers against baseline performance."""
        logger.info("=== Relative Performance Comparison ===")
        
        strategies_to_test = [
            OptimizationStrategy.SCIPY_LBFGS,  # Baseline
            OptimizationStrategy.HYBRID,
            OptimizationStrategy.JAX_ADAM_ADAPTIVE
        ]
        
        results = {}
        
        for strategy in strategies_to_test:
            logger.info(f"Benchmarking {strategy.value}...")
            metrics = benchmarker.benchmark_strategy_comprehensive(
                strategy=strategy,
                data_context=dipper_data,
                formula_spec=formula_specs['simple'],
                n_runs=3
            )
            results[strategy.value] = metrics
        
        # Calculate relative performance
        baseline = results['scipy_lbfgs']
        baseline_time = baseline.execution_time
        baseline_aic = baseline.final_aic
        
        print(f"\n📊 Relative Performance Analysis:")
        print(f"{'Strategy':<20} {'Speed Ratio':<12} {'AIC Delta':<10} {'Reliability':<12} {'Status'}")
        print("=" * 70)
        
        for strategy_name, metrics in results.items():
            if strategy_name == 'scipy_lbfgs':
                speed_ratio = 1.0
                aic_delta = 0.0
            else:
                speed_ratio = baseline_time / metrics.execution_time if metrics.execution_time > 0 else float('inf')
                aic_delta = metrics.final_aic - baseline_aic if metrics.final_aic != float('inf') else float('inf')
            
            reliability = metrics.convergence_reliability
            
            # Status assessment
            if reliability >= 0.8 and speed_ratio >= 0.8 and abs(aic_delta) < 2.0:
                status = "✅ EXCELLENT"
            elif reliability >= 0.7 and speed_ratio >= 0.5:
                status = "👍 GOOD"
            elif reliability >= 0.5:
                status = "⚠️ ACCEPTABLE"
            else:
                status = "❌ POOR"
            
            print(f"{strategy_name:<20} {speed_ratio:>10.2f}x {aic_delta:>9.2f} {reliability:>10.1%} {status}")
        
        # Assertions for new optimizers
        hybrid_metrics = results.get('hybrid')
        adaptive_metrics = results.get('jax_adam_adaptive')
        
        if hybrid_metrics:
            hybrid_speed = baseline_time / hybrid_metrics.execution_time
            assert hybrid_speed >= 0.5, f"HYBRID too slow: {hybrid_speed:.2f}x baseline"
            assert hybrid_metrics.convergence_reliability >= 0.8, f"HYBRID unreliable: {hybrid_metrics.convergence_reliability:.1%}"
        
        if adaptive_metrics:
            adaptive_speed = baseline_time / adaptive_metrics.execution_time
            assert adaptive_speed >= 0.3, f"ADAPTIVE ADAM too slow: {adaptive_speed:.2f}x baseline"
            assert adaptive_metrics.convergence_reliability >= 0.7, f"ADAPTIVE ADAM unreliable: {adaptive_metrics.convergence_reliability:.1%}"
    
    def test_scalability_analysis(self, dipper_data, formula_specs, benchmarker):
        """Test scalability of new optimizers with different dataset sizes."""
        logger.info("=== Scalability Analysis ===")
        
        # Test HYBRID scalability
        hybrid_scalability = benchmarker.benchmark_scalability(
            strategy=OptimizationStrategy.HYBRID,
            base_data_context=dipper_data,
            formula_spec=formula_specs['simple'],
            size_factors=[0.5, 1.0, 1.5]  # Conservative scaling for testing
        )
        
        print(f"\n📈 HYBRID Scalability Results:")
        print(f"   Scaling Exponent: {hybrid_scalability['scaling_exponent']:.2f}")
        print(f"   Scaling Quality: {hybrid_scalability['scaling_quality']}")
        
        for result in hybrid_scalability['results']:
            print(f"   Size {result['n_individuals']}: "
                  f"{result['execution_time']:.2f}s, "
                  f"{result['convergence_reliability']:.1%} reliable")
        
        # Scalability assertions
        assert hybrid_scalability['scaling_exponent'] < 2.0, f"Poor HYBRID scaling: {hybrid_scalability['scaling_exponent']:.2f}"
        assert hybrid_scalability['scaling_quality'] in ['excellent', 'good', 'acceptable'], \
            f"Poor HYBRID scaling quality: {hybrid_scalability['scaling_quality']}"
    
    def test_robustness_analysis(self, dipper_data, formula_specs, benchmarker):
        """Test robustness of new optimizers to different problem characteristics."""
        logger.info("=== Robustness Analysis ===")
        
        # Test HYBRID robustness across formula complexities
        test_formulas = [formula_specs['simple'], formula_specs['moderate'], formula_specs['complex']]
        
        hybrid_robustness = benchmarker.benchmark_robustness(
            strategy=OptimizationStrategy.HYBRID,
            data_context=dipper_data,
            formula_specs=test_formulas,
            perturbation_levels=[0.0, 0.1]  # Conservative perturbation for testing
        )
        
        print(f"\n🛡️ HYBRID Robustness Results:")
        print(f"   Average Reliability: {hybrid_robustness['avg_reliability']:.1%}")
        print(f"   Average Stability: {hybrid_robustness['avg_stability']:.4f}")
        print(f"   Robustness Score: {hybrid_robustness['robustness_score']:.3f}")
        print(f"   Robustness Level: {hybrid_robustness['robustness_level']}")
        
        # Robustness assertions
        assert hybrid_robustness['avg_reliability'] >= 0.7, f"Poor HYBRID reliability: {hybrid_robustness['avg_reliability']:.1%}"
        assert hybrid_robustness['robustness_level'] in ['excellent', 'good', 'acceptable'], \
            f"Poor HYBRID robustness: {hybrid_robustness['robustness_level']}"


@pytest.mark.benchmark
def test_comprehensive_new_optimizer_benchmark(dipper_data, formula_specs):
    """
    Comprehensive benchmark suite for new optimization strategies.
    
    Generates detailed performance reports comparing new optimizers across
    multiple dimensions: speed, reliability, robustness, and scalability.
    """
    benchmarker = NewOptimizerBenchmarker()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_id = f"new_optimizer_benchmark_{timestamp}"
    
    print(f"\n{'='*80}")
    print(f"🔥 COMPREHENSIVE NEW OPTIMIZER BENCHMARK")
    print(f"{'='*80}")
    print(f"Benchmark ID: {benchmark_id}")
    print(f"Timestamp: {timestamp}")
    
    # Test configuration
    strategies_to_test = [
        OptimizationStrategy.SCIPY_LBFGS,  # Baseline
        OptimizationStrategy.HYBRID,
        OptimizationStrategy.JAX_ADAM_ADAPTIVE
    ]
    
    formula_tests = ['simple', 'moderate', 'complex']
    
    all_metrics = []
    
    # Comprehensive benchmarking
    for strategy in strategies_to_test:
        for formula_name in formula_tests:
            print(f"\n🧪 Benchmarking {strategy.value} on {formula_name} model...")
            
            metrics = benchmarker.benchmark_strategy_comprehensive(
                strategy=strategy,
                data_context=dipper_data,
                formula_spec=formula_specs[formula_name],
                n_runs=3,
                measure_memory=True
            )
            
            # Add benchmark metadata
            metrics.dataset_name = "dipper"
            all_metrics.append(metrics)
            
            # Print summary
            status = "✅" if metrics.convergence_success else "❌"
            print(f"   {status} Time: {metrics.execution_time:.2f}s | "
                  f"Reliability: {metrics.convergence_reliability:.1%} | "
                  f"AIC: {metrics.final_aic:.2f}")
    
    # Generate performance analysis
    performance_summary = _analyze_performance_summary(all_metrics)
    
    # Create benchmark report
    report = BenchmarkReport(
        benchmark_id=benchmark_id,
        timestamp=timestamp,
        strategies_tested=[s.value for s in strategies_to_test],
        datasets_tested=["dipper"],
        formula_complexities=formula_tests,
        performance_metrics=all_metrics,
        performance_summary=performance_summary,
        scalability_analysis={},  # Would include full scalability analysis
        robustness_analysis={},   # Would include full robustness analysis
        best_performing_strategy=performance_summary['best_strategy'],
        most_reliable_strategy=performance_summary['most_reliable_strategy'],
        most_efficient_strategy=performance_summary['most_efficient_strategy'],
        recommendations=performance_summary['recommendations']
    )
    
    # Save results
    results_dir = Path(__file__).parent.parent.parent / "tests" / "benchmarks" / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed report
    report_file = results_dir / f"{benchmark_id}_report.json"
    with open(report_file, 'w') as f:
        # Convert dataclasses to dicts for JSON serialization
        report_dict = asdict(report)
        json.dump(report_dict, f, indent=2, default=str)
    
    # Save CSV summary for analysis
    metrics_df = pd.DataFrame([asdict(m) for m in all_metrics])
    csv_file = results_dir / f"{benchmark_id}_metrics.csv"
    metrics_df.to_csv(csv_file, index=False)
    
    print(f"\n📊 BENCHMARK SUMMARY")
    print(f"{'='*50}")
    print(f"Best Performing: {report.best_performing_strategy}")
    print(f"Most Reliable: {report.most_reliable_strategy}")
    print(f"Most Efficient: {report.most_efficient_strategy}")
    
    print(f"\n📁 Results saved to:")
    print(f"   {report_file}")
    print(f"   {csv_file}")
    
    # Validation assertions
    new_optimizer_metrics = [m for m in all_metrics if m.strategy in ['hybrid', 'jax_adam_adaptive']]
    successful_new_optimizers = [m for m in new_optimizer_metrics if m.convergence_success]
    
    assert len(successful_new_optimizers) > 0, "No new optimizers achieved successful convergence"
    
    avg_reliability = np.mean([m.convergence_reliability for m in new_optimizer_metrics])
    assert avg_reliability >= 0.7, f"Low average reliability for new optimizers: {avg_reliability:.1%}"


def _analyze_performance_summary(metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
    """Analyze performance metrics and generate summary statistics."""
    
    # Group by strategy
    strategy_groups = {}
    for metric in metrics:
        if metric.strategy not in strategy_groups:
            strategy_groups[metric.strategy] = []
        strategy_groups[metric.strategy].append(metric)
    
    # Calculate aggregate statistics
    strategy_stats = {}
    for strategy, group_metrics in strategy_groups.items():
        successful_metrics = [m for m in group_metrics if m.convergence_success]
        
        if successful_metrics:
            avg_time = np.mean([m.execution_time for m in successful_metrics])
            avg_reliability = np.mean([m.convergence_reliability for m in group_metrics])
            avg_memory = np.mean([m.memory_peak_mb for m in successful_metrics])
            best_aic = min([m.final_aic for m in successful_metrics])
        else:
            avg_time = float('inf')
            avg_reliability = 0.0
            avg_memory = float('inf')
            best_aic = float('inf')
        
        strategy_stats[strategy] = {
            'avg_execution_time': avg_time,
            'avg_reliability': avg_reliability,
            'avg_memory_mb': avg_memory,
            'best_aic': best_aic,
            'success_count': len(successful_metrics),
            'total_tests': len(group_metrics)
        }
    
    # Determine best performers
    valid_strategies = {k: v for k, v in strategy_stats.items() if v['avg_reliability'] > 0}
    
    if valid_strategies:
        best_strategy = min(valid_strategies.keys(), 
                          key=lambda s: valid_strategies[s]['avg_execution_time'])
        most_reliable = max(valid_strategies.keys(),
                          key=lambda s: valid_strategies[s]['avg_reliability'])
        most_efficient = min(valid_strategies.keys(),
                           key=lambda s: valid_strategies[s]['avg_memory_mb'])
    else:
        best_strategy = most_reliable = most_efficient = "none"
    
    # Generate recommendations
    recommendations = []
    
    # Check if new optimizers are competitive
    new_optimizer_stats = {k: v for k, v in strategy_stats.items() 
                          if k in ['hybrid', 'jax_adam_adaptive']}
    baseline_stats = strategy_stats.get('scipy_lbfgs', {})
    
    if new_optimizer_stats and baseline_stats:
        for strategy, stats in new_optimizer_stats.items():
            if stats['avg_reliability'] >= baseline_stats['avg_reliability'] * 0.9:
                recommendations.append(f"{strategy} shows competitive reliability")
            if stats['avg_execution_time'] <= baseline_stats['avg_execution_time'] * 1.2:
                recommendations.append(f"{strategy} shows competitive speed")
    
    return {
        'strategy_statistics': strategy_stats,
        'best_strategy': best_strategy,
        'most_reliable_strategy': most_reliable,
        'most_efficient_strategy': most_efficient,
        'recommendations': recommendations
    }


if __name__ == "__main__":
    # Run benchmark tests directly
    pytest.main([__file__, "-v", "-s", "--tb=short"])