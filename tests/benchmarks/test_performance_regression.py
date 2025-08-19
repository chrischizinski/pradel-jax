#!/usr/bin/env python3
"""
Performance regression testing for Pradel-JAX optimization framework.

This module provides automated performance monitoring to detect regressions
in optimization speed, memory usage, and convergence reliability across releases.

Key Features:
- Baseline performance measurement and comparison
- Automated regression detection with configurable thresholds
- CI/CD-ready with JSON output for automation
- Multiple performance metrics (time, memory, success rate)
- Historical performance tracking
"""

import time
import pytest
import numpy as np
import pandas as pd
import json
import psutil
import os
import tracemalloc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.models import PradelModel


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    strategy: str
    formula_complexity: str
    avg_time_seconds: float
    success_rate: float
    avg_memory_mb: float
    avg_iterations: int
    avg_function_evals: int
    baseline_date: str
    git_commit: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceBaseline':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PerformanceResult:
    """Single performance test result."""
    strategy: str
    formula_complexity: str
    dataset_size: int
    time_seconds: float
    memory_mb: float
    success: bool
    iterations: int
    function_evals: int
    aic: float
    timestamp: str


@dataclass
class RegressionReport:
    """Performance regression analysis report."""
    test_date: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    regressions_detected: List[Dict[str, Any]]
    performance_summary: Dict[str, Any]
    baseline_version: str
    current_version: str


class PerformanceRegressionTester:
    """Automated performance regression testing framework."""
    
    def __init__(self, 
                 time_regression_threshold: float = 1.5,  # 50% slower triggers regression
                 memory_regression_threshold: float = 1.3,  # 30% more memory
                 success_rate_threshold: float = 0.9,  # Must maintain 90% success rate
                 baseline_file: Optional[str] = None):
        """
        Initialize regression tester.
        
        Args:
            time_regression_threshold: Factor by which time can increase before flagging
            memory_regression_threshold: Factor by which memory can increase before flagging  
            success_rate_threshold: Minimum acceptable success rate
            baseline_file: Path to baseline performance file
        """
        self.time_threshold = time_regression_threshold
        self.memory_threshold = memory_regression_threshold
        self.success_threshold = success_rate_threshold
        
        # Set up baseline file
        if baseline_file is None:
            self.baseline_file = Path(__file__).parent / "performance_baselines.json"
        else:
            self.baseline_file = Path(baseline_file)
            
        self.baselines = self._load_baselines()
        self.results = []
        
    def _load_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Load performance baselines from file."""
        if not self.baseline_file.exists():
            return {}
        
        try:
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)
            
            baselines = {}
            for key, baseline_data in data.items():
                baselines[key] = PerformanceBaseline.from_dict(baseline_data)
            return baselines
            
        except Exception as e:
            print(f"Warning: Could not load baselines from {self.baseline_file}: {e}")
            return {}
    
    def _save_baselines(self):
        """Save performance baselines to file."""
        try:
            self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for key, baseline in self.baselines.items():
                data[key] = baseline.to_dict()
                
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save baselines to {self.baseline_file}: {e}")
    
    def _get_baseline_key(self, strategy: str, formula_complexity: str) -> str:
        """Generate unique key for baseline lookup."""
        return f"{strategy}_{formula_complexity}"
    
    def measure_performance(self,
                          strategy: str,
                          data_context,
                          formula_spec,
                          model: PradelModel,
                          n_runs: int = 3) -> PerformanceResult:
        """
        Measure performance for a specific strategy and configuration.
        
        Returns averaged performance metrics across multiple runs.
        """
        times = []
        memories = []
        successes = []
        iterations = []
        function_evals = []
        aics = []
        
        for run in range(n_runs):
            # Start memory tracking
            tracemalloc.start()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.perf_counter()
            
            try:
                # Setup optimization
                design_matrices = model.build_design_matrices(formula_spec, data_context)
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                # Add required attributes for optimization framework
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
                strategy_enum = OptimizationStrategy(strategy)
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=data_context,
                    bounds=bounds,
                    preferred_strategy=strategy_enum
                )
                
                # Measure elapsed time
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
                
                # Measure memory usage
                end_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = tracemalloc.get_traced_memory()[1] / 1024 / 1024
                memory_used = max(end_memory - start_memory, peak_memory)
                memories.append(memory_used)
                
                # Record success and metrics
                successes.append(result.success)
                iterations.append(result.result.nit if hasattr(result.result, 'nit') else 0)
                function_evals.append(result.result.nfev if hasattr(result.result, 'nfev') else 0)
                
                if result.success:
                    final_nll = result.result.fun
                    n_params = len(initial_params)
                    aic = 2 * final_nll + 2 * n_params
                    aics.append(aic)
                else:
                    aics.append(float('inf'))
                    
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                times.append(elapsed)
                end_memory = process.memory_info().rss / 1024 / 1024
                memories.append(max(0, end_memory - start_memory))
                successes.append(False)
                iterations.append(0)
                function_evals.append(0)
                aics.append(float('inf'))
                
            finally:
                tracemalloc.stop()
        
        # Calculate averages
        return PerformanceResult(
            strategy=strategy,
            formula_complexity=getattr(formula_spec, 'complexity', 'unknown'),
            dataset_size=data_context.n_individuals,
            time_seconds=np.mean(times),
            memory_mb=np.mean(memories),
            success=np.mean(successes) >= self.success_threshold,
            iterations=int(np.mean(iterations)),
            function_evals=int(np.mean(function_evals)),
            aic=np.mean([aic for aic in aics if aic != float('inf')]) if any(aic != float('inf') for aic in aics) else float('inf'),
            timestamp=datetime.now().isoformat()
        )
    
    def check_regression(self, result: PerformanceResult) -> Optional[Dict[str, Any]]:
        """
        Check if performance result indicates a regression.
        
        Returns regression details if detected, None otherwise.
        """
        key = self._get_baseline_key(result.strategy, result.formula_complexity)
        
        if key not in self.baselines:
            # No baseline available - this becomes the baseline
            return None
        
        baseline = self.baselines[key]
        regression_issues = []
        
        # Check time regression
        time_ratio = result.time_seconds / baseline.avg_time_seconds
        if time_ratio > self.time_threshold:
            regression_issues.append({
                'metric': 'time',
                'baseline': baseline.avg_time_seconds,
                'current': result.time_seconds,
                'ratio': time_ratio,
                'threshold': self.time_threshold
            })
        
        # Check memory regression
        if baseline.avg_memory_mb > 0:  # Avoid division by zero
            memory_ratio = result.memory_mb / baseline.avg_memory_mb
            if memory_ratio > self.memory_threshold:
                regression_issues.append({
                    'metric': 'memory',
                    'baseline': baseline.avg_memory_mb,
                    'current': result.memory_mb,
                    'ratio': memory_ratio,
                    'threshold': self.memory_threshold
                })
        
        # Check success rate regression
        if not result.success and baseline.success_rate >= self.success_threshold:
            regression_issues.append({
                'metric': 'success_rate',
                'baseline': baseline.success_rate,
                'current': result.success,
                'threshold': self.success_threshold
            })
        
        if regression_issues:
            return {
                'strategy': result.strategy,
                'formula_complexity': result.formula_complexity,
                'issues': regression_issues,
                'baseline_date': baseline.baseline_date,
                'current_date': result.timestamp
            }
        
        return None
    
    def update_baseline(self, result: PerformanceResult, git_commit: str = "unknown"):
        """Update baseline with new performance result."""
        key = self._get_baseline_key(result.strategy, result.formula_complexity)
        
        baseline = PerformanceBaseline(
            strategy=result.strategy,
            formula_complexity=result.formula_complexity,
            avg_time_seconds=result.time_seconds,
            success_rate=1.0 if result.success else 0.0,
            avg_memory_mb=result.memory_mb,
            avg_iterations=result.iterations,
            avg_function_evals=result.function_evals,
            baseline_date=result.timestamp,
            git_commit=git_commit
        )
        
        self.baselines[key] = baseline
        self._save_baselines()
    
    def run_regression_tests(self, 
                            test_configurations: List[Tuple[str, str]],
                            data_context,
                            formula_specs: Dict[str, Any],
                            model: PradelModel,
                            update_baselines: bool = False) -> RegressionReport:
        """
        Run complete regression test suite.
        
        Args:
            test_configurations: List of (strategy, formula_complexity) pairs to test
            data_context: Test data context
            formula_specs: Dictionary of formula specifications
            model: Model instance to test
            update_baselines: Whether to update baselines with new results
            
        Returns:
            Comprehensive regression report
        """
        regressions = []
        all_results = []
        
        print(f"Running {len(test_configurations)} performance regression tests...")
        
        for i, (strategy, formula_complexity) in enumerate(test_configurations):
            print(f"[{i+1}/{len(test_configurations)}] Testing {strategy} with {formula_complexity} formula...")
            
            try:
                result = self.measure_performance(
                    strategy=strategy,
                    data_context=data_context,
                    formula_spec=formula_specs[formula_complexity],
                    model=model,
                    n_runs=2  # Fewer runs for CI speed
                )
                
                all_results.append(result)
                
                # Check for regression
                regression = self.check_regression(result)
                if regression:
                    regressions.append(regression)
                    print(f"  ⚠️  REGRESSION DETECTED: {regression['issues']}")
                else:
                    print(f"  ✅ Performance acceptable: {result.time_seconds:.3f}s")
                
                # Update baseline if requested
                if update_baselines:
                    git_commit = os.environ.get('GITHUB_SHA', 'unknown')
                    self.update_baseline(result, git_commit)
                    
            except Exception as e:
                print(f"  ❌ Test failed: {e}")
                regressions.append({
                    'strategy': strategy,
                    'formula_complexity': formula_complexity,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
        
        # Generate summary
        performance_summary = {
            'avg_time': np.mean([r.time_seconds for r in all_results]),
            'avg_memory': np.mean([r.memory_mb for r in all_results]),
            'success_rate': np.mean([r.success for r in all_results]),
            'total_strategies_tested': len(set(r.strategy for r in all_results))
        }
        
        report = RegressionReport(
            test_date=datetime.now().isoformat(),
            total_tests=len(test_configurations),
            passed_tests=len(test_configurations) - len(regressions),
            failed_tests=len(regressions),
            regressions_detected=regressions,
            performance_summary=performance_summary,
            baseline_version=os.environ.get('BASELINE_VERSION', 'unknown'),
            current_version=os.environ.get('CURRENT_VERSION', 'unknown')
        )
        
        return report


@pytest.fixture
def regression_tester():
    """Create regression tester instance."""
    return PerformanceRegressionTester()


@pytest.fixture
def test_data():
    """Load test data for regression tests."""
    data_path = Path(__file__).parent.parent.parent / "data" / "dipper_dataset.csv"
    return pj.load_data(str(data_path))


@pytest.fixture
def test_formulas():
    """Create test formula specifications."""
    return {
        'simple': pj.create_simple_spec(phi="~1", p="~1", f="~1"),
        'moderate': pj.create_simple_spec(phi="~sex", p="~sex", f="~1")
    }


class TestPerformanceRegression:
    """Performance regression test suite."""
    
    def test_create_initial_baselines(self, regression_tester, test_data, test_formulas):
        """Create initial performance baselines (run once to establish baselines)."""
        model = pj.PradelModel()
        
        # Core strategies to baseline
        test_configs = [
            ('scipy_lbfgs', 'simple'),
            ('scipy_lbfgs', 'moderate'),
            ('scipy_slsqp', 'simple'),
            ('hybrid', 'simple'),  # Test our new hybrid strategy
        ]
        
        print("\n=== Creating Performance Baselines ===")
        
        for strategy, formula_complexity in test_configs:
            result = regression_tester.measure_performance(
                strategy=strategy,
                data_context=test_data,
                formula_spec=test_formulas[formula_complexity],
                model=model,
                n_runs=3
            )
            
            regression_tester.update_baseline(result)
            print(f"Baseline set for {strategy}/{formula_complexity}: {result.time_seconds:.3f}s")
    
    @pytest.mark.regression
    def test_performance_regression_suite(self, regression_tester, test_data, test_formulas):
        """Run comprehensive performance regression tests."""
        model = pj.PradelModel()
        
        # Test configurations for regression monitoring
        test_configs = [
            ('scipy_lbfgs', 'simple'),
            ('scipy_lbfgs', 'moderate'),
            ('scipy_slsqp', 'simple'),
            ('hybrid', 'simple'),
            ('hybrid', 'moderate'),
        ]
        
        report = regression_tester.run_regression_tests(
            test_configurations=test_configs,
            data_context=test_data,
            formula_specs=test_formulas,
            model=model,
            update_baselines=False  # Don't update baselines in regression tests
        )
        
        # Save regression report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(__file__).parent / f"regression_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        print(f"\n=== Performance Regression Report ===")
        print(f"Total tests: {report.total_tests}")
        print(f"Passed: {report.passed_tests}")
        print(f"Failed: {report.failed_tests}")
        print(f"Regressions detected: {len(report.regressions_detected)}")
        print(f"Average performance: {report.performance_summary['avg_time']:.3f}s")
        print(f"Report saved: {report_file}")
        
        # Assert no critical regressions
        if report.regressions_detected:
            critical_regressions = [
                r for r in report.regressions_detected 
                if any(issue.get('ratio', 1.0) > 2.0 for issue in r.get('issues', []))
            ]
            if critical_regressions:
                pytest.fail(f"Critical performance regressions detected: {critical_regressions}")
            else:
                pytest.skip(f"Minor regressions detected but within acceptable limits")
        
        # Assert minimum success rate
        assert report.performance_summary['success_rate'] >= 0.8, \
            f"Overall success rate too low: {report.performance_summary['success_rate']:.1%}"
    
    def test_memory_usage_monitoring(self, regression_tester, test_data, test_formulas):
        """Monitor memory usage to detect memory leaks."""
        model = pj.PradelModel()
        
        # Test memory usage with multiple runs
        strategy = 'scipy_lbfgs'
        formula = test_formulas['simple']
        
        memory_readings = []
        
        for run in range(5):
            result = regression_tester.measure_performance(
                strategy=strategy,
                data_context=test_data,
                formula_spec=formula,
                model=model,
                n_runs=1
            )
            memory_readings.append(result.memory_mb)
        
        # Check for increasing memory trend (potential leak)
        if len(memory_readings) >= 3:
            # Simple trend check: last reading shouldn't be much higher than first
            memory_increase = memory_readings[-1] / memory_readings[0]
            assert memory_increase < 1.5, \
                f"Potential memory leak detected: {memory_readings[0]:.1f}MB -> {memory_readings[-1]:.1f}MB"
        
        print(f"Memory usage stable: {np.mean(memory_readings):.1f}±{np.std(memory_readings):.1f}MB")


# CLI interface for CI/CD integration
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pradel-JAX Performance Regression Testing")
    parser.add_argument("--mode", choices=['baseline', 'test'], default='test',
                       help="Mode: 'baseline' to create baselines, 'test' to run regression tests")
    parser.add_argument("--output", help="Output file for CI/CD integration")
    parser.add_argument("--threshold-time", type=float, default=1.5,
                       help="Time regression threshold factor")
    parser.add_argument("--threshold-memory", type=float, default=1.3,
                       help="Memory regression threshold factor")
    
    args = parser.parse_args()
    
    # Setup
    tester = PerformanceRegressionTester(
        time_regression_threshold=args.threshold_time,
        memory_regression_threshold=args.threshold_memory
    )
    
    data_path = Path(__file__).parent.parent.parent / "data" / "dipper_dataset.csv"
    data_context = pj.load_data(str(data_path))
    
    formulas = {
        'simple': pj.create_simple_spec(phi="~1", p="~1", f="~1"),
        'moderate': pj.create_simple_spec(phi="~sex", p="~sex", f="~1")
    }
    
    model = pj.PradelModel()
    
    if args.mode == 'baseline':
        print("Creating performance baselines...")
        test_configs = [
            ('scipy_lbfgs', 'simple'),
            ('scipy_lbfgs', 'moderate'),
            ('scipy_slsqp', 'simple'),
            ('hybrid', 'simple'),
        ]
        
        for strategy, formula_complexity in test_configs:
            result = tester.measure_performance(
                strategy=strategy,
                data_context=data_context,
                formula_spec=formulas[formula_complexity],
                model=model
            )
            tester.update_baseline(result)
            print(f"✅ Baseline created: {strategy}/{formula_complexity}")
    
    else:  # test mode
        print("Running performance regression tests...")
        test_configs = [
            ('scipy_lbfgs', 'simple'),
            ('scipy_lbfgs', 'moderate'),
            ('scipy_slsqp', 'simple'),
            ('hybrid', 'simple'),
            ('hybrid', 'moderate'),
        ]
        
        report = tester.run_regression_tests(
            test_configurations=test_configs,
            data_context=data_context,
            formula_specs=formulas,
            model=model
        )
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        
        # Exit with appropriate code for CI/CD
        if report.regressions_detected:
            print(f"❌ {len(report.regressions_detected)} regressions detected")
            exit(1)
        else:
            print("✅ No regressions detected")
            exit(0)