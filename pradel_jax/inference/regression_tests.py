"""
Performance regression testing framework for pradel-jax.

Provides automated testing to detect statistical regressions and ensure
model estimates remain consistent across code changes.
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from ..core.exceptions import ModelSpecificationError, OptimizationError
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class RegressionTestResult:
    """Result from a single regression test."""
    test_name: str
    parameter_estimates: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    convergence_success: bool
    execution_time: float
    
    # Comparison with baseline (if available)
    parameter_differences: Optional[np.ndarray] = None
    log_likelihood_difference: Optional[float] = None
    relative_differences: Optional[np.ndarray] = None
    max_absolute_difference: Optional[float] = None
    
    # Test status
    passes_tolerance: bool = True
    warning_messages: List[str] = None
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.warning_messages is None:
            self.warning_messages = []
        if self.error_messages is None:
            self.error_messages = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        
        # Convert numpy arrays to lists
        if self.parameter_estimates is not None:
            result['parameter_estimates'] = self.parameter_estimates.tolist()
        if self.parameter_differences is not None:
            result['parameter_differences'] = self.parameter_differences.tolist()
        if self.relative_differences is not None:
            result['relative_differences'] = self.relative_differences.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RegressionTestResult':
        """Create from dictionary (JSON deserialization)."""
        # Convert lists back to numpy arrays
        if data['parameter_estimates'] is not None:
            data['parameter_estimates'] = np.array(data['parameter_estimates'])
        if data.get('parameter_differences') is not None:
            data['parameter_differences'] = np.array(data['parameter_differences'])
        if data.get('relative_differences') is not None:
            data['relative_differences'] = np.array(data['relative_differences'])
            
        return cls(**data)


@dataclass
class RegressionTestSuite:
    """Collection of regression tests."""
    suite_name: str
    test_results: List[RegressionTestResult]
    timestamp: str
    total_execution_time: float
    
    # Summary statistics
    n_tests: int
    n_passed: int
    n_failed: int
    n_warnings: int
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test suite results."""
        return {
            'suite_name': self.suite_name,
            'timestamp': self.timestamp,
            'total_execution_time': self.total_execution_time,
            'test_counts': {
                'total': self.n_tests,
                'passed': self.n_passed,
                'failed': self.n_failed,
                'warnings': self.n_warnings
            },
            'success_rate': self.n_passed / self.n_tests if self.n_tests > 0 else 0,
            'tests': [result.to_dict() for result in self.test_results]
        }


class RegressionTestDefinition:
    """Definition of a single regression test case."""
    
    def __init__(
        self,
        name: str,
        data_generator: Callable[[], Any],  # Returns data_context
        model_specification: Dict[str, str],  # Formula specifications
        expected_parameters: Optional[np.ndarray] = None,
        tolerance: float = 1e-3,
        relative_tolerance: float = 1e-2,
        max_execution_time: float = 60.0,
        description: str = ""
    ):
        self.name = name
        self.data_generator = data_generator
        self.model_specification = model_specification
        self.expected_parameters = expected_parameters
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.max_execution_time = max_execution_time
        self.description = description
    
    def generate_test_id(self) -> str:
        """Generate unique test ID based on test configuration."""
        config_string = f"{self.name}_{self.model_specification}_{self.tolerance}"
        return hashlib.md5(config_string.encode()).hexdigest()[:8]


class PerformanceRegressionTester:
    """
    Framework for automated performance regression testing.
    
    Ensures model estimates remain statistically consistent across
    code changes, with appropriate tolerance handling.
    """
    
    def __init__(self, baseline_directory: Optional[Path] = None):
        self.logger = get_logger(self.__class__.__name__)
        self.baseline_directory = baseline_directory or Path("tests/regression_baselines")
        self.baseline_directory.mkdir(parents=True, exist_ok=True)
    
    def run_regression_test(
        self,
        test_definition: RegressionTestDefinition,
        model_fitter: Callable[[Any, Dict[str, str]], Tuple[np.ndarray, float, Dict[str, Any]]],
        baseline_result: Optional[RegressionTestResult] = None
    ) -> RegressionTestResult:
        """
        Run a single regression test.
        
        Args:
            test_definition: Test configuration
            model_fitter: Function that fits model, returns (params, log_likelihood, diagnostics)
            baseline_result: Baseline result for comparison
            
        Returns:
            RegressionTestResult with test outcome
        """
        self.logger.info(f"Running regression test: {test_definition.name}")
        
        start_time = time.time()
        warning_messages = []
        error_messages = []
        
        try:
            # Generate test data
            data_context = test_definition.data_generator()
            
            # Fit model
            parameter_estimates, log_likelihood, diagnostics = model_fitter(
                data_context, test_definition.model_specification
            )
            
            # Check execution time
            execution_time = time.time() - start_time
            if execution_time > test_definition.max_execution_time:
                warning_messages.append(
                    f"Execution time ({execution_time:.2f}s) exceeded threshold "
                    f"({test_definition.max_execution_time:.2f}s)"
                )
            
            # Compute diagnostics
            aic = diagnostics.get('aic', np.nan)
            bic = diagnostics.get('bic', np.nan)
            convergence_success = diagnostics.get('success', True)
            
            # Compare with baseline if available
            parameter_differences = None
            log_likelihood_difference = None
            relative_differences = None
            max_absolute_difference = None
            passes_tolerance = True
            
            if baseline_result is not None:
                parameter_differences = parameter_estimates - baseline_result.parameter_estimates
                log_likelihood_difference = log_likelihood - baseline_result.log_likelihood
                
                # Relative differences (avoid division by zero)
                baseline_nonzero = np.abs(baseline_result.parameter_estimates) > 1e-10
                relative_differences = np.zeros_like(parameter_differences)
                relative_differences[baseline_nonzero] = (
                    parameter_differences[baseline_nonzero] / 
                    baseline_result.parameter_estimates[baseline_nonzero]
                )
                
                # Check tolerances
                max_absolute_difference = np.max(np.abs(parameter_differences))
                max_relative_difference = np.max(np.abs(relative_differences))
                
                if max_absolute_difference > test_definition.tolerance:
                    passes_tolerance = False
                    error_messages.append(
                        f"Maximum absolute difference ({max_absolute_difference:.6f}) "
                        f"exceeds tolerance ({test_definition.tolerance:.6f})"
                    )
                
                if max_relative_difference > test_definition.relative_tolerance:
                    passes_tolerance = False
                    error_messages.append(
                        f"Maximum relative difference ({max_relative_difference:.6f}) "
                        f"exceeds relative tolerance ({test_definition.relative_tolerance:.6f})"
                    )
                
                if abs(log_likelihood_difference) > test_definition.tolerance * 10:
                    warning_messages.append(
                        f"Log-likelihood difference ({log_likelihood_difference:.6f}) is large"
                    )
            
            # Check against expected parameters if provided
            elif test_definition.expected_parameters is not None:
                expected_differences = parameter_estimates - test_definition.expected_parameters
                max_expected_difference = np.max(np.abs(expected_differences))
                
                if max_expected_difference > test_definition.tolerance:
                    passes_tolerance = False
                    error_messages.append(
                        f"Parameters differ from expected by {max_expected_difference:.6f} "
                        f"(tolerance: {test_definition.tolerance:.6f})"
                    )
            
            result = RegressionTestResult(
                test_name=test_definition.name,
                parameter_estimates=parameter_estimates,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                convergence_success=convergence_success,
                execution_time=execution_time,
                parameter_differences=parameter_differences,
                log_likelihood_difference=log_likelihood_difference,
                relative_differences=relative_differences,
                max_absolute_difference=max_absolute_difference,
                passes_tolerance=passes_tolerance,
                warning_messages=warning_messages,
                error_messages=error_messages
            )
            
            self.logger.info(f"Test {test_definition.name}: {'PASSED' if passes_tolerance else 'FAILED'}")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_messages.append(f"Test failed with exception: {str(e)}")
            
            self.logger.error(f"Test {test_definition.name} failed: {e}")
            
            # Return failed result
            return RegressionTestResult(
                test_name=test_definition.name,
                parameter_estimates=np.array([]),
                log_likelihood=np.nan,
                aic=np.nan,
                bic=np.nan,
                convergence_success=False,
                execution_time=execution_time,
                passes_tolerance=False,
                warning_messages=warning_messages,
                error_messages=error_messages
            )
    
    def run_test_suite(
        self,
        test_definitions: List[RegressionTestDefinition],
        model_fitter: Callable[[Any, Dict[str, str]], Tuple[np.ndarray, float, Dict[str, Any]]],
        suite_name: str = "default",
        use_baselines: bool = True
    ) -> RegressionTestSuite:
        """
        Run a suite of regression tests.
        
        Args:
            test_definitions: List of test definitions
            model_fitter: Model fitting function
            suite_name: Name of the test suite
            use_baselines: Whether to compare against saved baselines
            
        Returns:
            RegressionTestSuite with all test results
        """
        self.logger.info(f"Running regression test suite: {suite_name} ({len(test_definitions)} tests)")
        
        start_time = time.time()
        test_results = []
        
        for test_def in test_definitions:
            # Load baseline if requested
            baseline_result = None
            if use_baselines:
                baseline_result = self.load_baseline_result(test_def)
            
            # Run test
            result = self.run_regression_test(test_def, model_fitter, baseline_result)
            test_results.append(result)
        
        total_time = time.time() - start_time
        
        # Compute summary statistics
        n_tests = len(test_results)
        n_passed = sum(1 for r in test_results if r.passes_tolerance)
        n_failed = n_tests - n_passed
        n_warnings = sum(1 for r in test_results if len(r.warning_messages) > 0)
        
        test_suite = RegressionTestSuite(
            suite_name=suite_name,
            test_results=test_results,
            timestamp=datetime.now().isoformat(),
            total_execution_time=total_time,
            n_tests=n_tests,
            n_passed=n_passed,
            n_failed=n_failed,
            n_warnings=n_warnings
        )
        
        self.logger.info(
            f"Test suite completed: {n_passed}/{n_tests} passed, "
            f"{n_warnings} warnings, {total_time:.2f}s"
        )
        
        return test_suite
    
    def save_baseline_results(self, test_suite: RegressionTestSuite) -> None:
        """Save test results as new baselines."""
        for result in test_suite.test_results:
            if result.convergence_success and result.passes_tolerance:
                baseline_path = self.baseline_directory / f"{result.test_name}.json"
                
                with open(baseline_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
                
                self.logger.debug(f"Saved baseline for {result.test_name}")
    
    def load_baseline_result(self, test_def: RegressionTestDefinition) -> Optional[RegressionTestResult]:
        """Load baseline result for a test."""
        baseline_path = self.baseline_directory / f"{test_def.name}.json"
        
        if not baseline_path.exists():
            self.logger.debug(f"No baseline found for {test_def.name}")
            return None
        
        try:
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            
            return RegressionTestResult.from_dict(baseline_data)
        
        except Exception as e:
            self.logger.warning(f"Failed to load baseline for {test_def.name}: {e}")
            return None
    
    def save_test_report(self, test_suite: RegressionTestSuite, output_path: Path) -> None:
        """Save comprehensive test report."""
        report = {
            'summary': test_suite.get_summary(),
            'detailed_results': [result.to_dict() for result in test_suite.test_results],
            'metadata': {
                'pradel_jax_version': '0.1.0',  # Would get from package
                'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                'numpy_version': np.__version__,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Test report saved to {output_path}")


# Standard test data generators
def generate_simple_dipper_data() -> Any:
    """Generate simple synthetic dipper-like data for testing."""
    np.random.seed(42)  # Reproducible
    
    n_individuals = 50
    n_occasions = 5
    
    # Simple capture probabilities
    true_phi = 0.7  # Survival
    true_p = 0.6    # Detection
    
    # Generate capture histories
    capture_matrix = np.zeros((n_individuals, n_occasions), dtype=np.int32)
    
    for i in range(n_individuals):
        alive = True
        first_capture = np.random.randint(0, n_occasions)
        capture_matrix[i, first_capture] = 1  # Ensure first capture
        
        for t in range(first_capture + 1, n_occasions):
            if alive:
                # Survival
                if np.random.random() > true_phi:
                    alive = False
                    break
                
                # Detection
                if np.random.random() < true_p:
                    capture_matrix[i, t] = 1
    
    # Create minimal data context
    from ..data.adapters import DataContext, CovariateInfo
    import jax.numpy as jnp
    
    return DataContext(
        capture_matrix=jnp.array(capture_matrix),
        covariates={},
        covariate_info={},
        n_individuals=n_individuals,
        n_occasions=n_occasions
    )


# Standard test definitions
def get_standard_regression_tests() -> List[RegressionTestDefinition]:
    """Get standard suite of regression tests."""
    
    return [
        RegressionTestDefinition(
            name="simple_constant_model",
            data_generator=generate_simple_dipper_data,
            model_specification={
                'phi': '~1',
                'p': '~1', 
                'f': '~1'
            },
            tolerance=1e-3,
            description="Simple constant parameter model"
        ),
        
        RegressionTestDefinition(
            name="time_varying_detection",
            data_generator=generate_simple_dipper_data,
            model_specification={
                'phi': '~1',
                'p': '~time',  # Assuming time covariate exists
                'f': '~1'
            },
            tolerance=5e-3,  # Higher tolerance for more complex model
            description="Model with time-varying detection probability"
        ),
    ]


# Convenience function
def run_performance_regression_tests(
    model_fitter: Callable[[Any, Dict[str, str]], Tuple[np.ndarray, float, Dict[str, Any]]],
    output_directory: Optional[Path] = None,
    create_baselines: bool = False
) -> RegressionTestSuite:
    """
    Run standard performance regression tests.
    
    Args:
        model_fitter: Model fitting function
        output_directory: Directory to save results
        create_baselines: Whether to create new baselines
        
    Returns:
        RegressionTestSuite with results
    """
    output_dir = output_directory or Path("test_results/regression")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tester = PerformanceRegressionTester(output_dir / "baselines")
    test_definitions = get_standard_regression_tests()
    
    # Run tests
    test_suite = tester.run_test_suite(
        test_definitions,
        model_fitter,
        suite_name="standard_regression_tests",
        use_baselines=not create_baselines
    )
    
    # Save results
    if create_baselines:
        tester.save_baseline_results(test_suite)
    
    # Save report
    report_path = output_dir / f"regression_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    tester.save_test_report(test_suite, report_path)
    
    return test_suite