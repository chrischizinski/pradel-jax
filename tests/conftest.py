"""
Shared pytest configuration and fixtures for Pradel-JAX tests.

This module provides common test fixtures, utilities, and configuration
used across the test suite.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Dict, Any
import tempfile
import csv

# Import test utilities
from pradel_jax.data.adapters import DataContext
from pradel_jax.formulas.spec import FormulaSpec, ParameterFormula, ParameterType


# Test configuration
pytest_plugins = []


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session") 
def sample_dipper_data():
    """Sample dipper dataset for testing."""
    # This mirrors the actual dipper data structure but with synthetic values
    capture_histories = [
        "1111000", "1010100", "0011110", "1001010", "0100101",
        "1100010", "0110100", "1011000", "0101010", "1000111"
    ]
    
    sex_values = ["Female", "Male", "Female", "Male", "Female",
                  "Male", "Female", "Male", "Female", "Male"]
    
    return {
        "capture_histories": capture_histories,
        "covariates": {"sex": sex_values},
        "n_individuals": len(capture_histories),
        "n_occasions": len(capture_histories[0])
    }


@pytest.fixture
def sample_data_context(sample_dipper_data):
    """Create a DataContext from sample dipper data."""
    
    class TestDataContext(DataContext):
        def __init__(self, data_dict):
            self.n_individuals = data_dict["n_individuals"]
            self.n_occasions = data_dict["n_occasions"]
            self.n_parameters = 3  # Default for Pradel model
            
            # Convert capture histories to binary matrix
            ch_matrix = []
            for ch in data_dict["capture_histories"]:
                ch_matrix.append([int(c) for c in ch])
            
            self.capture_matrix = jnp.array(ch_matrix, dtype=jnp.int32)
            
            # Convert covariates to numeric
            self.covariates = {}
            for name, values in data_dict["covariates"].items():
                if name == "sex":
                    # Convert to numeric: Female=0, Male=1
                    numeric_values = [0 if v == "Female" else 1 for v in values]
                    self.covariates[name] = jnp.array(numeric_values, dtype=jnp.float32)
        
        def get_condition_estimate(self):
            return 1e5
    
    return TestDataContext(sample_dipper_data)


@pytest.fixture
def simple_formula_spec():
    """Create a simple constant formula specification."""
    phi = ParameterFormula(ParameterType.PHI, "~1")
    p = ParameterFormula(ParameterType.P, "~1")
    f = ParameterFormula(ParameterType.F, "~1")
    
    return FormulaSpec(phi=phi, p=p, f=f)


@pytest.fixture  
def sex_formula_spec():
    """Create a formula specification with sex effects."""
    phi = ParameterFormula(ParameterType.PHI, "~1 + sex")
    p = ParameterFormula(ParameterType.P, "~1 + sex")
    f = ParameterFormula(ParameterType.F, "~1")
    
    return FormulaSpec(phi=phi, p=p, f=f)


@pytest.fixture
def temp_data_file(sample_dipper_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['ch', 'sex'])
        
        # Write data
        for ch, sex in zip(sample_dipper_data["capture_histories"], 
                          sample_dipper_data["covariates"]["sex"]):
            writer.writerow([ch, sex])
        
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def small_dataset():
    """Create a very small dataset for quick testing."""
    return {
        "capture_histories": ["101", "010", "110"],
        "covariates": {"sex": ["Female", "Male", "Female"]},
        "n_individuals": 3,
        "n_occasions": 3
    }


@pytest.fixture
def optimization_test_config():
    """Configuration for optimization tests."""
    return {
        "max_iterations": 100,
        "tolerance": 1e-6,
        "verbose": False,
        "strategy": "scipy_lbfgs"
    }


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    # JAX random key would be set here if needed


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test (fast)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test (medium speed)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (may take >10 seconds)"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )


# Test utilities
class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def create_synthetic_data(n_individuals=50, n_occasions=5, detection_prob=0.3):
        """Create synthetic capture-recapture data."""
        np.random.seed(42)  # Ensure reproducibility
        
        capture_matrix = np.random.binomial(
            1, detection_prob, (n_individuals, n_occasions)
        )
        
        sex_values = np.random.choice(["Female", "Male"], n_individuals)
        
        return {
            "capture_matrix": jnp.array(capture_matrix, dtype=jnp.int32),
            "covariates": {
                "sex": jnp.array([0 if s == "Female" else 1 for s in sex_values], 
                               dtype=jnp.float32)
            },
            "n_individuals": n_individuals,
            "n_occasions": n_occasions
        }
    
    @staticmethod
    def assert_optimization_result(result, should_succeed=True):
        """Assert common properties of optimization results."""
        if should_succeed:
            assert result.success, f"Optimization failed: {getattr(result, 'message', 'No message')}"
            assert hasattr(result, 'result')
            assert hasattr(result.result, 'x')  # Parameter values
            assert hasattr(result.result, 'fun')  # Objective value
        
        assert hasattr(result, 'strategy_used')
        assert hasattr(result, 'total_time')
    
    @staticmethod
    def assert_parameter_reasonable(params, expected_count=3):
        """Assert that parameters are reasonable for Pradel model."""
        assert len(params) == expected_count
        assert all(np.isfinite(params))
        
        # Basic sanity checks for transformed parameters
        # (these are on link scale, so can be negative)
        assert all(abs(p) < 20 for p in params)  # Not too extreme


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils


# Mock objects for testing
class MockOptimizer:
    """Mock optimizer for testing optimization framework."""
    
    def __init__(self, should_succeed=True, n_iterations=10):
        self.should_succeed = should_succeed
        self.n_iterations = n_iterations
    
    def optimize(self, objective_fn, initial_params, bounds=None):
        """Mock optimization that returns predictable results."""
        from scipy.optimize import OptimizeResult
        
        if self.should_succeed:
            # Return slightly perturbed initial parameters
            result_params = initial_params + 0.1 * np.random.randn(len(initial_params))
            result = OptimizeResult(
                x=result_params,
                fun=objective_fn(result_params),
                success=True,
                message="Mock optimization successful",
                nit=self.n_iterations,
                nfev=self.n_iterations * 2
            )
        else:
            result = OptimizeResult(
                x=initial_params,
                fun=objective_fn(initial_params),
                success=False,
                message="Mock optimization failed",
                nit=1,
                nfev=1
            )
        
        return result


@pytest.fixture
def mock_optimizer():
    """Provide mock optimizer."""
    return MockOptimizer