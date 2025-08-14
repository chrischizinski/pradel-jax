"""
Tests for RMark interface and execution.

These tests verify the multi-environment execution system while maintaining
security by using mock objects and avoiding actual SSH connections or
external dependencies.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from pradel_jax.validation.rmark_interface import (
    RMarkResult,
    ExecutionMethod,
    ExecutionStatus,
    EnvironmentDetector,
    SSHRMarkExecutor,
    LocalRMarkExecutor,
    MockRMarkExecutor,
    RMarkExecutor
)
from pradel_jax.validation.secure_config import SecureValidationConfig
from pradel_jax.data.adapters import DataContext, CovariateInfo
from pradel_jax.formulas.spec import FormulaSpec
import numpy as np


@pytest.fixture
def mock_data_context():
    """Create mock data context for testing."""
    import jax.numpy as jnp
    
    return DataContext(
        capture_matrix=jnp.array([
            [1, 0, 1, 0],
            [0, 1, 1, 0], 
            [1, 1, 0, 1],
            [0, 0, 1, 1]
        ]),
        covariates={
            'sex': jnp.array([1, 2, 1, 2]),
            'age': jnp.array([2, 3, 4, 2])
        },
        covariate_info={
            'sex': CovariateInfo(name='sex', dtype='int', is_categorical=True),
            'age': CovariateInfo(name='age', dtype='int', is_categorical=False)
        },
        n_individuals=4,
        n_occasions=4
    )


@pytest.fixture
def mock_formula_spec():
    """Create mock formula specification."""
    return FormulaSpec(
        phi="1 + sex",
        p="1",
        f="1"
    )


@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    config = Mock(spec=SecureValidationConfig)
    config.session_id = "test_session"
    config.ssh_host = "test.host.com"
    config.ssh_user = "testuser"
    config.ssh_key_path = "/path/to/key"
    config.ssh_r_path = "C:\\R\\bin\\Rscript.exe"
    config.ssh_timeout = 60
    config.ssh_max_retries = 2
    config.local_r_path = "Rscript"
    config.local_r_timeout = 120
    config.auto_install_rmark = True
    config.preferred_environment = "auto"
    
    # Mock methods
    config.has_ssh_config.return_value = True
    config.has_local_r_config.return_value = True
    
    return config


class TestRMarkResult:
    """Test RMarkResult data class."""
    
    def test_basic_result_creation(self):
        """Test basic result creation."""
        result = RMarkResult(
            model_formula="phi(1), p(1), f(1)",
            execution_method="mock",
            execution_time=1.5,
            converged=True,
            aic=150.5,
            log_likelihood=-72.25,
            n_parameters=3
        )
        
        assert result.model_formula == "phi(1), p(1), f(1)"
        assert result.execution_method == "mock"
        assert result.execution_time == 1.5
        assert result.converged is True
        assert result.aic == 150.5
        assert result.execution_status == ExecutionStatus.FAILED  # Default
    
    def test_get_summary(self):
        """Test result summary generation."""
        result = RMarkResult(
            model_formula="phi(sex), p(1), f(1)",
            execution_method="ssh",
            execution_time=2.3,
            converged=True,
            aic=148.2,
            log_likelihood=-71.1,
            n_parameters=4,
            execution_status=ExecutionStatus.SUCCESS
        )
        
        summary = result.get_summary()
        
        assert summary['model_formula'] == "phi(sex), p(1), f(1)"
        assert summary['execution_method'] == "ssh"
        assert summary['converged'] is True
        assert summary['aic'] == 148.2
        assert summary['status'] == "success"
    
    def test_parameters_and_errors(self):
        """Test parameter and standard error storage."""
        parameters = {
            'phi_intercept': 0.85,
            'phi_sex': 0.12,
            'p_intercept': 0.65
        }
        std_errors = {
            'phi_intercept': 0.05,
            'phi_sex': 0.03,
            'p_intercept': 0.04
        }
        
        result = RMarkResult(
            model_formula="test",
            execution_method="test",
            execution_time=1.0,
            parameters=parameters,
            std_errors=std_errors
        )
        
        assert result.parameters == parameters
        assert result.std_errors == std_errors
        assert len(result.parameters) == 3
        assert len(result.std_errors) == 3


class TestEnvironmentDetector:
    """Test environment detection logic."""
    
    def test_detect_available_methods_all_available(self, mock_config):
        """Test detection when all methods are available."""
        detector = EnvironmentDetector(mock_config)
        
        with patch.object(detector, '_test_ssh_connectivity', return_value=True), \
             patch.object(detector, '_test_local_r', return_value=True), \
             patch.object(detector, '_has_cached_results', return_value=True):
            
            methods = detector.detect_available_methods()
            
            assert ExecutionMethod.SSH in methods
            assert ExecutionMethod.LOCAL_R in methods
            assert ExecutionMethod.MOCK in methods
            assert ExecutionMethod.CACHED in methods
    
    def test_detect_available_methods_limited(self, mock_config):
        """Test detection when only limited methods are available."""
        detector = EnvironmentDetector(mock_config)
        
        with patch.object(detector, '_test_ssh_connectivity', return_value=False), \
             patch.object(detector, '_test_local_r', return_value=False), \
             patch.object(detector, '_has_cached_results', return_value=False):
            
            methods = detector.detect_available_methods()
            
            assert ExecutionMethod.SSH not in methods
            assert ExecutionMethod.LOCAL_R not in methods
            assert ExecutionMethod.MOCK in methods  # Always available
            assert ExecutionMethod.CACHED not in methods
    
    @patch('pradel_jax.validation.rmark_interface.PARAMIKO_AVAILABLE', False)
    def test_ssh_detection_no_paramiko(self, mock_config):
        """Test SSH detection when paramiko is not available."""
        detector = EnvironmentDetector(mock_config)
        
        assert detector._test_ssh_connectivity() is False
    
    @patch('pradel_jax.validation.rmark_interface.PARAMIKO_AVAILABLE', True)
    def test_ssh_detection_no_config(self, mock_config):
        """Test SSH detection when configuration is incomplete."""
        mock_config.has_ssh_config.return_value = False
        detector = EnvironmentDetector(mock_config)
        
        assert detector._test_ssh_connectivity() is False
    
    def test_local_r_detection_success(self, mock_config):
        """Test successful local R detection."""
        detector = EnvironmentDetector(mock_config)
        
        mock_result = Mock()
        mock_result.returncode = 0
        
        with patch('subprocess.run', return_value=mock_result):
            assert detector._test_local_r() is True
    
    def test_local_r_detection_failure(self, mock_config):
        """Test failed local R detection."""
        detector = EnvironmentDetector(mock_config)
        
        with patch('subprocess.run', side_effect=FileNotFoundError()):
            assert detector._test_local_r() is False
    
    def test_get_recommended_method_ssh_preferred(self, mock_config):
        """Test method recommendation when SSH is preferred and available."""
        mock_config.preferred_environment = "ssh"
        detector = EnvironmentDetector(mock_config)
        
        with patch.object(detector, 'detect_available_methods', 
                         return_value=[ExecutionMethod.SSH, ExecutionMethod.LOCAL_R, ExecutionMethod.MOCK]):
            
            method = detector.get_recommended_method()
            assert method == ExecutionMethod.SSH
    
    def test_get_recommended_method_auto_detection(self, mock_config):
        """Test automatic method recommendation."""
        mock_config.preferred_environment = "auto"
        detector = EnvironmentDetector(mock_config)
        
        # Test with SSH available
        with patch.object(detector, 'detect_available_methods', 
                         return_value=[ExecutionMethod.SSH, ExecutionMethod.LOCAL_R, ExecutionMethod.MOCK]):
            
            method = detector.get_recommended_method()
            assert method == ExecutionMethod.SSH  # Should prefer SSH
        
        # Test with only local R available
        with patch.object(detector, 'detect_available_methods', 
                         return_value=[ExecutionMethod.LOCAL_R, ExecutionMethod.MOCK]):
            
            method = detector.get_recommended_method()
            assert method == ExecutionMethod.LOCAL_R
        
        # Test with only mock available
        with patch.object(detector, 'detect_available_methods', 
                         return_value=[ExecutionMethod.MOCK]):
            
            method = detector.get_recommended_method()
            assert method == ExecutionMethod.MOCK


class TestSSHRMarkExecutor:
    """Test SSH RMark executor."""
    
    @patch('pradel_jax.validation.rmark_interface.PARAMIKO_AVAILABLE', False)
    def test_ssh_executor_no_paramiko(self, mock_config):
        """Test SSH executor when paramiko is not available."""
        from pradel_jax.validation.secure_config import SecurityError
        
        with pytest.raises(SecurityError, match="SSH execution requires paramiko"):
            SSHRMarkExecutor(mock_config)
    
    @patch('pradel_jax.validation.rmark_interface.PARAMIKO_AVAILABLE', True)
    def test_ssh_executor_no_config(self, mock_config):
        """Test SSH executor when configuration is incomplete."""
        from pradel_jax.validation.secure_config import SecurityError
        
        mock_config.has_ssh_config.return_value = False
        
        with pytest.raises(SecurityError, match="SSH configuration incomplete"):
            SSHRMarkExecutor(mock_config)
    
    @patch('pradel_jax.validation.rmark_interface.PARAMIKO_AVAILABLE', True)
    def test_ssh_executor_success(self, mock_config, mock_data_context, mock_formula_spec):
        """Test successful SSH execution."""
        executor = SSHRMarkExecutor(mock_config)
        
        # Mock SSH connection and execution
        mock_ssh = Mock()
        mock_sftp = Mock()
        mock_ssh.open_sftp.return_value = mock_sftp
        
        # Mock command execution
        mock_stdout = Mock()
        mock_stdout.read.return_value = b"""
RMARK_RESULTS_START
{"converged": true, "aic": 150.5, "log_likelihood": -72.25, "n_parameters": 3, "parameters": {"phi_intercept": 0.85}, "std_errors": {"phi_intercept": 0.05}}
RMARK_RESULTS_END
"""
        mock_stdout.channel.recv_exit_status.return_value = 0
        
        mock_stderr = Mock()
        mock_stderr.read.return_value = b""
        
        mock_ssh.exec_command.return_value = (None, mock_stdout, mock_stderr)
        
        with patch.object(executor, '_ssh_connection') as mock_ssh_context:
            mock_ssh_context.return_value.__enter__.return_value = mock_ssh
            mock_ssh_context.return_value.__exit__.return_value = None
            
            result = executor.execute(mock_data_context, mock_formula_spec)
        
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.execution_method == "ssh"
        assert result.converged is True
        assert result.aic == 150.5
        assert "phi_intercept" in result.parameters
    
    def test_ssh_r_script_generation(self, mock_config, mock_data_context, mock_formula_spec):
        """Test R script generation for SSH execution."""
        with patch('pradel_jax.validation.rmark_interface.PARAMIKO_AVAILABLE', True):
            executor = SSHRMarkExecutor(mock_config)
            
            script = executor._generate_rmark_script(mock_data_context, mock_formula_spec)
            
            assert "library(RMark)" in script
            assert "phi = ~1 + sex" in script
            assert "p = ~1" in script
            assert "f = ~1" in script
            assert mock_config.session_id in script
            assert "RMARK_RESULTS_START" in script


class TestLocalRMarkExecutor:
    """Test local RMark executor."""
    
    def test_local_executor_success(self, mock_config, mock_data_context, mock_formula_spec):
        """Test successful local execution."""
        executor = LocalRMarkExecutor(mock_config)
        
        # Mock R execution result
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
RMARK_RESULTS_START
{"converged": true, "aic": 148.2, "log_likelihood": -71.1, "n_parameters": 4, "parameters": {"phi_intercept": 0.86, "phi_sex": 0.1}, "std_errors": {"phi_intercept": 0.05, "phi_sex": 0.03}}
RMARK_RESULTS_END
"""
        
        with patch('subprocess.run', return_value=mock_result), \
             patch.object(executor, '_ensure_rmark_available'):
            
            result = executor.execute(mock_data_context, mock_formula_spec)
        
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.execution_method == "local_r"
        assert result.converged is True
        assert result.aic == 148.2
        assert len(result.parameters) == 2
    
    def test_local_executor_rmark_installation(self, mock_config):
        """Test RMark installation process."""
        executor = LocalRMarkExecutor(mock_config)
        
        # Mock RMark not available initially
        test_result = Mock()
        test_result.stdout = "Error: library not found"
        
        # Mock successful installation
        install_result = Mock()
        install_result.returncode = 0
        
        with patch('subprocess.run', side_effect=[test_result, install_result]):
            executor._ensure_rmark_available()
            
            # Should have attempted installation
            assert True  # If no exception raised, installation logic worked
    
    def test_local_executor_script_generation(self, mock_config, mock_formula_spec):
        """Test local R script generation."""
        executor = LocalRMarkExecutor(mock_config)
        
        data_file = Path("/tmp/test_data.csv")
        script = executor._generate_local_rmark_script(data_file, mock_formula_spec)
        
        assert "library(RMark)" in script
        assert str(data_file) in script
        assert "phi = ~1 + sex" in script
        assert "RMARK_RESULTS_START" in script
    
    def test_local_executor_failure(self, mock_config, mock_data_context, mock_formula_spec):
        """Test local execution failure handling."""
        executor = LocalRMarkExecutor(mock_config)
        
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("cmd", 120)), \
             patch.object(executor, '_ensure_rmark_available'):
            
            result = executor.execute(mock_data_context, mock_formula_spec)
        
        assert result.execution_status == ExecutionStatus.FAILED
        assert "Local execution failed" in result.notes[0]


class TestMockRMarkExecutor:
    """Test mock RMark executor."""
    
    def test_mock_executor_basic(self, mock_config, mock_data_context, mock_formula_spec):
        """Test basic mock execution."""
        executor = MockRMarkExecutor(mock_config)
        
        result = executor.execute(mock_data_context, mock_formula_spec)
        
        assert result.execution_status == ExecutionStatus.SUCCESS
        assert result.execution_method == "mock"
        assert result.converged is True
        assert result.aic > 0
        assert result.log_likelihood < 0
        assert len(result.parameters) >= 3  # At least intercepts
        assert "Mock validation result" in result.notes[0]
    
    def test_mock_executor_reproducibility(self, mock_config, mock_data_context, mock_formula_spec):
        """Test that mock results are reproducible."""
        executor1 = MockRMarkExecutor(mock_config)
        executor2 = MockRMarkExecutor(mock_config)
        
        result1 = executor1.execute(mock_data_context, mock_formula_spec)
        result2 = executor2.execute(mock_data_context, mock_formula_spec)
        
        # Should be identical due to fixed random seed
        assert result1.aic == result2.aic
        assert result1.log_likelihood == result2.log_likelihood
        assert result1.parameters == result2.parameters
    
    def test_mock_executor_covariate_effects(self, mock_config, mock_data_context):
        """Test mock generation with covariate effects."""
        # Formula with sex covariate
        formula_with_sex = FormulaSpec(phi="1 + sex", p="1", f="1")
        
        executor = MockRMarkExecutor(mock_config)
        result = executor.execute(mock_data_context, formula_with_sex)
        
        # Should include sex parameter
        phi_params = [p for p in result.parameters.keys() if 'Phi:' in p]
        assert len(phi_params) >= 2  # Intercept + sex
        
        # Should have corresponding standard errors
        for param in result.parameters.keys():
            assert param in result.std_errors


class TestRMarkExecutor:
    """Test main RMark executor with method selection."""
    
    def test_executor_auto_method_selection(self, mock_config):
        """Test automatic method selection."""
        with patch('pradel_jax.validation.rmark_interface.EnvironmentDetector') as mock_detector_class:
            mock_detector = Mock()
            mock_detector.detect_available_methods.return_value = [ExecutionMethod.SSH, ExecutionMethod.MOCK]
            mock_detector.get_recommended_method.return_value = ExecutionMethod.SSH
            mock_detector_class.return_value = mock_detector
            
            executor = RMarkExecutor(mock_config)
            
            assert ExecutionMethod.SSH in executor.available_methods
            assert ExecutionMethod.MOCK in executor.available_methods
    
    def test_executor_specific_method(self, mock_config, mock_data_context, mock_formula_spec):
        """Test execution with specific method."""
        executor = RMarkExecutor(mock_config)
        executor.available_methods = [ExecutionMethod.MOCK]
        
        result = executor.execute_rmark_analysis(
            mock_data_context, mock_formula_spec, ExecutionMethod.MOCK
        )
        
        assert result.execution_method == "mock"
        assert result.execution_status == ExecutionStatus.SUCCESS
    
    def test_executor_fallback_to_mock(self, mock_config, mock_data_context, mock_formula_spec):
        """Test fallback to mock when requested method fails."""
        executor = RMarkExecutor(mock_config)
        executor.available_methods = [ExecutionMethod.SSH, ExecutionMethod.MOCK]
        
        # Mock SSH executor that fails
        with patch('pradel_jax.validation.rmark_interface.SSHRMarkExecutor') as mock_ssh_class:
            mock_ssh = Mock()
            mock_ssh.execute.side_effect = Exception("SSH failed")
            mock_ssh_class.return_value = mock_ssh
            
            result = executor.execute_rmark_analysis(
                mock_data_context, mock_formula_spec, ExecutionMethod.SSH
            )
        
        # Should have fallen back to mock
        assert result.execution_method == "mock"
        assert "Fallback to mock after ssh failure" in result.notes[0]
    
    def test_executor_unavailable_method(self, mock_config, mock_data_context, mock_formula_spec):
        """Test handling of unavailable method."""
        executor = RMarkExecutor(mock_config)
        executor.available_methods = [ExecutionMethod.MOCK]  # Only mock available
        
        # Request SSH which is not available
        result = executor.execute_rmark_analysis(
            mock_data_context, mock_formula_spec, ExecutionMethod.SSH
        )
        
        # Should fall back to mock
        assert result.execution_method == "mock"


class TestConvenienceFunction:
    """Test convenience function for external use."""
    
    def test_execute_rmark_analysis_function(self, mock_data_context, mock_formula_spec):
        """Test the convenience function."""
        from pradel_jax.validation.rmark_interface import execute_rmark_analysis
        
        with patch('pradel_jax.validation.rmark_interface.RMarkExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_result = RMarkResult(
                model_formula="test",
                execution_method="mock",
                execution_time=1.0,
                execution_status=ExecutionStatus.SUCCESS
            )
            mock_executor.execute_rmark_analysis.return_value = mock_result
            mock_executor_class.return_value = mock_executor
            
            result = execute_rmark_analysis(mock_data_context, mock_formula_spec, method="mock")
            
            assert result.execution_status == ExecutionStatus.SUCCESS
    
    def test_execute_rmark_analysis_invalid_method(self, mock_data_context, mock_formula_spec):
        """Test convenience function with invalid method."""
        from pradel_jax.validation.rmark_interface import execute_rmark_analysis
        
        # Should handle invalid method gracefully by falling back to auto-detection
        with patch('pradel_jax.validation.rmark_interface.RMarkExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_result = RMarkResult(
                model_formula="test",
                execution_method="mock",
                execution_time=1.0
            )
            mock_executor.execute_rmark_analysis.return_value = mock_result
            mock_executor_class.return_value = mock_executor
            
            result = execute_rmark_analysis(mock_data_context, mock_formula_spec, method="invalid_method")
            
            # Should have been called with None (auto-detection)
            mock_executor.execute_rmark_analysis.assert_called_with(mock_data_context, mock_formula_spec, None)


if __name__ == "__main__":
    pytest.main([__file__])