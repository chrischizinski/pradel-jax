"""
Tests for secure configuration management.

These tests verify that the security-first configuration system correctly
loads settings from environment variables while protecting sensitive data.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from pradel_jax.validation.secure_config import (
    SecureValidationConfig,
    SecureConfigLoader,
    ValidationCriteria,
    SecuritySettings,
    SecurityError,
    get_secure_validation_config,
    reset_configuration
)


class TestSecureValidationConfig:
    """Test secure validation configuration."""
    
    def setup_method(self):
        """Setup for each test."""
        # Reset global configuration
        reset_configuration()
        
        # Clear environment variables to start clean
        self.original_env = {}
        env_vars = [
            'RMARK_SSH_HOST', 'RMARK_SSH_USER', 'RMARK_SSH_KEY_PATH', 'RMARK_R_PATH',
            'RMARK_SSH_PORT', 'RMARK_SSH_TIMEOUT', 'RMARK_LOCAL_R_PATH',
            'RMARK_LOCAL_R_TIMEOUT', 'RMARK_PREFERRED_ENVIRONMENT',
            'RMARK_MAX_EXECUTION_TIME', 'RMARK_SESSION_ID'
        ]
        
        for var in env_vars:
            self.original_env[var] = os.environ.pop(var, None)
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        for var, value in self.original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]
        
        # Reset configuration
        reset_configuration()
    
    def test_default_config(self):
        """Test default configuration with no environment variables."""
        config = SecureValidationConfig()
        
        assert config.ssh_host is None
        assert config.ssh_user is None
        assert config.ssh_key_path is None
        assert config.ssh_r_path is None
        assert config.ssh_port == 22
        assert config.local_r_path == "Rscript"
        assert config.preferred_environment == "auto"
        assert not config.has_ssh_config()
        assert config.has_local_r_config()
    
    def test_ssh_config_from_environment(self):
        """Test SSH configuration loading from environment variables."""
        os.environ['RMARK_SSH_HOST'] = '192.168.1.100'
        os.environ['RMARK_SSH_USER'] = 'testuser'
        os.environ['RMARK_SSH_KEY_PATH'] = '~/.ssh/test_key'
        os.environ['RMARK_R_PATH'] = 'C:\\Program Files\\R\\bin\\Rscript.exe'
        os.environ['RMARK_SSH_PORT'] = '2222'
        
        config = SecureValidationConfig()
        
        assert config.ssh_host == '192.168.1.100'
        assert config.ssh_user == 'testuser'
        assert config.ssh_key_path == '~/.ssh/test_key'
        assert config.ssh_r_path == 'C:\\Program Files\\R\\bin\\Rscript.exe'
        assert config.ssh_port == 2222
        assert config.has_ssh_config()
    
    def test_partial_ssh_config_validation(self):
        """Test validation catches partial SSH configuration."""
        os.environ['RMARK_SSH_HOST'] = '192.168.1.100'
        # Missing user and key path
        
        with patch('pradel_jax.validation.secure_config.logger') as mock_logger:
            config = SecureValidationConfig()
            
            # Should log warning about partial configuration
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Partial SSH configuration detected" in warning_call
            
            assert not config.has_ssh_config()
    
    def test_invalid_port_handling(self):
        """Test handling of invalid port values."""
        os.environ['RMARK_SSH_PORT'] = 'invalid'
        
        with patch('pradel_jax.validation.secure_config.logger') as mock_logger:
            config = SecureValidationConfig()
            
            # Should use default port and log warning
            assert config.ssh_port == 22
            mock_logger.warning.assert_called()
    
    def test_temp_directory_creation(self):
        """Test temporary directory creation and permissions."""
        config = SecureValidationConfig()
        
        assert config.temp_dir is not None
        assert config.temp_dir.exists()
        assert "pradel_validation" in str(config.temp_dir)
        
        # Check that it's process-specific
        config2 = SecureValidationConfig()
        assert config.temp_dir != config2.temp_dir
    
    def test_security_settings(self):
        """Test security settings configuration."""
        config = SecureValidationConfig()
        
        assert config.security.max_execution_time_seconds == 1800  # 30 minutes
        assert config.security.cleanup_temp_files_on_success is True
        assert config.security.enable_audit_logging is True
        assert config.security.use_process_specific_temp_dirs is True
    
    def test_cleanup(self):
        """Test configuration cleanup."""
        config = SecureValidationConfig()
        temp_dir = config.temp_dir
        
        assert temp_dir.exists()
        config.cleanup()
        assert not temp_dir.exists()
    
    def test_connection_summary_no_secrets(self):
        """Test that connection summary doesn't expose secrets."""
        os.environ['RMARK_SSH_HOST'] = 'secret.host.com'
        os.environ['RMARK_SSH_USER'] = 'secretuser'
        
        config = SecureValidationConfig()
        summary = config.get_connection_summary()
        
        # Should not contain actual credentials
        assert 'secret.host.com' not in str(summary)
        assert 'secretuser' not in str(summary)
        
        # Should contain status information
        assert summary['ssh_available'] is True
        assert summary['ssh_host_configured'] is True
        assert summary['ssh_user_configured'] is True


class TestSecureConfigLoader:
    """Test secure configuration loader."""
    
    def setup_method(self):
        """Setup for each test."""
        reset_configuration()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_configuration()
    
    def test_load_from_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_content = """
validation:
  criteria:
    parameter_absolute_tolerance: 0.002
    max_aic_difference: 1.5
  security:
    max_execution_time_minutes: 45
    cleanup_temp_files_on_success: false
  output:
    base_output_dir: "./custom_validation_results"
    generate_pdf_report: true
"""
        
        config_file = self.temp_path / "validation_config.yaml"
        config_file.write_text(config_content)
        
        loader = SecureConfigLoader()
        loader.config_search_paths = [config_file]
        
        config = loader.load_config()
        
        assert config.criteria.parameter_absolute_tolerance == 0.002
        assert config.criteria.max_aic_difference == 1.5
        assert config.security.max_execution_time_seconds == 45 * 60
        assert config.security.cleanup_temp_files_on_success is False
        assert config.output_base_dir == "./custom_validation_results"
        assert config.generate_pdf_report is True
    
    def test_load_with_no_config_file(self):
        """Test loading with no configuration file."""
        loader = SecureConfigLoader()
        loader.config_search_paths = [Path("nonexistent.yaml")]
        
        config = loader.load_config()
        
        # Should use defaults
        assert config.criteria.parameter_absolute_tolerance == 1e-3
        assert config.output_base_dir == "./validation_results"
    
    def test_load_with_invalid_yaml(self):
        """Test handling of invalid YAML file."""
        config_file = self.temp_path / "invalid_config.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        loader = SecureConfigLoader()
        loader.config_search_paths = [config_file]
        
        with patch('pradel_jax.validation.secure_config.logger') as mock_logger:
            config = loader.load_config()
            
            # Should log warning and use defaults
            mock_logger.warning.assert_called()
            assert config.criteria.parameter_absolute_tolerance == 1e-3


class TestGlobalConfiguration:
    """Test global configuration management."""
    
    def setup_method(self):
        """Setup for each test."""
        reset_configuration()
    
    def teardown_method(self):
        """Cleanup after each test."""
        reset_configuration()
    
    def test_singleton_pattern(self):
        """Test that configuration follows singleton pattern."""
        config1 = get_secure_validation_config()
        config2 = get_secure_validation_config()
        
        assert config1 is config2
    
    def test_reset_configuration(self):
        """Test configuration reset."""
        config1 = get_secure_validation_config()
        config1_id = id(config1)
        
        reset_configuration()
        
        config2 = get_secure_validation_config()
        config2_id = id(config2)
        
        assert config1_id != config2_id


class TestValidationCriteria:
    """Test validation criteria configuration."""
    
    def test_default_criteria(self):
        """Test default validation criteria."""
        criteria = ValidationCriteria()
        
        assert criteria.parameter_absolute_tolerance == 1e-3
        assert criteria.parameter_relative_tolerance_pct == 5.0
        assert criteria.equivalence_margin == 0.05
        assert criteria.max_aic_difference == 2.0
        assert criteria.min_ranking_concordance == 0.8
        assert criteria.min_convergence_rate == 0.95
        assert "phi_intercept" in criteria.critical_parameters
    
    def test_custom_criteria(self):
        """Test custom validation criteria."""
        criteria = ValidationCriteria(
            parameter_absolute_tolerance=0.005,
            max_aic_difference=1.0,
            critical_parameters=["custom_param"]
        )
        
        assert criteria.parameter_absolute_tolerance == 0.005
        assert criteria.max_aic_difference == 1.0
        assert criteria.critical_parameters == ["custom_param"]


class TestSecuritySettings:
    """Test security settings configuration."""
    
    def test_default_security_settings(self):
        """Test default security settings."""
        settings = SecuritySettings()
        
        assert settings.max_execution_time_seconds == 1800
        assert settings.cleanup_temp_files_on_success is True
        assert settings.enable_audit_logging is True
        assert settings.use_process_specific_temp_dirs is True
        assert settings.temp_dir_permissions == 0o700
    
    def test_custom_security_settings(self):
        """Test custom security settings."""
        settings = SecuritySettings(
            max_execution_time_seconds=3600,
            cleanup_temp_files_on_success=False,
            enable_audit_logging=False
        )
        
        assert settings.max_execution_time_seconds == 3600
        assert settings.cleanup_temp_files_on_success is False
        assert settings.enable_audit_logging is False


class TestSecurityValidation:
    """Test security validation features."""
    
    def test_template_value_detection(self):
        """Test detection of template values in configuration."""
        os.environ['RMARK_SSH_HOST'] = 'template_host'
        os.environ['RMARK_SSH_USER'] = 'template_user'
        os.environ['RMARK_SSH_KEY_PATH'] = '/path/to/template_key'
        
        with patch('pradel_jax.validation.secure_config.logger') as mock_logger:
            config = SecureValidationConfig()
            
            # Should detect template values and log warnings
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            template_warnings = [w for w in warning_calls if 'template' in w.lower()]
            
            assert len(template_warnings) >= 2  # Host and key path
    
    def test_long_execution_time_warning(self):
        """Test warning for excessively long execution times."""
        os.environ['RMARK_MAX_EXECUTION_TIME'] = '7200'  # 2 hours
        
        with patch('pradel_jax.validation.secure_config.logger') as mock_logger:
            config = SecureValidationConfig()
            
            # Should warn about long execution time
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            time_warnings = [w for w in warning_calls if 'execution time' in w]
            
            assert len(time_warnings) >= 1
    
    @patch('pradel_jax.validation.secure_config.logger')
    def test_audit_logging(self, mock_logger):
        """Test security audit logging."""
        config = SecureValidationConfig()
        
        # Should log configuration loading
        config._log_security_event("test_event", "test_details")
        
        # Verify audit log was written (mocked)
        assert mock_logger.info.called


if __name__ == "__main__":
    pytest.main([__file__])