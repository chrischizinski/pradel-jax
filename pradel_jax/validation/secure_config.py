"""
Secure Configuration Management for RMark Validation.

This module provides security-first configuration loading that ensures no
credentials or sensitive information is stored in code or committed to git.
All sensitive configuration is loaded from environment variables.

Design Principles:
    - Zero secrets in code or configuration files
    - Environment variable-based credential loading
    - Comprehensive security validation
    - Audit logging for security events
    - Process-specific temporary directories

Security Architecture:
    - Credentials loaded only from environment variables
    - Configuration files contain no sensitive data
    - Temporary directories are process-specific and cleaned up
    - Security events are logged for audit purposes
    - Configuration validation detects security issues
"""

import os
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import yaml

from ..utils.logging import get_logger

logger = get_logger(__name__)


class SecurityError(Exception):
    """Security-related configuration error."""
    
    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.suggestions = suggestions or []


@dataclass
class ValidationCriteria:
    """Statistical validation criteria with industry-standard defaults."""
    
    # Parameter-level tolerances (bioequivalence standards)
    parameter_absolute_tolerance: float = 1e-3  # 0.001 absolute difference
    parameter_relative_tolerance_pct: float = 5.0  # 5% relative difference
    equivalence_margin: float = 0.05  # ±5% for TOST equivalence testing
    equivalence_alpha: float = 0.05  # 95% confidence for equivalence
    require_confidence_overlap: bool = True
    
    # Model-level tolerances (ecological significance)
    max_aic_difference: float = 2.0  # Ecological significance threshold
    max_likelihood_relative_diff_pct: float = 1.0  # 1% log-likelihood difference
    min_ranking_concordance: float = 0.8  # Kendall's tau ≥ 0.8
    
    # System-level requirements
    min_convergence_rate: float = 0.95  # 95% of models must converge
    min_pass_rate_for_approval: float = 0.90  # 90% of tests must pass
    max_execution_time_minutes: float = 30.0  # Maximum execution time
    
    # Critical parameters that MUST pass validation
    critical_parameters: List[str] = field(default_factory=lambda: [
        "phi_intercept", "p_intercept", "f_intercept"
    ])


@dataclass
class SecuritySettings:
    """Security configuration with secure defaults."""
    
    # Execution security
    max_execution_time_seconds: int = 1800  # 30 minutes absolute maximum
    cleanup_temp_files_on_success: bool = True
    cleanup_temp_files_on_error: bool = True
    
    # Audit and logging
    enable_audit_logging: bool = True
    audit_log_path: str = "./logs/validation_security_audit.log"
    
    # Process isolation
    use_process_specific_temp_dirs: bool = True
    temp_dir_permissions: int = 0o700  # Owner read/write/execute only


@dataclass
class SecureValidationConfig:
    """
    Security-focused validation configuration.
    
    All sensitive data (SSH credentials, paths) are loaded from environment
    variables to ensure they are never committed to git.
    """
    
    # SSH configuration (loaded from environment)
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None
    ssh_key_path: Optional[str] = None
    ssh_r_path: Optional[str] = None
    ssh_port: int = 22
    ssh_timeout: int = 300
    ssh_max_retries: int = 3
    
    # Local R configuration  
    local_r_path: str = "Rscript"
    local_r_timeout: int = 180
    local_temp_dir_base: str = "/tmp"
    auto_install_rmark: bool = True
    
    # Environment detection
    preferred_environment: str = "auto"  # "auto", "ssh", "local_r", "mock"
    
    # Validation criteria
    criteria: ValidationCriteria = field(default_factory=ValidationCriteria)
    
    # Security settings
    security: SecuritySettings = field(default_factory=SecuritySettings)
    
    # Output configuration
    output_base_dir: str = "./validation_results"
    generate_html_report: bool = True
    generate_pdf_report: bool = False
    save_detailed_results: bool = True
    
    # Session metadata
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    temp_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Load secure configuration from environment and validate."""
        self._load_environment_config()
        self._create_session_temp_dir()
        self._validate_security_configuration()
        self._log_configuration_status()
    
    def _load_environment_config(self):
        """Load sensitive configuration from environment variables."""
        # SSH configuration (secure - from environment only)
        self.ssh_host = os.getenv("RMARK_SSH_HOST")
        self.ssh_user = os.getenv("RMARK_SSH_USER")
        self.ssh_key_path = os.getenv("RMARK_SSH_KEY_PATH")
        self.ssh_r_path = os.getenv("RMARK_R_PATH")
        
        # Optional SSH settings
        if port_str := os.getenv("RMARK_SSH_PORT"):
            try:
                self.ssh_port = int(port_str)
            except ValueError:
                logger.warning(f"Invalid SSH port '{port_str}', using default {self.ssh_port}")
        
        if timeout_str := os.getenv("RMARK_SSH_TIMEOUT"):
            try:
                self.ssh_timeout = int(timeout_str)
            except ValueError:
                logger.warning(f"Invalid SSH timeout '{timeout_str}', using default {self.ssh_timeout}")
        
        # Local R configuration (can be overridden by environment)
        self.local_r_path = os.getenv("RMARK_LOCAL_R_PATH", self.local_r_path)
        
        if local_timeout_str := os.getenv("RMARK_LOCAL_R_TIMEOUT"):
            try:
                self.local_r_timeout = int(local_timeout_str)
            except ValueError:
                logger.warning(f"Invalid local R timeout '{local_timeout_str}', using default {self.local_r_timeout}")
        
        # Environment preference
        self.preferred_environment = os.getenv("RMARK_PREFERRED_ENVIRONMENT", self.preferred_environment)
        
        # Security settings
        if max_time_str := os.getenv("RMARK_MAX_EXECUTION_TIME"):
            try:
                self.security.max_execution_time_seconds = int(max_time_str)
            except ValueError:
                logger.warning(f"Invalid max execution time '{max_time_str}', using default")
        
        # Session identification
        if session_id := os.getenv("RMARK_SESSION_ID"):
            self.session_id = session_id
    
    def _create_session_temp_dir(self):
        """Create process-specific temporary directory."""
        if self.security.use_process_specific_temp_dirs:
            # Create process-specific temp directory
            temp_base = Path(tempfile.gettempdir())
            self.temp_dir = temp_base / f"pradel_validation_{self.session_id}"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Set secure permissions (owner only)
            self.temp_dir.chmod(self.security.temp_dir_permissions)
        else:
            self.temp_dir = Path(self.local_temp_dir_base) / "pradel_validation"
            self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def _validate_security_configuration(self):
        """Validate configuration for security issues."""
        security_issues = []
        
        # Check SSH configuration consistency
        ssh_fields = [self.ssh_host, self.ssh_user, self.ssh_key_path]
        ssh_configured_count = sum(1 for field in ssh_fields if field)
        
        if 0 < ssh_configured_count < 3:
            security_issues.append(
                "Partial SSH configuration detected - all or none of "
                "RMARK_SSH_HOST, RMARK_SSH_USER, RMARK_SSH_KEY_PATH should be set"
            )
        
        # Check for potential template values that might be in git
        if self.ssh_host and "template" in self.ssh_host.lower():
            security_issues.append("SSH host appears to reference template value")
        
        if self.ssh_key_path and "template" in self.ssh_key_path.lower():
            security_issues.append("SSH key path appears to reference template value")
        
        # Check temp directory security
        if not self.security.use_process_specific_temp_dirs:
            security_issues.append("Not using process-specific temp directories (potential security risk)")
        
        # Check execution time limits
        if self.security.max_execution_time_seconds > 3600:  # 1 hour
            security_issues.append(f"Very long max execution time ({self.security.max_execution_time_seconds}s) may indicate security risk")
        
        # Log security issues
        for issue in security_issues:
            logger.warning(f"Security validation: {issue}")
            self._log_security_event("security_validation_warning", issue)
        
        if security_issues:
            logger.warning("Security validation completed with warnings - review configuration")
    
    def _log_configuration_status(self):
        """Log configuration status without exposing secrets."""
        status = {
            "session_id": self.session_id,
            "ssh_configured": self.has_ssh_config(),
            "local_r_configured": bool(self.local_r_path),
            "preferred_environment": self.preferred_environment,
            "temp_dir": str(self.temp_dir),
            "security_enabled": True
        }
        
        logger.info(f"Validation configuration loaded: {status}")
        self._log_security_event("configuration_loaded", str(status))
    
    def _log_security_event(self, event_type: str, details: str):
        """Log security event for audit purposes."""
        if not self.security.enable_audit_logging:
            return
        
        try:
            audit_log_path = Path(self.security.audit_log_path)
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            import datetime
            timestamp = datetime.datetime.now().isoformat()
            log_entry = f"{timestamp} [{self.session_id}] {event_type}: {details}\n"
            
            with open(audit_log_path, "a") as f:
                f.write(log_entry)
                
        except Exception as e:
            logger.warning(f"Failed to write security audit log: {e}")
    
    def has_ssh_config(self) -> bool:
        """Check if complete SSH configuration is available."""
        return all([
            self.ssh_host,
            self.ssh_user,
            self.ssh_key_path
        ])
    
    def has_local_r_config(self) -> bool:
        """Check if local R configuration is available."""
        return bool(self.local_r_path)
    
    def has_rmark_config(self) -> bool:
        """Check if RMark execution configuration is available."""
        # Check SSH configuration first
        if self.has_ssh_config():
            return True
        
        # Check local R configuration
        if self.has_local_r_config():
            return True
        
        # No valid RMark execution method available
        return False
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get connection summary without exposing credentials."""
        return {
            "ssh_available": self.has_ssh_config(),
            "ssh_host_configured": bool(self.ssh_host),
            "ssh_user_configured": bool(self.ssh_user),
            "ssh_key_configured": bool(self.ssh_key_path),
            "local_r_available": self.has_local_r_config(),
            "local_r_path": self.local_r_path,
            "preferred_environment": self.preferred_environment,
            "session_id": self.session_id,
            "temp_dir": str(self.temp_dir),
            "security_features": {
                "audit_logging": self.security.enable_audit_logging,
                "temp_dir_cleanup": self.security.cleanup_temp_files_on_success,
                "process_specific_dirs": self.security.use_process_specific_temp_dirs
            }
        }
    
    def cleanup(self):
        """Clean up temporary resources."""
        if self.temp_dir and self.temp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
                self._log_security_event("temp_dir_cleanup", str(self.temp_dir))
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {self.temp_dir}: {e}")


class SecureConfigLoader:
    """Load validation configuration with security-first principles."""
    
    def __init__(self):
        self.config_search_paths = [
            Path("config/validation_config.yaml"),
            Path("~/.config/pradel/validation.yaml").expanduser(),
            # Template is fallback for defaults only (no sensitive data)
            Path("config/validation_config_template.yaml")
        ]
    
    def load_config(self) -> SecureValidationConfig:
        """Load configuration with security validation."""
        logger.info("Loading secure validation configuration...")
        
        # Start with defaults
        config_data = {}
        
        # Try loading configuration files (non-sensitive settings only)
        config_file_loaded = False
        for config_path in self.config_search_paths:
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        file_config = yaml.safe_load(f)
                    
                    if file_config and 'validation' in file_config:
                        config_data = file_config['validation']
                        logger.info(f"Loaded configuration from: {config_path}")
                        config_file_loaded = True
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
        
        if not config_file_loaded:
            logger.info("No configuration file found, using defaults with environment variables")
        
        # Create secure config (automatically loads sensitive data from environment)
        secure_config = SecureValidationConfig()
        
        # Apply non-sensitive file-based settings
        if 'criteria' in config_data:
            criteria_config = config_data['criteria']
            secure_config.criteria = ValidationCriteria(
                parameter_absolute_tolerance=criteria_config.get('parameter_absolute_tolerance', 1e-3),
                parameter_relative_tolerance_pct=criteria_config.get('parameter_relative_tolerance_pct', 5.0),
                max_aic_difference=criteria_config.get('max_aic_difference', 2.0),
                max_likelihood_relative_diff_pct=criteria_config.get('max_likelihood_relative_diff_pct', 1.0),
                min_ranking_concordance=criteria_config.get('min_ranking_concordance', 0.8),
                min_convergence_rate=criteria_config.get('min_convergence_rate', 0.95),
                min_pass_rate_for_approval=criteria_config.get('min_pass_rate_for_approval', 0.90)
            )
        
        if 'security' in config_data:
            security_config = config_data['security']
            secure_config.security = SecuritySettings(
                max_execution_time_seconds=security_config.get('max_execution_time_minutes', 30) * 60,
                cleanup_temp_files_on_success=security_config.get('cleanup_temp_files_on_success', True),
                cleanup_temp_files_on_error=security_config.get('cleanup_temp_files_on_error', True),
                enable_audit_logging=security_config.get('enable_audit_logging', True)
            )
        
        if 'output' in config_data:
            output_config = config_data['output']
            secure_config.output_base_dir = output_config.get('base_output_dir', './validation_results')
            secure_config.generate_html_report = output_config.get('generate_html_report', True)
            secure_config.generate_pdf_report = output_config.get('generate_pdf_report', False)
            secure_config.save_detailed_results = output_config.get('save_detailed_results', True)
        
        # Apply local R settings (non-sensitive)
        if 'local_r' in config_data:
            local_r_config = config_data['local_r']
            # Only apply non-sensitive settings from file
            secure_config.local_r_timeout = local_r_config.get('timeout', 180)
            secure_config.auto_install_rmark = local_r_config.get('auto_install_rmark', True)
            # Note: local_r_path comes from environment or stays default
        
        logger.info("Secure validation configuration loaded successfully")
        return secure_config


# Singleton configuration instance
_secure_config: Optional[SecureValidationConfig] = None

def get_secure_validation_config() -> SecureValidationConfig:
    """Get singleton secure validation configuration."""
    global _secure_config
    if _secure_config is None:
        loader = SecureConfigLoader()
        _secure_config = loader.load_config()
    return _secure_config

def reset_configuration():
    """Reset configuration (for testing purposes)."""
    global _secure_config
    if _secure_config:
        _secure_config.cleanup()
    _secure_config = None