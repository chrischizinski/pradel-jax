"""
Flexible Configuration System for Validation Pipeline (Phase 3).

This module provides comprehensive configuration management for the automated
validation pipeline with quality gates. It extends the secure configuration
from Phase 1 with pipeline-specific settings, quality criteria, and
environment-specific configurations.

Key Features:
    - Hierarchical configuration with validation inheritance
    - Environment-specific settings (home, work, cloud, CI/CD)
    - Quality gate criteria with configurable thresholds
    - Pipeline orchestration settings
    - Performance and monitoring configuration

Usage:
    config = ValidationPipelineConfig.from_environment()
    pipeline = ValidationPipeline(config)
    report = pipeline.run_comprehensive_validation(datasets, models)
"""

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import logging

from .secure_config import SecureValidationConfig

logger = logging.getLogger(__name__)


class ValidationEnvironment(Enum):
    """Supported validation environments."""
    HOME_OFFICE = "home_office"
    WORK_OFFICE = "work_office" 
    CLOUD_SERVICE = "cloud_service"
    CI_CD_PIPELINE = "ci_cd"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class ValidationStatus(Enum):
    """Validation result status levels."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    ERROR = "error"
    SKIP = "skip"


class QualityGateDecision(Enum):
    """Quality gate decisions."""
    APPROVED = "approved"
    APPROVED_WITH_WARNINGS = "approved_with_warnings"
    REJECTED = "rejected"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


@dataclass
class ValidationCriteria:
    """
    Comprehensive validation criteria with industry standards.
    
    Implements bioequivalence guidelines for parameter validation and
    ecological significance thresholds for model comparison.
    """
    
    # Parameter-level thresholds (bioequivalence standards)
    parameter_absolute_tolerance: float = 1e-3  # 0.001 absolute difference
    parameter_relative_tolerance_pct: float = 5.0  # 5% relative difference
    equivalence_margin: float = 0.05  # ±5% for TOST equivalence
    require_confidence_overlap: bool = True
    equivalence_alpha: float = 0.05  # 95% confidence for equivalence
    
    # Model-level thresholds (ecological significance)
    max_aic_difference: float = 2.0  # Ecological significance threshold
    max_likelihood_relative_diff_pct: float = 1.0  # 1% log-likelihood difference
    min_ranking_concordance: float = 0.8  # Kendall's tau ≥ 0.8
    
    # System-level requirements
    min_convergence_rate: float = 0.95  # 95% of models must converge
    max_computation_time_ratio: float = 10.0  # JAX ≤ 10x RMark time
    require_reproducibility: bool = True
    max_numerical_instability: float = 1e-6  # Parameter stability across runs
    
    # Quality gate settings
    min_pass_rate_for_approval: float = 0.90  # 90% of tests must pass
    critical_parameters: List[str] = field(default_factory=lambda: [
        "phi_intercept", "p_intercept", "f_intercept"
    ])
    allow_warnings_in_approval: bool = True
    
    # Statistical power settings
    min_statistical_power: float = 0.80  # 80% power for detecting differences
    bootstrap_n_samples: int = 1000  # Bootstrap sample size
    cross_validation_folds: int = 5  # K-fold cross-validation
    
    def validate_criteria(self) -> None:
        """Validate that criteria values are reasonable."""
        if not 0 < self.parameter_relative_tolerance_pct < 100:
            raise ValueError("Parameter relative tolerance must be between 0 and 100 percent")
        
        if not 0 < self.equivalence_alpha < 1:
            raise ValueError("Equivalence alpha must be between 0 and 1")
        
        if not 0 < self.min_pass_rate_for_approval <= 1:
            raise ValueError("Minimum pass rate must be between 0 and 1")
        
        if self.bootstrap_n_samples < 100:
            raise ValueError("Bootstrap samples must be at least 100")


@dataclass 
class PerformanceConfig:
    """Performance and resource management configuration."""
    
    # Parallel processing
    max_parallel_jobs: int = 4  # Number of parallel validation jobs
    enable_multiprocessing: bool = True
    chunk_size: int = 10  # Models per chunk for parallel processing
    
    # Memory management
    max_memory_usage_gb: float = 8.0  # Maximum memory usage
    enable_memory_monitoring: bool = True
    garbage_collection_frequency: int = 100  # GC every N models
    
    # Timeout settings
    single_model_timeout_seconds: int = 300  # 5 minutes per model
    total_pipeline_timeout_seconds: int = 3600  # 1 hour total
    rmark_execution_timeout_seconds: int = 120  # 2 minutes for RMark
    
    # Performance optimization
    enable_jax_jit: bool = True
    jax_platform_name: Optional[str] = None  # "cpu", "gpu", or None for auto
    enable_result_caching: bool = True
    cache_directory: Optional[Path] = None
    
    def get_cache_directory(self) -> Path:
        """Get cache directory, creating if needed."""
        if self.cache_directory is None:
            self.cache_directory = Path.home() / ".pradel_jax" / "validation_cache"
        
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        return self.cache_directory


@dataclass
class ReportingConfig:
    """Report generation and output configuration."""
    
    # Output formats
    generate_html_report: bool = True
    generate_pdf_report: bool = False  # Requires additional dependencies
    generate_json_summary: bool = True
    generate_csv_results: bool = True
    
    # Report content
    include_diagnostic_plots: bool = True
    include_parameter_tables: bool = True
    include_model_comparison_tables: bool = True
    include_statistical_test_details: bool = True
    include_raw_data_dumps: bool = False  # For debugging only
    
    # Visualization settings
    plot_format: str = "png"  # "png", "svg", "pdf"
    plot_dpi: int = 300  # High resolution for publication
    plot_style: str = "seaborn-v0_8-whitegrid"  # Matplotlib style
    color_palette: str = "colorblind"  # Colorblind-friendly palette
    
    # Archive settings
    archive_results: bool = True
    archive_directory: Optional[Path] = None
    max_archive_age_days: int = 30  # Clean up old results
    
    def get_archive_directory(self) -> Path:
        """Get archive directory, creating if needed."""
        if self.archive_directory is None:
            self.archive_directory = Path.home() / ".pradel_jax" / "validation_archive"
        
        self.archive_directory.mkdir(parents=True, exist_ok=True)
        return self.archive_directory


@dataclass
class ValidationPipelineConfig:
    """
    Comprehensive configuration for validation pipeline orchestration.
    
    This configuration extends SecureValidationConfig with pipeline-specific
    settings for automated validation, quality gates, and reporting.
    """
    
    # Core configuration (inherits from Phase 1)
    secure_config: SecureValidationConfig = field(default_factory=SecureValidationConfig)
    
    # Environment and execution
    environment: ValidationEnvironment = ValidationEnvironment.DEVELOPMENT
    pipeline_name: str = "pradel_jax_validation"
    session_id: Optional[str] = None  # Auto-generated if None
    
    # Validation criteria and quality gates
    validation_criteria: ValidationCriteria = field(default_factory=ValidationCriteria)
    
    # Performance and resource management
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Reporting and output
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    
    # Pipeline behavior
    fail_fast: bool = False  # Continue validation even if some models fail
    retry_failed_models: bool = True
    max_retries: int = 3
    enable_quality_gates: bool = True
    
    # Logging and monitoring
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    enable_progress_bars: bool = True
    enable_mlflow_tracking: bool = False  # MLflow experiment tracking
    mlflow_experiment_name: str = "pradel_jax_validation"
    
    # Development and debugging
    enable_development_mode: bool = False  # Extra logging and validation
    save_intermediate_results: bool = False
    enable_profiling: bool = False
    
    @classmethod
    def from_environment(
        cls, 
        config_file: Optional[Path] = None,
        environment: Optional[ValidationEnvironment] = None
    ) -> "ValidationPipelineConfig":
        """
        Create configuration from environment variables and config files.
        
        Priority order:
        1. Environment variables (highest)
        2. Config file specified in argument
        3. Config file in PRADEL_JAX_CONFIG_FILE env var
        4. Default config file in ~/.pradel_jax/validation_config.yaml
        5. Built-in defaults (lowest)
        """
        
        # Start with defaults
        config = cls()
        
        # Auto-detect environment if not specified
        if environment is None:
            environment = cls._detect_environment()
        config.environment = environment
        
        # Load from config file
        config_path = cls._find_config_file(config_file)
        if config_path and config_path.exists():
            logger.info(f"Loading configuration from {config_path}")
            config = cls._load_from_file(config_path, config)
        
        # Override with environment variables
        config = cls._load_from_environment(config)
        
        # Apply environment-specific defaults
        config = cls._apply_environment_defaults(config)
        
        # Validate configuration
        config._validate_configuration()
        
        logger.info(f"Validation pipeline configuration loaded for {config.environment.value}")
        return config
    
    @classmethod
    def _detect_environment(cls) -> ValidationEnvironment:
        """Auto-detect the current environment."""
        
        # Check for CI/CD environment
        ci_indicators = [
            "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "TRAVIS", 
            "CIRCLECI", "AZURE_PIPELINES"
        ]
        if any(os.getenv(var) for var in ci_indicators):
            return ValidationEnvironment.CI_CD_PIPELINE
        
        # Check for explicit environment setting
        env_name = os.getenv("PRADEL_JAX_ENVIRONMENT", "").lower()
        if env_name:
            try:
                return ValidationEnvironment(env_name)
            except ValueError:
                logger.warning(f"Unknown environment '{env_name}', using development")
        
        # Default to development
        return ValidationEnvironment.DEVELOPMENT
    
    @classmethod
    def _find_config_file(cls, config_file: Optional[Path]) -> Optional[Path]:
        """Find configuration file in priority order."""
        
        # 1. Explicit argument
        if config_file:
            return config_file
        
        # 2. Environment variable
        env_config = os.getenv("PRADEL_JAX_CONFIG_FILE")
        if env_config:
            return Path(env_config)
        
        # 3. Default locations
        default_locations = [
            Path.cwd() / "validation_config.yaml",
            Path.home() / ".pradel_jax" / "validation_config.yaml",
            Path("/etc/pradel_jax/validation_config.yaml")
        ]
        
        for path in default_locations:
            if path.exists():
                return path
        
        return None
    
    @classmethod
    def _load_from_file(cls, config_path: Path, base_config: "ValidationPipelineConfig") -> "ValidationPipelineConfig":
        """Load configuration from YAML or JSON file."""
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Apply configuration data to base config
            # This is a simplified implementation - in production, you'd want
            # more sophisticated merging logic
            return cls._merge_config_dict(base_config, config_data)
            
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return base_config
    
    @classmethod
    def _load_from_environment(cls, config: "ValidationPipelineConfig") -> "ValidationPipelineConfig":
        """Override configuration with environment variables."""
        
        # Environment mapping
        env_mappings = {
            "PRADEL_JAX_LOG_LEVEL": ("log_level", str),
            "PRADEL_JAX_FAIL_FAST": ("fail_fast", lambda x: x.lower() == "true"),
            "PRADEL_JAX_MAX_PARALLEL_JOBS": ("performance.max_parallel_jobs", int),
            "PRADEL_JAX_ENABLE_QUALITY_GATES": ("enable_quality_gates", lambda x: x.lower() == "true"),
            "PRADEL_JAX_PARAMETER_TOLERANCE": ("validation_criteria.parameter_relative_tolerance_pct", float),
            "PRADEL_JAX_MIN_PASS_RATE": ("validation_criteria.min_pass_rate_for_approval", float),
        }
        
        for env_var, (attr_path, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    cls._set_nested_attr(config, attr_path, converted_value)
                    logger.debug(f"Set {attr_path} = {converted_value} from {env_var}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {value} ({e})")
        
        return config
    
    @classmethod
    def _apply_environment_defaults(cls, config: "ValidationPipelineConfig") -> "ValidationPipelineConfig":
        """Apply environment-specific defaults."""
        
        if config.environment == ValidationEnvironment.CI_CD_PIPELINE:
            # CI/CD optimizations
            config.fail_fast = False  # Don't stop on first failure
            config.enable_progress_bars = False  # No interactive output
            config.reporting.generate_pdf_report = False  # Faster generation
            config.performance.max_parallel_jobs = 2  # Conservative parallelism
            config.log_level = "INFO"
            
        elif config.environment == ValidationEnvironment.PRODUCTION:
            # Production optimizations
            config.enable_development_mode = False
            config.save_intermediate_results = False
            config.enable_profiling = False
            config.reporting.include_raw_data_dumps = False
            
        elif config.environment == ValidationEnvironment.DEVELOPMENT:
            # Development optimizations
            config.enable_development_mode = True
            config.save_intermediate_results = True
            config.log_level = "DEBUG"
            
        return config
    
    @staticmethod
    def _set_nested_attr(obj: Any, attr_path: str, value: Any) -> None:
        """Set nested attribute using dot notation (e.g., 'performance.max_parallel_jobs')."""
        attrs = attr_path.split('.')
        current = obj
        for attr in attrs[:-1]:
            current = getattr(current, attr)
        setattr(current, attrs[-1], value)
    
    @staticmethod
    def _merge_config_dict(config: "ValidationPipelineConfig", config_dict: Dict[str, Any]) -> "ValidationPipelineConfig":
        """Merge configuration dictionary into config object."""
        # Simplified implementation - in production you'd want recursive merging
        # This is a placeholder for the actual implementation
        return config
    
    def _validate_configuration(self) -> None:
        """Validate the complete configuration for consistency and correctness."""
        
        # Validate criteria
        self.validation_criteria.validate_criteria()
        
        # Validate performance settings
        if self.performance.max_parallel_jobs < 1:
            raise ValueError("max_parallel_jobs must be at least 1")
        
        if self.performance.single_model_timeout_seconds < 10:
            raise ValueError("single_model_timeout_seconds must be at least 10")
        
        # Validate reporting settings
        if not any([
            self.reporting.generate_html_report,
            self.reporting.generate_json_summary,
            self.reporting.generate_csv_results
        ]):
            raise ValueError("At least one report format must be enabled")
        
        # Environment-specific validation
        if self.environment == ValidationEnvironment.CI_CD_PIPELINE:
            if self.enable_progress_bars:
                logger.warning("Progress bars not recommended in CI/CD environment")
        
        logger.debug("Configuration validation passed")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        # This would be a full serialization implementation
        return {
            "environment": self.environment.value,
            "pipeline_name": self.pipeline_name,
            "validation_criteria": {
                "parameter_relative_tolerance_pct": self.validation_criteria.parameter_relative_tolerance_pct,
                "max_aic_difference": self.validation_criteria.max_aic_difference,
                "min_pass_rate_for_approval": self.validation_criteria.min_pass_rate_for_approval,
            },
            "performance": {
                "max_parallel_jobs": self.performance.max_parallel_jobs,
                "enable_multiprocessing": self.performance.enable_multiprocessing,
            },
            "reporting": {
                "generate_html_report": self.reporting.generate_html_report,
                "generate_json_summary": self.reporting.generate_json_summary,
            }
        }
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to file."""
        config_dict = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to {config_path}")


def get_validation_pipeline_config(**kwargs) -> ValidationPipelineConfig:
    """
    Convenience function to get validation pipeline configuration.
    
    Args:
        **kwargs: Override specific configuration values
    
    Returns:
        ValidationPipelineConfig: Configured validation pipeline
    """
    config = ValidationPipelineConfig.from_environment()
    
    # Apply any keyword overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    return config