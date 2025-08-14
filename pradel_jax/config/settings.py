"""
Configuration management system for pradel-jax.

Provides a flexible, hierarchical configuration system with support for
file-based configuration, environment variables, and runtime updates.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OptimizationStrategy(str, Enum):
    """Available optimization strategies."""
    AUTO = "auto"
    SCIPY_LBFGS = "scipy_lbfgs"
    SCIPY_SLSQP = "scipy_slsqp"
    JAX_ADAM = "jax_adam"
    JAXOPT_LBFGS = "jaxopt_lbfgs"
    MULTI_START = "multi_start"


class DataFormat(str, Enum):
    """Supported data formats."""
    AUTO = "auto"
    RMARK = "rmark"
    MARK = "mark"
    POPAN = "popan"
    GENERIC = "generic"


class ValidationLevel(str, Enum):
    """Data validation levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    MINIMAL = "minimal"
    NONE = "none"


class DataConfig(BaseModel):
    """Data processing configuration."""
    default_format: DataFormat = DataFormat.AUTO
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    cache_enabled: bool = True
    cache_directory: Optional[Path] = None
    auto_preprocessing: bool = True
    missing_data_strategy: str = "listwise_deletion"
    outlier_detection: bool = True
    standardize_covariates: bool = True
    
    @validator('cache_directory', pre=True)
    def validate_cache_directory(cls, v):
        if v is None:
            return Path.home() / ".pradel_jax" / "cache"
        return Path(v)


class OptimizationConfig(BaseModel):
    """Optimization configuration."""
    default_strategy: OptimizationStrategy = OptimizationStrategy.AUTO
    max_iterations: int = 1000
    convergence_tolerance: float = 1e-6
    gradient_tolerance: float = 1e-8
    enable_gpu: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    multi_start_attempts: int = 3
    random_seed: Optional[int] = None
    
    @validator('max_workers', pre=True)
    def validate_max_workers(cls, v):
        if v is None:
            return min(8, os.cpu_count() or 1)
        return max(1, v)


class ModelConfig(BaseModel):
    """Model specification configuration."""
    default_link_functions: Dict[str, str] = Field(default_factory=lambda: {
        "phi": "logit",
        "p": "logit", 
        "f": "log"
    })
    enable_model_selection: bool = True
    information_criterion: str = "AICc"
    model_averaging: bool = False
    confidence_level: float = 0.95
    profile_ci: bool = False


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    file_logging: bool = False
    log_file: Optional[Path] = None
    console_logging: bool = True
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @validator('log_file', pre=True)
    def validate_log_file(cls, v):
        if v is None and cls.file_logging:
            return Path.home() / ".pradel_jax" / "logs" / "pradel_jax.log"
        return Path(v) if v else None


class PerformanceConfig(BaseModel):
    """Performance and resource configuration."""
    memory_limit_gb: Optional[float] = None
    enable_jit_compilation: bool = True
    chunk_size: int = 10000
    batch_size: int = 32
    enable_progress_bars: bool = True
    profiling_enabled: bool = False


class PradelJaxConfig(BaseModel):
    """Main configuration class for pradel-jax."""
    
    data: DataConfig = Field(default_factory=DataConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    
    # Plugin and extension system
    enabled_plugins: List[str] = Field(default_factory=list)
    plugin_directories: List[Path] = Field(default_factory=list)
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        use_enum_values = True
        
    def __init__(self, config_file: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
            **kwargs: Override specific configuration values
        """
        # Load from file if provided
        config_data = {}
        if config_file:
            config_data = self._load_config_file(config_file)
        
        # Override with environment variables
        config_data.update(self._load_environment_variables())
        
        # Override with explicit kwargs
        config_data.update(kwargs)
        
        super().__init__(**config_data)
        
        # Ensure directories exist
        self._create_directories()
    
    def _load_config_file(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Define environment variable mappings
        env_mappings = {
            'PRADEL_JAX_LOG_LEVEL': ('logging', 'level'),
            'PRADEL_JAX_CACHE_DIR': ('data', 'cache_directory'),
            'PRADEL_JAX_MAX_WORKERS': ('optimization', 'max_workers'),
            'PRADEL_JAX_ENABLE_GPU': ('optimization', 'enable_gpu'),
            'PRADEL_JAX_VALIDATION_LEVEL': ('data', 'validation_level'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in config:
                    config[section] = {}
                
                # Type conversion based on key
                if key in ['max_workers']:
                    value = int(value)
                elif key in ['enable_gpu']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                config[section][key] = value
        
        return config
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data.cache_directory,
        ]
        
        if self.logging.log_file:
            directories.append(self.logging.log_file.parent)
        
        for directory in directories:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config_file: Union[str, Path]) -> None:
        """Save current configuration to YAML file."""
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    def update(self, **kwargs) -> None:
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Handle nested updates
                if '.' in key:
                    section, subkey = key.split('.', 1)
                    if hasattr(self, section):
                        section_obj = getattr(self, section)
                        if hasattr(section_obj, subkey):
                            setattr(section_obj, subkey, value)
    
    def get_user_config_path(self) -> Path:
        """Get the user's configuration file path."""
        return Path.home() / ".pradel_jax" / "config.yaml"
    
    def load_user_config(self) -> None:
        """Load user's configuration file if it exists."""
        user_config = self.get_user_config_path()
        if user_config.exists():
            config_data = self._load_config_file(user_config)
            for key, value in config_data.items():
                if hasattr(self, key):
                    current_value = getattr(self, key)
                    if isinstance(current_value, BaseModel):
                        current_value.__dict__.update(value)
                    else:
                        setattr(self, key, value)


# Default configuration instance
_default_config: Optional[PradelJaxConfig] = None

def get_default_config() -> PradelJaxConfig:
    """Get the default configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = PradelJaxConfig()
        _default_config.load_user_config()
    return _default_config