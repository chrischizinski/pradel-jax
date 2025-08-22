"""
Pradel-JAX: Flexible and robust capture-recapture analysis using JAX

A modern, extensible framework for capture-recapture analysis with support for
multiple data formats, flexible model specification, and robust optimization.
"""

__version__ = "2.0.0-alpha"
__author__ = "Ava Britton, Christopher Chizinski"

# Core data loading
from .data.adapters import load_data, DataContext, RMarkFormatAdapter, GenericFormatAdapter
from .data.sampling import (
    load_data_with_sampling, 
    stratified_sample, 
    train_validation_split,
    determine_tier_status,
    get_sampling_summary
)

# Formula system  
from .formulas import FormulaSpec, ParameterFormula, create_simple_spec

# Models
from .models import CaptureRecaptureModel, ModelResult, PradelModel
from .models.base import ModelType, register_model, get_model, list_available_models

# Configuration 
from .config.settings import PradelJaxConfig

# Export functionality
from .core.export import (
    ResultsExporter,
    export_model_results,
    create_timestamped_export
)

# Import key exception classes
from .core.exceptions import (
    PradelJaxError,
    DataFormatError,
    ModelSpecificationError,
    OptimizationError,
)

# Register built-in models
register_model(ModelType.PRADEL, PradelModel)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core data loading
    "load_data",
    "load_data_with_sampling",
    "stratified_sample", 
    "train_validation_split",
    "determine_tier_status",
    "get_sampling_summary",
    "DataContext", 
    
    # Formula system
    "FormulaSpec",
    "ParameterFormula", 
    "create_simple_spec",
    
    # Models
    "CaptureRecaptureModel",
    "ModelResult",
    "PradelModel",
    "ModelType",
    "register_model",
    "get_model",
    "list_available_models",
    
    # Configuration
    "PradelJaxConfig",
    
    # Exceptions
    "PradelJaxError",
    "DataFormatError", 
    "ModelSpecificationError",
    "OptimizationError",
    
    # Data adapters
    "RMarkFormatAdapter",
    "GenericFormatAdapter",
    
    # Export functionality
    "ResultsExporter",
    "export_model_results", 
    "create_timestamped_export",
]

# Set up default configuration
_default_config = None

def get_config() -> PradelJaxConfig:
    """Get the global configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = PradelJaxConfig()
    return _default_config

def configure(**kwargs) -> None:
    """Update global configuration."""
    config = get_config()
    for key, value in kwargs.items():
        setattr(config, key, value)