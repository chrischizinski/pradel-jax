"""
Pradel-JAX: Flexible and robust capture-recapture analysis using JAX

A modern, extensible framework for capture-recapture analysis with support for
multiple data formats, flexible model specification, and robust optimization.
"""

__version__ = "2.0.0-alpha"
__author__ = "Ava Britton, Christopher Chizinski"

# Core data loading
from .data.adapters import load_data, DataContext, RMarkFormatAdapter, GenericFormatAdapter

# Formula system  
from .formulas import FormulaSpec, ParameterFormula, create_simple_spec

# Models
from .models import CaptureRecaptureModel, ModelResult, PradelModel
from .models.base import ModelType, register_model, get_model, list_available_models

# Configuration 
from .config.settings import PradelJaxConfig

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