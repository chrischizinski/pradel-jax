"""
Main API functions for pradel-jax.

Placeholder implementation - will be expanded in later phases.
"""

from ..data.adapters import load_data

def fit_models(*args, **kwargs):
    """Placeholder function for fitting models."""
    raise NotImplementedError("Model fitting not yet implemented in redesign")

def select_best_model(*args, **kwargs):
    """Placeholder function for model selection."""
    raise NotImplementedError("Model selection not yet implemented in redesign")

def validate_against_rmark(*args, **kwargs):
    """Placeholder function for RMark validation."""
    raise NotImplementedError("RMark validation not yet implemented in redesign")

# Re-export load_data from data.adapters
__all__ = ["fit_models", "select_best_model", "validate_against_rmark", "load_data"]