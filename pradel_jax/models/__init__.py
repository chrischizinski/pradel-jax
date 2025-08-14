"""
Model implementations for pradel-jax.

Provides extensible framework for capture-recapture models.
"""

from .base import CaptureRecaptureModel, ModelResult, ModelRegistry
from .pradel import PradelModel

__all__ = [
    "CaptureRecaptureModel",
    "ModelResult", 
    "ModelRegistry",
    "PradelModel",
]