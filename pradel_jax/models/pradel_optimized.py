"""
Optimized Pradel model implementation with fully vectorized likelihood computation.

Key optimizations:
- Vectorized likelihood calculation (no Python loops)
- Efficient JAX operations for capture-recapture logic
- Memory-efficient design matrix operations
- Proper JAX compilation for maximum performance
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .pradel import PradelModel, inv_logit, exp_link
from ..formulas.design_matrix import DesignMatrixInfo
from ..data.adapters import DataContext
from ..utils.logging import get_logger

logger = get_logger(__name__)


class OptimizedPradelModel(PradelModel):
    """
    Backwards-compatible alias for :class:`PradelModel`.

    Historically this class carried a separate "optimized" likelihood, but that
    implementation was (a) numerically incorrect — it dropped the recruitment
    parameter f and the seniority/χ terms, reducing to a mis-specified CJS — and
    (b) broken at runtime, because it decorated an instance method with
    ``@jax.jit`` (passing ``self`` as a traced argument raised a TypeError so it
    never actually ran).

    :class:`PradelModel` is now itself fully vectorized via ``jax.vmap`` over a
    JIT-compiled per-individual likelihood, so no separate implementation is
    needed. This subclass is retained only so existing imports keep working; it
    inherits the correct, tested likelihood unchanged.
    """


# Create optimized convenience functions
def create_optimized_pradel_model() -> OptimizedPradelModel:
    """Create an optimized Pradel model instance."""
    return OptimizedPradelModel()
