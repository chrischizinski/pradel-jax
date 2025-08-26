"""
Hessian computation utilities for statistical inference.

Provides fallback methods for computing Hessian inverse and standard errors
when optimizers don't provide adequate Hessian information.
"""

import numpy as np
from typing import Callable, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


def compute_finite_difference_hessian_diagonal(
    objective_func: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    Compute diagonal elements of Hessian using finite differences.

    Uses central difference approximation:
    H_ii ≈ (f(x + eps*e_i) - 2*f(x) + f(x - eps*e_i)) / eps²

    Args:
        objective_func: Function to compute Hessian for
        x: Point at which to compute Hessian
        eps: Step size for finite differences

    Returns:
        Array of diagonal Hessian elements
    """
    n = len(x)
    hessian_diag = np.zeros(n)

    # Central value
    f_center = objective_func(x)

    for i in range(n):
        # Forward step
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = objective_func(x_plus)

        # Backward step
        x_minus = x.copy()
        x_minus[i] -= eps
        f_minus = objective_func(x_minus)

        # Central difference second derivative
        hessian_diag[i] = (f_plus - 2 * f_center + f_minus) / (eps**2)

    return hessian_diag


def compute_finite_difference_hessian_full(
    objective_func: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    Compute full Hessian matrix using finite differences.

    More expensive but gives full covariance information.
    Uses central difference for diagonal and mixed partials for off-diagonal.

    Args:
        objective_func: Function to compute Hessian for
        x: Point at which to compute Hessian
        eps: Step size for finite differences

    Returns:
        Full Hessian matrix
    """
    n = len(x)
    hessian = np.zeros((n, n))

    # Compute diagonal elements (pure second derivatives)
    hessian_diag = compute_finite_difference_hessian_diagonal(objective_func, x, eps)
    np.fill_diagonal(hessian, hessian_diag)

    # Compute off-diagonal elements (mixed partials)
    for i in range(n):
        for j in range(i + 1, n):
            # Mixed partial: ∂²f/∂x_i∂x_j
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps

            x_pm = x.copy()
            x_pm[i] += eps
            x_pm[j] -= eps

            x_mp = x.copy()
            x_mp[i] -= eps
            x_mp[j] += eps

            x_mm = x.copy()
            x_mm[i] -= eps
            x_mm[j] -= eps

            # Central difference for mixed partial
            mixed_partial = (
                objective_func(x_pp)
                - objective_func(x_pm)
                - objective_func(x_mp)
                + objective_func(x_mm)
            ) / (4 * eps**2)

            hessian[i, j] = mixed_partial
            hessian[j, i] = mixed_partial  # Symmetry

    return hessian


def compute_standard_errors_from_hessian(
    hessian: np.ndarray, is_inverse: bool = False
) -> Optional[np.ndarray]:
    """
    Compute standard errors from Hessian matrix.

    Args:
        hessian: Hessian matrix (or its inverse)
        is_inverse: Whether the input is already the inverse

    Returns:
        Standard errors or None if computation fails
    """
    try:
        if is_inverse:
            hess_inv = hessian
        else:
            # Invert the Hessian (with regularization for numerical stability)
            regularization = 1e-12 * np.eye(hessian.shape[0])
            hess_inv = np.linalg.inv(hessian + regularization)

        # Standard errors are sqrt of diagonal of covariance matrix (inverse Hessian)
        diagonal = np.diag(hess_inv)

        # Ensure positive diagonal elements
        diagonal = np.maximum(diagonal, 1e-12)

        return np.sqrt(diagonal)

    except (np.linalg.LinAlgError, ValueError) as e:
        logger.warning(f"Failed to compute standard errors from Hessian: {e}")
        return None


def compute_fallback_standard_errors(
    objective_func: Callable[[np.ndarray], float],
    x: np.ndarray,
    method: str = "diagonal",
    eps: float = 1e-6,
) -> Optional[np.ndarray]:
    """
    Compute standard errors using finite difference Hessian as fallback.

    Args:
        objective_func: Objective function (should be negative log-likelihood)
        x: Parameters at optimum
        method: "diagonal" for diagonal-only, "full" for full Hessian
        eps: Step size for finite differences

    Returns:
        Standard errors or None if computation fails
    """
    try:
        if method == "diagonal":
            # Faster: only compute diagonal elements
            hessian_diag = compute_finite_difference_hessian_diagonal(
                objective_func, x, eps
            )

            # Check for positive definite condition (diagonal elements should be positive)
            if np.any(hessian_diag <= 0):
                logger.warning(
                    "Hessian diagonal has non-positive elements - results may be unreliable"
                )
                hessian_diag = np.maximum(hessian_diag, 1e-6)

            # Standard errors for diagonal approximation
            return np.sqrt(1.0 / hessian_diag)

        elif method == "full":
            # More expensive but more accurate
            hessian = compute_finite_difference_hessian_full(objective_func, x, eps)
            return compute_standard_errors_from_hessian(hessian, is_inverse=False)

        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as e:
        logger.error(f"Fallback standard error computation failed: {e}")
        return None


def validate_hessian_quality(hessian_inv: Union[np.ndarray, object]) -> dict:
    """
    Assess quality of Hessian inverse for statistical inference.

    Args:
        hessian_inv: Hessian inverse matrix or approximation object

    Returns:
        Dictionary with quality assessment
    """
    quality_info = {
        "type": str(type(hessian_inv)),
        "available": hessian_inv is not None,
        "is_array": isinstance(hessian_inv, np.ndarray),
        "meaningful": False,
        "issues": [],
    }

    if hessian_inv is None:
        quality_info["issues"].append("No Hessian inverse provided")
        return quality_info

    try:
        # Try to extract diagonal
        if isinstance(hessian_inv, np.ndarray):
            diagonal = np.diag(hessian_inv)
        else:
            # Try unit vector approach for approximation objects
            n = getattr(hessian_inv, "shape", [0])[0]
            if n > 0:
                diagonal = np.array(
                    [hessian_inv @ np.eye(n)[i] @ np.eye(n)[i] for i in range(n)]
                )
            else:
                quality_info["issues"].append("Cannot determine size")
                return quality_info

        # Check for meaningful values (not all 1.0 or 0.0)
        if np.allclose(diagonal, 1.0):
            quality_info["issues"].append(
                "All diagonal elements are 1.0 (unit approximation)"
            )
        elif np.allclose(diagonal, 0.0):
            quality_info["issues"].append("All diagonal elements are 0.0")
        elif np.any(diagonal <= 0):
            quality_info["issues"].append("Some diagonal elements are negative or zero")
        else:
            quality_info["meaningful"] = True

        quality_info["diagonal_range"] = [
            float(np.min(diagonal)),
            float(np.max(diagonal)),
        ]

    except Exception as e:
        quality_info["issues"].append(f"Error extracting diagonal: {str(e)}")

    return quality_info
