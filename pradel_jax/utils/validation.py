"""
Validation utilities for pradel-jax.

Provides common validation functions for arrays, parameters, and data structures.
"""

import numpy as np
import jax.numpy as jnp
from typing import Union, Tuple, Optional, Any
from ..core.exceptions import ValidationError


def validate_array_dimensions(
    array: Union[np.ndarray, jnp.ndarray], 
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    name: str = "array"
) -> None:
    """
    Validate array dimensions.
    
    Args:
        array: Array to validate
        expected_shape: Expected exact shape (None values are ignored)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions  
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not hasattr(array, 'shape'):
        raise ValidationError(
            f"{name} must be an array-like object with shape attribute",
            suggestions=[
                "Ensure input is numpy or JAX array",
                "Convert lists to arrays using np.array()",
            ]
        )
    
    shape = array.shape
    ndims = len(shape)
    
    # Check dimension count
    if min_dims is not None and ndims < min_dims:
        raise ValidationError(
            f"{name} has {ndims} dimensions, expected at least {min_dims}",
            suggestions=[
                f"Add dimensions using reshape or expand_dims",
                f"Check that {name} has correct structure",
            ]
        )
    
    if max_dims is not None and ndims > max_dims:
        raise ValidationError(
            f"{name} has {ndims} dimensions, expected at most {max_dims}",
            suggestions=[
                f"Reduce dimensions using squeeze or indexing",
                f"Check that {name} has correct structure",
            ]
        )
    
    # Check exact shape if provided
    if expected_shape is not None:
        if len(expected_shape) != ndims:
            raise ValidationError(
                f"{name} has {ndims} dimensions, expected {len(expected_shape)}",
                suggestions=[
                    f"Expected shape: {expected_shape}",
                    f"Actual shape: {shape}",
                    "Check array construction and reshaping",
                ]
            )
        
        for i, (actual, expected) in enumerate(zip(shape, expected_shape)):
            if expected is not None and actual != expected:
                raise ValidationError(
                    f"{name} dimension {i} has size {actual}, expected {expected}",
                    suggestions=[
                        f"Expected shape: {expected_shape}",
                        f"Actual shape: {shape}",
                        f"Check dimension {i} construction",
                    ]
                )


def validate_positive(
    value: Union[float, int, np.ndarray, jnp.ndarray],
    name: str = "value",
    strict: bool = False
) -> None:
    """
    Validate that value(s) are positive.
    
    Args:
        value: Value or array to validate
        name: Name for error messages
        strict: If True, require strictly positive (> 0), else non-negative (>= 0)
        
    Raises:
        ValidationError: If validation fails
    """
    if hasattr(value, 'shape'):  # Array-like
        if strict:
            condition = jnp.all(value > 0) if hasattr(value, 'device') else np.all(value > 0)
            operator = ">"
        else:
            condition = jnp.all(value >= 0) if hasattr(value, 'device') else np.all(value >= 0)
            operator = ">="
        
        if not condition:
            min_val = jnp.min(value) if hasattr(value, 'device') else np.min(value)
            raise ValidationError(
                f"{name} contains negative values (min: {min_val})",
                suggestions=[
                    f"All values must be {operator} 0",
                    "Check data preprocessing steps",
                    "Consider data transformation if needed",
                ]
            )
    else:  # Scalar
        if strict and value <= 0:
            raise ValidationError(
                f"{name} must be positive, got {value}",
                suggestions=[
                    "Provide a positive value",
                    "Check parameter constraints",
                ]
            )
        elif not strict and value < 0:
            raise ValidationError(
                f"{name} must be non-negative, got {value}",
                suggestions=[
                    "Provide a non-negative value", 
                    "Check parameter constraints",
                ]
            )


def validate_probability(
    value: Union[float, np.ndarray, jnp.ndarray],
    name: str = "probability"
) -> None:
    """
    Validate that value(s) are valid probabilities (0 <= p <= 1).
    
    Args:
        value: Value or array to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    validate_positive(value, name, strict=False)
    
    if hasattr(value, 'shape'):  # Array-like
        condition = jnp.all(value <= 1) if hasattr(value, 'device') else np.all(value <= 1)
        if not condition:
            max_val = jnp.max(value) if hasattr(value, 'device') else np.max(value)
            raise ValidationError(
                f"{name} contains values > 1 (max: {max_val})",
                suggestions=[
                    "Probabilities must be between 0 and 1",
                    "Check if values need to be transformed",
                    "Verify probability calculation logic",
                ]
            )
    else:  # Scalar
        if value > 1:
            raise ValidationError(
                f"{name} must be <= 1, got {value}",
                suggestions=[
                    "Probabilities must be between 0 and 1",
                    "Check if value needs to be transformed",
                ]
            )


def validate_capture_matrix(
    capture_matrix: Union[np.ndarray, jnp.ndarray],
    min_individuals: int = 1,
    min_occasions: int = 3
) -> None:
    """
    Validate capture history matrix.
    
    Args:
        capture_matrix: Matrix of capture histories (individuals x occasions)
        min_individuals: Minimum number of individuals
        min_occasions: Minimum number of occasions
        
    Raises:
        ValidationError: If validation fails
    """
    validate_array_dimensions(
        capture_matrix, 
        min_dims=2, 
        max_dims=2,
        name="capture_matrix"
    )
    
    n_individuals, n_occasions = capture_matrix.shape
    
    if n_individuals < min_individuals:
        raise ValidationError(
            f"Capture matrix has {n_individuals} individuals, need at least {min_individuals}",
            suggestions=[
                "Provide more individual capture histories",
                "Check data loading and filtering",
            ]
        )
    
    if n_occasions < min_occasions:
        raise ValidationError(
            f"Capture matrix has {n_occasions} occasions, need at least {min_occasions}",
            suggestions=[
                "Pradel models require at least 3 capture occasions",
                "Check capture history length",
                "Verify data preprocessing",
            ]
        )
    
    # Check that values are 0 or 1
    unique_values = jnp.unique(capture_matrix) if hasattr(capture_matrix, 'device') else np.unique(capture_matrix)
    valid_values = set([0, 1])
    actual_values = set(unique_values.tolist())
    
    if not actual_values.issubset(valid_values):
        invalid_values = actual_values - valid_values
        raise ValidationError(
            f"Capture matrix contains invalid values: {invalid_values}",
            suggestions=[
                "Capture histories must contain only 0s and 1s",
                "Check data preprocessing and conversion",
                "Verify capture history format",
            ]
        )
    
    # Check that at least some individuals are captured
    total_captures = jnp.sum(capture_matrix) if hasattr(capture_matrix, 'device') else np.sum(capture_matrix)
    if total_captures == 0:
        raise ValidationError(
            "Capture matrix contains no captures (all zeros)",
            suggestions=[
                "Check data loading and filtering",
                "Verify capture history format",
                "Ensure data contains actual captures",
            ]
        )


def validate_design_matrix_compatibility(
    X: Union[np.ndarray, jnp.ndarray],
    capture_matrix: Union[np.ndarray, jnp.ndarray],
    parameter_name: str
) -> None:
    """
    Validate that design matrix is compatible with capture matrix.
    
    Args:
        X: Design matrix
        capture_matrix: Capture history matrix
        parameter_name: Name of parameter for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    validate_array_dimensions(X, min_dims=2, max_dims=2, name=f"X_{parameter_name}")
    validate_capture_matrix(capture_matrix)
    
    n_individuals, n_occasions = capture_matrix.shape
    n_rows, n_cols = X.shape
    
    # For Pradel models, design matrices should have dimensions based on intervals
    expected_rows = n_occasions - 1  # Intervals between occasions
    
    if n_rows != expected_rows:
        raise ValidationError(
            f"Design matrix X_{parameter_name} has {n_rows} rows, expected {expected_rows}",
            suggestions=[
                f"Pradel models use {n_occasions-1} intervals for {n_occasions} occasions",
                "Check design matrix construction",
                "Verify model specification",
            ],
            context={
                "parameter": parameter_name,
                "matrix_shape": (n_rows, n_cols),
                "expected_rows": expected_rows,
                "n_occasions": n_occasions,
            }
        )