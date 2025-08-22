"""
Design matrix construction for pradel-jax.

Converts formula terms into design matrices for statistical modeling.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

from .terms import Term, InterceptTerm, VariableTerm, InteractionTerm, FunctionTerm, PolynomialTerm
from .spec import ParameterFormula
from .time_varying import TimeVaryingDesignMatrixBuilder
from ..core.exceptions import ModelSpecificationError, DataFormatError
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class DesignMatrixInfo:
    """Information about a constructed design matrix."""
    matrix: jnp.ndarray
    column_names: List[str]
    parameter_count: int
    has_intercept: bool
    formula_string: str


class DesignMatrixBuilder:
    """
    Builds design matrices from formula terms and data.
    
    Handles various term types and creates appropriate design matrix columns.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.time_varying_builder = TimeVaryingDesignMatrixBuilder()
    
    def build_matrix(
        self,
        formula: ParameterFormula,
        data_context: Any,  # DataContext from data.adapters
        n_occasions: Optional[int] = None
    ) -> DesignMatrixInfo:
        """
        Build design matrix for a parameter formula.
        
        Args:
            formula: ParameterFormula object
            data_context: DataContext with covariates
            n_occasions: Number of time occasions (for time-varying parameters)
            
        Returns:
            DesignMatrixInfo with constructed matrix and metadata
        """
        self.logger.debug(f"Building design matrix for {formula.parameter.value}: {formula.formula_string}")
        
        # Validate covariates
        available_covariates = list(data_context.covariates.keys())
        formula.validate_covariates(available_covariates)
        
        # Get number of individuals
        n_individuals = data_context.n_individuals
        n_occasions = n_occasions or data_context.n_occasions
        
        # Build matrix columns
        matrix_columns = []
        column_names = []
        
        for term in formula.terms:
            columns, names = self._build_term_columns(
                term, data_context, n_individuals, n_occasions
            )
            matrix_columns.extend(columns)
            column_names.extend(names)
        
        if not matrix_columns:
            raise ModelSpecificationError(
                formula=formula.formula_string,
                parameter=formula.parameter.value,
                suggestions=[
                    "Formula produced no design matrix columns",
                    "Check formula syntax and available covariates",
                    "Use '1' for intercept-only models",
                ]
            )
        
        # Combine columns into matrix
        design_matrix = np.column_stack(matrix_columns)
        
        # Convert to JAX array
        design_matrix_jax = jnp.array(design_matrix, dtype=jnp.float32)
        
        self.logger.debug(
            f"Built design matrix: {design_matrix_jax.shape} "
            f"({len(column_names)} columns: {column_names})"
        )
        
        return DesignMatrixInfo(
            matrix=design_matrix_jax,
            column_names=column_names,
            parameter_count=len(column_names),
            has_intercept=formula.has_intercept,
            formula_string=formula.formula_string
        )
    
    def _build_term_columns(
        self,
        term: Term,
        data_context: Any,
        n_individuals: int,
        n_occasions: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Build design matrix columns for a single term.
        
        Args:
            term: Term object
            data_context: DataContext with covariates
            n_individuals: Number of individuals
            n_occasions: Number of occasions
            
        Returns:
            Tuple of (column arrays, column names)
        """
        if isinstance(term, InterceptTerm):
            return self._build_intercept_columns(n_individuals, n_occasions)
        
        elif isinstance(term, VariableTerm):
            return self._build_variable_columns(
                term, data_context, n_individuals, n_occasions
            )
        
        elif isinstance(term, InteractionTerm):
            return self._build_interaction_columns(
                term, data_context, n_individuals, n_occasions
            )
        
        elif isinstance(term, FunctionTerm):
            return self._build_function_columns(
                term, data_context, n_individuals, n_occasions
            )
        
        elif isinstance(term, PolynomialTerm):
            return self._build_polynomial_columns(
                term, data_context, n_individuals, n_occasions
            )
        
        else:
            raise ModelSpecificationError(
                formula=f"Unknown term type: {type(term)}",
                suggestions=[
                    "Supported terms: intercept, variable, interaction, function, polynomial",
                    "Check formula parsing logic",
                ]
            )
    
    def _build_intercept_columns(
        self, n_individuals: int, n_occasions: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build intercept column (all ones)."""
        intercept_col = np.ones(n_individuals, dtype=np.float32)
        return [intercept_col], ["(Intercept)"]
    
    def _build_variable_columns(
        self,
        term: VariableTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for a simple variable term."""
        var_name = term.variable_name
        
        if var_name not in data_context.covariates:
            raise DataFormatError(
                specific_issue=f"Covariate '{var_name}' not found in data",
                missing_covariates=[var_name],
                suggestions=[
                    f"Available covariates: {list(data_context.covariates.keys())}",
                    "Check variable name spelling",
                    "Ensure covariate exists in data",
                ]
            )
        
        # Check if this is a categorical variable
        is_categorical = data_context.covariates.get(f'{var_name}_is_categorical', False)
        
        if is_categorical:
            # Handle categorical variable with dummy coding
            categories = data_context.covariates.get(f'{var_name}_categories', [])
            categorical_codes = np.array(data_context.covariates[var_name])
            
            # Create dummy variables (drop first category for identifiability)
            if len(categories) <= 1:
                # Only one category - create intercept-like column
                column = np.ones(n_individuals, dtype=np.float32)
                return [column], [var_name]
            else:
                # Multiple categories - create dummy variables (drop first)
                columns = []
                names = []
                
                for i, category in enumerate(categories[1:], 1):  # Skip first category
                    dummy_col = (categorical_codes == i).astype(np.float32)
                    columns.append(dummy_col)
                    names.append(f'{var_name}_{category}')
                
                return columns, names
        else:
            # Handle numeric variable
            covariate_data = np.array(data_context.covariates[var_name])
            
            # Handle different data shapes
            if covariate_data.ndim == 1 and len(covariate_data) == n_individuals:
                # Individual-level covariate
                column = covariate_data.astype(np.float32)
            elif covariate_data.ndim == 2:
                # Time-varying covariate - handle properly
                self.logger.info(f"Processing time-varying covariate: {var_name}")
                
                # Check if we have time-varying information in data context
                if f"{var_name}_is_time_varying" in data_context.covariates:
                    # Use the time-varying matrix directly
                    # For Pradel models, parameters typically apply to intervals
                    # Use appropriate time indexing based on parameter type
                    
                    # For simplicity, use the first available time point for now
                    # TODO: Implement proper time-occasion mapping based on parameter type
                    column = covariate_data[:, 0].astype(np.float32)
                    self.logger.info(f"Using first time point for {var_name} (shape: {covariate_data.shape})")
                else:
                    # Legacy handling - use first time point with warning
                    self.logger.warning(f"Time-varying covariate {var_name} - using first time point (consider using time-varying framework)")
                    column = covariate_data[:, 0].astype(np.float32)
            else:
                raise DataFormatError(
                    specific_issue=f"Covariate '{var_name}' has unexpected shape: {covariate_data.shape}",
                    suggestions=[
                        f"Expected shape: ({n_individuals},) or ({n_individuals}, {n_occasions})",
                        "Check covariate data structure",
                        "Use time-varying covariate framework for multi-dimensional data",
                    ]
                )
            
            return [column], [var_name]
    
    def _build_interaction_columns(
        self,
        term: InteractionTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for interaction terms."""
        # Get all variable columns
        var_columns = []
        var_names = []
        
        for var_name in term.variables:
            # Create variable term and get its columns
            var_term = VariableTerm(var_name)
            columns, names = self._build_variable_columns(
                var_term, data_context, n_individuals, n_occasions
            )
            var_columns.extend(columns)
            var_names.extend(names)
        
        # Create interaction by multiplying variables
        if len(var_columns) != len(term.variables):
            raise ModelSpecificationError(
                formula=f"Interaction variable count mismatch: {term.variables}",
                suggestions=[
                    "Check for categorical variables in interactions",
                    "Categorical variables may create multiple columns",
                ]
            )
        
        # Element-wise multiplication for interaction
        interaction_col = var_columns[0].copy()
        for col in var_columns[1:]:
            interaction_col = interaction_col * col
        
        interaction_name = ":".join(term.variables)
        
        return [interaction_col], [interaction_name]
    
    def _build_function_columns(
        self,
        term: FunctionTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for function terms."""
        func_name = term.function_name
        
        if func_name == "I":
            # Identity function - evaluate expression
            return self._build_identity_function(term, data_context, n_individuals, n_occasions)
        
        elif func_name in ["log", "exp", "sqrt", "sin", "cos", "tan"]:
            # Standard mathematical functions
            return self._build_math_function(term, data_context, n_individuals, n_occasions)
        
        else:
            # Unknown function - treat as identity
            self.logger.warning(f"Unknown function '{func_name}' - treating as identity")
            return self._build_identity_function(term, data_context, n_individuals, n_occasions)
    
    def _build_identity_function(
        self,
        term: FunctionTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for I() function (identity/expression evaluation)."""
        if len(term.arguments) != 1:
            raise ModelSpecificationError(
                formula=f"I() function requires exactly one argument: {term.expression}",
                suggestions=[
                    "Use I(expression) for mathematical expressions",
                    "Examples: I(age^2), I(age*2), I(log(weight))",
                ]
            )
        
        expr = term.arguments[0]
        
        # Simple expression evaluation for common cases
        if "^2" in expr:
            # Quadratic term: var^2
            var_name = expr.replace("^2", "").strip()
            if var_name in data_context.covariates:
                var_data = np.array(data_context.covariates[var_name])
                if var_data.ndim == 1:
                    squared_col = (var_data ** 2).astype(np.float32)
                    return [squared_col], [f"I({expr})"]
        
        elif "^3" in expr:
            # Cubic term: var^3
            var_name = expr.replace("^3", "").strip()
            if var_name in data_context.covariates:
                var_data = np.array(data_context.covariates[var_name])
                if var_data.ndim == 1:
                    cubed_col = (var_data ** 3).astype(np.float32)
                    return [cubed_col], [f"I({expr})"]
        
        elif "*" in expr:
            # Multiplication: var1*var2 or var*constant
            parts = expr.split("*")
            if len(parts) == 2:
                left, right = parts[0].strip(), parts[1].strip()
                
                # Check if one is a number
                try:
                    const = float(right)
                    if left in data_context.covariates:
                        var_data = np.array(data_context.covariates[left])
                        if var_data.ndim == 1:
                            scaled_col = (var_data * const).astype(np.float32)
                            return [scaled_col], [f"I({expr})"]
                except ValueError:
                    # Both are variables - create interaction
                    if left in data_context.covariates and right in data_context.covariates:
                        left_data = np.array(data_context.covariates[left])
                        right_data = np.array(data_context.covariates[right])
                        if left_data.ndim == 1 and right_data.ndim == 1:
                            product_col = (left_data * right_data).astype(np.float32)
                            return [product_col], [f"I({expr})"]
        
        # Fallback: treat as simple variable if it exists
        if expr in data_context.covariates:
            var_term = VariableTerm(expr)
            return self._build_variable_columns(var_term, data_context, n_individuals, n_occasions)
        
        raise ModelSpecificationError(
            formula=f"Cannot evaluate expression: I({expr})",
            suggestions=[
                "Supported expressions: var^2, var^3, var*constant, var1*var2",
                "Ensure all variables exist in data",
                "Use simple variable names in expressions",
            ]
        )
    
    def _build_math_function(
        self,
        term: FunctionTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for mathematical functions."""
        if len(term.arguments) != 1:
            raise ModelSpecificationError(
                formula=f"{term.function_name}() requires exactly one argument",
                suggestions=[
                    f"Use {term.function_name}(variable_name)",
                    "Ensure variable exists in data",
                ]
            )
        
        var_name = term.arguments[0].strip()
        if var_name not in data_context.covariates:
            raise DataFormatError(
                specific_issue=f"Variable '{var_name}' not found for {term.function_name}() function",
                missing_covariates=[var_name],
                suggestions=[
                    f"Available variables: {list(data_context.covariates.keys())}",
                    "Check variable name in function",
                ]
            )
        
        var_data = np.array(data_context.covariates[var_name])
        if var_data.ndim != 1:
            raise ModelSpecificationError(
                formula=f"Function {term.function_name}() requires 1D variable data",
                suggestions=[
                    "Mathematical functions work on individual-level covariates",
                    "Check variable data structure",
                ]
            )
        
        # Apply mathematical function
        func_name = term.function_name
        try:
            if func_name == "log":
                # Check for non-positive values
                if np.any(var_data <= 0):
                    raise ModelSpecificationError(
                        formula=f"log() requires positive values in '{var_name}'",
                        suggestions=[
                            "Add constant: log(var + 1)",
                            "Transform data to ensure positive values",
                            "Check for zeros or negative values",
                        ]
                    )
                result_col = np.log(var_data).astype(np.float32)
            
            elif func_name == "exp":
                result_col = np.exp(var_data).astype(np.float32)
            
            elif func_name == "sqrt":
                if np.any(var_data < 0):
                    raise ModelSpecificationError(
                        formula=f"sqrt() requires non-negative values in '{var_name}'",
                        suggestions=[
                            "Check for negative values",
                            "Use absolute value: sqrt(abs(var))",
                        ]
                    )
                result_col = np.sqrt(var_data).astype(np.float32)
            
            elif func_name == "sin":
                result_col = np.sin(var_data).astype(np.float32)
            
            elif func_name == "cos":
                result_col = np.cos(var_data).astype(np.float32)
            
            elif func_name == "tan":
                result_col = np.tan(var_data).astype(np.float32)
            
            else:
                raise ModelSpecificationError(
                    formula=f"Unsupported function: {func_name}",
                    suggestions=[
                        "Supported functions: log, exp, sqrt, sin, cos, tan",
                        "Use I() for custom expressions",
                    ]
                )
            
            # Check for invalid results
            if np.any(~np.isfinite(result_col)):
                self.logger.warning(f"Function {func_name}({var_name}) produced non-finite values")
            
            return [result_col], [f"{func_name}({var_name})"]
        
        except Exception as e:
            raise ModelSpecificationError(
                formula=f"Error applying {func_name}() to '{var_name}': {e}",
                suggestions=[
                    "Check variable data range and values",
                    "Ensure function domain requirements are met",
                    "Consider data transformation",
                ]
            )
    
    def _build_polynomial_columns(
        self,
        term: PolynomialTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for polynomial terms."""
        var_name = term.variable_name
        degree = term.degree
        
        if var_name not in data_context.covariates:
            raise DataFormatError(
                specific_issue=f"Variable '{var_name}' not found for polynomial",
                missing_covariates=[var_name],
                suggestions=[
                    f"Available variables: {list(data_context.covariates.keys())}",
                    "Check variable name in poly() function",
                ]
            )
        
        var_data = np.array(data_context.covariates[var_name])
        if var_data.ndim != 1:
            raise ModelSpecificationError(
                formula=f"Polynomial requires 1D variable data for '{var_name}'",
                suggestions=[
                    "Polynomials work on individual-level covariates",
                    "Check variable data structure",
                ]
            )
        
        # Create polynomial columns
        columns = []
        names = []
        
        for power in range(1, degree + 1):
            poly_col = (var_data ** power).astype(np.float32)
            columns.append(poly_col)
            names.append(f"poly({var_name}, {degree}){power}")
        
        return columns, names


def build_design_matrix(
    formula: ParameterFormula,
    data_context: Any,
    n_occasions: Optional[int] = None
) -> DesignMatrixInfo:
    """
    Convenience function to build design matrix.
    
    Args:
        formula: ParameterFormula object
        data_context: DataContext with covariates
        n_occasions: Number of time occasions
        
    Returns:
        DesignMatrixInfo with constructed matrix
    """
    builder = DesignMatrixBuilder()
    return builder.build_matrix(formula, data_context, n_occasions)