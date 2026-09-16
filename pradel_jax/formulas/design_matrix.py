"""
Design matrix construction for pradel-jax.

Converts formula terms into design matrices for statistical modeling.
"""

import itertools

import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

from .terms import (
    Term,
    InterceptTerm,
    VariableTerm,
    InteractionTerm,
    FunctionTerm,
    PolynomialTerm,
)
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


def _reject_time_varying(
    var_name: str, n_columns: int, allow_time_varying: bool
) -> None:
    """Refuse to flatten a time-varying covariate into time-constant parameters.

    A 2D covariate expands to one design column per occasion, but a model whose
    parameters are a single value per individual then collapses those columns
    into one linear predictor.  The fit converges and reports an AIC, so the
    result looks ordinary while actually meaning "one time-constant parameter
    driven additively by every occasion at once" -- not a time-varying model.

    Builders that genuinely index parameters by occasion pass
    ``allow_time_varying=True``; until the Pradel model does so, this raises
    rather than letting the collapse happen silently.
    """
    if allow_time_varying:
        return
    raise ModelSpecificationError(
        formula=(
            f"covariate '{var_name}' varies over {n_columns} occasions, but this "
            f"model's parameters take a single value per individual. Expanding it "
            f"would silently collapse {n_columns} per-occasion columns into one "
            f"time-constant parameter. Use a time-constant summary of "
            f"'{var_name}' instead, or wait for occasion-specific parameter "
            f"support"
        ),
        parameter=var_name,
    )


class DesignMatrixBuilder:
    """
    Builds design matrices from formula terms and data.

    Handles various term types and creates appropriate design matrix columns.
    """

    def __init__(self, allow_time_varying: bool = False):
        self.logger = get_logger(self.__class__.__name__)
        self.time_varying_builder = TimeVaryingDesignMatrixBuilder()
        # Opt-in, because collapsing a time-varying covariate into a
        # time-constant parameter fails silently rather than loudly.
        self.allow_time_varying = allow_time_varying

    def build_matrix(
        self,
        formula: ParameterFormula,
        data_context: Any,  # DataContext from data.adapters
        n_occasions: Optional[int] = None,
        n_periods: Optional[int] = None,
    ) -> DesignMatrixInfo:
        """
        Build design matrix for a parameter formula.

        Args:
            formula: ParameterFormula object
            data_context: DataContext with covariates
            n_occasions: Number of time occasions (for time-varying parameters)
            n_periods: Length of this parameter's own time axis. ``phi`` and
                ``f`` are indexed by interval (``n_occasions - 1``) while ``p``
                is indexed by occasion (``n_occasions``), so the same covariate
                is sliced to a different length depending on which parameter it
                models. ``None`` means the caller wants a time-constant matrix
                and is the default, because most models are time-constant.

        Returns:
            DesignMatrixInfo with constructed matrix and metadata. The matrix is
            ``(n_individuals, n_columns)`` for a time-constant formula and
            ``(n_individuals, n_periods, n_columns)`` as soon as any term is
            occasion-specific. The column count -- and therefore the number of
            coefficients -- is the same either way.
        """
        self.logger.debug(
            f"Building design matrix for {formula.parameter.value}: {formula.formula_string}"
        )

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
                term, data_context, n_individuals, n_occasions, n_periods
            )
            matrix_columns.extend(columns)
            column_names.extend(names)

        # A categorical level that never occurs within this parameter's period
        # range produces an all-zero column, and its coefficient is then
        # unidentified -- the optimiser will wander on it and the Hessian will
        # be singular.  The commonest case is benign and easy to miss: a time
        # factor on phi carries a dummy for the final occasion, but phi spans
        # only the intervals *between* occasions, so that dummy can never be
        # 1.  Drop such columns and say so, rather than fitting a parameter the
        # data cannot inform.  The intercept is never dropped.
        kept_columns = []
        kept_names = []
        for column, name in zip(matrix_columns, column_names):
            if name != "(Intercept)" and not np.any(column):
                self.logger.warning(
                    f"dropping design column '{name}' from "
                    f"{formula.formula_string}: it is zero for every "
                    f"individual and period, so its coefficient is not "
                    f"identified"
                )
                continue
            kept_columns.append(column)
            kept_names.append(name)
        matrix_columns = kept_columns
        column_names = kept_names

        if not matrix_columns:
            raise ModelSpecificationError(
                formula=formula.formula_string,
                parameter=formula.parameter.value,
                suggestions=[
                    "Formula produced no design matrix columns",
                    "Check formula syntax and available covariates",
                    "Use '1' for intercept-only models",
                ],
            )

        # Combine columns into a matrix.  A time-constant column is
        # (n_individuals,); an occasion-specific one is (n_individuals,
        # n_periods).  If any column is occasion-specific the matrix gains a
        # period axis and the time-constant columns are broadcast along it, so
        # the coefficient vector is one value per column in both cases.  That is
        # what makes "phi ~ tier" a single tier coefficient applied at whatever
        # tier the individual held that year, rather than a separate coefficient
        # per year -- the latter is "phi ~ tier * year", a different model.
        if any(col.ndim == 2 for col in matrix_columns):
            widths = {col.shape[1] for col in matrix_columns if col.ndim == 2}
            if len(widths) != 1:
                raise ModelSpecificationError(
                    formula=(
                        f"occasion-specific terms in '{formula.formula_string}' "
                        f"disagree on the number of periods: {sorted(widths)}"
                    ),
                    parameter=formula.parameter.value,
                )
            width = widths.pop()
            design_matrix = np.stack(
                [
                    col if col.ndim == 2 else np.repeat(col[:, None], width, axis=1)
                    for col in matrix_columns
                ],
                axis=-1,
            )
        else:
            design_matrix = np.column_stack(matrix_columns)

        # Convert to JAX array
        design_matrix_jax = jnp.array(design_matrix, dtype=jnp.float64)

        self.logger.debug(
            f"Built design matrix: {design_matrix_jax.shape} "
            f"({len(column_names)} columns: {column_names})"
        )

        return DesignMatrixInfo(
            matrix=design_matrix_jax,
            column_names=column_names,
            parameter_count=len(column_names),
            has_intercept=formula.has_intercept,
            formula_string=formula.formula_string,
        )

    def _build_term_columns(
        self,
        term: Term,
        data_context: Any,
        n_individuals: int,
        n_occasions: int,
        n_periods: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Build design matrix columns for a single term.

        Args:
            term: Term object
            data_context: DataContext with covariates
            n_individuals: Number of individuals
            n_occasions: Number of occasions
            n_periods: Length of the parameter's time axis, or None for a
                time-constant matrix. See :meth:`build_matrix`.

        Returns:
            Tuple of (column arrays, column names). Each column is either
            (n_individuals,) or (n_individuals, n_periods).
        """
        if isinstance(term, InterceptTerm):
            return self._build_intercept_columns(n_individuals, n_occasions)

        elif isinstance(term, VariableTerm):
            return self._build_variable_columns(
                term, data_context, n_individuals, n_occasions, n_periods
            )

        elif isinstance(term, InteractionTerm):
            return self._build_interaction_columns(
                term, data_context, n_individuals, n_occasions, n_periods
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
                ],
            )

    def _build_intercept_columns(
        self, n_individuals: int, n_occasions: int
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build intercept column (all ones)."""
        intercept_col = np.ones(n_individuals, dtype=np.float64)
        return [intercept_col], ["(Intercept)"]

    def _slice_to_periods(
        self, var_name: str, data: np.ndarray, n_periods: Optional[int]
    ) -> np.ndarray:
        """Align an (n_individuals, T) covariate to a parameter's time axis.

        The value used for interval t is the one recorded at its *starting*
        occasion t.  This is the MARK convention for time-varying individual
        covariates: survival over 2019->2020 is driven by the status held in
        2019, because that is what is known when the interval begins.  Detection
        at occasion t uses occasion t directly, so ``p`` simply takes one more
        period than ``phi`` and ``f``.
        """
        width = data.shape[1]
        if n_periods is None:
            return np.asarray(data, dtype=np.float64)
        if width < n_periods:
            raise ModelSpecificationError(
                formula=(
                    f"covariate '{var_name}' is recorded for {width} occasions "
                    f"but this parameter is indexed over {n_periods} periods"
                ),
                parameter=var_name,
                suggestions=[
                    "Provide the covariate for every occasion in the study",
                    "phi and f need n_occasions - 1 values; p needs n_occasions",
                ],
            )
        return np.asarray(data[:, :n_periods], dtype=np.float64)

    def _build_variable_columns(
        self,
        term: VariableTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int,
        n_periods: Optional[int] = None,
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
                ],
            )

        # Check if this is a categorical variable
        metadata = getattr(data_context, "metadata", {}) or {}
        is_categorical = data_context.covariates.get(
            f"{var_name}_is_categorical", False
        )
        if not is_categorical:
            is_categorical = metadata.get(f"{var_name}_is_categorical", False)

        if is_categorical:
            # Handle categorical variable with dummy coding
            categories = data_context.covariates.get(f"{var_name}_categories")
            if categories is None:
                categories = metadata.get(f"{var_name}_categories", [])
            categorical_data = np.array(data_context.covariates[var_name])

            def _resolve_category_codes(
                labels: List[Any], raw_data: np.ndarray
            ) -> np.ndarray:
                if not labels:
                    return np.array([], dtype=float)

                flattened = raw_data.reshape(-1)
                # Filter NaNs for numeric arrays
                if flattened.dtype.kind in {"f", "i"}:
                    flattened = flattened.astype(float)
                    flattened = flattened[~np.isnan(flattened)]
                expected = np.arange(len(labels), dtype=float)
                if flattened.size:
                    unique_vals = np.unique(flattened)
                    if len(unique_vals) == len(labels) and np.allclose(
                        np.sort(unique_vals), expected
                    ):
                        return expected

                # Try to coerce labels to numeric codes
                try:
                    numeric_labels = np.array(
                        [float(label) for label in labels], dtype=float
                    )
                    if numeric_labels.shape[0] == len(labels):
                        return numeric_labels
                except (TypeError, ValueError):
                    pass

                return expected

            category_codes = _resolve_category_codes(categories, categorical_data)

            # Time-varying categorical (2D): one dummy per non-reference
            # level, each varying over the parameter's periods.  There is
            # deliberately NOT one dummy per (level, occasion) pair: that would
            # be the level-by-time interaction, and it is not what "~ tier"
            # asks for.
            if categorical_data.ndim == 2:
                _reject_time_varying(
                    var_name, categorical_data.shape[1], self.allow_time_varying
                )
                codes = self._slice_to_periods(var_name, categorical_data, n_periods)
                if len(categories) <= 1:
                    # Single level - intercept-like, and therefore constant
                    return (
                        [np.ones(codes.shape, dtype=np.float64)],
                        [var_name],
                    )
                columns = []
                names = []
                # Codes correspond to entries in `categories`; drop the first
                # level as the reference for identifiability.
                for code_value, category in zip(category_codes[1:], categories[1:]):
                    columns.append(np.isclose(codes, code_value).astype(np.float64))
                    names.append(f"{var_name}_{category}")
                return columns, names
            else:
                categorical_codes = categorical_data.astype(float)
                # Create dummy variables (drop first category for identifiability)
                if len(categories) <= 1:
                    # Only one category - create intercept-like column
                    column = np.ones(n_individuals, dtype=np.float64)
                    return [column], [var_name]
                else:
                    # Multiple categories - create dummy variables (drop first)
                    columns = []
                    names = []

                    for code_value, category in zip(
                        category_codes[1:], categories[1:]
                    ):  # Skip first category
                        dummy_col = np.isclose(categorical_codes, code_value).astype(
                            np.float64
                        )
                        columns.append(dummy_col)
                        names.append(f"{var_name}_{category}")

                    return columns, names
        else:
            # Handle numeric variable
            covariate_data = np.array(data_context.covariates[var_name])

            # Handle different data shapes
            if covariate_data.ndim == 1 and len(covariate_data) == n_individuals:
                # Individual-level covariate
                column = covariate_data.astype(np.float64)
                return [column], [var_name]
            elif covariate_data.ndim == 2:
                # Time-varying numeric covariate: a single coefficient applied
                # to whatever value the individual held in each period.
                _reject_time_varying(
                    var_name, covariate_data.shape[1], self.allow_time_varying
                )
                self.logger.info(f"Processing time-varying covariate: {var_name}")
                column = self._slice_to_periods(var_name, covariate_data, n_periods)
                if np.any(np.isnan(column)):
                    # Deliberately not imputed.  The row mean that used to stand
                    # here invents values never observed, which for a state-like
                    # variable is meaningless -- "tier 1.4" is not a thing.  A
                    # state variable belongs on the categorical path, where
                    # missing/inactive is an explicit level; a genuinely
                    # continuous covariate has to be completed by the caller,
                    # who knows what the gap means.
                    raise DataFormatError(
                        specific_issue=(
                            f"time-varying covariate '{var_name}' has missing "
                            f"values, and this model will not impute them"
                        ),
                        suggestions=[
                            "Fill the gaps explicitly with a value that means "
                            "something in this study",
                            "For state-like variables (e.g. tier) declare the "
                            "covariate categorical so missing becomes its own "
                            "level",
                        ],
                    )
                return [column], [var_name]
            else:
                raise DataFormatError(
                    specific_issue=f"Covariate '{var_name}' has unexpected shape: {covariate_data.shape}",
                    suggestions=[
                        f"Expected shape: ({n_individuals},) or ({n_individuals}, {n_occasions})",
                        "Check covariate data structure",
                        "Use time-varying covariate framework for multi-dimensional data",
                    ],
                )

    def _build_interaction_columns(
        self,
        term: InteractionTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int,
        n_periods: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for interaction terms.

        A categorical variable contributes one column per non-reference level,
        so an interaction involving one is the product of the *level sets*, not
        a single column: tier (two levels beyond the reference) crossed with
        year (one per occasion after the first) gives one column per
        tier-by-year cell.  Multiplying the flattened column lists instead --
        which is what this did before, when it did not simply refuse -- would
        collapse every level into one column and quietly fit a different model.
        """
        per_variable_columns = []
        per_variable_names = []

        for var_name in term.variables:
            var_term = VariableTerm(var_name)
            columns, names = self._build_variable_columns(
                var_term, data_context, n_individuals, n_occasions, n_periods
            )
            if not columns:
                raise ModelSpecificationError(
                    formula=(
                        f"variable '{var_name}' in interaction "
                        f"{':'.join(term.variables)} produced no columns"
                    ),
                    parameter=var_name,
                )
            per_variable_columns.append(columns)
            per_variable_names.append(names)

        # A time-constant factor multiplied by an occasion-specific one is
        # occasion-specific, so the 1D operand broadcasts along the period axis.
        width = next(
            (
                col.shape[1]
                for columns in per_variable_columns
                for col in columns
                if col.ndim == 2
            ),
            None,
        )

        def _align(col: np.ndarray) -> np.ndarray:
            if width is None or col.ndim == 2:
                return col
            return np.repeat(col[:, None], width, axis=1)

        columns = []
        names = []
        for combo in itertools.product(
            *(range(len(cols)) for cols in per_variable_columns)
        ):
            product_col = _align(per_variable_columns[0][combo[0]]).copy()
            for position, index in enumerate(combo[1:], start=1):
                product_col = product_col * _align(
                    per_variable_columns[position][index]
                )
            columns.append(product_col)
            names.append(
                ":".join(
                    per_variable_names[position][index]
                    for position, index in enumerate(combo)
                )
            )

        return columns, names

    def _build_function_columns(
        self,
        term: FunctionTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for function terms."""
        func_name = term.function_name

        if func_name == "I":
            # Identity function - evaluate expression
            return self._build_identity_function(
                term, data_context, n_individuals, n_occasions
            )

        elif func_name in ["log", "exp", "sqrt", "sin", "cos", "tan"]:
            # Standard mathematical functions
            return self._build_math_function(
                term, data_context, n_individuals, n_occasions
            )

        else:
            # Unknown function - treat as identity
            self.logger.warning(
                f"Unknown function '{func_name}' - treating as identity"
            )
            return self._build_identity_function(
                term, data_context, n_individuals, n_occasions
            )

    def _build_identity_function(
        self,
        term: FunctionTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for I() function (identity/expression evaluation)."""
        if len(term.arguments) != 1:
            raise ModelSpecificationError(
                formula=f"I() function requires exactly one argument: {term.expression}",
                suggestions=[
                    "Use I(expression) for mathematical expressions",
                    "Examples: I(age^2), I(age*2), I(log(weight))",
                ],
            )

        expr = term.arguments[0]

        # Simple expression evaluation for common cases
        if "^2" in expr:
            # Quadratic term: var^2
            var_name = expr.replace("^2", "").strip()
            if var_name in data_context.covariates:
                var_data = np.array(data_context.covariates[var_name])
                if var_data.ndim == 1:
                    squared_col = (var_data**2).astype(np.float64)
                    return [squared_col], [f"I({expr})"]

        elif "^3" in expr:
            # Cubic term: var^3
            var_name = expr.replace("^3", "").strip()
            if var_name in data_context.covariates:
                var_data = np.array(data_context.covariates[var_name])
                if var_data.ndim == 1:
                    cubed_col = (var_data**3).astype(np.float64)
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
                            scaled_col = (var_data * const).astype(np.float64)
                            return [scaled_col], [f"I({expr})"]
                except ValueError:
                    # Both are variables - create interaction
                    if (
                        left in data_context.covariates
                        and right in data_context.covariates
                    ):
                        left_data = np.array(data_context.covariates[left])
                        right_data = np.array(data_context.covariates[right])
                        if left_data.ndim == 1 and right_data.ndim == 1:
                            product_col = (left_data * right_data).astype(np.float64)
                            return [product_col], [f"I({expr})"]

        # Fallback: treat as simple variable if it exists
        if expr in data_context.covariates:
            var_term = VariableTerm(expr)
            return self._build_variable_columns(
                var_term, data_context, n_individuals, n_occasions
            )

        raise ModelSpecificationError(
            formula=f"Cannot evaluate expression: I({expr})",
            suggestions=[
                "Supported expressions: var^2, var^3, var*constant, var1*var2",
                "Ensure all variables exist in data",
                "Use simple variable names in expressions",
            ],
        )

    def _build_math_function(
        self,
        term: FunctionTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int,
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Build columns for mathematical functions."""
        if len(term.arguments) != 1:
            raise ModelSpecificationError(
                formula=f"{term.function_name}() requires exactly one argument",
                suggestions=[
                    f"Use {term.function_name}(variable_name)",
                    "Ensure variable exists in data",
                ],
            )

        var_name = term.arguments[0].strip()
        if var_name not in data_context.covariates:
            raise DataFormatError(
                specific_issue=f"Variable '{var_name}' not found for {term.function_name}() function",
                missing_covariates=[var_name],
                suggestions=[
                    f"Available variables: {list(data_context.covariates.keys())}",
                    "Check variable name in function",
                ],
            )

        var_data = np.array(data_context.covariates[var_name])
        if var_data.ndim != 1:
            raise ModelSpecificationError(
                formula=f"Function {term.function_name}() requires 1D variable data",
                suggestions=[
                    "Mathematical functions work on individual-level covariates",
                    "Check variable data structure",
                ],
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
                        ],
                    )
                result_col = np.log(var_data).astype(np.float64)

            elif func_name == "exp":
                result_col = np.exp(var_data).astype(np.float64)

            elif func_name == "sqrt":
                if np.any(var_data < 0):
                    raise ModelSpecificationError(
                        formula=f"sqrt() requires non-negative values in '{var_name}'",
                        suggestions=[
                            "Check for negative values",
                            "Use absolute value: sqrt(abs(var))",
                        ],
                    )
                result_col = np.sqrt(var_data).astype(np.float64)

            elif func_name == "sin":
                result_col = np.sin(var_data).astype(np.float64)

            elif func_name == "cos":
                result_col = np.cos(var_data).astype(np.float64)

            elif func_name == "tan":
                result_col = np.tan(var_data).astype(np.float64)

            else:
                raise ModelSpecificationError(
                    formula=f"Unsupported function: {func_name}",
                    suggestions=[
                        "Supported functions: log, exp, sqrt, sin, cos, tan",
                        "Use I() for custom expressions",
                    ],
                )

            # Check for invalid results
            if np.any(~np.isfinite(result_col)):
                self.logger.warning(
                    f"Function {func_name}({var_name}) produced non-finite values"
                )

            return [result_col], [f"{func_name}({var_name})"]

        except Exception as e:
            raise ModelSpecificationError(
                formula=f"Error applying {func_name}() to '{var_name}': {e}",
                suggestions=[
                    "Check variable data range and values",
                    "Ensure function domain requirements are met",
                    "Consider data transformation",
                ],
            )

    def _build_polynomial_columns(
        self,
        term: PolynomialTerm,
        data_context: Any,
        n_individuals: int,
        n_occasions: int,
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
                ],
            )

        var_data = np.array(data_context.covariates[var_name])
        if var_data.ndim != 1:
            raise ModelSpecificationError(
                formula=f"Polynomial requires 1D variable data for '{var_name}'",
                suggestions=[
                    "Polynomials work on individual-level covariates",
                    "Check variable data structure",
                ],
            )

        # Create polynomial columns
        columns = []
        names = []

        for power in range(1, degree + 1):
            poly_col = (var_data**power).astype(np.float64)
            columns.append(poly_col)
            names.append(f"poly({var_name}, {degree}){power}")

        return columns, names


def build_design_matrix(
    formula: ParameterFormula,
    data_context: Any,
    n_occasions: Optional[int] = None,
    allow_time_varying: bool = False,
    n_periods: Optional[int] = None,
) -> DesignMatrixInfo:
    """
    Convenience function to build design matrix.

    Args:
        formula: ParameterFormula object
        data_context: DataContext with covariates
        n_occasions: Number of time occasions
        allow_time_varying: Permit 2D covariates to become occasion-specific
            design columns. Only safe for models that index parameters by
            occasion; see _reject_time_varying.
        n_periods: Length of this parameter's time axis. See
            :meth:`DesignMatrixBuilder.build_matrix`.

    Returns:
        DesignMatrixInfo with constructed matrix
    """
    builder = DesignMatrixBuilder(allow_time_varying=allow_time_varying)
    return builder.build_matrix(formula, data_context, n_occasions, n_periods)
