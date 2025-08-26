"""
Formula term representations for pradel-jax.

Defines different types of terms that can appear in model formulas.
"""

from abc import ABC, abstractmethod
from typing import List, Set, Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum
import re

from ..core.exceptions import ModelSpecificationError


class TermType(str, Enum):
    """Types of formula terms."""

    INTERCEPT = "intercept"
    VARIABLE = "variable"
    INTERACTION = "interaction"
    FUNCTION = "function"
    POLYNOMIAL = "polynomial"


@dataclass
class Term(ABC):
    """Abstract base class for formula terms."""

    @abstractmethod
    def get_variable_names(self) -> Set[str]:
        """Get all variable names used in this term."""
        pass

    @abstractmethod
    def get_parameter_count(self) -> int:
        """Get number of parameters this term contributes."""
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Convert term to string representation."""
        pass

    @abstractmethod
    def validate_variables(self, available_variables: Set[str]) -> None:
        """Validate that all required variables are available."""
        pass


@dataclass
class InterceptTerm(Term):
    """Intercept term (constant)."""

    def get_variable_names(self) -> Set[str]:
        """Intercept doesn't use any variables."""
        return set()

    def get_parameter_count(self) -> int:
        """Intercept contributes one parameter."""
        return 1

    def to_string(self) -> str:
        """String representation."""
        return "1"

    def validate_variables(self, available_variables: Set[str]) -> None:
        """Intercept is always valid."""
        pass

    def is_intercept(self) -> bool:
        """Check if this is an intercept term."""
        return True


@dataclass
class VariableTerm(Term):
    """Simple variable term."""

    variable_name: str

    def __post_init__(self):
        """Validate variable name."""
        if not self.variable_name or not isinstance(self.variable_name, str):
            raise ModelSpecificationError(
                formula=f"Invalid variable name: {self.variable_name}",
                suggestions=[
                    "Variable names must be non-empty strings",
                    "Use valid Python identifier names",
                    "Examples: 'age', 'sex', 'weight'",
                ],
            )

    def get_variable_names(self) -> Set[str]:
        """Return the variable name."""
        return {self.variable_name}

    def get_parameter_count(self) -> int:
        """Simple variable contributes one parameter."""
        return 1

    def to_string(self) -> str:
        """String representation."""
        return self.variable_name

    def validate_variables(self, available_variables: Set[str]) -> None:
        """Check if variable is available."""
        if self.variable_name not in available_variables:
            raise ModelSpecificationError(
                formula=f"Variable '{self.variable_name}' not found",
                missing_covariates=[self.variable_name],
                suggestions=[
                    f"Available variables: {sorted(available_variables)}",
                    "Check variable name spelling",
                    "Ensure variable exists in data",
                ],
            )

    def is_intercept(self) -> bool:
        """Check if this is an intercept term."""
        return False


@dataclass
class InteractionTerm(Term):
    """Interaction between variables (e.g., age * sex)."""

    variables: List[str]

    def __post_init__(self):
        """Validate interaction."""
        if len(self.variables) < 2:
            raise ModelSpecificationError(
                formula=f"Interaction requires at least 2 variables: {self.variables}",
                suggestions=[
                    "Interactions need multiple variables",
                    "Use format: 'var1 * var2' or 'var1:var2'",
                    "For single variables, use VariableTerm",
                ],
            )

        # Check for duplicates
        if len(set(self.variables)) != len(self.variables):
            raise ModelSpecificationError(
                formula=f"Duplicate variables in interaction: {self.variables}",
                suggestions=[
                    "Remove duplicate variables",
                    "Each variable should appear once per interaction",
                ],
            )

    def get_variable_names(self) -> Set[str]:
        """Return all variables in the interaction."""
        return set(self.variables)

    def get_parameter_count(self) -> int:
        """Interaction contributes one parameter."""
        return 1

    def to_string(self) -> str:
        """String representation."""
        return " * ".join(self.variables)

    def validate_variables(self, available_variables: Set[str]) -> None:
        """Check if all variables are available."""
        missing = set(self.variables) - available_variables
        if missing:
            raise ModelSpecificationError(
                formula=f"Interaction variables not found: {sorted(missing)}",
                missing_covariates=list(missing),
                suggestions=[
                    f"Available variables: {sorted(available_variables)}",
                    "Check variable name spelling in interaction",
                    "Ensure all variables exist in data",
                ],
            )

    def is_intercept(self) -> bool:
        """Check if this is an intercept term."""
        return False


@dataclass
class FunctionTerm(Term):
    """Function transformation of variables (e.g., I(age^2), log(weight))."""

    function_name: str
    arguments: List[str]
    expression: str

    def __post_init__(self):
        """Validate function term."""
        if not self.function_name:
            raise ModelSpecificationError(
                formula=f"Function name cannot be empty: {self.expression}",
                suggestions=[
                    "Specify a valid function name",
                    "Examples: 'I', 'log', 'exp', 'sqrt'",
                ],
            )

        if not self.arguments:
            raise ModelSpecificationError(
                formula=f"Function requires arguments: {self.expression}",
                suggestions=[
                    "Provide at least one argument",
                    "Examples: I(age^2), log(weight), poly(age, 2)",
                ],
            )

    def get_variable_names(self) -> Set[str]:
        """Extract variable names from arguments."""
        variables = set()
        for arg in self.arguments:
            # Simple extraction - look for variable names in argument
            # This could be more sophisticated for complex expressions
            var_matches = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", arg)
            for var in var_matches:
                # Exclude common function names and numbers
                if var not in {"I", "log", "exp", "sqrt", "poly", "sin", "cos", "tan"}:
                    variables.add(var)
        return variables

    def get_parameter_count(self) -> int:
        """Function term contributes one parameter by default."""
        # For polynomial terms, this might be different
        if self.function_name == "poly":
            # Extract degree if specified
            try:
                if len(self.arguments) >= 2:
                    degree = int(self.arguments[1])
                    return degree
            except (ValueError, IndexError):
                pass
        return 1

    def to_string(self) -> str:
        """String representation."""
        return self.expression

    def validate_variables(self, available_variables: Set[str]) -> None:
        """Check if all variables used in function are available."""
        required_vars = self.get_variable_names()
        missing = required_vars - available_variables
        if missing:
            raise ModelSpecificationError(
                formula=f"Function variables not found: {sorted(missing)} in {self.expression}",
                missing_covariates=list(missing),
                suggestions=[
                    f"Available variables: {sorted(available_variables)}",
                    "Check variable names in function expression",
                    "Ensure all variables exist in data",
                ],
            )

    def is_intercept(self) -> bool:
        """Check if this is an intercept term."""
        return False


@dataclass
class PolynomialTerm(Term):
    """Polynomial term (e.g., poly(age, 2) for quadratic)."""

    variable_name: str
    degree: int

    def __post_init__(self):
        """Validate polynomial term."""
        if self.degree < 1:
            raise ModelSpecificationError(
                formula=f"Polynomial degree must be >= 1: {self.degree}",
                suggestions=[
                    "Use degree 1 for linear terms",
                    "Use degree 2 for quadratic terms",
                    "Higher degrees may cause fitting issues",
                ],
            )

        if self.degree > 5:
            raise ModelSpecificationError(
                formula=f"Polynomial degree too high: {self.degree}",
                suggestions=[
                    "High-degree polynomials can cause numerical issues",
                    "Consider using splines for flexible curves",
                    "Typical range: 1-3 degrees",
                ],
            )

    def get_variable_names(self) -> Set[str]:
        """Return the base variable name."""
        return {self.variable_name}

    def get_parameter_count(self) -> int:
        """Polynomial contributes degree parameters."""
        return self.degree

    def to_string(self) -> str:
        """String representation."""
        return f"poly({self.variable_name}, {self.degree})"

    def validate_variables(self, available_variables: Set[str]) -> None:
        """Check if base variable is available."""
        if self.variable_name not in available_variables:
            raise ModelSpecificationError(
                formula=f"Polynomial variable '{self.variable_name}' not found",
                missing_covariates=[self.variable_name],
                suggestions=[
                    f"Available variables: {sorted(available_variables)}",
                    "Check variable name spelling",
                    "Ensure variable exists in data",
                ],
            )

    def is_intercept(self) -> bool:
        """Check if this is an intercept term."""
        return False


def create_term(term_string: str) -> Term:
    """
    Create appropriate Term object from string representation.

    Args:
        term_string: String representation of the term

    Returns:
        Appropriate Term object

    Examples:
        "1" -> InterceptTerm()
        "age" -> VariableTerm("age")
        "age * sex" -> InteractionTerm(["age", "sex"])
        "I(age^2)" -> FunctionTerm("I", ["age^2"], "I(age^2)")
        "poly(age, 2)" -> PolynomialTerm("age", 2)
    """
    term_string = term_string.strip()

    # Intercept term
    if term_string == "1":
        return InterceptTerm()

    # Function terms
    if "(" in term_string and ")" in term_string:
        # Extract function name and arguments
        match = re.match(r"(\w+)\((.*)\)", term_string)
        if match:
            func_name = match.group(1)
            args_str = match.group(2)

            # Special handling for polynomial
            if func_name == "poly":
                args = [arg.strip() for arg in args_str.split(",")]
                if len(args) >= 2:
                    try:
                        variable = args[0]
                        degree = int(args[1])
                        return PolynomialTerm(variable, degree)
                    except ValueError:
                        pass

            # General function term
            args = [arg.strip() for arg in args_str.split(",")]
            return FunctionTerm(func_name, args, term_string)

    # Interaction terms
    if "*" in term_string:
        variables = [var.strip() for var in term_string.split("*")]
        return InteractionTerm(variables)

    # Simple variable term
    if term_string.isidentifier():
        return VariableTerm(term_string)

    # If we can't parse it, create a generic function term
    return FunctionTerm("unknown", [term_string], term_string)


def parse_terms(formula_string: str) -> List[Term]:
    """
    Parse a formula string into a list of terms.

    Args:
        formula_string: R-style formula string

    Returns:
        List of Term objects

    Examples:
        "1" -> [InterceptTerm()]
        "age + sex" -> [InterceptTerm(), VariableTerm("age"), VariableTerm("sex")]
        "age * sex" -> [InterceptTerm(), VariableTerm("age"), VariableTerm("sex"), InteractionTerm(["age", "sex"])]
        "age + I(age^2)" -> [InterceptTerm(), VariableTerm("age"), FunctionTerm("I", ["age^2"], "I(age^2)")]
    """
    # Remove ~ if present (formula response part)
    if "~" in formula_string:
        formula_string = formula_string.split("~", 1)[1].strip()

    # Handle special case: intercept only
    if formula_string.strip() == "1":
        return [InterceptTerm()]

    # Handle no intercept case
    has_intercept = True
    if formula_string.startswith("-1") or formula_string.startswith("0"):
        has_intercept = False
        # Remove the -1 or 0 part
        formula_string = re.sub(r"^(-1|0)\s*\+?\s*", "", formula_string).strip()
        if not formula_string:
            return []  # No terms at all

    terms = []

    # Add intercept if not explicitly removed
    if has_intercept:
        terms.append(InterceptTerm())

    # Split by + to get individual terms
    # This is simplified - real R formula parsing is more complex
    if formula_string:
        term_strings = [term.strip() for term in formula_string.split("+")]

        for term_str in term_strings:
            if term_str and term_str != "1":  # Skip empty and redundant intercept
                # Handle interactions - need to add main effects too
                if "*" in term_str:
                    # For a * b, add a, b, and a*b
                    variables = [var.strip() for var in term_str.split("*")]

                    # Add main effects (if not already present)
                    for var in variables:
                        var_term = VariableTerm(var)
                        if not any(
                            isinstance(t, VariableTerm) and t.variable_name == var
                            for t in terms
                        ):
                            terms.append(var_term)

                    # Add interaction
                    terms.append(InteractionTerm(variables))
                else:
                    # Regular term
                    terms.append(create_term(term_str))

    return terms
