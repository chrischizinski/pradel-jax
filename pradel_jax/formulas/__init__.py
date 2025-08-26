"""
Formula system for pradel-jax.

Provides R-style formula parsing and design matrix construction.
"""

from .parser import FormulaParser, parse_formula, create_simple_spec
from .terms import Term, InterceptTerm, VariableTerm, InteractionTerm, FunctionTerm
from .design_matrix import DesignMatrixBuilder, build_design_matrix
from .spec import FormulaSpec, ParameterFormula, ParameterType

__all__ = [
    # Main API
    "parse_formula",
    "build_design_matrix",
    "create_simple_spec",
    # Core classes
    "FormulaParser",
    "DesignMatrixBuilder",
    "FormulaSpec",
    "ParameterFormula",
    "ParameterType",
    # Term types
    "Term",
    "InterceptTerm",
    "VariableTerm",
    "InteractionTerm",
    "FunctionTerm",
]
