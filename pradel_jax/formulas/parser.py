"""
Formula parser for pradel-jax.

Provides R-style formula parsing for capture-recapture models.
"""

import re
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass

from .terms import Term, parse_terms
from .spec import ParameterFormula, FormulaSpec, ParameterType
from ..core.exceptions import ModelSpecificationError
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ParsedFormula:
    """Result of parsing a formula string."""

    response: Optional[str]
    terms: List[Term]
    has_intercept: bool
    original_string: str


class FormulaParser:
    """
    Parser for R-style model formulas.

    Supports standard R formula syntax:
    - Basic terms: age, sex, weight
    - Interactions: age * sex (expands to age + sex + age:sex)
    - Functions: I(age^2), log(weight), poly(age, 2)
    - Intercept control: +1 (default), -1 or 0 (no intercept)
    - Complex expressions: age + sex + age:sex + I(age^2)
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    def parse(self, formula_string: str) -> ParsedFormula:
        """
        Parse a formula string into components.

        Args:
            formula_string: R-style formula string

        Returns:
            ParsedFormula object with parsed components

        Examples:
            "~1" -> response=None, terms=[InterceptTerm()], has_intercept=True
            "y ~ age + sex" -> response="y", terms=[InterceptTerm(), VariableTerm("age"), VariableTerm("sex")]
            "phi ~ age * sex" -> response="phi", terms with main effects and interaction
        """
        original_string = formula_string.strip()
        self.logger.debug(f"Parsing formula: {original_string}")

        if not original_string:
            raise ModelSpecificationError(
                formula=original_string,
                suggestions=[
                    "Provide a non-empty formula",
                    "Use '~1' for intercept-only models",
                    "Examples: '~age', '~age + sex', '~1'",
                ],
            )

        # Split on ~ to separate response and predictors
        if "~" in formula_string:
            parts = formula_string.split("~", 1)
            if len(parts) != 2:
                raise ModelSpecificationError(
                    formula=original_string,
                    suggestions=[
                        "Formula should have format 'response ~ predictors'",
                        "Or just '~ predictors' for no response",
                        "Examples: 'phi ~ age', '~ age + sex'",
                    ],
                )

            response_part = parts[0].strip()
            predictor_part = parts[1].strip()

            # Handle empty response (formula starts with ~)
            response = response_part if response_part else None
        else:
            # No ~ found - treat as predictor-only formula
            response = None
            predictor_part = formula_string.strip()

        # Parse predictor terms
        try:
            terms = parse_terms(predictor_part)
        except Exception as e:
            raise ModelSpecificationError(
                formula=original_string,
                suggestions=[
                    f"Error parsing predictors: {e}",
                    "Check formula syntax",
                    "Supported: +, *, :, I(), poly(), functions",
                    "Examples: 'age + sex', 'age * sex', 'I(age^2)'",
                ],
            )

        # Determine if intercept is included
        has_intercept = any(
            term.__class__.__name__ == "InterceptTerm" for term in terms
        )

        self.logger.debug(
            f"Parsed formula: response={response}, {len(terms)} terms, "
            f"intercept={has_intercept}"
        )

        return ParsedFormula(
            response=response,
            terms=terms,
            has_intercept=has_intercept,
            original_string=original_string,
        )

    def create_parameter_formula(
        self, parameter_type: ParameterType, formula_string: str
    ) -> ParameterFormula:
        """
        Create a ParameterFormula from a formula string.

        Args:
            parameter_type: Type of parameter (phi, p, f, etc.)
            formula_string: Formula string for this parameter

        Returns:
            ParameterFormula object
        """
        parsed = self.parse(formula_string)

        # Create ParameterFormula
        param_formula = ParameterFormula(
            parameter=parameter_type,
            formula_string=formula_string,
            terms=parsed.terms,
            has_intercept=parsed.has_intercept,
        )

        self.logger.debug(
            f"Created {parameter_type.value} formula: {formula_string} "
            f"({len(parsed.terms)} terms)"
        )

        return param_formula

    def parse_model_spec(
        self,
        spec_dict: Dict[str, str],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> FormulaSpec:
        """
        Parse a complete model specification from dictionary.

        Args:
            spec_dict: Dictionary with parameter formulas
            name: Optional model name
            description: Optional model description

        Returns:
            Complete FormulaSpec object

        Examples:
            {"phi": "~1", "p": "~sex", "f": "~1"}
            {"phi": "~age+sex", "p": "~time", "f": "~1"}
        """
        self.logger.debug(f"Parsing model spec: {spec_dict}")

        # Parse required parameters
        try:
            phi = self.create_parameter_formula(
                ParameterType.PHI, spec_dict.get("phi", "~1")
            )
            p = self.create_parameter_formula(ParameterType.P, spec_dict.get("p", "~1"))
            f = self.create_parameter_formula(ParameterType.F, spec_dict.get("f", "~1"))
        except Exception as e:
            raise ModelSpecificationError(
                formula=str(spec_dict),
                suggestions=[
                    f"Error parsing required parameters: {e}",
                    "Required: phi, p, f formulas",
                    "Example: {'phi': '~age', 'p': '~1', 'f': '~1'}",
                ],
            )

        # Parse optional parameters
        psi = None
        if "psi" in spec_dict:
            try:
                psi = self.create_parameter_formula(ParameterType.PSI, spec_dict["psi"])
            except Exception as e:
                self.logger.warning(f"Error parsing psi formula: {e}")

        r = None
        if "r" in spec_dict:
            try:
                r = self.create_parameter_formula(ParameterType.R, spec_dict["r"])
            except Exception as e:
                self.logger.warning(f"Error parsing r formula: {e}")

        # Create FormulaSpec
        formula_spec = FormulaSpec(
            phi=phi, p=p, f=f, psi=psi, r=r, name=name, description=description
        )

        self.logger.info(
            f"Created model spec '{name or 'unnamed'}': "
            f"phi={phi.formula_string}, p={p.formula_string}, f={f.formula_string}"
        )

        return formula_spec


# Convenience functions for common use cases


def parse_formula(formula_string: str) -> ParsedFormula:
    """
    Parse a single formula string.

    Args:
        formula_string: R-style formula string

    Returns:
        ParsedFormula object
    """
    parser = FormulaParser()
    return parser.parse(formula_string)


def create_simple_spec(
    phi: str = "~1", p: str = "~1", f: str = "~1", name: Optional[str] = None
) -> FormulaSpec:
    """
    Create a simple model specification.

    Args:
        phi: Formula for survival probability
        p: Formula for detection probability
        f: Formula for recruitment probability
        name: Optional model name

    Returns:
        FormulaSpec object

    Examples:
        create_simple_spec("~age", "~sex", "~1", "Age+Sex model")
        create_simple_spec("~1", "~1", "~1", "Constant model")
    """
    parser = FormulaParser()
    return parser.parse_model_spec({"phi": phi, "p": p, "f": f}, name=name)


def parse_formula_list(formulas: List[Dict[str, str]]) -> List[FormulaSpec]:
    """
    Parse a list of formula specifications.

    Args:
        formulas: List of formula dictionaries

    Returns:
        List of FormulaSpec objects

    Examples:
        formulas = [
            {"phi": "~1", "p": "~1", "f": "~1"},
            {"phi": "~age", "p": "~sex", "f": "~1"},
            {"phi": "~age*sex", "p": "~time", "f": "~age"},
        ]
        specs = parse_formula_list(formulas)
    """
    parser = FormulaParser()
    specs = []

    for i, formula_dict in enumerate(formulas):
        try:
            name = formula_dict.get("name", f"Model_{i+1}")
            description = formula_dict.get("description")
            spec = parser.parse_model_spec(formula_dict, name, description)
            specs.append(spec)
        except Exception as e:
            logger.warning(f"Skipping invalid formula {i+1}: {e}")

    return specs


def validate_formula_syntax(formula_string: str) -> bool:
    """
    Validate formula syntax without full parsing.

    Args:
        formula_string: Formula string to validate

    Returns:
        True if syntax is valid

    Raises:
        ModelSpecificationError: If syntax is invalid
    """
    try:
        parse_formula(formula_string)
        return True
    except ModelSpecificationError:
        raise
    except Exception as e:
        raise ModelSpecificationError(
            formula=formula_string,
            suggestions=[
                f"Syntax error: {e}",
                "Check formula syntax",
                "Use R-style formula notation",
            ],
        )
