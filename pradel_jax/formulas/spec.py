"""
Formula specification classes for pradel-jax.

Defines the structure and validation of model formulas.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum

from ..core.exceptions import ModelSpecificationError


class ParameterType(str, Enum):
    """Types of parameters in capture-recapture models."""

    PHI = "phi"  # Survival probability
    P = "p"  # Detection probability
    F = "f"  # Recruitment probability
    PSI = "psi"  # Transition probability (multi-state)
    R = "r"  # Recovery probability


@dataclass
class ParameterFormula:
    """
    Formula specification for a single parameter.

    Examples:
        phi ~ 1                    # Intercept only
        phi ~ age + sex            # Additive effects
        phi ~ age * sex            # Interaction
        phi ~ I(age^2) + sex       # Function transformation
    """

    parameter: ParameterType
    formula_string: str
    terms: List["Term"] = field(default_factory=list)
    has_intercept: bool = True

    def __post_init__(self):
        """Validate and parse formula after initialization."""
        if not self.formula_string.strip():
            raise ModelSpecificationError(
                formula=self.formula_string,
                parameter=self.parameter.value,
                suggestions=[
                    "Provide a non-empty formula",
                    "Use '1' for intercept-only models",
                    "Example: 'age + sex' or '1'",
                ],
            )
            
        # Parse the formula string into terms
        try:
            from ..formulas.parser import parse_formula
            parsed = parse_formula(self.formula_string)
            self.terms = parsed.terms
            self.has_intercept = parsed.has_intercept
        except ImportError:
            # If parser is not available, create basic terms
            from ..formulas.terms import InterceptTerm
            if self.formula_string.strip() in ['~1', '1']:
                self.terms = [InterceptTerm()]
                self.has_intercept = True
        except Exception as e:
            # If parsing fails, handle gracefully for simple cases
            from ..formulas.terms import InterceptTerm
            if self.formula_string.strip() in ['~1', '1']:
                self.terms = [InterceptTerm()]
                self.has_intercept = True
            else:
                raise ModelSpecificationError(
                    formula=self.formula_string,
                    parameter=self.parameter.value,
                    suggestions=[
                        f"Failed to parse formula: {e}",
                        "Check formula syntax",
                        "Use '1' for intercept-only models",
                    ],
                )

    def validate_covariates(self, available_covariates: List[str]) -> None:
        """
        Validate that all covariates in formula are available.

        Args:
            available_covariates: List of available covariate names

        Raises:
            ModelSpecificationError: If missing covariates found
        """
        # Extract covariate names from terms
        required_covariates = set()
        for term in self.terms:
            required_covariates.update(term.get_variable_names())

        # Remove special terms
        required_covariates.discard("1")  # Intercept

        # Check for missing covariates
        missing = required_covariates - set(available_covariates)
        if missing:
            raise ModelSpecificationError(
                formula=self.formula_string,
                parameter=self.parameter.value,
                available_covariates=available_covariates,
                missing_covariates=list(missing),
            )

    def get_complexity(self) -> int:
        """Get formula complexity (number of parameters)."""
        complexity = 0
        for term in self.terms:
            complexity += term.get_parameter_count()
        return complexity


@dataclass
class FormulaSpec:
    """
    Complete model formula specification.

    For Pradel models, specifies formulas for phi, p, and f parameters.
    """

    phi: ParameterFormula
    p: ParameterFormula
    f: ParameterFormula

    # Optional parameters for extended models
    psi: Optional[ParameterFormula] = None  # Multi-state transitions
    r: Optional[ParameterFormula] = None  # Recovery probability

    # Metadata
    name: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate formula specification after initialization."""
        # Ensure required parameters are present
        required_params = {ParameterType.PHI, ParameterType.P, ParameterType.F}
        provided_params = {self.phi.parameter, self.p.parameter, self.f.parameter}

        if not required_params.issubset(provided_params):
            missing = required_params - provided_params
            raise ModelSpecificationError(
                formula="FormulaSpec",
                suggestions=[
                    f"Missing required parameters: {[p.value for p in missing]}",
                    "Pradel models require phi, p, and f formulas",
                    "Example: FormulaSpec(phi='~1', p='~1', f='~1')",
                ],
            )

    def validate_all_covariates(self, available_covariates: List[str]) -> None:
        """
        Validate all parameter formulas against available covariates.

        Args:
            available_covariates: List of available covariate names
        """
        for param_formula in [self.phi, self.p, self.f]:
            if param_formula:
                param_formula.validate_covariates(available_covariates)

        # Check optional parameters
        if self.psi:
            self.psi.validate_covariates(available_covariates)
        if self.r:
            self.r.validate_covariates(available_covariates)

    def get_total_parameters(self) -> int:
        """Get total number of parameters in the model."""
        total = 0
        for param_formula in [self.phi, self.p, self.f]:
            if param_formula:
                total += param_formula.get_complexity()

        if self.psi:
            total += self.psi.get_complexity()
        if self.r:
            total += self.r.get_complexity()

        return total

    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names in the model."""
        names = [
            self.phi.parameter.value,
            self.p.parameter.value,
            self.f.parameter.value,
        ]
        if self.psi:
            names.append(self.psi.parameter.value)
        if self.r:
            names.append(self.r.parameter.value)
        return names

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "phi": self.phi.formula_string,
            "p": self.p.formula_string,
            "f": self.f.formula_string,
        }

        if self.psi:
            result["psi"] = self.psi.formula_string
        if self.r:
            result["r"] = self.r.formula_string

        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Union[str, Dict]]) -> "FormulaSpec":
        """
        Create FormulaSpec from dictionary.

        Args:
            data: Dictionary with parameter formulas

        Examples:
            {"phi": "~1", "p": "~sex", "f": "~1"}
            {"phi": "~age+sex", "p": "~time", "f": "~1", "name": "Complex model"}
        """
        # Extract required parameters
        phi_formula = data.get("phi", "~1")
        p_formula = data.get("p", "~1")
        f_formula = data.get("f", "~1")

        phi = ParameterFormula(ParameterType.PHI, phi_formula)
        p = ParameterFormula(ParameterType.P, p_formula)
        f = ParameterFormula(ParameterType.F, f_formula)

        # Extract optional parameters
        psi = None
        if "psi" in data:
            psi = ParameterFormula(ParameterType.PSI, data["psi"])

        r = None
        if "r" in data:
            r = ParameterFormula(ParameterType.R, data["r"])

        # Extract metadata
        name = data.get("name")
        description = data.get("description")

        return cls(phi=phi, p=p, f=f, psi=psi, r=r, name=name, description=description)

    def __str__(self) -> str:
        """String representation of the formula specification."""
        parts = [
            f"phi {self.phi.formula_string}",
            f"p {self.p.formula_string}",
            f"f {self.f.formula_string}",
        ]

        if self.psi:
            parts.append(f"psi {self.psi.formula_string}")
        if self.r:
            parts.append(f"r {self.r.formula_string}")

        formula_str = ", ".join(parts)

        if self.name:
            return f"{self.name}: {formula_str}"
        else:
            return formula_str
