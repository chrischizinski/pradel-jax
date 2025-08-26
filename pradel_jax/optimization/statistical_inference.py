"""
Statistical inference utilities for pradel-jax optimization results.

Provides parameter naming, statistical computations, and model comparison tools.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

from ..formulas.spec import FormulaSpec, ParameterType
from ..data.adapters import DataContext


def generate_parameter_names(
    formula_spec: FormulaSpec, data_context: DataContext
) -> List[str]:
    """
    Generate parameter names from formula specification.

    Creates interpretable names that map to the parameter vector, following
    the same ordering used in design matrix construction.

    Args:
        formula_spec: Formula specification with parameter formulas
        data_context: Data context with covariate information

    Returns:
        List of parameter names in order corresponding to parameter vector

    Example:
        For phi="~1 + sex", p="~1 + sex", f="~1":
        Returns: ["phi_intercept", "phi_sex", "p_intercept", "p_sex", "f_intercept"]
    """
    parameter_names = []

    # Process each parameter type in consistent order
    # FormulaSpec has phi, p, f attributes, not a formulas dict
    formulas = {
        ParameterType.PHI: formula_spec.phi,
        ParameterType.P: formula_spec.p,
        ParameterType.F: formula_spec.f,
    }

    # Add optional parameters if present
    if hasattr(formula_spec, "psi") and formula_spec.psi is not None:
        formulas[ParameterType.PSI] = formula_spec.psi
    if hasattr(formula_spec, "r") and formula_spec.r is not None:
        formulas[ParameterType.R] = formula_spec.r

    parameter_order = [
        ParameterType.PHI,
        ParameterType.P,
        ParameterType.F,
        ParameterType.PSI,
        ParameterType.R,
    ]

    for param_type in parameter_order:
        if param_type not in formulas:
            continue

        param_formula = formulas[param_type]
        param_prefix = param_type.value

        # Generate names based on formula terms
        names = _generate_names_for_parameter(param_formula, param_prefix, data_context)
        parameter_names.extend(names)

    return parameter_names


def _generate_names_for_parameter(
    param_formula, param_prefix: str, data_context: DataContext
) -> List[str]:
    """Generate names for a single parameter type."""
    names = []

    # Handle intercept
    if param_formula.has_intercept:
        names.append(f"{param_prefix}_intercept")

    # Process each term in the formula
    for term in param_formula.terms:
        if term.is_intercept():
            continue  # Already handled

        # Get variable names from the term
        var_names = term.get_variable_names()

        if len(var_names) == 1:
            # Simple term: phi_sex, phi_age, etc.
            var_name = list(var_names)[0]
            names.append(f"{param_prefix}_{var_name}")

        elif len(var_names) > 1:
            # Interaction term: phi_sex:age, etc.
            interaction_name = ":".join(sorted(var_names))
            names.append(f"{param_prefix}_{interaction_name}")

        else:
            # Function or complex term - use generic naming
            names.append(f"{param_prefix}_term_{len(names)}")

    return names


@dataclass
class ModelComparisonResult:
    """Results of model comparison analysis."""

    models: Dict[str, Any]  # Model names to results
    aic_ranking: List[Tuple[str, float]]  # (name, AIC) sorted by AIC
    bic_ranking: List[Tuple[str, float]]  # (name, BIC) sorted by BIC
    delta_aic: Dict[str, float]  # AIC differences from best model
    delta_bic: Dict[str, float]  # BIC differences from best model
    best_aic_model: str
    best_bic_model: str


def compare_models(model_results: Dict[str, Any]) -> ModelComparisonResult:
    """
    Compare multiple models using information criteria.

    Args:
        model_results: Dictionary of model_name -> OptimizationResult

    Returns:
        ModelComparisonResult with rankings and comparisons
    """
    aic_values = {}
    bic_values = {}

    # Extract AIC/BIC values
    for name, result in model_results.items():
        if hasattr(result, "result") and hasattr(result.result, "aic"):
            aic_values[name] = result.result.aic
        elif hasattr(result, "aic"):
            aic_values[name] = result.aic

        if hasattr(result, "result") and hasattr(result.result, "bic"):
            bic_values[name] = result.result.bic
        elif hasattr(result, "bic"):
            bic_values[name] = result.bic

    # Create rankings
    aic_ranking = sorted(aic_values.items(), key=lambda x: x[1])
    bic_ranking = sorted(bic_values.items(), key=lambda x: x[1])

    # Calculate deltas from best model
    best_aic = min(aic_values.values()) if aic_values else 0
    best_bic = min(bic_values.values()) if bic_values else 0

    delta_aic = {name: aic - best_aic for name, aic in aic_values.items()}
    delta_bic = {name: bic - best_bic for name, bic in bic_values.items()}

    return ModelComparisonResult(
        models=model_results,
        aic_ranking=aic_ranking,
        bic_ranking=bic_ranking,
        delta_aic=delta_aic,
        delta_bic=delta_bic,
        best_aic_model=aic_ranking[0][0] if aic_ranking else "",
        best_bic_model=bic_ranking[0][0] if bic_ranking else "",
    )


def print_parameter_summary(result) -> None:
    """
    Print a formatted parameter summary table.

    Args:
        result: OptimizationResult with statistical inference
    """
    if not hasattr(result, "get_parameter_summary"):
        print("No parameter summary available")
        return

    summary = result.get_parameter_summary()
    if summary is None:
        print("No parameter summary available")
        return

    # Print header
    print("\nParameter Estimates:")
    print("-" * 70)
    print(
        f"{'Parameter':<15} {'Estimate':<12} {'Std.Error':<12} {'z-score':<10} {'95% CI':<20}"
    )
    print("-" * 70)

    # Print each parameter
    for name, info in summary.items():
        estimate = info.get("estimate", 0)
        std_error = info.get("std_error", None)
        z_score = info.get("z_score", None)
        ci_lower = info.get("ci_lower", None)
        ci_upper = info.get("ci_upper", None)

        # Format values
        est_str = f"{estimate:8.4f}"
        se_str = f"{std_error:8.4f}" if std_error is not None else "    --  "
        z_str = f"{z_score:6.2f}" if z_score is not None else "  --  "

        if ci_lower is not None and ci_upper is not None:
            ci_str = f"({ci_lower:6.3f}, {ci_upper:6.3f})"
        else:
            ci_str = "      --       "

        print(f"{name:<15} {est_str:<12} {se_str:<12} {z_str:<10} {ci_str:<20}")

    # Print model fit statistics
    if hasattr(result, "aic") and result.aic is not None:
        print("-" * 70)
        print(f"Log-likelihood: {result.log_likelihood:.4f}")
        print(f"AIC: {result.aic:.4f}")
        if hasattr(result, "bic") and result.bic is not None:
            print(f"BIC: {result.bic:.4f}")


def print_model_comparison(comparison: ModelComparisonResult) -> None:
    """
    Print formatted model comparison table.

    Args:
        comparison: ModelComparisonResult from compare_models()
    """
    print("\nModel Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'AIC':<12} {'ΔAIC':<10} {'BIC':<12} {'ΔBIC':<10}")
    print("-" * 60)

    # Print models sorted by AIC
    for model_name, aic in comparison.aic_ranking:
        delta_aic = comparison.delta_aic[model_name]

        # Get BIC info if available
        bic_info = ""
        delta_bic_info = ""
        for bic_name, bic in comparison.bic_ranking:
            if bic_name == model_name:
                bic_info = f"{bic:8.2f}"
                delta_bic_info = f"{comparison.delta_bic[model_name]:6.2f}"
                break

        print(
            f"{model_name:<20} {aic:8.2f}    {delta_aic:6.2f}    {bic_info:<12} {delta_bic_info:<10}"
        )

    print("-" * 60)
    print(f"Best AIC model: {comparison.best_aic_model}")
    if comparison.best_bic_model != comparison.best_aic_model:
        print(f"Best BIC model: {comparison.best_bic_model}")
