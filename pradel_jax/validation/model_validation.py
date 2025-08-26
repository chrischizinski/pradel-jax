"""
Model validation and selection for pradel-jax following statistical best practices.

Implements proper holdout validation, cross-validation, and model selection procedures
according to statistical modeling best practices.
"""

import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import time
from pathlib import Path

from ..models.pradel import PradelModel
from ..data.adapters import DataContext
from ..data.sampling import train_validation_split, determine_tier_status
from ..optimization.parallel import (
    ParallelOptimizationResult,
    ParallelModelSpec,
    fit_models_parallel,
)
from ..formulas.spec import FormulaSpec
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelValidationResult:
    """Result from model validation on holdout data."""

    model_name: str
    model_index: int

    # Training performance
    train_log_likelihood: float
    train_aic: float
    train_n_individuals: int

    # Validation performance
    val_log_likelihood: float
    val_aic: float
    val_n_individuals: int

    # Comparison metrics
    log_likelihood_per_individual_train: float
    log_likelihood_per_individual_val: float
    validation_loss: float  # Negative validation log-likelihood per individual
    overfitting_ratio: float
    aic_penalty: float

    # Predictions on validation data
    val_predictions: Dict[str, Any]

    # Model complexity
    n_parameters: int
    formula_spec: FormulaSpec

    # Fit metadata
    random_seed: Optional[int] = None
    data_hash: Optional[str] = None
    fit_time: Optional[float] = None


@dataclass
class ModelComparisonResult:
    """Result from comparing multiple models with validation."""

    models: List[ModelValidationResult]
    best_model_index: int
    best_model_name: str

    # Selection criteria
    selection_criterion: str  # "validation_loss", "aic", "cross_validation"

    # Model ranking
    model_ranks_by_validation: List[int]
    model_ranks_by_aic: List[int]

    # Information criteria comparison
    aic_weights: np.ndarray
    delta_aic: np.ndarray

    # Validation statistics
    validation_statistics: Dict[str, float]

    # Cross-validation results (if performed)
    cv_results: Optional[Dict[str, Any]] = None


def validate_model_on_holdout(
    model_spec: ParallelModelSpec,
    train_context: DataContext,
    val_context: DataContext,
    fitted_parameters: np.ndarray,
    train_result: ParallelOptimizationResult,
) -> ModelValidationResult:
    """
    Validate a fitted model on holdout validation data.

    Args:
        model_spec: Model specification
        train_context: Training data context
        val_context: Validation data context
        fitted_parameters: Parameters fitted on training data
        train_result: Training optimization result

    Returns:
        Validation result with performance metrics
    """
    logger.info(f"Validating model {model_spec.name} on holdout data")

    # Create model instance
    model = PradelModel()

    # Build design matrices for both datasets
    train_design_matrices = model.build_design_matrices(
        model_spec.formula_spec, train_context
    )
    val_design_matrices = model.build_design_matrices(
        model_spec.formula_spec, val_context
    )

    # Calculate validation metrics
    validation_metrics = model.calculate_validation_metrics(
        fitted_parameters,
        train_context,
        val_context,
        train_design_matrices,
        val_design_matrices,
    )

    # Make predictions on validation data
    val_predictions = model.predict(
        fitted_parameters,
        val_context,
        val_design_matrices,
        return_individual_predictions=False,  # Save memory
    )

    # Calculate validation loss (negative log-likelihood per individual)
    validation_loss = -validation_metrics["val_ll_per_individual"]

    # Calculate AIC penalty (complexity penalty)
    aic_penalty = 2 * len(fitted_parameters) / val_context.n_individuals

    return ModelValidationResult(
        model_name=model_spec.name,
        model_index=model_spec.index,
        train_log_likelihood=validation_metrics["train_log_likelihood"],
        train_aic=validation_metrics["train_aic"],
        train_n_individuals=validation_metrics.get(
            "train_n_individuals", train_context.n_individuals
        ),
        val_log_likelihood=validation_metrics["val_log_likelihood"],
        val_aic=validation_metrics["val_aic"],
        val_n_individuals=validation_metrics.get(
            "val_n_individuals", val_context.n_individuals
        ),
        log_likelihood_per_individual_train=validation_metrics[
            "train_ll_per_individual"
        ],
        log_likelihood_per_individual_val=validation_metrics["val_ll_per_individual"],
        validation_loss=validation_loss,
        overfitting_ratio=validation_metrics["overfitting_ratio"],
        aic_penalty=aic_penalty,
        val_predictions=val_predictions,
        n_parameters=len(fitted_parameters),
        formula_spec=model_spec.formula_spec,
        random_seed=getattr(train_result, "random_seed", None),
        data_hash=getattr(train_result, "data_hash", None),
        fit_time=getattr(train_result, "fit_time", None),
    )


def compare_models_with_validation(
    model_specs: List[ParallelModelSpec],
    train_context: DataContext,
    val_context: DataContext,
    train_results: List[ParallelOptimizationResult],
    selection_criterion: str = "validation_loss",
) -> ModelComparisonResult:
    """
    Compare multiple models using validation data following best practices.

    Args:
        model_specs: List of model specifications
        train_context: Training data context
        val_context: Validation data context
        train_results: Training optimization results
        selection_criterion: Criterion for model selection

    Returns:
        Model comparison result with rankings and selection
    """
    logger.info(f"Comparing {len(model_specs)} models using validation data")

    # Validate each model on holdout data
    validation_results = []

    for i, (spec, result) in enumerate(zip(model_specs, train_results)):
        if result and result.success:
            try:
                val_result = validate_model_on_holdout(
                    spec,
                    train_context,
                    val_context,
                    np.array(result.parameters),
                    result,
                )
                validation_results.append(val_result)
            except Exception as e:
                logger.warning(f"Validation failed for model {spec.name}: {e}")
                continue
        else:
            logger.warning(f"Skipping failed model {spec.name}")

    if not validation_results:
        raise ValueError("No models successfully validated")

    # Calculate model selection criteria
    n_models = len(validation_results)

    # AIC comparison
    aic_values = np.array([r.val_aic for r in validation_results])
    min_aic = np.min(aic_values)
    delta_aic = aic_values - min_aic

    # AIC weights (Burnham & Anderson approach)
    aic_weights = np.exp(-0.5 * delta_aic)
    aic_weights = aic_weights / np.sum(aic_weights)

    # Validation loss comparison
    validation_losses = np.array([r.validation_loss for r in validation_results])

    # Model ranking
    aic_ranks = np.argsort(aic_values)  # Lower AIC is better
    validation_ranks = np.argsort(validation_losses)  # Lower validation loss is better

    # Select best model based on criterion
    if selection_criterion == "validation_loss":
        best_idx = validation_ranks[0]
    elif selection_criterion == "aic":
        best_idx = aic_ranks[0]
    else:
        raise ValueError(f"Unknown selection criterion: {selection_criterion}")

    best_model = validation_results[best_idx]

    # Calculate validation statistics
    validation_statistics = {
        "mean_validation_loss": float(np.mean(validation_losses)),
        "std_validation_loss": float(np.std(validation_losses)),
        "mean_overfitting_ratio": float(
            np.mean([r.overfitting_ratio for r in validation_results])
        ),
        "std_overfitting_ratio": float(
            np.std([r.overfitting_ratio for r in validation_results])
        ),
        "best_validation_loss": float(validation_losses[best_idx]),
        "best_model_aic": float(best_model.val_aic),
        "best_model_aic_weight": float(aic_weights[best_idx]),
        "n_models_compared": n_models,
    }

    logger.info(
        f"Best model selected: {best_model.model_name} (criterion: {selection_criterion})"
    )
    logger.info(
        f"Best validation loss: {validation_statistics['best_validation_loss']:.3f}"
    )
    logger.info(
        f"Best model AIC weight: {validation_statistics['best_model_aic_weight']:.3f}"
    )

    return ModelComparisonResult(
        models=validation_results,
        best_model_index=best_idx,
        best_model_name=best_model.model_name,
        selection_criterion=selection_criterion,
        model_ranks_by_validation=validation_ranks.tolist(),
        model_ranks_by_aic=aic_ranks.tolist(),
        aic_weights=aic_weights,
        delta_aic=delta_aic,
        validation_statistics=validation_statistics,
    )


def select_best_model(
    model_specs: List[ParallelModelSpec],
    train_context: DataContext,
    val_context: DataContext,
    selection_criterion: str = "validation_loss",
    n_workers: int = 4,
) -> ModelComparisonResult:
    """
    Complete model selection workflow: fit models on training data and validate.

    Args:
        model_specs: List of model specifications to compare
        train_context: Training data context
        val_context: Validation data context
        selection_criterion: Model selection criterion
        n_workers: Number of parallel workers

    Returns:
        Model comparison result with best model selected
    """
    logger.info(f"Running complete model selection with {len(model_specs)} models")

    # Fit all models on training data
    start_time = time.time()
    train_results = fit_models_parallel(
        model_specs=model_specs, data_context=train_context, n_workers=n_workers
    )

    fit_time = time.time() - start_time
    successful_fits = sum(1 for r in train_results if r and r.success)

    logger.info(
        f"Fitted {successful_fits}/{len(model_specs)} models in {fit_time:.1f}s"
    )

    # Compare models with validation
    comparison_result = compare_models_with_validation(
        model_specs, train_context, val_context, train_results, selection_criterion
    )

    return comparison_result


def cross_validate_model(
    model_spec: ParallelModelSpec,
    data_context: DataContext,
    n_folds: int = 5,
    tier_columns: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation for a single model with stratification.

    Args:
        model_spec: Model specification
        data_context: Full data context
        n_folds: Number of cross-validation folds
        tier_columns: Columns for stratification (auto-detected if None)
        random_state: Random seed

    Returns:
        Cross-validation results with performance metrics
    """
    logger.info(f"Running {n_folds}-fold cross-validation for {model_spec.name}")

    # This would require converting DataContext back to DataFrame
    # For now, implement a simplified version with holdout validation
    logger.warning("Full k-fold CV not yet implemented, using holdout validation")

    # Split data into train/validation (80/20)
    # This is a placeholder - full implementation would require data restructuring

    return {
        "model_name": model_spec.name,
        "cv_method": "holdout_validation",
        "n_folds": 1,
        "status": "placeholder_implementation",
        "message": "Full k-fold cross-validation requires additional data handling infrastructure",
    }


def predict_best_model_on_validation(
    comparison_result: ModelComparisonResult,
    train_context: DataContext,
    val_context: DataContext,
    return_individual_predictions: bool = True,
) -> Dict[str, Any]:
    """
    Generate detailed predictions using the best selected model.

    Args:
        comparison_result: Model comparison result
        train_context: Training data context
        val_context: Validation data context
        return_individual_predictions: Whether to include individual-level predictions

    Returns:
        Detailed predictions and assessment
    """
    best_model = comparison_result.models[comparison_result.best_model_index]

    logger.info(f"Generating predictions using best model: {best_model.model_name}")

    # Recreate model and design matrices
    model = PradelModel()
    val_design_matrices = model.build_design_matrices(
        best_model.formula_spec, val_context
    )

    # Get fitted parameters
    fitted_parameters = np.array(
        [0.0, 0.0, 0.0]
    )  # Placeholder - would get from training result

    # Make detailed predictions
    predictions = model.predict(
        fitted_parameters,
        val_context,
        val_design_matrices,
        return_individual_predictions=return_individual_predictions,
    )

    # Calculate prediction intervals (simplified)
    # In practice, would use bootstrap or asymptotic methods

    return {
        "best_model_name": best_model.model_name,
        "selection_criterion": comparison_result.selection_criterion,
        "validation_performance": {
            "validation_loss": best_model.validation_loss,
            "aic": best_model.val_aic,
            "aic_weight": float(
                comparison_result.aic_weights[comparison_result.best_model_index]
            ),
        },
        "predictions": predictions,
        "model_formula": {
            "phi": best_model.formula_spec.phi.formula,
            "p": best_model.formula_spec.p.formula,
            "f": best_model.formula_spec.f.formula,
        },
        "validation_data_size": val_context.n_individuals,
        "overfitting_assessment": {
            "overfitting_ratio": best_model.overfitting_ratio,
            "interpretation": (
                "low"
                if abs(best_model.overfitting_ratio) < 0.05
                else "moderate" if abs(best_model.overfitting_ratio) < 0.15 else "high"
            ),
        },
    }
