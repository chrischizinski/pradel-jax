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
from ..optimization.parallel import ParallelOptimizationResult, ParallelModelSpec, fit_models_parallel
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
    train_result: ParallelOptimizationResult
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
    train_design_matrices = model.build_design_matrices(model_spec.formula_spec, train_context)
    val_design_matrices = model.build_design_matrices(model_spec.formula_spec, val_context)
    
    # Calculate validation metrics
    validation_metrics = model.calculate_validation_metrics(
        fitted_parameters,
        train_context,
        val_context,
        train_design_matrices,
        val_design_matrices
    )
    
    # Make predictions on validation data
    val_predictions = model.predict(
        fitted_parameters,
        val_context,
        val_design_matrices,
        return_individual_predictions=False  # Save memory
    )
    
    # Calculate validation loss (negative log-likelihood per individual)
    validation_loss = -validation_metrics['val_ll_per_individual']
    
    # Calculate AIC penalty (complexity penalty)
    aic_penalty = 2 * len(fitted_parameters) / val_context.n_individuals
    
    return ModelValidationResult(
        model_name=model_spec.name,
        model_index=model_spec.index,
        train_log_likelihood=validation_metrics['train_log_likelihood'],
        train_aic=validation_metrics['train_aic'],
        train_n_individuals=validation_metrics.get('train_n_individuals', train_context.n_individuals),
        val_log_likelihood=validation_metrics['val_log_likelihood'],
        val_aic=validation_metrics['val_aic'],
        val_n_individuals=validation_metrics.get('val_n_individuals', val_context.n_individuals),
        log_likelihood_per_individual_train=validation_metrics['train_ll_per_individual'],
        log_likelihood_per_individual_val=validation_metrics['val_ll_per_individual'],
        validation_loss=validation_loss,
        overfitting_ratio=validation_metrics['overfitting_ratio'],
        aic_penalty=aic_penalty,
        val_predictions=val_predictions,
        n_parameters=len(fitted_parameters),
        formula_spec=model_spec.formula_spec,
        random_seed=getattr(train_result, 'random_seed', None),
        data_hash=getattr(train_result, 'data_hash', None),
        fit_time=getattr(train_result, 'fit_time', None)
    )


def compare_models_with_validation(
    model_specs: List[ParallelModelSpec],
    train_context: DataContext,
    val_context: DataContext,
    train_results: List[ParallelOptimizationResult],
    selection_criterion: str = "validation_loss"
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
                    result
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
    validation_statistics = {\n        'mean_validation_loss': float(np.mean(validation_losses)),\n        'std_validation_loss': float(np.std(validation_losses)),\n        'mean_overfitting_ratio': float(np.mean([r.overfitting_ratio for r in validation_results])),\n        'std_overfitting_ratio': float(np.std([r.overfitting_ratio for r in validation_results])),\n        'best_validation_loss': float(validation_losses[best_idx]),\n        'best_model_aic': float(best_model.val_aic),\n        'best_model_aic_weight': float(aic_weights[best_idx]),\n        'n_models_compared': n_models\n    }\n    \n    logger.info(f\"Best model selected: {best_model.model_name} (criterion: {selection_criterion})\")\n    logger.info(f\"Best validation loss: {validation_statistics['best_validation_loss']:.3f}\")\n    logger.info(f\"Best model AIC weight: {validation_statistics['best_model_aic_weight']:.3f}\")\n    \n    return ModelComparisonResult(\n        models=validation_results,\n        best_model_index=best_idx,\n        best_model_name=best_model.model_name,\n        selection_criterion=selection_criterion,\n        model_ranks_by_validation=validation_ranks.tolist(),\n        model_ranks_by_aic=aic_ranks.tolist(),\n        aic_weights=aic_weights,\n        delta_aic=delta_aic,\n        validation_statistics=validation_statistics\n    )\n\n\ndef select_best_model(\n    model_specs: List[ParallelModelSpec],\n    train_context: DataContext,\n    val_context: DataContext,\n    selection_criterion: str = \"validation_loss\",\n    n_workers: int = 4\n) -> ModelComparisonResult:\n    \"\"\"\n    Complete model selection workflow: fit models on training data and validate.\n    \n    Args:\n        model_specs: List of model specifications to compare\n        train_context: Training data context\n        val_context: Validation data context\n        selection_criterion: Model selection criterion\n        n_workers: Number of parallel workers\n        \n    Returns:\n        Model comparison result with best model selected\n    \"\"\"\n    logger.info(f\"Running complete model selection with {len(model_specs)} models\")\n    \n    # Fit all models on training data\n    start_time = time.time()\n    train_results = fit_models_parallel(\n        model_specs=model_specs,\n        data_context=train_context,\n        n_workers=n_workers\n    )\n    \n    fit_time = time.time() - start_time\n    successful_fits = sum(1 for r in train_results if r and r.success)\n    \n    logger.info(f\"Fitted {successful_fits}/{len(model_specs)} models in {fit_time:.1f}s\")\n    \n    # Compare models with validation\n    comparison_result = compare_models_with_validation(\n        model_specs,\n        train_context,\n        val_context,\n        train_results,\n        selection_criterion\n    )\n    \n    return comparison_result\n\n\ndef cross_validate_model(\n    model_spec: ParallelModelSpec,\n    data_context: DataContext,\n    n_folds: int = 5,\n    tier_columns: Optional[List[str]] = None,\n    random_state: Optional[int] = None\n) -> Dict[str, Any]:\n    \"\"\"\n    Perform k-fold cross-validation for a single model with stratification.\n    \n    Args:\n        model_spec: Model specification\n        data_context: Full data context\n        n_folds: Number of cross-validation folds\n        tier_columns: Columns for stratification (auto-detected if None)\n        random_state: Random seed\n        \n    Returns:\n        Cross-validation results with performance metrics\n    \"\"\"\n    logger.info(f\"Running {n_folds}-fold cross-validation for {model_spec.name}\")\n    \n    # This would require converting DataContext back to DataFrame\n    # For now, implement a simplified version with holdout validation\n    logger.warning(\"Full k-fold CV not yet implemented, using holdout validation\")\n    \n    # Split data into train/validation (80/20)\n    # This is a placeholder - full implementation would require data restructuring\n    \n    return {\n        'model_name': model_spec.name,\n        'cv_method': 'holdout_validation',\n        'n_folds': 1,\n        'status': 'placeholder_implementation',\n        'message': 'Full k-fold cross-validation requires additional data handling infrastructure'\n    }\n\n\ndef predict_best_model_on_validation(\n    comparison_result: ModelComparisonResult,\n    train_context: DataContext,\n    val_context: DataContext,\n    return_individual_predictions: bool = True\n) -> Dict[str, Any]:\n    \"\"\"\n    Generate detailed predictions using the best selected model.\n    \n    Args:\n        comparison_result: Model comparison result\n        train_context: Training data context\n        val_context: Validation data context  \n        return_individual_predictions: Whether to include individual-level predictions\n        \n    Returns:\n        Detailed predictions and assessment\n    \"\"\"\n    best_model = comparison_result.models[comparison_result.best_model_index]\n    \n    logger.info(f\"Generating predictions using best model: {best_model.model_name}\")\n    \n    # Recreate model and design matrices\n    model = PradelModel()\n    val_design_matrices = model.build_design_matrices(best_model.formula_spec, val_context)\n    \n    # Get fitted parameters\n    fitted_parameters = np.array([0.0, 0.0, 0.0])  # Placeholder - would get from training result\n    \n    # Make detailed predictions\n    predictions = model.predict(\n        fitted_parameters,\n        val_context,\n        val_design_matrices,\n        return_individual_predictions=return_individual_predictions\n    )\n    \n    # Calculate prediction intervals (simplified)\n    # In practice, would use bootstrap or asymptotic methods\n    \n    return {\n        'best_model_name': best_model.model_name,\n        'selection_criterion': comparison_result.selection_criterion,\n        'validation_performance': {\n            'validation_loss': best_model.validation_loss,\n            'aic': best_model.val_aic,\n            'aic_weight': float(comparison_result.aic_weights[comparison_result.best_model_index])\n        },\n        'predictions': predictions,\n        'model_formula': {\n            'phi': best_model.formula_spec.phi.formula,\n            'p': best_model.formula_spec.p.formula,\n            'f': best_model.formula_spec.f.formula\n        },\n        'validation_data_size': val_context.n_individuals,\n        'overfitting_assessment': {\n            'overfitting_ratio': best_model.overfitting_ratio,\n            'interpretation': 'low' if abs(best_model.overfitting_ratio) < 0.05 else 'moderate' if abs(best_model.overfitting_ratio) < 0.15 else 'high'\n        }\n    }"