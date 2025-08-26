"""
Main API functions for pradel-jax.

High-level user interface for model fitting and analysis.
"""

from typing import Optional, Union, Dict, Any
from pathlib import Path

from ..data.adapters import load_data, DataContext
from ..formulas import FormulaSpec, ParameterFormula, create_simple_spec, ParameterType
from ..models import PradelModel, ModelResult, CaptureRecaptureModel
from ..models.base import OptimizationStatus, ModelType
from ..optimization.orchestrator import optimize_model
from ..optimization.strategy import OptimizationStrategy
from ..utils.logging import get_logger
from ..core.exceptions import ModelSpecificationError, OptimizationError

logger = get_logger(__name__)


def create_formula_spec(
    phi: Optional[str] = None,
    p: Optional[str] = None, 
    f: Optional[str] = None,
    **kwargs
) -> FormulaSpec:
    """
    Create a formula specification for Pradel model parameters.
    
    Args:
        phi: Formula for survival probability (default: "~1")
        p: Formula for detection probability (default: "~1") 
        f: Formula for recruitment rate (default: "~1")
        **kwargs: Additional parameter formulas
        
    Returns:
        FormulaSpec object ready for model fitting
        
    Examples:
        >>> # Constant parameters
        >>> spec = create_formula_spec()
        
        >>> # Survival and detection with sex effect
        >>> spec = create_formula_spec(
        ...     phi="~1 + sex",
        ...     p="~1 + sex", 
        ...     f="~1"
        ... )
    """
    # Set defaults
    phi = phi or "~1"
    p = p or "~1"
    f = f or "~1"
    
    # Create parameter formulas with correct constructor signature
    phi_formula = ParameterFormula(parameter=ParameterType.PHI, formula_string=phi)
    p_formula = ParameterFormula(parameter=ParameterType.P, formula_string=p)
    f_formula = ParameterFormula(parameter=ParameterType.F, formula_string=f)
    
    # Handle additional parameters from kwargs
    optional_formulas = {}
    for param_name, formula_str in kwargs.items():
        try:
            param_type = ParameterType[param_name.upper()]
            optional_formulas[param_name] = ParameterFormula(
                parameter=param_type, 
                formula_string=formula_str
            )
        except KeyError:
            logger.warning(f"Unknown parameter type: {param_name}")
    
    # Create FormulaSpec with correct structure
    return FormulaSpec(
        phi=phi_formula,
        p=p_formula,
        f=f_formula,
        psi=optional_formulas.get('psi'),
        r=optional_formulas.get('r'),
        name=kwargs.get('name'),
        description=kwargs.get('description')
    )


def fit_model(
    model: Optional[CaptureRecaptureModel] = None,
    formula: Optional[FormulaSpec] = None,
    data: Optional[Union[DataContext, str, Path]] = None,
    strategy: Optional[Union[str, OptimizationStrategy]] = None,
    **kwargs
) -> ModelResult:
    """
    Fit a capture-recapture model to data.
    
    Args:
        model: Model instance (default: PradelModel())
        formula: Formula specification (default: constant parameters)
        data: Data context or file path
        strategy: Optimization strategy ("auto", "lbfgs", "slsqp", "adam", "multi_start")
        **kwargs: Additional optimization parameters
        
    Returns:
        ModelResult with fitted parameters and diagnostics
        
    Examples:
        >>> # Simple fitting with defaults
        >>> result = fit_model(data="data/dipper_dataset.csv")
        
        >>> # Custom model specification
        >>> formula = create_formula_spec(
        ...     phi="~1 + sex",
        ...     p="~1 + sex"
        ... )
        >>> result = fit_model(
        ...     formula=formula,
        ...     data=data_context,
        ...     strategy="multi_start"
        ... )
    """
    # Set defaults
    if model is None:
        model = PradelModel()
        
    if formula is None:
        formula = create_formula_spec()
        logger.info("Using default constant parameter formulas")
        
    # Load data if needed
    if isinstance(data, (str, Path)):
        data = load_data(data)
    elif data is None:
        raise ModelSpecificationError(
            specific_issue="No data provided",
            suggestions=[
                "Provide a DataContext object",
                "Provide a file path to load data",
                "Use load_data() to create a DataContext first"
            ]
        )
        
    # Parse strategy
    if isinstance(strategy, str):
        if strategy.lower() == "auto":
            # Auto strategy selection - we'll use the orchestrator's automatic selection
            strategy = None  # Will trigger auto selection
        else:
            try:
                strategy = OptimizationStrategy(strategy.lower())
            except ValueError:
                valid_strategies = [s.value for s in OptimizationStrategy] + ["auto"]
                raise ModelSpecificationError(
                    specific_issue=f"Invalid optimization strategy: {strategy}",
                    suggestions=[
                        f"Valid strategies: {valid_strategies}",
                        "Use 'auto' for automatic strategy selection"
                    ]
                )
    elif strategy is None:
        # Default to automatic selection
        strategy = None
        
    strategy_name = strategy.value if strategy else "automatic"
    logger.info(f"Fitting {model.__class__.__name__} with {strategy_name} optimization")
    
    try:
        # Build design matrices
        design_matrices = model.build_design_matrices(formula, data)
        
        def objective_function(params):
            return -model.log_likelihood(params, data, design_matrices)
            
        # Get initial parameters and bounds
        initial_params = model.get_initial_parameters(data, design_matrices) 
        bounds = model.get_parameter_bounds(data, design_matrices)
        
        # Run optimization using the orchestrator
        optimization_result = optimize_model(
            objective_function=objective_function,
            initial_parameters=initial_params,
            context=data,
            bounds=bounds,
            preferred_strategy=strategy,
            **kwargs
        )
        
        if not optimization_result.success:
            raise OptimizationError(
                reason=optimization_result.result.message,
                optimizer=optimization_result.strategy_used
            )
            
        # Extract results from OptimizationResponse structure
        opt_result = optimization_result.result  
        
        # Map optimization result to ModelResult fields
        status = OptimizationStatus.SUCCESS if opt_result.success else OptimizationStatus.FAILED
        
        result = ModelResult(
            model_type=ModelType.PRADEL,
            formula_spec=formula,
            model_name=model.__class__.__name__,
            status=status,
            parameters=opt_result.x,
            log_likelihood=-opt_result.fun,  # Convert back to log-likelihood from objective
            design_matrices=design_matrices,
            n_iterations=getattr(opt_result, 'nit', None),
            optimizer_used=optimization_result.strategy_used,
            fit_time=optimization_result.total_time,
            metadata={
                'optimization_message': opt_result.message,
                'n_function_evaluations': opt_result.nfev,
                'data_summary': {
                    'n_individuals': data.n_individuals,
                    'n_occasions': data.n_occasions,
                    'n_covariates': len(data.covariates)
                }
            }
        )
        
        logger.info(f"Model fitting completed successfully in {result.fit_time:.4f}s")
        return result
        
    except Exception as e:
        if isinstance(e, (ModelSpecificationError, OptimizationError)):
            raise
        else:
            raise OptimizationError(
                reason=f"Model fitting failed: {str(e)}",
                optimizer="unknown"
            ) from e


def fit_models(*args, **kwargs):
    """Placeholder function for fitting multiple models (backwards compatibility)."""
    logger.warning("fit_models() is deprecated, use fit_model() instead")
    return fit_model(*args, **kwargs)


def select_best_model(*args, **kwargs):
    """Placeholder function for model selection."""
    raise NotImplementedError("Model selection not yet implemented in redesign")


def validate_against_rmark(*args, **kwargs):
    """Placeholder function for RMark validation."""
    raise NotImplementedError("RMark validation not yet implemented in redesign")


__all__ = [
    "fit_model", 
    "create_formula_spec",
    "fit_models", 
    "select_best_model", 
    "validate_against_rmark", 
    "load_data"
]
