"""
Base classes for capture-recapture models in pradel-jax.

Defines the common interface and infrastructure for all model implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable, Type
from enum import Enum
import jax.numpy as jnp
import numpy as np

from ..formulas.spec import FormulaSpec, ParameterFormula
from ..formulas.design_matrix import DesignMatrixInfo
from ..data.adapters import DataContext
from ..core.exceptions import ModelSpecificationError, OptimizationError
from ..utils.logging import get_logger


logger = get_logger(__name__)


class ModelType(str, Enum):
    """Types of capture-recapture models."""

    PRADEL = "pradel"
    CJS = "cjs"
    POPAN = "popan"
    MULTI_STATE = "multi_state"
    ROBUST_DESIGN = "robust_design"


class OptimizationStatus(str, Enum):
    """Optimization status codes."""

    SUCCESS = "success"
    FAILED = "failed"
    MAX_ITER = "max_iterations"
    NUMERICAL_ERROR = "numerical_error"
    CONVERGENCE_ERROR = "convergence_error"


@dataclass
class ModelResult:
    """Result of fitting a capture-recapture model."""

    # Model identification
    model_type: ModelType
    formula_spec: FormulaSpec
    model_name: Optional[str] = None

    # Optimization results
    status: OptimizationStatus = OptimizationStatus.FAILED
    parameters: Optional[jnp.ndarray] = None
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None

    # Parameter information
    parameter_names: Optional[List[str]] = None
    parameter_se: Optional[jnp.ndarray] = None
    parameter_ci: Optional[Dict[str, jnp.ndarray]] = None

    # Design matrix information
    design_matrices: Optional[Dict[str, DesignMatrixInfo]] = None

    # Optimization metadata
    n_parameters: Optional[int] = None
    n_iterations: Optional[int] = None
    optimizer_used: Optional[str] = None
    convergence_tolerance: Optional[float] = None

    # Diagnostics
    gradient_norm: Optional[float] = None
    hessian_condition: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    # Additional metadata
    fit_time: Optional[float] = None
    data_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived quantities after initialization."""
        if self.parameters is not None and self.n_parameters is None:
            self.n_parameters = len(self.parameters)

        if self.log_likelihood is not None and self.n_parameters is not None:
            # Calculate AIC if not provided
            if self.aic is None:
                self.aic = -2 * self.log_likelihood + 2 * self.n_parameters

            # Calculate BIC if sample size available
            if self.bic is None and "n_individuals" in self.metadata:
                n = self.metadata["n_individuals"]
                self.bic = -2 * self.log_likelihood + np.log(n) * self.n_parameters

    @property
    def success(self) -> bool:
        """Whether the optimization was successful."""
        return self.status == OptimizationStatus.SUCCESS

    @property
    def converged(self) -> bool:
        """Whether the optimization converged (synonym for success)."""
        return self.success

    def get_parameter_dict(self) -> Dict[str, float]:
        """Get parameters as a dictionary with names."""
        if self.parameters is None or self.parameter_names is None:
            return {}

        return dict(zip(self.parameter_names, self.parameters))

    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics for the model fit."""
        stats = {}

        if self.log_likelihood is not None:
            stats["log_likelihood"] = float(self.log_likelihood)

        if self.aic is not None:
            stats["aic"] = float(self.aic)

        if self.bic is not None:
            stats["bic"] = float(self.bic)

        if self.n_parameters is not None:
            stats["n_parameters"] = self.n_parameters

        if self.gradient_norm is not None:
            stats["gradient_norm"] = float(self.gradient_norm)

        return stats

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        result_dict = {
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "status": self.status.value,
            "success": self.success,
        }

        # Add formula information
        if self.formula_spec:
            result_dict["formulas"] = self.formula_spec.to_dict()

        # Add optimization results
        if self.parameters is not None:
            result_dict["parameters"] = self.parameters.tolist()

        result_dict.update(self.get_summary_stats())

        # Add metadata
        result_dict["metadata"] = self.metadata.copy()

        if self.warnings:
            result_dict["warnings"] = self.warnings.copy()

        return result_dict


class CaptureRecaptureModel(ABC):
    """
    Abstract base class for all capture-recapture models.

    Defines the interface that all model implementations must follow.
    """

    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def log_likelihood(
        self,
        parameters: jnp.ndarray,
        data_context: DataContext,
        design_matrices: Dict[str, DesignMatrixInfo],
    ) -> float:
        """
        Calculate log-likelihood for given parameters.

        Args:
            parameters: Model parameters
            data_context: Data and covariates
            design_matrices: Design matrices for each parameter

        Returns:
            Log-likelihood value
        """
        pass

    @abstractmethod
    def get_parameter_bounds(
        self, data_context: DataContext, design_matrices: Dict[str, DesignMatrixInfo]
    ) -> List[tuple]:
        """
        Get parameter bounds for optimization.

        Args:
            data_context: Data and covariates
            design_matrices: Design matrices for each parameter

        Returns:
            List of (lower, upper) bound tuples
        """
        pass

    @abstractmethod
    def get_initial_parameters(
        self, data_context: DataContext, design_matrices: Dict[str, DesignMatrixInfo]
    ) -> jnp.ndarray:
        """
        Get initial parameter values for optimization.

        Args:
            data_context: Data and covariates
            design_matrices: Design matrices for each parameter

        Returns:
            Initial parameter vector
        """
        pass

    @abstractmethod
    def build_design_matrices(
        self, formula_spec: FormulaSpec, data_context: DataContext
    ) -> Dict[str, DesignMatrixInfo]:
        """
        Build design matrices from formula specification.

        Args:
            formula_spec: Model formula specification
            data_context: Data and covariates

        Returns:
            Dictionary of design matrices by parameter name
        """
        pass

    def validate_data(self, data_context: DataContext) -> None:
        """
        Validate data for this model type.

        Args:
            data_context: Data to validate

        Raises:
            ModelSpecificationError: If data is invalid
        """
        # Default validation - can be overridden
        if data_context.n_individuals < 1:
            raise ModelSpecificationError(
                formula="Data validation",
                suggestions=["Data must contain at least 1 individual"],
            )

        if data_context.n_occasions < 2:
            raise ModelSpecificationError(
                formula="Data validation",
                suggestions=["Data must contain at least 2 capture occasions"],
            )

    def validate_formula(
        self, formula_spec: FormulaSpec, data_context: DataContext
    ) -> None:
        """
        Validate formula specification for this model.

        Args:
            formula_spec: Formula specification to validate
            data_context: Available data

        Raises:
            ModelSpecificationError: If formula is invalid
        """
        # Validate against available covariates
        available_covariates = list(data_context.covariates.keys())
        formula_spec.validate_all_covariates(available_covariates)

    def fit(
        self,
        formula_spec: FormulaSpec,
        data_context: DataContext,
        optimizer_config: Optional[Dict[str, Any]] = None,
    ) -> ModelResult:
        """
        Fit the model to data.

        Args:
            formula_spec: Model formula specification
            data_context: Data and covariates
            optimizer_config: Optional optimizer configuration

        Returns:
            ModelResult with fit results
        """
        self.logger.info(f"Fitting {self.model_type.value} model")

        try:
            # Validate inputs
            self.validate_data(data_context)
            self.validate_formula(formula_spec, data_context)

            # Build design matrices
            design_matrices = self.build_design_matrices(formula_spec, data_context)

            # Get optimization components
            initial_params = self.get_initial_parameters(data_context, design_matrices)
            bounds = self.get_parameter_bounds(data_context, design_matrices)

            # Create objective function
            def objective(params):
                return -self.log_likelihood(params, data_context, design_matrices)

            # TODO: Replace with actual optimization framework
            # This is a placeholder that will be replaced with the optimization system
            result = ModelResult(
                model_type=self.model_type,
                formula_spec=formula_spec,
                status=OptimizationStatus.FAILED,
                warnings=["Optimization framework not yet implemented"],
            )

            self.logger.warning("Model fitting not fully implemented yet")
            return result

        except Exception as e:
            self.logger.error(f"Model fitting failed: {e}")
            return ModelResult(
                model_type=self.model_type,
                formula_spec=formula_spec,
                status=OptimizationStatus.FAILED,
                warnings=[f"Fitting error: {str(e)}"],
            )


class ModelRegistry:
    """
    Registry for managing available model implementations.

    Provides a plugin-style system for registering and creating model instances.
    """

    def __init__(self):
        self._models: Dict[ModelType, Type[CaptureRecaptureModel]] = {}
        self.logger = get_logger(self.__class__.__name__)

    def register(
        self, model_type: ModelType, model_class: Type[CaptureRecaptureModel]
    ) -> None:
        """
        Register a model implementation.

        Args:
            model_type: Type of model
            model_class: Model implementation class
        """
        if not issubclass(model_class, CaptureRecaptureModel):
            raise TypeError(f"Model class must inherit from CaptureRecaptureModel")

        self._models[model_type] = model_class
        self.logger.info(
            f"Registered model: {model_type.value} -> {model_class.__name__}"
        )

    def get_model(self, model_type: Union[ModelType, str]) -> CaptureRecaptureModel:
        """
        Get a model instance by type.

        Args:
            model_type: Type of model to create

        Returns:
            Model instance

        Raises:
            ValueError: If model type not registered
        """
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                raise ValueError(f"Unknown model type: {model_type}")

        if model_type not in self._models:
            raise ValueError(
                f"Model type '{model_type.value}' not registered. "
                f"Available: {list(self._models.keys())}"
            )

        model_class = self._models[model_type]
        return model_class(model_type)

    def list_models(self) -> List[ModelType]:
        """Get list of available model types."""
        return list(self._models.keys())

    def is_registered(self, model_type: Union[ModelType, str]) -> bool:
        """Check if a model type is registered."""
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError:
                return False

        return model_type in self._models


# Global model registry instance
_registry = ModelRegistry()


def register_model(
    model_type: ModelType, model_class: Type[CaptureRecaptureModel]
) -> None:
    """Register a model with the global registry."""
    _registry.register(model_type, model_class)


def get_model(model_type: Union[ModelType, str]) -> CaptureRecaptureModel:
    """Get a model instance from the global registry."""
    return _registry.get_model(model_type)


def list_available_models() -> List[ModelType]:
    """List available model types in the global registry."""
    return _registry.list_models()
