"""
Pradel model implementation for pradel-jax.

Implements the Pradel capture-recapture model with JAX-based likelihood computation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .base import CaptureRecaptureModel, ModelType, ModelResult
from ..formulas.spec import FormulaSpec, ParameterFormula, ParameterType
from ..formulas.design_matrix import DesignMatrixInfo, build_design_matrix
from ..data.adapters import DataContext
from ..core.exceptions import ModelSpecificationError
from ..utils.logging import get_logger


logger = get_logger(__name__)


@jax.jit
def logit(x: jnp.ndarray) -> jnp.ndarray:
    """Logit link function."""
    return jnp.log(x / (1 - x))


@jax.jit
def inv_logit(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse logit (sigmoid) function."""
    return jax.nn.sigmoid(x)


@jax.jit
def log_link(x: jnp.ndarray) -> jnp.ndarray:
    """Log link function."""
    return jnp.log(x)


@jax.jit
def exp_link(x: jnp.ndarray) -> jnp.ndarray:
    """Exponential (inverse log) function."""
    return jnp.exp(x)


class PradelModel(CaptureRecaptureModel):
    """
    Pradel capture-recapture model implementation.
    
    The Pradel model estimates:
    - φ (phi): Apparent survival probability
    - p: Detection/capture probability  
    - f: Per-capita recruitment rate
    
    Uses logit link for φ and p (bounded 0-1), log link for f (positive).
    """
    
    def __init__(self, model_type: ModelType = ModelType.PRADEL):
        super().__init__(model_type)
        self.parameter_order = ["phi", "p", "f"]
    
    def build_design_matrices(
        self,
        formula_spec: FormulaSpec,
        data_context: DataContext
    ) -> Dict[str, DesignMatrixInfo]:
        """Build design matrices for Pradel model parameters."""
        self.logger.debug("Building design matrices for Pradel model")
        
        # Get number of occasions for time-varying parameters
        n_occasions = data_context.n_occasions
        
        design_matrices = {}
        
        # Build design matrix for each parameter
        for param_name in self.parameter_order:
            param_formula = getattr(formula_spec, param_name)
            if param_formula is None:
                raise ModelSpecificationError(
                    formula=f"Missing {param_name} formula in specification",
                    suggestions=[
                        f"Pradel models require {param_name} formula",
                        "Provide all three formulas: phi, p, f",
                    ]
                )
            
            # For time-varying parameters, we might need different handling
            # For now, treat all as individual-level
            design_info = build_design_matrix(param_formula, data_context)
            design_matrices[param_name] = design_info
            
            self.logger.debug(
                f"Built {param_name} design matrix: {design_info.matrix.shape} "
                f"({design_info.parameter_count} parameters)"
            )
        
        return design_matrices
    
    def get_parameter_bounds(
        self,
        data_context: DataContext,
        design_matrices: Dict[str, DesignMatrixInfo]
    ) -> List[tuple]:
        """Get parameter bounds for optimization."""
        bounds = []
        
        for param_name in self.parameter_order:
            design_info = design_matrices[param_name]
            n_params = design_info.parameter_count
            
            if param_name in ["phi", "p"]:
                # Logit-scale parameters: reasonable bounds on logit scale
                param_bounds = [(-10.0, 10.0)] * n_params
            elif param_name == "f":
                # Log-scale parameters: reasonable bounds on log scale
                param_bounds = [(-10.0, 5.0)] * n_params  # exp(-10) to exp(5)
            else:
                # Default bounds
                param_bounds = [(-10.0, 10.0)] * n_params
            
            bounds.extend(param_bounds)
        
        return bounds
    
    def get_initial_parameters(
        self,
        data_context: DataContext,
        design_matrices: Dict[str, DesignMatrixInfo]
    ) -> jnp.ndarray:
        """Get initial parameter values based on data."""
        initial_params = []
        
        # Calculate simple empirical estimates for initialization
        capture_matrix = data_context.capture_matrix
        
        # Empirical capture probability (proportion of 1s)
        p_empirical = jnp.mean(capture_matrix)
        p_logit = logit(jnp.clip(p_empirical, 0.01, 0.99))
        
        # Empirical survival (naive estimate from consecutive captures)
        phi_empirical = self._estimate_empirical_survival(capture_matrix)
        phi_logit = logit(jnp.clip(phi_empirical, 0.01, 0.99))
        
        # Empirical recruitment (simple estimate)
        f_empirical = self._estimate_empirical_recruitment(capture_matrix)
        f_log = log_link(jnp.maximum(f_empirical, 0.01))
        
        self.logger.debug(
            f"Empirical estimates: p={float(p_empirical):.3f}, phi={float(phi_empirical):.3f}, f={float(f_empirical):.3f}"
        )
        
        # Initialize parameters for each design matrix
        for param_name in self.parameter_order:
            design_info = design_matrices[param_name]
            n_params = design_info.parameter_count
            
            if param_name == "phi":
                if design_info.has_intercept:
                    # Intercept gets empirical estimate, others get small values
                    params = jnp.concatenate([
                        jnp.array([phi_logit]),
                        jnp.zeros(n_params - 1) * 0.1
                    ])
                else:
                    params = jnp.ones(n_params) * phi_logit
            
            elif param_name == "p":
                if design_info.has_intercept:
                    params = jnp.concatenate([
                        jnp.array([p_logit]),
                        jnp.zeros(n_params - 1) * 0.1
                    ])
                else:
                    params = jnp.ones(n_params) * p_logit
            
            elif param_name == "f":
                if design_info.has_intercept:
                    params = jnp.concatenate([
                        jnp.array([f_log]),
                        jnp.zeros(n_params - 1) * 0.1
                    ])
                else:
                    params = jnp.ones(n_params) * f_log
            
            initial_params.append(params)
        
        # Concatenate all parameters
        return jnp.concatenate(initial_params)
    
    def _estimate_empirical_survival(self, capture_matrix: jnp.ndarray) -> jnp.ndarray:
        """Estimate empirical survival probability using JAX-compatible operations."""
        # Simple approach: look at consecutive capture pairs
        n_individuals, n_occasions = capture_matrix.shape
        
        # Use jnp.where instead of if statement
        default_survival = 0.8
        
        # Count individuals captured in consecutive occasions using vectorized operations
        alive_and_captured = 0.0
        total_comparisons = 0.0
        
        for t in range(n_occasions - 1):
            captured_t = capture_matrix[:, t]
            captured_t1 = capture_matrix[:, t + 1]
            
            # Individuals captured at t
            n_captured_t = jnp.sum(captured_t)
            
            # Of those, how many were captured at t+1
            n_recaptured = jnp.sum(captured_t * captured_t1)
            
            # Use jnp.where to avoid control flow
            alive_and_captured += jnp.where(n_captured_t > 0, n_recaptured, 0.0)
            total_comparisons += jnp.where(n_captured_t > 0, n_captured_t, 0.0)
        
        # Calculate survival rate with JAX-compatible conditional
        survival_rate = jnp.where(
            total_comparisons > 0,
            alive_and_captured / total_comparisons,
            default_survival
        )
        
        return survival_rate
    
    def _estimate_empirical_recruitment(self, capture_matrix: jnp.ndarray) -> jnp.ndarray:
        """Estimate empirical recruitment rate using JAX-compatible operations."""
        # Simple approach: look at first captures as proxy for recruitment
        n_individuals, n_occasions = capture_matrix.shape
        
        default_recruitment = 0.1
        
        # Count new individuals in each period (first capture) using vectorized operations
        new_individuals = jnp.zeros(n_occasions - 1)
        
        for t in range(1, n_occasions):  # Start from second occasion
            # Individuals captured at t but not before
            captured_before = jnp.any(capture_matrix[:, :t], axis=1)
            captured_at_t = capture_matrix[:, t]
            
            new_at_t = captured_at_t * (1 - captured_before)
            new_individuals = new_individuals.at[t-1].set(jnp.sum(new_at_t))
        
        # Average number of new individuals per occasion
        avg_new = jnp.mean(new_individuals)
        
        # Estimate recruitment rate relative to population size using JAX-compatible conditional
        population_estimate = n_individuals  # Rough proxy
        f_estimate = jnp.where(
            population_estimate > 0,
            avg_new / population_estimate,
            default_recruitment
        )
        
        return jnp.maximum(f_estimate, 0.01)
    
    def _split_parameters(
        self, 
        parameters: jnp.ndarray, 
        design_matrices: Dict[str, DesignMatrixInfo]
    ) -> Dict[str, jnp.ndarray]:
        """Split concatenated parameter vector by parameter type."""
        param_dict = {}
        start_idx = 0
        
        for param_name in self.parameter_order:
            design_info = design_matrices[param_name]
            n_params = design_info.parameter_count
            
            param_dict[param_name] = parameters[start_idx:start_idx + n_params]
            start_idx += n_params
        
        return param_dict
    
    def log_likelihood(
        self,
        parameters: jnp.ndarray,
        data_context: DataContext,
        design_matrices: Dict[str, DesignMatrixInfo]
    ) -> float:
        """
        Calculate Pradel model log-likelihood using JAX-compatible operations.
        
        Args:
            parameters: Concatenated parameter vector
            data_context: Data and covariates
            design_matrices: Design matrices for each parameter
            
        Returns:
            Log-likelihood value
        """
        # Split parameters by type
        param_split = self._split_parameters(parameters, design_matrices)
        
        # Get design matrices
        X_phi = design_matrices["phi"].matrix
        X_p = design_matrices["p"].matrix
        X_f = design_matrices["f"].matrix
        
        # Calculate linear predictors
        eta_phi = X_phi @ param_split["phi"]
        eta_p = X_p @ param_split["p"]
        eta_f = X_f @ param_split["f"]
        
        # Apply link functions
        phi = inv_logit(eta_phi)  # Survival probability (0-1)
        p = inv_logit(eta_p)      # Detection probability (0-1)
        f = exp_link(eta_f)       # Recruitment rate (positive)
        
        # Get capture matrix
        capture_matrix = data_context.capture_matrix
        n_individuals, n_occasions = capture_matrix.shape
        
        # Vectorized likelihood calculation using JAX-compatible operations
        log_lik_contributions = jnp.zeros(n_individuals)
        
        # Process each individual's capture history
        for i in range(n_individuals):
            ch = capture_matrix[i, :]  # Capture history for individual i
            individual_loglik = 0.0
            
            # Process consecutive capture occasions
            for t in range(n_occasions - 1):
                # Use jnp.where instead of if statements for JAX compatibility
                captured_at_t = ch[t]
                captured_at_t1 = ch[t + 1]
                
                # Contribution when captured at time t
                recapture_prob = phi[i] * p[i]
                log_recapture = jnp.log(recapture_prob)
                log_no_recapture = jnp.log(1 - recapture_prob)
                
                # Use jnp.where for conditional logic
                contribution = jnp.where(
                    captured_at_t == 1,
                    jnp.where(
                        captured_at_t1 == 1,
                        log_recapture,
                        log_no_recapture
                    ),
                    0.0
                )
                individual_loglik += contribution
            
            # Add first capture probability using JAX-compatible operations
            # Check if individual was captured at least once
            was_captured = jnp.sum(ch) > 0
            first_capture_contribution = jnp.where(
                was_captured,
                jnp.log(p[i]),
                0.0
            )
            individual_loglik += first_capture_contribution
            
            log_lik_contributions = log_lik_contributions.at[i].set(individual_loglik)
        
        return jnp.sum(log_lik_contributions)
    
    def validate_data(self, data_context: DataContext) -> None:
        """Validate data for Pradel model."""
        super().validate_data(data_context)
        
        # Pradel-specific validation
        if data_context.n_occasions < 3:
            raise ModelSpecificationError(
                formula="Pradel model data validation",
                suggestions=[
                    "Pradel models require at least 3 capture occasions",
                    "Use CJS model for 2-occasion data",
                ]
            )
        
        # Check for empty capture histories
        capture_matrix = data_context.capture_matrix
        empty_histories = jnp.sum(capture_matrix, axis=1) == 0
        n_empty = jnp.sum(empty_histories)
        
        if n_empty > 0:
            logger.warning(f"Found {n_empty} individuals with no captures - these will be ignored")
    
    def get_parameter_names(self, design_matrices: Dict[str, DesignMatrixInfo]) -> List[str]:
        """Get names of all parameters in order."""
        names = []
        
        for param_name in self.parameter_order:
            design_info = design_matrices[param_name]
            
            # Add parameter prefix to column names
            param_names = [f"{param_name}_{col}" for col in design_info.column_names]
            names.extend(param_names)
        
        return names