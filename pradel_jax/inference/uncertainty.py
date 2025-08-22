"""
Parameter uncertainty estimation for pradel-jax.

Provides rigorous statistical methods for parameter uncertainty:
- Hessian-based standard errors (asymptotic theory)
- Fisher Information Matrix computation
- Bootstrap confidence intervals (non-parametric)
- Profile likelihood confidence intervals
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax import grad, hessian, vmap
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings

from ..core.exceptions import ModelSpecificationError, OptimizationError
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ParameterUncertainty:
    """
    Container for parameter uncertainty information.
    
    Based on standard statistical inference theory for maximum likelihood estimation.
    """
    parameter_names: List[str]
    estimates: np.ndarray
    standard_errors: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]  # Keys: '95%', '99%', etc.
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    
    # Diagnostic information
    fisher_information_matrix: Optional[np.ndarray] = None
    hessian_matrix: Optional[np.ndarray] = None
    hessian_condition_number: Optional[float] = None
    
    # Bootstrap results (if available)
    bootstrap_samples: Optional[np.ndarray] = None
    bootstrap_bias: Optional[np.ndarray] = None
    bootstrap_bias_corrected_estimates: Optional[np.ndarray] = None
    
    def get_parameter_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each parameter."""
        summary = {}
        
        for i, name in enumerate(self.parameter_names):
            summary[name] = {
                'estimate': float(self.estimates[i]),
                'std_error': float(self.standard_errors[i]),
                'z_score': float(self.estimates[i] / self.standard_errors[i]) if self.standard_errors[i] > 0 else np.inf,
                'p_value': 2 * (1 - stats.norm.cdf(abs(self.estimates[i] / self.standard_errors[i]))) if self.standard_errors[i] > 0 else 0.0
            }
            
            # Add confidence intervals
            for level, intervals in self.confidence_intervals.items():
                summary[name][f'ci_lower_{level}'] = float(intervals[i, 0])
                summary[name][f'ci_upper_{level}'] = float(intervals[i, 1])
        
        return summary


class HessianBasedUncertainty:
    """
    Computes parameter uncertainty using Hessian-based asymptotic theory.
    
    Uses the inverse of the negative Hessian (Fisher Information Matrix) 
    to estimate parameter covariance matrix under regularity conditions.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def compute_uncertainty(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        parameter_estimates: np.ndarray,
        parameter_names: List[str],
        confidence_levels: List[float] = [0.95, 0.99],
        use_finite_differences: bool = False,
        fd_step_size: float = 1e-8
    ) -> ParameterUncertainty:
        """
        Compute parameter uncertainty using Hessian-based methods.
        
        Args:
            log_likelihood_fn: Log-likelihood function
            parameter_estimates: MLE parameter estimates
            parameter_names: Names of parameters
            confidence_levels: Confidence levels for intervals
            use_finite_differences: Use finite differences instead of auto-diff
            fd_step_size: Step size for finite differences
            
        Returns:
            ParameterUncertainty object with all uncertainty information
        """
        self.logger.info("Computing Hessian-based parameter uncertainty")
        
        # Compute Hessian matrix
        if use_finite_differences:
            hessian_matrix = self._compute_finite_difference_hessian(
                log_likelihood_fn, parameter_estimates, fd_step_size
            )
        else:
            hessian_matrix = self._compute_autodiff_hessian(
                log_likelihood_fn, parameter_estimates
            )
        
        # Fisher Information Matrix is negative Hessian of log-likelihood
        fisher_information = -hessian_matrix
        
        # Check matrix condition
        condition_number = np.linalg.cond(fisher_information)
        if condition_number > 1e12:
            self.logger.warning(
                f"Fisher Information Matrix is ill-conditioned (condition number: {condition_number:.2e}). "
                "Standard errors may be unreliable."
            )
        
        # Compute covariance matrix (inverse of Fisher Information)
        try:
            covariance_matrix = np.linalg.inv(fisher_information)
        except np.linalg.LinAlgError:
            # Matrix is singular - use pseudo-inverse
            self.logger.warning("Fisher Information Matrix is singular. Using pseudo-inverse.")
            covariance_matrix = np.linalg.pinv(fisher_information)
        
        # Extract standard errors (diagonal elements)
        variances = np.diag(covariance_matrix)
        if np.any(variances < 0):
            self.logger.warning("Negative variances detected. Setting to small positive value.")
            variances = np.maximum(variances, 1e-10)
        
        standard_errors = np.sqrt(variances)
        
        # Compute correlation matrix
        correlation_matrix = self._compute_correlation_matrix(covariance_matrix)
        
        # Compute confidence intervals
        confidence_intervals = self._compute_confidence_intervals(
            parameter_estimates, standard_errors, confidence_levels
        )
        
        self.logger.info(f"Computed uncertainty for {len(parameter_names)} parameters")
        
        return ParameterUncertainty(
            parameter_names=parameter_names,
            estimates=parameter_estimates,
            standard_errors=standard_errors,
            confidence_intervals=confidence_intervals,
            correlation_matrix=correlation_matrix,
            covariance_matrix=covariance_matrix,
            fisher_information_matrix=fisher_information,
            hessian_matrix=hessian_matrix,
            hessian_condition_number=condition_number
        )
    
    def _compute_autodiff_hessian(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        parameters: np.ndarray
    ) -> np.ndarray:
        """Compute Hessian using JAX automatic differentiation."""
        
        # Convert function to JAX if needed
        def jax_ll_fn(params):
            return log_likelihood_fn(np.array(params))
        
        # Compute Hessian
        hessian_fn = hessian(jax_ll_fn)
        
        try:
            hessian_matrix = hessian_fn(jnp.array(parameters))
            return np.array(hessian_matrix)
        except Exception as e:
            self.logger.warning(f"Auto-diff Hessian computation failed: {e}. Using finite differences.")
            return self._compute_finite_difference_hessian(log_likelihood_fn, parameters)
    
    def _compute_finite_difference_hessian(
        self,
        log_likelihood_fn: Callable[[np.ndarray], float],
        parameters: np.ndarray,
        step_size: float = 1e-8
    ) -> np.ndarray:
        """Compute Hessian using finite differences."""
        
        n_params = len(parameters)
        hessian_matrix = np.zeros((n_params, n_params))
        
        # Compute diagonal elements (second derivatives)
        for i in range(n_params):
            h = step_size * max(1.0, abs(parameters[i]))
            
            params_plus = parameters.copy()
            params_plus[i] += h
            
            params_minus = parameters.copy() 
            params_minus[i] -= h
            
            ll_plus = log_likelihood_fn(params_plus)
            ll_minus = log_likelihood_fn(params_minus)
            ll_center = log_likelihood_fn(parameters)
            
            hessian_matrix[i, i] = (ll_plus - 2 * ll_center + ll_minus) / (h**2)
        
        # Compute off-diagonal elements (cross derivatives)
        for i in range(n_params):
            for j in range(i + 1, n_params):
                h_i = step_size * max(1.0, abs(parameters[i]))
                h_j = step_size * max(1.0, abs(parameters[j]))
                
                params_pp = parameters.copy()
                params_pp[i] += h_i
                params_pp[j] += h_j
                
                params_pm = parameters.copy()
                params_pm[i] += h_i
                params_pm[j] -= h_j
                
                params_mp = parameters.copy()
                params_mp[i] -= h_i
                params_mp[j] += h_j
                
                params_mm = parameters.copy()
                params_mm[i] -= h_i
                params_mm[j] -= h_j
                
                ll_pp = log_likelihood_fn(params_pp)
                ll_pm = log_likelihood_fn(params_pm)
                ll_mp = log_likelihood_fn(params_mp)
                ll_mm = log_likelihood_fn(params_mm)
                
                cross_deriv = (ll_pp - ll_pm - ll_mp + ll_mm) / (4 * h_i * h_j)
                
                hessian_matrix[i, j] = cross_deriv
                hessian_matrix[j, i] = cross_deriv  # Symmetric
        
        return hessian_matrix
    
    def _compute_correlation_matrix(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Compute correlation matrix from covariance matrix."""
        std_devs = np.sqrt(np.diag(covariance_matrix))
        correlation_matrix = covariance_matrix / np.outer(std_devs, std_devs)
        
        # Ensure diagonal is exactly 1
        np.fill_diagonal(correlation_matrix, 1.0)
        
        return correlation_matrix
    
    def _compute_confidence_intervals(
        self,
        estimates: np.ndarray,
        standard_errors: np.ndarray,
        confidence_levels: List[float]
    ) -> Dict[str, np.ndarray]:
        """Compute confidence intervals using normal approximation."""
        
        confidence_intervals = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower = estimates - z_score * standard_errors
            upper = estimates + z_score * standard_errors
            
            intervals = np.column_stack([lower, upper])
            confidence_intervals[f"{level:.0%}"] = intervals
        
        return confidence_intervals


class BootstrapUncertainty:
    """
    Computes parameter uncertainty using bootstrap methods.
    
    Provides non-parametric confidence intervals that don't rely on 
    asymptotic normality assumptions.
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def compute_bootstrap_uncertainty(
        self,
        data_context: Any,
        model_fit_fn: Callable[[Any], Tuple[np.ndarray, float]],
        n_bootstrap_samples: int = 1000,
        confidence_levels: List[float] = [0.95, 0.99],
        random_seed: Optional[int] = None
    ) -> ParameterUncertainty:
        """
        Compute parameter uncertainty using bootstrap resampling.
        
        Args:
            data_context: Original data context
            model_fit_fn: Function that fits model to data, returns (params, log_likelihood)
            n_bootstrap_samples: Number of bootstrap samples
            confidence_levels: Confidence levels for intervals
            random_seed: Random seed for reproducibility
            
        Returns:
            ParameterUncertainty object with bootstrap-based uncertainty
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.logger.info(f"Computing bootstrap uncertainty with {n_bootstrap_samples} samples")
        
        # Get original parameter estimates
        original_params, _ = model_fit_fn(data_context)
        n_params = len(original_params)
        
        # Storage for bootstrap samples
        bootstrap_samples = []
        successful_samples = 0
        
        # Bootstrap sampling
        for i in range(n_bootstrap_samples):
            try:
                # Create bootstrap sample of data
                bootstrap_data = self._create_bootstrap_sample(data_context)
                
                # Fit model to bootstrap sample
                bootstrap_params, _ = model_fit_fn(bootstrap_data)
                
                if len(bootstrap_params) == n_params:
                    bootstrap_samples.append(bootstrap_params)
                    successful_samples += 1
                
            except Exception as e:
                self.logger.debug(f"Bootstrap sample {i} failed: {e}")
                continue
            
            # Progress logging
            if (i + 1) % 100 == 0:
                self.logger.info(f"Completed {i+1}/{n_bootstrap_samples} bootstrap samples")
        
        if successful_samples < n_bootstrap_samples * 0.5:
            raise OptimizationError(
                f"Too many bootstrap samples failed ({n_bootstrap_samples - successful_samples}/"
                f"{n_bootstrap_samples}). Model may be unstable or data insufficient."
            )
        
        bootstrap_samples = np.array(bootstrap_samples)
        
        self.logger.info(f"Successfully completed {successful_samples} bootstrap samples")
        
        # Compute bootstrap statistics
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        bootstrap_std_errors = np.std(bootstrap_samples, axis=0, ddof=1)
        
        # Bias correction
        bootstrap_bias = bootstrap_means - original_params
        bias_corrected_estimates = original_params - bootstrap_bias
        
        # Bootstrap confidence intervals (percentile method)
        confidence_intervals = {}
        for level in confidence_levels:
            alpha = 1 - level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)
            
            lower = np.percentile(bootstrap_samples, lower_percentile, axis=0)
            upper = np.percentile(bootstrap_samples, upper_percentile, axis=0)
            
            confidence_intervals[f"{level:.0%}"] = np.column_stack([lower, upper])
        
        # Compute correlation and covariance matrices
        covariance_matrix = np.cov(bootstrap_samples.T)
        correlation_matrix = np.corrcoef(bootstrap_samples.T)
        
        # Parameter names (if not provided)
        parameter_names = [f"param_{i}" for i in range(n_params)]
        
        return ParameterUncertainty(
            parameter_names=parameter_names,
            estimates=original_params,
            standard_errors=bootstrap_std_errors,
            confidence_intervals=confidence_intervals,
            correlation_matrix=correlation_matrix,
            covariance_matrix=covariance_matrix,
            bootstrap_samples=bootstrap_samples,
            bootstrap_bias=bootstrap_bias,
            bootstrap_bias_corrected_estimates=bias_corrected_estimates
        )
    
    def _create_bootstrap_sample(self, data_context: Any) -> Any:
        """Create bootstrap sample by resampling individuals."""
        n_individuals = data_context.n_individuals
        
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_individuals, size=n_individuals, replace=True)
        
        # Create new capture matrix
        bootstrap_capture_matrix = data_context.capture_matrix[bootstrap_indices, :]
        
        # Create new covariate arrays
        bootstrap_covariates = {}
        for name, values in data_context.covariates.items():
            if isinstance(values, (np.ndarray, jnp.ndarray)) and len(values.shape) > 0:
                if values.shape[0] == n_individuals:
                    bootstrap_covariates[name] = values[bootstrap_indices]
                else:
                    # Keep non-individual-level data unchanged
                    bootstrap_covariates[name] = values
            else:
                # Keep scalar or metadata unchanged
                bootstrap_covariates[name] = values
        
        # Create new data context
        return type(data_context)(
            capture_matrix=bootstrap_capture_matrix,
            covariates=bootstrap_covariates,
            covariate_info=data_context.covariate_info,
            n_individuals=n_individuals,  # Same sample size
            n_occasions=data_context.n_occasions,
            occasion_names=data_context.occasion_names,
            individual_ids=None,  # Don't track individual IDs in bootstrap
            metadata=data_context.metadata
        )


# Convenience functions
def compute_hessian_standard_errors(
    log_likelihood_fn: Callable[[np.ndarray], float],
    parameter_estimates: np.ndarray,
    parameter_names: List[str],
    **kwargs
) -> ParameterUncertainty:
    """
    Convenience function for Hessian-based standard errors.
    
    Args:
        log_likelihood_fn: Log-likelihood function
        parameter_estimates: MLE parameter estimates  
        parameter_names: Parameter names
        **kwargs: Additional arguments for HessianBasedUncertainty
        
    Returns:
        ParameterUncertainty object
    """
    uncertainty_computer = HessianBasedUncertainty()
    return uncertainty_computer.compute_uncertainty(
        log_likelihood_fn, parameter_estimates, parameter_names, **kwargs
    )


def bootstrap_confidence_intervals(
    data_context: Any,
    model_fit_fn: Callable[[Any], Tuple[np.ndarray, float]], 
    **kwargs
) -> ParameterUncertainty:
    """
    Convenience function for bootstrap confidence intervals.
    
    Args:
        data_context: Data context
        model_fit_fn: Model fitting function
        **kwargs: Additional arguments for BootstrapUncertainty
        
    Returns:
        ParameterUncertainty object
    """
    uncertainty_computer = BootstrapUncertainty()
    return uncertainty_computer.compute_bootstrap_uncertainty(
        data_context, model_fit_fn, **kwargs
    )


def compute_fisher_information(
    log_likelihood_fn: Callable[[np.ndarray], float],
    parameters: np.ndarray
) -> np.ndarray:
    """
    Compute Fisher Information Matrix.
    
    Args:
        log_likelihood_fn: Log-likelihood function
        parameters: Parameter values
        
    Returns:
        Fisher Information Matrix (negative Hessian of log-likelihood)
    """
    uncertainty_computer = HessianBasedUncertainty()
    hessian_matrix = uncertainty_computer._compute_autodiff_hessian(log_likelihood_fn, parameters)
    return -hessian_matrix