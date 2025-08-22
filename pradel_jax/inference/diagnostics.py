"""
Model diagnostics and selection criteria for pradel-jax.

Provides comprehensive model diagnostics based on statistical theory:
- Information criteria (AIC, AICc, BIC, QAIC)
- Goodness-of-fit tests (Chi-square, Deviance)
- Residual analysis (Pearson, Deviance residuals)
- Diagnostic plots and visualizations
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass
from scipy import stats
import warnings

from ..core.exceptions import ModelSpecificationError
from ..utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class ModelSelectionCriteria:
    """Container for model selection criteria."""
    log_likelihood: float
    aic: float
    aicc: float  # Corrected AIC for small samples
    bic: float
    qaic: Optional[float] = None  # Quasi-AIC for overdispersed models
    qaicc: Optional[float] = None  # Corrected QAIC
    
    # Additional metrics
    deviance: Optional[float] = None
    effective_sample_size: Optional[int] = None
    overdispersion_parameter: Optional[float] = None
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all criteria."""
        summary = {
            'log_likelihood': self.log_likelihood,
            'aic': self.aic,
            'aicc': self.aicc,
            'bic': self.bic,
        }
        
        if self.qaic is not None:
            summary['qaic'] = self.qaic
        if self.qaicc is not None:
            summary['qaicc'] = self.qaicc
        if self.deviance is not None:
            summary['deviance'] = self.deviance
        if self.overdispersion_parameter is not None:
            summary['c_hat'] = self.overdispersion_parameter
            
        return summary


@dataclass 
class GoodnessOfFitResults:
    """Results from goodness-of-fit tests."""
    chi_square_statistic: float
    chi_square_df: int
    chi_square_p_value: float
    
    deviance_statistic: float
    deviance_df: int
    deviance_p_value: float
    
    # Overdispersion assessment
    overdispersion_estimate: float
    is_overdispersed: bool
    
    # Residual diagnostics
    pearson_residuals: np.ndarray
    deviance_residuals: np.ndarray
    standardized_residuals: np.ndarray
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of goodness-of-fit results."""
        return {
            'chi_square': {
                'statistic': self.chi_square_statistic,
                'df': self.chi_square_df,
                'p_value': self.chi_square_p_value
            },
            'deviance': {
                'statistic': self.deviance_statistic,
                'df': self.deviance_df,
                'p_value': self.deviance_p_value
            },
            'overdispersion': {
                'c_hat': self.overdispersion_estimate,
                'is_overdispersed': self.is_overdispersed
            },
            'residuals_summary': {
                'n_residuals': len(self.pearson_residuals),
                'pearson_mean': float(np.mean(self.pearson_residuals)),
                'pearson_std': float(np.std(self.pearson_residuals)),
                'deviance_mean': float(np.mean(self.deviance_residuals)),
                'deviance_std': float(np.std(self.deviance_residuals))
            }
        }


@dataclass
class ModelDiagnostics:
    """Complete model diagnostics container."""
    selection_criteria: ModelSelectionCriteria
    goodness_of_fit: GoodnessOfFitResults
    
    # Model specification
    n_parameters: int
    n_observations: int
    model_name: str
    
    # Additional diagnostics
    convergence_info: Optional[Dict[str, Any]] = None
    parameter_estimates: Optional[np.ndarray] = None
    parameter_names: Optional[List[str]] = None
    
    def compare_with(self, other: 'ModelDiagnostics') -> Dict[str, float]:
        """Compare this model with another model."""
        comparison = {}
        
        # Information criteria differences (negative = this model is better)
        comparison['delta_aic'] = self.selection_criteria.aic - other.selection_criteria.aic
        comparison['delta_aicc'] = self.selection_criteria.aicc - other.selection_criteria.aicc  
        comparison['delta_bic'] = self.selection_criteria.bic - other.selection_criteria.bic
        
        # Log-likelihood difference
        comparison['delta_log_likelihood'] = (
            self.selection_criteria.log_likelihood - other.selection_criteria.log_likelihood
        )
        
        # Akaike weights
        aic_diff = comparison['delta_aic']
        comparison['akaike_weight_ratio'] = np.exp(-0.5 * aic_diff) if aic_diff != 0 else 1.0
        
        return comparison


class ModelSelectionCriteriaComputer:
    """
    Computes model selection criteria based on information theory.
    
    Implements standard criteria with proper corrections for small samples
    and overdispersion following Burnham & Anderson (2002).
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def compute_criteria(
        self,
        log_likelihood: float,
        n_parameters: int,
        n_observations: int,
        overdispersion_parameter: Optional[float] = None
    ) -> ModelSelectionCriteria:
        """
        Compute model selection criteria.
        
        Args:
            log_likelihood: Maximum log-likelihood value
            n_parameters: Number of estimated parameters
            n_observations: Number of observations (individuals)
            overdispersion_parameter: Overdispersion parameter (c-hat)
            
        Returns:
            ModelSelectionCriteria object
        """
        # AIC: Akaike Information Criterion
        aic = -2 * log_likelihood + 2 * n_parameters
        
        # AICc: Corrected AIC for small samples
        if n_observations > n_parameters + 1:
            correction = (2 * n_parameters * (n_parameters + 1)) / (n_observations - n_parameters - 1)
            aicc = aic + correction
        else:
            aicc = np.inf  # AICc undefined for small samples
            self.logger.warning("AICc undefined: n_observations <= n_parameters + 1")
        
        # BIC: Bayesian Information Criterion
        bic = -2 * log_likelihood + n_parameters * np.log(n_observations)
        
        # Deviance (for goodness-of-fit)
        deviance = -2 * log_likelihood
        
        # QAIC and QAICc for overdispersed models
        qaic = None
        qaicc = None
        if overdispersion_parameter is not None and overdispersion_parameter > 1:
            qaic = aic / overdispersion_parameter
            if aicc != np.inf:
                qaicc = aicc / overdispersion_parameter
        
        self.logger.debug(
            f"Model criteria: AIC={aic:.2f}, AICc={aicc:.2f}, BIC={bic:.2f}, "
            f"LogLik={log_likelihood:.2f}"
        )
        
        return ModelSelectionCriteria(
            log_likelihood=log_likelihood,
            aic=aic,
            aicc=aicc,
            bic=bic,
            qaic=qaic,
            qaicc=qaicc,
            deviance=deviance,
            effective_sample_size=n_observations,
            overdispersion_parameter=overdispersion_parameter
        )


class GoodnessOfFitTester:
    """
    Performs goodness-of-fit tests for capture-recapture models.
    
    Implements standard tests appropriate for capture-recapture data:
    - Pearson chi-square test
    - Deviance test  
    - Overdispersion assessment
    """
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    def compute_goodness_of_fit(
        self,
        observed_data: np.ndarray,
        expected_data: np.ndarray,
        model_predictions: np.ndarray,
        n_parameters: int
    ) -> GoodnessOfFitResults:
        """
        Compute comprehensive goodness-of-fit statistics.
        
        Args:
            observed_data: Observed capture frequencies
            expected_data: Expected capture frequencies under model
            model_predictions: Model predicted probabilities
            n_parameters: Number of model parameters
            
        Returns:
            GoodnessOfFitResults object
        """
        n_observations = len(observed_data)
        df = n_observations - n_parameters
        
        if df <= 0:
            self.logger.warning("Degrees of freedom <= 0. Goodness-of-fit tests may be unreliable.")
            df = max(1, df)  # Avoid division by zero
        
        # Pearson chi-square test
        chi_square_stat, chi_square_p = self._compute_pearson_chi_square(
            observed_data, expected_data, df
        )
        
        # Deviance test
        deviance_stat, deviance_p = self._compute_deviance_test(
            observed_data, expected_data, df
        )
        
        # Residual analysis
        pearson_residuals = self._compute_pearson_residuals(observed_data, expected_data)
        deviance_residuals = self._compute_deviance_residuals(observed_data, expected_data)
        standardized_residuals = self._compute_standardized_residuals(
            pearson_residuals, expected_data
        )
        
        # Overdispersion assessment
        overdispersion_estimate = max(chi_square_stat / df, 1.0)
        is_overdispersed = overdispersion_estimate > 1.5  # Common threshold
        
        if is_overdispersed:
            self.logger.info(f"Model shows evidence of overdispersion (c-hat = {overdispersion_estimate:.3f})")
        
        return GoodnessOfFitResults(
            chi_square_statistic=chi_square_stat,
            chi_square_df=df,
            chi_square_p_value=chi_square_p,
            deviance_statistic=deviance_stat,
            deviance_df=df,
            deviance_p_value=deviance_p,
            overdispersion_estimate=overdispersion_estimate,
            is_overdispersed=is_overdispersed,
            pearson_residuals=pearson_residuals,
            deviance_residuals=deviance_residuals,
            standardized_residuals=standardized_residuals
        )
    
    def _compute_pearson_chi_square(
        self,
        observed: np.ndarray,
        expected: np.ndarray,
        df: int
    ) -> Tuple[float, float]:
        """Compute Pearson chi-square statistic."""
        # Avoid division by zero
        expected_safe = np.maximum(expected, 1e-10)
        
        chi_square_stat = np.sum((observed - expected)**2 / expected_safe)
        p_value = 1 - stats.chi2.cdf(chi_square_stat, df) if df > 0 else np.nan
        
        return float(chi_square_stat), float(p_value)
    
    def _compute_deviance_test(
        self,
        observed: np.ndarray,
        expected: np.ndarray, 
        df: int
    ) -> Tuple[float, float]:
        """Compute deviance test statistic."""
        # Avoid log(0)
        observed_safe = np.maximum(observed, 1e-10)
        expected_safe = np.maximum(expected, 1e-10)
        
        # Only include non-zero observed values in deviance
        mask = observed > 0
        if np.sum(mask) > 0:
            deviance_components = observed[mask] * np.log(observed_safe[mask] / expected_safe[mask])
            deviance_stat = 2 * np.sum(deviance_components)
        else:
            deviance_stat = 0.0
        
        p_value = 1 - stats.chi2.cdf(deviance_stat, df) if df > 0 else np.nan
        
        return float(deviance_stat), float(p_value)
    
    def _compute_pearson_residuals(
        self,
        observed: np.ndarray,
        expected: np.ndarray
    ) -> np.ndarray:
        """Compute Pearson residuals."""
        expected_safe = np.maximum(expected, 1e-10)
        return (observed - expected) / np.sqrt(expected_safe)
    
    def _compute_deviance_residuals(
        self,
        observed: np.ndarray,
        expected: np.ndarray
    ) -> np.ndarray:
        """Compute deviance residuals."""
        observed_safe = np.maximum(observed, 1e-10)
        expected_safe = np.maximum(expected, 1e-10)
        
        # Sign of residual
        signs = np.sign(observed - expected)
        
        # Deviance components
        deviance_components = np.zeros_like(observed)
        mask = observed > 0
        
        if np.sum(mask) > 0:
            deviance_components[mask] = 2 * observed[mask] * np.log(observed_safe[mask] / expected_safe[mask])
        
        # For zero observed values, use different formula
        zero_mask = observed == 0
        if np.sum(zero_mask) > 0:
            deviance_components[zero_mask] = 2 * expected[zero_mask]
        
        deviance_residuals = signs * np.sqrt(np.abs(deviance_components))
        
        return deviance_residuals
    
    def _compute_standardized_residuals(
        self,
        pearson_residuals: np.ndarray,
        expected: np.ndarray
    ) -> np.ndarray:
        """Compute standardized residuals."""
        # Simple standardization (more complex versions exist)
        variance = np.maximum(expected * (1 - expected / np.sum(expected)), 1e-10)
        return pearson_residuals / np.sqrt(variance)


# Convenience functions
def compute_model_selection_criteria(
    log_likelihood: float,
    n_parameters: int,
    n_observations: int,
    overdispersion_parameter: Optional[float] = None
) -> ModelSelectionCriteria:
    """
    Convenience function to compute model selection criteria.
    
    Args:
        log_likelihood: Maximum log-likelihood
        n_parameters: Number of parameters
        n_observations: Number of observations
        overdispersion_parameter: Overdispersion parameter
        
    Returns:
        ModelSelectionCriteria object
    """
    computer = ModelSelectionCriteriaComputer()
    return computer.compute_criteria(
        log_likelihood, n_parameters, n_observations, overdispersion_parameter
    )


def compute_goodness_of_fit_tests(
    observed_data: np.ndarray,
    expected_data: np.ndarray,
    model_predictions: np.ndarray,
    n_parameters: int
) -> GoodnessOfFitResults:
    """
    Convenience function to compute goodness-of-fit tests.
    
    Args:
        observed_data: Observed data
        expected_data: Expected data under model
        model_predictions: Model predictions
        n_parameters: Number of parameters
        
    Returns:
        GoodnessOfFitResults object
    """
    tester = GoodnessOfFitTester()
    return tester.compute_goodness_of_fit(
        observed_data, expected_data, model_predictions, n_parameters
    )


def compute_complete_model_diagnostics(
    log_likelihood: float,
    n_parameters: int,
    n_observations: int,
    observed_data: np.ndarray,
    expected_data: np.ndarray,
    model_predictions: np.ndarray,
    model_name: str = "Unknown",
    parameter_estimates: Optional[np.ndarray] = None,
    parameter_names: Optional[List[str]] = None,
    convergence_info: Optional[Dict[str, Any]] = None
) -> ModelDiagnostics:
    """
    Compute complete model diagnostics.
    
    Args:
        log_likelihood: Maximum log-likelihood
        n_parameters: Number of parameters
        n_observations: Number of observations
        observed_data: Observed data
        expected_data: Expected data
        model_predictions: Model predictions
        model_name: Name of the model
        parameter_estimates: Parameter estimates
        parameter_names: Parameter names
        convergence_info: Optimization convergence information
        
    Returns:
        ModelDiagnostics object with complete diagnostics
    """
    # Compute goodness-of-fit first to get overdispersion estimate
    gof_results = compute_goodness_of_fit_tests(
        observed_data, expected_data, model_predictions, n_parameters
    )
    
    # Compute selection criteria with overdispersion if present
    overdispersion = gof_results.overdispersion_estimate if gof_results.is_overdispersed else None
    selection_criteria = compute_model_selection_criteria(
        log_likelihood, n_parameters, n_observations, overdispersion
    )
    
    return ModelDiagnostics(
        selection_criteria=selection_criteria,
        goodness_of_fit=gof_results,
        n_parameters=n_parameters,
        n_observations=n_observations,
        model_name=model_name,
        convergence_info=convergence_info,
        parameter_estimates=parameter_estimates,
        parameter_names=parameter_names
    )


def plot_diagnostic_plots(diagnostics: ModelDiagnostics, save_path: Optional[str] = None):
    """
    Create diagnostic plots for model assessment.
    
    Args:
        diagnostics: ModelDiagnostics object
        save_path: Optional path to save plots
        
    Note: This function requires matplotlib. Implementation would include:
    - Residual plots (Q-Q plot, residuals vs fitted)
    - Goodness-of-fit visualization
    - Parameter uncertainty plots
    """
    # This would require matplotlib dependency
    # For now, just log that plots would be created
    logger.info("Diagnostic plots would be created here (requires matplotlib)")
    logger.info(f"Model: {diagnostics.model_name}")
    logger.info(f"AIC: {diagnostics.selection_criteria.aic:.2f}")
    logger.info(f"BIC: {diagnostics.selection_criteria.bic:.2f}")
    logger.info(f"Overdispersion: {diagnostics.goodness_of_fit.overdispersion_estimate:.3f}")
    
    if save_path:
        logger.info(f"Plots would be saved to: {save_path}")