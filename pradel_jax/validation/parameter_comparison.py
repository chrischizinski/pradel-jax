"""
Parameter Comparison Utilities for RMark Validation.

This module provides comprehensive statistical comparison of parameter estimates
between JAX-based Pradel models and RMark results, following industry standards
for numerical validation and bioequivalence testing.

Key Features:
    - Parameter-by-parameter statistical comparison
    - Confidence interval overlap analysis  
    - Relative and absolute difference calculation
    - Model-level comparison with AIC concordance
    - Publication-ready comparison summaries

Statistical Methods:
    - Absolute and relative tolerance testing
    - Confidence interval overlap detection
    - Effect size calculation (Cohen's d)
    - Statistical significance testing
    - Bioequivalence-style tolerance bounds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ComparisonStatus(Enum):
    """Status of parameter comparison."""
    EXCELLENT = "excellent"      # <0.1% difference
    GOOD = "good"               # <1% difference  
    ACCEPTABLE = "acceptable"   # <5% difference
    POOR = "poor"              # >5% difference
    FAILED = "failed"          # Major discrepancy or error


@dataclass
class ParameterComparisonResult:
    """Result of comparing a single parameter between JAX and RMark."""
    
    # Parameter identification
    parameter_name: str
    parameter_type: str  # "phi", "p", "f"
    formula_term: Optional[str] = None  # e.g., "intercept", "sex", "age"
    
    # Point estimates and uncertainty
    jax_estimate: float = 0.0
    jax_std_error: float = 0.0
    jax_confidence_interval: Optional[Tuple[float, float]] = None
    
    rmark_estimate: float = 0.0
    rmark_std_error: float = 0.0  
    rmark_confidence_interval: Optional[Tuple[float, float]] = None
    
    # Comparison metrics
    absolute_difference: float = 0.0
    relative_difference_pct: float = 0.0
    standardized_difference: float = 0.0  # Cohen's d-like measure
    
    # Statistical tests
    confidence_intervals_overlap: bool = False
    overlap_proportion: float = 0.0  # Proportion of CI overlap
    within_absolute_tolerance: bool = False
    within_relative_tolerance: bool = False
    
    # Quality assessment
    comparison_status: ComparisonStatus = ComparisonStatus.FAILED
    precision_level: str = "unknown"
    notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self._calculate_differences()
        self._calculate_confidence_interval_overlap()
        self._assess_comparison_quality()
    
    def _calculate_differences(self):
        """Calculate absolute, relative, and standardized differences."""
        # Absolute difference
        self.absolute_difference = abs(self.jax_estimate - self.rmark_estimate)
        
        # Relative difference (percentage)
        if abs(self.rmark_estimate) > 1e-10:  # Avoid division by zero
            self.relative_difference_pct = (
                abs(self.jax_estimate - self.rmark_estimate) / abs(self.rmark_estimate) * 100
            )
        else:
            self.relative_difference_pct = float('inf') if self.absolute_difference > 1e-10 else 0.0
        
        # Standardized difference (Cohen's d-like)
        if self.jax_std_error > 0 and self.rmark_std_error > 0:
            pooled_se = np.sqrt((self.jax_std_error**2 + self.rmark_std_error**2) / 2)
            if pooled_se > 0:
                self.standardized_difference = abs(self.jax_estimate - self.rmark_estimate) / pooled_se
        
    def _calculate_confidence_interval_overlap(self):
        """Calculate confidence interval overlap."""
        if self.jax_confidence_interval and self.rmark_confidence_interval:
            jax_ci = self.jax_confidence_interval
            rmark_ci = self.rmark_confidence_interval
            
            # Check for any overlap
            overlap_start = max(jax_ci[0], rmark_ci[0])
            overlap_end = min(jax_ci[1], rmark_ci[1])
            
            if overlap_start <= overlap_end:
                self.confidence_intervals_overlap = True
                
                # Calculate proportion of overlap
                overlap_length = overlap_end - overlap_start
                jax_length = jax_ci[1] - jax_ci[0]
                rmark_length = rmark_ci[1] - rmark_ci[0]
                avg_length = (jax_length + rmark_length) / 2
                
                self.overlap_proportion = overlap_length / avg_length if avg_length > 0 else 0.0
            else:
                self.confidence_intervals_overlap = False
                self.overlap_proportion = 0.0
    
    def _assess_comparison_quality(self):
        """Assess overall comparison quality and status."""
        # Determine precision level based on relative difference
        if self.relative_difference_pct < 0.1:
            self.precision_level = "excellent"
            self.comparison_status = ComparisonStatus.EXCELLENT
        elif self.relative_difference_pct < 1.0:
            self.precision_level = "good"
            self.comparison_status = ComparisonStatus.GOOD
        elif self.relative_difference_pct < 5.0:
            self.precision_level = "acceptable"
            self.comparison_status = ComparisonStatus.ACCEPTABLE
        elif self.relative_difference_pct < 20.0:
            self.precision_level = "poor"
            self.comparison_status = ComparisonStatus.POOR
        else:
            self.precision_level = "failed"
            self.comparison_status = ComparisonStatus.FAILED
        
        # Add contextual notes
        if self.comparison_status == ComparisonStatus.EXCELLENT:
            self.notes.append("Parameters are numerically equivalent")
        elif self.comparison_status == ComparisonStatus.GOOD:
            self.notes.append("Parameters show excellent agreement")
        elif self.comparison_status == ComparisonStatus.ACCEPTABLE:
            self.notes.append("Parameters show acceptable agreement for practical purposes")
        elif self.comparison_status == ComparisonStatus.POOR:
            self.notes.append("Parameters show poor agreement - investigate implementation differences")
        else:
            self.notes.append("Parameters show major discrepancies - review model specifications")
        
        # Add specific notes based on analysis
        if not self.confidence_intervals_overlap and self.jax_confidence_interval and self.rmark_confidence_interval:
            self.notes.append("Confidence intervals do not overlap - statistically significant difference")
        elif self.confidence_intervals_overlap and self.overlap_proportion > 0.8:
            self.notes.append("Confidence intervals show substantial overlap - estimates are compatible")
        
        if self.standardized_difference > 2.0:
            self.notes.append("Large standardized difference (>2 SD) - practically significant difference")


@dataclass  
class ModelComparisonResult:
    """Result of comparing complete model results between JAX and RMark."""
    
    # Model identification
    model_formula: str
    model_specification: Dict[str, str] = field(default_factory=dict)
    
    # Model fit statistics
    jax_aic: float = 0.0
    jax_log_likelihood: float = 0.0
    jax_n_parameters: int = 0
    jax_convergence: bool = False
    
    rmark_aic: float = 0.0
    rmark_log_likelihood: float = 0.0
    rmark_n_parameters: int = 0
    rmark_convergence: bool = False
    
    # Comparison metrics
    aic_difference: float = 0.0
    aic_relative_difference_pct: float = 0.0
    likelihood_difference: float = 0.0
    likelihood_relative_difference_pct: float = 0.0
    
    # Parameter comparisons
    parameter_comparisons: List[ParameterComparisonResult] = field(default_factory=list)
    
    # Model ranking (within dataset)
    jax_aic_rank: Optional[int] = None
    rmark_aic_rank: Optional[int] = None
    ranking_agreement: bool = False
    
    # Overall assessment
    overall_status: ComparisonStatus = ComparisonStatus.FAILED
    parameter_pass_rate: float = 0.0
    critical_parameter_pass_rate: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self._calculate_model_differences()
        self._assess_parameter_pass_rates()
        self._assess_overall_status()
    
    def _calculate_model_differences(self):
        """Calculate model-level comparison metrics."""
        # AIC differences
        self.aic_difference = abs(self.jax_aic - self.rmark_aic)
        if abs(self.rmark_aic) > 1e-10:
            self.aic_relative_difference_pct = self.aic_difference / abs(self.rmark_aic) * 100
        
        # Log-likelihood differences  
        self.likelihood_difference = abs(self.jax_log_likelihood - self.rmark_log_likelihood)
        if abs(self.rmark_log_likelihood) > 1e-10:
            self.likelihood_relative_difference_pct = (
                self.likelihood_difference / abs(self.rmark_log_likelihood) * 100
            )
        
        # Ranking agreement
        if self.jax_aic_rank is not None and self.rmark_aic_rank is not None:
            self.ranking_agreement = (self.jax_aic_rank == self.rmark_aic_rank)
    
    def _assess_parameter_pass_rates(self):
        """Calculate parameter-level pass rates."""
        if not self.parameter_comparisons:
            return
        
        # Overall pass rate (acceptable or better)
        passing_params = [
            p for p in self.parameter_comparisons 
            if p.comparison_status in [ComparisonStatus.EXCELLENT, ComparisonStatus.GOOD, ComparisonStatus.ACCEPTABLE]
        ]
        self.parameter_pass_rate = len(passing_params) / len(self.parameter_comparisons)
        
        # Critical parameter pass rate (intercept parameters)
        critical_params = [
            p for p in self.parameter_comparisons
            if "intercept" in p.parameter_name.lower() or p.formula_term == "intercept"
        ]
        if critical_params:
            passing_critical = [
                p for p in critical_params
                if p.comparison_status in [ComparisonStatus.EXCELLENT, ComparisonStatus.GOOD, ComparisonStatus.ACCEPTABLE]
            ]
            self.critical_parameter_pass_rate = len(passing_critical) / len(critical_params)
        else:
            self.critical_parameter_pass_rate = 1.0  # No critical parameters to fail
    
    def _assess_overall_status(self):
        """Assess overall model comparison status."""
        # Check AIC agreement (ecological significance threshold)
        aic_concordant = self.aic_difference < 2.0
        
        # Check likelihood agreement
        likelihood_concordant = self.likelihood_relative_difference_pct < 1.0
        
        # Check parameter agreement
        params_good = self.parameter_pass_rate >= 0.8
        critical_params_good = self.critical_parameter_pass_rate >= 0.9
        
        # Determine overall status
        if aic_concordant and likelihood_concordant and params_good and critical_params_good:
            self.overall_status = ComparisonStatus.EXCELLENT
            self.recommendations.append("Model results show excellent agreement - validation passed")
        elif (aic_concordant or likelihood_concordant) and params_good:
            self.overall_status = ComparisonStatus.GOOD  
            self.recommendations.append("Model results show good agreement - minor discrepancies acceptable")
        elif params_good and critical_params_good:
            self.overall_status = ComparisonStatus.ACCEPTABLE
            self.recommendations.append("Parameter estimates acceptable, but check model fit statistics")
        elif critical_params_good:
            self.overall_status = ComparisonStatus.POOR
            self.recommendations.append("Critical parameters acceptable, but overall model has issues")
        else:
            self.overall_status = ComparisonStatus.FAILED
            self.recommendations.append("Significant discrepancies detected - review model implementation")
        
        # Add specific recommendations
        if not aic_concordant:
            self.recommendations.append(f"AIC difference ({self.aic_difference:.2f}) exceeds ecological significance threshold (2.0)")
        
        if not likelihood_concordant:
            self.recommendations.append(f"Log-likelihood difference ({self.likelihood_relative_difference_pct:.1f}%) exceeds 1% threshold")
        
        if self.parameter_pass_rate < 0.8:
            self.recommendations.append(f"Parameter pass rate ({self.parameter_pass_rate:.1%}) below 80% threshold")
        
        if not self.ranking_agreement and self.jax_aic_rank is not None:
            self.recommendations.append(f"Model ranking disagreement: JAX rank {self.jax_aic_rank}, RMark rank {self.rmark_aic_rank}")


def compare_parameter_estimates(
    jax_params: Dict[str, float],
    jax_std_errors: Dict[str, float],
    rmark_params: Dict[str, float], 
    rmark_std_errors: Dict[str, float],
    absolute_tolerance: float = 1e-3,
    relative_tolerance_pct: float = 5.0,
    confidence_level: float = 0.95
) -> List[ParameterComparisonResult]:
    """
    Compare parameter estimates between JAX and RMark implementations.
    
    Args:
        jax_params: Dictionary of JAX parameter estimates
        jax_std_errors: Dictionary of JAX standard errors
        rmark_params: Dictionary of RMark parameter estimates
        rmark_std_errors: Dictionary of RMark standard errors
        absolute_tolerance: Absolute tolerance for parameter differences
        relative_tolerance_pct: Relative tolerance percentage
        confidence_level: Confidence level for interval calculations
        
    Returns:
        List of parameter comparison results
    """
    logger.info(f"Comparing {len(jax_params)} JAX parameters with {len(rmark_params)} RMark parameters")
    
    results = []
    
    # Calculate confidence interval multiplier
    from scipy import stats
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha/2)
    
    # Compare common parameters
    common_params = set(jax_params.keys()) & set(rmark_params.keys())
    logger.info(f"Found {len(common_params)} common parameters for comparison")
    
    for param_name in common_params:
        # Extract parameter information
        jax_est = jax_params[param_name]
        jax_se = jax_std_errors.get(param_name, 0.0)
        rmark_est = rmark_params[param_name]
        rmark_se = rmark_std_errors.get(param_name, 0.0)
        
        # Calculate confidence intervals if standard errors available
        jax_ci = None
        rmark_ci = None
        
        if jax_se > 0:
            margin = z_score * jax_se
            jax_ci = (jax_est - margin, jax_est + margin)
        
        if rmark_se > 0:
            margin = z_score * rmark_se  
            rmark_ci = (rmark_est - margin, rmark_est + margin)
        
        # Determine parameter type from name
        param_type = "unknown"
        formula_term = None
        
        if "phi" in param_name.lower():
            param_type = "phi"
        elif "p" in param_name.lower():
            param_type = "p"
        elif "f" in param_name.lower():
            param_type = "f"
        
        if "intercept" in param_name.lower():
            formula_term = "intercept"
        elif any(term in param_name.lower() for term in ["sex", "age", "time", "year"]):
            # Extract term name
            for term in ["sex", "age", "time", "year"]:
                if term in param_name.lower():
                    formula_term = term
                    break
        
        # Create comparison result
        comparison = ParameterComparisonResult(
            parameter_name=param_name,
            parameter_type=param_type,
            formula_term=formula_term,
            jax_estimate=jax_est,
            jax_std_error=jax_se,
            jax_confidence_interval=jax_ci,
            rmark_estimate=rmark_est,
            rmark_std_error=rmark_se,
            rmark_confidence_interval=rmark_ci
        )
        
        # Apply tolerance tests
        comparison.within_absolute_tolerance = comparison.absolute_difference <= absolute_tolerance
        comparison.within_relative_tolerance = comparison.relative_difference_pct <= relative_tolerance_pct
        
        results.append(comparison)
        
        # Log comparison summary
        logger.debug(
            f"Parameter {param_name}: JAX={jax_est:.4f}, RMark={rmark_est:.4f}, "
            f"diff={comparison.absolute_difference:.4f} ({comparison.relative_difference_pct:.2f}%), "
            f"status={comparison.comparison_status.value}"
        )
    
    # Check for missing parameters
    jax_only = set(jax_params.keys()) - set(rmark_params.keys())
    rmark_only = set(rmark_params.keys()) - set(jax_params.keys())
    
    if jax_only:
        logger.warning(f"Parameters only in JAX results: {sorted(jax_only)}")
    if rmark_only:
        logger.warning(f"Parameters only in RMark results: {sorted(rmark_only)}")
    
    logger.info(f"Parameter comparison completed: {len(results)} comparisons")
    return results


def compare_model_results(
    jax_result: Dict[str, Any],
    rmark_result: Dict[str, Any],
    model_formula: str,
    absolute_tolerance: float = 1e-3,
    relative_tolerance_pct: float = 5.0
) -> ModelComparisonResult:
    """
    Compare complete model results between JAX and RMark.
    
    Args:
        jax_result: JAX optimization result dictionary
        rmark_result: RMark result dictionary  
        model_formula: Model formula specification
        absolute_tolerance: Absolute tolerance for parameters
        relative_tolerance_pct: Relative tolerance percentage
        
    Returns:
        Complete model comparison result
    """
    logger.info(f"Comparing model results for formula: {model_formula}")
    
    # Extract model fit statistics
    jax_aic = jax_result.get('aic', 0.0)
    jax_ll = jax_result.get('log_likelihood', 0.0)
    jax_n_params = jax_result.get('n_parameters', 0)
    jax_converged = jax_result.get('success', False)
    
    rmark_aic = rmark_result.get('aic', 0.0)  
    rmark_ll = rmark_result.get('log_likelihood', 0.0)
    rmark_n_params = rmark_result.get('n_parameters', 0)
    rmark_converged = rmark_result.get('converged', False)
    
    # Extract parameters
    jax_params = jax_result.get('parameters', {})
    jax_std_errors = jax_result.get('std_errors', {})
    rmark_params = rmark_result.get('parameters', {})
    rmark_std_errors = rmark_result.get('std_errors', {})
    
    # Compare parameters
    parameter_comparisons = compare_parameter_estimates(
        jax_params, jax_std_errors,
        rmark_params, rmark_std_errors,
        absolute_tolerance, relative_tolerance_pct
    )
    
    # Create model comparison result
    comparison = ModelComparisonResult(
        model_formula=model_formula,
        model_specification=jax_result.get('model_specification', {}),
        jax_aic=jax_aic,
        jax_log_likelihood=jax_ll,
        jax_n_parameters=jax_n_params,
        jax_convergence=jax_converged,
        rmark_aic=rmark_aic,
        rmark_log_likelihood=rmark_ll,
        rmark_n_parameters=rmark_n_params,
        rmark_convergence=rmark_converged,
        parameter_comparisons=parameter_comparisons
    )
    
    logger.info(
        f"Model comparison completed: AIC diff={comparison.aic_difference:.2f}, "
        f"param pass rate={comparison.parameter_pass_rate:.1%}, "
        f"status={comparison.overall_status.value}"
    )
    
    return comparison


def create_comparison_summary_table(
    comparisons: List[Union[ParameterComparisonResult, ModelComparisonResult]]
) -> pd.DataFrame:
    """Create summary table of comparison results."""
    
    if not comparisons:
        return pd.DataFrame()
    
    if isinstance(comparisons[0], ParameterComparisonResult):
        # Parameter comparison summary
        data = []
        for comp in comparisons:
            data.append({
                'Parameter': comp.parameter_name,
                'Type': comp.parameter_type,
                'JAX_Estimate': comp.jax_estimate,
                'RMark_Estimate': comp.rmark_estimate,
                'Absolute_Diff': comp.absolute_difference,
                'Relative_Diff_Pct': comp.relative_difference_pct,
                'CI_Overlap': comp.confidence_intervals_overlap,
                'Status': comp.comparison_status.value,
                'Precision': comp.precision_level
            })
        
        return pd.DataFrame(data)
    
    else:
        # Model comparison summary
        data = []
        for comp in comparisons:
            data.append({
                'Model_Formula': comp.model_formula,
                'JAX_AIC': comp.jax_aic,
                'RMark_AIC': comp.rmark_aic,
                'AIC_Diff': comp.aic_difference,
                'JAX_LogLik': comp.jax_log_likelihood,
                'RMark_LogLik': comp.rmark_log_likelihood,
                'LogLik_Diff_Pct': comp.likelihood_relative_difference_pct,
                'Param_Pass_Rate': comp.parameter_pass_rate,
                'Status': comp.overall_status.value
            })
        
        return pd.DataFrame(data)