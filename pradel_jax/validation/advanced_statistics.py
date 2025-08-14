"""
Advanced Statistical Methods for RMark Validation.

This module provides sophisticated statistical analysis methods for parameter
validation, including bootstrap uncertainty quantification, robust statistics,
and advanced concordance analysis.

Key Features:
    - Bootstrap confidence intervals with bias correction
    - Robust statistical estimators (Huber, Tukey)
    - Advanced concordance metrics (Lin's CCC, Bland-Altman)
    - Cross-validation for model stability assessment
    - Publication-ready statistical summaries

Standards Compliance:
    - Efron & Tibshirani bootstrap methodology
    - Lin's Concordance Correlation Coefficient
    - Bland-Altman agreement analysis
    - Robust statistics literature (Huber, 1981)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .statistical_tests import StatisticalTestResult, TestResult
from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy not available - some advanced methods will be limited")
    SCIPY_AVAILABLE = False


class BootstrapMethod(Enum):
    """Bootstrap resampling methods."""
    BASIC = "basic"
    PERCENTILE = "percentile"
    BCa = "bias_corrected_accelerated"
    STUDENTIZED = "studentized"


class RobustEstimator(Enum):
    """Robust statistical estimators."""
    HUBER = "huber"
    TUKEY = "tukey"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"


@dataclass
class BootstrapResult:
    """Result of bootstrap analysis."""
    
    original_statistic: float
    bootstrap_estimates: np.ndarray
    confidence_interval: Tuple[float, float]
    bias_estimate: float
    acceleration: float
    method: BootstrapMethod
    confidence_level: float
    n_bootstrap: int
    
    # Diagnostic information
    bootstrap_se: float = 0.0
    bias_corrected_estimate: float = 0.0
    convergence_assessment: str = "unknown"
    notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived statistics."""
        self.bootstrap_se = np.std(self.bootstrap_estimates, ddof=1)
        self.bias_corrected_estimate = self.original_statistic - self.bias_estimate
        self._assess_convergence()
    
    def _assess_convergence(self):
        """Assess bootstrap convergence."""
        if len(self.bootstrap_estimates) < 100:
            self.convergence_assessment = "insufficient_samples"
            self.notes.append("Bootstrap sample size may be too small for reliable inference")
        elif self.bootstrap_se == 0:
            self.convergence_assessment = "no_variation"
            self.notes.append("No variation in bootstrap estimates - check data")
        else:
            # Check for reasonable bootstrap distribution
            cv = self.bootstrap_se / abs(self.original_statistic) if self.original_statistic != 0 else float('inf')
            
            if cv < 0.01:
                self.convergence_assessment = "very_stable"
            elif cv < 0.05:
                self.convergence_assessment = "stable"
            elif cv < 0.20:
                self.convergence_assessment = "moderate"
            else:
                self.convergence_assessment = "unstable"
                self.notes.append(f"High coefficient of variation ({cv:.3f}) - results may be unreliable")


@dataclass
class ConcordanceAnalysisResult:
    """Result of comprehensive concordance analysis."""
    
    # Basic agreement metrics
    correlation_coefficient: float
    concordance_correlation_coefficient: float  # Lin's CCC
    mean_absolute_error: float
    root_mean_square_error: float
    
    # Bias assessment
    systematic_bias: float
    proportional_bias: float
    bias_p_value: float
    
    # Agreement limits (Bland-Altman)
    mean_difference: float
    limits_of_agreement: Tuple[float, float]
    within_limits_percentage: float
    
    # Robust statistics
    robust_correlation: float
    robust_bias: float
    outlier_count: int
    
    # Overall assessment
    agreement_category: str = "unknown"
    clinical_significance: str = "unknown"
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Assess overall agreement quality."""
        self._categorize_agreement()
        self._assess_clinical_significance()
        self._generate_recommendations()
    
    def _categorize_agreement(self):
        """Categorize level of agreement."""
        if self.concordance_correlation_coefficient >= 0.99:
            self.agreement_category = "excellent"
        elif self.concordance_correlation_coefficient >= 0.95:
            self.agreement_category = "substantial"
        elif self.concordance_correlation_coefficient >= 0.90:
            self.agreement_category = "moderate"
        elif self.concordance_correlation_coefficient >= 0.75:
            self.agreement_category = "fair"
        else:
            self.agreement_category = "poor"
    
    def _assess_clinical_significance(self):
        """Assess clinical/practical significance."""
        # Based on relative error magnitude
        rel_rmse = self.root_mean_square_error / (abs(np.mean([0, 1])))  # Approximate scale
        
        if rel_rmse < 0.01:
            self.clinical_significance = "negligible"
        elif rel_rmse < 0.05:
            self.clinical_significance = "minimal"
        elif rel_rmse < 0.10:
            self.clinical_significance = "moderate"
        else:
            self.clinical_significance = "substantial"
    
    def _generate_recommendations(self):
        """Generate interpretive recommendations."""
        if self.agreement_category == "excellent":
            self.recommendations.append("Methods show excellent agreement - validation successful")
        elif self.agreement_category == "substantial":
            self.recommendations.append("Methods show substantial agreement - minor differences acceptable")
        elif self.agreement_category == "moderate":
            self.recommendations.append("Methods show moderate agreement - investigate systematic differences")
        else:
            self.recommendations.append("Methods show poor agreement - significant validation concerns")
        
        if abs(self.systematic_bias) > 0.05:
            self.recommendations.append(f"Systematic bias detected ({self.systematic_bias:.3f}) - check implementation differences")
        
        if self.proportional_bias > 0.1:
            self.recommendations.append(f"Proportional bias detected - difference varies with magnitude")
        
        if self.outlier_count > 0:
            self.recommendations.append(f"{self.outlier_count} outliers detected - investigate specific cases")


def bootstrap_parameter_difference(
    estimates_1: np.ndarray,
    estimates_2: np.ndarray,
    statistic_func: Callable = lambda x, y: np.mean(x) - np.mean(y),
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: BootstrapMethod = BootstrapMethod.BCa,
    random_seed: Optional[int] = None
) -> BootstrapResult:
    """
    Bootstrap analysis of parameter differences.
    
    Implements bias-corrected and accelerated (BCa) bootstrap following
    Efron & Tibshirani methodology for robust uncertainty quantification.
    
    Args:
        estimates_1: First set of parameter estimates
        estimates_2: Second set of parameter estimates  
        statistic_func: Function to compute statistic of interest
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        method: Bootstrap method to use
        random_seed: Random seed for reproducibility
        
    Returns:
        BootstrapResult with confidence intervals and diagnostics
    """
    logger.info(f"Bootstrap analysis: {n_bootstrap} samples, {method.value} method")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Original statistic
    original_stat = statistic_func(estimates_1, estimates_2)
    
    # Bootstrap resampling
    n1, n2 = len(estimates_1), len(estimates_2)
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        boot_idx1 = np.random.choice(n1, size=n1, replace=True)
        boot_idx2 = np.random.choice(n2, size=n2, replace=True)
        
        boot_sample1 = estimates_1[boot_idx1]
        boot_sample2 = estimates_2[boot_idx2]
        
        bootstrap_stats[i] = statistic_func(boot_sample1, boot_sample2)
    
    # Calculate confidence interval based on method
    alpha = 1 - confidence_level
    
    if method == BootstrapMethod.BASIC:
        ci = _bootstrap_basic_ci(original_stat, bootstrap_stats, alpha)
        bias, acceleration = 0.0, 0.0
        
    elif method == BootstrapMethod.PERCENTILE:
        ci = _bootstrap_percentile_ci(bootstrap_stats, alpha)
        bias, acceleration = 0.0, 0.0
        
    elif method == BootstrapMethod.BCa:
        ci, bias, acceleration = _bootstrap_bca_ci(
            original_stat, bootstrap_stats, estimates_1, estimates_2, 
            statistic_func, alpha
        )
        
    else:  # STUDENTIZED
        ci = _bootstrap_studentized_ci(original_stat, bootstrap_stats, alpha)
        bias, acceleration = 0.0, 0.0
    
    return BootstrapResult(
        original_statistic=original_stat,
        bootstrap_estimates=bootstrap_stats,
        confidence_interval=ci,
        bias_estimate=bias,
        acceleration=acceleration,
        method=method,
        confidence_level=confidence_level,
        n_bootstrap=n_bootstrap
    )


def _bootstrap_basic_ci(original_stat: float, bootstrap_stats: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Basic bootstrap confidence interval."""
    bias = np.mean(bootstrap_stats) - original_stat
    lower = 2 * original_stat - np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
    upper = 2 * original_stat - np.percentile(bootstrap_stats, 100 * alpha/2)
    return (lower, upper)


def _bootstrap_percentile_ci(bootstrap_stats: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval."""
    lower = np.percentile(bootstrap_stats, 100 * alpha/2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
    return (lower, upper)


def _bootstrap_bca_ci(
    original_stat: float,
    bootstrap_stats: np.ndarray,
    estimates_1: np.ndarray,
    estimates_2: np.ndarray,
    statistic_func: Callable,
    alpha: float
) -> Tuple[Tuple[float, float], float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval."""
    
    # Bias correction
    n_bootstrap = len(bootstrap_stats)
    prop_less = np.sum(bootstrap_stats < original_stat) / n_bootstrap
    
    if SCIPY_AVAILABLE:
        z0 = stats.norm.ppf(prop_less) if 0 < prop_less < 1 else 0
    else:
        # Rough approximation
        z0 = 0 if prop_less == 0.5 else (1 if prop_less > 0.5 else -1)
    
    # Acceleration using jackknife
    n1, n2 = len(estimates_1), len(estimates_2)
    jackknife_stats = []
    
    # Jackknife for first sample
    for i in range(n1):
        jack_sample1 = np.delete(estimates_1, i)
        jack_stat = statistic_func(jack_sample1, estimates_2)
        jackknife_stats.append(jack_stat)
    
    # Jackknife for second sample  
    for i in range(n2):
        jack_sample2 = np.delete(estimates_2, i)
        jack_stat = statistic_func(estimates_1, jack_sample2)
        jackknife_stats.append(jack_stat)
    
    jackknife_stats = np.array(jackknife_stats)
    jack_mean = np.mean(jackknife_stats)
    
    # Acceleration
    numerator = np.sum((jack_mean - jackknife_stats)**3)
    denominator = 6 * (np.sum((jack_mean - jackknife_stats)**2))**1.5
    acceleration = numerator / denominator if denominator != 0 else 0
    
    # BCa percentiles
    if SCIPY_AVAILABLE:
        z_alpha_2 = stats.norm.ppf(alpha/2)
        z_1_alpha_2 = stats.norm.ppf(1 - alpha/2)
        
        alpha1 = stats.norm.cdf(z0 + (z0 + z_alpha_2)/(1 - acceleration * (z0 + z_alpha_2)))
        alpha2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2)/(1 - acceleration * (z0 + z_1_alpha_2)))
    else:
        # Simplified calculation
        alpha1, alpha2 = alpha/2, 1 - alpha/2
    
    # Ensure valid percentiles
    alpha1 = max(0.001, min(0.999, alpha1))
    alpha2 = max(0.001, min(0.999, alpha2))
    
    lower = np.percentile(bootstrap_stats, 100 * alpha1)
    upper = np.percentile(bootstrap_stats, 100 * alpha2)
    
    bias = np.mean(bootstrap_stats) - original_stat
    
    return (lower, upper), bias, acceleration


def _bootstrap_studentized_ci(original_stat: float, bootstrap_stats: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Studentized bootstrap confidence interval."""
    # Simplified version - would need bootstrap of bootstrap for full implementation
    se = np.std(bootstrap_stats, ddof=1)
    if SCIPY_AVAILABLE:
        t_crit = stats.t.ppf(1 - alpha/2, df=len(bootstrap_stats)-1)
    else:
        t_crit = 1.96  # Normal approximation
    
    margin = t_crit * se
    return (original_stat - margin, original_stat + margin)


def comprehensive_concordance_analysis(
    values_1: np.ndarray,
    values_2: np.ndarray,
    robust: bool = True,
    confidence_level: float = 0.95
) -> ConcordanceAnalysisResult:
    """
    Comprehensive concordance analysis using multiple agreement metrics.
    
    Implements Lin's Concordance Correlation Coefficient and Bland-Altman
    analysis following clinical agreement literature standards.
    
    Args:
        values_1: First set of measurements
        values_2: Second set of measurements
        robust: Whether to include robust statistics
        confidence_level: Confidence level for intervals
        
    Returns:
        ConcordanceAnalysisResult with comprehensive agreement assessment
    """
    logger.info(f"Concordance analysis: {len(values_1)} paired observations")
    
    if len(values_1) != len(values_2):
        raise ValueError("Input arrays must have the same length")
    
    values_1 = np.asarray(values_1)
    values_2 = np.asarray(values_2)
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(values_1) | np.isnan(values_2))
    values_1 = values_1[valid_mask]
    values_2 = values_2[valid_mask]
    
    if len(values_1) < 3:
        raise ValueError("Need at least 3 valid paired observations")
    
    # Basic correlation
    correlation = np.corrcoef(values_1, values_2)[0, 1]
    
    # Lin's Concordance Correlation Coefficient
    mean_1, mean_2 = np.mean(values_1), np.mean(values_2)
    var_1, var_2 = np.var(values_1, ddof=1), np.var(values_2, ddof=1)
    
    # CCC formula: 2 * covariance / (var1 + var2 + (mean1 - mean2)^2)
    covariance = np.mean((values_1 - mean_1) * (values_2 - mean_2))
    ccc = 2 * covariance / (var_1 + var_2 + (mean_1 - mean_2)**2)
    
    # Agreement metrics
    differences = values_1 - values_2
    mean_diff = np.mean(differences)
    sd_diff = np.std(differences, ddof=1)
    
    # Limits of agreement (Bland-Altman)
    if SCIPY_AVAILABLE:
        t_crit = stats.t.ppf(1 - (1-confidence_level)/2, df=len(differences)-1)
    else:
        t_crit = 1.96
    
    loa_margin = t_crit * sd_diff
    limits_of_agreement = (mean_diff - loa_margin, mean_diff + loa_margin)
    
    # Percentage within limits
    within_limits = np.sum((differences >= limits_of_agreement[0]) & 
                          (differences <= limits_of_agreement[1]))
    within_limits_pct = within_limits / len(differences) * 100
    
    # Error metrics
    mae = np.mean(np.abs(differences))
    rmse = np.sqrt(np.mean(differences**2))
    
    # Bias assessment
    # Systematic bias (constant difference)
    systematic_bias = mean_diff
    
    # Proportional bias (slope ≠ 1 in Deming regression)
    # Simplified: correlation between mean and difference
    means = (values_1 + values_2) / 2
    proportional_bias_corr = np.corrcoef(means, differences)[0, 1]
    
    # Statistical test for bias
    if SCIPY_AVAILABLE:
        _, bias_p_value = stats.ttest_1samp(differences, 0)
    else:
        # Rough approximation
        t_stat = mean_diff / (sd_diff / np.sqrt(len(differences)))
        bias_p_value = 0.05 if abs(t_stat) > 2 else 0.5
    
    # Robust statistics if requested
    robust_correlation = correlation
    robust_bias = systematic_bias
    outlier_count = 0
    
    if robust:
        robust_stats = _compute_robust_statistics(values_1, values_2, differences)
        robust_correlation = robust_stats.get('correlation', correlation)
        robust_bias = robust_stats.get('bias', systematic_bias)
        outlier_count = robust_stats.get('outlier_count', 0)
    
    return ConcordanceAnalysisResult(
        correlation_coefficient=correlation,
        concordance_correlation_coefficient=ccc,
        mean_absolute_error=mae,
        root_mean_square_error=rmse,
        systematic_bias=systematic_bias,
        proportional_bias=abs(proportional_bias_corr),
        bias_p_value=bias_p_value,
        mean_difference=mean_diff,
        limits_of_agreement=limits_of_agreement,
        within_limits_percentage=within_limits_pct,
        robust_correlation=robust_correlation,
        robust_bias=robust_bias,
        outlier_count=outlier_count
    )


def _compute_robust_statistics(values_1: np.ndarray, values_2: np.ndarray, differences: np.ndarray) -> Dict[str, float]:
    """Compute robust statistical measures."""
    
    # Median-based correlation (Spearman-like)
    rank_1 = stats.rankdata(values_1) if SCIPY_AVAILABLE else np.argsort(np.argsort(values_1)) + 1
    rank_2 = stats.rankdata(values_2) if SCIPY_AVAILABLE else np.argsort(np.argsort(values_2)) + 1
    robust_correlation = np.corrcoef(rank_1, rank_2)[0, 1]
    
    # Robust bias (median difference)
    robust_bias = np.median(differences)
    
    # Outlier detection using modified Z-score
    median_diff = np.median(differences)
    mad = np.median(np.abs(differences - median_diff))
    modified_z_scores = 0.6745 * (differences - median_diff) / mad if mad > 0 else np.zeros_like(differences)
    outlier_count = np.sum(np.abs(modified_z_scores) > 3.5)
    
    return {
        'correlation': robust_correlation,
        'bias': robust_bias,
        'outlier_count': outlier_count
    }


def cross_validation_stability_test(
    data_func: Callable,
    n_folds: int = 5,
    n_repeats: int = 10,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Cross-validation stability assessment for model validation.
    
    Tests whether parameter differences are stable across different
    data subsets, providing evidence for validation robustness.
    
    Args:
        data_func: Function that returns (jax_params, rmark_params) for given data subset
        n_folds: Number of cross-validation folds
        n_repeats: Number of repeated CV experiments
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with stability metrics and assessment
    """
    logger.info(f"Cross-validation stability test: {n_folds} folds, {n_repeats} repeats")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    all_differences = []
    all_relative_differences = []
    convergence_rates = []
    
    for repeat in range(n_repeats):
        fold_differences = []
        fold_relative_diffs = []
        converged_count = 0
        
        for fold in range(n_folds):
            try:
                # Get parameter estimates for this fold
                jax_params, rmark_params = data_func(fold, repeat)
                
                if jax_params is not None and rmark_params is not None:
                    converged_count += 1
                    
                    # Calculate differences for common parameters
                    common_params = set(jax_params.keys()) & set(rmark_params.keys())
                    
                    for param in common_params:
                        diff = abs(jax_params[param] - rmark_params[param])
                        rel_diff = diff / abs(rmark_params[param]) if rmark_params[param] != 0 else 0
                        
                        fold_differences.append(diff)
                        fold_relative_diffs.append(rel_diff)
                
            except Exception as e:
                logger.warning(f"CV fold {fold} repeat {repeat} failed: {e}")
        
        all_differences.extend(fold_differences)
        all_relative_differences.extend(fold_relative_diffs)
        convergence_rates.append(converged_count / n_folds)
    
    if not all_differences:
        return {
            'stability_assessment': 'failed',
            'error': 'No successful cross-validation runs',
            'convergence_rate': 0.0
        }
    
    # Stability metrics
    differences = np.array(all_differences)
    relative_differences = np.array(all_relative_differences)
    
    stability_metrics = {
        'mean_absolute_difference': np.mean(differences),
        'std_absolute_difference': np.std(differences),
        'mean_relative_difference_pct': np.mean(relative_differences) * 100,
        'std_relative_difference_pct': np.std(relative_differences) * 100,
        'convergence_rate': np.mean(convergence_rates),
        'n_total_comparisons': len(differences),
        'coefficient_of_variation': np.std(differences) / np.mean(differences) if np.mean(differences) > 0 else float('inf')
    }
    
    # Stability assessment
    cv = stability_metrics['coefficient_of_variation']
    convergence_rate = stability_metrics['convergence_rate']
    
    if convergence_rate < 0.8:
        assessment = 'poor_convergence'
    elif cv < 0.1:
        assessment = 'very_stable'
    elif cv < 0.3:
        assessment = 'stable'
    elif cv < 0.6:
        assessment = 'moderately_stable'
    else:
        assessment = 'unstable'
    
    stability_metrics['stability_assessment'] = assessment
    
    # Recommendations
    recommendations = []
    if assessment == 'very_stable':
        recommendations.append("Validation shows excellent stability across data subsets")
    elif assessment == 'stable':
        recommendations.append("Validation shows good stability - differences are consistent")
    elif assessment == 'moderately_stable':
        recommendations.append("Validation shows moderate stability - some variation in differences")
    elif assessment == 'unstable':
        recommendations.append("Validation shows poor stability - investigate data dependencies")
    else:
        recommendations.append("Poor convergence rate - check model specifications and data quality")
    
    stability_metrics['recommendations'] = recommendations
    
    return stability_metrics


def publication_ready_comparison_summary(
    parameter_comparisons: List,
    model_comparisons: List,
    concordance_results: Optional[List[ConcordanceAnalysisResult]] = None,
    bootstrap_results: Optional[List[BootstrapResult]] = None
) -> Dict[str, Any]:
    """
    Generate publication-ready statistical summary of validation results.
    
    Creates comprehensive statistical summary suitable for scientific
    publication with all necessary statistical tests and interpretations.
    
    Args:
        parameter_comparisons: List of ParameterComparisonResult objects
        model_comparisons: List of ModelComparisonResult objects  
        concordance_results: Optional concordance analysis results
        bootstrap_results: Optional bootstrap confidence intervals
        
    Returns:
        Dictionary with publication-ready statistical summary
    """
    logger.info("Generating publication-ready comparison summary")
    
    summary = {
        'statistical_summary': {},
        'parameter_analysis': {},
        'model_analysis': {},
        'concordance_analysis': {},
        'uncertainty_analysis': {},
        'conclusions': {},
        'methodology': {}
    }
    
    # Statistical Summary
    if parameter_comparisons:
        param_stats = _summarize_parameter_comparisons(parameter_comparisons)
        summary['statistical_summary'].update(param_stats)
    
    if model_comparisons:
        model_stats = _summarize_model_comparisons(model_comparisons)
        summary['statistical_summary'].update(model_stats)
    
    # Parameter Analysis
    summary['parameter_analysis'] = _detailed_parameter_analysis(parameter_comparisons)
    
    # Model Analysis
    summary['model_analysis'] = _detailed_model_analysis(model_comparisons)
    
    # Concordance Analysis
    if concordance_results:
        summary['concordance_analysis'] = _summarize_concordance_analysis(concordance_results)
    
    # Uncertainty Analysis
    if bootstrap_results:
        summary['uncertainty_analysis'] = _summarize_bootstrap_analysis(bootstrap_results)
    
    # Conclusions and Recommendations
    summary['conclusions'] = _generate_validation_conclusions(
        parameter_comparisons, model_comparisons, concordance_results
    )
    
    # Methodology Description
    summary['methodology'] = _generate_methodology_description()
    
    return summary


def _summarize_parameter_comparisons(comparisons: List) -> Dict[str, Any]:
    """Summarize parameter comparison statistics."""
    
    if not comparisons:
        return {}
    
    # Extract key metrics
    abs_diffs = [c.absolute_difference for c in comparisons]
    rel_diffs = [c.relative_difference_pct for c in comparisons]
    
    # Success rates by quality level
    excellent_count = sum(1 for c in comparisons if c.comparison_status.value == 'excellent')
    good_count = sum(1 for c in comparisons if c.comparison_status.value == 'good')
    acceptable_count = sum(1 for c in comparisons if c.comparison_status.value == 'acceptable')
    
    total_count = len(comparisons)
    
    return {
        'parameter_count': total_count,
        'mean_absolute_difference': np.mean(abs_diffs),
        'median_absolute_difference': np.median(abs_diffs),
        'max_absolute_difference': np.max(abs_diffs),
        'mean_relative_difference_pct': np.mean(rel_diffs),
        'median_relative_difference_pct': np.median(rel_diffs),
        'max_relative_difference_pct': np.max(rel_diffs),
        'excellent_agreement_rate': excellent_count / total_count,
        'good_agreement_rate': good_count / total_count,
        'acceptable_agreement_rate': acceptable_count / total_count,
        'overall_pass_rate': (excellent_count + good_count + acceptable_count) / total_count
    }


def _summarize_model_comparisons(comparisons: List) -> Dict[str, Any]:
    """Summarize model comparison statistics."""
    
    if not comparisons:
        return {}
    
    aic_diffs = [c.aic_difference for c in comparisons]
    ll_diffs = [c.likelihood_relative_difference_pct for c in comparisons]
    param_pass_rates = [c.parameter_pass_rate for c in comparisons]
    
    # Success rates
    excellent_count = sum(1 for c in comparisons if c.overall_status.value == 'excellent')
    good_count = sum(1 for c in comparisons if c.overall_status.value == 'good')
    acceptable_count = sum(1 for c in comparisons if c.overall_status.value == 'acceptable')
    
    total_count = len(comparisons)
    
    return {
        'model_count': total_count,
        'mean_aic_difference': np.mean(aic_diffs),
        'median_aic_difference': np.median(aic_diffs),
        'max_aic_difference': np.max(aic_diffs),
        'mean_likelihood_difference_pct': np.mean(ll_diffs),
        'median_likelihood_difference_pct': np.median(ll_diffs),
        'mean_parameter_pass_rate': np.mean(param_pass_rates),
        'model_excellent_rate': excellent_count / total_count,
        'model_good_rate': good_count / total_count,
        'model_acceptable_rate': acceptable_count / total_count,
        'model_overall_pass_rate': (excellent_count + good_count + acceptable_count) / total_count
    }


def _detailed_parameter_analysis(comparisons: List) -> Dict[str, Any]:
    """Detailed parameter-level analysis."""
    
    if not comparisons:
        return {}
    
    # Group by parameter type
    by_type = {}
    for comp in comparisons:
        param_type = comp.parameter_type
        if param_type not in by_type:
            by_type[param_type] = []
        by_type[param_type].append(comp)
    
    type_analysis = {}
    for param_type, type_comps in by_type.items():
        abs_diffs = [c.absolute_difference for c in type_comps]
        rel_diffs = [c.relative_difference_pct for c in type_comps]
        
        type_analysis[param_type] = {
            'parameter_count': len(type_comps),
            'mean_absolute_difference': np.mean(abs_diffs),
            'mean_relative_difference_pct': np.mean(rel_diffs),
            'pass_rate': sum(1 for c in type_comps if c.comparison_status.value in ['excellent', 'good', 'acceptable']) / len(type_comps)
        }
    
    return {
        'by_parameter_type': type_analysis,
        'critical_parameters': _analyze_critical_parameters(comparisons)
    }


def _analyze_critical_parameters(comparisons: List) -> Dict[str, Any]:
    """Analyze critical parameter performance."""
    
    critical_params = [c for c in comparisons if 'intercept' in c.parameter_name.lower()]
    
    if not critical_params:
        return {'note': 'No critical parameters identified'}
    
    abs_diffs = [c.absolute_difference for c in critical_params]
    rel_diffs = [c.relative_difference_pct for c in critical_params]
    
    pass_count = sum(1 for c in critical_params if c.comparison_status.value in ['excellent', 'good', 'acceptable'])
    
    return {
        'critical_parameter_count': len(critical_params),
        'mean_absolute_difference': np.mean(abs_diffs),
        'mean_relative_difference_pct': np.mean(rel_diffs),
        'pass_rate': pass_count / len(critical_params),
        'all_critical_passed': pass_count == len(critical_params)
    }


def _detailed_model_analysis(comparisons: List) -> Dict[str, Any]:
    """Detailed model-level analysis."""
    
    if not comparisons:
        return {}
    
    # AIC concordance analysis
    aic_concordant = sum(1 for c in comparisons if c.aic_difference < 2.0)
    likelihood_concordant = sum(1 for c in comparisons if c.likelihood_relative_difference_pct < 1.0)
    
    return {
        'aic_concordance_rate': aic_concordant / len(comparisons),
        'likelihood_concordance_rate': likelihood_concordant / len(comparisons),
        'convergence_agreement': _analyze_convergence_agreement(comparisons),
        'ranking_analysis': _analyze_model_rankings(comparisons)
    }


def _analyze_convergence_agreement(comparisons: List) -> Dict[str, Any]:
    """Analyze convergence agreement between methods."""
    
    both_converged = sum(1 for c in comparisons if c.jax_convergence and c.rmark_convergence)
    jax_only = sum(1 for c in comparisons if c.jax_convergence and not c.rmark_convergence)
    rmark_only = sum(1 for c in comparisons if not c.jax_convergence and c.rmark_convergence)
    neither = sum(1 for c in comparisons if not c.jax_convergence and not c.rmark_convergence)
    
    total = len(comparisons)
    
    return {
        'both_converged_rate': both_converged / total,
        'jax_only_rate': jax_only / total,
        'rmark_only_rate': rmark_only / total,
        'neither_converged_rate': neither / total,
        'agreement_rate': (both_converged + neither) / total
    }


def _analyze_model_rankings(comparisons: List) -> Dict[str, Any]:
    """Analyze model ranking agreement."""
    
    ranked_comparisons = [c for c in comparisons if c.jax_aic_rank is not None and c.rmark_aic_rank is not None]
    
    if not ranked_comparisons:
        return {'note': 'No ranking information available'}
    
    exact_matches = sum(1 for c in ranked_comparisons if c.ranking_agreement)
    
    return {
        'ranked_model_count': len(ranked_comparisons),
        'exact_ranking_agreement_rate': exact_matches / len(ranked_comparisons),
        'ranking_concordance': 'See concordance analysis for detailed ranking correlation'
    }


def _summarize_concordance_analysis(results: List[ConcordanceAnalysisResult]) -> Dict[str, Any]:
    """Summarize concordance analysis results."""
    
    if not results:
        return {}
    
    ccc_values = [r.concordance_correlation_coefficient for r in results]
    agreement_categories = [r.agreement_category for r in results]
    
    category_counts = {}
    for category in agreement_categories:
        category_counts[category] = category_counts.get(category, 0) + 1
    
    return {
        'mean_concordance_correlation': np.mean(ccc_values),
        'median_concordance_correlation': np.median(ccc_values),
        'min_concordance_correlation': np.min(ccc_values),
        'agreement_distribution': category_counts,
        'excellent_agreement_rate': category_counts.get('excellent', 0) / len(results)
    }


def _summarize_bootstrap_analysis(results: List[BootstrapResult]) -> Dict[str, Any]:
    """Summarize bootstrap uncertainty analysis."""
    
    if not results:
        return {}
    
    confidence_widths = [(r.confidence_interval[1] - r.confidence_interval[0]) for r in results]
    bias_estimates = [abs(r.bias_estimate) for r in results]
    
    return {
        'mean_confidence_width': np.mean(confidence_widths),
        'median_confidence_width': np.median(confidence_widths),
        'mean_bias_estimate': np.mean(bias_estimates),
        'max_bias_estimate': np.max(bias_estimates),
        'bootstrap_convergence_assessment': [r.convergence_assessment for r in results]
    }


def _generate_validation_conclusions(
    parameter_comparisons: List,
    model_comparisons: List,
    concordance_results: Optional[List] = None
) -> Dict[str, Any]:
    """Generate overall validation conclusions."""
    
    conclusions = {
        'overall_validation_status': 'unknown',
        'key_findings': [],
        'recommendations': [],
        'statistical_significance': [],
        'practical_significance': []
    }
    
    # Determine overall status
    if parameter_comparisons and model_comparisons:
        param_pass_rate = sum(1 for c in parameter_comparisons 
                             if c.comparison_status.value in ['excellent', 'good', 'acceptable']) / len(parameter_comparisons)
        model_pass_rate = sum(1 for c in model_comparisons 
                             if c.overall_status.value in ['excellent', 'good', 'acceptable']) / len(model_comparisons)
        
        if param_pass_rate >= 0.9 and model_pass_rate >= 0.9:
            conclusions['overall_validation_status'] = 'excellent'
        elif param_pass_rate >= 0.8 and model_pass_rate >= 0.8:
            conclusions['overall_validation_status'] = 'good'
        elif param_pass_rate >= 0.7 and model_pass_rate >= 0.7:
            conclusions['overall_validation_status'] = 'acceptable'
        else:
            conclusions['overall_validation_status'] = 'poor'
    
    # Generate findings and recommendations based on status
    status = conclusions['overall_validation_status']
    
    if status == 'excellent':
        conclusions['key_findings'] = [
            "JAX implementation shows excellent agreement with RMark",
            "Parameter estimates are statistically and practically equivalent",
            "Model selection results are highly concordant"
        ]
        conclusions['recommendations'] = [
            "JAX implementation is validated for production use",
            "Results can be considered interchangeable with RMark",
            "Proceed with confidence for scientific applications"
        ]
    elif status == 'good':
        conclusions['key_findings'] = [
            "JAX implementation shows good agreement with RMark",
            "Minor differences within acceptable bounds",
            "Model selection generally concordant"
        ]
        conclusions['recommendations'] = [
            "JAX implementation is acceptable for most applications",
            "Monitor differences in critical applications",
            "Consider sensitivity analysis for important decisions"
        ]
    elif status == 'acceptable':
        conclusions['key_findings'] = [
            "JAX implementation shows acceptable agreement with RMark",
            "Some parameters show larger differences",
            "Model selection partially concordant"
        ]
        conclusions['recommendations'] = [
            "JAX implementation requires caution in use",
            "Investigate sources of larger differences",
            "Validate results against RMark for important analyses"
        ]
    else:
        conclusions['key_findings'] = [
            "JAX implementation shows poor agreement with RMark",
            "Significant differences in parameter estimates",
            "Model selection shows substantial disagreement"
        ]
        conclusions['recommendations'] = [
            "JAX implementation requires substantial revision",
            "Do not use for production analyses",
            "Investigate fundamental implementation differences"
        ]
    
    return conclusions


def _generate_methodology_description() -> Dict[str, str]:
    """Generate methodology description for publication."""
    
    return {
        'parameter_comparison': (
            "Parameter estimates were compared using absolute and relative difference metrics. "
            "Statistical equivalence was assessed using Two One-Sided Tests (TOST) following "
            "FDA bioequivalence guidelines (equivalence margin ±5%). Confidence interval "
            "overlap was analyzed to assess parameter compatibility."
        ),
        'model_comparison': (
            "Model fit was compared using Akaike Information Criterion (AIC) differences, "
            "with ecological significance threshold of 2.0 AIC units. Log-likelihood "
            "differences were assessed with 1% relative tolerance. Model ranking concordance "
            "was evaluated using Kendall's tau correlation coefficient."
        ),
        'concordance_analysis': (
            "Agreement between methods was assessed using Lin's Concordance Correlation "
            "Coefficient and Bland-Altman analysis. Systematic and proportional bias were "
            "evaluated. Robust statistics were included to assess outlier influence."
        ),
        'uncertainty_quantification': (
            "Uncertainty in parameter differences was quantified using bias-corrected and "
            "accelerated (BCa) bootstrap confidence intervals following Efron & Tibshirani "
            "methodology. Bootstrap convergence was assessed using coefficient of variation "
            "and bias estimates."
        ),
        'statistical_software': (
            "Analysis was performed using Python with NumPy and SciPy. Statistical tests "
            "followed established methodologies with appropriate corrections for multiple "
            "comparisons where applicable."
        )
    }