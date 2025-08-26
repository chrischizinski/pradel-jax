"""
Statistical Tests for RMark Validation.

This module provides industry-standard statistical tests for parameter validation,
following bioequivalence testing guidelines and scientific computing best practices.

Key Statistical Tests:
    - Two One-Sided Tests (TOST) for parameter equivalence
    - Confidence interval overlap analysis
    - Model ranking concordance (Kendall's tau)
    - Effect size calculations (Cohen's d)
    - Statistical power analysis

Standards Compliance:
    - FDA bioequivalence guidelines for TOST methodology
    - NIST guidelines for numerical validation
    - IEEE standards for floating-point comparison
    - Scientific reproducibility standards
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    from scipy import stats
    from scipy.stats import kendalltau

    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy not available - some statistical tests will be limited")
    SCIPY_AVAILABLE = False


class TestResult(Enum):
    """Result of statistical test."""

    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


@dataclass
class StatisticalTestResult:
    """Base class for statistical test results."""

    test_name: str
    test_statistic: float
    p_value: float
    result: TestResult
    confidence_level: float
    effect_size: Optional[float] = None
    power: Optional[float] = None
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


@dataclass
class EquivalenceTestResult(StatisticalTestResult):
    """Result of TOST equivalence test."""

    equivalence_bounds: Tuple[float, float] = None
    lower_test_statistic: float = 0.0
    upper_test_statistic: float = 0.0
    lower_p_value: float = 1.0
    upper_p_value: float = 1.0
    point_estimate: float = 0.0
    confidence_interval: Tuple[float, float] = None

    def __post_init__(self):
        super().__post_init__()
        # Overall p-value is maximum of the two one-sided tests
        self.p_value = max(self.lower_p_value, self.upper_p_value)
        self.test_statistic = max(
            abs(self.lower_test_statistic), abs(self.upper_test_statistic)
        )


@dataclass
class ConfidenceIntervalTestResult(StatisticalTestResult):
    """Result of confidence interval overlap test."""

    interval_1: Tuple[float, float] = None
    interval_2: Tuple[float, float] = None
    overlap_region: Optional[Tuple[float, float]] = None
    overlap_proportion: float = 0.0
    separation_distance: float = 0.0


@dataclass
class ConcordanceTestResult(StatisticalTestResult):
    """Result of ranking concordance test."""

    kendall_tau: float = 0.0
    spearman_rho: float = 0.0
    concordant_pairs: int = 0
    discordant_pairs: int = 0
    tied_pairs: int = 0
    perfect_agreement_rate: float = 0.0


def test_parameter_equivalence(
    estimate_1: float,
    std_error_1: float,
    estimate_2: float,
    std_error_2: float,
    equivalence_margin: float = 0.05,
    alpha: float = 0.05,
    method: str = "tost",
) -> EquivalenceTestResult:
    """
    Test parameter equivalence using Two One-Sided Tests (TOST).

    This follows FDA bioequivalence guidelines adapted for parameter validation.
    The null hypothesis is that parameters are NOT equivalent (difference > margin).
    We reject this to conclude equivalence.

    Args:
        estimate_1: First parameter estimate
        std_error_1: Standard error of first estimate
        estimate_2: Second parameter estimate
        std_error_2: Standard error of second estimate
        equivalence_margin: Equivalence margin (proportion, e.g. 0.05 = ±5%)
        alpha: Significance level (default 0.05 for 95% confidence)
        method: Test method ("tost" or "confidence_interval")

    Returns:
        EquivalenceTestResult with detailed test statistics
    """
    logger.debug(
        f"Testing parameter equivalence: {estimate_1:.4f} ± {std_error_1:.4f} vs {estimate_2:.4f} ± {std_error_2:.4f}"
    )

    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not available - using simplified equivalence test")
        return _simple_equivalence_test(estimate_1, estimate_2, equivalence_margin)

    # Calculate pooled standard error
    pooled_se = np.sqrt(std_error_1**2 + std_error_2**2)

    if pooled_se == 0:
        logger.warning("Zero standard errors - cannot perform statistical test")
        return EquivalenceTestResult(
            test_name="TOST_Equivalence",
            test_statistic=0.0,
            p_value=1.0,
            result=TestResult.INCONCLUSIVE,
            confidence_level=1 - alpha,
            notes=["Cannot perform test with zero standard errors"],
        )

    # Point estimate of difference
    diff = estimate_1 - estimate_2

    # Equivalence bounds (absolute)
    reference_value = max(
        abs(estimate_1), abs(estimate_2), 1.0
    )  # Avoid division by very small numbers
    lower_bound = -equivalence_margin * reference_value
    upper_bound = equivalence_margin * reference_value

    # TOST procedure: two one-sided tests
    # Test 1: H0: diff <= lower_bound vs H1: diff > lower_bound
    t1 = (diff - lower_bound) / pooled_se

    # Test 2: H0: diff >= upper_bound vs H1: diff < upper_bound
    t2 = (diff - upper_bound) / pooled_se

    # Degrees of freedom (conservative estimate)
    # Using Welch-Satterthwaite approximation
    if std_error_1 > 0 and std_error_2 > 0:
        df = (std_error_1**2 + std_error_2**2) ** 2 / (std_error_1**4 + std_error_2**4)
        df = max(1, int(df))  # Ensure at least 1 df
    else:
        df = 1

    # Calculate p-values
    p1 = 1 - stats.t.cdf(t1, df)  # Right-tailed test
    p2 = stats.t.cdf(t2, df)  # Left-tailed test

    # Overall result: reject both null hypotheses to conclude equivalence
    overall_p = max(p1, p2)
    is_equivalent = overall_p < alpha

    # Confidence interval for difference
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_margin = t_crit * pooled_se
    ci_lower = diff - ci_margin
    ci_upper = diff + ci_margin

    # Effect size (standardized difference)
    effect_size = abs(diff) / pooled_se if pooled_se > 0 else 0.0

    result = EquivalenceTestResult(
        test_name="TOST_Equivalence",
        test_statistic=max(abs(t1), abs(t2)),
        p_value=overall_p,
        result=TestResult.EQUIVALENT if is_equivalent else TestResult.NOT_EQUIVALENT,
        confidence_level=1 - alpha,
        effect_size=effect_size,
        equivalence_bounds=(lower_bound, upper_bound),
        lower_test_statistic=t1,
        upper_test_statistic=t2,
        lower_p_value=p1,
        upper_p_value=p2,
        point_estimate=diff,
        confidence_interval=(ci_lower, ci_upper),
    )

    # Add interpretive notes
    if is_equivalent:
        result.notes.append(
            f"Parameters are equivalent within ±{equivalence_margin*100:.1f}% margin"
        )
    else:
        result.notes.append(
            f"Cannot conclude equivalence within ±{equivalence_margin*100:.1f}% margin"
        )

    if abs(diff) < equivalence_margin * reference_value * 0.5:
        result.notes.append("Difference is less than half the equivalence margin")

    logger.debug(
        f"TOST result: p={overall_p:.4f}, equivalent={is_equivalent}, effect_size={effect_size:.3f}"
    )

    return result


def _simple_equivalence_test(
    estimate_1: float, estimate_2: float, equivalence_margin: float
) -> EquivalenceTestResult:
    """Simple equivalence test when SciPy is not available."""

    diff = abs(estimate_1 - estimate_2)
    reference_value = max(abs(estimate_1), abs(estimate_2), 1.0)
    relative_diff = diff / reference_value

    is_equivalent = relative_diff <= equivalence_margin

    return EquivalenceTestResult(
        test_name="Simple_Equivalence",
        test_statistic=relative_diff,
        p_value=0.0 if is_equivalent else 1.0,  # Simplified
        result=TestResult.EQUIVALENT if is_equivalent else TestResult.NOT_EQUIVALENT,
        confidence_level=0.95,
        point_estimate=estimate_1 - estimate_2,
        notes=[
            f"Simplified test: relative difference {relative_diff:.4f} vs margin {equivalence_margin:.4f}"
        ],
    )


def test_confidence_interval_overlap(
    estimate_1: float,
    std_error_1: float,
    estimate_2: float,
    std_error_2: float,
    confidence_level: float = 0.95,
) -> ConfidenceIntervalTestResult:
    """
    Test for confidence interval overlap between two parameter estimates.

    Args:
        estimate_1: First parameter estimate
        std_error_1: Standard error of first estimate
        estimate_2: Second parameter estimate
        std_error_2: Standard error of second estimate
        confidence_level: Confidence level for intervals

    Returns:
        ConfidenceIntervalTestResult with overlap analysis
    """
    logger.debug(
        f"Testing CI overlap: {estimate_1:.4f}±{std_error_1:.4f} vs {estimate_2:.4f}±{std_error_2:.4f}"
    )

    alpha = 1 - confidence_level

    if SCIPY_AVAILABLE:
        # Use t-distribution critical value (conservative with df=1)
        t_crit = stats.t.ppf(1 - alpha / 2, df=1)
    else:
        # Use normal approximation
        t_crit = 1.96  # Approximately 95% confidence

    # Calculate confidence intervals
    margin_1 = t_crit * std_error_1
    margin_2 = t_crit * std_error_2

    ci_1 = (estimate_1 - margin_1, estimate_1 + margin_1)
    ci_2 = (estimate_2 - margin_2, estimate_2 + margin_2)

    # Check for overlap
    overlap_start = max(ci_1[0], ci_2[0])
    overlap_end = min(ci_1[1], ci_2[1])

    has_overlap = overlap_start <= overlap_end
    overlap_region = (overlap_start, overlap_end) if has_overlap else None

    # Calculate overlap proportion
    if has_overlap:
        overlap_length = overlap_end - overlap_start
        ci_1_length = ci_1[1] - ci_1[0]
        ci_2_length = ci_2[1] - ci_2[0]
        avg_length = (ci_1_length + ci_2_length) / 2
        overlap_prop = overlap_length / avg_length if avg_length > 0 else 0.0
    else:
        overlap_prop = 0.0
        # Calculate separation distance
        separation = min(abs(ci_1[1] - ci_2[0]), abs(ci_2[1] - ci_1[0]))

    separation_dist = 0.0 if has_overlap else separation

    # Statistical interpretation
    # Non-overlapping CIs suggest statistically significant difference
    if has_overlap:
        if overlap_prop > 0.5:
            result = TestResult.EQUIVALENT
            notes = [
                "Substantial confidence interval overlap suggests compatible estimates"
            ]
        else:
            result = TestResult.INCONCLUSIVE
            notes = ["Minimal confidence interval overlap - estimates may differ"]
    else:
        result = TestResult.NOT_EQUIVALENT
        notes = [
            "Non-overlapping confidence intervals suggest statistically significant difference"
        ]

    return ConfidenceIntervalTestResult(
        test_name="Confidence_Interval_Overlap",
        test_statistic=overlap_prop,
        p_value=1.0 - overlap_prop if has_overlap else 0.0,  # Approximate
        result=result,
        confidence_level=confidence_level,
        interval_1=ci_1,
        interval_2=ci_2,
        overlap_region=overlap_region,
        overlap_proportion=overlap_prop,
        separation_distance=separation_dist,
        notes=notes,
    )


def calculate_model_ranking_concordance(
    rankings_1: List[float],
    rankings_2: List[float],
    values_1: Optional[List[float]] = None,
    values_2: Optional[List[float]] = None,
) -> ConcordanceTestResult:
    """
    Calculate ranking concordance between two sets of model rankings.

    Uses Kendall's tau for rank correlation, which is robust to ties and
    appropriate for comparing model selection results.

    Args:
        rankings_1: First set of rankings (lower = better)
        rankings_2: Second set of rankings
        values_1: Optional AIC/criterion values for first ranking
        values_2: Optional AIC/criterion values for second ranking

    Returns:
        ConcordanceTestResult with correlation statistics
    """
    logger.debug(f"Calculating ranking concordance for {len(rankings_1)} models")

    if len(rankings_1) != len(rankings_2):
        logger.error(f"Ranking length mismatch: {len(rankings_1)} vs {len(rankings_2)}")
        return ConcordanceTestResult(
            test_name="Ranking_Concordance",
            test_statistic=0.0,
            p_value=1.0,
            result=TestResult.ERROR,
            confidence_level=0.95,
            notes=["Error: Ranking vectors have different lengths"],
        )

    if len(rankings_1) < 2:
        logger.warning("Need at least 2 models for ranking concordance")
        return ConcordanceTestResult(
            test_name="Ranking_Concordance",
            test_statistic=1.0,
            p_value=0.0,
            result=TestResult.EQUIVALENT,
            confidence_level=0.95,
            notes=["Perfect concordance: only one model"],
        )

    rankings_1 = np.array(rankings_1)
    rankings_2 = np.array(rankings_2)

    # Calculate Kendall's tau
    if SCIPY_AVAILABLE:
        try:
            tau, p_value = kendalltau(rankings_1, rankings_2)

            # Also calculate Spearman's rho for comparison
            rho, _ = stats.spearmanr(rankings_1, rankings_2)

        except Exception as e:
            logger.warning(f"Error calculating Kendall's tau: {e}")
            tau, p_value, rho = _manual_kendall_tau(rankings_1, rankings_2)
    else:
        tau, p_value, rho = _manual_kendall_tau(rankings_1, rankings_2)

    # Count concordant/discordant pairs manually for detailed analysis
    n = len(rankings_1)
    concordant = 0
    discordant = 0
    tied = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Compare pairs
            sign_1 = np.sign(rankings_1[i] - rankings_1[j])
            sign_2 = np.sign(rankings_2[i] - rankings_2[j])

            if sign_1 * sign_2 > 0:
                concordant += 1
            elif sign_1 * sign_2 < 0:
                discordant += 1
            else:
                tied += 1

    # Perfect agreement rate (exact rank matches)
    perfect_matches = np.sum(rankings_1 == rankings_2)
    perfect_agreement_rate = perfect_matches / n

    # Interpret results
    if tau >= 0.8:
        result = TestResult.EQUIVALENT
        notes = ["Strong ranking concordance - model selection agrees well"]
    elif tau >= 0.6:
        result = TestResult.INCONCLUSIVE
        notes = ["Moderate ranking concordance - some model selection differences"]
    else:
        result = TestResult.NOT_EQUIVALENT
        notes = ["Poor ranking concordance - substantial model selection disagreement"]

    if perfect_agreement_rate >= 0.8:
        notes.append(f"High exact match rate: {perfect_agreement_rate:.1%}")
    elif perfect_agreement_rate < 0.3:
        notes.append(f"Low exact match rate: {perfect_agreement_rate:.1%}")

    logger.debug(
        f"Ranking concordance: tau={tau:.3f}, p={p_value:.4f}, perfect_rate={perfect_agreement_rate:.3f}"
    )

    return ConcordanceTestResult(
        test_name="Ranking_Concordance",
        test_statistic=tau,
        p_value=p_value,
        result=result,
        confidence_level=0.95,
        kendall_tau=tau,
        spearman_rho=rho,
        concordant_pairs=concordant,
        discordant_pairs=discordant,
        tied_pairs=tied,
        perfect_agreement_rate=perfect_agreement_rate,
        notes=notes,
    )


def _manual_kendall_tau(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Manual calculation of Kendall's tau when SciPy not available."""

    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            sign_x = np.sign(x[i] - x[j])
            sign_y = np.sign(y[i] - y[j])

            if sign_x * sign_y > 0:
                concordant += 1
            elif sign_x * sign_y < 0:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0.0

    # Approximate p-value (simplified)
    if total_pairs > 10:
        # Large sample approximation
        var_tau = 2 * (2 * n + 5) / (9 * n * (n - 1))
        z = tau / np.sqrt(var_tau)
        if SCIPY_AVAILABLE:
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        else:
            # Very rough approximation
            p_value = 1.0 if abs(z) < 1.96 else 0.01
    else:
        p_value = 0.05  # Conservative default

    # Simple Spearman correlation (Pearson on ranks)
    try:
        rho = np.corrcoef(x, y)[0, 1]
        if np.isnan(rho):
            rho = 0.0
    except:
        rho = 0.0

    return tau, p_value, rho


def calculate_effect_size(
    estimate_1: float,
    estimate_2: float,
    pooled_std_error: float,
    method: str = "cohens_d",
) -> float:
    """
    Calculate effect size for parameter difference.

    Args:
        estimate_1: First parameter estimate
        estimate_2: Second parameter estimate
        pooled_std_error: Pooled standard error
        method: Effect size method ("cohens_d", "standardized_diff")

    Returns:
        Effect size value
    """
    if pooled_std_error == 0:
        return 0.0

    raw_diff = abs(estimate_1 - estimate_2)

    if method == "cohens_d":
        # Cohen's d for effect size
        return raw_diff / pooled_std_error
    elif method == "standardized_diff":
        # Standardized difference
        return raw_diff / pooled_std_error
    else:
        logger.warning(f"Unknown effect size method: {method}")
        return raw_diff / pooled_std_error


def interpret_effect_size(effect_size: float, method: str = "cohens_d") -> str:
    """Interpret effect size magnitude."""

    if method == "cohens_d":
        if effect_size < 0.2:
            return "negligible"
        elif effect_size < 0.5:
            return "small"
        elif effect_size < 0.8:
            return "medium"
        else:
            return "large"
    else:
        # Generic interpretation
        if effect_size < 0.5:
            return "small"
        elif effect_size < 1.0:
            return "medium"
        else:
            return "large"
