"""
Tests for statistical testing components.

These tests verify that statistical tests follow industry standards for
bioequivalence testing, confidence interval analysis, and ranking concordance.
"""

import pytest
import numpy as np
from unittest.mock import patch

from pradel_jax.validation.statistical_tests import (
    StatisticalTestResult,
    EquivalenceTestResult,
    ConfidenceIntervalTestResult,
    ConcordanceTestResult,
    TestResult,
    test_parameter_equivalence,
    test_confidence_interval_overlap,
    calculate_model_ranking_concordance,
    calculate_effect_size,
    interpret_effect_size
)


class TestParameterEquivalence:
    """Test Two One-Sided Tests (TOST) for parameter equivalence."""
    
    def test_equivalent_parameters(self):
        """Test TOST with truly equivalent parameters."""
        result = test_parameter_equivalence(
            estimate_1=0.85,
            std_error_1=0.05,
            estimate_2=0.851,  # Very small difference
            std_error_2=0.05,
            equivalence_margin=0.05,  # ±5%
            alpha=0.05
        )
        
        assert isinstance(result, EquivalenceTestResult)
        assert result.test_name == "TOST_Equivalence"
        assert result.result == TestResult.EQUIVALENT
        assert result.p_value < 0.05  # Should reject null hypothesis of non-equivalence
        assert result.confidence_level == 0.95
        
        # Check TOST-specific fields
        assert result.equivalence_bounds is not None
        assert result.point_estimate == pytest.approx(-0.001, abs=1e-6)
        assert result.confidence_interval is not None
    
    def test_non_equivalent_parameters(self):
        """Test TOST with non-equivalent parameters."""
        result = test_parameter_equivalence(
            estimate_1=0.85,
            std_error_1=0.05,
            estimate_2=0.75,  # Large difference
            std_error_2=0.05,
            equivalence_margin=0.05,  # ±5%
            alpha=0.05
        )
        
        assert result.result == TestResult.NOT_EQUIVALENT
        assert result.p_value >= 0.05  # Should not reject null hypothesis
        assert result.point_estimate == pytest.approx(0.1, abs=1e-6)
    
    def test_boundary_case(self):
        """Test parameter difference exactly at equivalence boundary."""
        result = test_parameter_equivalence(
            estimate_1=1.0,
            std_error_1=0.02,
            estimate_2=1.05,  # Exactly 5% difference
            std_error_2=0.02,
            equivalence_margin=0.05,
            alpha=0.05
        )
        
        # Result may be inconclusive or non-equivalent depending on statistical power
        assert result.result in [TestResult.NOT_EQUIVALENT, TestResult.INCONCLUSIVE]
    
    def test_zero_standard_errors(self):
        """Test handling of zero standard errors."""
        result = test_parameter_equivalence(
            estimate_1=0.5,
            std_error_1=0.0,
            estimate_2=0.5,
            std_error_2=0.0,
            equivalence_margin=0.05
        )
        
        assert result.result == TestResult.INCONCLUSIVE
        assert "Cannot perform test with zero standard errors" in result.notes[0]
    
    def test_different_equivalence_margins(self):
        """Test with different equivalence margins."""
        # Test with strict margin
        result_strict = test_parameter_equivalence(
            estimate_1=0.85,
            std_error_1=0.05,
            estimate_2=0.86,  # 1.2% difference
            std_error_2=0.05,
            equivalence_margin=0.01  # ±1%
        )
        
        # Test with loose margin
        result_loose = test_parameter_equivalence(
            estimate_1=0.85,
            std_error_1=0.05,
            estimate_2=0.86,  # Same 1.2% difference
            std_error_2=0.05,
            equivalence_margin=0.05  # ±5%
        )
        
        # Same data should be non-equivalent with strict margin, equivalent with loose margin
        assert result_strict.result == TestResult.NOT_EQUIVALENT
        assert result_loose.result == TestResult.EQUIVALENT
    
    @patch('pradel_jax.validation.statistical_tests.SCIPY_AVAILABLE', False)
    def test_simple_equivalence_fallback(self):
        """Test simple equivalence test when SciPy not available."""
        result = test_parameter_equivalence(
            estimate_1=0.85,
            std_error_1=0.05,
            estimate_2=0.851,
            std_error_2=0.05,
            equivalence_margin=0.05
        )
        
        assert result.test_name == "Simple_Equivalence"
        assert result.result in [TestResult.EQUIVALENT, TestResult.NOT_EQUIVALENT]
        assert "Simplified test" in result.notes[0]


class TestConfidenceIntervalOverlap:
    """Test confidence interval overlap analysis."""
    
    def test_overlapping_intervals(self):
        """Test parameters with overlapping confidence intervals."""
        result = test_confidence_interval_overlap(
            estimate_1=0.85,
            std_error_1=0.05,
            estimate_2=0.87,  # Close estimate
            std_error_2=0.05,
            confidence_level=0.95
        )
        
        assert isinstance(result, ConfidenceIntervalTestResult)
        assert result.test_name == "Confidence_Interval_Overlap"
        assert result.result == TestResult.EQUIVALENT  # Should have substantial overlap
        assert result.overlap_proportion > 0.5
        assert result.separation_distance == 0.0
        
        # Check interval fields
        assert result.interval_1 is not None
        assert result.interval_2 is not None
        assert result.overlap_region is not None
    
    def test_non_overlapping_intervals(self):
        """Test parameters with non-overlapping confidence intervals."""
        result = test_confidence_interval_overlap(
            estimate_1=0.5,
            std_error_1=0.02,
            estimate_2=0.7,  # Far apart
            std_error_2=0.02,
            confidence_level=0.95
        )
        
        assert result.result == TestResult.NOT_EQUIVALENT
        assert result.overlap_proportion == 0.0
        assert result.overlap_region is None
        assert result.separation_distance > 0
        assert "Non-overlapping confidence intervals" in result.notes[0]
    
    def test_minimal_overlap(self):
        """Test parameters with minimal confidence interval overlap."""
        result = test_confidence_interval_overlap(
            estimate_1=0.6,
            std_error_1=0.03,
            estimate_2=0.65,  # Moderate separation
            std_error_2=0.03,
            confidence_level=0.95
        )
        
        # Should have minimal overlap
        if result.overlap_proportion > 0:
            assert result.result == TestResult.INCONCLUSIVE
            assert result.overlap_proportion < 0.5
            assert "Minimal confidence interval overlap" in result.notes[0]
    
    def test_different_confidence_levels(self):
        """Test with different confidence levels."""
        # Test with 90% confidence (wider intervals, more overlap)
        result_90 = test_confidence_interval_overlap(
            estimate_1=0.6,
            std_error_1=0.04,
            estimate_2=0.65,
            std_error_2=0.04,
            confidence_level=0.90
        )
        
        # Test with 99% confidence (narrower intervals, less overlap)
        result_99 = test_confidence_interval_overlap(
            estimate_1=0.6,
            std_error_1=0.04,
            estimate_2=0.65,
            std_error_2=0.04,
            confidence_level=0.99
        )
        
        # 90% intervals should have more overlap than 99% intervals
        if result_90.overlap_proportion > 0 and result_99.overlap_proportion > 0:
            assert result_90.overlap_proportion > result_99.overlap_proportion


class TestModelRankingConcordance:
    """Test model ranking concordance analysis."""
    
    def test_perfect_concordance(self):
        """Test rankings with perfect agreement."""
        rankings_1 = [1, 2, 3, 4, 5]
        rankings_2 = [1, 2, 3, 4, 5]
        
        result = calculate_model_ranking_concordance(rankings_1, rankings_2)
        
        assert isinstance(result, ConcordanceTestResult)
        assert result.test_name == "Ranking_Concordance"
        assert result.result == TestResult.EQUIVALENT
        assert result.kendall_tau == pytest.approx(1.0, abs=1e-6)
        assert result.perfect_agreement_rate == 1.0
        assert result.concordant_pairs == 10  # C(5,2) = 10 pairs
        assert result.discordant_pairs == 0
    
    def test_perfect_disagreement(self):
        """Test rankings with perfect disagreement."""
        rankings_1 = [1, 2, 3, 4, 5]
        rankings_2 = [5, 4, 3, 2, 1]  # Reverse order
        
        result = calculate_model_ranking_concordance(rankings_1, rankings_2)
        
        assert result.result == TestResult.NOT_EQUIVALENT
        assert result.kendall_tau == pytest.approx(-1.0, abs=1e-6)
        assert result.perfect_agreement_rate == 0.0
        assert result.concordant_pairs == 0
        assert result.discordant_pairs == 10  # All pairs are discordant
    
    def test_moderate_concordance(self):
        """Test rankings with moderate agreement."""
        rankings_1 = [1, 2, 3, 4, 5]
        rankings_2 = [1, 3, 2, 5, 4]  # Some swaps
        
        result = calculate_model_ranking_concordance(rankings_1, rankings_2)
        
        assert result.result in [TestResult.INCONCLUSIVE, TestResult.EQUIVALENT]
        assert 0.5 < result.kendall_tau < 1.0  # Positive but not perfect
        assert 0.2 < result.perfect_agreement_rate < 1.0  # Some exact matches
        assert result.concordant_pairs > result.discordant_pairs
    
    def test_tied_rankings(self):
        """Test rankings with ties."""
        rankings_1 = [1, 2, 2, 3, 3]  # Tied rankings
        rankings_2 = [1, 2, 3, 2, 3]
        
        result = calculate_model_ranking_concordance(rankings_1, rankings_2)
        
        assert result.tied_pairs > 0
        assert result.concordant_pairs + result.discordant_pairs + result.tied_pairs == 10
    
    def test_single_model(self):
        """Test with only one model (edge case)."""
        rankings_1 = [1]
        rankings_2 = [1]
        
        result = calculate_model_ranking_concordance(rankings_1, rankings_2)
        
        assert result.result == TestResult.EQUIVALENT
        assert result.test_statistic == 1.0
        assert "Perfect concordance: only one model" in result.notes[0]
    
    def test_length_mismatch(self):
        """Test error handling for mismatched ranking lengths."""
        rankings_1 = [1, 2, 3]
        rankings_2 = [1, 2]  # Different length
        
        result = calculate_model_ranking_concordance(rankings_1, rankings_2)
        
        assert result.result == TestResult.ERROR
        assert "Ranking vectors have different lengths" in result.notes[0]
    
    @patch('pradel_jax.validation.statistical_tests.SCIPY_AVAILABLE', False)
    def test_manual_kendall_tau_calculation(self):
        """Test manual Kendall's tau calculation when SciPy not available."""
        rankings_1 = [1, 2, 3, 4]
        rankings_2 = [1, 2, 3, 4]
        
        result = calculate_model_ranking_concordance(rankings_1, rankings_2)
        
        # Should still work with manual calculation
        assert result.kendall_tau == pytest.approx(1.0, abs=1e-6)
        assert result.perfect_agreement_rate == 1.0


class TestEffectSizeCalculation:
    """Test effect size calculation and interpretation."""
    
    def test_cohens_d_calculation(self):
        """Test Cohen's d calculation."""
        effect_size = calculate_effect_size(
            estimate_1=0.85,
            estimate_2=0.75,
            pooled_std_error=0.05,
            method="cohens_d"
        )
        
        expected = abs(0.85 - 0.75) / 0.05  # 0.1 / 0.05 = 2.0
        assert effect_size == pytest.approx(expected, abs=1e-6)
    
    def test_zero_pooled_error(self):
        """Test handling of zero pooled standard error."""
        effect_size = calculate_effect_size(
            estimate_1=0.85,
            estimate_2=0.75,
            pooled_std_error=0.0
        )
        
        assert effect_size == 0.0
    
    def test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        assert interpret_effect_size(0.1) == "negligible"
        assert interpret_effect_size(0.3) == "small"
        assert interpret_effect_size(0.6) == "medium"
        assert interpret_effect_size(1.2) == "large"
    
    def test_standardized_difference_method(self):
        """Test standardized difference method."""
        effect_size = calculate_effect_size(
            estimate_1=1.0,
            estimate_2=0.8,
            pooled_std_error=0.1,
            method="standardized_diff"
        )
        
        expected = abs(1.0 - 0.8) / 0.1  # 0.2 / 0.1 = 2.0
        assert effect_size == pytest.approx(expected, abs=1e-6)


class TestStatisticalTestResult:
    """Test base StatisticalTestResult class."""
    
    def test_basic_result_creation(self):
        """Test basic statistical test result creation."""
        result = StatisticalTestResult(
            test_name="Test",
            test_statistic=2.5,
            p_value=0.02,
            result=TestResult.EQUIVALENT,
            confidence_level=0.95,
            effect_size=0.5
        )
        
        assert result.test_name == "Test"
        assert result.test_statistic == 2.5
        assert result.p_value == 0.02
        assert result.result == TestResult.EQUIVALENT
        assert result.confidence_level == 0.95
        assert result.effect_size == 0.5
        assert result.notes == []  # Should initialize empty list
    
    def test_result_with_notes(self):
        """Test result creation with notes."""
        notes = ["Note 1", "Note 2"]
        result = StatisticalTestResult(
            test_name="Test",
            test_statistic=1.0,
            p_value=0.05,
            result=TestResult.INCONCLUSIVE,
            confidence_level=0.95,
            notes=notes
        )
        
        assert result.notes == notes


class TestEquivalenceTestResult:
    """Test EquivalenceTestResult specific functionality."""
    
    def test_p_value_calculation(self):
        """Test that overall p-value is maximum of two one-sided tests."""
        result = EquivalenceTestResult(
            test_name="TOST",
            test_statistic=0.0,  # Will be overridden
            p_value=0.0,  # Will be overridden
            result=TestResult.EQUIVALENT,
            confidence_level=0.95,
            lower_p_value=0.02,
            upper_p_value=0.03
        )
        
        assert result.p_value == 0.03  # Maximum of 0.02 and 0.03
        assert result.test_statistic == 0.0  # Will be updated in __post_init__
    
    def test_test_statistic_calculation(self):
        """Test that test statistic is maximum absolute value."""
        result = EquivalenceTestResult(
            test_name="TOST",
            test_statistic=0.0,
            p_value=0.0,
            result=TestResult.EQUIVALENT,
            confidence_level=0.95,
            lower_test_statistic=-2.5,
            upper_test_statistic=1.8
        )
        
        assert result.test_statistic == 2.5  # max(abs(-2.5), abs(1.8))


if __name__ == "__main__":
    pytest.main([__file__])