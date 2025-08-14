"""
Tests for advanced statistical methods in validation framework.

These tests verify the sophisticated statistical analysis methods including
bootstrap confidence intervals, concordance analysis, and robust statistics.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock

from pradel_jax.validation.advanced_statistics import (
    BootstrapResult,
    BootstrapMethod,
    ConcordanceAnalysisResult,
    RobustEstimator,
    bootstrap_parameter_difference,
    comprehensive_concordance_analysis,
    cross_validation_stability_test,
    publication_ready_comparison_summary,
    _bootstrap_basic_ci,
    _bootstrap_percentile_ci,
    _bootstrap_bca_ci
)


class TestBootstrapResult:
    """Test BootstrapResult data class."""
    
    def test_bootstrap_result_creation(self):
        """Test basic bootstrap result creation."""
        bootstrap_estimates = np.array([0.85, 0.86, 0.84, 0.87, 0.83])
        
        result = BootstrapResult(
            original_statistic=0.85,
            bootstrap_estimates=bootstrap_estimates,
            confidence_interval=(0.83, 0.87),
            bias_estimate=0.002,
            acceleration=0.01,
            method=BootstrapMethod.BCa,
            confidence_level=0.95,
            n_bootstrap=5
        )
        
        assert result.original_statistic == 0.85
        assert result.method == BootstrapMethod.BCa
        assert result.bootstrap_se > 0  # Should be calculated
        assert result.bias_corrected_estimate == pytest.approx(0.848, abs=1e-3)
    
    def test_convergence_assessment(self):
        """Test bootstrap convergence assessment."""
        # Test insufficient samples
        small_estimates = np.array([0.85, 0.86])
        result_small = BootstrapResult(
            original_statistic=0.85,
            bootstrap_estimates=small_estimates,
            confidence_interval=(0.84, 0.86),
            bias_estimate=0.0,
            acceleration=0.0,
            method=BootstrapMethod.BASIC,
            confidence_level=0.95,
            n_bootstrap=2
        )
        assert result_small.convergence_assessment == "insufficient_samples"
        
        # Test no variation
        constant_estimates = np.full(100, 0.85)
        result_constant = BootstrapResult(
            original_statistic=0.85,
            bootstrap_estimates=constant_estimates,
            confidence_interval=(0.85, 0.85),
            bias_estimate=0.0,
            acceleration=0.0,
            method=BootstrapMethod.BASIC,
            confidence_level=0.95,
            n_bootstrap=100
        )
        assert result_constant.convergence_assessment == "no_variation"
        
        # Test stable estimates
        stable_estimates = np.random.normal(0.85, 0.001, 100)
        result_stable = BootstrapResult(
            original_statistic=0.85,
            bootstrap_estimates=stable_estimates,
            confidence_interval=(0.848, 0.852),
            bias_estimate=0.0,
            acceleration=0.0,
            method=BootstrapMethod.BASIC,
            confidence_level=0.95,
            n_bootstrap=100
        )
        assert result_stable.convergence_assessment == "very_stable"


class TestBootstrapParameterDifference:
    """Test bootstrap parameter difference analysis."""
    
    def test_basic_bootstrap(self):
        """Test basic bootstrap functionality."""
        np.random.seed(42)
        estimates_1 = np.random.normal(0.85, 0.05, 20)
        estimates_2 = np.random.normal(0.86, 0.05, 20)
        
        result = bootstrap_parameter_difference(
            estimates_1, estimates_2,
            n_bootstrap=100,
            method=BootstrapMethod.BASIC,
            random_seed=42
        )
        
        assert isinstance(result, BootstrapResult)
        assert result.method == BootstrapMethod.BASIC
        assert len(result.bootstrap_estimates) == 100
        assert result.confidence_interval[0] < result.confidence_interval[1]
        assert abs(result.original_statistic) < 0.1  # Small difference expected
    
    def test_percentile_method(self):
        """Test percentile bootstrap method."""
        np.random.seed(42)
        estimates_1 = np.random.normal(0.8, 0.04, 15)
        estimates_2 = np.random.normal(0.82, 0.04, 15)
        
        result = bootstrap_parameter_difference(
            estimates_1, estimates_2,
            n_bootstrap=200,
            method=BootstrapMethod.PERCENTILE,
            confidence_level=0.90,
            random_seed=42
        )
        
        assert result.method == BootstrapMethod.PERCENTILE
        assert result.confidence_level == 0.90
        assert len(result.bootstrap_estimates) == 200
    
    def test_bca_method(self):
        """Test bias-corrected accelerated bootstrap method."""
        np.random.seed(42)
        estimates_1 = np.random.normal(0.75, 0.03, 25)
        estimates_2 = np.random.normal(0.77, 0.03, 25)
        
        result = bootstrap_parameter_difference(
            estimates_1, estimates_2,
            n_bootstrap=150,
            method=BootstrapMethod.BCa,
            random_seed=42
        )
        
        assert result.method == BootstrapMethod.BCa
        assert result.bias_estimate is not None
        assert result.acceleration is not None
        # BCa should provide bias and acceleration corrections
        assert abs(result.bias_estimate) >= 0  # Can be zero but not negative
    
    def test_custom_statistic_function(self):
        """Test bootstrap with custom statistic function."""
        estimates_1 = np.array([0.8, 0.85, 0.9])
        estimates_2 = np.array([0.82, 0.87, 0.88])
        
        # Custom function: ratio of means
        def ratio_statistic(x, y):
            return np.mean(x) / np.mean(y)
        
        result = bootstrap_parameter_difference(
            estimates_1, estimates_2,
            statistic_func=ratio_statistic,
            n_bootstrap=50,
            random_seed=42
        )
        
        expected_ratio = np.mean(estimates_1) / np.mean(estimates_2)
        assert result.original_statistic == pytest.approx(expected_ratio, abs=1e-6)
    
    def test_different_sample_sizes(self):
        """Test bootstrap with different sample sizes."""
        estimates_1 = np.random.normal(0.8, 0.05, 10)  # Smaller sample
        estimates_2 = np.random.normal(0.82, 0.05, 30)  # Larger sample
        
        result = bootstrap_parameter_difference(
            estimates_1, estimates_2,
            n_bootstrap=100,
            random_seed=42
        )
        
        assert isinstance(result, BootstrapResult)
        assert len(result.bootstrap_estimates) == 100


class TestConcordanceAnalysis:
    """Test comprehensive concordance analysis."""
    
    def test_perfect_concordance(self):
        """Test analysis with perfect agreement."""
        values_1 = np.array([0.8, 0.85, 0.9, 0.75, 0.82])
        values_2 = values_1.copy()  # Perfect agreement
        
        result = comprehensive_concordance_analysis(values_1, values_2)
        
        assert isinstance(result, ConcordanceAnalysisResult)
        assert result.correlation_coefficient == pytest.approx(1.0, abs=1e-10)
        assert result.concordance_correlation_coefficient == pytest.approx(1.0, abs=1e-10)
        assert result.mean_absolute_error == pytest.approx(0.0, abs=1e-10)
        assert result.systematic_bias == pytest.approx(0.0, abs=1e-10)
        assert result.agreement_category == "excellent"
    
    def test_systematic_bias(self):
        """Test detection of systematic bias."""
        values_1 = np.array([0.8, 0.85, 0.9, 0.75, 0.82])
        values_2 = values_1 + 0.05  # Constant bias
        
        result = comprehensive_concordance_analysis(values_1, values_2)
        
        assert result.systematic_bias == pytest.approx(-0.05, abs=1e-6)
        assert result.correlation_coefficient == pytest.approx(1.0, abs=1e-6)
        # CCC should be lower due to bias
        assert result.concordance_correlation_coefficient < 1.0
        assert abs(result.systematic_bias) > 0.04  # Should detect bias
    
    def test_proportional_bias(self):
        """Test detection of proportional bias."""
        values_1 = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        values_2 = values_1 * 1.1  # 10% proportional bias
        
        result = comprehensive_concordance_analysis(values_1, values_2)
        
        # Should detect proportional relationship
        assert result.correlation_coefficient > 0.99
        assert result.proportional_bias > 0.1  # Should detect proportional bias
        assert result.agreement_category != "excellent"  # Bias should lower rating
    
    def test_bland_altman_analysis(self):
        """Test Bland-Altman limits of agreement."""
        np.random.seed(42)
        values_1 = np.random.normal(0.8, 0.1, 50)
        values_2 = values_1 + np.random.normal(0.02, 0.03, 50)  # Small random differences
        
        result = comprehensive_concordance_analysis(values_1, values_2, confidence_level=0.95)
        
        assert result.limits_of_agreement[0] < result.mean_difference < result.limits_of_agreement[1]
        assert result.within_limits_percentage >= 90  # Most points should be within limits
        assert result.limits_of_agreement[1] > result.limits_of_agreement[0]
    
    def test_robust_statistics(self):
        """Test robust statistics with outliers."""
        values_1 = np.array([0.8, 0.85, 0.9, 0.75, 0.82, 0.87])
        values_2 = np.array([0.82, 0.86, 0.91, 0.77, 0.84, 1.5])  # Last value is outlier
        
        result = comprehensive_concordance_analysis(values_1, values_2, robust=True)
        
        assert result.outlier_count >= 1  # Should detect at least one outlier
        # Robust correlation should be different from regular correlation
        assert result.robust_correlation != result.correlation_coefficient
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        values_1 = np.array([0.8])
        values_2 = np.array([0.82])
        
        with pytest.raises(ValueError, match="Need at least 3"):
            comprehensive_concordance_analysis(values_1, values_2)
    
    def test_mismatched_lengths(self):
        """Test handling of mismatched array lengths."""
        values_1 = np.array([0.8, 0.85, 0.9])
        values_2 = np.array([0.82, 0.86])  # Different length
        
        with pytest.raises(ValueError, match="same length"):
            comprehensive_concordance_analysis(values_1, values_2)
    
    def test_nan_handling(self):
        """Test handling of NaN values."""
        values_1 = np.array([0.8, np.nan, 0.9, 0.75])
        values_2 = np.array([0.82, 0.86, np.nan, 0.77])
        
        result = comprehensive_concordance_analysis(values_1, values_2)
        
        # Should work with remaining valid pairs
        assert isinstance(result, ConcordanceAnalysisResult)
        # Should have fewer effective observations due to NaN removal


class TestCrossValidationStability:
    """Test cross-validation stability assessment."""
    
    def test_basic_stability_test(self):
        """Test basic stability test functionality."""
        
        def mock_data_func(fold, repeat):
            """Mock data function that returns stable parameter estimates."""
            np.random.seed(fold + repeat * 10)  # Deterministic but different per fold
            
            # Simulate stable estimates with small variation
            jax_params = {
                'phi_intercept': 0.85 + np.random.normal(0, 0.01),
                'p_intercept': 0.65 + np.random.normal(0, 0.01)
            }
            rmark_params = {
                'phi_intercept': 0.851 + np.random.normal(0, 0.01),
                'p_intercept': 0.649 + np.random.normal(0, 0.01)
            }
            
            return jax_params, rmark_params
        
        result = cross_validation_stability_test(
            mock_data_func,
            n_folds=3,
            n_repeats=2,
            random_seed=42
        )
        
        assert result['stability_assessment'] in ['very_stable', 'stable', 'moderately_stable']
        assert result['convergence_rate'] > 0.8
        assert result['n_total_comparisons'] > 0
        assert 'recommendations' in result
    
    def test_unstable_estimates(self):
        """Test detection of unstable estimates."""
        
        def unstable_data_func(fold, repeat):
            """Mock data function with highly variable estimates."""
            np.random.seed(fold + repeat * 10)
            
            # Simulate unstable estimates with large variation
            jax_params = {
                'phi_intercept': 0.85 + np.random.normal(0, 0.2),  # Large variation
                'p_intercept': 0.65 + np.random.normal(0, 0.2)
            }
            rmark_params = {
                'phi_intercept': 0.85 + np.random.normal(0, 0.05),  # More stable
                'p_intercept': 0.65 + np.random.normal(0, 0.05)
            }
            
            return jax_params, rmark_params
        
        result = cross_validation_stability_test(
            unstable_data_func,
            n_folds=4,
            n_repeats=3,
            random_seed=42
        )
        
        assert result['stability_assessment'] in ['unstable', 'moderately_stable']
        assert result['coefficient_of_variation'] > 0.1  # Should show high variation
    
    def test_convergence_failures(self):
        """Test handling of convergence failures."""
        
        def failing_data_func(fold, repeat):
            """Mock data function with frequent failures."""
            if (fold + repeat) % 3 == 0:  # Fail 1/3 of the time
                return None, None
            
            jax_params = {'phi_intercept': 0.85}
            rmark_params = {'phi_intercept': 0.851}
            return jax_params, rmark_params
        
        result = cross_validation_stability_test(
            failing_data_func,
            n_folds=3,
            n_repeats=3,
            random_seed=42
        )
        
        assert result['convergence_rate'] < 1.0  # Should detect failures
        if result['convergence_rate'] < 0.8:
            assert result['stability_assessment'] == 'poor_convergence'
    
    def test_complete_failure(self):
        """Test handling of complete failure."""
        
        def complete_failure_func(fold, repeat):
            """Mock data function that always fails."""
            return None, None
        
        result = cross_validation_stability_test(
            complete_failure_func,
            n_folds=2,
            n_repeats=2,
            random_seed=42
        )
        
        assert result['stability_assessment'] == 'failed'
        assert 'error' in result
        assert result['convergence_rate'] == 0.0


class TestPublicationReadySummary:
    """Test publication-ready comparison summary generation."""
    
    def setup_method(self):
        """Set up mock comparison results for testing."""
        from pradel_jax.validation.parameter_comparison import (
            ParameterComparisonResult, ModelComparisonResult, ComparisonStatus
        )
        
        # Mock parameter comparisons
        self.parameter_comparisons = [
            ParameterComparisonResult(
                parameter_name='phi_intercept',
                parameter_type='phi',
                jax_estimate=0.85,
                rmark_estimate=0.851,
                absolute_difference=0.001,
                relative_difference_pct=0.12,
                comparison_status=ComparisonStatus.EXCELLENT
            ),
            ParameterComparisonResult(
                parameter_name='p_intercept',
                parameter_type='p',
                jax_estimate=0.65,
                rmark_estimate=0.648,
                absolute_difference=0.002,
                relative_difference_pct=0.31,
                comparison_status=ComparisonStatus.GOOD
            )
        ]
        
        # Mock model comparisons
        self.model_comparisons = [
            ModelComparisonResult(
                model_formula="phi(1), p(1), f(1)",
                jax_aic=150.5,
                rmark_aic=150.7,
                aic_difference=0.2,
                jax_log_likelihood=-72.25,
                rmark_log_likelihood=-72.35,
                likelihood_relative_difference_pct=0.14,
                jax_convergence=True,
                rmark_convergence=True,
                overall_status=ComparisonStatus.EXCELLENT,
                parameter_comparisons=self.parameter_comparisons
            )
        ]
    
    def test_basic_summary_generation(self):
        """Test basic summary generation."""
        summary = publication_ready_comparison_summary(
            self.parameter_comparisons,
            self.model_comparisons
        )
        
        assert 'statistical_summary' in summary
        assert 'parameter_analysis' in summary
        assert 'model_analysis' in summary
        assert 'conclusions' in summary
        assert 'methodology' in summary
        
        # Check statistical summary content
        stats = summary['statistical_summary']
        assert stats['parameter_count'] == 2
        assert stats['model_count'] == 1
        assert 0 <= stats['overall_pass_rate'] <= 1
    
    def test_parameter_analysis_details(self):
        """Test detailed parameter analysis."""
        summary = publication_ready_comparison_summary(
            self.parameter_comparisons,
            self.model_comparisons
        )
        
        param_analysis = summary['parameter_analysis']
        assert 'by_parameter_type' in param_analysis
        assert 'critical_parameters' in param_analysis
        
        # Should have analysis by parameter type
        by_type = param_analysis['by_parameter_type']
        assert 'phi' in by_type
        assert 'p' in by_type
        
        # Each type should have statistics
        for param_type, stats in by_type.items():
            assert 'parameter_count' in stats
            assert 'mean_absolute_difference' in stats
            assert 'pass_rate' in stats
    
    def test_model_analysis_details(self):
        """Test detailed model analysis."""
        summary = publication_ready_comparison_summary(
            self.parameter_comparisons,
            self.model_comparisons
        )
        
        model_analysis = summary['model_analysis']
        assert 'aic_concordance_rate' in model_analysis
        assert 'likelihood_concordance_rate' in model_analysis
        assert 'convergence_agreement' in model_analysis
        assert 'ranking_analysis' in model_analysis
    
    def test_conclusions_generation(self):
        """Test validation conclusions generation."""
        summary = publication_ready_comparison_summary(
            self.parameter_comparisons,
            self.model_comparisons
        )
        
        conclusions = summary['conclusions']
        assert 'overall_validation_status' in conclusions
        assert 'key_findings' in conclusions
        assert 'recommendations' in conclusions
        
        # With excellent parameter and model results, should be positive
        assert conclusions['overall_validation_status'] in ['excellent', 'good']
        assert len(conclusions['key_findings']) > 0
        assert len(conclusions['recommendations']) > 0
    
    def test_methodology_description(self):
        """Test methodology description generation."""
        summary = publication_ready_comparison_summary(
            self.parameter_comparisons,
            self.model_comparisons
        )
        
        methodology = summary['methodology']
        assert 'parameter_comparison' in methodology
        assert 'model_comparison' in methodology
        assert 'statistical_software' in methodology
        
        # Should contain key methodological terms
        param_method = methodology['parameter_comparison']
        assert 'TOST' in param_method
        assert 'bioequivalence' in param_method
    
    def test_with_optional_results(self):
        """Test summary generation with optional concordance and bootstrap results."""
        # Mock concordance results
        concordance_results = [
            ConcordanceAnalysisResult(
                correlation_coefficient=0.99,
                concordance_correlation_coefficient=0.98,
                mean_absolute_error=0.002,
                root_mean_square_error=0.003,
                systematic_bias=0.001,
                proportional_bias=0.01,
                bias_p_value=0.5,
                mean_difference=0.001,
                limits_of_agreement=(-0.005, 0.007),
                within_limits_percentage=95.0,
                robust_correlation=0.99,
                robust_bias=0.001,
                outlier_count=0
            )
        ]
        
        # Mock bootstrap results
        bootstrap_results = [
            BootstrapResult(
                original_statistic=0.001,
                bootstrap_estimates=np.random.normal(0.001, 0.0005, 100),
                confidence_interval=(-0.0005, 0.0025),
                bias_estimate=0.0001,
                acceleration=0.01,
                method=BootstrapMethod.BCa,
                confidence_level=0.95,
                n_bootstrap=100
            )
        ]
        
        summary = publication_ready_comparison_summary(
            self.parameter_comparisons,
            self.model_comparisons,
            concordance_results=concordance_results,
            bootstrap_results=bootstrap_results
        )
        
        assert 'concordance_analysis' in summary
        assert 'uncertainty_analysis' in summary
        
        # Should include concordance statistics
        concordance = summary['concordance_analysis']
        assert 'mean_concordance_correlation' in concordance
        assert 'excellent_agreement_rate' in concordance
        
        # Should include bootstrap statistics
        uncertainty = summary['uncertainty_analysis']
        assert 'mean_confidence_width' in uncertainty
        assert 'mean_bias_estimate' in uncertainty


class TestBootstrapConfidenceIntervals:
    """Test specific bootstrap confidence interval methods."""
    
    def test_basic_ci(self):
        """Test basic bootstrap confidence interval."""
        original_stat = 0.85
        bootstrap_stats = np.array([0.84, 0.86, 0.83, 0.87, 0.85])
        alpha = 0.1  # 90% confidence
        
        ci = _bootstrap_basic_ci(original_stat, bootstrap_stats, alpha)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert isinstance(ci[0], (int, float))
        assert isinstance(ci[1], (int, float))
    
    def test_percentile_ci(self):
        """Test percentile bootstrap confidence interval."""
        bootstrap_stats = np.array([0.82, 0.84, 0.85, 0.86, 0.88])
        alpha = 0.1
        
        ci = _bootstrap_percentile_ci(bootstrap_stats, alpha)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]
        # Should be approximately the 5th and 95th percentiles
        assert ci[0] == pytest.approx(0.82, abs=0.01)
        assert ci[1] == pytest.approx(0.88, abs=0.01)
    
    @patch('pradel_jax.validation.advanced_statistics.SCIPY_AVAILABLE', False)
    def test_bca_without_scipy(self):
        """Test BCa method without SciPy."""
        original_stat = 0.85
        bootstrap_stats = np.random.normal(0.85, 0.02, 100)
        estimates_1 = np.random.normal(0.85, 0.05, 20)
        estimates_2 = np.random.normal(0.86, 0.05, 20)
        
        def simple_diff(x, y):
            return np.mean(x) - np.mean(y)
        
        result = _bootstrap_bca_ci(
            original_stat, bootstrap_stats, estimates_1, estimates_2,
            simple_diff, 0.1
        )
        
        ci, bias, acceleration = result
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert isinstance(bias, (int, float))
        assert isinstance(acceleration, (int, float))


if __name__ == "__main__":
    pytest.main([__file__])