"""
Tests for parameter comparison utilities.

These tests verify that parameter comparison functions correctly calculate
differences, confidence intervals, and quality assessments according to
industry standards for numerical validation.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict

from pradel_jax.validation.parameter_comparison import (
    ParameterComparisonResult,
    ModelComparisonResult,
    ComparisonStatus,
    compare_parameter_estimates,
    compare_model_results,
    create_comparison_summary_table
)


class TestParameterComparisonResult:
    """Test ParameterComparisonResult class."""
    
    def test_identical_parameters(self):
        """Test comparison of identical parameters."""
        result = ParameterComparisonResult(
            parameter_name="phi_intercept",
            parameter_type="phi",
            jax_estimate=0.8,
            jax_std_error=0.05,
            rmark_estimate=0.8,
            rmark_std_error=0.05
        )
        
        assert result.absolute_difference == 0.0
        assert result.relative_difference_pct == 0.0
        assert result.comparison_status == ComparisonStatus.EXCELLENT
        assert result.precision_level == "excellent"
    
    def test_small_difference(self):
        """Test comparison with small difference."""
        result = ParameterComparisonResult(
            parameter_name="p_intercept",
            parameter_type="p",
            jax_estimate=0.65,
            jax_std_error=0.04,
            rmark_estimate=0.653,
            rmark_std_error=0.041
        )
        
        assert result.absolute_difference == pytest.approx(0.003, abs=1e-6)
        assert result.relative_difference_pct == pytest.approx(0.46, abs=0.01)
        assert result.comparison_status == ComparisonStatus.GOOD
    
    def test_large_difference(self):
        """Test comparison with large difference."""
        result = ParameterComparisonResult(
            parameter_name="f_intercept",
            parameter_type="f", 
            jax_estimate=0.2,
            jax_std_error=0.03,
            rmark_estimate=0.25,
            rmark_std_error=0.035
        )
        
        assert result.absolute_difference == pytest.approx(0.05, abs=1e-6)
        assert result.relative_difference_pct == pytest.approx(20.0, abs=0.1)
        assert result.comparison_status == ComparisonStatus.POOR
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation and overlap."""
        result = ParameterComparisonResult(
            parameter_name="phi_sex",
            parameter_type="phi",
            jax_estimate=0.1,
            jax_std_error=0.05,
            rmark_estimate=0.12,
            rmark_std_error=0.048
        )
        
        # Check confidence intervals were calculated
        assert result.jax_confidence_interval is not None
        assert result.rmark_confidence_interval is not None
        
        # Check interval structure
        jax_ci = result.jax_confidence_interval
        rmark_ci = result.rmark_confidence_interval
        
        assert jax_ci[0] < result.jax_estimate < jax_ci[1]
        assert rmark_ci[0] < result.rmark_estimate < rmark_ci[1]
        
        # Should have overlap given small difference
        assert result.confidence_intervals_overlap is True
        assert result.overlap_proportion > 0
    
    def test_zero_standard_errors(self):
        """Test handling of zero standard errors."""
        result = ParameterComparisonResult(
            parameter_name="test_param",
            parameter_type="phi",
            jax_estimate=0.5,
            jax_std_error=0.0,
            rmark_estimate=0.5,
            rmark_std_error=0.0
        )
        
        # Should handle gracefully
        assert result.jax_confidence_interval is None
        assert result.rmark_confidence_interval is None
        assert result.standardized_difference == 0.0
    
    def test_division_by_zero_protection(self):
        """Test protection against division by zero in relative difference."""
        result = ParameterComparisonResult(
            parameter_name="zero_param",
            parameter_type="p",
            jax_estimate=0.0,
            jax_std_error=0.01,
            rmark_estimate=0.0,
            rmark_std_error=0.01
        )
        
        assert result.relative_difference_pct == 0.0
        assert result.comparison_status == ComparisonStatus.EXCELLENT
        
        # Test with one zero, one non-zero
        result2 = ParameterComparisonResult(
            parameter_name="near_zero_param",
            parameter_type="p",
            jax_estimate=1e-12,
            jax_std_error=0.01,
            rmark_estimate=0.1,
            rmark_std_error=0.01
        )
        
        # Should handle very small denominators
        assert result2.relative_difference_pct > 100  # Large relative difference


class TestCompareParameterEstimates:
    """Test compare_parameter_estimates function."""
    
    def test_basic_comparison(self):
        """Test basic parameter comparison."""
        jax_params = {
            'phi_intercept': 0.85,
            'p_intercept': 0.65,
            'f_intercept': 0.25
        }
        jax_se = {
            'phi_intercept': 0.05,
            'p_intercept': 0.04,
            'f_intercept': 0.03
        }
        rmark_params = {
            'phi_intercept': 0.851,
            'p_intercept': 0.648,
            'f_intercept': 0.252
        }
        rmark_se = {
            'phi_intercept': 0.051,
            'p_intercept': 0.041,
            'f_intercept': 0.031
        }
        
        results = compare_parameter_estimates(
            jax_params, jax_se, rmark_params, rmark_se
        )
        
        assert len(results) == 3
        
        # Check that all comparisons are high quality
        for result in results:
            assert result.comparison_status in [ComparisonStatus.EXCELLENT, ComparisonStatus.GOOD]
            assert result.parameter_type in ['phi', 'p', 'f']
    
    def test_missing_parameters(self):
        """Test handling of missing parameters."""
        jax_params = {
            'phi_intercept': 0.85,
            'p_intercept': 0.65,
            'jax_only_param': 0.1
        }
        jax_se = {
            'phi_intercept': 0.05,
            'p_intercept': 0.04,
            'jax_only_param': 0.02
        }
        rmark_params = {
            'phi_intercept': 0.85,
            'p_intercept': 0.65,
            'rmark_only_param': 0.2
        }
        rmark_se = {
            'phi_intercept': 0.05,
            'p_intercept': 0.04,
            'rmark_only_param': 0.03
        }
        
        results = compare_parameter_estimates(
            jax_params, jax_se, rmark_params, rmark_se
        )
        
        # Should only compare common parameters
        assert len(results) == 2
        param_names = [r.parameter_name for r in results]
        assert 'phi_intercept' in param_names
        assert 'p_intercept' in param_names
        assert 'jax_only_param' not in param_names
        assert 'rmark_only_param' not in param_names
    
    def test_tolerance_settings(self):
        """Test custom tolerance settings."""
        jax_params = {'param1': 1.0}
        jax_se = {'param1': 0.1}
        rmark_params = {'param1': 1.002}  # 0.2% difference
        rmark_se = {'param1': 0.1}
        
        # Test with strict tolerance
        results_strict = compare_parameter_estimates(
            jax_params, jax_se, rmark_params, rmark_se,
            absolute_tolerance=0.001,
            relative_tolerance_pct=0.1
        )
        
        assert len(results_strict) == 1
        result = results_strict[0]
        assert result.within_absolute_tolerance is False  # 0.002 > 0.001
        assert result.within_relative_tolerance is False  # 0.2% > 0.1%
        
        # Test with loose tolerance
        results_loose = compare_parameter_estimates(
            jax_params, jax_se, rmark_params, rmark_se,
            absolute_tolerance=0.01,
            relative_tolerance_pct=1.0
        )
        
        result = results_loose[0]
        assert result.within_absolute_tolerance is True
        assert result.within_relative_tolerance is True
    
    def test_parameter_type_detection(self):
        """Test automatic parameter type detection."""
        params = {
            'Phi:(Intercept)': 0.8,
            'p:sex': 0.1,
            'f:age': -0.05,
            'unknown_param': 0.5
        }
        se = {k: 0.05 for k in params.keys()}
        
        results = compare_parameter_estimates(params, se, params, se)
        
        type_map = {r.parameter_name: r.parameter_type for r in results}
        assert type_map['Phi:(Intercept)'] == 'phi'
        assert type_map['p:sex'] == 'p'
        assert type_map['f:age'] == 'f'
        assert type_map['unknown_param'] == 'unknown'


class TestModelComparisonResult:
    """Test ModelComparisonResult class."""
    
    def test_excellent_model_agreement(self):
        """Test model with excellent agreement."""
        parameter_comparisons = [
            ParameterComparisonResult(
                parameter_name='phi_intercept',
                parameter_type='phi',
                jax_estimate=0.85,
                rmark_estimate=0.851,
                comparison_status=ComparisonStatus.EXCELLENT
            ),
            ParameterComparisonResult(
                parameter_name='p_intercept', 
                parameter_type='p',
                jax_estimate=0.65,
                rmark_estimate=0.651,
                comparison_status=ComparisonStatus.GOOD
            )
        ]
        
        result = ModelComparisonResult(
            model_formula="phi(1), p(1), f(1)",
            jax_aic=150.5,
            jax_log_likelihood=-72.25,
            jax_convergence=True,
            rmark_aic=150.7,
            rmark_log_likelihood=-72.35,
            rmark_convergence=True,
            parameter_comparisons=parameter_comparisons
        )
        
        assert result.aic_difference == pytest.approx(0.2, abs=0.01)
        assert result.likelihood_difference == pytest.approx(0.1, abs=0.01)
        assert result.parameter_pass_rate == 1.0  # Both parameters passed
        assert result.overall_status == ComparisonStatus.EXCELLENT
    
    def test_poor_model_agreement(self):
        """Test model with poor agreement."""
        parameter_comparisons = [
            ParameterComparisonResult(
                parameter_name='phi_intercept',
                parameter_type='phi',
                jax_estimate=0.85,
                rmark_estimate=0.75,  # Large difference
                comparison_status=ComparisonStatus.FAILED
            ),
            ParameterComparisonResult(
                parameter_name='p_intercept',
                parameter_type='p', 
                jax_estimate=0.65,
                rmark_estimate=0.45,  # Large difference
                comparison_status=ComparisonStatus.POOR
            )
        ]
        
        result = ModelComparisonResult(
            model_formula="phi(1), p(1), f(1)",
            jax_aic=150.0,
            jax_log_likelihood=-72.0,
            jax_convergence=True,
            rmark_aic=160.0,  # Large AIC difference
            rmark_log_likelihood=-77.0,  # Large likelihood difference
            rmark_convergence=True,
            parameter_comparisons=parameter_comparisons
        )
        
        assert result.aic_difference == 10.0
        assert result.parameter_pass_rate == 0.0  # No parameters passed
        assert result.overall_status == ComparisonStatus.FAILED
        assert any("AIC difference" in rec for rec in result.recommendations)
    
    def test_critical_parameter_detection(self):
        """Test critical parameter identification."""
        parameter_comparisons = [
            ParameterComparisonResult(
                parameter_name='phi_intercept',  # Critical
                parameter_type='phi',
                formula_term='intercept',
                comparison_status=ComparisonStatus.EXCELLENT
            ),
            ParameterComparisonResult(
                parameter_name='phi_sex',  # Non-critical
                parameter_type='phi',
                formula_term='sex',
                comparison_status=ComparisonStatus.FAILED
            )
        ]
        
        result = ModelComparisonResult(
            model_formula="phi(sex), p(1), f(1)",
            jax_aic=150.0,
            rmark_aic=150.1,
            jax_log_likelihood=-72.0,
            rmark_log_likelihood=-72.05,
            jax_convergence=True,
            rmark_convergence=True,
            parameter_comparisons=parameter_comparisons
        )
        
        # Critical parameter pass rate should be 100% (intercept passed)
        assert result.critical_parameter_pass_rate == 1.0
        # Overall pass rate should be 50% (1 of 2 parameters)
        assert result.parameter_pass_rate == 0.5


class TestCompareModelResults:
    """Test compare_model_results function."""
    
    def test_complete_model_comparison(self):
        """Test complete model result comparison."""
        jax_result = {
            'aic': 150.5,
            'log_likelihood': -72.25,
            'n_parameters': 3,
            'success': True,
            'parameters': {
                'phi_intercept': 0.85,
                'p_intercept': 0.65,
                'f_intercept': 0.25
            },
            'std_errors': {
                'phi_intercept': 0.05,
                'p_intercept': 0.04,
                'f_intercept': 0.03
            },
            'model_specification': {
                'phi': '1',
                'p': '1', 
                'f': '1'
            }
        }
        
        rmark_result = {
            'aic': 150.7,
            'log_likelihood': -72.35,
            'n_parameters': 3,
            'converged': True,
            'parameters': {
                'phi_intercept': 0.851,
                'p_intercept': 0.649,
                'f_intercept': 0.252
            },
            'std_errors': {
                'phi_intercept': 0.051,
                'p_intercept': 0.041,
                'f_intercept': 0.031
            }
        }
        
        result = compare_model_results(
            jax_result, rmark_result, "phi(1), p(1), f(1)"
        )
        
        assert result.model_formula == "phi(1), p(1), f(1)"
        assert result.jax_aic == 150.5
        assert result.rmark_aic == 150.7
        assert result.aic_difference == pytest.approx(0.2, abs=0.01)
        assert len(result.parameter_comparisons) == 3
        assert result.overall_status in [ComparisonStatus.EXCELLENT, ComparisonStatus.GOOD]
    
    def test_convergence_handling(self):
        """Test handling of convergence status."""
        jax_result = {
            'aic': 150.0,
            'log_likelihood': -72.0,
            'success': False,  # JAX failed
            'parameters': {},
            'std_errors': {}
        }
        
        rmark_result = {
            'aic': 150.0,
            'log_likelihood': -72.0,
            'converged': True,  # RMark succeeded
            'parameters': {},
            'std_errors': {}
        }
        
        result = compare_model_results(
            jax_result, rmark_result, "test_formula"
        )
        
        assert result.jax_convergence is False
        assert result.rmark_convergence is True


class TestCreateComparisonSummaryTable:
    """Test summary table creation."""
    
    def test_parameter_summary_table(self):
        """Test parameter comparison summary table."""
        comparisons = [
            ParameterComparisonResult(
                parameter_name='phi_intercept',
                parameter_type='phi',
                jax_estimate=0.85,
                rmark_estimate=0.851,
                absolute_difference=0.001,
                relative_difference_pct=0.12,
                confidence_intervals_overlap=True,
                comparison_status=ComparisonStatus.EXCELLENT,
                precision_level='excellent'
            ),
            ParameterComparisonResult(
                parameter_name='p_intercept',
                parameter_type='p',
                jax_estimate=0.65,
                rmark_estimate=0.649,
                absolute_difference=0.001,
                relative_difference_pct=0.15,
                confidence_intervals_overlap=True,
                comparison_status=ComparisonStatus.GOOD,
                precision_level='good'
            )
        ]
        
        table = create_comparison_summary_table(comparisons)
        
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        assert 'Parameter' in table.columns
        assert 'JAX_Estimate' in table.columns
        assert 'RMark_Estimate' in table.columns
        assert 'Status' in table.columns
        
        # Check values
        phi_row = table[table['Parameter'] == 'phi_intercept'].iloc[0]
        assert phi_row['JAX_Estimate'] == 0.85
        assert phi_row['Status'] == 'excellent'
    
    def test_model_summary_table(self):
        """Test model comparison summary table."""
        comparisons = [
            ModelComparisonResult(
                model_formula="phi(1), p(1), f(1)",
                jax_aic=150.5,
                rmark_aic=150.7,
                aic_difference=0.2,
                jax_log_likelihood=-72.25,
                rmark_log_likelihood=-72.35,
                likelihood_relative_difference_pct=0.14,
                parameter_pass_rate=1.0,
                overall_status=ComparisonStatus.EXCELLENT
            )
        ]
        
        table = create_comparison_summary_table(comparisons)
        
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 1
        assert 'Model_Formula' in table.columns
        assert 'JAX_AIC' in table.columns
        assert 'RMark_AIC' in table.columns
        assert 'Status' in table.columns
        
        row = table.iloc[0]
        assert row['Model_Formula'] == "phi(1), p(1), f(1)"
        assert row['JAX_AIC'] == 150.5
        assert row['Status'] == 'excellent'
    
    def test_empty_comparison_list(self):
        """Test handling of empty comparison list."""
        table = create_comparison_summary_table([])
        
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 0


if __name__ == "__main__":
    pytest.main([__file__])