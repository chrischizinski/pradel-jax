"""
Phase 2 Integration Tests for Advanced Statistical Testing Framework.

This module provides comprehensive integration tests that validate the complete
Phase 2 statistical framework using realistic capture-recapture datasets and
comprehensive statistical analysis workflows.

Test Coverage:
- End-to-end bootstrap confidence interval analysis
- Complete concordance analysis workflows  
- Cross-validation stability assessment
- Publication-ready reporting generation
- Multi-dataset validation scenarios
- Performance and convergence benchmarking
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import json

# Import validation framework components
from pradel_jax.validation.advanced_statistics import (
    BootstrapMethod,
    BootstrapResult,
    ConcordanceAnalysisResult,
    bootstrap_parameter_difference,
    comprehensive_concordance_analysis,
    cross_validation_stability_test,
    publication_ready_comparison_summary
)

from pradel_jax.validation.parameter_comparison import (
    ParameterComparisonResult,
    ModelComparisonResult,
    ComparisonStatus,
    compare_parameter_estimates
)

from pradel_jax.validation.statistical_tests import (
    test_parameter_equivalence,
    test_confidence_interval_overlap,
    TestResult
)

# Import core framework components
from pradel_jax.data.adapters import DataContext, CovariateInfo
from pradel_jax.formulas.spec import FormulaSpec
import jax.numpy as jnp


@pytest.fixture
def realistic_capture_data():
    """Create realistic capture-recapture dataset for testing."""
    np.random.seed(42)  # Reproducible results
    
    # Simulate 100 individuals across 5 capture occasions
    n_individuals = 100
    n_occasions = 5
    
    # Create realistic capture matrix with missing data patterns
    capture_matrix = np.random.binomial(1, 0.7, size=(n_individuals, n_occasions))
    
    # Ensure each individual has at least one capture
    for i in range(n_individuals):
        if capture_matrix[i, :].sum() == 0:
            capture_matrix[i, np.random.randint(n_occasions)] = 1
    
    # Create covariates
    sex = np.random.choice([1, 2], size=n_individuals)  # 1=male, 2=female
    age = np.random.choice([1, 2, 3], size=n_individuals)  # 1=juvenile, 2=adult, 3=old
    weight = np.random.normal(50, 10, size=n_individuals)  # Continuous covariate
    
    return DataContext(
        capture_matrix=jnp.array(capture_matrix),
        covariates={
            'sex': jnp.array(sex),
            'age': jnp.array(age),
            'weight': jnp.array(weight)
        },
        covariate_info={
            'sex': CovariateInfo(name='sex', dtype='int', is_categorical=True),
            'age': CovariateInfo(name='age', dtype='int', is_categorical=True),
            'weight': CovariateInfo(name='weight', dtype='float', is_categorical=False)
        },
        n_individuals=n_individuals,
        n_occasions=n_occasions
    )


@pytest.fixture
def complex_formula_specs():
    """Create various formula specifications for comprehensive testing."""
    return {
        'simple': FormulaSpec(phi="1", p="1", f="1"),
        'sex_effect': FormulaSpec(phi="1 + sex", p="1", f="1"),
        'age_sex_interaction': FormulaSpec(phi="1 + sex + age + sex:age", p="1 + sex", f="1"),
        'continuous_covariate': FormulaSpec(phi="1 + weight", p="1", f="1"),
        'complex_model': FormulaSpec(phi="1 + sex + age", p="1 + sex", f="1 + age")
    }


@pytest.fixture
def simulated_model_results():
    """Create simulated JAX and RMark model results for comparison."""
    np.random.seed(42)
    
    # Simulate parameter estimates that are close but not identical
    base_params = {
        'phi_intercept': 0.85,
        'phi_sex': 0.12,
        'phi_age': -0.08,
        'p_intercept': 0.65,
        'p_sex': 0.05,
        'f_intercept': 0.15,
        'f_age': -0.02
    }
    
    # JAX results (with small random variation)
    jax_params = {}
    jax_errors = {}
    for param, value in base_params.items():
        jax_params[param] = value + np.random.normal(0, 0.002)
        jax_errors[param] = abs(np.random.normal(0.03, 0.005))
    
    # RMark results (with slightly different random variation)
    rmark_params = {}
    rmark_errors = {}
    for param, value in base_params.items():
        rmark_params[param] = value + np.random.normal(0, 0.003)
        rmark_errors[param] = abs(np.random.normal(0.03, 0.005))
    
    return {
        'jax': {'parameters': jax_params, 'std_errors': jax_errors},
        'rmark': {'parameters': rmark_params, 'std_errors': rmark_errors}
    }


class TestBootstrapIntegration:
    """Integration tests for bootstrap confidence interval analysis."""
    
    def test_multi_parameter_bootstrap_analysis(self, simulated_model_results):
        """Test bootstrap analysis across multiple parameters."""
        jax_results = simulated_model_results['jax']
        rmark_results = simulated_model_results['rmark']
        
        bootstrap_results = []
        
        # Test all parameters using different bootstrap methods
        methods = [BootstrapMethod.BASIC, BootstrapMethod.PERCENTILE, BootstrapMethod.BCa]
        
        for i, (param_name, jax_est) in enumerate(jax_results['parameters'].items()):
            if param_name in rmark_results['parameters']:
                rmark_est = rmark_results['parameters'][param_name]
                
                # Simulate parameter estimates from multiple bootstrap samples
                jax_bootstrap = np.random.normal(jax_est, jax_results['std_errors'][param_name], 200)
                rmark_bootstrap = np.random.normal(rmark_est, rmark_results['std_errors'][param_name], 200)
                
                # Use different method for each parameter to test variety
                method = methods[i % len(methods)]
                
                result = bootstrap_parameter_difference(
                    jax_bootstrap, rmark_bootstrap,
                    n_bootstrap=150,
                    method=method,
                    confidence_level=0.95,
                    random_seed=42 + i
                )
                
                bootstrap_results.append(result)
        
        # Validate results
        assert len(bootstrap_results) == len(jax_results['parameters'])
        
        for result in bootstrap_results:
            assert isinstance(result, BootstrapResult)
            assert result.confidence_interval[0] < result.confidence_interval[1]
            assert len(result.bootstrap_estimates) == 150
            assert result.convergence_assessment in ['very_stable', 'stable', 'moderately_stable']
            
            # Bias-corrected estimate should be reasonable
            assert abs(result.bias_corrected_estimate - result.original_statistic) < 0.1
    
    def test_bootstrap_method_comparison(self, simulated_model_results):
        """Test and compare different bootstrap methods on same data."""
        jax_results = simulated_model_results['jax']
        rmark_results = simulated_model_results['rmark']
        
        # Use phi_intercept for comparison
        param_name = 'phi_intercept'
        jax_est = jax_results['parameters'][param_name]
        rmark_est = rmark_results['parameters'][param_name]
        
        # Generate same bootstrap samples
        np.random.seed(42)
        jax_bootstrap = np.random.normal(jax_est, jax_results['std_errors'][param_name], 500)
        rmark_bootstrap = np.random.normal(rmark_est, rmark_results['std_errors'][param_name], 500)
        
        # Test all methods
        methods_results = {}
        for method in BootstrapMethod:
            result = bootstrap_parameter_difference(
                jax_bootstrap, rmark_bootstrap,
                n_bootstrap=300,
                method=method,
                confidence_level=0.95,
                random_seed=42
            )
            methods_results[method] = result
        
        # All methods should provide valid results
        for method, result in methods_results.items():
            assert result.method == method
            assert result.confidence_interval[0] < result.confidence_interval[1]
            assert result.confidence_level == 0.95
            assert len(result.bootstrap_estimates) == 300
        
        # BCa should have bias and acceleration corrections
        bca_result = methods_results[BootstrapMethod.BCa]
        assert bca_result.bias_estimate is not None
        assert bca_result.acceleration is not None
        
        # Confidence intervals should be similar but not identical
        basic_ci = methods_results[BootstrapMethod.BASIC].confidence_interval
        percentile_ci = methods_results[BootstrapMethod.PERCENTILE].confidence_interval
        bca_ci = methods_results[BootstrapMethod.BCa].confidence_interval
        
        # All confidence intervals should overlap substantially
        assert max(basic_ci[0], percentile_ci[0], bca_ci[0]) < min(basic_ci[1], percentile_ci[1], bca_ci[1])


class TestConcordanceIntegration:
    """Integration tests for comprehensive concordance analysis."""
    
    def test_multi_parameter_concordance_analysis(self, simulated_model_results):
        """Test concordance analysis across multiple parameters."""
        jax_results = simulated_model_results['jax']
        rmark_results = simulated_model_results['rmark']
        
        concordance_results = []
        
        # Analyze concordance for each parameter
        for param_name in jax_results['parameters'].keys():
            if param_name in rmark_results['parameters']:
                # Simulate multiple estimates for concordance analysis
                n_estimates = 50
                jax_estimates = np.random.normal(
                    jax_results['parameters'][param_name],
                    jax_results['std_errors'][param_name],
                    n_estimates
                )
                rmark_estimates = np.random.normal(
                    rmark_results['parameters'][param_name],
                    rmark_results['std_errors'][param_name],
                    n_estimates
                )
                
                result = comprehensive_concordance_analysis(
                    jax_estimates, rmark_estimates,
                    confidence_level=0.95,
                    robust=True
                )
                
                concordance_results.append(result)
        
        # Validate results
        assert len(concordance_results) > 0
        
        for result in concordance_results:
            assert isinstance(result, ConcordanceAnalysisResult)
            assert 0 <= result.correlation_coefficient <= 1
            assert 0 <= result.concordance_correlation_coefficient <= 1
            assert result.mean_absolute_error >= 0
            assert result.root_mean_square_error >= 0
            assert result.agreement_category in ['excellent', 'good', 'moderate', 'poor']
            
            # Bland-Altman analysis should be complete
            assert len(result.limits_of_agreement) == 2
            assert result.limits_of_agreement[0] < result.limits_of_agreement[1]
            assert 0 <= result.within_limits_percentage <= 100
    
    def test_concordance_with_outliers(self):
        """Test concordance analysis robustness with outliers."""
        np.random.seed(42)
        
        # Create data with outliers
        n_points = 40
        clean_values_1 = np.random.normal(0.8, 0.05, n_points - 3)
        clean_values_2 = clean_values_1 + np.random.normal(0, 0.02, n_points - 3)
        
        # Add outliers
        outlier_values_1 = np.array([0.3, 1.4, 0.1])  # Extreme outliers
        outlier_values_2 = np.array([0.82, 0.81, 0.79])  # Corresponding values
        
        values_1 = np.concatenate([clean_values_1, outlier_values_1])
        values_2 = np.concatenate([clean_values_2, outlier_values_2])
        
        # Test robust vs non-robust analysis
        robust_result = comprehensive_concordance_analysis(values_1, values_2, robust=True)
        standard_result = comprehensive_concordance_analysis(values_1, values_2, robust=False)
        
        # Robust analysis should detect outliers
        assert robust_result.outlier_count >= 2  # Should detect at least 2 outliers
        assert standard_result.outlier_count == 0  # Standard analysis doesn't detect outliers
        
        # Robust correlation should be different from standard correlation
        assert abs(robust_result.robust_correlation - standard_result.correlation_coefficient) > 0.05
        
        # Both should provide valid agreement assessments
        assert robust_result.agreement_category in ['excellent', 'good', 'moderate', 'poor']
        assert standard_result.agreement_category in ['excellent', 'good', 'moderate', 'poor']


class TestCrossValidationIntegration:
    """Integration tests for cross-validation stability assessment."""
    
    def test_parameter_stability_assessment(self, realistic_capture_data):
        """Test parameter stability across cross-validation folds."""
        
        def mock_model_fitting_function(fold, repeat):
            """Mock function that simulates model fitting with realistic variation."""
            np.random.seed(fold + repeat * 10 + 42)  # Reproducible but different per fold
            
            # Base parameters with realistic estimation uncertainty
            base_phi = 0.85
            base_p = 0.65
            base_f = 0.15
            
            # Simulate realistic parameter uncertainty
            uncertainty_factor = 0.02 + 0.005 * np.random.random()  # 2-2.5% uncertainty
            
            jax_params = {
                'phi_intercept': base_phi + np.random.normal(0, uncertainty_factor),
                'p_intercept': base_p + np.random.normal(0, uncertainty_factor),
                'f_intercept': base_f + np.random.normal(0, uncertainty_factor)
            }
            
            rmark_params = {
                'phi_intercept': base_phi + np.random.normal(0, uncertainty_factor * 1.1),
                'p_intercept': base_p + np.random.normal(0, uncertainty_factor * 1.1),
                'f_intercept': base_f + np.random.normal(0, uncertainty_factor * 1.1)
            }
            
            return jax_params, rmark_params
        
        # Run stability test
        stability_result = cross_validation_stability_test(
            mock_model_fitting_function,
            n_folds=5,
            n_repeats=3,
            random_seed=42
        )
        
        # Validate stability assessment
        assert stability_result['stability_assessment'] in [
            'very_stable', 'stable', 'moderately_stable', 'unstable', 'poor_convergence'
        ]
        assert 0 <= stability_result['convergence_rate'] <= 1
        assert stability_result['n_total_comparisons'] == 5 * 3  # 5 folds × 3 repeats
        assert 'recommendations' in stability_result
        assert isinstance(stability_result['parameter_stability'], dict)
        
        # Should provide detailed parameter-level analysis
        for param_name, param_stats in stability_result['parameter_stability'].items():
            assert 'coefficient_of_variation' in param_stats
            assert 'stability_category' in param_stats
            assert param_stats['coefficient_of_variation'] >= 0
    
    def test_stability_with_convergence_failures(self):
        """Test stability assessment with some convergence failures."""
        
        def unreliable_fitting_function(fold, repeat):
            """Mock function with occasional convergence failures."""
            if (fold + repeat) % 4 == 0:  # Fail 25% of the time
                return None, None
            
            # Successful convergence
            np.random.seed(fold + repeat * 5)
            jax_params = {'phi_intercept': 0.85 + np.random.normal(0, 0.03)}
            rmark_params = {'phi_intercept': 0.851 + np.random.normal(0, 0.03)}
            
            return jax_params, rmark_params
        
        stability_result = cross_validation_stability_test(
            unreliable_fitting_function,
            n_folds=4,
            n_repeats=4,
            random_seed=42
        )
        
        # Should handle convergence failures appropriately
        assert stability_result['convergence_rate'] < 1.0
        assert stability_result['convergence_rate'] >= 0.7  # Should be around 75%
        
        if stability_result['convergence_rate'] < 0.8:
            assert 'convergence' in stability_result['stability_assessment']


class TestPublicationReadyIntegration:
    """Integration tests for publication-ready statistical reporting."""
    
    def test_comprehensive_comparison_summary(self, simulated_model_results):
        """Test complete publication-ready summary generation."""
        jax_results = simulated_model_results['jax']
        rmark_results = simulated_model_results['rmark']
        
        # Generate parameter comparisons
        parameter_comparisons = []
        for param_name in jax_results['parameters'].keys():
            if param_name in rmark_results['parameters']:
                comparison = compare_parameter_estimates(
                    jax_estimate=jax_results['parameters'][param_name],
                    jax_std_error=jax_results['std_errors'][param_name],
                    rmark_estimate=rmark_results['parameters'][param_name],
                    rmark_std_error=rmark_results['std_errors'][param_name],
                    parameter_name=param_name
                )
                parameter_comparisons.append(comparison)
        
        # Generate model comparisons
        model_comparisons = [
            ModelComparisonResult(
                model_formula="phi(1 + sex), p(1 + sex), f(1)",
                jax_aic=145.2,
                rmark_aic=145.5,
                aic_difference=0.3,
                jax_log_likelihood=-69.6,
                rmark_log_likelihood=-69.75,
                likelihood_relative_difference_pct=0.22,
                jax_convergence=True,
                rmark_convergence=True,
                overall_status=ComparisonStatus.EXCELLENT,
                parameter_comparisons=parameter_comparisons
            )
        ]
        
        # Generate concordance results
        concordance_results = []
        bootstrap_results = []
        
        for param_name in list(jax_results['parameters'].keys())[:3]:  # Test subset
            # Concordance analysis
            n_estimates = 30
            jax_estimates = np.random.normal(
                jax_results['parameters'][param_name],
                jax_results['std_errors'][param_name],
                n_estimates
            )
            rmark_estimates = np.random.normal(
                rmark_results['parameters'][param_name],
                rmark_results['std_errors'][param_name],
                n_estimates
            )
            
            concordance_result = comprehensive_concordance_analysis(jax_estimates, rmark_estimates)
            concordance_results.append(concordance_result)
            
            # Bootstrap analysis
            bootstrap_result = bootstrap_parameter_difference(
                jax_estimates, rmark_estimates,
                n_bootstrap=100,
                method=BootstrapMethod.BCa,
                random_seed=42
            )
            bootstrap_results.append(bootstrap_result)
        
        # Generate comprehensive summary
        summary = publication_ready_comparison_summary(
            parameter_comparisons,
            model_comparisons,
            concordance_results=concordance_results,
            bootstrap_results=bootstrap_results
        )
        
        # Validate comprehensive summary structure
        required_sections = [
            'statistical_summary',
            'parameter_analysis',
            'model_analysis',
            'concordance_analysis',
            'uncertainty_analysis',
            'conclusions',
            'methodology'
        ]
        
        for section in required_sections:
            assert section in summary, f"Missing section: {section}"
        
        # Validate statistical summary
        stats = summary['statistical_summary']
        assert stats['parameter_count'] == len(parameter_comparisons)
        assert stats['model_count'] == 1
        assert 0 <= stats['overall_pass_rate'] <= 1
        assert 'bootstrap_analyses_count' in stats
        assert 'concordance_analyses_count' in stats
        
        # Validate parameter analysis
        param_analysis = summary['parameter_analysis']
        assert 'by_parameter_type' in param_analysis
        assert 'critical_parameters' in param_analysis
        
        # Validate concordance analysis
        concordance_analysis = summary['concordance_analysis']
        assert 'mean_concordance_correlation' in concordance_analysis
        assert 'excellent_agreement_rate' in concordance_analysis
        assert 'systematic_bias_detected' in concordance_analysis
        
        # Validate uncertainty analysis
        uncertainty_analysis = summary['uncertainty_analysis']
        assert 'mean_confidence_width' in uncertainty_analysis
        assert 'mean_bias_estimate' in uncertainty_analysis
        assert 'stability_assessment' in uncertainty_analysis
        
        # Validate conclusions
        conclusions = summary['conclusions']
        assert conclusions['overall_validation_status'] in ['excellent', 'good', 'adequate', 'poor', 'failed']
        assert len(conclusions['key_findings']) > 0
        assert len(conclusions['recommendations']) > 0
        
        # Validate methodology
        methodology = summary['methodology']
        assert 'parameter_comparison' in methodology
        assert 'model_comparison' in methodology
        assert 'concordance_analysis' in methodology
        assert 'bootstrap_analysis' in methodology
    
    def test_summary_with_mixed_quality_results(self):
        """Test summary generation with mixed quality comparison results."""
        # Create mixed quality parameter comparisons
        parameter_comparisons = [
            # Excellent comparison
            ParameterComparisonResult(
                parameter_name='phi_intercept',
                parameter_type='phi',
                jax_estimate=0.85,
                rmark_estimate=0.851,
                absolute_difference=0.001,
                relative_difference_pct=0.12,
                comparison_status=ComparisonStatus.EXCELLENT
            ),
            # Poor comparison
            ParameterComparisonResult(
                parameter_name='p_age',
                parameter_type='p',
                jax_estimate=0.65,
                rmark_estimate=0.72,
                absolute_difference=0.07,
                relative_difference_pct=10.8,
                comparison_status=ComparisonStatus.FAILED
            ),
            # Good comparison
            ParameterComparisonResult(
                parameter_name='f_intercept',
                parameter_type='f',
                jax_estimate=0.15,
                rmark_estimate=0.148,
                absolute_difference=0.002,
                relative_difference_pct=1.3,
                comparison_status=ComparisonStatus.GOOD
            )
        ]
        
        # Model comparison with mixed convergence
        model_comparisons = [
            ModelComparisonResult(
                model_formula="phi(1 + age), p(1 + age), f(1)",
                jax_aic=152.1,
                rmark_aic=148.3,
                aic_difference=3.8,  # Large difference
                jax_log_likelihood=-73.05,
                rmark_log_likelihood=-71.15,
                likelihood_relative_difference_pct=2.7,
                jax_convergence=True,
                rmark_convergence=True,
                overall_status=ComparisonStatus.POOR,
                parameter_comparisons=parameter_comparisons
            )
        ]
        
        summary = publication_ready_comparison_summary(
            parameter_comparisons,
            model_comparisons
        )
        
        # With mixed results, overall status should reflect problems
        assert summary['statistical_summary']['overall_pass_rate'] < 1.0
        assert summary['conclusions']['overall_validation_status'] in ['adequate', 'poor']
        
        # Should identify problematic parameters
        critical_params = summary['parameter_analysis']['critical_parameters']
        assert len(critical_params) >= 1
        assert any('p_age' in str(param) for param in critical_params)
        
        # Should provide specific recommendations
        recommendations = summary['conclusions']['recommendations']
        assert len(recommendations) > 0
        assert any('investigation' in rec.lower() or 'review' in rec.lower() for rec in recommendations)


class TestMultiDatasetValidation:
    """Integration tests with multiple datasets and scenarios."""
    
    def test_validation_across_sample_sizes(self):
        """Test validation framework across different sample sizes."""
        sample_sizes = [20, 50, 100, 200]
        validation_results = {}
        
        for n_size in sample_sizes:
            # Generate dataset of specified size
            np.random.seed(42)
            n_occasions = 4
            
            # Simulate realistic parameter differences that scale with sample size
            uncertainty = 0.1 / np.sqrt(n_size)  # Uncertainty decreases with sample size
            
            jax_estimates = np.random.normal(0.8, uncertainty, 10)
            rmark_estimates = jax_estimates + np.random.normal(0, uncertainty * 0.5, 10)
            
            # Concordance analysis
            concordance_result = comprehensive_concordance_analysis(jax_estimates, rmark_estimates)
            
            # Bootstrap analysis
            bootstrap_result = bootstrap_parameter_difference(
                jax_estimates, rmark_estimates,
                n_bootstrap=min(200, n_size * 2),  # Scale bootstrap samples
                method=BootstrapMethod.PERCENTILE,
                random_seed=42
            )
            
            validation_results[n_size] = {
                'concordance': concordance_result,
                'bootstrap': bootstrap_result
            }
        
        # Validate that results improve with larger sample sizes
        for i, n_size in enumerate(sample_sizes[:-1]):
            current_ccc = validation_results[n_size]['concordance'].concordance_correlation_coefficient
            next_ccc = validation_results[sample_sizes[i+1]]['concordance'].concordance_correlation_coefficient
            
            # CCC should generally improve with larger samples (allowing some variation)
            assert next_ccc >= current_ccc - 0.1  # Allow for some random variation
    
    def test_validation_with_different_model_complexities(self, complex_formula_specs):
        """Test validation across different model complexities."""
        complexity_results = {}
        
        # Define complexity levels based on number of parameters
        complexity_levels = {
            'simple': 3,           # phi(1), p(1), f(1)
            'sex_effect': 4,       # phi(1+sex), p(1), f(1)
            'continuous_covariate': 4,  # phi(1+weight), p(1), f(1)
            'complex_model': 6     # phi(1+sex+age), p(1+sex), f(1+age)
        }
        
        for model_name, n_params in complexity_levels.items():
            # Simulate parameter estimates with complexity-dependent uncertainty
            uncertainty_base = 0.02
            uncertainty = uncertainty_base * (1 + 0.1 * n_params)  # More complex = more uncertain
            
            jax_params = {}
            rmark_params = {}
            
            for i in range(n_params):
                param_name = f'param_{i}'
                base_value = 0.7 + 0.1 * np.sin(i)  # Varied base values
                
                jax_params[param_name] = base_value + np.random.normal(0, uncertainty)
                rmark_params[param_name] = base_value + np.random.normal(0, uncertainty * 1.1)
            
            # Calculate parameter comparisons
            comparisons = []
            for param_name in jax_params.keys():
                comparison = ParameterComparisonResult(
                    parameter_name=param_name,
                    parameter_type='phi',  # Simplified for test
                    jax_estimate=jax_params[param_name],
                    rmark_estimate=rmark_params[param_name],
                    absolute_difference=abs(jax_params[param_name] - rmark_params[param_name]),
                    relative_difference_pct=abs(jax_params[param_name] - rmark_params[param_name]) / abs(jax_params[param_name]) * 100,
                    comparison_status=ComparisonStatus.GOOD  # Simplified for test
                )
                comparisons.append(comparison)
            
            complexity_results[model_name] = {
                'n_parameters': n_params,
                'comparisons': comparisons,
                'mean_absolute_difference': np.mean([c.absolute_difference for c in comparisons])
            }
        
        # Validate that results are reasonable across complexity levels
        for model_name, results in complexity_results.items():
            assert results['n_parameters'] == complexity_levels[model_name]
            assert len(results['comparisons']) == results['n_parameters']
            assert results['mean_absolute_difference'] > 0
            
            # More complex models may have larger differences, but should still be reasonable
            assert results['mean_absolute_difference'] < 0.2  # Sanity check


class TestPerformanceBenchmarking:
    """Performance and scalability tests for the validation framework."""
    
    def test_bootstrap_performance_scaling(self):
        """Test bootstrap performance with increasing sample sizes."""
        import time
        
        sample_sizes = [100, 500, 1000]
        bootstrap_sizes = [50, 100, 200]
        performance_results = {}
        
        for n_samples in sample_sizes:
            for n_bootstrap in bootstrap_sizes:
                # Generate test data
                estimates_1 = np.random.normal(0.8, 0.05, n_samples)
                estimates_2 = np.random.normal(0.81, 0.05, n_samples)
                
                # Time bootstrap analysis
                start_time = time.time()
                result = bootstrap_parameter_difference(
                    estimates_1, estimates_2,
                    n_bootstrap=n_bootstrap,
                    method=BootstrapMethod.BASIC,
                    random_seed=42
                )
                execution_time = time.time() - start_time
                
                performance_results[(n_samples, n_bootstrap)] = {
                    'execution_time': execution_time,
                    'samples_per_second': n_bootstrap / execution_time if execution_time > 0 else float('inf')
                }
                
                # Validate that result is reasonable
                assert isinstance(result, BootstrapResult)
                assert result.bootstrap_se > 0
        
        # Performance should be reasonable (less than 10 seconds for largest test)
        max_time = max(r['execution_time'] for r in performance_results.values())
        assert max_time < 10.0, f"Bootstrap analysis took too long: {max_time:.2f} seconds"
    
    def test_concordance_analysis_performance(self):
        """Test concordance analysis performance with large datasets."""
        import time
        
        # Test with increasingly large datasets
        data_sizes = [100, 500, 1000]
        performance_results = {}
        
        for n_points in data_sizes:
            # Generate test data
            values_1 = np.random.normal(0.8, 0.1, n_points)
            values_2 = values_1 + np.random.normal(0, 0.02, n_points)
            
            # Time concordance analysis
            start_time = time.time()
            result = comprehensive_concordance_analysis(
                values_1, values_2,
                robust=True,
                confidence_level=0.95
            )
            execution_time = time.time() - start_time
            
            performance_results[n_points] = {
                'execution_time': execution_time,
                'points_per_second': n_points / execution_time if execution_time > 0 else float('inf')
            }
            
            # Validate result quality
            assert isinstance(result, ConcordanceAnalysisResult)
            assert result.correlation_coefficient > 0.9  # Should be high correlation
            assert result.agreement_category in ['excellent', 'good']
        
        # Performance should scale reasonably
        largest_time = performance_results[max(data_sizes)]['execution_time']
        assert largest_time < 5.0, f"Concordance analysis took too long: {largest_time:.2f} seconds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])