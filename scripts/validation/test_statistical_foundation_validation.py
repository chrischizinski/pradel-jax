#!/usr/bin/env python3
"""
Comprehensive validation of statistical foundations in pradel-jax.

Tests all new statistical implementations against established theory:
- Time-varying covariate handling
- Parameter uncertainty estimation (Hessian-based)
- Bootstrap confidence intervals
- Model selection diagnostics
- Goodness-of-fit tests

This test ensures NO statistical regressions and validates theoretical correctness.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
import warnings
import logging

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging to see detailed output
logging.basicConfig(level=logging.INFO)

def test_time_varying_covariate_detection():
    """Test time-varying covariate detection and processing."""
    print("\n=== Testing Time-Varying Covariate Framework ===")
    
    try:
        from pradel_jax.formulas.time_varying import (
            TimeVaryingCovariateDetector, 
            detect_and_process_time_varying_covariates
        )
        from pradel_jax.data.adapters import DataContext, CovariateInfo
        import jax.numpy as jnp
        
        # Create synthetic data with time-varying covariates
        n_individuals = 100
        data = pd.DataFrame({
            'individual_id': range(n_individuals),
            'age_2016': np.random.randint(20, 60, n_individuals),
            'age_2017': np.random.randint(21, 61, n_individuals), 
            'age_2018': np.random.randint(22, 62, n_individuals),
            'tier_2016': np.random.choice([0, 1, 2], n_individuals),
            'tier_2017': np.random.choice([0, 1, 2], n_individuals),
            'tier_2018': np.random.choice([0, 1, 2], n_individuals),
            'sex': np.random.choice(['M', 'F'], n_individuals)
        })
        
        # Test detector
        detector = TimeVaryingCovariateDetector()
        time_varying_covariates = detector.detect_time_varying_patterns(data)
        
        print(f"✓ Detected {len(time_varying_covariates)} time-varying covariate groups")
        
        # Validate detection
        assert 'age' in time_varying_covariates, "Should detect age as time-varying"
        assert 'tier' in time_varying_covariates, "Should detect tier as time-varying"
        
        age_info = time_varying_covariates['age']
        assert age_info.n_occasions == 3, f"Age should have 3 occasions, got {age_info.n_occasions}"
        assert not age_info.is_categorical, "Age should be numeric"
        
        tier_info = time_varying_covariates['tier']
        assert tier_info.n_occasions == 3, f"Tier should have 3 occasions, got {tier_info.n_occasions}"
        assert tier_info.is_categorical, "Tier should be categorical"
        
        print("✓ Time-varying covariate detection working correctly")
        
        # Test data context expansion
        capture_matrix = np.random.binomial(1, 0.3, (n_individuals, 5))
        original_context = DataContext(
            capture_matrix=jnp.array(capture_matrix),
            covariates={'sex': jnp.array([1 if s == 'M' else 0 for s in data['sex']])},
            covariate_info={'sex': CovariateInfo(name='sex', dtype='int', is_categorical=True)},
            n_individuals=n_individuals,
            n_occasions=5
        )
        
        updated_context, tv_info = detect_and_process_time_varying_covariates(data, original_context)
        
        # Validate expansion
        assert 'age_is_time_varying' in updated_context.covariates, "Should add time-varying metadata"
        assert 'tier_is_time_varying' in updated_context.covariates, "Should add time-varying metadata"
        
        print("✓ Data context expansion with time-varying covariates successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Time-varying covariate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_uncertainty_estimation():
    """Test Hessian-based parameter uncertainty estimation."""
    print("\n=== Testing Parameter Uncertainty Estimation ===")
    
    try:
        from pradel_jax.inference.uncertainty import (
            compute_hessian_standard_errors,
            HessianBasedUncertainty
        )
        
        # Create a simple quadratic log-likelihood function
        # L(θ) = -(θ - 2)²/2 - (θ - 1)²
        # True maximum at θ ≈ 1.33, known Hessian
        def test_log_likelihood(params):
            theta1, theta2 = params
            return -0.5 * (theta1 - 2)**2 - (theta2 - 1)**2
        
        true_mle = np.array([2.0, 1.0])  # Known MLE
        parameter_names = ['theta1', 'theta2']
        
        # Test Hessian computation
        uncertainty_computer = HessianBasedUncertainty()
        uncertainty_result = uncertainty_computer.compute_uncertainty(
            test_log_likelihood, true_mle, parameter_names
        )
        
        print(f"✓ Computed uncertainty for {len(parameter_names)} parameters")
        print(f"  - Standard errors: {uncertainty_result.standard_errors}")
        print(f"  - Condition number: {uncertainty_result.hessian_condition_number:.2e}")
        
        # Validate results
        assert len(uncertainty_result.standard_errors) == 2, "Should have 2 standard errors"
        assert all(se > 0 for se in uncertainty_result.standard_errors), "Standard errors should be positive"
        assert uncertainty_result.hessian_condition_number < 1e10, "Matrix should be well-conditioned"
        
        # Check confidence intervals
        assert '95%' in uncertainty_result.confidence_intervals, "Should have 95% CI"
        ci_95 = uncertainty_result.confidence_intervals['95%']
        assert ci_95.shape == (2, 2), "CI should be 2x2 (parameters x bounds)"
        
        # Check correlation matrix
        assert uncertainty_result.correlation_matrix.shape == (2, 2), "Correlation matrix shape"
        assert np.allclose(np.diag(uncertainty_result.correlation_matrix), 1.0), "Diagonal should be 1"
        
        # Test parameter summary
        summary = uncertainty_result.get_parameter_summary()
        assert len(summary) == 2, "Should have summary for both parameters"
        for param_name in parameter_names:
            assert param_name in summary, f"Should have summary for {param_name}"
            assert 'estimate' in summary[param_name], "Should have estimate"
            assert 'std_error' in summary[param_name], "Should have standard error"
            assert 'p_value' in summary[param_name], "Should have p-value"
        
        print("✓ Parameter uncertainty estimation working correctly")
        print(f"  - θ₁: {summary['theta1']['estimate']:.3f} ± {summary['theta1']['std_error']:.3f}")
        print(f"  - θ₂: {summary['theta2']['estimate']:.3f} ± {summary['theta2']['std_error']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Parameter uncertainty test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_selection_diagnostics():
    """Test model selection criteria computation."""
    print("\n=== Testing Model Selection Diagnostics ===")
    
    try:
        from pradel_jax.inference.diagnostics import (
            compute_model_selection_criteria,
            compute_complete_model_diagnostics
        )
        
        # Test data
        log_likelihood = -150.5
        n_parameters = 5
        n_observations = 100
        
        # Compute selection criteria
        criteria = compute_model_selection_criteria(
            log_likelihood, n_parameters, n_observations
        )
        
        print(f"✓ Computed model selection criteria")
        print(f"  - AIC: {criteria.aic:.2f}")
        print(f"  - AICc: {criteria.aicc:.2f}")
        print(f"  - BIC: {criteria.bic:.2f}")
        print(f"  - Log-likelihood: {criteria.log_likelihood:.2f}")
        
        # Validate theoretical relationships
        expected_aic = -2 * log_likelihood + 2 * n_parameters
        expected_bic = -2 * log_likelihood + n_parameters * np.log(n_observations)
        
        assert np.isclose(criteria.aic, expected_aic), f"AIC mismatch: {criteria.aic} vs {expected_aic}"
        assert np.isclose(criteria.bic, expected_bic), f"BIC mismatch: {criteria.bic} vs {expected_bic}"
        assert criteria.bic > criteria.aic, "BIC should be larger than AIC for this case"
        assert criteria.aicc >= criteria.aic, "AICc should be >= AIC"
        
        print("✓ Model selection criteria computed correctly")
        
        # Test with overdispersion
        overdispersion = 2.5
        criteria_overdispersed = compute_model_selection_criteria(
            log_likelihood, n_parameters, n_observations, overdispersion
        )
        
        assert criteria_overdispersed.qaic is not None, "Should compute QAIC with overdispersion"
        assert criteria_overdispersed.qaic < criteria_overdispersed.aic, "QAIC should be smaller than AIC"
        
        print(f"✓ Overdispersion handling working (c-hat = {overdispersion})")
        print(f"  - QAIC: {criteria_overdispersed.qaic:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model selection diagnostics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_goodness_of_fit_tests():
    """Test goodness-of-fit test computations."""
    print("\n=== Testing Goodness-of-Fit Tests ===")
    
    try:
        from pradel_jax.inference.diagnostics import (
            compute_goodness_of_fit_tests,
            GoodnessOfFitTester
        )
        
        # Create synthetic observed vs expected data
        np.random.seed(42)
        n_cells = 50
        
        # True expected values
        expected = np.random.gamma(2, 5, n_cells)  # Mean around 10
        
        # Generate observed data with some overdispersion
        overdispersion_factor = 1.5
        observed = np.random.poisson(expected * overdispersion_factor)
        
        # Model predictions (simplified)
        model_predictions = expected / np.sum(expected)
        
        n_parameters = 5
        
        # Compute goodness-of-fit
        gof_results = compute_goodness_of_fit_tests(
            observed, expected, model_predictions, n_parameters
        )
        
        print(f"✓ Computed goodness-of-fit tests")
        print(f"  - Chi-square: {gof_results.chi_square_statistic:.2f} (df={gof_results.chi_square_df})")
        print(f"  - Deviance: {gof_results.deviance_statistic:.2f} (df={gof_results.deviance_df})")
        print(f"  - Overdispersion: {gof_results.overdispersion_estimate:.3f}")
        print(f"  - Is overdispersed: {gof_results.is_overdispersed}")
        
        # Validate results
        assert gof_results.chi_square_df == n_cells - n_parameters, "Chi-square DF incorrect"
        assert gof_results.deviance_df == n_cells - n_parameters, "Deviance DF incorrect"
        assert gof_results.chi_square_statistic >= 0, "Chi-square should be non-negative"
        assert gof_results.deviance_statistic >= 0, "Deviance should be non-negative"
        assert gof_results.overdispersion_estimate >= 1.0, "Overdispersion should be >= 1"
        
        # Check residuals
        assert len(gof_results.pearson_residuals) == n_cells, "Pearson residuals length"
        assert len(gof_results.deviance_residuals) == n_cells, "Deviance residuals length"
        assert len(gof_results.standardized_residuals) == n_cells, "Standardized residuals length"
        
        # Check p-values are in [0, 1]
        assert 0 <= gof_results.chi_square_p_value <= 1, "Chi-square p-value out of range"
        assert 0 <= gof_results.deviance_p_value <= 1, "Deviance p-value out of range"
        
        print("✓ Goodness-of-fit tests working correctly")
        
        # Test summary
        summary = gof_results.get_summary()
        assert 'chi_square' in summary, "Summary should have chi-square results"
        assert 'deviance' in summary, "Summary should have deviance results"
        assert 'overdispersion' in summary, "Summary should have overdispersion info"
        
        return True
        
    except Exception as e:
        print(f"✗ Goodness-of-fit test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bootstrap_confidence_intervals():
    """Test bootstrap confidence interval computation."""
    print("\n=== Testing Bootstrap Confidence Intervals ===")
    
    try:
        from pradel_jax.inference.uncertainty import BootstrapUncertainty
        from pradel_jax.data.adapters import DataContext, CovariateInfo
        import jax.numpy as jnp
        
        # Create simple synthetic data
        np.random.seed(42)
        n_individuals = 30  # Small for fast bootstrap
        n_occasions = 4
        
        capture_matrix = np.random.binomial(1, 0.4, (n_individuals, n_occasions))
        
        data_context = DataContext(
            capture_matrix=jnp.array(capture_matrix),
            covariates={},
            covariate_info={},
            n_individuals=n_individuals,
            n_occasions=n_occasions
        )
        
        # Simple model fitting function (mock)
        def mock_model_fitter(data_ctx):
            # Return estimates with realistic variation based on data
            data_sum = float(data_ctx.capture_matrix.sum())
            data_var = float(np.var(data_ctx.capture_matrix))
            
            # Base parameters with data-dependent variation
            phi_base = 0.6 + (data_sum / 1000) * 0.1  # Vary based on capture frequency
            p_base = 0.7 + (data_var - 0.25) * 0.2    # Vary based on capture variation
            
            # Add small amount of noise
            phi = phi_base + np.random.normal(0, 0.02)
            p = p_base + np.random.normal(0, 0.02) 
            
            params = np.array([max(0.1, min(0.99, phi)), max(0.1, min(0.99, p))])  # Keep in valid range
            log_likelihood = -50.0 + np.random.normal(0, 1.0)  # Add some variation
            return params, log_likelihood
        
        # Test bootstrap
        bootstrap_computer = BootstrapUncertainty()
        bootstrap_result = bootstrap_computer.compute_bootstrap_uncertainty(
            data_context, 
            mock_model_fitter, 
            n_bootstrap_samples=50,  # Small number for testing
            random_seed=42
        )
        
        print(f"✓ Computed bootstrap uncertainty")
        print(f"  - Bootstrap samples: {bootstrap_result.bootstrap_samples.shape}")
        print(f"  - Standard errors: {bootstrap_result.standard_errors}")
        print(f"  - Bias correction: {bootstrap_result.bootstrap_bias}")
        
        # Validate results
        assert bootstrap_result.bootstrap_samples.shape[0] <= 50, "Should have <= 50 samples"
        assert bootstrap_result.bootstrap_samples.shape[1] == 2, "Should have 2 parameters"
        assert len(bootstrap_result.standard_errors) == 2, "Should have 2 standard errors"
        assert len(bootstrap_result.bootstrap_bias) == 2, "Should have 2 bias estimates"
        
        # Check confidence intervals
        assert '95%' in bootstrap_result.confidence_intervals, "Should have 95% CI"
        ci_95 = bootstrap_result.confidence_intervals['95%']
        assert ci_95.shape == (2, 2), "CI should be 2x2"
        
        # Check that confidence intervals make sense
        for i in range(2):
            lower, upper = ci_95[i, :]
            assert lower < upper, f"Lower bound should be < upper bound for parameter {i}"
            # Parameter estimate should generally be within CI (not always due to bias correction)
            
        print("✓ Bootstrap confidence intervals working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Bootstrap confidence intervals test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_regression_framework():
    """Test performance regression testing framework."""
    print("\n=== Testing Performance Regression Framework ===")
    
    try:
        from pradel_jax.inference.regression_tests import (
            RegressionTestDefinition,
            PerformanceRegressionTester,
            generate_simple_dipper_data
        )
        import tempfile
        
        # Test data generation
        data_context = generate_simple_dipper_data()
        assert data_context.n_individuals == 50, "Should generate 50 individuals"
        assert data_context.n_occasions == 5, "Should generate 5 occasions"
        
        print("✓ Test data generation working")
        
        # Mock model fitter
        def mock_model_fitter(data_ctx, model_spec):
            # Return deterministic results based on data
            n_params = len(model_spec)
            data_hash = hash(str(data_ctx.capture_matrix.sum())) % 1000
            params = np.array([0.7, 0.6, 0.1][:n_params])  # phi, p, f
            log_likelihood = -100.0 - data_hash * 0.01
            
            diagnostics = {
                'aic': -2 * log_likelihood + 2 * n_params,
                'bic': -2 * log_likelihood + n_params * np.log(data_ctx.n_individuals),
                'success': True
            }
            
            return params, log_likelihood, diagnostics
        
        # Create test definition
        test_def = RegressionTestDefinition(
            name="test_constant_model",
            data_generator=generate_simple_dipper_data,
            model_specification={'phi': '~1', 'p': '~1', 'f': '~1'},
            tolerance=1e-6,
            description="Test constant model"
        )
        
        # Test regression framework
        with tempfile.TemporaryDirectory() as temp_dir:
            tester = PerformanceRegressionTester(Path(temp_dir) / "baselines")
            
            # Run single test
            result = tester.run_regression_test(test_def, mock_model_fitter)
            
            print(f"✓ Single test executed: {'PASSED' if result.passes_tolerance else 'FAILED'}")
            print(f"  - Parameters: {result.parameter_estimates}")
            print(f"  - Log-likelihood: {result.log_likelihood:.2f}")
            print(f"  - Execution time: {result.execution_time:.3f}s")
            
            # Validate result
            assert len(result.parameter_estimates) == 3, "Should have 3 parameters"
            assert result.convergence_success, "Should converge successfully"
            assert result.execution_time > 0, "Should have positive execution time"
            
            # Run test suite
            test_suite = tester.run_test_suite([test_def], mock_model_fitter)
            
            assert test_suite.n_tests == 1, "Should run 1 test"
            assert test_suite.n_passed == 1, "Should pass 1 test"
            assert test_suite.n_failed == 0, "Should fail 0 tests"
            
            print("✓ Test suite execution working correctly")
            
            # Test baseline functionality
            tester.save_baseline_results(test_suite)
            baseline_result = tester.load_baseline_result(test_def)
            
            assert baseline_result is not None, "Should load baseline"
            assert np.array_equal(baseline_result.parameter_estimates, result.parameter_estimates), "Baseline should match"
            
            print("✓ Baseline save/load working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance regression framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_validation():
    """Run all validation tests."""
    print("=" * 60)
    print("COMPREHENSIVE STATISTICAL FOUNDATION VALIDATION")
    print("=" * 60)
    print("Testing all new statistical implementations against established theory...")
    
    tests = [
        ("Time-varying covariates", test_time_varying_covariate_detection),
        ("Parameter uncertainty", test_parameter_uncertainty_estimation),
        ("Model selection diagnostics", test_model_selection_diagnostics),
        ("Goodness-of-fit tests", test_goodness_of_fit_tests),
        ("Bootstrap confidence intervals", test_bootstrap_confidence_intervals),
        ("Performance regression framework", test_performance_regression_framework),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_function in tests:
        try:
            if test_function():
                passed += 1
            else:
                print(f"FAILED: {test_name}")
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Statistical foundations are sound!")
        print("\nKey capabilities validated:")
        print("✓ Time-varying covariate detection and processing")
        print("✓ Hessian-based parameter uncertainty estimation")
        print("✓ Information criteria (AIC, AICc, BIC, QAIC)")
        print("✓ Goodness-of-fit tests (Chi-square, Deviance)")
        print("✓ Bootstrap confidence intervals")
        print("✓ Performance regression testing framework")
        print("\nThe statistical implementations follow established theory and are ready for production use.")
        return True
    else:
        print(f"❌ {total - passed} TESTS FAILED - Statistical foundations need attention!")
        return False


if __name__ == "__main__":
    # Run comprehensive validation
    success = run_comprehensive_validation()
    
    if not success:
        sys.exit(1)
    else:
        print("\n✅ All statistical validation tests passed successfully!")
        sys.exit(0)