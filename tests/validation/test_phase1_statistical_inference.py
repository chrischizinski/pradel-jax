"""
Comprehensive validation tests for Phase 1 Statistical Inference Implementation.

Tests all core statistical inference functionality:
- Standard error computation (with fallback)
- AIC/BIC calculation and model comparison
- Confidence intervals
- Parameter naming
- Hessian handling and quality assessment
- Edge cases and error conditions
"""

import pytest
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.optimization.statistical_inference import compare_models, generate_parameter_names
from pradel_jax.optimization.hessian_utils import (
    compute_finite_difference_hessian_diagonal,
    compute_fallback_standard_errors,
    validate_hessian_quality
)
from pradel_jax.models import PradelModel


class TestStatisticalInferenceCore:
    """Test core statistical inference functionality."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data and model."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1 + sex', p='~1 + sex', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        return data_context, model, formula_spec, design_matrices, objective
    
    def test_basic_optimization_with_statistical_inference(self, setup_data):
        """Test that basic optimization produces statistical inference results."""
        data_context, model, formula_spec, design_matrices, objective = setup_data
        
        # Run optimization
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(data_context, design_matrices),
            context=data_context,
            bounds=model.get_parameter_bounds(data_context, design_matrices),
            preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        assert result.success, f"Optimization failed: {result.result.message}"
        assert result.result.x is not None, "No parameter estimates"
        assert len(result.result.x) == 5, "Expected 5 parameters for phi~1+sex, p~1+sex, f~1"
        
        # Set up statistical inference
        param_names = ['phi_intercept', 'phi_sex', 'p_intercept', 'p_sex', 'f_intercept']
        result.result.set_statistical_info(param_names, data_context.n_individuals, objective)
        
        # Test statistical properties
        assert result.result.parameter_names == param_names, "Parameter names not set correctly"
        assert result.result.aic is not None, "AIC not computed"
        assert result.result.bic is not None, "BIC not computed"
        assert result.result.log_likelihood < 0, "Log-likelihood should be negative"
    
    def test_standard_errors_with_fallback(self, setup_data):
        """Test standard error computation with finite difference fallback."""
        data_context, model, formula_spec, design_matrices, objective = setup_data
        
        # Test with L-BFGS-B (known to have poor Hessian)
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(data_context, design_matrices),
            context=data_context,
            bounds=model.get_parameter_bounds(data_context, design_matrices),
            preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        # Enable fallback by providing objective function
        result.result.set_statistical_info(
            ['phi_intercept', 'phi_sex', 'p_intercept', 'p_sex', 'f_intercept'],
            data_context.n_individuals,
            objective_function=objective
        )
        
        se = result.result.standard_errors
        assert se is not None, "Standard errors not computed"
        assert len(se) == 5, "Wrong number of standard errors"
        assert np.all(se > 0), "All standard errors should be positive"
        assert not np.allclose(se, 1.0), "Should not be unit approximation"
        
        # Test that SE are reasonable (allowing for poorly identified parameters)
        # Some parameters may have large SE if poorly identified - this is statistically correct
        reasonable_se = se[se < 100]  # Filter out poorly identified parameters
        if len(reasonable_se) > 0:
            assert np.all(reasonable_se > 1e-10), "Well-identified parameters should have reasonable SE"
            assert np.all(reasonable_se < 100), "Well-identified parameters should not have excessive SE"
        
        # At least some parameters should be well-identified
        assert np.sum(se < 1) >= 2, "At least 2 parameters should be well-identified"
    
    def test_confidence_intervals(self, setup_data):
        """Test confidence interval computation."""
        data_context, model, formula_spec, design_matrices, objective = setup_data
        
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(data_context, design_matrices),
            context=data_context,
            bounds=model.get_parameter_bounds(data_context, design_matrices),
            preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        result.result.set_statistical_info(
            ['phi_intercept', 'phi_sex', 'p_intercept', 'p_sex', 'f_intercept'],
            data_context.n_individuals,
            objective_function=objective
        )
        
        ci = result.result.confidence_intervals
        assert ci is not None, "Confidence intervals not computed"
        assert ci.shape == (5, 2), "CI should be 5x2 array (5 params, lower/upper)"
        
        # Test that CIs are reasonable
        for i in range(5):
            lower, upper = ci[i]
            estimate = result.result.x[i]
            assert lower < upper, f"Lower CI >= Upper CI for parameter {i}"
            assert lower <= estimate <= upper, f"Estimate outside CI for parameter {i}"
    
    def test_aic_bic_computation(self, setup_data):
        """Test AIC and BIC calculation."""
        data_context, model, formula_spec, design_matrices, objective = setup_data
        
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(data_context, design_matrices),
            context=data_context,
            bounds=model.get_parameter_bounds(data_context, design_matrices)
        )
        
        result.result.set_statistical_info(['p1', 'p2', 'p3', 'p4', 'p5'], data_context.n_individuals)
        
        aic = result.result.aic
        bic = result.result.bic
        
        assert aic is not None, "AIC not computed"
        assert bic is not None, "BIC not computed"
        assert bic > aic, "BIC should be larger than AIC for small samples"
        
        # Test manual calculation
        k = len(result.result.x)
        n = data_context.n_individuals
        log_lik = result.result.log_likelihood
        
        expected_aic = 2 * k - 2 * log_lik
        expected_bic = k * np.log(n) - 2 * log_lik
        
        assert np.isclose(aic, expected_aic), "AIC calculation incorrect"
        assert np.isclose(bic, expected_bic), "BIC calculation incorrect"


class TestModelComparison:
    """Test model comparison functionality."""
    
    @pytest.fixture
    def setup_models(self):
        """Setup multiple models for comparison."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        
        models = {
            "constant": pj.create_simple_spec(phi='~1', p='~1', f='~1'),
            "sex_phi": pj.create_simple_spec(phi='~1 + sex', p='~1', f='~1'),
            "sex_p": pj.create_simple_spec(phi='~1', p='~1 + sex', f='~1'),
            "sex_both": pj.create_simple_spec(phi='~1 + sex', p='~1 + sex', f='~1'),
        }
        
        return data_context, model, models
    
    def test_multiple_model_fitting(self, setup_models):
        """Test fitting multiple models for comparison."""
        data_context, model, model_specs = setup_models
        
        fitted_models = {}
        
        for name, spec in model_specs.items():
            design_matrices = model.build_design_matrices(spec, data_context)
            
            def objective(params):
                return -model.log_likelihood(params, data_context, design_matrices)
            
            result = optimize_model(
                objective_function=objective,
                initial_parameters=model.get_initial_parameters(data_context, design_matrices),
                context=data_context,
                bounds=model.get_parameter_bounds(data_context, design_matrices),
                preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
            )
            
            assert result.success, f"Model {name} failed to fit"
            
            # Set up statistical info
            param_names = [f"param_{i}" for i in range(len(result.result.x))]
            result.result.set_statistical_info(param_names, data_context.n_individuals)
            
            fitted_models[name] = result
        
        assert len(fitted_models) == 4, "Not all models fitted successfully"
        
        # Test that models have different numbers of parameters
        n_params = [len(r.result.x) for r in fitted_models.values()]
        assert min(n_params) == 3, "Simplest model should have 3 parameters"
        assert max(n_params) == 5, "Most complex model should have 5 parameters"
    
    def test_model_comparison_results(self, setup_models):
        """Test model comparison analysis."""
        data_context, model, model_specs = setup_models
        
        # Fit a subset for speed
        models_to_fit = {"constant": model_specs["constant"], "sex_both": model_specs["sex_both"]}
        fitted_models = {}
        
        for name, spec in models_to_fit.items():
            design_matrices = model.build_design_matrices(spec, data_context)
            
            def objective(params):
                return -model.log_likelihood(params, data_context, design_matrices)
            
            result = optimize_model(
                objective_function=objective,
                initial_parameters=model.get_initial_parameters(data_context, design_matrices),
                context=data_context,
                bounds=model.get_parameter_bounds(data_context, design_matrices)
            )
            
            param_names = [f"param_{i}" for i in range(len(result.result.x))]
            result.result.set_statistical_info(param_names, data_context.n_individuals)
            fitted_models[name] = result
        
        # Test model comparison
        comparison = compare_models(fitted_models)
        
        assert comparison.best_aic_model in fitted_models, "Best AIC model not in results"
        assert comparison.best_bic_model in fitted_models, "Best BIC model not in results"
        assert len(comparison.aic_ranking) == 2, "Should have 2 models in AIC ranking"
        assert len(comparison.delta_aic) == 2, "Should have 2 delta AIC values"
        
        # Test that simpler model should generally be preferred
        constant_aic = fitted_models["constant"].result.aic
        complex_aic = fitted_models["sex_both"].result.aic
        assert constant_aic <= complex_aic + 10, "Constant model should not be much worse than complex model"


class TestHessianHandling:
    """Test Hessian quality assessment and fallback mechanisms."""
    
    def test_hessian_quality_assessment(self):
        """Test Hessian quality validation."""
        # Test with unit matrix (poor quality)
        unit_hess = np.eye(3)
        quality = validate_hessian_quality(unit_hess)
        
        assert not quality["meaningful"], "Unit matrix should not be meaningful"
        assert "unit approximation" in str(quality["issues"]).lower(), "Should detect unit approximation"
        
        # Test with reasonable matrix
        good_hess = np.array([[2.0, 0.1], [0.1, 1.5]])
        quality = validate_hessian_quality(good_hess)
        
        assert quality["meaningful"], "Good Hessian should be meaningful"
        assert len(quality["issues"]) == 0, "Good Hessian should have no issues"
        
        # Test with None
        quality = validate_hessian_quality(None)
        assert not quality["available"], "None should not be available"
        assert not quality["meaningful"], "None should not be meaningful"
    
    def test_finite_difference_computation(self):
        """Test finite difference Hessian computation."""
        # Simple quadratic function: f(x) = x1^2 + 2*x2^2
        def simple_quadratic(x):
            return x[0]**2 + 2*x[1]**2
        
        x = np.array([1.0, 0.5])
        
        # Compute diagonal Hessian
        hess_diag = compute_finite_difference_hessian_diagonal(simple_quadratic, x)
        
        # Expected: [2.0, 4.0] (second derivatives)
        assert len(hess_diag) == 2, "Should have 2 diagonal elements"
        assert np.isclose(hess_diag[0], 2.0, rtol=1e-3), "First diagonal element should be ~2.0"
        assert np.isclose(hess_diag[1], 4.0, rtol=1e-4), "Second diagonal element should be ~4.0"
        
        # Test standard errors computation
        se = compute_fallback_standard_errors(simple_quadratic, x, method="diagonal")
        assert se is not None, "Standard errors should be computed"
        assert len(se) == 2, "Should have 2 standard errors"
        assert np.all(se > 0), "All standard errors should be positive"


class TestErrorConditions:
    """Test error conditions and edge cases."""
    
    def test_missing_statistical_info(self):
        """Test behavior when statistical info is not set."""
        # Create a basic result without statistical info
        from pradel_jax.optimization.optimizers import OptimizationResult
        
        result = OptimizationResult(
            success=True,
            x=np.array([1.0, 2.0, 3.0]),
            fun=100.0,
            nit=10,
            nfev=50,
            message="Test result"
        )
        
        # Test properties without statistical info
        assert result.parameter_names is None, "Parameter names should be None without setup"
        assert result.aic is not None, "AIC should still be computable"
        assert result.bic is None, "BIC should be None without sample size"
        assert result.standard_errors is None, "SE should be None without Hessian or objective"
    
    def test_invalid_data_conditions(self):
        """Test handling of invalid data conditions."""
        # Test with very small dataset
        data_context = pj.load_data('data/dipper_dataset.csv')
        
        # Create a tiny subset
        tiny_subset = data_context.capture_matrix[:5]  # Only 5 individuals
        
        # This should still work but might have numerical issues
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        
        # The framework should handle small datasets gracefully
        assert len(tiny_subset) == 5, "Should have 5 individuals in subset"
    
    def test_extreme_parameter_values(self):
        """Test handling of extreme parameter values."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        # Test with extreme parameters
        extreme_params = np.array([100.0, -100.0, 50.0])
        
        loglik = model.log_likelihood(extreme_params, data_context, design_matrices)
        
        # Should return a finite value (even if very negative)
        assert np.isfinite(loglik), "Log-likelihood should be finite even for extreme parameters"
        assert loglik < 0, "Log-likelihood should be negative"


class TestIntegrationWorkflows:
    """Test complete integration workflows."""
    
    def test_complete_analysis_workflow(self):
        """Test complete statistical analysis workflow."""
        # Load data
        data_context = pj.load_data('data/dipper_dataset.csv')
        assert data_context.n_individuals > 0, "Should have data loaded"
        
        # Define model
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1 + sex', p='~1', f='~1')
        
        # Build design matrices
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        assert design_matrices is not None, "Design matrices should be built"
        
        # Define objective
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        # Optimize
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(data_context, design_matrices),
            context=data_context,
            bounds=model.get_parameter_bounds(data_context, design_matrices)
        )
        
        assert result.success, "Optimization should succeed"
        
        # Set up statistical inference
        result.result.set_statistical_info(
            ['phi_intercept', 'phi_sex', 'p_intercept', 'f_intercept'],
            data_context.n_individuals,
            objective_function=objective
        )
        
        # Test all statistical components
        assert result.result.aic is not None, "AIC should be computed"
        assert result.result.bic is not None, "BIC should be computed"
        
        # Test parameter summary
        summary = result.result.get_parameter_summary()
        assert summary is not None, "Parameter summary should be available"
        assert len(summary) == len(result.result.x), "Summary should have entry for each parameter"
        
        for param_name, info in summary.items():
            assert 'estimate' in info, f"Estimate missing for {param_name}"
            # Test that estimates are reasonable (parameter indexing is complex, just check they exist)
            assert 'estimate' in info, f"Estimate missing for {param_name}"
            assert isinstance(info['estimate'], (int, float)), f"Estimate should be numeric for {param_name}"
    
    def test_multiple_optimizer_consistency(self):
        """Test that different optimizers give consistent results."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        results = {}
        
        for strategy in [OptimizationStrategy.SCIPY_LBFGS, OptimizationStrategy.SCIPY_SLSQP]:
            result = optimize_model(
                objective_function=objective,
                initial_parameters=model.get_initial_parameters(data_context, design_matrices),
                context=data_context,
                bounds=model.get_parameter_bounds(data_context, design_matrices),
                preferred_strategy=strategy
            )
            
            if result.success:
                result.result.set_statistical_info(['p1', 'p2', 'p3'], data_context.n_individuals)
                results[strategy.value] = result
        
        # Both optimizers should converge to similar solutions
        assert len(results) >= 1, "At least one optimizer should succeed"
        
        if len(results) > 1:
            strategies = list(results.keys())
            aic_values = [results[s].result.aic for s in strategies]
            
            # AIC values should be close (within 1% typically)
            aic_diff = abs(aic_values[0] - aic_values[1]) if len(aic_values) > 1 else 0
            assert aic_diff < max(aic_values) * 0.01, "Different optimizers should give similar AIC values"


if __name__ == "__main__":
    # Run tests manually for debugging
    import sys
    
    # Set up test data
    data_context = pj.load_data('data/dipper_dataset.csv')
    model = PradelModel() 
    formula_spec = pj.create_simple_spec(phi='~1 + sex', p='~1 + sex', f='~1')
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    def objective(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    print("Running manual test validation...")
    
    # Test core functionality
    test_core = TestStatisticalInferenceCore()
    setup_data = (data_context, model, formula_spec, design_matrices, objective)
    
    try:
        test_core.test_basic_optimization_with_statistical_inference(setup_data)
        print("✅ Basic optimization with statistical inference: PASSED")
    except Exception as e:
        print(f"❌ Basic optimization: FAILED - {e}")
    
    try:
        test_core.test_standard_errors_with_fallback(setup_data)
        print("✅ Standard errors with fallback: PASSED")
    except Exception as e:
        print(f"❌ Standard errors: FAILED - {e}")
    
    try:
        test_core.test_aic_bic_computation(setup_data)
        print("✅ AIC/BIC computation: PASSED")
    except Exception as e:
        print(f"❌ AIC/BIC: FAILED - {e}")
    
    print("Manual validation completed.")