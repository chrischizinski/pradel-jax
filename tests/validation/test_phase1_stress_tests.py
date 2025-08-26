"""
Stress tests and edge case validation for Phase 1 Statistical Inference.

Tests the robustness and reliability of the statistical inference implementation
under challenging conditions including:
- Very small datasets
- Highly correlated parameters  
- Extreme parameter values
- Numerical boundary conditions
- Performance under load
"""

import pytest
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.optimization.statistical_inference import compare_models
from pradel_jax.models import PradelModel


class TestNumericalStability:
    """Test numerical stability under challenging conditions."""
    
    def test_very_small_dataset(self):
        """Test behavior with very small datasets."""
        # Use only 10 individuals from the dataset
        data_context = pj.load_data('data/dipper_dataset.csv')
        
        # Create a synthetic small dataset 
        small_capture_matrix = data_context.capture_matrix[:10]
        # Handle covariates properly - some might be boolean scalars
        small_covariates = {}
        for k, v in data_context.covariates.items():
            if hasattr(v, '__getitem__') and hasattr(v, '__len__'):
                small_covariates[k] = v[:10] if len(v) > 10 else v
            else:
                small_covariates[k] = v  # Scalar covariate, keep as is
        
        # Create a new data context with small data
        from pradel_jax.data.adapters import DataContext
        small_context = DataContext(
            capture_matrix=small_capture_matrix,
            covariates=small_covariates,
            covariate_info=data_context.covariate_info,
            n_individuals=10,
            n_occasions=data_context.n_occasions
        )
        
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, small_context)
        
        def objective(params):
            return -model.log_likelihood(params, small_context, design_matrices)
        
        # Should still work but might have numerical issues
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(small_context, design_matrices),
            context=small_context,
            bounds=model.get_parameter_bounds(small_context, design_matrices),
            preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        # Framework should handle gracefully
        assert result.success, "Should handle small datasets gracefully"
        
        # Set up statistical inference
        result.result.set_statistical_info(['phi', 'p', 'f'], 10, objective_function=objective)
        
        # Statistical inference should work but may have large uncertainties
        assert result.result.aic is not None, "AIC should be computable for small datasets"
        assert result.result.bic is not None, "BIC should be computable for small datasets"
        
        se = result.result.standard_errors
        if se is not None:
            # Expect large standard errors for small datasets - this is statistically correct
            assert np.all(se > 0), "All standard errors should be positive"
    
    def test_extreme_initial_values(self):
        """Test robustness with extreme initial parameter values."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        # Test with various extreme starting points
        extreme_starts = [
            np.array([100.0, 100.0, 100.0]),   # Very large positive
            np.array([-100.0, -100.0, -100.0]), # Very large negative
            np.array([1e-10, 1e-10, 1e-10]),   # Very small positive
            np.array([0.0, 0.0, 0.0]),         # Zeros
        ]
        
        successes = 0
        for i, start_params in enumerate(extreme_starts):
            result = optimize_model(
                objective_function=objective,
                initial_parameters=start_params,
                context=data_context,
                bounds=model.get_parameter_bounds(data_context, design_matrices),
                preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
            )
            
            if result.success:
                successes += 1
                # Should converge to reasonable values regardless of starting point
                assert np.all(np.isfinite(result.result.x)), f"Parameters should be finite for start {i}"
                assert np.all(np.abs(result.result.x) < 100), f"Parameters should be reasonable for start {i}"
        
        # At least half the extreme starts should succeed
        assert successes >= 2, f"Should succeed with at least 2/4 extreme starts, got {successes}"
    
    def test_parameter_correlation_handling(self):
        """Test handling of highly correlated parameters."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        
        # Create a model with potentially correlated parameters
        formula_spec = pj.create_simple_spec(phi='~1 + sex', p='~1 + sex', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(data_context, design_matrices),
            context=data_context,
            bounds=model.get_parameter_bounds(data_context, design_matrices),
            preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        assert result.success, "Should handle correlated parameters"
        
        result.result.set_statistical_info(
            ['phi_int', 'phi_sex', 'p_int', 'p_sex', 'f_int'],
            data_context.n_individuals,
            objective_function=objective
        )
        
        # Standard errors should detect correlation (some may be large)
        se = result.result.standard_errors
        if se is not None:
            # Should have reasonable values for at least some parameters
            reasonable_se = se[se < 10]
            assert len(reasonable_se) >= 2, "At least some parameters should be well-identified"


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    def test_repeated_optimization_consistency(self):
        """Test that repeated optimizations give consistent results."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        results = []
        
        # Run optimization 5 times
        for i in range(5):
            result = optimize_model(
                objective_function=objective,
                initial_parameters=model.get_initial_parameters(data_context, design_matrices),
                context=data_context,
                bounds=model.get_parameter_bounds(data_context, design_matrices),
                preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
            )
            
            if result.success:
                result.result.set_statistical_info(['phi', 'p', 'f'], data_context.n_individuals)
                results.append(result)
        
        assert len(results) >= 4, "At least 4/5 runs should succeed"
        
        # Check consistency of results
        aic_values = [r.result.aic for r in results]
        log_liks = [r.result.log_likelihood for r in results]
        
        # All runs should give very similar results
        aic_std = np.std(aic_values)
        loglik_std = np.std(log_liks)
        
        assert aic_std < 0.1, f"AIC should be consistent across runs, std={aic_std}"
        assert loglik_std < 0.05, f"Log-likelihood should be consistent, std={loglik_std}"
    
    def test_multiple_strategies_comparison(self):
        """Test that multiple optimization strategies work and give similar results."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        strategies = [OptimizationStrategy.SCIPY_LBFGS, OptimizationStrategy.SCIPY_SLSQP]
        results = {}
        
        for strategy in strategies:
            result = optimize_model(
                objective_function=objective,
                initial_parameters=model.get_initial_parameters(data_context, design_matrices),
                context=data_context,
                bounds=model.get_parameter_bounds(data_context, design_matrices),
                preferred_strategy=strategy
            )
            
            if result.success:
                result.result.set_statistical_info(['phi', 'p', 'f'], data_context.n_individuals)
                results[strategy.value] = result
        
        assert len(results) >= 1, "At least one strategy should succeed"
        
        if len(results) > 1:
            # Compare results between strategies
            strategies = list(results.keys())
            aic1 = results[strategies[0]].result.aic
            aic2 = results[strategies[1]].result.aic
            
            # Should get similar AIC values (within 1%)
            rel_diff = abs(aic1 - aic2) / max(aic1, aic2)
            assert rel_diff < 0.01, f"Different strategies should give similar results: AIC1={aic1}, AIC2={aic2}"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_missing_data_patterns(self):
        """Test handling of various missing data patterns."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        
        # Test with data that has all-zero capture histories
        capture_matrix = data_context.capture_matrix.copy()
        
        # Make first few individuals never captured (use JAX-compatible assignment)
        import numpy as np
        capture_matrix = np.array(capture_matrix)  # Convert to numpy for modification
        capture_matrix[:3] = 0
        
        from pradel_jax.data.adapters import DataContext
        modified_context = DataContext(
            capture_matrix=capture_matrix,
            covariates=data_context.covariates,
            covariate_info=data_context.covariate_info,
            n_individuals=data_context.n_individuals,
            n_occasions=data_context.n_occasions
        )
        
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, modified_context)
        
        def objective(params):
            return -model.log_likelihood(params, modified_context, design_matrices)
        
        # Should handle gracefully
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(modified_context, design_matrices),
            context=modified_context,
            bounds=model.get_parameter_bounds(modified_context, design_matrices),
            preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        # Framework should either succeed or fail gracefully
        if result.success:
            result.result.set_statistical_info(['phi', 'p', 'f'], modified_context.n_individuals)
            assert result.result.aic is not None, "Should compute AIC even with challenging data"
        else:
            # Failure is acceptable for pathological data
            assert isinstance(result.result.message, str), "Should provide error message"
    
    def test_boundary_parameter_values(self):
        """Test behavior when parameters approach boundaries."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        # Get the bounds
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Test near boundaries (but not exactly at them to avoid numerical issues)
        if bounds is not None:
            boundary_params = []
            for i, (lower, upper) in enumerate(bounds):
                # Test near lower bound
                near_lower = np.zeros(len(bounds))
                near_lower[i] = lower + 0.01 if lower > -np.inf else -10
                boundary_params.append(near_lower)
                
                # Test near upper bound
                near_upper = np.zeros(len(bounds))
                near_upper[i] = upper - 0.01 if upper < np.inf else 10
                boundary_params.append(near_upper)
            
            # Test that boundary values produce finite likelihoods
            for params in boundary_params:
                loglik = objective(params)
                assert np.isfinite(loglik), f"Likelihood should be finite near boundaries: params={params}, loglik={loglik}"
    
    def test_statistical_inference_without_objective(self):
        """Test statistical inference behavior when objective function is not provided."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(data_context, design_matrices),
            context=data_context,
            bounds=model.get_parameter_bounds(data_context, design_matrices),
            preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        # Set up statistical info WITHOUT objective function
        result.result.set_statistical_info(['phi', 'p', 'f'], data_context.n_individuals)
        
        # Should still compute some statistics
        assert result.result.aic is not None, "AIC should work without objective function"
        assert result.result.bic is not None, "BIC should work without objective function"
        
        # Standard errors might not be available without fallback
        se = result.result.standard_errors
        # This is OK - SE might be None without objective function for fallback


class TestComprehensiveValidation:
    """Comprehensive validation tests covering the complete workflow."""
    
    def test_full_model_selection_workflow(self):
        """Test complete model selection workflow with multiple models."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        
        # Define a reasonable set of models
        model_specs = {
            "null": pj.create_simple_spec(phi='~1', p='~1', f='~1'),
            "phi_sex": pj.create_simple_spec(phi='~1 + sex', p='~1', f='~1'),
            "p_sex": pj.create_simple_spec(phi='~1', p='~1 + sex', f='~1')
        }
        
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
            
            if result.success:
                n_params = len(result.result.x)
                param_names = [f"param_{i}" for i in range(n_params)]
                result.result.set_statistical_info(param_names, data_context.n_individuals)
                fitted_models[name] = result
        
        assert len(fitted_models) >= 2, "At least 2 models should fit successfully"
        
        # Test model comparison
        comparison = compare_models(fitted_models)
        
        # Validate comparison results
        assert comparison.best_aic_model in fitted_models, "Best AIC model should be in fitted models"
        assert len(comparison.aic_ranking) == len(fitted_models), "AIC ranking should include all models"
        
        # Test that rankings make sense
        aic_values = [aic for _, aic in comparison.aic_ranking]
        assert aic_values == sorted(aic_values), "AIC ranking should be sorted"
        
        # Test delta calculations
        best_aic = min(aic_values)
        for model_name, delta in comparison.delta_aic.items():
            model_aic = fitted_models[model_name].result.aic
            expected_delta = model_aic - best_aic
            assert abs(delta - expected_delta) < 1e-6, f"Delta AIC calculation error for {model_name}"
        
        print(f"✅ Model selection workflow: {len(fitted_models)} models fitted, best = {comparison.best_aic_model}")
    
    def test_statistical_properties_consistency(self):
        """Test that statistical properties are mathematically consistent."""
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(data_context, design_matrices),
            context=data_context,
            bounds=model.get_parameter_bounds(data_context, design_matrices),
            preferred_strategy=OptimizationStrategy.SCIPY_LBFGS
        )
        
        result.result.set_statistical_info(['phi', 'p', 'f'], data_context.n_individuals, objective)
        
        # Test mathematical relationships
        k = len(result.result.x)
        n = data_context.n_individuals
        log_lik = result.result.log_likelihood
        
        # Manual AIC/BIC calculation
        expected_aic = 2 * k - 2 * log_lik
        expected_bic = k * np.log(n) - 2 * log_lik
        
        assert abs(result.result.aic - expected_aic) < 1e-6, "AIC calculation should be exact"
        assert abs(result.result.bic - expected_bic) < 1e-6, "BIC calculation should be exact"
        
        # Test that log-likelihood is negative of objective at optimum
        obj_at_optimum = objective(result.result.x)
        assert abs(result.result.log_likelihood + obj_at_optimum) < 1e-6, "Log-likelihood should be negative of objective"
        
        # Test confidence intervals are symmetric around estimate (for normal approximation)
        ci = result.result.confidence_intervals
        if ci is not None:
            for i in range(len(result.result.x)):
                estimate = result.result.x[i]
                lower, upper = ci[i]
                midpoint = (lower + upper) / 2
                assert abs(midpoint - estimate) < 1e-6, f"CI should be centered on estimate for parameter {i}"
        
        print("✅ Statistical properties consistency validated")


if __name__ == "__main__":
    # Run manual stress tests
    print("=== RUNNING MANUAL STRESS TESTS ===")
    
    # Test numerical stability
    test_numerical = TestNumericalStability()
    
    try:
        test_numerical.test_very_small_dataset()
        print("✅ Very small dataset handling: PASSED")
    except Exception as e:
        print(f"❌ Very small dataset: FAILED - {e}")
    
    try:
        test_numerical.test_extreme_initial_values()
        print("✅ Extreme initial values: PASSED")
    except Exception as e:
        print(f"❌ Extreme initial values: FAILED - {e}")
    
    # Test comprehensive validation
    test_comprehensive = TestComprehensiveValidation()
    
    try:
        test_comprehensive.test_full_model_selection_workflow()
        print("✅ Full model selection workflow: PASSED")
    except Exception as e:
        print(f"❌ Model selection workflow: FAILED - {e}")
    
    try:
        test_comprehensive.test_statistical_properties_consistency()
        print("✅ Statistical properties consistency: PASSED")
    except Exception as e:
        print(f"❌ Statistical consistency: FAILED - {e}")
    
    print("=== MANUAL STRESS TESTS COMPLETED ===")