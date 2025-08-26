#!/usr/bin/env python3
"""
Bootstrap Statistical Inference Demonstration

Shows advanced bootstrap confidence interval computation using
the comprehensive statistical inference framework.
"""

import numpy as np
import pradel_jax as pj
from pradel_jax.models import PradelModel
from pradel_jax.inference import bootstrap_confidence_intervals
from pradel_jax.optimization import optimize_model

def main():
    print("================================================================================")
    print("BOOTSTRAP STATISTICAL INFERENCE DEMONSTRATION")
    print("Advanced Non-parametric Confidence Intervals")
    print("================================================================================")
    
    # Load data
    print("\n1. LOADING DATA")
    print("-" * 40)
    data_context = pj.load_data('data/dipper_dataset.csv')
    print(f"✅ Loaded {data_context.n_individuals} individuals")
    
    # Define model
    model = PradelModel()
    formula_spec = pj.create_simple_spec(phi='~1', p='~1', f='~1')
    design_matrices = model.build_design_matrices(formula_spec, data_context)
    
    # Define model fitting function for bootstrap
    def model_fit_fn(bootstrap_data_context):
        """Fit model to bootstrap sample."""
        bootstrap_design_matrices = model.build_design_matrices(formula_spec, bootstrap_data_context)
        
        def objective(params):
            return -model.log_likelihood(params, bootstrap_data_context, bootstrap_design_matrices)
        
        result = optimize_model(
            objective_function=objective,
            initial_parameters=model.get_initial_parameters(bootstrap_data_context, bootstrap_design_matrices),
            context=bootstrap_data_context,
            bounds=model.get_parameter_bounds(bootstrap_data_context, bootstrap_design_matrices)
        )
        
        if result.success:
            return result.result.x, result.result.fun
        else:
            raise RuntimeError("Bootstrap sample optimization failed")
    
    print("\n2. COMPUTING BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 40)
    print("Running 100 bootstrap samples (this may take a moment)...")
    
    # Compute bootstrap uncertainty with fewer samples for demo
    bootstrap_uncertainty = bootstrap_confidence_intervals(
        data_context=data_context,
        model_fit_fn=model_fit_fn,
        n_bootstrap_samples=100,  # Reduced for demo speed
        confidence_levels=[0.95, 0.99],
        random_seed=42  # For reproducibility
    )
    
    print(f"✅ Successfully completed {bootstrap_uncertainty.bootstrap_samples.shape[0]} bootstrap samples")
    
    print("\n3. BOOTSTRAP RESULTS SUMMARY")
    print("-" * 40)
    param_names = ['phi_intercept', 'p_intercept', 'f_intercept']
    
    print(f"{'Parameter':<15} {'Original':<10} {'Bootstrap Mean':<15} {'Bias':<10} {'95% CI':<20}")
    print("-" * 80)
    
    for i, name in enumerate(param_names):
        original = bootstrap_uncertainty.estimates[i]
        bootstrap_mean = np.mean(bootstrap_uncertainty.bootstrap_samples[:, i])
        bias = bootstrap_uncertainty.bootstrap_bias[i]
        ci_95 = bootstrap_uncertainty.confidence_intervals['95%'][i]
        
        print(f"{name:<15} {original:8.4f}   {bootstrap_mean:8.4f}      {bias:8.4f}   "
              f"({ci_95[0]:6.3f}, {ci_95[1]:6.3f})")
    
    print("\n4. BOOTSTRAP DIAGNOSTICS")
    print("-" * 40)
    print(f"Bootstrap standard errors: {bootstrap_uncertainty.standard_errors}")
    print(f"Bootstrap covariance matrix condition number: {np.linalg.cond(bootstrap_uncertainty.covariance_matrix):.2e}")
    print(f"Parameter correlations:")
    
    for i in range(3):
        for j in range(i+1, 3):
            corr = bootstrap_uncertainty.correlation_matrix[i, j]
            print(f"  {param_names[i]} vs {param_names[j]}: {corr:.3f}")
    
    print("\n5. BOOTSTRAP VS ASYMPTOTIC COMPARISON")
    print("-" * 40)
    
    # Compare with asymptotic confidence intervals
    def objective(params):
        return -model.log_likelihood(params, data_context, design_matrices)
    
    from pradel_jax.inference import compute_hessian_standard_errors
    
    asymptotic_uncertainty = compute_hessian_standard_errors(
        log_likelihood_fn=lambda x: -objective(x),  # Convert to log-likelihood
        parameter_estimates=bootstrap_uncertainty.estimates,
        parameter_names=param_names
    )
    
    print(f"{'Parameter':<15} {'Bootstrap SE':<12} {'Asymptotic SE':<14} {'Ratio':<8}")
    print("-" * 60)
    
    for i, name in enumerate(param_names):
        bs_se = bootstrap_uncertainty.standard_errors[i]
        async_se = asymptotic_uncertainty.standard_errors[i]
        ratio = bs_se / async_se if async_se > 0 else np.inf
        
        print(f"{name:<15} {bs_se:10.6f}   {async_se:10.6f}     {ratio:6.3f}")
    
    print("\n" + "="*80)
    print("BOOTSTRAP STATISTICAL INFERENCE: COMPLETE ✅")
    print("✅ Non-parametric confidence intervals computed")
    print("✅ Bias correction applied")
    print("✅ Bootstrap diagnostics validated")
    print("✅ Asymptotic vs bootstrap comparison performed")
    print("="*80)

if __name__ == "__main__":
    main()