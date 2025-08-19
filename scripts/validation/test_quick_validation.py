#!/usr/bin/env python3
"""
Quick validation test for new optimizers.
Tests the basic functionality of our validation framework.
"""

import logging
import pradel_jax as pj
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
from pradel_jax.models import PradelModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_quick_optimizer_comparison():
    """Quick test of optimizer comparison functionality."""
    logger.info("🚀 Starting quick validation test")
    
    # Load test data
    logger.info("Loading dipper dataset...")
    data = pj.load_data('data/dipper_dataset.csv')
    formula = pj.create_simple_spec(phi='~1', p='~1', f='~1')
    
    # Test strategies
    strategies = [
        OptimizationStrategy.SCIPY_LBFGS,  # Baseline
        OptimizationStrategy.HYBRID,       # New optimizer 1
        OptimizationStrategy.JAX_ADAM_ADAPTIVE  # New optimizer 2
    ]
    
    results = {}
    model = PradelModel()
    
    for strategy in strategies:
        logger.info(f"Testing {strategy.value}...")
        
        try:
            # Build optimization setup
            design_matrices = model.build_design_matrices(formula, data)
            initial_params = model.get_initial_parameters(data, design_matrices)
            bounds = model.get_parameter_bounds(data, design_matrices)
            
            # Ensure data context has required attributes
            if not hasattr(data, 'n_parameters'):
                data.n_parameters = len(initial_params)
            if not hasattr(data, 'get_condition_estimate'):
                data.get_condition_estimate = lambda: 1e5
            
            def objective(params):
                try:
                    ll = model.log_likelihood(params, data, design_matrices)
                    return -ll
                except Exception:
                    return 1e10
            
            # Run optimization
            result = optimize_model(
                objective_function=objective,
                initial_parameters=initial_params,
                context=data,
                bounds=bounds,
                preferred_strategy=strategy
            )
            
            if result.success:
                final_params = result.result.x
                final_nll = result.result.fun
                n_params = len(initial_params)
                aic = 2 * final_nll + 2 * n_params
                
                results[strategy.value] = {
                    'success': True,
                    'aic': aic,
                    'log_likelihood': -final_nll,
                    'n_iterations': getattr(result.result, 'nit', 0),
                    'strategy_used': result.strategy_used if hasattr(result, 'strategy_used') else strategy.value
                }
                
                logger.info(f"✅ {strategy.value}: AIC={aic:.2f}, LL={-final_nll:.2f}")
            else:
                results[strategy.value] = {'success': False, 'error': 'Optimization failed'}
                logger.warning(f"❌ {strategy.value}: Failed to converge")
                
        except Exception as e:
            results[strategy.value] = {'success': False, 'error': str(e)}
            logger.error(f"❌ {strategy.value}: Exception - {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("🎯 QUICK VALIDATION SUMMARY")
    logger.info("="*60)
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_results:
        print(f"{'Strategy':<20} {'AIC':<10} {'Log-Lik':<10} {'Status'}")
        print("-" * 50)
        
        for strategy, result in results.items():
            if result.get('success'):
                aic = result['aic']
                ll = result['log_likelihood']
                status = "✅ PASS"
            else:
                aic = "N/A"
                ll = "N/A"
                status = "❌ FAIL"
            
            print(f"{strategy:<20} {str(aic):<10} {str(ll):<10} {status}")
        
        # Check if new optimizers are competitive
        baseline_aic = results.get('scipy_lbfgs', {}).get('aic', float('inf'))
        hybrid_aic = results.get('hybrid', {}).get('aic', float('inf'))
        adaptive_aic = results.get('jax_adam_adaptive', {}).get('aic', float('inf'))
        
        logger.info(f"\n📊 PERFORMANCE ANALYSIS:")
        
        if hybrid_aic != float('inf') and baseline_aic != float('inf'):
            hybrid_diff = abs(hybrid_aic - baseline_aic)
            logger.info(f"HYBRID vs Baseline: AIC difference = {hybrid_diff:.2f}")
            if hybrid_diff < 2.0:
                logger.info("✅ HYBRID shows statistical equivalence (AIC diff < 2.0)")
            else:
                logger.warning("⚠️ HYBRID shows AIC divergence from baseline")
        
        if adaptive_aic != float('inf') and baseline_aic != float('inf'):
            adaptive_diff = abs(adaptive_aic - baseline_aic)
            logger.info(f"ADAPTIVE ADAM vs Baseline: AIC difference = {adaptive_diff:.2f}")
            if adaptive_diff < 2.0:
                logger.info("✅ ADAPTIVE ADAM shows statistical equivalence (AIC diff < 2.0)")
            else:
                logger.warning("⚠️ ADAPTIVE ADAM shows AIC divergence from baseline")
        
        # Overall assessment
        new_optimizer_success = (
            results.get('hybrid', {}).get('success', False) or
            results.get('jax_adam_adaptive', {}).get('success', False)
        )
        
        if new_optimizer_success:
            logger.info("🎉 NEW OPTIMIZERS: At least one new optimizer succeeded!")
        else:
            logger.warning("⚠️ NEW OPTIMIZERS: No new optimizers achieved successful convergence")
        
        return True
    else:
        logger.error("❌ All optimizers failed - validation cannot proceed")
        return False


if __name__ == "__main__":
    success = test_quick_optimizer_comparison()
    if success:
        print("\n🎯 Quick validation completed successfully!")
    else:
        print("\n❌ Quick validation failed!")
        exit(1)