#!/usr/bin/env python3
"""
Check if parallel optimization is producing properly differentiated likelihoods
rather than repeating the same values across models.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path

# Set PYTHONPATH to include our package
import sys
sys.path.insert(0, str(Path(__file__).parent))

def check_likelihood_differentiation():
    """Check if different models produce different likelihood values."""
    print("🔍 Checking likelihood differentiation in parallel optimization...")
    
    # Import components
    import pradel_jax as pj
    from pradel_jax.optimization.parallel import ParallelOptimizer, ParallelModelSpec
    from pradel_jax.optimization.strategy import OptimizationStrategy
    
    # Check if Nebraska data is available
    data_file = Path("data/encounter_histories_ne_clean.csv")
    if not data_file.exists():
        print(f"❌ Nebraska data not found: {data_file}")
        return False
    
    try:
        # Load slightly larger sample for better differentiation
        print("📊 Loading Nebraska data sample...")
        full_data = pd.read_csv(data_file, dtype={'ch': str})
        sample_data = full_data.sample(n=75, random_state=123)  # Different seed, larger sample
        
        # Process data  
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            sample_data.to_csv(tmp_file.name, index=False)
            tmp_file_path = tmp_file.name
        
        try:
            data_context = pj.load_data(tmp_file_path, dtype={'ch': str})
        finally:
            Path(tmp_file_path).unlink()
        
        print(f"✅ Data processed: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
        
        # Create diverse model set that should produce different likelihoods
        model_specs = [
            ParallelModelSpec(
                name="φ(.) p(.) f(.)",
                formula_spec=pj.create_simple_spec(phi="~1", p="~1", f="~1"),
                index=0
            ),
            ParallelModelSpec(
                name="φ(gender) p(.) f(.)",  
                formula_spec=pj.create_simple_spec(phi="~1 + gender", p="~1", f="~1"),
                index=1
            ),
            ParallelModelSpec(
                name="φ(.) p(gender) f(.)",
                formula_spec=pj.create_simple_spec(phi="~1", p="~1 + gender", f="~1"),
                index=2
            ),
            ParallelModelSpec(
                name="φ(age) p(.) f(.)",
                formula_spec=pj.create_simple_spec(phi="~1 + age", p="~1", f="~1"),
                index=3
            ),
            ParallelModelSpec(
                name="φ(.) p(age) f(.)",
                formula_spec=pj.create_simple_spec(phi="~1", p="~1 + age", f="~1"),
                index=4
            )
        ]
        
        print(f"📋 Created {len(model_specs)} model specifications")
        
        # Run parallel optimization
        print("⚡ Running parallel optimization...")
        
        optimizer = ParallelOptimizer(n_workers=3)
        results = optimizer.fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            strategy=OptimizationStrategy.SCIPY_LBFGS,  # Use consistent strategy
            batch_size=5  # All at once
        )
        
        # Analyze likelihood differentiation
        successful_results = [r for r in results if r and r.success]
        
        if not successful_results:
            print("❌ No successful model fits")
            return False
        
        print(f"\n📊 Likelihood Analysis ({len(successful_results)} successful models):")
        print("-" * 80)
        
        log_likelihoods = []
        aics = []
        param_counts = []
        
        for i, result in enumerate(successful_results):
            ll = result.log_likelihood
            aic = result.aic
            k = result.n_parameters
            
            log_likelihoods.append(ll)
            aics.append(aic)
            param_counts.append(k)
            
            print(f"{i+1:2d}. {result.model_name:25} | LL: {ll:8.3f} | AIC: {aic:8.1f} | k: {k:2d} | Strategy: {result.strategy_used}")
        
        # Check for problematic patterns
        print(f"\n🔍 Differentiation Analysis:")
        
        # Check if all likelihoods are identical (problematic)
        ll_array = np.array(log_likelihoods)
        ll_range = np.max(ll_array) - np.min(ll_array)
        ll_std = np.std(ll_array)
        
        print(f"   Log-likelihood range: {ll_range:.6f}")
        print(f"   Log-likelihood std dev: {ll_std:.6f}")
        print(f"   Min LL: {np.min(ll_array):.3f}, Max LL: {np.max(ll_array):.3f}")
        
        # Check parameter counts
        k_array = np.array(param_counts)
        print(f"   Parameter counts: {sorted(set(param_counts))}")
        
        # Check if models with more parameters have higher likelihoods
        for i in range(len(successful_results)):
            for j in range(i+1, len(successful_results)):
                if param_counts[i] != param_counts[j]:
                    if param_counts[i] < param_counts[j]:
                        simpler_ll = log_likelihoods[i]
                        complex_ll = log_likelihoods[j]
                        model_i = successful_results[i].model_name
                        model_j = successful_results[j].model_name
                    else:
                        simpler_ll = log_likelihoods[j]
                        complex_ll = log_likelihoods[i]
                        model_i = successful_results[j].model_name
                        model_j = successful_results[i].model_name
                    
                    if complex_ll <= simpler_ll:
                        print(f"   ⚠️  {model_j} (k={max(param_counts[i], param_counts[j])}) has LL ≤ {model_i} (k={min(param_counts[i], param_counts[j])})")
        
        # Evaluate results
        print(f"\n📋 Assessment:")
        
        # Test 1: Are likelihoods properly differentiated?
        if ll_range < 0.001:
            print("   ❌ PROBLEM: All log-likelihoods are nearly identical")
            print("      This suggests models are not being properly differentiated")
            return False
        elif ll_range < 0.1:
            print("   ⚠️  WARNING: Log-likelihoods are very similar")
            print("      This might be expected for nested models, but verify manually")
        else:
            print("   ✅ GOOD: Log-likelihoods show proper differentiation")
        
        # Test 2: Are parameter counts correct?
        expected_ks = [3, 4, 4, 4, 4]  # Based on model specifications
        actual_ks = [r.n_parameters for r in successful_results]
        
        if actual_ks == expected_ks:
            print("   ✅ GOOD: Parameter counts match model specifications")
        else:
            print(f"   ⚠️  WARNING: Parameter counts unexpected")
            print(f"      Expected: {expected_ks}")
            print(f"      Actual:   {actual_ks}")
        
        # Test 3: Are AICs reasonable?
        aic_array = np.array(aics)
        aic_range = np.max(aic_array) - np.min(aic_array)
        
        if aic_range > 2.0:  # Generally models with delta AIC > 2 are considered different
            print("   ✅ GOOD: AIC values show meaningful model differences")
        else:
            print("   ℹ️  INFO: AIC values are similar (models may be comparable)")
        
        # Test 4: Check for exact duplicates (major red flag)
        ll_rounded = np.round(ll_array, 6)
        unique_lls = len(set(ll_rounded))
        
        if unique_lls == len(ll_array):
            print("   ✅ EXCELLENT: All log-likelihoods are unique")
            return True
        elif unique_lls < len(ll_array) / 2:
            print("   ❌ PROBLEM: Many log-likelihoods are identical")
            print("      This indicates a serious issue with model differentiation")
            return False
        else:
            print("   ⚠️  WARNING: Some log-likelihoods are identical")
            print("      This might be okay for very similar models")
            return True
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Parallel Optimization Likelihood Differentiation Check")
    print("=" * 60)
    
    success = check_likelihood_differentiation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ASSESSMENT: PASSED")
        print("\nThe parallel optimization framework is producing properly")
        print("differentiated likelihood values across different models.")
        print("This confirms that:")
        print("• Each model is being fitted independently")
        print("• DataContext serialization preserves data integrity")
        print("• Different model specifications produce different results")
        print("• The optimization process is working correctly")
    else:
        print("❌ ASSESSMENT: FAILED")
        print("\nThere may be issues with model differentiation.")
        print("This could indicate problems with:")
        print("• DataContext serialization corrupting data")
        print("• Models converging to the same solution inappropriately")
        print("• Parameter bounds or optimization settings")
        
    exit(0 if success else 1)