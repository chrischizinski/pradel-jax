#!/usr/bin/env python3
"""
Focused test of model differentiation - the key question you raised.

This tests whether different models actually produce different likelihoods and parameters
as they should if optimization is working correctly.
"""

import sys
import numpy as np
import jax.numpy as jnp
import pandas as pd
from scipy.optimize import minimize
import time

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

from nebraska_data_loader import load_and_prepare_nebraska_data
from nebraska_ultra_conservative_validation import UltraConservativeValidator

def test_model_differentiation_detailed():
    """
    Detailed test of whether different models produce meaningfully different results.
    
    This addresses the critical question: Should different models have different likelihoods?
    Answer: YES - if they don't, something is fundamentally wrong.
    """
    print("🔍 FOCUSED MODEL DIFFERENTIATION TEST")
    print("="*60)
    
    # Load one sample for detailed analysis
    print("Loading Nebraska data sample...")
    data_context, df = load_and_prepare_nebraska_data(n_sample=300, random_state=42)
    
    if data_context is None:
        print("❌ Failed to load data")
        return
    
    print(f"Data loaded: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
    
    # Create validator and get formula specs
    validator = UltraConservativeValidator()
    formulas = validator.create_formula_specifications()
    
    # Test these specific models for clear differentiation
    test_models = {
        'null_model': formulas['null_model'],
        'gender_survival': formulas['gender_survival'], 
        'additive_model': formulas['additive_model']
    }
    
    print(f"\nTesting {len(test_models)} different model specifications:")
    model_descriptions = {
        'null_model': 'phi=~1, p=~1, f=~1 (intercept only)',
        'gender_survival': 'phi=~1+gender, p=~1, f=~1 (gender affects survival)',
        'additive_model': 'phi=~1+gender+age, p=~1+gender+age, f=~1+gender (full additive)'
    }
    for name in test_models.keys():
        print(f"  {name}: {model_descriptions[name]}")
    
    # Fit each model multiple times and collect detailed results
    all_results = {}
    
    for model_name, formula_spec in test_models.items():
        print(f"\n📊 Fitting {model_name}...")
        
        model_results = []
        
        # Run each model 3 times to check for consistency
        for run in range(3):
            print(f"  Run {run + 1}: ", end="")
            
            result = validator.fit_single_model_ultra_rigorous(
                data_context, formula_spec, model_name, "test_sample", run
            )
            
            if result['success']:
                print(f"✅ LL={result['final_log_likelihood']:.4f}, AIC={result['aic']:.1f}")
            else:
                print(f"❌ Failed")
            
            model_results.append(result)
        
        all_results[model_name] = model_results
    
    # Analyze differentiation
    print(f"\n" + "="*60)
    print("DIFFERENTIATION ANALYSIS")
    print("="*60)
    
    # Extract successful results
    successful_results = {}
    for model_name, results in all_results.items():
        successful = [r for r in results if r['success']]
        if successful:
            # Use mean across runs
            mean_ll = np.mean([r['final_log_likelihood'] for r in successful])
            mean_aic = np.mean([r['aic'] for r in successful])
            std_ll = np.std([r['final_log_likelihood'] for r in successful])
            n_params = successful[0]['n_parameters']
            
            successful_results[model_name] = {
                'mean_ll': mean_ll,
                'std_ll': std_ll,
                'mean_aic': mean_aic,
                'n_parameters': n_params,
                'n_successful': len(successful),
                'example_params': successful[0]['parameters_natural']
            }
            
            print(f"{model_name}:")
            print(f"  Log-likelihood: {mean_ll:.4f} ± {std_ll:.6f}")
            print(f"  AIC: {mean_aic:.1f}")
            print(f"  Parameters: {n_params}")
            print(f"  Success rate: {len(successful)}/3")
    
    # Critical analysis: Are models actually different?
    print(f"\n🔍 CRITICAL DIFFERENTIATION CHECK:")
    
    model_names = list(successful_results.keys())
    
    if len(model_names) < 2:
        print("❌ INSUFFICIENT SUCCESSFUL MODELS for comparison")
        return
    
    # Compare each pair of models
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]
            
            ll1 = successful_results[model1]['mean_ll']
            ll2 = successful_results[model2]['mean_ll']
            
            ll_diff = abs(ll1 - ll2)
            aic1 = successful_results[model1]['mean_aic']
            aic2 = successful_results[model2]['mean_aic']
            aic_diff = abs(aic1 - aic2)
            
            print(f"\n  {model1} vs {model2}:")
            print(f"    ΔLog-likelihood: {ll_diff:.6f}")
            print(f"    ΔAIC: {aic_diff:.2f}")
            
            # Assess whether difference is meaningful
            if ll_diff < 1e-6:
                print(f"    ❌ CRITICAL: Identical likelihoods (diff < 1e-6)")
                verdict = "IDENTICAL"
            elif ll_diff < 0.01:
                print(f"    ⚠️  WARNING: Very similar likelihoods (diff < 0.01)")
                verdict = "TOO_SIMILAR"
            elif ll_diff < 0.1:
                print(f"    🟡 MARGINAL: Small likelihood difference (diff < 0.1)")
                verdict = "MARGINAL"
            else:
                print(f"    ✅ GOOD: Meaningful likelihood difference")
                verdict = "MEANINGFUL"
            
            # Check AIC difference (should be > 2 for meaningful model differences)
            if aic_diff < 2:
                print(f"    📊 AIC suggests models are statistically equivalent (ΔAIC < 2)")
            elif aic_diff < 7:
                print(f"    📊 AIC suggests moderate evidence for difference (2 ≤ ΔAIC < 7)")
            else:
                print(f"    📊 AIC suggests strong evidence for difference (ΔAIC ≥ 7)")
    
    # Parameter differentiation check
    print(f"\n📋 PARAMETER DIFFERENTIATION CHECK:")
    
    # Find parameters that should be different between models
    for model_name, results in successful_results.items():
        params = results['example_params']
        print(f"\n  {model_name} parameters (natural scale):")
        for param_name, value in params.items():
            print(f"    {param_name}: {value:.6f}")
    
    # Final verdict
    print(f"\n" + "="*60)
    print("FINAL MODEL DIFFERENTIATION VERDICT")
    print("="*60)
    
    # Count meaningful differences
    meaningful_count = 0
    total_comparisons = 0
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            total_comparisons += 1
            model1, model2 = model_names[i], model_names[j]
            ll1 = successful_results[model1]['mean_ll']
            ll2 = successful_results[model2]['mean_ll']
            ll_diff = abs(ll1 - ll2)
            
            if ll_diff >= 0.1:
                meaningful_count += 1
    
    meaningful_rate = meaningful_count / total_comparisons if total_comparisons > 0 else 0.0
    
    print(f"Meaningful model differences: {meaningful_count}/{total_comparisons} ({meaningful_rate:.1%})")
    
    if meaningful_rate >= 0.8:
        print("✅ EXCELLENT: Models are properly differentiated")
    elif meaningful_rate >= 0.5:
        print("🟡 ACCEPTABLE: Most models are differentiated")
    else:
        print("❌ CONCERNING: Models are insufficiently differentiated")
        print("   This suggests potential issues with:")
        print("   - Model specification")
        print("   - Optimization convergence")
        print("   - Parameter identifiability")
        print("   - Data informativeness")
    
    return successful_results

if __name__ == "__main__":
    test_model_differentiation_detailed()