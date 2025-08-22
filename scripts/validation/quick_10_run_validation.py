#!/usr/bin/env python3
"""
Quick 10-run validation focusing on the core question:
Are models consistently differentiated across multiple validation runs?
"""

import sys
import numpy as np
import time

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

from nebraska_data_loader import load_and_prepare_nebraska_data
from nebraska_ultra_conservative_validation import UltraConservativeValidator

def quick_validation_run(run_id: int):
    """Quick validation run focusing on model differentiation."""
    
    print(f"Run {run_id + 1}: ", end="")
    
    # Load smaller sample for speed
    data_context, df = load_and_prepare_nebraska_data(n_sample=200, random_state=42 + run_id * 100)
    
    if data_context is None:
        print("❌ Data loading failed")
        return None
    
    # Test core models
    validator = UltraConservativeValidator()
    formulas = validator.create_formula_specifications()
    
    test_models = ['null_model', 'gender_survival', 'age_linear']
    
    results = {}
    
    # Fit each model once
    for model_name in test_models:
        try:
            result = validator.fit_single_model_ultra_rigorous(
                data_context, formulas[model_name], model_name, f"run_{run_id}", 0
            )
            
            if result['success']:
                results[model_name] = result['final_log_likelihood']
            else:
                results[model_name] = None
                
        except Exception as e:
            results[model_name] = None
    
    # Check differentiation
    successful_models = {k: v for k, v in results.items() if v is not None}
    
    if len(successful_models) >= 2:
        # Calculate minimum difference
        lls = list(successful_models.values())
        min_diff = float('inf')
        max_diff = 0
        
        for i in range(len(lls)):
            for j in range(i+1, len(lls)):
                diff = abs(lls[i] - lls[j])
                min_diff = min(min_diff, diff)
                max_diff = max(max_diff, diff)
        
        # Report
        success_rate = len(successful_models) / len(test_models)
        
        if min_diff < 0.01:
            status = "❌ TOO_SIMILAR"
        elif min_diff < 0.1:
            status = "⚠️  MARGINAL"
        else:
            status = "✅ GOOD"
        
        print(f"{status} Success: {success_rate:.0%}, MinΔLL: {min_diff:.3f}, MaxΔLL: {max_diff:.3f}")
        
        return {
            'run_id': run_id,
            'success_rate': success_rate,
            'min_ll_diff': min_diff,
            'max_ll_diff': max_diff,
            'results': successful_models
        }
    else:
        print(f"❌ FAILED - Only {len(successful_models)} successful models")
        return None

def run_10_validation_comparison():
    """Run 10 quick validation runs and compare results."""
    
    print("QUICK 10-RUN VALIDATION: MODEL DIFFERENTIATION TEST")
    print("="*60)
    print("Testing whether models produce consistently different likelihoods")
    print("across multiple independent validation runs.\n")
    
    all_results = []
    
    for run_id in range(10):
        result = quick_validation_run(run_id)
        if result:
            all_results.append(result)
    
    print(f"\n" + "="*60)
    print("SUMMARY ACROSS 10 VALIDATION RUNS")
    print("="*60)
    
    if not all_results:
        print("❌ ALL VALIDATION RUNS FAILED")
        return
    
    # Success rate analysis
    success_rates = [r['success_rate'] for r in all_results]
    min_diffs = [r['min_ll_diff'] for r in all_results]
    max_diffs = [r['max_ll_diff'] for r in all_results]
    
    print(f"Successful validation runs: {len(all_results)}/10")
    print(f"Model success rate: {np.mean(success_rates):.1%} ± {np.std(success_rates):.3f}")
    print(f"Minimum LL difference: {np.mean(min_diffs):.3f} ± {np.std(min_diffs):.3f}")
    print(f"Maximum LL difference: {np.mean(max_diffs):.1f} ± {np.std(max_diffs):.1f}")
    
    # Consistency analysis
    consistent_differentiation = len([r for r in all_results if r['min_ll_diff'] >= 0.1])
    marginal_differentiation = len([r for r in all_results if 0.01 <= r['min_ll_diff'] < 0.1])
    poor_differentiation = len([r for r in all_results if r['min_ll_diff'] < 0.01])
    
    print(f"\nModel differentiation quality:")
    print(f"  ✅ Good differentiation (MinΔLL ≥ 0.1): {consistent_differentiation}/10 ({consistent_differentiation*10}%)")
    print(f"  ⚠️  Marginal differentiation (0.01 ≤ MinΔLL < 0.1): {marginal_differentiation}/10 ({marginal_differentiation*10}%)")
    print(f"  ❌ Poor differentiation (MinΔLL < 0.01): {poor_differentiation}/10 ({poor_differentiation*10}%)")
    
    # Cross-run model consistency
    print(f"\nCross-run model consistency:")
    
    # Collect model-specific results
    model_results = {'null_model': [], 'gender_survival': [], 'age_linear': []}
    
    for result in all_results:
        for model_name, ll in result['results'].items():
            if ll is not None:
                model_results[model_name].append(ll)
    
    for model_name, lls in model_results.items():
        if len(lls) > 1:
            ll_range = np.max(lls) - np.min(lls) 
            ll_cv = np.std(lls) / abs(np.mean(lls))
            print(f"  {model_name}: Range={ll_range:.1f}, CV={ll_cv:.6f}")
            
            if ll_cv > 0.01:
                print(f"    ⚠️  High variation across runs")
    
    # Final verdict
    print(f"\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    
    overall_success_rate = np.mean(success_rates)
    good_differentiation_rate = consistent_differentiation / 10
    
    if overall_success_rate >= 0.95 and good_differentiation_rate >= 0.8:
        verdict = "✅ EXCELLENT: Consistent model differentiation across validation runs"
    elif overall_success_rate >= 0.9 and good_differentiation_rate >= 0.6:
        verdict = "🟡 GOOD: Generally consistent with some variation"
    elif overall_success_rate >= 0.8:
        verdict = "⚠️  ACCEPTABLE: Some inconsistencies detected"
    else:
        verdict = "❌ CONCERNING: Significant inconsistencies in model differentiation"
    
    print(verdict)
    
    return all_results

if __name__ == "__main__":
    run_10_validation_comparison()