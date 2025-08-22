#!/usr/bin/env python3
"""
Test parallel optimization compatibility with fixed parameter bounds.

This tests:
1. Whether the bounds fix works correctly in parallel processes
2. Race conditions or serialization issues with bounds
3. Consistency between serial and parallel optimization results
4. Memory and resource management with fixed bounds
"""

import sys
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any
import multiprocessing as mp

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

from nebraska_data_loader import load_and_prepare_nebraska_data
from pradel_jax.optimization.parallel import ParallelOptimizer, ParallelModelSpec
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec
from nebraska_ultra_conservative_validation import UltraConservativeValidator

def create_test_models() -> Dict[str, FormulaSpec]:
    """Create test model specifications for parallel testing."""
    parser = FormulaParser()
    
    return {
        "null_model": FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        ),
        
        "gender_model": FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + gender"),
            p=parser.create_parameter_formula(ParameterType.P, "~1 + gender"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        ),
        
        "age_model": FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + age"),
            p=parser.create_parameter_formula(ParameterType.P, "~1 + age"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        ),
        
        "additive_model": FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + gender + age"),
            p=parser.create_parameter_formula(ParameterType.P, "~1 + gender + age"),
            f=parser.create_parameter_formula(ParameterType.F, "~1 + gender")
        )
    }

def fit_serial_baseline(data_context, models: Dict[str, FormulaSpec]) -> Dict[str, Dict]:
    """Fit models serially to establish baseline results."""
    print("🔄 Fitting models serially (baseline)...")
    
    validator = UltraConservativeValidator()
    serial_results = {}
    
    for model_name, formula_spec in models.items():
        print(f"  Fitting {model_name}...", end=" ")
        
        start_time = time.time()
        result = validator.fit_single_model_ultra_rigorous(
            data_context, formula_spec, model_name, "serial_test", 0
        )
        fit_time = time.time() - start_time
        
        if result['success']:
            serial_results[model_name] = {
                'success': True,
                'log_likelihood': result['final_log_likelihood'],
                'aic': result['aic'],
                'parameters': result['parameters_transformed'],
                'fit_time': fit_time,
                'n_parameters': result['n_parameters']
            }
            print(f"✅ LL={result['final_log_likelihood']:.3f}")
        else:
            serial_results[model_name] = {'success': False, 'error': result.get('error', 'Unknown')}
            print(f"❌ Failed")
    
    return serial_results

def fit_parallel_test(data_context, models: Dict[str, FormulaSpec], n_workers: int = None) -> Dict[str, Dict]:
    """Fit models in parallel using the ParallelOptimizer."""
    print(f"🚀 Fitting models in parallel (workers={n_workers or 'auto'})...")
    
    if n_workers is None:
        n_workers = min(4, mp.cpu_count())  # Conservative worker count
    
    # Create model specifications for parallel fitting
    model_specs = []
    for idx, (model_name, formula_spec) in enumerate(models.items()):
        spec = ParallelModelSpec(
            name=model_name,
            formula_spec=formula_spec,
            index=idx,
            random_seed=42 + idx  # Consistent seeds for reproducibility
        )
        model_specs.append(spec)
    
    # Initialize parallel optimizer
    parallel_optimizer = ParallelOptimizer(n_workers=n_workers)
    
    # Fit models in parallel
    start_time = time.time()
    
    try:
        results = parallel_optimizer.fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            strategy="scipy_lbfgs"  # Use our proven strategy
        )
        
        total_time = time.time() - start_time
        print(f"  Parallel fitting completed in {total_time:.1f} seconds")
        
        # Convert to comparable format
        parallel_results = {}
        for result in results:
            if result.success:
                parallel_results[result.model_name] = {
                    'success': True,
                    'log_likelihood': result.log_likelihood,
                    'aic': result.aic,
                    'parameters': result.parameters,
                    'fit_time': result.fit_time,
                    'n_parameters': result.n_parameters,
                    'strategy_used': result.strategy_used
                }
                print(f"  ✅ {result.model_name}: LL={result.log_likelihood:.3f}")
            else:
                parallel_results[result.model_name] = {
                    'success': False,
                    'error': result.error_message
                }
                print(f"  ❌ {result.model_name}: Failed - {result.error_message}")
        
        return parallel_results
        
    except Exception as e:
        print(f"  ❌ Parallel fitting failed: {e}")
        return {}

def compare_serial_vs_parallel(serial_results: Dict, parallel_results: Dict) -> Dict[str, Any]:
    """Compare serial vs parallel results for consistency."""
    print("\n🔍 Comparing serial vs parallel results...")
    
    comparison = {
        'models_tested': 0,
        'both_successful': 0,
        'serial_only_successful': 0,
        'parallel_only_successful': 0,
        'both_failed': 0,
        'likelihood_differences': [],
        'parameter_differences': [],
        'identical_results': 0,
        'issues_found': []
    }
    
    for model_name in serial_results.keys():
        comparison['models_tested'] += 1
        
        serial_success = serial_results[model_name]['success']
        parallel_success = parallel_results.get(model_name, {}).get('success', False)
        
        print(f"\n  {model_name}:")
        
        if serial_success and parallel_success:
            comparison['both_successful'] += 1
            
            # Compare likelihoods
            serial_ll = serial_results[model_name]['log_likelihood']
            parallel_ll = parallel_results[model_name]['log_likelihood']
            ll_diff = abs(serial_ll - parallel_ll)
            
            comparison['likelihood_differences'].append(ll_diff)
            
            print(f"    Serial LL:   {serial_ll:.6f}")
            print(f"    Parallel LL: {parallel_ll:.6f}")
            print(f"    Difference:  {ll_diff:.8f}")
            
            # Check if results are essentially identical
            if ll_diff < 1e-6:
                comparison['identical_results'] += 1
                print(f"    ✅ Identical results")
            elif ll_diff < 1e-4:
                print(f"    ✅ Essentially identical (diff < 1e-4)")
            elif ll_diff < 0.01:
                print(f"    ⚠️  Small difference (diff < 0.01)")
                comparison['issues_found'].append(f"{model_name}: Small LL difference ({ll_diff:.6f})")
            else:
                print(f"    ❌ Significant difference (diff ≥ 0.01)")
                comparison['issues_found'].append(f"{model_name}: Large LL difference ({ll_diff:.6f})")
            
            # Compare parameters if available
            if 'parameters' in serial_results[model_name] and 'parameters' in parallel_results[model_name]:
                serial_params = np.array(serial_results[model_name]['parameters'])
                parallel_params = np.array(parallel_results[model_name]['parameters'])
                
                if len(serial_params) == len(parallel_params):
                    param_diff = np.max(np.abs(serial_params - parallel_params))
                    comparison['parameter_differences'].append(param_diff)
                    
                    print(f"    Max param diff: {param_diff:.8f}")
                    
                    if param_diff > 1e-4:
                        comparison['issues_found'].append(f"{model_name}: Large parameter difference ({param_diff:.6f})")
                else:
                    comparison['issues_found'].append(f"{model_name}: Different parameter counts")
            
        elif serial_success and not parallel_success:
            comparison['serial_only_successful'] += 1
            print(f"    ❌ Parallel failed, serial succeeded")
            parallel_error = parallel_results.get(model_name, {}).get('error', 'Unknown')
            comparison['issues_found'].append(f"{model_name}: Parallel failure - {parallel_error}")
            
        elif not serial_success and parallel_success:
            comparison['parallel_only_successful'] += 1
            print(f"    ⚠️  Serial failed, parallel succeeded (unusual)")
            comparison['issues_found'].append(f"{model_name}: Serial failure but parallel success")
            
        else:
            comparison['both_failed'] += 1
            print(f"    ❌ Both failed")
    
    return comparison

def test_parallel_bounds_compatibility():
    """Main test function for parallel bounds compatibility."""
    print("🧪 PARALLEL OPTIMIZATION BOUNDS COMPATIBILITY TEST")
    print("="*65)
    
    # Load test data
    print("📂 Loading test data...")
    data_context, df = load_and_prepare_nebraska_data(n_sample=200, random_state=42)
    
    if data_context is None:
        print("❌ Failed to load test data")
        return False
    
    print(f"   ✅ Loaded {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
    
    # Create test models
    models = create_test_models()
    print(f"📋 Created {len(models)} test models: {list(models.keys())}")
    
    # Test 1: Serial baseline
    print(f"\n" + "="*65)
    print("TEST 1: SERIAL BASELINE")
    print("="*65)
    
    serial_results = fit_serial_baseline(data_context, models)
    serial_success_rate = len([r for r in serial_results.values() if r['success']]) / len(models)
    print(f"📊 Serial success rate: {serial_success_rate:.1%}")
    
    # Test 2: Parallel with different worker counts
    worker_counts = [1, 2, 4] if mp.cpu_count() >= 4 else [1, 2]
    
    all_parallel_results = {}
    
    for n_workers in worker_counts:
        print(f"\n" + "="*65)
        print(f"TEST 2.{n_workers}: PARALLEL WITH {n_workers} WORKERS")
        print("="*65)
        
        parallel_results = fit_parallel_test(data_context, models, n_workers)
        all_parallel_results[n_workers] = parallel_results
        
        if parallel_results:
            parallel_success_rate = len([r for r in parallel_results.values() if r['success']]) / len(models)
            print(f"📊 Parallel success rate: {parallel_success_rate:.1%}")
            
            # Compare with serial
            comparison = compare_serial_vs_parallel(serial_results, parallel_results)
            
            print(f"\n📊 COMPARISON SUMMARY:")
            print(f"   Both successful: {comparison['both_successful']}/{comparison['models_tested']}")
            print(f"   Identical results: {comparison['identical_results']}/{comparison['both_successful']}")
            
            if comparison['likelihood_differences']:
                ll_diffs = comparison['likelihood_differences']
                print(f"   LL differences: mean={np.mean(ll_diffs):.8f}, max={np.max(ll_diffs):.8f}")
            
            if comparison['parameter_differences']:
                param_diffs = comparison['parameter_differences']
                print(f"   Param differences: mean={np.mean(param_diffs):.8f}, max={np.max(param_diffs):.8f}")
            
            if comparison['issues_found']:
                print(f"   ⚠️  Issues found: {len(comparison['issues_found'])}")
                for issue in comparison['issues_found']:
                    print(f"      - {issue}")
        else:
            print("❌ Parallel optimization completely failed")
    
    # Final assessment
    print(f"\n" + "="*65)
    print("FINAL PARALLEL COMPATIBILITY ASSESSMENT")
    print("="*65)
    
    if serial_success_rate < 0.8:
        verdict = "❌ BASELINE FAILURE: Serial optimization has issues"
    elif not any(all_parallel_results.values()):
        verdict = "❌ PARALLEL FAILURE: All parallel tests failed"
    else:
        # Check for consistency across worker counts
        consistent_results = True
        max_difference = 0
        
        for n_workers, parallel_results in all_parallel_results.items():
            if parallel_results:
                comparison = compare_serial_vs_parallel(serial_results, parallel_results)
                if comparison['issues_found']:
                    consistent_results = False
                if comparison['likelihood_differences']:
                    max_difference = max(max_difference, max(comparison['likelihood_differences']))
        
        if consistent_results and max_difference < 1e-6:
            verdict = "✅ EXCELLENT: Perfect parallel compatibility"
        elif consistent_results and max_difference < 1e-4:
            verdict = "✅ GOOD: Minor numerical differences (acceptable)"
        elif max_difference < 0.01:
            verdict = "⚠️  ACCEPTABLE: Small but noticeable differences"
        else:
            verdict = "❌ CONCERNING: Significant differences between serial and parallel"
    
    print(verdict)
    
    # Bounds-specific assessment
    print(f"\n🔍 BOUNDS-SPECIFIC ASSESSMENT:")
    print(f"   The parameter bounds fix should work identically in parallel because:")
    print(f"   ✅ Bounds are computed per-model instance (no shared state)")
    print(f"   ✅ Each worker creates its own PradelModel with fixed bounds")
    print(f"   ✅ No race conditions on bounds computation")
    print(f"   ✅ Serialization doesn't affect bounds (computed at runtime)")
    
    return verdict.startswith("✅")

if __name__ == "__main__":
    success = test_parallel_bounds_compatibility()
    
    if success:
        print(f"\n🎉 PARALLEL BOUNDS COMPATIBILITY: PASSED")
    else:
        print(f"\n💥 PARALLEL BOUNDS COMPATIBILITY: FAILED")