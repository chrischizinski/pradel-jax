#!/usr/bin/env python3
"""
Run the ultra-conservative validation 20 times to test consistency across validation runs.

This meta-validation tests:
1. Consistency of validation results across multiple runs
2. Stability of optimization performance over time
3. Robustness of the validation framework itself
4. Detection of any intermittent issues

Each validation run uses completely independent random samples.
"""

import sys
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import warnings

# Add path and suppress warnings
sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')
warnings.filterwarnings("ignore", ".*TPU.*")

from nebraska_ultra_conservative_validation import UltraConservativeValidator

class MetaValidationRunner:
    """Run multiple validation runs and analyze meta-consistency."""
    
    def __init__(self, n_validation_runs: int = 20):
        self.n_validation_runs = n_validation_runs
        self.validation_results = []
        self.meta_analysis = {}
        
    def run_single_validation(self, run_id: int) -> Dict[str, Any]:
        """Run a single validation and extract key metrics."""
        print(f"\n" + "="*60)
        print(f"VALIDATION RUN {run_id + 1} of {self.n_validation_runs}")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Create validator with different random seed for each run
            validator = UltraConservativeValidator()
            
            # Temporarily modify the validation to use different random seeds
            # We'll create a custom version that changes seeds
            
            # Run validation
            detailed_results = self.run_custom_validation(validator, run_id)
            
            run_time = time.time() - start_time
            
            # Extract summary metrics
            total_runs = len(detailed_results)
            successful_runs = len([r for r in detailed_results if r.get('success', False)])
            
            # Group by model
            models = set(r['model_name'] for r in detailed_results)
            model_stats = {}
            
            for model_name in models:
                model_results = [r for r in detailed_results if r['model_name'] == model_name]
                model_successful = [r for r in model_results if r.get('success', False)]
                
                if model_successful:
                    lls = [r['final_log_likelihood'] for r in model_successful]
                    aics = [r['aic'] for r in model_successful]
                    
                    model_stats[model_name] = {
                        'success_rate': len(model_successful) / len(model_results),
                        'mean_ll': np.mean(lls),
                        'std_ll': np.std(lls),
                        'mean_aic': np.mean(aics),
                        'std_aic': np.std(aics),
                        'n_successful': len(model_successful),
                        'n_total': len(model_results)
                    }
                else:
                    model_stats[model_name] = {
                        'success_rate': 0.0,
                        'n_successful': 0,
                        'n_total': len(model_results)
                    }
            
            # Count ultra-consistent results
            consistency_reports = []
            
            # Group results by sample and model for consistency analysis
            samples = set((r['sample_id'], r['model_name']) for r in detailed_results)
            ultra_consistent_count = 0
            
            for sample_id, model_name in samples:
                sample_model_results = [r for r in detailed_results 
                                      if r['sample_id'] == sample_id and r['model_name'] == model_name]
                
                consistency = validator.analyze_within_model_consistency(sample_model_results)
                consistency_reports.append(consistency)
                
                if consistency['status'] == 'ULTRA_CONSISTENT':
                    ultra_consistent_count += 1
            
            validation_summary = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'run_time_seconds': run_time,
                'total_optimization_runs': total_runs,
                'successful_optimization_runs': successful_runs,
                'overall_success_rate': successful_runs / total_runs if total_runs > 0 else 0.0,
                'model_statistics': model_stats,
                'ultra_consistent_count': ultra_consistent_count,
                'total_consistency_tests': len(consistency_reports),
                'ultra_consistent_rate': ultra_consistent_count / len(consistency_reports) if consistency_reports else 0.0,
                'validation_passed': (successful_runs / total_runs >= 0.95 and 
                                    ultra_consistent_count / len(consistency_reports) >= 0.8 if consistency_reports else False)
            }
            
            print(f"\nVALIDATION RUN {run_id + 1} SUMMARY:")
            print(f"  Success rate: {validation_summary['overall_success_rate']:.1%}")
            print(f"  Ultra-consistent rate: {validation_summary['ultra_consistent_rate']:.1%}")
            print(f"  Validation passed: {validation_summary['validation_passed']}")
            print(f"  Run time: {run_time:.1f} seconds")
            
            return validation_summary
            
        except Exception as e:
            error_summary = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'run_time_seconds': time.time() - start_time,
                'error': str(e),
                'validation_passed': False
            }
            
            print(f"\n❌ VALIDATION RUN {run_id + 1} FAILED: {e}")
            
            return error_summary
    
    def run_custom_validation(self, validator: UltraConservativeValidator, run_id: int) -> List[Dict[str, Any]]:
        """Run custom validation with different random seeds per validation run."""
        from nebraska_data_loader import load_and_prepare_nebraska_data
        
        # Custom validation parameters - smaller for 20 runs
        sample_sizes = [200, 500]  
        n_independent_samples = 2  # Reduced from 3 to 2
        n_optimization_runs = 3   # Reduced from 5 to 3
        
        formulas = validator.create_formula_specifications()
        test_models = ['null_model', 'gender_survival', 'age_linear']  # Reduced from 4 to 3
        
        # Use different base random seed for each validation run
        base_seed = 42 + run_id * 1000
        
        all_results = []
        
        # For each sample size
        for size_idx, sample_size in enumerate(sample_sizes):
            
            # Generate multiple independent samples with run-specific seeds
            samples_data = []
            for sample_idx in range(n_independent_samples):
                seed = base_seed + size_idx * 100 + sample_idx * 10
                
                data_context, df = load_and_prepare_nebraska_data(
                    n_sample=sample_size, 
                    random_state=seed
                )
                
                if data_context is not None:
                    samples_data.append((data_context, df))
            
            # Test each model on each sample
            for model_name in test_models:
                formula_spec = formulas[model_name]
                
                # Test on each independent sample
                for sample_idx, (data_context, df) in enumerate(samples_data):
                    sample_id = f"size_{sample_size}_sample_{sample_idx}_run_{run_id}"
                    
                    # Multiple optimization runs on same data
                    for opt_run_id in range(n_optimization_runs):
                        result = validator.fit_single_model_ultra_rigorous(
                            data_context, formula_spec, model_name, sample_id, opt_run_id
                        )
                        all_results.append(result)
        
        return all_results
    
    def analyze_consistency_simple(self, results: List[Dict]) -> Dict[str, Any]:
        """Simple consistency analysis for meta-validation."""
        successful_runs = [r for r in results if r.get('success', False)]
        n_successful = len(successful_runs)
        n_total = len(results)
        
        if n_successful == 0:
            return {'status': 'ALL_FAILED'}
        
        if n_successful < n_total:
            return {'status': 'PARTIAL_FAILURE'}
        
        # Check log-likelihood consistency
        if len(successful_runs) > 1:
            log_likelihoods = [r['final_log_likelihood'] for r in successful_runs]
            ll_std = np.std(log_likelihoods)
            ll_mean = np.mean(log_likelihoods)
            ll_cv = ll_std / abs(ll_mean) if ll_mean != 0 else np.inf
            
            if ll_cv < 1e-6:
                return {'status': 'ULTRA_CONSISTENT'}
            elif ll_cv < 1e-4:
                return {'status': 'MARGINALLY_CONSISTENT'}
            else:
                return {'status': 'INCONSISTENT'}
        else:
            return {'status': 'ULTRA_CONSISTENT'}
    
    def run_all_validations(self):
        """Run all validation runs and collect results."""
        print("="*80)
        print(f"RUNNING {self.n_validation_runs} ULTRA-CONSERVATIVE VALIDATION RUNS")
        print("="*80)
        
        print(f"Each validation run will test:")
        print(f"  - 2 sample sizes (200, 500 individuals)")
        print(f"  - 2 independent samples per size") 
        print(f"  - 3 model types (null, gender_survival, age_linear)")
        print(f"  - 3 optimization runs per model")
        print(f"  - Total: 36 optimization runs per validation")
        print(f"  - Grand total: {36 * self.n_validation_runs} optimization runs")
        
        overall_start_time = time.time()
        
        for run_id in range(self.n_validation_runs):
            validation_result = self.run_single_validation(run_id)
            self.validation_results.append(validation_result)
            
            # Progress update
            completed = run_id + 1
            progress = completed / self.n_validation_runs
            elapsed = time.time() - overall_start_time
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            
            print(f"\nPROGRESS: {completed}/{self.n_validation_runs} ({progress:.1%}) - "
                  f"Elapsed: {elapsed:.0f}s, Estimated remaining: {remaining:.0f}s")
        
        total_time = time.time() - overall_start_time
        
        print(f"\n" + "="*80)
        print(f"ALL {self.n_validation_runs} VALIDATION RUNS COMPLETED")
        print(f"Total time: {total_time:.0f} seconds ({total_time/60:.1f} minutes)")
        print("="*80)
        
        # Generate meta-analysis
        self.generate_meta_analysis()
        
        return self.validation_results
    
    def generate_meta_analysis(self):
        """Generate comprehensive meta-analysis of all validation runs."""
        
        print("\n" + "="*80)
        print("META-ANALYSIS: CONSISTENCY ACROSS 20 VALIDATION RUNS")
        print("="*80)
        
        # Overall statistics
        successful_validations = [r for r in self.validation_results if r.get('validation_passed', False)]
        n_successful_validations = len(successful_validations)
        
        print(f"VALIDATION-LEVEL RESULTS:")
        print(f"  Successful validation runs: {n_successful_validations}/{self.n_validation_runs} ({n_successful_validations/self.n_validation_runs:.1%})")
        
        # Collect all optimization success rates
        success_rates = [r['overall_success_rate'] for r in self.validation_results if 'overall_success_rate' in r]
        ultra_consistent_rates = [r['ultra_consistent_rate'] for r in self.validation_results if 'ultra_consistent_rate' in r]
        
        if success_rates:
            print(f"  Optimization success rate: {np.mean(success_rates):.3f} ± {np.std(success_rates):.6f}")
            print(f"    Min: {np.min(success_rates):.3f}, Max: {np.max(success_rates):.3f}")
        
        if ultra_consistent_rates:
            print(f"  Ultra-consistent rate: {np.mean(ultra_consistent_rates):.3f} ± {np.std(ultra_consistent_rates):.6f}")
            print(f"    Min: {np.min(ultra_consistent_rates):.3f}, Max: {np.max(ultra_consistent_rates):.3f}")
        
        # CRITICAL ANALYSIS: Model differentiation and likelihood differences
        print(f"\n🔍 CRITICAL ANALYSIS: MODEL DIFFERENTIATION")
        print("="*60)
        
        self.analyze_model_differentiation()
        
        # Model-specific meta-analysis
        print(f"\nMODEL-SPECIFIC META-ANALYSIS:")
        
        # Collect model statistics across all runs
        all_models = set()
        for result in self.validation_results:
            if 'model_statistics' in result:
                all_models.update(result['model_statistics'].keys())
        
        for model_name in sorted(all_models):
            model_success_rates = []
            model_lls = []
            model_aics = []
            
            for result in self.validation_results:
                if 'model_statistics' in result and model_name in result['model_statistics']:
                    stats = result['model_statistics'][model_name]
                    model_success_rates.append(stats['success_rate'])
                    
                    if 'mean_ll' in stats:
                        model_lls.append(stats['mean_ll'])
                    if 'mean_aic' in stats:
                        model_aics.append(stats['mean_aic'])
            
            print(f"  {model_name}:")
            if model_success_rates:
                print(f"    Success rate: {np.mean(model_success_rates):.3f} ± {np.std(model_success_rates):.6f}")
            if model_lls:
                print(f"    Log-likelihood: {np.mean(model_lls):.1f} ± {np.std(model_lls):.3f}")
                print(f"      Range: [{np.min(model_lls):.1f}, {np.max(model_lls):.1f}]")
            if model_aics:
                print(f"    AIC: {np.mean(model_aics):.1f} ± {np.std(model_aics):.3f}")
        
        # Runtime analysis
        run_times = [r['run_time_seconds'] for r in self.validation_results if 'run_time_seconds' in r]
        if run_times:
            print(f"\nPERFORMANCE ANALYSIS:")
            print(f"  Validation run time: {np.mean(run_times):.1f} ± {np.std(run_times):.1f} seconds")
            print(f"    Min: {np.min(run_times):.1f}s, Max: {np.max(run_times):.1f}s")
        
        # Meta-validation verdict
        print(f"\n" + "="*80)
        print("META-VALIDATION FINAL VERDICT")
        print("="*80)
        
        if n_successful_validations < self.n_validation_runs * 0.95:
            verdict = f"🚫 META-VALIDATION FAILED - Only {n_successful_validations}/{self.n_validation_runs} validation runs passed"
        elif success_rates and np.min(success_rates) < 0.95:
            verdict = f"⚠️  META-VALIDATION CONCERNING - Minimum optimization success rate: {np.min(success_rates):.3f}"
        elif ultra_consistent_rates and np.min(ultra_consistent_rates) < 0.8:
            verdict = f"🔶 META-VALIDATION QUESTIONABLE - Minimum ultra-consistent rate: {np.min(ultra_consistent_rates):.3f}"
        else:
            verdict = "✅ META-VALIDATION PASSED - Consistent performance across all validation runs"
        
        print(verdict)
        
        # Stability assessment
        success_rate_cv = np.std(success_rates) / np.mean(success_rates) if success_rates and np.mean(success_rates) > 0 else np.inf
        ultra_consistent_cv = np.std(ultra_consistent_rates) / np.mean(ultra_consistent_rates) if ultra_consistent_rates and np.mean(ultra_consistent_rates) > 0 else np.inf
        
        print(f"\nSTABILITY ASSESSMENT:")
        print(f"  Success rate CV: {success_rate_cv:.6f}")
        print(f"  Ultra-consistent rate CV: {ultra_consistent_cv:.6f}")
        
        if success_rate_cv < 0.01 and ultra_consistent_cv < 0.01:
            print("  🎯 ULTRA-STABLE: Extremely consistent performance")
        elif success_rate_cv < 0.05 and ultra_consistent_cv < 0.05:
            print("  ✅ STABLE: Consistent performance across runs")
        else:
            print("  ⚠️  VARIABLE: Some variation in performance detected")
        
        # Store meta-analysis
        self.meta_analysis = {
            'n_validation_runs': self.n_validation_runs,
            'successful_validation_runs': n_successful_validations,
            'validation_success_rate': n_successful_validations / self.n_validation_runs,
            'optimization_success_rates': success_rates,
            'ultra_consistent_rates': ultra_consistent_rates,
            'success_rate_stability': success_rate_cv,
            'ultra_consistent_stability': ultra_consistent_cv,
            'verdict': verdict
        }
        
        return self.meta_analysis
    
    def analyze_model_differentiation(self):
        """
        Critical analysis: Different models should have meaningfully different likelihoods.
        This tests whether the optimization is actually finding different solutions for different models.
        """
        print("Analyzing model differentiation within validation runs...")
        
        # Collect detailed results from each validation run
        differentiation_issues = []
        sample_analyses = []
        
        for validation_idx, validation_result in enumerate(self.validation_results):
            if 'model_statistics' not in validation_result:
                continue
                
            models = validation_result['model_statistics']
            model_names = list(models.keys())
            
            # For each pair of models, check if their likelihoods are meaningfully different
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    
                    if 'mean_ll' in models[model1] and 'mean_ll' in models[model2]:
                        ll1 = models[model1]['mean_ll']
                        ll2 = models[model2]['mean_ll']
                        ll_diff = abs(ll1 - ll2)
                        
                        # Models should differ by at least 0.1 log-likelihood units for meaningful difference
                        # (This is a conservative threshold - in practice differences should be larger)
                        if ll_diff < 0.1:
                            differentiation_issues.append({
                                'validation_run': validation_idx,
                                'model1': model1,
                                'model2': model2,
                                'll1': ll1,
                                'll2': ll2,
                                'll_difference': ll_diff,
                                'issue': 'TOO_SIMILAR'
                            })
                        
                        sample_analyses.append({
                            'validation_run': validation_idx,
                            'model1': model1,
                            'model2': model2,
                            'll_difference': ll_diff,
                            'meaningful_difference': ll_diff >= 0.1
                        })
        
        # Report findings
        total_comparisons = len(sample_analyses)
        meaningful_differences = len([a for a in sample_analyses if a['meaningful_difference']])
        
        print(f"  Total model comparisons across all validation runs: {total_comparisons}")
        print(f"  Meaningful differences (ΔLL ≥ 0.1): {meaningful_differences} ({meaningful_differences/total_comparisons:.1%})")
        
        if differentiation_issues:
            print(f"  ⚠️  DIFFERENTIATION ISSUES FOUND: {len(differentiation_issues)} cases")
            print(f"     Models with suspiciously similar likelihoods:")
            
            for issue in differentiation_issues[:5]:  # Show first 5
                print(f"       Run {issue['validation_run']}: {issue['model1']} vs {issue['model2']}")
                print(f"         LL1={issue['ll1']:.4f}, LL2={issue['ll2']:.4f}, Diff={issue['ll_difference']:.6f}")
                
            if len(differentiation_issues) > 5:
                print(f"       ... and {len(differentiation_issues)-5} more")
                
        else:
            print(f"  ✅ All model pairs show meaningful likelihood differences")
        
        # Statistical analysis of differences
        if sample_analyses:
            ll_differences = [a['ll_difference'] for a in sample_analyses]
            
            print(f"\n  LIKELIHOOD DIFFERENCE STATISTICS:")
            print(f"    Mean difference: {np.mean(ll_differences):.3f}")
            print(f"    Median difference: {np.median(ll_differences):.3f}")
            print(f"    Min difference: {np.min(ll_differences):.6f}")
            print(f"    Max difference: {np.max(ll_differences):.3f}")
            
            # Check for suspiciously identical results
            identical_results = len([d for d in ll_differences if d < 1e-6])
            if identical_results > 0:
                print(f"    ❌ CRITICAL: {identical_results} model pairs with identical likelihoods (diff < 1e-6)")
            
            very_similar = len([d for d in ll_differences if d < 0.01])
            if very_similar > identical_results:
                print(f"    ⚠️  WARNING: {very_similar-identical_results} model pairs with very similar likelihoods (diff < 0.01)")
        
        # Model-specific analysis across all runs
        print(f"\n  CROSS-RUN MODEL CONSISTENCY:")
        
        all_models = set()
        for result in self.validation_results:
            if 'model_statistics' in result:
                all_models.update(result['model_statistics'].keys())
        
        for model_name in sorted(all_models):
            model_lls = []
            for result in self.validation_results:
                if 'model_statistics' in result and model_name in result['model_statistics']:
                    stats = result['model_statistics'][model_name]
                    if 'mean_ll' in stats:
                        model_lls.append(stats['mean_ll'])
            
            if len(model_lls) > 1:
                ll_range = np.max(model_lls) - np.min(model_lls)
                ll_cv = np.std(model_lls) / abs(np.mean(model_lls)) if np.mean(model_lls) != 0 else np.inf
                
                print(f"    {model_name}: Range={ll_range:.3f}, CV={ll_cv:.6f}")
                
                # Check for concerning patterns
                if ll_range < 0.1:
                    print(f"      ⚠️  Small range across validation runs - may indicate issues")
                if ll_cv > 0.01:
                    print(f"      ⚠️  High coefficient of variation - inconsistent results")
        
        # Additional check: Are parameters actually different between models?
        print(f"\n  PARAMETER DIFFERENTIATION CHECK:")
        self.analyze_parameter_differentiation()

        return self.meta_analysis
    
    def analyze_parameter_differentiation(self):
        """
        Analyze whether different models are actually producing different parameter estimates.
        If all models converge to the same parameters, there's a fundamental issue.
        """
        
        # This is a simplified analysis since we don't have access to detailed parameter data
        # In the actual validation run, we'd need to modify the data collection to capture
        # individual parameter estimates for comparison
        
        print("    ⚠️  LIMITATION: Full parameter analysis requires detailed parameter data")
        print("    Current analysis focuses on likelihood differentiation as proxy")
        print("    📋 RECOMMENDATION: Enhance data collection for future parameter-level analysis")
        
        # What we can check: Look for patterns that suggest parameter issues
        # 1. If different models have identical likelihoods, they likely have identical parameters
        # 2. If within-model parameter CVs are all zero, optimization might be stuck
        
        identical_ll_pairs = 0
        total_pairs = 0
        
        for validation_result in self.validation_results:
            if 'model_statistics' not in validation_result:
                continue
                
            models = validation_result['model_statistics']
            model_names = list(models.keys())
            
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    
                    if 'mean_ll' in models[model1] and 'mean_ll' in models[model2]:
                        ll1 = models[model1]['mean_ll']
                        ll2 = models[model2]['mean_ll']
                        
                        total_pairs += 1
                        
                        if abs(ll1 - ll2) < 1e-8:  # Essentially identical
                            identical_ll_pairs += 1
        
        if total_pairs > 0:
            identical_rate = identical_ll_pairs / total_pairs
            print(f"    Model pairs with identical likelihoods: {identical_ll_pairs}/{total_pairs} ({identical_rate:.1%})")
            
            if identical_rate > 0.1:
                print(f"    ❌ CRITICAL: High rate of identical likelihoods suggests parameter optimization issues")
            elif identical_rate > 0:
                print(f"    ⚠️  WARNING: Some models have identical likelihoods - investigate further")
            else:
                print(f"    ✅ Good: No identical likelihoods between different models")

        return self.meta_analysis

def main():
    """Run the meta-validation."""
    runner = MetaValidationRunner(n_validation_runs=10)  # Reduced from 20 to 10 as requested
    results = runner.run_all_validations()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"meta_validation_results_{timestamp}.json"
    
    # Convert numpy types to native Python for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean results for JSON
    clean_results = []
    for result in results:
        clean_result = {}
        for key, value in result.items():
            clean_result[key] = convert_numpy(value)
        clean_results.append(clean_result)
    
    with open(filename, 'w') as f:
        json.dump({
            'meta_analysis': runner.meta_analysis,
            'validation_results': clean_results
        }, f, indent=2, default=convert_numpy)
    
    print(f"\nResults saved to: {filename}")
    
    return results

if __name__ == "__main__":
    main()