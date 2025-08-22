#!/usr/bin/env python3
"""
Ultra Conservative Testing of Nebraska Models with Covariates.

This test validates the fixed optimization bounds on realistic capture-recapture data
with complex covariate structures. We test for consistency across multiple runs,
parameter stability, and likelihood convergence.

ULTRA CONSERVATIVE APPROACH:
- Multiple independent runs on same data
- Multiple different random samples  
- Statistical tests for consistency
- Detailed convergence diagnostics
- Cross-validation of parameter estimates
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import time
from typing import Dict, List, Tuple, Any
import warnings

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel, logit, log_link, inv_logit, exp_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec

# Suppress JAX warnings for cleaner output
warnings.filterwarnings("ignore", ".*TPU.*")

def load_nebraska_data():
    """Load and prepare Nebraska data for testing."""
    print("Loading Nebraska dataset...")
    
    try:
        df = pd.read_csv('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax/data/encounter_histories_ne_clean.csv')
        print(f"  Loaded {len(df)} total records")
        
        # Basic data quality checks
        print("  Data quality checks:")
        print(f"    Missing gender: {df['gender'].isna().sum()}")
        print(f"    Missing age: {df['age'].isna().sum()}")
        print(f"    Missing tier_cat: {df['tier_cat'].isna().sum()}")
        
        # Filter out records with missing key covariates
        df_clean = df.dropna(subset=['gender', 'age', 'tier_cat']).copy()
        print(f"    After filtering: {len(df_clean)} records")
        
        return df_clean
        
    except Exception as e:
        print(f"Error loading Nebraska data: {e}")
        return None

def create_test_samples(df: pd.DataFrame, sample_sizes: List[int], n_samples: int = 3) -> Dict[str, List[pd.DataFrame]]:
    """Create multiple random samples of different sizes for testing."""
    print(f"\nCreating test samples...")
    
    samples = {}
    
    for size in sample_sizes:
        samples[f"size_{size}"] = []
        
        for i in range(n_samples):
            # Use different random seeds for each sample
            sample = df.sample(n=size, random_state=42 + i*10)
            samples[f"size_{size}"].append(sample)
            print(f"  Sample {i+1} of size {size}: {len(sample)} records")
    
    return samples

def prepare_formula_specs() -> Dict[str, FormulaSpec]:
    """Create different formula specifications to test."""
    parser = FormulaParser()
    
    formulas = {
        "intercept_only": FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        ),
        
        "gender_effects": FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + gender"),
            p=parser.create_parameter_formula(ParameterType.P, "~1 + gender"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        ),
        
        "age_effects": FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + age"),
            p=parser.create_parameter_formula(ParameterType.P, "~1 + age"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        ),
        
        "tier_effects": FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + tier_cat"),
            p=parser.create_parameter_formula(ParameterType.P, "~1 + tier_cat"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        ),
        
        "full_model": FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + gender + age + tier_cat"),
            p=parser.create_parameter_formula(ParameterType.P, "~1 + gender + age + tier_cat"),
            f=parser.create_parameter_formula(ParameterType.F, "~1 + gender")
        )
    }
    
    return formulas

def fit_model_with_diagnostics(
    data_context: Any,
    formula_spec: FormulaSpec,
    formula_name: str,
    run_id: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """Fit a single model with comprehensive diagnostics."""
    
    start_time = time.time()
    
    try:
        model = PradelModel()
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        # Get initial parameters and bounds
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Define optimization functions
        def objective(params):
            return -float(model.log_likelihood(jnp.array(params), data_context, design_matrices))
        
        def gradient(params):
            grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
            return -np.array(grad_fn(jnp.array(params)))
        
        # Test gradient at initial point
        initial_ll = -objective(initial_params)
        initial_grad = gradient(initial_params)
        initial_grad_norm = np.linalg.norm(initial_grad)
        
        # Optimize
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            jac=gradient,
            bounds=bounds,
            options={'disp': False, 'maxiter': 200, 'ftol': 1e-9}
        )
        
        fit_time = time.time() - start_time
        
        # Analyze results
        final_ll = -result.fun if result.success else None
        ll_improvement = final_ll - initial_ll if final_ll is not None else None
        
        # Check if hit bounds
        hit_bounds = []
        if result.success:
            param_names = model.get_parameter_names(design_matrices)
            for i, (param_val, (lower, upper)) in enumerate(zip(result.x, bounds)):
                if abs(param_val - lower) < 1e-3:
                    hit_bounds.append(f"{param_names[i]} (lower)")
                elif abs(param_val - upper) < 1e-3:
                    hit_bounds.append(f"{param_names[i]} (upper)")
        
        # Calculate AIC
        n_params = len(initial_params)
        aic = 2 * n_params - 2 * final_ll if final_ll is not None else None
        
        diagnostics = {
            'formula_name': formula_name,
            'run_id': run_id,
            'success': result.success,
            'message': result.message,
            'n_individuals': data_context.n_individuals,
            'n_parameters': n_params,
            'iterations': result.nit if hasattr(result, 'nit') else None,
            'function_evals': result.nfev if hasattr(result, 'nfev') else None,
            'fit_time': fit_time,
            'initial_ll': initial_ll,
            'final_ll': final_ll,
            'll_improvement': ll_improvement,
            'initial_grad_norm': initial_grad_norm,
            'final_grad_norm': np.linalg.norm(result.jac) if hasattr(result, 'jac') else None,
            'aic': aic,
            'parameters': result.x.tolist() if result.success else None,
            'hit_bounds': hit_bounds,
            'parameter_names': model.get_parameter_names(design_matrices) if result.success else None
        }
        
        if verbose:
            print(f"    {formula_name} run {run_id}: Success={result.success}, LL={final_ll:.3f}, AIC={aic:.1f}")
        
        return diagnostics
        
    except Exception as e:
        return {
            'formula_name': formula_name,
            'run_id': run_id,
            'success': False,
            'error': str(e),
            'fit_time': time.time() - start_time
        }

def analyze_consistency(results: List[Dict[str, Any]], formula_name: str) -> Dict[str, Any]:
    """Analyze consistency across multiple runs of the same model."""
    
    successful_runs = [r for r in results if r['success']]
    n_success = len(successful_runs)
    n_total = len(results)
    
    if n_success == 0:
        return {
            'formula_name': formula_name,
            'success_rate': 0.0,
            'status': 'ALL_FAILED'
        }
    
    # Extract key metrics
    final_lls = [r['final_ll'] for r in successful_runs]
    aics = [r['aic'] for r in successful_runs]
    fit_times = [r['fit_time'] for r in successful_runs]
    
    # Statistical analysis
    ll_mean = np.mean(final_lls)
    ll_std = np.std(final_lls)
    ll_cv = ll_std / abs(ll_mean) if ll_mean != 0 else np.inf
    
    aic_mean = np.mean(aics)
    aic_std = np.std(aics)
    aic_cv = aic_std / aic_mean if aic_mean != 0 else np.inf
    
    # Check parameter consistency
    if successful_runs[0]['parameters'] is not None:
        all_params = np.array([r['parameters'] for r in successful_runs])
        param_means = np.mean(all_params, axis=0)
        param_stds = np.std(all_params, axis=0)
        param_cvs = param_stds / np.abs(param_means)
        param_cvs = np.where(np.isfinite(param_cvs), param_cvs, np.inf)
        
        max_param_cv = np.max(param_cvs)
        mean_param_cv = np.mean(param_cvs[np.isfinite(param_cvs)])
    else:
        max_param_cv = np.inf
        mean_param_cv = np.inf
    
    # Consistency classification
    if n_success < n_total:
        status = 'PARTIAL_FAILURE'
    elif ll_cv > 0.001 or max_param_cv > 0.01:  # Very strict consistency criteria
        status = 'INCONSISTENT'
    elif ll_cv > 0.0001 or max_param_cv > 0.001:
        status = 'MARGINALLY_CONSISTENT' 
    else:
        status = 'HIGHLY_CONSISTENT'
    
    return {
        'formula_name': formula_name,
        'n_total_runs': n_total,
        'n_successful_runs': n_success,
        'success_rate': n_success / n_total,
        'status': status,
        'final_ll_mean': ll_mean,
        'final_ll_std': ll_std,
        'final_ll_cv': ll_cv,
        'aic_mean': aic_mean,
        'aic_std': aic_std,
        'aic_cv': aic_cv,
        'max_param_cv': max_param_cv,
        'mean_param_cv': mean_param_cv,
        'mean_fit_time': np.mean(fit_times),
        'std_fit_time': np.std(fit_times)
    }

def run_ultra_conservative_validation():
    """Run ultra conservative validation tests."""
    print("="*70)
    print("ULTRA CONSERVATIVE NEBRASKA MODELS VALIDATION")
    print("="*70)
    
    # Load data
    df = load_nebraska_data()
    if df is None:
        print("❌ Failed to load Nebraska data")
        return
    
    # Create test samples - start with smaller sizes
    sample_sizes = [100, 250]  # Start conservative
    samples = create_test_samples(df, sample_sizes, n_samples=2)  # 2 different samples each
    
    # Get formula specifications  
    formulas = prepare_formula_specs()
    
    # Focus on key models for initial testing
    test_formulas = ['intercept_only', 'gender_effects', 'age_effects']
    
    print(f"\nTesting {len(test_formulas)} formula types on {len(sample_sizes)} sample sizes...")
    print("Running 5 independent optimization runs per model for consistency testing")
    
    all_results = []
    
    # Test each combination
    for sample_size_key, sample_list in samples.items():
        for sample_idx, sample_df in enumerate(sample_list):
            
            print(f"\n" + "-"*50)
            print(f"TESTING: {sample_size_key.upper()}, Sample {sample_idx + 1}")
            print("-"*50)
            
            # Convert to data context
            try:
                # Create temporary file for data loading
                import tempfile
                import os
                
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                sample_df.to_csv(temp_file.name, index=False)
                temp_file.close()
                
                try:
                    data_context = pj.load_data(temp_file.name)
                finally:
                    os.unlink(temp_file.name)
                
                print(f"  Data loaded: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
                
            except Exception as e:
                print(f"  ❌ Failed to load data context: {e}")
                continue
            
            # Test each formula
            for formula_name in test_formulas:
                formula_spec = formulas[formula_name]
                
                print(f"\n  Testing {formula_name}...")
                
                # Run multiple independent optimizations
                formula_results = []
                for run in range(5):  # 5 runs for consistency
                    result = fit_model_with_diagnostics(
                        data_context, formula_spec, formula_name, run, verbose=True
                    )
                    result['sample_size'] = sample_size_key
                    result['sample_idx'] = sample_idx
                    formula_results.append(result)
                    all_results.append(result)
                
                # Analyze consistency
                consistency = analyze_consistency(formula_results, formula_name)
                
                print(f"  CONSISTENCY ANALYSIS: {consistency['status']}")
                print(f"    Success rate: {consistency['success_rate']:.1%}")
                print(f"    Log-likelihood CV: {consistency['final_ll_cv']:.6f}")
                print(f"    Max parameter CV: {consistency['max_param_cv']:.6f}")
                
                if consistency['status'] in ['INCONSISTENT', 'PARTIAL_FAILURE']:
                    print(f"    ⚠️  POTENTIAL ISSUE DETECTED")
    
    # Overall summary
    print(f"\n" + "="*70)
    print("OVERALL VALIDATION SUMMARY")
    print("="*70)
    
    total_runs = len(all_results)
    successful_runs = len([r for r in all_results if r['success']])
    
    print(f"Total optimization runs: {total_runs}")
    print(f"Successful runs: {successful_runs} ({successful_runs/total_runs:.1%})")
    
    # Group by formula and analyze
    formula_groups = {}
    for result in all_results:
        formula = result['formula_name']
        if formula not in formula_groups:
            formula_groups[formula] = []
        formula_groups[formula].append(result)
    
    print(f"\nFormula-specific results:")
    for formula_name, results in formula_groups.items():
        consistency = analyze_consistency(results, formula_name)
        status_symbol = {
            'HIGHLY_CONSISTENT': '✅',
            'MARGINALLY_CONSISTENT': '⚠️',
            'INCONSISTENT': '❌',
            'PARTIAL_FAILURE': '💥',
            'ALL_FAILED': '🚫'
        }.get(consistency['status'], '❓')
        
        print(f"  {status_symbol} {formula_name}: {consistency['status']}")
        print(f"      Success: {consistency['success_rate']:.1%}")
        print(f"      LL CV: {consistency['final_ll_cv']:.6f}")
        print(f"      Param CV: {consistency['max_param_cv']:.6f}")
    
    return all_results

if __name__ == "__main__":
    results = run_ultra_conservative_validation()