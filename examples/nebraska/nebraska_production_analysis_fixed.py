#!/usr/bin/env python3
"""
Nebraska Data Analysis Using Production Pradel-JAX System (FIXED VERSION)
Fixes issues with identical log-likelihoods and adds parameter export.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from scipy.optimize import minimize
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

def prepare_nebraska_data(data_file: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Prepare Nebraska data for the Pradel-JAX system."""
    print(f"📂 Loading Nebraska data from: {data_file}")
    
    # Load data
    if sample_size:
        # Efficient sampling for large files
        total_rows = sum(1 for _ in open(data_file)) - 1
        skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                          size=total_rows - sample_size, 
                                          replace=False))
        data = pd.read_csv(data_file, skiprows=skip_rows)
        print(f"   Loaded {len(data):,} individuals (sampled from {total_rows:,})")
    else:
        data = pd.read_csv(data_file)
        print(f"   Loaded {len(data):,} individuals")
    
    print("🔧 Converting to Pradel-JAX format...")
    
    # Generate column names dynamically
    years = list(range(2016, 2025))
    year_columns = [f'Y{year}' for year in years]
    age_columns = [f'age_{year}' for year in years]
    tier_columns = [f'tier_{year}' for year in years]
    
    # Create encounter history matrix (binary)
    encounter_data = (data[year_columns].values > 0).astype(int)
    
    # Prepare data in format expected by Pradel-JAX
    formatted_data = []
    
    for i, (idx, row) in enumerate(data.iterrows()):
        individual_record = {
            'individual': row['person_id'],
            'gender': row['gender'] if pd.notna(row['gender']) else 0,  # 0=unknown, 1=male, 2=female
        }
        
        # Add encounter histories
        for j in range(len(year_columns)):
            individual_record[f'occasion_{j+1}'] = encounter_data[i, j]
        
        # Add time-varying covariates  
        for j, (age_col, tier_col) in enumerate(zip(age_columns, tier_columns)):
            individual_record[f'age_{j+1}'] = row[age_col] if pd.notna(row[age_col]) else 0
            individual_record[f'tier_{j+1}'] = row[tier_col] if pd.notna(row[tier_col]) else 0
            
        formatted_data.append(individual_record)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(formatted_data)
    
    # Data quality summary
    occasions = [col for col in result_df.columns if col.startswith('occasion_')]
    total_detections = result_df[occasions].sum().sum()
    total_possible = len(result_df) * len(occasions)
    detection_rate = total_detections / total_possible
    
    print(f"   - Occasions: {len(occasions)}")
    print(f"   - Detection rate: {detection_rate:.3f}")
    print(f"   - Never detected: {(result_df[occasions].sum(axis=1) == 0).sum()}")
    
    return result_df

def fit_single_model(model, formula_spec, data_context, model_name: str) -> Dict[str, Any]:
    """
    Fit a single model with proper error handling and validation.
    Returns detailed results including parameters.
    """
    try:
        # Build design matrices
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Print debug info for first few models
        if model_name.endswith("f(1)") and ("φ(1)" in model_name or "φ(age)" in model_name):
            print(f"   🔍 Debug {model_name}:")
            print(f"      - Initial params shape: {len(initial_params)}")
            print(f"      - Design matrix shapes: {[dm.shape for dm in design_matrices.values()]}")
        
        # Objective function with better error handling
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                if not np.isfinite(ll):
                    return 1e10
                return -ll
            except Exception as e:
                # Return penalty for numerical issues
                return 1e10
        
        # Try multiple optimization strategies
        optimizers = [
            ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-9}),
            ('SLSQP', {'maxiter': 1000, 'ftol': 1e-9})
        ]
        
        best_result = None
        best_objective = np.inf
        
        for method, options in optimizers:
            try:
                # Add small random perturbation to initial parameters to avoid local minima
                perturbed_init = initial_params + np.random.normal(0, 0.01, len(initial_params))
                
                opt_result = minimize(
                    objective, 
                    perturbed_init, 
                    method=method, 
                    bounds=bounds, 
                    options=options
                )
                
                if opt_result.success and opt_result.fun < best_objective:
                    best_result = opt_result
                    best_objective = opt_result.fun
                    
            except Exception as e:
                continue
        
        if best_result is not None and best_result.success:
            # Calculate final log-likelihood and AIC
            final_ll = -best_result.fun
            k = len(best_result.x)
            aic = 2 * k - 2 * final_ll
            
            # Validate log-likelihood calculation
            try:
                validation_ll = model.log_likelihood(best_result.x, data_context, design_matrices)
                if abs(final_ll - validation_ll) > 1e-6:
                    print(f"   ⚠️  Warning: LL validation failed for {model_name}")
                    print(f"      Optimization LL: {final_ll:.6f}")
                    print(f"      Validation LL: {validation_ll:.6f}")
            except:
                validation_ll = final_ll
            
            return {
                'success': True,
                'parameters': best_result.x.tolist(),
                'log_likelihood': final_ll,
                'aic': aic,
                'n_evaluations': best_result.nfev,
                'n_parameters': k,
                'convergence_message': best_result.message,
                'formula_spec': {
                    'phi': formula_spec.phi,
                    'p': formula_spec.p, 
                    'f': formula_spec.f
                },
                'parameter_names': _get_parameter_names(formula_spec, design_matrices),
                'validation_ll': validation_ll if 'validation_ll' in locals() else final_ll
            }
        else:
            return {
                'success': False,
                'error': 'Optimization failed',
                'n_evaluations': best_result.nfev if best_result else 0
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'n_evaluations': 0
        }

def _get_parameter_names(formula_spec, design_matrices) -> list:
    """Generate parameter names based on design matrices."""
    param_names = []
    
    # Add phi parameters
    phi_dm = design_matrices.get('phi')
    if phi_dm is not None:
        n_phi = phi_dm.shape[1] if len(phi_dm.shape) > 1 else 1
        for i in range(n_phi):
            if i == 0:
                param_names.append('phi_intercept')
            else:
                param_names.append(f'phi_coef_{i}')
    
    # Add p parameters  
    p_dm = design_matrices.get('p')
    if p_dm is not None:
        n_p = p_dm.shape[1] if len(p_dm.shape) > 1 else 1
        for i in range(n_p):
            if i == 0:
                param_names.append('p_intercept')
            else:
                param_names.append(f'p_coef_{i}')
    
    # Add f parameters
    f_dm = design_matrices.get('f') 
    if f_dm is not None:
        n_f = f_dm.shape[1] if len(f_dm.shape) > 1 else 1
        for i in range(n_f):
            if i == 0:
                param_names.append('f_intercept')
            else:
                param_names.append(f'f_coef_{i}')
    
    return param_names

def export_results(results: list, output_file: str = "nebraska_model_results.json"):
    """Export detailed results to JSON file."""
    print(f"\n💾 Exporting results to {output_file}")
    
    export_data = {
        'analysis_info': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_models': len(results),
            'successful_models': sum(1 for r in results if r['result']['success'])
        },
        'models': []
    }
    
    for model_info in results:
        model_data = {
            'name': model_info['name'],
            'success': model_info['result']['success']
        }
        
        if model_info['result']['success']:
            result = model_info['result']
            model_data.update({
                'aic': result['aic'],
                'log_likelihood': result['log_likelihood'],
                'n_parameters': result['n_parameters'],
                'parameters': result['parameters'],
                'parameter_names': result['parameter_names'],
                'formula_spec': result['formula_spec'],
                'n_evaluations': result['n_evaluations'],
                'convergence_message': result.get('convergence_message', '')
            })
        else:
            model_data['error'] = model_info['result']['error']
        
        export_data['models'].append(model_data)
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"   ✅ Results exported successfully")

def main():
    """Run production Nebraska analysis with fixes."""
    
    print("🔬 Nebraska Analysis Using Production Pradel-JAX System (FIXED)")
    print("=" * 70)
    
    # Configuration
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Dataset size selection - use smaller sample for debugging
    print("📊 Dataset Size Options (64 model combinations):")
    print("   1. Tiny (100 individuals) - ~1 minute [DEBUG]")
    print("   2. Small (1K individuals) - ~5 minutes")
    print("   3. Medium (5K individuals) - ~15 minutes")
    print("   4. Large (10K individuals) - ~30 minutes")
    print("   5. Custom size")
    
    try:
        choice = input(f"\nSelect option (1-5) [default: 1 for debugging]: ").strip()
        if not choice:
            choice = '1'
            
        size_map = {
            '1': 100,   # Debug size
            '2': 1000, 
            '3': 5000,
            '4': 10000, 
            '5': 'custom'
        }
        
        sample_size = size_map.get(choice, 100)
        
        if sample_size == 'custom':
            custom_size = input("Enter number of individuals: ").strip()
            try:
                sample_size = int(custom_size)
                if sample_size <= 0:
                    raise ValueError("Must be positive")
            except ValueError:
                print("Invalid input, using debug size (100)")
                sample_size = 100
        
        print(f"✅ Selected: {sample_size:,} individuals")
            
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Analysis cancelled")
        return
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        # Prepare data
        np.random.seed(42)
        nebraska_data = prepare_nebraska_data(data_file, sample_size)
        
        # Save and load through Pradel-JAX system
        temp_file = f"temp_production_nebraska_{len(nebraska_data)}.csv"
        nebraska_data.to_csv(temp_file, index=False)
        
        print("\n🚀 Loading data into Pradel-JAX system...")
        data_context = pj.load_data(temp_file)
        print(f"   ✅ Loaded: {data_context.n_individuals:,} individuals, {data_context.n_occasions} occasions")
        
        # Define model formulas (reduced set for debugging)
        phi_formulas = [
            "~1",                           # intercept only
            "~1 + age_1",                   # age effect  
            "~1 + gender + age_1",          # gender + age
        ]
        
        f_formulas = [
            "~1",                           # intercept only
            "~1 + gender",                  # gender effect
        ]
        
        # Generate model specifications
        def create_model_name(formula):
            """Create readable name from formula string."""
            if formula == "~1":
                return "1"
            vars_str = formula.replace("~1 + ", "").replace("age_1", "age").replace("tier_1", "tier")
            return vars_str
        
        model_specifications = [
            {
                'name': f"φ({create_model_name(phi)}) f({create_model_name(f)})",
                'formula': pj.create_simple_spec(phi=phi, p="~1", f=f, name=f"phi_{i}_f_{j}")
            }
            for i, phi in enumerate(phi_formulas)
            for j, f in enumerate(f_formulas)
        ]
        
        print(f"\n📊 Generated {len(model_specifications)} model combinations:")
        print(f"   - φ (survival): {len(phi_formulas)} formulations")
        print(f"   - f (recruitment): {len(f_formulas)} formulations")  
        print(f"   - p (detection): constant across all models")
        
        print(f"\n⚡ Fitting {len(model_specifications)} models...")
        
        # Fit models with improved error handling
        model = pj.PradelModel()
        results = []
        fit_start = time.time()
        
        for i, spec in enumerate(model_specifications, 1):
            print(f"\n🔧 Model {i}/{len(model_specifications)}: {spec['name']}")
            
            result = fit_single_model(model, spec['formula'], data_context, spec['name'])
            results.append({'name': spec['name'], 'result': result})
            
            if result['success']:
                print(f"   ✅ Success - AIC: {result['aic']:.1f}, LL: {result['log_likelihood']:.1f} ({result['n_evaluations']} evals)")
            else:
                print(f"   ❌ Failed: {result['error']}")
        
        fit_time = time.time() - fit_start
        
        # Export results
        export_results(results, f"nebraska_results_{sample_size}_individuals.json")
        
        # Results analysis
        print(f"\n🎯 Analysis Results")
        print("=" * 50)
        print(f"Total fitting time: {fit_time:.1f} seconds")
        
        successful_results = [r for r in results if r['result']['success']]
        print(f"Successful models: {len(successful_results)}/{len(results)}")
        
        if successful_results:
            # Model comparison table
            print(f"\n📊 Model Selection Results:")
            print("-" * 80)
            print(f"{'Model':<20} {'AIC':<12} {'ΔAIC':<8} {'LogLik':<12} {'K':<5} {'Status':<10}")
            print("-" * 80)
            
            # Sort by AIC
            model_stats = [(r['name'], r['result']) for r in successful_results]
            model_stats.sort(key=lambda x: x[1]['aic'])
            best_aic = model_stats[0][1]['aic']
            
            for name, result in model_stats:
                delta_aic = result['aic'] - best_aic
                status = "✅" if delta_aic < 2 else "⚠️" if delta_aic < 10 else "❌"
                print(f"{name:<20} {result['aic']:<12.1f} {delta_aic:<8.1f} {result['log_likelihood']:<12.1f} {result['n_parameters']:<5} {status:<10}")
            
            print("-" * 80)
            
            # Best model details
            best_name, best_result = model_stats[0]
            print(f"\n🏆 Best Model: {best_name}")
            print(f"   AIC: {best_result['aic']:.3f}")
            print(f"   Log-likelihood: {best_result['log_likelihood']:.3f}")
            print(f"   Parameters: {best_result['n_parameters']}")
            print(f"   Parameter estimates: {[f'{p:.4f}' for p in best_result['parameters']]}")
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time:.1f} seconds")
        print(f"📊 Results exported to nebraska_results_{sample_size}_individuals.json")
        
        # Cleanup
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()