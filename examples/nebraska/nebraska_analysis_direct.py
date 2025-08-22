#!/usr/bin/env python3
"""
Nebraska Data Analysis - Direct Optimization Version
Uses scipy optimization directly to bypass framework issues.
Exports parameters and provides working model comparison.
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

def prepare_nebraska_data(data_file: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Prepare Nebraska data for the Pradel-JAX system."""
    print(f"📂 Loading Nebraska data from: {data_file}")
    
    # Load data
    if sample_size:
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
            'gender': row['gender'] if pd.notna(row['gender']) else 0,
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

def fit_model_direct(model, formula_spec, data_context, model_name: str) -> Dict[str, Any]:
    """
    Fit model using direct scipy optimization.
    """
    try:
        # Build design matrices
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Objective function
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                if not np.isfinite(ll):
                    return 1e10
                return -ll
            except Exception:
                return 1e10
        
        # Try multiple optimization methods
        methods = [
            ('L-BFGS-B', {'maxiter': 1000, 'ftol': 1e-9}),
            ('SLSQP', {'maxiter': 1000, 'ftol': 1e-9}),
            ('TNC', {'maxiter': 1000, 'ftol': 1e-9})
        ]
        
        best_result = None
        best_objective = np.inf
        method_used = None
        
        for method, options in methods:
            try:
                # Add small perturbation to avoid local minima
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
                    method_used = method
                    
            except Exception:
                continue
        
        if best_result is not None and best_result.success:
            # Calculate final statistics
            final_ll = -best_result.fun
            k = len(best_result.x)
            aic = 2 * k - 2 * final_ll
            
            # Verify the log-likelihood calculation
            try:
                validation_ll = model.log_likelihood(best_result.x, data_context, design_matrices)
                if abs(final_ll - validation_ll) > 1e-6:
                    print(f"   ⚠️  LL validation difference: {abs(final_ll - validation_ll):.8f}")
            except:
                validation_ll = final_ll
            
            return {
                'success': True,
                'parameters': best_result.x.tolist(),
                'log_likelihood': final_ll,
                'aic': aic,
                'n_evaluations': best_result.nfev,
                'n_parameters': k,
                'method_used': method_used,
                'formula_spec': {
                    'phi': formula_spec.phi,
                    'p': formula_spec.p, 
                    'f': formula_spec.f
                },
                'convergence_message': best_result.message,
                'validation_ll': validation_ll
            }
        else:
            return {
                'success': False,
                'error': 'All optimization methods failed',
                'n_evaluations': best_result.nfev if best_result else 0
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'n_evaluations': 0
        }

def export_results(results: list, base_name: str):
    """Export results in both CSV and JSON formats."""
    
    # CSV export for easy analysis
    csv_file = f"{base_name}.csv"
    print(f"\n💾 Exporting results to {csv_file}")
    
    csv_data = []
    for model_info in results:
        row = {
            'model': model_info['name'],
            'success': model_info['result']['success']
        }
        
        if model_info['result']['success']:
            result = model_info['result']
            row.update({
                'aic': result['aic'],
                'log_likelihood': result['log_likelihood'],
                'n_parameters': result['n_parameters'],
                'method_used': result.get('method_used', 'unknown'),
                'n_evaluations': result.get('n_evaluations', 0),
                'convergence': 'success'
            })
            
            # Add parameters with names
            for i, param in enumerate(result['parameters']):
                row[f'param_{i+1}'] = param
                
        else:
            row.update({
                'aic': np.nan,
                'log_likelihood': np.nan,
                'n_parameters': np.nan,
                'method_used': 'failed',
                'convergence': 'failed',
                'error': result.get('error', 'Unknown')
            })
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    # JSON export for detailed analysis
    json_file = f"{base_name}.json"
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
            model_data.update(model_info['result'])
        else:
            model_data['error'] = model_info['result']['error']
        
        export_data['models'].append(model_data)
    
    with open(json_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"   ✅ Results exported to {csv_file} and {json_file}")

def main():
    """Run direct optimization Nebraska analysis."""
    
    print("🔬 Nebraska Analysis - Direct Optimization with Parameter Export")
    print("=" * 70)
    
    # Configuration
    data_file = "data/encounter_histories_ne_clean.csv"
    
    print("📊 Dataset Size Options:")
    print("   1. Test (100 individuals) - ~3 minutes [RECOMMENDED]")
    print("   2. Small (500 individuals) - ~20 minutes")
    print("   3. Medium (1K individuals) - ~1 hour")
    print("   4. Custom size")
    print("\n🎯 Recommendation: Start with option 1 to verify the fix")
    
    try:
        choice = input(f"\nSelect option (1-4) [default: 1]: ").strip()
        if not choice:
            choice = '1'
            
        size_map = {
            '1': 100,
            '2': 500, 
            '3': 1000,
            '4': 'custom'
        }
        
        sample_size = size_map.get(choice, 100)
        
        if sample_size == 'custom':
            custom_size = input("Enter number of individuals: ").strip()
            try:
                sample_size = int(custom_size)
                if sample_size <= 0:
                    raise ValueError("Must be positive")
            except ValueError:
                print("Invalid input, using test size (100)")
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
        temp_file = f"temp_nebraska_direct_{len(nebraska_data)}.csv"
        nebraska_data.to_csv(temp_file, index=False)
        
        print("\n🚀 Loading data into Pradel-JAX system...")
        data_context = pj.load_data(temp_file)
        print(f"   ✅ Loaded: {data_context.n_individuals:,} individuals, {data_context.n_occasions} occasions")
        
        # Define model formulas
        phi_formulas = ["~1", "~1 + age_1"]
        f_formulas = ["~1", "~1 + gender"]
        
        # Generate model specifications
        def create_model_name(formula):
            if formula == "~1":
                return "1"
            return formula.replace("~1 + ", "").replace("age_1", "age").replace("tier_1", "tier")
        
        model_specifications = [
            {
                'name': f"φ({create_model_name(phi)}) f({create_model_name(f)})",
                'formula': pj.create_simple_spec(phi=phi, p="~1", f=f, name=f"phi_{i}_f_{j}")
            }
            for i, phi in enumerate(phi_formulas)
            for j, f in enumerate(f_formulas)
        ]
        
        print(f"\n📊 Model specifications:")
        print(f"   - Total models: {len(model_specifications)}")
        for spec in model_specifications:
            print(f"   - {spec['name']}")
        
        print(f"\n⚡ Fitting models using direct optimization...")
        
        # Fit models
        model = pj.PradelModel()
        results = []
        fit_start = time.time()
        
        for i, spec in enumerate(model_specifications, 1):
            elapsed = time.time() - fit_start
            
            print(f"\n🔧 Model {i}/{len(model_specifications)}: {spec['name']}")
            if i > 1:
                avg_time = elapsed / (i - 1)
                remaining = avg_time * (len(model_specifications) - i)
                print(f"   ⏱️  Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
            
            result = fit_model_direct(model, spec['formula'], data_context, spec['name'])
            results.append({'name': spec['name'], 'result': result})
            
            if result['success']:
                print(f"   ✅ Success - AIC: {result['aic']:.2f}, LL: {result['log_likelihood']:.4f}")
                print(f"      Method: {result.get('method_used', 'unknown')}, Evals: {result.get('n_evaluations', 0)}")
                print(f"      Parameters: [{', '.join([f'{p:.4f}' for p in result['parameters']])}]")
            else:
                print(f"   ❌ Failed: {result['error']}")
        
        fit_time = time.time() - fit_start
        
        # Export results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_name = f"nebraska_direct_{sample_size}ind_{timestamp}"
        export_results(results, base_name)
        
        # Analysis summary
        print(f"\n🎯 Analysis Summary")
        print("=" * 50)
        print(f"Total fitting time: {fit_time/60:.1f} minutes")
        
        successful_results = [r for r in results if r['result']['success']]
        print(f"Successful models: {len(successful_results)}/{len(results)}")
        
        if successful_results:
            print(f"\n📊 Model Comparison:")
            print("-" * 80)
            print(f"{'Model':<20} {'AIC':<10} {'ΔAIC':<8} {'LogLik':<12} {'K':<3} {'Method':<8}")
            print("-" * 80)
            
            # Sort by AIC
            model_stats = [(r['name'], r['result']) for r in successful_results]
            model_stats.sort(key=lambda x: x[1]['aic'])
            best_aic = model_stats[0][1]['aic']
            
            for name, result in model_stats:
                delta_aic = result['aic'] - best_aic
                method = result.get('method_used', 'unknown')[:6]
                print(f"{name:<20} {result['aic']:<10.2f} {delta_aic:<8.2f} "
                      f"{result['log_likelihood']:<12.4f} {result['n_parameters']:<3} {method:<8}")
            
            print("-" * 80)
            
            # Check for the original problem
            unique_lls = set(round(r[1]['log_likelihood'], 6) for r in model_stats)
            if len(unique_lls) == 1:
                print(f"\n⚠️  WARNING: All models have identical log-likelihood!")
                print("   This indicates a potential issue with model specification or data.")
            else:
                print(f"\n✅ SUCCESS: Models show different log-likelihoods")
                print(f"   Range: {min(r[1]['log_likelihood'] for r in model_stats):.4f} to {max(r[1]['log_likelihood'] for r in model_stats):.4f}")
            
            # Best model summary
            best_name, best_result = model_stats[0]
            print(f"\n🏆 Best Model: {best_name}")
            print(f"   AIC: {best_result['aic']:.4f}")
            print(f"   Log-likelihood: {best_result['log_likelihood']:.6f}")
            print(f"   Parameters: {best_result['parameters']}")
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time/60:.1f} minutes")
        print(f"📊 Results exported to {base_name}.csv and {base_name}.json")
        
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