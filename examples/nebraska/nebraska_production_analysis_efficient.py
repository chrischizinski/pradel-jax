#!/usr/bin/env python3
"""
Nebraska Data Analysis - Efficient Version with Parameter Export
Fixes design matrix access issues and improves performance.
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

def fit_single_model_efficient(model, formula_spec, data_context, model_name: str) -> Dict[str, Any]:
    """
    Fit a single model efficiently with proper error handling.
    Uses the production optimization framework.
    """
    try:
        # Use the production optimization system directly
        result = pj.fit_model(
            model=model,
            formula=formula_spec,
            data=data_context
        )
        
        if result.success:
            # Extract parameters from the result
            parameters = result.parameters.tolist() if hasattr(result.parameters, 'tolist') else result.parameters
            
            return {
                'success': True,
                'parameters': parameters,
                'log_likelihood': result.log_likelihood,
                'aic': result.aic,
                'n_evaluations': getattr(result, 'n_evaluations', 0),
                'n_parameters': len(parameters),
                'strategy_used': getattr(result, 'strategy_used', 'unknown'),
                'formula_spec': {
                    'phi': formula_spec.phi,
                    'p': formula_spec.p, 
                    'f': formula_spec.f
                },
                'convergence_message': getattr(result, 'message', 'Success')
            }
        else:
            return {
                'success': False,
                'error': 'Optimization failed',
                'message': getattr(result, 'message', 'Unknown error')
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'n_evaluations': 0
        }

def export_results_csv(results: list, output_file: str = "nebraska_model_comparison.csv"):
    """Export results to CSV for easy analysis."""
    print(f"\n💾 Exporting results to {output_file}")
    
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
                'strategy_used': result.get('strategy_used', 'unknown'),
                'n_evaluations': result.get('n_evaluations', 0),
                'convergence_message': result.get('convergence_message', '')
            })
            
            # Add individual parameters
            for i, param in enumerate(result['parameters']):
                row[f'param_{i+1}'] = param
        else:
            row.update({
                'aic': np.nan,
                'log_likelihood': np.nan,
                'n_parameters': np.nan,
                'error': result.get('error', 'Unknown error')
            })
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"   ✅ CSV results exported successfully")

def export_detailed_results(results: list, output_file: str = "nebraska_detailed_results.json"):
    """Export detailed results to JSON."""
    print(f"💾 Exporting detailed results to {output_file}")
    
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
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"   ✅ Detailed results exported successfully")

def main():
    """Run efficient Nebraska analysis."""
    
    print("🔬 Nebraska Analysis - Efficient Version with Parameter Export")
    print("=" * 65)
    
    # Configuration
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Realistic time estimates based on your feedback
    print("📊 Dataset Size Options - REALISTIC TIME ESTIMATES:")
    print("   1. Tiny (100 individuals) - ~2 minutes [DEBUG/TEST]")
    print("   2. Small (500 individuals) - ~15 minutes")
    print("   3. Medium (1K individuals) - ~45 minutes")
    print("   4. Large (2K individuals) - ~2 hours") 
    print("   5. X-Large (5K individuals) - ~6-8 hours [Your previous run]")
    print("   6. Custom size")
    print("\n⚠️  NOTE: Times scale roughly quadratically with sample size")
    print("🎯 Recommendation: Start with option 1 or 2 to test the system")
    
    try:
        choice = input(f"\nSelect option (1-6) [default: 1]: ").strip()
        if not choice:
            choice = '1'
            
        size_map = {
            '1': 100,   # Debug size
            '2': 500,   # Small test
            '3': 1000,  # Medium test
            '4': 2000,  # Large test
            '5': 5000,  # Your previous size
            '6': 'custom'
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
        
        # Estimate time
        time_estimate = (sample_size / 100) ** 1.8 * 2  # Rough scaling
        if time_estimate < 60:
            print(f"⏱️  Estimated time: ~{time_estimate:.0f} minutes")
        else:
            hours = time_estimate / 60
            print(f"⏱️  Estimated time: ~{hours:.1f} hours")
            
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
        temp_file = f"temp_nebraska_analysis_{len(nebraska_data)}.csv"
        nebraska_data.to_csv(temp_file, index=False)
        
        print("\n🚀 Loading data into Pradel-JAX system...")
        data_context = pj.load_data(temp_file)
        print(f"   ✅ Loaded: {data_context.n_individuals:,} individuals, {data_context.n_occasions} occasions")
        
        # Define model formulas - start with simple models
        if sample_size <= 100:
            # Simple models for testing
            phi_formulas = ["~1", "~1 + age_1"]
            f_formulas = ["~1", "~1 + gender"]
        elif sample_size <= 1000:
            # Medium complexity
            phi_formulas = ["~1", "~1 + age_1", "~1 + gender", "~1 + gender + age_1"]
            f_formulas = ["~1", "~1 + gender", "~1 + age_1"]
        else:
            # Full model set
            phi_formulas = [
                "~1", "~1 + gender", "~1 + age_1", "~1 + tier_1",
                "~1 + gender + age_1", "~1 + gender + tier_1", 
                "~1 + age_1 + tier_1", "~1 + gender + age_1 + tier_1"
            ]
            f_formulas = [
                "~1", "~1 + gender", "~1 + age_1", "~1 + tier_1",
                "~1 + gender + age_1", "~1 + gender + tier_1", 
                "~1 + age_1 + tier_1", "~1 + gender + age_1 + tier_1"
            ]
        
        # Generate model specifications
        def create_model_name(formula):
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
        
        print(f"\n📊 Model specifications:")
        print(f"   - φ (survival): {len(phi_formulas)} formulations")
        print(f"   - f (recruitment): {len(f_formulas)} formulations")  
        print(f"   - p (detection): constant (~1)")
        print(f"   - Total models: {len(model_specifications)}")
        
        print(f"\n⚡ Fitting {len(model_specifications)} models using production system...")
        
        # Fit models efficiently
        model = pj.PradelModel()
        results = []
        fit_start = time.time()
        
        for i, spec in enumerate(model_specifications, 1):
            elapsed = time.time() - fit_start
            remaining = ((elapsed / i) * (len(model_specifications) - i)) / 60 if i > 0 else 0
            
            print(f"\n🔧 Model {i}/{len(model_specifications)}: {spec['name']}")
            if i > 1:
                print(f"   ⏱️  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining:.1f}min")
            
            result = fit_single_model_efficient(model, spec['formula'], data_context, spec['name'])
            results.append({'name': spec['name'], 'result': result})
            
            if result['success']:
                print(f"   ✅ Success - AIC: {result['aic']:.1f}, LL: {result['log_likelihood']:.1f}")
                print(f"      Strategy: {result.get('strategy_used', 'unknown')}")
            else:
                print(f"   ❌ Failed: {result['error']}")
        
        fit_time = time.time() - fit_start
        
        # Export results in multiple formats
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_name = f"nebraska_analysis_{sample_size}ind_{timestamp}"
        
        export_results_csv(results, f"{base_name}.csv")
        export_detailed_results(results, f"{base_name}.json")
        
        # Results summary
        print(f"\n🎯 Analysis Summary")
        print("=" * 50)
        print(f"Total fitting time: {fit_time/60:.1f} minutes ({fit_time:.1f} seconds)")
        
        successful_results = [r for r in results if r['result']['success']]
        print(f"Successful models: {len(successful_results)}/{len(results)}")
        
        if successful_results:
            # Model comparison table
            print(f"\n📊 Top Models (by AIC):")
            print("-" * 80)
            print(f"{'Model':<25} {'AIC':<10} {'ΔAIC':<8} {'LogLik':<12} {'K':<3} {'Strategy':<10}")
            print("-" * 80)
            
            # Sort by AIC
            model_stats = [(r['name'], r['result']) for r in successful_results]
            model_stats.sort(key=lambda x: x[1]['aic'])
            best_aic = model_stats[0][1]['aic']
            
            # Show top 10 models
            for name, result in model_stats[:10]:
                delta_aic = result['aic'] - best_aic
                strategy = result.get('strategy_used', 'unknown')[:8]
                print(f"{name:<25} {result['aic']:<10.1f} {delta_aic:<8.1f} "
                      f"{result['log_likelihood']:<12.1f} {result['n_parameters']:<3} {strategy:<10}")
            
            if len(model_stats) > 10:
                print(f"   ... and {len(model_stats) - 10} more models")
            
            print("-" * 80)
            
            # Best model details
            best_name, best_result = model_stats[0]
            print(f"\n🏆 Best Model: {best_name}")
            print(f"   AIC: {best_result['aic']:.3f}")
            print(f"   Log-likelihood: {best_result['log_likelihood']:.3f}")
            print(f"   Parameters: {best_result['n_parameters']}")
            print(f"   Strategy: {best_result.get('strategy_used', 'unknown')}")
            
            # Show parameter estimates for best model
            params = best_result['parameters']
            print(f"   Parameter estimates: [{', '.join([f'{p:.4f}' for p in params[:5]])}{'...' if len(params) > 5 else ''}]")
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time/60:.1f} minutes")
        print(f"📊 Results exported to {base_name}.csv and {base_name}.json")
        
        # Performance feedback
        actual_rate = total_time / sample_size
        print(f"\n📈 Performance: {actual_rate:.2f} seconds per individual")
        if sample_size >= 1000:
            print(f"🔮 Estimated time for 5K individuals: {(actual_rate * 5000) / 3600:.1f} hours")
        
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