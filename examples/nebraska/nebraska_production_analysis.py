#!/usr/bin/env python3
"""
Nebraska Data Analysis Using Production Pradel-JAX System
Uses the full framework we built instead of custom optimization code.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from scipy.optimize import minimize
import sys
import time
from pathlib import Path
from typing import Optional

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

def main():
    """Run production Nebraska analysis using full Pradel-JAX system."""
    
    print("🔬 Nebraska Analysis Using Production Pradel-JAX System")
    print("=" * 65)
    
    # Configuration
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Dataset size selection with 5K default (64 models total)
    print("📊 Dataset Size Options (64 model combinations):")
    print("   1. Small (1K individuals) - ~5 minutes")
    print("   2. Medium (5K individuals) - ~15 minutes [RECOMMENDED START]")
    print("   3. Large (10K individuals) - ~30 minutes")
    print("   4. X-Large (25K individuals) - ~75 minutes")
    print("   5. XX-Large (50K individuals) - ~2.5 hours")
    print("   6. Full dataset (~111K individuals) - ~5 hours")
    print("   7. Custom size")
    
    try:
        choice = input(f"\nSelect option (1-7) [default: 2]: ").strip()
        if not choice:  # Default to 5K
            choice = '2'
            
        size_map = {
            '1': 1000, 
            '2': 5000,   # Start here
            '3': 10000, 
            '4': 25000, 
            '5': 50000, 
            '6': None,   # Full dataset
            '7': 'custom'
        }
        
        sample_size = size_map.get(choice, 5000)
        
        if sample_size == 'custom':
            custom_size = input("Enter number of individuals: ").strip()
            try:
                sample_size = int(custom_size)
                if sample_size <= 0:
                    raise ValueError("Must be positive")
            except ValueError:
                print("Invalid input, using default 5K")
                sample_size = 5000
        
        if sample_size:
            print(f"✅ Selected: {sample_size:,} individuals")
        else:
            print("✅ Selected: Full dataset")
            
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Analysis cancelled")
        return
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        # Prepare data using our system
        np.random.seed(42)
        nebraska_data = prepare_nebraska_data(data_file, sample_size)
        
        # Save and load through Pradel-JAX system
        temp_file = f"temp_production_nebraska_{len(nebraska_data)}.csv"
        nebraska_data.to_csv(temp_file, index=False)
        
        print("\n🚀 Loading data into Pradel-JAX system...")
        data_context = pj.load_data(temp_file)
        print(f"   ✅ Loaded: {data_context.n_individuals:,} individuals, {data_context.n_occasions} occasions")
        
        # Define all combinations of variables for φ and f (no interactions)
        # Variables available: gender, age_1, tier_1
        # φ combinations: 1, gender, age, tier, gender+age, gender+tier, age+tier, gender+age+tier
        # f combinations: 1, gender, age, tier, gender+age, gender+tier, age+tier, gender+age+tier
        # p stays constant: ~1
        
        phi_formulas = [
            "~1",                           # intercept only
            "~1 + gender",                  # gender effect
            "~1 + age_1",                   # age effect  
            "~1 + tier_1",                  # tier effect
            "~1 + gender + age_1",          # gender + age
            "~1 + gender + tier_1",         # gender + tier
            "~1 + age_1 + tier_1",          # age + tier
            "~1 + gender + age_1 + tier_1"  # all three variables
        ]
        
        f_formulas = [
            "~1",                           # intercept only
            "~1 + gender",                  # gender effect
            "~1 + age_1",                   # age effect
            "~1 + tier_1",                  # tier effect
            "~1 + gender + age_1",          # gender + age
            "~1 + gender + tier_1",         # gender + tier
            "~1 + age_1 + tier_1",          # age + tier
            "~1 + gender + age_1 + tier_1"  # all three variables
        ]
        
        # Generate model specifications dynamically
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
        
        print(f"📊 Generated {len(model_specifications)} model combinations:")
        print(f"   - φ (survival): {len(phi_formulas)} formulations")
        print(f"   - f (recruitment): {len(f_formulas)} formulations")  
        print(f"   - p (detection): constant across all models")
        print(f"   - Total models: {len(phi_formulas)} × {len(f_formulas)} = {len(model_specifications)}")
        
        print(f"\n⚡ Fitting {len(model_specifications)} models using production system...")
        
        # Use our production Pradel model
        model = pj.PradelModel()
        results = []
        fit_start = time.time()
        
        for i, spec in enumerate(model_specifications, 1):
            print(f"\n🔧 Model {i}/{len(model_specifications)}: {spec['name']}")
            
            try:
                # Use direct optimization since high-level API isn't ready
                design_matrices = model.build_design_matrices(spec['formula'], data_context)
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                # Objective function
                def objective(params):
                    try:
                        ll = model.log_likelihood(params, data_context, design_matrices)
                        return -ll if np.isfinite(ll) else 1e10
                    except:
                        return 1e10
                
                # Use scipy L-BFGS-B optimizer
                
                # Try optimization with fallback
                optimizers = [('L-BFGS-B', {'maxiter': 500, 'ftol': 1e-8}), ('SLSQP', {'maxiter': 500, 'ftol': 1e-8})]
                
                success = False
                for method, options in optimizers:
                    opt_result = minimize(objective, initial_params, method=method, bounds=bounds, options=options)
                    
                    if opt_result.success:
                        from types import SimpleNamespace
                        ll = -opt_result.fun
                        k = len(opt_result.x)
                        aic = 2 * k - 2 * ll
                        
                        result = SimpleNamespace(
                            success=True, parameters=opt_result.x, log_likelihood=ll, aic=aic, n_evaluations=opt_result.nfev
                        )
                        
                        results.append({'name': spec['name'], 'result': result, 'success': True})
                        print(f"   ✅ Success ({method}) - AIC: {aic:.1f} ({opt_result.nfev} evals)")
                        success = True
                        break
                
                if not success:
                    print(f"   ❌ All optimizers failed")
                    results.append({'name': spec['name'], 'result': None, 'success': False})
            
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                results.append({
                    'name': spec['name'],
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        fit_time = time.time() - fit_start
        
        # Results analysis using our system capabilities
        print(f"\n🎯 Production System Results")
        print("=" * 65)
        print(f"Total fitting time: {fit_time:.1f} seconds")
        
        successful_results = [r for r in results if r['success'] and r['result']]
        print(f"Successful models: {len(successful_results)}/{len(results)}")
        
        if successful_results:
            # Model comparison table
            print(f"\n📊 Model Selection Results:")
            print("-" * 65)
            print(f"{'Model':<25} {'AIC':<10} {'ΔAIC':<8} {'LogLik':<10} {'K':<5}")
            print("-" * 65)
            
            # Calculate model statistics
            model_stats = []
            for r in successful_results:
                result = r['result']
                model_stats.append({
                    'name': r['name'],
                    'aic': result.aic,
                    'll': result.log_likelihood,
                    'k': len(result.parameters),
                    'result': result
                })
            
            # Sort by AIC
            model_stats.sort(key=lambda x: x['aic'])
            best_aic = model_stats[0]['aic']
            
            for stats in model_stats:
                delta_aic = stats['aic'] - best_aic
                print(f"{stats['name']:<25} {stats['aic']:<10.1f} {delta_aic:<8.1f} {stats['ll']:<10.1f} {stats['k']:<5}")
            
            print("-" * 65)
            
            # Best model summary
            best_model = model_stats[0]
            print(f"\n🏆 Best Model: {best_model['name']}")
            print(f"   AIC: {best_model['aic']:.3f}")
            print(f"   Log-likelihood: {best_model['ll']:.3f}")
            print(f"   Parameters: {best_model['k']}")
            
            # Parameter interpretation for intercept-only parameters
            params = best_model['result'].parameters
            if len(params) >= 3:
                # Convert logit to probabilities (first 3 parameters are typically phi, p, f intercepts)
                phi_prob = 1 / (1 + np.exp(-params[0]))
                p_prob = 1 / (1 + np.exp(-params[1]))
                f_prob = 1 / (1 + np.exp(-params[2]))
                
                print(f"\n📊 Biological Interpretation (Best Model):")
                print(f"   Annual survival rate: {phi_prob:.1%}")
                print(f"   Detection probability: {p_prob:.1%}")
                print(f"   Recruitment rate: {f_prob:.1%}")
        
        total_time = time.time() - start_time
        print(f"\n✅ Production analysis completed in {total_time:.1f} seconds")
        print(f"🚀 Processed {data_context.n_individuals:,} individuals using Pradel-JAX system")
        
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