#!/usr/bin/env python3
"""
Simplified Large-Scale Nebraska Analysis
Uses the basic optimization that we know works, optimized for larger datasets.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
import sys
import time
import gc
from pathlib import Path
from typing import Optional

def create_efficient_data_loader(data_file: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Efficiently load large datasets with memory management."""
    print(f"📂 Loading data efficiently from: {data_file}")
    
    if sample_size:
        # Read first few rows to get column info
        sample_data = pd.read_csv(data_file, nrows=5)
        total_rows = sum(1 for _ in open(data_file)) - 1  # -1 for header
        
        print(f"   Total rows in dataset: {total_rows:,}")
        print(f"   Sampling {sample_size:,} rows...")
        
        # Use skiprows for efficient random sampling
        np.random.seed(42)
        skip_indices = sorted(np.random.choice(range(1, total_rows + 1), 
                                             size=total_rows - sample_size, 
                                             replace=False))
        data = pd.read_csv(data_file, skiprows=skip_indices)
        print(f"   Loaded sample: {len(data):,} rows")
    else:
        print(f"   Loading full dataset...")
        data = pd.read_csv(data_file)
        print(f"   Loaded: {len(data):,} rows")
    
    return data

def process_data_efficiently(data: pd.DataFrame) -> pd.DataFrame:
    """Process encounter histories with memory optimization and covariates."""
    print("🔧 Processing encounter histories and covariates efficiently...")
    
    year_columns = ['Y2016', 'Y2017', 'Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024']
    age_columns = ['age_2016', 'age_2017', 'age_2018', 'age_2019', 'age_2020', 'age_2021', 'age_2022', 'age_2023', 'age_2024']
    tier_columns = ['tier_2016', 'tier_2017', 'tier_2018', 'tier_2019', 'tier_2020', 'tier_2021', 'tier_2022', 'tier_2023', 'tier_2024']
    
    n_individuals = len(data)
    n_occasions = len(year_columns)
    
    # Pre-allocate arrays for better memory efficiency
    individuals = data['person_id'].values
    encounter_data = np.zeros((n_individuals, n_occasions), dtype=np.int8)
    
    # Process encounter histories
    for i, year_col in enumerate(year_columns):
        encounter_data[:, i] = (data[year_col].values > 0).astype(np.int8)
    
    # Process covariates
    gender_data = data['gender'].fillna(0).values  # 0=unknown, 1=male, 2=female
    age_data = data[age_columns].fillna(0).values.astype(np.float32)
    tier_data = data[tier_columns].fillna(0).values.astype(np.int8)
    
    # Create DataFrame efficiently
    occasion_cols = [f'occasion_{i+1}' for i in range(n_occasions)]
    age_cols = [f'age_{i+1}' for i in range(n_occasions)]
    tier_cols = [f'tier_{i+1}' for i in range(n_occasions)]
    
    processed_data = pd.DataFrame({
        'individual': individuals,
        'gender': gender_data,
        **{col: encounter_data[:, i] for i, col in enumerate(occasion_cols)},
        **{col: age_data[:, i] for i, col in enumerate(age_cols)},
        **{col: tier_data[:, i] for i, col in enumerate(tier_cols)}
    })
    
    # Memory cleanup
    del encounter_data, individuals, gender_data, age_data, tier_data
    gc.collect()
    
    return processed_data

def estimate_single_model_time(n_individuals: int) -> float:
    """Estimate runtime in minutes for a single model based on dataset size."""
    if n_individuals < 1000:
        return 0.2
    elif n_individuals < 5000:
        return 0.8
    elif n_individuals < 20000:
        return 3.0
    elif n_individuals < 50000:
        return 8.0
    else:
        return 15.0

def main():
    """Run simplified large-scale analysis."""
    
    print("🔬 Simplified Large-Scale Nebraska Pradel Analysis")
    print("=" * 60)
    
    # Configuration
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Ask user for sample size
    print("📊 Dataset Size Options:")
    print("   1. Small test (1,000 individuals) - ~1 minute")
    print("   2. Medium (5,000 individuals) - ~3 minutes")  
    print("   3. Large (20,000 individuals) - ~10 minutes")
    print("   4. Extra Large (50,000 individuals) - ~30 minutes")
    print("   5. Full dataset (~111,000 individuals) - ~1 hour")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        size_map = {
            '1': 1000,
            '2': 5000, 
            '3': 20000,
            '4': 50000,
            '5': None  # Full dataset
        }
        
        sample_size = size_map.get(choice, 1000)
        
        if sample_size:
            print(f"✅ Selected: {sample_size:,} individuals")
        else:
            print("✅ Selected: Full dataset")
            
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Analysis cancelled")
        return
    
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        # Efficient data loading
        sampled_data = create_efficient_data_loader(data_file, sample_size)
        
        # Efficient processing
        processed_data = process_data_efficiently(sampled_data)
        occasions = [col for col in processed_data.columns if col.startswith('occasion_')]
        
        # Memory cleanup
        del sampled_data
        gc.collect()
        
        # Summary statistics
        total_detections = processed_data[occasions].sum().sum()
        total_possible = len(processed_data) * len(occasions)
        never_detected = (processed_data[occasions].sum(axis=1) == 0).sum()
        
        print(f"\n📊 Data Summary:")
        print(f"   - Individuals: {len(processed_data):,}")
        print(f"   - Occasions: {len(occasions)}")
        print(f"   - Total detections: {total_detections:,} / {total_possible:,} ({total_detections/total_possible:.3f})")
        print(f"   - Never detected: {never_detected:,} ({never_detected/len(processed_data):.3f})")
        
        # Save and load into pradel-jax
        temp_file = f"temp_nebraska_n{len(processed_data)}.csv"
        print(f"💾 Saving processed data to {temp_file}...")
        processed_data.to_csv(temp_file, index=False)
        
        print("🔧 Loading data into pradel-jax...")
        data_context = pj.load_data(temp_file)
        print(f"   ✅ Loaded: {data_context.n_individuals:,} individuals, {data_context.n_occasions} occasions")
        
        # Model setup using the approach we know works
        print("⚡ Setting up Pradel model...")
        model = pj.PradelModel()
        formula_spec = pj.create_simple_spec(phi="~1", p="~1", f="~1", name="Intercept-only")
        
        # Build design matrices
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        print(f"   Parameters to estimate: {len(initial_params)}")
        print(f"   Initial parameters: {initial_params}")
        
        # Objective function with robust error handling
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                if np.isnan(ll) or np.isinf(ll):
                    return 1e10
                return -ll
            except Exception:
                return 1e10
        
        # Define multiple model specifications
        model_specs = [
            {
                'name': 'Intercept-only model',
                'formula': pj.create_simple_spec(phi="~1", p="~1", f="~1", name="Intercept-only")
            },
            {
                'name': 'Gender effect model',
                'formula': pj.create_simple_spec(phi="~1 + gender", p="~1", f="~1", name="Gender")
            },
            {
                'name': 'Age effect model',
                'formula': pj.create_simple_spec(phi="~1 + age_1", p="~1", f="~1", name="Age")
            },
            {
                'name': 'Tier effect model',
                'formula': pj.create_simple_spec(phi="~1 + tier_1", p="~1", f="~1", name="Tier")
            },
            {
                'name': 'Gender + Age model',
                'formula': pj.create_simple_spec(phi="~1 + gender + age_1", p="~1", f="~1", name="Gender+Age")
            },
            {
                'name': 'Age + Tier model',
                'formula': pj.create_simple_spec(phi="~1 + age_1 + tier_1", p="~1", f="~1", name="Age+Tier")
            },
            {
                'name': 'Full model (Gender + Age + Tier)',
                'formula': pj.create_simple_spec(phi="~1 + gender + age_1 + tier_1", p="~1", f="~1", name="Full")
            }
        ]
        
        # Estimate total time for all models
        base_time_per_model = estimate_single_model_time(data_context.n_individuals)
        total_expected_time = base_time_per_model * len(model_specs)
        
        print(f"🚀 Fitting {len(model_specs)} models to {data_context.n_individuals:,} individuals...")
        print(f"   Models: {[spec['name'] for spec in model_specs]}")
        print(f"   Expected total time: {total_expected_time:.1f} minutes ({base_time_per_model:.1f} min/model)")
        
        # Use scipy directly for reliability with large datasets
        from scipy.optimize import minimize
        
        all_results = []
        total_fit_start = time.time()
        
        for i, spec in enumerate(model_specs, 1):
            model_start_time = time.time()
            elapsed_total = model_start_time - total_fit_start
            remaining_models = len(model_specs) - i + 1
            
            if i > 1:
                avg_time_per_model = elapsed_total / (i - 1)
                estimated_remaining = avg_time_per_model * remaining_models
                print(f"\n⚡ Model {i}/{len(model_specs)}: {spec['name']} (Est. remaining: {estimated_remaining/60:.1f} min)")
            else:
                print(f"\n⚡ Model {i}/{len(model_specs)}: {spec['name']}")
            
            try:
                # Build design matrices for this model
                design_matrices = model.build_design_matrices(spec['formula'], data_context)
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                # Objective function for this model
                def objective(params):
                    try:
                        ll = model.log_likelihood(params, data_context, design_matrices)
                        if np.isnan(ll) or np.isinf(ll):
                            return 1e10
                        return -ll
                    except Exception:
                        return 1e10
                
                fit_start = time.time()
                
                # Try L-BFGS-B first (good for large problems)
                result_lbfgs = minimize(
                    objective, 
                    initial_params, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )
                
                if result_lbfgs.success:
                    result = result_lbfgs
                    method = "L-BFGS-B"
                else:
                    # Fallback to SLSQP
                    result_slsqp = minimize(
                        objective,
                        initial_params,
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-9}
                    )
                    
                    if result_slsqp.success:
                        result = result_slsqp 
                        method = "SLSQP"
                    else:
                        # Last resort - Nelder-Mead (no bounds)
                        result_nm = minimize(
                            objective,
                            initial_params,
                            method='Nelder-Mead',
                            options={'maxiter': 2000}
                        )
                        result = result_nm
                        method = "Nelder-Mead"
                
                fit_time = time.time() - fit_start
                
                # Store results
                model_result = {
                    'name': spec['name'],
                    'result': result,
                    'method': method,
                    'fit_time': fit_time,
                    'n_params': len(initial_params)
                }
                all_results.append(model_result)
                
                # Quick status update
                if result.success:
                    ll = -result.fun
                    aic = 2 * len(result.x) - 2 * ll
                    print(f"   ✅ Success ({method}, {fit_time:.1f}s) - AIC: {aic:.1f}")
                else:
                    print(f"   ❌ Failed ({method}, {fit_time:.1f}s)")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                model_result = {
                    'name': spec['name'],
                    'result': None,
                    'method': 'Error',
                    'fit_time': 0,
                    'n_params': 0,
                    'error': str(e)
                }
                all_results.append(model_result)
        
        total_fit_time = time.time() - total_fit_start
        
        # Results Summary
        print(f"\n🎯 Model Comparison Results")
        print("=" * 80)
        print(f"Total fitting time: {total_fit_time:.1f} seconds")
        print(f"Successful models: {sum(1 for r in all_results if r.get('result') and r['result'].success)}/{len(all_results)}")
        
        # Create results table
        successful_results = [r for r in all_results if r.get('result') and r['result'].success]
        
        if successful_results:
            print(f"\n📊 Model Selection Table:")
            print("-" * 80)
            print(f"{'Model':<25} {'AIC':<10} {'ΔAICc':<8} {'LogLik':<10} {'K':<3} {'Time':<6}")
            print("-" * 80)
            
            # Calculate AIC values
            model_stats = []
            for r in successful_results:
                result = r['result']
                ll = -result.fun
                k = len(result.x)
                n = data_context.n_individuals
                aic = 2 * k - 2 * ll
                # AICc correction for finite sample size
                aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else aic
                
                model_stats.append({
                    'name': r['name'],
                    'aic': aic,
                    'aicc': aicc,
                    'll': ll,
                    'k': k,
                    'time': r['fit_time'],
                    'result': result
                })
            
            # Sort by AICc
            model_stats.sort(key=lambda x: x['aicc'])
            best_aicc = model_stats[0]['aicc']
            
            # Print table
            for stats in model_stats:
                delta_aicc = stats['aicc'] - best_aicc
                print(f"{stats['name']:<25} {stats['aic']:<10.1f} {delta_aicc:<8.1f} {stats['ll']:<10.1f} {stats['k']:<3} {stats['time']:<6.1f}")
            
            print("-" * 80)
            print("K = number of parameters, ΔAICc = difference from best model")
            
            # Best model details
            best_model = model_stats[0]
            print(f"\n🏆 Best Model: {best_model['name']}")
            print(f"   AICc: {best_model['aicc']:.3f}")
            print(f"   Log-likelihood: {best_model['ll']:.3f}")
            print(f"   Parameters: {best_model['k']}")
            
            # Parameter estimates for best model
            print(f"\n📋 Parameter Estimates (Best Model):")
            best_result = best_model['result']
            n_phi_params = best_model['k'] // 3 if best_model['k'] % 3 == 0 else (best_model['k'] - 2) // 3 + 1
            
            phi_params = best_result.x[:n_phi_params] if n_phi_params > 0 else []
            p_params = [best_result.x[n_phi_params]] if best_model['k'] > n_phi_params else []
            f_params = best_result.x[n_phi_params+1:] if best_model['k'] > n_phi_params + 1 else []
            
            print(f"   Survival (φ) parameters: {len(phi_params)}")
            for i, param in enumerate(phi_params):
                prob = 1 / (1 + np.exp(-param))
                param_name = 'intercept' if i == 0 else f'covariate_{i}'
                print(f"     φ({param_name}): {param:.4f} (prob: {prob:.4f})")
            
            if p_params:
                prob_p = 1 / (1 + np.exp(-p_params[0]))
                print(f"   Detection (p): {p_params[0]:.4f} (prob: {prob_p:.4f})")
            
            if f_params:
                prob_f = 1 / (1 + np.exp(-f_params[0]))
                print(f"   Recruitment (f): {f_params[0]:.4f} (prob: {prob_f:.4f})")
            
        # Failed models
        failed_results = [r for r in all_results if not (r.get('result') and r['result'].success)]
        if failed_results:
            print(f"\n❌ Failed Models:")
            for r in failed_results:
                if 'error' in r:
                    print(f"   {r['name']}: {r['error']}")
                else:
                    print(f"   {r['name']}: Optimization failed")
        
        print(f"\n🎯 Model Selection Recommendations:")
        if successful_results:
            if len(model_stats) > 1:
                delta_aicc_2nd = model_stats[1]['aicc'] - best_aicc
                if delta_aicc_2nd < 2:
                    print(f"   ⚠️  Models within ΔAICc < 2: Consider model averaging")
                elif delta_aicc_2nd < 7:
                    print(f"   ✅ Clear best model, but {model_stats[1]['name']} has some support")
                else:
                    print(f"   ✅ Strong support for {best_model['name']}")
            print(f"   💡 Best model explains biological variation in survival with AICc = {best_model['aicc']:.1f}")
        else:
            print(f"   ❌ No models converged successfully")
            print(f"   💡 Try simpler models or check data quality")
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time:.1f} seconds")
        print(f"📊 Successfully processed {data_context.n_individuals:,} individuals")
        
        # Cleanup
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
    except MemoryError:
        print("❌ Memory error: Dataset too large for available RAM")
        print("💡 Try a smaller sample size")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()