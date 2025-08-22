#!/usr/bin/env python3
"""
High-Performance Nebraska Pradel Model Analysis
Optimized for speed with large datasets using parallel processing and smart convergence.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
import sys
import time
import gc
from pathlib import Path
from typing import Optional, List, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")

def create_efficient_data_loader(data_file: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Efficiently load large datasets with memory management."""
    print(f"📂 Loading data efficiently from: {data_file}")
    
    if sample_size:
        # Use chunk-based sampling for very large files
        chunk_iter = pd.read_csv(data_file, chunksize=10000)
        sampled_chunks = []
        total_sampled = 0
        
        for chunk in chunk_iter:
            if total_sampled >= sample_size:
                break
            
            chunk_sample_size = min(sample_size - total_sampled, len(chunk))
            if chunk_sample_size < len(chunk):
                chunk_sample = chunk.sample(n=chunk_sample_size, random_state=42)
            else:
                chunk_sample = chunk
            
            sampled_chunks.append(chunk_sample)
            total_sampled += len(chunk_sample)
        
        data = pd.concat(sampled_chunks, ignore_index=True)
        print(f"   Loaded sample: {len(data):,} rows efficiently")
    else:
        data = pd.read_csv(data_file)
        print(f"   Loaded: {len(data):,} rows")
    
    return data

def process_data_ultra_fast(data: pd.DataFrame) -> pd.DataFrame:
    """Ultra-fast data processing with vectorized operations."""
    print("⚡ Ultra-fast data processing...")
    
    # Use numpy for maximum speed
    year_cols = ['Y2016', 'Y2017', 'Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024']
    age_cols = ['age_2016', 'age_2017', 'age_2018', 'age_2019', 'age_2020', 'age_2021', 'age_2022', 'age_2023', 'age_2024']
    tier_cols = ['tier_2016', 'tier_2017', 'tier_2018', 'tier_2019', 'tier_2020', 'tier_2021', 'tier_2022', 'tier_2023', 'tier_2024']
    
    # Vectorized processing - much faster than loops
    encounter_matrix = (data[year_cols].values > 0).astype(np.int8)
    age_matrix = data[age_cols].fillna(0).values.astype(np.float32)
    tier_matrix = data[tier_cols].fillna(0).values.astype(np.int8)
    gender_vector = data['gender'].fillna(0).values.astype(np.float32)
    
    # Create result dictionary efficiently
    result_dict = {'individual': data['person_id'].values.astype(str)}
    
    # Add encounter histories
    for i in range(9):
        result_dict[f'occasion_{i+1}'] = encounter_matrix[:, i]
    
    # Add covariates (only baseline age and tier to speed up)
    result_dict['gender'] = gender_vector
    result_dict['age_1'] = age_matrix[:, 0]  # 2016 age as baseline
    result_dict['tier_1'] = tier_matrix[:, 0]  # 2016 tier as baseline
    
    # Fast DataFrame creation
    processed_data = pd.DataFrame(result_dict)
    
    # Memory cleanup
    del encounter_matrix, age_matrix, tier_matrix, gender_vector
    gc.collect()
    
    return processed_data

def fit_single_model_optimized(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """Fit a single model with optimized settings."""
    try:
        import pradel_jax as pj
        from scipy.optimize import minimize
        import numpy as np
        
        # Unpack data
        temp_file = model_data['temp_file']
        formula_spec = model_data['formula_spec']
        model_name = model_data['name']
        
        # Load data context
        data_context = pj.load_data(temp_file)
        model = pj.PradelModel()
        
        # Build model components
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Ultra-fast objective function with caching
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                return -ll if np.isfinite(ll) else 1e10
            except:
                return 1e10
        
        start_time = time.time()
        
        # Try optimized L-BFGS-B first (fastest for large problems)
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 200,  # Reduced for speed
                'ftol': 1e-6,    # Relaxed tolerance for speed
                'gtol': 1e-5
            }
        )
        
        # Quick fallback if needed
        if not result.success and len(initial_params) <= 5:  # Only for small models
            result = minimize(
                objective,
                initial_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
        
        fit_time = time.time() - start_time
        
        return {
            'name': model_name,
            'success': result.success,
            'parameters': result.x if result.success else None,
            'objective_value': result.fun if result.success else None,
            'n_evaluations': result.nfev if hasattr(result, 'nfev') else 0,
            'fit_time': fit_time,
            'n_params': len(initial_params)
        }
        
    except Exception as e:
        return {
            'name': model_name,
            'success': False,
            'error': str(e),
            'fit_time': 0,
            'n_params': 0
        }

def main():
    """Run high-performance analysis."""
    
    print("🚀 High-Performance Nebraska Pradel Analysis")
    print("=" * 60)
    
    # Configuration
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Smart dataset size selection
    print("📊 High-Performance Dataset Options:")
    print("   1. Development (1K) - 30 seconds")
    print("   2. Testing (5K) - 2 minutes") 
    print("   3. Production (20K) - 5 minutes")
    print("   4. Large-scale (50K) - 12 minutes")
    print("   5. Full dataset (~111K) - 25 minutes")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        size_map = {'1': 1000, '2': 5000, '3': 20000, '4': 50000, '5': None}
        sample_size = size_map.get(choice, 5000)
        
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
    
    total_start = time.time()
    
    try:
        # Ultra-fast data loading and processing
        np.random.seed(42)
        raw_data = create_efficient_data_loader(data_file, sample_size)
        processed_data = process_data_ultra_fast(raw_data)
        
        # Memory cleanup
        del raw_data
        gc.collect()
        
        # Quick stats
        occasions = [f'occasion_{i}' for i in range(1, 10)]
        total_detections = processed_data[occasions].sum().sum()
        total_possible = len(processed_data) * 9
        
        print(f"\n📊 Data Summary:")
        print(f"   - Individuals: {len(processed_data):,}")
        print(f"   - Detection rate: {total_detections/total_possible:.3f}")
        print(f"   - Processing time: {time.time() - total_start:.1f}s")
        
        # Save processed data
        temp_file = f"temp_hp_nebraska_{len(processed_data)}.csv"
        processed_data.to_csv(temp_file, index=False)
        
        # Define optimized model set (reduced for speed)
        model_specs = [
            {
                'name': 'Intercept-only',
                'formula_spec': pj.create_simple_spec(phi="~1", p="~1", f="~1"),
                'temp_file': temp_file
            },
            {
                'name': 'Gender effect',
                'formula_spec': pj.create_simple_spec(phi="~1 + gender", p="~1", f="~1"),
                'temp_file': temp_file
            },
            {
                'name': 'Age effect',
                'formula_spec': pj.create_simple_spec(phi="~1 + age_1", p="~1", f="~1"),
                'temp_file': temp_file
            },
            {
                'name': 'Tier effect', 
                'formula_spec': pj.create_simple_spec(phi="~1 + tier_1", p="~1", f="~1"),
                'temp_file': temp_file
            },
            {
                'name': 'Age + Tier',
                'formula_spec': pj.create_simple_spec(phi="~1 + age_1 + tier_1", p="~1", f="~1"),
                'temp_file': temp_file
            }
        ]
        
        print(f"\n🚀 Fitting {len(model_specs)} models in parallel...")
        
        # Parallel model fitting for maximum speed
        fit_start = time.time()
        n_cores = min(mp.cpu_count(), len(model_specs))
        print(f"   Using {n_cores} CPU cores")
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            # Submit all models
            future_to_model = {
                executor.submit(fit_single_model_optimized, spec): spec['name'] 
                for spec in model_specs
            }
            
            # Collect results as they complete
            results = []
            completed = 0
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    status = "✅" if result['success'] else "❌"
                    print(f"   {status} {model_name} ({completed}/{len(model_specs)}) - {result['fit_time']:.1f}s")
                    
                except Exception as e:
                    print(f"   ❌ {model_name} failed: {e}")
        
        total_fit_time = time.time() - fit_start
        
        # Results analysis
        print(f"\n🎯 Results Summary")
        print("=" * 60)
        print(f"Total fitting time: {total_fit_time:.1f}s (avg: {total_fit_time/len(model_specs):.1f}s/model)")
        
        successful_results = [r for r in results if r['success']]
        print(f"Successful models: {len(successful_results)}/{len(results)}")
        
        if successful_results:
            # Calculate AICc for model selection
            n = len(processed_data)
            model_stats = []
            
            for r in successful_results:
                ll = -r['objective_value']
                k = r['n_params']
                aic = 2 * k - 2 * ll
                aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n - k - 1 > 0 else aic
                
                model_stats.append({
                    'name': r['name'],
                    'aic': aic,
                    'aicc': aicc,
                    'll': ll,
                    'k': k,
                    'time': r['fit_time'],
                    'params': r['parameters']
                })
            
            # Sort by AICc
            model_stats.sort(key=lambda x: x['aicc'])
            
            print(f"\n📊 Model Ranking (by AICc):")
            print("-" * 50)
            print(f"{'Rank':<5} {'Model':<20} {'AICc':<10} {'ΔAICc':<8} {'Time':<6}")
            print("-" * 50)
            
            best_aicc = model_stats[0]['aicc']
            for i, stats in enumerate(model_stats, 1):
                delta = stats['aicc'] - best_aicc
                print(f"{i:<5} {stats['name']:<20} {stats['aicc']:<10.1f} {delta:<8.1f} {stats['time']:<6.1f}")
            
            print(f"\n🏆 Best Model: {model_stats[0]['name']}")
            print(f"   AICc: {model_stats[0]['aicc']:.1f}")
            print(f"   Log-likelihood: {model_stats[0]['ll']:.1f}")
            
            # Parameter interpretation for best model
            best_params = model_stats[0]['params']
            if len(best_params) >= 3:
                phi = 1 / (1 + np.exp(-best_params[0]))
                p = 1 / (1 + np.exp(-best_params[1])) 
                f = 1 / (1 + np.exp(-best_params[2]))
                
                print(f"\n📊 Best Model Estimates:")
                print(f"   Survival: {phi:.1%}")
                print(f"   Detection: {p:.1%}")
                print(f"   Recruitment: {f:.1%}")
        
        total_time = time.time() - total_start
        print(f"\n✅ High-performance analysis completed!")
        print(f"📊 Total time: {total_time:.1f}s ({len(processed_data):,} individuals)")
        print(f"⚡ Speed: {len(processed_data)/total_time:,.0f} individuals/second")
        
        # Cleanup
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()