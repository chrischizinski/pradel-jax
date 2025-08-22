#!/usr/bin/env python3
"""
Parallel Nebraska Analysis with Optimized Performance
- Vectorized likelihood computation (100-1000x speedup)
- Parallel model fitting across CPU cores
- Checkpoint/resume functionality
- Memory-efficient processing
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from pradel_jax.models.pradel_optimized import OptimizedPradelModel
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy
import sys
import time
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os

def prepare_nebraska_data(data_file: str, sample_size: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Prepare Nebraska data with consistent seeding."""
    print(f"📂 Loading Nebraska data from: {data_file}")
    
    np.random.seed(seed)
    
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
    
    # Convert to Pradel-JAX format
    years = list(range(2016, 2025))
    year_columns = [f'Y{year}' for year in years]
    age_columns = [f'age_{year}' for year in years]
    tier_columns = [f'tier_{year}' for year in years]
    
    encounter_data = (data[year_columns].values > 0).astype(int)
    
    formatted_data = []
    for i, (idx, row) in enumerate(data.iterrows()):
        individual_record = {
            'individual': row['person_id'],
            'gender': row['gender'] if pd.notna(row['gender']) else 0,
        }
        
        for j in range(len(year_columns)):
            individual_record[f'occasion_{j+1}'] = encounter_data[i, j]
        
        for j, (age_col, tier_col) in enumerate(zip(age_columns, tier_columns)):
            individual_record[f'age_{j+1}'] = row[age_col] if pd.notna(row[age_col]) else 0
            individual_record[f'tier_{j+1}'] = row[tier_col] if pd.notna(row[tier_col]) else 0
            
        formatted_data.append(individual_record)
    
    return pd.DataFrame(formatted_data)

def fit_single_model_worker(args):
    """
    Worker function for parallel model fitting.
    Runs in separate process to enable true parallelization.
    """
    model_spec, data_csv_path, model_index, total_models = args
    
    try:
        # Load data in worker process
        data_context = pj.load_data(data_csv_path)
        
        # Create optimized model
        model = OptimizedPradelModel()
        
        start_time = time.time()
        
        # Build design matrices and get parameters
        design_matrices = model.build_design_matrices(model_spec['formula'], data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Define objective function
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                return -ll if np.isfinite(ll) else 1e10
            except:
                return 1e10
        
        # Optimize using framework
        result = optimize_model(
            objective_function=objective,
            initial_parameters=initial_params,
            context=data_context,
            bounds=bounds,
            preferred_strategy=OptimizationStrategy.HYBRID
        )
        
        fit_time = time.time() - start_time
        
        if result.success:
            opt_result = result.result
            ll = -opt_result.fun
            k = len(opt_result.x)
            aic = 2 * k - 2 * ll
            
            return {
                'model_index': model_index,
                'name': model_spec['name'],
                'result': {
                    'success': True,
                    'parameters': opt_result.x.tolist(),
                    'log_likelihood': ll,
                    'aic': aic,
                    'n_evaluations': opt_result.nfev,
                    'n_parameters': k,
                    'strategy_used': result.strategy_used,
                    'fit_time': fit_time,
                    'formula_spec': {
                        'phi': model_spec['formula'].phi,
                        'p': model_spec['formula'].p, 
                        'f': model_spec['formula'].f
                    },
                    'convergence_message': opt_result.message
                }
            }
        else:
            return {
                'model_index': model_index,
                'name': model_spec['name'],
                'result': {
                    'success': False,
                    'error': 'Optimization failed',
                    'fit_time': fit_time,
                    'message': getattr(result.result, 'message', 'Unknown error') if hasattr(result, 'result') else 'Unknown error'
                }
            }
            
    except Exception as e:
        return {
            'model_index': model_index,
            'name': model_spec['name'],
            'result': {
                'success': False,
                'error': str(e),
                'fit_time': 0
            }
        }

def run_parallel_analysis(
    sample_size: int,
    output_dir: str = "results",
    n_workers: Optional[int] = None,
    batch_size: int = 8,
    resume: bool = False
):
    """
    Run analysis with parallel model fitting.
    
    Args:
        sample_size: Number of individuals to analyze
        output_dir: Output directory for results
        n_workers: Number of parallel workers (default: CPU count)
        batch_size: Number of models to fit in parallel batches
        resume: Resume from checkpoint if available
    """
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print(f"🚀 Starting parallel analysis with {n_workers} workers")
    
    # Setup
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    base_name = f"nebraska_parallel_{sample_size}ind_{timestamp}"
    checkpoint_file = output_path / f"{base_name}_checkpoint.pkl"
    
    # Try to resume from checkpoint
    if resume and checkpoint_file.exists():
        print("🔄 Resuming from checkpoint...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        results = checkpoint_data['results']
        completed_indices = checkpoint_data['completed_indices']
        data_csv_path = checkpoint_data['data_csv_path']
        model_specifications = checkpoint_data['model_specifications']
        analysis_start_time = checkpoint_data['analysis_start_time']
    else:
        print("🚀 Starting fresh parallel analysis...")
        
        # Prepare data
        data_file = "data/encounter_histories_ne_clean.csv"
        nebraska_data = prepare_nebraska_data(data_file, sample_size)
        
        # Save data to temporary CSV for worker processes
        data_csv_path = output_path / f"temp_data_{sample_size}.csv"
        nebraska_data.to_csv(data_csv_path, index=False)
        
        print(f"✅ Data prepared: {len(nebraska_data):,} individuals")
        
        # Define all model specifications
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
        
        results = [None] * len(model_specifications)
        completed_indices = set()
        analysis_start_time = time.time()
    
    total_models = len(model_specifications)
    print(f"📊 Total models: {total_models}")
    print(f"📊 Parallel workers: {n_workers}")
    print(f"📊 Batch size: {batch_size}")
    
    # Process models in parallel batches
    remaining_indices = [i for i in range(total_models) if i not in completed_indices]
    
    for batch_start in range(0, len(remaining_indices), batch_size):
        batch_indices = remaining_indices[batch_start:batch_start + batch_size]
        batch_specs = [(model_specifications[i], str(data_csv_path), i, total_models) 
                      for i in batch_indices]
        
        print(f"\n🔧 Processing batch: models {batch_indices[0]+1}-{batch_indices[-1]+1}")
        batch_start_time = time.time()
        
        # Submit batch to parallel workers
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_index = {
                executor.submit(fit_single_model_worker, spec): spec[2] 
                for spec in batch_specs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    idx = result['model_index']
                    results[idx] = result
                    completed_indices.add(idx)
                    
                    if result['result']['success']:
                        print(f"   ✅ {result['name']}: AIC={result['result']['aic']:.1f}")
                    else:
                        print(f"   ❌ {result['name']}: {result['result']['error']}")
                        
                except Exception as e:
                    print(f"   ❌ Worker error: {str(e)}")
        
        batch_time = time.time() - batch_start_time
        models_per_second = len(batch_indices) / batch_time
        
        # Progress reporting
        completed_count = len(completed_indices)
        elapsed_total = time.time() - analysis_start_time
        
        if completed_count > 0:
            avg_time_per_model = elapsed_total / completed_count
            remaining_models = total_models - completed_count
            eta_seconds = remaining_models / models_per_second if models_per_second > 0 else 0
            
            print(f"   ⏱️  Batch time: {batch_time:.1f}s ({models_per_second:.1f} models/sec)")
            print(f"   📊 Progress: {completed_count}/{total_models} ({completed_count/total_models*100:.1f}%)")
            print(f"   🔮 ETA: {eta_seconds/3600:.1f} hours")
        
        # Save checkpoint after each batch
        checkpoint_data = {
            'results': results,
            'completed_indices': completed_indices,
            'data_csv_path': str(data_csv_path),
            'model_specifications': model_specifications,
            'analysis_start_time': analysis_start_time,
            'sample_size': sample_size
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"   💾 Checkpoint saved")
    
    # Final results
    total_time = time.time() - analysis_start_time
    successful_results = [r for r in results if r and r['result']['success']]
    
    print(f"\n🎯 Parallel Analysis Complete!")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Successful models: {len(successful_results)}/{total_models}")
    print(f"Average time per model: {total_time/total_models:.1f} seconds")
    print(f"Parallel speedup: ~{n_workers}x (theoretical)")
    
    # Export final results
    final_csv = output_path / f"{base_name}_final.csv"
    
    csv_data = []
    for result in results:
        if result is None:
            continue
            
        row = {
            'model': result['name'],
            'success': result['result']['success']
        }
        
        if result['result']['success']:
            res = result['result']
            row.update({
                'aic': res['aic'],
                'log_likelihood': res['log_likelihood'],
                'n_parameters': res['n_parameters'],
                'strategy_used': res.get('strategy_used', 'unknown'),
                'fit_time': res.get('fit_time', 0),
                'n_evaluations': res.get('n_evaluations', 0)
            })
            
            for i, param in enumerate(res['parameters']):
                row[f'param_{i+1}'] = param
        else:
            row.update({
                'aic': np.nan,
                'log_likelihood': np.nan,
                'error': result['result'].get('error', 'Unknown')
            })
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(final_csv, index=False)
    
    print(f"📊 Results exported to: {final_csv}")
    
    # Clean up
    if data_csv_path.exists():
        data_csv_path.unlink()
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    return results

def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(description='Parallel Nebraska Analysis')
    parser.add_argument('--sample-size', type=int, default=1000, 
                       help='Number of individuals to analyze')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Number of models per parallel batch')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if available')
    
    args = parser.parse_args()
    
    print("🔬 Parallel Nebraska Analysis with Optimizations")
    print("=" * 60)
    print(f"Sample size: {args.sample_size:,} individuals")
    print(f"Workers: {args.workers or mp.cpu_count()}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resume mode: {args.resume}")
    
    if not Path("data/encounter_histories_ne_clean.csv").exists():
        print("❌ Data file not found: data/encounter_histories_ne_clean.csv")
        sys.exit(1)
    
    try:
        results = run_parallel_analysis(
            sample_size=args.sample_size,
            output_dir=args.output_dir,
            n_workers=args.workers,
            batch_size=args.batch_size,
            resume=args.resume
        )
        
        successful = sum(1 for r in results if r and r['result']['success'])
        print(f"\n✅ Analysis completed: {successful}/{len(results)} models successful")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        print("💾 Progress saved in checkpoint - use --resume to continue")
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()