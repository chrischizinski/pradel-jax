#!/usr/bin/env python3
"""
Nebraska Data Analysis - Working Version with Parameter Export
Uses the optimization framework directly since high-level API is not ready.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from pradel_jax.optimization import (
    optimize_model, 
    fit_models_parallel, 
    create_model_specs_from_formulas,
    ParallelOptimizer
)
from pradel_jax.optimization.strategy import OptimizationStrategy
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings
import multiprocessing as mp

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

def fit_single_model_working(model, formula_spec, data_context, model_name: str) -> Dict[str, Any]:
    """
    Fit a single model using the optimization framework directly.
    """
    try:
        # Build design matrices and get initial parameters
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Create objective function
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                return -ll if np.isfinite(ll) else 1e10
            except:
                return 1e10
        
        # Use the optimization framework
        result = optimize_model(
            objective_function=objective,
            initial_parameters=initial_params,
            context=data_context,
            bounds=bounds,
            preferred_strategy=OptimizationStrategy.HYBRID  # Use the robust hybrid strategy
        )
        
        if result.success:
            # Extract the actual optimization result
            opt_result = result.result
            
            # Calculate log-likelihood and AIC
            ll = -opt_result.fun  # Objective was negative log-likelihood
            k = len(opt_result.x)  # Number of parameters
            aic = 2 * k - 2 * ll
            
            return {
                'success': True,
                'parameters': opt_result.x.tolist() if hasattr(opt_result.x, 'tolist') else list(opt_result.x),
                'log_likelihood': ll,
                'aic': aic,
                'n_evaluations': opt_result.nfev,
                'n_parameters': k,
                'strategy_used': result.strategy_used,
                'formula_spec': {
                    'phi': formula_spec.phi,
                    'p': formula_spec.p, 
                    'f': formula_spec.f
                },
                'convergence_message': opt_result.message,
                'optimization_time': opt_result.optimization_time
            }
        else:
            return {
                'success': False,
                'error': 'Optimization failed',
                'message': getattr(result.result, 'message', 'Unknown error') if hasattr(result, 'result') else 'Unknown error',
                'strategy_attempted': result.strategy_used if hasattr(result, 'strategy_used') else 'unknown'
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
            model_result = model_info['result']
            row.update({
                'aic': model_result['aic'],
                'log_likelihood': model_result['log_likelihood'],
                'n_parameters': model_result['n_parameters'],
                'strategy_used': model_result.get('strategy_used', 'unknown'),
                'n_evaluations': model_result.get('n_evaluations', 0),
                'optimization_time': model_result.get('optimization_time', 0),
                'convergence_message': model_result.get('convergence_message', '')
            })
            
            # Add individual parameters
            for i, param in enumerate(model_result['parameters']):
                row[f'param_{i+1}'] = param
        else:
            row.update({
                'aic': np.nan,
                'log_likelihood': np.nan,
                'n_parameters': np.nan,
                'error': model_info['result'].get('error', 'Unknown error')
            })
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"   ✅ CSV results exported successfully")

def run_parallel_analysis(sample_size: int, n_workers: Optional[int] = None, output_dir: str = "results"):
    """Run analysis using core parallel optimization framework."""
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)
    
    print(f"🚀 Running parallel analysis with {n_workers} workers")
    
    # Prepare data
    data_file = "data/encounter_histories_ne_clean.csv"
    nebraska_data = prepare_nebraska_data(data_file, sample_size)
    
    # Load into Pradel-JAX
    temp_file = f"temp_working_{sample_size}.csv"
    nebraska_data.to_csv(temp_file, index=False)
    data_context = pj.load_data(temp_file)
    
    print(f"✅ Loaded: {data_context.n_individuals:,} individuals, {data_context.n_occasions} occasions")
    
    # Define model formulas using core functionality
    phi_formulas = ["~1", "~1 + age_1"]
    p_formulas = ["~1"]  # Keep detection constant for simplicity
    f_formulas = ["~1", "~1 + gender"]
    
    # Create model specifications using core function
    model_specs = create_model_specs_from_formulas(phi_formulas, p_formulas, f_formulas)
    
    print(f"📊 Created {len(model_specs)} model specifications")
    
    # Run parallel optimization using core framework
    start_time = time.time()
    
    results = fit_models_parallel(
        model_specs=model_specs,
        data_context=data_context,
        n_workers=n_workers,
        strategy=OptimizationStrategy.HYBRID,
        batch_size=min(4, n_workers),
        checkpoint_interval=2,
        checkpoint_name=f"nebraska_{sample_size}",
        resume=False
    )
    
    total_time = time.time() - start_time
    
    # Process results
    successful_results = [r for r in results if r and r.success]
    
    print(f"\n🎯 Parallel Analysis Complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful models: {len(successful_results)}/{len(results)}")
    print(f"Parallel speedup: ~{n_workers}x theoretical")
    
    # Export results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_file = output_path / f"nebraska_parallel_{sample_size}_{timestamp}.csv"
    
    csv_data = []
    for result in results:
        if result is None:
            continue
            
        row = {
            'model': result.model_name,
            'success': result.success
        }
        
        if result.success:
            row.update({
                'aic': result.aic,
                'log_likelihood': result.log_likelihood,
                'n_parameters': result.n_parameters,
                'strategy_used': result.strategy_used,
                'fit_time': result.fit_time
            })
            
            for i, param in enumerate(result.parameters):
                row[f'param_{i+1}'] = param
        else:
            row.update({
                'aic': np.nan,
                'log_likelihood': np.nan,
                'error': result.error_message
            })
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    
    print(f"📊 Results exported to: {csv_file}")
    
    # Show top models
    if successful_results:
        print(f"\n📊 Top Models:")
        print("-" * 60)
        
        # Sort by AIC
        sorted_results = sorted(successful_results, key=lambda x: x.aic)
        
        for i, result in enumerate(sorted_results[:5]):
            delta_aic = result.aic - sorted_results[0].aic
            print(f"{i+1}. {result.model_name}: AIC={result.aic:.1f} (Δ{delta_aic:.1f})")
        
        # Check for performance validation
        unique_lls = set(round(r.log_likelihood, 6) for r in sorted_results)
        if len(unique_lls) == 1:
            print(f"\n⚠️  WARNING: All models have identical log-likelihood!")
        else:
            print(f"\n✅ SUCCESS: Models show different log-likelihoods")
            print(f"   Range: {min(r.log_likelihood for r in sorted_results):.4f} to {max(r.log_likelihood for r in sorted_results):.4f}")
    
    # Cleanup
    Path(temp_file).unlink()
    
    return results


def main():
    """Main entry point with performance optimization."""
    
    parser = argparse.ArgumentParser(description='Nebraska Analysis with Core Optimizations')
    parser.add_argument('--sample-size', type=int, default=1000, 
                       help='Number of individuals to analyze')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--mode', choices=['sequential', 'parallel'], default='parallel',
                       help='Analysis mode')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("🔬 Nebraska Analysis - Core Optimized Version")
    print("=" * 65)
    print(f"Sample size: {args.sample_size:,} individuals")
    print(f"Mode: {args.mode}")
    print(f"Workers: {args.workers or mp.cpu_count()}")
    
    if args.mode == 'parallel':
        results = run_parallel_analysis(
            sample_size=args.sample_size,
            n_workers=args.workers,
            output_dir=args.output_dir
        )
        return
    
    # Original sequential code below (keeping for comparison)
    print("🔬 Nebraska Analysis - Working Version with Parameter Export")
    print("=" * 65)
    
    # Configuration
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Realistic time estimates
    print("📊 Dataset Size Options - REALISTIC TIME ESTIMATES:")
    print("   1. Tiny (100 individuals) - ~3 minutes [RECOMMENDED TEST]")
    print("   2. Small (500 individuals) - ~20 minutes")
    print("   3. Medium (1K individuals) - ~1 hour")
    print("   4. Large (2K individuals) - ~3 hours") 
    print("   5. X-Large (5K individuals) - ~8-12 hours")
    print("   6. Custom size")
    print("\n⚠️  Performance scales roughly O(n²) with sample size")
    
    try:
        choice = input(f"\nSelect option (1-6) [default: 1]: ").strip()
        if not choice:
            choice = '1'
            
        size_map = {
            '1': 100,   # Test size
            '2': 500,   # Small  
            '3': 1000,  # Medium
            '4': 2000,  # Large
            '5': 5000,  # X-Large
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
        temp_file = f"temp_nebraska_working_{len(nebraska_data)}.csv"
        nebraska_data.to_csv(temp_file, index=False)
        
        print("\n🚀 Loading data into Pradel-JAX system...")
        data_context = pj.load_data(temp_file)
        print(f"   ✅ Loaded: {data_context.n_individuals:,} individuals, {data_context.n_occasions} occasions")
        
        # Define simple model set for testing
        phi_formulas = ["~1", "~1 + age_1"]
        f_formulas = ["~1", "~1 + gender"]
        
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
        
        print(f"\n⚡ Fitting {len(model_specifications)} models using optimization framework...")
        
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
                print(f"   ⏱️  Elapsed: {elapsed/60:.1f}min, Est. remaining: {remaining/60:.1f}min")
            
            result = fit_single_model_working(model, spec['formula'], data_context, spec['name'])
            results.append({'name': spec['name'], 'result': result})
            
            if result['success']:
                print(f"   ✅ Success - AIC: {result['aic']:.1f}, LL: {result['log_likelihood']:.2f}")
                print(f"      Strategy: {result.get('strategy_used', 'unknown')}, Time: {result.get('optimization_time', 0):.1f}s")
            else:
                print(f"   ❌ Failed: {result['error']}")
        
        fit_time = time.time() - fit_start
        
        # Export results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_name = f"nebraska_working_{sample_size}ind_{timestamp}"
        
        export_results_csv(results, f"{base_name}.csv")
        
        # Results summary
        print(f"\n🎯 Analysis Summary")
        print("=" * 50)
        print(f"Total fitting time: {fit_time/60:.1f} minutes")
        
        successful_results = [r for r in results if r['result']['success']]
        print(f"Successful models: {len(successful_results)}/{len(results)}")
        
        if successful_results:
            # Model comparison table
            print(f"\n📊 Model Results:")
            print("-" * 80)
            print(f"{'Model':<20} {'AIC':<10} {'ΔAIC':<8} {'LogLik':<12} {'K':<3} {'Strategy':<10}")
            print("-" * 80)
            
            # Sort by AIC
            model_stats = [(r['name'], r['result']) for r in successful_results]
            model_stats.sort(key=lambda x: x[1]['aic'])
            best_aic = model_stats[0][1]['aic']
            
            for name, result in model_stats:
                delta_aic = result['aic'] - best_aic
                strategy = result.get('strategy_used', 'unknown')[:8]
                print(f"{name:<20} {result['aic']:<10.1f} {delta_aic:<8.1f} "
                      f"{result['log_likelihood']:<12.2f} {result['n_parameters']:<3} {strategy:<10}")
            
            print("-" * 80)
            
            # Best model details
            best_name, best_result = model_stats[0]
            print(f"\n🏆 Best Model: {best_name}")
            print(f"   AIC: {best_result['aic']:.3f}")
            print(f"   Log-likelihood: {best_result['log_likelihood']:.3f}")
            print(f"   Parameters: {best_result['n_parameters']}")
            print(f"   Strategy: {best_result.get('strategy_used', 'unknown')}")
            
            # Show parameter estimates
            params = best_result['parameters']
            print(f"   Parameter estimates: [{', '.join([f'{p:.4f}' for p in params])}]")
            
            # Check if log-likelihoods are different (debugging the original issue)
            unique_lls = set(r[1]['log_likelihood'] for r in model_stats)
            if len(unique_lls) == 1:
                print(f"\n⚠️  WARNING: All models have identical log-likelihood ({list(unique_lls)[0]:.6f})")
                print("   This suggests a potential issue with model fitting.")
            else:
                print(f"\n✅ Models show different log-likelihoods (range: {min(unique_lls):.3f} to {max(unique_lls):.3f})")
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time/60:.1f} minutes")
        print(f"📊 Results exported to {base_name}.csv")
        
        # Performance metrics
        if successful_results:
            avg_time_per_model = fit_time / len(model_specifications)
            print(f"📈 Average time per model: {avg_time_per_model:.1f} seconds")
            
            # Extrapolate for larger sizes
            if sample_size < 5000:
                scaling_factor = (5000 / sample_size) ** 1.8  # Rough quadratic scaling
                estimated_5k_time = avg_time_per_model * scaling_factor
                print(f"🔮 Estimated time per model for 5K individuals: {estimated_5k_time/60:.1f} minutes")
                print(f"🔮 Estimated total time for 64 models on 5K: {(estimated_5k_time * 64)/3600:.1f} hours")
        
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