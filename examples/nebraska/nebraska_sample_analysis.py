#!/usr/bin/env python3
"""
Comprehensive Capture-Recapture Data Analysis Script
Randomly samples specified number of rows from capture-recapture encounter history data 
and fits all combinations of Pradel models with φ and f covariates, p(1).

Supports multiple datasets:
- Nebraska (NE): encounter_histories_ne_clean.csv
- South Dakota (SD): encounter_histories_sd_clean.csv
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from pradel_jax.optimization import create_model_specs_from_formulas, fit_models_parallel
from pradel_jax.optimization.parallel import ParallelModelSpec, fit_models_parallel as parallel_fit
from pradel_jax.formulas import create_simple_spec
import sys
import argparse
import time
import gc
from pathlib import Path
from itertools import combinations
import multiprocessing as mp

# Dataset configurations
DATASET_CONFIGS = {
    'nebraska': {
        'file': 'data/encounter_histories_ne_clean.csv',
        'name': 'Nebraska',
        'abbrev': 'NE',
        'covariates': ['gender', 'age', 'tier'],
        'age_column': 'age',  # Standardized age column
    },
    'south_dakota': {
        'file': 'data/encounter_histories_sd_clean.csv', 
        'name': 'South Dakota',
        'abbrev': 'SD',
        'covariates': ['gender', 'tier', 'age_2020'],
        'age_column': 'age_2020',  # Use 2020 age as proxy
    }
}

def detect_dataset(data_file_path):
    """Auto-detect which dataset is being used based on file path."""
    data_file_str = str(data_file_path).lower()
    
    if 'ne_clean' in data_file_str or 'nebraska' in data_file_str:
        return 'nebraska'
    elif 'sd_clean' in data_file_str or 'south_dakota' in data_file_str:
        return 'south_dakota'
    else:
        # Default fallback
        return 'nebraska'

def get_available_datasets():
    """Get list of available datasets."""
    available = []
    for dataset_key, config in DATASET_CONFIGS.items():
        if Path(config['file']).exists():
            available.append((dataset_key, config))
    return available

def generate_formula_combinations(covariates):
    """Generate all possible combinations of covariates for model formulas."""
    formulas = ["~1"]  # Always include intercept-only
    
    # Single covariate effects
    for cov in covariates:
        formulas.append(f"~1 + {cov}")
    
    # Two-way additive combinations
    for cov_pair in combinations(covariates, 2):
        formulas.append(f"~1 + {' + '.join(cov_pair)}")
    
    # Three-way additive combinations (if not too many)
    if len(covariates) <= 5:  # Limit to avoid overparameterization
        for cov_triple in combinations(covariates, 3):
            formulas.append(f"~1 + {' + '.join(cov_triple)}")
    
    return formulas

def main():
    """Run comprehensive Pradel model analysis on capture-recapture data."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Capture-Recapture Pradel Model Analysis')
    parser.add_argument('--dataset', '-d', type=str, choices=['nebraska', 'south_dakota', 'auto'], 
                       default='auto', help='Dataset to analyze (default: auto-detect)')
    parser.add_argument('--data-file', type=str, help='Custom data file path (overrides --dataset)')
    parser.add_argument('--sample-size', '-n', type=int, default=1000,
                       help='Number of individuals to sample (default: 1000, use 0 for full dataset)')
    parser.add_argument('--max-models', type=int, default=64,
                       help='Maximum number of models to fit (default: 64 for all combinations)')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Use parallel processing for large datasets')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for memory-efficient processing (default: 10000)')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List available datasets and exit')
    args = parser.parse_args()
    
    # Handle list datasets option
    if args.list_datasets:
        print("📊 Available Datasets:")
        print("=" * 40)
        available = get_available_datasets()
        if not available:
            print("❌ No datasets found in data/ directory")
            print("Expected files:")
            for key, config in DATASET_CONFIGS.items():
                print(f"  - {config['file']} ({config['name']})")
        else:
            for dataset_key, config in available:
                file_size = Path(config['file']).stat().st_size / (1024**2)  # MB
                print(f"✅ {config['name']} ({config['abbrev']})")
                print(f"   File: {config['file']}")
                print(f"   Size: {file_size:.1f} MB")
                print(f"   Covariates: {', '.join(config['covariates'])}")
                print()
        sys.exit(0)
    
    # Determine dataset and configuration
    if args.data_file:
        # Custom data file provided
        data_file = args.data_file
        dataset_key = detect_dataset(data_file)
        dataset_config = DATASET_CONFIGS[dataset_key]
        print(f"🔬 Custom Dataset Analysis: {data_file}")
    elif args.dataset == 'auto':
        # Auto-detect from available datasets
        available = get_available_datasets()
        if not available:
            print("❌ No datasets found! Use --list-datasets to see expected files.")
            sys.exit(1)
        dataset_key, dataset_config = available[0]  # Use first available
        data_file = dataset_config['file']
        print(f"🔬 Auto-detected: {dataset_config['name']} Dataset Analysis")
    else:
        # Specific dataset requested
        dataset_key = args.dataset
        dataset_config = DATASET_CONFIGS[dataset_key]
        data_file = dataset_config['file']
        print(f"🔬 {dataset_config['name']} Dataset Analysis")
    
    # Add sample size info
    if args.sample_size == 0:
        print("   📊 Mode: Full Dataset")
    else:
        print(f"   📊 Mode: Random Sample ({args.sample_size:,} individuals)")
    print("=" * 60)
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        print("Use --list-datasets to see available datasets.")
        sys.exit(1)
    
    try:
        # Load dataset with large-scale optimization
        print(f"📂 Loading data from: {data_file}")
        import pandas as pd  # Re-import to be sure
        
        if args.sample_size == 0:
            # Full dataset processing
            print("   Loading full dataset for large-scale analysis...")
            if args.chunk_size and args.chunk_size < 50000:
                # Use chunked loading for memory efficiency
                print(f"   Using chunked loading (chunk size: {args.chunk_size:,})...")
                chunk_iter = pd.read_csv(data_file, chunksize=args.chunk_size)
                chunks = list(chunk_iter)
                sampled_data = pd.concat(chunks, ignore_index=True)
                print(f"   Loaded {len(chunks)} chunks, total shape: {sampled_data.shape}")
            else:
                # Direct loading for sufficient memory
                sampled_data = pd.read_csv(data_file)
                print(f"   Full dataset shape: {sampled_data.shape}")
            sample_size = len(sampled_data)
            print(f"🚀 Processing full dataset: {sample_size:,} individuals")
        else:
            # Sample-based processing
            full_data = pd.read_csv(data_file)
            print(f"   Full dataset shape: {full_data.shape}")
            
            # Random sample of specified size
            sample_size = min(args.sample_size, len(full_data))
            print(f"🎲 Randomly sampling {sample_size:,} rows...")
            
            # Set random seed for reproducibility
            np.random.seed(42)
            sampled_data = full_data.sample(n=sample_size, random_state=42)
            print(f"   Sample shape: {sampled_data.shape}")
        
        # Save sampled data to temporary file for loading
        print("🔧 Converting to pradel-jax format...")
        temp_file = f"temp_{dataset_config['abbrev'].lower()}_sample.csv"
        sampled_data.to_csv(temp_file, index=False)
        
        # CRITICAL FIX: Preprocess covariates before loading
        print("🔧 Preprocessing covariates for proper modeling...")
        
        # Fix gender coding and missing values
        if 'gender' in sampled_data.columns:
            # Convert 1.0->Male, 2.0->Female, NaN->Male (default)
            sampled_data['gender'] = sampled_data['gender'].fillna(1.0)  # Fill missing with Male
            sampled_data['gender'] = sampled_data['gender'].map({1.0: 'Male', 2.0: 'Female'})
            print(f"   ✅ Gender: {sampled_data['gender'].value_counts().to_dict()}")
        
        # Standardize age for numerical stability (use dataset-specific age column)
        age_column = dataset_config['age_column']
        if age_column in sampled_data.columns:
            original_age = sampled_data[age_column].copy()
            # Create standardized 'age' column for modeling
            sampled_data['age'] = (sampled_data[age_column] - sampled_data[age_column].mean()) / sampled_data[age_column].std()
            print(f"   ✅ Age ({age_column}): standardized (mean={original_age.mean():.1f}, std={original_age.std():.1f})")
        
        # Simplify tier_history to meaningful categories
        if 'tier_history' in sampled_data.columns:
            # Create simple tier categories based on tier_history patterns
            def categorize_tier(tier_val):
                if pd.isna(tier_val):
                    return 'Unknown'
                tier_str = str(int(tier_val))
                if tier_str.startswith('1'):
                    return 'Tier1'
                elif tier_str.startswith('2'):
                    return 'Tier2'  
                elif len(tier_str) >= 8:  # Long codes like 100000000
                    return 'LongTerm'
                else:
                    return 'Other'
            
            sampled_data['tier'] = sampled_data['tier_history'].apply(categorize_tier)
            print(f"   ✅ Tier: {sampled_data['tier'].value_counts().to_dict()}")
            # Keep tier_history as backup but use tier for modeling
        
        # Save preprocessed data
        sampled_data.to_csv(temp_file, index=False)
        
        # Force use of GenericFormatAdapter to avoid issues with 'ch' column
        # Use Y-columns instead (Y2016-Y2024 are more reliable)
        from pradel_jax.data.adapters import GenericFormatAdapter
        generic_adapter = GenericFormatAdapter()
        data_context = pj.load_data(temp_file, adapter=generic_adapter)
        
        print("   Data summary:")
        print(f"   - Number of individuals: {data_context.n_individuals}")
        print(f"   - Number of occasions: {data_context.n_occasions}")
        if hasattr(data_context, 'covariates') and data_context.covariates:
            # Show only the main covariates (not metadata)
            covariate_keys = list(data_context.covariates.keys())
            main_covariates = [k for k in covariate_keys if not k.endswith('_categories') and not k.endswith('_is_categorical')]
            print(f"   - Available covariates: {main_covariates[:10]}")  # Show first 10
            if len(main_covariates) > 10:
                print(f"     (and {len(main_covariates)-10} more...)")
        
        # Create model specifications for fitting
        print("📊 Setting up comprehensive Pradel model set...")
        
        # Get available covariates (excluding metadata)
        main_covariates = [k for k in data_context.covariates.keys() 
                          if not k.endswith('_categories') and not k.endswith('_is_categorical')]
        
        # Define target covariates for modeling based on dataset configuration
        target_covariates = []
        for cov in dataset_config['covariates']:
            if cov in main_covariates:
                target_covariates.append(cov)
        
        # Handle dataset-specific covariate fallbacks
        if 'tier' not in target_covariates and 'tier_history' in main_covariates:
            target_covariates.append('tier_history')
        
        # For South Dakota: add age covariates if available
        if dataset_key == 'south_dakota':
            age_vars = [col for col in main_covariates if col.startswith('age_') and col != age_column]
            for age_var in age_vars[:2]:  # Limit to prevent overparameterization
                if age_var not in target_covariates:
                    target_covariates.append(age_var)
            
        print(f"   Target covariates for modeling: {target_covariates}")
        
        # Generate all formula combinations for survival (φ) and recruitment (f)
        phi_formulas = generate_formula_combinations(target_covariates)
        f_formulas = generate_formula_combinations(target_covariates)
        p_formulas = ["~1"]  # Keep detection constant for all models
        
        # Limit total models to avoid excessive computation
        total_models = len(phi_formulas) * len(f_formulas) * len(p_formulas)
        print(f"   Potential models: φ({len(phi_formulas)}) × f({len(f_formulas)}) × p({len(p_formulas)}) = {total_models}")
        
        if total_models > args.max_models:
            print(f"   ⚠️  Too many models ({total_models}), limiting to {args.max_models}")
            # Prioritize simpler models first
            phi_formulas = phi_formulas[:int(np.sqrt(args.max_models))]
            f_formulas = f_formulas[:int(np.sqrt(args.max_models))]
            total_models = len(phi_formulas) * len(f_formulas)
            print(f"   Reduced to: φ({len(phi_formulas)}) × f({len(f_formulas)}) = {total_models}")
        
        print(f"\n   📋 Survival (φ) formulas ({len(phi_formulas)}):")
        for i, formula in enumerate(phi_formulas, 1):
            print(f"     {i}. φ{formula}")
        
        print(f"\n   📋 Recruitment (f) formulas ({len(f_formulas)}):")
        for i, formula in enumerate(f_formulas, 1):
            print(f"     {i}. f{formula}")
        
        print(f"\n   📋 Detection (p) formula: p~1 (constant)")
        
        if target_covariates:
            print(f"\n   Available covariates: {', '.join(target_covariates)}")
        else:
            print("   No target covariates detected - using intercept-only models")
        
        # Create model specifications using the new API
        from pradel_jax.optimization import optimize_model
        from pradel_jax.models import PradelModel
        
        # Configure optimization for dataset size
        if sample_size >= 50000:
            print("⚡ Fitting models using large-scale optimization (50K+ individuals)...")
            use_large_scale = True
        elif sample_size >= 10000 or args.parallel:
            print("⚡ Fitting models using parallel optimization...")
            use_parallel = True
        else:
            print("⚡ Fitting models using standard optimization...")
            use_large_scale = False
            use_parallel = False
        
        # Create results list to store all model fitting results
        results = []
        total_combinations = len(phi_formulas) * len(f_formulas) * len(p_formulas)
        
        # Performance and memory optimization
        start_time = time.time()
        if sample_size >= 10000:
            print(f"   🧠 Large dataset detected ({sample_size:,} individuals)")
            print(f"   💾 Enabling memory optimization and garbage collection")
            gc.collect()  # Clean memory before starting
        
        # Choose optimization approach based on dataset size
        if 'use_parallel' in locals() and use_parallel and sample_size >= 5000:
            # Parallel processing for medium to large datasets
            print(f"   🚀 Using parallel processing ({mp.cpu_count()} cores available)")
            
            # Create model specifications for parallel processing
            model_specs = []
            model_counter = 0
            for i, phi_formula in enumerate(phi_formulas):
                for j, f_formula in enumerate(f_formulas):
                    for k, p_formula in enumerate(p_formulas):
                        model_name = f"phi{phi_formula}_p{p_formula}_f{f_formula}"
                        formula_spec = create_simple_spec(
                            phi=phi_formula,
                            p=p_formula,
                            f=f_formula
                        )
                        model_specs.append(ParallelModelSpec(
                            name=model_name,
                            formula_spec=formula_spec,
                            index=model_counter
                        ))
                        model_counter += 1
            
            # Run parallel optimization
            try:
                from pradel_jax.optimization.parallel import fit_models_parallel
                parallel_results = fit_models_parallel(
                    model_specs=model_specs,
                    data_context=data_context,
                    max_workers=min(mp.cpu_count(), 8),  # Limit to 8 cores max
                    timeout_per_model=300  # 5 minutes per model
                )
                
                # Convert parallel results to standard format
                for parallel_result in parallel_results:
                    if parallel_result.success:
                        model_result = type('ModelResult', (), {
                            'success': True,
                            'model_name': parallel_result.model_name,
                            'log_likelihood': parallel_result.log_likelihood,
                            'parameters': parallel_result.parameters,
                            'n_parameters': parallel_result.n_parameters,
                            'strategy_used': parallel_result.strategy_used,
                            'aic': parallel_result.aic,
                            'lambda_mean': None
                        })()
                    else:
                        model_result = type('ModelResult', (), {
                            'success': False,
                            'model_name': parallel_result.model_name,
                            'error_message': parallel_result.error_message
                        })()
                    results.append(model_result)
                    
                print(f"   ✅ Completed {len(results)} models using parallel processing")
                    
            except Exception as e:
                print(f"   ⚠️  Parallel processing failed: {e}")
                print(f"   🔄 Falling back to sequential processing...")
                use_parallel = False
        
        # Sequential processing (fallback or for smaller datasets)
        if not ('use_parallel' in locals() and use_parallel) or len(results) == 0:
            for i, phi_formula in enumerate(phi_formulas):
                for j, f_formula in enumerate(f_formulas):
                    for k, p_formula in enumerate(p_formulas):
                        model_name = f"phi{phi_formula}_p{p_formula}_f{f_formula}"
                        
                        try:
                            # Create formula spec
                            formula_spec = create_simple_spec(
                                phi=phi_formula,
                                p=p_formula,
                                f=f_formula
                            )
                            
                            # Create and fit model
                            model = PradelModel()
                            design_matrices = model.build_design_matrices(formula_spec, data_context)
                            
                            # Define objective function
                            def objective(params):
                                return -model.log_likelihood(params, data_context, design_matrices)
                            
                            # Configure optimization for large datasets
                            optimization_kwargs = {
                                'objective_function': objective,
                                'initial_parameters': model.get_initial_parameters(data_context, design_matrices),
                                'context': data_context,
                                'bounds': model.get_parameter_bounds(data_context, design_matrices)
                            }
                            
                            # Add large-scale config for very large datasets
                            if sample_size >= 50000:
                                optimization_kwargs['config_overrides'] = {
                                    'batch_size': min(10000, sample_size // 10),
                                    'num_workers': min(4, mp.cpu_count())
                                }
                            
                            # Optimize
                            result = optimize_model(**optimization_kwargs)
                        
                            # Convert to compatible result format with statistical inference
                            if result.success:
                                # Generate parameter names from formula specification
                                from pradel_jax.optimization.statistical_inference import generate_parameter_names
                                param_names = generate_parameter_names(formula_spec, data_context)
                                
                                # Set up statistical inference (standard errors, confidence intervals)
                                result.result.set_statistical_info(
                                    param_names, 
                                    data_context.n_individuals, 
                                    objective_function=objective
                                )
                                
                                # Create enhanced model result with statistical inference
                                model_result = type('ModelResult', (), {
                                    'success': True,
                                    'model_name': model_name,
                                    'log_likelihood': -result.result.fun,
                                    'parameters': result.result.x,
                                    'n_parameters': len(result.result.x),
                                    'strategy_used': result.strategy_used,
                                    'aic': result.result.aic,  # Use computed AIC from statistical inference
                                    'bic': result.result.bic,  # BIC available too
                                    'parameter_names': param_names,
                                    'standard_errors': result.result.standard_errors,
                                    'confidence_intervals': result.result.confidence_intervals,
                                    'parameter_summary': result.result.get_parameter_summary(),
                                    'lambda_mean': None  # Would need to calculate from parameters
                                })()
                            else:
                                model_result = type('ModelResult', (), {
                                    'success': False,
                                    'model_name': model_name,
                                    'error_message': f"Optimization failed: {result.message}"
                                })()
                            
                            results.append(model_result)
                            
                            # Progress indicator with performance info
                            completed = i * len(f_formulas) * len(p_formulas) + j * len(p_formulas) + k + 1
                            elapsed_time = time.time() - start_time
                            avg_time_per_model = elapsed_time / completed
                            remaining_models = total_combinations - completed
                            estimated_remaining = avg_time_per_model * remaining_models
                            
                            status_icon = '✅' if result.success else '❌'
                            print(f"   Progress: {completed}/{total_combinations} - {model_name} {status_icon}")
                            
                            # Show time estimates for large datasets
                            if sample_size >= 10000 and completed % 5 == 0:  # Every 5 models
                                print(f"      ⏱️  Elapsed: {elapsed_time/60:.1f}min, Est. remaining: {estimated_remaining/60:.1f}min")
                            
                            # Memory cleanup for large datasets
                            if sample_size >= 50000 and completed % 10 == 0:  # Every 10 models
                                gc.collect()
                                print(f"      🧹 Memory cleanup performed")
                        
                        except Exception as e:
                            completed = i * len(f_formulas) * len(p_formulas) + j * len(p_formulas) + k + 1
                            print(f"   Progress: {completed}/{total_combinations} - {model_name} ❌ (Error: {str(e)[:50]})")
                            error_result = type('ModelResult', (), {
                                'success': False,
                                'model_name': model_name,
                                'error_message': str(e)
                            })()
                            results.append(error_result)
        
        # Display results for all models
        print(f"\n🎯 Model Results ({len(results)} models fitted)")
        print("=" * 60)
        
        successful_results = [r for r in results if r and r.success]
        if successful_results:
            # Sort by AIC (lower is better)
            successful_results.sort(key=lambda x: x.aic)
            
            print(f"✅ {len(successful_results)} model(s) converged successfully:")
            print()
            
            for i, result in enumerate(successful_results):
                rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                
                print(f"{rank_symbol} {result.model_name}")
                print(f"   Log-likelihood: {result.log_likelihood:.3f}")
                print(f"   AIC: {result.aic:.3f}")
                if hasattr(result, 'bic') and result.bic is not None:
                    print(f"   BIC: {result.bic:.3f}")
                print(f"   Parameters: {result.n_parameters}")
                print(f"   Strategy: {result.strategy_used}")
                
                # Statistical inference information
                if hasattr(result, 'standard_errors') and result.standard_errors is not None:
                    se_range = f"[{np.min(result.standard_errors):.4f}, {np.max(result.standard_errors):.4f}]"
                    print(f"   Standard Error Range: {se_range}")
                
                if hasattr(result, 'parameter_summary') and result.parameter_summary is not None:
                    n_significant = sum(1 for param_info in result.parameter_summary.values() 
                                      if param_info.get('p_value', 1.0) < 0.05)
                    print(f"   Significant Parameters (p<0.05): {n_significant}/{result.n_parameters}")
                
                # Population growth rate information
                if hasattr(result, 'lambda_mean') and result.lambda_mean is not None:
                    print(f"   Lambda (growth rate): {result.lambda_mean:.4f}")
                
                print()  # Blank line between models
            
            # Best model summary with detailed statistical inference
            best_model = successful_results[0]
            print("🏆 Best Model Detailed Summary:")
            print(f"   Model: {best_model.model_name}")
            print(f"   AIC: {best_model.aic:.3f}")
            if hasattr(best_model, 'bic') and best_model.bic is not None:
                print(f"   BIC: {best_model.bic:.3f}")
            print(f"   Log-likelihood: {best_model.log_likelihood:.3f}")
            print(f"   Strategy: {best_model.strategy_used}")
            
            # Detailed parameter table for best model
            if hasattr(best_model, 'parameter_summary') and best_model.parameter_summary is not None:
                print(f"\n   📋 Parameter Estimates:")
                print(f"   {'Parameter':<15} {'Estimate':<10} {'SE':<10} {'95% CI':<20} {'p-value':<8}")
                print(f"   {'-'*70}")
                
                for param_name, info in best_model.parameter_summary.items():
                    estimate = info.get('estimate', 0)
                    se = info.get('std_error', None)
                    p_val = info.get('p_value', None)
                    
                    # Find 95% CI
                    ci_lower = info.get('ci_lower_95%', None)
                    ci_upper = info.get('ci_upper_95%', None)
                    
                    # Format values
                    est_str = f"{estimate:8.4f}"
                    se_str = f"{se:8.4f}" if se is not None else "   --   "
                    ci_str = f"({ci_lower:6.3f}, {ci_upper:6.3f})" if ci_lower is not None else "      --       "
                    p_str = f"{p_val:.4f}" if p_val is not None else "  --  "
                    
                    # Add significance indicator
                    sig_indicator = " *" if p_val is not None and p_val < 0.05 else ""
                    
                    print(f"   {param_name:<15} {est_str:<10} {se_str:<10} {ci_str:<20} {p_str:<8}{sig_indicator}")
                
                print(f"   {'-'*70}")
                print(f"   * p < 0.05")
            else:
                print(f"   Parameter estimates: {[f'{p:.4f}' for p in best_model.parameters]}")
                
            # Model comparison summary if multiple models
            if len(successful_results) > 1:
                print(f"\n   📊 Model Support:")
                second_best = successful_results[1]
                delta_aic = second_best.aic - best_model.aic
                print(f"   Δ AIC vs next best: {delta_aic:.3f}")
                if delta_aic > 2:
                    print(f"   Strong evidence for best model (Δ AIC > 2)")
                elif delta_aic > 4:
                    print(f"   Very strong evidence for best model (Δ AIC > 4)")
                elif delta_aic > 10:
                    print(f"   Overwhelming evidence for best model (Δ AIC > 10)")
            
        else:
            print("❌ No models converged successfully")
            if results:
                for result in results:
                    if hasattr(result, 'error_message') and result.error_message:
                        print(f"   Error: {result.error_message}")
            else:
                print("   No result objects returned")
        
        print(f"\n✅ Analysis completed successfully!")
        
        # Comprehensive results export package
        print(f"\n📄 Generating Comprehensive Analysis Package...")
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Full results export (MARK-compatible CSV)
            print(f"   📊 Creating comprehensive results table...")
            dataset_prefix = dataset_config['abbrev'].lower()
            full_results_file = f"{dataset_prefix}_full_results_{sample_size}ind_{timestamp}.csv"
            
            # Create comprehensive results DataFrame with statistical inference
            import pandas as pd
            results_data = []
            for result in results:
                if hasattr(result, 'success') and result.success:
                    # Base model information
                    result_dict = {
                        'model_name': result.model_name,
                        'log_likelihood': result.log_likelihood,
                        'aic': result.aic,
                        'bic': getattr(result, 'bic', None),
                        'n_parameters': result.n_parameters,
                        'strategy_used': result.strategy_used,
                        'success': result.success
                    }
                    
                    # Statistical inference information
                    if hasattr(result, 'standard_errors') and result.standard_errors is not None:
                        result_dict['se_min'] = np.min(result.standard_errors)
                        result_dict['se_max'] = np.max(result.standard_errors)
                        result_dict['se_mean'] = np.mean(result.standard_errors)
                    
                    if hasattr(result, 'parameter_summary') and result.parameter_summary is not None:
                        # Count significant parameters
                        n_significant = sum(1 for info in result.parameter_summary.values() 
                                          if info.get('p_value', 1.0) < 0.05)
                        result_dict['n_significant_params'] = n_significant
                        result_dict['pct_significant'] = n_significant / result.n_parameters * 100
                        
                        # Add individual parameter estimates with names
                        if hasattr(result, 'parameter_names') and result.parameter_names is not None:
                            for i, (param_name, param_value) in enumerate(zip(result.parameter_names, result.parameters)):
                                result_dict[f'param_{param_name}'] = param_value
                                if param_name in result.parameter_summary:
                                    param_info = result.parameter_summary[param_name]
                                    result_dict[f'se_{param_name}'] = param_info.get('std_error', None)
                                    result_dict[f'pval_{param_name}'] = param_info.get('p_value', None)
                                    result_dict[f'ci_lower_{param_name}'] = param_info.get('ci_lower_95%', None)
                                    result_dict[f'ci_upper_{param_name}'] = param_info.get('ci_upper_95%', None)
                    
                    results_data.append(result_dict)
                else:
                    results_data.append({
                        'model_name': getattr(result, 'model_name', 'Unknown'),
                        'log_likelihood': None,
                        'aic': None,
                        'bic': None,
                        'n_parameters': None,
                        'strategy_used': None,
                        'success': False,
                        'error': getattr(result, 'error_message', 'Unknown error')
                    })
            
            export_df = pd.DataFrame(results_data)
            export_df.to_csv(full_results_file, index=False)
            
            # 2. Model comparison table (publication-ready) using statistical inference framework
            print(f"   🏆 Creating model comparison table with statistical inference...")
            successful_df = export_df[export_df['success'] == True].copy()
            if len(successful_df) > 0:
                # Use the comprehensive model comparison from statistical inference framework
                from pradel_jax.optimization.statistical_inference import compare_models, print_model_comparison
                
                # Create model results dictionary for comparison
                model_results_for_comparison = {}
                for result in results:
                    if hasattr(result, 'success') and result.success:
                        model_results_for_comparison[result.model_name] = result
                
                # Generate comprehensive model comparison
                if len(model_results_for_comparison) > 1:
                    comparison_result = compare_models(model_results_for_comparison)
                    
                    # Create enhanced comparison DataFrame
                    successful_df = successful_df.sort_values('aic')
                    successful_df['delta_aic'] = successful_df['aic'] - successful_df['aic'].min()
                    
                    # Add BIC comparisons if available
                    if 'bic' in successful_df.columns and successful_df['bic'].notna().any():
                        successful_df['delta_bic'] = successful_df['bic'] - successful_df['bic'].min()
                    
                    # AIC weights and evidence ratios
                    successful_df['aic_weight'] = np.exp(-0.5 * successful_df['delta_aic'])
                    successful_df['aic_weight'] = successful_df['aic_weight'] / successful_df['aic_weight'].sum()
                    successful_df['substantial_support'] = successful_df['delta_aic'] <= 2.0
                    successful_df['strong_support'] = successful_df['delta_aic'] <= 4.0
                    successful_df['very_strong_support'] = successful_df['delta_aic'] <= 7.0
                    
                    # Evidence ratio vs best model
                    if len(successful_df) > 1:
                        best_weight = successful_df.iloc[0]['aic_weight']
                        successful_df['evidence_ratio'] = best_weight / successful_df['aic_weight']
                    
                    comparison_df = successful_df
                else:
                    comparison_df = successful_df
                
                comparison_file = f"{dataset_prefix}_model_comparison_{sample_size}ind_{timestamp}.csv"
                comparison_df.to_csv(comparison_file, index=False)
            else:
                import pandas as pd
                comparison_df = pd.DataFrame()  # Empty comparison if no successful models
            
            # 3. Detailed parameter tables with statistical inference
            print(f"   📋 Creating comprehensive parameter analysis...")
            if len(comparison_df) > 0:
                # A. Model selection summary
                model_cols = ['model_name', 'aic', 'bic', 'n_parameters', 'delta_aic', 'aic_weight', 'evidence_ratio', 'substantial_support', 'strong_support', 'n_significant_params', 'pct_significant']
                available_model_cols = [col for col in model_cols if col in comparison_df.columns]
                model_summary = comparison_df[available_model_cols].copy()
                model_summary_file = f"{dataset_prefix}_model_selection_{sample_size}ind_{timestamp}.csv"
                model_summary.to_csv(model_summary_file, index=False)
                
                # B. Best model parameter details
                if len(results) > 0 and hasattr(results[0], 'parameter_summary'):
                    best_result = None
                    for result in results:
                        if hasattr(result, 'success') and result.success and hasattr(result, 'parameter_summary') and result.parameter_summary:
                            best_result = result
                            break
                    
                    if best_result:
                        param_details_data = []
                        for param_name, param_info in best_result.parameter_summary.items():
                            param_details_data.append({
                                'parameter': param_name,
                                'estimate': param_info.get('estimate', None),
                                'std_error': param_info.get('std_error', None),
                                'z_score': param_info.get('z_score', None),
                                'p_value': param_info.get('p_value', None),
                                'ci_lower_95': param_info.get('ci_lower_95%', None),
                                'ci_upper_95': param_info.get('ci_upper_95%', None),
                                'significant_05': param_info.get('p_value', 1.0) < 0.05,
                                'significant_01': param_info.get('p_value', 1.0) < 0.01,
                                'model_name': best_result.model_name
                            })
                        
                        param_details_df = pd.DataFrame(param_details_data)
                        param_details_file = f"{dataset_prefix}_best_model_parameters_{sample_size}ind_{timestamp}.csv"
                        param_details_df.to_csv(param_details_file, index=False)
                    else:
                        param_details_file = None
                else:
                    param_details_file = None
                
                # C. Legacy parameter summary for compatibility
                param_cols = ['model_name', 'aic', 'n_parameters', 'delta_aic', 'aic_weight', 'substantial_support']
                available_cols = [col for col in param_cols if col in comparison_df.columns]
                param_summary = comparison_df[available_cols].copy()
                param_summary_file = f"{dataset_prefix}_parameters_{sample_size}ind_{timestamp}.csv"
                param_summary.to_csv(param_summary_file, index=False)
            else:
                model_summary_file = None
                param_details_file = None
                param_summary_file = None
            
            # 4. Print enhanced analysis summary
            print(f"\n📊 {dataset_config['name']} Capture-Recapture Analysis Summary")
            print(f"=" * 60)
            print(f"Dataset: {sample_size} individuals, {data_context.n_occasions} occasions")
            print(f"Analysis completed: {timestamp}")
            print(f"Models fitted: {len(results)} ({len([r for r in results if r.success])} successful)")
            
            if len(comparison_df) > 0:
                print(f"\n🏆 Model Selection Results:")
                best_model = comparison_df.iloc[0]
                print(f"   Best Model: {best_model['model_name']}")
                print(f"   AIC: {best_model['aic']:.3f}")
                print(f"   AIC Weight: {best_model['aic_weight']:.3f}")
                if len(comparison_df) > 1:
                    evidence_ratio = best_model['aic_weight'] / comparison_df.iloc[1]['aic_weight']
                    print(f"   Evidence Ratio: {evidence_ratio:.1f}x better than next model")
                
                # Model support summary
                substantial_models = comparison_df[comparison_df['substantial_support'] == True]
                print(f"\n🎯 Model Support Summary:")
                print(f"   Models with substantial support (ΔAIC ≤ 2): {len(substantial_models)}")
                print(f"   Top model cumulative weight: {comparison_df['aic_weight'].iloc[0]:.3f}")
            else:
                print(f"\n❌ No models converged successfully - no model comparison available")
                
            # 5. Files generated summary with statistical inference
            print(f"\n📁 Generated Files with Statistical Inference:")
            print(f"   📊 Full Results: {full_results_file}")
            print(f"       • Complete parameter estimates, standard errors, confidence intervals")
            print(f"       • Statistical significance tests and p-values")
            print(f"       • All optimization metadata and diagnostics")
            
            if len(comparison_df) > 0:
                print(f"   🏆 Model Selection: {model_summary_file}")
                print(f"       • AIC/BIC rankings with evidence ratios")
                print(f"       • Statistical support indicators and weights")
                print(f"       • Parameter significance summaries")
                
                print(f"   🏆 Model Comparison: {comparison_file}")
                print(f"       • Detailed model comparison with statistical tests")
                print(f"       • Publication-ready table with all criteria")
                
                if param_details_file:
                    print(f"   📋 Best Model Parameters: {param_details_file}")
                    print(f"       • Detailed parameter table with standard errors")
                    print(f"       • 95% confidence intervals and significance tests")
                    print(f"       • Z-scores and p-values for hypothesis testing")
                
                if param_summary_file:
                    print(f"   📋 Parameter Summary: {param_summary_file}")
                    print(f"       • Quick reference parameter table (legacy format)")
            else:
                print(f"   ⚠️  No statistical inference files generated (no successful fits)")
            
            # Performance Summary
            total_time = time.time() - start_time
            successful_models = len([r for r in results if r.success])
            print(f"\n📊 Performance Summary:")
            print(f"   ⏱️  Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
            print(f"   💾 Dataset size: {sample_size:,} individuals, {data_context.n_occasions} occasions")
            print(f"   🏆 Models fitted: {len(results)} total, {successful_models} successful ({successful_models/len(results)*100:.1f}%)")
            print(f"   ⚡ Average time per model: {total_time/len(results):.1f} seconds")
            if sample_size >= 10000:
                individuals_per_second = sample_size * len(results) / total_time
                print(f"   🚀 Processing rate: {individuals_per_second:,.0f} individual-models per second")
            
            print(f"\n✅ Complete analysis package generated!")
            print(f"🔬 Ready for publication, further analysis, or RMark comparison")
            
        except Exception as e:
            print(f"⚠️  Export package generation failed: {e}")
            print(f"📄 Basic results still available in result objects")
            import traceback
            traceback.print_exc()
        
        # Clean up temporary file
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Provide helpful debugging information
        print("\n🔍 Troubleshooting tips:")
        print("1. Check that data file contains encounter history data")
        print("2. Verify data format matches expected pradel-jax input")
        print("3. Try with a smaller sample size")
        print("4. Check data for missing values or formatting issues")
        
        sys.exit(1)

if __name__ == "__main__":
    main()