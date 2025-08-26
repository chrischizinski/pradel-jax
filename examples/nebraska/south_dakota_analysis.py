#!/usr/bin/env python3
"""
Comprehensive Nebraska Data Analysis Script
Randomly samples specified number of rows from Nebraska encounter history data 
and fits all combinations of Pradel models with φ and f covariates, p(1).
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
    """Run comprehensive Pradel model analysis on a random sample of Nebraska data."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Nebraska Pradel Model Analysis')
    parser.add_argument('--sample-size', '-n', type=int, default=1000,
                       help='Number of individuals to sample (default: 1000, use 0 for full dataset)')
    parser.add_argument('--max-models', type=int, default=64,
                       help='Maximum number of models to fit (default: 64 for all combinations)')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Use parallel processing for large datasets')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for memory-efficient processing (default: 10000)')
    args = parser.parse_args()
    
    if args.sample_size == 0:
        print("🔬 South Dakota Capture-Recapture Analysis - Full Dataset")
    else:
        print("🔬 South Dakota Capture-Recapture Analysis - Random Sample")
    print("=" * 60)
    
    # Data file path (relative to project root)
    data_file = "data/encounter_histories_sd_clean.csv"
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        print("Available data files:")
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.glob("*.csv"):
                print(f"  - {f}")
        else:
            print("Data directory not found. Make sure you're running from examples/nebraska/")
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
        temp_file = "temp_south_dakota_sample.csv"
        sampled_data.to_csv(temp_file, index=False)
        
        # CRITICAL FIX: Preprocess covariates before loading
        print("🔧 Preprocessing covariates for proper modeling...")
        
        # Fix gender coding and missing values
        if 'gender' in sampled_data.columns:
            # Convert 1.0->Male, 2.0->Female, NaN->Male (default)
            sampled_data['gender'] = sampled_data['gender'].fillna(1.0)  # Fill missing with Male
            sampled_data['gender'] = sampled_data['gender'].map({1.0: 'Male', 2.0: 'Female'})
            print(f"   ✅ Gender: {sampled_data['gender'].value_counts().to_dict()}")
        
        # Standardize age for numerical stability
        if 'age' in sampled_data.columns:
            original_age = sampled_data['age'].copy()
            sampled_data['age'] = (sampled_data['age'] - sampled_data['age'].mean()) / sampled_data['age'].std()
            print(f"   ✅ Age: standardized (mean={original_age.mean():.1f}, std={original_age.std():.1f})")
        
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
        
        # Define target covariates for modeling (including tier)
        target_covariates = []
        for cov in ['gender', 'age', 'tier']:
            if cov in main_covariates:
                target_covariates.append(cov)
        
        # Add tier_history if tier is not available but tier_history is
        if 'tier' not in target_covariates and 'tier_history' in main_covariates:
            target_covariates.append('tier_history')
            
        # For South Dakota: use age_2020 as proxy for age if age is not available
        if 'age' not in target_covariates and 'age_2020' in main_covariates:
            target_covariates.append('age_2020')
            
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
                        
                            # Convert to compatible result format
                            if result.success:
                                model_result = type('ModelResult', (), {
                                    'success': True,
                                    'model_name': model_name,
                                    'log_likelihood': -result.result.fun,
                                'parameters': result.result.x,
                                'n_parameters': len(result.result.x),
                                'strategy_used': result.strategy_used,
                                'aic': 2 * len(result.result.x) + 2 * result.result.fun,
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
                print(f"   Parameters: {result.n_parameters}")
                print(f"   Strategy: {result.strategy_used}")
                
                # Population growth rate information
                if hasattr(result, 'lambda_mean') and result.lambda_mean is not None:
                    print(f"   Lambda (growth rate): {result.lambda_mean:.4f}")
                
                print()  # Blank line between models
            
            # Best model summary
            best_model = successful_results[0]
            print("🏆 Best Model Summary:")
            print(f"   Model: {best_model.model_name}")
            print(f"   AIC: {best_model.aic:.3f}")
            print(f"   Parameter estimates: {[f'{p:.4f}' for p in best_model.parameters]}")
            
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
            full_results_file = f"south_dakota_full_results_{sample_size}ind_{timestamp}.csv"
            
            # Create results DataFrame manually since the old export API may not work
            import pandas as pd
            results_data = []
            for result in results:
                if hasattr(result, 'success') and result.success:
                    results_data.append({
                        'model_name': result.model_name,
                        'log_likelihood': result.log_likelihood,
                        'aic': result.aic,
                        'n_parameters': result.n_parameters,
                        'strategy_used': result.strategy_used,
                        'success': result.success
                    })
                else:
                    results_data.append({
                        'model_name': getattr(result, 'model_name', 'Unknown'),
                        'log_likelihood': None,
                        'aic': None,
                        'n_parameters': None,
                        'strategy_used': None,
                        'success': False,
                        'error': getattr(result, 'error_message', 'Unknown error')
                    })
            
            export_df = pd.DataFrame(results_data)
            export_df.to_csv(full_results_file, index=False)
            
            # 2. Model comparison table (publication-ready)
            print(f"   🏆 Creating model comparison table...")
            successful_df = export_df[export_df['success'] == True].copy()
            if len(successful_df) > 0:
                successful_df = successful_df.sort_values('aic')
                successful_df['delta_aic'] = successful_df['aic'] - successful_df['aic'].min()
                successful_df['aic_weight'] = np.exp(-0.5 * successful_df['delta_aic'])
                successful_df['aic_weight'] = successful_df['aic_weight'] / successful_df['aic_weight'].sum()
                successful_df['substantial_support'] = successful_df['delta_aic'] <= 2.0
                
                comparison_df = successful_df
                comparison_file = f"south_dakota_model_comparison_{sample_size}ind_{timestamp}.csv"
                comparison_df.to_csv(comparison_file, index=False)
            else:
                import pandas as pd
                comparison_df = pd.DataFrame()  # Empty comparison if no successful models
            
            # 3. Parameter summary (simplified for quick reference)
            print(f"   📋 Creating parameter summary...")
            if len(comparison_df) > 0:
                param_cols = ['model_name', 'aic', 'n_parameters', 'delta_aic', 'aic_weight', 'substantial_support']
                available_cols = [col for col in param_cols if col in comparison_df.columns]
                param_summary = comparison_df[available_cols].copy()
                param_summary_file = f"south_dakota_parameters_{sample_size}ind_{timestamp}.csv"
                param_summary.to_csv(param_summary_file, index=False)
            else:
                param_summary_file = None
            
            # 4. Print enhanced analysis summary
            print(f"\n📊 South Dakota Capture-Recapture Analysis Summary")
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
                
            # 5. Files generated summary
            print(f"\n📁 Generated Files:")
            print(f"   📊 Full Results: {full_results_file}")
            print(f"       • All parameters, statistics, and metadata")
            if len(comparison_df) > 0:
                print(f"   🏆 Model Comparison: {comparison_file}")
                print(f"       • Publication-ready model selection table")
                print(f"       • AIC weights, evidence ratios, support indicators")
                if param_summary_file:
                    print(f"   📋 Parameter Summary: {param_summary_file}")
                    print(f"       • Simplified parameter table for quick reference")
            else:
                print(f"   ⚠️  No model comparison files generated (no successful fits)")
            
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