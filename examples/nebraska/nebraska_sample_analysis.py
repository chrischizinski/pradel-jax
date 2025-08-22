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
import sys
import argparse
from pathlib import Path
from itertools import combinations

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
                       help='Number of individuals to sample (default: 1000)')
    parser.add_argument('--max-models', type=int, default=50,
                       help='Maximum number of models to fit (default: 50)')
    args = parser.parse_args()
    
    print("🔬 Nebraska Capture-Recapture Analysis - Random Sample")
    print("=" * 60)
    
    # Data file path (relative to project root when run from examples/nebraska/)
    data_file = "../../data/encounter_histories_ne_clean.csv"
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        print("Available data files:")
        data_dir = Path("../../data")
        if data_dir.exists():
            for f in data_dir.glob("*.csv"):
                print(f"  - {f}")
        else:
            print("Data directory not found. Make sure you're running from examples/nebraska/")
        sys.exit(1)
    
    try:
        # Load the full dataset
        print(f"📂 Loading data from: {data_file}")
        full_data = pd.read_csv(data_file)
        print(f"   Full dataset shape: {full_data.shape}")
        
        # Random sample of specified size
        sample_size = min(args.sample_size, len(full_data))  # Handle case where dataset < specified rows
        print(f"🎲 Randomly sampling {sample_size} rows...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        sampled_data = full_data.sample(n=sample_size, random_state=42)
        print(f"   Sample shape: {sampled_data.shape}")
        
        # Save sampled data to temporary file for loading
        print("🔧 Converting to pradel-jax format...")
        temp_file = "temp_nebraska_sample.csv"
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
        
        # Create model specifications
        model_specs = create_model_specs_from_formulas(
            phi_formulas=phi_formulas,
            p_formulas=p_formulas, 
            f_formulas=f_formulas,
            random_seed_base=42
        )
        
        print(f"   Created {len(model_specs)} model(s) for comparison:")
        for i, spec in enumerate(model_specs):
            print(f"   - Model {i+1}: {spec.name}")
        
        # Fit the model using parallel optimization
        print("⚡ Fitting model with automatic optimization...")
        results = fit_models_parallel(
            model_specs=model_specs,
            data_context=data_context,
            n_workers=1  # Single worker for this simple case
        )
        
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
            full_results_file = f"nebraska_full_results_{sample_size}ind_{timestamp}.csv"
            export_df = pj.export_model_results(
                results=results,
                model_specs=model_specs,
                data_context=data_context,
                export_file=full_results_file,
                print_summary=False  # We'll print our own summary
            )
            
            # 2. Model comparison table (publication-ready)
            print(f"   🏆 Creating model comparison table...")
            exporter = pj.ResultsExporter()
            comparison_df = exporter.create_model_comparison_table(export_df, sort_by='aic')
            comparison_file = f"nebraska_model_comparison_{sample_size}ind_{timestamp}.csv"
            comparison_df.to_csv(comparison_file, index=False)
            
            # 3. Parameter summary (simplified for quick reference)
            print(f"   📋 Creating parameter summary...")
            # Use comparison_df which has delta_aic and aic_weight columns
            param_cols = ['model_name', 'aic', 'n_parameters', 'lambda_mean']
            # Add delta_aic and aic_weight if they exist in comparison_df
            if 'delta_aic' in comparison_df.columns:
                param_cols.append('delta_aic')
            if 'aic_weight' in comparison_df.columns:
                param_cols.append('aic_weight')
            # Add parameter columns from original export_df
            param_cols.extend([col for col in export_df.columns if col.startswith(('phi_', 'p_', 'f_')) and not col.endswith('_estimate')])
            
            # Create parameter summary from comparison_df (which has the model selection metrics)
            available_cols = [col for col in param_cols if col in comparison_df.columns]
            param_summary = comparison_df[available_cols].copy()
            param_summary_file = f"nebraska_parameters_{sample_size}ind_{timestamp}.csv"
            param_summary.to_csv(param_summary_file, index=False)
            
            # 4. Print enhanced analysis summary
            print(f"\n📊 Nebraska Capture-Recapture Analysis Summary")
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
                print(f"   Evidence Ratio: {best_model.get('evidence_ratio', 'N/A'):.1f}x better than next model" if len(comparison_df) > 1 else "")
                
                # Population dynamics interpretation
                if 'lambda_mean' in best_model and best_model['lambda_mean'] is not None:
                    lambda_val = best_model['lambda_mean']
                    print(f"\n📈 Population Dynamics (Best Model):")
                    print(f"   Population Growth Rate (λ): {lambda_val:.4f}")
                    if lambda_val < 0.95:
                        trend = "🔻 DECLINING"
                    elif lambda_val > 1.05:
                        trend = "🔺 INCREASING" 
                    else:
                        trend = "➡️  STABLE"
                    print(f"   Population Trend: {trend}")
                
                # Model support summary
                substantial_models = comparison_df[comparison_df['substantial_support'] == True]
                print(f"\n🎯 Model Support Summary:")
                print(f"   Models with substantial support (ΔAIC ≤ 2): {len(substantial_models)}")
                print(f"   Top model cumulative weight: {comparison_df['aic_weight'].iloc[0]:.3f}")
                
            # 5. Files generated summary
            print(f"\n📁 Generated Files:")
            print(f"   📊 Full Results: {full_results_file}")
            print(f"       • All parameters, statistics, and metadata")
            print(f"       • MARK-compatible format with 40+ columns")
            print(f"   🏆 Model Comparison: {comparison_file}")
            print(f"       • Publication-ready model selection table")
            print(f"       • AIC weights, evidence ratios, support indicators")
            print(f"   📋 Parameter Summary: {param_summary_file}")
            print(f"       • Simplified parameter table for quick reference")
            
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