#!/usr/bin/env python3
"""
Test script to validate covariate preprocessing fixes.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from pradel_jax.optimization import create_model_specs_from_formulas, fit_models_parallel
from pradel_jax.data.adapters import GenericFormatAdapter

def main():
    print("🔧 Testing covariate preprocessing fixes...")
    
    # Load and sample data
    data_file = "data/encounter_histories_ne_clean.csv"
    full_data = pd.read_csv(data_file)
    sampled_data = full_data.sample(n=100, random_state=42)
    
    print(f"Original data shape: {sampled_data.shape}")
    
    # CRITICAL FIX: Preprocess covariates before loading
    print("🔧 Preprocessing covariates for proper modeling...")
    
    # Fix gender coding and missing values
    if 'gender' in sampled_data.columns:
        print(f"   Gender before: {sampled_data['gender'].value_counts().to_dict()}")
        # Convert 1.0->Male, 2.0->Female, NaN->Male (default)
        sampled_data['gender'] = sampled_data['gender'].fillna(1.0)  # Fill missing with Male
        sampled_data['gender'] = sampled_data['gender'].map({1.0: 'Male', 2.0: 'Female'})
        print(f"   ✅ Gender after: {sampled_data['gender'].value_counts().to_dict()}")
    
    # Standardize age for numerical stability
    if 'age' in sampled_data.columns:
        original_age = sampled_data['age'].copy()
        sampled_data['age'] = (sampled_data['age'] - sampled_data['age'].mean()) / sampled_data['age'].std()
        print(f"   ✅ Age: standardized (mean={original_age.mean():.1f}, std={original_age.std():.1f})")
        print(f"      New age range: {sampled_data['age'].min():.2f} to {sampled_data['age'].max():.2f}")
    
    # Simplify tier_history to meaningful categories
    if 'tier_history' in sampled_data.columns:
        print(f"   Tier_history unique values: {sampled_data['tier_history'].unique()[:10]}")
        
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
    
    # Save and load through adapter
    temp_file = "temp_test_sample.csv"
    sampled_data.to_csv(temp_file, index=False)
    
    adapter = GenericFormatAdapter()
    data_context = pj.load_data(temp_file, adapter=adapter)
    
    print(f"\n📊 Loaded data:")
    print(f"   - Individuals: {data_context.n_individuals}")
    print(f"   - Occasions: {data_context.n_occasions}")
    
    # Check processed covariates
    print(f"\n🔍 Processed covariates:")
    main_covariates = [k for k in data_context.covariates.keys() 
                      if not k.endswith('_categories') and not k.endswith('_is_categorical')]
    
    for cov in ['gender', 'age', 'tier']:
        if cov in main_covariates:
            values = data_context.covariates[cov]
            print(f"   {cov}: unique values = {np.unique(values)}")
    
    # Test simple model fitting
    print(f"\n⚡ Testing simple model fitting...")
    
    target_covariates = []
    for cov in ['gender', 'age', 'tier']:
        if cov in main_covariates:
            target_covariates.append(cov)
    
    print(f"   Available for modeling: {target_covariates}")
    
    # Create two simple models to test
    phi_formulas = ["~1"]
    if target_covariates:
        phi_formulas.append(f"~1 + {target_covariates[0]}")  # Single covariate test
    
    model_specs = create_model_specs_from_formulas(
        phi_formulas=phi_formulas,
        p_formulas=["~1"],
        f_formulas=["~1"],
        random_seed_base=42
    )
    
    print(f"   Testing {len(model_specs)} models...")
    
    results = fit_models_parallel(
        model_specs=model_specs,
        data_context=data_context,
        n_workers=1
    )
    
    # Check if models have different likelihoods
    print(f"\n🎯 Results:")
    for i, result in enumerate(results):
        if result and result.success:
            print(f"   Model {i+1}: log-likelihood = {result.log_likelihood:.3f}, AIC = {result.aic:.3f}")
            print(f"              Parameters: {[f'{p:.4f}' for p in result.parameters]}")
        else:
            print(f"   Model {i+1}: FAILED")
    
    # Check if we have different likelihoods
    if len(results) >= 2:
        ll1 = results[0].log_likelihood if results[0] and results[0].success else None
        ll2 = results[1].log_likelihood if results[1] and results[1].success else None
        
        if ll1 and ll2:
            diff = abs(ll1 - ll2)
            if diff > 0.001:
                print(f"\n✅ SUCCESS: Models have different likelihoods (diff = {diff:.6f})")
                print("   Covariate preprocessing is working!")
            else:
                print(f"\n❌ ISSUE: Models still have identical likelihoods (diff = {diff:.6f})")
                print("   Covariates may still not be working properly")
    
    # Cleanup
    import os
    if os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    main()