#!/usr/bin/env python3
"""
Time-Varying Covariate Investigation
====================================

Investigate and implement proper time-varying covariate support for tier and age
in the Pradel-JAX capture-recapture models.

Requirements:
- Age should vary by year (age increases each capture occasion)
- Tier should be able to vary by year (hunters can change tier status)
- Both covariates should be properly incorporated into the design matrices

Author: Workflow Validation System
Date: August 2025
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def investigate_time_varying_structure():
    """Investigate the time-varying structure in our datasets."""
    
    print("🕐 TIME-VARYING COVARIATE INVESTIGATION")
    print("=" * 60)
    
    # Load both datasets
    datasets = {
        'nebraska': 'data/encounter_histories_ne_clean.csv',
        'south_dakota': 'data/encounter_histories_sd_clean.csv'
    }
    
    for dataset_name, file_path in datasets.items():
        if not Path(file_path).exists():
            print(f"❌ Dataset not found: {file_path}")
            continue
            
        print(f"\n📊 DATASET: {dataset_name.upper()}")
        print("-" * 40)
        
        # Load data
        data = pd.read_csv(file_path)
        print(f"Dataset shape: {data.shape}")
        
        # 1. INVESTIGATE AGE STRUCTURE
        print("\n🎂 AGE COVARIATE ANALYSIS:")
        
        # Find age columns
        age_columns = [col for col in data.columns if col.startswith('age_')]
        if age_columns:
            print(f"  Time-varying age columns found: {age_columns}")
            
            # Sample 10 individuals to show age progression
            sample_data = data.head(10)
            age_data = sample_data[['person_id' if 'person_id' in data.columns else 'customer_id'] + age_columns]
            print(f"\n  Age progression examples:")
            print(age_data.to_string(index=False))
            
            # Check for proper age progression (should increase by ~1 each year)
            if len(age_columns) >= 2:
                age_diff = data[age_columns[1]] - data[age_columns[0]]
                print(f"\n  Age progression validation:")
                print(f"    Expected difference: ~1 year")
                print(f"    Actual mean difference: {age_diff.mean():.2f}")
                print(f"    Difference std: {age_diff.std():.2f}")
                print(f"    Reasonable progression: {'✅' if 0.8 <= age_diff.mean() <= 1.2 else '❌'}")
        else:
            print("  ❌ No time-varying age columns found")
            
        # Check for single age column
        if 'age' in data.columns:
            print(f"  Single 'age' column found: {data['age'].describe()}")
        
        # 2. INVESTIGATE TIER STRUCTURE  
        print("\n🎯 TIER COVARIATE ANALYSIS:")
        
        # Find tier-related columns
        tier_columns = [col for col in data.columns if 'tier' in col.lower()]
        print(f"  Tier-related columns: {tier_columns}")
        
        for col in tier_columns:
            if col in data.columns:
                unique_vals = data[col].unique()
                print(f"  {col}: {len(unique_vals)} unique values")
                if len(unique_vals) <= 20:  # Show values if not too many
                    print(f"    Values: {sorted([x for x in unique_vals if pd.notna(x)])}")
                
                # Check for time-variation within individuals
                if 'person_id' in data.columns:
                    id_col = 'person_id'
                elif 'customer_id' in data.columns:
                    id_col = 'customer_id'
                else:
                    id_col = None
                    
                if id_col and len(data) > 1000:  # Only if we have ID column
                    sample_ids = data[id_col].unique()[:100]  # Sample 100 individuals
                    sample_data = data[data[id_col].isin(sample_ids)]
                    
                    # This would need yearly tier data to properly analyze
                    print(f"    Individual tier variation analysis needs yearly tier data")
        
        # 3. ENCOUNTER HISTORY ANALYSIS
        print("\n📅 ENCOUNTER HISTORY TEMPORAL STRUCTURE:")
        
        # Find Y-columns (yearly encounter indicators)
        y_columns = [col for col in data.columns if col.startswith('Y')]
        if y_columns:
            print(f"  Encounter year columns: {y_columns}")
            years = sorted([int(col[1:]) for col in y_columns if col[1:].isdigit()])
            print(f"  Study years: {min(years)}-{max(years)} ({len(years)} years)")
            
            # Check encounter rates by year
            encounter_rates = {}
            for y_col in y_columns:
                if y_col[1:].isdigit():
                    year = int(y_col[1:])
                    rate = data[y_col].mean()
                    encounter_rates[year] = rate
            
            print(f"  Encounter rates by year:")
            for year in sorted(encounter_rates.keys()):
                print(f"    {year}: {encounter_rates[year]:.3f}")
        
        # 4. CHECK FOR PROPER TIME-VARYING COVARIATE SUPPORT
        print(f"\n🔧 TIME-VARYING COVARIATE REQUIREMENTS:")
        print(f"  ✅ Need: Age varies by capture occasion")
        print(f"  ✅ Need: Tier can vary by capture occasion") 
        print(f"  ✅ Need: Design matrices accommodate time-varying effects")
        
        # Show current covariate availability
        covariates_available = []
        if age_columns:
            covariates_available.append(f"Age (time-varying): {len(age_columns)} years")
        elif 'age' in data.columns:
            covariates_available.append("Age (single value - needs time-varying implementation)")
            
        if 'tier_history' in data.columns:
            covariates_available.append("Tier (as tier_history - may need expansion to yearly values)")
        if 'tier' in data.columns:
            covariates_available.append("Tier (processed - may need time-varying expansion)")
            
        print(f"\n  Current covariate availability:")
        for cov in covariates_available:
            print(f"    • {cov}")

def check_pradel_jax_time_varying_support():
    """Check current Pradel-JAX support for time-varying covariates."""
    
    print(f"\n🔍 PRADEL-JAX TIME-VARYING SUPPORT INVESTIGATION")
    print("=" * 60)
    
    try:
        # Check formula system support
        from pradel_jax.formulas import create_simple_spec
        from pradel_jax.formulas.time_varying import TimeVaryingFormulaSpec
        print("✅ Time-varying formula module found")
        
        # Check design matrix support
        from pradel_jax.formulas.design_matrix import build_design_matrices_time_varying
        print("✅ Time-varying design matrix builder found")
        
    except ImportError as e:
        print(f"⚠️ Time-varying support may be limited: {e}")
        
    # Test basic time-varying formula
    try:
        import pradel_jax as pj
        
        # Create a small test dataset
        print(f"\n🧪 Testing time-varying covariate processing...")
        
        # Load a small sample to test
        ne_file = 'data/encounter_histories_ne_clean.csv'
        if Path(ne_file).exists():
            data = pd.read_csv(ne_file)
            sample = data.head(20)  # Small sample for testing
            
            # Save test sample
            test_file = "temp_time_varying_test.csv"
            sample.to_csv(test_file, index=False)
            
            # Load through pradel-jax
            data_context = pj.load_data(test_file)
            
            print(f"  Data context created: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
            
            # Check available covariates
            if hasattr(data_context, 'covariates'):
                time_varying_candidates = []
                for cov_name, cov_data in data_context.covariates.items():
                    if isinstance(cov_data, np.ndarray) and len(cov_data.shape) > 1:
                        if cov_data.shape[1] > 1:  # Multiple time points
                            time_varying_candidates.append((cov_name, cov_data.shape))
                
                print(f"  Time-varying covariate candidates:")
                for name, shape in time_varying_candidates:
                    print(f"    • {name}: shape {shape}")
                    
                if not time_varying_candidates:
                    print(f"    ❌ No time-varying covariates detected in current processing")
                    print(f"    💡 May need to modify data adapter to preserve yearly covariate structure")
            
            # Clean up
            import os
            if os.path.exists(test_file):
                os.remove(test_file)
                
        else:
            print(f"  ⚠️ Cannot test - data file not available")
            
    except Exception as e:
        print(f"❌ Error testing time-varying support: {e}")

def generate_time_varying_recommendations():
    """Generate recommendations for implementing time-varying covariates."""
    
    print(f"\n💡 TIME-VARYING COVARIATE RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = [
        "1. **Data Adapter Enhancement**:",
        "   - Modify GenericFormatAdapter to preserve yearly age columns (age_2016, age_2017, etc.)",
        "   - Create yearly tier variables from tier_history or administrative records",
        "   - Ensure time-varying covariates are stored as (n_individuals, n_occasions) matrices",
        "",
        "2. **Formula System Enhancement**:",
        "   - Extend formula parser to handle time-varying syntax: φ~age(t) + tier(t)",
        "   - Support for interaction between time-varying covariates and time",
        "   - Validate that time-varying formulas match data structure",
        "",
        "3. **Design Matrix Construction**:",
        "   - Update design matrix builder to handle time-varying covariates properly",
        "   - Ensure proper indexing for time-varying effects in likelihood computation",
        "   - Handle missing values in time-varying covariates gracefully",
        "",
        "4. **Model Implementation**:",
        "   - Verify Pradel likelihood computation supports time-varying covariates",
        "   - Test parameter estimation with time-varying effects",
        "   - Validate against known results with time-varying covariates",
        "",
        "5. **Validation Requirements**:",
        "   - Compare results with RMark time-varying models",
        "   - Test with simulated data where true time-varying effects are known",
        "   - Ensure biological realism of parameter estimates"
    ]
    
    for rec in recommendations:
        print(rec)
    
    print(f"\n🎯 PRIORITY ACTIONS:")
    print(f"1. Investigate current time-varying support in design matrix construction")
    print(f"2. Modify data adapter to preserve yearly covariate structure")
    print(f"3. Test time-varying formula syntax and processing")
    print(f"4. Validate against known time-varying model results")

def main():
    """Main investigation function."""
    
    investigate_time_varying_structure()
    check_pradel_jax_time_varying_support() 
    generate_time_varying_recommendations()
    
    print(f"\n✅ Time-varying covariate investigation completed!")
    print(f"📋 Next steps: Implement recommendations and test time-varying models")

if __name__ == "__main__":
    main()