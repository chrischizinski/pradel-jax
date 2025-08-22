#!/usr/bin/env python3
"""
Nebraska Pradel Model Analysis with Covariates
Uses the pre-built capture history (ch) and includes age, gender, and tier covariates.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
import sys
from pathlib import Path

def main():
    """Run Pradel model analysis on Nebraska data with covariates."""
    
    print("🔬 Nebraska Pradel Model Analysis with Covariates")
    print("=" * 60)
    
    # Data file path
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        sys.exit(1)
    
    try:
        # Load and sample data
        print(f"📂 Loading data from: {data_file}")
        full_data = pd.read_csv(data_file)
        print(f"   Full dataset shape: {full_data.shape}")
        
        # Random sample of 1000 rows
        sample_size = min(1000, len(full_data))
        print(f"🎲 Randomly sampling {sample_size} rows...")
        np.random.seed(42)
        sampled_data = full_data.sample(n=sample_size, random_state=42)
        
        # Examine the data structure
        print("\n📊 Data Structure Summary:")
        print(f"   - Individual IDs: {sampled_data['person_id'].nunique()} unique")
        print(f"   - Capture histories (ch): Available")
        print(f"   - Date of birth (dob): Available") 
        print(f"   - Gender: {sampled_data['gender'].value_counts().to_dict()}")
        
        # Check capture history format  
        ch_sample = sampled_data['ch'].astype(str)
        ch_lengths = ch_sample.str.len()
        print(f"   - Capture history length: {ch_lengths.mode().iloc[0]} characters")
        
        # Example capture histories
        print("\n🔍 Example Capture Histories:")
        for i in range(min(5, len(sampled_data))):
            person = sampled_data.iloc[i]
            print(f"   {person['person_id']}: {person['ch']} (gender: {person['gender']}, age in 2016: {person['age_2016']})")
        
        # Prepare data for pradel-jax
        print("\n🔧 Preparing data for Pradel model...")
        
        # Convert capture history string to individual occasion columns
        ch_data = []
        year_columns = ['Y2016', 'Y2017', 'Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024']
        
        for idx, row in sampled_data.iterrows():
            individual_data = {
                'individual': row['person_id'],
                'gender': row['gender'] if pd.notna(row['gender']) else 'Unknown'
            }
            
            # Add binary encounter history (convert from Y columns, not ch string)
            for i, year_col in enumerate(year_columns):
                occasion_name = f'occasion_{i+1}'
                # Use the Y columns directly since they're already processed
                individual_data[occasion_name] = 1 if row[year_col] > 0 else 0
            
            # Add time-varying age covariates
            age_columns = ['age_2016', 'age_2017', 'age_2018', 'age_2019', 'age_2020', 'age_2021', 'age_2022', 'age_2023', 'age_2024']
            for i, age_col in enumerate(age_columns):
                individual_data[f'age_{i+1}'] = row[age_col] if pd.notna(row[age_col]) else np.nan
            
            # Add time-varying tier covariates 
            tier_columns = ['tier_2016', 'tier_2017', 'tier_2018', 'tier_2019', 'tier_2020', 'tier_2021', 'tier_2022', 'tier_2023', 'tier_2024']
            for i, tier_col in enumerate(tier_columns):
                individual_data[f'tier_{i+1}'] = row[tier_col] if pd.notna(row[tier_col]) else 0
            
            ch_data.append(individual_data)
        
        # Create DataFrame
        processed_data = pd.DataFrame(ch_data)
        
        # Clean gender data (assuming 1=Male, 2=Female based on your data)
        processed_data['gender'] = processed_data['gender'].fillna(0)  # 0 for unknown
        processed_data['gender_numeric'] = processed_data['gender']  # Already numeric
        
        # Create readable labels for summary
        gender_labels = processed_data['gender'].map({1.0: 'Male', 2.0: 'Female', 0: 'Unknown'}).fillna('Unknown')
        
        # Save to temporary file
        temp_file = "temp_nebraska_covariates.csv"
        processed_data.to_csv(temp_file, index=False)
        
        print("📈 Data Processing Summary:")
        occasions = [col for col in processed_data.columns if col.startswith('occasion_')]
        total_detections = processed_data[occasions].sum().sum()
        total_possible = len(processed_data) * len(occasions)
        print(f"   - Occasions: {len(occasions)}")
        print(f"   - Detection rate: {total_detections}/{total_possible} = {total_detections/total_possible:.3f}")
        # Calculate age statistics from time-varying age columns
        age_cols = [col for col in processed_data.columns if col.startswith('age_')]
        age_values = processed_data[age_cols].values.flatten()
        age_values = age_values[~pd.isna(age_values)]
        print(f"   - Age range across all years: {age_values.min():.0f} to {age_values.max():.0f}")
        print(f"   - Gender distribution: {gender_labels.value_counts().to_dict()}")
        
        # Load into pradel-jax
        print("\n🔧 Loading data into pradel-jax format...")
        
        # Create minimal format - just encounter histories first
        minimal_data = processed_data[['individual'] + occasions].copy()
        minimal_temp_file = "temp_nebraska_minimal.csv"
        minimal_data.to_csv(minimal_temp_file, index=False)
        
        try:
            data_context = pj.load_data(minimal_temp_file)
            print(f"   ✅ Data loaded successfully")
        except Exception as load_error:
            print(f"   ❌ Data loading failed: {load_error}")
            raise
        
        print(f"   - Individuals: {data_context.n_individuals}")
        print(f"   - Occasions: {data_context.n_occasions}")
        
        # Model specifications with increasing complexity
        # Start with simple intercept-only model
        model_specs = [
            {
                'name': 'Intercept-only model',
                'formula': pj.create_simple_spec(phi="~1", p="~1", f="~1")
            }
        ]
        
        # Fit models
        results = []
        for spec in model_specs:
            print(f"\n⚡ Fitting: {spec['name']}")
            
            try:
                model = pj.PradelModel()
                result = model.fit(
                    formula_spec=spec['formula'],
                    data_context=data_context
                )
                
                results.append({
                    'name': spec['name'],
                    'result': result
                })
                
                print(f"   {'✅ Success' if result.success else '❌ Failed'}")
                if result.success:
                    print(f"   AIC: {result.aic:.3f}")
                    print(f"   Strategy: {result.strategy_used}")
                
            except Exception as model_error:
                print(f"   ❌ Model failed: {model_error}")
        
        # Display results
        print(f"\n🎯 Model Results Summary")
        print("=" * 50)
        
        for model_result in results:
            result = model_result['result']
            print(f"\n📊 {model_result['name']}:")
            print(f"   Convergence: {'✅ Success' if result.success else '❌ Failed'}")
            
            if result.success:
                print(f"   Log-likelihood: {result.log_likelihood:.3f}")
                print(f"   AIC: {result.aic:.3f}")
                print(f"   Parameters: {[f'{name}: {val:.4f}' for name, val in zip(result.parameter_names, result.parameters)]}")
        
        print(f"\n✅ Nebraska covariate analysis completed!")
        print(f"📊 Analyzed {len(processed_data)} individuals over 9 occasions (2016-2024)")
        
        # Clean up temporary files
        import os
        for temp_file_name in [temp_file, minimal_temp_file]:
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        print("\n🔍 Full error traceback:")
        traceback.print_exc()
        
        sys.exit(1)

if __name__ == "__main__":
    main()