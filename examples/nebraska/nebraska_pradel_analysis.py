#!/usr/bin/env python3
"""
Nebraska Pradel Model Analysis
Converts Nebraska data to binary format and runs Pradel model analysis.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
import sys
from pathlib import Path

def main():
    """Run Pradel model analysis on Nebraska data."""
    
    print("🔬 Nebraska Pradel Model Analysis")
    print("=" * 50)
    
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
        
        # Extract year columns and convert to binary
        year_columns = [col for col in sampled_data.columns if col.startswith('Y20')]
        print(f"📊 Processing {len(year_columns)} capture occasions: {year_columns[0]} to {year_columns[-1]}")
        
        # Convert encounter histories to binary (any detection > 0 becomes 1)
        encounter_histories = sampled_data[year_columns].copy()
        encounter_histories = (encounter_histories > 0).astype(int)
        
        # Create individual IDs 
        individual_ids = sampled_data['person_id'].values
        
        # Create formatted data for pradel-jax
        formatted_data = pd.DataFrame()
        formatted_data['individual'] = individual_ids
        
        # Add binary encounter histories
        for i, year_col in enumerate(year_columns, 1):
            formatted_data[f'occasion_{i}'] = encounter_histories[year_col].values
        
        # Save to temporary file
        temp_file = "temp_nebraska_binary.csv"
        formatted_data.to_csv(temp_file, index=False)
        
        print("🔧 Loading data into pradel-jax format...")
        try:
            data_context = pj.load_data(temp_file)
        except Exception as load_error:
            print(f"❌ Data loading failed: {load_error}")
            print("🔧 Trying alternative format...")
            
            # Create a simpler format - just the encounter history columns
            simple_data = encounter_histories.copy()
            simple_data.index = individual_ids
            simple_temp_file = "temp_nebraska_simple.csv" 
            simple_data.to_csv(simple_temp_file, index_label='individual')
            data_context = pj.load_data(simple_temp_file)
        
        print("   Data summary:")
        print(f"   - Number of individuals: {data_context.n_individuals}")
        print(f"   - Number of occasions: {data_context.n_occasions}")
        
        # Binary encounter history statistics
        total_detections = encounter_histories.sum().sum()
        total_possible = len(encounter_histories) * len(year_columns)
        detection_rate = total_detections / total_possible
        
        print(f"   - Total detections: {total_detections} / {total_possible} ({detection_rate:.3f})")
        
        individuals_detected = (encounter_histories.sum(axis=1) > 0).sum()
        print(f"   - Individuals ever detected: {individuals_detected} / {len(encounter_histories)} ({individuals_detected/len(encounter_histories):.3f})")
        
        # Create simple model specification
        print("📊 Setting up Pradel model (intercept-only)...")
        formula_spec = pj.create_formula_spec(
            phi="~1",    # Survival probability (constant)
            p="~1",      # Detection probability (constant)  
            f="~1"       # Recruitment probability (constant)
        )
        
        print("   Model specification:")
        print("   - Survival (φ): Intercept only")
        print("   - Detection (p): Intercept only")
        print("   - Recruitment (f): Intercept only")
        
        # Fit the model
        print("⚡ Fitting Pradel model...")
        result = pj.fit_model(
            model=pj.PradelModel(),
            formula=formula_spec,
            data=data_context
        )
        
        # Display results
        print("\n🎯 Model Results")
        print("=" * 40)
        print(f"Convergence: {'✅ Success' if result.success else '❌ Failed'}")
        print(f"Optimization strategy: {result.strategy_used}")
        print(f"Function evaluations: {result.n_function_evaluations}")
        print(f"Log-likelihood: {result.log_likelihood:.3f}")
        print(f"AIC: {result.aic:.3f}")
        
        if result.success:
            print(f"\nParameter estimates:")
            for param_name, value in zip(result.parameter_names, result.parameters):
                print(f"  {param_name}: {value:.4f}")
            
            # Convert to probabilities for interpretation
            print(f"\nProbability estimates:")
            if len(result.parameters) >= 3:
                phi_logit, p_logit, f_logit = result.parameters[:3]
                phi_prob = 1 / (1 + np.exp(-phi_logit))  # Survival probability
                p_prob = 1 / (1 + np.exp(-p_logit))      # Detection probability  
                f_prob = 1 / (1 + np.exp(-f_logit))      # Recruitment probability
                
                print(f"  Survival (φ): {phi_prob:.4f}")
                print(f"  Detection (p): {p_prob:.4f}")
                print(f"  Recruitment (f): {f_prob:.4f}")
        
        print(f"\n✅ Nebraska Pradel model analysis completed!")
        print(f"📊 Model fitted to {data_context.n_individuals} individuals over {data_context.n_occasions} occasions")
        
        # Clean up temporary files
        import os
        for temp_file_name in [temp_file, "temp_nebraska_simple.csv"]:
            if os.path.exists(temp_file_name):
                os.remove(temp_file_name)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        # Provide helpful debugging information
        print("\n🔍 Troubleshooting tips:")
        print("1. Data has been converted to binary format (0/1)")
        print("2. Using simple intercept-only model")
        print("3. Check console output for specific error details")
        
        sys.exit(1)

if __name__ == "__main__":
    main()