#!/usr/bin/env python3
"""
Sample Nebraska Data Analysis Script
Randomly samples 1000 rows from Nebraska encounter history data and fits a Pradel model.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
import sys
from pathlib import Path

def main():
    """Run Pradel model analysis on a random sample of Nebraska data."""
    
    print("🔬 Nebraska Capture-Recapture Analysis - Random Sample")
    print("=" * 60)
    
    # Data file path
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        print("Available data files:")
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.glob("*.csv"):
                print(f"  - {f}")
        sys.exit(1)
    
    try:
        # Load the full dataset
        print(f"📂 Loading data from: {data_file}")
        full_data = pd.read_csv(data_file)
        print(f"   Full dataset shape: {full_data.shape}")
        
        # Random sample of 1000 rows
        sample_size = min(1000, len(full_data))  # Handle case where dataset < 1000 rows
        print(f"🎲 Randomly sampling {sample_size} rows...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        sampled_data = full_data.sample(n=sample_size, random_state=42)
        print(f"   Sample shape: {sampled_data.shape}")
        
        # Load data into pradel-jax format
        print("🔧 Converting to pradel-jax format...")
        data_context = pj.load_data(sampled_data)  # Pass DataFrame directly
        
        print("   Data summary:")
        print(f"   - Number of individuals: {data_context.n_individuals}")
        print(f"   - Number of occasions: {data_context.n_occasions}")
        if hasattr(data_context, 'covariates') and data_context.covariates:
            print(f"   - Available covariates: {list(data_context.covariates.keys())}")
        
        # Create simple model specification (start with intercept-only model)
        print("📊 Setting up Pradel model...")
        formula_spec = pj.create_formula_spec(
            phi="~1",    # Survival probability (intercept only)
            p="~1",      # Detection probability (intercept only)  
            f="~1"       # Recruitment/seniority probability (intercept only)
        )
        
        print("   Model specification:")
        print("   - Survival (φ): Intercept only")
        print("   - Detection (p): Intercept only")
        print("   - Recruitment (f): Intercept only")
        
        # Fit the model
        print("⚡ Fitting model with automatic optimization...")
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
            
            if hasattr(result, 'standard_errors') and result.standard_errors is not None:
                print(f"\nStandard errors:")
                for param_name, se in zip(result.parameter_names, result.standard_errors):
                    print(f"  SE({param_name}): {se:.4f}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📄 Results saved in result object for further analysis")
        
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