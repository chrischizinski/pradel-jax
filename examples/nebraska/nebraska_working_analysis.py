#!/usr/bin/env python3
"""
Working Nebraska Pradel Model Analysis
Uses the optimization framework directly to fit models successfully.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from pradel_jax.optimization import optimize_model, ModelContext
import sys
from pathlib import Path

def main():
    """Run working Pradel model analysis on Nebraska data."""
    
    print("🔬 Nebraska Pradel Model Analysis (Working Version)")
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
        
        # Random sample of 500 rows (smaller for initial testing)
        sample_size = min(500, len(full_data))
        print(f"🎲 Randomly sampling {sample_size} rows...")
        np.random.seed(42)
        sampled_data = full_data.sample(n=sample_size, random_state=42)
        
        # Process data
        print("🔧 Processing encounter histories...")
        year_columns = ['Y2016', 'Y2017', 'Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024']
        
        # Create simple encounter history data
        ch_data = []
        for idx, row in sampled_data.iterrows():
            individual_data = {'individual': row['person_id']}
            
            # Add binary encounter history
            for i, year_col in enumerate(year_columns):
                occasion_name = f'occasion_{i+1}'
                individual_data[occasion_name] = 1 if row[year_col] > 0 else 0
            
            ch_data.append(individual_data)
        
        processed_data = pd.DataFrame(ch_data)
        occasions = [col for col in processed_data.columns if col.startswith('occasion_')]
        
        # Summary statistics
        total_detections = processed_data[occasions].sum().sum()
        total_possible = len(processed_data) * len(occasions)
        print(f"📊 Data Summary:")
        print(f"   - Individuals: {len(processed_data)}")
        print(f"   - Occasions: {len(occasions)}")
        print(f"   - Detection rate: {total_detections}/{total_possible} = {total_detections/total_possible:.3f}")
        
        # Save and load into pradel-jax
        temp_file = "temp_nebraska_working.csv"
        processed_data.to_csv(temp_file, index=False)
        
        print("🔧 Loading data into pradel-jax...")
        data_context = pj.load_data(temp_file)
        print(f"   ✅ Loaded: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
        
        # Create model and setup optimization
        print("⚡ Setting up Pradel model...")
        model = pj.PradelModel()
        
        # Create simple formula spec
        formula_spec = pj.create_simple_spec(phi="~1", p="~1", f="~1", name="Intercept-only")
        print(f"   Formula: {formula_spec}")
        
        # Build design matrices
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        print(f"   Design matrices built successfully")
        
        # Get initial parameters and bounds
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        print(f"   Initial parameters: {initial_params}")
        print(f"   Parameter bounds: {bounds}")
        
        # Define objective function (negative log-likelihood)
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                return -ll  # Negative for minimization
            except Exception as e:
                print(f"   Warning: Objective evaluation failed: {e}")
                return np.inf
        
        # Create model context for optimization
        model_context = ModelContext(
            n_parameters=len(initial_params),
            n_individuals=data_context.n_individuals,
            n_occasions=data_context.n_occasions
        )
        
        print("🚀 Fitting model using optimization framework...")
        result = optimize_model(
            objective_function=objective,
            initial_parameters=initial_params,
            context=model_context,
            bounds=bounds
        )
        
        # Display results
        print(f"\n🎯 Model Results")
        print("=" * 40)
        print(f"Convergence: {'✅ Success' if result.success else '❌ Failed'}")
        
        if result.success:
            print(f"Strategy used: {result.strategy_used}")
            print(f"Function evaluations: {result.n_function_evaluations}")
            print(f"Final objective value: {result.final_objective_value:.3f}")
            print(f"Log-likelihood: {-result.final_objective_value:.3f}")
            
            # Calculate AIC
            k = len(result.parameters)  # Number of parameters
            aic = 2 * k - 2 * (-result.final_objective_value)
            print(f"AIC: {aic:.3f}")
            
            print(f"\nParameter estimates (logit scale):")
            param_names = ['phi_intercept', 'p_intercept', 'f_intercept']
            for i, (name, value) in enumerate(zip(param_names, result.parameters)):
                print(f"  {name}: {value:.4f}")
            
            # Convert to probabilities
            print(f"\nProbability estimates:")
            if len(result.parameters) >= 3:
                phi_prob = 1 / (1 + np.exp(-result.parameters[0]))  # Survival
                p_prob = 1 / (1 + np.exp(-result.parameters[1]))    # Detection
                f_prob = 1 / (1 + np.exp(-result.parameters[2]))    # Recruitment
                
                print(f"  Survival (φ): {phi_prob:.4f}")
                print(f"  Detection (p): {p_prob:.4f}")  
                print(f"  Recruitment (f): {f_prob:.4f}")
        
        else:
            print(f"Optimization failed: {result.message}")
            if hasattr(result, 'parameters') and result.parameters is not None:
                print(f"Last parameters: {result.parameters}")
        
        print(f"\n✅ Nebraska Pradel analysis completed!")
        
        # Clean up
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        print("\n🔍 Full error traceback:")
        traceback.print_exc()
        
        sys.exit(1)

if __name__ == "__main__":
    main()