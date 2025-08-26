#!/usr/bin/env python3
"""
Debug Real Data Error
=====================

Test with actual Nebraska data to reproduce the JAX assignment error.
"""

import traceback
import numpy as np
import pandas as pd
from pathlib import Path

def test_real_data_error():
    """Test with real Nebraska data to reproduce error."""
    
    print("🔍 DEBUG REAL DATA JAX ERROR")
    print("=" * 60)
    
    try:
        import pradel_jax as pj
        from pradel_jax.data.adapters import GenericFormatAdapter  
        from pradel_jax.models import PradelModel
        from pradel_jax.formulas import create_simple_spec
        
        # Load real data
        data_file = "data/encounter_histories_ne_clean.csv"
        if not Path(data_file).exists():
            print(f"❌ Data file not found: {data_file}")
            return
            
        print("🔧 Loading real Nebraska data...")
        full_data = pd.read_csv(data_file)
        sample_data = full_data.sample(n=50, random_state=42)  # Small but real sample
        
        print(f"✅ Real data loaded: {sample_data.shape}")
        print(f"   Columns: {list(sample_data.columns)[:10]}...")
        
        # Save temp file
        temp_file = "temp_real_debug.csv"
        sample_data.to_csv(temp_file, index=False)
        
        # Load through pradel-jax
        print("🔧 Processing through GenericFormatAdapter...")
        adapter = GenericFormatAdapter()
        data_context = pj.load_data(temp_file, adapter=adapter)
        
        print("✅ Data processed successfully")
        print(f"   Individuals: {data_context.n_individuals}")
        print(f"   Occasions: {data_context.n_occasions}")
        print(f"   Covariates: {len(data_context.covariates)}")
        
        # Test intercept-only model first
        print("🔧 Testing intercept-only model...")
        formula_spec = create_simple_spec(phi="~1", p="~1", f="~1")
        model = PradelModel()
        
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        print("✅ Intercept-only model works!")
        
        # Test with gender covariate - this might trigger the error
        print("🔧 Testing model with gender covariate...")
        formula_spec = create_simple_spec(phi="~1 + gender", p="~1", f="~1")
        
        try:
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            print("✅ Gender covariate model works!")
            
            for param, info in design_matrices.items():
                print(f"   {param}: {info.matrix.shape}")
                
        except Exception as cov_error:
            print(f"❌ Error with gender covariate: {cov_error}")
            print("This might be where the JAX immutable error occurs")
            traceback.print_exc()
        
        # Test with age covariate 
        print("🔧 Testing model with age covariate...")
        try:
            formula_spec = create_simple_spec(phi="~1 + age", p="~1", f="~1")
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            print("✅ Age covariate model works!")
            
        except Exception as age_error:
            print(f"❌ Error with age covariate: {age_error}")
            traceback.print_exc()
        
        # Now test optimization - this is where the real error was occurring
        print("🔧 Testing optimization...")
        try:
            from pradel_jax.optimization import optimize_model
            
            # Simple intercept model for optimization test
            formula_spec = create_simple_spec(phi="~1", p="~1", f="~1")
            model = PradelModel()
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            
            def objective(params):
                return -model.log_likelihood(params, data_context, design_matrices)
            
            initial_params = model.get_initial_parameters(data_context, design_matrices)
            bounds = model.get_parameter_bounds(data_context, design_matrices)
            
            print(f"   Initial params: {initial_params}")
            print(f"   Bounds: {bounds}")
            
            # This is likely where the JAX error occurs
            result = optimize_model(
                objective_function=objective,
                initial_parameters=initial_params,
                context=data_context,
                bounds=bounds
            )
            
            print(f"✅ Optimization successful: {result.success}")
            
        except Exception as opt_error:
            print(f"❌ Optimization error: {opt_error}")
            print("🔍 This is likely the source of the JAX immutable array error!")
            traceback.print_exc()
        
        # Clean up
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    except Exception as e:
        print(f"❌ Top-level error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_real_data_error()