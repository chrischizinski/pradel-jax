#!/usr/bin/env python3
"""
Debug JAX Assignment Error
==========================

Simple test to isolate the JAX array assignment error.
"""

import traceback
import numpy as np
import pandas as pd

def test_simple_model_build():
    """Test simple model building to isolate JAX error."""
    
    print("🐛 DEBUG JAX ASSIGNMENT ERROR")
    print("=" * 60)
    
    try:
        import pradel_jax as pj
        from pradel_jax.data.adapters import GenericFormatAdapter  
        from pradel_jax.models import PradelModel
        from pradel_jax.formulas import create_simple_spec
        
        print("✅ Imports successful")
        
        # Create minimal test data
        test_data = pd.DataFrame({
            'Y2016': [1, 0, 1],
            'Y2017': [0, 1, 1], 
            'Y2018': [1, 1, 0],
            'gender': ['Male', 'Female', 'Male'],
            'age': [25, 30, 35]
        })
        
        print("✅ Test data created")
        print(f"   Shape: {test_data.shape}")
        
        # Save temp file
        temp_file = "temp_debug.csv"
        test_data.to_csv(temp_file, index=False)
        
        # Load through pradel-jax
        print("🔧 Loading data through pradel-jax...")
        adapter = GenericFormatAdapter()
        data_context = pj.load_data(temp_file, adapter=adapter)
        
        print("✅ Data loaded successfully")
        print(f"   Individuals: {data_context.n_individuals}")
        print(f"   Occasions: {data_context.n_occasions}")
        
        # Create simple formula
        print("🔧 Creating formula...")
        formula_spec = create_simple_spec(phi="~1", p="~1", f="~1")
        print("✅ Formula created")
        
        # Create model
        print("🔧 Creating Pradel model...")
        model = PradelModel()
        print("✅ Model created")
        
        # Build design matrices - this is where the error likely occurs
        print("🔧 Building design matrices...")
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        print("✅ Design matrices built successfully!")
        
        for param, info in design_matrices.items():
            print(f"   {param}: {info.matrix.shape}")
        
        # Clean up
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Also try to get more context on the JAX error
        if "JAX" in str(e) and "immutable" in str(e):
            print(f"\n🔍 JAX IMMUTABLE ARRAY ERROR DETECTED")
            print("This suggests an in-place assignment operation on a JAX array")
            print("Common causes:")
            print("  1. array[index] = value  # ❌ Wrong")
            print("  2. Should be: array = array.at[index].set(value)  # ✅ Correct")

if __name__ == "__main__":
    test_simple_model_build()