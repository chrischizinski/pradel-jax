#!/usr/bin/env python3
"""
Debug Data Loading Issue
========================

Diagnose the specific JAX string handling issue in data loading.
"""

import pandas as pd
import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path
import traceback

sys.path.append('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

from pradel_jax.data.adapters import load_data

def debug_data_loading():
    """Debug the data loading issue step by step."""
    datasets = [
        ('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax/data/dipper_dataset.csv', 'dipper'),
    ]
    
    for filepath, name in datasets:
        print(f"\n{'='*60}")
        print(f"DEBUGGING DATASET: {name.upper()}")
        print(f"{'='*60}")
        
        try:
            print("Step 1: Loading data...")
            data_context = load_data(filepath)
            print(f"✅ Data loaded successfully")
            print(f"   - Individuals: {data_context.n_individuals}")
            print(f"   - Occasions: {data_context.n_occasions}")
            print(f"   - Covariates: {len(data_context.covariates) if data_context.covariates else 0}")
            
            print("\nStep 2: Examining covariates...")
            if data_context.covariates:
                for cov_name, cov_data in data_context.covariates.items():
                    print(f"   Covariate '{cov_name}':")
                    print(f"     Type: {type(cov_data)}")
                    print(f"     Shape: {cov_data.shape if hasattr(cov_data, 'shape') else 'No shape'}")
                    print(f"     Dtype: {cov_data.dtype if hasattr(cov_data, 'dtype') else 'No dtype'}")
                    
                    # Check if it contains strings
                    if isinstance(cov_data, (list, np.ndarray)) and len(cov_data) > 0:
                        sample = cov_data[0] if hasattr(cov_data, '__getitem__') else cov_data
                        print(f"     Sample value: {sample} (type: {type(sample)})")
                        
                        # Try to check for strings
                        try:
                            if isinstance(sample, str):
                                print(f"     ⚠️  Contains string data!")
                            elif hasattr(cov_data, '__iter__'):
                                has_strings = any(isinstance(x, str) for x in cov_data)
                                if has_strings:
                                    print(f"     ⚠️  Contains string data!")
                        except:
                            print(f"     Could not check for strings")
            
            print("\nStep 3: Testing JAX operations...")
            try:
                # Test basic operations
                capture_sum = jnp.sum(data_context.capture_matrix)
                print(f"   ✅ Capture matrix sum: {capture_sum}")
                
                # Test covariate operations
                if data_context.covariates:
                    for cov_name, cov_data in data_context.covariates.items():
                        try:
                            if isinstance(cov_data, jnp.ndarray):
                                # This should fail if there are strings
                                result = jnp.isnan(cov_data)
                                print(f"   ✅ '{cov_name}' NaN check passed")
                            elif isinstance(cov_data, list):
                                print(f"   ⚠️  '{cov_name}' is a list, not JAX array")
                                # Try to find strings
                                string_items = [item for item in cov_data if isinstance(item, str)]
                                if string_items:
                                    print(f"        Contains strings: {string_items[:5]}...")
                            else:
                                print(f"   ⚠️  '{cov_name}' unexpected type: {type(cov_data)}")
                                
                        except Exception as e:
                            print(f"   ❌ '{cov_name}' failed JAX operation: {str(e)}")
                
            except Exception as e:
                print(f"   ❌ JAX operations failed: {str(e)}")
                traceback.print_exc()
            
            print("\nStep 4: Manual quality assessment...")
            try:
                # This is likely where it fails
                for cov_name, cov_data in data_context.covariates.items():
                    print(f"   Testing '{cov_name}'...")
                    
                    if isinstance(cov_data, list) and any(isinstance(x, str) for x in cov_data):
                        print(f"     ❌ List contains strings - will fail JAX")
                        continue
                        
                    # Try the operation that's failing
                    if hasattr(cov_data, 'shape') and len(cov_data) > 0:
                        try:
                            nan_check = jnp.isnan(cov_data)
                            nan_count = jnp.sum(nan_check)
                            print(f"     ✅ NaN count: {nan_count}")
                        except Exception as e:
                            print(f"     ❌ NaN check failed: {str(e)}")
                            print(f"     Data sample: {cov_data[:5] if len(cov_data) > 5 else cov_data}")
                            
            except Exception as e:
                print(f"   ❌ Manual assessment failed: {str(e)}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"❌ Overall data loading failed: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    debug_data_loading()