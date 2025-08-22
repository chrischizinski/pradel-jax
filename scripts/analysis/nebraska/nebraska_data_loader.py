#!/usr/bin/env python3
"""
Proper Nebraska data loader that handles the multi-state capture data correctly.

The Nebraska data contains:
- Y2016-Y2024: Capture occasions (0=not captured, 1=captured in tier 1, 2=captured in tier 2)
- gender: M/F  
- age: Age in years
- tier_cat: Tier category (1, 2, etc.)

For the Pradel model, we need to convert multi-state captures (0/1/2) to binary (0/1).
"""

import pandas as pd
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')
import pradel_jax as pj

def load_and_prepare_nebraska_data(n_sample: int = None, random_state: int = 42):
    """
    Load and properly prepare Nebraska data for Pradel modeling.
    
    Args:
        n_sample: Number of individuals to sample (None for all data)
        random_state: Random seed for sampling
        
    Returns:
        tuple: (data_context, original_dataframe)
    """
    print("Loading Nebraska dataset...")
    
    # Load raw data
    df = pd.read_csv('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax/data/encounter_histories_ne_clean.csv')
    print(f"  Loaded {len(df)} total records")
    
    # Filter out records with missing key covariates
    df_clean = df.dropna(subset=['gender', 'age', 'tier_cat']).copy()
    print(f"  After filtering missing data: {len(df_clean)} records")
    
    # Handle gender encoding - convert to binary if needed
    if 'M' in df_clean['gender'].values or 'F' in df_clean['gender'].values:
        df_clean['gender_binary'] = (df_clean['gender'] == 'M').astype(int)
        print("  Converted gender M/F to binary 1/0")
    else:
        df_clean['gender_binary'] = df_clean['gender']
    
    # Sample if requested
    if n_sample is not None and n_sample < len(df_clean):
        df_clean = df_clean.sample(n=n_sample, random_state=random_state)
        print(f"  Sampled {n_sample} individuals")
    
    # Extract capture occasions (Y2016-Y2024)
    year_columns = ['Y2016', 'Y2017', 'Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024']
    
    # Convert multi-state captures to binary (any capture = 1)
    capture_data = df_clean[year_columns].copy()
    
    # Check the range of values
    unique_values = np.unique(capture_data.values)
    print(f"  Unique capture values: {unique_values}")
    
    # Convert to binary: 0 stays 0, any positive value becomes 1
    capture_binary = (capture_data > 0).astype(int)
    
    print(f"  Converted multi-state to binary captures")
    print(f"  Capture occasions: {len(year_columns)} (2016-2024)")
    
    # Create capture history strings
    capture_histories = []
    for idx, row in capture_binary.iterrows():
        ch = ''.join(row.astype(str))
        capture_histories.append(ch)
    
    # Prepare final dataframe for pradel-jax
    final_df = pd.DataFrame({
        'individual_id': range(len(df_clean)),
        'ch': capture_histories,
        'gender': df_clean['gender_binary'].values,
        'age': df_clean['age'].values,
        'tier_cat': df_clean['tier_cat'].values
    })
    
    # Basic statistics
    n_captured = (capture_binary.sum(axis=1) > 0).sum()
    print(f"  Final dataset: {len(final_df)} individuals")
    print(f"  Individuals ever captured: {n_captured} ({n_captured/len(final_df):.1%})")
    
    # Mean captures per individual
    mean_captures = capture_binary.sum(axis=1).mean()
    print(f"  Mean captures per individual: {mean_captures:.2f}")
    
    # Create temporary file for data loading
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    final_df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        data_context = pj.load_data(temp_file.name)
        print(f"  Successfully loaded into pradel-jax context")
        return data_context, df_clean
        
    except Exception as e:
        print(f"  ❌ Failed to load data context: {e}")
        return None, df_clean
        
    finally:
        os.unlink(temp_file.name)

def test_data_loading():
    """Test the data loading function."""
    print("Testing Nebraska data loading...")
    
    # Test with small sample
    data_context, df = load_and_prepare_nebraska_data(n_sample=100, random_state=42)
    
    if data_context is not None:
        print(f"\n✅ Data loading successful!")
        print(f"  Data context: {data_context.n_individuals} individuals, {data_context.n_occasions} occasions")
        print(f"  Available covariates: {list(data_context.covariates.keys())}")
        
        # Show some basic stats
        capture_matrix = data_context.capture_matrix
        total_captures = capture_matrix.sum()
        print(f"  Total captures in matrix: {total_captures}")
        print(f"  Capture rate: {total_captures / (capture_matrix.shape[0] * capture_matrix.shape[1]):.3f}")
        
        return True
    else:
        print(f"\n❌ Data loading failed")
        return False

if __name__ == "__main__":
    test_data_loading()