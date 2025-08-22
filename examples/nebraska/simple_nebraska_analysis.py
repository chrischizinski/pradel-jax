#!/usr/bin/env python3
"""
Simple Nebraska Data Analysis Script
Loads Nebraska data and creates a basic encounter history analysis.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def main():
    """Run basic Nebraska data analysis."""
    
    print("🔬 Nebraska Capture-Recapture Data Analysis")
    print("=" * 50)
    
    # Data file path
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        sys.exit(1)
    
    try:
        # Load the full dataset
        print(f"📂 Loading data from: {data_file}")
        full_data = pd.read_csv(data_file)
        print(f"   Full dataset shape: {full_data.shape}")
        print(f"   Columns: {list(full_data.columns[:10])}...")  # Show first 10 columns
        
        # Random sample of 1000 rows
        sample_size = min(1000, len(full_data))
        print(f"🎲 Randomly sampling {sample_size} rows...")
        np.random.seed(42)
        sampled_data = full_data.sample(n=sample_size, random_state=42)
        
        # Look at the encounter history columns (Y2016-Y2024)
        year_columns = [col for col in sampled_data.columns if col.startswith('Y20')]
        print(f"   Year columns found: {year_columns}")
        
        if year_columns:
            # Extract encounter histories
            encounter_histories = sampled_data[year_columns]
            print(f"   Encounter history shape: {encounter_histories.shape}")
            
            # Basic statistics
            print("\n📊 Basic Statistics:")
            print(f"   - Total individuals: {len(encounter_histories)}")
            print(f"   - Number of occasions: {len(year_columns)}")
            
            # Count detections per individual
            detections_per_individual = encounter_histories.sum(axis=1)
            print(f"   - Individuals never detected: {(detections_per_individual == 0).sum()}")
            print(f"   - Individuals detected once: {(detections_per_individual == 1).sum()}")
            print(f"   - Individuals detected 2+ times: {(detections_per_individual >= 2).sum()}")
            
            # Detection rates per year
            print("\n📈 Detection Rates by Year:")
            for year_col in year_columns:
                detection_rate = encounter_histories[year_col].mean()
                n_detected = encounter_histories[year_col].sum()
                print(f"   {year_col}: {n_detected} individuals ({detection_rate:.3f})")
            
            # Show some example encounter histories
            print("\n🔍 Example Encounter Histories:")
            for i in range(min(5, len(encounter_histories))):
                history = encounter_histories.iloc[i].values
                history_str = ''.join(map(str, history.astype(int)))
                person_id = sampled_data.iloc[i]['person_id'] if 'person_id' in sampled_data.columns else f"Individual_{i}"
                print(f"   {person_id}: {history_str}")
                
            print(f"\n✅ Data exploration completed!")
            print(f"📊 This data has {len(year_columns)} capture occasions from {year_columns[0]} to {year_columns[-1]}")
            print(f"🎯 Ready for Pradel model analysis once data format issues are resolved")
            
        else:
            print("❌ No year columns (Y20XX) found in the data")
            print("Available columns:", list(sampled_data.columns))
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)

if __name__ == "__main__":
    main()