#!/usr/bin/env python3
"""
Large-Scale Nebraska Pradel Model Analysis
Optimized for handling large datasets (10K+ individuals) efficiently.
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from pradel_jax.optimization import optimize_model, ModelCharacteristics, OptimizationStrategy
import sys
import time
import gc
from pathlib import Path
from typing import Optional

def create_efficient_data_loader(data_file: str, sample_size: Optional[int] = None, 
                               chunk_size: int = 10000) -> pd.DataFrame:
    """
    Efficiently load large datasets with memory management.
    """
    print(f"📂 Loading data efficiently from: {data_file}")
    
    # Get total rows first
    total_rows = sum(1 for _ in open(data_file)) - 1  # -1 for header
    print(f"   Total rows in dataset: {total_rows:,}")
    
    if sample_size and sample_size < total_rows:
        print(f"   Sampling {sample_size:,} rows...")
        # Use pandas skiprows for efficient random sampling
        skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                          size=total_rows - sample_size, 
                                          replace=False))
        data = pd.read_csv(data_file, skiprows=skip_rows)
        print(f"   Loaded sample: {len(data):,} rows")
    else:
        print(f"   Loading full dataset...")
        data = pd.read_csv(data_file)
        print(f"   Loaded: {len(data):,} rows")
    
    return data

def process_data_efficiently(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process encounter histories with memory optimization.
    """
    print("🔧 Processing encounter histories efficiently...")
    
    year_columns = ['Y2016', 'Y2017', 'Y2018', 'Y2019', 'Y2020', 'Y2021', 'Y2022', 'Y2023', 'Y2024']
    n_individuals = len(data)
    n_occasions = len(year_columns)
    
    # Pre-allocate arrays for better memory efficiency
    individuals = data['person_id'].values
    encounter_data = np.zeros((n_individuals, n_occasions), dtype=np.int8)  # Use int8 to save memory
    
    # Vectorized processing
    for i, year_col in enumerate(year_columns):
        encounter_data[:, i] = (data[year_col].values > 0).astype(np.int8)
    
    # Create DataFrame efficiently
    occasion_cols = [f'occasion_{i+1}' for i in range(n_occasions)]
    processed_data = pd.DataFrame({
        'individual': individuals,
        **{col: encounter_data[:, i] for i, col in enumerate(occasion_cols)}
    })
    
    # Memory cleanup
    del encounter_data, individuals
    gc.collect()
    
    return processed_data

def get_large_scale_optimizer_config(n_individuals: int) -> dict:
    """
    Get optimizer configuration optimized for dataset size.
    """
    if n_individuals < 1000:
        strategy = OptimizationStrategy.LBFGS_B
        max_iter = 1000
    elif n_individuals < 5000:
        strategy = OptimizationStrategy.HYBRID
        max_iter = 2000
    elif n_individuals < 20000:
        strategy = OptimizationStrategy.MULTI_START
        max_iter = 3000
    else:
        strategy = OptimizationStrategy.JAX_ADAM_ADAPTIVE
        max_iter = 5000
    
    return {
        'preferred_strategy': strategy,
        'max_iterations': max_iter,
        'tolerance': 1e-6 if n_individuals < 10000 else 1e-5
    }

def main():
    """Run large-scale Pradel model analysis on Nebraska data."""
    
    print("🔬 Large-Scale Nebraska Pradel Model Analysis")
    print("=" * 60)
    
    # Configuration
    data_file = "data/encounter_histories_ne_clean.csv"
    
    # Ask user for sample size
    print("📊 Dataset Size Options:")
    print("   1. Small test (1,000 individuals)")
    print("   2. Medium (5,000 individuals)")  
    print("   3. Large (20,000 individuals)")
    print("   4. Extra Large (50,000 individuals)")
    print("   5. Full dataset (all individuals)")
    
    try:
        choice = input("\nSelect option (1-5): ").strip()
        
        size_map = {
            '1': 1000,
            '2': 5000, 
            '3': 20000,
            '4': 50000,
            '5': None  # Full dataset
        }
        
        sample_size = size_map.get(choice, 1000)
        
        if sample_size:
            print(f"Selected: {sample_size:,} individuals")
        else:
            print("Selected: Full dataset")
            
    except KeyboardInterrupt:
        print("\n👋 Analysis cancelled")
        return
    
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        sys.exit(1)
    
    start_time = time.time()
    
    try:
        # Efficient data loading
        np.random.seed(42)  # Reproducible sampling
        sampled_data = create_efficient_data_loader(data_file, sample_size)
        
        # Efficient processing
        processed_data = process_data_efficiently(sampled_data)
        occasions = [col for col in processed_data.columns if col.startswith('occasion_')]
        
        # Memory cleanup
        del sampled_data
        gc.collect()
        
        # Summary statistics
        total_detections = processed_data[occasions].sum().sum()
        total_possible = len(processed_data) * len(occasions)
        never_detected = (processed_data[occasions].sum(axis=1) == 0).sum()
        
        print(f"\n📊 Data Summary:")
        print(f"   - Individuals: {len(processed_data):,}")
        print(f"   - Occasions: {len(occasions)}")
        print(f"   - Total detections: {total_detections:,} / {total_possible:,} ({total_detections/total_possible:.3f})")
        print(f"   - Never detected: {never_detected:,} ({never_detected/len(processed_data):.3f})")
        
        # Save and load into pradel-jax
        temp_file = "temp_nebraska_large_scale.csv"
        print(f"💾 Saving processed data...")
        processed_data.to_csv(temp_file, index=False)
        
        print("🔧 Loading data into pradel-jax...")
        data_context = pj.load_data(temp_file)
        print(f"   ✅ Loaded: {data_context.n_individuals:,} individuals, {data_context.n_occasions} occasions")
        
        # Get optimizer configuration for this dataset size
        optimizer_config = get_large_scale_optimizer_config(data_context.n_individuals)
        print(f"🚀 Using {optimizer_config['preferred_strategy'].value} optimization strategy")
        
        # Model setup
        print("⚡ Setting up Pradel model...")
        model = pj.PradelModel()
        formula_spec = pj.create_simple_spec(phi="~1", p="~1", f="~1", name="Intercept-only")
        
        # Build design matrices
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        print(f"   Parameters to estimate: {len(initial_params)}")
        print(f"   Initial parameters: {initial_params}")
        
        # Objective function with error handling
        def objective(params):
            try:
                ll = model.log_likelihood(params, data_context, design_matrices)
                if np.isnan(ll) or np.isinf(ll):
                    return 1e10
                return -ll
            except Exception:
                return 1e10
        
        # Model characteristics for optimization  
        data_sparsity = 1.0 - (total_detections / total_possible)  # Use previously calculated values
        parameter_ratio = len(initial_params) / data_context.n_individuals
        
        model_characteristics = ModelCharacteristics(
            n_parameters=len(initial_params),
            n_individuals=data_context.n_individuals,
            n_occasions=data_context.n_occasions,
            parameter_ratio=parameter_ratio,
            data_sparsity=data_sparsity,
            temporal_covariates=False,
            has_interactions=False
        )
        
        # Optimization with progress tracking
        print(f"🔥 Fitting model to {data_context.n_individuals:,} individuals...")
        print(f"   Expected time: {estimate_runtime(data_context.n_individuals)} minutes")
        
        fit_start = time.time()
        result = optimize_model(
            objective_function=objective,
            initial_parameters=initial_params,
            context=model_characteristics,
            bounds=bounds,
            preferred_strategy=optimizer_config['preferred_strategy']
        )
        fit_time = time.time() - fit_start
        
        # Results
        print(f"\n🎯 Model Results")
        print("=" * 50)
        print(f"Convergence: {'✅ Success' if result.success else '❌ Failed'}")
        print(f"Optimization time: {fit_time:.1f} seconds")
        
        if result.success:
            print(f"Strategy used: {result.strategy_used}")
            print(f"Function evaluations: {result.n_function_evaluations:,}")
            
            # Model statistics
            ll = -result.final_objective_value
            k = len(result.parameters)
            n = data_context.n_individuals
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            
            print(f"Log-likelihood: {ll:.3f}")
            print(f"AIC: {aic:.3f}")
            print(f"BIC: {bic:.3f}")
            
            # Parameter estimates
            print(f"\nParameter Estimates:")
            param_names = ['Survival φ(intercept)', 'Detection p(intercept)', 'Recruitment f(intercept)']
            for name, value in zip(param_names, result.parameters):
                prob = 1 / (1 + np.exp(-value))
                print(f"  {name}: {value:.4f} (prob: {prob:.4f})")
        
        else:
            print(f"❌ Optimization failed: {result.message}")
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time:.1f} seconds")
        print(f"📊 Processed {data_context.n_individuals:,} individuals successfully")
        
        # Cleanup
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
    except MemoryError:
        print("❌ Memory error: Dataset too large for available RAM")
        print("💡 Try a smaller sample size or upgrade system memory")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        traceback.print_exc()
        sys.exit(1)

def estimate_runtime(n_individuals: int) -> float:
    """Estimate runtime in minutes based on dataset size."""
    if n_individuals < 1000:
        return 0.5
    elif n_individuals < 5000:
        return 2
    elif n_individuals < 20000:
        return 8
    elif n_individuals < 50000:
        return 25
    else:
        return 60

if __name__ == "__main__":
    main()