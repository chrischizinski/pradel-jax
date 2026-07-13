#!/usr/bin/env python3
"""
Comprehensive Capture-Recapture Data Analysis Script
Randomly samples specified number of rows from capture-recapture encounter history data 
and fits all combinations of Pradel models with φ and f covariates, p(1).

Supports multiple datasets:
- Nebraska (NE): encounter_histories_ne_clean.csv
- South Dakota (SD): encounter_histories_sd_clean.csv
"""

import pandas as pd
import numpy as np
import pradel_jax as pj
from pradel_jax.optimization import create_model_specs_from_formulas, fit_models_parallel
from pradel_jax.optimization.parallel import ParallelModelSpec, fit_models_parallel as parallel_fit
from pradel_jax.formulas import create_simple_spec
import sys
import argparse
import time
import gc
from pathlib import Path
from itertools import combinations
import multiprocessing as mp

# Optional bootstrap (best model)
try:
    from pradel_jax.inference.uncertainty import bootstrap_confidence_intervals
except Exception:
    bootstrap_confidence_intervals = None

# Strategy enum
from pradel_jax.optimization.strategy import OptimizationStrategy, OptimizationConfig
from pradel_jax.optimization.optimizers import create_optimizer

# Import our data loaders
sys.path.append('/Users/cchizinski2/gitlab/student_work/ava_britton/pradel-jax')
from nebraska_data_loader import load_and_prepare_nebraska_data
from south_dakota_data_loader import load_and_prepare_south_dakota_data

# Dataset configurations
DATASET_CONFIGS = {
    'nebraska': {
        'file': 'data/20250904_ne_hip_tier_data.csv',
        'name': 'Nebraska',
        'abbrev': 'NE',
        'loader_func': load_and_prepare_nebraska_data,
        'covariates': ['gender', 'age_baseline'],  # Individual-level covariates
        'time_varying': {
            'age': [f'age_t{i}' for i in range(9)],   # Time-varying age
            'tier': [f'tier_t{i}' for i in range(9)]  # Time-varying tier
        }
    },
    'south_dakota': {
        'file': 'data/20250903_sd_hip_tier_data.csv',
        'name': 'South Dakota',
        'abbrev': 'SD',
        'loader_func': load_and_prepare_south_dakota_data,
        'covariates': ['gender', 'age_baseline'],  # Individual-level covariates
        'time_varying': {
            'age': [f'age_t{i}' for i in range(9)],   # Time-varying age
            'tier': [f'tier_t{i}' for i in range(9)]  # Time-varying tier
        }
    }
}

def detect_dataset(data_file_path):
    """Auto-detect which dataset is being used based on file path."""
    data_file_str = str(data_file_path).lower()
    
    if 'ne_clean' in data_file_str or 'nebraska' in data_file_str:
        return 'nebraska'
    elif 'sd_clean' in data_file_str or 'south_dakota' in data_file_str:
        return 'south_dakota'
    else:
        # Default fallback
        return 'nebraska'

def get_available_datasets():
    """Get list of available datasets."""
    available = []
    for dataset_key, config in DATASET_CONFIGS.items():
        if Path(config['file']).exists():
            available.append((dataset_key, config))
    return available

def generate_formula_combinations(dataset_config, include_time_varying=True, max_time_varying=3):
    """Generate all possible combinations of covariates for model formulas including time-varying."""
    formulas = ["~1"]  # Always include intercept-only

    # Individual-level covariates
    individual_covs = dataset_config['covariates']

    # Single individual covariate effects
    for cov in individual_covs:
        formulas.append(f"~1 + {cov}")

    # Two-way combinations of individual covariates
    for cov_pair in combinations(individual_covs, 2):
        formulas.append(f"~1 + {' + '.join(cov_pair)}")

    if include_time_varying:
        # Time-varying covariates (select representative time points to avoid explosion)
        time_varying = dataset_config['time_varying']

        # Select key time points for analysis (start, middle, end)
        key_time_points = [0, 4, 8]  # t0, t4, t8

        for var_type, var_list in time_varying.items():
            # Single time-varying effects at key time points
            for t in key_time_points[:max_time_varying]:
                if t < len(var_list):
                    tv_var = var_list[t]
                    formulas.append(f"~1 + {tv_var}")

                    # Combined with individual covariates
                    for ind_cov in individual_covs[:1]:  # Limit to first individual covariate
                        formulas.append(f"~1 + {ind_cov} + {tv_var}")

    # Remove duplicates and return
    return list(set(formulas))

def main():
    """Run comprehensive Pradel model analysis on capture-recapture data."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Capture-Recapture Pradel Model Analysis')
    parser.add_argument('--dataset', '-d', type=str, choices=['nebraska', 'south_dakota', 'auto'], 
                       default='auto', help='Dataset to analyze (default: auto-detect)')
    parser.add_argument('--data-file', type=str, help='Custom data file path (overrides --dataset)')
    parser.add_argument('--sample-size', '-n', type=int, default=1000,
                       help='Number of individuals to sample (default: 1000, use 0 for full dataset)')
    parser.add_argument('--max-models', type=int, default=64,
                       help='Maximum number of models to fit (default: 64 for all combinations)')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Use parallel processing for large datasets')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='Chunk size for memory-efficient processing (default: 10000)')
    parser.add_argument('--list-datasets', action='store_true',
                       help='List available datasets and exit')

    # Optimization controls
    parser.add_argument(
        '--strategy', type=str, default=None,
        choices=['scipy_lbfgs', 'scipy_slsqp', 'jax_adam', 'hybrid', 'multi_start'],
        help='Optimization strategy to use (default: auto)'
    )
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Maximum number of iterations for optimizer (default: 1000)')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Convergence tolerance (default: 1e-6)')
    parser.add_argument('--multi-start-attempts', type=int, default=5,
                        help='Number of starts for multi_start strategy (default: 5)')

    # Penalty and warm-start options
    parser.add_argument('--penalty', type=str, default='none', choices=['none', 'ridge', 'firth'],
                        help='Apply penalty to stabilize estimation (ridge supported; firth placeholder)')
    parser.add_argument('--lambda-penalty', type=float, default=1e-4,
                        help='Penalty strength for ridge (default: 1e-4)')
    parser.add_argument('--penalty-include-intercept', action='store_true',
                        help='Include intercept terms in ridge penalty (default: exclude)')
    parser.add_argument('--warm-start', type=str, default='intercept', choices=['none', 'intercept'],
                        help='Warm-start strategy for target models (default: intercept)')
    parser.add_argument('--warm-start-iter', type=int, default=300,
                        help='Max iterations for warm-start intercept fit (default: 300)')

    # Boundary behavior priors (to reduce piling up at 0/1 for probabilities)
    parser.add_argument('--boundary-prior', type=str, default='none', choices=['none','jeffreys','barrier'],
                        help='Apply a boundary-aware prior/penalty on φ and p (default: none)')
    parser.add_argument('--boundary-weight', type=float, default=1e-4,
                        help='Weight for boundary prior/penalty (default: 1e-4)')

    # Bootstrap controls
    parser.add_argument('--bootstrap', action='store_true',
                        help='Compute bootstrap CIs for the best model')
    parser.add_argument('--bootstrap-samples', type=int, default=200,
                        help='Number of bootstrap samples for --bootstrap (default: 200)')
    
    # Robust SE computation
    parser.add_argument('--robust-se', action='store_true',
                        help='Use robust SVD-based Hessian SEs and print Fisher condition numbers')
    parser.add_argument('--robust-se-on', type=str, default='top', choices=['none','top','all'],
                        help='Apply robust SEs to none, top models, or all (default: top)')
    parser.add_argument('--robust-se-top-k', type=int, default=5,
                        help='Number of top models (by AIC) to apply robust SEs when --robust-se-on=top')

    # Parallel tuning
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of workers for parallel fitting (default: auto/min(cpu,8))')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for parallel fitting (default: 8)')

    # Aggregation and Firth refinement
    parser.add_argument('--aggregate', type=str, default='none', choices=['none','by_history'],
                        help='Aggregate identical capture histories (valid for intercept-only models)')
    parser.add_argument('--firth-refine', action='store_true',
                        help='Run post-MLE Firth bias-reduction refinement on best model')
    parser.add_argument('--firth-steps', type=int, default=2,
                        help='Number of refinement steps for Firth (default: 2)')
    parser.add_argument('--firth-weight', type=float, default=1.0,
                        help='Weight for Firth penalty (Jeffreys prior) (default: 1.0)')

    # Prefer only time-varying base covariates (e.g., use 'age' not 'age_YYYY')
    parser.add_argument('--prefer-tv-only', action='store_true',
                        help="Prefer only time-varying base covariates when available (exclude single-year 'age_YYYY'/'tier_YYYY')")
    args = parser.parse_args()
    
    # Map strategy argument once for reuse
    selected_strategy_global = None
    if args.strategy:
        try:
            selected_strategy_global = OptimizationStrategy(args.strategy)
        except ValueError:
            selected_strategy_global = None
    
    # Handle list datasets option
    if args.list_datasets:
        print("📊 Available Datasets:")
        print("=" * 40)
        available = get_available_datasets()
        if not available:
            print("❌ No datasets found in data/ directory")
            print("Expected files:")
            for key, config in DATASET_CONFIGS.items():
                print(f"  - {config['file']} ({config['name']})")
        else:
            for dataset_key, config in available:
                file_size = Path(config['file']).stat().st_size / (1024**2)  # MB
                print(f"✅ {config['name']} ({config['abbrev']})")
                print(f"   File: {config['file']}")
                print(f"   Size: {file_size:.1f} MB")
                print(f"   Covariates: {', '.join(config['covariates'])}")
                print()
        sys.exit(0)
    
    # Determine dataset and configuration
    if args.data_file:
        # Custom data file provided
        data_file = args.data_file
        dataset_key = detect_dataset(data_file)
        dataset_config = DATASET_CONFIGS[dataset_key]
        print(f"🔬 Custom Dataset Analysis: {data_file}")
    elif args.dataset == 'auto':
        # Auto-detect from available datasets
        available = get_available_datasets()
        if not available:
            print("❌ No datasets found! Use --list-datasets to see expected files.")
            sys.exit(1)
        dataset_key, dataset_config = available[0]  # Use first available
        data_file = dataset_config['file']
        print(f"🔬 Auto-detected: {dataset_config['name']} Dataset Analysis")
    else:
        # Specific dataset requested
        dataset_key = args.dataset
        dataset_config = DATASET_CONFIGS[dataset_key]
        data_file = dataset_config['file']
        print(f"🔬 {dataset_config['name']} Dataset Analysis")
    
    # Add sample size info
    if args.sample_size == 0:
        print("   📊 Mode: Full Dataset")
    else:
        print(f"   📊 Mode: Random Sample ({args.sample_size:,} individuals)")
    print("=" * 60)
    
    # Check if data file exists
    if not Path(data_file).exists():
        print(f"❌ Error: Data file not found: {data_file}")
        print("Use --list-datasets to see available datasets.")
        sys.exit(1)
    
    try:
        # Load dataset with large-scale optimization
        print(f"📂 Loading data from: {data_file}")
        import pandas as pd  # Re-import to be sure
        
        # Use our updated data loaders for proper time-varying covariate handling
        print(f"🔄 Loading data using {dataset_config['name']} data loader...")

        # Determine sample size for the data loader
        loader_sample_size = None if args.sample_size == 0 else args.sample_size

        # Load data using the appropriate loader function
        data_context, sampled_data = dataset_config['loader_func'](
            n_sample=loader_sample_size,
            random_state=42  # Use fixed seed for reproducibility
        )

        if data_context is None:
            print(f"❌ Failed to load data using {dataset_config['name']} data loader")
            return

        # Update sample size from actual loaded data
        sample_size = len(sampled_data)
        print(f"✅ Loaded {sample_size:,} individuals with time-varying covariates")
        print(f"   Available covariates: {list(data_context.covariates.keys())}")

        # Skip the old pandas loading and preprocessing - our loaders handle it
        
        # Data is already loaded and preprocessed by our data loader
        # No need for additional file conversion or preprocessing
        
        print("   Data summary:")
        print(f"   - Number of individuals: {data_context.n_individuals}")
        print(f"   - Number of occasions: {data_context.n_occasions}")
        if hasattr(data_context, 'covariates') and data_context.covariates:
            # Show only the main covariates (not metadata)
            covariate_keys = list(data_context.covariates.keys())
            main_covariates = [k for k in covariate_keys if not k.endswith('_categories') and not k.endswith('_is_categorical')]
            print(f"   - Available covariates: {main_covariates[:10]}")  # Show first 10
            if len(main_covariates) > 10:
                print(f"     (and {len(main_covariates)-10} more...)")
        
        # Create model specifications for fitting
        print("📊 Setting up comprehensive Pradel model set...")
        
        # Get available covariates (excluding metadata)
        main_covariates = [k for k in data_context.covariates.keys() 
                          if not k.endswith('_categories') and not k.endswith('_is_categorical')]
        
        # Define target covariates for modeling based on dataset configuration
        target_covariates = []
        for cov in dataset_config['covariates']:
            # Prefer time-varying base names when present (e.g., 'age' vs 'age_2017')
            if cov.startswith('age') and 'age' in main_covariates and isinstance(np.array(data_context.covariates['age']), np.ndarray) and np.array(data_context.covariates['age']).ndim == 2:
                if 'age' not in target_covariates:
                    target_covariates.append('age')
                continue
            if cov.startswith('tier') and 'tier' in main_covariates and isinstance(np.array(data_context.covariates['tier']), np.ndarray) and np.array(data_context.covariates['tier']).ndim == 2:
                if 'tier' not in target_covariates:
                    target_covariates.append('tier')
                continue
            if cov in main_covariates and cov not in target_covariates:
                target_covariates.append(cov)
        
        # Handle dataset-specific covariate fallbacks
        if 'tier' not in target_covariates and 'tier_history' in main_covariates:
            # Prefer time-varying tier if present; else fall back to tier_history (static)
            if 'tier' in main_covariates and isinstance(np.array(data_context.covariates['tier']), np.ndarray) and np.array(data_context.covariates['tier']).ndim == 2:
                target_covariates.append('tier')
            else:
                target_covariates.append('tier_history')
        
        # For South Dakota: add age covariates if available
        if dataset_key == 'south_dakota':
            age_vars = [col for col in main_covariates if col.startswith('age_') and col != 'age_baseline']
            for age_var in age_vars[:2]:  # Limit to prevent overparameterization
                if age_var not in target_covariates:
                    target_covariates.append(age_var)
            
        print(f"   Target covariates for modeling: {target_covariates}")

        # If requested, prefer only time-varying base covariates when available
        if args.prefer_tv_only:
            filtered = []
            for cov in target_covariates:
                if cov.startswith('age_') and 'age' in target_covariates:
                    continue
                if cov.startswith('tier_') and 'tier' in target_covariates:
                    continue
                filtered.append(cov)
            target_covariates = filtered
            print(f"   (TV-only) Using covariates: {target_covariates}")
        
        # Generate all formula combinations for survival (φ), detection (p), and recruitment (f)
        # Use our updated function that handles time-varying covariates
        include_tv = not args.prefer_tv_only  # Include time-varying unless specifically requested not to
        phi_formulas = generate_formula_combinations(dataset_config, include_time_varying=include_tv)
        p_formulas = generate_formula_combinations(dataset_config, include_time_varying=include_tv)
        f_formulas = generate_formula_combinations(dataset_config, include_time_varying=include_tv)
        
        # Limit total models to avoid excessive computation
        total_models = len(phi_formulas) * len(f_formulas) * len(p_formulas)
        print(f"   Potential models: φ({len(phi_formulas)}) × f({len(f_formulas)}) × p({len(p_formulas)}) = {total_models}")
        
        if total_models > args.max_models:
            print(f"   ⚠️  Too many models ({total_models}), limiting to {args.max_models}")
            # Prioritize simpler models first
            phi_formulas = phi_formulas[:int(np.sqrt(args.max_models))]
            f_formulas = f_formulas[:int(np.sqrt(args.max_models))]
            total_models = len(phi_formulas) * len(f_formulas)
            print(f"   Reduced to: φ({len(phi_formulas)}) × f({len(f_formulas)}) = {total_models}")
        
        print(f"\n   📋 Survival (φ) formulas ({len(phi_formulas)}):")
        for i, formula in enumerate(phi_formulas, 1):
            print(f"     {i}. φ{formula}")
        
        print(f"\n   📋 Recruitment (f) formulas ({len(f_formulas)}):")
        for i, formula in enumerate(f_formulas, 1):
            print(f"     {i}. f{formula}")
        
        print(f"\n   📋 Detection (p) formula: p~1 (constant)")
        
        if target_covariates:
            print(f"\n   Available covariates: {', '.join(target_covariates)}")
        else:
            print("   No target covariates detected - using intercept-only models")
        
        # Create model specifications using the new API
        from pradel_jax.optimization import optimize_model
        from pradel_jax.models import PradelModel
        
        # Configure optimization for dataset size
        if sample_size >= 50000:
            print("⚡ Fitting models using large-scale optimization (50K+ individuals)...")
            use_large_scale = True
        elif sample_size >= 10000 or args.parallel:
            print("⚡ Fitting models using parallel optimization...")
            use_parallel = True
        else:
            print("⚡ Fitting models using standard optimization...")
            use_large_scale = False
            use_parallel = False
        
        # Create results list to store all model fitting results
        results = []
        total_combinations = len(phi_formulas) * len(f_formulas) * len(p_formulas)
        
        # Performance and memory optimization
        start_time = time.time()
        if sample_size >= 10000:
            print(f"   🧠 Large dataset detected ({sample_size:,} individuals)")
            print(f"   💾 Enabling memory optimization and garbage collection")
            gc.collect()  # Clean memory before starting
        
        # If aggregation requested, only allow when all formulas are intercept-only
        if args.aggregate == 'by_history':
            only_intercepts = (phi_formulas == ["~1"]) and (f_formulas == ["~1"]) and (p_formulas == ["~1"])
            if not only_intercepts:
                print("   ⚠️  Aggregation by history skipped (formulas include covariates)")
            else:
                try:
                    cm = np.array(data_context.capture_matrix)
                    uniq, idx, counts = np.unique(cm, axis=0, return_index=True, return_counts=True)
                    # Rebuild minimal covariates with weights
                    new_covariates = {"weights": jnp.array(counts)}
                    from pradel_jax.data.adapters import CovariateInfo, DataContext as DC
                    cov_info = {"weights": CovariateInfo(name="weights", dtype="float", is_time_varying=False)}
                    data_context = DC(
                        capture_matrix=jnp.array(uniq),
                        covariates=new_covariates,
                        covariate_info=cov_info,
                        n_individuals=uniq.shape[0],
                        n_occasions=cm.shape[1],
                        occasion_names=data_context.occasion_names,
                        individual_ids=None,
                        metadata={"adapter": "AggregatedByHistory"},
                    )
                    print(f"   ✅ Aggregated identical histories: {cm.shape[0]} -> {uniq.shape[0]} rows")
                except Exception as ee:
                    print(f"   ⚠️  Aggregation failed: {ee}")

        # Choose optimization approach based on dataset size
        if 'use_parallel' in locals() and use_parallel and sample_size >= 5000:
            # Parallel processing for medium to large datasets
            print(f"   🚀 Using parallel processing ({mp.cpu_count()} cores available)")
            
            # Create model specifications for parallel processing
            model_specs = []
            model_counter = 0
            for i, phi_formula in enumerate(phi_formulas):
                for j, f_formula in enumerate(f_formulas):
                    for k, p_formula in enumerate(p_formulas):
                        model_name = f"phi{phi_formula}_p{p_formula}_f{f_formula}"
                        formula_spec = create_simple_spec(
                            phi=phi_formula,
                            p=p_formula,
                            f=f_formula
                        )
                        model_specs.append(ParallelModelSpec(
                            name=model_name,
                            formula_spec=formula_spec,
                            index=model_counter
                        ))
                        model_counter += 1
            
            # Run parallel optimization
            try:
                from pradel_jax.optimization.parallel import fit_models_parallel
                parallel_results = fit_models_parallel(
                    model_specs=model_specs,
                    data_context=data_context,
                    n_workers=(args.n_workers or min(mp.cpu_count(), 8)),
                    strategy=selected_strategy_global or OptimizationStrategy.HYBRID,
                    batch_size=max(1, int(args.batch_size)),
                    worker_options={
                        'penalty': args.penalty,
                        'lambda_penalty': args.lambda_penalty,
                        'penalty_include_intercept': args.penalty_include_intercept,
                        'warm_start': args.warm_start,
                        'warm_start_iter': args.warm_start_iter,
                        'boundary_prior': args.boundary_prior,
                        'boundary_weight': args.boundary_weight,
                    },
                )
                
                # Convert parallel results to standard format
                for parallel_result in parallel_results:
                    if parallel_result.success:
                        model_result = type('ModelResult', (), {
                            'success': True,
                            'model_name': parallel_result.model_name,
                            'log_likelihood': parallel_result.log_likelihood,
                            'parameters': parallel_result.parameters,
                            'n_parameters': parallel_result.n_parameters,
                            'strategy_used': parallel_result.strategy_used,
                            'aic': parallel_result.aic,
                            'lambda_mean': None
                        })()
                    else:
                        model_result = type('ModelResult', (), {
                            'success': False,
                            'model_name': parallel_result.model_name,
                            'error_message': parallel_result.error_message
                        })()
                    results.append(model_result)
                    
                print(f"   ✅ Completed {len(results)} models using parallel processing")
                    
            except Exception as e:
                print(f"   ⚠️  Parallel processing failed: {e}")
                print(f"   🔄 Falling back to sequential processing...")
                use_parallel = False
        
        # Sequential processing (fallback or for smaller datasets)
        if not ('use_parallel' in locals() and use_parallel) or len(results) == 0:
            for i, phi_formula in enumerate(phi_formulas):
                for j, f_formula in enumerate(f_formulas):
                    for k, p_formula in enumerate(p_formulas):
                        model_name = f"phi{phi_formula}_p{p_formula}_f{f_formula}"
                        
                        try:
                            # Create formula spec
                            formula_spec = create_simple_spec(
                                phi=phi_formula,
                                p=p_formula,
                                f=f_formula
                            )
                            
                            # Create and fit model
                            model = PradelModel()
                            design_matrices = model.build_design_matrices(formula_spec, data_context)

                            # Helper: build ridge mask to optionally exclude intercepts
                            def _ridge_mask(design_mats):
                                sizes = [design_mats['phi'].parameter_count,
                                         design_mats['p'].parameter_count,
                                         design_mats['f'].parameter_count]
                                mask = np.ones(sum(sizes), dtype=float)
                                if not args.penalty_include_intercept:
                                    # Zero-out intercept positions
                                    idx = 0
                                    for pname in ['phi','p','f']:
                                        cols = design_mats[pname].column_names
                                        if design_mats[pname].has_intercept:
                                            try:
                                                loc = cols.index('(Intercept)')
                                                mask[idx + loc] = 0.0
                                            except ValueError:
                                                pass
                                        idx += design_mats[pname].parameter_count
                                return mask

                            ridge_mask = _ridge_mask(design_matrices)

                            # Define objective function with optional penalty
                            def objective(params):
                                nll = -model.log_likelihood(params, data_context, design_matrices)
                                if args.penalty == 'ridge':
                                    # Quadratic penalty (optionally exclude intercepts)
                                    pen = args.lambda_penalty * float(np.sum((np.asarray(params) ** 2) * ridge_mask))
                                    return nll + pen
                                elif args.penalty == 'firth':
                                    # Placeholder: full Firth inside objective is computationally expensive for large N
                                    # Keeping as base nll for now; robust SE and bootstrap recommended for inference
                                    pass
                                # Optional boundary-aware prior on φ and p to reduce boundary pile-up
                                if args.boundary_prior != 'none' and args.boundary_weight > 0:
                                    # Split params and compute linear predictors for phi and p
                                    sizes = [design_matrices['phi'].parameter_count,
                                             design_matrices['p'].parameter_count,
                                             design_matrices['f'].parameter_count]
                                    phi_params = np.asarray(params[:sizes[0]])
                                    p_params = np.asarray(params[sizes[0]:sizes[0]+sizes[1]])
                                    X_phi = np.asarray(design_matrices['phi'].matrix)
                                    X_p = np.asarray(design_matrices['p'].matrix)
                                    eta_phi = X_phi @ phi_params
                                    eta_p = X_p @ p_params
                                    phi_prob = 1.0 / (1.0 + np.exp(-eta_phi))
                                    p_prob = 1.0 / (1.0 + np.exp(-eta_p))
                                    eps = 1e-12
                                    if args.boundary_prior == 'barrier':
                                        # Log-barrier on probabilities
                                        prior_term = - (np.log(phi_prob*(1-phi_prob) + eps) + np.log(p_prob*(1-p_prob) + eps)).mean()
                                    else:
                                        # 'jeffreys' ~ p^{-1/2}(1-p)^{-1/2} -> penalty ~ -1/2[log p + log(1-p)]
                                        prior_term = -0.5 * (np.log(phi_prob + eps) + np.log(1-phi_prob + eps) +
                                                             np.log(p_prob + eps) + np.log(1-p_prob + eps)).mean()
                                    nll = nll + float(args.boundary_weight) * prior_term
                                return nll

                            # Configure optimization (with optional warm-start intercept)
                            initial_params = model.get_initial_parameters(data_context, design_matrices)

                            if args.warm_start == 'intercept':
                                try:
                                    # Build intercept-only spec and fit quickly
                                    ispec = create_simple_spec(phi='~1', p='~1', f='~1')
                                    dm_i = model.build_design_matrices(ispec, data_context)
                                    init_i = model.get_initial_parameters(data_context, dm_i)
                                    bnd_i = model.get_parameter_bounds(data_context, dm_i)

                                    def obj_i(p):
                                        return -model.log_likelihood(p, data_context, dm_i)

                                    # Quick LBFGS for warm start (few iterations)
                                    from pradel_jax.optimization import optimize_model as optm
                                    warm_resp = optm(
                                        objective_function=obj_i,
                                        initial_parameters=np.array(init_i),
                                        context=data_context,
                                        bounds=bnd_i,
                                        preferred_strategy=selected_strategy_global or OptimizationStrategy.SCIPY_LBFGS,
                                        config_overrides={'max_iter': int(args.warm_start_iter), 'tolerance': max(1e-6, args.tolerance)}
                                    )
                                    if warm_resp.success:
                                        theta_i = np.array(warm_resp.result.x)
                                        # Map intercepts into full start vector
                                        # Determine indices of intercepts in target design matrices
                                        def _segment_lengths(dm):
                                            return [dm['phi'].parameter_count, dm['p'].parameter_count, dm['f'].parameter_count]
                                        seg_i = _segment_lengths(dm_i)
                                        seg_t = _segment_lengths(design_matrices)
                                        # Extract intercepts from theta_i by segments (each segment has 1 param if intercept-only)
                                        phi_i, p_i, f_i = theta_i[0], theta_i[1], theta_i[2]
                                        # Build start for target
                                        start = np.array(initial_params, dtype=float)
                                        idx = 0
                                        # phi
                                        if design_matrices['phi'].has_intercept:
                                            try:
                                                loc = design_matrices['phi'].column_names.index('(Intercept)')
                                                start[idx + loc] = phi_i
                                            except ValueError:
                                                pass
                                        idx += seg_t[0]
                                        # p
                                        if design_matrices['p'].has_intercept:
                                            try:
                                                loc = design_matrices['p'].column_names.index('(Intercept)')
                                                start[idx + loc] = p_i
                                            except ValueError:
                                                pass
                                        idx += seg_t[1]
                                        # f
                                        if design_matrices['f'].has_intercept:
                                            try:
                                                loc = design_matrices['f'].column_names.index('(Intercept)')
                                                start[idx + loc] = f_i
                                            except ValueError:
                                                pass
                                        initial_params = start
                                except Exception as ee:
                                    print(f"      ⚠️  Warm-start failed, using default initials: {ee}")
                            bounds = model.get_parameter_bounds(data_context, design_matrices)

                            # Prepare config overrides
                            config_overrides = {
                                'max_iter': args.max_iter,
                                'tolerance': args.tolerance,
                            }
                            
                            # Optimize
                            if selected_strategy_global == OptimizationStrategy.MULTI_START:
                                # Use direct optimizer to expose n_starts control
                                base_config = OptimizationConfig(max_iter=args.max_iter, tolerance=args.tolerance)
                                optimizer = create_optimizer(
                                    OptimizationStrategy.MULTI_START,
                                    base_config,
                                    n_starts=max(1, int(args.multi_start_attempts)),
                                )
                                opt_res = optimizer.minimize(objective, np.array(initial_params), bounds=bounds)

                                class SimpleResponse:
                                    success = opt_res.success
                                    result = opt_res
                                    strategy_used = 'multi_start'

                                result = SimpleResponse()
                            else:
                                result = optimize_model(
                                    objective_function=objective,
                                    initial_parameters=initial_params,
                                    context=data_context,
                                    bounds=bounds,
                                    preferred_strategy=selected_strategy_global,
                                    config_overrides=config_overrides,
                                )

                            # Convert to compatible result format with statistical inference
                            if result.success:
                                # Generate parameter names from formula specification
                                from pradel_jax.optimization.statistical_inference import generate_parameter_names
                                param_names = generate_parameter_names(formula_spec, data_context)
                                
                                # Set up statistical inference (standard errors, confidence intervals)
                                result.result.set_statistical_info(
                                    param_names, 
                                    data_context.n_individuals, 
                                    objective_function=objective
                                )
                                
                                # Create enhanced model result with statistical inference
                                model_result = type('ModelResult', (), {
                                    'success': True,
                                    'model_name': model_name,
                                    'log_likelihood': -result.result.fun,
                                    'parameters': result.result.x,
                                    'n_parameters': len(result.result.x),
                                    'strategy_used': result.strategy_used,
                                    'aic': result.result.aic,  # Use computed AIC from statistical inference
                                    'bic': result.result.bic,  # BIC available too
                                    'parameter_names': param_names,
                                    'standard_errors': result.result.standard_errors,
                                    'confidence_intervals': result.result.confidence_intervals,
                                    'parameter_summary': result.result.get_parameter_summary(),
                                    'fisher_condition_number': None,
                                    'lambda_mean': None  # Would need to calculate from parameters
                                })()

                                # Optional: robust SE via full FD Hessian + SVD
                                if args.robust_se and args.robust_se_on == 'all':
                                    try:
                                        from pradel_jax.optimization.hessian_utils import compute_finite_difference_hessian_full
                                        H = compute_finite_difference_hessian_full(objective, np.array(result.result.x), eps=1e-5)
                                        fisher = H  # objective = -loglik, so Hessian(objective) ≈ Fisher
                                        # Condition number
                                        cond = float(np.linalg.cond(fisher))
                                        # SVD-based stabilized inverse
                                        U, s, Vt = np.linalg.svd(fisher, full_matrices=False)
                                        floor = max(1e-12, s[0] * 1e-12)
                                        s_reg = np.maximum(s, floor)
                                        fisher_inv = (Vt.T * (1.0 / s_reg)) @ U.T
                                        diag = np.maximum(np.diag(fisher_inv), 1e-12)
                                        robust_se = np.sqrt(diag)

                                        # Update result fields
                                        model_result.standard_errors = robust_se
                                        z = 1.96
                                        lower = result.result.x - z * robust_se
                                        upper = result.result.x + z * robust_se
                                        model_result.confidence_intervals = np.column_stack([lower, upper])
                                        # Parameter summary
                                        summary = {}
                                        for idx, name in enumerate(param_names):
                                            se_i = float(robust_se[idx])
                                            est = float(result.result.x[idx])
                                            summary[name] = {
                                                'estimate': est,
                                                'std_error': se_i,
                                                'ci_lower_95%': float(lower[idx]),
                                                'ci_upper_95%': float(upper[idx]),
                                                'z_score': (est / se_i) if se_i > 0 else None,
                                            }
                                        model_result.parameter_summary = summary
                                        model_result.fisher_condition_number = cond
                                    except Exception as ee:
                                        print(f"      ⚠️  Robust SE computation failed: {ee}")
                            else:
                                model_result = type('ModelResult', (), {
                                    'success': False,
                                    'model_name': model_name,
                                    'error_message': f"Optimization failed: {result.message}"
                                })()
                            
                            results.append(model_result)
                            
                            # Progress indicator with performance info
                            completed = i * len(f_formulas) * len(p_formulas) + j * len(p_formulas) + k + 1
                            elapsed_time = time.time() - start_time
                            avg_time_per_model = elapsed_time / completed
                            remaining_models = total_combinations - completed
                            estimated_remaining = avg_time_per_model * remaining_models
                            
                            status_icon = '✅' if result.success else '❌'
                            print(f"   Progress: {completed}/{total_combinations} - {model_name} {status_icon}")
                            
                            # Show time estimates for large datasets
                            if sample_size >= 10000 and completed % 5 == 0:  # Every 5 models
                                print(f"      ⏱️  Elapsed: {elapsed_time/60:.1f}min, Est. remaining: {estimated_remaining/60:.1f}min")
                            
                            # Memory cleanup for large datasets
                            if sample_size >= 50000 and completed % 10 == 0:  # Every 10 models
                                gc.collect()
                                print(f"      🧹 Memory cleanup performed")
                        
                        except Exception as e:
                            completed = i * len(f_formulas) * len(p_formulas) + j * len(p_formulas) + k + 1
                            print(f"   Progress: {completed}/{total_combinations} - {model_name} ❌ (Error: {str(e)[:50]})")
                            error_result = type('ModelResult', (), {
                                'success': False,
                                'model_name': model_name,
                                'error_message': str(e)
                            })()
                            results.append(error_result)
        
        # Display results for all models
        print(f"\n🎯 Model Results ({len(results)} models fitted)")
        print("=" * 60)
        
        successful_results = [r for r in results if r and r.success]
        if successful_results:
            # Sort by AIC (lower is better)
            successful_results.sort(key=lambda x: x.aic)
            
            print(f"✅ {len(successful_results)} model(s) converged successfully:")
            print()
            
            for i, result in enumerate(successful_results):
                rank_symbol = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                
                print(f"{rank_symbol} {result.model_name}")
                print(f"   Log-likelihood: {result.log_likelihood:.3f}")
                print(f"   AIC: {result.aic:.3f}")
                if hasattr(result, 'bic') and result.bic is not None:
                    print(f"   BIC: {result.bic:.3f}")
                print(f"   Parameters: {result.n_parameters}")
                print(f"   Strategy: {result.strategy_used}")
                if hasattr(result, 'fisher_condition_number') and result.fisher_condition_number is not None:
                    print(f"   Fisher cond.: {result.fisher_condition_number:.2e}")
                
                # Statistical inference information
                if hasattr(result, 'standard_errors') and result.standard_errors is not None:
                    se_range = f"[{np.min(result.standard_errors):.4f}, {np.max(result.standard_errors):.4f}]"
                    print(f"   Standard Error Range: {se_range}")
                
                if hasattr(result, 'parameter_summary') and result.parameter_summary is not None:
                    n_significant = sum(1 for param_info in result.parameter_summary.values() 
                                      if param_info.get('p_value', 1.0) < 0.05)
                    print(f"   Significant Parameters (p<0.05): {n_significant}/{result.n_parameters}")
                
                # Population growth rate information
                if hasattr(result, 'lambda_mean') and result.lambda_mean is not None:
                    print(f"   Lambda (growth rate): {result.lambda_mean:.4f}")
                
                print()  # Blank line between models
            
            # Best model summary with detailed statistical inference
            best_model = successful_results[0]
            print("🏆 Best Model Detailed Summary:")
            print(f"   Model: {best_model.model_name}")
            print(f"   AIC: {best_model.aic:.3f}")
            if hasattr(best_model, 'bic') and best_model.bic is not None:
                print(f"   BIC: {best_model.bic:.3f}")
            print(f"   Log-likelihood: {best_model.log_likelihood:.3f}")
            print(f"   Strategy: {best_model.strategy_used}")
            if hasattr(best_model, 'fisher_condition_number') and best_model.fisher_condition_number is not None:
                print(f"   Fisher cond.: {best_model.fisher_condition_number:.2e}")
            # Boundary proximity warning for best model
            try:
                mdl, spec, dm, obj_fn, init, bnd = _spec_and_obj_from_model_name(best_model.model_name)
                bounds = mdl.get_parameter_bounds(data_context, dm)
                theta = np.array(best_model.parameters)
                tol = 5e-2
                names = []
                for pname in ['phi','p','f']:
                    names.extend([f"{pname}_{c}" for c in dm[pname].column_names])
                near = []
                for idx, (lo, hi) in enumerate(bounds[:len(theta)]):
                    val = theta[idx]
                    if (abs(val - lo) <= tol) or (abs(val - hi) <= tol):
                        nm = names[idx] if idx < len(names) else f"param_{idx}"
                        near.append(nm)
                if near:
                    print(f"   ⚠️  Near-bound parameters: {', '.join(near)}")
            except Exception:
                pass
            
            # Detailed parameter table for best model
            if hasattr(best_model, 'parameter_summary') and best_model.parameter_summary is not None:
                print(f"\n   📋 Parameter Estimates:")
                print(f"   {'Parameter':<15} {'Estimate':<10} {'SE':<10} {'95% CI':<20} {'p-value':<8}")
                print(f"   {'-'*70}")
                
                for param_name, info in best_model.parameter_summary.items():
                    estimate = info.get('estimate', 0)
                    se = info.get('std_error', None)
                    p_val = info.get('p_value', None)
                    
                    # Find 95% CI
                    ci_lower = info.get('ci_lower_95%', None)
                    ci_upper = info.get('ci_upper_95%', None)
                    
                    # Format values
                    est_str = f"{estimate:8.4f}"
                    se_str = f"{se:8.4f}" if se is not None else "   --   "
                    ci_str = f"({ci_lower:6.3f}, {ci_upper:6.3f})" if ci_lower is not None else "      --       "
                    p_str = f"{p_val:.4f}" if p_val is not None else "  --  "
                    
                    # Add significance indicator
                    sig_indicator = " *" if p_val is not None and p_val < 0.05 else ""
                    
                    print(f"   {param_name:<15} {est_str:<10} {se_str:<10} {ci_str:<20} {p_str:<8}{sig_indicator}")
                
                print(f"   {'-'*70}")
                print(f"   * p < 0.05")
            else:
                print(f"   Parameter estimates: {[f'{p:.4f}' for p in best_model.parameters]}")

            # Helpers to (re)construct model spec and objective from a result
            def _spec_and_obj_from_model_name(model_name: str):
                mdl = PradelModel()
                parts = model_name.split('_')
                phi = parts[0].replace('phi', '')
                p = parts[1].replace('p', '')
                f = parts[2].replace('f', '')
                spec = create_simple_spec(phi=phi, p=p, f=f)
                dm = mdl.build_design_matrices(spec, data_context)
                def obj_fn(params):
                    return -mdl.log_likelihood(params, data_context, dm)
                init = mdl.get_initial_parameters(data_context, dm)
                bnd = mdl.get_parameter_bounds(data_context, dm)
                return mdl, spec, dm, obj_fn, init, bnd

            # Apply robust SEs to top-k models if requested (post-fit)
            if args.robust_se and args.robust_se_on == 'top' and len(successful_results) > 0:
                top_k = min(max(1, int(args.robust_se_top_k)), len(successful_results))
                print(f"\n🔧 Computing robust SEs for top {top_k} model(s)...")
                for idx in range(top_k):
                    res = successful_results[idx]
                    try:
                        mdl, spec, dm, obj_fn, init, bnd = _spec_and_obj_from_model_name(res.model_name)
                        from pradel_jax.optimization.hessian_utils import compute_finite_difference_hessian_full
                        H = compute_finite_difference_hessian_full(obj_fn, np.array(res.parameters), eps=1e-5)
                        fisher = H
                        cond = float(np.linalg.cond(fisher))
                        U, s, Vt = np.linalg.svd(fisher, full_matrices=False)
                        floor = max(1e-12, s[0] * 1e-12)
                        s_reg = np.maximum(s, floor)
                        fisher_inv = (Vt.T * (1.0 / s_reg)) @ U.T
                        diag = np.maximum(np.diag(fisher_inv), 1e-12)
                        robust_se = np.sqrt(diag)
                        # Update result entry
                        res.standard_errors = robust_se
                        z = 1.96
                        lower = np.array(res.parameters) - z * robust_se
                        upper = np.array(res.parameters) + z * robust_se
                        res.confidence_intervals = np.column_stack([lower, upper])
                        res.fisher_condition_number = cond
                        print(f"   {idx+1}. {res.model_name} -> Fisher cond.: {cond:.2e}")
                    except Exception as ee:
                        print(f"   {idx+1}. {res.model_name} -> robust SE failed: {ee}")

            # Bootstrap best or top-k models as requested
            if args.bootstrap:
                if bootstrap_confidence_intervals is None:
                    print("\n⚠️  Bootstrap utilities unavailable; skipping --bootstrap")
                else:
                    def _bootstrap_for_model(res):
                        try:
                            mdl, spec, dm, obj_fn, init, bnd = _spec_and_obj_from_model_name(res.model_name)
                            def fit_fn(dc):
                                # rebuild design matrices for bootstrap dc
                                dm2 = mdl.build_design_matrices(spec, dc)
                                init2 = mdl.get_initial_parameters(dc, dm2)
                                bnd2 = mdl.get_parameter_bounds(dc, dm2)
                                def obj2(params):
                                    return -mdl.log_likelihood(params, dc, dm2)
                                if selected_strategy_global == OptimizationStrategy.MULTI_START:
                                    base_cfg = OptimizationConfig(max_iter=args.max_iter, tolerance=args.tolerance)
                                    opt = create_optimizer(OptimizationStrategy.MULTI_START, base_cfg, n_starts=max(1, int(args.multi_start_attempts)))
                                    r = opt.minimize(obj2, np.array(init2), bounds=bnd2)
                                    return np.array(r.x), -float(r.fun)
                                else:
                                    resp = optimize_model(
                                        objective_function=obj2,
                                        initial_parameters=np.array(init2),
                                        context=dc,
                                        bounds=bnd2,
                                        preferred_strategy=selected_strategy_global,
                                        config_overrides={'max_iter': args.max_iter, 'tolerance': args.tolerance},
                                    )
                                    r = resp.result
                                    return np.array(r.x), -float(r.fun)
                            boot = bootstrap_confidence_intervals(
                                data_context, fit_fn, n_bootstrap_samples=int(args.bootstrap_samples)
                            )
                            se_range = (float(np.min(boot.standard_errors)), float(np.max(boot.standard_errors)))
                            print(f"   Bootstrap SE range for {res.model_name}: [{se_range[0]:.4f}, {se_range[1]:.4f}]")
                        except Exception as e:
                            print(f"   ⚠️  Bootstrap failed for {res.model_name}: {e}")

                    if getattr(args, 'bootstrap_on', None) in ('top','best'):
                        pass
                    else:
                        # default behavior retained below
                        args.bootstrap_on = 'best'

                    print("\n🔁 Running bootstrap:")
                    if args.bootstrap_on == 'best':
                        _bootstrap_for_model(best_model)
                    elif args.bootstrap_on == 'top':
                        top_k = min( max(1, int(getattr(args,'bootstrap_top_k', 3))), len(successful_results))
                        print(f"   Top {top_k} models (by AIC)")
                        for idx in range(top_k):
                            _bootstrap_for_model(successful_results[idx])

            # Post-MLE Firth refinement (best model)
            if args.firth_refine and len(successful_results) > 0:
                try:
                    print("\n🔧 Firth refinement on best model (post-MLE)...")
                    res = successful_results[0]
                    mdl, spec, dm, obj_fn, init, bnd = _spec_and_obj_from_model_name(res.model_name)

                    # Build base negative log-likelihood (exclude penalties)
                    def nll_theta(theta):
                        return -mdl.log_likelihood(theta, data_context, dm)

                    # Firth objective: nll + 0.5 * firth_weight * logdet(Fisher)
                    def firth_obj(theta):
                        from pradel_jax.optimization.hessian_utils import compute_finite_difference_hessian_full
                        H = compute_finite_difference_hessian_full(nll_theta, np.array(theta), eps=1e-5)
                        # Fisher ≈ Hessian(nll)
                        U, s, Vt = np.linalg.svd(H, full_matrices=False)
                        # Stabilize small singular values
                        floor = max(1e-12, s[0] * 1e-12)
                        s_reg = np.maximum(s, floor)
                        logdet = float(np.sum(np.log(s_reg)))
                        return float(nll_theta(theta)) + 0.5 * float(args.firth_weight) * logdet

                    # Run a few refinement steps starting from current params
                    theta0 = np.array(res.parameters)
                    from pradel_jax.optimization import optimize_model as optm
                    refine = optm(
                        objective_function=firth_obj,
                        initial_parameters=theta0,
                        context=data_context,
                        bounds=bnd,
                        preferred_strategy=OptimizationStrategy.SCIPY_LBFGS,
                        config_overrides={"max_iter": max(20, int(args.firth_steps) * 10), "tolerance": max(1e-7, args.tolerance)}
                    )
                    if refine.success:
                        refined = np.array(refine.result.x)
                        # Update best result fields
                        res.parameters = refined
                        res.log_likelihood = float(-refine.result.fun)  # approx
                        print("   ✅ Firth refinement complete")
                    else:
                        print("   ⚠️  Firth refinement did not converge; keeping original MLE")
                except Exception as ee:
                    print(f"   ⚠️  Firth refinement failed: {ee}")
                
            # Model comparison summary if multiple models
            if len(successful_results) > 1:
                print(f"\n   📊 Model Support:")
                second_best = successful_results[1]
                delta_aic = second_best.aic - best_model.aic
                print(f"   Δ AIC vs next best: {delta_aic:.3f}")
                if delta_aic > 2:
                    print(f"   Strong evidence for best model (Δ AIC > 2)")
                elif delta_aic > 4:
                    print(f"   Very strong evidence for best model (Δ AIC > 4)")
                elif delta_aic > 10:
                    print(f"   Overwhelming evidence for best model (Δ AIC > 10)")
            
        else:
            print("❌ No models converged successfully")
            if results:
                for result in results:
                    if hasattr(result, 'error_message') and result.error_message:
                        print(f"   Error: {result.error_message}")
            else:
                print("   No result objects returned")
        
        print(f"\n✅ Analysis completed successfully!")
        
        # Comprehensive results export package
        print(f"\n📄 Generating Comprehensive Analysis Package...")
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Full results export (MARK-compatible CSV)
            print(f"   📊 Creating comprehensive results table...")
            dataset_prefix = dataset_config['abbrev'].lower()
            full_results_file = f"{dataset_prefix}_full_results_{sample_size}ind_{timestamp}.csv"
            
            # Create comprehensive results DataFrame with statistical inference
            import pandas as pd
            results_data = []
            for result in results:
                if hasattr(result, 'success') and result.success:
                    # Base model information
                    result_dict = {
                        'model_name': result.model_name,
                        'log_likelihood': result.log_likelihood,
                        'aic': result.aic,
                        'bic': getattr(result, 'bic', None),
                        'n_parameters': result.n_parameters,
                        'strategy_used': result.strategy_used,
                        'success': result.success
                    }
                    
                    # Statistical inference information
                    if hasattr(result, 'standard_errors') and result.standard_errors is not None:
                        result_dict['se_min'] = np.min(result.standard_errors)
                        result_dict['se_max'] = np.max(result.standard_errors)
                        result_dict['se_mean'] = np.mean(result.standard_errors)
                    
                    if hasattr(result, 'parameter_summary') and result.parameter_summary is not None:
                        # Count significant parameters
                        n_significant = sum(1 for info in result.parameter_summary.values() 
                                          if info.get('p_value', 1.0) < 0.05)
                        result_dict['n_significant_params'] = n_significant
                        result_dict['pct_significant'] = n_significant / result.n_parameters * 100
                        
                        # Add individual parameter estimates with names
                        if hasattr(result, 'parameter_names') and result.parameter_names is not None:
                            for i, (param_name, param_value) in enumerate(zip(result.parameter_names, result.parameters)):
                                result_dict[f'param_{param_name}'] = param_value
                                if param_name in result.parameter_summary:
                                    param_info = result.parameter_summary[param_name]
                                    result_dict[f'se_{param_name}'] = param_info.get('std_error', None)
                                    result_dict[f'pval_{param_name}'] = param_info.get('p_value', None)
                                    result_dict[f'ci_lower_{param_name}'] = param_info.get('ci_lower_95%', None)
                                    result_dict[f'ci_upper_{param_name}'] = param_info.get('ci_upper_95%', None)
                    
                    results_data.append(result_dict)
                else:
                    results_data.append({
                        'model_name': getattr(result, 'model_name', 'Unknown'),
                        'log_likelihood': None,
                        'aic': None,
                        'bic': None,
                        'n_parameters': None,
                        'strategy_used': None,
                        'success': False,
                        'error': getattr(result, 'error_message', 'Unknown error')
                    })
            
            export_df = pd.DataFrame(results_data)
            export_df.to_csv(full_results_file, index=False)
            
            # 2. Model comparison table (publication-ready) using statistical inference framework
            print(f"   🏆 Creating model comparison table with statistical inference...")
            successful_df = export_df[export_df['success'] == True].copy()
            if len(successful_df) > 0:
                # Use the comprehensive model comparison from statistical inference framework
                from pradel_jax.optimization.statistical_inference import compare_models, print_model_comparison
                
                # Create model results dictionary for comparison
                model_results_for_comparison = {}
                for result in results:
                    if hasattr(result, 'success') and result.success:
                        model_results_for_comparison[result.model_name] = result
                
                # Generate comprehensive model comparison
                if len(model_results_for_comparison) > 1:
                    comparison_result = compare_models(model_results_for_comparison)
                    
                    # Create enhanced comparison DataFrame
                    successful_df = successful_df.sort_values('aic')
                    successful_df['delta_aic'] = successful_df['aic'] - successful_df['aic'].min()
                    
                    # Add BIC comparisons if available
                    if 'bic' in successful_df.columns and successful_df['bic'].notna().any():
                        successful_df['delta_bic'] = successful_df['bic'] - successful_df['bic'].min()
                    
                    # AIC weights and evidence ratios
                    successful_df['aic_weight'] = np.exp(-0.5 * successful_df['delta_aic'])
                    successful_df['aic_weight'] = successful_df['aic_weight'] / successful_df['aic_weight'].sum()
                    successful_df['substantial_support'] = successful_df['delta_aic'] <= 2.0
                    successful_df['strong_support'] = successful_df['delta_aic'] <= 4.0
                    successful_df['very_strong_support'] = successful_df['delta_aic'] <= 7.0
                    
                    # Evidence ratio vs best model
                    if len(successful_df) > 1:
                        best_weight = successful_df.iloc[0]['aic_weight']
                        successful_df['evidence_ratio'] = best_weight / successful_df['aic_weight']
                    
                    comparison_df = successful_df

                # Enrich comparison with diagnostics: Fisher condition (if computed) and boundary proximity
                def _near_bound_info(model_name: str):
                    try:
                        mdl, spec, dm, obj_fn, init, bnd = _spec_and_obj_from_model_name(model_name)
                        bounds = mdl.get_parameter_bounds(data_context, dm)
                        res = next((x for x in results if getattr(x, 'success', False) and getattr(x, 'model_name','') == model_name), None)
                        if res is None or not hasattr(res, 'parameters'):
                            return False, 0, ''
                        theta = np.array(res.parameters)
                        near_names = []
                        tol = 5e-2  # link-scale proximity tolerance
                        # Construct param names
                        names = []
                        for pname in ['phi','p','f']:
                            names.extend([f"{pname}_{c}" for c in dm[pname].column_names])
                        for idx, (lo, hi) in enumerate(bounds[:len(theta)]):
                            val = theta[idx]
                            if (abs(val - lo) <= tol) or (abs(val - hi) <= tol):
                                nm = names[idx] if idx < len(names) else f"param_{idx}"
                                near_names.append(nm)
                        return len(near_names) > 0, len(near_names), ";".join(near_names)
                    except Exception:
                        return False, 0, ''

                # Fisher condition numbers mapped if available
                fisher_map = {}
                for r in results:
                    if getattr(r, 'success', False) and hasattr(r, 'fisher_condition_number') and r.fisher_condition_number is not None:
                        fisher_map[r.model_name] = r.fisher_condition_number
                comparison_df['fisher_condition'] = comparison_df['model_name'].map(fisher_map)

                # Boundary flags
                nb_tuple = comparison_df['model_name'].apply(lambda m: _near_bound_info(m))
                comparison_df['near_bound_any'] = nb_tuple.apply(lambda t: t[0])
                comparison_df['near_bound_count'] = nb_tuple.apply(lambda t: t[1])
                comparison_df['near_bound_params'] = nb_tuple.apply(lambda t: t[2])
                
                comparison_file = f"{dataset_prefix}_model_comparison_{sample_size}ind_{timestamp}.csv"
                comparison_df.to_csv(comparison_file, index=False)
            else:
                import pandas as pd
                comparison_df = pd.DataFrame()  # Empty comparison if no successful models
            
            # 3. Detailed parameter tables with statistical inference
            print(f"   📋 Creating comprehensive parameter analysis...")
            if len(comparison_df) > 0:
                # A. Model selection summary
                model_cols = ['model_name', 'aic', 'bic', 'n_parameters', 'delta_aic', 'aic_weight', 'evidence_ratio', 'substantial_support', 'strong_support', 'n_significant_params', 'pct_significant']
                available_model_cols = [col for col in model_cols if col in comparison_df.columns]
                model_summary = comparison_df[available_model_cols].copy()
                model_summary_file = f"{dataset_prefix}_model_selection_{sample_size}ind_{timestamp}.csv"
                model_summary.to_csv(model_summary_file, index=False)
                
                # B. Best model parameter details (ensure it matches best AIC model)
                best_model_name = comparison_df.iloc[0]['model_name'] if len(comparison_df) > 0 else None
                best_result = None
                for r in results:
                    if getattr(r, 'success', False) and getattr(r, 'model_name', '') == best_model_name:
                        best_result = r
                        break
                # Build summary if missing or unreliable
                def _needs_robust_summary(res):
                    if res is None or not hasattr(res, 'parameter_summary') or not res.parameter_summary:
                        return True
                    try:
                        ses = [v.get('std_error', None) for v in res.parameter_summary.values()]
                        if any((s is None) or (isinstance(s, (int,float)) and s >= 1e3) for s in ses):
                            return True
                    except Exception:
                        return True
                    return False

                if best_result is not None and _needs_robust_summary(best_result):
                    try:
                        mdl, spec, dm, obj_fn, init, bnd = _spec_and_obj_from_model_name(best_result.model_name)
                        # Compute robust SEs for best model
                        from pradel_jax.optimization.hessian_utils import compute_finite_difference_hessian_full
                        theta = np.array(best_result.parameters)
                        H = compute_finite_difference_hessian_full(obj_fn, theta, eps=1e-5)
                        U, s, Vt = np.linalg.svd(H, full_matrices=False)
                        floor = max(1e-12, s[0] * 1e-12)
                        s_reg = np.maximum(s, floor)
                        H_inv = (Vt.T * (1.0 / s_reg)) @ U.T
                        diag = np.maximum(np.diag(H_inv), 1e-12)
                        se = np.sqrt(diag)
                        z = 1.96
                        lower = theta - z * se
                        upper = theta + z * se
                        # Parameter names from design matrices
                        param_names = []
                        for pname in ['phi','p','f']:
                            cols = dm[pname].column_names
                            param_names.extend([f"{pname}_{c}" for c in cols])
                        summary = {}
                        for idx, name in enumerate(param_names[:len(theta)]):
                            summary[name] = {
                                'estimate': float(theta[idx]),
                                'std_error': float(se[idx]),
                                'z_score': float(theta[idx]/se[idx]) if se[idx] > 0 else None,
                                'ci_lower_95%': float(lower[idx]),
                                'ci_upper_95%': float(upper[idx]),
                            }
                        best_result.parameter_summary = summary
                    except Exception as ee:
                        print(f"   ⚠️  Could not compute parameter summary for best model: {ee}")

                if best_result is not None and hasattr(best_result, 'parameter_summary') and best_result.parameter_summary:
                    param_details_data = []
                    for param_name, param_info in best_result.parameter_summary.items():
                        param_details_data.append({
                            'parameter': param_name,
                            'estimate': param_info.get('estimate', None),
                            'std_error': param_info.get('std_error', None),
                            'z_score': param_info.get('z_score', None),
                            'p_value': param_info.get('p_value', None),
                            'ci_lower_95': param_info.get('ci_lower_95%', None),
                            'ci_upper_95': param_info.get('ci_upper_95%', None),
                            'significant_05': param_info.get('p_value', 1.0) < 0.05,
                            'significant_01': param_info.get('p_value', 1.0) < 0.01,
                            'model_name': best_result.model_name
                        })

                    param_details_df = pd.DataFrame(param_details_data)
                    param_details_file = f"{dataset_prefix}_best_model_parameters_{sample_size}ind_{timestamp}.csv"
                    param_details_df.to_csv(param_details_file, index=False)
                else:
                    param_details_file = None
                
                # C. Legacy parameter summary for compatibility
                param_cols = ['model_name', 'aic', 'n_parameters', 'delta_aic', 'aic_weight', 'substantial_support']
                available_cols = [col for col in param_cols if col in comparison_df.columns]
                param_summary = comparison_df[available_cols].copy()
                param_summary_file = f"{dataset_prefix}_parameters_{sample_size}ind_{timestamp}.csv"
                param_summary.to_csv(param_summary_file, index=False)
            else:
                model_summary_file = None
                param_details_file = None
                param_summary_file = None
            
            # 4. Print enhanced analysis summary
            print(f"\n📊 {dataset_config['name']} Capture-Recapture Analysis Summary")
            print(f"=" * 60)
            print(f"Dataset: {sample_size} individuals, {data_context.n_occasions} occasions")
            print(f"Analysis completed: {timestamp}")
            print(f"Models fitted: {len(results)} ({len([r for r in results if r.success])} successful)")
            
            if len(comparison_df) > 0:
                print(f"\n🏆 Model Selection Results:")
                best_model = comparison_df.iloc[0]
                print(f"   Best Model: {best_model['model_name']}")
                print(f"   AIC: {best_model['aic']:.3f}")
                print(f"   AIC Weight: {best_model['aic_weight']:.3f}")
                if len(comparison_df) > 1:
                    second_best_weight = comparison_df.iloc[1]['aic_weight']
                    if second_best_weight > 0:
                        evidence_ratio = best_model['aic_weight'] / second_best_weight
                        print(f"   Evidence Ratio: {evidence_ratio:.1f}x better than next model")
                    else:
                        print(f"   Evidence Ratio: >1000x better than next model (extreme support)")
                
                # Model support summary
                substantial_models = comparison_df[comparison_df['substantial_support'] == True]
                print(f"\n🎯 Model Support Summary:")
                print(f"   Models with substantial support (ΔAIC ≤ 2): {len(substantial_models)}")
                print(f"   Top model cumulative weight: {comparison_df['aic_weight'].iloc[0]:.3f}")
            else:
                print(f"\n❌ No models converged successfully - no model comparison available")
                
            # 5. Files generated summary with statistical inference
            print(f"\n📁 Generated Files with Statistical Inference:")
            print(f"   📊 Full Results: {full_results_file}")
            print(f"       • Complete parameter estimates, standard errors, confidence intervals")
            print(f"       • Statistical significance tests and p-values")
            print(f"       • All optimization metadata and diagnostics")
            
            if len(comparison_df) > 0:
                print(f"   🏆 Model Selection: {model_summary_file}")
                print(f"       • AIC/BIC rankings with evidence ratios")
                print(f"       • Statistical support indicators and weights")
                print(f"       • Parameter significance summaries")
                
                print(f"   🏆 Model Comparison: {comparison_file}")
                print(f"       • Detailed model comparison with statistical tests")
                print(f"       • Publication-ready table with all criteria")
                
                if param_details_file:
                    print(f"   📋 Best Model Parameters: {param_details_file}")
                    print(f"       • Detailed parameter table with standard errors")
                    print(f"       • 95% confidence intervals and significance tests")
                    print(f"       • Z-scores and p-values for hypothesis testing")
                
                if param_summary_file:
                    print(f"   📋 Parameter Summary: {param_summary_file}")
                    print(f"       • Quick reference parameter table (legacy format)")
            else:
                print(f"   ⚠️  No statistical inference files generated (no successful fits)")
            
            # Performance Summary
            total_time = time.time() - start_time
            successful_models = len([r for r in results if r.success])
            print(f"\n📊 Performance Summary:")
            print(f"   ⏱️  Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
            print(f"   💾 Dataset size: {sample_size:,} individuals, {data_context.n_occasions} occasions")
            print(f"   🏆 Models fitted: {len(results)} total, {successful_models} successful ({successful_models/len(results)*100:.1f}%)")
            print(f"   ⚡ Average time per model: {total_time/len(results):.1f} seconds")
            if sample_size >= 10000:
                individuals_per_second = sample_size * len(results) / total_time
                print(f"   🚀 Processing rate: {individuals_per_second:,.0f} individual-models per second")
            
            print(f"\n✅ Complete analysis package generated!")
            print(f"🔬 Ready for publication, further analysis, or RMark comparison")
            
        except Exception as e:
            print(f"⚠️  Export package generation failed: {e}")
            print(f"📄 Basic results still available in result objects")
            import traceback
            traceback.print_exc()
        
        # Clean up temporary file
        import os
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
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
