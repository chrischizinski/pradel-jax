#!/usr/bin/env python3
"""
Focused Workflow Validation
===========================

Focused validation specifically designed to examine the workflow with proper
adapter selection and identify key statistical and process errors.

This validation targets specific issues without overfitting to data.
"""

import numpy as np
import pandas as pd
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocusedValidator:
    """Focused validation for critical workflow components."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        
    def validate_data_adapter_consistency(self, datasets: List[str], sample_size: int = 1000):
        """Test data adapter consistency and accuracy."""
        
        print("🔧 DATA ADAPTER CONSISTENCY VALIDATION")
        print("=" * 60)
        
        import pradel_jax as pj
        from pradel_jax.data.adapters import GenericFormatAdapter
        
        results = {}
        
        for dataset in datasets:
            print(f"\n📊 Testing dataset: {dataset.upper()}")
            
            # Load dataset
            if dataset == 'nebraska':
                data_file = 'data/encounter_histories_ne_clean.csv'
            elif dataset == 'south_dakota':
                data_file = 'data/encounter_histories_sd_clean.csv'
            else:
                continue
                
            if not Path(data_file).exists():
                print(f"❌ Data file not found: {data_file}")
                continue
            
            # Load sample
            full_data = pd.read_csv(data_file)
            sample_data = full_data.sample(n=min(sample_size, len(full_data)), random_state=42)
            
            print(f"   Sample size: {len(sample_data):,}")
            print(f"   Columns: {len(sample_data.columns)}")
            
            # Test with explicit GenericFormatAdapter
            try:
                temp_file = f"temp_focused_{dataset}.csv"
                sample_data.to_csv(temp_file, index=False)
                
                # Force GenericFormatAdapter
                adapter = GenericFormatAdapter()
                data_context = pj.load_data(temp_file, adapter=adapter)
                
                print(f"   ✅ GenericFormatAdapter succeeded")
                print(f"      Individuals: {data_context.n_individuals}")
                print(f"      Occasions: {data_context.n_occasions}")
                print(f"      Covariates: {len(data_context.covariates)}")
                
                # Test capture matrix properties
                if hasattr(data_context, 'capture_matrix'):
                    cap_matrix = data_context.capture_matrix
                    print(f"      Capture matrix: {cap_matrix.shape}")
                    print(f"      Encounter rate: {np.mean(cap_matrix):.3f}")
                    print(f"      Values range: {np.min(cap_matrix):.0f} to {np.max(cap_matrix):.0f}")
                    
                    # Critical validation: should only contain 0s and 1s
                    unique_vals = np.unique(cap_matrix)
                    valid_encounters = all(v in [0, 1] for v in unique_vals)
                    print(f"      Valid encounter values: {'✅' if valid_encounters else '❌'} {unique_vals}")
                    
                results[dataset] = {
                    'adapter_success': True,
                    'n_individuals': data_context.n_individuals,
                    'n_occasions': data_context.n_occasions, 
                    'n_covariates': len(data_context.covariates),
                    'encounter_rate': float(np.mean(cap_matrix)) if hasattr(data_context, 'capture_matrix') else None,
                    'valid_encounters': valid_encounters if hasattr(data_context, 'capture_matrix') else False
                }
                
                # Clean up
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
            except Exception as e:
                print(f"   ❌ GenericFormatAdapter failed: {e}")
                results[dataset] = {'adapter_success': False, 'error': str(e)}
        
        return results
    
    def validate_model_fitting_consistency(self, datasets: List[str], sample_size: int = 200):
        """Test model fitting consistency across multiple runs."""
        
        print(f"\n🎯 MODEL FITTING CONSISTENCY VALIDATION")
        print("=" * 60)
        
        import pradel_jax as pj
        from pradel_jax.data.adapters import GenericFormatAdapter
        from pradel_jax.models import PradelModel
        from pradel_jax.formulas import create_simple_spec
        from pradel_jax.optimization import optimize_model
        
        results = {}
        
        for dataset in datasets:
            print(f"\n📊 Testing model consistency: {dataset.upper()}")
            
            # Load dataset
            if dataset == 'nebraska':
                data_file = 'data/encounter_histories_ne_clean.csv'
            elif dataset == 'south_dakota':
                data_file = 'data/encounter_histories_sd_clean.csv'
            else:
                continue
                
            if not Path(data_file).exists():
                print(f"❌ Data file not found: {data_file}")
                continue
            
            dataset_results = {
                'runs': [],
                'consistency_metrics': {}
            }
            
            # Multiple runs with same data, different seeds
            for run_id in range(5):
                try:
                    print(f"   🔄 Run {run_id + 1}/5...")
                    
                    # Load sample (same sample each time for consistency testing)
                    full_data = pd.read_csv(data_file)
                    sample_data = full_data.sample(n=min(sample_size, len(full_data)), random_state=42)
                    
                    temp_file = f"temp_consist_{dataset}_{run_id}.csv"
                    sample_data.to_csv(temp_file, index=False)
                    
                    # Load data
                    adapter = GenericFormatAdapter()
                    data_context = pj.load_data(temp_file, adapter=adapter)
                    
                    # Simple intercept-only model for consistency testing
                    formula_spec = create_simple_spec(phi="~1", p="~1", f="~1")
                    model = PradelModel()
                    design_matrices = model.build_design_matrices(formula_spec, data_context)
                    
                    # Objective function
                    def objective(params):
                        return -model.log_likelihood(params, data_context, design_matrices)
                    
                    # Initial parameters (add small random perturbation for each run)
                    np.random.seed(run_id * 123)  # Different seed per run
                    base_params = model.get_initial_parameters(data_context, design_matrices)
                    # JAX arrays are immutable - create new array with perturbation
                    perturbation = np.random.normal(0, 0.01, size=base_params.shape)
                    initial_params = base_params + perturbation  # This creates a new JAX array
                    
                    bounds = model.get_parameter_bounds(data_context, design_matrices)
                    
                    # Clip to bounds (JAX-compatible)
                    import jax.numpy as jnp
                    clipped_params = []
                    for i, (low, high) in enumerate(bounds):
                        clipped_val = jnp.clip(initial_params[i], low, high)
                        clipped_params.append(clipped_val)
                    initial_params = jnp.array(clipped_params)
                    
                    # Optimize
                    start_time = time.time()
                    result = optimize_model(
                        objective_function=objective,
                        initial_parameters=initial_params,
                        context=data_context,
                        bounds=bounds
                    )
                    opt_time = time.time() - start_time
                    
                    if result.success:
                        log_likelihood = -result.result.fun
                        aic = 2 * len(result.result.x) + 2 * result.result.fun
                        
                        run_result = {
                            'run_id': run_id,
                            'success': True,
                            'log_likelihood': float(log_likelihood),
                            'aic': float(aic),
                            'parameters': result.result.x.tolist(),
                            'n_iterations': getattr(result.result, 'nit', 0),
                            'n_function_evals': getattr(result.result, 'nfev', 0),
                            'optimization_time': opt_time,
                            'strategy_used': result.strategy_used
                        }
                        
                        print(f"      ✅ Run {run_id + 1}: LL={log_likelihood:.3f}, AIC={aic:.3f}, Strategy={result.strategy_used}")
                        
                    else:
                        run_result = {
                            'run_id': run_id,
                            'success': False,
                            'error': result.message if hasattr(result, 'message') else 'Unknown error'
                        }
                        print(f"      ❌ Run {run_id + 1} failed: {run_result['error']}")
                    
                    dataset_results['runs'].append(run_result)
                    
                    # Clean up
                    import os
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                except Exception as e:
                    print(f"      ❌ Run {run_id + 1} error: {e}")
                    dataset_results['runs'].append({
                        'run_id': run_id,
                        'success': False,
                        'error': str(e)
                    })
            
            # Analyze consistency
            successful_runs = [r for r in dataset_results['runs'] if r.get('success', False)]
            
            if len(successful_runs) >= 2:
                # Log-likelihood consistency
                log_likelihoods = [r['log_likelihood'] for r in successful_runs]
                ll_std = np.std(log_likelihoods)
                ll_mean = np.mean(log_likelihoods)
                ll_cv = ll_std / abs(ll_mean) if ll_mean != 0 else float('inf')
                
                # Parameter consistency
                param_arrays = np.array([r['parameters'] for r in successful_runs])
                param_stds = np.std(param_arrays, axis=0)
                param_means = np.mean(param_arrays, axis=0)
                param_cvs = []
                for i in range(len(param_means)):
                    if abs(param_means[i]) > 1e-8:
                        param_cvs.append(param_stds[i] / abs(param_means[i]))
                    else:
                        param_cvs.append(param_stds[i])
                
                consistency_metrics = {
                    'success_rate': len(successful_runs) / len(dataset_results['runs']),
                    'll_consistency': ll_cv < 1e-6,  # Very tight consistency for same data
                    'll_cv': ll_cv,
                    'll_range': [min(log_likelihoods), max(log_likelihoods)],
                    'param_consistency': all(cv < 1e-4 for cv in param_cvs),
                    'max_param_cv': max(param_cvs) if param_cvs else 0,
                    'avg_optimization_time': np.mean([r['optimization_time'] for r in successful_runs])
                }
                
                dataset_results['consistency_metrics'] = consistency_metrics
                
                print(f"   📊 Consistency Analysis:")
                print(f"      Success rate: {consistency_metrics['success_rate']:.1%}")
                print(f"      Log-likelihood CV: {ll_cv:.2e} ({'✅' if ll_cv < 1e-6 else '⚠️'})")
                print(f"      Parameter consistency: {'✅' if consistency_metrics['param_consistency'] else '⚠️'}")
                print(f"      Avg optimization time: {consistency_metrics['avg_optimization_time']:.2f}s")
                
            else:
                print(f"   ❌ Insufficient successful runs ({len(successful_runs)}) for consistency analysis")
                dataset_results['consistency_metrics'] = {'success_rate': len(successful_runs) / len(dataset_results['runs'])}
            
            results[dataset] = dataset_results
        
        return results
    
    def validate_parameter_reasonableness(self, datasets: List[str], sample_size: int = 1000):
        """Validate that parameter estimates are biologically reasonable."""
        
        print(f"\n📈 PARAMETER REASONABLENESS VALIDATION")
        print("=" * 60)
        
        import pradel_jax as pj
        from pradel_jax.data.adapters import GenericFormatAdapter
        from pradel_jax.models import PradelModel
        from pradel_jax.formulas import create_simple_spec
        from pradel_jax.optimization import optimize_model
        
        results = {}
        
        # Define reasonable parameter ranges for capture-recapture
        REASONABLE_RANGES = {
            'survival_rate': (0.1, 0.99),      # φ: survival rates
            'detection_rate': (0.01, 0.95),    # p: detection probabilities  
            'recruitment_rate': (0.0, 2.0)     # f: recruitment (can be >1 for growing populations)
        }
        
        for dataset in datasets:
            print(f"\n📊 Testing parameter reasonableness: {dataset.upper()}")
            
            # Load dataset
            if dataset == 'nebraska':
                data_file = 'data/encounter_histories_ne_clean.csv'
            elif dataset == 'south_dakota':
                data_file = 'data/encounter_histories_sd_clean.csv'
            else:
                continue
                
            if not Path(data_file).exists():
                continue
            
            try:
                # Load sample
                full_data = pd.read_csv(data_file)
                sample_data = full_data.sample(n=min(sample_size, len(full_data)), random_state=42)
                
                temp_file = f"temp_param_{dataset}.csv"
                sample_data.to_csv(temp_file, index=False)
                
                # Load data
                adapter = GenericFormatAdapter()
                data_context = pj.load_data(temp_file, adapter=adapter)
                
                # Fit intercept-only model
                formula_spec = create_simple_spec(phi="~1", p="~1", f="~1")
                model = PradelModel()
                design_matrices = model.build_design_matrices(formula_spec, data_context)
                
                def objective(params):
                    return -model.log_likelihood(params, data_context, design_matrices)
                
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                # Optimize
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=data_context,
                    bounds=bounds
                )
                
                if result.success:
                    # Transform logit parameters back to probabilities
                    params = result.result.x
                    
                    # Assuming standard Pradel parameterization: [logit(φ), logit(p), logit(f)]
                    survival_rate = 1 / (1 + np.exp(-params[0]))     # φ
                    detection_rate = 1 / (1 + np.exp(-params[1]))    # p
                    
                    # f (recruitment) is more complex, but logit transform is common
                    recruitment_rate = 1 / (1 + np.exp(-params[2])) if len(params) > 2 else None
                    
                    # Check reasonableness
                    survival_reasonable = (
                        REASONABLE_RANGES['survival_rate'][0] <= survival_rate <= REASONABLE_RANGES['survival_rate'][1]
                    )
                    detection_reasonable = (
                        REASONABLE_RANGES['detection_rate'][0] <= detection_rate <= REASONABLE_RANGES['detection_rate'][1]
                    )
                    
                    recruitment_reasonable = True
                    if recruitment_rate is not None:
                        recruitment_reasonable = (
                            REASONABLE_RANGES['recruitment_rate'][0] <= recruitment_rate <= REASONABLE_RANGES['recruitment_rate'][1]
                        )
                    
                    param_results = {
                        'success': True,
                        'raw_parameters': params.tolist(),
                        'survival_rate': float(survival_rate),
                        'detection_rate': float(detection_rate),
                        'recruitment_rate': float(recruitment_rate) if recruitment_rate is not None else None,
                        'survival_reasonable': survival_reasonable,
                        'detection_reasonable': detection_reasonable,
                        'recruitment_reasonable': recruitment_reasonable,
                        'all_reasonable': survival_reasonable and detection_reasonable and recruitment_reasonable,
                        'log_likelihood': float(-result.result.fun),
                        'aic': float(2 * len(params) + 2 * result.result.fun)
                    }
                    
                    print(f"   📊 Parameter Estimates:")
                    print(f"      Survival rate (φ): {survival_rate:.3f} ({'✅' if survival_reasonable else '❌'})")
                    print(f"      Detection rate (p): {detection_rate:.3f} ({'✅' if detection_reasonable else '❌'})")
                    if recruitment_rate is not None:
                        print(f"      Recruitment rate (f): {recruitment_rate:.3f} ({'✅' if recruitment_reasonable else '❌'})")
                    print(f"      Overall reasonableness: {'✅' if param_results['all_reasonable'] else '❌'}")
                    print(f"      Log-likelihood: {param_results['log_likelihood']:.3f}")
                    
                else:
                    param_results = {
                        'success': False,
                        'error': result.message if hasattr(result, 'message') else 'Optimization failed'
                    }
                    print(f"   ❌ Parameter estimation failed: {param_results['error']}")
                
                results[dataset] = param_results
                
                # Clean up
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
            except Exception as e:
                print(f"   ❌ Parameter validation error: {e}")
                results[dataset] = {'success': False, 'error': str(e)}
        
        return results
    
    def run_focused_validation(self):
        """Run focused validation suite."""
        
        print("🎯 FOCUSED WORKFLOW VALIDATION")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print()
        
        datasets = ['nebraska', 'south_dakota']
        
        # Phase 1: Data adapter consistency  
        adapter_results = self.validate_data_adapter_consistency(datasets, sample_size=1000)
        self.results['data_adapter'] = adapter_results
        
        # Phase 2: Model fitting consistency
        fitting_results = self.validate_model_fitting_consistency(datasets, sample_size=200)
        self.results['model_fitting'] = fitting_results
        
        # Phase 3: Parameter reasonableness
        param_results = self.validate_parameter_reasonableness(datasets, sample_size=1000)
        self.results['parameter_reasonableness'] = param_results
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Generate validation summary."""
        
        print(f"\n📋 FOCUSED VALIDATION SUMMARY")
        print("=" * 60)
        
        # Data adapter summary
        if 'data_adapter' in self.results:
            adapter_successes = sum(1 for r in self.results['data_adapter'].values() 
                                  if r.get('adapter_success', False))
            total_datasets = len(self.results['data_adapter'])
            print(f"📊 Data Adapter: {adapter_successes}/{total_datasets} datasets processed successfully")
        
        # Model fitting summary
        if 'model_fitting' in self.results:
            consistent_datasets = 0
            for dataset, results in self.results['model_fitting'].items():
                if results.get('consistency_metrics', {}).get('success_rate', 0) >= 0.8:
                    consistent_datasets += 1
            total_datasets = len(self.results['model_fitting'])
            print(f"🎯 Model Fitting: {consistent_datasets}/{total_datasets} datasets show consistent convergence")
        
        # Parameter reasonableness summary
        if 'parameter_reasonableness' in self.results:
            reasonable_datasets = sum(1 for r in self.results['parameter_reasonableness'].values()
                                    if r.get('all_reasonable', False))
            total_datasets = len(self.results['parameter_reasonableness'])
            print(f"📈 Parameter Reasonableness: {reasonable_datasets}/{total_datasets} datasets produce reasonable estimates")
        
        print(f"\n✅ Focused validation completed at {self.timestamp}")


def main():
    """Run focused workflow validation."""
    
    validator = FocusedValidator()
    results = validator.run_focused_validation()
    
    return results

if __name__ == "__main__":
    main()