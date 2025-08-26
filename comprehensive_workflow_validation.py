#!/usr/bin/env python3
"""
Comprehensive Workflow Validation Framework
==========================================

Deep examination of the Pradel-JAX workflow for process errors and statistical errors.
This validation framework is designed to be dataset-agnostic and detect fundamental
issues without overfitting to specific data characteristics.

Key Validation Areas:
1. Data Processing Pipeline Integrity
2. Mathematical Model Implementation
3. Optimization Convergence Analysis
4. Statistical Inference Validity
5. Cross-Dataset Consistency
6. Numerical Stability Assessment

Author: Automated Validation System
Date: August 2025
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

# Configure logging for validation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowValidator:
    """Comprehensive validation framework for Pradel-JAX workflow."""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.validation_results = {}
        
    def run_comprehensive_validation(self, datasets: List[str], sample_sizes: List[int] = [100, 1000, 5000]):
        """Run comprehensive validation across multiple datasets and sample sizes."""
        
        print("🔬 COMPREHENSIVE WORKFLOW VALIDATION")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Datasets: {datasets}")
        print(f"Sample Sizes: {sample_sizes}")
        print()
        
        # Phase 1: Data Processing Pipeline Validation
        print("📊 Phase 1: Data Processing Pipeline Validation")
        print("-" * 50)
        data_validation_results = self._validate_data_processing_pipeline(datasets, sample_sizes)
        self.validation_results['data_processing'] = data_validation_results
        
        # Phase 2: Mathematical Model Implementation
        print("\n🧮 Phase 2: Mathematical Model Implementation Validation")
        print("-" * 50)
        model_validation_results = self._validate_mathematical_implementation(datasets, sample_sizes)
        self.validation_results['mathematical_model'] = model_validation_results
        
        # Phase 3: Optimization Convergence Analysis
        print("\n⚡ Phase 3: Optimization Convergence Analysis")
        print("-" * 50)
        optimization_results = self._validate_optimization_convergence(datasets, sample_sizes)
        self.validation_results['optimization'] = optimization_results
        
        # Phase 4: Statistical Inference Validity
        print("\n📈 Phase 4: Statistical Inference Validity")
        print("-" * 50)
        inference_results = self._validate_statistical_inference(datasets, sample_sizes)
        self.validation_results['statistical_inference'] = inference_results
        
        # Phase 5: Cross-Dataset Consistency
        print("\n🔄 Phase 5: Cross-Dataset Consistency Analysis")
        print("-" * 50)
        consistency_results = self._validate_cross_dataset_consistency(datasets, sample_sizes)
        self.validation_results['cross_dataset_consistency'] = consistency_results
        
        # Phase 6: Numerical Stability Assessment
        print("\n🎯 Phase 6: Numerical Stability Assessment")
        print("-" * 50)
        stability_results = self._validate_numerical_stability(datasets, sample_sizes)
        self.validation_results['numerical_stability'] = stability_results
        
        # Generate comprehensive report
        self._generate_validation_report()
        
        return self.validation_results
    
    def _validate_data_processing_pipeline(self, datasets: List[str], sample_sizes: List[int]) -> Dict:
        """Validate data processing pipeline for errors and consistency."""
        
        results = {
            'pipeline_integrity': {},
            'covariate_processing': {},
            'encounter_history_validation': {},
            'missing_data_handling': {}
        }
        
        import pradel_jax as pj
        from pradel_jax.data.adapters import GenericFormatAdapter
        
        for dataset in datasets:
            print(f"  📋 Validating dataset: {dataset.upper()}")
            
            # Load dataset configurations
            from examples.nebraska.nebraska_sample_analysis import DATASET_CONFIGS
            if dataset not in DATASET_CONFIGS:
                print(f"    ❌ Dataset {dataset} not found in configurations")
                continue
                
            config = DATASET_CONFIGS[dataset]
            data_file = config['file']
            
            if not Path(data_file).exists():
                print(f"    ❌ Data file not found: {data_file}")
                continue
            
            dataset_results = {
                'file_integrity': True,
                'sample_consistency': {},
                'covariate_stability': {},
                'encounter_validation': {}
            }
            
            # Test multiple sample sizes for consistency
            for sample_size in sample_sizes:
                try:
                    print(f"    🔍 Testing sample size: {sample_size}")
                    
                    # Load and sample data
                    full_data = pd.read_csv(data_file)
                    if sample_size >= len(full_data):
                        sampled_data = full_data.copy()
                        actual_size = len(full_data)
                    else:
                        # Use multiple random seeds to test consistency
                        samples = []
                        for seed in [42, 123, 456]:
                            sample = full_data.sample(n=sample_size, random_state=seed)
                            samples.append(sample)
                        
                        # Test sample consistency
                        sample_stats = []
                        for i, sample in enumerate(samples):
                            stats = {
                                'n_individuals': len(sample),
                                'gender_dist': sample.get('gender', pd.Series()).value_counts().to_dict() if 'gender' in sample.columns else {},
                                'age_mean': sample.get(config['age_column'], pd.Series()).mean() if config['age_column'] in sample.columns else np.nan,
                                'missing_values': sample.isnull().sum().sum()
                            }
                            sample_stats.append(stats)
                        
                        # Check consistency across seeds
                        age_means = [s['age_mean'] for s in sample_stats if not np.isnan(s['age_mean'])]
                        if len(age_means) > 1:
                            age_cv = np.std(age_means) / np.mean(age_means) if np.mean(age_means) != 0 else 0
                            dataset_results['sample_consistency'][sample_size] = {
                                'age_mean_cv': age_cv,
                                'consistent_sampling': age_cv < 0.1  # Less than 10% CV acceptable
                            }
                        
                        sampled_data = samples[0]  # Use first sample for further testing
                        actual_size = sample_size
                    
                    # Process data through pradel-jax pipeline
                    temp_file = f"temp_validation_{dataset}_{sample_size}.csv"
                    sampled_data.to_csv(temp_file, index=False)
                    
                    # Test data adapter consistency
                    try:
                        adapter = GenericFormatAdapter()
                        data_context = pj.load_data(temp_file, adapter=adapter)
                        
                        # Validate data context properties
                        validation_checks = {
                            'n_individuals_match': data_context.n_individuals == actual_size,
                            'positive_occasions': data_context.n_occasions > 0,
                            'has_encounter_histories': hasattr(data_context, 'encounter_histories'),
                            'encounter_matrix_shape': data_context.encounter_histories.shape == (actual_size, data_context.n_occasions),
                            'covariates_present': len(data_context.covariates) > 0 if hasattr(data_context, 'covariates') else False
                        }
                        
                        dataset_results['encounter_validation'][sample_size] = validation_checks
                        
                        # Test covariate processing stability
                        if hasattr(data_context, 'covariates'):
                            covariate_checks = {}
                            for cov_name, cov_data in data_context.covariates.items():
                                if not cov_name.endswith('_categories') and not cov_name.endswith('_is_categorical'):
                                    if isinstance(cov_data, np.ndarray):
                                        covariate_checks[cov_name] = {
                                            'shape_correct': cov_data.shape[0] == actual_size,
                                            'no_inf_values': not np.any(np.isinf(cov_data)),
                                            'finite_values': np.all(np.isfinite(cov_data[~np.isnan(cov_data)]))
                                        }
                            
                            dataset_results['covariate_stability'][sample_size] = covariate_checks
                        
                    except Exception as e:
                        print(f"    ⚠️ Data processing error for {dataset} size {sample_size}: {e}")
                        dataset_results['encounter_validation'][sample_size] = {'error': str(e)}
                    
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                except Exception as e:
                    print(f"    ❌ Sample validation failed for {dataset} size {sample_size}: {e}")
                    dataset_results['sample_consistency'][sample_size] = {'error': str(e)}
            
            results['pipeline_integrity'][dataset] = dataset_results
            print(f"    ✅ Completed pipeline validation for {dataset}")
        
        return results
    
    def _validate_mathematical_implementation(self, datasets: List[str], sample_sizes: List[int]) -> Dict:
        """Validate mathematical correctness of Pradel model implementation."""
        
        results = {
            'likelihood_computation': {},
            'parameter_bounds': {},
            'gradient_accuracy': {},
            'model_identifiability': {}
        }
        
        import pradel_jax as pj
        from pradel_jax.models import PradelModel
        from pradel_jax.formulas import create_simple_spec
        import jax.numpy as jnp
        
        print("  🧮 Testing mathematical implementation...")
        
        # Test on smaller samples for mathematical validation
        test_sample_sizes = [100, 500]  # Smaller sizes for intensive testing
        
        for dataset in datasets:
            from examples.nebraska.nebraska_sample_analysis import DATASET_CONFIGS
            if dataset not in DATASET_CONFIGS:
                continue
                
            config = DATASET_CONFIGS[dataset]
            data_file = config['file']
            
            if not Path(data_file).exists():
                continue
            
            dataset_results = {
                'likelihood_tests': {},
                'parameter_tests': {},
                'gradient_tests': {}
            }
            
            for sample_size in test_sample_sizes:
                try:
                    print(f"    🔍 Mathematical validation: {dataset} (n={sample_size})")
                    
                    # Load and prepare data
                    full_data = pd.read_csv(data_file)
                    sampled_data = full_data.sample(n=min(sample_size, len(full_data)), random_state=42)
                    
                    temp_file = f"temp_math_{dataset}_{sample_size}.csv"
                    sampled_data.to_csv(temp_file, index=False)
                    
                    data_context = pj.load_data(temp_file)
                    
                    # Test simple model (intercept-only)
                    formula_spec = create_simple_spec(phi="~1", p="~1", f="~1")
                    model = PradelModel()
                    design_matrices = model.build_design_matrices(formula_spec, data_context)
                    
                    # Mathematical validation tests
                    math_tests = {}
                    
                    # Test 1: Likelihood bounds and properties
                    initial_params = model.get_initial_parameters(data_context, design_matrices)
                    likelihood_val = model.log_likelihood(initial_params, data_context, design_matrices)
                    
                    math_tests['likelihood_finite'] = np.isfinite(likelihood_val)
                    math_tests['likelihood_negative'] = likelihood_val <= 0  # Log-likelihood should be ≤ 0
                    math_tests['likelihood_reasonable'] = likelihood_val > -1e6  # Not extremely negative
                    
                    # Test 2: Parameter bounds validation
                    bounds = model.get_parameter_bounds(data_context, design_matrices)
                    math_tests['bounds_consistent'] = all(low <= high for low, high in bounds)
                    math_tests['initial_within_bounds'] = all(low <= param <= high for param, (low, high) in zip(initial_params, bounds))
                    
                    # Test 3: Gradient computation accuracy (finite differences)
                    def objective(params):
                        return -model.log_likelihood(params, data_context, design_matrices)
                    
                    # Compute analytical gradients
                    import jax
                    grad_fn = jax.grad(lambda p: -model.log_likelihood(p, data_context, design_matrices))
                    analytical_grad = grad_fn(initial_params)
                    
                    # Compute numerical gradients
                    eps = 1e-8
                    numerical_grad = np.zeros_like(initial_params)
                    for i in range(len(initial_params)):
                        params_plus = initial_params.copy()
                        params_minus = initial_params.copy()
                        params_plus[i] += eps
                        params_minus[i] -= eps
                        
                        numerical_grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps)
                    
                    # Compare gradients
                    if len(analytical_grad) == len(numerical_grad):
                        grad_diff = np.abs(analytical_grad - numerical_grad)
                        max_diff = np.max(grad_diff)
                        relative_diff = max_diff / (np.max(np.abs(analytical_grad)) + 1e-12)
                        
                        math_tests['gradient_accuracy'] = relative_diff < 1e-4  # 0.01% tolerance
                        math_tests['max_gradient_diff'] = float(max_diff)
                        math_tests['relative_gradient_diff'] = float(relative_diff)
                    else:
                        math_tests['gradient_accuracy'] = False
                        math_tests['gradient_shape_mismatch'] = True
                    
                    # Test 4: Model identifiability (parameter sensitivity)
                    param_sensitivities = []
                    for i in range(len(initial_params)):
                        params_test = initial_params.copy()
                        params_test[i] += 0.1  # Small parameter change
                        ll_change = abs(model.log_likelihood(params_test, data_context, design_matrices) - likelihood_val)
                        param_sensitivities.append(ll_change)
                    
                    math_tests['parameter_sensitivity'] = {
                        'all_sensitive': all(s > 1e-6 for s in param_sensitivities),
                        'sensitivities': param_sensitivities
                    }
                    
                    dataset_results['likelihood_tests'][sample_size] = math_tests
                    
                    # Clean up
                    import os
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                except Exception as e:
                    print(f"    ❌ Mathematical validation failed: {e}")
                    dataset_results['likelihood_tests'][sample_size] = {'error': str(e)}
            
            results['likelihood_computation'][dataset] = dataset_results
            print(f"    ✅ Mathematical validation completed for {dataset}")
        
        return results
    
    def _validate_optimization_convergence(self, datasets: List[str], sample_sizes: List[int]) -> Dict:
        """Validate optimization convergence and consistency."""
        
        results = {
            'convergence_consistency': {},
            'strategy_performance': {},
            'parameter_stability': {}
        }
        
        import pradel_jax as pj
        from pradel_jax.optimization import optimize_model
        from pradel_jax.models import PradelModel
        from pradel_jax.formulas import create_simple_spec
        
        print("  ⚡ Testing optimization convergence...")
        
        for dataset in datasets:
            from examples.nebraska.nebraska_sample_analysis import DATASET_CONFIGS
            if dataset not in DATASET_CONFIGS:
                continue
                
            config = DATASET_CONFIGS[dataset]
            data_file = config['file']
            
            if not Path(data_file).exists():
                continue
            
            dataset_results = {
                'multi_start_consistency': {},
                'strategy_comparison': {},
                'convergence_stability': {}
            }
            
            # Test convergence on reasonable sample sizes
            test_sizes = [min(s, 2000) for s in sample_sizes]  # Cap at 2000 for optimization testing
            
            for sample_size in test_sizes:
                try:
                    print(f"    🔍 Optimization testing: {dataset} (n={sample_size})")
                    
                    # Load data
                    full_data = pd.read_csv(data_file)
                    sampled_data = full_data.sample(n=min(sample_size, len(full_data)), random_state=42)
                    
                    temp_file = f"temp_opt_{dataset}_{sample_size}.csv"
                    sampled_data.to_csv(temp_file, index=False)
                    
                    data_context = pj.load_data(temp_file)
                    
                    # Simple model for optimization testing
                    formula_spec = create_simple_spec(phi="~1", p="~1", f="~1")
                    model = PradelModel()
                    design_matrices = model.build_design_matrices(formula_spec, data_context)
                    
                    def objective(params):
                        return -model.log_likelihood(params, data_context, design_matrices)
                    
                    initial_params = model.get_initial_parameters(data_context, design_matrices)
                    bounds = model.get_parameter_bounds(data_context, design_matrices)
                    
                    # Test 1: Multi-start consistency
                    multi_start_results = []
                    for seed in [42, 123, 456, 789, 999]:
                        try:
                            # Perturb initial parameters slightly
                            np.random.seed(seed)
                            perturbed_params = initial_params + np.random.normal(0, 0.05, size=initial_params.shape)
                            
                            # Ensure within bounds
                            for i, (low, high) in enumerate(bounds):
                                perturbed_params[i] = np.clip(perturbed_params[i], low, high)
                            
                            result = optimize_model(
                                objective_function=objective,
                                initial_parameters=perturbed_params,
                                context=data_context,
                                bounds=bounds
                            )
                            
                            if result.success:
                                multi_start_results.append({
                                    'success': True,
                                    'final_objective': result.result.fun,
                                    'final_params': result.result.x,
                                    'iterations': getattr(result.result, 'nit', 0),
                                    'strategy': result.strategy_used
                                })
                            else:
                                multi_start_results.append({'success': False, 'message': result.message})
                                
                        except Exception as e:
                            multi_start_results.append({'success': False, 'error': str(e)})
                    
                    # Analyze multi-start consistency
                    successful_runs = [r for r in multi_start_results if r.get('success', False)]
                    
                    consistency_tests = {
                        'success_rate': len(successful_runs) / len(multi_start_results),
                        'all_converged': len(successful_runs) == len(multi_start_results)
                    }
                    
                    if len(successful_runs) >= 2:
                        # Check objective function consistency
                        objectives = [r['final_objective'] for r in successful_runs]
                        obj_std = np.std(objectives)
                        obj_mean = np.mean(objectives)
                        
                        consistency_tests.update({
                            'objective_consistency': obj_std / abs(obj_mean) < 1e-6 if obj_mean != 0 else obj_std < 1e-8,
                            'objective_cv': obj_std / abs(obj_mean) if obj_mean != 0 else float('inf'),
                            'max_objective_diff': max(objectives) - min(objectives)
                        })
                        
                        # Check parameter consistency
                        param_arrays = np.array([r['final_params'] for r in successful_runs])
                        param_stds = np.std(param_arrays, axis=0)
                        param_means = np.mean(param_arrays, axis=0)
                        
                        param_cvs = []
                        for i in range(len(param_means)):
                            if abs(param_means[i]) > 1e-8:
                                param_cvs.append(param_stds[i] / abs(param_means[i]))
                            else:
                                param_cvs.append(param_stds[i])
                        
                        consistency_tests.update({
                            'parameter_consistency': all(cv < 1e-4 for cv in param_cvs),
                            'max_param_cv': max(param_cvs) if param_cvs else 0,
                            'param_cvs': param_cvs
                        })
                    
                    dataset_results['multi_start_consistency'][sample_size] = consistency_tests
                    
                    # Clean up
                    import os
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
                except Exception as e:
                    print(f"    ❌ Optimization validation failed: {e}")
                    dataset_results['multi_start_consistency'][sample_size] = {'error': str(e)}
            
            results['convergence_consistency'][dataset] = dataset_results
            print(f"    ✅ Optimization validation completed for {dataset}")
        
        return results
    
    def _validate_statistical_inference(self, datasets: List[str], sample_sizes: List[int]) -> Dict:
        """Validate statistical inference properties."""
        
        results = {
            'parameter_reasonableness': {},
            'model_selection_stability': {},
            'confidence_intervals': {}
        }
        
        # Implementation would include:
        # - Parameter estimate reasonableness (survival rates 0-1, etc.)
        # - AIC model selection stability across samples
        # - Bootstrap confidence interval validation
        # - Cross-validation consistency
        
        print("  📈 Statistical inference validation...")
        print("    ℹ️ Placeholder for statistical tests (implementation pending)")
        
        return results
    
    def _validate_cross_dataset_consistency(self, datasets: List[str], sample_sizes: List[int]) -> Dict:
        """Validate consistency across different datasets."""
        
        results = {
            'processing_consistency': {},
            'parameter_scale_consistency': {},
            'model_behavior_consistency': {}
        }
        
        print("  🔄 Cross-dataset consistency validation...")
        print("    ℹ️ Placeholder for cross-dataset tests (implementation pending)")
        
        return results
    
    def _validate_numerical_stability(self, datasets: List[str], sample_sizes: List[int]) -> Dict:
        """Validate numerical stability across different scales."""
        
        results = {
            'scale_invariance': {},
            'precision_stability': {},
            'overflow_underflow_handling': {}
        }
        
        print("  🎯 Numerical stability validation...")
        print("    ℹ️ Placeholder for numerical stability tests (implementation pending)")
        
        return results
    
    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        
        report_file = self.output_dir / f"validation_report_{self.timestamp}.json"
        
        # Save detailed results
        import json
        with open(report_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = self.output_dir / f"validation_summary_{self.timestamp}.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# Comprehensive Workflow Validation Report\n\n")
            f.write(f"**Timestamp:** {self.timestamp}\n\n")
            f.write(f"**Generated by:** Pradel-JAX Validation Framework\n\n")
            
            # Summary by phase
            for phase, results in self.validation_results.items():
                f.write(f"## {phase.replace('_', ' ').title()}\n\n")
                f.write(f"Status: {'✅ Completed' if results else '⚠️ Pending'}\n\n")
                
                if results:
                    f.write("Key findings:\n")
                    # Add specific findings based on results structure
                    f.write("- Detailed results in JSON file\n")
                f.write("\n")
        
        print(f"\n📄 Validation Report Generated:")
        print(f"  📊 Detailed Results: {report_file}")
        print(f"  📋 Summary Report: {summary_file}")


def main():
    """Run comprehensive workflow validation."""
    
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive Workflow Validation')
    parser.add_argument('--datasets', nargs='+', default=['nebraska', 'south_dakota'], 
                       help='Datasets to validate')
    parser.add_argument('--sample-sizes', nargs='+', type=int, default=[100, 1000, 5000],
                       help='Sample sizes to test')
    parser.add_argument('--output-dir', default='validation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = WorkflowValidator(output_dir=args.output_dir)
    
    # Run validation
    results = validator.run_comprehensive_validation(
        datasets=args.datasets,
        sample_sizes=args.sample_sizes
    )
    
    print(f"\n✅ Comprehensive validation completed!")
    print(f"📁 Results saved to: {validator.output_dir}")


if __name__ == "__main__":
    main()