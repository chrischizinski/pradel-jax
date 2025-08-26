#!/usr/bin/env python3
"""
Focused Workflow Examination
===========================

Comprehensive examination of the Pradel-JAX workflow components that are currently
implemented, focusing on identifying process and statistical issues.
"""

import pandas as pd
import numpy as np
import jax.numpy as jnp
import warnings
from pathlib import Path
import json
from datetime import datetime
import sys
import logging
from typing import Dict, List, Any, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
sys.path.append('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

# Import what's actually available
import pradel_jax as pj
from pradel_jax.data.adapters import load_data
from pradel_jax.models.pradel import PradelModel
from pradel_jax.formulas import create_simple_spec, FormulaSpec
from pradel_jax.optimization import optimize_model

class FocusedWorkflowExaminer:
    """Focused examination of currently working workflow components."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'datasets_examined': [],
            'data_loading_results': {},
            'formula_parsing_results': {},
            'likelihood_computation_results': {},
            'optimization_results': {},
            'statistical_validation_results': {},
            'critical_issues': [],
            'recommendations': []
        }
    
    def examine_data_loading(self, filepath: str, dataset_name: str) -> Dict[str, Any]:
        """Comprehensive data loading examination."""
        logger.info(f"=== EXAMINING DATA LOADING: {dataset_name} ===")
        
        results = {
            'dataset': dataset_name,
            'filepath': filepath,
            'loading_successful': False,
            'data_quality_metrics': {},
            'potential_issues': [],
            'data_context': None
        }
        
        try:
            # Load data and capture detailed information
            data_context = load_data(filepath)
            results['loading_successful'] = True
            results['data_context'] = data_context
            
            # Extract detailed metrics
            results['data_quality_metrics'] = {
                'n_individuals': data_context.n_individuals,
                'n_occasions': data_context.n_occasions,
                'capture_matrix_shape': data_context.capture_matrix.shape,
                'n_covariates': len(data_context.covariates) if data_context.covariates else 0,
                'covariate_names': list(data_context.covariates.keys()) if data_context.covariates else [],
                'total_encounters': int(jnp.sum(data_context.capture_matrix)),
                'individuals_never_encountered': int(jnp.sum(jnp.sum(data_context.capture_matrix, axis=1) == 0)),
                'empty_occasions': int(jnp.sum(jnp.sum(data_context.capture_matrix, axis=0) == 0))
            }
            
            # Data quality checks
            encounter_rate = results['data_quality_metrics']['total_encounters'] / (data_context.n_individuals * data_context.n_occasions)
            if encounter_rate < 0.05:
                results['potential_issues'].append(f"Very low overall encounter rate: {encounter_rate:.3f}")
            
            if results['data_quality_metrics']['individuals_never_encountered'] > data_context.n_individuals * 0.1:
                results['potential_issues'].append(f"High proportion of individuals never encountered: {results['data_quality_metrics']['individuals_never_encountered']}")
            
            if results['data_quality_metrics']['empty_occasions'] > 0:
                results['potential_issues'].append(f"Empty time occasions detected: {results['data_quality_metrics']['empty_occasions']}")
            
            # Check for matrix sparsity issues
            sparsity = 1 - (results['data_quality_metrics']['total_encounters'] / (data_context.n_individuals * data_context.n_occasions))
            if sparsity > 0.95:
                results['potential_issues'].append(f"Very sparse data (sparsity: {sparsity:.3f}) may cause optimization issues")
            
            # Covariate quality checks
            if data_context.covariates:
                for cov_name, cov_data in data_context.covariates.items():
                    if jnp.any(jnp.isnan(cov_data)):
                        nan_count = int(jnp.sum(jnp.isnan(cov_data)))
                        results['potential_issues'].append(f"Covariate '{cov_name}' has {nan_count} NaN values")
                    
                    # Check for constant covariates
                    if jnp.all(cov_data == cov_data[0]):
                        results['potential_issues'].append(f"Covariate '{cov_name}' is constant - will cause singularity issues")
            
        except Exception as e:
            results['loading_successful'] = False
            results['error'] = str(e)
            logger.error(f"Data loading failed for {dataset_name}: {e}")
        
        return results
    
    def examine_formula_system(self, data_context, dataset_name: str) -> Dict[str, Any]:
        """Examine formula parsing and design matrix construction."""
        logger.info(f"=== EXAMINING FORMULA SYSTEM: {dataset_name} ===")
        
        results = {
            'basic_formulas_work': False,
            'covariate_formulas_work': False,
            'design_matrix_issues': [],
            'formula_errors': []
        }
        
        try:
            # Test basic intercept-only formula
            basic_spec = create_simple_spec()
            model = PradelModel()
            
            try:
                design_matrices = model.build_design_matrices(basic_spec, data_context)
                results['basic_formulas_work'] = True
                
                # Validate design matrix structure
                if not isinstance(design_matrices, dict):
                    results['design_matrix_issues'].append("Design matrices not returned as dictionary")
                else:
                    expected_params = ['phi', 'p', 'f']
                    for param in expected_params:
                        if param not in design_matrices:
                            results['design_matrix_issues'].append(f"Missing design matrix for parameter '{param}'")
                        else:
                            dm = design_matrices[param]
                            if dm.shape[0] != data_context.n_individuals:
                                results['design_matrix_issues'].append(f"Design matrix for '{param}' has wrong number of rows: {dm.shape[0]} vs {data_context.n_individuals}")
                            
                            # Check for numerical issues
                            if jnp.any(jnp.isnan(dm)):
                                results['design_matrix_issues'].append(f"Design matrix for '{param}' contains NaN values")
                            if jnp.any(jnp.isinf(dm)):
                                results['design_matrix_issues'].append(f"Design matrix for '{param}' contains infinite values")
                
            except Exception as e:
                results['formula_errors'].append(f"Basic formula creation failed: {str(e)}")
            
            # Test covariate formulas if available
            if data_context.covariates:
                covariate_names = list(data_context.covariates.keys())
                test_covariates = covariate_names[:2]  # Test first two covariates
                
                for cov_name in test_covariates:
                    try:
                        # Create formula with single covariate
                        cov_spec = FormulaSpec(
                            phi=f"~1 + {cov_name}",
                            p="~1", 
                            f="~1"
                        )
                        
                        design_matrices = model.build_design_matrices(cov_spec, data_context)
                        
                        # Check if covariate was incorporated
                        phi_dm = design_matrices['phi']
                        if phi_dm.shape[1] < 2:
                            results['design_matrix_issues'].append(f"Covariate '{cov_name}' not properly incorporated in design matrix")
                        
                        results['covariate_formulas_work'] = True
                        
                    except Exception as e:
                        results['formula_errors'].append(f"Covariate formula with '{cov_name}' failed: {str(e)}")
            
        except Exception as e:
            results['formula_errors'].append(f"Formula system examination failed: {str(e)}")
        
        return results
    
    def examine_likelihood_computation(self, data_context, dataset_name: str) -> Dict[str, Any]:
        """Examine model likelihood computation for numerical stability."""
        logger.info(f"=== EXAMINING LIKELIHOOD COMPUTATION: {dataset_name} ===")
        
        results = {
            'likelihood_computable': False,
            'gradient_computable': False,
            'numerical_stability_issues': [],
            'parameter_sensitivity': {}
        }
        
        try:
            model = PradelModel()
            formula_spec = create_simple_spec()
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            
            # Test likelihood computation with reasonable parameters
            test_params_sets = [
                np.array([0.0, 0.0, 0.0]),      # Neutral (0.5 probabilities)
                np.array([1.0, 1.0, 1.0]),      # High probabilities
                np.array([-1.0, -1.0, -1.0]),   # Low probabilities
                np.array([0.5, -0.5, 2.0]),     # Mixed realistic values
            ]
            
            likelihood_values = []
            
            for i, params in enumerate(test_params_sets):
                try:
                    ll = model.log_likelihood(params, data_context, design_matrices)
                    likelihood_values.append(ll)
                    
                    # Check for numerical issues
                    if np.isnan(ll):
                        results['numerical_stability_issues'].append(f"NaN likelihood with parameter set {i+1}")
                    elif np.isinf(ll):
                        results['numerical_stability_issues'].append(f"Infinite likelihood with parameter set {i+1}")
                    elif ll > 0:
                        results['numerical_stability_issues'].append(f"Positive log-likelihood ({ll:.3f}) with parameter set {i+1}")
                    
                    results['likelihood_computable'] = True
                    
                except Exception as e:
                    results['numerical_stability_issues'].append(f"Likelihood computation failed with parameter set {i+1}: {str(e)}")
            
            # Test gradient computation
            try:
                import jax
                grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
                
                for i, params in enumerate(test_params_sets[:2]):  # Test first two sets
                    try:
                        gradient = grad_fn(params)
                        
                        if np.any(np.isnan(gradient)):
                            results['numerical_stability_issues'].append(f"NaN gradient with parameter set {i+1}")
                        elif np.any(np.isinf(gradient)):
                            results['numerical_stability_issues'].append(f"Infinite gradient with parameter set {i+1}")
                        else:
                            results['gradient_computable'] = True
                            
                    except Exception as e:
                        results['numerical_stability_issues'].append(f"Gradient computation failed with parameter set {i+1}: {str(e)}")
                        
            except Exception as e:
                results['numerical_stability_issues'].append(f"Gradient function creation failed: {str(e)}")
            
            # Analyze likelihood surface characteristics
            if len(likelihood_values) >= 2 and not any(np.isnan(v) or np.isinf(v) for v in likelihood_values):
                ll_range = max(likelihood_values) - min(likelihood_values)
                results['parameter_sensitivity']['likelihood_range'] = ll_range
                
                if ll_range < 1e-6:
                    results['numerical_stability_issues'].append("Likelihood surface appears flat - potential identifiability issues")
                elif ll_range > 1000:
                    results['numerical_stability_issues'].append("Extreme likelihood range suggests numerical instability")
            
        except Exception as e:
            results['numerical_stability_issues'].append(f"Likelihood examination failed: {str(e)}")
        
        return results
    
    def examine_optimization(self, data_context, dataset_name: str) -> Dict[str, Any]:
        """Examine optimization behavior and convergence."""
        logger.info(f"=== EXAMINING OPTIMIZATION: {dataset_name} ===")
        
        results = {
            'optimization_attempts': [],
            'convergence_issues': [],
            'parameter_estimates': {},
            'optimization_stable': False
        }
        
        try:
            model = PradelModel()
            formula_spec = create_simple_spec()
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            
            # Define objective function
            def objective(params):
                return -model.log_likelihood(params, data_context, design_matrices)
            
            # Get initial parameters and bounds
            initial_params = model.get_initial_parameters(data_context, design_matrices)
            bounds = model.get_parameter_bounds(data_context, design_matrices)
            
            # Test optimization with different starting points
            starting_points = [
                initial_params,
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 1.0]),
                np.array([-1.0, -1.0, -1.0])
            ]
            
            successful_optimizations = []
            
            for i, start_point in enumerate(starting_points):
                try:
                    result = optimize_model(
                        objective_function=objective,
                        initial_parameters=start_point,
                        context=data_context,
                        bounds=bounds
                    )
                    
                    opt_result = {
                        'starting_point': i,
                        'success': result.success if hasattr(result, 'success') else False,
                        'final_params': result.x if hasattr(result, 'x') else None,
                        'final_likelihood': result.fun if hasattr(result, 'fun') else None,
                        'iterations': result.nit if hasattr(result, 'nit') else None
                    }
                    
                    results['optimization_attempts'].append(opt_result)
                    
                    if opt_result['success']:
                        successful_optimizations.append(opt_result)
                        
                        # Convert to probability scale for interpretation
                        if opt_result['final_params'] is not None:
                            params = opt_result['final_params']
                            phi_prob = 1 / (1 + np.exp(-params[0]))
                            p_prob = 1 / (1 + np.exp(-params[1]))  
                            f_prob = 1 / (1 + np.exp(-params[2]))
                            
                            opt_result['probabilities'] = {
                                'survival': phi_prob,
                                'detection': p_prob,
                                'entry': f_prob
                            }
                            
                            # Check biological plausibility
                            if phi_prob < 0.01 or phi_prob > 0.99:
                                results['convergence_issues'].append(f"Extreme survival estimate: {phi_prob:.4f}")
                            if p_prob < 0.01 or p_prob > 0.99:
                                results['convergence_issues'].append(f"Extreme detection estimate: {p_prob:.4f}")
                            if f_prob < 0.001 or f_prob > 0.9:
                                results['convergence_issues'].append(f"Extreme entry estimate: {f_prob:.4f}")
                    
                except Exception as e:
                    results['convergence_issues'].append(f"Optimization failed with starting point {i}: {str(e)}")
            
            # Analyze optimization stability
            if len(successful_optimizations) >= 2:
                param_estimates = np.array([opt['final_params'] for opt in successful_optimizations])
                param_std = np.std(param_estimates, axis=0)
                
                if np.any(param_std > 0.5):  # Large standard deviation in logit scale
                    results['convergence_issues'].append(f"Unstable parameter estimates across starting points: std = {param_std}")
                else:
                    results['optimization_stable'] = True
                    results['parameter_estimates'] = {
                        'mean_params': np.mean(param_estimates, axis=0).tolist(),
                        'param_std': param_std.tolist()
                    }
            
        except Exception as e:
            results['convergence_issues'].append(f"Optimization examination failed: {str(e)}")
        
        return results
    
    def run_focused_examination(self) -> Dict[str, Any]:
        """Run focused examination on available datasets."""
        logger.info("=== STARTING FOCUSED WORKFLOW EXAMINATION ===")
        
        datasets = [
            ('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax/data/dipper_dataset.csv', 'dipper'),
            ('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax/data/encounter_histories_ne_clean.csv', 'nebraska'),
            ('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax/data/encounter_histories_sd_clean.csv', 'south_dakota')
        ]
        
        for filepath, name in datasets:
            if Path(filepath).exists():
                logger.info(f"\n{'='*60}")
                logger.info(f"EXAMINING DATASET: {name.upper()}")
                logger.info(f"{'='*60}")
                
                # 1. Data loading examination
                data_results = self.examine_data_loading(filepath, name)
                self.results['data_loading_results'][name] = data_results
                
                if data_results['loading_successful']:
                    data_context = data_results['data_context']
                    
                    # 2. Formula system examination
                    formula_results = self.examine_formula_system(data_context, name)
                    self.results['formula_parsing_results'][name] = formula_results
                    
                    # 3. Likelihood computation examination
                    likelihood_results = self.examine_likelihood_computation(data_context, name)
                    self.results['likelihood_computation_results'][name] = likelihood_results
                    
                    # 4. Optimization examination
                    opt_results = self.examine_optimization(data_context, name)
                    self.results['optimization_results'][name] = opt_results
                    
                else:
                    logger.warning(f"Skipping further examination of {name} due to data loading failure")
        
        # Synthesize findings
        self.synthesize_critical_issues()
        self.generate_targeted_recommendations()
        
        return self.results
    
    def synthesize_critical_issues(self):
        """Identify critical workflow issues across datasets."""
        critical_issues = []
        
        # Check data loading issues
        loading_failures = [name for name, results in self.results['data_loading_results'].items() 
                          if not results['loading_successful']]
        if loading_failures:
            critical_issues.append(f"Data loading failed for: {', '.join(loading_failures)}")
        
        # Check formula system issues
        formula_failures = [name for name, results in self.results['formula_parsing_results'].items()
                          if not results.get('basic_formulas_work', False)]
        if formula_failures:
            critical_issues.append(f"Basic formula parsing failed for: {', '.join(formula_failures)}")
        
        # Check likelihood computation issues
        likelihood_failures = [name for name, results in self.results['likelihood_computation_results'].items()
                             if not results.get('likelihood_computable', False)]
        if likelihood_failures:
            critical_issues.append(f"Likelihood computation failed for: {', '.join(likelihood_failures)}")
        
        # Check optimization issues
        opt_failures = [name for name, results in self.results['optimization_results'].items()
                       if not results.get('optimization_stable', False)]
        if opt_failures:
            critical_issues.append(f"Optimization unstable for: {', '.join(opt_failures)}")
        
        # Cross-dataset issues
        all_datasets = list(self.results['data_loading_results'].keys())
        
        # Check for sparse data issues
        sparse_datasets = []
        for name in all_datasets:
            data_results = self.results['data_loading_results'][name]
            if data_results['loading_successful']:
                issues = data_results.get('potential_issues', [])
                if any('sparse' in issue.lower() for issue in issues):
                    sparse_datasets.append(name)
        
        if sparse_datasets:
            critical_issues.append(f"Sparse data detected in: {', '.join(sparse_datasets)}")
        
        # Check for biological implausibility
        implausible_datasets = []
        for name in all_datasets:
            if name in self.results['optimization_results']:
                issues = self.results['optimization_results'][name].get('convergence_issues', [])
                if any('extreme' in issue.lower() for issue in issues):
                    implausible_datasets.append(name)
        
        if implausible_datasets:
            critical_issues.append(f"Biologically implausible estimates in: {', '.join(implausible_datasets)}")
        
        self.results['critical_issues'] = critical_issues
    
    def generate_targeted_recommendations(self):
        """Generate specific, actionable recommendations."""
        recommendations = []
        
        # Data quality recommendations
        sparse_data_issues = any('sparse' in issue for issue in self.results['critical_issues'])
        if sparse_data_issues:
            recommendations.append("Implement data adequacy checks before model fitting")
            recommendations.append("Add warnings for datasets with low encounter rates")
        
        # Formula system recommendations  
        formula_issues = any('formula' in issue.lower() for issue in self.results['critical_issues'])
        if formula_issues:
            recommendations.append("Improve formula system error handling and user feedback")
        
        # Likelihood computation recommendations
        likelihood_issues = any('likelihood' in issue.lower() for issue in self.results['critical_issues'])
        if likelihood_issues:
            recommendations.append("Add numerical stability safeguards to likelihood computation")
            recommendations.append("Implement parameter bounds checking in likelihood function")
        
        # Optimization recommendations
        opt_issues = any('optimization' in issue.lower() for issue in self.results['critical_issues'])
        if opt_issues:
            recommendations.append("Implement multi-start optimization by default for stability")
            recommendations.append("Add convergence diagnostics and warnings")
        
        # Biological plausibility recommendations
        bio_issues = any('implausible' in issue.lower() for issue in self.results['critical_issues'])
        if bio_issues:
            recommendations.append("Add biological bounds to optimization (e.g., survival < 0.99)")
            recommendations.append("Implement post-optimization biological plausibility checks")
        
        # General recommendations
        recommendations.extend([
            "Add comprehensive input validation for all workflow steps",
            "Implement automatic model diagnostics and warnings",
            "Create user-friendly error messages with actionable advice",
            "Add data preprocessing recommendations based on detected issues"
        ])
        
        self.results['recommendations'] = recommendations
    
    def save_report(self, filename: str = None):
        """Save examination report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"focused_workflow_examination_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Focused examination report saved to: {filename}")
        return filename

def main():
    """Run focused workflow examination."""
    examiner = FocusedWorkflowExaminer()
    results = examiner.run_focused_examination()
    
    # Save detailed report
    report_filename = examiner.save_report()
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("FOCUSED WORKFLOW EXAMINATION SUMMARY")
    print("="*80)
    
    # Data loading results
    print("\n📊 DATA LOADING RESULTS:")
    for name, result in results['data_loading_results'].items():
        status = "✅" if result['loading_successful'] else "❌"
        print(f"  {status} {name.title()}")
        if result['loading_successful']:
            metrics = result['data_quality_metrics']
            print(f"     • {metrics['n_individuals']:,} individuals, {metrics['n_occasions']} occasions")
            print(f"     • {metrics['total_encounters']:,} total encounters")
            print(f"     • {metrics['n_covariates']} covariates")
            if result['potential_issues']:
                print(f"     ⚠️  Issues: {len(result['potential_issues'])}")
        else:
            print(f"     ❌ {result.get('error', 'Unknown error')}")
    
    # Formula system results
    print("\n🔧 FORMULA SYSTEM RESULTS:")
    for name, result in results['formula_parsing_results'].items():
        basic_status = "✅" if result.get('basic_formulas_work', False) else "❌"
        cov_status = "✅" if result.get('covariate_formulas_work', False) else "❌"
        print(f"  {name.title()}:")
        print(f"     • Basic formulas: {basic_status}")
        print(f"     • Covariate formulas: {cov_status}")
        if result.get('formula_errors'):
            print(f"     ❌ Errors: {len(result['formula_errors'])}")
    
    # Likelihood computation results
    print("\n🧮 LIKELIHOOD COMPUTATION RESULTS:")
    for name, result in results['likelihood_computation_results'].items():
        ll_status = "✅" if result.get('likelihood_computable', False) else "❌"
        grad_status = "✅" if result.get('gradient_computable', False) else "❌"
        print(f"  {name.title()}:")
        print(f"     • Likelihood: {ll_status}")
        print(f"     • Gradients: {grad_status}")
        if result.get('numerical_stability_issues'):
            print(f"     ⚠️  Stability issues: {len(result['numerical_stability_issues'])}")
    
    # Optimization results
    print("\n🎯 OPTIMIZATION RESULTS:")
    for name, result in results['optimization_results'].items():
        stable = result.get('optimization_stable', False)
        status = "✅" if stable else "❌"
        print(f"  {status} {name.title()}")
        attempts = result.get('optimization_attempts', [])
        successful = [a for a in attempts if a.get('success', False)]
        print(f"     • {len(successful)}/{len(attempts)} optimization attempts successful")
        if result.get('convergence_issues'):
            print(f"     ⚠️  Issues: {len(result['convergence_issues'])}")
    
    # Critical issues
    print(f"\n🚨 CRITICAL ISSUES ({len(results['critical_issues'])}):")
    for issue in results['critical_issues']:
        print(f"  • {issue}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS ({len(results['recommendations'])}):")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i:2}. {rec}")
    
    print(f"\n📋 Detailed report: {report_filename}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()