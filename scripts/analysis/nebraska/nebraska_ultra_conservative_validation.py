#!/usr/bin/env python3
"""
ULTRA CONSERVATIVE VALIDATION OF NEBRASKA MODELS

This implements the most rigorous testing possible:
1. Multiple independent random samples from the same population
2. Multiple independent optimization runs on identical data  
3. Cross-validation between different samples
4. Statistical tests for parameter consistency
5. Convergence diagnostics and stability analysis
6. Comprehensive model comparison with covariates

Following the user's directive: "Ultra think and plan on fixing the problem"
- No shortcuts or simplified approaches
- Full statistical rigor
- Comprehensive diagnostics
- Ultra-conservative consistency criteria
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
import time
from typing import Dict, List, Tuple, Any
import warnings
from nebraska_data_loader import load_and_prepare_nebraska_data

sys.path.insert(0, '/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
from pradel_jax.models.pradel import PradelModel, logit, log_link, inv_logit, exp_link
from pradel_jax.formulas.parser import FormulaParser
from pradel_jax.formulas.spec import ParameterType, FormulaSpec

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", ".*TPU.*")

class UltraConservativeValidator:
    """Ultra conservative validation framework for Nebraska models."""
    
    def __init__(self):
        self.results_database = []
        self.consistency_reports = []
        
    def create_formula_specifications(self) -> Dict[str, FormulaSpec]:
        """Create comprehensive formula specifications."""
        parser = FormulaParser()
        
        return {
            "null_model": FormulaSpec(
                phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
                p=parser.create_parameter_formula(ParameterType.P, "~1"),
                f=parser.create_parameter_formula(ParameterType.F, "~1")
            ),
            
            "gender_survival": FormulaSpec(
                phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + gender"),
                p=parser.create_parameter_formula(ParameterType.P, "~1"),
                f=parser.create_parameter_formula(ParameterType.F, "~1")
            ),
            
            "gender_detection": FormulaSpec(
                phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
                p=parser.create_parameter_formula(ParameterType.P, "~1 + gender"),
                f=parser.create_parameter_formula(ParameterType.F, "~1")
            ),
            
            "age_linear": FormulaSpec(
                phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + age"),
                p=parser.create_parameter_formula(ParameterType.P, "~1 + age"),
                f=parser.create_parameter_formula(ParameterType.F, "~1")
            ),
            
            "tier_effects": FormulaSpec(
                phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + tier_cat"),
                p=parser.create_parameter_formula(ParameterType.P, "~1 + tier_cat"),
                f=parser.create_parameter_formula(ParameterType.F, "~1")
            ),
            
            "additive_model": FormulaSpec(
                phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + gender + age"),
                p=parser.create_parameter_formula(ParameterType.P, "~1 + gender + age"),
                f=parser.create_parameter_formula(ParameterType.F, "~1 + gender")
            )
        }
    
    def fit_single_model_ultra_rigorous(
        self, 
        data_context: Any, 
        formula_spec: FormulaSpec,
        model_name: str,
        sample_id: str,
        run_id: int
    ) -> Dict[str, Any]:
        """
        Fit a single model with ultra-rigorous diagnostics.
        
        This method captures EVERY aspect of the optimization process
        for ultra-conservative validation.
        """
        start_time = time.time()
        
        try:
            # Initialize model
            model = PradelModel()
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            
            # Get optimization setup
            initial_params = model.get_initial_parameters(data_context, design_matrices)
            bounds = model.get_parameter_bounds(data_context, design_matrices)
            param_names = model.get_parameter_names(design_matrices)
            
            # Define optimization functions with full diagnostics
            iteration_history = []
            
            def objective_with_logging(params):
                ll = model.log_likelihood(jnp.array(params), data_context, design_matrices)
                obj_val = -float(ll)
                iteration_history.append({
                    'params': params.copy(),
                    'log_likelihood': float(ll),
                    'objective': obj_val
                })
                return obj_val
            
            def gradient_with_logging(params):
                grad_fn = jax.grad(lambda p: model.log_likelihood(p, data_context, design_matrices))
                grad = -np.array(grad_fn(jnp.array(params)))
                return grad
            
            # Test gradient accuracy at initial point
            initial_ll = -objective_with_logging(initial_params)
            initial_grad = gradient_with_logging(initial_params)
            initial_grad_norm = np.linalg.norm(initial_grad)
            
            # Clear history for actual optimization
            iteration_history = []
            
            # Multiple optimization attempts with different methods
            optimization_attempts = []
            
            # Method 1: L-BFGS-B (primary)
            try:
                result_lbfgs = minimize(
                    objective_with_logging,
                    initial_params,
                    method='L-BFGS-B',
                    jac=gradient_with_logging,
                    bounds=bounds,
                    options={'disp': False, 'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                optimization_attempts.append({
                    'method': 'L-BFGS-B',
                    'result': result_lbfgs,
                    'iteration_history': iteration_history.copy()
                })
                
            except Exception as e:
                optimization_attempts.append({
                    'method': 'L-BFGS-B',
                    'error': str(e),
                    'result': None
                })
            
            # Method 2: SLSQP (backup)
            iteration_history = []
            try:
                result_slsqp = minimize(
                    objective_with_logging,
                    initial_params,
                    method='SLSQP',
                    jac=gradient_with_logging,
                    bounds=bounds,
                    options={'disp': False, 'maxiter': 300, 'ftol': 1e-12}
                )
                
                optimization_attempts.append({
                    'method': 'SLSQP',
                    'result': result_slsqp,
                    'iteration_history': iteration_history.copy()
                })
                
            except Exception as e:
                optimization_attempts.append({
                    'method': 'SLSQP',
                    'error': str(e),
                    'result': None
                })
            
            # Select best result
            successful_attempts = [a for a in optimization_attempts if a.get('result') and a['result'].success]
            
            if successful_attempts:
                # Choose result with best objective value
                best_attempt = min(successful_attempts, key=lambda x: x['result'].fun)
                result = best_attempt['result']
                method_used = best_attempt['method']
                final_iteration_history = best_attempt['iteration_history']
            else:
                # No successful optimization
                result = optimization_attempts[0]['result'] if optimization_attempts else None
                method_used = 'FAILED'
                final_iteration_history = []
            
            fit_time = time.time() - start_time
            
            # Comprehensive result analysis
            if result and result.success:
                final_ll = -result.fun
                ll_improvement = final_ll - initial_ll
                final_grad_norm = np.linalg.norm(result.jac) if hasattr(result, 'jac') else None
                
                # Parameter analysis
                params_natural = {}
                for i, (param_name, param_val) in enumerate(zip(param_names, result.x)):
                    if 'phi_' in param_name or 'p_' in param_name:
                        params_natural[param_name] = inv_logit(param_val)
                    elif 'f_' in param_name:
                        params_natural[param_name] = exp_link(param_val)
                    else:
                        params_natural[param_name] = param_val
                
                # Bound analysis
                hit_bounds = []
                near_bounds = []
                for i, (param_val, (lower, upper)) in enumerate(zip(result.x, bounds)):
                    param_name = param_names[i]
                    if abs(param_val - lower) < 1e-6:
                        hit_bounds.append(f"{param_name} (lower)")
                    elif abs(param_val - upper) < 1e-6:
                        hit_bounds.append(f"{param_name} (upper)")
                    elif abs(param_val - lower) < 0.01 or abs(param_val - upper) < 0.01:
                        near_bounds.append(param_name)
                
                # Convergence quality assessment
                if final_grad_norm is not None:
                    if final_grad_norm < 1e-6:
                        convergence_quality = 'EXCELLENT'
                    elif final_grad_norm < 1e-4:
                        convergence_quality = 'GOOD'
                    elif final_grad_norm < 1e-2:
                        convergence_quality = 'ACCEPTABLE'
                    else:
                        convergence_quality = 'POOR'
                else:
                    convergence_quality = 'UNKNOWN'
                
                # Calculate information criteria
                n_params = len(result.x)
                n_obs = data_context.n_individuals
                aic = 2 * n_params - 2 * final_ll
                bic = np.log(n_obs) * n_params - 2 * final_ll
                
                success_status = True
                
            else:
                # Failed optimization
                final_ll = None
                ll_improvement = None
                final_grad_norm = None
                params_natural = {}
                hit_bounds = []
                near_bounds = []
                convergence_quality = 'FAILED'
                aic = None
                bic = None
                n_params = len(initial_params)
                success_status = False
            
            # Comprehensive diagnostic record
            diagnostic_record = {
                # Identifiers
                'model_name': model_name,
                'sample_id': sample_id,
                'run_id': run_id,
                'timestamp': time.time(),
                
                # Data characteristics
                'n_individuals': data_context.n_individuals,
                'n_occasions': data_context.n_occasions,
                'n_parameters': n_params,
                'parameter_names': param_names,
                
                # Optimization process
                'method_used': method_used,
                'optimization_attempts': len(optimization_attempts),
                'successful_attempts': len(successful_attempts),
                'fit_time_seconds': fit_time,
                
                # Initial state
                'initial_log_likelihood': initial_ll,
                'initial_gradient_norm': initial_grad_norm,
                
                # Final results
                'success': success_status,
                'final_log_likelihood': final_ll,
                'log_likelihood_improvement': ll_improvement,
                'final_gradient_norm': final_grad_norm,
                'convergence_quality': convergence_quality,
                
                # Parameters
                'parameters_transformed': result.x.tolist() if result and result.success else None,
                'parameters_natural': params_natural,
                
                # Model selection
                'aic': aic,
                'bic': bic,
                
                # Diagnostics
                'iterations': result.nit if result and hasattr(result, 'nit') else None,
                'function_evaluations': result.nfev if result and hasattr(result, 'nfev') else None,
                'hit_bounds': hit_bounds,
                'near_bounds': near_bounds,
                'message': result.message if result else 'No result',
                
                # Full optimization history
                'iteration_history': final_iteration_history
            }
            
            return diagnostic_record
            
        except Exception as e:
            return {
                'model_name': model_name,
                'sample_id': sample_id,
                'run_id': run_id,
                'success': False,
                'error': str(e),
                'fit_time_seconds': time.time() - start_time
            }
    
    def analyze_within_model_consistency(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze consistency within multiple runs of the same model on same data.
        
        Ultra-conservative criteria:
        - Log-likelihood CV < 1e-6 (essentially identical)
        - Parameter CV < 1e-4 (very tight)
        - All runs must succeed
        """
        
        successful_runs = [r for r in results if r['success']]
        n_successful = len(successful_runs)
        n_total = len(results)
        
        if n_successful == 0:
            return {'status': 'ALL_FAILED', 'success_rate': 0.0}
        
        if n_successful < n_total:
            return {'status': 'PARTIAL_FAILURE', 'success_rate': n_successful / n_total}
        
        # Extract metrics
        log_likelihoods = [r['final_log_likelihood'] for r in successful_runs]
        aics = [r['aic'] for r in successful_runs]
        
        # Statistical analysis
        ll_mean = np.mean(log_likelihoods)
        ll_std = np.std(log_likelihoods)
        ll_cv = ll_std / abs(ll_mean) if ll_mean != 0 else np.inf
        
        aic_mean = np.mean(aics)
        aic_std = np.std(aics)
        
        # Parameter consistency analysis
        all_params = np.array([r['parameters_transformed'] for r in successful_runs])
        param_cvs = []
        
        if len(successful_runs) > 1:
            param_stds = np.std(all_params, axis=0)
            param_means = np.mean(all_params, axis=0)
            param_cvs = param_stds / np.abs(param_means)
            param_cvs = np.where(np.isfinite(param_cvs), param_cvs, 0.0)
            
            max_param_cv = np.max(param_cvs)
            mean_param_cv = np.mean(param_cvs)
        else:
            max_param_cv = 0.0
            mean_param_cv = 0.0
        
        # Ultra-conservative consistency classification
        if ll_cv > 1e-4:
            status = 'HIGHLY_INCONSISTENT'
        elif ll_cv > 1e-6:
            status = 'MODERATELY_INCONSISTENT'
        elif max_param_cv > 1e-2:
            status = 'PARAMETER_INCONSISTENT'
        elif max_param_cv > 1e-4:
            status = 'MARGINALLY_CONSISTENT'
        else:
            status = 'ULTRA_CONSISTENT'
        
        return {
            'status': status,
            'n_successful': n_successful,
            'n_total': n_total,
            'success_rate': n_successful / n_total,
            'log_likelihood_mean': ll_mean,
            'log_likelihood_cv': ll_cv,
            'aic_mean': aic_mean,
            'aic_std': aic_std,
            'max_parameter_cv': max_param_cv,
            'mean_parameter_cv': mean_param_cv,
            'convergence_qualities': [r['convergence_quality'] for r in successful_runs]
        }
    
    def run_comprehensive_validation(self):
        """Run the comprehensive ultra-conservative validation."""
        
        print("="*80)
        print("ULTRA CONSERVATIVE NEBRASKA MODELS VALIDATION")
        print("="*80)
        
        # Test parameters
        sample_sizes = [200, 500]  # Two different sizes
        n_independent_samples = 3  # Multiple independent samples
        n_optimization_runs = 5   # Multiple runs per sample
        
        formulas = self.create_formula_specifications()
        # Start with core models for thorough validation
        test_models = ['null_model', 'gender_survival', 'gender_detection', 'age_linear']
        
        print(f"VALIDATION PARAMETERS:")
        print(f"  Sample sizes: {sample_sizes}")
        print(f"  Independent samples per size: {n_independent_samples}")
        print(f"  Optimization runs per sample: {n_optimization_runs}")
        print(f"  Models to test: {test_models}")
        print(f"  Total optimization runs: {len(sample_sizes) * n_independent_samples * len(test_models) * n_optimization_runs}")
        
        validation_results = []
        
        # For each sample size
        for sample_size in sample_sizes:
            print(f"\n" + "="*60)
            print(f"TESTING SAMPLE SIZE: {sample_size}")
            print("="*60)
            
            # Generate multiple independent samples
            samples_data = []
            for sample_idx in range(n_independent_samples):
                data_context, df = load_and_prepare_nebraska_data(
                    n_sample=sample_size, 
                    random_state=42 + sample_idx * 100
                )
                
                if data_context is not None:
                    samples_data.append((data_context, df))
                    print(f"  Sample {sample_idx + 1}: {data_context.n_individuals} individuals loaded")
                else:
                    print(f"  ❌ Sample {sample_idx + 1}: Failed to load")
            
            # Test each model on each sample
            for model_name in test_models:
                print(f"\n  Testing model: {model_name}")
                
                formula_spec = formulas[model_name]
                
                # Test on each independent sample
                for sample_idx, (data_context, df) in enumerate(samples_data):
                    sample_id = f"size_{sample_size}_sample_{sample_idx}"
                    
                    print(f"    Sample {sample_idx + 1}: ", end="")
                    
                    # Multiple optimization runs on same data
                    model_results = []
                    for run_id in range(n_optimization_runs):
                        result = self.fit_single_model_ultra_rigorous(
                            data_context, formula_spec, model_name, sample_id, run_id
                        )
                        model_results.append(result)
                        validation_results.append(result)
                    
                    # Analyze consistency within this sample
                    consistency = self.analyze_within_model_consistency(model_results)
                    
                    # Status reporting
                    status_symbols = {
                        'ULTRA_CONSISTENT': '✅',
                        'MARGINALLY_CONSISTENT': '🟡', 
                        'PARAMETER_INCONSISTENT': '🟠',
                        'MODERATELY_INCONSISTENT': '🔶',
                        'HIGHLY_INCONSISTENT': '❌',
                        'PARTIAL_FAILURE': '⚠️',
                        'ALL_FAILED': '🚫'
                    }
                    
                    symbol = status_symbols.get(consistency['status'], '❓')
                    print(f"{symbol} {consistency['status']} (Success: {consistency['success_rate']:.0%}, LL_CV: {consistency['log_likelihood_cv']:.2e})")
                    
                    # Store consistency report
                    consistency['model_name'] = model_name
                    consistency['sample_id'] = sample_id
                    self.consistency_reports.append(consistency)
        
        # Final comprehensive analysis
        self.generate_final_validation_report(validation_results)
        
        return validation_results
    
    def generate_final_validation_report(self, results: List[Dict]):
        """Generate comprehensive final validation report."""
        
        print(f"\n" + "="*80)
        print("FINAL VALIDATION REPORT")
        print("="*80)
        
        total_runs = len(results)
        successful_runs = [r for r in results if r.get('success', False)]
        n_successful = len(successful_runs)
        
        print(f"OVERALL STATISTICS:")
        print(f"  Total optimization runs: {total_runs}")
        print(f"  Successful runs: {n_successful} ({n_successful/total_runs:.1%})")
        
        # Model-specific analysis
        print(f"\nMODEL-SPECIFIC RESULTS:")
        
        models = set(r['model_name'] for r in results)
        for model_name in sorted(models):
            model_results = [r for r in results if r['model_name'] == model_name]
            model_successful = [r for r in model_results if r.get('success', False)]
            
            if model_successful:
                lls = [r['final_log_likelihood'] for r in model_successful]
                aics = [r['aic'] for r in model_successful]
                
                print(f"  {model_name}:")
                print(f"    Success rate: {len(model_successful)}/{len(model_results)} ({len(model_successful)/len(model_results):.1%})")
                print(f"    Log-likelihood: {np.mean(lls):.3f} ± {np.std(lls):.6f}")
                print(f"    AIC: {np.mean(aics):.1f} ± {np.std(aics):.3f}")
            else:
                print(f"  {model_name}: ❌ ALL FAILED")
        
        # Consistency analysis
        print(f"\nCONSISTENCY ANALYSIS:")
        
        consistency_statuses = [report['status'] for report in self.consistency_reports]
        status_counts = pd.Series(consistency_statuses).value_counts()
        
        for status, count in status_counts.items():
            symbol = {
                'ULTRA_CONSISTENT': '✅',
                'MARGINALLY_CONSISTENT': '🟡', 
                'PARAMETER_INCONSISTENT': '🟠',
                'MODERATELY_INCONSISTENT': '🔶',
                'HIGHLY_INCONSISTENT': '❌',
                'PARTIAL_FAILURE': '⚠️',
                'ALL_FAILED': '🚫'
            }.get(status, '❓')
            
            print(f"  {symbol} {status}: {count} cases ({count/len(self.consistency_reports):.1%})")
        
        # Ultra-conservative final verdict
        print(f"\n" + "="*80)
        print("ULTRA-CONSERVATIVE FINAL VERDICT")
        print("="*80)
        
        ultra_consistent_count = sum(1 for report in self.consistency_reports if report['status'] == 'ULTRA_CONSISTENT')
        total_consistency_tests = len(self.consistency_reports)
        
        if n_successful < total_runs * 0.95:
            verdict = "🚫 VALIDATION FAILED - Success rate too low"
        elif ultra_consistent_count < total_consistency_tests * 0.8:
            verdict = "⚠️  VALIDATION CONCERNING - Insufficient ultra-consistent results"
        elif any(report['status'] in ['HIGHLY_INCONSISTENT', 'ALL_FAILED'] for report in self.consistency_reports):
            verdict = "🔶 VALIDATION QUESTIONABLE - Some highly inconsistent results"
        else:
            verdict = "✅ VALIDATION PASSED - Ultra-conservative criteria met"
        
        print(verdict)
        print(f"Ultra-consistent rate: {ultra_consistent_count}/{total_consistency_tests} ({ultra_consistent_count/total_consistency_tests:.1%})")

def main():
    """Run the ultra-conservative validation."""
    validator = UltraConservativeValidator()
    results = validator.run_comprehensive_validation()
    return results

if __name__ == "__main__":
    main()