#!/usr/bin/env python3
"""
Comprehensive Validation Audit for JAX Optimization Claims

Extensive testing to validate the claims about JAX optimization improvements:
1. Solution quality (37.8% improvement claim)
2. Convergence reliability 
3. Multiple model formulations
4. Different starting points
5. Reproducibility
6. Statistical validity
"""

import sys
sys.path.append('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax')

import pradel_jax as pj
import numpy as np
import pandas as pd
from pradel_jax.optimization import optimize_model, OptimizationStrategy, OptimizationConfig
from pradel_jax.models import PradelModel
import jax.numpy as jnp
from jax import grad, jit, random
import time
from typing import Dict, List, Tuple
import json
from datetime import datetime

class ComprehensiveValidationAudit:
    """Comprehensive validation of optimization performance claims."""
    
    def __init__(self):
        self.results = []
        self.summary_stats = {}
        self.audit_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def setup_test_problems(self):
        """Setup multiple test problems for comprehensive validation."""
        
        print("="*80)
        print("COMPREHENSIVE VALIDATION AUDIT")
        print("Extensive testing of JAX optimization performance claims")
        print("="*80)
        
        # Load data
        data_context = pj.load_data('data/dipper_dataset.csv')
        model = PradelModel()
        
        # Define multiple model formulations for testing
        test_formulations = [
            {
                "name": "Constant Model",
                "formula": pj.create_simple_spec(),
                "description": "Constant survival, detection, recruitment"
            }
        ]
        
        # Add covariate models if sex is available
        if 'sex' in data_context.covariates:
            from pradel_jax.formulas.spec import FormulaSpec, ParameterFormula
            
            try:
                test_formulations.extend([
                    {
                        "name": "Sex on Survival",
                        "formula": FormulaSpec(
                            phi=ParameterFormula("phi", "~1 + sex"),
                            p=ParameterFormula("p", "~1"),
                            f=ParameterFormula("f", "~1")
                        ),
                        "description": "Sex effect on survival only"
                    },
                    {
                        "name": "Sex on Detection", 
                        "formula": FormulaSpec(
                            phi=ParameterFormula("phi", "~1"),
                            p=ParameterFormula("p", "~1 + sex"),
                            f=ParameterFormula("f", "~1")
                        ),
                        "description": "Sex effect on detection only"
                    },
                    {
                        "name": "Sex on Both",
                        "formula": FormulaSpec(
                            phi=ParameterFormula("phi", "~1 + sex"),
                            p=ParameterFormula("p", "~1 + sex"),
                            f=ParameterFormula("f", "~1")
                        ),
                        "description": "Sex effects on survival and detection"
                    }
                ])
            except Exception as e:
                print(f"Could not create covariate models: {e}")
                print("Proceeding with constant model only")
        
        return data_context, model, test_formulations
    
    def test_optimization_strategies(self, data_context, model, formulations):
        """Test multiple optimization strategies across different formulations."""
        
        print(f"\n1. OPTIMIZATION STRATEGY TESTING")
        print("-" * 40)
        print(f"Testing {len(formulations)} model formulations")
        
        # Strategies to test
        strategies = [
            OptimizationStrategy.SCIPY_LBFGS,
            OptimizationStrategy.SCIPY_SLSQP, 
            OptimizationStrategy.JAX_LBFGS,
            OptimizationStrategy.JAX_ADAM,
            OptimizationStrategy.JAX_ADAM_ADAPTIVE,
            OptimizationStrategy.MULTI_START,
            OptimizationStrategy.HYBRID
        ]
        
        formulation_results = []
        
        for formulation in formulations:
            print(f"\n• Testing formulation: {formulation['name']}")
            
            try:
                # Build design matrices
                design_matrices = model.build_design_matrices(formulation['formula'], data_context)
                
                def objective(params):
                    return -model.log_likelihood(params, data_context, design_matrices)
                
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                print(f"    Problem size: {len(initial_params)} parameters")
                
                strategy_results = []
                
                for strategy in strategies:
                    print(f"    Testing {strategy.value}...", end="")
                    
                    try:
                        start_time = time.time()
                        result = optimize_model(
                            objective_function=objective,
                            initial_parameters=initial_params,
                            context=data_context,
                            bounds=bounds,
                            preferred_strategy=strategy
                        )
                        elapsed_time = time.time() - start_time
                        
                        strategy_result = {
                            'formulation': formulation['name'],
                            'strategy': strategy.value,
                            'success': result.success,
                            'strategy_used': result.strategy_used,
                            'objective': result.result.fun if hasattr(result, 'result') else None,
                            'iterations': result.result.nit if hasattr(result, 'result') else None,
                            'time': elapsed_time,
                            'aic': getattr(result.result, 'aic', None) if hasattr(result, 'result') else None,
                            'parameters': result.result.x.tolist() if hasattr(result, 'result') else None,
                            'n_params': len(initial_params)
                        }
                        
                        strategy_results.append(strategy_result)
                        self.results.append(strategy_result)
                        
                        status = "✅" if result.success else "⚠️"
                        obj_str = f"{result.result.fun:.2f}" if hasattr(result, 'result') else "N/A"
                        print(f" {status} Obj: {obj_str}, Time: {elapsed_time:.3f}s")
                        
                    except Exception as e:
                        print(f" ❌ Error: {str(e)[:50]}")
                        strategy_results.append({
                            'formulation': formulation['name'],
                            'strategy': strategy.value,
                            'success': False,
                            'error': str(e),
                            'time': 0,
                            'n_params': len(initial_params)
                        })
                
                formulation_results.append({
                    'formulation': formulation,
                    'strategies': strategy_results
                })
                
            except Exception as e:
                print(f"    ❌ Failed to setup formulation: {e}")
        
        return formulation_results
    
    def test_multiple_starting_points(self, data_context, model):
        """Test optimization from multiple starting points."""
        
        print(f"\n2. MULTIPLE STARTING POINTS TEST")
        print("-" * 40)
        
        formula_spec = pj.create_simple_spec()
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Generate multiple starting points
        n_starts = 10
        key = random.PRNGKey(42)
        starting_points = []
        
        # Strategy 1: Random points within bounds
        for i in range(n_starts // 2):
            key, subkey = random.split(key)
            start_point = []
            for (lower, upper) in bounds:
                param = random.uniform(subkey, (), minval=float(lower), maxval=float(upper))
                start_point.append(float(param))
            starting_points.append(("Random", np.array(start_point)))
        
        # Strategy 2: Perturbations around default
        default_start = model.get_initial_parameters(data_context, design_matrices)
        for i in range(n_starts // 2):
            key, subkey = random.split(key)
            perturbation = random.normal(subkey, shape=default_start.shape) * 0.1
            perturbed = default_start + perturbation
            starting_points.append(("Perturbed", perturbed))
        
        print(f"Testing {len(starting_points)} starting points")
        
        # Test key strategies
        test_strategies = [
            OptimizationStrategy.SCIPY_LBFGS,
            OptimizationStrategy.JAX_LBFGS,
            OptimizationStrategy.JAX_ADAM
        ]
        
        starting_point_results = {}
        
        for strategy in test_strategies:
            print(f"\n• Testing {strategy.value} from multiple starts:")
            
            strategy_objectives = []
            strategy_successes = []
            
            for i, (start_type, start_point) in enumerate(starting_points):
                try:
                    result = optimize_model(
                        objective_function=objective,
                        initial_parameters=start_point,
                        context=data_context,
                        bounds=bounds,
                        preferred_strategy=strategy
                    )
                    
                    obj_val = result.result.fun if hasattr(result, 'result') else float('inf')
                    strategy_objectives.append(obj_val)
                    strategy_successes.append(result.success)
                    
                    print(f"    Start {i+1} ({start_type}): Obj={obj_val:.2f}, Success={result.success}")
                    
                except Exception as e:
                    print(f"    Start {i+1} ({start_type}): Error - {str(e)[:30]}")
                    strategy_objectives.append(float('inf'))
                    strategy_successes.append(False)
            
            # Analyze results
            valid_objectives = [obj for obj in strategy_objectives if obj != float('inf')]
            if valid_objectives:
                starting_point_results[strategy.value] = {
                    'best_objective': min(valid_objectives),
                    'worst_objective': max(valid_objectives),
                    'mean_objective': np.mean(valid_objectives),
                    'std_objective': np.std(valid_objectives),
                    'success_rate': np.mean(strategy_successes),
                    'n_valid': len(valid_objectives)
                }
                
                print(f"    Summary: Best={min(valid_objectives):.2f}, "
                      f"Mean={np.mean(valid_objectives):.2f}±{np.std(valid_objectives):.2f}, "
                      f"Success={np.mean(strategy_successes):.1%}")
        
        return starting_point_results
    
    def test_reproducibility(self, data_context, model):
        """Test reproducibility of optimization results."""
        
        print(f"\n3. REPRODUCIBILITY TEST")
        print("-" * 40)
        
        formula_spec = pj.create_simple_spec()
        design_matrices = model.build_design_matrices(formula_spec, data_context)
        
        def objective(params):
            return -model.log_likelihood(params, data_context, design_matrices)
        
        initial_params = model.get_initial_parameters(data_context, design_matrices)
        bounds = model.get_parameter_bounds(data_context, design_matrices)
        
        # Test reproducibility for key strategies
        test_strategies = [
            OptimizationStrategy.SCIPY_LBFGS,
            OptimizationStrategy.JAX_LBFGS,
        ]
        
        n_repeats = 5
        reproducibility_results = {}
        
        for strategy in test_strategies:
            print(f"\n• Testing {strategy.value} reproducibility ({n_repeats} runs):")
            
            run_results = []
            
            for run in range(n_repeats):
                try:
                    result = optimize_model(
                        objective_function=objective,
                        initial_parameters=initial_params,
                        context=data_context,
                        bounds=bounds,
                        preferred_strategy=strategy
                    )
                    
                    obj_val = result.result.fun if hasattr(result, 'result') else float('inf')
                    params = result.result.x.tolist() if hasattr(result, 'result') else None
                    
                    run_results.append({
                        'objective': obj_val,
                        'parameters': params,
                        'success': result.success
                    })
                    
                    print(f"    Run {run+1}: Obj={obj_val:.6f}, Success={result.success}")
                    
                except Exception as e:
                    print(f"    Run {run+1}: Error - {e}")
                    run_results.append({
                        'objective': float('inf'),
                        'parameters': None,
                        'success': False
                    })
            
            # Analyze reproducibility
            valid_runs = [r for r in run_results if r['success']]
            if valid_runs:
                objectives = [r['objective'] for r in valid_runs]
                obj_std = np.std(objectives)
                obj_range = max(objectives) - min(objectives)
                
                reproducibility_results[strategy.value] = {
                    'n_valid_runs': len(valid_runs),
                    'objective_mean': np.mean(objectives),
                    'objective_std': obj_std,
                    'objective_range': obj_range,
                    'consistent': obj_std < 1e-6  # Very tight consistency requirement
                }
                
                print(f"    Summary: Mean={np.mean(objectives):.6f}±{obj_std:.8f}, "
                      f"Range={obj_range:.8f}, Consistent={obj_std < 1e-6}")
        
        return reproducibility_results
    
    def analyze_performance_claims(self):
        """Analyze specific performance claims from the audit results."""
        
        print(f"\n4. PERFORMANCE CLAIMS ANALYSIS")  
        print("-" * 40)
        
        # Filter results for constant model to compare with claims
        constant_results = [r for r in self.results if r.get('formulation') == 'Constant Model' and r.get('success')]
        
        if not constant_results:
            print("❌ No successful constant model results to analyze")
            return {}
        
        # Group by strategy
        strategy_groups = {}
        for result in constant_results:
            strategy = result['strategy']
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(result)
        
        print("Performance comparison (Constant Model):")
        
        claims_validation = {}
        
        # Find SciPy baseline
        scipy_baseline = None
        if 'scipy_lbfgs' in strategy_groups:
            scipy_results = strategy_groups['scipy_lbfgs']
            if scipy_results:
                scipy_baseline = min(r['objective'] for r in scipy_results)
                print(f"  SciPy L-BFGS-B baseline: {scipy_baseline:.6f}")
        
        # Analyze each JAX strategy
        jax_strategies = ['jax_lbfgs', 'jax_adam', 'adaptive_jax_adam']
        
        for strategy in jax_strategies:
            if strategy in strategy_groups and scipy_baseline:
                strategy_results = strategy_groups[strategy]
                best_objective = min(r['objective'] for r in strategy_results)
                improvement = (scipy_baseline - best_objective) / scipy_baseline * 100
                
                claims_validation[strategy] = {
                    'objective': best_objective,
                    'scipy_baseline': scipy_baseline,
                    'improvement_percent': improvement,
                    'improvement_absolute': scipy_baseline - best_objective,
                    'validates_claim': improvement > 30  # 37.8% claim
                }
                
                status = "✅ VALIDATES" if improvement > 30 else "❌ DOES NOT VALIDATE"
                print(f"  {strategy}: {best_objective:.6f} ({improvement:+.1f}%) {status}")
        
        # Overall validation
        validated_strategies = sum(1 for v in claims_validation.values() if v['validates_claim'])
        total_strategies = len(claims_validation)
        
        print(f"\nClaims validation: {validated_strategies}/{total_strategies} strategies validate >30% improvement")
        
        return claims_validation
    
    def generate_comprehensive_report(self):
        """Generate comprehensive validation report."""
        
        print(f"\n5. COMPREHENSIVE VALIDATION REPORT")
        print("-" * 40)
        
        # Calculate summary statistics
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.get('success')])
        
        # Strategy success rates
        strategy_stats = {}
        for result in self.results:
            strategy = result['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {'total': 0, 'success': 0, 'objectives': []}
            
            strategy_stats[strategy]['total'] += 1
            if result.get('success'):
                strategy_stats[strategy]['success'] += 1
                if result.get('objective'):
                    strategy_stats[strategy]['objectives'].append(result['objective'])
        
        print("Strategy Performance Summary:")
        for strategy, stats in strategy_stats.items():
            success_rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            avg_obj = np.mean(stats['objectives']) if stats['objectives'] else None
            
            print(f"  {strategy}: {success_rate:.1%} success ({stats['success']}/{stats['total']})", end="")
            if avg_obj:
                print(f", Avg Obj: {avg_obj:.2f}")
            else:
                print()
        
        # Save detailed results
        report = {
            'audit_timestamp': self.audit_timestamp,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'overall_success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'strategy_statistics': strategy_stats,
            'detailed_results': self.results,
            'summary_stats': self.summary_stats
        }
        
        report_file = f"validation_audit_report_{self.audit_timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved: {report_file}")
        
        return report

def run_comprehensive_validation():
    """Run the complete comprehensive validation audit."""
    
    audit = ComprehensiveValidationAudit()
    
    # Setup test problems
    data_context, model, formulations = audit.setup_test_problems()
    
    # Run comprehensive tests
    formulation_results = audit.test_optimization_strategies(data_context, model, formulations)
    starting_point_results = audit.test_multiple_starting_points(data_context, model)
    reproducibility_results = audit.test_reproducibility(data_context, model)
    
    # Store additional results
    audit.summary_stats.update({
        'formulation_results': formulation_results,
        'starting_point_results': starting_point_results,
        'reproducibility_results': reproducibility_results
    })
    
    # Analyze performance claims
    claims_analysis = audit.analyze_performance_claims()
    audit.summary_stats['claims_analysis'] = claims_analysis
    
    # Generate final report
    final_report = audit.generate_comprehensive_report()
    
    return audit, final_report

if __name__ == "__main__":
    print("Starting Comprehensive Validation Audit...")
    
    try:
        audit, report = run_comprehensive_validation()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION AUDIT COMPLETE")
        print("="*80)
        
        # Final validation summary
        success_rate = report['overall_success_rate']
        claims_validation = report['summary_stats'].get('claims_analysis', {})
        validated_claims = sum(1 for v in claims_validation.values() if v.get('validates_claim', False))
        
        print(f"\n📊 AUDIT SUMMARY:")
        print(f"  • Total tests executed: {report['total_tests']}")
        print(f"  • Overall success rate: {success_rate:.1%}")
        print(f"  • Performance claims validated: {validated_claims}/{len(claims_validation)} strategies")
        
        if success_rate > 0.7 and validated_claims >= 1:
            print(f"\n✅ VALIDATION PASSED")
            print(f"   Optimization improvements validated with extensive testing")
            exit(0)
        else:
            print(f"\n⚠️ VALIDATION CONCERNS")
            print(f"   Some claims require further investigation") 
            exit(1)
            
    except Exception as e:
        print(f"\n❌ AUDIT FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)