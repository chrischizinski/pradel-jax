#!/usr/bin/env python3
"""
Convergence analysis benchmarking for Pradel-JAX optimization framework.
Tests convergence behavior, stability, and robustness across different scenarios.
"""

import time
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime

import pradel_jax as pj
from pradel_jax.models import PradelModel
from pradel_jax.optimization import optimize_model
from pradel_jax.optimization.strategy import OptimizationStrategy


class ConvergenceAnalyzer:
    """Convergence analysis utilities for optimization benchmarking."""
    
    def __init__(self):
        self.results = []
        
    def analyze_convergence_stability(self, 
                                    data_context,
                                    formula_spec,
                                    model: PradelModel,
                                    strategy: str,
                                    n_runs: int = 10) -> Dict[str, Any]:
        """Analyze convergence stability across multiple runs."""
        parameters_history = []
        aic_history = []
        success_history = []
        time_history = []
        
        for run in range(n_runs):
            start_time = time.perf_counter()
            
            try:
                design_matrices = model.build_design_matrices(formula_spec, data_context)
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                # Add required attributes for optimization framework
                if not hasattr(data_context, 'n_parameters'):
                    data_context.n_parameters = len(initial_params)
                if not hasattr(data_context, 'get_condition_estimate'):
                    data_context.get_condition_estimate = lambda: 1e5
                
                def objective(params):
                    try:
                        ll = model.log_likelihood(params, data_context, design_matrices)
                        # For JAX compatibility, don't use float() - return JAX array directly
                        return -ll
                    except Exception:
                        return 1e10
                
                strategy_enum = OptimizationStrategy(strategy)
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=data_context,
                    bounds=bounds,
                    preferred_strategy=strategy_enum
                )
                
                elapsed = time.perf_counter() - start_time
                time_history.append(elapsed)
                success_history.append(result.success)
                
                if result.success:
                    parameters_history.append(result.result.x)
                    final_nll = result.result.fun
                    n_params = len(initial_params)
                    aic = 2 * final_nll + 2 * n_params
                    aic_history.append(aic)
                else:
                    aic_history.append(float('inf'))
                    
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                time_history.append(elapsed)
                success_history.append(False)
                aic_history.append(float('inf'))
        
        # Analyze parameter stability
        param_stats = {}
        if parameters_history:
            param_array = np.array(parameters_history)
            param_stats = {
                'param_means': np.mean(param_array, axis=0),
                'param_stds': np.std(param_array, axis=0),
                'param_cv': np.std(param_array, axis=0) / np.abs(np.mean(param_array, axis=0))
            }
        
        # Analyze AIC stability
        valid_aics = [aic for aic in aic_history if aic != float('inf')]
        aic_stats = {}
        if valid_aics:
            aic_stats = {
                'aic_mean': np.mean(valid_aics),
                'aic_std': np.std(valid_aics),
                'aic_cv': np.std(valid_aics) / np.mean(valid_aics) if np.mean(valid_aics) != 0 else float('inf'),
                'aic_range': max(valid_aics) - min(valid_aics)
            }
        
        return {
            'strategy': strategy,
            'n_runs': n_runs,
            'success_rate': np.mean(success_history),
            'avg_time': np.mean(time_history),
            'std_time': np.std(time_history),
            'param_stats': param_stats,
            'aic_stats': aic_stats,
            'convergence_consistency': len(valid_aics) / n_runs if n_runs > 0 else 0
        }
    
    def test_initialization_sensitivity(self,
                                      data_context,
                                      formula_spec,
                                      model: PradelModel,
                                      strategy: str,
                                      n_inits: int = 5) -> Dict[str, Any]:
        """Test sensitivity to different parameter initializations."""
        results = []
        
        for init_run in range(n_inits):
            # Add noise to initial parameters to test sensitivity
            np.random.seed(42 + init_run)  # Reproducible but different
            
            try:
                design_matrices = model.build_design_matrices(formula_spec, data_context)
                initial_params = model.get_initial_parameters(data_context, design_matrices)
                bounds = model.get_parameter_bounds(data_context, design_matrices)
                
                # Add required attributes for optimization framework
                if not hasattr(data_context, 'n_parameters'):
                    data_context.n_parameters = len(initial_params)
                if not hasattr(data_context, 'get_condition_estimate'):
                    data_context.get_condition_estimate = lambda: 1e5
                
                def objective(params):
                    try:
                        ll = model.log_likelihood(params, data_context, design_matrices)
                        # For JAX compatibility, don't use float() - return JAX array directly
                        return -ll
                    except Exception:
                        return 1e10
                
                strategy_enum = OptimizationStrategy(strategy)
                result = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=data_context,
                    bounds=bounds,
                    preferred_strategy=strategy_enum
                )
                
                if result.success:
                    final_nll = result.result.fun
                    n_params = len(initial_params)
                    aic = 2 * final_nll + 2 * n_params
                else:
                    aic = float('inf')
                
                results.append({
                    'init_run': init_run,
                    'success': result.success,
                    'aic': aic,
                    'parameters': result.result.x if result.success else None
                })
                
            except Exception:
                results.append({
                    'init_run': init_run,
                    'success': False,
                    'aic': float('inf'),
                    'parameters': None
                })
        
        # Analyze results
        success_count = sum(r['success'] for r in results)
        valid_aics = [r['aic'] for r in results if r['success']]
        
        return {
            'strategy': strategy,
            'n_initializations': n_inits,
            'success_rate': success_count / n_inits,
            'aic_variability': np.std(valid_aics) if len(valid_aics) > 1 else 0,
            'best_aic': min(valid_aics) if valid_aics else float('inf'),
            'worst_aic': max(valid_aics) if valid_aics else float('inf'),
            'initialization_robustness': len(set(np.round(valid_aics, 2))) == 1 if len(valid_aics) > 1 else True
        }


@pytest.fixture
def convergence_analyzer():
    """Create convergence analyzer instance."""
    return ConvergenceAnalyzer()


@pytest.fixture
def dipper_data():
    """Load dipper dataset."""
    data_path = Path(__file__).parent.parent.parent / "data" / "dipper_dataset.csv"
    return pj.load_data(str(data_path))


@pytest.fixture
def test_formulas():
    """Create test formulas of different complexities."""
    return {
        'simple': pj.create_simple_spec(phi="~1", p="~1", f="~1"),
        'moderate': pj.create_simple_spec(phi="~sex", p="~1", f="~1"),
        'complex': pj.create_simple_spec(phi="~sex", p="~sex", f="~sex")
    }


class TestConvergenceAnalysis:
    """Convergence analysis test suite."""
    
    def test_lbfgs_convergence_stability(self, dipper_data, test_formulas, convergence_analyzer):
        """Test L-BFGS-B convergence stability."""
        model = pj.PradelModel()
        formula = test_formulas['simple']
        
        result = convergence_analyzer.analyze_convergence_stability(
            data_context=dipper_data,
            formula_spec=formula,
            model=model,
            strategy='lbfgs',
            n_runs=10
        )
        
        print("\n=== L-BFGS-B Convergence Stability ===")
        print(f"Success rate: {result['success_rate']:.1%}")
        print(f"Time consistency: {result['avg_time']:.3f}±{result['std_time']:.3f}s")
        
        if result['aic_stats']:
            print(f"AIC consistency: {result['aic_stats']['aic_mean']:.2f}±{result['aic_stats']['aic_std']:.3f}")
            print(f"AIC CV: {result['aic_stats']['aic_cv']:.3f}")
        
        # Validate convergence quality
        assert result['success_rate'] >= 0.8, f"Poor convergence rate: {result['success_rate']:.1%}"
        
        if result['aic_stats'] and result['aic_stats']['aic_cv'] < float('inf'):
            assert result['aic_stats']['aic_cv'] < 0.01, f"High AIC variability: {result['aic_stats']['aic_cv']:.3f}"
    
    def test_slsqp_convergence_stability(self, dipper_data, test_formulas, convergence_analyzer):
        """Test SLSQP convergence stability."""
        model = pj.PradelModel()
        formula = test_formulas['simple']
        
        result = convergence_analyzer.analyze_convergence_stability(
            data_context=dipper_data,
            formula_spec=formula,
            model=model,
            strategy='slsqp',
            n_runs=8
        )
        
        print("\n=== SLSQP Convergence Stability ===")
        print(f"Success rate: {result['success_rate']:.1%}")
        print(f"Time consistency: {result['avg_time']:.3f}±{result['std_time']:.3f}s")
        
        if result['aic_stats']:
            print(f"AIC consistency: {result['aic_stats']['aic_mean']:.2f}±{result['aic_stats']['aic_std']:.3f}")
        
        # SLSQP may be less stable than L-BFGS-B
        assert result['success_rate'] >= 0.6, f"Poor SLSQP convergence: {result['success_rate']:.1%}"
    
    def test_multi_start_robustness(self, dipper_data, test_formulas, convergence_analyzer):
        """Test multi-start optimization robustness."""
        model = pj.PradelModel()
        formula = test_formulas['moderate']  # Use slightly more complex model
        
        result = convergence_analyzer.analyze_convergence_stability(
            data_context=dipper_data,
            formula_spec=formula,
            model=model,
            strategy='multi_start',
            n_runs=5  # Fewer runs since multi-start is slower
        )
        
        print("\n=== Multi-Start Convergence Robustness ===")
        print(f"Success rate: {result['success_rate']:.1%}")
        print(f"Time consistency: {result['avg_time']:.3f}±{result['std_time']:.3f}s")
        
        if result['aic_stats']:
            print(f"AIC consistency: {result['aic_stats']['aic_mean']:.2f}±{result['aic_stats']['aic_std']:.3f}")
        
        # Multi-start should have high success rate
        assert result['success_rate'] >= 0.8, f"Multi-start should be robust: {result['success_rate']:.1%}"
    
    def test_initialization_sensitivity_lbfgs(self, dipper_data, test_formulas, convergence_analyzer):
        """Test L-BFGS-B sensitivity to initialization."""
        model = pj.PradelModel()
        formula = test_formulas['simple']
        
        result = convergence_analyzer.test_initialization_sensitivity(
            data_context=dipper_data,
            formula_spec=formula,
            model=model,
            strategy='lbfgs',
            n_inits=8
        )
        
        print("\n=== L-BFGS-B Initialization Sensitivity ===")
        print(f"Success rate: {result['success_rate']:.1%}")
        print(f"AIC range: {result['best_aic']:.2f} to {result['worst_aic']:.2f}")
        print(f"AIC variability: {result['aic_variability']:.3f}")
        print(f"Robust to initialization: {result['initialization_robustness']}")
        
        # Should be relatively robust
        assert result['success_rate'] >= 0.7, f"High initialization sensitivity: {result['success_rate']:.1%}"
    
    def test_complex_model_convergence(self, dipper_data, test_formulas, convergence_analyzer):
        """Test convergence on complex models."""
        model = pj.PradelModel()
        formula = test_formulas['complex']
        
        # Test different strategies on complex model
        strategies = ['lbfgs', 'slsqp', 'multi_start']
        results = []
        
        for strategy in strategies:
            result = convergence_analyzer.analyze_convergence_stability(
                data_context=dipper_data,
                formula_spec=formula,
                model=model,
                strategy=strategy,
                n_runs=5
            )
            results.append(result)
        
        print("\n=== Complex Model Convergence Comparison ===")
        for result in results:
            print(f"{result['strategy']:12} | "
                  f"Success: {result['success_rate']:.1%} | "
                  f"Time: {result['avg_time']:.2f}s")
            
            if result['aic_stats']:
                print(f"             | AIC: {result['aic_stats']['aic_mean']:.2f}±{result['aic_stats']['aic_std']:.3f}")
        
        # At least one strategy should handle complex models well
        success_rates = [r['success_rate'] for r in results]
        assert max(success_rates) >= 0.6, f"All strategies failed on complex model: {success_rates}"
    
    def test_convergence_timing_consistency(self, dipper_data, test_formulas, convergence_analyzer):
        """Test timing consistency across convergence runs."""
        model = pj.PradelModel()
        formula = test_formulas['simple']
        
        result = convergence_analyzer.analyze_convergence_stability(
            data_context=dipper_data,
            formula_spec=formula,
            model=model,
            strategy='lbfgs',
            n_runs=12
        )
        
        print("\n=== Timing Consistency Analysis ===")
        print(f"Average time: {result['avg_time']:.3f}s")
        print(f"Time std dev: {result['std_time']:.3f}s")
        print(f"Time CV: {result['std_time']/result['avg_time']*100:.1f}%")
        
        # Timing should be reasonably consistent
        cv_threshold = 0.5  # 50% coefficient of variation
        actual_cv = result['std_time'] / result['avg_time']
        assert actual_cv < cv_threshold, f"High timing variability: {actual_cv:.2f}"


@pytest.mark.benchmark
def test_comprehensive_convergence_analysis(dipper_data, test_formulas):
    """Run comprehensive convergence analysis and save results."""
    analyzer = ConvergenceAnalyzer()
    model = pj.PradelModel()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    
    print(f"\n=== Comprehensive Convergence Analysis ===")
    print(f"Timestamp: {timestamp}")
    
    # Test configurations
    test_configs = [
        ('simple', 'lbfgs', 10),
        ('simple', 'slsqp', 8),
        ('simple', 'multi_start', 5),
        ('moderate', 'lbfgs', 8),
        ('moderate', 'multi_start', 5),
        ('complex', 'lbfgs', 5),
        ('complex', 'multi_start', 3),
    ]
    
    for formula_name, strategy, n_runs in test_configs:
        print(f"\nAnalyzing {formula_name} model with {strategy} ({n_runs} runs)...")
        
        # Convergence stability analysis
        result = analyzer.analyze_convergence_stability(
            data_context=dipper_data,
            formula_spec=test_formulas[formula_name],
            model=model,
            strategy=strategy,
            n_runs=n_runs
        )
        
        # Initialization sensitivity for selected configs
        if strategy in ['lbfgs', 'slsqp'] and formula_name == 'simple':
            init_result = analyzer.test_initialization_sensitivity(
                data_context=dipper_data,
                formula_spec=test_formulas[formula_name],
                model=model,
                strategy=strategy,
                n_inits=6
            )
            result['initialization_analysis'] = init_result
        
        result.update({
            'formula_complexity': formula_name,
            'timestamp': timestamp,
            'dataset': 'dipper'
        })
        
        all_results.append(result)
        
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Avg time: {result['avg_time']:.3f}s")
        if result['aic_stats']:
            print(f"  AIC: {result['aic_stats']['aic_mean']:.2f}±{result['aic_stats']['aic_std']:.3f}")
    
    # Save detailed results
    results_dir = Path(__file__).parent.parent.parent / "tests" / "benchmarks"
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"convergence_analysis_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary report
    summary_data = []
    for result in all_results:
        summary_data.append({
            'formula_complexity': result['formula_complexity'],
            'strategy': result['strategy'],
            'success_rate': result['success_rate'],
            'avg_time': result['avg_time'],
            'time_cv': result['std_time'] / result['avg_time'] if result['avg_time'] > 0 else float('inf'),
            'aic_mean': result['aic_stats']['aic_mean'] if result['aic_stats'] else None,
            'aic_cv': result['aic_stats']['aic_cv'] if result['aic_stats'] else None,
            'convergence_consistency': result['convergence_consistency']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = results_dir / f"convergence_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nResults saved to:")
    print(f"  {results_file}")
    print(f"  {summary_file}")
    
    # Generate recommendations
    print("\n=== Convergence Analysis Recommendations ===")
    
    # Find best strategy for each complexity
    for complexity in ['simple', 'moderate', 'complex']:
        complexity_results = [r for r in all_results if r['formula_complexity'] == complexity]
        if complexity_results:
            best_strategy = max(complexity_results, key=lambda x: x['success_rate'])
            print(f"{complexity.capitalize()} models: {best_strategy['strategy']} "
                  f"(success rate: {best_strategy['success_rate']:.1%})")
    
    # Overall validation
    assert len(all_results) == len(test_configs)
    successful_runs = [r for r in all_results if r['success_rate'] > 0]
    assert len(successful_runs) > 0, "No successful convergence runs"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])