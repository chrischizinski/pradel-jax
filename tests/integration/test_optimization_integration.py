#!/usr/bin/env python3
"""
Comprehensive optimization framework integration test with real Pradel models.

Tests the seamless integration between optimization strategies and core pradel-jax models
using actual capture-recapture data. Validates performance, convergence, and robustness.
"""

import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import pandas as pd
import time
import sys
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import json

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))

from pradel_jax.optimization import (
    optimize_model,
    compare_optimization_strategies,
    OptimizationStrategy,
    OptimizationConfig,
    OptimizationRequest,
    OptimizationOrchestrator,
    recommend_strategy,
    diagnose_optimization_difficulty,
    optimization_experiment
)

from pradel_jax.models import PradelModel
from pradel_jax.data.adapters import DataContext
from pradel_jax.formulas.spec import FormulaSpec, ParameterFormula, ParameterType

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealDataContext(DataContext):
    """DataContext implementation for real capture-recapture data."""
    
    def __init__(self, data_file: str, capture_cols: List[str], covariate_cols: List[str] = None):
        """Initialize with real data from file."""
        self.data_file = data_file
        self.capture_cols = capture_cols
        self.covariate_cols = covariate_cols or []
        
        # Load and process data
        self._load_data()
        
        # Set up context properties
        self.n_individuals = self.capture_matrix.shape[0]
        self.n_occasions = self.capture_matrix.shape[1]
        self.n_parameters = 3  # Default for simple Pradel model (phi, p, f)
        
        logger.info(f"Loaded data: {self.n_individuals} individuals, {self.n_occasions} occasions")
        logger.info(f"Capture rate: {np.mean(self.capture_matrix):.3f}")
        
    def _load_data(self):
        """Load and process capture-recapture data."""
        df = pd.read_csv(self.data_file)
        
        # Extract capture matrix
        if len(self.capture_cols) > 0:
            # Use specified capture columns
            capture_data = df[self.capture_cols].values
        elif 'ch' in df.columns:
            # Parse capture history string
            ch_strings = df['ch'].astype(str)
            max_len = max(len(ch) for ch in ch_strings)
            capture_data = np.array([[int(ch[i]) if i < len(ch) else 0 
                                   for i in range(max_len)] for ch in ch_strings])
        else:
            raise ValueError("No capture data found - specify capture_cols or include 'ch' column")
        
        self.capture_matrix = jnp.array(capture_data, dtype=jnp.int32)
        
        # Extract covariates if available
        self.covariates = {}
        for col in self.covariate_cols:
            if col in df.columns:
                self.covariates[col] = jnp.array(df[col].values)
        
        # Add sex/gender covariate if available
        if 'sex' in df.columns:
            sex_values = df['sex'].map({'Male': 0, 'Female': 1, 'M': 0, 'F': 1}).fillna(0)
            self.covariates['sex'] = jnp.array(sex_values.values, dtype=jnp.int32)
        elif 'gender' in df.columns:
            # Convert gender to numeric
            gender_values = pd.to_numeric(df['gender'], errors='coerce').fillna(0)
            self.covariates['sex'] = jnp.array(gender_values.values, dtype=jnp.int32)
    
    def get_condition_estimate(self):
        """Estimate condition number of the problem."""
        # Simple condition estimate based on data sparsity and structure
        sparsity = np.mean(self.capture_matrix == 0)
        
        # Higher sparsity typically means worse conditioning
        if sparsity > 0.9:
            return 1e8  # Poor conditioning
        elif sparsity > 0.7:
            return 1e6  # Moderate conditioning
        else:
            return 1e4  # Good conditioning


class OptimizationIntegrationTester:
    """Comprehensive tester for optimization framework integration."""
    
    def __init__(self):
        self.results = {}
        self.test_configs = []
        self.model = PradelModel()
        
    def load_test_datasets(self) -> Dict[str, RealDataContext]:
        """Load available test datasets."""
        datasets = {}
        base_path = Path(__file__).parent / "data"
        
        # Try different data files
        data_files = [
            ("dipper_processed", "data/dipper_processed.csv", ['capture_1', 'capture_2', 'capture_3', 'capture_4', 'capture_5', 'capture_6', 'capture_7'], []),
            ("wf_dat", "data/test_datasets/wf.dat.csv", [], ['gender']),
            ("dipper_minimal", "data/dipper_minimal.csv", [], ['sex'])
        ]
        
        for name, filepath, capture_cols, covariate_cols in data_files:
            try:
                full_path = Path(__file__).parent / filepath
                if full_path.exists():
                    logger.info(f"Loading dataset: {name}")
                    datasets[name] = RealDataContext(str(full_path), capture_cols, covariate_cols)
                    logger.info(f"Successfully loaded {name}: {datasets[name].n_individuals} individuals")
                else:
                    logger.warning(f"Dataset file not found: {full_path}")
            except Exception as e:
                logger.warning(f"Failed to load {name}: {e}")
        
        if not datasets:
            logger.warning("No datasets loaded, creating synthetic data")
            datasets["synthetic"] = self._create_synthetic_dataset()
        
        return datasets
    
    def _create_synthetic_dataset(self) -> RealDataContext:
        """Create synthetic capture-recapture data for testing."""
        # Generate realistic capture-recapture data
        np.random.seed(42)
        n_individuals = 500
        n_occasions = 6
        
        # Simulate capture probabilities
        p_base = 0.3
        phi_base = 0.8
        
        capture_matrix = np.zeros((n_individuals, n_occasions))
        
        for i in range(n_individuals):
            alive = True
            for t in range(n_occasions):
                if alive:
                    # Capture probability
                    if np.random.random() < p_base:
                        capture_matrix[i, t] = 1
                    
                    # Survival to next occasion
                    if t < n_occasions - 1:
                        if np.random.random() > phi_base:
                            alive = False
        
        # Create temporary file
        temp_file = Path(__file__).parent / "temp_synthetic_data.csv"
        
        # Create DataFrame with capture histories
        ch_strings = [''.join(map(str, row.astype(int))) for row in capture_matrix]
        sex_values = np.random.choice(['Male', 'Female'], n_individuals)
        
        df = pd.DataFrame({
            'individual_id': [f'id_{i:04d}' for i in range(n_individuals)],
            'ch': ch_strings,
            'sex': sex_values
        })
        
        df.to_csv(temp_file, index=False)
        
        return RealDataContext(str(temp_file), [], ['sex'])
    
    def create_formula_specs(self) -> List[Tuple[str, FormulaSpec]]:
        """Create different model formula specifications."""
        specs = []
        
        # Import parser to create formulas properly
        from pradel_jax.formulas.parser import FormulaParser
        parser = FormulaParser()
        
        # 1. Constant parameters (simplest)
        specs.append(("constant", FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1"),
            p=parser.create_parameter_formula(ParameterType.P, "~1"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )))
        
        # 2. Sex effects (if sex covariate available)
        specs.append(("sex_effects", FormulaSpec(
            phi=parser.create_parameter_formula(ParameterType.PHI, "~1 + sex"),
            p=parser.create_parameter_formula(ParameterType.P, "~1 + sex"),
            f=parser.create_parameter_formula(ParameterType.F, "~1")
        )))
        
        return specs
    
    def test_single_optimization(
        self, 
        dataset_name: str, 
        data_context: RealDataContext,
        formula_spec: FormulaSpec,
        strategy: OptimizationStrategy = None
    ) -> Dict:
        """Test single optimization run."""
        logger.info(f"Testing optimization for {dataset_name} with formula {formula_spec}")
        
        try:
            # Validate data
            self.model.validate_data(data_context)
            
            # Build design matrices
            design_matrices = self.model.build_design_matrices(formula_spec, data_context)
            
            # Get initial parameters and bounds
            initial_params = self.model.get_initial_parameters(data_context, design_matrices)
            bounds = self.model.get_parameter_bounds(data_context, design_matrices)
            
            logger.info(f"Model has {len(initial_params)} parameters")
            
            # Define objective function (negative log-likelihood)
            def objective(params):
                try:
                    ll = self.model.log_likelihood(params, data_context, design_matrices)
                    return -float(ll)  # Minimize negative log-likelihood
                except Exception as e:
                    logger.warning(f"Objective function error: {e}")
                    return 1e10  # Large penalty for invalid parameters
            
            # Define gradient function
            grad_fn = jit(grad(lambda p: -self.model.log_likelihood(p, data_context, design_matrices)))
            
            def gradient(params):
                try:
                    return np.array(grad_fn(params))
                except Exception as e:
                    logger.warning(f"Gradient function error: {e}")
                    return np.zeros_like(params)
            
            start_time = time.time()
            
            if strategy is None:
                # Use automatic strategy selection
                response = optimize_model(
                    objective_function=objective,
                    initial_parameters=initial_params,
                    context=data_context,
                    bounds=bounds,
                    gradient_function=gradient
                )
                strategy_used = response.strategy_used
            else:
                # Use specified strategy
                config = OptimizationConfig(max_iter=1000, tolerance=1e-8)
                
                from pradel_jax.optimization.optimizers import minimize_with_strategy
                result = minimize_with_strategy(
                    strategy, objective, initial_params, bounds, gradient, config
                )
                
                # Create response-like object
                class MockResponse:
                    def __init__(self, result, strategy):
                        self.result = result
                        self.success = result.success if hasattr(result, 'success') else True
                        self.strategy_used = strategy
                        self.recommendations = []
                
                response = MockResponse(result, strategy)
                strategy_used = strategy
            
            optimization_time = time.time() - start_time
            
            # Extract results
            test_result = {
                'dataset': dataset_name,
                'formula': str(formula_spec),
                'strategy': strategy_used.value if hasattr(strategy_used, 'value') else str(strategy_used),
                'success': response.success,
                'optimization_time': optimization_time,
                'n_parameters': len(initial_params),
                'n_individuals': data_context.n_individuals,
                'n_occasions': data_context.n_occasions,
                'capture_rate': float(np.mean(data_context.capture_matrix)),
            }
            
            if response.success and hasattr(response, 'result'):
                test_result.update({
                    'final_nll': float(response.result.fun),
                    'n_iterations': getattr(response.result, 'nit', 0),
                    'final_params': response.result.x.tolist(),
                    'parameter_names': self.model.get_parameter_names(design_matrices)
                })
                
                logger.info(f"Optimization successful: NLL={response.result.fun:.6f}, "
                          f"Time={optimization_time:.2f}s, Strategy={strategy_used}")
            else:
                logger.warning(f"Optimization failed for {dataset_name}")
                if hasattr(response, 'result') and hasattr(response.result, 'message'):
                    test_result['error_message'] = response.result.message
            
            return test_result
            
        except Exception as e:
            logger.error(f"Test failed for {dataset_name}: {e}")
            return {
                'dataset': dataset_name,
                'formula': str(formula_spec),
                'success': False,
                'error': str(e)
            }
    
    def test_strategy_comparison(
        self, 
        dataset_name: str, 
        data_context: RealDataContext,
        formula_spec: FormulaSpec
    ) -> Dict:
        """Compare multiple optimization strategies on same problem."""
        logger.info(f"Comparing strategies for {dataset_name}")
        
        strategies = [
            OptimizationStrategy.SCIPY_LBFGS,
            OptimizationStrategy.SCIPY_SLSQP,
            OptimizationStrategy.JAX_ADAM,
            OptimizationStrategy.MULTI_START
        ]
        
        comparison_results = {}
        
        for strategy in strategies:
            try:
                result = self.test_single_optimization(dataset_name, data_context, formula_spec, strategy)
                comparison_results[strategy.value] = result
            except Exception as e:
                logger.error(f"Strategy {strategy.value} failed: {e}")
                comparison_results[strategy.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return comparison_results
    
    def test_convergence_diagnostics(
        self,
        dataset_name: str,
        data_context: RealDataContext,
        formula_spec: FormulaSpec
    ) -> Dict:
        """Test convergence diagnostics and problem analysis."""
        logger.info(f"Running convergence diagnostics for {dataset_name}")
        
        try:
            # Get strategy recommendation
            recommendation = recommend_strategy(data_context)
            
            # Get problem diagnosis
            diagnosis = diagnose_optimization_difficulty(data_context)
            
            # Test with multiple starting values
            design_matrices = self.model.build_design_matrices(formula_spec, data_context)
            base_initial = self.model.get_initial_parameters(data_context, design_matrices)
            
            multi_start_results = []
            
            for i in range(3):  # Test 3 different starting points
                # Perturb initial values
                np.random.seed(42 + i)
                perturbed_initial = base_initial + np.random.normal(0, 0.5, size=base_initial.shape)
                
                # Run optimization
                def objective(params):
                    return -float(self.model.log_likelihood(params, data_context, design_matrices))
                
                bounds = self.model.get_parameter_bounds(data_context, design_matrices)
                
                try:
                    response = optimize_model(
                        objective_function=objective,
                        initial_parameters=perturbed_initial,
                        context=data_context,
                        bounds=bounds
                    )
                    
                    multi_start_results.append({
                        'start_point': i,
                        'success': response.success,
                        'final_nll': float(response.result.fun) if response.success else None,
                        'strategy_used': response.strategy_used.value if hasattr(response.strategy_used, 'value') else str(response.strategy_used)
                    })
                except Exception as e:
                    multi_start_results.append({
                        'start_point': i,
                        'success': False,
                        'error': str(e)
                    })
            
            return {
                'dataset': dataset_name,
                'recommendation': {
                    'strategy': recommendation.strategy.value,
                    'confidence': recommendation.confidence,
                    'rationale': recommendation.rationale
                },
                'diagnosis': diagnosis,
                'multi_start_results': multi_start_results,
                'convergence_consistency': self._assess_convergence_consistency(multi_start_results)
            }
            
        except Exception as e:
            logger.error(f"Convergence diagnostics failed: {e}")
            return {'error': str(e)}
    
    def _assess_convergence_consistency(self, results: List[Dict]) -> Dict:
        """Assess consistency of convergence across multiple starting points."""
        successful_results = [r for r in results if r.get('success', False)]
        
        if len(successful_results) < 2:
            return {'consistent': False, 'reason': 'Too few successful optimizations'}
        
        nll_values = [r['final_nll'] for r in successful_results]
        nll_std = np.std(nll_values)
        nll_mean = np.mean(nll_values)
        
        # Consider consistent if standard deviation is less than 1% of mean
        consistent = nll_std < 0.01 * abs(nll_mean)
        
        return {
            'consistent': consistent,
            'nll_mean': nll_mean,
            'nll_std': nll_std,
            'coefficient_of_variation': nll_std / abs(nll_mean) if nll_mean != 0 else float('inf'),
            'n_successful': len(successful_results)
        }
    
    def run_comprehensive_test(self) -> Dict:
        """Run comprehensive integration test."""
        logger.info("Starting comprehensive optimization framework integration test")
        
        # Load datasets
        datasets = self.load_test_datasets()
        logger.info(f"Loaded {len(datasets)} datasets")
        
        # Get formula specifications
        formula_specs = self.create_formula_specs()
        logger.info(f"Testing {len(formula_specs)} formula specifications")
        
        results = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'datasets_tested': list(datasets.keys()),
            'single_optimizations': [],
            'strategy_comparisons': [],
            'convergence_diagnostics': [],
            'summary': {}
        }
        
        # Test each dataset with each formula
        for dataset_name, data_context in datasets.items():
            for formula_name, formula_spec in formula_specs:
                
                # Skip sex effects if no sex covariate
                if formula_name == "sex_effects" and 'sex' not in data_context.covariates:
                    continue
                
                logger.info(f"Testing {dataset_name} with {formula_name} formula")
                
                # Single optimization test
                single_result = self.test_single_optimization(
                    f"{dataset_name}_{formula_name}", 
                    data_context, 
                    formula_spec
                )
                results['single_optimizations'].append(single_result)
                
                # Strategy comparison (only for successful cases)
                if single_result.get('success', False):
                    comparison_result = self.test_strategy_comparison(
                        f"{dataset_name}_{formula_name}",
                        data_context,
                        formula_spec
                    )
                    results['strategy_comparisons'].append(comparison_result)
                    
                    # Convergence diagnostics
                    convergence_result = self.test_convergence_diagnostics(
                        f"{dataset_name}_{formula_name}",
                        data_context,
                        formula_spec
                    )
                    results['convergence_diagnostics'].append(convergence_result)
        
        # Generate summary
        results['summary'] = self._generate_test_summary(results)
        
        return results
    
    def _generate_test_summary(self, results: Dict) -> Dict:
        """Generate summary statistics from test results."""
        single_optimizations = results['single_optimizations']
        strategy_comparisons = results['strategy_comparisons']
        
        # Success rates
        total_tests = len(single_optimizations)
        successful_tests = sum(1 for r in single_optimizations if r.get('success', False))
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        # Strategy performance
        strategy_performance = {}
        for comp in strategy_comparisons:
            for strategy, result in comp.items():
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {'total': 0, 'successful': 0, 'times': []}
                
                strategy_performance[strategy]['total'] += 1
                if result.get('success', False):
                    strategy_performance[strategy]['successful'] += 1
                    if 'optimization_time' in result:
                        strategy_performance[strategy]['times'].append(result['optimization_time'])
        
        # Calculate success rates and average times for each strategy
        for strategy in strategy_performance:
            perf = strategy_performance[strategy]
            perf['success_rate'] = perf['successful'] / perf['total'] if perf['total'] > 0 else 0
            perf['avg_time'] = np.mean(perf['times']) if perf['times'] else None
            perf['std_time'] = np.std(perf['times']) if len(perf['times']) > 1 else None
        
        return {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'overall_success_rate': success_rate,
            'strategy_performance': strategy_performance,
            'datasets_tested': len(results.get('datasets_tested', [])),
            'avg_optimization_time': np.mean([r.get('optimization_time', 0) 
                                            for r in single_optimizations 
                                            if r.get('success', False) and 'optimization_time' in r])
        }
    
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"optimization_integration_test_results_{timestamp}.json"
        
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        
        # Also save summary to text file
        summary_file = filepath.with_suffix('.txt')
        self._save_summary_report(results, summary_file)
        
        return filepath
    
    def _save_summary_report(self, results: Dict, filepath: Path):
        """Save human-readable summary report."""
        summary = results['summary']
        
        report = f"""
Optimization Framework Integration Test Results
==============================================

Test Timestamp: {results['test_timestamp']}
Datasets Tested: {', '.join(results['datasets_tested'])}

Overall Performance:
- Total Tests: {summary['total_tests']}
- Successful Tests: {summary['successful_tests']}
- Success Rate: {summary['overall_success_rate']:.1%}
- Average Optimization Time: {summary.get('avg_optimization_time', 0):.2f}s

Strategy Performance:
"""
        
        for strategy, perf in summary['strategy_performance'].items():
            report += f"\n{strategy}:\n"
            report += f"  Success Rate: {perf['success_rate']:.1%} ({perf['successful']}/{perf['total']})\n"
            if perf['avg_time'] is not None:
                report += f"  Average Time: {perf['avg_time']:.2f}±{perf.get('std_time', 0):.2f}s\n"
        
        report += f"\n\nDetailed Results:\n"
        report += f"Single optimizations: {len(results['single_optimizations'])}\n"
        report += f"Strategy comparisons: {len(results['strategy_comparisons'])}\n"
        report += f"Convergence diagnostics: {len(results['convergence_diagnostics'])}\n"
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Summary report saved to {filepath}")


def main():
    """Run the comprehensive integration test."""
    logger.info("Pradel-JAX Optimization Framework Integration Test")
    logger.info("=" * 60)
    
    tester = OptimizationIntegrationTester()
    
    try:
        # Run comprehensive test
        results = tester.run_comprehensive_test()
        
        # Save results
        results_file = tester.save_results(results)
        
        # Print summary
        summary = results['summary']
        print(f"\n{'='*60}")
        print("INTEGRATION TEST COMPLETE")
        print(f"{'='*60}")
        print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
        print(f"Total Tests Run: {summary['total_tests']}")
        print(f"Datasets Tested: {summary['datasets_tested']}")
        print(f"Results saved to: {results_file}")
        
        if summary['overall_success_rate'] > 0.8:
            print("✅ Integration test PASSED - Optimization framework working well!")
        else:
            print("⚠️  Integration test shows issues - Check detailed results")
        
        return results
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise


if __name__ == "__main__":
    results = main()