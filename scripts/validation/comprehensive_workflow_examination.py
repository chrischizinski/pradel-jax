#!/usr/bin/env python3
"""
Comprehensive Workflow Examination
==================================

Deep analysis of the entire Pradel-JAX workflow looking for:
1. Process errors in data handling, parsing, optimization
2. Statistical errors in likelihood computation, inference, parameter interpretation
3. Edge cases and robustness issues
4. Biological plausibility of results

This analysis is designed to be dataset-agnostic and not fit to specific data.
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

import pradel_jax as pj
from pradel_jax.data.adapters import load_data, GenericFormatAdapter
from pradel_jax.models.pradel import PradelModel
from pradel_jax.formulas.spec import FormulaSpec
from pradel_jax.optimization import optimize_model

class WorkflowExaminer:
    """Comprehensive workflow examination tool."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'datasets_examined': [],
            'process_issues': [],
            'statistical_issues': [],
            'edge_cases': [],
            'validation_results': {},
            'recommendations': []
        }
        
    def examine_dataset(self, filepath: str, dataset_name: str) -> Dict[str, Any]:
        """Examine a single dataset comprehensively."""
        logger.info(f"=== EXAMINING DATASET: {dataset_name} ===")
        
        dataset_results = {
            'name': dataset_name,
            'filepath': filepath,
            'data_quality': {},
            'loading_issues': [],
            'formula_issues': [],
            'optimization_issues': [],
            'statistical_issues': [],
            'parameter_plausibility': {}
        }
        
        try:
            # 1. Data Loading and Quality Assessment
            dataset_results['data_quality'] = self.assess_data_quality(filepath)
            
            # 2. Test Data Loading Process
            data_context, loading_issues = self.test_data_loading(filepath)
            dataset_results['loading_issues'] = loading_issues
            
            if data_context is None:
                dataset_results['critical_error'] = "Failed to load data"
                return dataset_results
            
            # 3. Test Formula Parsing and Design Matrix Construction
            formula_results = self.test_formula_system(data_context)
            dataset_results['formula_issues'] = formula_results
            
            # 4. Test Model Likelihood Computation
            likelihood_results = self.test_likelihood_computation(data_context)
            dataset_results['likelihood_issues'] = likelihood_results
            
            # 5. Test Optimization Process
            opt_results = self.test_optimization_process(data_context)
            dataset_results['optimization_issues'] = opt_results
            
            # 6. Test Statistical Inference
            inference_results = self.test_statistical_inference(data_context)
            dataset_results['statistical_issues'] = inference_results
            
            # 7. Assess Parameter Biological Plausibility
            plausibility_results = self.assess_parameter_plausibility(data_context)
            dataset_results['parameter_plausibility'] = plausibility_results
            
        except Exception as e:
            dataset_results['critical_error'] = str(e)
            logger.error(f"Critical error examining {dataset_name}: {e}")
        
        return dataset_results
    
    def assess_data_quality(self, filepath: str) -> Dict[str, Any]:
        """Assess fundamental data quality issues."""
        try:
            df = pd.read_csv(filepath)
            
            quality_metrics = {
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'missing_data': {},
                'data_types': {},
                'encounter_patterns': {},
                'temporal_issues': []
            }
            
            # Missing data analysis
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    quality_metrics['missing_data'][col] = {
                        'count': int(missing_count),
                        'percentage': float(missing_count / len(df) * 100)
                    }
            
            # Data type analysis
            quality_metrics['data_types'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Encounter history analysis (look for Y columns or ch column)
            encounter_cols = [col for col in df.columns if col.startswith('Y') or col == 'ch']
            if encounter_cols:
                quality_metrics['encounter_patterns'] = self.analyze_encounter_patterns(df, encounter_cols)
            
            return quality_metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_encounter_patterns(self, df: pd.DataFrame, encounter_cols: List[str]) -> Dict[str, Any]:
        """Analyze encounter history patterns for data quality issues."""
        patterns = {
            'total_encounters': 0,
            'individuals_with_encounters': 0,
            'temporal_gaps': [],
            'suspicious_patterns': []
        }
        
        if 'ch' in encounter_cols:
            # Character-based encounter histories
            ch_series = df['ch'].dropna()
            patterns['total_encounters'] = ch_series.str.count('1').sum()
            patterns['individuals_with_encounters'] = (ch_series.str.count('1') > 0).sum()
            
            # Check for suspicious patterns
            all_zeros = (ch_series.str.count('1') == 0).sum()
            if all_zeros > len(ch_series) * 0.1:  # More than 10% never encountered
                patterns['suspicious_patterns'].append(f"High proportion never encountered: {all_zeros}/{len(ch_series)}")
                
        else:
            # Y-column format
            y_cols = [col for col in encounter_cols if col.startswith('Y')]
            if y_cols:
                encounter_matrix = df[y_cols].fillna(0)
                patterns['total_encounters'] = encounter_matrix.sum().sum()
                patterns['individuals_with_encounters'] = (encounter_matrix.sum(axis=1) > 0).sum()
                
                # Check for temporal consistency
                years = [int(col[1:]) if col[1:].isdigit() else 0 for col in y_cols]
                if len(set(years)) == len(years) and min(years) > 0:  # Consecutive years
                    sorted_years = sorted(years)
                    for i in range(1, len(sorted_years)):
                        if sorted_years[i] - sorted_years[i-1] > 1:
                            patterns['temporal_gaps'].append(f"Gap between {sorted_years[i-1]} and {sorted_years[i]}")
        
        return patterns
    
    def test_data_loading(self, filepath: str) -> Tuple[Optional[Any], List[str]]:
        """Test the data loading process for errors."""
        issues = []
        data_context = None
        
        try:
            # Test with automatic format detection
            data_context = load_data(filepath)
            
            # Validate data context structure
            if not hasattr(data_context, 'capture_matrix'):
                issues.append("Data context missing capture_matrix attribute")
            else:
                if data_context.capture_matrix.size == 0:
                    issues.append("Empty capture matrix")
                
                # Check matrix dimensions
                if len(data_context.capture_matrix.shape) != 2:
                    issues.append(f"Capture matrix has wrong dimensions: {data_context.capture_matrix.shape}")
            
            if not hasattr(data_context, 'n_individuals') or data_context.n_individuals == 0:
                issues.append("No individuals in dataset")
                
            if not hasattr(data_context, 'n_occasions') or data_context.n_occasions == 0:
                issues.append("No time occasions in dataset")
            
            if hasattr(data_context, 'covariates'):
                if data_context.covariates is None or len(data_context.covariates) == 0:
                    issues.append("No covariates loaded")
            
        except Exception as e:
            issues.append(f"Data loading failed: {str(e)}")
            
        return data_context, issues
    
    def test_formula_system(self, data_context) -> List[str]:
        """Test formula parsing and design matrix construction."""
        issues = []
        
        try:
            # Test basic formulas
            basic_formulas = [
                ("phi", "~1"),
                ("p", "~1"), 
                ("f", "~1")
            ]
            
            # Test with covariates if available
            if hasattr(data_context, 'covariates') and data_context.covariates:
                covariate_names = list(data_context.covariates.keys())
                if covariate_names:
                    # Test single covariate formulas
                    for param in ['phi', 'p', 'f']:
                        for cov in covariate_names[:2]:  # Test first two covariates
                            try:
                                formula_spec = pj.create_formula_spec(**{param: f"~1 + {cov}"})
                                model = PradelModel()
                                design_matrices = model.build_design_matrices(formula_spec, data_context)
                            except Exception as e:
                                issues.append(f"Formula '{param} ~ 1 + {cov}' failed: {str(e)}")
            
            # Test for common formula parsing issues
            problematic_formulas = [
                "~nonexistent_covariate",
                "~1 + ",
                "~ +covariate",
                "~covariate1 + covariate1"  # Duplicate terms
            ]
            
            for formula in problematic_formulas:
                try:
                    formula_spec = pj.create_formula_spec(phi=formula, p="~1", f="~1")
                    # If this doesn't fail, it might be an issue
                    issues.append(f"Problematic formula '{formula}' was accepted (should it be?)")
                except:
                    # This is expected for problematic formulas
                    pass
                    
        except Exception as e:
            issues.append(f"Formula system testing failed: {str(e)}")
            
        return issues
    
    def test_likelihood_computation(self, data_context) -> List[str]:
        """Test model likelihood computation for accuracy and stability."""
        issues = []
        
        try:
            model = PradelModel()
            formula_spec = pj.create_formula_spec(phi="~1", p="~1", f="~1")
            design_matrices = model.build_design_matrices(formula_spec, data_context)
            
            # Test with reasonable parameter values
            test_params = np.array([0.0, 0.0, 0.0])  # Logit scale, should give ~0.5 probabilities
            
            try:
                likelihood = model.log_likelihood(test_params, data_context, design_matrices)
                
                # Check for numerical issues
                if np.isnan(likelihood):
                    issues.append("Likelihood computation returns NaN")
                elif np.isinf(likelihood):
                    issues.append("Likelihood computation returns infinity")
                elif likelihood > 0:
                    issues.append(f"Log-likelihood is positive: {likelihood} (should be <= 0)")
                
                # Test gradient computation
                try:
                    import jax
                    grad_fn = jax.grad(lambda params: model.log_likelihood(params, data_context, design_matrices))
                    gradient = grad_fn(test_params)
                    
                    if np.any(np.isnan(gradient)):
                        issues.append("Gradient computation contains NaN")
                    if np.any(np.isinf(gradient)):
                        issues.append("Gradient computation contains infinity")
                        
                except Exception as e:
                    issues.append(f"Gradient computation failed: {str(e)}")
                    
            except Exception as e:
                issues.append(f"Likelihood computation failed: {str(e)}")
            
            # Test with extreme parameter values
            extreme_params = [
                np.array([10.0, 10.0, 10.0]),   # High values
                np.array([-10.0, -10.0, -10.0]), # Low values
                np.array([0.0, 10.0, -10.0])    # Mixed values
            ]
            
            for i, params in enumerate(extreme_params):
                try:
                    likelihood = model.log_likelihood(params, data_context, design_matrices)
                    if np.isnan(likelihood) or np.isinf(likelihood):
                        issues.append(f"Likelihood unstable with extreme parameters {i+1}")
                except:
                    issues.append(f"Likelihood computation crashes with extreme parameters {i+1}")
                    
        except Exception as e:
            issues.append(f"Likelihood testing failed: {str(e)}")
            
        return issues
    
    def test_optimization_process(self, data_context) -> List[str]:
        """Test optimization process for convergence and stability."""
        issues = []
        
        try:
            # Test with simple model
            result = pj.fit_model(
                model=PradelModel(),
                formula=pj.create_formula_spec(phi="~1", p="~1", f="~1"),
                data=data_context
            )
            
            # Check optimization result
            if not result.success:
                issues.append(f"Optimization failed to converge: {result.message}")
            
            if hasattr(result, 'parameters'):
                params = result.parameters
                
                # Check for unrealistic parameter values (on logit scale)
                if np.any(np.abs(params) > 10):
                    issues.append(f"Extreme parameter values detected: {params}")
                
                # Convert to probability scale for interpretation
                probs = 1 / (1 + np.exp(-params))
                if np.any(probs < 0.01) or np.any(probs > 0.99):
                    issues.append(f"Extreme probability values: {probs}")
            
            # Test optimization stability (multiple runs)
            results = []
            for i in range(3):
                try:
                    result_i = pj.fit_model(
                        model=PradelModel(),
                        formula=pj.create_formula_spec(phi="~1", p="~1", f="~1"),
                        data=data_context
                    )
                    if result_i.success and hasattr(result_i, 'parameters'):
                        results.append(result_i.parameters)
                except:
                    pass
            
            # Check consistency across runs
            if len(results) >= 2:
                param_diff = np.abs(np.array(results[0]) - np.array(results[1]))
                if np.any(param_diff > 0.5):  # Large differences in logit scale
                    issues.append(f"Optimization instability detected: parameter differences {param_diff}")
                    
        except Exception as e:
            issues.append(f"Optimization testing failed: {str(e)}")
            
        return issues
    
    def test_statistical_inference(self, data_context) -> List[str]:
        """Test statistical inference components."""
        issues = []
        
        try:
            # Fit model and check for standard errors/confidence intervals
            result = pj.fit_model(
                model=PradelModel(),
                formula=pj.create_formula_spec(phi="~1", p="~1", f="~1"),
                data=data_context
            )
            
            if result.success:
                # Check if standard errors are available
                if not hasattr(result, 'standard_errors') and not hasattr(result, 'hessian'):
                    issues.append("No uncertainty quantification available (standard errors or Hessian)")
                
                # Check AIC/BIC calculation
                if not hasattr(result, 'aic'):
                    issues.append("AIC not calculated")
                elif result.aic is None or np.isnan(result.aic):
                    issues.append("AIC calculation failed")
                
                # Test model comparison framework
                try:
                    # Compare with more complex model if covariates available
                    if hasattr(data_context, 'covariates') and data_context.covariates:
                        covariate_names = list(data_context.covariates.keys())
                        if covariate_names:
                            complex_formula = pj.create_formula_spec(
                                phi=f"~1 + {covariate_names[0]}",
                                p="~1",
                                f="~1"
                            )
                            complex_result = pj.fit_model(
                                model=PradelModel(),
                                formula=complex_formula,
                                data=data_context
                            )
                            
                            if complex_result.success and hasattr(complex_result, 'aic'):
                                if complex_result.aic <= result.aic:
                                    # More complex model should have higher AIC unless justified
                                    param_diff = len(complex_result.parameters) - len(result.parameters)
                                    aic_improvement = result.aic - complex_result.aic
                                    if aic_improvement < 2 * param_diff:
                                        issues.append("Model selection may not be penalizing complexity appropriately")
                except Exception as e:
                    issues.append(f"Model comparison testing failed: {str(e)}")
                    
        except Exception as e:
            issues.append(f"Statistical inference testing failed: {str(e)}")
            
        return issues
    
    def assess_parameter_plausibility(self, data_context) -> Dict[str, Any]:
        """Assess biological plausibility of parameter estimates."""
        plausibility = {
            'survival_rates': {},
            'detection_rates': {},
            'entry_rates': {},
            'biological_warnings': []
        }
        
        try:
            result = pj.fit_model(
                model=PradelModel(),
                formula=pj.create_formula_spec(phi="~1", p="~1", f="~1"),
                data=data_context
            )
            
            if result.success and hasattr(result, 'parameters'):
                # Convert to probability scale
                phi_prob = 1 / (1 + np.exp(-result.parameters[0]))  # Survival
                p_prob = 1 / (1 + np.exp(-result.parameters[1]))    # Detection
                f_prob = 1 / (1 + np.exp(-result.parameters[2]))    # Entry/Recruitment
                
                plausibility['survival_rates']['estimate'] = float(phi_prob)
                plausibility['detection_rates']['estimate'] = float(p_prob)
                plausibility['entry_rates']['estimate'] = float(f_prob)
                
                # Biological plausibility checks
                if phi_prob < 0.1:
                    plausibility['biological_warnings'].append(f"Very low survival rate: {phi_prob:.3f}")
                elif phi_prob > 0.99:
                    plausibility['biological_warnings'].append(f"Unrealistically high survival rate: {phi_prob:.3f}")
                
                if p_prob < 0.05:
                    plausibility['biological_warnings'].append(f"Very low detection rate: {p_prob:.3f}")
                elif p_prob > 0.95:
                    plausibility['biological_warnings'].append(f"Unrealistically high detection rate: {p_prob:.3f}")
                
                if f_prob < 0.01:
                    plausibility['biological_warnings'].append(f"Very low entry rate: {f_prob:.3f}")
                elif f_prob > 0.8:
                    plausibility['biological_warnings'].append(f"Unrealistically high entry rate: {f_prob:.3f}")
                
                # Check parameter correlations and identifiability
                if abs(phi_prob - p_prob) < 0.01:
                    plausibility['biological_warnings'].append("Survival and detection rates very similar - potential identifiability issue")
                    
        except Exception as e:
            plausibility['error'] = str(e)
            
        return plausibility
    
    def run_comprehensive_examination(self) -> Dict[str, Any]:
        """Run comprehensive examination on all available real datasets."""
        logger.info("=== STARTING COMPREHENSIVE WORKFLOW EXAMINATION ===")
        
        # Define datasets to examine
        datasets = [
            ('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax/data/dipper_dataset.csv', 'dipper'),
            ('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax/data/encounter_histories_ne_clean.csv', 'nebraska'),
            ('/Users/cchizinski2/Documents/git2/student_work/ava_britton/pradel-jax/data/encounter_histories_sd_clean.csv', 'south_dakota')
        ]
        
        for filepath, name in datasets:
            if Path(filepath).exists():
                logger.info(f"Examining dataset: {name}")
                dataset_result = self.examine_dataset(filepath, name)
                self.results['datasets_examined'].append(dataset_result)
            else:
                logger.warning(f"Dataset not found: {filepath}")
        
        # Synthesize cross-dataset issues
        self.synthesize_findings()
        
        # Generate recommendations
        self.generate_recommendations()
        
        return self.results
    
    def synthesize_findings(self):
        """Synthesize findings across all datasets."""
        all_process_issues = []
        all_statistical_issues = []
        all_edge_cases = []
        
        for dataset in self.results['datasets_examined']:
            all_process_issues.extend(dataset.get('loading_issues', []))
            all_process_issues.extend(dataset.get('formula_issues', []))
            all_process_issues.extend(dataset.get('optimization_issues', []))
            all_statistical_issues.extend(dataset.get('statistical_issues', []))
            all_statistical_issues.extend(dataset.get('likelihood_issues', []))
            
            # Extract edge cases from parameter plausibility
            plausibility = dataset.get('parameter_plausibility', {})
            if 'biological_warnings' in plausibility:
                all_edge_cases.extend(plausibility['biological_warnings'])
        
        # Remove duplicates while preserving order
        self.results['process_issues'] = list(dict.fromkeys(all_process_issues))
        self.results['statistical_issues'] = list(dict.fromkeys(all_statistical_issues))
        self.results['edge_cases'] = list(dict.fromkeys(all_edge_cases))
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on findings."""
        recommendations = []
        
        # Process-based recommendations
        if any('loading' in issue.lower() for issue in self.results['process_issues']):
            recommendations.append("Enhance data loading validation and error handling")
        
        if any('formula' in issue.lower() for issue in self.results['process_issues']):
            recommendations.append("Improve formula parsing error messages and validation")
        
        if any('optimization' in issue.lower() for issue in self.results['process_issues']):
            recommendations.append("Implement more robust optimization convergence criteria")
        
        # Statistical recommendations
        if any('likelihood' in issue.lower() for issue in self.results['statistical_issues']):
            recommendations.append("Add numerical stability checks to likelihood computation")
        
        if any('uncertainty' in issue.lower() or 'standard error' in issue.lower() for issue in self.results['statistical_issues']):
            recommendations.append("Implement robust uncertainty quantification methods")
        
        # Biological plausibility recommendations
        if any('unrealistic' in warning.lower() for warning in self.results['edge_cases']):
            recommendations.append("Add biological bounds and plausibility checks to optimization")
        
        if any('identifiability' in warning.lower() for warning in self.results['edge_cases']):
            recommendations.append("Implement identifiability diagnostics and warnings")
        
        # General recommendations
        recommendations.append("Add comprehensive unit tests for edge cases identified")
        recommendations.append("Implement real-time model diagnostics during optimization")
        recommendations.append("Create user-friendly warnings for biological implausibility")
        
        self.results['recommendations'] = recommendations
    
    def save_report(self, filename: str = None):
        """Save comprehensive examination report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_workflow_examination_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Comprehensive examination report saved to: {filename}")
        return filename

def main():
    """Run comprehensive workflow examination."""
    examiner = WorkflowExaminer()
    results = examiner.run_comprehensive_examination()
    
    # Save detailed report
    report_filename = examiner.save_report()
    
    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE WORKFLOW EXAMINATION SUMMARY")
    print("="*80)
    
    print(f"\nDatasets Examined: {len(results['datasets_examined'])}")
    for dataset in results['datasets_examined']:
        print(f"  - {dataset['name']}: {'✓' if 'critical_error' not in dataset else '✗'}")
    
    print(f"\nProcess Issues Found: {len(results['process_issues'])}")
    for issue in results['process_issues'][:5]:  # Show first 5
        print(f"  - {issue}")
    if len(results['process_issues']) > 5:
        print(f"  ... and {len(results['process_issues']) - 5} more")
    
    print(f"\nStatistical Issues Found: {len(results['statistical_issues'])}")
    for issue in results['statistical_issues'][:5]:  # Show first 5
        print(f"  - {issue}")
    if len(results['statistical_issues']) > 5:
        print(f"  ... and {len(results['statistical_issues']) - 5} more")
    
    print(f"\nEdge Cases/Warnings: {len(results['edge_cases'])}")
    for case in results['edge_cases'][:5]:  # Show first 5
        print(f"  - {case}")
    if len(results['edge_cases']) > 5:
        print(f"  ... and {len(results['edge_cases']) - 5} more")
    
    print(f"\nRecommendations: {len(results['recommendations'])}")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\n📋 Detailed report saved to: {report_filename}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()